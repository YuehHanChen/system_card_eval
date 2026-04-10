"""Main evaluation pipeline: send system card text + prompts to 3 LLM judges via OpenRouter.

For small documents (<= context limit): send full text in one shot.
For large documents (> context limit): agentic mode — send TOC first, let judge
request sections across up to 3 turns, then score.
"""

import json
import re
import time
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from config import (
    JUDGES, METRICS, MODELS, RUNS_PER_JUDGE,
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    PROMPTS_DIR, RESULTS_DIR,
)
from extract_text import build_model_text, get_relevant_pages


# ── Setup ──────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

SYSTEM_PROMPT = (PROMPTS_DIR / "system_prompt.txt").read_text()

# Context window limits per judge (input tokens, conservative)
JUDGE_CONTEXT_LIMITS = {
    "sonnet_4_6": 190_000,
    "gpt_5_4": 120_000,
    "gemini_3_1_pro": 950_000,
}

MAX_AGENT_TURNS = 2


def load_metric_prompt(metric_key: str) -> str:
    metric_info = METRICS[metric_key]
    prompt_path = PROMPTS_DIR / metric_info["type"] / f"{metric_key}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    return prompt_path.read_text()


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _extract_json(content: str) -> dict:
    """Extract JSON from model response, handling markdown fences."""
    content = content.strip()
    if not content:
        raise json.JSONDecodeError("Empty response from API", "", 0)
    if content.startswith("```"):
        lines = content.split("\n")
        start, end = 0, len(lines)
        for i, line in enumerate(lines):
            if line.strip().startswith("```") and i == 0:
                start = i + 1
            elif line.strip() == "```" and i > 0:
                end = i
        content = "\n".join(lines[start:end])
    return json.loads(content)


def _api_call(judge_key: str, messages: list, retry_count: int = 3) -> tuple[str, dict]:
    """Make an API call and return (content, usage_dict). Retries on failure."""
    judge_config = JUDGES[judge_key]
    extra_body = dict(judge_config.get("extra_body", {}))

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=judge_config["model"],
                messages=messages,
                max_tokens=judge_config["max_tokens"],
                temperature=0.3,
                extra_body=extra_body if extra_body else None,
            )
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            }
            return content, usage
        except Exception as e:
            print(f"      [ERROR] attempt {attempt + 1}: {e}")
            if attempt < retry_count - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def _format_toc(toc: list[dict]) -> str:
    """Format TOC entries for display."""
    lines = []
    for entry in toc:
        indent = "  " * (entry["level"] - 1)
        page = f"p. {entry['page']}" if entry["page"] else ""
        lines.append(f"{indent}{entry['title']}  {page}")
    return "\n".join(lines)


def _get_pages_by_range(pages: list[dict], page_numbers: list[int]) -> str:
    """Get text for specific page numbers."""
    page_map = {p["page"]: p["text"] for p in pages}
    parts = []
    for pn in sorted(page_numbers):
        if pn in page_map:
            parts.append(f"--- PAGE {pn} ---\n{page_map[pn]}")
        else:
            parts.append(f"--- PAGE {pn} ---\n[Page not found]")
    return "\n".join(parts)


def _parse_page_requests(content: str) -> list[int]:
    """Parse page numbers from the judge's response requesting sections."""
    pages = set()
    # Match patterns like "pages 1-5, 10, 23-30" or "p. 1, 2, 3" or "Pages: 1-5"
    for match in re.finditer(r'(\d+)\s*[-–]\s*(\d+)', content):
        start, end = int(match.group(1)), int(match.group(2))
        pages.update(range(start, min(end + 1, start + 50)))  # cap range to 50 pages
    for match in re.finditer(r'(?:page|p\.?)\s*(\d+)', content, re.IGNORECASE):
        pages.add(int(match.group(1)))
    # Also match bare numbers in comma-separated lists after "pages:" or "sections:"
    for match in re.finditer(r'(?:pages|sections)\s*:?\s*([\d,\s]+)', content, re.IGNORECASE):
        for num in re.findall(r'\d+', match.group(1)):
            pages.add(int(num))
    return sorted(pages)


# ── Single-shot mode (small docs) ─────────────────────────────────────

def judge_single_shot(
    judge_key: str,
    metric_key: str,
    model_key: str,
    model_text_data: dict,
) -> dict:
    """Score a small document in one API call (full text fits in context)."""
    metric_prompt = load_metric_prompt(metric_key)
    full_text = "\n".join(
        f"--- PAGE {p['page']} ---\n{p['text']}"
        for p in model_text_data["combined_pages"]
    )

    display_name = MODELS[model_key]["display_name"]
    company = MODELS[model_key]["company"]
    total_pages = model_text_data["total_pages"]
    num_companions = len(model_text_data["companion_texts"])

    user_msg = (
        f"MODEL BEING EVALUATED: {display_name} by {company}\n"
        f"DOCUMENT: System card ({total_pages} pages total"
        f"{f', plus {num_companions} companion report(s)' if num_companions > 0 else ''})\n"
        f"NOTE: You are seeing the full document.\n"
        f"\n{'='*80}\n"
        f"METRIC TO EVALUATE:\n\n{metric_prompt}\n"
        f"\n{'='*80}\n"
        f"DOCUMENT TEXT:\n\n{full_text}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    content, usage = _api_call(judge_key, messages)
    parsed = _extract_json(content)
    parsed["_mode"] = "single_shot"
    parsed["_usage"] = usage
    return parsed


# ── Agentic mode (large docs) ─────────────────────────────────────────

AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT + """

AGENTIC MODE: The document is too large to send in full. You will interact in 2 turns:

Turn 1: You receive the TABLE OF CONTENTS and the metric to evaluate. Respond with a JSON object specifying which pages you want to read:
{"action": "request_pages", "pages": [1, 2, 3, 10, 11, 12, 34, 35, 36], "reasoning": "I need pages X-Y for ... and pages Z for ..."}

Turn 2: You receive the requested pages. You MUST provide your final score as the standard JSON response format for the metric.

Request at most 60 pages per turn to stay within context limits. Be strategic — use the TOC to identify the most relevant sections for this metric.
"""


def _prefetch_pages(
    judge_keys: list[str],
    metric_key: str,
    model_key: str,
    model_text_data: dict,
) -> list[int]:
    """Prefetch step: ask each judge which pages it wants, union the results.

    This ensures all judges and all runs see the same page set, eliminating
    page-selection variance from the error bars.
    """
    metric_prompt = load_metric_prompt(metric_key)
    toc_str = _format_toc(model_text_data["combined_toc"])

    display_name = MODELS[model_key]["display_name"]
    company = MODELS[model_key]["company"]
    total_pages = model_text_data["total_pages"]
    num_companions = len(model_text_data["companion_texts"])

    prefetch_system = SYSTEM_PROMPT + """

You are in PREFETCH mode. The document is too large to send in full. Your job is to select which pages are most relevant for evaluating the given metric.

Respond with ONLY a JSON object:
{"action": "request_pages", "pages": [1, 2, 3, 10, 11, 12, ...], "reasoning": "I need pages X-Y for ... and pages Z for ..."}

Request at most 60 pages. Be strategic — use the TOC to identify the most relevant sections for this metric.
"""

    turn1_msg = (
        f"MODEL BEING EVALUATED: {display_name} by {company}\n"
        f"DOCUMENT: System card ({total_pages} pages total"
        f"{f', plus {num_companions} companion report(s)' if num_companions > 0 else ''})\n"
        f"\n{'='*80}\n"
        f"METRIC TO EVALUATE:\n\n{metric_prompt}\n"
        f"\n{'='*80}\n"
        f"TABLE OF CONTENTS:\n\n{toc_str}\n"
        f"\nTotal pages available: {total_pages}\n"
        f"Which pages are most relevant for this metric?"
    )

    all_pages = set()

    for jk in judge_keys:
        # Only prefetch from judges that need agentic mode
        full_text_tokens = _estimate_tokens(
            "".join(p["text"] for p in model_text_data["combined_pages"])
        )
        context_limit = JUDGE_CONTEXT_LIMITS.get(jk, 190_000)
        if full_text_tokens < context_limit - 5_000:
            continue  # this judge gets full text, no prefetch needed

        print(f"      [PREFETCH] {jk}...", end=" ", flush=True)
        messages = [
            {"role": "system", "content": prefetch_system},
            {"role": "user", "content": turn1_msg},
        ]

        try:
            content, _ = _api_call(jk, messages)
            parsed = _extract_json(content)
            pages = parsed.get("pages", [])[:60]
            all_pages.update(pages)
            print(f"{len(pages)} pages requested")
        except Exception as e:
            print(f"ERROR: {e}")
            # Fallback: parse page numbers from raw text
            pages = _parse_page_requests(content if 'content' in dir() else "")
            all_pages.update(pages[:60])

        time.sleep(1)

    # Always include first 5 pages for context (title, intro, TOC)
    all_pages.update(range(1, min(6, model_text_data["total_pages"] + 1)))

    return sorted(all_pages)


def judge_agentic_with_fixed_pages(
    judge_key: str,
    metric_key: str,
    model_key: str,
    model_text_data: dict,
    fixed_pages: list[int],
) -> dict:
    """Score a large document using a pre-determined set of pages (single shot)."""
    metric_prompt = load_metric_prompt(metric_key)
    page_text = _get_pages_by_range(model_text_data["combined_pages"], fixed_pages)

    display_name = MODELS[model_key]["display_name"]
    company = MODELS[model_key]["company"]
    total_pages = model_text_data["total_pages"]
    num_companions = len(model_text_data["companion_texts"])

    user_msg = (
        f"MODEL BEING EVALUATED: {display_name} by {company}\n"
        f"DOCUMENT: System card ({total_pages} pages total"
        f"{f', plus {num_companions} companion report(s)' if num_companions > 0 else ''})\n"
        f"NOTE: You are seeing {len(fixed_pages)} of {total_pages} pages, selected as the most relevant sections for this metric.\n"
        f"\n{'='*80}\n"
        f"METRIC TO EVALUATE:\n\n{metric_prompt}\n"
        f"\n{'='*80}\n"
        f"DOCUMENT TEXT:\n\n{page_text}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    content, usage = _api_call(judge_key, messages)
    parsed = _extract_json(content)
    parsed["_mode"] = "agentic_fixed_pages"
    parsed["_pages_read"] = fixed_pages
    parsed["_usage"] = usage
    return parsed


# ── Main dispatch ──────────────────────────────────────────────────────

def _needs_agentic(model_text_data: dict, judge_key: str) -> bool:
    """Check if this model+judge combo needs agentic mode."""
    full_text_tokens = _estimate_tokens(
        "".join(p["text"] for p in model_text_data["combined_pages"])
    )
    context_limit = JUDGE_CONTEXT_LIMITS.get(judge_key, 190_000)
    return full_text_tokens >= context_limit - 5_000


def judge_model_metric(
    judge_key: str,
    metric_key: str,
    model_key: str,
    model_text_data: dict,
    fixed_pages: list[int] | None = None,
) -> dict:
    """Score a model on a metric — dispatches to single-shot or fixed-page mode."""
    if not _needs_agentic(model_text_data, judge_key):
        return judge_single_shot(judge_key, metric_key, model_key, model_text_data)
    elif fixed_pages:
        return judge_agentic_with_fixed_pages(
            judge_key, metric_key, model_key, model_text_data, fixed_pages,
        )
    else:
        # Shouldn't happen — prefetch should have been done
        raise RuntimeError(
            f"Agentic mode needed for {model_key}+{judge_key} but no fixed_pages provided. "
            "Run prefetch first."
        )


def run_eval(
    model_keys: list[str] | None = None,
    metric_keys: list[str] | None = None,
    judge_keys: list[str] | None = None,
    dry_run: bool = False,
):
    """Run the full evaluation pipeline."""
    model_keys = model_keys or list(MODELS.keys())
    metric_keys = metric_keys or list(METRICS.keys())
    judge_keys = judge_keys or list(JUDGES.keys())

    raw_dir = RESULTS_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_calls = len(model_keys) * len(metric_keys) * len(judge_keys) * RUNS_PER_JUDGE
    print(f"Total API calls planned: {total_calls}")
    print(f"Models: {len(model_keys)}, Metrics: {len(metric_keys)}, "
          f"Judges: {len(judge_keys)}, Runs/judge: {RUNS_PER_JUDGE}")
    print()

    if dry_run:
        print("[DRY RUN] Would make the following calls:")
        for mk in model_keys:
            for metric in metric_keys:
                for jk in judge_keys:
                    for run in range(RUNS_PER_JUDGE):
                        print(f"  {MODELS[mk]['display_name']} × {metric} × {jk} (run {run+1})")
        return

    call_count = 0

    for mk in model_keys:
        print(f"\n{'='*60}")
        print(f"Model: {MODELS[mk]['display_name']} ({MODELS[mk]['company']})")
        print(f"{'='*60}")

        # Extract text once per model
        model_text_data = build_model_text(mk)
        print(f"  Total pages: {model_text_data['total_pages']}")

        # Check if any judge needs agentic mode for this model
        needs_agentic = any(_needs_agentic(model_text_data, jk) for jk in judge_keys)

        # Prefetch: collect page sets per metric (shared across all judges & runs)
        # Cache key: (model_key, metric_key) -> fixed_pages
        prefetch_cache = {}

        if needs_agentic:
            print(f"\n  [PREFETCH] Large document — collecting page requests per metric (parallel)...")
            prefetch_dir = RESULTS_DIR / "prefetch"
            prefetch_dir.mkdir(parents=True, exist_ok=True)

            # Check which metrics need prefetching (not cached)
            prefetch_tasks = []
            for metric in metric_keys:
                prefetch_path = prefetch_dir / f"{mk}__{metric}__pages.json"
                if prefetch_path.exists():
                    fixed_pages = json.loads(prefetch_path.read_text())
                    prefetch_cache[metric] = fixed_pages
                    print(f"    {metric}: {len(fixed_pages)} pages (cached)")
                else:
                    prefetch_tasks.append((metric, prefetch_path))

            if prefetch_tasks:
                def _do_prefetch(args):
                    metric, ppath = args
                    pages = _prefetch_pages(judge_keys, metric, mk, model_text_data)
                    ppath.write_text(json.dumps(pages))
                    return metric, pages

                with ThreadPoolExecutor(max_workers=50) as executor:
                    futures = {executor.submit(_do_prefetch, t): t for t in prefetch_tasks}
                    for future in as_completed(futures):
                        metric, pages = future.result()
                        prefetch_cache[metric] = pages
                        print(f"    {metric}: {len(pages)} pages (prefetched)")

        # Build all tasks for this model
        tasks = []
        for metric in metric_keys:
            fixed_pages = prefetch_cache.get(metric)
            for jk in judge_keys:
                for run_idx in range(RUNS_PER_JUDGE):
                    result_path = raw_dir / f"{mk}__{metric}__{jk}__run{run_idx}.json"
                    if result_path.exists():
                        call_count += 1
                        print(f"    [{call_count}/{total_calls}] {mk} × {metric} × {jk} run {run_idx+1} — CACHED")
                        continue
                    tasks.append((metric, jk, run_idx, fixed_pages, result_path))

        if not tasks:
            print("  All results cached, skipping.")
            continue

        print(f"\n  Running {len(tasks)} API calls in parallel (max 50 workers)...")

        def _run_task(task_args):
            metric, jk, run_idx, fixed_pages, result_path = task_args
            try:
                result = judge_model_metric(
                    jk, metric, mk, model_text_data, fixed_pages,
                )
            except Exception as e:
                result = {"_error": str(e), "_judge": jk}

            result["_judge"] = jk
            result["_model_id"] = JUDGES[jk]["model"]
            result["_model_key"] = mk
            result["_metric"] = metric
            result["_run_idx"] = run_idx
            result_path.write_text(json.dumps(result, indent=2))
            return metric, jk, run_idx, result

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = {executor.submit(_run_task, t): t for t in tasks}
            for future in as_completed(futures):
                call_count += 1
                metric, jk, run_idx, result = future.result()
                score = result.get("score")
                mode = result.get("_mode", "?")
                if score is not None:
                    label = f"score={score}"
                elif "count_present" in result:
                    label = f"count={result['count_present']}"
                elif "count" in result:
                    label = f"count={result['count']}"
                else:
                    label = f"ERROR: {result.get('_error', 'unknown')[:60]}"
                print(f"    [{call_count}/{total_calls}] {metric} × {jk} run {run_idx+1} → {label} ({mode})")

    print(f"\n\nDone! {call_count} calls completed. Results in {raw_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run system card evaluation")
    parser.add_argument("--models", nargs="+", help="Model keys to evaluate")
    parser.add_argument("--metrics", nargs="+", help="Metric keys to evaluate")
    parser.add_argument("--judges", nargs="+", help="Judge keys to use")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without calling APIs")
    args = parser.parse_args()

    run_eval(
        model_keys=args.models,
        metric_keys=args.metrics,
        judge_keys=args.judges,
        dry_run=args.dry_run,
    )
