"""Extract text from system card PDFs and markdown files.

For large PDFs, also extracts a table of contents (section headings + page numbers)
so we can send only relevant sections to judges per metric.
"""

import json
import re
from pathlib import Path

import pymupdf
import pymupdf4llm

from config import MODELS, SYSTEM_CARDS_DIR, COMPANION_DIR, RESULTS_DIR


def extract_pdf_text(pdf_path: Path) -> dict:
    """Extract full text and TOC from a PDF using pymupdf4llm for proper table extraction.

    Returns:
        {
            "full_text": str,
            "pages": [{"page": 1, "text": "..."}, ...],
            "toc": [{"level": 1, "title": "Introduction", "page": 1}, ...],
            "total_pages": int,
        }
    """
    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)

    # Extract full markdown using pymupdf4llm (preserves tables, headings, lists)
    # page_chunks=True returns a list of dicts with per-page text
    page_chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
    )

    pages = []
    full_text_parts = []
    for i, chunk in enumerate(page_chunks):
        md_text = chunk["text"]
        pages.append({"page": i + 1, "text": md_text})
        full_text_parts.append(f"\n--- PAGE {i + 1} ---\n{md_text}")

    # Extract TOC from PDF metadata
    toc_entries = doc.get_toc()  # [[level, title, page], ...]
    toc = [{"level": lvl, "title": title, "page": pg} for lvl, title, pg in toc_entries]

    # If no TOC in metadata, try to infer headings from markdown
    if not toc:
        toc = _infer_headings(pages)

    doc.close()

    return {
        "full_text": "\n".join(full_text_parts),
        "pages": pages,
        "toc": toc,
        "total_pages": total_pages,
    }


def _infer_headings(pages: list[dict]) -> list[dict]:
    """Heuristic: lines that are short, title-cased, and followed by longer text are headings."""
    headings = []
    # Common section header patterns
    patterns = [
        re.compile(r"^(\d+\.?\s+[A-Z][A-Za-z\s&:,/-]+)$", re.MULTILINE),  # "1. Introduction"
        re.compile(r"^([A-Z][A-Za-z\s&:,/-]{3,60})$", re.MULTILINE),  # "Executive Summary"
    ]

    for page_info in pages:
        for pattern in patterns:
            for match in pattern.finditer(page_info["text"]):
                title = match.group(1).strip()
                if len(title) > 3 and not title.isupper():  # skip ALL-CAPS lines (likely headers/footers)
                    headings.append({
                        "level": 1,
                        "title": title,
                        "page": page_info["page"],
                    })
    return headings


def extract_md_text(md_path: Path) -> dict:
    """Extract text and headings from a markdown file."""
    text = md_path.read_text(encoding="utf-8")

    # Extract headings
    toc = []
    for match in re.finditer(r"^(#{1,4})\s+(.+)$", text, re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        toc.append({"level": level, "title": title, "page": None})

    return {
        "full_text": text,
        "pages": [{"page": 1, "text": text}],
        "toc": toc,
        "total_pages": 1,
    }


def get_relevant_pages(
    toc: list[dict],
    pages: list[dict],
    metric: str,
    max_tokens: int | None = None,
) -> str:
    """Given a TOC and metric name, return text from the most relevant sections.

    By default, sends the FULL document text. Only falls back to keyword-based
    section selection if max_tokens is set and the full text would exceed it.
    """
    full_text = "\n".join(f"--- PAGE {p['page']} ---\n{p['text']}" for p in pages)
    estimated_tokens = len(full_text) // 4  # rough char-to-token estimate

    # If full text fits within the judge's context, send everything
    if max_tokens is None or estimated_tokens < max_tokens - 5000:  # 5K buffer for prompt
        return full_text

    # Keywords that map metrics to likely relevant sections
    METRIC_KEYWORDS = {
        "topic_coverage": None,  # needs full document
        "dangerous_capability_reporting": [
            "dangerous", "capability", "cbrn", "chemical", "biological", "nuclear",
            "cyber", "autonomous", "replication", "persuasion", "manipulation",
            "weapon", "uplift", "dual-use", "misuse",
        ],
        "alignment_controllability": [
            "alignment", "controllability", "refusal", "jailbreak", "adversarial",
            "instruction", "hierarchy", "robustness", "safety", "guardrail",
            "red-team", "prompt injection",
        ],
        "risk_category_breadth": None,  # needs full document
        "stakeholder_diversity": None,  # needs full document
        "evidence_sufficiency": None,  # needs full document
        "eval_reporting_quality": [
            "evaluation", "benchmark", "eval", "methodology", "dataset",
            "results", "performance", "testing", "metric",
        ],
        "reasoning_depth": None,  # needs full document
        "limitation_specificity": [
            "limitation", "failure", "weakness", "shortcoming", "challenge",
            "struggle", "degrade", "error", "known issue",
        ],
        "reasoning_consistency": None,  # needs full document
        "external_validator_count": [
            "external", "third-party", "audit", "red-team", "partner",
            "independent", "organization", "METR", "Apollo",
        ],
        "eval_type_diversity": [
            "evaluation", "benchmark", "red-team", "audit", "bug bounty",
            "adversarial", "user study", "expert review", "academic",
        ],
        "post_deployment_monitoring": [
            "deployment", "monitoring", "incident", "response", "update",
            "rollback", "feedback", "post-deployment", "post-launch",
        ],
    }

    keywords = METRIC_KEYWORDS.get(metric)
    if keywords is None:
        # Needs full document — return all text
        return "\n".join(f"--- PAGE {p['page']} ---\n{p['text']}" for p in pages)

    # Score each page by keyword hits
    scored_pages = []
    for page in pages:
        text_lower = page["text"].lower()
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        scored_pages.append((hits, page))

    # Include pages with at least 1 hit, plus first 3 pages for context
    relevant = set(range(3))  # always include first 3 pages
    for i, (hits, _) in enumerate(scored_pages):
        if hits > 0:
            relevant.add(i)
            # Also include adjacent pages for context
            if i > 0:
                relevant.add(i - 1)
            if i < len(scored_pages) - 1:
                relevant.add(i + 1)

    selected = [scored_pages[i][1] for i in sorted(relevant)]

    if len(selected) < 5:
        # Too few pages found — fall back to full text
        return "\n".join(f"--- PAGE {p['page']} ---\n{p['text']}" for p in pages)

    return "\n".join(
        f"--- PAGE {p['page']} ---\n{p['text']}" for p in selected
    )


def build_model_text(model_key: str) -> dict:
    """Build the full text for a model by combining its system card + companion reports.

    Returns:
        {
            "card_text": extracted card data,
            "companion_texts": [extracted companion data, ...],
            "combined_full_text": str,
            "combined_pages": [...],
            "combined_toc": [...],
            "total_pages": int,
        }
    """
    model = MODELS[model_key]

    # Extract main card
    card_path = SYSTEM_CARDS_DIR / model["card_file"]
    if card_path.suffix == ".md":
        card_data = extract_md_text(card_path)
    else:
        card_data = extract_pdf_text(card_path)

    # Extract companion reports
    companion_data_list = []
    page_offset = card_data["total_pages"]

    for comp_file in model.get("companions", []):
        comp_path = COMPANION_DIR / comp_file
        comp_data = extract_pdf_text(comp_path)
        # Offset page numbers
        for p in comp_data["pages"]:
            p["page"] += page_offset
        for t in comp_data["toc"]:
            t["page"] += page_offset
        page_offset += comp_data["total_pages"]
        companion_data_list.append(comp_data)

    # Combine
    combined_text_parts = [
        f"=== SYSTEM CARD: {model['display_name']} ===\n{card_data['full_text']}"
    ]
    combined_pages = list(card_data["pages"])
    combined_toc = list(card_data["toc"])

    for i, comp_data in enumerate(companion_data_list):
        comp_file = model["companions"][i]
        combined_text_parts.append(
            f"\n=== COMPANION REPORT: {comp_file} ===\n{comp_data['full_text']}"
        )
        combined_pages.extend(comp_data["pages"])
        combined_toc.extend(comp_data["toc"])

    return {
        "card_text": card_data,
        "companion_texts": companion_data_list,
        "combined_full_text": "\n".join(combined_text_parts),
        "combined_pages": combined_pages,
        "combined_toc": combined_toc,
        "total_pages": page_offset,
    }


def extract_all():
    """Extract text from all system cards and save to results/extracted/."""
    out_dir = RESULTS_DIR / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_key in MODELS:
        print(f"Extracting: {model_key}...")
        data = build_model_text(model_key)

        # Save metadata (not full text — too large for JSON)
        meta = {
            "model_key": model_key,
            "display_name": MODELS[model_key]["display_name"],
            "company": MODELS[model_key]["company"],
            "total_pages": data["total_pages"],
            "toc": data["combined_toc"],
            "num_companions": len(data["companion_texts"]),
        }
        meta_path = out_dir / f"{model_key}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        # Save full text
        text_path = out_dir / f"{model_key}_full.txt"
        text_path.write_text(data["combined_full_text"])

        print(f"  → {data['total_pages']} pages, TOC entries: {len(data['combined_toc'])}")

    print("\nDone! Extracted text saved to results/extracted/")


if __name__ == "__main__":
    extract_all()
