"""Aggregate raw judge results into scores, agreement metrics, and bias analysis."""

import json
import os
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import METRICS, MODELS, JUDGES, DIMENSIONS, ALL_DIMENSIONS, RESULTS_DIR


# ── Judge-to-company mapping (for bias analysis) ──────────────────────
JUDGE_COMPANY = {
    "sonnet_4_6": "Anthropic",
    "gpt_5_4": "OpenAI",
    "gemini_3_1_pro": "Google",
}


def load_raw_results() -> list[dict]:
    """Load all raw JSON results from results/raw/."""
    raw_dir = RESULTS_DIR / "raw"
    results = []
    for f in sorted(os.listdir(raw_dir)):
        if not f.endswith(".json"):
            continue
        data = json.loads((raw_dir / f).read_text())
        results.append(data)
    return results


def extract_score(result: dict) -> float | None:
    """Extract the numeric score from a raw result."""
    if "_error" in result:
        return None
    # Rubric metrics return "score" directly
    if "score" in result and result["score"] is not None:
        return float(result["score"])
    # Extractive metrics with score field
    if "count_present" in result:
        return float(result.get("score", 0))
    # external_validator_count returns "count" — needs post-hoc normalization
    if result.get("_metric") == "external_validator_count" and "count" in result:
        return float(result["count"])  # raw count, normalized later
    return None


def aggregate_scores(results: list[dict]) -> dict:
    """Aggregate raw results into per-model, per-metric scores.

    Returns:
        {
            model_key: {
                metric_key: {
                    "scores": [float, ...],  # all 6 data points
                    "per_judge": {judge_key: [float, ...]},
                    "mean": float,
                    "median": float,
                    "std": float,
                }
            }
        }
    """
    # Collect scores: model -> metric -> judge -> [scores]
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for r in results:
        mk = r.get("_model_key")
        metric = r.get("_metric")
        jk = r.get("_judge")
        if not mk or not metric or not jk:
            continue
        if mk not in MODELS:
            continue
        score = extract_score(r)
        if score is not None:
            raw[mk][metric][jk].append(score)

    # Normalize external_validator_count by max observed
    max_ext_count = 0
    for mk in raw:
        for jk_scores in raw[mk].get("external_validator_count", {}).values():
            for s in jk_scores:
                max_ext_count = max(max_ext_count, s)

    if max_ext_count > 0:
        for mk in raw:
            if "external_validator_count" in raw[mk]:
                for jk in raw[mk]["external_validator_count"]:
                    raw[mk]["external_validator_count"][jk] = [
                        (s / max_ext_count) * 100
                        for s in raw[mk]["external_validator_count"][jk]
                    ]

    # Build aggregated structure
    agg = {}
    for mk in raw:
        agg[mk] = {}
        for metric in raw[mk]:
            all_scores = []
            per_judge = {}
            for jk in raw[mk][metric]:
                per_judge[jk] = raw[mk][metric][jk]
                all_scores.extend(raw[mk][metric][jk])

            if all_scores:
                agg[mk][metric] = {
                    "scores": all_scores,
                    "per_judge": per_judge,
                    "mean": float(np.mean(all_scores)),
                    "median": float(np.median(all_scores)),
                    "std": float(np.std(all_scores)),
                    "n": len(all_scores),
                }

    return agg


def compute_dimension_scores(agg: dict) -> dict:
    """Compute dimension and overall scores per model.

    Returns:
        {
            model_key: {
                "Comprehensiveness": {"mean": float, "std": float, "metrics": [...]},
                "Reasoning Quality": {...},
                "3rd-party Verification": {...},
                "Overall": {"mean": float, "std": float},
            }
        }
    """
    dim_scores = {}

    for mk in agg:
        dim_scores[mk] = {}

        # Collect metric means from scored dimensions for overall
        all_metric_means = []

        for dim in ALL_DIMENSIONS:
            metric_keys = [k for k, v in METRICS.items() if v["dimension"] == dim]
            metric_means = []
            for metric in metric_keys:
                if metric in agg[mk]:
                    metric_means.append(agg[mk][metric]["mean"])

            if metric_means:
                dim_mean = float(np.mean(metric_means))
                dim_std = float(np.std(metric_means))
                dim_scores[mk][dim] = {
                    "mean": dim_mean,
                    "std": dim_std,
                    "metric_means": metric_means,
                    "n_metrics": len(metric_means),
                }
                # Only include in overall if dimension is in DIMENSIONS (not excluded)
                if dim in DIMENSIONS:
                    all_metric_means.extend(metric_means)

        if all_metric_means:
            # Overall = mean across ALL metrics (not mean of dimensions)
            # This weights each metric equally regardless of which dimension it's in
            # Breadth (5 metrics) gets 5/13 weight, Depth 5/13, 3rd-party 3/13
            dim_scores[mk]["Overall"] = {
                "mean": float(np.mean(all_metric_means)),
                "std": float(np.std(all_metric_means)),
            }

    return dim_scores


def compute_krippendorff_alpha(scores_by_judge: dict) -> float:
    """Compute Krippendorff's alpha for ordinal data across judges.

    Args:
        scores_by_judge: {judge_key: [score_for_model_1, score_for_model_2, ...]}

    Returns alpha (float). Returns NaN if insufficient data.
    """
    judges = list(scores_by_judge.keys())
    if len(judges) < 2:
        return float("nan")

    # Build reliability matrix: judges × models
    n_models = max(len(v) for v in scores_by_judge.values())
    if n_models < 2:
        return float("nan")

    # Collect all values into a matrix
    # Each judge might have different number of scores (due to runs)
    # Average runs per judge first to get one score per judge per model
    # But we don't have model alignment here — this function is called per-metric
    # across all models, so scores_by_judge[jk] = [mean_score_for_each_model]

    values = []
    for jk in judges:
        values.append(scores_by_judge[jk])

    # Pad shorter lists with NaN
    max_len = max(len(v) for v in values)
    padded = []
    for v in values:
        padded.append(v + [float("nan")] * (max_len - len(v)))

    matrix = np.array(padded)  # judges × items

    # Simple ordinal alpha calculation
    n_judges, n_items = matrix.shape

    # Collect all observed values (excluding NaN)
    all_vals = matrix[~np.isnan(matrix)]
    if len(all_vals) < 4:
        return float("nan")

    # Observed disagreement
    Do = 0.0
    n_pairs = 0
    for i in range(n_items):
        col = matrix[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        for a in range(len(valid)):
            for b in range(a + 1, len(valid)):
                Do += (valid[a] - valid[b]) ** 2
                n_pairs += 1

    if n_pairs == 0:
        return float("nan")
    Do /= n_pairs

    # Expected disagreement
    De = 0.0
    n_total = len(all_vals)
    n_exp_pairs = 0
    for a in range(n_total):
        for b in range(a + 1, n_total):
            De += (all_vals[a] - all_vals[b]) ** 2
            n_exp_pairs += 1

    if n_exp_pairs == 0:
        return float("nan")
    De /= n_exp_pairs

    if De == 0:
        return 1.0  # perfect agreement

    alpha = 1.0 - Do / De
    return float(alpha)


def compute_agreement(agg: dict) -> dict:
    """Compute Krippendorff's alpha per metric across all models.

    Returns:
        {metric_key: {"alpha": float, "reliable": bool}}
    """
    agreement = {}
    model_keys = sorted(agg.keys())

    for metric in METRICS:
        # For each judge, collect mean score per model
        scores_by_judge = defaultdict(list)
        for mk in model_keys:
            if metric not in agg[mk]:
                continue
            per_judge = agg[mk][metric]["per_judge"]
            for jk in per_judge:
                # Average runs for this judge on this model
                scores_by_judge[jk].append(float(np.mean(per_judge[jk])))

        alpha = compute_krippendorff_alpha(dict(scores_by_judge))
        agreement[metric] = {
            "alpha": round(alpha, 3) if not np.isnan(alpha) else None,
            "reliable": alpha >= 0.4 if not np.isnan(alpha) else False,
            "n_judges": len(scores_by_judge),
            "n_models": len(model_keys),
        }

    return agreement


def compute_bias_analysis(agg: dict) -> dict:
    """Check if judges systematically inflate their own company's scores.

    For each judge, compare:
    - Mean score given to OWN company's models
    - Mean score given to OTHER companies' models

    Returns:
        {
            judge_key: {
                "own_company": str,
                "mean_own": float,
                "mean_other": float,
                "delta": float,  # positive = inflates own
                "per_metric": {metric: {"own": float, "other": float, "delta": float}}
            }
        }
    """
    bias = {}

    for jk, judge_company in JUDGE_COMPANY.items():
        own_scores = []
        other_scores = []
        per_metric = {}

        for metric in METRICS:
            own_metric = []
            other_metric = []

            for mk in agg:
                if metric not in agg[mk]:
                    continue
                per_judge = agg[mk][metric].get("per_judge", {})
                if jk not in per_judge:
                    continue

                model_company = MODELS[mk]["company"]
                judge_scores = per_judge[jk]
                mean_score = float(np.mean(judge_scores))

                if model_company == judge_company:
                    own_scores.append(mean_score)
                    own_metric.append(mean_score)
                else:
                    other_scores.append(mean_score)
                    other_metric.append(mean_score)

            if own_metric and other_metric:
                per_metric[metric] = {
                    "own": round(float(np.mean(own_metric)), 1),
                    "other": round(float(np.mean(other_metric)), 1),
                    "delta": round(float(np.mean(own_metric)) - float(np.mean(other_metric)), 1),
                }

        mean_own = float(np.mean(own_scores)) if own_scores else 0
        mean_other = float(np.mean(other_scores)) if other_scores else 0

        bias[jk] = {
            "own_company": judge_company,
            "mean_own": round(mean_own, 1),
            "mean_other": round(mean_other, 1),
            "delta": round(mean_own - mean_other, 1),
            "n_own": len(own_scores),
            "n_other": len(other_scores),
            "per_metric": per_metric,
        }

    return bias


def save_results(agg: dict, dim_scores: dict, agreement: dict, bias: dict):
    """Save all aggregated results."""
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── scores.json ──
    scores_out = {}
    for mk in agg:
        scores_out[mk] = {
            "display_name": MODELS[mk]["display_name"],
            "company": MODELS[mk]["company"],
            "metrics": {},
        }
        for metric in agg[mk]:
            m = agg[mk][metric]
            scores_out[mk]["metrics"][metric] = {
                "mean": round(m["mean"], 1),
                "median": round(m["median"], 1),
                "std": round(m["std"], 1),
                "n": m["n"],
                "per_judge": {
                    jk: [round(s, 1) for s in scores]
                    for jk, scores in m["per_judge"].items()
                },
            }
        if mk in dim_scores:
            scores_out[mk]["dimensions"] = {}
            for dim in DIMENSIONS + ["Overall"]:
                if dim in dim_scores[mk]:
                    scores_out[mk]["dimensions"][dim] = {
                        "mean": round(dim_scores[mk][dim]["mean"], 1),
                    }

    (out_dir / "scores.json").write_text(json.dumps(scores_out, indent=2))

    # ── summary.csv ──
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["model", "company"]
        for metric in METRICS:
            header.extend([f"{metric}_mean", f"{metric}_std"])
        for dim in DIMENSIONS:
            header.append(f"{dim}_mean")
        header.append("Overall_mean")
        writer.writerow(header)

        for mk in sorted(agg.keys()):
            row = [MODELS[mk]["display_name"], MODELS[mk]["company"]]
            for metric in METRICS:
                if metric in agg[mk]:
                    row.extend([round(agg[mk][metric]["mean"], 1), round(agg[mk][metric]["std"], 1)])
                else:
                    row.extend(["", ""])
            for dim in DIMENSIONS:
                if mk in dim_scores and dim in dim_scores[mk]:
                    row.append(round(dim_scores[mk][dim]["mean"], 1))
                else:
                    row.append("")
            if mk in dim_scores and "Overall" in dim_scores[mk]:
                row.append(round(dim_scores[mk]["Overall"]["mean"], 1))
            else:
                row.append("")
            writer.writerow(row)

    # ── agreement.csv ──
    with open(out_dir / "agreement.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "dimension", "krippendorff_alpha", "reliable", "n_judges", "n_models"])
        for metric in METRICS:
            if metric in agreement:
                a = agreement[metric]
                writer.writerow([
                    metric,
                    METRICS[metric]["dimension"],
                    a["alpha"],
                    a["reliable"],
                    a["n_judges"],
                    a["n_models"],
                ])

    # ── bias_analysis.json ──
    (out_dir / "bias_analysis.json").write_text(json.dumps(bias, indent=2))

    # ── Print summary ──
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Per-model dimension scores
    print(f"\n{'Model':<25s} {'Breadth':>8s} {'Depth':>13s} {'3rd-party':>10s} {'Overall':>8s}")
    print("-" * 70)
    for mk in sorted(dim_scores.keys(), key=lambda k: dim_scores[k].get("Overall", {}).get("mean", 0), reverse=True):
        name = MODELS[mk]["display_name"]
        b = dim_scores[mk].get("Comprehensiveness", {}).get("mean", 0)
        t = dim_scores[mk].get("Reasoning Quality", {}).get("mean", 0)
        v = dim_scores[mk].get("3rd-party Verification", {}).get("mean", 0)
        o = dim_scores[mk].get("Overall", {}).get("mean", 0)
        print(f"{name:<25s} {b:>8.1f} {t:>13.1f} {v:>10.1f} {o:>8.1f}")

    # Agreement
    print(f"\n{'Metric':<40s} {'Alpha':>6s} {'Reliable':>9s}")
    print("-" * 60)
    for metric in METRICS:
        if metric in agreement:
            a = agreement[metric]
            alpha_str = f"{a['alpha']:.3f}" if a["alpha"] is not None else "N/A"
            rel = "YES" if a["reliable"] else "NO"
            print(f"{metric:<40s} {alpha_str:>6s} {rel:>9s}")

    # Bias
    print(f"\n{'Judge':<20s} {'Company':>10s} {'Own':>6s} {'Other':>6s} {'Delta':>6s}")
    print("-" * 55)
    for jk in bias:
        b = bias[jk]
        print(f"{jk:<20s} {b['own_company']:>10s} {b['mean_own']:>6.1f} {b['mean_other']:>6.1f} {b['delta']:>+6.1f}")


def main():
    print("Loading raw results...")
    results = load_raw_results()
    print(f"  Loaded {len(results)} results")

    print("Aggregating scores...")
    agg = aggregate_scores(results)
    print(f"  {len(agg)} models, {sum(len(v) for v in agg.values())} model×metric pairs")

    print("Computing dimension scores...")
    dim_scores = compute_dimension_scores(agg)

    print("Computing inter-annotator agreement...")
    agreement = compute_agreement(agg)

    print("Computing self-evaluation bias analysis...")
    bias = compute_bias_analysis(agg)

    print("Saving results...")
    save_results(agg, dim_scores, agreement, bias)

    print("\nDone!")


if __name__ == "__main__":
    main()
