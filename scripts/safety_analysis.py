"""Safety-researcher-focused analyses."""

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import METRICS, MODELS, DIMENSIONS, RESULTS_DIR

COMPANY_COLORS = {
    "Anthropic": "#D97706",
    "OpenAI": "#10B981",
    "Google": "#3B82F6",
}


def load_data():
    scores = json.loads((RESULTS_DIR / "scores.json").read_text())
    raw_results = []
    for f in sorted(os.listdir(RESULTS_DIR / "raw")):
        if f.endswith(".json"):
            data = json.loads((RESULTS_DIR / "raw" / f).read_text())
            if "_error" not in data:
                raw_results.append(data)
    return scores, raw_results


# ── 1. Breadth vs Depth tradeoff ──────────────────────────────────────
def plot_breadth_vs_depth(scores):
    """Do cards that cover more topics also cover them well?"""
    fig, ax = plt.subplots(figsize=(10, 8))

    for mk, data in scores.items():
        breadth = data.get("dimensions", {}).get("Comprehensiveness", {}).get("mean", 0)
        transparency = data.get("dimensions", {}).get("Transparency", {}).get("mean", 0)
        company = data["company"]

        # Compute stderr
        b_metrics = [data["metrics"][m]["mean"] for m in data["metrics"]
                     if METRICS.get(m, {}).get("dimension") == "Comprehensiveness"]
        t_metrics = [data["metrics"][m]["mean"] for m in data["metrics"]
                     if METRICS.get(m, {}).get("dimension") == "Transparency"]
        b_se = np.std(b_metrics) / np.sqrt(len(b_metrics)) if len(b_metrics) > 1 else 0
        t_se = np.std(t_metrics) / np.sqrt(len(t_metrics)) if len(t_metrics) > 1 else 0

        ax.errorbar(breadth, transparency, xerr=b_se, yerr=t_se,
                    color=COMPANY_COLORS[company], fmt="o",
                    markersize=10, zorder=5, markeredgecolor="white", markeredgewidth=0.5,
                    capsize=3, elinewidth=1)
        ax.annotate(data["display_name"], (breadth, transparency),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, color=COMPANY_COLORS[company])

    # Diagonal line: breadth = depth
    ax.plot([0, 100], [0, 100], "--", color="gray", alpha=0.3, linewidth=1)
    ax.text(85, 78, "Breadth = Depth", fontsize=8, color="gray", alpha=0.5, rotation=35)

    # Quadrant labels
    ax.text(30, 85, "Narrow but deep", ha="center", fontsize=9, color="gray", alpha=0.4)
    ax.text(85, 85, "Broad AND deep", ha="center", fontsize=9, color="gray", alpha=0.4)
    ax.text(30, 25, "Thin card", ha="center", fontsize=9, color="gray", alpha=0.4)
    ax.text(85, 25, "Broad but shallow", ha="center", fontsize=9, color="gray", alpha=0.4)

    ax.axhline(y=55, color="gray", linestyle="--", alpha=0.15)
    ax.axvline(x=55, color="gray", linestyle="--", alpha=0.15)

    ax.set_xlabel("Comprehensiveness Score (what topics are covered)")
    ax.set_ylabel("Transparency Score (how well claims are backed up)")
    ax.set_title("The Breadth vs. Depth Tradeoff\nDo cards that cover more topics also cover them well?")
    ax.set_xlim(20, 100)
    ax.set_ylim(20, 100)
    ax.grid(True, alpha=0.2)

    legend_elements = [Patch(facecolor=c, label=co) for co, c in COMPANY_COLORS.items()]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "safety_breadth_vs_depth.png", dpi=200)
    plt.close()
    print("  Saved safety_breadth_vs_depth.png")


# ── 2. External audit network ─────────────────────────────────────────
def plot_external_audit_network(raw_results):
    """Which external orgs evaluated which models?"""
    # Collect all validators per model
    model_validators = defaultdict(set)
    validator_models = defaultdict(set)

    for r in raw_results:
        if r.get("_metric") != "external_validator_count":
            continue
        mk = r["_model_key"]
        for v in r.get("validators", []):
            name = v.get("name", "").strip()
            if name and len(name) > 1:
                model_validators[mk].add(name)
                validator_models[name].add(MODELS[mk]["display_name"])

    # Only keep validators that appear in at least 1 model
    # Sort by how many models they evaluated
    sorted_validators = sorted(validator_models.items(), key=lambda x: len(x[1]), reverse=True)

    if not sorted_validators:
        print("  No external validators found, skipping.")
        return

    # Build heatmap: validators × models
    top_validators = [v[0] for v in sorted_validators[:20]]  # top 20
    model_keys_sorted = sorted(MODELS.keys(), key=lambda mk: MODELS[mk]["date"])

    matrix = np.zeros((len(top_validators), len(model_keys_sorted)))
    for i, validator in enumerate(top_validators):
        for j, mk in enumerate(model_keys_sorted):
            if validator in model_validators[mk]:
                matrix[i, j] = 1

    fig, ax = plt.subplots(figsize=(14, max(6, len(top_validators) * 0.4)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(model_keys_sorted)))
    ax.set_xticklabels([MODELS[mk]["display_name"] for mk in model_keys_sorted],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_validators)))
    ax.set_yticklabels(top_validators, fontsize=8)

    # Color x-axis labels by company
    for i, mk in enumerate(model_keys_sorted):
        color = COMPANY_COLORS[MODELS[mk]["company"]]
        ax.get_xticklabels()[i].set_color(color)

    ax.set_title("Who Audits Whom?\nExternal organizations that evaluated each model")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "safety_audit_network.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved safety_audit_network.png")


# ── 3. Safety metric deep dive ────────────────────────────────────────
def plot_safety_metrics_comparison(scores):
    """Compare the 4 most safety-relevant metrics across all models."""
    safety_metrics = [
        "dangerous_capability_reporting",
        "alignment_controllability",
        "evidence_sufficiency",
        "reasoning_consistency",
    ]
    metric_labels = [
        "Dangerous\nCapability\nReporting",
        "Alignment &\nControllability",
        "Evidence\nSufficiency",
        "Reasoning\nConsistency",
    ]

    # Sort models by overall
    model_keys = sorted(
        scores.keys(),
        key=lambda mk: scores[mk].get("dimensions", {}).get("Overall", {}).get("mean", 0),
        reverse=True,
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 8), sharey=True)

    for idx, (metric, label) in enumerate(zip(safety_metrics, metric_labels)):
        ax = axes[idx]
        names = []
        means = []
        stderrs = []
        colors = []

        for mk in model_keys:
            data = scores[mk]
            m = data["metrics"].get(metric, {})
            if not m:
                continue
            names.append(data["display_name"])
            means.append(m["mean"])
            n = m.get("n", 6)
            stderrs.append(m["std"] / np.sqrt(n) if n > 0 else 0)
            colors.append(COMPANY_COLORS[data["company"]])

        y_pos = np.arange(len(names))
        ax.barh(y_pos, means, xerr=stderrs, color=colors, alpha=0.8, height=0.7,
                capsize=3, error_kw={"linewidth": 1})
        ax.set_yticks(y_pos)
        if idx == 0:
            ax.set_yticklabels(names, fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_xlim(0, 105)
        ax.set_title(label, fontsize=11)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.2)

    fig.suptitle("Safety-Critical Metrics Comparison", fontsize=16, y=1.02)

    legend_elements = [Patch(facecolor=c, label=co) for co, c in COMPANY_COLORS.items()]
    axes[-1].legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "safety_metrics_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved safety_metrics_comparison.png")


def main():
    print("Loading data...")
    scores, raw_results = load_data()

    print("\nGenerating safety-focused analyses...")
    plot_breadth_vs_depth(scores)
    plot_external_audit_network(raw_results)
    plot_safety_metrics_comparison(scores)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
