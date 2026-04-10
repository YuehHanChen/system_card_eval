"""Final publication-quality plots."""

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from adjustText import adjust_text

from config import METRICS, MODELS, DIMENSIONS, ALL_DIMENSIONS, RESULTS_DIR, TOPIC_CHECKLIST

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COMPANY_COLORS = {"Anthropic": "#D97706", "OpenAI": "#10B981", "Google": "#3B82F6"}
COMPANY_MARKERS = {"Anthropic": "o", "OpenAI": "s", "Google": "D"}

METRIC_LABELS = {
    "topic_coverage": "Topic\nCoverage",
    "dangerous_capability_reporting": "Dangerous\nCapability",
    "alignment_controllability": "Alignment &\nControllability",
    "risk_category_breadth": "Risk Category\nBreadth",
    "stakeholder_diversity": "Stakeholder\nDiversity",
    "evidence_sufficiency": "Evidence\nSufficiency",
    "eval_reporting_quality": "Eval Reporting\nQuality",
    "reasoning_depth": "Reasoning\nDepth",
    "limitation_specificity": "Limitation\nSpecificity",
    "reasoning_consistency": "Reasoning\nConsistency",
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


def get_overall_stderr(scores, mk):
    metric_means = [scores[mk]["metrics"][m]["mean"] for m in scores[mk]["metrics"]
                    if METRICS.get(m, {}).get("dimension") in DIMENSIONS]
    if len(metric_means) < 2:
        return 0
    return float(np.std(metric_means) / np.sqrt(len(metric_means)))


# ── 1. REPORT CARD HEATMAP ────────────────────────────────────────────
def plot_report_card(scores):
    metric_keys = [k for k, v in METRICS.items() if v["dimension"] in DIMENSIONS]
    model_keys = sorted(
        scores.keys(),
        key=lambda mk: scores[mk].get("dimensions", {}).get("Overall", {}).get("mean", 0),
        reverse=True,
    )

    matrix = np.array([
        [scores[mk]["metrics"].get(m, {}).get("mean", 0) for m in metric_keys]
        for mk in model_keys
    ])
    overalls = [scores[mk]["dimensions"]["Overall"]["mean"] for mk in model_keys]

    fig, (ax_main, ax_overall) = plt.subplots(
        1, 2, figsize=(16, 8),
        gridspec_kw={"width_ratios": [len(metric_keys), 1.5], "wspace": 0.05},
    )

    im = ax_main.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax_main.set_xticks(range(len(metric_keys)))
    ax_main.set_xticklabels(
        [METRIC_LABELS.get(m, m) for m in metric_keys],
        fontsize=7, rotation=0, ha="center",
    )
    ax_main.set_yticks(range(len(model_keys)))
    ax_main.set_yticklabels(
        [scores[mk]["display_name"] for mk in model_keys], fontsize=10,
    )
    for i, mk in enumerate(model_keys):
        ax_main.get_yticklabels()[i].set_color(COMPANY_COLORS[scores[mk]["company"]])
        ax_main.get_yticklabels()[i].set_fontweight("bold")

    ax_main.axvline(x=4.5, color="white", linewidth=3)

    for i in range(len(model_keys)):
        for j in range(len(metric_keys)):
            val = matrix[i, j]
            color = "white" if val < 35 or val > 80 else "black"
            ax_main.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    ax_main.text(2, -0.8, "COMPREHENSIVENESS", ha="center", fontsize=10, fontweight="bold", color="#666")
    ax_main.text(7, -0.8, "REASONING QUALITY", ha="center", fontsize=10, fontweight="bold", color="#666")

    im2 = ax_overall.imshow(np.array(overalls).reshape(-1, 1), cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax_overall.set_xticks([0])
    ax_overall.set_xticklabels(["OVERALL"], fontsize=10, fontweight="bold")
    ax_overall.set_yticks([])
    stderrs = [get_overall_stderr(scores, mk) for mk in model_keys]
    for i, (val, se) in enumerate(zip(overalls, stderrs)):
        color = "white" if val < 35 or val > 80 else "black"
        ax_overall.text(0, i, f"{val:.1f}±{se:.1f}", ha="center", va="center",
                       fontsize=10, color=color, fontweight="bold")

    plt.colorbar(im, ax=[ax_main, ax_overall], label="Score (0-100)", shrink=0.8, pad=0.02)
    fig.suptitle("System Card Eval Score by Metric", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "report_card.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved report_card.png")


# ── 2. OVERALL RANKING ────────────────────────────────────────────────
def plot_overall_ranking(scores):
    model_keys = sorted(
        scores.keys(),
        key=lambda mk: scores[mk].get("dimensions", {}).get("Overall", {}).get("mean", 0),
        reverse=True,
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    names = [scores[mk]["display_name"] for mk in model_keys]
    overall = [scores[mk]["dimensions"]["Overall"]["mean"] for mk in model_keys]
    stderrs = [get_overall_stderr(scores, mk) for mk in model_keys]
    colors = [COMPANY_COLORS[scores[mk]["company"]] for mk in model_keys]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, overall, xerr=stderrs, color=colors, alpha=0.85, height=0.65,
            capsize=3, error_kw={"linewidth": 1, "color": "#666"})

    for i, (v, se) in enumerate(zip(overall, stderrs)):
        ax.text(v + se + 1.5, i, f"{v:.1f}", va="center", fontsize=10, fontweight="bold", color="#333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    for i, mk in enumerate(model_keys):
        ax.get_yticklabels()[i].set_color(COMPANY_COLORS[scores[mk]["company"]])
        ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlim(0, 100)
    ax.set_xlabel("")
    ax.set_title("System Card Score (Comprehensiveness + Reasoning Quality)", fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.2)

    legend_elements = [Patch(facecolor=c, label=co) for co, c in COMPANY_COLORS.items()]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "overall_ranking.png", dpi=200)
    plt.close()
    print("  Saved overall_ranking.png")


# ── 3. TOPIC COVERAGE (heatmap: topics x 3 companies) ─────────────────
def plot_topic_coverage(raw_results):
    topic_by_company = defaultdict(lambda: defaultdict(list))
    for r in raw_results:
        if r.get("_metric") != "topic_coverage":
            continue
        mk = r["_model_key"]
        if mk not in MODELS:
            continue
        company = MODELS[mk]["company"]
        for t in r.get("topics", []):
            topic_by_company[t["topic"]][company].append(1 if t["present"] else 0)

    topics = [t for t in TOPIC_CHECKLIST if t in topic_by_company]
    companies = list(COMPANY_COLORS.keys())

    # Compute means and sort by total coverage
    topic_totals = []
    for topic in topics:
        total = sum(np.mean(topic_by_company[topic].get(co, [0])) for co in companies)
        topic_totals.append((topic, total))
    topic_totals.sort(key=lambda x: x[1], reverse=True)
    topics = [t[0] for t in topic_totals]

    # Build matrix: topics x companies
    matrix = np.zeros((len(topics), len(companies)))
    for i, topic in enumerate(topics):
        for j, company in enumerate(companies):
            vals = topic_by_company[topic].get(company, [0])
            matrix[i, j] = np.mean(vals) * 100

    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.xaxis.tick_top()
    ax.set_xticks(range(len(companies)))
    ax.set_xticklabels(companies, fontsize=12, fontweight="bold")
    # Black company labels
    for j, company in enumerate(companies):
        ax.get_xticklabels()[j].set_color("black")

    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels([t[:50] for t in topics], fontsize=9)

    # Add text annotations
    for i in range(len(topics)):
        for j in range(len(companies)):
            val = matrix[i, j]
            color = "white" if val < 35 or val > 80 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="% of Models Covering Topic", shrink=0.8, pad=0.02)
    ax.set_title("What Do System Cards Actually Cover?\nTopic Coverage by Company", fontweight="bold")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "topic_coverage.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved topic_coverage.png")


# ── 4. SHOW YOUR WORK (cleaned scatter) ───────────────────────────────
def plot_show_your_work(scores):
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot numbered points
    model_list = []
    for mk, data in sorted(scores.items(), key=lambda x: -x[1].get("dimensions", {}).get("Overall", {}).get("mean", 0)):
        if mk not in MODELS:
            continue
        dc_data = data["metrics"].get("dangerous_capability_reporting", {})
        ev_data = data["metrics"].get("evidence_sufficiency", {})
        x = dc_data.get("mean", 0)
        y = ev_data.get("mean", 0)
        xe = dc_data.get("std", 0) / np.sqrt(dc_data.get("n", 6)) if dc_data.get("n", 0) > 0 else 0
        ye = ev_data.get("std", 0) / np.sqrt(ev_data.get("n", 6)) if ev_data.get("n", 0) > 0 else 0
        company = data["company"]
        model_list.append((data["display_name"], company, x, y, xe, ye))

    for i, (name, company, x, y, xe, ye) in enumerate(model_list):
        ax.errorbar(x, y, xerr=xe, yerr=ye,
                    color=COMPANY_COLORS[company], fmt=COMPANY_MARKERS[company],
                    markersize=12, zorder=5, markeredgecolor="white", markeredgewidth=0.5,
                    capsize=3, elinewidth=1, alpha=0.9)
        # Number on the point
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=7,
                color="white", fontweight="bold", zorder=6)

    # Legend table
    legend_lines = []
    for i, (name, company, x, y, _, _) in enumerate(model_list):
        legend_lines.append(f"{i+1}. {name}")
    legend_text = "\n".join(legend_lines)
    ax.text(0.02, 0.45, legend_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#ccc"))

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.12)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.12)
    ax.text(15, 98, "Evidence without\ndanger coverage", ha="center", fontsize=8, color="#bbb")
    ax.text(90, 98, "Transparent", ha="center", fontsize=8, color="#bbb", fontweight="bold")
    ax.text(15, 15, "Opaque", ha="center", fontsize=8, color="#bbb", fontweight="bold")
    ax.text(90, 15, "Danger coverage\nwithout evidence", ha="center", fontsize=8, color="#bbb")

    ax.set_xlabel("Dangerous Capability Reporting Score")
    ax.set_ylabel("Evidence Sufficiency Score")
    ax.set_title('"Show Your Work"\nDo companies back up safety claims with evidence?', fontweight="bold")
    ax.set_xlim(10, 105)
    ax.set_ylim(10, 105)
    ax.grid(True, alpha=0.15)
    legend_elements = [Patch(facecolor=c, label=co) for co, c in COMPANY_COLORS.items()]
    ax.legend(handles=legend_elements, framealpha=0.9, loc="upper left")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "show_your_work.png", dpi=200)
    plt.close()
    print("  Saved show_your_work.png")


# ── 5 & 6. OVER TIME (breadth + transparency) ────────────────────────
def _plot_over_time(scores, dimension, filename, ylabel, title):
    fig, ax = plt.subplots(figsize=(13, 6))

    texts = []
    for mk, data in scores.items():
        company = data["company"]
        dim_data = data.get("dimensions", {}).get(dimension, {})
        if not dim_data:
            continue
        mean = dim_data["mean"]
        date = datetime.strptime(MODELS[mk]["date"], "%Y-%m")

        metric_means = [data["metrics"][m]["mean"] for m in data["metrics"]
                        if METRICS.get(m, {}).get("dimension") == dimension]
        se = np.std(metric_means) / np.sqrt(len(metric_means)) if len(metric_means) > 1 else 0

        ax.errorbar(date, mean, yerr=se,
                    color=COMPANY_COLORS[company], fmt=COMPANY_MARKERS[company],
                    markersize=8, zorder=5, markeredgecolor="white", markeredgewidth=0.5,
                    capsize=3, elinewidth=1, alpha=0.9)
        texts.append(ax.text(date, mean, data["display_name"], fontsize=7.5,
                            color=COMPANY_COLORS[company], fontweight="bold"))

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, lw=0.5),
                expand=(1.8, 1.8), force_text=(0.6, 0.6))

    ax.set_xlabel("Release Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylim(20, 100)
    ax.grid(True, alpha=0.2)
    legend_elements = [Patch(facecolor=c, label=co) for co, c in COMPANY_COLORS.items()]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=200)
    plt.close()
    print(f"  Saved {filename}")


def plot_overall_over_time(scores):
    """Overall score over time — uses all metrics for stderr."""
    fig, ax = plt.subplots(figsize=(13, 6))

    texts = []
    for mk, data in scores.items():
        if mk not in MODELS:
            continue
        company = data["company"]
        overall = data.get("dimensions", {}).get("Overall", {}).get("mean", 0)
        se = get_overall_stderr(scores, mk)
        date = datetime.strptime(MODELS[mk]["date"], "%Y-%m")

        ax.errorbar(date, overall, yerr=se,
                    color=COMPANY_COLORS[company], fmt=COMPANY_MARKERS[company],
                    markersize=8, zorder=5, markeredgecolor="white", markeredgewidth=0.5,
                    capsize=3, elinewidth=1, alpha=0.9)
        texts.append(ax.text(date, overall, data["display_name"], fontsize=7.5,
                            color=COMPANY_COLORS[company], fontweight="bold"))

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, lw=0.5),
                expand=(1.8, 1.8), force_text=(0.6, 0.6))

    ax.set_xlabel("Release Date")
    ax.set_ylabel("Overall Score\n(Comprehensiveness + Reasoning Quality)")
    ax.set_title("System Card Eval Score Over Time", fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylim(20, 100)
    ax.grid(True, alpha=0.2)
    legend_elements = [Patch(facecolor=c, label=co) for co, c in COMPANY_COLORS.items()]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "overall_over_time.png", dpi=200)
    plt.close()
    print("  Saved overall_over_time.png")


def plot_comprehensiveness_over_time(scores):
    _plot_over_time(scores, "Comprehensiveness", "comprehensiveness_over_time.png",
                    "Comprehensiveness Score", "System Card Comprehensiveness Over Time\nWhat topics are covered?")


def plot_transparency_over_time(scores):
    _plot_over_time(scores, "Reasoning Quality", "reasoning_quality_over_time.png",
                    "Reasoning Quality Score", "System Card Reasoning Quality Over Time\nHow well are claims backed up?")


# ── 7. WHAT CHANGED ───────────────────────────────────────────────────
def plot_what_changed(scores):
    pairs = {
        "Anthropic": ("claude_4", "mythos_preview"),
        "OpenAI": ("gpt_4o", "gpt_5_4_thinking"),
        "Google": ("gemini_2_5_pro", "gemini_3_1_pro"),
    }
    metric_keys = [k for k, v in METRICS.items() if v["dimension"] in DIMENSIONS]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for idx, (company, (old_mk, new_mk)) in enumerate(pairs.items()):
        ax = axes[idx]
        old_data = scores[old_mk]
        new_data = scores[new_mk]

        old_scores = [old_data["metrics"].get(m, {}).get("mean", 0) for m in metric_keys]
        new_scores = [new_data["metrics"].get(m, {}).get("mean", 0) for m in metric_keys]
        deltas = [n - o for n, o in zip(new_scores, old_scores)]

        labels = [METRIC_LABELS.get(m, m) for m in metric_keys]
        y_pos = np.arange(len(labels))
        colors = ["#10B981" if d >= 0 else "#EF4444" for d in deltas]

        ax.barh(y_pos, deltas, color=colors, alpha=0.8, height=0.6)
        for i, d in enumerate(deltas):
            side = 1 if d >= 0 else -1
            ax.text(d + side * 1.5, i, f"{d:+.0f}", va="center", fontsize=8,
                    fontweight="bold", color=colors[i])

        ax.axvline(x=0, color="#666", linewidth=0.8)
        ax.set_yticks(y_pos)
        if idx == 0:
            ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{company}\n{old_data['display_name']} \u2192 {new_data['display_name']}",
                     fontsize=11, fontweight="bold", color=COMPANY_COLORS[company])
        ax.set_xlim(-50, 50)
        ax.grid(True, axis="x", alpha=0.2)
        ax.invert_yaxis()

    fig.suptitle("What Changed? Oldest vs. Newest System Card", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "what_changed.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved what_changed.png")


def main():
    print("Loading data...")
    scores, raw_results = load_data()

    print("\nGenerating final plots...")
    plot_report_card(scores)
    plot_overall_ranking(scores)
    plot_overall_over_time(scores)
    plot_topic_coverage(raw_results)
    plot_show_your_work(scores)
    plot_what_changed(scores)

    print("\nDone! 7 plots saved.")


if __name__ == "__main__":
    main()
