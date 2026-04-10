"""Generate all visualizations for the system card eval."""

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from config import METRICS, MODELS, DIMENSIONS, ALL_DIMENSIONS, RESULTS_DIR

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

COMPANY_COLORS = {
    "Anthropic": "#D97706",
    "OpenAI": "#10B981",
    "Google": "#3B82F6",
}

COMPANY_MARKERS = {
    "Anthropic": "o",
    "OpenAI": "s",
    "Google": "D",
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
    """Compute stderr for overall score: std across all metric means / sqrt(n_metrics)."""
    metric_means = []
    for metric, m in scores[mk]["metrics"].items():
        if METRICS[metric]["dimension"] in DIMENSIONS:
            metric_means.append(m["mean"])
    if len(metric_means) < 2:
        return 0
    return float(np.std(metric_means) / np.sqrt(len(metric_means)))


def get_metric_stderr(metric_data):
    """Compute stderr for a metric: std / sqrt(n)."""
    n = metric_data.get("n", 6)
    std = metric_data.get("std", 0)
    return std / np.sqrt(n) if n > 0 else 0


# ── 1. Dimension scores over time (separate plots) ───────────────────
def _plot_dimension_over_time(scores, dimension, filename, ylabel, title):
    """Helper: plot a single dimension over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for mk, data in scores.items():
        company = data["company"]
        dim_data = data.get("dimensions", {}).get(dimension, {})
        if not dim_data:
            continue
        mean = dim_data["mean"]
        date_str = MODELS[mk]["date"]
        date = datetime.strptime(date_str, "%Y-%m")

        # Compute stderr from metric means within this dimension
        metric_means = [data["metrics"][m]["mean"] for m in data["metrics"]
                        if METRICS.get(m, {}).get("dimension") == dimension]
        se = np.std(metric_means) / np.sqrt(len(metric_means)) if len(metric_means) > 1 else 0

        ax.errorbar(date, mean, yerr=se,
                    color=COMPANY_COLORS[company], fmt=COMPANY_MARKERS[company],
                    markersize=8, zorder=5, markeredgecolor="white", markeredgewidth=0.5,
                    capsize=3, elinewidth=1)
        ax.annotate(data["display_name"], (date, mean),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, color=COMPANY_COLORS[company])

    ax.set_xlabel("Release Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    legend_elements = [Patch(facecolor=c, label=co) for co, c in COMPANY_COLORS.items()]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=200)
    plt.close()
    print(f"  Saved {filename}")


def plot_comprehensiveness_over_time(scores):
    _plot_dimension_over_time(
        scores, "Comprehensiveness", "comprehensiveness_over_time.png",
        "Comprehensiveness Score", "System Card Comprehensiveness Over Time\nWhat topics are covered?"
    )


def plot_transparency_over_time(scores):
    _plot_dimension_over_time(
        scores, "Transparency", "transparency_over_time.png",
        "Transparency Score", "System Card Transparency Over Time\nHow well are claims backed up?"
    )


# ── 2. Company radar chart ────────────────────────────────────────────
def plot_company_radar(scores):
    # Average metrics per company
    company_metrics = defaultdict(lambda: defaultdict(list))
    for mk, data in scores.items():
        company = data["company"]
        for metric, m in data["metrics"].items():
            company_metrics[company][metric].append(m["mean"])

    company_means = {}
    for company in company_metrics:
        company_means[company] = {}
        for metric in company_metrics[company]:
            company_means[company][metric] = np.mean(company_metrics[company][metric])

    # Only use Breadth + Transparency metrics
    metric_keys = [k for k, v in METRICS.items() if v["dimension"] in DIMENSIONS]
    metric_labels = [k.replace("_", " ").title() for k in metric_keys]

    N = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for company, color in COMPANY_COLORS.items():
        if company not in company_means:
            continue
        values = [company_means[company].get(m, 0) for m in metric_keys]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=company)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=8)
    ax.set_ylim(0, 100)
    ax.set_title("System Card Quality by Company\n(Breadth + Transparency)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "company_radar.png", dpi=200)
    plt.close()
    print("  Saved company_radar.png")


# ── 3. Pages vs score scatter ─────────────────────────────────────────
def plot_pages_vs_score(scores):
    PAGE_COUNTS = {
        "claude_4": 212, "claude_opus_4_5": 153, "claude_haiku_4_5": 39,
        "claude_sonnet_4_6": 135, "claude_opus_4_6": 369, "mythos_preview": 303,
        "gpt_4o": 33, "o1": 43, "gpt_5": 60, "gpt_5_4_thinking": 39,
        "gemini_2_5_pro": 21, "gemini_3_pro": 35, "gemini_3_flash": 6, "gemini_3_1_pro": 9,
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for mk, data in scores.items():
        company = data["company"]
        overall = data.get("dimensions", {}).get("Overall", {}).get("mean", 0)
        se = get_overall_stderr(scores, mk)
        pages = PAGE_COUNTS.get(mk, 0)

        ax.errorbar(pages, overall, yerr=se,
                    color=COMPANY_COLORS[company], fmt=COMPANY_MARKERS[company],
                    markersize=8, zorder=5, markeredgecolor="white", markeredgewidth=0.5,
                    capsize=3, elinewidth=1)
        ax.annotate(data["display_name"], (pages, overall),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, color=COMPANY_COLORS[company])

    ax.set_xlabel("Total Pages (System Card + Companions)")
    ax.set_ylabel("Overall Score")
    ax.set_title("Pages vs. Transparency Score")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    legend_elements = [Patch(facecolor=c, label=co) for co, c in COMPANY_COLORS.items()]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "pages_vs_score.png", dpi=200)
    plt.close()
    print("  Saved pages_vs_score.png")


# ── 4. Per-metric bar chart with error bars ───────────────────────────
def plot_metric_bars(scores):
    metric_keys = [k for k, v in METRICS.items() if v["dimension"] in DIMENSIONS]

    # Sort models by overall score
    model_keys = sorted(
        scores.keys(),
        key=lambda mk: scores[mk].get("dimensions", {}).get("Overall", {}).get("mean", 0),
        reverse=True,
    )

    fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharey=True)
    axes = axes.flatten()

    for idx, metric in enumerate(metric_keys):
        ax = axes[idx]
        names = []
        means = []
        stds = []
        colors = []

        for mk in model_keys:
            data = scores[mk]
            m = data["metrics"].get(metric, {})
            if not m:
                continue
            names.append(data["display_name"])
            means.append(m["mean"])
            stds.append(m["std"])
            colors.append(COMPANY_COLORS[data["company"]])

        y_pos = np.arange(len(names))
        ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.8, height=0.7,
                capsize=3, error_kw={"linewidth": 1})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlim(0, 105)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.invert_yaxis()

    # Hide unused axes
    for idx in range(len(metric_keys), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Per-Metric Scores by Model (Breadth + Transparency)", fontsize=16, y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "metric_bars.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved metric_bars.png")


# ── 5. Topic coverage heatmap ─────────────────────────────────────────
def plot_topic_heatmap(raw_results):
    """Which of the 17 topics does each model cover?"""
    from config import TOPIC_CHECKLIST

    # Collect topic presence across all judges for each model
    topic_data = defaultdict(lambda: defaultdict(list))

    for r in raw_results:
        if r.get("_metric") != "topic_coverage":
            continue
        mk = r["_model_key"]
        topics = r.get("topics", [])
        for t in topics:
            topic_data[mk][t["topic"]].append(1 if t["present"] else 0)

    # Average across judges: fraction of judges that found the topic
    model_keys = sorted(topic_data.keys(),
                        key=lambda mk: MODELS[mk]["date"])

    matrix = []
    for mk in model_keys:
        row = []
        for topic in TOPIC_CHECKLIST:
            vals = topic_data[mk].get(topic, [0])
            row.append(np.mean(vals) * 100)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(matrix.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(model_keys)))
    ax.set_xticklabels([MODELS[mk]["display_name"] for mk in model_keys],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(TOPIC_CHECKLIST)))
    ax.set_yticklabels([t[:50] for t in TOPIC_CHECKLIST], fontsize=8)

    # Add text annotations
    for i in range(len(model_keys)):
        for j in range(len(TOPIC_CHECKLIST)):
            val = matrix[i, j]
            text = f"{val:.0f}"
            color = "white" if val < 40 or val > 80 else "black"
            ax.text(i, j, text, ha="center", va="center", fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="% of judges finding topic present")
    ax.set_title("Topic Coverage Heatmap")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "topic_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved topic_heatmap.png")


# ── 6. Which metrics differentiate most ───────────────────────────────
def plot_metric_variance(scores):
    """Metrics with highest cross-model variance = most discriminating."""
    metric_keys = list(METRICS.keys())

    variances = []
    for metric in metric_keys:
        model_means = []
        for mk, data in scores.items():
            m = data["metrics"].get(metric, {})
            if m:
                model_means.append(m["mean"])
        variances.append((metric, np.std(model_means) if model_means else 0))

    variances.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [v[0].replace("_", " ").title() for v in variances]
    vals = [v[1] for v in variances]
    colors = []
    for v in variances:
        dim = METRICS[v[0]]["dimension"]
        if dim == "Comprehensiveness":
            colors.append("#D97706")
        elif dim == "Transparency":
            colors.append("#10B981")
        else:
            colors.append("#3B82F6")

    ax.barh(range(len(names)), vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Standard Deviation Across Models")
    ax.set_title("Most Discriminating Metrics\n(higher = more spread across models)")
    ax.invert_yaxis()

    legend_elements = [
        Patch(facecolor="#D97706", label="Comprehensiveness"),
        Patch(facecolor="#10B981", label="Transparency"),
        Patch(facecolor="#3B82F6", label="3rd-party Verification"),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "metric_variance.png", dpi=200)
    plt.close()
    print("  Saved metric_variance.png")


# ── 7. Company transparency gap ───────────────────────────────────────
def plot_transparency_gap(scores):
    """For each company, show the range from worst to best model."""
    fig, ax = plt.subplots(figsize=(10, 5))

    companies = {}
    for mk, data in scores.items():
        company = data["company"]
        overall = data.get("dimensions", {}).get("Overall", {}).get("mean", 0)
        if company not in companies:
            companies[company] = {"models": [], "scores": []}
        companies[company]["models"].append(data["display_name"])
        companies[company]["scores"].append(overall)

    y_pos = 0
    for company in ["Anthropic", "OpenAI", "Google"]:
        if company not in companies:
            continue
        c_scores = companies[company]["scores"]
        c_models = companies[company]["models"]
        min_s, max_s = min(c_scores), max(c_scores)
        min_m = c_models[c_scores.index(min_s)]
        max_m = c_models[c_scores.index(max_s)]
        gap = max_s - min_s
        color = COMPANY_COLORS[company]

        # Draw range bar
        ax.barh(y_pos, gap, left=min_s, color=color, alpha=0.3, height=0.5)
        # Draw individual model dots
        for s, m in zip(c_scores, c_models):
            ax.scatter(s, y_pos, color=color, s=80, zorder=5, edgecolors="white")
            ax.annotate(m, (s, y_pos), textcoords="offset points",
                        xytext=(0, 12), fontsize=7, ha="center", color=color)

        ax.text(min_s - 2, y_pos, f"{min_s:.0f}", ha="right", va="center", fontsize=9, color=color)
        ax.text(max_s + 2, y_pos, f"{max_s:.0f}", ha="left", va="center", fontsize=9, color=color)
        ax.text(105, y_pos, f"gap: {gap:.0f}", ha="left", va="center", fontsize=10,
                fontweight="bold", color=color)

        y_pos += 1

    ax.set_yticks(range(len(companies)))
    ax.set_yticklabels(["Anthropic", "OpenAI", "Google"])
    ax.set_xlim(0, 120)
    ax.set_xlabel("Overall Score")
    ax.set_title("Transparency Gap Within Each Company\n(range from worst to best model)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "transparency_gap.png", dpi=200)
    plt.close()
    print("  Saved transparency_gap.png")


# ── 8. Judge severity profile ─────────────────────────────────────────
def plot_judge_severity(raw_results):
    """Mean score given by each judge across all models."""
    judge_scores = defaultdict(list)

    for r in raw_results:
        score = r.get("score")
        if score is None:
            continue
        judge_scores[r["_judge"]].append(float(score))

    fig, ax = plt.subplots(figsize=(8, 5))

    judges = sorted(judge_scores.keys())
    means = [np.mean(judge_scores[j]) for j in judges]
    stds = [np.std(judge_scores[j]) for j in judges]
    labels = [f"{j}\n(n={len(judge_scores[j])})" for j in judges]

    bars = ax.bar(range(len(judges)), means, yerr=stds, capsize=5,
                  color=["#D97706", "#3B82F6", "#10B981"], alpha=0.8)
    ax.set_xticks(range(len(judges)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Score Given")
    ax.set_title("Judge Severity Profile\n(lower = harsher judge)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 2, f"{m:.1f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "judge_severity.png", dpi=200)
    plt.close()
    print("  Saved judge_severity.png")


# ── 9. Overall ranking bar chart ──────────────────────────────────────
def plot_overall_ranking(scores):
    """Horizontal bar chart of overall scores, colored by company."""
    model_keys = sorted(
        scores.keys(),
        key=lambda mk: scores[mk].get("dimensions", {}).get("Overall", {}).get("mean", 0),
        reverse=True,
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    names = [scores[mk]["display_name"] for mk in model_keys]
    overall = [scores[mk]["dimensions"]["Overall"]["mean"] for mk in model_keys]
    stderrs = [get_overall_stderr(scores, mk) for mk in model_keys]
    colors = [COMPANY_COLORS[scores[mk]["company"]] for mk in model_keys]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, overall, xerr=stderrs, color=colors, alpha=0.85, height=0.7,
                   edgecolor="white", capsize=3, error_kw={"linewidth": 1})

    # Add score labels
    for i, (v, se, mk) in enumerate(zip(overall, stderrs, model_keys)):
        ax.text(v + se + 1.5, i, f"{v:.1f}", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Overall Score (Breadth + Transparency)")
    ax.set_title("System Card Transparency Ranking")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    legend_elements = [Patch(facecolor=c, label=co) for co, c in COMPANY_COLORS.items()]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "overall_ranking.png", dpi=200)
    plt.close()
    print("  Saved overall_ranking.png")


def main():
    print("Loading data...")
    scores, raw_results = load_data()
    print(f"  {len(scores)} models, {len(raw_results)} raw results")

    print("\nGenerating visualizations...")
    plot_overall_ranking(scores)
    plot_comprehensiveness_over_time(scores)
    plot_transparency_over_time(scores)
    plot_pages_vs_score(scores)
    plot_metric_bars(scores)
    plot_topic_heatmap(raw_results)
    plot_transparency_gap(scores)

    print(f"\nDone! All charts saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
