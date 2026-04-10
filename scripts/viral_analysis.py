"""Viral-worthy analyses for Twitter."""

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import METRICS, MODELS, DIMENSIONS, RESULTS_DIR, TOPIC_CHECKLIST

COMPANY_COLORS = {
    "Anthropic": "#D97706",
    "OpenAI": "#10B981",
    "Google": "#3B82F6",
}


def get_overall_stderr(scores, mk):
    """Compute stderr for overall score."""
    metric_means = []
    for metric, m in scores[mk]["metrics"].items():
        if METRICS[metric]["dimension"] in DIMENSIONS:
            metric_means.append(m["mean"])
    if len(metric_means) < 2:
        return 0
    return float(np.std(metric_means) / np.sqrt(len(metric_means)))


def load_data():
    scores = json.loads((RESULTS_DIR / "scores.json").read_text())
    raw_results = []
    for f in sorted(os.listdir(RESULTS_DIR / "raw")):
        if f.endswith(".json"):
            data = json.loads((RESULTS_DIR / "raw" / f).read_text())
            if "_error" not in data:
                raw_results.append(data)
    return scores, raw_results


# ── 1. "What AI companies don't want you to know" ─────────────────────
# Which topics do ALL companies skip? Which does only 1 company cover?
def plot_topic_by_company(raw_results):
    """Bar chart: for each topic, which companies cover it?"""
    topic_by_company = defaultdict(lambda: defaultdict(list))

    for r in raw_results:
        if r.get("_metric") != "topic_coverage":
            continue
        mk = r["_model_key"]
        company = MODELS[mk]["company"]
        for t in r.get("topics", []):
            topic_by_company[t["topic"]][company].append(1 if t["present"] else 0)

    fig, ax = plt.subplots(figsize=(14, 9))

    topics = list(TOPIC_CHECKLIST)
    companies = list(COMPANY_COLORS.keys())

    # Compute per-company mean and stderr for each topic
    company_means = {co: [] for co in companies}
    company_ses = {co: [] for co in companies}
    for topic in topics:
        for company in companies:
            vals = topic_by_company[topic].get(company, [0])
            mean = np.mean(vals) * 100
            se = (np.std(vals) / np.sqrt(len(vals))) * 100 if len(vals) > 1 else 0
            company_means[company].append(mean)
            company_ses[company].append(se)

    # Sort by total coverage (sum across companies)
    totals = [sum(company_means[co][i] for co in companies) for i in range(len(topics))]
    sorted_idx = sorted(range(len(topics)), key=lambda i: totals[i], reverse=True)
    topics = [topics[i] for i in sorted_idx]
    for co in companies:
        company_means[co] = [company_means[co][i] for i in sorted_idx]
        company_ses[co] = [company_ses[co][i] for i in sorted_idx]

    y_pos = np.arange(len(topics))

    # Stacked horizontal bars with stderr
    left = np.zeros(len(topics))
    for company, color in COMPANY_COLORS.items():
        means = np.array(company_means[company])
        ses = np.array(company_ses[company])
        ax.barh(y_pos, means, left=left, color=color, alpha=0.85, label=company, height=0.7)
        # Add stderr caps at the right edge of each company's segment
        for i in range(len(topics)):
            if ses[i] > 0:
                x = left[i] + means[i]
                ax.errorbar(x, y_pos[i], xerr=[[0], [ses[i]]], color=color,
                            capsize=2, elinewidth=1, fmt="none", alpha=0.7)
        left += means

    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[:55] for t in topics], fontsize=9)
    ax.set_xlim(0, 320)
    ax.set_xlabel("% of Models Covering This Topic (stacked by company)")
    ax.set_title("What Do System Cards Actually Cover?\nTopic Coverage by Company", fontsize=14)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "viral_topic_coverage.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved viral_topic_coverage.png")


# ── 2. "The transparency tax" — dollars per page of transparency ──────
def plot_transparency_per_page(scores):
    """Score per page — who gets the most transparency per unit of effort?"""
    PAGE_COUNTS = {
        "claude_4": 212, "claude_opus_4_5": 153, "claude_haiku_4_5": 39,
        "claude_sonnet_4_6": 135, "claude_opus_4_6": 369, "mythos_preview": 303,
        "gpt_4o": 33, "o1": 43, "gpt_5": 60, "gpt_5_4_thinking": 39,
        "gemini_2_5_pro": 21, "gemini_3_pro": 35, "gemini_3_flash": 6, "gemini_3_1_pro": 9,
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    models = []
    for mk, data in scores.items():
        overall = data.get("dimensions", {}).get("Overall", {}).get("mean", 0)
        pages = PAGE_COUNTS.get(mk, 1)
        efficiency = overall / pages  # score per page
        models.append((mk, data["display_name"], data["company"], overall, pages, efficiency))

    models.sort(key=lambda x: x[5], reverse=True)

    names = [m[1] for m in models]
    efficiencies = [m[5] for m in models]
    colors = [COMPANY_COLORS[m[2]] for m in models]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, efficiencies, color=colors, alpha=0.85, height=0.7)

    for i, m in enumerate(models):
        ax.text(m[5] + 0.05, i, f"{m[5]:.2f}  ({m[3]:.0f} pts / {m[4]} pg)",
                va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Transparency Score Per Page")
    ax.set_title("Transparency Efficiency\nWho says the most per page?", fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    legend_elements = [Patch(facecolor=c, label=co) for co, c in COMPANY_COLORS.items()]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "viral_transparency_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved viral_transparency_efficiency.png")


# ── 3. "Show your work" — evidence vs claims ──────────────────────────
def plot_evidence_vs_claims(scores):
    """Scatter: evidence sufficiency vs dangerous capability reporting.
    Who talks big on safety but doesn't show the data?"""
    fig, ax = plt.subplots(figsize=(10, 8))

    for mk, data in scores.items():
        ev_data = data["metrics"].get("evidence_sufficiency", {})
        dc_data = data["metrics"].get("dangerous_capability_reporting", {})
        evidence = ev_data.get("mean", 0)
        dangerous = dc_data.get("mean", 0)
        ev_se = ev_data.get("std", 0) / np.sqrt(ev_data.get("n", 6)) if ev_data.get("n", 0) > 0 else 0
        dc_se = dc_data.get("std", 0) / np.sqrt(dc_data.get("n", 6)) if dc_data.get("n", 0) > 0 else 0
        company = data["company"]

        ax.errorbar(dangerous, evidence, xerr=dc_se, yerr=ev_se,
                    color=COMPANY_COLORS[company], fmt="o",
                    markersize=10, zorder=5, markeredgecolor="white", markeredgewidth=0.5,
                    capsize=3, elinewidth=1)
        ax.annotate(data["display_name"], (dangerous, evidence),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, color=COMPANY_COLORS[company])

    # Add quadrant labels
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.3)
    ax.text(25, 85, "Provides evidence\nbut avoids dangerous\ncapability topics", ha="center",
            fontsize=8, color="gray", alpha=0.5)
    ax.text(75, 85, "Transparent:\ncovers dangerous capabilities\nwith evidence", ha="center",
            fontsize=8, color="gray", alpha=0.5)
    ax.text(25, 15, "Opaque:\nlittle coverage,\nlittle evidence", ha="center",
            fontsize=8, color="gray", alpha=0.5)
    ax.text(75, 15, "Talks about danger\nbut doesn't\nshow the data", ha="center",
            fontsize=8, color="gray", alpha=0.5)

    ax.set_xlabel("Dangerous Capability Reporting Score")
    ax.set_ylabel("Evidence Sufficiency Score")
    ax.set_title("\"Show Your Work\"\nDo companies back up their safety claims with evidence?", fontsize=14)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    legend_elements = [Patch(facecolor=c, label=co) for co, c in COMPANY_COLORS.items()]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "viral_show_your_work.png", dpi=200)
    plt.close()
    print("  Saved viral_show_your_work.png")


# ── 4. "The missing pieces" — what NO card covers ─────────────────────
def plot_biggest_gaps(raw_results):
    """What topics/metrics have the biggest gaps across the industry?"""
    # Find topics with lowest average coverage
    topic_coverage = defaultdict(list)
    for r in raw_results:
        if r.get("_metric") != "topic_coverage":
            continue
        for t in r.get("topics", []):
            topic_coverage[t["topic"]].append(1 if t["present"] else 0)

    gaps = [(topic, np.mean(vals) * 100) for topic, vals in topic_coverage.items()]
    gaps.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(12, 7))

    topics = [g[0][:50] for g in gaps]
    coverage = [g[1] for g in gaps]
    colors = ["#EF4444" if c < 30 else "#F59E0B" if c < 60 else "#10B981" for c in coverage]

    ax.barh(range(len(topics)), coverage, color=colors, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% of Models Covering This Topic (across all judges)")
    ax.set_title("The Biggest Gaps in AI Transparency\nWhat frontier model system cards DON'T tell you",
                 fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Add color legend
    legend_elements = [
        Patch(facecolor="#EF4444", label="Rarely covered (<30%)"),
        Patch(facecolor="#F59E0B", label="Sometimes covered (30-60%)"),
        Patch(facecolor="#10B981", label="Usually covered (>60%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "viral_biggest_gaps.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved viral_biggest_gaps.png")


# ── 5. "Flagship vs forgotten" — do small models get less transparency?
def plot_flagship_vs_small(scores):
    """Compare flagship models vs smaller/cheaper models per company."""
    FLAGSHIP = {
        "Anthropic": ["claude_opus_4_6", "mythos_preview"],
        "OpenAI": ["gpt_5", "gpt_5_4_thinking"],
        "Google": ["gemini_3_pro", "gemini_3_1_pro"],
    }
    SMALL = {
        "Anthropic": ["claude_haiku_4_5"],
        "OpenAI": ["gpt_4o"],
        "Google": ["gemini_3_flash"],
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(3)
    width = 0.35
    companies = ["Anthropic", "OpenAI", "Google"]

    flagship_means = []
    small_means = []

    for company in companies:
        f_scores = [scores[mk]["dimensions"]["Overall"]["mean"]
                    for mk in FLAGSHIP[company] if mk in scores]
        s_scores = [scores[mk]["dimensions"]["Overall"]["mean"]
                    for mk in SMALL[company] if mk in scores]
        flagship_means.append(np.mean(f_scores) if f_scores else 0)
        small_means.append(np.mean(s_scores) if s_scores else 0)

    bars1 = ax.bar(x - width/2, flagship_means, width, label="Flagship models",
                   color=[COMPANY_COLORS[c] for c in companies], alpha=0.9)
    bars2 = ax.bar(x + width/2, small_means, width, label="Smaller models",
                   color=[COMPANY_COLORS[c] for c in companies], alpha=0.4)

    for i in range(3):
        gap = flagship_means[i] - small_means[i]
        ax.text(i, max(flagship_means[i], small_means[i]) + 2,
                f"+{gap:.0f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(companies)
    ax.set_ylabel("Overall Score")
    ax.set_title("Flagship vs. Smaller Models\nDo companies put equal transparency effort into all models?",
                 fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "viral_flagship_vs_small.png", dpi=200)
    plt.close()
    print("  Saved viral_flagship_vs_small.png")


# ── 6. Simple headline chart — the one chart for Twitter ──────────────
def plot_headline(scores):
    """Clean, simple overall comparison by company average."""
    company_scores = defaultdict(list)
    for mk, data in scores.items():
        overall = data.get("dimensions", {}).get("Overall", {}).get("mean", 0)
        company_scores[data["company"]].append(overall)

    companies = ["Anthropic", "OpenAI", "Google"]
    means = [np.mean(company_scores[c]) for c in companies]
    stderrs = [np.std(company_scores[c]) / np.sqrt(len(company_scores[c])) for c in companies]
    colors = [COMPANY_COLORS[c] for c in companies]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(companies, means, yerr=stderrs, color=colors, alpha=0.85,
                  capsize=8, error_kw={"linewidth": 2}, width=0.6,
                  edgecolor="white", linewidth=2)

    for i, (m, s) in enumerate(zip(means, stderrs)):
        ax.text(i, m + s + 2, f"{m:.0f}", ha="center", fontsize=20, fontweight="bold")

    ax.set_ylabel("Transparency Score", fontsize=13)
    ax.set_title("How Transparent Are AI System Cards?", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=14)

    # Subtitle
    ax.text(0.5, -0.12, "Scored across 14 system cards on 10 metrics (breadth + transparency)\n3 LLM judges × 2 runs each",
            transform=ax.transAxes, ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "viral_headline.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved viral_headline.png")


def main():
    print("Loading data...")
    scores, raw_results = load_data()

    print("\nGenerating viral analyses...")
    plot_headline(scores)
    plot_topic_by_company(raw_results)
    plot_transparency_per_page(scores)
    plot_evidence_vs_claims(scores)
    plot_biggest_gaps(raw_results)
    plot_flagship_vs_small(scores)

    print(f"\nDone! All charts saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
