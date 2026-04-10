"""Configuration for system card evaluation pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
SYSTEM_CARDS_DIR = ROOT_DIR / "system_cards"
COMPANION_DIR = SYSTEM_CARDS_DIR / "companion_reports"
PROMPTS_DIR = ROOT_DIR / "prompts"
RESULTS_DIR = ROOT_DIR / "results"

# ── OpenRouter ─────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Judge models ───────────────────────────────────────────────────────
JUDGES = {
    "sonnet_4_6": {
        "model": "anthropic/claude-sonnet-4.6",
        "max_tokens": 128_000,
        "extra_body": {
            "reasoning": {"effort": "high"},
        },
    },
    "gpt_5_4": {
        "model": "openai/gpt-5.4",
        "max_tokens": 128_000,
        "extra_body": {
            "reasoning": {"effort": "high"},
        },
    },
    "gemini_3_1_pro": {
        "model": "google/gemini-3.1-pro-preview",
        "max_tokens": 65_536,
        "extra_body": {
            "reasoning": {"effort": "high"},
        },
    },
}

RUNS_PER_JUDGE = 2  # 2 runs × 3 judges = 6 data points per metric

# ── System cards & companion report mapping ────────────────────────────
MODELS = {
    # Anthropic
    "claude_4": {
        "display_name": "Claude 4",
        "company": "Anthropic",
        "date": "2025-05",
        "card_file": "claude_4.pdf",
        "companions": [
            "anthropic_asl3_report.pdf",
            "anthropic_pilot_sabotage_risk_report_2025.pdf",
        ],
    },
    "claude_opus_4_5": {
        "display_name": "Claude Opus 4.5",
        "company": "Anthropic",
        "date": "2025-08",
        "card_file": "claude_opus_4_5.pdf",
        "companions": [],
    },
    # claude_haiku_4_5, claude_sonnet_4_6 excluded — smaller models
    "claude_opus_4_6": {
        "display_name": "Claude Opus 4.6",
        "company": "Anthropic",
        "date": "2026-02",
        "card_file": "claude_opus_4_6.pdf",
        "companions": [
            "anthropic_feb_2026_risk_report.pdf",
            "anthropic_opus_4_6_sabotage_risk_report.pdf",
        ],
    },
    "mythos_preview": {
        "display_name": "Mythos Preview",
        "company": "Anthropic",
        "date": "2026-04",
        "card_file": "mythos_preview_system_card.pdf",
        "companions": [
            "anthropic_mythos_risk_report.pdf",
        ],
    },
    # OpenAI
    "gpt_4o": {
        "display_name": "GPT-4o",
        "company": "OpenAI",
        "date": "2024-08",
        "card_file": "gpt_4o.pdf",
        "companions": [],
    },
    "o1": {
        "display_name": "o1",
        "company": "OpenAI",
        "date": "2024-09",
        "card_file": "o1.pdf",
        "companions": [],
    },
    "gpt_5": {
        "display_name": "GPT-5",
        "company": "OpenAI",
        "date": "2025-08",
        "card_file": "gpt_5.pdf",
        "companions": [],
    },
    "gpt_5_3_codex": {
        "display_name": "GPT-5.3 Codex",
        "company": "OpenAI",
        "date": "2026-02",
        "card_file": "gpt_5_3_codex.pdf",
        "companions": [],
    },
    "gpt_5_4_thinking": {
        "display_name": "GPT-5.4 Thinking",
        "company": "OpenAI",
        "date": "2026-03",
        "card_file": "gpt_5_4_thinking.pdf",
        "companions": [],
    },
    # Google
    "gemini_2_5_pro": {
        "display_name": "Gemini 2.5 Pro",
        "company": "Google",
        "date": "2025-06",
        "card_file": "gemini_2_5_pro.pdf",
        "companions": [
            "gemini_2_5_technical_report.pdf",
        ],
    },
    "gemini_3_pro": {
        "display_name": "Gemini 3 Pro",
        "company": "Google",
        "date": "2025-12",
        "card_file": "gemini_3_pro.pdf",
        "companions": [
            "gemini_3_pro_fsf_report.pdf",
            "gemini_3_pro_evaluation.pdf",
        ],
    },
    # gemini_3_flash excluded — 6-page card, not comparable
    "gemini_3_1_pro": {
        "display_name": "Gemini 3.1 Pro",
        "company": "Google",
        "date": "2026-02",
        "card_file": "gemini_3_1_pro.pdf",
        "companions": [
            "gemini_3_1_pro_evaluation.pdf",
        ],
    },
    # ── xAI and Meta excluded for now (can re-enable later) ──
    # "grok_4": { ... },
    # "grok_4_fast": { ... },
    # "grok_code_fast_1": { ... },
    # "grok_4_1": { ... },
    # "llama_4": { ... },
}

# ── Topic checklist (for topic_coverage metric) ────────────────────────
# Canonical topics derived bottom-up from all 14 system cards.
# Each topic appears in at least 2 companies' cards.
# "Model introduction & overview" removed (100% coverage, no discriminative power).
# See results/canonical_topics.json for full derivation.
TOPIC_CHECKLIST = [
    "Safety & harmful content evaluations",
    "Dangerous capability evaluations (CBRN, cyber, autonomy)",
    "Alignment assessment",
    "External red teaming & third-party assessments",
    "Capability benchmarks & performance",
    "Agentic safety",
    "Model data & training",
    "Bias & fairness evaluations",
    "Intended use, limitations & societal impacts",
    "Honesty & hallucination evaluations",
    "Model welfare & moral status",
    "Sabotage risk assessment",
    "Implementation, deployment & sustainability",
    "Multilingual performance",
    "Chain-of-thought & reasoning transparency",
    "Reward hacking evaluations",
]

# ── Reference lists (for extractive metrics) ───────────────────────────
RISK_CATEGORIES = [
    "CBRN (chemical, biological, radiological, nuclear)",
    "Cybersecurity (offensive cyber capabilities, vulnerability exploitation)",
    "Persuasion / manipulation (influence operations, deceptive content)",
    "Bias / fairness (demographic bias, stereotyping, representation)",
    "Societal impact (labor displacement, information ecosystem, democratic processes)",
    "Economic impact (market effects, competitive dynamics)",
    "Privacy (data leakage, personal information, memorization)",
    "Environmental (energy usage, carbon footprint, compute costs)",
    "Legal / regulatory (compliance, intellectual property, liability)",
]

STAKEHOLDER_GROUPS = [
    "Developers",
    "End-users",
    "Researchers",
    "Policymakers",
    "Affected communities",
    "Children / minors",
    "Enterprises",
    "Educators",
]

EVAL_METHODOLOGIES = [
    "Automated benchmarks",
    "Human red-teaming",
    "External audits",
    "Domain-expert review",
    "Academic partnerships",
    "Bug bounties",
    "Adversarial testing",
    "User studies",
]

# ── Metric definitions ─────────────────────────────────────────────────
METRICS = {
    # ── Breadth ──
    "topic_coverage": {
        "dimension": "Comprehensiveness",
        "type": "extractive",
        "reference_list": TOPIC_CHECKLIST,
        "max_count": len(TOPIC_CHECKLIST),
    },
    "dangerous_capability_reporting": {
        "dimension": "Comprehensiveness",
        "type": "rubric",
    },
    "alignment_controllability": {
        "dimension": "Comprehensiveness",
        "type": "rubric",
    },
    "risk_category_breadth": {
        "dimension": "Comprehensiveness",
        "type": "extractive",
        "reference_list": RISK_CATEGORIES,
        "max_count": 9,
    },
    # stakeholder_diversity: dropped — not safety-relevant enough
    # ── Transparency ──
    "evidence_sufficiency": {
        "dimension": "Reasoning Quality",
        "type": "rubric",
    },
    "eval_reporting_quality": {
        "dimension": "Reasoning Quality",
        "type": "rubric",
    },
    "reasoning_depth": {
        "dimension": "Reasoning Quality",
        "type": "rubric",
    },
    "limitation_specificity": {
        "dimension": "Reasoning Quality",
        "type": "rubric",
    },
    "reasoning_consistency": {
        "dimension": "Reasoning Quality",
        "type": "rubric",
    },
    # ── 3rd-party Verification ──
    "external_validator_count": {
        "dimension": "3rd-party Verification",
        "type": "extractive",
        "reference_list": None,  # no fixed list — extract all
        "max_count": None,  # normalized by max_observed across models in aggregation step
        "score_field": "count",  # raw count — normalized to 0-100 post-hoc
    },
    "eval_type_diversity": {
        "dimension": "3rd-party Verification",
        "type": "extractive",
        "reference_list": EVAL_METHODOLOGIES,
        "max_count": len(EVAL_METHODOLOGIES),
    },
    "post_deployment_monitoring": {
        "dimension": "3rd-party Verification",
        "type": "rubric",
    },
}

DIMENSIONS = ["Comprehensiveness", "Reasoning Quality"]
ALL_DIMENSIONS = ["Comprehensiveness", "Reasoning Quality", "3rd-party Verification"]
# "3rd-party Verification" excluded from overall score — too influenced by
# companion report availability and document length. Still computed for reference.
