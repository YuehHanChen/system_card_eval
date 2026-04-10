# System Card Eval: Full Report

## Summary

We evaluated 15 frontier model system cards from Anthropic (6), OpenAI (5), and Google (4) across 9 metrics spanning two dimensions: **Comprehensiveness** (what topics are covered) and **Reasoning Quality** (how well claims are backed up). Each metric was scored by 3 LLM judges (Claude Sonnet 4.6, GPT-5.4, Gemini 3.1 Pro).

**Overall scores:** Anthropic 79, OpenAI 71, Google 57.

---

## 1. Methodology

### 1.1 System Cards Collected

| Company | Models | Cards | Companion Reports |
|---|---|---|---|
| Anthropic (6) | Claude 4, Opus 4.5, Haiku 4.5, Sonnet 4.6, Opus 4.6, Mythos Preview | 120–244 pages | ASL-3 Report, Pilot Sabotage Report, Feb 2026 Risk Report, Opus 4.6 Sabotage Report, Mythos Risk Report |
| OpenAI (5) | GPT-4o, o1, GPT-5, GPT-5.3 Codex, GPT-5.4 Thinking | 31–60 pages | None |
| Google (4) | Gemini 2.5 Pro, 3 Pro, 3 Flash, 3.1 Pro | 6–21 pages | Gemini 2.5 Technical Report (73p), FSF Report (26p), Model Evaluation Reports |

Where available, companion reports (risk reports, safety framework reports, technical papers) were concatenated with the system card and evaluated as a single unit.

### 1.2 Metrics

**Comprehensiveness** — what topics are covered:

| Metric | Type | Description |
|---|---|---|
| Topic coverage | Extractive | Fraction of 16 canonical topics present (derived bottom-up from all 12 cards) |
| Dangerous capability reporting | Rubric | How thoroughly CBRN, cyber, autonomy, persuasion are assessed (0/25/50/75/100) |
| Alignment & controllability | Rubric | Coverage of refusals, jailbreaks, instruction hierarchy (0/25/50/75/100) |
| Risk category comprehensiveness | Extractive | How many of 9 risk categories are discussed |
| Stakeholder diversity | Extractive | How many of 8 stakeholder groups are addressed |

**Reasoning Quality** — how well are claims backed up:

| Metric | Type | Description |
|---|---|---|
| Evidence sufficiency | Rubric | What fraction of claims are backed by data (0/25/50/75/100) |
| Eval reporting quality | Rubric | Are evals documented with methodology and breakdowns (0/25/50/75/100) |
| Reasoning reasoning depth | Rubric | Is the *why* behind decisions explained with tradeoffs (0/25/50/75/100) |
| Limitation specificity | Rubric | Are limitations concrete or generic disclaimers (0/25/50/75/100) |
| Reasoning consistency | Rubric | Internal consistency across sections (0/25/50/75/100) |

### 1.3 Scoring

- **Extractive metrics**: LLM judges extract items with verbatim quotes, then count. Score = (count / reference max) x 100.
- **Rubric metrics**: LLM judges select one of {0, 25, 50, 75, 100} from a rubric with concrete anchors.
- **Overall score**: Mean of all 10 metric scores (equal weight per metric, not per dimension).
- **Error bars**: Standard error = std across 6 data points / sqrt(6).

### 1.4 Topic Checklist Derivation

The 16-topic checklist was derived bottom-up, not prescribed:
1. Extracted 635 section headings (level 1-2) across all 12 system cards
2. Clustered into canonical topics using an LLM
3. Retained topics appearing in at least 2 companies' cards
4. Removed "Model introduction & overview" (100% coverage, no discriminative power)

### 1.5 Handling Large Documents

Documents exceeding a judge's context window trigger **agentic prefetch mode**:
1. Each judge reviews the table of contents and requests relevant pages
2. Page requests are unioned across all judges into a fixed page set per (model, metric) pair
3. All judges score using the same pages — eliminating page-selection variance

---

## 2. Overall Results

![Report Card](results/report_card.png)

### 2.1 Company Rankings

| Company | Overall | Comprehensiveness | Reasoning Quality | Range |
|---|---|---|---|---|
| Anthropic | 78.8 ± 4.4 | 86.0 | 76.2 | 70.2 – 82.8 |
| OpenAI | 70.7 ± 3.5 | 73.8 | 70.5 | 66.2 – 77.0 |
| Google | 56.8 ± 17.1 | 64.4 | 54.8 | 32.9 – 76.4 |

Anthropic leads on both dimensions. Google has the highest variance (std=17.1) — their best card (Gemini 2.5 Pro, 76.4) is competitive with Anthropic's mid-tier cards, but their worst (Gemini 3 Flash, 32.9) is far below.

### 2.2 Model Rankings

![Overall Ranking](results/overall_ranking.png)

| Rank | Model | Company | Overall | Comprehensiveness | Reasoning Quality |
|---|---|---|---|---|---|
| 1 | Claude Opus 4.5 | Anthropic | 82.8 | 84.7 | 80.8 |
| 2 | Claude Opus 4.6 | Anthropic | 81.9 | 84.7 | 79.2 |
| 3 | Claude 4 | Anthropic | 80.9 | 86.0 | 75.8 |
| 4 | Mythos Preview | Anthropic | 80.8 | 85.4 | 76.2 |
| 5 | GPT-5 | OpenAI | 77.0 | 78.9 | 75.0 |
| 6 | Claude Sonnet 4.6 | Anthropic | 76.4 | 78.6 | 74.2 |
| 7 | Gemini 2.5 Pro | Google | 76.4 | 80.7 | 72.0 |
| 8 | GPT-5.4 Thinking | OpenAI | 70.3 | 69.0 | 71.7 |
| 9 | Claude Haiku 4.5 | Anthropic | 70.2 | 67.1 | 73.3 |
| 10 | o1 | OpenAI | 70.2 | 70.4 | 70.0 |
| 11 | GPT-4o | OpenAI | 69.8 | 71.3 | 68.3 |
| 12 | Gemini 3 Pro | Google | 69.1 | 68.5 | 69.7 |
| 13 | GPT-5.3 Codex | OpenAI | 66.2 | 60.0 | 72.3 |
| 14 | Gemini 3.1 Pro | Google | 48.6 | 49.7 | 47.5 |
| 15 | Gemini 3 Flash | Google | 32.9 | 34.1 | 31.7 |

---

## 3. Key Findings

### 3.1 System cards are not getting better over time

![Overall Over Time](results/overall_over_time.png)

Despite growing public attention to AI safety documentation, system card quality is not consistently improving:

- **Google's Gemini 3.1 Pro** (Feb 2026) scored 48.6 — worse than **Gemini 2.5 Pro** (Jun 2025) at 76.4. The older card had a 73-page companion technical report; the newer one has only a 3-page evaluation addendum.
- **OpenAI's GPT-5.3 Codex** (Feb 2026) scored 66.2 — below **GPT-4o** (Aug 2024) at 69.8 and **o1** (Sep 2024) at 70.2. Codex-specific cards tend to be narrower in scope.
- **Anthropic** is the only company that has maintained scores consistently above 75, with all 6 cards scoring between 70.2 and 82.8.

### 3.2 Topic coverage varies dramatically by company

![Topic Coverage](results/topic_coverage.png)

Across 16 safety-relevant topics derived from the cards themselves:

**Anthropic (92% average)**: Covers 100% on 12 of 16 topics. The only company that consistently addresses model welfare & moral status, reward hacking evaluations, and chain-of-thought reasoning transparency. Weakest on environmental impact (44%) and intended use/limitations (65%).

**OpenAI (74% average)**: Strong on safety evaluations, dangerous capabilities, and alignment. But inconsistent — GPT-5 covers 85% while GPT-5.3 Codex covers only 58%. Zero coverage on model welfare. Weak on implementation/sustainability (37%) and reward hacking (33%).

**Google (64% average)**: Strong where others are weak — 100% on environmental/sustainability and multilingual performance. But weak on bias evaluations (22%), reward hacking (22%), chain-of-thought transparency (28%), and model welfare (0%).

**Universal gaps**: No company consistently covers all topics. Model welfare (only Anthropic), implementation/sustainability (only Google at 100%), and reward hacking remain the most under-reported.

### 3.3 Per-metric breakdown

**Metrics where companies differ most** (highest cross-company variance):

| Metric | Anthropic | OpenAI | Google | Spread |
|---|---|---|---|---|
| Topic coverage | 92.5 | 73.8 | 64.4 | 28.1 |
| Limitation specificity | 81.6 | 75.8 | 42.7 | 38.9 |
| Stakeholder diversity | 74.5 | 46.3 | 45.0 | 29.5 |
| Reasoning reasoning depth | 76.0 | 67.5 | 49.6 | 26.4 |
| External validator count | 56.8 | 40.3 | 12.2 | 44.6 |

**Metrics where companies are similar** (lowest variance):

| Metric | Anthropic | OpenAI | Google | Spread |
|---|---|---|---|---|
| Reasoning consistency | 71.9 | 67.5 | 62.9 | 9.0 |
| Evidence sufficiency | 72.6 | 69.8 | 57.3 | 15.3 |
| Dangerous capability reporting | 92.0 | 94.2 | 72.9 | 21.3 |

Notable: **OpenAI leads on dangerous capability reporting** (94.2 vs Anthropic's 92.0). This is the only metric where OpenAI outscores Anthropic.

### 3.4 "Show Your Work" — evidence vs safety claims

![Show Your Work](results/show_your_work.png)

Plotting evidence sufficiency against dangerous capability reporting reveals a clear pattern:
- Most Anthropic and OpenAI models cluster in the **top-right** (cover dangerous capabilities AND back up claims with evidence)
- Google's Gemini Flash and 3.1 Pro sit in the **bottom-left** (limited coverage AND limited evidence)
- No models sit in the **bottom-right** (talking about danger without evidence) — companies that discuss dangerous capabilities tend to also provide data

### 3.5 What changed? Oldest vs newest card

![What Changed](results/what_changed.png)

Comparing each company's oldest and newest system card:

- **Anthropic** (Claude 4 → Mythos Preview): Improved on dangerous capability reporting (+4), risk category comprehensiveness (+9), and eval reporting quality (+6). But regressed on stakeholder diversity (-21) and evidence sufficiency (-8).
- **OpenAI** (GPT-4o → GPT-5.4 Thinking): Big gains on risk category comprehensiveness (+8) and stakeholder diversity (+17). Regressions on evidence sufficiency and eval reporting quality.
- **Google** (Gemini 2.5 Pro → 3.1 Pro): Improved on stakeholder diversity (+11). But regressed on most other metrics — evidence sufficiency, reasoning reasoning depth, and limitation specificity all dropped.

---

## 4. Reliability & Bias Checks

### 4.1 Inter-Annotator Agreement

12 of 13 metrics achieved reliable agreement (Krippendorff's alpha >= 0.4):

| Metric | Alpha | Reliable |
|---|---|---|
| Topic coverage | 0.921 | Yes |
| Eval type diversity | 0.921 | Yes |
| External validator count | 0.885 | Yes |
| Limitation specificity | 0.876 | Yes |
| Stakeholder diversity | 0.854 | Yes |
| Risk category comprehensiveness | 0.813 | Yes |
| Post-deployment monitoring | 0.806 | Yes |
| Dangerous capability reporting | 0.798 | Yes |
| Eval reporting quality | 0.782 | Yes |
| Reasoning reasoning depth | 0.749 | Yes |
| Alignment & controllability | 0.432 | Yes |
| Evidence sufficiency | 0.421 | Yes |
| Reasoning consistency | 0.068 | **No** |

**Reasoning consistency** (alpha=0.068) is essentially unreliable — judges disagree strongly on whether cards contain internal contradictions. This metric should be interpreted with caution.

Extractive metrics (topic coverage, eval type diversity, external validator count) have the highest agreement, as expected — counting items is more objective than judging quality.

### 4.2 Self-Evaluation Bias Check

Each judge evaluates cards from its own company. We tested whether judges inflate their own company's scores:

| Target Company | Sonnet 4.6 (Anthropic) | GPT-5.4 (OpenAI) | Gemini 3.1 Pro (Google) |
|---|---|---|---|
| Anthropic cards | 76.4 | 73.5 | **79.6** |
| OpenAI cards | 68.8 | 69.9 | 70.3 |
| Google cards | 52.4 | **54.6** | 53.8 |

**No self-inflation detected.** If anything, the pattern is the opposite:
- Gemini rates Anthropic's cards highest (79.6), not its own
- GPT-5.4 rates Google slightly higher than the other judges do
- All three judges roughly agree on OpenAI (68.8-70.3)

Score differences across companies reflect genuine quality differences, not judge bias.

---

## 5. Limitations

1. **Companion report asymmetry**: Anthropic cards benefit from 2-5 companion reports (150+ extra pages). Google's scores jump 22 points (Gemini 2.5 Pro) when including the technical report. This structural difference affects scores.

2. **One unreliable metric**: Reasoning consistency (alpha=0.068) shows near-zero inter-annotator agreement and should be treated as noise.

3. **No human validation**: All scoring is LLM-based. We report inter-annotator agreement but have no human baseline to calibrate against.

4. **Ordinal rubric treated as interval**: Rubric levels (0/25/50/75/100) are ordinal, but we compute means and standard deviations as if they're interval data.

5. **Judge contamination**: LLM judges were likely trained on these system cards. For large documents in agentic mode, judges only see a subset of pages but might "recall" content from training data.

6. **Sample size**: 12 system cards across 3 companies. Adding cards from xAI (Grok), Meta (Llama), and others would strengthen generalizability.

---

## 6. Conclusion

Anthropic consistently produces the most comprehensive and well-evidenced system cards. OpenAI is a solid second, with GPT-5 being their strongest card. Google trails significantly, though their 73-page Gemini 2.5 technical report shows they can match Anthropic-level comprehensiveness when they choose to.

The most concerning finding is that system card quality is not improving over time. Companies appear to invest heavily in flagship model cards while treating incremental releases (Codex variants, Flash models) with lighter documentation.

Three topics remain chronically under-covered across the industry: model welfare & moral status, environmental impact, and reward hacking evaluations. These represent the biggest opportunities for improving AI transparency.
