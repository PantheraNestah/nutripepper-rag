# NutriPepper Backend — Analysis

This document covers the problem analysis, requirements, knowledge base, the
evaluation methodology, the current measured results, and a critical analysis of
the system's strengths, gaps, and risks.

## 1. Problem Statement

General-purpose LLMs hallucinate, are not localized to Kenyan foods/contexts, and
can drift into unsafe medical advice. NutriPepper constrains an LLM with a curated,
authoritative knowledge base and a verification layer so that an embodied robot can
teach nutrition and food safety **safely, locally, and traceably**.

## 2. Stakeholders & Users

- **Learners** — children, students, community members asking nutrition questions.
- **Facilitators / teachers** — use the long-form text lessons to teach.
- **The robot (Pepper)** — consumes structured beats (speech + gesture + tablet).
- **Researchers / developers** — rely on logged interactions and the evaluation
  harness to measure grounding and retrieval quality.

## 3. Requirements

### 3.1 Functional

| ID | Requirement | Where realized |
|---|---|---|
| F1 | Answer a free-text nutrition question grounded in sources | `/query`, retriever, generator |
| F2 | Skip retrieval for conversational/greeting inputs | `is_conversational()` |
| F3 | Produce a robot-deliverable structured lesson (beats) | `/learn/structured` |
| F4 | Produce a long-form narrative lesson + quiz hints | `/learn/text` |
| F5 | Cite sources inline and report citation presence/count | `instrumentation.py` |
| F6 | Verify response grounding against retrieved context | `verifier.py` (background) |
| F7 | Persist every interaction with provenance | `interaction_logger.py`, DB models |
| F8 | Correlate interactions by session | `X-Session-ID` header |
| F9 | Expose request latency | `ProcessTimeMiddleware` |

### 3.2 Non-Functional (targets from the evaluation harness)

| Metric | Target | Source |
|---|---|---|
| Precision@3 | ≥ 0.85 | `evaluate.py` |
| Recall@5 | ≥ 0.90 | `evaluate.py` |
| MRR | ≥ 0.80 | `evaluate.py` |
| Source attribution rate | ≥ 0.90 | `evaluate.py` |
| Hallucination rate | < 0.05 | `evaluate.py` |
| Kenya domain relevance | ≥ 0.80 | `evaluate.py` |
| P95 latency | < 5000 ms | `evaluate.py` |
| Requests under 3 s | > 95% | `evaluate.py` |

Additional NFRs implied by design: modularity/explainability, safety (no medical
advice), and reproducible knowledge ingestion.

## 4. Knowledge Base Analysis

Six authoritative PDFs in `data/raw/`, tagged in `data/scripts/source_metadata.py`:

| Document | Authority | Year | Domain | Kenya-specific |
|---|---|---|---|---|
| Kenya Food Composition Tables | FAO/MoH Kenya | 2018 | nutrition | yes |
| Kenya Nutrient Profile Model (KNPM) | Kenya MoH | 2025 | nutrition | yes |
| Kenya Nutrition Action Plan 2018–2022 | Kenya MoH | 2018 | nutrition | yes |
| National Guidelines for Healthy Diets & Physical Activity | Kenya MoH | 2017 | nutrition | yes |
| WHO Healthy Diet Fact Sheet | WHO | 2024 | nutrition | no |
| FAO/WHO General Principles of Food Hygiene (Codex) | FAO/WHO Codex | 2023 | food_safety | no |

**Coverage observations:**
- Domain split is essentially **nutrition** vs **food_safety**; "sustainability" is
  referenced in code (`evaluate.py`) but is not a tagged domain in the corpus.
- Kenya specificity is concentrated in the four MoH/FAO-Kenya documents.
- PDF extraction + aggressive cleaning (see `03-implementation.md`) strips standalone
  numbers, which degrades tabular/quantitative content (e.g. food composition values).

## 5. Evaluation Methodology

A three-script offline pipeline:

1. **`generate_test_set.py`** — pulls all chunks from ChromaDB, stratified-samples
   ~30 across domains, and uses GPT-3.5 to write one natural question per chunk
   (giving a guaranteed ground-truth chunk id). Adds 10 hand-written Kenya-specific
   queries → 40-query `test_set.json` (seeded `random.seed(42)` for reproducibility).
2. **`seed_interactions.py`** — replays the test set against the live `/query`
   endpoint so the verifier populates grounding columns in the DB.
3. **`evaluate.py`** — computes four families of metrics:
   - **Retrieval** (auto queries only): Precision@K, Recall@K, MRR, plus a per-domain
     breakdown. Note: with one ground-truth chunk per query, Precision@K == Recall@K
     here (binary hit), so these mirror each other by construction.
   - **Grounding** (from DB): source-attribution rate, grounding-score stats,
     contradiction rate, label distribution, `estimated_hallucination_rate = 1 − mean_score`.
   - **System performance** (from DB): mean/median/p95 latency, % under 3 s, per endpoint.
   - **Kenya-specific** (manual queries): domain relevance, Kenyan-term match rate,
     source diversity.

## 6. Current Measured Results

From `evaluation/results/evaluation_summary.txt` (generated 2026-06-10):

### Retrieval quality (30 auto queries)

| Metric | Value | Target | Met? |
|---|---|---|---|
| Precision@1 | 0.367 | — | — |
| Precision@3 | 0.533 | ≥ 0.85 | ✗ |
| Precision@5 | 0.567 | — | — |
| Recall@3 | 0.533 | — | — |
| Recall@5 | 0.567 | ≥ 0.90 | ✗ |
| MRR | 0.446 | ≥ 0.80 | ✗ |

Per domain: **food_safety** MRR 0.636 / Recall@3 0.733 is markedly stronger than
**nutrition** MRR 0.256 / Recall@3 0.333.

### Grounding (only 5 verified interactions)

| Metric | Value | Target | Met? |
|---|---|---|---|
| Source attribution | 0.60 | ≥ 0.90 | ✗ |
| Mean grounding score | 0.457 | — | — |
| Contradiction rate | 0.20 | — | — |
| Est. hallucination | 0.543 | < 0.05 | ✗ |
| Avg claims checked | 8.33 | — | — |
| Avg claims supported | 0.67 | — | — |
| Label dist. | grounded 2 / partial 1 / ungrounded 2 | — | — |

### System performance

| Endpoint | Requests | Mean | Median | P95 | < 3 s |
|---|---|---|---|---|---|
| query | 7 | 44,592 ms | 6,109 ms | 276,051 ms | 0% |
| learn | 4 | 11,495 ms | 7,167 ms | 27,068 ms | 0% |

### Kenya-specific (10 manual queries)

| Metric | Value | Target | Met? |
|---|---|---|---|
| Domain relevance | 1.00 | ≥ 0.80 | ✓ |
| Kenya content match | 0.666 | — | — |
| Avg source diversity | 1.0 | — | — |

## 7. Interpretation of Results

- **Retrieval below target.** Likely contributors: aggressive number-stripping in
  cleaning harms nutrition/quantitative chunks; single-ground-truth metric is harsh
  (a semantically-correct neighbouring chunk scores 0); nutrition corpus is larger and
  more tabular, depressing nutrition-domain scores vs the more prose-like food-safety
  corpus.
- **Grounding sample is tiny (n=5).** The headline 0.54 "hallucination" figure is
  `1 − mean_grounding_score`, a conservative proxy heavily influenced by the strict
  NLI thresholds and tabular chunks that NLI handles poorly. It should be read as a
  directional signal, not a true hallucination rate.
- **Latency far over target.** The biggest gap. LLM calls run **synchronously inside
  the request**; cold model loads (MPNet, cross-encoder, NLI) and network round-trips
  to OpenAI dominate. p95 of 276 s indicates at least one cold-start/timeout outlier.
- **Kenya relevance strong.** Domain relevance 1.0 and source diversity confirm the
  corpus + retriever return on-topic Kenyan content for local queries.

## 8. Strengths

- Clear separation of concerns; each stage is independently testable.
- Genuinely hybrid retrieval (semantic + lexical + cross-encoder) — a strong baseline.
- Grounding verification + instrumentation give real, queryable telemetry.
- Honest, target-driven evaluation harness with reproducible test-set generation.
- Locally relevant, authoritative corpus with provenance preserved end-to-end.

## 9. Gaps, Risks & Limitations

| Area | Issue | Impact |
|---|---|---|
| Latency | Synchronous LLM + cold model loads | Fails real-time robot UX targets |
| Retrieval | Number-stripping in cleaning; harsh single-GT metric | Low nutrition Precision/Recall |
| Grounding eval | n=5 sample; NLI weak on tables | Unreliable hallucination estimate |
| Safety | Enforced only by prompt text, no classifier | Possible unsafe outputs slip through |
| Config | `DEBUG` is a string; minimal settings | Limited environment control |
| Provenance | Metadata key mismatch (`source` vs `source_document`) | Risk of `source = "unknown"` (see impl doc) |
| Docs drift | README/legacy doc describe a dummy pipeline | Misleading without this docs set |
| Security | No auth, open CORS, no rate limiting | Not production-hardened |

## 10. Recommendations (prioritised)

1. **Cut latency**: warm models at startup (already partly done in `evaluate.py`),
   cache the OpenAI client, consider async/streaming, and move heavy model loads out
   of the first request.
2. **Improve nutrition retrieval**: relax number-stripping in `clean_text`, preserve
   numeric/tabular content, and evaluate with graded/multi-chunk relevance instead of
   a single ground-truth id.
3. **Grow grounding sample**: seed all 40 queries and re-run; report on ≥ 30 verified
   interactions before drawing conclusions.
4. **Add an explicit safety layer**: intent classification for medical/diagnosis/
   out-of-scope requests, applied before generation.
5. **Fix provenance metadata** so `source/authority/domain` are consistently present
   (see `03-implementation.md` §"Known issues").
6. **Harden config/security** for any deployment beyond the lab.
