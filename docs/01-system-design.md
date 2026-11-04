# NutriPepper Backend — System Design

> Status: reflects the codebase as implemented (not the older skeleton described in
> `README.md` / `nutripepper_backend_comprehensive_documentation.md`). Where the code
> and the legacy docs disagree, this document follows the code.

## 1. Purpose

NutriPepper is the backend and Retrieval-Augmented Generation (RAG) orchestration
service for an embodied nutrition-education robot ("Pepper") at the JKUAT Social
Robotics Lab. It answers nutrition, food-safety, and sustainable-food-practice
questions and produces teaching lessons that are **grounded strictly in an
authoritative, Kenya-focused knowledge base**.

Three design pillars drive every decision:

1. **Grounding / anti-hallucination** — answers must be traceable to retrieved source text.
2. **Safety** — the assistant is an educator, never a clinician (no diagnosis/prescription).
3. **Transparency & evaluability** — every interaction is logged, verified, and measurable.

The pipeline is **hand-orchestrated** (no LangChain/LlamaIndex) so that retrieval,
generation, verification, and logging stay individually inspectable.

## 2. High-Level Architecture

```text
                          ┌──────────────────────────────────────────────┐
                          │                 Client / Robot                │
                          │  (HTTP + optional X-Session-ID header)         │
                          └───────────────────────┬──────────────────────┘
                                                  │  POST /query
                                                  │  POST /learn/structured
                                                  │  POST /learn/text
                                                  ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ FastAPI app (app/main.py)                                                       │
│   • CORS middleware (open)        • ProcessTimeMiddleware (X-Process-Time)       │
│   • lifespan → init_db()          • router (app/api/routes.py)                   │
└───────────────────────────────────────────┬───────────────────────────────────┘
                                             │ Depends(get_pipeline)
                                             ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ RAGPipeline (app/services/pipeline.py)                                          │
│                                                                                 │
│  preprocess → [conversational shortcut?] → retrieve → generate → instrument     │
└───────┬───────────────────────────────┬───────────────────────────┬───────────┘
        │                               │                           │
        ▼                               ▼                           ▼
┌───────────────┐          ┌──────────────────────────┐   ┌────────────────────┐
│ retriever.py  │          │ generator.py             │   │ instrumentation.py │
│ 3-stage hybrid│          │ OpenAI chat completions  │   │ citation regex     │
│ Semantic+BM25 │          │ (gpt-3.5 / gpt-4-turbo)  │   │ (count + present)  │
│ +RRF+reranker │          └──────────────────────────┘   └────────────────────┘
│      │        │
│      ▼        │
│ ChromaDB +    │
│ MPNet + BM25  │
└───────────────┘
                                             │
              response returned to client    │  (synchronously)
              ───────────────────────────────┘
                                             │
                                             ▼  background_tasks (after response sent)
┌───────────────────────────────────────────────────────────────────────────────┐
│ interaction_logger.py                                                           │
│   /query  → verifier.verify_response() [NLI grounding] → persist                │
│   /learn  → persist lesson + chunks                                             │
│                                                                                 │
│   → SQLAlchemy async → SQLite (nutripepper.db)                                  │
│        Interaction · QueryDetail · LearnDetail · RetrievedChunk                 │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Offline / build-time path (not in the request loop)

```text
PDFs (data/raw/*.pdf)
  → ingest_documents.py  (PyMuPDF extract → clean → NLTK sentence chunk 100–220 words → metadata)
  → data/knowledge_base/nutrition_knowledge_base.json
  → build_vector_store.py (MPNet embed → ChromaDB collection "nutrition_kb")
  → app/chroma_storage/  (persistent vector store used at runtime)
```

### Evaluation path (offline, separate from serving)

```text
generate_test_set.py → test_set.json → seed_interactions.py (hits live API)
  → DB populated → evaluate.py → evaluation/results/{evaluation_report.json, evaluation_summary.txt}
```

## 3. Component Responsibilities

| Layer | Module | Responsibility |
|---|---|---|
| App / wiring | `app/main.py` | FastAPI app, middleware, lifespan DB init |
| API | `app/api/routes.py` | 3 endpoints, timing, schedules background logging |
| Orchestration | `app/services/pipeline.py` | `RAGPipeline` — preprocess → retrieve → generate → instrument |
| Retrieval | `app/services/retriever.py` | 3-stage hybrid retrieval (semantic + BM25 + RRF + cross-encoder) |
| Generation | `app/services/generator.py` | OpenAI prompting for query answers + two lesson modes |
| Verification | `app/core/verifier.py` | NLI-based claim grounding (async, post-response) |
| Instrumentation | `app/core/instrumentation.py` | Cheap citation metrics at response time |
| Persistence | `app/database/` | Async SQLAlchemy models, engine, session, `init_db` |
| Logging sink | `app/services/interaction_logger.py` | Background verify + persist for every request |
| Schemas | `app/schemas/` | Pydantic request/response contracts |
| Config / logging | `app/core/config.py`, `app/core/logger.py` | Settings + logger |
| Middleware | `app/middleware/timer.py` | `X-Process-Time` header |
| Embedding util | `app/services/embedder.py` | Standalone MPNet encoder helper |
| Knowledge build | `data/scripts/` | PDF ingestion + vector store builder |
| Evaluation | `evaluation/` | Test-set generation, seeding, metrics |

## 4. Request Flows

### 4.1 `POST /query`

1. Route records `start = time.monotonic()`.
2. `RAGPipeline.run_query()`:
   - `preprocess()` → strip + lowercase.
   - `is_conversational()` — if the query matches a greeting/meta trigger
     (`hello`, `who are you`, `thanks`, …), retrieval is **skipped** and the LLM
     answers with empty context.
   - Otherwise `retrieve()` returns top-K grounded chunks, then `generate_response()`.
   - `instrument_response()` adds `citations_present` / `citation_count`.
3. Route computes `latency_ms` and returns the clean JSON payload.
4. A **background task** (`log_query_interaction`) runs `verify_response()` (NLI
   grounding) and writes `Interaction` + `QueryDetail` + `RetrievedChunk` rows.
   Verification deliberately runs **after** the response is sent so model latency
   never blocks the robot's reply.

### 4.2 `POST /learn/structured`

- `run_structured_lesson()` retrieves `top_k=6` chunks and asks the LLM (JSON mode)
  for a 4–5 "beat" lesson. Each beat carries `speech` (robot-speakable text),
  `gesture` (NAOqi animation tag), and a `tablet` display object.
- `_sanitize_beats()` guarantees every beat has a valid tablet object.
- Background task persists a `LearnDetail` (mode `structured`).

### 4.3 `POST /learn/text`

- `run_text_lesson()` retrieves `top_k=10` chunks and asks GPT-4-turbo (JSON mode)
  for a long-form (~2,500–3,000 word) narrative lesson plus `quiz_hints`.
- Optional `description` steers audience/focus.
- Background task persists a `LearnDetail` (mode `text`).

## 5. Retrieval Design (the core IR contribution)

Three sequential stages, all models loaded once as module-level singletons:

| Stage | Technique | Model / params | Output |
|---|---|---|---|
| 1. Semantic | Dense vector similarity | `all-mpnet-base-v2` via ChromaDB | top 10 |
| 2. Lexical + fusion | BM25 keyword search fused with stage 1 | `rank_bm25` + Reciprocal Rank Fusion | top 15 candidates |
| 3. Rerank | Cross-encoder relevance | `cross-encoder/ms-marco-MiniLM-L-6-v2` | top `k` |

**Reciprocal Rank Fusion (RRF):**
`score = 0.6 · 1/(60 + semantic_rank) + 0.4 · 1/(60 + bm25_rank)`.
The constant `k=60` damps outlier ranks; the 0.6/0.4 weighting favours semantic
understanding while letting exact-term (BM25) matches influence ordering — important
for nutrient names, food terms, and tabular data.

The final chunks are stripped to `{text, source, authority, domain}` before being
handed to the generator (internal scores are dropped).

## 6. Generation Design

`generator.py` calls models through `app/services/llm.py`, a thin provider layer
(Anthropic, DeepSeek, OpenAI). Each task asks for a **tier**; the tier→model mapping is
configured with `LLM_MODEL_COMPLEX|MEDIUM|EASY=provider:model[:effort]`. Three prompt
programs share a context-block builder that numbers sources `[1], [2], …` so the model
can cite inline.

| Task | Tier | Default model | Max tokens* | Format |
|---|---|---|---|---|
| `/query` (grounded) | medium | `claude-haiku-5-5`, effort medium | 2000 | free text w/ inline citations |
| `/query` (small talk, no context) | easy | `claude-haiku-5-5`, effort medium | 2000 | free text |
| `/learn/structured` | medium | `claude-haiku-5-5`, effort medium | 6000 | JSON object (beats) |
| `/learn/text` | complex | `claude-sonnet-5-5`, effort medium | 10000 | JSON object (lesson + hints) |

\*Includes reasoning tokens. Anthropic's current models accept no `temperature`/`top_p`;
the generator's temperatures apply only to OpenAI-format providers (DeepSeek with effort
`off`, OpenAI). Evaluation results in this documentation were produced with the earlier
OpenAI models (gpt-3.5-turbo / gpt-4-turbo) and have not been re-run on the new defaults.

All system prompts enforce: answer only from VERIFIED CONTEXT, never fabricate
values, no medical advice, prefer Kenyan food examples, and (for lessons) emit
robot-speakable plain text. Every path has a graceful fallback on LLM/JSON failure.

## 7. Verification & Grounding Design

`verifier.py` implements a post-hoc grounding check:

1. **Claim extraction** — split response into sentences, drop short/meta sentences,
   strip citation markers.
2. **NLI scoring** — `cross-encoder/nli-MiniLM2-L6-H768` scores each (chunk, claim)
   pair as `[contradiction, entailment, neutral]`.
   - contradiction > 0.60 → `contradicted`
   - entailment > 0.45 → `entailed`
3. **Keyword-overlap fallback** — when NLI is unconfident (common for tabular
   chunks), content-word overlap ≥ 0.35 counts as weakly `entailed`.
4. **Grounding report** — aggregate to `grounding_score = supported/total` and a
   label (`grounded` ≥ 0.75, `partial` ≥ 0.45 or any contradiction, else `ungrounded`).

This produces the columns the evaluation harness reads to estimate hallucination
rate, contradiction rate, and source attribution.

## 8. Persistence Design

Async SQLAlchemy 2.0 over SQLite (`aiosqlite`), schema managed by Alembic.

```text
Interaction (1) ──┬── (0..1) QueryDetail        # /query payload + grounding metrics
                  ├── (0..1) LearnDetail         # /learn payload (structured or text)
                  └── (0..N) RetrievedChunk       # provenance of each retrieved chunk
```

`Interaction` carries `endpoint`, `session_id`, `latency_ms`, `is_conversational`.
`QueryDetail` stores instrumentation (`citations_present/count`) plus verifier output
(`grounding_score/label`, `has_contradiction`, `claims_checked/supported`).
`LearnDetail` stores either beat JSON (`lesson_json`, `total_beats`, `beat_types`) or
narrative (`lesson_text`, `quiz_hints`, `description`).

## 9. Key Design Decisions & Rationale

- **Manual pipeline over frameworks** — full control of retrieval/grounding/safety;
  easier to evaluate and reason about for a health-adjacent domain.
- **Hybrid retrieval + reranking** — dense embeddings alone miss exact terms;
  BM25 + RRF + cross-encoder improves precision on nutrient/food terminology.
- **Background verification** — NLI is expensive; running it after the response keeps
  the robot responsive while still producing grounding telemetry.
- **Singletons for ML models** — embedding/reranker/NLI models load once at module
  import, not per request.
- **Structured vs. text lessons** — structured serves the robot (speech + gesture +
  tablet beats); text serves long-form study/evaluation.
- **Session correlation via header** — `X-Session-ID` ties multiple interactions to a
  session without a user/auth system.

## 10. Non-Goals / Out of Scope (current)

- No authentication/authorization or rate limiting.
- No streaming responses; LLM calls are synchronous within the request.
- No live robot (NAOqi) bridge — gestures/tablet are produced as data only.
- No safety classifier module yet (safety is enforced only via prompt instructions).

See `02-analysis.md` for requirements/evaluation analysis and `03-implementation.md`
for file-level implementation detail and known issues.
