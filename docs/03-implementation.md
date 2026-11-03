# NutriPepper Backend — Implementation

File-level reference for how the system is built, plus setup, the HTTP contract, the
database schema/migrations, and known issues. Pairs with `01-system-design.md`
(architecture) and `02-analysis.md` (requirements/evaluation).

## 1. Tech Stack

| Concern | Choice |
|---|---|
| Web framework | FastAPI `0.136` (ASGI, async) + Uvicorn |
| Validation | Pydantic v2 |
| Vector DB | ChromaDB `1.5` (PersistentClient) |
| Embeddings | `sentence-transformers` → `all-mpnet-base-v2` |
| Lexical search | `rank_bm25` (BM25Okapi) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| NLI verifier | `cross-encoder/nli-MiniLM2-L6-H768` |
| LLM | OpenAI Chat Completions (`gpt-3.5-turbo`, `gpt-4-turbo`) |
| ORM / DB | SQLAlchemy 2.0 async + `aiosqlite` (SQLite) |
| Migrations | Alembic |
| PDF / NLP | PyMuPDF (`fitz`), NLTK |
| Config | `python-dotenv` |

> `requirements.txt` also pins `google-generativeai`, but the runtime generator uses
> OpenAI. Gemini is not wired in.

## 2. Project Layout

```text
nutripepper_backend/
├── app/
│   ├── main.py                     # FastAPI app, middleware, lifespan
│   ├── api/routes.py               # /query, /learn/structured, /learn/text
│   ├── services/
│   │   ├── pipeline.py             # RAGPipeline orchestrator
│   │   ├── retriever.py            # 3-stage hybrid retrieval
│   │   ├── generator.py            # OpenAI prompting (query + 2 lesson modes)
│   │   ├── embedder.py             # standalone MPNet helper
│   │   └── interaction_logger.py   # background verify + persist
│   ├── core/
│   │   ├── config.py               # Settings (env)
│   │   ├── logger.py               # logging config
│   │   ├── verifier.py             # NLI grounding
│   │   ├── instrumentation.py      # citation metrics
│   │   └── exceptions.py           # error JSON helper
│   ├── database/
│   │   ├── db.py                   # async engine/session, init_db
│   │   └── models.py               # ORM models
│   ├── schemas/
│   │   ├── query.py                # request models
│   │   └── response.py             # response + tablet/beat models
│   ├── dependencies/pipeline.py    # get_pipeline DI provider
│   ├── middleware/timer.py         # X-Process-Time
│   ├── utils/preprocess.py         # query normalisation
│   └── chroma_storage/             # runtime ChromaDB store (used by retriever)
├── data/
│   ├── raw/                        # 6 source PDFs
│   ├── knowledge_base/nutrition_knowledge_base.json
│   └── scripts/
│       ├── ingest_documents.py     # PDF → chunks JSON
│       ├── source_metadata.py      # per-document metadata
│       ├── build_vector_store.py   # JSON → ChromaDB
│       └── embedder2.py            # MPNet helper for the builder
├── evaluation/                     # test set, seeding, metrics, results
├── alembic/                        # migrations
├── chromadb/                       # secondary chroma sqlite (see known issues)
├── nutripepper.db                  # runtime SQLite DB
├── requirements.txt · run.sh · VERSION · .env
```

## 3. Application Wiring (`app/main.py`)

- Creates `FastAPI(title, version, lifespan)`. `settings` come from `app/core/config.py`.
- `lifespan` calls `await init_db()` once on startup to create tables.
- Middleware: open `CORSMiddleware` (all origins/methods/headers) +
  `ProcessTimeMiddleware`.
- Router included at root (no `/api` prefix).

## 4. API Contract (`app/api/routes.py`)

All endpoints accept an optional `X-Session-ID` header and schedule a background
logging task after returning. Latency is timed with `time.monotonic()`.

### `POST /query`

Request:

```json
{ "user_query": "Is ugali healthy for breakfast?" }
```

Response (`QueryResponse`):

```json
{
  "status": "success",
  "message": "Query processed successfully",
  "data": {
    "original_query": "Is ugali healthy for breakfast?",
    "processed_query": "is ugali healthy for breakfast?",
    "retrieved_context": [
      { "text": "...", "source": "...", "authority": "...", "domain": "..." }
    ],
    "generated_response": "... [1] ...",
    "citations_present": true
  }
}
```

`grounding_*` fields are **not** in the response — they are computed asynchronously
and stored in the DB only.

### `POST /learn/structured`

Request: `{ "concept": "iron deficiency in children" }` →
`data` is a lesson object: `{ concept, total_beats, beats[], sources[] }`, where each
beat is `{ beat_number, type, speech, gesture, tablet }`.

Beat `type` ∈ `introduction | kenyan_context | key_facts | safety_tip |
practical_advice | quiz`. `gesture` ∈ `explaining | pointing_tablet | tip_gesture |
warning_gesture | thinking | welcoming`. Tablet `display_type` ∈ `key_fact |
food_list | tip | warning | quiz`.

### `POST /learn/text`

Request: `{ "concept": "...", "description": "optional focus" }` →
`data` is `{ concept, lesson_text, quiz_hints[], sources[] }`.

## 5. Orchestration (`app/services/pipeline.py`)

`RAGPipeline` exposes three async methods. `preprocess()` (`app/utils/preprocess.py`)
just strips + lowercases. `CONVERSATIONAL_TRIGGERS` is a set of greeting/meta phrases;
`is_conversational()` matches against the normalised query to bypass retrieval.
`top_k` differs per task: `/query` → 3, structured → 6, text → 10.

## 6. Retrieval (`app/services/retriever.py`)

- **Lazy singletons**: `_collection`, `_embedding_model`, `_reranker`, `_bm25`
  (+ `_bm25_chunks`). The BM25 index is built once from all ChromaDB documents.
- `_semantic_search`: encodes the query with MPNet, queries Chroma, converts cosine
  distance to `similarity = 1 − distance`.
- `_bm25_search`: tokenises by lowercase whitespace split; drops zero-score chunks.
- `_reciprocal_rank_fusion`: combines the two ranked lists (k=60, 0.6/0.4 weights).
- `_rerank`: cross-encoder scores `(query, chunk_text)` pairs and sorts.
- `retrieve(query, top_k=3)`: semantic top-10 + BM25 top-10 → RRF → top-15 →
  rerank → top-`k`, returning `{text, source, authority, domain}`.

`CHROMADB_PATH = <project>/app/chroma_storage`; `EMBEDDING_MODEL = all-mpnet-base-v2`;
`RERANKER_MODEL = cross-encoder/ms-marco-MiniLM-L-6-v2`.

## 7. Generation (`app/services/generator.py`)

- `OpenAI(api_key=os.getenv("OPENAI_API_KEY"))` client at module load.
- `_build_context_block()` numbers sources `[i] Source: … | Authority: …` so the LLM
  can cite inline.
- `generate_response()` → query answers (gpt-3.5, temp 0.2, max 600).
- `generate_lesson()` → structured beats (gpt-3.5, JSON mode, temp 0.3, max 1500);
  `_sanitize_beats()` backfills any missing tablet object; sources defaulted from
  context. `_fallback_structured_lesson()` on error.
- `generate_text_lesson()` → narrative (gpt-4-turbo, JSON mode, temp 0.4, max 4000);
  `_fallback_text_lesson()` on error.

System prompts encode Pepper's persona (JKUAT Social Robotics Lab), strict grounding,
no medical advice, Kenyan food emphasis, and output formatting rules.

## 8. Verification (`app/core/verifier.py`)

- `extract_claims()` — sentence split, strip `[n]` citations, drop sentences < 6 words
  and meta-prefixed sentences ("based on", "in summary", …).
- `_score_claim_against_chunks()` — NLI softmax over `[contradiction, entailment,
  neutral]`; thresholds contradiction > 0.60, entailment > 0.45; else keyword-overlap
  fallback (≥ 0.35 → weak entailment).
- `_assign_grounding_label()` — `grounded` ≥ 0.75, `partial` ≥ 0.45 or any
  contradiction, else `ungrounded`.
- `verify_response()` returns a `GroundingReport`. No context / no claims → trivially
  `grounded` (score 1.0).

## 9. Instrumentation (`app/core/instrumentation.py`)

Regex `\[\d+(?:,\s*\d+)*\]` over the response text; returns `citations_present`
(bool) and `citation_count` (unique source indices). Zero LLM cost.

## 10. Persistence (`app/database/`)

`db.py`: `DATABASE_URL = sqlite+aiosqlite:///./nutripepper.db` (overridable via env);
async engine + `AsyncSessionLocal` (`expire_on_commit=False`); `init_db()` creates
tables; `get_db()` is an unused-but-available request-scoped session dependency.

`models.py` tables:

- **`interactions`** — `id`, `session_id`, `endpoint` (enum `query|learn`),
  `created_at`, `latency_ms`, `is_conversational`; relationships to the three child
  tables (cascade delete).
- **`query_details`** — query text + response + `citations_present/count` +
  verifier columns (`grounding_score/label`, `has_contradiction`,
  `claims_checked/supported`).
- **`learn_details`** — `concept`, `lesson_mode`, `sources_used`; structured fields
  (`total_beats`, `beat_types`, `lesson_json`) and text fields (`lesson_text`,
  `quiz_hints`, `description`).
- **`retrieved_chunks`** — per-chunk provenance (`chunk_index`, `source`, `authority`,
  `domain`, `text_preview` capped at 500 chars).

`interaction_logger.py` runs as a FastAPI background task: for `/query` it first calls
`verify_response()` (failures are non-fatal/logged) then writes rows in a single
transaction; for `/learn` it serialises beats/hints and writes the lesson + chunks.

### Migrations (`alembic/`)

`alembic/env.py` targets `Base.metadata`; online migrations swap `aiosqlite` for the
sync `sqlite` driver. Existing revisions:

- `f1a75e759958` — add grounding instrumentation to `query_details`.
- `ab0ece7a381a` — add text-lesson columns to `learn_details`.
- `61126318d2a0` — add `description` to `learn_details`.

## 11. Knowledge Build Pipeline (`data/scripts/`)

1. **`ingest_documents.py`** — for each PDF in `data/raw/`: `extract_pdf_text`
   (PyMuPDF) → `clean_text` (collapse whitespace, strip `Page N`, strip standalone
   numbers, fix hyphenation, collapse dots) → `chunk_text` (NLTK `sent_tokenize`,
   100–220 words/chunk, skip < 5-word sentences) → attach `source_metadata` →
   write `nutrition_knowledge_base.json` (`chunk_id`, `source_document`, `text`,
   `authority`, `year`, `domain`, `kenya_relevance`).
2. **`source_metadata.py`** — the per-file authority/year/domain/Kenya table.
3. **`build_vector_store.py`** — load JSON, embed each chunk (`embedder2.py` → MPNet),
   `collection.add` into Chroma collection `nutrition_kb`.

## 12. Evaluation Harness (`evaluation/`)

- `generate_test_set.py` → `test_set.json` (30 auto + 10 manual Kenya queries).
- `seed_interactions.py` → replays queries against `http://localhost:8000/query`.
- `evaluate.py` → retrieval (Precision@K/Recall@K/MRR), grounding (from DB), system
  latency, Kenya-specific analysis → `results/evaluation_report.json` + `…_summary.txt`.
  It imports the production `retrieve` and re-attaches chunk ids via the BM25 store.

## 13. Setup & Run

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Secrets (.env)
#    OPENAI_API_KEY=sk-...
#    APP_NAME=NutriPepper Backend
#    APP_VERSION=1.0.0
#    DEBUG=True
#    (optional) DATABASE_URL=sqlite+aiosqlite:///./nutripepper.db

# 3. (Re)build the knowledge base + vector store, if needed
python data/scripts/ingest_documents.py
python data/scripts/build_vector_store.py    # see known-issues before running

# 4. Run the API (creates DB tables on startup)
./run.sh          # uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. (optional) Evaluate
python evaluation/generate_test_set.py
python evaluation/seed_interactions.py        # server must be running
python evaluation/evaluate.py
```

Interactive docs: `http://localhost:8000/docs`.

## 14. Known Issues / Inconsistencies

These are real mismatches found in the code; fix before relying on the affected path.

1. **Metadata key mismatch (provenance).** `retriever.py` reads `meta.get("source")`,
   but `build_vector_store.py` writes the key `source_document` (and adds `year`,
   `kenya_relevance`). If the runtime store was built with that script, retrieved
   chunks would carry `source = "unknown"`. The fact that evaluation shows real source
   names implies `app/chroma_storage` was built by a different/updated builder that
   uses `source`. Standardise on one metadata schema across builder, retriever, and
   `generate_test_set.py` (which reads `source_document`).
2. **Hardcoded, wrong KB path in `build_vector_store.py`.** `KB_PATH` points to
   `.../4thyrProject/nutripepper_backend/...`, missing the `nutripepper/` parent of
   the actual layout. Make it relative to the project root.
3. **Two Chroma stores.** A second `chromadb/chroma.sqlite3` exists at the repo root
   alongside the runtime `app/chroma_storage/`. Only `app/chroma_storage` is used by
   the retriever; the root one is stale/duplicate.
4. **Output-vs-builder field gap.** `build_vector_store.py` writes
   `source_document/year/kenya_relevance`; the retriever only surfaces
   `source/authority/domain`. Align them.
5. **Latency.** LLM calls are synchronous in the request; model cold-starts dominate
   first-call latency (see `02-analysis.md`).
6. **`DEBUG` is a string** (`os.getenv(... , "True")`), not a bool — guard accordingly.
7. **Docs drift.** `README.md` and `nutripepper_backend_comprehensive_documentation.md`
   describe an earlier "dummy pipeline" phase and are superseded by this `docs/` set.
8. **No safety classifier / auth / rate limiting** — safety is prompt-only; the API is
   open. Not production-hardened.
