# NutriPepper Backend — Documentation

NutriPepper is the backend and Retrieval-Augmented Generation (RAG) service for an
embodied nutrition-education robot at the JKUAT Social Robotics Lab. It answers
nutrition and food-safety questions and generates teaching lessons grounded strictly
in an authoritative, Kenya-focused knowledge base, with safety constraints and full
interaction logging/verification.

This documentation set reflects the **current codebase** and supersedes the older
`README.md` and `nutripepper_backend_comprehensive_documentation.md`, which describe
an earlier skeleton/dummy-pipeline phase.

## Contents

| Doc | What it covers |
|---|---|
| [01 — System Design](01-system-design.md) | Architecture, components, request flows, retrieval/generation/verification design, key decisions |
| [02 — Analysis](02-analysis.md) | Problem, stakeholders, requirements, knowledge base, evaluation methodology, measured results, gaps & recommendations |
| [03 — Implementation](03-implementation.md) | File-level reference, API contract, DB schema/migrations, build pipeline, setup/run, known issues |

## At a Glance

- **API**: FastAPI with `POST /query`, `POST /learn/structured`, `POST /learn/text`.
- **Retrieval**: 3-stage hybrid — dense (MPNet/ChromaDB) + BM25, fused via Reciprocal
  Rank Fusion, then cross-encoder reranking.
- **Generation**: OpenAI Chat Completions (`gpt-3.5-turbo` for queries/structured
  lessons, `gpt-4-turbo` for long-form lessons), strictly grounded in retrieved context.
- **Verification**: NLI-based claim grounding (runs async after the response is sent).
- **Persistence**: async SQLAlchemy over SQLite, every interaction logged with provenance.
- **Evaluation**: reproducible test-set generation + retrieval/grounding/latency metrics.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# add OPENAI_API_KEY to .env
./run.sh   # http://localhost:8000/docs
```

See [03 — Implementation](03-implementation.md) for full setup, the knowledge-base
build pipeline, and the evaluation workflow.
