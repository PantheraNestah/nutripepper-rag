# NutriPepper Backend System Documentation

## Project Overview

NutriPepper is a Retrieval-Augmented Generation (RAG) powered embodied AI system designed to provide safe, grounded, and culturally localized nutrition education within Kenyan contexts. The project integrates semantic information retrieval, Large Language Models (LLMs), verification mechanisms, and robotic embodiment to create an educational humanoid assistant capable of answering nutrition and food safety questions.

The system is designed as a safety-aware, modular backend architecture centered around a manually orchestrated RAG pipeline rather than a framework-heavy abstraction approach. This decision prioritizes:

- Transparency
- Retrieval control
- Evaluation traceability
- Safety enforcement
- Research flexibility
- Explainability

The backend system is implemented using FastAPI and follows a service-oriented modular architecture.

---

# 1. Research and Technical Objectives

## Primary Objective

To design and implement a grounded Retrieval-Augmented Generation pipeline integrated with an embodied AI system for culturally localized nutrition education.

## Technical Objectives

The implementation seeks to achieve the following:

1. Semantic retrieval over authoritative nutrition documents
2. Context-grounded response generation
3. Metadata-aware retrieval filtering
4. Reduced hallucination rates
5. Traceable and explainable outputs
6. Safety-aware conversational boundaries
7. Real-time API-driven interaction
8. Persistent vector memory using ChromaDB
9. Extensible architecture for future robotic integration
10. Evaluation-ready logging and observability

---

# 2. Current System State

## Current Development Version

```text
v0.1.0-alpha
```

## Implemented Components

### Backend Infrastructure
- FastAPI backend
- Modular project structure
- API routing
- Environment configuration
- Logging system
- Response timing middleware
- Exception handling
- Dependency injection

### RAG Infrastructure
- PDF ingestion pipeline
- Document cleaning pipeline
- Semantic chunking
- Metadata tagging
- MPNet embedding generation
- ChromaDB persistent vector storage
- Semantic similarity retrieval

### Knowledge Base
- Kenya nutrition corpus
- WHO nutrition guidelines
- FAO food safety guidelines
- Kenya MoH nutrition policy documents
- Structured chunked JSON knowledge base

### Evaluation Infrastructure
- Interaction logging
- Request timing
- Retrieval traceability
- Structured API responses

---

# 3. High-Level System Architecture

The backend follows a layered architecture.

```text
User Query
    ↓
FastAPI Endpoint
    ↓
Pipeline Orchestration
    ↓
Preprocessing
    ↓
Embedding Generation
    ↓
ChromaDB Semantic Retrieval
    ↓
Retrieved Context
    ↓
Prompt Construction
    ↓
LLM Generation (Planned)
    ↓
Verification Layer (Planned)
    ↓
Safety Filtering (Planned)
    ↓
Structured Response
```

---

# 4. Project Directory Structure

```text
nutripepper_backend/
│
├── app/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── logger.py
│   │
│   ├── dependencies/
│   │   └── pipeline.py
│   │
│   ├── middleware/
│   │   └── timer.py
│   │
│   ├── schemas/
│   │   ├── query.py
│   │   └── response.py
│   │
│   ├── services/
│   │   ├── embedder.py
│   │   ├── generator.py
│   │   ├── pipeline.py
│   │   └── retriever.py
│   │
│   ├── utils/
│   │   └── preprocess.py
│   │
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── knowledge_base/
│   └── scripts/
│       ├── ingest_documents.py
│       └── source_metadata.py
│
├── chroma_storage/
├── requirements.txt
├── README.md
├── VERSION
├── .env
└── run.sh
```

---

# 5. Backend Infrastructure

## 5.1 FastAPI Application Layer

### File

```text
app/main.py
```

### Responsibilities

- Initializes FastAPI application
- Registers middleware
- Includes API routers
- Loads configuration settings

### Core Features

- OpenAPI/Swagger documentation
- ASGI asynchronous architecture
- Modular router registration
- Middleware support

### Current Middleware

- Process timing middleware

---

# 6. API Layer

## 6.1 Query Endpoint

### Endpoint

```http
POST /query
```

### Purpose

Accepts user nutrition-related queries and executes the manual RAG pipeline.

### Request Example

```json
{
  "user_query": "Is ugali healthy for breakfast?"
}
```

### Response Structure

```json
{
  "status": "success",
  "message": "Query processed successfully",
  "data": {
    "original_query": "Is ugali healthy for breakfast?",
    "processed_query": "is ugali healthy for breakfast?",
    "retrieved_context": [],
    "generated_response": "..."
  }
}
```

### Processing Flow

```text
Receive Query
    ↓
Preprocess Query
    ↓
Generate Query Embedding
    ↓
Semantic Retrieval
    ↓
Retrieve Top-K Documents
    ↓
Generate Response
    ↓
Return Structured JSON
```

---

# 7. Core Infrastructure Components

## 7.1 Configuration System

### File

```text
app/core/config.py
```

### Purpose

Centralized environment and configuration management.

### Current Configurations

```env
APP_NAME=NutriPepper Backend
APP_VERSION=1.0.0
DEBUG=True
```

### Design Goals

- Centralized settings
- Environment portability
- Production readiness
- Simplified deployment

---

## 7.2 Logging System

### File

```text
app/core/logger.py
```

### Purpose

Provides centralized structured logging across the backend.

### Current Logged Events

- Incoming queries
- Processed queries
- Retrieval counts
- Response generation events
- System errors

### Logging Importance

The logging system is critical for:

- Retrieval debugging
- Hallucination analysis
- Evaluation metrics
- Performance profiling
- Research reproducibility

---

## 7.3 Response Timing Middleware

### File

```text
app/middleware/timer.py
```

### Purpose

Measures request processing latency.

### Output

Adds:

```http
X-Process-Time
```

header to responses.

### Importance

Used later for:

- Retrieval latency evaluation
- Real-time interaction assessment
- Performance benchmarking

---

# 8. RAG Pipeline Architecture

## 8.1 Pipeline Philosophy

Unlike many RAG systems that rely heavily on orchestration frameworks such as LangChain, NutriPepper implements a manually controlled pipeline.

### Reasons for Manual Pipeline

1. Fine-grained retrieval control
2. Easier debugging
3. Transparent evaluation
4. Better safety integration
5. Easier verification insertion
6. Research traceability
7. Lower architectural abstraction

This design choice aligns with the project’s research emphasis on grounding and safety.

---

# 9. Knowledge Engineering Pipeline

## 9.1 Corpus Sources

The current knowledge base contains authoritative nutrition and food safety documents.

### Current Documents

#### Kenya-Focused Documents

1. Kenya Food Composition Tables (FAO/MoH Kenya)
2. Kenya Nutrient Profile Model 2025
3. Kenya Nutrition Action Plan 2018–2022
4. National Guidelines for Healthy Diets and Physical Activity 2017

#### International Authoritative Sources

5. WHO Healthy Diet Fact Sheet
6. FAO/WHO General Principles of Food Hygiene

---

# 10. Data Engineering Pipeline

## 10.1 Document Ingestion

### Script

```text
data/scripts/ingest_documents.py
```

### Responsibilities

- PDF reading
- Text extraction
- Cleaning
- Sentence segmentation
- Chunk generation
- Metadata attachment
- JSON export

---

## 10.2 PDF Text Extraction

### Technology

```python
PyMuPDF (fitz)
```

### Purpose

Extracts raw text from uploaded PDF documents.

### Extraction Flow

```text
PDF Document
    ↓
Page Iteration
    ↓
Text Extraction
    ↓
Combined Raw Text
```

---

## 10.3 Text Cleaning

### Current Cleaning Operations

- Remove excessive whitespace
- Remove page artifacts
- Remove broken line breaks
- Remove repeated dots
- Remove numeric noise
- Fix token fragmentation

### Goal

Improve embedding quality and semantic coherence.

---

## 10.4 Semantic Chunking

### Current Chunking Strategy

Sentence-aware chunk aggregation.

### Chunk Constraints

- Minimum words: 100
- Maximum words: 220

### Why Chunking Matters

Chunking affects:

- Retrieval quality
- Context precision
- Embedding fidelity
- LLM grounding quality

### Current Output

```text
810 knowledge chunks generated
```

---

## 10.5 Metadata Tagging

### Current Metadata Fields

Each chunk contains:

```json
{
  "chunk_id": "...",
  "source_document": "...",
  "authority": "...",
  "year": 2025,
  "domain": "nutrition",
  "kenya_relevance": true
}
```

### Purpose of Metadata

Metadata enables:

- Filtered retrieval
- Source attribution
- Citation generation
- Domain-aware search
- Evaluation analysis

---

# 11. Embedding System

## 11.1 Embedding Model

### Current Model

```text
sentence-transformers/all-mpnet-base-v2
```

### Why MPNet Was Chosen

MPNet provides:

- Strong semantic contextualization
- High retrieval quality
- Better sentence-level understanding
- Strong academic credibility for IR tasks

### Tradeoff

| Advantage | Cost |
|---|---|
| Better semantic retrieval | Higher compute usage |
| Better contextual embeddings | Slower embedding generation |

The project prioritizes retrieval quality over lightweight inference.

---

## 11.2 Embedding Pipeline

### File

```text
app/services/embedder.py
```

### Responsibilities

- Load MPNet model
- Generate vector embeddings
- Provide reusable embedding interface

### Embedding Flow

```text
Text Chunk
    ↓
MPNet Encoder
    ↓
Dense Vector Representation
```

---

# 12. Vector Database Layer

## 12.1 ChromaDB Integration

### Current Database

```text
ChromaDB PersistentClient
```

### Storage Location

```text
chroma_storage/
```

### Purpose

Stores:

- embeddings
- documents
- metadata
- vector indexes

---

## 12.2 Vector Store Builder

### Script

```text
data/scripts/build_vector_store.py
```

### Responsibilities

- Load JSON chunks
- Generate embeddings
- Insert embeddings into ChromaDB
- Attach metadata
- Persist vector storage

### Processing Flow

```text
Knowledge JSON
    ↓
Generate MPNet Embeddings
    ↓
Store in ChromaDB
    ↓
Persistent Semantic Memory
```

---

# 13. Semantic Retrieval Engine

## 13.1 Retriever Service

### File

```text
app/services/retriever.py
```

### Current Retrieval Strategy

Dense vector similarity search.

### Retrieval Flow

```text
User Query
    ↓
Generate Query Embedding
    ↓
ChromaDB Similarity Search
    ↓
Retrieve Top-K Context Chunks
```

### Current Retrieval Output

Returns:

- text
- source document
- authority
- domain

### Current Retrieval Technique

```text
Dense Vector Similarity Search
```

### Planned Future Enhancements

- Metadata filtering
- Cross-encoder reranking
- Hybrid retrieval
- Retrieval confidence scoring
- Citation ranking

---

# 14. Generation Layer

## 14.1 Generator Service

### File

```text
app/services/generator.py
```

### Current State

Placeholder generation engine.

### Current Functionality

- Combines retrieved context
- Constructs synthetic grounded response

### Planned Upgrade

Future integration with:

```text
Gemini-2.5-pro
```

or equivalent LLM API.

---

# 15. Pipeline Orchestration Layer

## 15.1 Pipeline Service

### File

```text
app/services/pipeline.py
```

### Purpose

Coordinates the entire RAG workflow.

### Current Responsibilities

1. Receive query
2. Preprocess query
3. Retrieve context
4. Generate response
5. Return structured output

### Architectural Importance

This file acts as the central orchestration engine and will later coordinate:

- verification
- reranking
- safety filtering
- citation injection
- robot communication

---

# 16. Dependency Injection

## 16.1 Dependency Layer

### File

```text
app/dependencies/pipeline.py
```

### Purpose

Provides service abstraction and dependency injection.

### Benefits

- Easier testing
- Swappable services
- Cleaner architecture
- Better modularity

---

# 17. Request and Response Schemas

## 17.1 Query Schema

### File

```text
app/schemas/query.py
```

### Purpose

Validates incoming request structure.

---

## 17.2 Response Schema

### File

```text
app/schemas/response.py
```

### Purpose

Defines standardized API responses.

### Benefits

- API consistency
- Easier frontend integration
- Better debugging
- Evaluation standardization

---

# 18. Current Technologies Stack

| Layer | Technology |
|---|---|
| Backend Framework | FastAPI |
| API Server | Uvicorn |
| Vector Database | ChromaDB |
| Embedding Model | all-mpnet-base-v2 |
| PDF Extraction | PyMuPDF |
| NLP Utilities | NLTK |
| Environment Management | python-dotenv |
| Logging | Python logging |
| Serialization | JSON |
| Validation | Pydantic |

---

# 19. Current Implementation Phases

## Phase 1 — Backend Skeleton

### Completed

- FastAPI initialization
- Modular structure
- API endpoint
- Dummy RAG flow
- Response schema

---

## Phase 1.5 — Backend Professionalization

### Completed

- Logging
- Middleware
- Dependency injection
- Configuration management
- Exception handling
- Response standardization
- Git/project hygiene

---

## Phase 2A — Knowledge Base Construction

### Completed

- PDF ingestion
- Text extraction
- Cleaning pipeline
- Semantic chunking
- Metadata tagging
- Structured JSON knowledge base

### Current Output

```text
810 structured knowledge chunks
```

---

## Phase 2B — Semantic Retrieval Engine

### In Progress

- MPNet embedding generation
- ChromaDB persistent storage
- Semantic retrieval integration

---

# 20. Planned Future Implementation Phases

## Phase 3 — LLM Integration

### Planned Features

- Gemini API integration
- Prompt engineering
- Grounded prompt templates
- Source-conditioned generation

---

## Phase 4 — Verification Layer

### Planned Features

- Claim extraction
- Claim-to-source matching
- Unsupported claim removal
- Response reconstruction

### Importance

This is the core anti-hallucination layer.

---

## Phase 5 — Safety Layer

### Planned Features

- Diagnosis blocking
- Prescription blocking
- Unsafe food handling filtering
- Out-of-scope query rejection

### Architectural Goal

Transform NutriPepper into a safety-aware embodied AI system.

---

## Phase 6 — Embodied Integration

### Planned Features

- Pepper robot integration
- Speech recognition
- Text-to-speech
- Gesture synchronization
- Tablet synchronization

---

## Phase 7 — Evaluation Infrastructure

### Planned Metrics

- Precision@K
- Recall@K
- MRR
- NDCG
- Hallucination rate
- Latency
- Safety compliance

---

# 21. Engineering Philosophy

NutriPepper intentionally avoids over-reliance on high-level orchestration frameworks.

The system prioritizes:

- Explicit architecture
- Traceable pipelines
- Transparent retrieval
- Modular AI engineering
- Research reproducibility
- Explainability
- Safety integration readiness

This philosophy is especially important because the project operates within a health-adjacent educational domain.

---

# 22. Current Limitations

## Retrieval Limitations

Current system does not yet include:

- reranking
- metadata-aware filtering
- hybrid search
- retrieval confidence estimation

## Generation Limitations

Current generation layer is placeholder-based and not yet connected to a real LLM.

## Safety Limitations

No safety filtering yet implemented.

## Verification Limitations

No claim verification yet implemented.

---

# 23. Recommended Next Engineering Tasks

## Immediate Next Priority

Complete:

```text
build_vector_store.py
```

and validate semantic retrieval.

---

## Recommended Sequence

### Step 1

Finish ChromaDB vector population.

### Step 2

Test semantic retrieval quality.

### Step 3

Integrate Gemini API.

### Step 4

Implement prompt engineering.

### Step 5

Implement verification engine.

### Step 6

Implement safety layer.

### Step 7

Add metadata-aware retrieval.

### Step 8

Implement cross-encoder reranking.

---

# 24. Guidance for Future AI Assistance

When using AI systems to continue development of NutriPepper, the following architectural principles should always be preserved.

## Mandatory Architectural Constraints

### Preserve Manual Pipeline Control

Do not replace the orchestrated pipeline with opaque framework abstractions.

### Maintain Modular Services

Each major capability should remain isolated:

- retrieval
- generation
- verification
- safety
- embodiment

### Prioritize Explainability

Every retrieval and generation step should remain inspectable.

### Maintain Metadata Preservation

Metadata is essential for:

- evaluation
- attribution
- safety
- explainability

### Avoid Unsafe Generative Freedom

Future LLM integrations must remain grounded strictly on retrieved context.

---

# 25. Suggested Future Enhancements

## Retrieval Enhancements

- BM25 hybrid retrieval
- Cross-encoder reranking
- Semantic caching
- Query expansion
- Multi-hop retrieval

## Generation Enhancements

- Structured prompting
- Citation injection
- Hallucination scoring
- Confidence estimation

## Safety Enhancements

- Medical intent classification
- Toxicity filtering
- Adversarial prompt handling

## Robotics Enhancements

- Real-time speech streaming
- Gesture classification
- Emotion-aware responses
- Gaze synchronization

## Infrastructure Enhancements

- Dockerization
- Redis caching
- Async retrieval
- Background embedding jobs
- GPU acceleration

---

# 26. Conclusion

NutriPepper currently exists as a professionally structured Retrieval-Augmented Generation backend system designed for grounded nutrition education in Kenyan contexts.

The project has successfully established:

- modular backend infrastructure,
- a structured corpus engineering pipeline,
- semantic chunking,
- metadata-aware knowledge organization,
- MPNet embedding infrastructure,
- and ChromaDB semantic retrieval foundations.

The architecture has been intentionally designed for:

- safety integration,
- verification extensibility,
- explainability,
- evaluation,
- and embodied AI deployment.

The current system provides a strong research-grade foundation for the future implementation of:

- grounded LLM generation,
- hallucination mitigation,
- verification pipelines,
- and humanoid robotic interaction.

NutriPepper therefore represents both:

1. a practical AI engineering artifact, and
2. a research platform for grounded embodied AI systems.

