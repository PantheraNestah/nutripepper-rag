# NutriPepper Backend

NutriPepper Backend is the core API and Retrieval-Augmented Generation orchestration service for the NutriPepper embodied nutrition education robot.

## Current Development Phase
Phase 1 — Backend Skeleton and Manual Pipeline Foundation

## Implemented Components
- FastAPI backend service
- /query endpoint
- Manual pipeline orchestration
- Dummy retrieval service
- Dummy generation service
- Logging middleware
- Response timing middleware
- Dependency injection
- Standard API responses

## Planned Next Components
- ChromaDB semantic vector retrieval
- SentenceTransformer embeddings
- Gemini LLM integration
- Verification layer
- Safety filtering
- Pepper robot communication bridge

## Run Server
```bash
./run.sh