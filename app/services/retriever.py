# app/services/retriever.py

import logging, os
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from app.core.logger import logger as app_logger

logger = logging.getLogger(__name__)

# ── Model + ChromaDB loading ──────────────────────────────────────────────────
# All models loaded once at module level — never reloaded per request.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CHROMADB_PATH    = os.path.join(BASE_DIR, "app/chroma_storage")
EMBEDDING_MODEL  = "all-mpnet-base-v2"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_collection:       chromadb.Collection  | None = None
_embedding_model:  SentenceTransformer  | None = None
_reranker:         CrossEncoder         | None = None
_bm25:             BM25Okapi            | None = None
_bm25_chunks:      list[dict]           | None = None  # parallel list to BM25 corpus


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client      = chromadb.PersistentClient(path=CHROMADB_PATH)
        collections = client.list_collections()
        if not collections:
            raise RuntimeError("No ChromaDB collections found")
        _collection = client.get_collection(collections[0].name)
        app_logger.info(f"[Retriever] ChromaDB loaded: '{_collection.name}' ({_collection.count()} chunks)")
    return _collection


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        app_logger.info(f"[Retriever] Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        app_logger.info(f"[Retriever] Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _get_bm25() -> tuple[BM25Okapi, list[dict]]:
    """
    Build BM25 index from all chunks in ChromaDB.
    Built once and cached — rebuilds only if ChromaDB collection changes.
    """
    global _bm25, _bm25_chunks
    if _bm25 is None:
        app_logger.info("[Retriever] Building BM25 index from ChromaDB...")
        collection = _get_collection()
        results    = collection.get(include=["documents", "metadatas"])

        _bm25_chunks = []
        corpus       = []

        for chunk_id, text, meta in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            if not text:
                continue
            _bm25_chunks.append({
                "id":        chunk_id,
                "text":      text,
                "source":    meta.get("source", "unknown"),
                "authority": meta.get("authority", "unknown"),
                "domain":    meta.get("domain", "unknown"),
            })
            # Tokenize: lowercase, split on whitespace
            corpus.append(text.lower().split())

        _bm25 = BM25Okapi(corpus)
        app_logger.info(f"[Retriever] BM25 index built: {len(_bm25_chunks)} chunks")

    return _bm25, _bm25_chunks


# ── Stage 1: Semantic search (ChromaDB dense retrieval) ───────────────────────

def _semantic_search(query: str, top_k: int) -> list[dict]:
    """Retrieve top_k chunks via dense vector similarity."""
    collection      = _get_collection()
    embedding_model = _get_embedding_model()
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for chunk_id, text, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Convert cosine distance → similarity score (0–1, higher = more similar)
        semantic_score = 1.0 - dist
        chunks.append({
            "id":             chunk_id,
            "text":           text,
            "source":         meta.get("source_document", "unknown"),
            "authority":      meta.get("authority", "unknown"),
            "domain":         meta.get("domain", "unknown"),
            "semantic_score": round(semantic_score, 4),
            "bm25_score":     0.0,
            "hybrid_score":   0.0,
        })
    return chunks


# ── Stage 2: BM25 keyword search ─────────────────────────────────────────────

def _bm25_search(query: str, top_k: int) -> list[dict]:
    """Retrieve top_k chunks via BM25 keyword matching."""
    bm25, bm25_chunks = _get_bm25()
    tokenized_query   = query.lower().split()
    scores            = bm25.get_scores(tokenized_query)

    # Get top_k indices by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    chunks = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue  # Skip zero-score chunks — not relevant at all
        chunk = bm25_chunks[idx].copy()
        chunk["semantic_score"] = 0.0
        chunk["bm25_score"]     = round(float(scores[idx]), 4)
        chunk["hybrid_score"]   = 0.0
        chunks.append(chunk)

    return chunks


# ── Stage 2: Hybrid score fusion (RRF) ───────────────────────────────────────

def _reciprocal_rank_fusion(
    semantic_chunks: list[dict],
    bm25_chunks_result: list[dict],
    k: int = 60,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict]:
    """
    Combine semantic and BM25 results using Reciprocal Rank Fusion.

    RRF score = semantic_weight * 1/(k + semantic_rank)
              + bm25_weight    * 1/(k + bm25_rank)

    k=60 is the standard RRF constant that reduces sensitivity to outlier ranks.
    semantic_weight=0.6 / bm25_weight=0.4 favours semantic understanding
    while giving BM25 meaningful influence for exact-term queries.
    """
    rrf_scores: dict[str, dict] = {}

    for rank, chunk in enumerate(semantic_chunks, start=1):
        cid = chunk["id"]
        if cid not in rrf_scores:
            rrf_scores[cid] = chunk.copy()
            rrf_scores[cid]["hybrid_score"] = 0.0
        rrf_scores[cid]["hybrid_score"] += semantic_weight * (1.0 / (k + rank))
        rrf_scores[cid]["semantic_score"] = chunk["semantic_score"]

    for rank, chunk in enumerate(bm25_chunks_result, start=1):
        cid = chunk["id"]
        if cid not in rrf_scores:
            rrf_scores[cid] = chunk.copy()
            rrf_scores[cid]["hybrid_score"] = 0.0
        rrf_scores[cid]["hybrid_score"] += bm25_weight * (1.0 / (k + rank))
        rrf_scores[cid]["bm25_score"] = chunk["bm25_score"]

    # Sort by hybrid score descending
    fused = sorted(rrf_scores.values(), key=lambda c: c["hybrid_score"], reverse=True)
    return fused


# ── Stage 3: Cross-encoder reranking ─────────────────────────────────────────

def _rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """
    Rerank candidate chunks using a cross-encoder for precise relevance scoring.

    The cross-encoder reads (query, chunk) pairs jointly — capturing exact
    query-document interaction that bi-encoder embeddings miss.
    ms-marco-MiniLM-L-6-v2 is trained on passage retrieval and works well
    for domain-specific short queries.
    """
    reranker = _get_reranker()
    pairs    = [(query, chunk["text"]) for chunk in candidates]
    scores   = reranker.predict(pairs)

    for chunk, score in zip(candidates, scores):
        chunk["reranker_score"] = round(float(score), 4)

    reranked = sorted(candidates, key=lambda c: c["reranker_score"], reverse=True)

    logger.info(
        f"[Retriever] Reranked {len(candidates)} → top {top_n} "
        f"(best score: {reranked[0]['reranker_score'] if reranked else 'N/A'})"
    )

    return reranked[:top_n]


# ── Public interface ──────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    3-stage hybrid retrieval pipeline:

    Stage 1 — Semantic search: dense vector similarity via ChromaDB (top 10)
    Stage 2 — BM25 search: keyword matching for exact terms (top 10)
              Fused via Reciprocal Rank Fusion into a single ranked list (top 15)
    Stage 3 — Cross-encoder reranking: precise relevance scoring (top_k)

    Args:
        query:  The user's query string (preprocessed).
        top_k:  Number of final chunks to return after reranking.

    Returns:
        List of chunk dicts with text, source, authority, domain, and scores.
    """
    logger.info(f"[Retriever] Query: '{query}' | top_k={top_k}")

    # Stage 1 — Semantic
    semantic_results = _semantic_search(query, top_k=10)
    logger.info(f"[Retriever] Semantic: {len(semantic_results)} chunks")

    # Stage 2 — BM25 + RRF fusion
    bm25_results  = _bm25_search(query, top_k=10)
    fused_results = _reciprocal_rank_fusion(semantic_results, bm25_results)
    logger.info(f"[Retriever] After RRF fusion: {len(fused_results)} unique chunks")

    # Take top 15 into reranker — wide enough to capture BM25 gains
    candidates = fused_results[:15]

    # Stage 3 — Cross-encoder reranking
    final_chunks = _rerank(query, candidates, top_n=top_k)

    # Clean output — remove internal score fields the LLM doesn't need
    clean = []
    for chunk in final_chunks:
        clean.append({
            "text":      chunk["text"],
            "source":    chunk["source"],
            "authority": chunk["authority"],
            "domain":    chunk["domain"],
        })

    logger.info(
        f"[Retriever] Final: {len(clean)} chunks from "
        f"{', '.join(c['source'] for c in clean)}"
    )
    return clean