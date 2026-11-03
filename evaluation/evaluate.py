
"""
evaluation/evaluate.py

RAG Pipeline Evaluation Runner for NutriPepper.

Computes:
  - Retrieval metrics: Precision@K, Recall@K, MRR (against ground truth chunk IDs)
  - Grounding metrics: source attribution rate, grounding score distribution,
    hallucination rate, contradiction rate (from SQLite DB)
  - System metrics: average latency, p95 latency, success rate (from SQLite DB)
  - Kenya-specific query performance (subset analysis)

Outputs:
  - evaluation/results/evaluation_report.json
  - evaluation/results/evaluation_summary.txt

Usage:
    cd nutripepper_backend
    python evaluation/evaluate.py
"""

import sys
import json
import logging
import sqlite3
import statistics
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# ── Path setup — must happen before importing app modules ─────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Now safe to import production modules ─────────────────────────────────────

import chromadb
from app.services.retriever import retrieve, _get_bm25, _get_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("evaluator")

DB_PATH       = PROJECT_ROOT / "nutripepper.db"
TEST_SET_PATH = PROJECT_ROOT / "evaluation" / "test_set.json"
RESULTS_DIR   = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

KENYAN_TERMS = {
    "ugali", "sukuma", "omena", "githeri", "mukimo", "matoke",
    "terere", "managu", "nyama", "uji", "chapati", "kenya",
    "kenyan", "nairobi", "kiambu", "kisumu", "mombasa",
}


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def retrieve_with_ids(query: str, k: int) -> list[dict]:
    """
    Retrieve top-k chunks using the production 3-stage pipeline,
    but also return the chunk IDs needed for metric computation.

    The production retrieve() strips IDs from the clean output.
    We fetch IDs separately from the BM25 index (which holds all chunk metadata)
    by matching on text content after retrieval.
    """
    # Get the clean chunks from the production retriever
    chunks = retrieve(query, top_k=k)

    # Enrich with IDs by matching text against the BM25 chunk store
    # (BM25 index holds all chunks with their IDs)
    _, bm25_chunks = _get_bm25()
    text_to_id = {c["text"]: c["id"] for c in bm25_chunks}

    for chunk in chunks:
        chunk["id"] = text_to_id.get(chunk["text"], "unknown")

    return chunks


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_id: str) -> float:
    for i, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id == relevant_id:
            return 1.0 / i
    return 0.0


# ── Phase 1: Retrieval evaluation ────────────────────────────────────────────

def evaluate_retrieval(test_set: list[dict], k_values: list[int] = [1, 3, 5]) -> dict:
    """
    Run retrieval evaluation over all auto-generated queries with known ground truth.
    Uses the production 3-stage hybrid retriever (semantic + BM25 + reranking).
    """
    auto_queries = [q for q in test_set if q.get("ground_truth_chunk_id")]
    logger.info(f"Retrieval evaluation: {len(auto_queries)} auto-generated queries")

    precision = {k: [] for k in k_values}
    recall    = {k: [] for k in k_values}
    rr_scores = []
    per_query = []

    for i, item in enumerate(auto_queries):
        query_id    = item["query_id"]
        query       = item["query"]
        gt_chunk_id = item["ground_truth_chunk_id"]
        gt_source   = item.get("ground_truth_source", "")

        logger.info(f"  [{i+1}/{len(auto_queries)}] {query_id}: '{query[:60]}'")

        try:
            retrieved     = retrieve_with_ids(query, k=max(k_values))
            retrieved_ids = [c.get("id", "unknown") for c in retrieved]
        except Exception as e:
            logger.warning(f"  Retrieval failed for {query_id}: {e}")
            retrieved_ids = []

        rr = reciprocal_rank(retrieved_ids, gt_chunk_id)
        rr_scores.append(rr)

        query_metrics = {
            "query_id":      query_id,
            "query":         query,
            "domain":        item.get("domain"),
            "gt_chunk_id":   gt_chunk_id,
            "gt_source":     gt_source,
            "rr":            round(rr, 4),
            "found_at_rank": None,
            "retrieved_ids": retrieved_ids,
        }

        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid == gt_chunk_id:
                query_metrics["found_at_rank"] = rank
                break

        for k in k_values:
            p = precision_at_k(retrieved_ids, gt_chunk_id, k)
            r = recall_at_k(retrieved_ids, gt_chunk_id, k)
            precision[k].append(p)
            recall[k].append(r)
            query_metrics[f"precision@{k}"] = p
            query_metrics[f"recall@{k}"]    = r

        per_query.append(query_metrics)

    # Aggregate
    aggregated = {
        "num_queries": len(auto_queries),
        "MRR":         round(statistics.mean(rr_scores), 4) if rr_scores else 0.0,
    }
    for k in k_values:
        aggregated[f"Precision@{k}"] = round(statistics.mean(precision[k]), 4)
        aggregated[f"Recall@{k}"]    = round(statistics.mean(recall[k]), 4)

    # Domain breakdown
    domains: dict[str, dict] = {}
    for q in per_query:
        d = q.get("domain", "unknown")
        if d not in domains:
            domains[d] = {"rr": [], "recall@3": []}
        domains[d]["rr"].append(q["rr"])
        domains[d]["recall@3"].append(q.get("recall@3", 0.0))

    domain_breakdown = {
        d: {
            "count":    len(v["rr"]),
            "MRR":      round(statistics.mean(v["rr"]), 4),
            "Recall@3": round(statistics.mean(v["recall@3"]), 4),
        }
        for d, v in domains.items()
    }

    logger.info(f"Retrieval aggregated: {aggregated}")
    return {
        "aggregated":       aggregated,
        "domain_breakdown": domain_breakdown,
        "per_query":        per_query,
    }


# ── Phase 2: Grounding metrics from SQLite ────────────────────────────────────

def evaluate_grounding(db_path: Path) -> dict:
    if not db_path.exists():
        logger.warning(f"DB not found at {db_path} — skipping grounding metrics")
        return {}

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM query_details WHERE grounding_score IS NOT NULL")
    total_verified = cursor.fetchone()[0]

    if total_verified == 0:
        logger.warning("No verified interactions in DB — run seed_interactions.py first")
        conn.close()
        return {"total_verified": 0}

    cursor.execute("""
        SELECT grounding_label, COUNT(*) FROM query_details
        WHERE grounding_label IS NOT NULL GROUP BY grounding_label
    """)
    label_dist = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN citations_present = 1 THEN 1 ELSE 0 END)
        FROM query_details WHERE citations_present IS NOT NULL
    """)
    row = cursor.fetchone()
    citation_rate = (row[1] / row[0]) if row[0] else 0.0

    cursor.execute("""
        SELECT AVG(grounding_score), MIN(grounding_score), MAX(grounding_score)
        FROM query_details WHERE grounding_score IS NOT NULL
    """)
    row = cursor.fetchone()
    avg_score = round(row[0], 4) if row[0] else 0.0

    cursor.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN has_contradiction = 1 THEN 1 ELSE 0 END)
        FROM query_details WHERE has_contradiction IS NOT NULL
    """)
    row = cursor.fetchone()
    contradiction_rate = (row[1] / row[0]) if row[0] else 0.0

    cursor.execute("""
        SELECT AVG(claims_checked), AVG(claims_supported)
        FROM query_details WHERE claims_checked IS NOT NULL AND claims_checked > 0
    """)
    row = cursor.fetchone()

    conn.close()

    results = {
        "total_verified_interactions": total_verified,
        "source_attribution_rate":     round(citation_rate, 4),
        "grounding_score": {
            "mean": avg_score,
            "min":  round(row[0] if row and row[0] else 0.0, 4),
            "max":  round(row[1] if row and row[1] else 0.0, 4),
        },
        "grounding_label_distribution": label_dist,
        "contradiction_rate":           round(contradiction_rate, 4),
        "avg_claims_checked":           round(row[0], 2) if row and row[0] else 0.0,
        "avg_claims_supported":         round(row[1], 2) if row and row[1] else 0.0,
        "estimated_hallucination_rate": round(1.0 - avg_score, 4),
    }

    logger.info(f"Grounding metrics: {results}")
    return results


# ── Phase 3: System performance from SQLite ───────────────────────────────────

def evaluate_system_performance(db_path: Path) -> dict:
    if not db_path.exists():
        return {}

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT endpoint, latency_ms FROM interactions WHERE latency_ms IS NOT NULL"
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {"note": "No latency data in DB yet"}

    by_endpoint: dict[str, list[float]] = {}
    for endpoint, latency in rows:
        by_endpoint.setdefault(endpoint, []).append(latency)

    performance = {}
    for endpoint, latencies in by_endpoint.items():
        sorted_lat = sorted(latencies)
        n          = len(sorted_lat)
        p95_idx    = min(int(0.95 * n), n - 1)
        performance[endpoint] = {
            "count":        n,
            "mean_ms":      round(statistics.mean(latencies), 1),
            "median_ms":    round(statistics.median(latencies), 1),
            "p95_ms":       round(sorted_lat[p95_idx], 1),
            "min_ms":       round(sorted_lat[0], 1),
            "max_ms":       round(sorted_lat[-1], 1),
            "under_3s_pct": round(
                sum(1 for l in latencies if l < 3000) / n * 100, 1
            ),
        }

    logger.info(f"Performance: {performance}")
    return performance


# ── Phase 4: Kenya-specific query analysis ────────────────────────────────────

def evaluate_kenyan_queries(test_set: list[dict]) -> dict:
    kenyan_queries = [q for q in test_set if q.get("query_type") == "kenyan_specific"]
    logger.info(f"Kenya-specific evaluation: {len(kenyan_queries)} queries")

    per_query = []

    for item in kenyan_queries:
        query = item["query"]

        try:
            retrieved = retrieve(query, top_k=3)
        except Exception as e:
            logger.warning(f"Retrieval failed for {item['query_id']}: {e}")
            retrieved = []

        relevant_domains = {"nutrition", "food_safety", "sustainability"}
        domain_hits      = sum(1 for c in retrieved if c.get("domain") in relevant_domains)
        domain_relevance = domain_hits / len(retrieved) if retrieved else 0.0

        sources      = {c.get("source") for c in retrieved}
        kenya_hits   = sum(
            1 for c in retrieved
            if any(term in c.get("text", "").lower() for term in KENYAN_TERMS)
        )
        kenya_relevance = kenya_hits / len(retrieved) if retrieved else 0.0

        per_query.append({
            "query_id":          item["query_id"],
            "query":             query,
            "domain":            item.get("domain"),
            "domain_relevance":  round(domain_relevance, 2),
            "kenya_relevance":   round(kenya_relevance, 2),
            "source_diversity":  len(sources),
            "retrieved_sources": list(sources),
        })

    aggregated = {
        "num_queries":          len(kenyan_queries),
        "avg_domain_relevance": round(statistics.mean(q["domain_relevance"] for q in per_query), 4) if per_query else 0.0,
        "avg_kenya_relevance":  round(statistics.mean(q["kenya_relevance"]  for q in per_query), 4) if per_query else 0.0,
        "avg_source_diversity": round(statistics.mean(q["source_diversity"] for q in per_query), 4) if per_query else 0.0,
    }

    logger.info(f"Kenya-specific metrics: {aggregated}")
    return {"aggregated": aggregated, "per_query": per_query}


# ── Report generation ─────────────────────────────────────────────────────────

def build_summary_text(results: dict) -> str:
    r   = results
    ret = r.get("retrieval", {}).get("aggregated", {})
    grd = r.get("grounding", {})
    sys = r.get("system_performance", {})
    ken = r.get("kenyan_queries", {}).get("aggregated", {})

    lines = [
        "=" * 65,
        "  NUTRIPEPPER RAG PIPELINE — EVALUATION REPORT",
        f"  Generated: {r.get('generated_at', '')}",
        "=" * 65,
        "",
        "── RETRIEVAL QUALITY ────────────────────────────────────────",
        f"  Queries evaluated : {ret.get('num_queries', 'N/A')}",
        f"  Precision@1       : {ret.get('Precision@1', 'N/A')}",
        f"  Precision@3       : {ret.get('Precision@3', 'N/A')}  (target ≥ 0.85)",
        f"  Precision@5       : {ret.get('Precision@5', 'N/A')}",
        f"  Recall@3          : {ret.get('Recall@3', 'N/A')}",
        f"  Recall@5          : {ret.get('Recall@5', 'N/A')}  (target ≥ 0.90)",
        f"  MRR               : {ret.get('MRR', 'N/A')}  (target ≥ 0.80)",
        "",
    ]

    domain_bd = r.get("retrieval", {}).get("domain_breakdown", {})
    if domain_bd:
        lines.append("── RETRIEVAL BY DOMAIN ──────────────────────────────────────")
        for domain, metrics in domain_bd.items():
            lines.append(
                f"  {domain:<20} n={metrics['count']}  "
                f"MRR={metrics['MRR']}  Recall@3={metrics['Recall@3']}"
            )
        lines.append("")

    lines += [
        "── RESPONSE GROUNDING ───────────────────────────────────────",
        f"  Verified interactions : {grd.get('total_verified_interactions', 'N/A')}",
        f"  Source attribution    : {grd.get('source_attribution_rate', 'N/A')}  (target ≥ 0.90)",
        f"  Mean grounding score  : {grd.get('grounding_score', {}).get('mean', 'N/A')}",
        f"  Contradiction rate    : {grd.get('contradiction_rate', 'N/A')}",
        f"  Est. hallucination    : {grd.get('estimated_hallucination_rate', 'N/A')}  (target < 0.05)",
        f"  Avg claims checked    : {grd.get('avg_claims_checked', 'N/A')}",
        f"  Avg claims supported  : {grd.get('avg_claims_supported', 'N/A')}",
    ]

    label_dist = grd.get("grounding_label_distribution", {})
    if label_dist:
        lines.append(f"  Label distribution    : {label_dist}")
    lines.append("")

    if sys:
        lines.append("── SYSTEM PERFORMANCE ───────────────────────────────────────")
        for endpoint, metrics in sys.items():
            lines += [
                f"  [{endpoint}]",
                f"    Requests      : {metrics.get('count')}",
                f"    Mean latency  : {metrics.get('mean_ms')} ms",
                f"    Median latency: {metrics.get('median_ms')} ms",
                f"    P95 latency   : {metrics.get('p95_ms')} ms  (target < 5000ms)",
                f"    Under 3s      : {metrics.get('under_3s_pct')}%  (target > 95%)",
            ]
        lines.append("")

    lines += [
        "── KENYA-SPECIFIC QUERY PERFORMANCE (Objective 4) ──────────",
        f"  Queries evaluated     : {ken.get('num_queries', 'N/A')}",
        f"  Domain relevance      : {ken.get('avg_domain_relevance', 'N/A')}  (target ≥ 0.80)",
        f"  Kenya content match   : {ken.get('avg_kenya_relevance', 'N/A')}",
        f"  Avg source diversity  : {ken.get('avg_source_diversity', 'N/A')}",
        "",
        "── TARGETS vs ACTUALS ───────────────────────────────────────",
    ]

    targets = [
        ("Precision@3 ≥ 0.85",        ret.get("Precision@3", 0),                       0.85),
        ("Recall@5 ≥ 0.90",           ret.get("Recall@5", 0),                           0.90),
        ("MRR ≥ 0.80",                ret.get("MRR", 0),                                0.80),
        ("Source attribution ≥ 0.90", grd.get("source_attribution_rate", 0),            0.90),
        ("Hallucination < 0.05",      1 - grd.get("estimated_hallucination_rate", 1),   0.95),
        ("Kenya relevance ≥ 0.80",    ken.get("avg_domain_relevance", 0),               0.80),
    ]

    for label, actual, target in targets:
        met        = "✓ MET" if isinstance(actual, (int, float)) and actual >= target else "✗ NOT MET"
        actual_str = f"{actual:.4f}" if isinstance(actual, float) else str(actual)
        lines.append(f"  {label:<35} actual={actual_str}  {met}")

    lines += ["", "=" * 65]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== NutriPepper RAG Evaluation Runner ===")
    logger.info(f"Pipeline: Hybrid (Semantic + BM25 + Cross-encoder Reranking)")

    if not TEST_SET_PATH.exists():
        logger.error(f"Test set not found: {TEST_SET_PATH}")
        logger.error("Run: python evaluation/generate_test_set.py first")
        sys.exit(1)

    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)
    logger.info(f"Loaded {len(test_set)} test queries")

    # Warm up the retriever — loads ChromaDB, builds BM25 index, loads reranker
    logger.info("Warming up retriever (loading models and BM25 index)...")
    retrieve("nutrition warm-up query", top_k=1)
    logger.info("Retriever ready")

    logger.info("── Phase 1: Retrieval evaluation ──")
    retrieval_results = evaluate_retrieval(test_set)

    logger.info("── Phase 2: Grounding metrics from DB ──")
    grounding_results = evaluate_grounding(DB_PATH)

    logger.info("── Phase 3: System performance from DB ──")
    performance_results = evaluate_system_performance(DB_PATH)

    logger.info("── Phase 4: Kenya-specific query analysis ──")
    kenyan_results = evaluate_kenyan_queries(test_set)

    timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_results = {
        "generated_at":       timestamp,
        "retriever_pipeline": "Hybrid (Semantic + BM25 RRF + Cross-encoder Reranking)",
        "test_set_size":      len(test_set),
        "retrieval":          retrieval_results,
        "grounding":          grounding_results,
        "system_performance": performance_results,
        "kenyan_queries":     kenyan_results,
    }

    json_path = RESULTS_DIR / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Full report saved: {json_path}")

    summary      = build_summary_text(full_results)
    summary_path = RESULTS_DIR / "evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info(f"Summary saved: {summary_path}")

    print("\n" + summary)


if __name__ == "__main__":
    main()