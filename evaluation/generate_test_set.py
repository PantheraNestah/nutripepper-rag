
"""
evaluation/generate_test_set.py

Generates a structured test query dataset for RAG pipeline evaluation.

Strategy:
- Samples chunks directly from ChromaDB across domains and authorities
- Uses an LLM (EASY tier, see app/services/llm.py) to generate natural questions from each chunk (ground truth guaranteed)
- Appends 10 manually defined Kenya-specific queries
- Saves the full test set to evaluation/test_set.json

Usage:
    python evaluation/generate_test_set.py

Output:
    evaluation/test_set.json  — 40 test queries with ground truth chunk references
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from dotenv import load_dotenv
import chromadb

# ── Setup ─────────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("test_set_generator")

# Resolve project root (this script lives in evaluation/ one level below root)
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
CHROMADB_PATH = PROJECT_ROOT / "app/chroma_storage"
OUTPUT_DIR    = PROJECT_ROOT / "evaluation"
OUTPUT_FILE   = OUTPUT_DIR / "test_set.json"

OUTPUT_DIR.mkdir(exist_ok=True)

# Model access goes through the shared provider layer (see app/services/llm.py).
sys.path.insert(0, str(PROJECT_ROOT))
from app.services.llm import Tier, complete  # noqa: E402


# ── ChromaDB access ───────────────────────────────────────────────────────────

def load_chromadb_collection(chroma_path: Path) -> chromadb.Collection:
    """
    Connect to the local ChromaDB persistent store and return the collection.
    Tries common collection names used in NutriPepper — update if yours differs.
    """
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))

    # List available collections so we can auto-detect the name
    collections = chroma_client.list_collections()
    if not collections:
        logger.error(f"No collections found in ChromaDB at {chroma_path}")
        sys.exit(1)

    collection_names = [c.name for c in collections]
    logger.info(f"Available ChromaDB collections: {collection_names}")

    # Use the first collection — update this if you have multiple
    collection = chroma_client.get_collection(collection_names[0])
    logger.info(f"Using collection: '{collection.name}' ({collection.count()} chunks)")
    return collection


def fetch_all_chunks(collection: chromadb.Collection) -> list[dict]:
    """
    Fetch all chunks from the collection with their metadata.
    Returns a list of dicts with: id, text, source, authority, domain.
    """
    results = collection.get(include=["documents", "metadatas"])

    chunks = []
    for chunk_id, text, meta in zip(
        results["ids"],
        results["documents"],
        results["metadatas"],
    ):
        if not text or len(text.strip()) < 100:
            continue  # Skip very short chunks — unlikely to generate good queries

        chunks.append({
            "id":        chunk_id,
            "text":      text.strip(),
            "source":    meta.get("source_document", "unknown"),
            "authority": meta.get("authority", "unknown"),
            "domain":    meta.get("domain", "unknown"),
        })

    logger.info(f"Fetched {len(chunks)} usable chunks from ChromaDB")
    return chunks


# ── Stratified sampling ───────────────────────────────────────────────────────

def stratified_sample(chunks: list[dict], target: int = 30) -> list[dict]:
    """
    Sample chunks proportionally across domains and authorities so the
    test set covers the knowledge base evenly.

    Target is 30 auto-generated + 10 manual Kenya-specific = 40 total.
    """
    # Group by domain
    by_domain: dict[str, list] = {}
    for chunk in chunks:
        domain = chunk.get("domain", "unknown")
        by_domain.setdefault(domain, []).append(chunk)

    logger.info(f"Chunks by domain: { {k: len(v) for k, v in by_domain.items()} }")

    sampled = []
    domains = list(by_domain.keys())

    # Distribute target evenly across domains
    per_domain = max(1, target // len(domains))
    remainder  = target - (per_domain * len(domains))

    for i, domain in enumerate(domains):
        pool  = by_domain[domain]
        count = per_domain + (1 if i < remainder else 0)
        count = min(count, len(pool))
        sampled.extend(random.sample(pool, count))

    # If we're short (some domains had fewer chunks), top up from the full pool
    if len(sampled) < target:
        remaining = [c for c in chunks if c not in sampled]
        extra     = min(target - len(sampled), len(remaining))
        sampled.extend(random.sample(remaining, extra))

    random.shuffle(sampled)
    logger.info(f"Stratified sample: {len(sampled)} chunks selected")
    return sampled[:target]


# ── Query generation ──────────────────────────────────────────────────────────

QUERY_GEN_PROMPT = """You are generating evaluation data for a nutrition education RAG system \
deployed in Kenya.

Given a text chunk from a nutrition or food safety document, generate ONE natural question \
that a Kenyan student, teacher, or community member might ask, and that this chunk directly answers.

Rules:
- The question must be answerable using ONLY the provided chunk.
- Write the question as a real person would ask it — conversational, not academic.
- Do not reference the document or chunk explicitly (e.g. do not say "according to this text").
- Keep it to one sentence.
- Return ONLY the question string, nothing else.

Chunk:
{chunk_text}
"""


def generate_query_for_chunk(chunk: dict) -> str | None:
    """Generate a natural question from a chunk using the EASY-tier model."""
    prompt = QUERY_GEN_PROMPT.format(chunk_text=chunk["text"][:800])

    try:
        query = complete(
            Tier.EASY,
            system="You write short, natural questions for a nutrition education quiz.",
            user=prompt,
            max_tokens=500,      # headroom for reasoning tokens; the question itself is ~30
            temperature=0.7,     # some variety in phrasing (ignored by Anthropic models)
        ).strip('"')
        return query if len(query) > 10 else None

    except Exception as e:
        logger.warning(f"Query generation failed for chunk {chunk['id']}: {e}")
        return None


# ── Manual Kenya-specific queries ─────────────────────────────────────────────

KENYA_SPECIFIC_QUERIES = [
    {
        "query_id": "q_manual_001",
        "query": "Is ugali a healthy meal for a growing child?",
        "domain": "nutrition",
        "authority": "manual",
        "query_type": "kenyan_specific",
        "ground_truth_chunk_id": None,
        "ground_truth_source": None,
        "ground_truth_text_preview": None,
        "notes": "Tests cultural food context — ugali nutritional value",
    },
    {
        "query_id": "q_manual_002",
        "query": "How can I make sukuma wiki more nutritious for my family?",
        "domain": "nutrition",
        "authority": "manual",
        "query_type": "kenyan_specific",
        "ground_truth_chunk_id": None,
        "ground_truth_source": None,
        "ground_truth_text_preview": None,
        "notes": "Tests local vegetable knowledge and preparation advice",
    },
    {
        "query_id": "q_manual_003",
        "query": "What is the best local food to eat when a child has anaemia?",
        "domain": "nutrition",
        "authority": "manual",
        "query_type": "kenyan_specific",
        "ground_truth_chunk_id": None,
        "ground_truth_source": None,
        "ground_truth_text_preview": None,
        "notes": "Tests iron deficiency + local food knowledge",
    },
    {
        "query_id": "q_manual_004",
        "query": "How long can I safely store cooked githeri before it goes bad?",
        "domain": "food_safety",
        "authority": "manual",
        "query_type": "kenyan_specific",
        "ground_truth_chunk_id": None,
        "ground_truth_source": None,
        "ground_truth_text_preview": None,
        "notes": "Tests food safety knowledge for common Kenyan dish",
    },
    {
        "query_id": "q_manual_005",
        "query": "Is omena safe to eat every day for protein?",
        "domain": "nutrition",
        "authority": "manual",
        "query_type": "kenyan_specific",
        "ground_truth_chunk_id": None,
        "ground_truth_source": None,
        "ground_truth_text_preview": None,
        "notes": "Tests knowledge of silver sardines as protein source",
    },
    {
        "query_id": "q_manual_006",
        "query": "What foods should a pregnant woman in Kenya eat for a healthy baby?",
        "domain": "nutrition",
        "authority": "manual",
        "query_type": "kenyan_specific",
        "ground_truth_chunk_id": None,
        "ground_truth_source": None,
        "ground_truth_text_preview": None,
        "notes": "Tests life stage nutrition in Kenyan context",
    },
    {
        "query_id": "q_manual_007",
        "query": "How do I know if maize flour has gone bad before cooking ugali?",
        "domain": "food_safety",
        "authority": "manual",
        "query_type": "kenyan_specific",
        "ground_truth_chunk_id": None,
        "ground_truth_source": None,
        "ground_truth_text_preview": None,
        "notes": "Tests aflatoxin / grain storage knowledge",
    },
    {
        "query_id": "q_manual_008",
        "query": "What vegetables can I grow at home in Kenya to improve my family's diet?",
        "domain": "nutrition",
        "authority": "manual",
        "query_type": "kenyan_specific",
        "ground_truth_chunk_id": None,
        "ground_truth_source": None,
        "ground_truth_text_preview": None,
        "notes": "Tests sustainable food practices + local food context",
    },
    {
        "query_id": "q_manual_009",
        "query": "Why do children in Kenya often lack enough vitamins even when they eat regularly?",
        "domain": "nutrition",
        "authority": "manual",
        "query_type": "kenyan_specific",
        "ground_truth_chunk_id": None,
        "ground_truth_source": None,
        "ground_truth_text_preview": None,
        "notes": "Tests micronutrient deficiency knowledge in Kenyan context",
    },
    {
        "query_id": "q_manual_010",
        "query": "Is it safe to reheat nyama choma that was left out overnight?",
        "domain": "food_safety",
        "authority": "manual",
        "query_type": "kenyan_specific",
        "ground_truth_chunk_id": None,
        "ground_truth_source": None,
        "ground_truth_text_preview": None,
        "notes": "Tests food safety knowledge for common Kenyan meat dish",
    },
]


# ── Test set assembly ─────────────────────────────────────────────────────────

def build_test_set(chunks: list[dict]) -> list[dict]:
    """
    Generate 30 auto-generated queries + append 10 manual Kenya-specific queries.
    Returns the full 40-item test set.
    """
    sampled = stratified_sample(chunks, target=30)
    test_set = []

    for i, chunk in enumerate(sampled, start=1):
        query_id = f"q_{i:03d}"
        logger.info(f"Generating query {i}/30 for chunk {chunk['id']} ...")

        query = generate_query_for_chunk(chunk)
        if not query:
            logger.warning(f"Skipping chunk {chunk['id']} — query generation returned None")
            continue

        test_set.append({
            "query_id":                  query_id,
            "query":                     query,
            "domain":                    chunk["domain"],
            "authority":                 chunk["authority"],
            "query_type":                "auto_generated",
            "ground_truth_chunk_id":     chunk["id"],
            "ground_truth_source":       chunk["source"],
            "ground_truth_text_preview": chunk["text"][:300],
            "notes":                     f"Auto-generated from {chunk['source']}",
        })

    # Append manual Kenya-specific queries
    test_set.extend(KENYA_SPECIFIC_QUERIES)

    logger.info(f"Test set assembled: {len(test_set)} queries total")
    return test_set


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logger.info("=== NutriPepper Test Set Generator ===")
    logger.info(f"ChromaDB path: {CHROMADB_PATH}")
    logger.info(f"Output file:   {OUTPUT_FILE}")

    if not CHROMADB_PATH.exists():
        logger.error(f"ChromaDB directory not found at {CHROMADB_PATH}")
        logger.error("Make sure you're running this from the project root or adjust CHROMADB_PATH.")
        sys.exit(1)

    # Load ChromaDB
    collection = load_chromadb_collection(CHROMADB_PATH)
    chunks     = fetch_all_chunks(collection)

    if len(chunks) < 30:
        logger.warning(
            f"Only {len(chunks)} usable chunks found — "
            f"test set will have fewer than 30 auto-generated queries."
        )

    # Build test set
    random.seed(42)   # Reproducible sampling
    test_set = build_test_set(chunks)

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    logger.info(f"Test set saved to {OUTPUT_FILE}")

    # Print summary
    domains    = {}
    query_types = {}
    for item in test_set:
        domains[item["domain"]]         = domains.get(item["domain"], 0) + 1
        query_types[item["query_type"]] = query_types.get(item["query_type"], 0) + 1

    logger.info("=== Summary ===")
    logger.info(f"Total queries:  {len(test_set)}")
    logger.info(f"By domain:      {domains}")
    logger.info(f"By query type:  {query_types}")
    logger.info("Done.")


if __name__ == "__main__":
    main()