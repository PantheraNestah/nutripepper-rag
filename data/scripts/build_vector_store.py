"""
Build the ChromaDB vector store from the extracted knowledge-base JSON.

Run from anywhere:

    python data/scripts/build_vector_store.py

Defaults (relative to the repository root, so no editing is needed):
    input   data/knowledge_base/nutrition_knowledge_base.json
    output  app/chroma_storage        (where the app looks for it)

Safe to re-run: chunks are matched by ID. Chunks whose text is unchanged are not
re-embedded (only their metadata is refreshed), so a re-run is quick; new or edited
chunks are embedded. Embedding all 810 chunks from scratch took about 22 minutes on a
CPU-only machine. Use --reset to delete the collection first (needed after a re-ingest
that removed or renamed chunks, so stale ones do not linger).
After rebuilding, restart the server: its BM25 index is built once at startup.
"""

import argparse
import json
from pathlib import Path

import chromadb
from tqdm import tqdm

from embedder2 import embedding_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB = REPO_ROOT / "data" / "knowledge_base" / "nutrition_knowledge_base.json"
DEFAULT_OUTPUT = REPO_ROOT / "app" / "chroma_storage"
COLLECTION_NAME = "nutrition_kb"   # the retriever uses the first collection in the store


def build_store(kb_path: Path, output_dir: Path, batch_size: int, reset: bool) -> int:
    with open(kb_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client = chromadb.PersistentClient(path=str(output_dir))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass   # nothing to delete
    # Default distance settings are kept on purpose: the retriever was written against them.
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    def metadata_for(c):
        return {
            "source_document": c["source_document"],
            "authority": c["authority"],
            "year": c["year"],
            "domain": c["domain"],
            "kenya_relevance": str(c["kenya_relevance"]),
        }

    # Re-embed only chunks that are new or whose text changed.
    existing = collection.get(include=["documents"])
    known = dict(zip(existing["ids"], existing["documents"]))
    to_embed = [c for c in chunks if known.get(c["chunk_id"]) != c["text"]]
    unchanged = [c for c in chunks if known.get(c["chunk_id"]) == c["text"]]
    stale = set(known) - {c["chunk_id"] for c in chunks}
    print(f"{len(chunks)} chunks in {kb_path.name}: {len(to_embed)} to embed, {len(unchanged)} unchanged.")
    if stale:
        print(f"Warning: {len(stale)} chunks in the store are not in the JSON; run with --reset to remove them.")

    for start in range(0, len(unchanged), 500):                 # refresh metadata only (cheap)
        batch = unchanged[start:start + 500]
        collection.update(ids=[c["chunk_id"] for c in batch], metadatas=[metadata_for(c) for c in batch])

    for start in tqdm(range(0, len(to_embed), batch_size), disable=not to_embed):
        batch = to_embed[start:start + batch_size]
        embeddings = embedding_model.encode([c["text"] for c in batch]).tolist()
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[metadata_for(c) for c in batch],
        )

    total = collection.count()
    print(f"Vector store ready: collection '{COLLECTION_NAME}' holds {total} chunks.")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the NutriPepper vector store.")
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB, help="knowledge-base JSON (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Chroma folder (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=32, help="chunks embedded per batch (default: %(default)s)")
    parser.add_argument("--reset", action="store_true", help="delete the collection before building")
    args = parser.parse_args()
    build_store(args.kb, args.output, args.batch_size, args.reset)
