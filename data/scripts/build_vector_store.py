import os
import json
import chromadb
from tqdm import tqdm
from embedder2 import generate_embedding

KB_PATH = "/home/nesta/Documents/COMP SCIE/4thyrProject/nutripepper_backend/data/knowledge_base/nutrition_knowledge_base.json"
CHROMA_DIR = "chroma_storage"

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(name="nutrition_kb")


def build_store():
    with open(KB_PATH, "r") as f:
        knowledge_chunks = json.load(f)

    print(f"Embedding {len(knowledge_chunks)} chunks into ChromaDB...")

    for item in tqdm(knowledge_chunks):
        embedding = generate_embedding(item["text"])

        metadata = {
            "source_document": item["source_document"],
            "authority": item["authority"],
            "year": item["year"],
            "domain": item["domain"],
            "kenya_relevance": str(item["kenya_relevance"])
        }

        collection.add(
            ids=[item["chunk_id"]],
            embeddings=[embedding],
            documents=[item["text"]],
            metadatas=[metadata]
        )

    print("Vector store built successfully.")


if __name__ == "__main__":
    build_store()