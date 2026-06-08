import os

import chromadb
from app.services.embedder import generate_embedding

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "app/chroma_storage")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="nutrition_kb")


def retrieve_documents(query: str, top_k: int = 3):
    print(f"\n\n*********\n{CHROMA_DIR} \n\n {client}\n*********\n")

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    retrieved_context = []

    for doc, meta in zip(documents, metadatas):
        retrieved_context.append({
            "text": doc,
            "source": meta["source_document"],
            "authority": meta["authority"],
            "domain": meta["domain"]
        })

    return retrieved_context