from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def generate_embedding(text: str):
    return embedding_model.encode(text).tolist()