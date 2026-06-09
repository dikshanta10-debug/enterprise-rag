from sentence_transformers import SentenceTransformer
import os

# Load model once globally
model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return list of embeddings, each a list of floats."""
    return model.encode(texts).tolist()