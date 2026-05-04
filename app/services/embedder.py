from sentence_transformers import SentenceTransformer
from app.core.config import settings
import numpy as np

# Don't load at import time — load on first use
_model = None

def _get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        print("Embedding model loaded.")
    return _model


def embed_chunks(chunks: list[str]) -> np.ndarray:
    model = _get_model()
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    model = _get_model()
    vector = model.encode([query], convert_to_numpy=True)
    return vector.astype("float32")