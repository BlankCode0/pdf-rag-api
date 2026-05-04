import faiss
import numpy as np
import json
import os
from pathlib import Path
from app.services.embedder import embed_chunks

STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)


def _index_path(doc_id: str) -> str:
    return str(STORAGE_DIR / f"{doc_id}.index")


def _chunks_path(doc_id: str) -> str:
    return str(STORAGE_DIR / f"{doc_id}.json")


def store(doc_id: str, chunks: list[str], embeddings: np.ndarray) -> int:
    """
    Build a FAISS index from embeddings and save it to disk.
    Also saves the raw chunks so we can retrieve text later.
    Returns the number of chunks stored.
    """
    dim = embeddings.shape[1]

    # IndexFlatL2 = exact nearest-neighbour search (good for small-medium docs)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, _index_path(doc_id))

    # Save chunks as JSON (needed to return text during retrieval)
    with open(_chunks_path(doc_id), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    return index.ntotal


def search(doc_id: str, query_vector: np.ndarray, top_k: int) -> list[str]:
    """
    Load the FAISS index for a document and return the top-k
    most similar chunks to the query vector.
    Raises FileNotFoundError if the doc_id doesn't exist.
    """
    idx_path = _index_path(doc_id)
    chunks_path = _chunks_path(doc_id)

    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"No index found for doc_id: {doc_id}")

    index = faiss.read_index(idx_path)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # distances: similarity scores, indices: chunk positions
    distances, indices = index.search(query_vector, top_k)

    # Filter out -1 (returned when fewer chunks than top_k exist)
    results = [
        chunks[i] for i in indices[0] if i != -1
    ]
    return results