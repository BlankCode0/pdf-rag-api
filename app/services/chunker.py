from app.core.config import settings


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks using a sliding window.

    chunk_size:    max characters per chunk  (from .env, default 500)
    chunk_overlap: characters shared between
                   consecutive chunks        (from .env, default 50)

    Overlap ensures context isn't lost at chunk boundaries.
    """
    size    = settings.chunk_size
    overlap = settings.chunk_overlap
    chunks  = []
    start   = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap  # slide forward, keeping overlap

    return chunks