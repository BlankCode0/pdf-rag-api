import uuid
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.pdf import extract_text
from app.services.chunker import chunk_text
from app.services.embedder import embed_chunks
from app.services.vector_store import store
from app.models.schemas import UploadResponse
from app.core.config import settings

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {settings.max_file_size_mb}MB."
        )

    # Save PDF temporarily
    doc_id = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(content)

    try:
        # Pipeline: extract → chunk → embed → store
        text       = extract_text(str(pdf_path))
        chunks     = chunk_text(text)
        embeddings = embed_chunks(chunks)
        num_chunks = store(doc_id, chunks, embeddings)

    except ValueError as e:
        # Clean up temp file on failure
        os.remove(pdf_path)
        raise HTTPException(status_code=422, detail=str(e))

    finally:
        # Always clean up the temp PDF — we only need the FAISS index
        if pdf_path.exists():
            os.remove(pdf_path)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_chunks=num_chunks,
        message="PDF processed successfully. Use doc_id to ask questions."
    )