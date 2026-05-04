from fastapi import APIRouter, HTTPException
from app.models.schemas import AskRequest, AskResponse
from app.services.embedder import embed_query
from app.services.vector_store import search
from app.services.llm import ask_llm
from app.core.config import settings

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequest):
    """
    Ask a question about a previously uploaded PDF.
    
    Flow:
      1. Embed the question using the same model used during upload
      2. Search FAISS for top-k most relevant chunks
      3. Send question + chunks to Groq LLM
      4. Return the answer
    """
    if not body.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Step 1: embed the question
    query_vector = embed_query(body.question)

    # Step 2: retrieve relevant chunks from FAISS
    try:
        chunks = search(body.doc_id, query_vector, top_k=settings.top_k)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{body.doc_id}' not found. Please upload it first."
        )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the document for this question."
        )

    # Step 3: ask LLM with context
    answer = ask_llm(body.question, chunks)

    return AskResponse(
        doc_id=body.doc_id,
        question=body.question,
        answer=answer,
        chunks_used=len(chunks)
    )