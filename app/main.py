from fastapi import FastAPI
from app.routes import upload, query

app = FastAPI(
    title="PDF RAG API",
    description="Upload a PDF and ask questions about it using RAG + Groq LLM",
    version="0.1.0",
)

app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "version": "0.1.0"}

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "PDF RAG API is running",
        "docs": "/docs",
        "health": "/health"
    }