from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    message: str


class AskRequest(BaseModel):
    doc_id: str
    question: str


class AskResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    chunks_used: int