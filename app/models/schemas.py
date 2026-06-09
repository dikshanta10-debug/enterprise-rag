from pydantic import BaseModel

class UploadResponse(BaseModel):
    status: str
    document_name: str
    chunks_indexed: int

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources_count: int