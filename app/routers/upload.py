import io
from fastapi import APIRouter, UploadFile, File, Depends
from PyPDF2 import PdfReader
from app.services.chunker import semantic_chunk
from app.services.embedder import embed_texts
from app.database import get_connection
from app.models.schemas import UploadResponse

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    text = ""
    if file.filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file.filename.endswith(".md") or file.filename.endswith(".txt"):
        text = content.decode("utf-8")
    else:
        return UploadResponse(status="unsupported file type", document_name=file.filename, chunks_indexed=0)

    if not text.strip():
        return UploadResponse(status="no extractable text", document_name=file.filename, chunks_indexed=0)

    chunks = semantic_chunk(text)
    embeddings = embed_texts(chunks)

    conn = get_connection()
    # We need to ensure vector type is registered before inserting
    from pgvector.psycopg2 import register_vector
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)

    with conn.cursor() as cur:
        for chunk, emb in zip(chunks, embeddings):
            cur.execute(
                "INSERT INTO chunks (document_name, chunk_text, embedding) VALUES (%s, %s, %s)",
                (file.filename, chunk, emb)
            )
    conn.commit()
    conn.close()

    return UploadResponse(status="indexed", document_name=file.filename, chunks_indexed=len(chunks))