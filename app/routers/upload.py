import io
import time
from fastapi import APIRouter, UploadFile, File
from app.services.chunker import semantic_chunk
from app.services.embedder import embed_texts
from app.database import get_connection
from app.models.schemas import UploadResponse
from pgvector.psycopg2 import register_vector
import pdfplumber

router = APIRouter()


def clean_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer to keep only the book content."""
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    start_idx = text.find(start_marker)
    if start_idx != -1:
        text = text[start_idx + len(start_marker):]
        # Remove the rest of that line (title line)
        next_newline = text.find("\n")
        if next_newline != -1:
            text = text[next_newline + 1:]

    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]

    return text.strip()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    text = ""

    if file.filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join(
                page.extract_text() for page in pdf.pages if page.extract_text()
            )
    elif file.filename.endswith(".md") or file.filename.endswith(".txt"):
        text = content.decode("utf-8")
        text = clean_gutenberg(text)          # <-- strip Gutenberg boilerplate
    else:
        return UploadResponse(
            status="unsupported file type",
            document_name=file.filename,
            chunks_indexed=0,
        )

    if not text.strip():
        return UploadResponse(
            status="no extractable text",
            document_name=file.filename,
            chunks_indexed=0,
        )

    # Chunk, embed, store
    chunks = semantic_chunk(text)
    embeddings = embed_texts(chunks)

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(conn)

    start_insert = time.time()
    with conn.cursor() as cur:
        for chunk, emb in zip(chunks, embeddings):
            cur.execute(
                "INSERT INTO chunks (document_name, chunk_text, embedding) VALUES (%s, %s, %s)",
                (file.filename, chunk, emb),
            )
    insert_elapsed = time.time() - start_insert
    throughput = len(chunks) / insert_elapsed if insert_elapsed > 0 else 0
    print(
        f"Indexed {len(chunks)} chunks in {insert_elapsed:.2f}s ({throughput:.1f} chunks/sec)"
    )
    conn.commit()
    conn.close()

    return UploadResponse(
        status="indexed",
        document_name=file.filename,
        chunks_indexed=len(chunks),
    )