import time
from fastapi import APIRouter
from app.models.schemas import QueryRequest, QueryResponse
from app.services.embedder import embed_texts
from app.services.retriever import retrieve_chunks
from app.services.synthesizer import synthesize_answer

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    # 1. Embed the question
    q_emb = embed_texts([request.question])[0]

    # 2. Retrieve chunks with latency tracking
    start = time.time()
    chunks_with_scores, chunks_retrieved, chunks_after_filter = retrieve_chunks(q_emb)
    retrieval_latency_ms = round((time.time() - start) * 1000, 2)

    # 3. Synthesize answer
    answer = synthesize_answer(request.question, chunks_with_scores)

    return QueryResponse(
        answer=answer,
        sources_count=len(chunks_with_scores),
        chunks_retrieved=chunks_retrieved,
        chunks_after_filter=chunks_after_filter,
        retrieval_latency_ms=retrieval_latency_ms
    )