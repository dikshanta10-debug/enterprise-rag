import re

def clean_chunk(text: str) -> str:
    # Collapse multiple whitespace into single space, trim
    return re.sub(r'\s+', ' ', text).strip()

def synthesize_answer(question: str, chunks_with_scores: list[tuple[str, float]]) -> str:
    if not chunks_with_scores:
        return "I couldn't find relevant information in the knowledge base."

    # For now, combine top 3 chunks into a readable answer (no LLM needed)
    parts = []
    for i, (chunk, score) in enumerate(chunks_with_scores[:3], 1):
        parts.append(f"[Source {i}, relevance {score:.2f}] {clean_chunk(chunk)}")
    return "\n\n".join(parts)