from app.database import get_connection
from pgvector.psycopg2 import register_vector

def retrieve_chunks(query_embedding: list[float], top_k: int = 10, threshold: float = 0.5):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(conn)

    # Convert the Python list to a PostgreSQL vector literal string
    vector_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_text, 1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vector_str, vector_str, top_k))
        rows = cur.fetchall()
    conn.close()

    chunks_retrieved = len(rows)

    # Apply relevance guardrail
    relevant = [(row["chunk_text"], row["similarity"]) for row in rows if row["similarity"] >= threshold]
    chunks_after_filter = len(relevant)

    return relevant, chunks_retrieved, chunks_after_filter