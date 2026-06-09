from app.services.chunker import semantic_chunk

def test_returns_list_of_strings():
    text = "Hello world. This is a test."
    chunks = semantic_chunk(text, chunk_size=50, overlap=0)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)

def test_no_empty_chunks():
    text = "Short text."
    chunks = semantic_chunk(text, chunk_size=1000, overlap=200)
    assert len(chunks) > 0
    assert all(len(c) > 0 for c in chunks)