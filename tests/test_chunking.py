from app.ingest import chunk_text


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = list(chunk_text(text, 100, 20))
    assert len(chunks) >= 5
    assert all(len(c.split()) <= 100 for c in chunks)
    assert chunks[1].split()[0] == "w80"