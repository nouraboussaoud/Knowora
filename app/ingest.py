import pymupdf  # au lieu de : import fitz
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app import config


def extract_pages(path):
    with pymupdf.open(path) as doc:  # au lieu de : fitz.open(path)
        return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


def chunk_text(text, size, overlap):
    words = text.split()
    step = size - overlap
    for start in range(0, max(len(words), 1), step):
        piece = words[start:start + size]
        if len(piece) >= 30:  # ignore les morceaux trop courts (en-têtes, pieds de page)
            yield " ".join(piece)
        if start + size >= len(words):
            break


def build_chunks(path):
    chunks = []
    for page, text in extract_pages(path):
        for piece in chunk_text(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP):
            chunks.append({"page": page, "text": piece})
    return chunks


def index_document(path, model=None, client=None):
    """Replace the active collection with chunks from one PDF document."""
    chunks = build_chunks(path)
    if not chunks:
        raise ValueError("The PDF does not contain enough extractable text to index.")

    model = model or TextEmbedding(config.EMBED_MODEL)
    client = client or QdrantClient(path=config.QDRANT_PATH)
    vectors = list(model.embed([c["text"] for c in chunks]))
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        config.COLLECTION,
        vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=i, vector=v.tolist(), payload=c)
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    for i in range(0, len(points), 128):
        client.upsert(config.COLLECTION, points=points[i:i + 128])
    return {"chunks": len(chunks), "pages": len({chunk["page"] for chunk in chunks})}


def main():
    result = index_document(config.PDF_PATH)
    print(f"Index built: {result['chunks']} chunks from {result['pages']} pages.")


if __name__ == "__main__":
    main()
