import pymupdf  # au lieu de : import fitz
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app import config


def extract_pages(path):
    doc = pymupdf.open(path)  # au lieu de : fitz.open(path)
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


def main():
    chunks = build_chunks(config.PDF_PATH)
    print(f"{len(chunks)} morceaux extraits")

    model = TextEmbedding(config.EMBED_MODEL)
    vectors = list(model.embed([c["text"] for c in chunks]))

    client = QdrantClient(path=config.QDRANT_PATH)
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
    print("Index construit.")


if __name__ == "__main__":
    main()