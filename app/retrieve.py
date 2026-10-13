from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from app import config

_model = None
_client = None


def _load():
    global _model, _client
    if _model is None:
        _model = TextEmbedding(config.EMBED_MODEL)
        _client = QdrantClient(path=config.QDRANT_PATH)
    return _model, _client


def get_resources():
    """Return the cached embedding model and Qdrant client for indexing or retrieval."""
    return _load()


def retrieve(question, k=None):
    model, client = _load()
    vector = next(iter(model.query_embed(question))).tolist()
    result = client.query_points(
        config.COLLECTION, query=vector, limit=k or config.TOP_K
    )
    return [
        {"page": p.payload["page"], "text": p.payload["text"], "score": p.score}
        for p in result.points
    ]
