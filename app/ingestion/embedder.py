"""Generate embeddings for text chunks using sentence-transformers."""

import time
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level model cache — loaded once, reused across calls
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model (cached singleton)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        start = time.time()
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info(f"Embedding model loaded in {time.time() - start:.2f}s")
    return _model


def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """Add embedding vectors to each chunk dict.

    Args:
        chunks: List of chunk dicts from chunker.

    Returns:
        Same list with 'embedding' key added to each chunk.
    """
    model = _get_model()
    texts = [c["text"] for c in chunks]

    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    elapsed = time.time() - start

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    logger.info(
        f"Generated {len(chunks)} embeddings in {elapsed:.2f}s "
        f"({elapsed / max(len(chunks), 1) * 1000:.1f}ms/chunk)"
    )
    return chunks


def embed_query(query: str) -> list[float]:
    """Generate embedding for a single query string.

    Args:
        query: The search query text.

    Returns:
        Embedding vector as a list of floats.
    """
    model = _get_model()
    embedding = model.encode(query, show_progress_bar=False, convert_to_numpy=True)
    return embedding.tolist()
