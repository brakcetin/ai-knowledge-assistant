"""High-level retrieval interface — query documents by natural language."""

import time
from app.ingestion.embedder import embed_query
from app.retrieval.vector_store import VectorStore
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve(query: str, vector_store: VectorStore, top_k: int | None = None) -> list[dict]:
    """Retrieve the most relevant chunks for a given question.

    Args:
        query: User's natural language question.
        vector_store: Initialized VectorStore instance.
        top_k: Number of results. Uses config default if None.

    Returns:
        List of result dicts with: chunk_id, text, source, chunk_index, similarity_score.

    Raises:
        ValueError: If no documents are loaded.
    """
    top_k = top_k or settings.TOP_K

    if vector_store.count == 0:
        raise ValueError(
            "No documents are loaded. Please upload documents before asking questions."
        )

    start = time.time()

    # Embed the query using the same model as ingestion
    query_embedding = embed_query(query)

    # Search ChromaDB
    raw_results = vector_store.query(query_embedding, top_k=top_k)

    # Format results
    results = []
    for r in raw_results:
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score: 1 - (distance / 2)
        similarity = 1 - (r["distance"] / 2)
        results.append(
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "source": r["metadata"].get("source", "unknown"),
                "chunk_index": r["metadata"].get("chunk_index", -1),
                "similarity_score": round(similarity, 4),
            }
        )

    elapsed = time.time() - start
    logger.info(
        f"Retrieved {len(results)} chunks for query '{query[:50]}...' "
        f"in {elapsed * 1000:.0f}ms"
    )
    return results
