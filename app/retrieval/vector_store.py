"""ChromaDB vector store for storing and querying document embeddings."""

import os
import chromadb
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Manages a ChromaDB collection for document chunk embeddings."""

    COLLECTION_NAME = "knowledge_base"

    def __init__(self, persist_dir: str | None = None):
        """Initialize ChromaDB client with persistent storage.

        Args:
            persist_dir: Directory for ChromaDB persistence. Uses config default if None.
        """
        self._persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        os.makedirs(self._persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB initialized at '{self._persist_dir}' — "
            f"collection '{self.COLLECTION_NAME}' has {self._collection.count()} items"
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add embedded chunks to the collection.

        Args:
            chunks: List of dicts with keys: chunk_id, text, metadata, embedding.
        """
        if not chunks:
            return

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            f"Added {len(chunks)} chunks to collection "
            f"(total: {self._collection.count()})"
        )

    def query(self, query_embedding: list[float], top_k: int | None = None) -> list[dict]:
        """Find the most similar chunks to a query embedding.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return. Uses config default if None.

        Returns:
            List of result dicts with: chunk_id, text, metadata, distance.
        """
        top_k = top_k or settings.TOP_K

        if self._collection.count() == 0:
            logger.warning("Query on empty collection — no documents loaded.")
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        logger.info(f"Query returned {len(output)} results")
        return output

    def get_all_documents(self) -> list[str]:
        """Return a list of unique source file names in the collection."""
        if self._collection.count() == 0:
            return []

        all_data = self._collection.get(include=["metadatas"])
        sources = set()
        for meta in all_data["metadatas"]:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    def get_document_chunk_count(self, file_name: str) -> int:
        """Return the number of chunks for a specific document."""
        results = self._collection.get(
            where={"source": file_name}, include=["metadatas"]
        )
        return len(results["ids"])

    def document_exists(self, file_name: str) -> bool:
        """Check if a document has already been ingested."""
        results = self._collection.get(
            where={"source": file_name}, include=[]
        )
        return len(results["ids"]) > 0

    def delete_collection(self) -> None:
        """Delete the entire collection and recreate it empty."""
        self._client.delete_collection(name=self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection deleted and recreated.")

    @property
    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._collection.count()
