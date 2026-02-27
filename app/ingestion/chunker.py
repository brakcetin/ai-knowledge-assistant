"""Split document text into overlapping chunks with metadata."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_text(text: str, file_name: str) -> list[dict]:
    """Split text into chunks and attach metadata.

    Args:
        text: Full document text.
        file_name: Source file name for metadata.

    Returns:
        List of chunk dicts with keys: chunk_id, text, metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)
    total = len(raw_chunks)

    chunks = []
    for i, chunk_text_content in enumerate(raw_chunks):
        chunks.append(
            {
                "chunk_id": f"{file_name}_chunk_{i}",
                "text": chunk_text_content,
                "metadata": {
                    "source": file_name,
                    "chunk_index": i,
                    "total_chunks": total,
                },
            }
        )

    avg_size = sum(len(c["text"]) for c in chunks) / max(len(chunks), 1)
    logger.info(
        f"Chunked '{file_name}': {total} chunks, "
        f"avg {avg_size:.0f} chars/chunk "
        f"(size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})"
    )
    return chunks
