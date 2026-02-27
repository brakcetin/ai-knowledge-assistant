"""Read uploaded files (PDF, TXT, MD) and extract text content."""

from PyPDF2 import PdfReader
from app.utils.logger import get_logger

logger = get_logger(__name__)


def read_file(uploaded_file) -> str:
    """Extract text content from an uploaded file.

    Args:
        uploaded_file: A Streamlit UploadedFile object with .name and .read().

    Returns:
        Extracted text as a single string.

    Raises:
        ValueError: If file type is not supported.
        RuntimeError: If file cannot be read.
    """
    file_name = uploaded_file.name.lower()
    logger.info(f"Reading file: {uploaded_file.name} ({uploaded_file.size} bytes)")

    try:
        if file_name.endswith(".pdf"):
            return _read_pdf(uploaded_file)
        elif file_name.endswith(".txt"):
            return _read_text(uploaded_file)
        elif file_name.endswith(".md"):
            return _read_text(uploaded_file)
        else:
            ext = file_name.rsplit(".", 1)[-1] if "." in file_name else "unknown"
            raise ValueError(
                f"Unsupported file type: .{ext}. Please upload PDF, TXT, or MD files."
            )
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to read {uploaded_file.name}: {e}")
        raise RuntimeError(f"Could not read file '{uploaded_file.name}': {e}")


def _read_pdf(uploaded_file) -> str:
    """Extract text from a PDF file page by page."""
    reader = PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text)
    total_text = "\n\n".join(pages)
    logger.info(
        f"PDF '{uploaded_file.name}': {len(reader.pages)} pages, "
        f"{len(total_text)} chars extracted"
    )
    if not total_text.strip():
        raise RuntimeError(
            f"PDF '{uploaded_file.name}' contains no extractable text. "
            "It may be a scanned document."
        )
    return total_text


def _read_text(uploaded_file) -> str:
    """Read a plain text or markdown file."""
    content = uploaded_file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    logger.info(f"Text file '{uploaded_file.name}': {len(content)} chars")
    if not content.strip():
        raise RuntimeError(f"File '{uploaded_file.name}' is empty.")
    return content
