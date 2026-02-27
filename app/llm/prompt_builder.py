"""Build structured prompts with retrieved context for the LLM."""

from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful AI Knowledge Assistant. Answer the user's question based ONLY on the provided context from uploaded documents.

Rules:
1. Only use information from the provided context to answer.
2. If the context does not contain enough information to answer, say: "I don't have enough information in the uploaded documents to answer this question."
3. Always cite your sources by referencing the document name and chunk number in your answer (e.g., [Source: document.pdf, Chunk #3]).
4. Be concise and accurate.
5. Format your answer clearly with proper structure."""


def build_prompt(question: str, context_chunks: list[dict]) -> list[dict]:
    """Build a chat-format message list for the LLM.

    Args:
        question: User's question.
        context_chunks: Retrieved chunks with keys: text, source, chunk_index.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    # Format context blocks
    context_parts = []
    for chunk in context_chunks:
        source = chunk.get("source", "unknown")
        idx = chunk.get("chunk_index", "?")
        score = chunk.get("similarity_score", 0)
        context_parts.append(
            f"[Source: {source}, Chunk #{idx}] (relevance: {score:.0%})\n"
            f"{chunk['text']}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"Context from uploaded documents:\n\n"
        f"---\n\n{context_text}\n\n---\n\n"
        f"Question: {question}\n\n"
        f"Please answer based on the context above and cite your sources."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    total_chars = sum(len(m["content"]) for m in messages)
    logger.info(f"Prompt built: {total_chars} chars, {len(context_chunks)} context chunks")

    return messages
