"""LLM answer generation using Groq or OpenAI API."""

import time
from groq import Groq
from app.config import settings
from app.llm.prompt_builder import build_prompt
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level client cache
_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    """Get or create a cached Groq client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
        logger.info("Groq client initialized")
    return _groq_client


def generate_answer(question: str, context_chunks: list[dict]) -> dict:
    """Generate an answer using the LLM with retrieved context.

    Args:
        question: User's question.
        context_chunks: Retrieved chunks from the retrieval system.

    Returns:
        Dict with keys: answer, sources, model, inference_time.

    Raises:
        RuntimeError: If the LLM API call fails.
    """
    messages = build_prompt(question, context_chunks)

    # Extract sources from context
    sources = [
        {
            "source": c.get("source", "unknown"),
            "chunk_index": c.get("chunk_index", -1),
        }
        for c in context_chunks
    ]

    start = time.time()

    try:
        if settings.LLM_PROVIDER == "groq":
            answer = _call_groq(messages)
        else:
            raise RuntimeError(
                f"Provider '{settings.LLM_PROVIDER}' is not supported. Use 'groq'."
            )
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise RuntimeError(
            f"Failed to get a response from the AI model. "
            f"Please check your API key and try again. Error: {e}"
        )

    elapsed = time.time() - start

    logger.info(
        f"LLM response generated in {elapsed:.2f}s "
        f"(model: {settings.LLM_MODEL}, provider: {settings.LLM_PROVIDER})"
    )

    return {
        "answer": answer,
        "sources": sources,
        "model": settings.LLM_MODEL,
        "inference_time": round(elapsed, 2),
    }


def _call_groq(messages: list[dict]) -> str:
    """Call the Groq API and return the response text."""
    client = _get_groq_client()

    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
        timeout=10,
    )

    return response.choices[0].message.content


def generate_answer_stream(question: str, context_chunks: list[dict]):
    """Generate a streaming answer using the LLM.

    Yields:
        str: Token chunks as they arrive.

    Also returns metadata via the .sources, .model attributes on the generator
    (access after iteration).
    """
    messages = build_prompt(question, context_chunks)

    sources = [
        {
            "source": c.get("source", "unknown"),
            "chunk_index": c.get("chunk_index", -1),
        }
        for c in context_chunks
    ]

    try:
        if settings.LLM_PROVIDER == "groq":
            client = _get_groq_client()
            stream = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        else:
            raise RuntimeError(f"Provider '{settings.LLM_PROVIDER}' not supported.")
    except Exception as e:
        logger.error(f"Streaming LLM call failed: {e}")
        yield f"\n\n⚠️ Error: {e}"
