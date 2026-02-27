"""Unit tests for the AI Knowledge Assistant core modules."""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings, ConfigError
from app.ingestion.chunker import chunk_text
from app.llm.prompt_builder import build_prompt


# ── Config Tests ─────────────────────────────────────────────────────────

def test_config_validates_with_groq_key(monkeypatch):
    """Config should pass validation when GROQ_API_KEY is set."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test_key_123")
    s = Settings()
    assert s.validate() is True


def test_config_fails_without_api_key(monkeypatch):
    """Config should raise ConfigError when API key is missing."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")
    s = Settings()
    try:
        s.validate()
        assert False, "Should have raised ConfigError"
    except ConfigError:
        pass


def test_config_fails_invalid_provider(monkeypatch):
    """Config should reject unknown LLM providers."""
    monkeypatch.setenv("LLM_PROVIDER", "invalid_provider")
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    s = Settings()
    try:
        s.validate()
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        assert "invalid_provider" in str(e)


def test_config_active_api_key(monkeypatch):
    """active_api_key should return the correct key for the provider."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "groq_key_123")
    monkeypatch.setenv("OPENAI_API_KEY", "openai_key_456")
    s = Settings()
    assert s.active_api_key == "groq_key_123"


# ── Chunker Tests ────────────────────────────────────────────────────────

def test_chunk_text_single_chunk():
    """Short text should produce a single chunk."""
    chunks = chunk_text("Hello world, this is a test.", "test.txt")
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["source"] == "test.txt"
    assert chunks[0]["chunk_id"] == "test.txt_chunk_0"


def test_chunk_text_multiple_chunks():
    """Long text should produce multiple chunks."""
    long_text = "This is a sentence. " * 200  # ~4000 chars
    chunks = chunk_text(long_text, "long.pdf")
    assert len(chunks) > 1
    # Verify sequential chunk IDs
    for i, chunk in enumerate(chunks):
        assert chunk["metadata"]["chunk_index"] == i
        assert chunk["metadata"]["total_chunks"] == len(chunks)


def test_chunk_text_metadata():
    """Each chunk should have correct metadata structure."""
    chunks = chunk_text("Some test content here.", "doc.md")
    chunk = chunks[0]
    assert "chunk_id" in chunk
    assert "text" in chunk
    assert "metadata" in chunk
    assert chunk["metadata"]["source"] == "doc.md"
    assert chunk["metadata"]["chunk_index"] == 0


# ── Prompt Builder Tests ─────────────────────────────────────────────────

def test_build_prompt_structure():
    """Prompt should have system and user messages."""
    context = [
        {"text": "Some context", "source": "doc.pdf", "chunk_index": 0, "similarity_score": 0.9}
    ]
    messages = build_prompt("What is this?", context)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_prompt_includes_question():
    """User message should contain the question."""
    context = [
        {"text": "Context text", "source": "file.txt", "chunk_index": 1, "similarity_score": 0.8}
    ]
    messages = build_prompt("What is the meaning?", context)
    assert "What is the meaning?" in messages[1]["content"]


def test_build_prompt_includes_sources():
    """User message should reference source documents."""
    context = [
        {"text": "Data about cats", "source": "animals.pdf", "chunk_index": 3, "similarity_score": 0.85}
    ]
    messages = build_prompt("Tell me about cats", context)
    assert "animals.pdf" in messages[1]["content"]
    assert "Chunk #3" in messages[1]["content"]


def test_build_prompt_hallucination_guard():
    """System prompt should instruct the model to refuse when context is insufficient."""
    context = [{"text": "x", "source": "x.txt", "chunk_index": 0, "similarity_score": 0.5}]
    messages = build_prompt("anything", context)
    system_msg = messages[0]["content"].lower()
    assert "don't have enough information" in system_msg or "only" in system_msg
