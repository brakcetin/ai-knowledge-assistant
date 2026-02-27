"""AI Knowledge Assistant — Streamlit application entry point."""

import time
import streamlit as st

from app.config import settings, ConfigError
from app.ingestion.file_reader import read_file
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import generate_embeddings
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import retrieve
from app.llm.generator import generate_answer, generate_answer_stream
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
)

# ── Validate Config ──────────────────────────────────────────────────────
try:
    settings.validate()
except ConfigError as e:
    st.error(f"⚙️ Configuration Error: {e}")
    st.stop()

# ── Session State Init ───────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "loaded_documents" not in st.session_state:
    # Rebuild from ChromaDB on restart
    st.session_state.loaded_documents = st.session_state.vector_store.get_all_documents()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

vs: VectorStore = st.session_state.vector_store


# ── Sidebar: Document Management ─────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document Manager")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, Markdown",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Skip if already ingested
            if vs.document_exists(uploaded_file.name):
                st.info(f"ℹ️ '{uploaded_file.name}' is already loaded.")
                continue

            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # 1. Read
                    text = read_file(uploaded_file)
                    # 2. Chunk
                    chunks = chunk_text(text, uploaded_file.name)
                    # 3. Embed
                    chunks = generate_embeddings(chunks)
                    # 4. Store
                    vs.add_chunks(chunks)

                    st.session_state.loaded_documents = vs.get_all_documents()
                    st.success(
                        f"✅ **{uploaded_file.name}** — {len(chunks)} chunks ingested"
                    )
                    logger.info(f"Ingested '{uploaded_file.name}': {len(chunks)} chunks")
                except Exception as e:
                    st.error(f"❌ Failed to process '{uploaded_file.name}': {e}")
                    logger.error(f"Ingestion error for '{uploaded_file.name}': {e}")

    # Loaded documents list
    st.divider()
    st.subheader("📚 Loaded Documents")

    docs = st.session_state.loaded_documents
    if docs:
        for doc_name in docs:
            chunk_count = vs.get_document_chunk_count(doc_name)
            st.write(f"• **{doc_name}** ({chunk_count} chunks)")
        st.caption(f"Total: {vs.count} chunks across {len(docs)} documents")
    else:
        st.caption("No documents loaded yet.")

    # Reset button
    st.divider()
    if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
        vs.delete_collection()
        st.session_state.loaded_documents = []
        st.session_state.chat_history = []
        st.success("All documents and history cleared.")
        st.rerun()

    # Config info
    st.divider()
    st.caption(
        f"**Model:** {settings.LLM_MODEL}\n\n"
        f"**Provider:** {settings.LLM_PROVIDER}\n\n"
        f"**Top-K:** {settings.TOP_K} chunks"
    )


# ── Main Area: Q&A ──────────────────────────────────────────────────────
st.title("🧠 AI Knowledge Assistant")
st.markdown(
    "Upload documents in the sidebar, then ask questions about their content. "
    "Answers are grounded in your documents with source citations."
)

# Display chat history
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        with st.expander("📎 Sources"):
            for s in entry["sources"]:
                st.write(f"• **{s['source']}** — Chunk #{s['chunk_index']}")
        st.caption(f"⚡ Answered in {entry['inference_time']}s using {entry['model']}")

# Question input
question = st.chat_input("Ask a question about your documents...")

if question:
    # Validation
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
        st.stop()

    if not st.session_state.loaded_documents:
        st.warning("⚠️ Please upload at least one document first.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):
        total_start = time.time()

        try:
            # Retrieve relevant chunks
            with st.spinner("🔍 Searching documents..."):
                context_chunks = retrieve(question, vs)

            if not context_chunks:
                st.info(
                    "No relevant content found in the uploaded documents "
                    "for this question."
                )
                st.stop()

            # Check relevance (low similarity warning)
            avg_similarity = sum(
                c["similarity_score"] for c in context_chunks
            ) / len(context_chunks)

            if avg_similarity < 0.3:
                st.warning(
                    "⚠️ Low confidence — the documents may not contain "
                    "information relevant to this question."
                )

            # Stream LLM response
            answer_placeholder = st.empty()
            full_answer = ""

            for token in generate_answer_stream(question, context_chunks):
                full_answer += token
                answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)

            total_elapsed = round(time.time() - total_start, 2)

            # Sources
            sources = [
                {
                    "source": c["source"],
                    "chunk_index": c["chunk_index"],
                }
                for c in context_chunks
            ]

            with st.expander("📎 Sources"):
                for s in sources:
                    st.write(f"• **{s['source']}** — Chunk #{s['chunk_index']}")

            st.caption(
                f"⚡ Answered in {total_elapsed}s using {settings.LLM_MODEL}"
            )

            # Save to history
            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": full_answer,
                    "sources": sources,
                    "model": settings.LLM_MODEL,
                    "inference_time": total_elapsed,
                }
            )

        except ValueError as e:
            st.warning(f"⚠️ {e}")
        except RuntimeError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")
            logger.error(f"Unexpected error: {e}", exc_info=True)
