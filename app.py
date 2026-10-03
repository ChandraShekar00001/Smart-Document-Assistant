"""Smart Document Assistant - Streamlit Application.

A production-grade GenAI application featuring Retrieval-Augmented Generation (RAG)
grounded in uploaded PDF, TXT, and DOCX documents with ChromaDB vector search, Groq LLM,
source attribution, conversation history, and automatic document summarization.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
import streamlit as st

from config.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K,
    GROQ_API_KEY,
    INDEX_FILE_PATH,
    METADATA_FILE_PATH,
    SAMPLE_DOCS_DIR,
    UPLOADS_DIR,
    get_groq_api_key,
    is_api_key_configured,
)
from src.embeddings import EmbeddingService
from src.llm_service import LLMService, UNKNOWN_ANSWER_MESSAGE
from src.rag_pipeline import RAGPipeline
from src.source_formatter import SourceFormatter
from src.summarizer import DocumentSummarizer
from src.vector_store import VectorStore

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Smart Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
    .source-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_embedding_service(model_name: str = DEFAULT_EMBEDDING_MODEL) -> EmbeddingService:
    """Load and cache the SentenceTransformer embedding model."""
    return EmbeddingService.get_instance(model_name)


def init_session_state():
    """Initialize application session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "document_registry" not in st.session_state:
        st.session_state.document_registry = {}

    if "summary_store" not in st.session_state:
        st.session_state.summary_store = {}


init_session_state()

# Load shared services
embedding_svc = get_embedding_service()
vector_store = VectorStore(
    dimension=embedding_svc.dimension,
    index_path=INDEX_FILE_PATH,
    metadata_path=METADATA_FILE_PATH,
)
vector_store.load()

llm_service = LLMService(
    api_key=GROQ_API_KEY,
    model_name=DEFAULT_GROQ_MODEL,
)

rag_pipeline = RAGPipeline(
    vector_store=vector_store,
    embedding_service=embedding_svc,
    llm_service=llm_service,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    top_k=DEFAULT_TOP_K,
    relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD,
)

summarizer = DocumentSummarizer(llm_service=llm_service)

# ==========================================
# SIDEBAR: Document Management & Config
# ==========================================
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/description/materialicons/48dp/1x/baseline_description_black_48dp.png",
        width=48,
    )
    st.title("Document Manager")

    # API Status Indicator
    st.subheader("🔑 LLM Configuration")
    if is_api_key_configured():
        st.success("🟢 Groq API configured")
    else:
        st.warning("⚠️ GROQ_API_KEY is not configured. Add it to your .env file.")

    st.markdown("---")

    # File Upload Section
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload multiple documents to build your local retrieval knowledge base.",
    )

    col_proc, col_sample = st.columns([1, 1])

    with col_proc:
        process_clicked = st.button("⚡ Index Files", type="primary", use_container_width=True)

    with col_sample:
        load_sample_clicked = st.button("📁 Load Samples", use_container_width=True)

    # Handle Sample Document Loading
    if load_sample_clicked:
        if SAMPLE_DOCS_DIR.exists():
            sample_files = list(SAMPLE_DOCS_DIR.glob("*.*"))
            valid_samples = [f for f in sample_files if f.suffix.lower() in {".pdf", ".txt", ".docx"}]
            if valid_samples:
                with st.spinner("Indexing synthetic sample documents..."):
                    for s_file in valid_samples:
                        try:
                            res = rag_pipeline.process_document(s_file, s_file.name)
                            st.session_state.document_registry[s_file.name] = res
                        except Exception as e:
                            st.error(f"Failed to process sample {s_file.name}: {e}")
                st.success(f"Loaded {len(valid_samples)} sample documents!")
                st.rerun()
            else:
                st.info("No sample files found in data/sample_documents.")
        else:
            st.error("Sample directory does not exist.")

    # Handle Uploaded Files Processing
    if process_clicked:
        if not uploaded_files:
            st.warning("Please select at least one document to index.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}...")
                # Save to disk temporarily
                dest_path = UPLOADS_DIR / uploaded_file.name
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    res = rag_pipeline.process_document(dest_path, uploaded_file.name)
                    st.session_state.document_registry[uploaded_file.name] = res
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("Indexing completed!")
            st.success("Documents successfully processed and indexed into ChromaDB.")
            st.rerun()

    st.markdown("---")

    # Document Collection Management
    st.subheader("🗂️ Indexed Documents")
    indexed_docs = vector_store.get_indexed_documents()

    if indexed_docs:
        for doc in indexed_docs:
            col_info, col_del = st.columns([3, 1])
            with col_info:
                p_text = f" ({doc['total_pages']} pp)" if doc["total_pages"] else ""
                st.markdown(f"**{doc['filename']}**{p_text} — `{doc['chunk_count']} chunks`")
            with col_del:
                if st.button("🗑️", key=f"del_{doc['filename']}", help=f"Remove {doc['filename']}"):
                    vector_store.delete_document(doc["filename"], embedding_service=embedding_svc)
                    if doc["filename"] in st.session_state.document_registry:
                        del st.session_state.document_registry[doc["filename"]]
                    st.rerun()

        if st.button("🧹 Clear All Documents", use_container_width=True):
            vector_store.clear()
            st.session_state.document_registry.clear()
            st.session_state.chat_history.clear()
            st.session_state.summary_store.clear()
            st.success("All documents and vector indices cleared.")
            st.rerun()
    else:
        st.info("No documents currently indexed. Upload files or click 'Load Samples' above.")

    st.markdown("---")

    # System Status Metrics
    st.subheader("📊 Knowledge Base Stats")
    st.metric("Total Indexed Documents", len(indexed_docs))
    st.metric("Total Indexed Chunks", vector_store.count())
    st.caption(f"**Embedding Model:** `{DEFAULT_EMBEDDING_MODEL}` (384-dim)")
    st.caption(f"**Vector Store:** `ChromaDB PersistentClient` (Cosine Sim)")
    st.caption(f"**LLM Model:** Groq `{DEFAULT_GROQ_MODEL}`")


# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
st.markdown('<div class="main-header">Smart Document Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Ask natural-language questions grounded strictly in your documents with verified source citations, '
    'or generate structured executive summaries.'
    '</div>',
    unsafe_allow_html=True,
)

tab_qa, tab_summary, tab_arch = st.tabs([
    "💬 Ask Questions (RAG)",
    "📑 Automatic Document Summary",
    "🔍 System Architecture & Inspector",
])

# ------------------------------------------
# TAB 1: QUESTION ANSWERING USING RAG
# ------------------------------------------
with tab_qa:
    if vector_store.count() == 0:
        st.info(
            "👋 **Getting Started**: No documents are currently loaded. "
            "Upload files via the sidebar or click **'📁 Load Samples'** to load sample documents with company policies.",
            icon="ℹ️",
        )

    # Demo Quick Question Prompts
    st.markdown("**Sample Questions to try:**")
    prompt_cols = st.columns(3)
    sample_queries = [
        "What is the maximum reimbursement for home office equipment?",
        "How many days of paid vacation do employees receive annually?",
        "What is the company policy regarding pet bereavement leave?",
    ]

    selected_prompt = None
    for idx, prompt_text in enumerate(sample_queries):
        with prompt_cols[idx]:
            if st.button(prompt_text, key=f"sample_q_{idx}", use_container_width=True):
                selected_prompt = prompt_text

    # Question Input Form
    with st.form("qa_form", clear_on_submit=False):
        user_query = st.text_input(
            "Enter your question about the documents:",
            value=selected_prompt or "",
            placeholder="e.g., What are the core working hours for remote workers?",
        )
        col_submit, col_clear = st.columns([1, 6])
        with col_submit:
            submit_question = st.form_submit_width = st.form_submit_button("Ask Question", type="primary")

    if submit_question and user_query:
        if not is_api_key_configured():
            st.error("⚠️ **GROQ_API_KEY is not configured. Add it to your .env file.**")
        else:
            with st.spinner("Searching documents and generating grounded answer..."):
                # Prepare conversation history turns for LLM context
                conv_history = [
                    {"role": "user", "content": item["question"]}
                    if i % 2 == 0 else {"role": "assistant", "content": item["answer"]}
                    for i, item in enumerate(st.session_state.chat_history[-4:])
                ]

                response = rag_pipeline.ask(
                    question=user_query,
                    conversation_history=conv_history,
                )

                # Append to session history
                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer": response.answer,
                    "sources": response.sources,
                    "has_sufficient_context": response.has_sufficient_context,
                })

    # Render Current / Latest Response
    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]

        st.markdown("### Answer")
        if latest["has_sufficient_context"]:
            st.markdown(latest["answer"])
        else:
            st.warning(latest["answer"], icon="🔍")

        # Source Citations Section
        st.markdown("### 📚 Source Citations & Excerpts")
        if latest["sources"]:
            for s_idx, src in enumerate(latest["sources"], 1):
                formatted = SourceFormatter.format_source_item(src)
                with st.expander(
                    f"**Source [{s_idx}]**: {formatted['source']} — {formatted['page_label']} "
                    f"(Cosine Similarity: {formatted['score']:.2f})",
                    expanded=(s_idx == 1),
                ):
                    st.markdown(f"**Original Excerpt:**\n> {formatted['content']}")
                    st.caption(
                        f"Chunk ID: `{formatted['chunk_id']}` | "
                        f"Metric: `{formatted['score_label']}` (Vector cosine similarity, not calibrated probability)"
                    )
        else:
            st.caption("_No supporting document sources were retrieved for this query._")

        st.markdown("---")

        # Session Conversation History
        with st.expander(f"📜 Session History ({len(st.session_state.chat_history)} Q&A pairs)", expanded=False):
            for i, chat in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                if chat["sources"]:
                    st.caption(f"Sources used: {', '.join({s['source'] for s in chat['sources']})}")
                st.markdown("---")

            if st.button("Clear Conversation History"):
                st.session_state.chat_history.clear()
                st.rerun()

# ------------------------------------------
# TAB 2: AUTOMATIC DOCUMENT SUMMARY
# ------------------------------------------
with tab_summary:
    st.subheader("📑 Document Summarization")
    st.write(
        "Generate a structured, grounded summary of any document in your collection. "
        "For large documents, the assistant analyzes sections hierarchically and synthesizes a cohesive executive brief."
    )

    indexed_docs = vector_store.get_indexed_documents()
    available_doc_names = [d["filename"] for d in indexed_docs]

    # Also include sample files if not yet indexed
    if SAMPLE_DOCS_DIR.exists():
        for sf in SAMPLE_DOCS_DIR.glob("*.*"):
            if sf.name not in available_doc_names and sf.suffix.lower() in {".pdf", ".txt", ".docx"}:
                available_doc_names.append(sf.name)

    if not available_doc_names:
        st.warning("Please upload or load at least one document to generate summaries.")
    else:
        col_select, col_btn = st.columns([3, 1])
        with col_select:
            selected_doc = st.selectbox("Select document to summarize:", options=available_doc_names)

        with col_btn:
            st.write("")  # Vertical spacing
            gen_summary_clicked = st.button("✨ Generate Summary", type="primary", use_container_width=True)

        if gen_summary_clicked:
            if not is_api_key_configured():
                st.error("⚠️ **GROQ_API_KEY is not configured. Add it to your .env file.**")
            else:
                # Find path to document
                doc_path = UPLOADS_DIR / selected_doc
                if not doc_path.exists():
                    doc_path = SAMPLE_DOCS_DIR / selected_doc

                if doc_path.exists():
                    with st.spinner(f"Analyzing and summarizing '{selected_doc}'..."):
                        summary_text, was_cached = summarizer.summarize_document(doc_path, selected_doc)
                        st.session_state.summary_store[selected_doc] = {
                            "text": summary_text,
                            "cached": was_cached,
                        }
                else:
                    # Document might only exist as indexed chunks in vector store
                    chunks_for_doc = [c.content for c in vector_store.chunks if c.source == selected_doc]
                    if chunks_for_doc:
                        with st.spinner(f"Summarizing indexed chunks for '{selected_doc}'..."):
                            combined_text = "\n\n".join(chunks_for_doc)
                            summary_text, was_cached = summarizer.summarize_text(combined_text, selected_doc)
                            st.session_state.summary_store[selected_doc] = {
                                "text": summary_text,
                                "cached": was_cached,
                            }
                    else:
                        st.error(f"Cannot find source file or chunks for '{selected_doc}'.")

        # Display Summary
        if selected_doc in st.session_state.summary_store:
            stored = st.session_state.summary_store[selected_doc]
            cache_tag = "⚡ Loaded from cache" if stored.get("cached") else "✨ Freshly generated"
            st.success(f"Summary for **{selected_doc}** ({cache_tag})")
            st.markdown(stored["text"])

# ------------------------------------------
# TAB 3: SYSTEM ARCHITECTURE & INSPECTION
# ------------------------------------------
with tab_arch:
    st.subheader("🏗️ Architecture & Data Flow")
    st.write(
        "Smart Document Assistant uses a decoupled architecture separating offline ingestion "
        "and indexing from online retrieval and grounded generation."
    )

    st.markdown(
        """
        ```mermaid
        flowchart TD
            subgraph Ingestion["Offline Ingestion Pipeline"]
                Doc[Document: PDF / TXT / DOCX] --> Extract[Text Extraction & Cleaning]
                Extract --> Chunk[Recursive Text Splitter]
                Chunk --> Embed[Sentence-Transformers all-MiniLM-L6-v2]
                Embed --> Chroma[ChromaDB Vector Store: PersistentClient]
                Chunk --> Chroma
            end

            subgraph Online["Online RAG Query Pipeline"]
                User([User Query]) --> QEmbed[Query Embedding]
                QEmbed --> Search[ChromaDB Similarity Search]
                Chroma -.-> Search
                Search --> Filter[Relevance Threshold Filter: score >= 0.35]
                Filter --> Ctx[Context Builder: Max 6,000 chars]
                Ctx --> Prompt[Anti-Injection Grounded Prompt]
                Prompt --> Groq[Groq LLM: openai/gpt-oss-120b]
                Groq --> Answer([Grounded Answer + Sources])
            end

            subgraph Summarization["Document Summarization Pipeline"]
                DocText --> CacheCheck{"Summary cached?"}
                CacheCheck -- "Yes" --> DocSummary([Document Insights: Summary, Key Points, Numbers, Action Items])
                CacheCheck -- "No" --> TokenBudget{"Estimated input <= safe budget?\n(Default: 5,138 tokens)"}
                TokenBudget -- "Within budget" --> GroqSummary[Single Groq Summary Call]
                TokenBudget -- "Over budget" --> LocalCompress[Local Heuristic Compression: No API Call]
                LocalCompress --> CompressedFits{"Compressed text within budget?"}
                CompressedFits -- "Yes" --> GroqSummary
                CompressedFits -- "No" --> TooLarge[User-Friendly Warning: No API Call]
                GroqSummary -- "Valid response" --> DocSummary
                GroqSummary -- "429 / rate limit" --> RateLimitStop[Return Warning; Stop Without Retry]
            end
        ```
        """
    )

    st.markdown("---")
    st.subheader("🔎 Raw Vector Store Chunks Inspector")
    if vector_store.count() > 0:
        st.write(f"Displaying {min(10, vector_store.count())} of {vector_store.count()} indexed chunks:")
        for i, chk in enumerate(vector_store.chunks[:10]):
            with st.expander(f"Chunk {i+1}: {chk.source} (Page: {chk.page_number or 'N/A'}, ID: {chk.chunk_id})"):
                st.code(chk.content, language="text")
                st.json(chk.metadata)
    else:
        st.info("No chunks indexed to inspect.")
