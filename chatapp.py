import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4
MIN_RELEVANCE_SCORE = 0.25
UNKNOWN_ANSWER = "I couldn't find sufficient information to answer this question in the uploaded documents."
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"


def get_api_key():
    return os.getenv("GOOGLE_API_KEY", "").strip()


def extract_text_documents(uploaded_files):
    documents = []
    errors = []
    for uploaded_file in uploaded_files:
        filename = Path(uploaded_file.name).name
        extension = Path(filename).suffix.lower()
        try:
            if extension == ".pdf":
                reader = PdfReader(uploaded_file)
                file_documents = []
                for page_number, page in enumerate(reader.pages, start=1):
                    page_text = (page.extract_text() or "").strip()
                    if page_text:
                        file_documents.append(Document(
                            page_content=page_text,
                            metadata={"source": filename, "page": page_number, "file_type": "PDF"},
                        ))
                if not file_documents:
                    errors.append(f"{filename}: no extractable text was found.")
                else:
                    documents.extend(file_documents)
            elif extension == ".txt":
                text = uploaded_file.getvalue().decode("utf-8", errors="replace").strip()
                if not text:
                    errors.append(f"{filename}: the text file is empty.")
                else:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": filename, "section": "Text document", "file_type": "TXT"},
                    ))
            else:
                errors.append(f"{filename}: unsupported file type. Upload PDF or TXT files only.")
        except Exception as exc:
            errors.append(f"{filename}: could not be processed ({exc}).")
    return documents, errors


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embedding=embeddings)


def get_relevant_documents(vector_store, question):
    try:
        scored_documents = vector_store.similarity_search_with_relevance_scores(question, k=RETRIEVAL_K)
        return [document for document, score in scored_documents if score >= MIN_RELEVANCE_SCORE]
    except Exception:
        return vector_store.similarity_search(question, k=RETRIEVAL_K)


def format_source(metadata):
    source = metadata.get("source", "Unknown source")
    if metadata.get("page") is not None:
        return f"{source} - Page {metadata['page']}"
    if metadata.get("section"):
        return f"{source} - {metadata['section']}"
    return source


def get_unique_sources(documents):
    sources = []
    seen = set()
    for document in documents:
        source_label = format_source(document.metadata)
        if source_label not in seen:
            sources.append(source_label)
            seen.add(source_label)
    return sources


def get_chat_model():
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)


def answer_question(question, relevant_documents):
    if not relevant_documents:
        return UNKNOWN_ANSWER
    context = "\n\n".join(
        f"Source: {format_source(document.metadata)}\n{document.page_content}"
        for document in relevant_documents
    )
    prompt = f"""
You are a document question-answering assistant. Answer only from the supplied context.
Do not use general model knowledge, invent facts, invent citations, or infer unsupported details.
If the context does not contain enough information, reply exactly:
{UNKNOWN_ANSWER}
Do not include a source list; the application displays retrieved sources separately.

Context:
{context}

Question: {question}

Answer:
"""
    response = get_chat_model().invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def summarize_documents(documents):
    text = "\n\n".join(document.page_content for document in documents)
    if not text.strip():
        return "The selected document has no extractable text."
    if len(text) > 10000:
        section_summaries = []
        for index in range(0, len(text), 2500):
            response = get_chat_model().invoke(
                "Summarize only the key facts in this document section. Do not add information:\n\n"
                + text[index:index + 2500]
            )
            section_summaries.append(response.content if hasattr(response, "content") else str(response))
        text = "\n\n".join(section_summaries[:5])
    prompt = f"""
Create a concise, factual summary using only the document content below.
Do not invent details or use outside knowledge. Mention important entities, dates, decisions,
requirements, and risks when they appear.

Document content:
{text[:16000]}

Summary:
"""
    response = get_chat_model().invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def reset_session():
    st.session_state.vector_store = None
    st.session_state.documents = []
    st.session_state.chunks = []
    st.session_state.processed_files = []
    st.session_state.processing_errors = []
    st.session_state.chat_history = []
    st.session_state.last_question = ""


def initialize_session():
    defaults = {
        "vector_store": None,
        "documents": [],
        "chunks": [],
        "processed_files": [],
        "processing_errors": [],
        "chat_history": [],
        "last_question": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_uploaded_files(uploaded_files):
    if not uploaded_files:
        reset_session()
        st.session_state.processing_errors = ["Select at least one PDF or TXT file first."]
        return
    documents, errors = extract_text_documents(uploaded_files)
    if not documents:
        reset_session()
        st.session_state.processing_errors = errors or ["No usable document text was found."]
        return
    if not get_api_key():
        reset_session()
        st.session_state.processing_errors = errors + [
            "GOOGLE_API_KEY is missing. Add it to the .env file before processing documents."
        ]
        return
    chunks = split_documents(documents)
    try:
        vector_store = create_vector_store(chunks)
    except Exception as exc:
        reset_session()
        st.session_state.processing_errors = errors + [f"Could not create embeddings: {exc}"]
        return
    st.session_state.vector_store = vector_store
    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.processed_files = sorted({document.metadata["source"] for document in documents})
    st.session_state.processing_errors = errors
    st.session_state.chat_history = []


def render_sources(sources):
    if sources:
        st.markdown("**Sources**")
        for source in sources:
            st.markdown(f"- {source}")
    else:
        st.caption("No source metadata was available for this answer.")


def main():
    st.set_page_config(page_title="Smart Document Assistant", page_icon=":page_facing_up:")
    initialize_session()
    st.title("Smart Document Assistant")
    st.caption("Upload PDF or TXT files, then ask grounded questions about their contents.")
    with st.sidebar:
        st.image(str(BASE_DIR / "img" / "Robot.jpg"))
        st.divider()
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
        )
        if st.button("Process documents", type="primary", use_container_width=True):
            with st.spinner("Extracting text and creating embeddings..."):
                process_uploaded_files(uploaded_files)
        if st.session_state.processed_files:
            st.success(f"Processed {len(st.session_state.processed_files)} document(s).")
            st.markdown("**Uploaded documents**")
            for filename in st.session_state.processed_files:
                st.caption(filename)
        for error in st.session_state.processing_errors:
            st.error(error)
        st.divider()
        if st.button("Reset documents and chat", use_container_width=True):
            reset_session()
            st.rerun()
        st.divider()
        st.caption("AI app by Chandra Shekar")
    if not get_api_key():
        st.warning("Add GOOGLE_API_KEY to .env to process documents and generate answers.")
    question = st.text_input(
        "Ask a question", placeholder="What does the document say about...?",
        disabled=st.session_state.vector_store is None,
    )
    if (
        question.strip()
        and question.strip() != st.session_state.last_question
        and st.session_state.vector_store is not None
    ):
        with st.spinner("Searching the documents..."):
            try:
                relevant_documents = get_relevant_documents(st.session_state.vector_store, question.strip())
                answer = answer_question(question.strip(), relevant_documents)
                st.session_state.chat_history.append({
                    "question": question.strip(), "answer": answer,
                    "sources": get_unique_sources(relevant_documents),
                })
                st.session_state.last_question = question.strip()
            except Exception as exc:
                st.error(f"Could not answer the question: {exc}")
    for item in st.session_state.chat_history:
        st.markdown(f"**Question:** {item['question']}")
        st.markdown("**Answer**")
        st.write(item["answer"])
        render_sources(item["sources"])
        st.divider()
    if st.session_state.documents:
        st.subheader("Document summary")
        summary_options = ["All processed documents"] + st.session_state.processed_files
        selected_summary = st.selectbox("Choose a document to summarize", summary_options)
        if st.button("Generate summary"):
            summary_documents = st.session_state.documents
            if selected_summary != "All processed documents":
                summary_documents = [
                    document for document in st.session_state.documents
                    if document.metadata.get("source") == selected_summary
                ]
            with st.spinner("Generating a grounded summary..."):
                try:
                    st.write(summarize_documents(summary_documents))
                except Exception as exc:
                    st.error(f"Could not generate the summary: {exc}")
    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/ChandraShekar00001" target="_blank">Chandra Shekar</a> | Smart Document Assistant
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
