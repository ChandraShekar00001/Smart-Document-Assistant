# Smart Document Assistant — Engineering Time Log

This log documents the time investment and engineering phases for the development of the **Smart Document Assistant** take-home project.

---

## Summary Overview

* **Total Estimated Hours**: 18.5 Hours
* **Methodology**: Iterative Test-Driven Development (TDD) with modular component isolation.
* **Target Environment**: Windows 11 / Python 3.13, local CPU embedding inference, Google Gemini LLM API.

---

## Detailed Breakdown by Phase

### Phase 1: Requirements Analysis & System Architecture (2.0 Hours)
- Analyzed technical objectives, constraints, and hallucination reduction requirements.
- Evaluated library ecosystem for Python on Windows (`PyMuPDF`, `python-docx`, `faiss-cpu`, `sentence-transformers`, `google-generativeai`).
- Architected the three decoupled pipelines:
  1. Offline Ingestion & Vector Indexing.
  2. Online Grounded RAG Query Pipeline.
  3. Automatic Document Summarization Pipeline.
- Created Mermaid data flow diagrams and initial component specifications.

### Phase 2: Document Ingestion & Text Extraction (2.5 Hours)
- Implemented `src/document_loader.py` supporting multi-page PDF (`PyMuPDF` with `pypdf` fallback), plain text (`.txt` with multi-encoding fallback), and Word documents (`.docx`).
- Built robust text cleaning pipeline to handle null bytes, control characters, and line normalizations.
- Added comprehensive document metadata attachment (`source`, `page_number`, `total_pages`, `file_type`).
- Added robust error handling: `UnsupportedFormatError`, `EmptyDocumentError`, `DocumentParsingError`.

### Phase 3: Recursive Chunking & Embedding Generation (2.5 Hours)
- Implemented `src/text_splitter.py` with hierarchical recursive splitting (`\n\n` -> `\n` -> sentences -> words -> characters).
- Added enriched chunk metadata (`chunk_id`, `chunk_index`, `char_count`, inherited page numbers).
- Built `src/embeddings.py` using `sentence-transformers` (`all-MiniLM-L6-v2`).
- Implemented unit L2 vector normalization to enable exact cosine similarity using Inner Product.
- Added singleton model caching to eliminate redundant weight loading across requests.

### Phase 4: FAISS Vector Store & Local Persistence (2.0 Hours)
- Implemented `src/vector_store.py` wrapping `faiss.IndexFlatIP`.
- Implemented binary index saving (`faiss.write_index`) and JSON metadata serialization.
- Built synchronization logic for document deletion, index rebuilding, and collection clearing.
- Handled empty stores, vector dimension validation, and boundary conditions.

### Phase 5: Semantic Retrieval & Grounded LLM Service (3.0 Hours)
- Implemented `src/retriever.py` with top-k retrieval and cosine similarity thresholding (`score >= 0.35`).
- Implemented `src/llm_service.py` using `google-generativeai`.
- Formulated anti-hallucination system prompt and untrusted data boundary fencing (`<document_context>`).
- Implemented deterministic unknown-answer triggering:
  *"I couldn't find sufficient information to answer this question in the uploaded documents."*
- Added error handling for missing API keys, quota exhaustion (HTTP 429), and network exceptions.
- Implemented `src/source_formatter.py` to format citations without hallucinated references.

### Phase 6: Automatic Document Summarizer (1.5 Hours)
- Implemented `src/summarizer.py` supporting short-document single-pass summaries and long-document hierarchical map-reduce summarization.
- Designed structured summary output schema (Executive Summary, Key Policies, Figures & Dates, Requirements).
- Added SHA-256 in-memory summary caching to eliminate duplicate API costs.

### Phase 7: Streamlit User Interface & Session Experience (2.5 Hours)
- Built multi-tab Streamlit interface in `app.py`.
- Developed sidebar document manager: multi-file upload, sample data loader, document table, deletion buttons, and knowledge base stats.
- Created Q&A tab with interactive prompt suggestions, markdown answer rendering, visual source cards, and session conversation history.
- Built Document Summarization tab with document selector and cache indicator.
- Created Architecture & Vector Chunk Inspector tab.

### Phase 8: Testing, Synthetic Data & Verification (2.5 Hours)
- Built synthetic sample documents: `company_policy.txt`, `leave_policy.txt`, `employee_handbook.pdf`, `employee_handbook.docx`.
- Developed comprehensive test suite (`tests/` with 26 unit tests) mocking Gemini API and verifying end-to-end flows.
- Debugged recursive text splitter edge cases and verified 100% test pass rate.
- Generated high-resolution architecture diagram (`docs/architecture.png`) and presentation script (`docs/demo_script.md`).

---

## Total Engineering Time: 18.5 Hours
