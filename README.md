# Smart Document Assistant

Smart Document Assistant is a Streamlit RAG application for asking grounded questions about multiple PDF and TXT documents. It extracts document text, preserves source metadata, retrieves relevant chunks with FAISS, and uses Gemini to answer with readable citations.

## Problem Statement

Users often need answers from a small collection of documents without manually searching every page. This application provides a simple upload, retrieval, question-answering, and summary workflow while making the supporting source locations visible.

## Features

- Multiple PDF and TXT uploads in one session.
- PDF page metadata and TXT filename/section metadata.
- Per-file success and error reporting, including empty or unsupported files.
- Metadata-aware chunking and FAISS similarity retrieval.
- Grounded Gemini answers with deduplicated source citations.
- Explicit insufficient-information response for low-confidence retrieval or unsupported questions.
- Question history and a reset button.
- Grounded summaries for all documents or one selected document.

## Supported Files

- `.pdf`: extracted with PyPDF2; page numbers are preserved when text is available.
- `.txt`: decoded as UTF-8 with replacement for invalid bytes; the filename is used as the source and `Text document` as the section.

## Architecture and Data Flow

1. Streamlit accepts multiple PDF/TXT uploads.
2. `extract_text_documents` creates LangChain `Document` objects with source metadata.
3. `split_documents` uses `RecursiveCharacterTextSplitter` with 1,200-character chunks and 200-character overlap.
4. Gemini `models/gemini-embedding-001` creates embeddings for each chunk. The installed wrapper uses its query task default when embedding questions.
5. FAISS stores vectors and the original documents, including metadata, in Streamlit session state.
6. A question retrieves up to four chunks and filters low relevance scores where the vector store supports scores.
7. Gemini 2.5 Flash receives only the retrieved context and is instructed not to invent facts or citations.
8. The UI displays the answer separately from deduplicated retrieved sources.
9. The same Gemini integration generates summaries; long text is summarized in sections before a final synthesis.

See the Mermaid diagram in [docs/architecture.md](docs/architecture.md).

## Technologies

- **Python and Streamlit:** lightweight interactive web application.
- **PyPDF2:** PDF text extraction with page boundaries.
- **LangChain:** document objects and recursive text splitting.
- **Gemini via `langchain-google-genai`:** embeddings and grounded generation.
- **FAISS via `langchain-community`:** local in-memory vector similarity search.
- **python-dotenv:** loads the API key from `.env` without hardcoding secrets.

## Installation and Configuration

Use Python 3.10 or newer. From the project directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the same directory as `chatapp.py`:

```text
GOOGLE_API_KEY=your-google-api-key
```

Never commit `.env` or expose the key in the UI or logs. The application checks for a missing key and displays an actionable message.

## Run

```powershell
.\venv\Scripts\python.exe -m streamlit run chatapp.py
```

Then open the local URL shown by Streamlit, normally `http://localhost:8501`.

## RAG Details

### Chunking and embeddings

Each PDF page is first retained as a separate source document. TXT content starts as one source document. The recursive splitter creates chunks of approximately 1,200 characters with 200 characters of overlap. This keeps chunks focused while preserving nearby context. Every resulting chunk inherits its source filename and page or section metadata.

### FAISS retrieval

FAISS stores the Gemini embedding vectors and the LangChain documents together. Each question retrieves at most four similar chunks. When relevance scores are available, chunks below `0.25` are removed. This threshold is a heuristic: it reduces obviously unrelated context but cannot guarantee that every retained chunk answers the question.

### Source citations

The answer prompt receives source labels such as `company_policy.pdf - Page 3`. The UI separately lists the metadata of the chunks actually retrieved, removes duplicate labels, and never creates a page number that was not present in the parser metadata.

### Unknown questions and hallucination mitigation

The system uses three safeguards:

1. It filters low-relevance retrieval results when scores are available.
2. It instructs Gemini to answer only from supplied context and return a fixed insufficient-information message when context is inadequate.
3. It renders citations from retrieved metadata rather than asking the model to invent a source list.

These measures reduce unsupported answers but do not make the system hallucination-free. OCR-poor PDFs, ambiguous questions, weak embeddings, and model errors remain possible.

## Document Summary

The summary panel can summarize all processed documents or one selected document. Long inputs are summarized in 2,500-character sections, up to five section summaries, and then synthesized into a final grounded summary. This limits prompt size but may omit details from very long documents and uses additional Gemini calls.

## Testing and Validation

Static/local checks:

```powershell
.\venv\Scripts\python.exe -m py_compile chatapp.py
.\venv\Scripts\python.exe -c "import chatapp; print('import ok')"
```

Manual checks should cover:

1. One PDF with a known answer and page citation.
2. One TXT file with a known answer and filename citation.
3. Multiple PDF/TXT files.
4. Empty TXT, image-only PDF, malformed PDF, and unsupported extension.
5. An unrelated or vague question and the insufficient-information response.
6. Summary generation for one file and all files.
7. Missing API key and API/network failures.

Embedding, retrieval, Gemini answers, and summaries require a valid API key and external service access. They are not claimed as passed by static checks alone.

## Security Considerations

- API keys are loaded from environment configuration and are not hardcoded.
- Uploaded files are processed in memory; the app does not expose them through a download route.
- Source filenames are normalized to their basename before displaying metadata.
- Do not upload confidential documents unless the organization permits sending their content to the configured model provider.
- Avoid logging prompts, document contents, API keys, or generated secrets.

## AI Tools Used During Development

Development assistance used GitHub Copilot, Claude and ChatGPT for code inspection, incremental implementation, documentation, and validation guidance. Generated code was reviewed against the local dependency versions
## Known Limitations

- Scanned/image-only PDFs need OCR, which is not included.
- Gemini API quotas, model availability, and network failures can interrupt processing.
- The relevance threshold and top-four retrieval limit may need tuning for a different corpus.
- The summary workflow limits long-document map steps to five sections to control API usage.

