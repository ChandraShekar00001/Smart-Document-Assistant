# Smart Document Assistant Architecture

```mermaid
flowchart TD
    U[User] --> UI[Streamlit UI]
    UI --> Upload[Upload PDF or TXT]
    Upload --> Parser[Document parser\nPyPDF2 or UTF-8 TXT]
    Parser --> Chunks[Recursive chunking\n1200 characters / 200 overlap]
    Chunks --> Metadata[Source metadata\nfilename + PDF page or TXT section]
    Metadata --> Embed[Gemini embeddings]
    Embed --> FAISS[FAISS vector store\nkept in Streamlit session]
    FAISS --> Retrieve[Similarity retrieval\nup to 4 chunks + threshold]
    Retrieve --> Grounded[Grounded Gemini answer]
    Grounded --> Answer[Answer + deduplicated citations]
    Retrieve --> Unknown[No relevant chunks]
    Unknown --> Fallback[Insufficient-information response]
    Metadata --> Summary[Gemini document summary\nmap/reduce for long text]
    Summary --> SummaryOutput[Grounded summary]
```

The relevance threshold is a practical safeguard, not a guarantee of factual correctness. The model is also instructed to use only retrieved context and the UI displays only metadata attached to retrieved chunks.
