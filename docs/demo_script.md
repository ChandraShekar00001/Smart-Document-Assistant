# Smart Document Assistant — Video Demonstration Script (5–8 Minutes)

This script provides a structured, professional, and natural walkthrough for recording a 5–8 minute demonstration video of the **Smart Document Assistant**.

---

## Pre-Recording Checklist
- [ ] Terminal open at the project root
- [ ] Application running via: `streamlit run app.py`
- [ ] Browser window open at `http://localhost:8501`
- [ ] Groq API key configured in `.env` (`GROQ_API_KEY`)
- [ ] Sample documents available in `data/sample_documents/`

---

## Video Timeline Overview

| Section | Target Duration | Key Topics |
| :--- | :--- | :--- |
| **1. Introduction & Problem Statement** | 0:00 – 1:00 (1 min) | Project goal, why standard LLMs fail on private docs, RAG concept |
| **2. Document Upload & Ingestion** | 1:00 – 2:00 (1 min) | Uploading PDF/TXT/DOCX, text cleaning, chunking, ChromaDB indexing |
| **3. Grounded Question Answering & Sources** | 2:00 – 3:30 (1.5 min) | Single-doc Q&A, multi-doc synthesis, inspecting source citations |
| **4. Hallucination Safeguards & Absent Info** | 3:30 – 4:30 (1 min) | Asking unanswerable question, relevance thresholding, unknown response |
| **5. Automatic Document Summarization** | 4:30 – 5:30 (1 min) | Executive brief generation, long doc sectioning, caching |
| **6. System Architecture & Safeguards** | 5:30 – 6:45 (1.25 min)| Offline indexing vs online retrieval, ChromaDB persistent client, Groq prompt |
| **7. Limitations & 2-Day Roadmap** | 6:45 – 7:45 (1 min) | Hybrid search, reranking, OCR, evaluation benchmarks |

---

## Detailed Speaking Script

### 1. Introduction & Problem Statement (0:00 – 1:00)
> *"Hello! Today I'm excited to present the **Smart Document Assistant**, a complete, production-grade Generative AI application built with Python, Streamlit, ChromaDB, and Groq.*
>
> *Large Language Models are remarkably capable, but when applied to private enterprise knowledge bases, they suffer from two major vulnerabilities: **knowledge cutoff** and **hallucination**. Asking a generic model about an internal company policy or confidential contract either produces guesswork or completely fabricated answers.*
>
> *To solve this, I built a Retrieval-Augmented Generation (RAG) system that grounds every response strictly in verified document chunks, displays full source attribution with exact excerpts, and explicitly refuses to guess when sufficient evidence is missing."*

---

### 2. Document Upload & Ingestion (1:00 – 2:00)
*(Screen Action: Point to the sidebar, enter API Key, then click **'📁 Load Samples'** or drag & drop documents from `data/sample_documents/`)*

> *"Let's take a look at the interface. In the sidebar, we have our Document Manager. The app accepts PDFs, plain text files, and Word DOCX documents.*
>
> *I'll click **'Load Samples'**, which immediately processes our three synthetic enterprise documents: `company_policy.txt`, `leave_policy.txt`, and `employee_handbook.pdf`.*
>
> *Behind the scenes, our pipeline extracted the raw text, cleaned out null bytes and formatting artifacts, split the text into 600-character chunks with a 100-character overlap using recursive separator splitting, generated 384-dimensional dense embeddings using `all-MiniLM-L6-v2`, and indexed them in a local ChromaDB collection.*
>
> *Notice our stats in the sidebar: we now have 3 documents and their respective indexed chunks ready for instant querying."*

---

### 3. Grounded Question Answering & Sources (2:00 – 3:30)
*(Screen Action: Navigate to Tab 1 **'💬 Ask Questions (RAG)'**. Click the sample prompt: 'What is the maximum reimbursement for home office equipment?')*

> *"Now let's ask a factual question: **'What is the maximum reimbursement for home office equipment?'***
>
> *(Wait 1-2 seconds for Groq response)*
>
> *The model answers accurately: **'$1,200 (USD) for desk, ergonomic chair, monitor, and accessories.'***
>
> *Crucially, notice the **Source Citations** section below the answer. We don't just get an answer; we get verifiable proof. When I expand Source [1], it shows:
> - Filename: `company_policy.txt`
> - Page label: `N/A (TXT Document)`
> - Relevance score: `0.72 (Cosine Similarity)`
> - The exact verbatim text excerpt from the policy.*
>
> *Next, let's test a cross-document question:
> **'How many days of paid vacation do employees receive, and what are the core working hours for remote workers?'***
>
> *Notice that this question spans two separate documents: `leave_policy.txt` and `company_policy.txt`. The RAG pipeline retrieves the top-ranked chunks from both documents, feeds them into the bounded context window, and Groq synthesizes both facts into a single coherent answer, citing both sources."*

---

### 4. Hallucination Safeguards & Absent Information (3:30 – 4:30)
*(Screen Action: Enter the question: 'What is the company policy regarding pet bereavement leave?')*

> *"Now, let's put our anti-hallucination safeguards to the test. Let's ask:
> **'What is the company policy regarding pet bereavement leave?'***
>
> *In our synthetic leave policy, bereavement leave exists for immediate family, but **pet bereavement** is deliberately unmentioned.*
>
> *Notice the result: Instead of guessing or assuming a standard 1-2 days, the application returns our deterministic safeguard message:
> **'I couldn't find sufficient information to answer this question in the uploaded documents.'**
>
> *How does this work?
> 1. First, our **Relevance Threshold Safeguard** filters out chunks with cosine similarity below 0.35. If no chunk meets the threshold, no context is passed.
> 2. Second, our system prompt strictly instructs Groq that if the facts aren't explicitly inside the `<document_context>` XML fence, it must never invent answers.
> 3. Third, all document excerpts are treated as untrusted data, safeguarding against prompt injection attacks."*

---

### 5. Automatic Document Summarization (4:30 – 5:30)
*(Screen Action: Click on Tab 2 **'📑 Automatic Document Summary'**. Select `employee_handbook.pdf` from the dropdown and click **'✨ Generate Summary'**)*

> *"Now let's inspect our creative feature: **Automatic Document Summary**.*
>
> *This feature operates independently of the Q&A retrieval flow while reusing our document extraction and LLM service.
> I'll select `employee_handbook.pdf`—which is a multi-page PDF—and click **Generate Summary**.*
>
> *(Wait a moment for summary rendering)*
>
> *Look at how structured and actionable this summary is:
> - **Executive Summary / Purpose**
> - **Key Policies**: Probationary period of 60 days, equal opportunity standards.
> - **Important Figures & Deadlines**: 14-character passwords, 90-day rotation, 24/7 whistleblower hotline.
> - **Information Security Requirements**: MFA mandatory, USB devices blocked, no public GenAI without zero-retention agreements.
>
> *For long documents that exceed our 4,000-character section limit, the summarizer automatically runs a map-reduce sectioning flow: summarizing individual sections first and then synthesizing them into a cohesive brief.*
>
> *If I click Generate Summary again, notice the notification: **'⚡ Loaded from cache'**. We compute a SHA-256 hash of the document text to avoid redundant API calls and save token costs."*

---

### 6. System Architecture & Technical Decisions (5:30 – 6:45)
*(Screen Action: Click on Tab 3 **'🔍 System Architecture & Inspector'**)*

> *"Let's examine the architecture. We can see our Mermaid workflow and chunk inspector right here in the app.*
>
> *Why these technology choices?
> 1. **PyMuPDF**: It is significantly faster than standard PDF extractors and allows accurate page-by-page mapping so our citations point to exact pages.
> 2. **Recursive Character Splitter**: Rather than arbitrary token slicing, it respects natural language hierarchies—paragraphs first, then sentences, then words.
> 3. **sentence-transformers (`all-MiniLM-L6-v2`)**: It runs 100% locally on CPU, has a 384-dimensional footprint, and requires zero paid embedding API calls.
> 4. **ChromaDB `PersistentClient`**: Efficient HNSW indexing with cosine distance space and native persistent storage, eliminating manual file sync.
> 5. **Groq (`openai/gpt-oss-120b`)**: High speed, ultra-low latency, and exceptional instruction-following capability on LPU hardware."*

---

### 7. Limitations & 2-Day Roadmap (6:45 – 7:45)
> *"To conclude, let's discuss current limitations and what I would build with an additional two days:
> 1. **Hybrid Search (BM25 + Dense Vectors)**: Dense embeddings excel at semantic similarity, but keyword search (BM25) is superior for exact identifiers like product codes, SKUs, or error numbers. A hybrid ranker with Reciprocal Rank Fusion (RRF) would improve precision.
> 2. **Cross-Encoder Reranking**: Adding a secondary reranker model (like `bge-reranker-base`) to score top-20 retrieved candidates before feeding top-4 into Groq.
> 3. **OCR for Scanned PDFs**: Integrating Tesseract or Google Cloud Vision OCR to handle image-only scanned documents.
> 4. **RAG Triad Automated Evaluation**: Integrating TruLens or Ragas to automatically benchmark Context Relevance, Groundedness, and Answer Relevance across changes.
>
> *All 26 automated unit tests are passing, and the application is completely ready to run on any standard Windows machine. Thank you!"*
