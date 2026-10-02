"""Script to generate high-resolution architecture diagram docs/architecture.png."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure docs directory exists
docs_dir = Path("docs")
docs_dir.mkdir(parents=True, exist_ok=True)

# Create figure
fig, ax = plt.subplots(figsize=(16, 11), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis("off")

# Title
ax.text(
    8,
    10.5,
    "Smart Document Assistant — System Architecture & Data Flow",
    fontsize=18,
    fontweight="bold",
    ha="center",
    va="center",
    color="#1E3A8A",
)
ax.text(
    8,
    10.15,
    "Retrieval-Augmented Generation (RAG) & Automatic Document Summarization Pipelines",
    fontsize=11,
    ha="center",
    va="center",
    color="#4B5563",
)


def draw_box(x, y, w, h, title, subtitle, color, border="#1E40AF"):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor=border,
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        va="center",
        color="#1F2937",
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            fontsize=7.5,
            ha="center",
            va="center",
            color="#4B5563",
        )


def draw_arrow(x1, y1, x2, y2, label=""):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            color="#2563EB",
            lw=1.6,
            mutation_scale=12,
        ),
    )
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.15,
            label,
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="bottom",
            color="#1D4ED8",
        )


# Background Pipeline Containers
ingest_bg = patches.FancyBboxPatch(
    (0.5, 4.4),
    4.6,
    5.3,
    boxstyle="round,pad=0.15",
    facecolor="#F0FDF4",
    edgecolor="#16A34A",
    linestyle="--",
    linewidth=1.2,
)
ax.add_patch(ingest_bg)
ax.text(
    2.8,
    9.45,
    "1. Offline Ingestion & Indexing",
    fontsize=11,
    fontweight="bold",
    ha="center",
    color="#15803D",
)

rag_bg = patches.FancyBboxPatch(
    (5.7, 1.8),
    4.6,
    7.9,
    boxstyle="round,pad=0.15",
    facecolor="#EFF6FF",
    edgecolor="#2563EB",
    linestyle="--",
    linewidth=1.2,
)
ax.add_patch(rag_bg)
ax.text(
    8.0,
    9.45,
    "2. Online RAG Query Workflow",
    fontsize=11,
    fontweight="bold",
    ha="center",
    color="#1D4ED8",
)

summ_bg = patches.FancyBboxPatch(
    (10.9, 4.0),
    4.6,
    5.7,
    boxstyle="round,pad=0.15",
    facecolor="#FAF5FF",
    edgecolor="#9333EA",
    linestyle="--",
    linewidth=1.2,
)
ax.add_patch(summ_bg)
ax.text(
    13.2,
    9.45,
    "3. Automatic Summarization",
    fontsize=11,
    fontweight="bold",
    ha="center",
    color="#7E22CE",
)

# 1. Ingestion Pipeline Nodes
draw_box(1.0, 8.5, 3.6, 0.7, "Uploaded Documents", "PDF, TXT, DOCX Files", "#DCFCE7", "#16A34A")
draw_box(1.0, 7.3, 3.6, 0.7, "Document Loader & Cleaner", "PyMuPDF, python-docx, text cleaning", "#DCFCE7", "#16A34A")
draw_box(1.0, 6.1, 3.6, 0.7, "Recursive Text Splitter", "Chunk size: 600 | Overlap: 100", "#DCFCE7", "#16A34A")
draw_box(1.0, 4.9, 3.6, 0.7, "Dense Embeddings", "sentence-transformers: all-MiniLM-L6-v2", "#DCFCE7", "#16A34A")

draw_arrow(2.8, 8.5, 2.8, 8.0)
draw_arrow(2.8, 7.3, 2.8, 6.8)
draw_arrow(2.8, 6.1, 2.8, 5.6)

# Shared Vector Store (Bottom Left)
draw_box(
    1.0,
    1.8,
    3.6,
    1.4,
    "Local Vector Store & Metadata",
    "ChromaDB PersistentClient (Cosine)\ndata/chroma_db (SQLite + HNSW)",
    "#FEF3C7",
    "#D97706",
)
draw_arrow(2.8, 4.9, 2.8, 3.2, "Persist")

# 2. Online RAG Pipeline Nodes
draw_box(6.2, 8.5, 3.6, 0.7, "User Natural Query", "Streamlit UI Input", "#DBEAFE", "#2563EB")
draw_box(6.2, 7.3, 3.6, 0.7, "Query Embedding", "all-MiniLM-L6-v2 (Normalized)", "#DBEAFE", "#2563EB")
draw_box(6.2, 6.1, 3.6, 0.7, "ChromaDB Similarity Search", "Retrieve Top-K Chunks", "#DBEAFE", "#2563EB")
draw_box(6.2, 4.9, 3.6, 0.7, "Relevance Threshold Filter", "Score >= 0.35 (Anti-Hallucination)", "#FEE2E2", "#DC2626")
draw_box(6.2, 3.7, 3.6, 0.7, "Context Builder & Budget", "Max 6,000 Chars | XML Encapsulation", "#DBEAFE", "#2563EB")
draw_box(6.2, 2.3, 3.6, 0.8, "Groq LLM Generation", "openai/gpt-oss-120b (Strict Grounding)", "#E0E7FF", "#4F46E5")

draw_arrow(8.0, 8.5, 8.0, 8.0)
draw_arrow(8.0, 7.3, 8.0, 6.8)
draw_arrow(8.0, 6.1, 8.0, 5.6)
draw_arrow(8.0, 4.9, 8.0, 4.4)
draw_arrow(8.0, 3.7, 8.0, 3.1)

# Vector store connection to Search
draw_arrow(4.6, 2.5, 6.2, 6.45, "Dense Vectors")

# Low-relevance Fallback
draw_box(
    6.2,
    0.4,
    3.6,
    0.8,
    "Unknown-Answer Fallback",
    "I couldn't find sufficient information...",
    "#FEE2E2",
    "#DC2626",
)
ax.annotate(
    "",
    xy=(8.0, 1.2),
    xytext=(9.8, 5.25),
    arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.4, linestyle=":"),
)
ax.text(9.1, 3.3, "Below\nThreshold", fontsize=7.5, color="#DC2626", ha="center")

# 3. Summarization Nodes
draw_box(11.4, 8.5, 3.6, 0.7, "Document Text Extraction", "Full document text extraction", "#F3E8FF", "#9333EA")
draw_box(11.4, 7.3, 3.6, 0.7, "In-Memory Summary Cache", "Hash check (avoid redundant calls)", "#F3E8FF", "#9333EA")
draw_box(11.4, 6.1, 3.6, 0.7, "Document Insights Router", "<=100k chars: 1 Call | >100k: Fallback", "#F3E8FF", "#9333EA")
draw_box(11.4, 4.7, 3.6, 0.8, "Single-Pass / Synthesis", "1 Call (Summary, Key Points, etc.)", "#E0E7FF", "#4F46E5")

draw_arrow(13.2, 8.5, 13.2, 8.0)
draw_arrow(13.2, 7.3, 13.2, 6.8)
draw_arrow(13.2, 6.1, 13.2, 5.5)

# LLM connection to Summarizer
ax.annotate(
    "",
    xy=(11.4, 5.1),
    xytext=(9.8, 2.7),
    arrowprops=dict(arrowstyle="<->", color="#4F46E5", lw=1.3, linestyle="--"),
)
ax.text(10.6, 3.8, "Shared LLM\nService", fontsize=7.5, color="#4F46E5", ha="center")

# Final Outputs
draw_box(11.4, 2.3, 3.6, 0.8, "Structured Summary", "Executive Brief, Headings & Dates", "#F3E8FF", "#9333EA")
draw_arrow(13.2, 4.7, 13.2, 3.1)

draw_box(11.4, 0.4, 3.6, 0.8, "Verified Output to UI", "Answers, Citations & Source Excerpts", "#DBEAFE", "#2563EB")
draw_arrow(9.8, 2.7, 11.4, 0.8)

plt.tight_layout()
output_file = Path("docs/architecture.png")
plt.savefig(str(output_file), dpi=200, bbox_inches="tight")
plt.close()
print(f"Generated architecture diagram successfully: {output_file.resolve()}")
