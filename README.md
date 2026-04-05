# Epstein Files Intelligence System

An end-to-end AI-powered document intelligence system built to process, index, and query the declassified Epstein court documents using RAG, vector search, OCR, and LLM-based Q&A.

---

## Overview

EFIS is a local AI pipeline that ingests hundreds of classified PDF documents, extracts and indexes their content, and exposes an interactive interface for natural language querying, entity exploration, and timeline analysis — all running fully offline.

---

## Features

- **Intelligent PDF Classification** — Automatically detects whether a PDF is digital, scanned, or mixed using text density analysis and OCR sampling
- **Multi-modal Text Extraction** — Uses `pdfplumber` for digital PDFs, `Tesseract OCR` for scanned pages, and page-level hybrid extraction for mixed documents
- **Image Understanding** — Extracts embedded images, routes text-heavy scans to OCR and visual content to BLIP image captioning, making images fully searchable
- **RAG Pipeline** — Chunks documents with overlap, embeds using `all-MiniLM-L6-v2` (Sentence Transformers), indexes with FAISS, and retrieves semantically relevant passages per query
- **LLM Q&A** — Answers questions using LLaMA3 via Ollama with context-grounded prompts; gracefully falls back to keyword search if Ollama is unavailable
- **Entity Relationship Graph** — Extracts named entities across all chunks and maps co-occurrences to reveal connections between people, places, and organizations
- **Timeline Engine** — Detects and sorts year-anchored events chronologically across all documents
- **Interactive Dashboard** — Streamlit UI with Chat, Entity Graph, and Timeline tabs; includes source attribution with file name and page number

---

## Architecture

```
PDF Files
    │
    ▼
PDF Classifier (digital / scanned / mixed)
    │
    ├── pdfplumber (digital)
    ├── Tesseract OCR (scanned)
    └── Hybrid page-level (mixed)
    │
    ▼
Image Handler (BLIP captioning + OCR routing)
    │
    ▼
Chunker (paragraph-aware, with overlap)
    │
    ▼
Embedder (Sentence Transformers → FAISS Index)
    │
    ▼
Query Engine (semantic search + LLaMA3 via Ollama)
    │
    ▼
Streamlit App (Chat | Entity Graph | Timeline)
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Text Extraction | pdfplumber, PyMuPDF (fitz), pdf2image |
| OCR | Tesseract, pytesseract |
| Image Captioning | BLIP (Salesforce), Transformers |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| LLM | LLaMA3 via Ollama |
| Frontend | Streamlit |
| Core | Python, NumPy, Pandas |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/VimalBarik/epstein-files-intelligence-system.git
cd epstein-files-intelligence-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Make sure Tesseract is installed on your system:
> - Ubuntu: `sudo apt install tesseract-ocr`
> - Mac: `brew install tesseract`
> - Windows: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)

### 3. Add your PDF files

Place all PDF files inside a folder named `epstein files/` in the project root.

### 4. Run the pipeline

```bash
# Step 1: Classify PDFs
python pdf_classifier.py

# Step 2: Extract text + images
python pipeline.py

# Step 3: Chunk the data
python chunker.py

# Step 4: Build vector store
python vector_store.py

# Step 5: Launch the app
streamlit run app.py
```

### 5. (Optional) Set up Ollama for LLM Q&A

```bash
ollama pull llama3
ollama serve
```

If Ollama is not running, the app falls back to keyword-based search automatically.

---

## Project Structure

```
├── app.py                  # Streamlit dashboard
├── pipeline.py             # Main extraction pipeline
├── pdf_classifier.py       # PDF type detection
├── extract_text.py         # Digital PDF text extraction
├── ocr.py                  # Scanned PDF OCR
├── image_handler.py        # Image extraction + BLIP captioning
├── chunker.py              # Text chunking with overlap
├── embedder.py             # Sentence Transformer embeddings
├── vector_store.py         # FAISS index builder
├── query.py                # Search + LLM Q&A engine
├── enhanced_query.py       # Filtered search wrapper
├── entity_extractor.py     # Named entity + graph builder
├── timeline.py             # Chronological event extractor
├── project_paths.py        # Centralized path config
└── epstein files/          # PDF documents (not included)
```

---

## Disclaimer

This project is built for **research and educational purposes only**. All documents processed are publicly available declassified court records. No private or illegally obtained data is used.

---

## Author

**Vimal Barik**
