# DOJ Epstein Disclosure Archive — AI Research Tool

A Streamlit application for searching and analyzing the DOJ's Epstein disclosure documents using semantic search, OCR, and an LLM-powered Q&A interface.

---

## Features

- **Semantic Search** — Embeds documents using `BAAI/bge-small-en-v1.5` and stores vectors in Pinecone for fast retrieval
- **LLM Q&A** — Uses Groq (Llama 3.1 8B) to answer questions grounded in the indexed documents
- **Reranking** — Combines vector similarity, keyword overlap, and entity matching to surface the most relevant passages
- **Entity Graph** — Extracts co-occurring named entities across documents to map relationships
- **Timeline View** — Parses years from documents to build a chronological record
- **Reference Heatmap** — Shows which documents are cited most often in a given query
- **Follow-up Detection** — Enriches short or ambiguous queries using conversation history
- **Export** — Download any answer + its sources as a `.md` file

---

## Project Structure

```
├── app.py                  # Streamlit UI
├── query.py                # Search, reranking, Groq LLM, answer generation
├── enhanced_query.py       # Filtered search wrapper
├── reranker.py             # Keyword + entity + vector reranking
├── embedder.py             # Sentence-transformer embedding (BGE small)
├── vector_store.py         # Pinecone upsert pipeline with checkpointing
├── chunker.py              # Text chunking with overlap
├── pipeline.py             # End-to-end PDF extraction pipeline
├── extract_text.py         # Digital PDF text extraction (pdfplumber)
├── ocr.py                  # Scanned PDF OCR (pdf2image + pytesseract)
├── image_handler.py        # Image captioning (BLIP) from PDFs
├── pdf_classifier.py       # Classifies PDFs as digital / scanned / mixed
├── entity_extractor.py     # Named entity co-occurrence graph
├── timeline.py             # Year-based chronological sorting
├── epstein_download.py     # Selenium scraper for DOJ disclosure pages
├── project_paths.py        # Centralised path constants
└── requirements.txt
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `pytesseract` requires Tesseract to be installed separately.
> - macOS: `brew install tesseract`
> - Ubuntu: `sudo apt install tesseract-ocr`
> - Windows: download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### 3. Set environment variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX=epstein-index
GROQ_API_KEY=your-groq-api-key
```

---

## Running the Pipeline (First Time)

Run these steps once to download, process, and index the documents.

```bash
# 1. Download PDFs from DOJ disclosure pages
python epstein_download.py

# 2. Classify each PDF as digital, scanned, or mixed
python pdf_classifier.py

# 3. Extract text (OCR for scanned, pdfplumber for digital)
python pipeline.py

# 4. Chunk the extracted text
python chunker.py

# 5. Embed and upload to Pinecone
python vector_store.py
```

---

## Running the App

```bash
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. In **Settings → Secrets**, add your API keys in TOML format:

```toml
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX = "epstein-index"
GROQ_API_KEY = "your-groq-api-key"
```

4. Deploy — Streamlit Cloud will install `requirements.txt` automatically

---

## API Keys Required

| Service  | Purpose                        | Get it at                        |
|----------|--------------------------------|----------------------------------|
| Pinecone | Vector database                | [pinecone.io](https://pinecone.io) |
| Groq     | LLM inference (Llama 3.1 8B)   | [console.groq.com](https://console.groq.com) |
