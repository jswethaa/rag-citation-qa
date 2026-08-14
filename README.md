# RAG Citation QA System

A Retrieval-Augmented Generation (RAG) system built to answer questions from a document corpus with precise citations back to the original source pages, featuring hybrid retrieval (vector + BM25 keyword search) and cross-encoder reranking.

## Project Architecture & Roadmap
1. **Ingestion & Chunking (Day 1)**: Extract text from PDF via `pdfplumber`, tokenize & chunk with overlap using `tiktoken`, save structured JSON.
2. **Embeddings & Vector Store**: Compute embeddings (`all-MiniLM-L6-v2`) and index into `ChromaDB` alongside `rank_bm25`.
3. **Hybrid Retrieval & Reranking**: Parallel vector + keyword search merged with Reciprocal Rank Fusion (RRF), rescored with cross-encoder (`ms-marco-MiniLM-L-6-v2`).
4. **Generation & Grounded Prompts**: Generate answers with source citations using Anthropic Claude API.
5. **Interactive UI**: FastAPI backend with Streamlit / React user interface.

## Setup Instructions

### 1. Environment Setup
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Add Source PDF
Place your IBM annual report PDF at:
`data/raw/ibm_annual_report.pdf`

### 3. Run Ingestion Pipeline
```bash
python src/ingest.py
```
This extracts text page-by-page, chunks it into ~500–800 token segments with ~100 token sentence overlap, and outputs to `data/processed/chunks.json`.
