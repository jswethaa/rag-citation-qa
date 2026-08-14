"""
Document Ingestion & Chunking Pipeline for RAG (Retrieval-Augmented Generation)

This module handles:
1. Loading source PDF documents using `pdfplumber`.
2. Extracting text page-by-page while tracking page metadata and handling errors.
3. Tokenizing text using `tiktoken` (`cl100k_base` encoding).
4. Chunking text into ~500-800 token segments with ~100 token sentence overlap.
5. Saving structured chunk artifacts to `data/processed/chunks.json`.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pdfplumber
import tiktoken

# Initialize the tiktoken encoding.
# 'cl100k_base' is the standard BPE tokenizer used by OpenAI models (GPT-3.5/GPT-4),
# which serves as an accurate general-purpose token counter for RAG pipelines.
ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Counts the exact number of BPE tokens in a string using tiktoken."""
    return len(ENCODER.encode(text))


def split_into_sentences(text: str) -> List[str]:
    """
    Splits raw text into individual sentences using regex boundary matching.
    
    Using regex sentence segmentation ensures we do not cut sentences in half
    during chunking, preserving grammatical and semantic structure.
    """
    # Clean up whitespace and newlines for cleaner text
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    if not cleaned_text:
        return []
    
    # Split on sentence-ending punctuation followed by whitespace or quote
    sentence_end_pattern = r'(?<=[.!?])\s+(?=[A-Z0-9"])'
    sentences = re.split(sentence_end_pattern, cleaned_text)
    return [s.strip() for s in sentences if s.strip()]


def extract_text_from_pdf(pdf_path: str) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Opens a PDF file and extracts text page-by-page.
    
    Returns:
        pages_data: List of dicts containing 'page_number' and 'text'
        failed_pages: List of page numbers that failed to extract
    """
    pages_data = []
    failed_pages = []

    print(f"Opening PDF document: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages found: {total_pages}")

        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages_data.append({
                        "page_number": i,
                        "text": text.strip()
                    })
                else:
                    # Page had no readable text (blank page, image-only, or formatting issue)
                    failed_pages.append(i)
            except Exception as e:
                print(f"Error extracting page {i}: {e}")
                failed_pages.append(i)

    return pages_data, failed_pages


"""
===============================================================================
OVERLAP MECHANISM EXPLANATION:
===============================================================================
When chunking documents for RAG, cutting text strictly at a fixed token limit 
(e.g., exactly at 700 tokens) risks splitting sentences mid-phrase or separating
a key statement from its preceding context.

To solve this:
1. We group text into full sentences.
2. We accumulate sentences into a Chunk N until its size reaches 500-800 tokens.
3. When Chunk N is completed, we don't start Chunk N+1 from the very next sentence.
   Instead, we step backwards by ~100 tokens worth of full sentences from the end 
   of Chunk N.
4. Chunk N+1 begins with these overlapping sentences, providing "sliding context"
   so that queries matching text near a chunk boundary still retrieve full context.
===============================================================================
"""

def create_chunks_with_overlap(
    pages_data: List[Dict[str, Any]],
    source_doc_name: str,
    target_min_tokens: int = 500,
    target_max_tokens: int = 800,
    overlap_tokens: int = 100
) -> List[Dict[str, Any]]:
    """
    Converts page-level text into sentence-bounded chunks with token overlap.
    """
    # Step 1: Flatten all sentences across all pages while recording page attribution
    all_sentences = []
    for page in pages_data:
        page_num = page["page_number"]
        sentences = split_into_sentences(page["text"])
        for s in sentences:
            tokens = count_tokens(s)
            all_sentences.append({
                "text": s,
                "page_number": page_num,
                "token_count": tokens
            })

    chunks = []
    chunk_index = 1
    total_sentences = len(all_sentences)
    current_start = 0

    while current_start < total_sentences:
        current_chunk_sentences = []
        current_token_count = 0
        end_idx = current_start
        start_page = all_sentences[current_start]["page_number"]

        # Step 2: Accumulate sentences up to target_max_tokens
        while end_idx < total_sentences:
            sentence_info = all_sentences[end_idx]
            sentence_tokens = sentence_info["token_count"]

            # If adding this sentence exceeds max tokens and we've already hit target_min_tokens, stop
            if (current_token_count + sentence_tokens > target_max_tokens) and (current_token_count >= target_min_tokens):
                break

            current_chunk_sentences.append(sentence_info)
            current_token_count += sentence_tokens
            end_idx += 1

            # If single sentence is very long, force move forward
            if current_token_count >= target_max_tokens:
                break

        if not current_chunk_sentences:
            break

        # Assemble text for the chunk
        chunk_text = " ".join(s["text"] for s in current_chunk_sentences)
        actual_token_count = count_tokens(chunk_text)

        chunks.append({
            "chunk_id": f"chunk_{chunk_index:04d}",
            "source_doc": source_doc_name,
            "page_number": start_page,
            "text": chunk_text,
            "token_count": actual_token_count
        })

        chunk_index += 1

        # If we reached the end of all sentences, exit loop
        if end_idx >= total_sentences:
            break

        # Step 3: Calculate overlap for the next chunk's starting index
        # Work backwards from end_idx accumulating ~overlap_tokens
        accumulated_overlap = 0
        next_start = end_idx

        for rev_idx in range(end_idx - 1, current_start, -1):
            accumulated_overlap += all_sentences[rev_idx]["token_count"]
            if accumulated_overlap >= overlap_tokens:
                next_start = rev_idx
                break

        # Ensure forward progress to prevent infinite loop
        if next_start <= current_start:
            next_start = current_start + 1

        current_start = next_start

    return chunks


def process_ingestion(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    target_filename: str = "ibm_annual_report.pdf"
):
    """Executes the full ingestion and chunking pipeline."""
    pdf_path = Path(raw_dir) / target_filename
    output_path = Path(processed_dir) / "chunks.json"

    if not pdf_path.exists():
        print(f"\n[ERROR] Source PDF not found at: {pdf_path.resolve()}")
        print(f"Please place your PDF at '{pdf_path}' and run this script again.")
        return

    # Create processed directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extract text from PDF
    pages_data, failed_pages = extract_text_from_pdf(str(pdf_path))

    if not pages_data:
        print("[ERROR] No text could be extracted from the PDF.")
        return

    # 2. Generate chunks with overlap
    print("Chunking extracted text into ~500-800 token segments with sentence overlap...")
    chunks = create_chunks_with_overlap(
        pages_data=pages_data,
        source_doc_name=target_filename,
        target_min_tokens=500,
        target_max_tokens=800,
        overlap_tokens=100
    )

    # 3. Save chunks to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # 4. Print Summary Metrics
    total_chunks = len(chunks)
    avg_tokens = sum(c["token_count"] for c in chunks) / total_chunks if total_chunks > 0 else 0
    min_tokens = min(c["token_count"] for c in chunks) if total_chunks > 0 else 0
    max_tokens = max(c["token_count"] for c in chunks) if total_chunks > 0 else 0

    print("\n" + "=" * 55)
    print("      DOCUMENT INGESTION & CHUNKING SUMMARY")
    print("=" * 55)
    print(f"Source Document       : {target_filename}")
    print(f"Total Pages Processed : {len(pages_data) + len(failed_pages)}")
    print(f"Failed Pages          : {failed_pages if failed_pages else 'None'}")
    print(f"Total Chunks Created  : {total_chunks}")
    print(f"Average Token Count   : {avg_tokens:.1f} tokens/chunk")
    print(f"Token Count Range     : Min {min_tokens} | Max {max_tokens}")
    print(f"Output File Saved     : {output_path.resolve()}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    process_ingestion()
