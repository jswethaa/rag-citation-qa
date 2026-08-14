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


def split_into_units(text: str, max_unit_tokens: int = 150) -> List[str]:
    """
    Splits text into small semantic units (sentences or table lines).
    
    If text contains financial tables or unpunctuated lines, splitting strictly 
    on [.!?] punctuation will lump an entire page-long table into a single unit.
    This function splits by lines and sentences, and enforces a hard token ceiling
    on any individual unit.
    """
    if not text or not text.strip():
        return []

    lines = text.split('\n')
    units = []

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue

        # Split line on standard sentence-ending punctuation if present
        sentence_end_pattern = r'(?<=[.!?])\s+(?=[A-Z0-9"])'
        sentences = re.split(sentence_end_pattern, cleaned_line)

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            s_tokens = count_tokens(s_clean)
            # If a unit (e.g. unpunctuated financial table block) exceeds max_unit_tokens,
            # break it down into sub-units by word groups.
            if s_tokens > max_unit_tokens:
                words = s_clean.split(' ')
                sub_unit = []
                sub_count = 0
                for w in words:
                    wt = count_tokens(w + ' ')
                    if sub_count + wt > max_unit_tokens and sub_unit:
                        units.append(' '.join(sub_unit))
                        sub_unit = [w]
                        sub_count = wt
                    else:
                        sub_unit.append(w)
                        sub_count += wt
                if sub_unit:
                    units.append(' '.join(sub_unit))
            else:
                units.append(s_clean)

    return units


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


def create_chunks_with_overlap(
    pages_data: List[Dict[str, Any]],
    source_doc_name: str,
    target_min_tokens: int = 500,
    target_max_tokens: int = 800,
    overlap_tokens: int = 100
) -> List[Dict[str, Any]]:
    """
    Converts page-level text into bounded chunks with token overlap.
    Handles standard prose as well as dense unpunctuated financial tables cleanly.
    """
    # Step 1: Flatten all units across all pages while recording page attribution
    all_units = []
    for page in pages_data:
        page_num = page["page_number"]
        units = split_into_units(page["text"], max_unit_tokens=150)
        for u in units:
            all_units.append({
                "text": u,
                "page_number": page_num,
                "token_count": count_tokens(u)
            })

    chunks = []
    chunk_index = 1
    total_units = len(all_units)
    current_start = 0

    while current_start < total_units:
        current_chunk_units = []
        current_token_count = 0
        end_idx = current_start
        start_page = all_units[current_start]["page_number"]

        # Step 2: Accumulate units up to target_max_tokens
        while end_idx < total_units:
            unit_info = all_units[end_idx]
            unit_tokens = unit_info["token_count"]

            if (current_token_count + unit_tokens > target_max_tokens) and (current_token_count >= target_min_tokens):
                break

            current_chunk_units.append(unit_info)
            current_token_count += unit_tokens
            end_idx += 1

            if current_token_count >= target_max_tokens:
                break

        if not current_chunk_units:
            current_chunk_units = [all_units[current_start]]
            end_idx = current_start + 1

        chunk_text = " ".join(u["text"] for u in current_chunk_units)
        actual_token_count = count_tokens(chunk_text)

        chunks.append({
            "chunk_id": f"chunk_{chunk_index:04d}",
            "source_doc": source_doc_name,
            "page_number": start_page,
            "text": chunk_text,
            "token_count": actual_token_count
        })

        chunk_index += 1

        if end_idx >= total_units:
            break

        # Step 3: Calculate overlap for next chunk
        accumulated_overlap = 0
        next_start = end_idx

        for rev_idx in range(end_idx - 1, current_start, -1):
            accumulated_overlap += all_units[rev_idx]["token_count"]
            if accumulated_overlap >= overlap_tokens:
                next_start = rev_idx
                break

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
    raw_path = Path(raw_dir)
    pdf_path = raw_path / target_filename

    # If default target_filename does not exist, search for any PDF file in raw_dir
    if not pdf_path.exists():
        pdf_files = list(raw_path.glob("*.pdf"))
        if pdf_files:
            pdf_path = pdf_files[0]
            target_filename = pdf_path.name
            print(f"[INFO] Using detected PDF file: '{target_filename}'")
        else:
            print(f"\n[ERROR] Source PDF not found in: {raw_path.resolve()}")
            print("Please place your PDF in 'data/raw/' and run this script again.")
            return

    output_path = Path(processed_dir) / "chunks.json"

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
