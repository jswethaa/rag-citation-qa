"""
Embedding Generation & ChromaDB Vector Storage Pipeline

This module handles:
1. Loading processed chunks from `data/processed/chunks.json`.
2. Initializing `sentence-transformers` with model `all-MiniLM-L6-v2`.
   - This model maps text into a 384-dimensional dense vector space.
   - It captures semantic context (meanings, concepts, synonyms) rather than exact keyword matches.
3. Initializing a persistent ChromaDB vector store saved to `./chroma_db`.
4. Storing vectors, text contents, document IDs, and original page metadata into a Chroma collection.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Model configuration
# 'all-MiniLM-L6-v2' produces 384-dimensional embeddings. It is lightweight, fast,
# and highly accurate for sentence/paragraph-level semantic similarity tasks.
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ibm_annual_report_chunks"
PERSIST_DIRECTORY = "chroma_db"
CHUNKS_JSON_PATH = "data/processed/chunks.json"


def load_chunks(json_path: str = CHUNKS_JSON_PATH) -> List[Dict[str, Any]]:
    """Loads processed chunks JSON generated during Day 1 ingestion."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed chunks file not found at '{path.resolve()}'. "
            "Please run 'python src/ingest.py' first."
        )
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def build_and_store_embeddings(
    chunks_path: str = CHUNKS_JSON_PATH,
    persist_dir: str = PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
    model_name: str = MODEL_NAME
):
    """
    Encodes chunk texts into 384-dimensional dense vectors and persists them
    into a local ChromaDB collection alongside full text and metadata.
    """
    # Step 1: Load pre-processed chunks
    chunks = load_chunks(chunks_path)
    if not chunks:
        print("[ERROR] No chunks found to embed.")
        return

    # Step 2: Initialize SentenceTransformer model
    # SentenceTransformers converts raw strings into float32 arrays of shape (384,)
    print(f"Loading SentenceTransformer embedding model: '{model_name}'...")
    model = SentenceTransformer(model_name)

    # Extract text contents and construct metadata lists
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "source_doc": c["source_doc"],
            "page_number": c["page_number"],
            "token_count": c["token_count"]
        }
        for c in chunks
    ]

    # Step 3: Compute dense vector embeddings
    # show_progress_bar gives visual feedback while embedding 273 chunks
    print(f"Generating 384-dimensional vector embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    # Step 4: Initialize persistent ChromaDB client
    # ChromaDB stores vectors, text payloads, and metadata in an SQLite + HNSW index on disk.
    # Specifying path=persist_dir ensures embeddings persist across python sessions.
    print(f"Connecting to persistent ChromaDB at directory: '{persist_dir}'...")
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    # Create or retrieve collection configured with Cosine Similarity space
    # metadata={"hnsw:space": "cosine"} ensures HNSW index calculates cosine distance
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Clear existing collection items if re-running to avoid duplicate IDs
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Found existing collection with {existing_count} items. Refreshing collection...")
        chroma_client.delete_collection(name=collection_name)
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # Step 5: Upsert embeddings, text payload, and metadata into ChromaDB
    print(f"Upserting {len(embeddings)} vectors into collection '{collection_name}'...")
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    final_count = collection.count()
    print("\n" + "=" * 55)
    print("      CHROMADB EMBEDDING & STORAGE SUMMARY")
    print("=" * 55)
    print(f"Embedding Model Used  : {model_name}")
    print(f"Vector Dimension      : {len(embeddings[0])} dimensions")
    print(f"Total Chunks Stored   : {final_count}")
    print(f"ChromaDB Storage Dir  : {Path(persist_dir).resolve()}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    build_and_store_embeddings()
