"""
Vector Retrieval Pipeline using Cosine Similarity Search

This module handles:
1. Embedding user query strings using the exact same embedding model (`all-MiniLM-L6-v2`).
2. Querying the persistent ChromaDB collection (`chroma_db/`) for nearest vector neighbors.
3. Calculating Cosine Similarity scores (Similarity = 1.0 - Cosine Distance).
4. Returning top-k most relevant chunks along with source metadata and page attribution.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Any

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ibm_annual_report_chunks"
PERSIST_DIRECTORY = "chroma_db"


def retrieve_similar_chunks(
    query: str,
    top_k: int = 5,
    persist_dir: str = PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
    model_name: str = MODEL_NAME
) -> List[Dict[str, Any]]:
    """
    Embeds a user query string and retrieves top-k most semantically similar chunks
    from the persistent ChromaDB vector collection.
    """
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"ChromaDB persistent directory not found at '{persist_path.resolve()}'. "
            "Please run 'python src/embed_store.py' first to build the vector store."
        )

    # Step 1: Connect to persistent ChromaDB
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(
            f"Collection '{collection_name}' not found in ChromaDB. "
            "Run 'python src/embed_store.py' to generate embeddings."
        ) from e

    # Step 2: Embed the query string using the EXACT SAME model used during chunk indexing
    # Crucial Rule: Query and document vectors must originate from the identical model space!
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()

    # Step 3: Search ChromaDB for nearest top-k vector neighbors
    # Chroma performs efficient HNSW vector indexing to find vectors with smallest cosine distance
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Step 4: Format retrieved chunks, metadatas, and convert distance to similarity score
    retrieved_chunks = []
    
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, doc_text, meta, distance in zip(ids, documents, metadatas, distances):
        # ChromaDB cosine space returns Cosine Distance (0 = identical, 1 = orthogonal, 2 = opposite)
        # Cosine Similarity = 1.0 - Cosine Distance
        similarity_score = max(0.0, 1.0 - distance)

        retrieved_chunks.append({
            "chunk_id": chunk_id,
            "source_doc": meta.get("source_doc", "unknown"),
            "page_number": meta.get("page_number", 0),
            "token_count": meta.get("token_count", 0),
            "text": doc_text,
            "distance": round(distance, 4),
            "similarity_score": round(similarity_score, 4)
        })

    return retrieved_chunks


def print_retrieval_results(query: str, results: List[Dict[str, Any]]):
    """Prints clean, formatted top-k retrieval results for evaluation."""
    print("\n" + "=" * 70)
    print(f"QUERY: \"{query}\"")
    print(f"RETRIEVED TOP-{len(results)} MOST SIMILAR CHUNKS:")
    print("=" * 70)

    for idx, r in enumerate(results, start=1):
        print(f"\n--- [Rank {idx}] {r['chunk_id']} | Page {r['page_number']} | Similarity Score: {r['similarity_score']:.4f} (Distance: {r['distance']:.4f}) ---")
        print(f"Source Doc  : {r['source_doc']}")
        print(f"Token Count : {r['token_count']} tokens")
        print(f"Text Snippet:\n{r['text'][:350]}...")
        print("-" * 70)
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query ChromaDB vector store for top-k similar document chunks.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What was IBM's total revenue and financial performance in 2025?",
        help="Query string to search for in vector store"
    )
    parser.add_argument(
        "--top_k",
        "-k",
        type=int,
        default=5,
        help="Number of top similar chunks to retrieve"
    )
    args = parser.parse_args()

    results = retrieve_similar_chunks(query=args.query, top_k=args.top_k)
    print_retrieval_results(args.query, results)
