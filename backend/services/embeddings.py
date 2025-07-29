import numpy as np
from backend.services.data_loader import load_all_embedding_chunks

# Module-level cache for loaded embedding chunks
_cached_embedding_chunks = None

def get_embedding_chunks():
    """
    Lazily load and cache embedding chunks from the designated files.
    Returns the cached list on subsequent calls.
    """
    global _cached_embedding_chunks
    if _cached_embedding_chunks is None:
        _cached_embedding_chunks = load_all_embedding_chunks()
    return _cached_embedding_chunks

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two numpy arrays."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def deduplicate_by_embedding(chunks: list, threshold=0.9):
    """
    Deduplicate chunks by comparing their precomputed embeddings.

    Args:
        chunks (list): List of chunk dicts, each with 'embedding' under 'metadata' or at root.
        threshold (float): Cosine similarity threshold above which chunks are considered duplicates.

    Returns:
        list: Filtered list of deduplicated chunk dicts.
    """
    filtered = []
    embeddings = []
    
    print(f"Deduplicating {len(chunks)} chunks with threshold {threshold}")

    for chunk in chunks:
        # Try to get embedding from 'metadata' or root key
        embedding = chunk.get('metadata', {}).get('embedding') or chunk.get('embedding')
        if embedding is None:
            # Skip chunks without embedding to avoid runtime errors
            continue

        emb = np.array(embedding, dtype=np.float32)
        if not embeddings:
            filtered.append(chunk)
            embeddings.append(emb)
            continue

        similarities = [cosine_similarity(emb, e) for e in embeddings]
        if max(similarities) < threshold:
            filtered.append(chunk)
            embeddings.append(emb)

    return filtered

def load_and_deduplicate(threshold=0.9):
    """
    Load all embedding chunks from designated files and deduplicate them lazily.

    Args:
        threshold (float): Cosine similarity threshold for deduplication.

    Returns:
        list: Deduplicated list of chunks.
    """
    chunks = get_embedding_chunks()
    deduped_chunks = deduplicate_by_embedding(chunks, threshold=threshold)
    return deduped_chunks
