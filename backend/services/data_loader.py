import json
from typing import List, Dict


def load_embedding_chunks(file_path: str) -> List[Dict]:
    """
    Load chunks with precomputed embeddings from a .jsonl file.
    """
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk)
    return chunks

def load_all_embedding_chunks():
    """
    Load all embedding chunk files (dense + sparse) explicitly.
    Returns a tuple (dense_chunks, sparse_chunks).
    """
    dense_files = [
        "data/Google_changia_embs.jsonl",
        "data/Google_jewel_embs.jsonl"
    ]
    sparse_files = [
        "data/Google_changia_sparse_embs.jsonl",
        "data/Google_jewel_sparse_embs.jsonl"
    ]

    dense_chunks = []
    for f in dense_files:
        dense_chunks.extend(load_embedding_chunks(f))

    sparse_chunks = []
    for f in sparse_files:
        sparse_chunks.extend(load_embedding_chunks(f))

    return dense_chunks, sparse_chunks
