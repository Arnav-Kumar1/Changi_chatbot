import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

query_encoder = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def deduplicate_by_embedding(chunks: list, model: SentenceTransformer, threshold=0.9):
    filtered = []
    embeddings = []

    for c in chunks:
        text = c['metadata']['text']
        emb = model.encode(text)
        if not embeddings:
            filtered.append(c)
            embeddings.append(emb)
            continue
        sim_vals = [cosine_similarity(emb, e) for e in embeddings]
        if max(sim_vals) < threshold:
            filtered.append(c)
            embeddings.append(emb)
    return filtered
