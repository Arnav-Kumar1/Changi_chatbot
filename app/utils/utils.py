import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_by_embedding(chunks, model, threshold=0.95):
    """
    Deduplicates chunks based on cosine similarity of their embeddings.
    """
    unique = []
    seen_embeddings = []

    for chunk in chunks:
        chunk_text = chunk["page_content"]
        embedding = model.encode(chunk_text)

        if seen_embeddings:
            similarities = cosine_similarity([embedding], seen_embeddings)[0]
            if np.max(similarities) > threshold:
                continue

        seen_embeddings.append(embedding)
        unique.append(chunk)

    return unique
