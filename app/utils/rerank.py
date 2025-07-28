from sentence_transformers import SentenceTransformer, util

# Load the same model used for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

def rerank(query: str, chunks: list, top_n: int = 6) -> list:
    """
    Re-rank a list of chunks based on semantic similarity to the query.

    Args:
        query (str): The user's question.
        chunks (list): A list of dicts, each with a "text" field.
        top_n (int): Number of top results to return.

    Returns:
        list: Re-ranked list of chunks.
    """
    query_emb = model.encode(query, convert_to_tensor=True)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embs = model.encode(chunk_texts, convert_to_tensor=True)

    cosine_scores = util.cos_sim(query_emb, chunk_embs)[0]
    scored = sorted(zip(chunks, cosine_scores), key=lambda x: x[1], reverse=True)

    top_chunks = [chunk for chunk, score in scored[:top_n]]
    return top_chunks
