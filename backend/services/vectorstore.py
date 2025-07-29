import os
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Google and Pinecone credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env file")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise ValueError("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME in .env file.")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def embed_query(query: str) -> list:
    """Embed user query using Google Generative AI embedding API (768-dim)."""
    response = genai.embed_content(model="embedding-001", content=query)
    return response['embedding']

def vector_search(query: str, top_k=50):
    query_emb = embed_query(query)
    results = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True,
        include_values=True  # <--- Add this line to get vectors returned!
    )

    chunks = []
    for match in results['matches']:
        metadata = match.get('metadata', {}) or {}
        # Attach the vector returned by Pinecone as 'embedding' to metadata
        metadata['embedding'] = match.get('values') or match.get('vector') or []
        chunk = {
            'chunk_id': match['id'],
            'metadata': metadata,
            'embedding': metadata['embedding'],
        }
        chunks.append(chunk)

    # Debug print
    print(f"[DEBUG] Pinecone returned {len(chunks)} chunks with embeddings attached.")
    for i, chunk in enumerate(chunks[:len(chunks)+1]):
        emb = chunk.get('embedding') or chunk.get('metadata', {}).get('embedding')
        print(f" Chunk {i}: embedding present? {'Yes' if emb else 'No'}, length: {len(emb) if emb else 'N/A'}")

    return chunks

