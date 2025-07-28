import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

if not pinecone_api_key or not pinecone_index_name:
    raise ValueError("Missing Pinecone credentials in .env")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

query_encoder = SentenceTransformer('all-MiniLM-L6-v2')

def vector_search(query: str, top_k=50):
    query_emb = query_encoder.encode(query)
    results = index.query(vector=query_emb.tolist(), top_k=top_k, include_metadata=True)
    return results['matches']
