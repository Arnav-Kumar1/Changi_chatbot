import json
import hashlib
from pinecone import Pinecone, ServerlessSpec
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

index_name = "developer-quickstart-py"
dimension = 384
BATCH_SIZE = 100
MAX_WORKERS = 8

def load_chunks(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def safe_id(chunk_id):
    return hashlib.sha256(chunk_id.encode('utf-8')).hexdigest() if len(chunk_id) > 512 else chunk_id

def make_batches(chunks):
    batch = []
    for chunk in chunks:
        batch.append((
            safe_id(chunk['chunk_id']),
            chunk['embedding'],
            {
                'orig_chunk_id': chunk['chunk_id'],
                'section': chunk['section'],
                'title': chunk['title'],
                'text': chunk['text']
            }
        ))
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = []
    if batch:
        yield batch

# For thread-safe printing and counting
lock = Lock()
progress = {"completed": 0, "total": 0}

def upsert_batch(index, batch, batch_num):
    index.upsert(batch)
    with lock:
        progress["completed"] += 1
        pct = (progress["completed"] / progress["total"]) * 100
        print(f"[{progress['completed']:>3}/{progress['total']}] Batch {batch_num} uploaded ({len(batch)} vectors) - {pct:.2f}% done")

def store_embeddings_multithreaded(chunks, index):
    batches = list(make_batches(chunks))
    progress["total"] = len(batches)
    print(f"\nUploading {progress['total']} batches of upserts with {MAX_WORKERS} workers...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(upsert_batch, index, batch, i + 1): i + 1
            for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                with lock:
                    print(f"❌ Error in batch {futures[future]}: {e}")

if __name__ == "__main__":
    changia_chunks = load_chunks('changia_embs.jsonl')
    jewel_chunks = load_chunks('jewel_embs.jsonl')
    all_chunks = changia_chunks + jewel_chunks

    pc = Pinecone(api_key="pcsk_3bj289_SdURc4yMW1nUUwaJfrArv2x3RzpD4ooitUhupmrmFk3Aa83SPTtYQWAdEMJ3b5V")

    if index_name not in pc.list_indexes().names():
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Index '{index_name}' already exists. Reusing it.")

    index = pc.Index(index_name)
    store_embeddings_multithreaded(all_chunks, index)
