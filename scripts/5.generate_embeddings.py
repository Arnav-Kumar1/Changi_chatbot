import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, small, good quality

def get_embedding(text):
    return model.encode(text).tolist()

def add_embeddings(input_jsonl, output_jsonl):
    with open(input_jsonl, 'r', encoding='utf-8') as fin, open(output_jsonl, 'w', encoding='utf-8') as fout:
        for line in fin:
            record = json.loads(line)
            embedding = get_embedding(record['text'])
            record['embedding'] = embedding
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Embeddings written to {output_jsonl}")

if __name__ == "__main__":
    add_embeddings('changia_embedding_ready_raw_chunks.jsonl', 'changia_embs.jsonl')
    add_embeddings('jewel_embedding_ready_raw_chunks.jsonl', 'jewel_embs.jsonl')