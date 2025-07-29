import json

def chunk_text(text, max_words=300):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def chunk_content_only(input_file, output_file, max_words=300):
    chunk_count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            record = json.loads(line)
            chunks = chunk_text(record['text'], max_words=max_words)
            for i, chunk in enumerate(chunks):
                # No cleaning/sanitization here, just chunk and save verbatim
                chunk_record = {
                    "chunk_id": f"{record['url']}_chunk{i}",
                    "url": record['url'],
                    "section": record.get('section', ''),
                    "title": record.get('title', ''),
                    "text": chunk
                }
                fout.write(json.dumps(chunk_record, ensure_ascii=False) + '\n')
                chunk_count += 1
    print(f"Total chunks prepared and saved (no cleaning): {chunk_count}")

if __name__ == "__main__":
    # Change filenames and chunk size as needed
    chunk_content_only('changia_content_sanitized.jsonl', 'changia_embedding_ready_raw_chunks.jsonl', max_words=300)
    chunk_content_only('jewel_content_sanitized.jsonl', 'jewel_embedding_ready_raw_chunks.jsonl', max_words=300)
