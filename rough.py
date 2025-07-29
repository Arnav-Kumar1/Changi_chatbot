import json

file_path = "data/Google_changia_sparse_embs.jsonl.jsonl"  # replace with actual path if needed

total_values = 0
with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
            if "embedding" in obj:
                count = len(obj["embedding"])
                total_values += count
                print(f"Line {i}: embedding length = {count}")
            else:
                print(f"Line {i}: No 'embedding' key found")
        except json.JSONDecodeError as e:
            print(f"Line {i}: ❌ JSON Decode Error: {e}")

print("\n✅ Total embedding values across all lines:", total_values)
