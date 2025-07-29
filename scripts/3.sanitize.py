import json
import unicodedata
import re
import urllib.parse

def clean_text(text):
    # cleaning function here (e.g., for removing accented chars, Chinese chars, unwanted symbols)
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sanitize_url(url: str) -> str:
    """
    URL-encode non-ASCII in query parameter values (e.g., searchTerm)
    """
    parsed = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)

    encoded_query = []
    for key, values in query_params.items():
        for value in values:
            encoded_value = urllib.parse.quote(value, safe='')
            encoded_query.append(f"{key}={encoded_value}")

    new_query = "&".join(encoded_query)
    sanitized_url = urllib.parse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))
    return sanitized_url

def should_exclude_url(url):
    # Auto exclude URLs by patterns or keywords you don't want
    exclude_keywords = [
        'search.html',
        'login',
        '.xls',
        '.xlsx',
        '.doc',
        '/media/',
        '/bin/',
        '/cgi-bin/',
        '/admin',
        'logout',
    ]
    url_lower = url.lower()
    if any(kw in url_lower for kw in exclude_keywords):
        return True
    return False

def sanitize_content_file(input_path, output_path):
    filtered_count = 0
    total_count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            total_count += 1
            record = json.loads(line)
            url = record.get('url', '')
            if should_exclude_url(url):
                print(f"Excluded URL by pattern: {url}")
                filtered_count += 1
                continue
            record['url'] = sanitize_url(url)
            # Clean text here
            record['text'] = clean_text(record.get('text', ''))
            if len(record['text'].split()) < 50:  # optional minimum length filter
                print(f"Excluded short content at URL: {url}")
                filtered_count += 1
                continue
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Sanitized content file created: {output_path}")
    print(f"Total records: {total_count}, Filtered out: {filtered_count}, Kept: {total_count - filtered_count}")

if __name__ == "__main__":
    sanitize_content_file('changia_content.jsonl', 'changia_content_sanitized.jsonl')
    sanitize_content_file('jewel_content.jsonl', 'jewel_content_sanitized.jsonl')
