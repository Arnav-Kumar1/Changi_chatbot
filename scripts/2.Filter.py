import re
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import json
import time

# ----- Step 0: URL Language Normalization ----- #
def to_english_url(url):
    url = re.sub(r"/cn/zh/", "/en/", url)
    url = re.sub(r"/zh/", "/en/", url)
    url = re.sub(r"//([^/]+)/zh", r"//\1/en", url)
    return url

# ----- Step 1: Load URLs, Normalize to English, Deduplicate ----- #
def load_and_normalize_urls(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_urls = [line.strip() for line in f if line.strip()]
    normalized_urls = [to_english_url(url) for url in raw_urls]
    unique_urls = list(set(normalized_urls))
    print(f"Loaded and normalized {len(unique_urls)} unique English URLs from {filepath}")
    return unique_urls

# ----- Step 2: Filter Non-HTML/static URLs ----- #
def filter_urls(url_list):
    skip_exts = ('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.css', '.js', '.zip', '.xls', '.xlsx')
    filtered = [url for url in url_list if not url.lower().endswith(skip_exts)]
    print(f"Filtered down to {len(filtered)} HTML content URLs")
    return filtered

# ----- Step 3: Categorize URLs ----- #
def categorize_url(url):
    if '/attractions/' in url or '/attractions' in url:
        return 'Attractions'
    if '/dine' in url:
        return 'Dining'
    if '/shop' in url:
        return 'Shopping'
    if '/promotion' in url:
        return 'Promotions'
    if '/faqs' in url:
        return 'FAQs'
    if '/careers' in url:
        return 'Careers'
    if '/media' in url or '/news' in url:
        return 'Media'
    return 'General'

# ----- Setup HTTP Session with Retry ----- #
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0 Safari/537.36"
    )
}

# ----- Refined HTML Cleanup (header/footer/nav tag removal) ----- #
def clean_html_soup(soup):
    for selector in ["header", "footer", "nav", ".footer", ".nav", "#footer", "#header"]:
        for tag in soup.select(selector):
            tag.decompose()
    return soup

# ----- Improved Text Cleaning: Remove Common Navigation/Footer Phrases ----- #
def remove_navigation_phrases(text):
    common_phrases = [
        "Changi Airport", "Flight Information", "Arrival Guide", "Departure Guide",
        "Lounges", "Map", "Terminal Guides", "Transport & Directions",
        "Special Assistance", "Facilities & Services", "Hotels", "Jewel Changi Airport",
        "Plan Your Events", "Dine & Shop", "Dining", "Shopping", "Changi Pay", "Rewards",
        "Shop Online", "Attractions", "Free Tours", "Events", "Promotions",
        "Changi Rewards", "Benefits & Privileges", "Changi Monarch", "Help", "App & Help",
        "Assistance", "Changi App", "Contact Information", "Download Changi App",
        "Sign Up", "Corporate", "Careers", "Facebook", "Instagram", "LinkedIn",
        "TikTok", "YouTube", "WeChat", "changiairport.com", "jewelchangiairport.com",
        "Sign up for a Changi Account"
    ]

    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        phrase_hits = sum(1 for phrase in common_phrases if phrase.lower() in line_clean.lower())
        if phrase_hits >= 2 or (phrase_hits == 1 and len(line_clean) < 80):
            continue
        filtered_lines.append(line_clean)
    return '\n'.join(filtered_lines)

# ----- Step 4: Robust Content Extraction with HTML Cleanup and Phrase Filtering ----- #
def extract_content(url):
    try:
        resp = session.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            reason = f"Non-200 status code {resp.status_code}"
            print(f"{reason} for {url}")
            return None, None, reason

        soup = BeautifulSoup(resp.text, "html.parser")
        soup = clean_html_soup(soup)

        title = soup.title.text.strip() if soup.title else "No Title"
        main_content = soup.find('main') or soup.body
        if not main_content:
            reason = "No main or body content found"
            return title, None, reason

        text = main_content.get_text(separator="\n", strip=True)
        text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
        text = remove_navigation_phrases(text)

        word_count = len(text.split())
        if word_count < 100:
            reason = f"Content too short ({word_count} words)"
            return title, None, reason

        return title, text, None
    except Exception as e:
        reason = f"Exception during fetch/parse: {str(e)}"
        print(f"{reason} for {url}")
        return None, None, reason

# ----- Step 5: Process URLs and Incrementally Save Content (JSONL) ----- #
def process_and_save_content(categorized_urls, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (url, section) in enumerate(categorized_urls):
            title, content, skip_reason = extract_content(url)
            if content:
                record = {
                    "url": url,
                    "section": section,
                    "title": title,
                    "text": content
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
            else:
                print(f"Skipping URL: {url} — Reason: {skip_reason}")
            if (idx + 1) % 20 == 0 or idx == len(categorized_urls) - 1:
                print(f"Processed {idx + 1} / {len(categorized_urls)} pages")
            time.sleep(1)

# ----- MAIN SCRIPT EXECUTION (Example Usage) ----- #
if __name__ == "__main__":
    changia_urls = load_and_normalize_urls('changai_crawled_urls.txt')
    jewel_changia_urls = load_and_normalize_urls('Jewel_changai_crawled_urls.txt')

    filtered_changia_urls = filter_urls(changia_urls)
    filtered_jewel_urls = filter_urls(jewel_changia_urls)

    categorized_changia_urls = [(url, categorize_url(url)) for url in filtered_changia_urls]
    categorized_jewel_urls = [(url, categorize_url(url)) for url in filtered_jewel_urls]

    process_and_save_content(categorized_changia_urls, 'changia_content.jsonl')
    process_and_save_content(categorized_jewel_urls, 'jewel_content.jsonl')
