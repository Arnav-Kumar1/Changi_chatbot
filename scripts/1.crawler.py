import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

base_url = "https://www.changiairport.com/"
visited = set()
to_visit = [base_url]

# Open a text file to save visited URLs
with open("changai_crawled_urls.txt", "w", encoding="utf-8") as file:
    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, "html.parser")

            # Write visited URL to the file
            file.write(url + "\n")
            file.flush()  # Ensure immediate write to disk

            # Extract internal links on the page
            for a_tag in soup.find_all("a", href=True):
                href = a_tag['href']
                full_url = urljoin(base_url, href)
                parsed_url = urlparse(full_url)
                if parsed_url.netloc == urlparse(base_url).netloc:
                    if full_url not in visited and full_url not in to_visit:
                        to_visit.append(full_url)

            visited.add(url)

            # Respectful crawling delay
            time.sleep(1)
        except Exception as e:
            # Handle errors (network issues, timeouts, etc.) silently or log if needed
            pass

print(f"Crawling finished. URLs saved to 'changai_crawled_urls.txt'.")
