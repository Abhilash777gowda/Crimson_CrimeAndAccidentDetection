import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

sites = {
    "Prajavani": "https://www.prajavani.net/",
    "Udayavani": "https://www.udayavani.com/category/state",
    "Vijayakarnataka": "https://vijaykarnataka.com/",
    "Vijayavani": "https://www.vijayavani.net/"
}

with open("tests/scraped_headlines.txt", "w", encoding="utf-8") as f:
    for name, url in sites.items():
        f.write(f"\n--- {name} ---\n")
        try:
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.content, "html.parser")
            
            # Let's find h1, h2, h3 tags first
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
            extracted = []
            for h in headings:
                a_tag = h.find('a')
                if a_tag and a_tag.get_text(strip=True):
                    extracted.append(a_tag.get_text(strip=True))
            
            if extracted:
                f.write(f"Found {len(extracted)} headings:\n")
                for e in extracted[:5]:
                    f.write(f"  - {e}\n")
            else:
                f.write("No headings with links found.\n")
                
            # Let's also find all links over a certain length
            links = soup.find_all('a')
            long_links = [a.get_text(strip=True) for a in links if len(a.get_text(strip=True)) > 25]
            f.write(f"Found {len(long_links)} long links (>25 chars):\n")
            for l in long_links[:5]:
                f.write(f"  - {l}\n")
                
        except Exception as e:
            f.write(f"Error: {e}\n")
