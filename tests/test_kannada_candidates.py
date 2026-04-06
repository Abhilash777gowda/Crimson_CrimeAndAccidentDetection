from scraper.rss_scraper import RSSNewsScraper
from preprocessing.text_cleaner import TextCleaner
from utils.geocoder import extract_location, geocode_location
import feedparser

def test_kannada_candidates():
    cleaner = TextCleaner()
    
    candidates = {
        "Vijayakarnataka (Common)": "https://vijaykarnataka.com",
        "Vijayakarnataka (Karnataka)": "https://vijaykarnataka.com",
        "Vijayakarnataka (State)": "https://vijaykarnataka.com",
        "Prajavani (Latest)": "https://www.prajavani.net/news",
        "Prajavani (Karnataka)": "https://www.prajavani.net",
        "Udayavani (Feed)": "https://www.udayavani.com",
        "Vijayavani (Feed)": "https://www.vijayavani.net",
    }
    
    for name, url in candidates.items():
        print(f"\nChecking {name}: {url}")
        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                print(f"  [!] No entries found.")
                continue
            
            print(f"  [✓] Success! {len(feed.entries)} entries found.")
            entry = feed.entries[0]
            title = entry.get("title", "")
            print(f"  Top Title: {title[:100]}")
            
            cleaned = cleaner.clean_text(title)
            loc = extract_location(cleaned)
            if loc:
                print(f"  Location Found: {loc}")
                
        except Exception as e:
            print(f"  [X] Error: {e}")

if __name__ == "__main__":
    test_kannada_candidates()
