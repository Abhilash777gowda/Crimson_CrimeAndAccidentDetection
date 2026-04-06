from scraper.rss_scraper import RSSNewsScraper
from preprocessing.text_cleaner import TextCleaner
from utils.geocoder import extract_location, geocode_location

def test_kannada_pipeline():
    scraper = RSSNewsScraper()
    cleaner = TextCleaner()
    
    print("Testing Kannada RSS Feeds...")
    # Filter for Kannada sources
    kannada_feeds = {k: v for k, v in scraper.feeds.items() if k in ["Vijayakarnataka", "Prajavani", "Udayavani", "Vijayavani"]}
    
    for source, url in kannada_feeds.items():
        print(f"\nChecking {source}...")
        try:
            import feedparser
            feed = feedparser.parse(url)
            if not feed.entries:
                print(f"  [!] No entries found for {source}. URL might be outdated or blocked.")
                continue
            
            entry = feed.entries[0]
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            full_text = f"{title}. {summary}"
            
            cleaned = cleaner.clean_text(full_text)
            lang = cleaner.detect_language(cleaned)
            loc = extract_location(cleaned)
            
            print(f"  Title: {title[:50]}...")
            print(f"  Cleaned: {cleaned[:50]}...")
            print(f"  Detected Lang: {lang}")
            print(f"  Extracted Loc: {loc}")
            
            if loc:
                lat, lon = geocode_location(loc)
                print(f"  Coordinates: {lat}, {lon}")
                
        except Exception as e:
            print(f"  [X] Error testing {source}: {e}")

if __name__ == "__main__":
    test_kannada_pipeline()
