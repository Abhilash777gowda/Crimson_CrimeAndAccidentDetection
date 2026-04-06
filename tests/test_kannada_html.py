from scraper.kannada_scraper import KannadaWebScraper

def test_kannada_html_scraper():
    scraper = KannadaWebScraper()
    print("Testing direct HTML scraping for Kannada sources...")
    df = scraper.scrape_all(max_per_source=5)
    
    print("\n--- Scraping Results ---")
    if df.empty:
        print("No articles fetched!")
    else:
        print(f"Total articles fetched: {len(df)}")
        for source in df['source'].unique():
            print(f"  {source}: {len(df[df['source']==source])} articles")
        
        df.to_csv("tests/test_kannada.csv", index=False, encoding='utf-8')
        print("Articles saved to tests/test_kannada.csv")

if __name__ == "__main__":
    test_kannada_html_scraper()
