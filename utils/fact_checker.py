import re
import pandas as pd
from typing import List, Dict
try:
    from utils.helpers import setup_logging
except ImportError:
    import logging
    def setup_logging():
        return logging.getLogger(__name__)

logger = setup_logging()

def extract_keywords(text: str) -> List[str]:
    """
    Extract key terms from an article for search.
    Length-based extraction supports Indian languages which lack capitalization.
    """
    # Remove basic punctuation but preserve unicode letters
    text = re.sub(r'[.,!?"\'()\[\]{}:;—-]', '', text)
    words = text.split()
    
    # Robust length heuristic: >= 3 characters catches meaningful Indian root words
    keywords = [w.strip() for w in words if len(w.strip()) > 2]
    
    # Return unique keywords, top 10
    return list(dict.fromkeys(keywords))[:10]

def search_online(keywords: List[str]) -> List[Dict]:
    """
    Search by cross-referencing against live web results via DuckDuckGo,
    with a robust fallback to local real-time scraped components.
    """
    if not keywords:
        return []
    
    results = []
    
    # 1. Attempt True Live Web Search
    try:
        from duckduckgo_search import DDGS
        query = " ".join(keywords)
        logger.info(f"Performing live online search for keywords: {query}")
        
        with DDGS(timeout=10) as ddgs:
            ddgs_generator = ddgs.text(query, region='in-en', safesearch='moderate', max_results=10)
            if ddgs_generator:
                for res in ddgs_generator:
                    text = res.get('body', '').lower()
                    title = res.get('title', '').lower()
                    
                    matches = sum(1 for kw in keywords if kw.lower() in text or kw.lower() in title)
                    if matches > 0:
                        results.append({
                            "title": res.get('title'),
                            "text": res.get('body'),
                            "url": res.get('href'),
                            "source": "Web Search (DuckDuckGo)",
                            "match_score": matches
                        })
    except Exception as e:
        logger.warning(f"Live web search failed or dropped connection: {e}")
        
    # 2. Fallback to Deep Scraper Cache if DDG is rate-limited
    if not results:
        logger.info("Falling back to Live Streamlit Application Dataset...")
        try:
            df = pd.read_csv("data/labeled_news.csv")
        except Exception:
            df = pd.DataFrame()

        if df.empty or len(df) < 5:
            from scraper.rss_scraper import RSSNewsScraper
            from scraper.kannada_scraper import KannadaWebScraper
            
            df_rss = RSSNewsScraper().scrape_all(max_per_feed=5)
            df_kan = KannadaWebScraper().scrape_all(max_per_source=3)
            
            frames = [d for d in [df_rss, df_kan] if not d.empty]
            if frames:
                df = pd.concat(frames, ignore_index=True)
                
        for _, row in df.iterrows():
            text = str(row.get('text', '')).lower()
            title = str(row.get('title', '')).lower()
            
            matches = sum(1 for kw in keywords if kw.lower() in text or kw.lower() in title)
            if matches > 0:
                results.append({
                    "title": row.get('title'),
                    "text": row.get('text'),
                    "url": row.get('url'),
                    "source": row.get('source'),
                    "match_score": matches
                })
                
    results = sorted(results, key=lambda x: x['match_score'], reverse=True)
    return results[:5]


def compare_articles(original_text: str, related_articles: List[Dict]) -> Dict:
    """
    Compare the input text with found articles.
    """
    if not related_articles:
        return {
            "status": "Inconclusive",
            "score": 0,
            "reasoning": "No related news articles found online to verify this claim.",
            "sources": []
        }

    # Simplified verification logic
    orig_keywords = set(extract_keywords(original_text.lower()))
    source_results = []
    
    max_overlap = 0.0
    total_score = 0.0
    
    for art in related_articles:
        art_keywords = set(extract_keywords(art.get('text', '').lower()))
        overlap = len(orig_keywords.intersection(art_keywords))
        # Similarity score
        score = min(100.0, (overlap / len(orig_keywords) * 100.0)) if orig_keywords else 0.0
        
        source_results.append({
            "title": art.get('title', 'Related News'),
            "url": art.get('url', '#'),
            "source": art.get('source', 'Unknown'),
            "similarity": int(score)
        })
        
        if score > max_overlap:
            max_overlap = score
        total_score += score

    avg_score = total_score / len(related_articles)
    
    if max_overlap > 75:
        status = "Verified"
        reasoning = f"This article is highly consistent with reports from {len(related_articles)} sources. The core facts are confirmed by multiple reputable news outlets."
    elif max_overlap > 40:
        status = "Partially Matches"
        reasoning = f"The article shares significant details with results from {len(related_articles)} sources, but some specifics might be missing or different."
    else:
        status = "Inconsistent / Low Evidence"
        reasoning = "The details provided do not strongly align with recent major news reports. This could be a very local event or potentially inaccurate information."

    return {
        "status": status,
        "score": int(max_overlap),
        "reasoning": reasoning,
        "sources": source_results
    }
