"""
SentimentAlpha — News & Data Ingestion
========================================
Scrapes financial news from RSS feeds (Moneycontrol, Economic Times, LiveMint),
Reddit, and can fetch earnings call transcripts.
"""

import re
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from config import RSS_FEEDS, NIFTY50_STOCKS, STOCK_NAMES
from database import db

logger = logging.getLogger("sentimentalpha.ingestion")

# ── Helpers ────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def _parse_date(date_str: str) -> Optional[datetime]:
    """Try to parse various date formats from RSS feeds."""
    if not date_str:
        return None
    try:
        return dateparser.parse(date_str)
    except (ValueError, TypeError):
        return None


def _clean_html(html_text: str) -> str:
    """Strip HTML tags and clean whitespace."""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _generate_url_hash(url: str) -> str:
    """Generate a consistent hash for deduplication."""
    return hashlib.md5(url.encode()).hexdigest()


def _detect_category(title: str, summary: str = "") -> str:
    """Simple rule-based category detection."""
    text = (title + " " + (summary or "")).lower()
    if any(w in text for w in ["nifty", "sensex", "market", "index", "rally", "crash"]):
        return "market"
    if any(w in text for w in ["sector", "industry", "banking", "it ", "pharma", "auto"]):
        return "sector"
    if any(w in text for w in ["rbi", "sebi", "government", "policy", "budget", "gdp", "inflation"]):
        return "macro"
    return "stock"


# ── RSS Feed Scraper ───────────────────────────────────────────────────────

class RSSFeedScraper:
    """Scrapes financial news from configured RSS feeds."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def scrape_all_feeds(self) -> int:
        """Scrape all configured RSS feeds. Returns count of new articles."""
        total_new = 0
        for source_name, feed_urls in RSS_FEEDS.items():
            for feed_url in feed_urls:
                try:
                    new_count = self._scrape_feed(source_name, feed_url)
                    total_new += new_count
                    logger.info(f"[{source_name}] {feed_url} → {new_count} new articles")
                except Exception as e:
                    logger.error(f"[{source_name}] Failed to scrape {feed_url}: {e}")
                time.sleep(1)  # Be polite
        logger.info(f"Total new articles scraped: {total_new}")
        return total_new

    def _scrape_feed(self, source_name: str, feed_url: str) -> int:
        """Scrape a single RSS feed."""
        try:
            resp = self.session.get(feed_url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"HTTP error for {feed_url}: {e}")
            return 0

        feed = feedparser.parse(resp.content)
        new_count = 0

        for entry in feed.entries:
            title = _clean_html(entry.get("title", ""))
            if not title:
                continue

            url = entry.get("link", "")
            summary = _clean_html(entry.get("summary", entry.get("description", "")))
            published = _parse_date(
                entry.get("published", entry.get("updated", ""))
            )
            category = _detect_category(title, summary)

            # Try to get full text for some sources
            full_text = None
            if source_name in ("economic_times",):
                full_text = self._fetch_full_text(url)

            article_id = db.insert_article(
                source=source_name,
                url=url,
                title=title,
                summary=summary,
                full_text=full_text,
                published_at=published,
                category=category,
            )
            if article_id:
                new_count += 1

        return new_count

    def _fetch_full_text(self, url: str) -> Optional[str]:
        """Attempt to fetch full article text (best-effort)."""
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Try common article body selectors
            for selector in [
                "div.artText", "div.article_content", "div.story-element",
                "div.paywall", "article", "div.content_wrapper",
            ]:
                body = soup.select_one(selector)
                if body:
                    text = body.get_text(separator=" ", strip=True)
                    if len(text) > 100:
                        return text[:5000]  # Cap at 5000 chars
            return None
        except Exception:
            return None


# ── Reddit Scraper ─────────────────────────────────────────────────────────

class RedditScraper:
    """Scrapes stock-related discussions from Reddit's public JSON API."""

    SUBREDDITS = ["IndianStreetBets", "IndiaInvestments", "DalalStreetTalks"]
    BASE_URL = "https://www.reddit.com/r/{subreddit}/hot.json?limit=25"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            **HEADERS,
            "Accept": "application/json",
        })

    def scrape_all(self) -> int:
        """Scrape all configured subreddits. Returns count of new articles."""
        total_new = 0
        for subreddit in self.SUBREDDITS:
            try:
                new_count = self._scrape_subreddit(subreddit)
                total_new += new_count
                logger.info(f"[Reddit/{subreddit}] {new_count} new posts")
            except Exception as e:
                logger.error(f"[Reddit/{subreddit}] Failed: {e}")
            time.sleep(2)
        return total_new

    def _scrape_subreddit(self, subreddit: str) -> int:
        url = self.BASE_URL.format(subreddit=subreddit)
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Reddit API error: {e}")
            return 0

        new_count = 0
        for post in data.get("data", {}).get("children", []):
            pdata = post.get("data", {})
            title = pdata.get("title", "")
            if not title:
                continue

            permalink = f"https://www.reddit.com{pdata.get('permalink', '')}"
            selftext = pdata.get("selftext", "")[:2000]
            created_utc = pdata.get("created_utc")
            published = datetime.utcfromtimestamp(created_utc) if created_utc else None
            category = _detect_category(title, selftext)

            article_id = db.insert_article(
                source="reddit",
                url=permalink,
                title=title,
                summary=selftext[:500] if selftext else None,
                full_text=selftext if len(selftext) > 500 else None,
                published_at=published,
                category=category,
            )
            if article_id:
                new_count += 1

        return new_count


# ── Earnings Transcript Scraper ────────────────────────────────────────────

class EarningsTranscriptScraper:
    """
    Scrapes earnings call transcript summaries from publicly available sources.
    Uses Trendlyne / Screener as fallback sources.
    """

    SCREENER_URL = "https://www.screener.in/company/{symbol}/consolidated/"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def scrape_transcripts(self, tickers: list = None) -> int:
        """Scrape earnings transcripts for given tickers."""
        if tickers is None:
            tickers = list(NIFTY50_STOCKS.keys())

        total_new = 0
        for ticker in tickers[:10]:  # Rate-limit to 10 per run
            try:
                symbol = ticker.replace(".NS", "")
                new_count = self._scrape_screener(symbol, ticker)
                total_new += new_count
            except Exception as e:
                logger.error(f"[Earnings/{ticker}] Failed: {e}")
            time.sleep(2)
        return total_new

    def _scrape_screener(self, symbol: str, ticker: str) -> int:
        """Try to get latest earnings info from screener.in."""
        try:
            url = self.SCREENER_URL.format(symbol=symbol)
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                return 0

            soup = BeautifulSoup(resp.text, "html.parser")
            # Look for quarterly results section
            results_section = soup.find("section", {"id": "quarters"})
            if not results_section:
                return 0

            # Extract headline financial data as a pseudo-transcript
            text = results_section.get_text(separator=" ", strip=True)[:2000]
            if not text:
                return 0

            article_id = db.insert_article(
                source="earnings_transcript",
                url=url,
                title=f"Quarterly Results - {NIFTY50_STOCKS.get(ticker, {}).get('name', symbol)}",
                summary=text[:500],
                full_text=text,
                published_at=datetime.utcnow(),
                category="stock",
            )
            return 1 if article_id else 0

        except Exception as e:
            logger.warning(f"Screener scrape failed for {symbol}: {e}")
            return 0


# ── Master Ingestion Runner ───────────────────────────────────────────────

class NewsIngestionPipeline:
    """Orchestrates all news ingestion sources."""

    def __init__(self):
        self.rss_scraper = RSSFeedScraper()
        self.reddit_scraper = RedditScraper()
        self.earnings_scraper = EarningsTranscriptScraper()

    def run_full_ingestion(self) -> dict:
        """Run all scrapers and return counts."""
        results = {}

        logger.info("=" * 60)
        logger.info("Starting full news ingestion pipeline")
        logger.info("=" * 60)

        # 1. RSS Feeds
        logger.info("Phase 1: RSS Feeds")
        results["rss"] = self.rss_scraper.scrape_all_feeds()

        # 2. Reddit
        logger.info("Phase 2: Reddit")
        results["reddit"] = self.reddit_scraper.scrape_all()

        # 3. Earnings Transcripts
        logger.info("Phase 3: Earnings Transcripts")
        results["earnings"] = self.earnings_scraper.scrape_transcripts()

        total = sum(results.values())
        logger.info(f"Ingestion complete: {total} new items ({results})")
        return results

    def run_rss_only(self) -> int:
        """Quick run — RSS feeds only."""
        return self.rss_scraper.scrape_all_feeds()


# ── CLI Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    pipeline = NewsIngestionPipeline()
    results = pipeline.run_full_ingestion()
    print(f"\nIngestion complete: {results}")
