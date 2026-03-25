"""
News service using GNews API (free tier).
Fetches the latest news (within 24 hours) for a given stock ticker.

GNews Free Tier:
  - 100 requests/day
  - 10 articles per request
  - Search by keyword
  - Filter by time range

Called every time the model prediction is triggered to get the latest news feed.
"""

import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

# GNews API endpoint
GNEWS_BASE_URL = "https://gnews.io/api/v4/search"

# Company name mappings for better search results
TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google Alphabet",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "META": "Meta Facebook",
    "NVDA": "Nvidia",
    "JPM": "JPMorgan",
    "V": "Visa",
    "JNJ": "Johnson Johnson",
    "WMT": "Walmart",
    "PG": "Procter Gamble",
    "MA": "Mastercard",
    "UNH": "UnitedHealth",
    "HD": "Home Depot",
    "DIS": "Disney",
    "BAC": "Bank of America",
    "XOM": "Exxon Mobil",
    "NFLX": "Netflix",
    "KO": "Coca Cola",
    "PEP": "PepsiCo",
    "ADBE": "Adobe",
    "CRM": "Salesforce",
    "AMD": "AMD semiconductor",
    "INTC": "Intel",
    "CSCO": "Cisco",
    "ORCL": "Oracle",
    "BA": "Boeing",
    "GS": "Goldman Sachs",
    "CVX": "Chevron",
}


class NewsService:
    """Fetches recent news articles for stock tickers via GNews API."""

    def __init__(self):
        self.api_key = settings.gnews_api_key
        if not self.api_key or self.api_key == "your_gnews_api_key_here":
            logger.warning(
                "GNews API key not configured. "
                "Sentiment features will default to neutral (0.0). "
                "Get a free key at https://gnews.io"
            )

    def get_news(self, ticker: str, max_articles: int = 10) -> List[Dict]:
        """
        Fetch news articles from the past 24 hours for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            max_articles: Maximum articles to return (GNews free max = 10)

        Returns:
            List of dicts with keys: title, description, url, published_at
        """
        if not self.api_key or self.api_key == "your_gnews_api_key_here":
            logger.debug(f"No API key — returning empty news for {ticker}")
            return []

        # Build search query: use company name if known, else ticker + "stock"
        company = TICKER_TO_COMPANY.get(ticker, ticker)
        search_query = f"{company} stock"

        # Calculate 24-hour window
        now = datetime.now(timezone.utc)
        from_time = (now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "q": search_query,
            "lang": "en",
            "country": "us",
            "max": min(max_articles, 10),
            "from": from_time,
            "sortby": "relevance",
            "apikey": self.api_key,
        }

        try:
            response = requests.get(
                GNEWS_BASE_URL,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                })

            logger.info(
                f"Fetched {len(articles)} articles for {ticker} "
                f"(query: '{search_query}')"
            )
            return articles

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                logger.error(
                    f"GNews API key invalid or rate limit exceeded: {e}"
                )
            else:
                logger.error(f"GNews HTTP error for {ticker}: {e}")
            return []
        except requests.exceptions.Timeout:
            logger.error(f"GNews request timed out for {ticker}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching news for {ticker}: {e}")
            return []

    def get_news_batch(self, tickers: List[str]) -> Dict[str, List[Dict]]:
        """
        Fetch news for multiple tickers.
        Returns a dict mapping ticker -> list of articles.
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.get_news(ticker)
        return results