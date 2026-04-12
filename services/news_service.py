"""
News service using Finnhub API (free tier).
Fetches company-specific news for a given stock ticker.

Finnhub Free Tier:
  - 60 API calls/minute
  - Company news by ticker + date range (no keyword guessing)
  - Returns 50+ articles per ticker per day typically
  - Aggregates from Reuters, Bloomberg, CNBC, Seeking Alpha, etc.

Called every time the model prediction is triggered to get the latest news feed.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1/company-news"


class NewsService:
    """Fetches recent news articles for stock tickers via Finnhub API."""

    def __init__(self):
        self.api_key = settings.finnhub_api_key
        if not self.api_key or self.api_key == "your_finnhub_api_key_here":
            logger.warning(
                "Finnhub API key not configured. "
                "Sentiment features will default to neutral (0.0). "
                "Get a free key at https://finnhub.io"
            )

    def get_news(self, ticker: str, max_articles: int = 50, days_back: int = 1) -> List[Dict]:
        """
        Fetch news articles for a ticker from the past N days.

        Finnhub's company-news endpoint returns articles filtered by
        ticker symbol and date range — no keyword guessing needed.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            max_articles: Maximum articles to return
            days_back: Days of history to pull (default 1 = last 24 hours)

        Returns:
            List of dicts with keys: title, description, url, published_at, source
        """
        if not self.api_key or self.api_key == "your_finnhub_api_key_here":
            logger.debug(f"No API key — returning empty news for {ticker}")
            return []

        today = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = {
            "symbol": ticker,
            "from": from_date,
            "to": today,
            "token": self.api_key,
        }

        try:
            response = requests.get(FINNHUB_BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, list):
                logger.warning(f"Unexpected response format for {ticker}: {type(data)}")
                return []

            articles = []
            for article in data[:max_articles]:
                dt_val = article.get("datetime", 0)
                published = (
                    datetime.fromtimestamp(dt_val).isoformat()
                    if dt_val
                    else ""
                )
                articles.append({
                    "title": article.get("headline", ""),
                    "description": article.get("summary", ""),
                    "url": article.get("url", ""),
                    "published_at": published,
                    "source": article.get("source", "Unknown"),
                })

            logger.info(f"Fetched {len(articles)} articles for {ticker}")
            return articles

        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                if e.response.status_code == 401:
                    logger.error(f"Finnhub API key invalid: {e}")
                elif e.response.status_code == 429:
                    logger.error(f"Finnhub rate limit exceeded: {e}")
                else:
                    logger.error(f"Finnhub HTTP error for {ticker}: {e}")
            return []
        except requests.exceptions.Timeout:
            logger.error(f"Finnhub request timed out for {ticker}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching news for {ticker}: {e}")
            return []

    def get_news_batch(self, tickers: List[str]) -> Dict[str, List[Dict]]:
        """Fetch news for multiple tickers."""
        results = {}
        for ticker in tickers:
            results[ticker] = self.get_news(ticker)
        return results