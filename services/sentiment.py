"""
Sentiment analysis engine using an ensemble of:
  1. VADER (Valence Aware Dictionary and sEntiment Reasoner) — rule-based
  2. TextBlob — NLP pattern-based

Both are free, local libraries with no API limits.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from typing import Dict, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """Computes sentiment scores from news article text."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze a single text string and return sentiment scores.

        Returns:
            {
                "compound": float,   # VADER compound [-1, 1]
                "positive": float,   # VADER positive [0, 1]
                "negative": float,   # VADER negative [0, 1]
                "neutral": float,    # VADER neutral [0, 1]
                "polarity": float,   # TextBlob polarity [-1, 1]
                "subjectivity": float # TextBlob subjectivity [0, 1]
            }
        """
        if not text or not text.strip():
            return self._neutral_scores()

        # VADER scores
        vader_scores = self.vader.polarity_scores(text)

        # TextBlob scores
        blob = TextBlob(text)

        return {
            "compound": vader_scores["compound"],
            "positive": vader_scores["pos"],
            "negative": vader_scores["neg"],
            "neutral": vader_scores["neu"],
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
        }

    def analyze_articles(self, articles: List[Dict]) -> Dict[str, float]:
        """
        Analyze a list of news articles and return aggregated sentiment.

        Each article should have 'title' and optionally 'description' keys.
        We analyze both title and description, weighted:
            - Title: 60% weight (headlines carry strong signal)
            - Description: 40% weight

        Returns aggregated (mean) sentiment scores + article count.
        """
        if not articles:
            scores = self._neutral_scores()
            scores["news_volume"] = 0
            scores["latest_headline"] = ""
            return scores

        all_scores = []

        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")

            # Analyze title
            title_scores = self.analyze_text(title)

            # Analyze description
            desc_scores = self.analyze_text(description) if description else self._neutral_scores()

            # Weighted combination
            combined = {}
            for key in title_scores:
                combined[key] = title_scores[key] * 0.6 + desc_scores[key] * 0.4

            all_scores.append(combined)

        # Aggregate: mean across all articles
        aggregated = {}
        keys = all_scores[0].keys()
        for key in keys:
            values = [s[key] for s in all_scores]
            aggregated[key] = sum(values) / len(values)

        # Add metadata
        aggregated["news_volume"] = len(articles)
        aggregated["latest_headline"] = articles[0].get("title", "")

        logger.info(
            f"Sentiment analysis: {len(articles)} articles, "
            f"compound={aggregated['compound']:.3f}, "
            f"news_volume={aggregated['news_volume']}"
        )

        return aggregated

    @staticmethod
    def _neutral_scores() -> Dict[str, float]:
        """Return neutral sentiment scores (no signal)."""
        return {
            "compound": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "polarity": 0.0,
            "subjectivity": 0.0,
        }
