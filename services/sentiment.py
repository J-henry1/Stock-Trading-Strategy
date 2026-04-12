"""
Sentiment analysis engine using FinBERT.

FinBERT is a BERT model fine-tuned on financial text (earnings calls,
analyst reports, financial news). It understands financial phrases like
"beats estimates" (bullish) and "guidance lowered" (bearish) that
general-purpose tools like VADER miss.

Model: ProsusAI/finbert (HuggingFace)
  - 3 classes: positive, negative, neutral
  - Trained on Reuters TRC-2 financial corpus + Financial PhraseBank
  - ~65% accuracy on financial sentiment vs ~50% for VADER

First run downloads ~440MB model weights, cached thereafter.
Runs on CPU; ~1-2 seconds per batch of headlines.
"""

from typing import Dict, List
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """Computes FinBERT sentiment scores from news article text."""

    _model = None
    _tokenizer = None

    def __init__(self):
        if SentimentAnalyzer._model is None:
            self._load_model()

    @classmethod
    def _load_model(cls):
        """Lazy-load FinBERT on first use."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            logger.info("Loading FinBERT model (first run downloads ~440MB)...")
            model_name = "ProsusAI/finbert"
            cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            cls._model.eval()
            cls._torch = torch
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            raise

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Score a single text string.

        Returns:
            {
                "positive": float [0, 1],
                "negative": float [0, 1],
                "neutral":  float [0, 1],
                "compound": float [-1, 1]  (positive - negative, for XGBoost compatibility)
            }
        """
        if not text or not text.strip():
            return self._neutral_scores()

        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            with self._torch.no_grad():
                outputs = self._model(**inputs)
                probs = self._torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = probs[0].tolist()

            # FinBERT label order: [positive, negative, neutral]
            positive, negative, neutral = probs[0], probs[1], probs[2]
            compound = positive - negative

            return {
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "compound": compound,
            }
        except Exception as e:
            logger.error(f"FinBERT scoring error: {e}")
            return self._neutral_scores()

    def analyze_articles(self, articles: List[Dict]) -> Dict[str, float]:
        """
        Score a list of articles and aggregate.

        Title gets 60% weight, description 40% — headlines carry stronger signal.
        Returns mean scores across all articles + news volume + latest headline.
        """
        if not articles:
            scores = self._neutral_scores()
            scores["subjectivity"] = 0.0
            scores["news_volume"] = 0
            scores["latest_headline"] = ""
            return scores

        all_scores = []
        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")

            title_scores = self.analyze_text(title)
            desc_scores = (
                self.analyze_text(description)
                if description
                else self._neutral_scores()
            )

            combined = {
                key: title_scores[key] * 0.6 + desc_scores[key] * 0.4
                for key in title_scores
            }
            all_scores.append(combined)

        # Average across articles
        aggregated = {}
        for key in all_scores[0]:
            vals = [s[key] for s in all_scores]
            aggregated[key] = sum(vals) / len(vals)

        # Subjectivity proxy: 1 - neutral (opinionated text has lower neutral prob)
        aggregated["subjectivity"] = 1.0 - aggregated.get("neutral", 1.0)
        aggregated["news_volume"] = len(articles)
        aggregated["latest_headline"] = articles[0].get("title", "")

        logger.info(
            f"FinBERT sentiment: {len(articles)} articles, "
            f"compound={aggregated['compound']:.3f}, "
            f"news_volume={aggregated['news_volume']}"
        )
        return aggregated

    @staticmethod
    def _neutral_scores() -> Dict[str, float]:
        return {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "compound": 0.0,
        }