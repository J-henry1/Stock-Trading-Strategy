"""
Prediction engine — orchestrates the full inference pipeline:
  1. Fetch latest prices (called fresh every time)
  2. Fetch latest news (called fresh every time — last 24 hours)
  3. Compute sentiment scores
  4. Build features
  5. Normalize with saved scaler
  6. Run model inference
  7. Return structured predictions

Designed to handle single ticker or batch of up to 50 tickers.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import xgboost as xgb
import numpy as np

from config.settings import settings
from services.news_service import NewsService
from services.price_service import PriceService
from services.sentiment import SentimentAnalyzer
from utils.feature_engine import FeatureEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Production inference engine.
    Loads model + scaler once, then processes prediction requests.
    """

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_engine = FeatureEngine()
        self.price_service = PriceService()
        self.news_service = NewsService()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.metadata: Dict = {}
        self._is_loaded = False

    def load(self):
        """Load the trained model, scaler, and metadata from disk."""
        model_path = settings.model_path
        scaler_path = settings.scaler_path
        metadata_path = settings.metadata_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run 'python scripts/train_model.py' first."
            )

        # Load model
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load scaler
        self.feature_engine.load_scaler(scaler_path)

        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)

        self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict(self, tickers: List[str]) -> Dict:
        """
        Generate Buy/Sell predictions for a list of tickers.

        For each ticker, this method:
          1. Calls yfinance to get latest historical prices
          2. Calls GNews API to get latest 24-hour news
          3. Runs sentiment analysis on the news
          4. Builds and normalizes the feature vector
          5. Runs the XGBoost model for prediction + confidence

        Args:
            tickers: List of validated ticker symbols (max 50)

        Returns:
            Dict with "predictions" list and metadata
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call predictor.load() first.")

        predictions = []

        for ticker in tickers:
            try:
                prediction = self._predict_single(ticker)
                if prediction:
                    predictions.append(prediction)
                else:
                    predictions.append(self._error_prediction(ticker, "Insufficient data"))
            except Exception as e:
                logger.error(f"Prediction failed for {ticker}: {e}")
                predictions.append(self._error_prediction(ticker, str(e)))

        return {
            "predictions": predictions,
            "model_version": settings.model_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _predict_single(self, ticker: str) -> Optional[Dict]:
        """
        Run the full prediction pipeline for one ticker.
        Both price and news APIs are called fresh each time.
        """
        # --- Step 1: Fetch latest prices ---
        logger.info(f"[{ticker}] Fetching latest prices...")
        hist = self.price_service.get_historical_prices(ticker)
        if hist is None:
            logger.warning(f"[{ticker}] No historical data available")
            return None

        current = self.price_service.get_current_price(ticker)
        if current is None:
            logger.warning(f"[{ticker}] No current price available")
            return None

        # Compute technical features
        tech_df = self.price_service.compute_technical_features(hist)

        # --- Step 2: Fetch latest news (last 24 hours) ---
        logger.info(f"[{ticker}] Fetching latest news...")
        articles = self.news_service.get_news(ticker)

        # --- Step 3: Compute sentiment ---
        sentiment = self.sentiment_analyzer.analyze_articles(articles)

        # --- Step 4: Build features ---
        features = self.feature_engine.build_inference_features(tech_df, sentiment)
        if features is None or features.empty:
            logger.warning(f"[{ticker}] Feature construction failed")
            return None

        # --- Step 5: Normalize ---
        features_scaled = self.feature_engine.transform(features)

        # --- Step 6: Predict ---
        prob = self.model.predict_proba(features_scaled)[0]
        prediction_class = int(np.argmax(prob))
        confidence = float(np.max(prob))

        signal = "Buy" if prediction_class == 1 else "Sell"

        result = {
            "ticker": ticker,
            "current_price": current["close"],
            "signal": signal,
            "confidence": round(confidence, 4),
            "sentiment_score": round(sentiment.get("compound", 0.0), 4),
            "latest_headline": sentiment.get("latest_headline", ""),
            "price_change_5d_pct": round(
                float(features["price_change_5d"].iloc[0]) if "price_change_5d" in features.columns else 0.0,
                2
            ),
            "volume_ratio": round(
                float(features["volume_ratio"].iloc[0]) if "volume_ratio" in features.columns else 0.0,
                2
            ),
            "rsi_14": round(
                float(features["rsi_14"].iloc[0]) if "rsi_14" in features.columns else 50.0,
                2
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"[{ticker}] → {signal} (confidence={confidence:.2%}, "
            f"sentiment={sentiment.get('compound', 0):.3f}, "
            f"price=${current['close']})"
        )

        return result

    @staticmethod
    def _error_prediction(ticker: str, error: str) -> Dict:
        """Return a structured error for a failed ticker."""
        return {
            "ticker": ticker,
            "current_price": None,
            "signal": "Error",
            "confidence": 0.0,
            "sentiment_score": 0.0,
            "latest_headline": "",
            "price_change_5d_pct": 0.0,
            "volume_ratio": 0.0,
            "rsi_14": 0.0,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
