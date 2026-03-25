"""
Feature engineering pipeline.

Responsibilities:
  1. Combine technical indicators (from price_service) with sentiment scores
  2. Handle missing values
  3. Normalize features using StandardScaler
  4. Define the canonical feature list used by training and inference

All features used here are available from yfinance + GNews (free sources).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import pickle
import os
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

# -----------------------------------------------------------------
# Canonical feature list — order matters for consistency
# These are the columns fed into the model. Every one of them is
# derivable from yfinance OHLCV data + GNews sentiment scores.
# -----------------------------------------------------------------
FEATURE_COLUMNS = [
    # Sentiment features (from GNews + VADER + TextBlob)
    "sentiment_compound",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_subjectivity",
    "news_volume",
    # Price return features (from yfinance)
    "price_change_1d",
    "price_change_5d",
    "price_change_20d",
    # Volatility features (from yfinance)
    "volatility_10d",
    "volatility_20d",
    # Volume feature (from yfinance)
    "volume_ratio",
    # Technical indicators (from yfinance)
    "rsi_14",
    "macd_signal",
    "bb_position",
    "sma_20_ratio",
    "sma_50_ratio",
]

TARGET_COLUMN = "target"


class FeatureEngine:
    """Builds, normalizes, and manages features for the ML pipeline."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False

    def build_training_features(
        self,
        price_df: pd.DataFrame,
        sentiment_scores: Dict[str, float],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build feature matrix + target for a single ticker's training data.

        Args:
            price_df: DataFrame with technical features (from PriceService.compute_technical_features)
            sentiment_scores: Dict of aggregated sentiment scores

        Returns:
            (X, y) — feature DataFrame and target Series, NaN rows dropped
        """
        df = price_df.copy()

        # Map sentiment scores to all rows (same sentiment for all historical rows
        # during training — in production we'd have per-day sentiment, but for
        # initial training we use the current sentiment as a constant feature)
        df["sentiment_compound"] = sentiment_scores.get("compound", 0.0)
        df["sentiment_positive"] = sentiment_scores.get("positive", 0.0)
        df["sentiment_negative"] = sentiment_scores.get("negative", 0.0)
        df["sentiment_subjectivity"] = sentiment_scores.get("subjectivity", 0.0)
        df["news_volume"] = sentiment_scores.get("news_volume", 0)

        # Ensure all feature columns exist
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        # Drop rows with NaN in features or target
        subset = FEATURE_COLUMNS + [TARGET_COLUMN]
        df = df.dropna(subset=[c for c in subset if c in df.columns])

        if df.empty:
            logger.warning("No valid rows after dropping NaN")
            return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=float)

        X = df[FEATURE_COLUMNS].copy()
        y = df[TARGET_COLUMN].copy()

        return X, y

    def build_inference_features(
        self,
        price_df: pd.DataFrame,
        sentiment_scores: Dict[str, float],
    ) -> Optional[pd.DataFrame]:
        """
        Build feature vector for a single ticker at inference time.
        Uses only the most recent row of price data.

        Returns a single-row DataFrame of features, or None on failure.
        """
        df = price_df.copy()

        # Attach sentiment
        df["sentiment_compound"] = sentiment_scores.get("compound", 0.0)
        df["sentiment_positive"] = sentiment_scores.get("positive", 0.0)
        df["sentiment_negative"] = sentiment_scores.get("negative", 0.0)
        df["sentiment_subjectivity"] = sentiment_scores.get("subjectivity", 0.0)
        df["news_volume"] = sentiment_scores.get("news_volume", 0)

        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        # Take the last row (most recent trading day)
        latest = df.iloc[[-1]][FEATURE_COLUMNS].copy()

        # Fill any remaining NaN with 0
        latest = latest.fillna(0.0)

        return latest

    def fit_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the StandardScaler on training data and transform.
        Normalization: zero mean, unit variance for each feature.
        """
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )
        self._is_fitted = True
        logger.info(
            f"Scaler fitted on {len(X)} samples, "
            f"{len(X.columns)} features"
        )
        return X_scaled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the already-fitted scaler.
        Used at inference time.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Scaler not fitted. Call fit_scaler() first or load a saved scaler."
            )

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )
        return X_scaled

    def save_scaler(self, path: str = None):
        """Save the fitted scaler to disk."""
        path = path or settings.scaler_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {path}")

    def load_scaler(self, path: str = None):
        """Load a previously fitted scaler from disk."""
        path = path or settings.scaler_path
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)
        self._is_fitted = True
        logger.info(f"Scaler loaded from {path}")
