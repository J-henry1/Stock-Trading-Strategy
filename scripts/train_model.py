#!/usr/bin/env python3
"""
Model Training Script
=====================

Orchestrates the full training pipeline:
  1. Fetches historical prices for a set of training tickers via yfinance
  2. Fetches recent news for each ticker via GNews API
  3. Computes sentiment scores (VADER + TextBlob)
  4. Builds feature matrix with technical indicators + sentiment
  5. Normalizes features with StandardScaler
  6. Runs stratified K-fold cross-validation
  7. Trains the final XGBoost model
  8. Saves model, scaler, and metadata to models/

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --tickers AAPL MSFT GOOGL
    python scripts/train_model.py --folds 10

After training, start the API:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
import argparse
import time

# Add project root to path so imports work when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from config.settings import settings
from services.news_service import NewsService
from services.price_service import PriceService
from services.sentiment import SentimentAnalyzer
from utils.feature_engine import FeatureEngine, FEATURE_COLUMNS, TARGET_COLUMN
from utils.trainer import Trainer
from utils.logger import get_logger

logger = get_logger("train")

# -----------------------------------------------------------------
# Default training tickers — diversified across sectors
# -----------------------------------------------------------------
DEFAULT_TICKERS = [
    "AAPL",   # Tech
    "MSFT",   # Tech
    "GOOGL",  # Tech
    "AMZN",   # Consumer / Tech
    "TSLA",   # Auto / Tech
    "META",   # Tech / Social
    "NVDA",   # Semiconductors
    "JPM",    # Finance
    "V",      # Finance
    "JNJ",    # Healthcare
    "WMT",    # Retail
    "PG",     # Consumer Staples
    "UNH",    # Healthcare
    "HD",     # Retail
    "DIS",    # Entertainment
    "BAC",    # Finance
    "XOM",    # Energy
    "NFLX",   # Entertainment
    "KO",     # Consumer Staples
    "AMD",    # Semiconductors
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the stock sentiment predictor model")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols to use for training (default: 20 diversified stocks)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=settings.cv_folds,
        help=f"Number of cross-validation folds (default: {settings.cv_folds})",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=settings.history_days,
        help=f"Days of historical price data (default: {settings.history_days})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 70)
    print("  STOCK SENTIMENT PREDICTOR — MODEL TRAINING")
    print("=" * 70)
    print(f"  Tickers:       {len(args.tickers)} stocks")
    print(f"  History:       {args.history_days} days")
    print(f"  CV Folds:      {args.folds}")
    print(f"  Model Version: {settings.model_version}")
    print("=" * 70)

    # Initialize services
    price_service = PriceService(history_days=args.history_days)
    news_service = NewsService()
    sentiment_analyzer = SentimentAnalyzer()
    feature_engine = FeatureEngine()
    trainer = Trainer(feature_engine)

    # -----------------------------------------------------------------
    # Step 1 & 2: Fetch data for all tickers
    # -----------------------------------------------------------------
    all_X = []
    all_y = []
    successful_tickers = []
    failed_tickers = []

    for i, ticker in enumerate(args.tickers, 1):
        print(f"\n[{i}/{len(args.tickers)}] Processing {ticker}...")

        # Fetch historical prices (called fresh)
        hist = price_service.get_historical_prices(ticker)
        if hist is None:
            print(f"  ✗ Skipping {ticker} — no historical data")
            failed_tickers.append(ticker)
            continue

        # Compute technical features
        tech_df = price_service.compute_technical_features(hist)

        # Fetch latest news (called fresh — last 24 hours)
        articles = news_service.get_news(ticker)
        sentiment = sentiment_analyzer.analyze_articles(articles)

        print(
            f"  ✓ {len(hist)} price rows, {len(articles)} articles, "
            f"sentiment={sentiment.get('compound', 0):.3f}"
        )

        # Build features
        X, y = feature_engine.build_training_features(tech_df, sentiment)

        if X.empty:
            print(f"  ✗ Skipping {ticker} — no valid features after NaN removal")
            failed_tickers.append(ticker)
            continue

        all_X.append(X)
        all_y.append(y)
        successful_tickers.append(ticker)
        print(f"  ✓ {len(X)} training samples extracted")

    # -----------------------------------------------------------------
    # Combine all tickers into one dataset
    # -----------------------------------------------------------------
    if not all_X:
        print("\n✗ ERROR: No valid training data. Check your tickers and API keys.")
        sys.exit(1)

    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)

    print(f"\n{'=' * 70}")
    print(f"  DATA SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Successful tickers: {len(successful_tickers)} / {len(args.tickers)}")
    print(f"  Failed tickers:     {failed_tickers if failed_tickers else 'None'}")
    print(f"  Total samples:      {len(X_combined)}")
    print(f"  Features:           {len(FEATURE_COLUMNS)}")
    print(f"  Class distribution: Buy={int(y_combined.sum())}, Sell={int(len(y_combined) - y_combined.sum())}")
    print(f"  Buy ratio:          {y_combined.mean():.2%}")

    # -----------------------------------------------------------------
    # Step 3: Normalize features
    # -----------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  NORMALIZING FEATURES (StandardScaler)")
    print(f"{'=' * 70}")

    X_scaled = feature_engine.fit_scaler(X_combined)

    print(f"  ✓ Features normalized to zero mean, unit variance")
    print(f"  Feature stats after scaling:")
    print(f"    Mean range: [{X_scaled.mean().min():.6f}, {X_scaled.mean().max():.6f}]")
    print(f"    Std range:  [{X_scaled.std().min():.4f}, {X_scaled.std().max():.4f}]")

    # -----------------------------------------------------------------
    # Step 4: Cross-validation
    # -----------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  CROSS-VALIDATION ({args.folds}-Fold Stratified)")
    print(f"{'=' * 70}")

    cv_results = trainer.cross_validate(X_scaled, y_combined, n_folds=args.folds)

    print(f"\n  Results:")
    print(f"  {'Metric':<12} {'Mean':>8} {'± Std':>10}")
    print(f"  {'-' * 32}")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        mean = cv_results[f"cv_{metric}_mean"]
        std = cv_results[f"cv_{metric}_std"]
        train_mean = cv_results[f"train_{metric}_mean"]
        print(f"  {metric:<12} {mean:>8.4f} {'±':>3} {std:<6.4f}  (train: {train_mean:.4f})")

    # -----------------------------------------------------------------
    # Step 5: Train final model
    # -----------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  TRAINING FINAL MODEL")
    print(f"{'=' * 70}")

    model = trainer.train(X_scaled, y_combined)

    # -----------------------------------------------------------------
    # Step 6: Save everything
    # -----------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  SAVING MODEL & ARTIFACTS")
    print(f"{'=' * 70}")

    trainer.save_model()
    feature_engine.save_scaler()

    print(f"  ✓ Model:    {settings.model_path}")
    print(f"  ✓ Scaler:   {settings.scaler_path}")
    print(f"  ✓ Metadata: {settings.metadata_path}")

    # -----------------------------------------------------------------
    # Feature importance
    # -----------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  FEATURE IMPORTANCE (Top 10)")
    print(f"{'=' * 70}")

    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    for _, row in importance_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:<25} {row['importance']:.4f} {bar}")

    # -----------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  ✓ TRAINING COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 70}")
    print(f"\n  Next step: Start the API server:")
    print(f"    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    print(f"\n  Then test:")
    print(f'    curl -X POST http://localhost:8000/api/predict \\')
    print(f'      -H "Content-Type: application/json" \\')
    print(f'      -d \'{{"tickers": ["AAPL", "MSFT"]}}\'\n')


if __name__ == "__main__":
    main()
