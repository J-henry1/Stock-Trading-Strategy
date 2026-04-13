# Stock-Trading-Strategy
Stock Trading Model used for determining "Buy"/"Sell" indicators. 

Here's a polished README. Replace everything in README.md with this:
markdown# Stock Trading Strategy — Sentiment-Enhanced ML API

A machine learning API that generates **Buy/Sell signals** for stocks by combining real-time news sentiment with historical price patterns. Built with XGBoost, FinBERT, and FastAPI.

End Point for Testing: [https://stock-trading-strategy.onrender.com/docs](https://stock-trading-strategy.onrender.com/docs)

---

## Overview

Traditional stock analysis requires combining historical price trends with breaking news — a task that's difficult for retail investors to do in real time. This system automates that by:

1. Pulling the latest price history for any ticker via Yahoo Finance
2. Fetching news from the past 24 hours via Finnhub
3. Scoring sentiment with FinBERT (a BERT model fine-tuned on financial text)
4. Running the combined features through an XGBoost classifier
5. Returning a Buy/Sell signal with confidence score

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| ML Model | XGBoost (gradient boosting) |
| Sentiment Analysis | FinBERT (HuggingFace: `ProsusAI/finbert`) |
| Stock Data | yfinance (Yahoo Finance) |
| News Data | Finnhub API (free tier, 60 req/min) |
| Feature Scaling | scikit-learn StandardScaler |
| Deployment | Render (Docker-ready) |

---

## Features

- Accepts **1 to 50 stock tickers** per request
- Returns Buy/Sell signal, confidence score, sentiment score, latest headline, and technical indicators
- Handles any US-listed ticker (not limited to trained set)
- Works on stocks not seen during training — model learns patterns, not memorization
- CORS-enabled for React frontend integration
- Interactive Swagger UI at `/docs`

---

## Model Architecture

**Target variable:** Binary classification — `1 (Buy)` if tomorrow's Close > today's Open, else `0 (Sell)`

**16 Features:**
- **Sentiment (5):** FinBERT compound, positive, negative scores + subjectivity + news volume
- **Technical (11):** 1d/5d/20d price returns, 10d/20d volatility, volume ratio, RSI, MACD, Bollinger Band position, SMA ratios (20d/50d)

**Overfitting prevention:**
- StandardScaler normalization
- L1 + L2 regularization (`reg_alpha=0.1`, `reg_lambda=1.0`)
- Max tree depth = 4
- 80% row + column subsampling
- Early stopping on validation loss
- 5-fold stratified cross-validation

**Training dataset:** 4,000 samples across 20 diversified stocks (tech, finance, healthcare, energy, retail, entertainment) spanning 365 days of history.

---

## Model Performance

| Metric | Cross-Validation | Train | Validation Set |
|--------|-----------------|-------|----------------|
| Accuracy | 67.8% ± 1.6% | 82.2% | 67.9% |
| Precision | 68.8% ± 1.3% | 82.4% | 68% |
| Recall | 70.0% ± 2.8% | 83.7% | 68% |
| F1 Score | 69.3% ± 1.8% | 83.1% | 70% |

---

## API Endpoints

### `POST /api/predict`

**Request:**
```json
{
  "tickers": ["AAPL", "NVDA", "TSLA"]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "ticker": "AAPL",
      "current_price": 251.64,
      "signal": "Buy",
      "confidence": 0.73,
      "sentiment_score": 0.035,
      "latest_headline": "Apple announces new AI features...",
      "price_change_5d_pct": 2.1,
      "volume_ratio": 1.3,
      "rsi_14": 58.2,
      "timestamp": "2026-04-12T23:30:00Z"
    }
  ],
  "model_version": "xgb_v1",
  "generated_at": "2026-04-12T23:30:00Z"
}
```

### `GET /api/health`
Health check — confirms server and model status.

### `GET /api/model-info`
Returns training metadata, CV scores, and feature list.

---
