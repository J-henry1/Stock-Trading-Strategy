# Stock Sentiment Predictor — Buy/Sell Signal Engine

A production-grade ML pipeline that combines **real-time news sentiment** with **historical stock prices** to generate Buy/Sell signals for up to 50 stock tickers. Exposed via a FastAPI REST API, ready for a React frontend.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  React Frontend (separate repo)                         │
│  Calls: POST /api/predict   GET /api/health             │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP / JSON
┌──────────────────▼──────────────────────────────────────┐
│  FastAPI Application  (main.py)                         │
│  ├── /api/predict      — single or batch ticker input   │
│  ├── /api/health       — health check                   │
│  └── /api/model-info   — model metadata                 │
├─────────────────────────────────────────────────────────┤
│  Services Layer                                         │
│  ├── news_service.py   — GNews API (free, 24hr window)  │
│  ├── price_service.py  — yfinance (free, Yahoo Finance) │
│  └── sentiment.py      — VADER + TextBlob ensemble      │
├─────────────────────────────────────────────────────────┤
│  ML Pipeline                                            │
│  ├── feature_engine.py — normalization, feature build    │
│  ├── trainer.py        — XGBoost w/ cross-validation    │
│  └── predictor.py      — inference engine                │
├─────────────────────────────────────────────────────────┤
│  Utils                                                  │
│  ├── validators.py     — input validation                │
│  └── logger.py         — structured logging              │
└─────────────────────────────────────────────────────────┘
```

---

## Data Sources (All Free)

| Source | What It Provides | Limit |
|--------|-----------------|-------|
| **yfinance** | Historical OHLCV prices, volume, dividends | Unlimited (Yahoo Finance scraper) |
| **GNews API** | News articles from last 24 hours per ticker | 100 requests/day (free tier) |
| **VADER Sentiment** | Rule-based sentiment scoring | No limit (local library) |
| **TextBlob** | NLP-based polarity/subjectivity | No limit (local library) |

---

## Step-by-Step Setup

### Prerequisites
- Python 3.10+
- pip (Python package manager)
- A free GNews API key from https://gnews.io (takes 30 seconds)

### Step 1 — Clone / Extract the Project
```bash
unzip stock-predictor.zip
cd stock-predictor
```

### Step 2 — Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### Step 3 — Install All Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables
```bash
cp .env.example .env
# Edit .env and add your free GNews API key:
#   GNEWS_API_KEY=your_key_here
```

### Step 5 — Train the Model (First Time)
```bash
python scripts/train_model.py
```
This will:
1. Download 6 months of historical data for the default 20 tickers
2. Fetch recent news and compute sentiment
3. Build features with normalization (StandardScaler)
4. Train an XGBoost classifier with 5-fold stratified cross-validation
5. Save the model + scaler to `models/`
6. Print cross-validation accuracy, precision, recall, F1

### Step 6 — Start the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 7 — Test It
```bash
# Single ticker
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL"]}'

# Multiple tickers (up to 50)
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]}'
```

### Step 8 — View Interactive Docs
Open http://localhost:8000/docs for the Swagger UI.

---

## API Endpoints

### POST /api/predict
**Request:**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "ticker": "AAPL",
      "current_price": 187.44,
      "signal": "Buy",
      "confidence": 0.73,
      "sentiment_score": 0.42,
      "latest_headline": "Apple announces new AI features...",
      "price_change_5d_pct": 2.1,
      "volume_ratio": 1.3,
      "timestamp": "2026-03-24T14:30:00Z"
    }
  ],
  "model_version": "xgb_v1",
  "generated_at": "2026-03-24T14:30:00Z"
}
```

### GET /api/health
Returns server status and model readiness.

### GET /api/model-info
Returns model metadata: features used, CV scores, training date.

---

## Feature Engineering Details

The model uses these features (all derivable from yfinance + GNews):

| Feature | Source | Description |
|---------|--------|-------------|
| `sentiment_compound` | GNews + VADER | Mean compound sentiment of last-24hr headlines |
| `sentiment_positive` | GNews + VADER | Mean positive sentiment score |
| `sentiment_negative` | GNews + VADER | Mean negative sentiment score |
| `sentiment_subjectivity` | GNews + TextBlob | Mean subjectivity of headlines |
| `news_volume` | GNews | Count of articles in last 24 hours |
| `price_change_1d` | yfinance | 1-day price return (%) |
| `price_change_5d` | yfinance | 5-day price return (%) |
| `price_change_20d` | yfinance | 20-day price return (%) |
| `volatility_10d` | yfinance | 10-day rolling std dev of returns |
| `volatility_20d` | yfinance | 20-day rolling std dev of returns |
| `volume_ratio` | yfinance | Today's volume / 20-day avg volume |
| `rsi_14` | yfinance | 14-day Relative Strength Index |
| `macd_signal` | yfinance | MACD minus Signal line |
| `bb_position` | yfinance | Price position within Bollinger Bands |
| `sma_20_ratio` | yfinance | Price / 20-day SMA |
| `sma_50_ratio` | yfinance | Price / 50-day SMA |

**Target Variable:** `1` (Buy) if next day's Close > today's Open, else `0` (Sell).

---

## Overfitting Prevention

1. **L1/L2 Regularization** — XGBoost `reg_alpha` and `reg_lambda` parameters
2. **Early Stopping** — Training halts when validation loss stops improving
3. **Max Depth Limiting** — Trees capped at depth 4
4. **Subsampling** — Both row and column subsampling at 80%
5. **Stratified K-Fold CV** — 5-fold cross-validation preserves class balance
6. **Feature Scaling** — StandardScaler normalization before training

---

## Productionization Checklist

- [x] Cross-validated model with saved scaler
- [x] API with CORS enabled for React frontend
- [x] Input validation (1-50 tickers, valid symbols)
- [x] Structured logging
- [x] Health check endpoint
- [x] Environment-based configuration
- [x] Requirements pinned in requirements.txt
- [x] Dockerfile included for container deployment
- [ ] Add Redis caching for repeated ticker lookups
- [ ] Add rate limiting middleware
- [ ] Add authentication layer
- [ ] Set up CI/CD pipeline

---

## Project Structure

```
stock-predictor/
├── app/
│   ├── __init__.py
│   └── main.py                 # FastAPI application & routes
├── config/
│   ├── __init__.py
│   └── settings.py             # Pydantic settings management
├── models/
│   └── (generated after training)
├── services/
│   ├── __init__.py
│   ├── news_service.py         # GNews API client
│   ├── price_service.py        # yfinance client
│   └── sentiment.py            # VADER + TextBlob scoring
├── utils/
│   ├── __init__.py
│   ├── feature_engine.py       # Feature engineering pipeline
│   ├── predictor.py            # Inference engine
│   ├── trainer.py              # Training with cross-validation
│   ├── validators.py           # Input validation
│   └── logger.py               # Logging setup
├── scripts/
│   └── train_model.py          # CLI training script
├── tests/
│   ├── __init__.py
│   └── test_api.py             # API integration tests
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── LICENSE
```
