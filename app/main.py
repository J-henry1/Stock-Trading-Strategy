"""
FastAPI Application — Stock Sentiment Predictor API

Endpoints:
  POST /api/predict    — Generate Buy/Sell signals for 1-50 tickers
  GET  /api/health     — Health check & model status
  GET  /api/model-info — Model metadata (features, CV scores, training date)

CORS is enabled for React frontend integration.
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from config.settings import settings
from utils.predictor import Predictor
from utils.validators import validate_tickers
from utils.logger import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------
# Global predictor instance (loaded once at startup)
# -----------------------------------------------------------------
predictor = Predictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup, clean up on shutdown."""
    logger.info("Starting Stock Sentiment Predictor API...")
    try:
        predictor.load()
        logger.info("Model loaded successfully — API is ready")
    except FileNotFoundError as e:
        logger.warning(
            f"Model not found: {e}. "
            "Run 'python scripts/train_model.py' to train first. "
            "API will start but predictions will fail until model is trained."
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    yield  # App runs here

    logger.info("Shutting down API...")


# -----------------------------------------------------------------
# App instance
# -----------------------------------------------------------------
app = FastAPI(
    title="Stock Sentiment Predictor API",
    description=(
        "Generates Buy/Sell signals by combining real-time news sentiment "
        "with historical stock price technical indicators. "
        "Powered by XGBoost with cross-validated training."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# -----------------------------------------------------------------
# CORS — allow React frontend to call this API
# -----------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------
class PredictRequest(BaseModel):
    """Request body for /api/predict."""
    tickers: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of stock ticker symbols (1-50). Example: ['AAPL', 'MSFT']",
        json_schema_extra={"examples": [["AAPL", "MSFT", "GOOGL"]]},
    )

    @field_validator("tickers")
    @classmethod
    def validate_ticker_list(cls, v):
        return validate_tickers(v)


class PredictionItem(BaseModel):
    """Single stock prediction result."""
    ticker: str
    current_price: Optional[float] = None
    signal: str  # "Buy", "Sell", or "Error"
    confidence: float = 0.0
    sentiment_score: float = 0.0
    latest_headline: str = ""
    price_change_5d_pct: float = 0.0
    volume_ratio: float = 0.0
    rsi_14: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""


class PredictResponse(BaseModel):
    """Response body for /api/predict."""
    predictions: List[PredictionItem]
    model_version: str
    generated_at: str


class HealthResponse(BaseModel):
    """Response body for /api/health."""
    status: str
    model_loaded: bool
    model_version: str
    uptime: str
    timestamp: str


# Track startup time
_startup_time = datetime.now(timezone.utc)


# -----------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------
@app.post(
    "/api/predict",
    response_model=PredictResponse,
    summary="Generate Buy/Sell signals",
    description=(
        "Accepts 1-50 stock tickers and returns Buy/Sell signals with confidence scores. "
        "Each prediction fetches fresh price data from Yahoo Finance and fresh news "
        "from GNews (last 24 hours) at the time of the request."
    ),
)
async def predict(request: PredictRequest):
    """
    Main prediction endpoint.

    For each ticker:
      1. Fetches the latest historical prices from yfinance
      2. Fetches the latest 24-hour news from GNews API
      3. Computes sentiment scores (VADER + TextBlob)
      4. Builds normalized feature vector
      5. Runs XGBoost inference → Buy/Sell + confidence
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Train the model first: "
                "python scripts/train_model.py"
            ),
        )

    logger.info(f"Prediction request: {request.tickers}")

    try:
        result = predictor.predict(request.tickers)
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health():
    """Returns server status and model readiness."""
    now = datetime.now(timezone.utc)
    uptime = now - _startup_time

    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded,
        model_version=settings.model_version,
        uptime=str(uptime),
        timestamp=now.isoformat(),
    )


@app.get(
    "/api/model-info",
    summary="Model metadata",
    description="Returns model metadata including features, CV scores, and training date.",
)
async def model_info():
    """Returns training metadata and model configuration."""
    metadata_path = settings.metadata_path

    if not os.path.exists(metadata_path):
        raise HTTPException(
            status_code=404,
            detail="Model metadata not found. Train the model first.",
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata
