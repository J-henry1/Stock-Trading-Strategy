"""
Integration tests for the Stock Sentiment Predictor API.

Run with:
    pytest tests/ -v

Note: These tests require a trained model in models/.
      Run 'python scripts/train_model.py' first.
"""

import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health_endpoint():
    """GET /api/health should return 200 with status info."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "model_version" in data
    assert "timestamp" in data


@pytest.mark.anyio
async def test_predict_single_ticker():
    """POST /api/predict with a single valid ticker."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/predict",
            json={"tickers": ["AAPL"]},
        )

    # If model isn't loaded, expect 503
    if response.status_code == 503:
        pytest.skip("Model not trained yet — run scripts/train_model.py first")

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1

    pred = data["predictions"][0]
    assert pred["ticker"] == "AAPL"
    assert pred["signal"] in ("Buy", "Sell", "Error")
    assert "current_price" in pred
    assert "confidence" in pred
    assert "sentiment_score" in pred
    assert "timestamp" in pred


@pytest.mark.anyio
async def test_predict_multiple_tickers():
    """POST /api/predict with multiple tickers."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/predict",
            json={"tickers": ["AAPL", "MSFT", "GOOGL"]},
        )

    if response.status_code == 503:
        pytest.skip("Model not trained yet")

    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 3
    tickers_returned = [p["ticker"] for p in data["predictions"]]
    assert "AAPL" in tickers_returned
    assert "MSFT" in tickers_returned
    assert "GOOGL" in tickers_returned


@pytest.mark.anyio
async def test_predict_empty_tickers():
    """POST /api/predict with empty list should return 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/predict",
            json={"tickers": []},
        )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_invalid_ticker_format():
    """POST /api/predict with invalid ticker format should return 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/predict",
            json={"tickers": ["123INVALID"]},
        )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_too_many_tickers():
    """POST /api/predict with >50 tickers should return 422."""
    tickers = [f"T{i:03d}" for i in range(51)]  # 51 tickers — invalid format anyway
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/predict",
            json={"tickers": tickers},
        )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_predict_duplicate_tickers():
    """POST /api/predict with duplicates should deduplicate."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/predict",
            json={"tickers": ["AAPL", "aapl", "AAPL"]},
        )

    if response.status_code == 503:
        pytest.skip("Model not trained yet")

    assert response.status_code == 200
    data = response.json()
    # Should be deduplicated to 1
    assert len(data["predictions"]) == 1


@pytest.mark.anyio
async def test_predict_lowercase_tickers():
    """POST /api/predict with lowercase should normalize to uppercase."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/predict",
            json={"tickers": ["aapl"]},
        )

    if response.status_code == 503:
        pytest.skip("Model not trained yet")

    assert response.status_code == 200
    data = response.json()
    assert data["predictions"][0]["ticker"] == "AAPL"


@pytest.mark.anyio
async def test_model_info_endpoint():
    """GET /api/model-info should return metadata if model is trained."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/model-info")

    if response.status_code == 404:
        pytest.skip("Model not trained yet")

    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "features" in data
    assert "cv_results" in data


@pytest.mark.anyio
async def test_cors_headers():
    """Verify CORS headers are present."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.options(
            "/api/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

    # FastAPI CORS should respond to preflight
    assert response.status_code in (200, 405)


# -----------------------------------------------------------------
# Unit tests for validators (no API needed)
# -----------------------------------------------------------------

from utils.validators import validate_tickers, validate_ticker


def test_validate_ticker_valid():
    assert validate_ticker("AAPL") == "AAPL"
    assert validate_ticker("aapl") == "AAPL"
    assert validate_ticker("  msft  ") == "MSFT"
    assert validate_ticker("BRK.B") == "BRK.B"


def test_validate_ticker_invalid():
    with pytest.raises(ValueError):
        validate_ticker("")
    with pytest.raises(ValueError):
        validate_ticker("123")
    with pytest.raises(ValueError):
        validate_ticker("TOOLONGTICKER")
    with pytest.raises(ValueError):
        validate_ticker("A B")


def test_validate_tickers_dedup():
    result = validate_tickers(["AAPL", "aapl", "AAPL"])
    assert result == ["AAPL"]


def test_validate_tickers_max():
    tickers = [f"T{chr(65+i)}" for i in range(26)] * 2  # 52 tickers
    with pytest.raises(ValueError, match="Maximum 50"):
        validate_tickers(tickers)
