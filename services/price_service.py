"""
Historical and current stock price service using yfinance.
Free, unlimited — scrapes Yahoo Finance.

Called every time the model is triggered to get the latest trading prices.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class PriceService:
    """Fetches historical and current stock price data from Yahoo Finance."""

    def __init__(self, history_days: int = None):
        self.history_days = history_days or settings.history_days

    def get_current_price(self, ticker: str) -> Optional[Dict]:
        """
        Get the most recent price data for a ticker.
        Returns dict with open, close, high, low, volume, or None on failure.
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")

            if hist.empty:
                logger.warning(f"No price data returned for {ticker}")
                return None

            latest = hist.iloc[-1]
            return {
                "ticker": ticker,
                "open": round(float(latest["Open"]), 2),
                "high": round(float(latest["High"]), 2),
                "low": round(float(latest["Low"]), 2),
                "close": round(float(latest["Close"]), 2),
                "volume": int(latest["Volume"]),
                "date": str(hist.index[-1].date()),
            }
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {e}")
            return None

    def get_historical_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for the configured number of days.
        Called fresh each time the model prediction is triggered.

        Returns a DataFrame with columns:
            Open, High, Low, Close, Volume, Date (index)
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.history_days)

            stock = yf.Ticker(ticker)
            hist = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )

            if hist.empty or len(hist) < 30:
                logger.warning(
                    f"Insufficient historical data for {ticker}: "
                    f"{len(hist)} rows (need >= 30)"
                )
                return None

            # Clean the dataframe
            hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            hist = hist.dropna()

            logger.info(
                f"Fetched {len(hist)} days of history for {ticker} "
                f"({hist.index[0].date()} to {hist.index[-1].date()})"
            )
            return hist

        except Exception as e:
            logger.error(f"Error fetching historical prices for {ticker}: {e}")
            return None

    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicator features from OHLCV data.

        Features computed:
            - price_change_1d, 5d, 20d  (percentage returns)
            - volatility_10d, 20d        (rolling std of returns)
            - volume_ratio               (current / 20-day average)
            - rsi_14                     (Relative Strength Index)
            - macd_signal                (MACD - Signal line)
            - bb_position                (position within Bollinger Bands)
            - sma_20_ratio, sma_50_ratio (price / SMA ratios)
        """
        data = df.copy()

        # --- Price Returns ---
        data["returns"] = data["Close"].pct_change()
        data["price_change_1d"] = data["returns"] * 100
        data["price_change_5d"] = data["Close"].pct_change(5) * 100
        data["price_change_20d"] = data["Close"].pct_change(20) * 100

        # --- Volatility ---
        data["volatility_10d"] = data["returns"].rolling(10).std() * 100
        data["volatility_20d"] = data["returns"].rolling(20).std() * 100

        # --- Volume Ratio ---
        data["volume_avg_20d"] = data["Volume"].rolling(20).mean()
        data["volume_ratio"] = data["Volume"] / data["volume_avg_20d"]

        # --- RSI (14-day) ---
        data["rsi_14"] = self._compute_rsi(data["Close"], window=14)

        # --- MACD ---
        ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        data["macd_signal"] = macd_line - signal_line

        # --- Bollinger Band Position ---
        sma_20 = data["Close"].rolling(20).mean()
        std_20 = data["Close"].rolling(20).std()
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        band_width = upper_band - lower_band
        # Position: 0 = at lower band, 1 = at upper band
        data["bb_position"] = np.where(
            band_width > 0,
            (data["Close"] - lower_band) / band_width,
            0.5
        )

        # --- SMA Ratios ---
        data["sma_20_ratio"] = data["Close"] / sma_20
        sma_50 = data["Close"].rolling(50).mean()
        data["sma_50_ratio"] = data["Close"] / sma_50

        # --- Target: 1 if next Close > current Open ---
        data["target"] = (data["Close"].shift(-1) > data["Open"]).astype(int)

        # Drop intermediate columns
        data = data.drop(columns=["returns", "volume_avg_20d"], errors="ignore")

        return data

    @staticmethod
    def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi
