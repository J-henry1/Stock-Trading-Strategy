"""
Input validation for ticker symbols and request payloads.
"""

import re
from typing import List


# Maximum tickers per request
MAX_TICKERS = 50

# Valid US stock ticker pattern: 1-5 uppercase letters, optionally with a dot (e.g., BRK.B)
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")


def validate_ticker(ticker: str) -> str:
    """
    Validate and normalize a single ticker symbol.
    Returns the normalized ticker or raises ValueError.
    """
    cleaned = ticker.strip().upper()

    if not cleaned:
        raise ValueError("Ticker symbol cannot be empty")

    if not TICKER_PATTERN.match(cleaned):
        raise ValueError(
            f"Invalid ticker symbol: '{ticker}'. "
            f"Must be 1-5 uppercase letters (e.g., AAPL, MSFT, BRK.B)"
        )

    return cleaned


def validate_tickers(tickers: List[str]) -> List[str]:
    """
    Validate a list of ticker symbols.
    Returns list of normalized tickers or raises ValueError.
    """
    if not tickers:
        raise ValueError("At least one ticker symbol is required")

    if len(tickers) > MAX_TICKERS:
        raise ValueError(
            f"Maximum {MAX_TICKERS} tickers per request. Got {len(tickers)}."
        )

    validated = []
    errors = []

    for t in tickers:
        try:
            validated.append(validate_ticker(t))
        except ValueError as e:
            errors.append(str(e))

    if errors:
        raise ValueError(f"Invalid tickers: {'; '.join(errors)}")

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for t in validated:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique
