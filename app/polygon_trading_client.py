"""Minimal Polygon.io trading client with simple rate-limit pacing & robust retries."""

from __future__ import annotations

import os
import ssl
import time
from datetime import datetime
from typing import Dict, List, Optional

from polygon import RESTClient

# requests + urllib3 errors (both layers may surface)
from requests.exceptions import (
    HTTPError,
    RetryError as RequestsRetryError,
    SSLError as RequestsSSLError,
    ConnectionError as RequestsConnectionError,
    ReadTimeout,
    ConnectTimeout,
)
try:
    from urllib3.exceptions import (
        SSLError as Urllib3SSLError,
        MaxRetryError as Urllib3MaxRetryError,
        ProtocolError as Urllib3ProtocolError,
        NewConnectionError as Urllib3NewConnectionError,
    )
except Exception:
    # Fallback if urllib3 exceptions can’t be imported (unlikely)
    Urllib3SSLError = Urllib3MaxRetryError = Urllib3ProtocolError = Urllib3NewConnectionError = tuple()


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded too many times."""


# A unified tuple of retryable network-ish errors
RETRYABLE_NET_ERRORS = (
    RequestsSSLError,
    RequestsRetryError,
    RequestsConnectionError,
    ReadTimeout,
    ConnectTimeout,
    Urllib3SSLError,
    Urllib3MaxRetryError,
    Urllib3ProtocolError,
    Urllib3NewConnectionError,
    ssl.SSLError,
    ConnectionResetError,
    TimeoutError,
)


class PolygonTradingClient:
    """
    Minimal Polygon client:
      - Paces requests by a fixed delay (free tier ≈ 5 req/min → ~13s spacing).
      - Retries on common TLS/network errors (requests *and* urllib3 layers).
      - Retries 429 with backoff.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        rate_limit_delay: float = 20.0,
        max_attempts: int = 5,
    ):
        """
        Initialize Polygon API client with rate limiting and retry logic.
        
        Args:
            api_key: Polygon API key (defaults to POLYGON_API_KEY env var)
            rate_limit_delay: Minimum seconds between API calls (default: 20s for free tier)
            max_attempts: Maximum retry attempts for failed requests (default: 5)
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY is not set (env) and no api_key was provided.")

        self.client = RESTClient(self.api_key)
        self.rate_limit_delay = float(rate_limit_delay)
        self.max_attempts = int(max_attempts)
        self._last_call = 0.0

    # ---------- helpers ----------

    def _pace(self) -> None:
        """Enforce minimum delay between API calls to respect rate limits."""
        elapsed = time.time() - self._last_call
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_call = time.time()

    @staticmethod
    def _retry_backoff(attempt: int, base: float) -> float:
        """Calculate exponential backoff delay for retries."""
        return base if attempt <= 2 else base * (2 ** (attempt - 2))

    def _handle_api_error(self, e: Exception, label: str, ticker: str, attempt: int) -> tuple[bool, float]:
        """
        Shared error handling for API calls. Returns (should_retry, delay_seconds).
        
        Args:
            e: Exception that occurred
            label: Label for logging (e.g., "daily", "news")
            ticker: Ticker symbol (or "ALL" for news)
            attempt: Current attempt number
            
        Returns:
            Tuple of (should_retry, delay_seconds). If should_retry is False, raise the exception.
        """
        if isinstance(e, HTTPError):
            status = getattr(e.response, "status_code", None)
            if status == 429:
                if attempt > self.max_attempts:
                    raise RateLimitError(f"[{label}] 429 after {self.max_attempts} attempts for {ticker}") from e
                retry_after = e.response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self._retry_backoff(attempt, self.rate_limit_delay)
                print(f"[{label}] 429 for {ticker}; sleeping {delay:.1f}s before retry #{attempt}")
                return True, delay
            if status == 404:
                raise ValueError(f"[{label}] Ticker {ticker} not found") from e
            # Other HTTP errors - don't retry
            raise
        elif isinstance(e, RETRYABLE_NET_ERRORS):
            if attempt > self.max_attempts:
                raise
            delay = self._retry_backoff(attempt, self.rate_limit_delay)
            print(f"[{label}] Network/TLS error for {ticker}; sleeping {delay:.1f}s before retry #{attempt} ({e})")
            return True, delay
        # Unknown error - don't retry
        raise

    def _retry_api_call(self, callable_func, label: str, ticker: str):
        """
        Generic retry wrapper for API calls with error handling.
        
        Args:
            callable_func: Function to call (should return the result)
            label: Label for logging (e.g., "daily", "news")
            ticker: Ticker symbol for logging
            
        Returns:
            Result from callable_func
            
        Raises:
            RateLimitError, ValueError, or other exceptions from _handle_api_error
        """
        attempt = 1
        while True:
            try:
                self._pace()
                return callable_func()
            except (HTTPError, *RETRYABLE_NET_ERRORS) as e:
                should_retry, delay = self._handle_api_error(e, label, ticker, attempt)
                if not should_retry:
                    raise
                attempt += 1
                time.sleep(delay)
                continue

    # ---------- public API ----------

    def get_daily_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        *,
        multiplier: int = 1,
        timespan: str = "day",
    ) -> List[Dict]:
        """
        Fetch daily OHLCV bars from Polygon API.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            multiplier: Bar size multiplier (default: 1)
            timespan: Time unit (default: "day")
            
        Returns:
            List of dicts with ticker, date, timestamp, open, high, low, close, volume, transactions, vwap
        """
        return self._fetch_aggs(
            label="daily",
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            multiplier=multiplier,
            timespan=timespan,
            include_date_field=True,
        )

    def get_intraday_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        *,
        multiplier: int = 5,
        timespan: str = "minute",
    ) -> List[Dict]:
        """
        Fetch intraday OHLCV bars from Polygon API.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            multiplier: Bar size in minutes (default: 5)
            timespan: Time unit (default: "minute")
            
        Returns:
            List of dicts with ticker, timestamp, open, high, low, close, volume, transactions, vwap
        """
        return self._fetch_aggs(
            label="intraday",
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            multiplier=multiplier,
            timespan=timespan,
            include_date_field=False,
        )

    def get_news(
        self,
        ticker: Optional[str] = None,
        *,
        limit: int = 100,
        order: str = "desc",
        published_utc_gte: Optional[str] = None,
        published_utc_lte: Optional[str] = None,
        only_with_insights: bool = True,
    ) -> List[Dict]:
        """
        Fetch news articles from Polygon API.
        
        Uses shared error handling via `_handle_api_error()` for consistency with other API methods.
        
        Args:
            ticker: Stock symbol (None for all tickers)
            limit: Max articles to return (default: 100)
            order: Sort order "asc" or "desc" (default: "desc")
            published_utc_gte: Filter articles published on/after this date (YYYY-MM-DD)
            published_utc_lte: Filter articles published on/before this date (YYYY-MM-DD)
            only_with_insights: Only return articles with sentiment insights (default: True)
            
        Returns:
            List of dicts with article metadata and sentiment scores
        """
        ticker_label = ticker or "ALL"
        
        def _fetch_news():
            """Inner function to fetch and process news articles."""
            params = {
                "ticker": ticker,
                "limit": limit,
                "order": order,
                "published_utc_gte": published_utc_gte,
                "published_utc_lte": published_utc_lte,
            }
            # filter out None
            params = {k: v for k, v in params.items() if v is not None}
            
            results = list(self.client.list_ticker_news(**params))
            items: List[Dict] = []
            
            for news in results:
                insights = getattr(news, "insights", None) or []
                if only_with_insights and not insights:
                    continue

                first = insights[0] if insights else None
                published_at = getattr(news, "published_utc", None)
                if hasattr(published_at, "isoformat"):
                    published_at = published_at.isoformat()

                items.append({
                    "id": getattr(news, "id", None),
                    "ticker": ticker,
                    "published_at": published_at,
                    "title": getattr(news, "title", "") or "",
                    "description": getattr(news, "description", "") or "",
                    "url": getattr(news, "article_url", "") or "",
                    "author": getattr(news, "author", "") or "",
                    "type": "article",
                    "sentiment_score": getattr(first, "score", None) if first else None,
                    "sentiment_label": getattr(first, "sentiment", None) if first else None,
                    "sentiment_reasoning": getattr(first, "sentiment_reasoning", None) if first else None,
                    "tickers": getattr(news, "tickers", []) or ([ticker] if ticker else []),
                    "keywords": getattr(news, "keywords", []) or [],
                })
            
            print(f"Fetched {len(items)} news items for {ticker_label}")
            return items
        
        return self._retry_api_call(_fetch_news, "news", ticker_label)

    # ---------- internal fetcher for aggs ----------

    def _fetch_aggs(
        self,
        *,
        label: str,
        ticker: str,
        start_date: str,
        end_date: str,
        multiplier: int,
        timespan: str,
        include_date_field: bool,
    ) -> List[Dict]:
        """
        Internal method to fetch aggregated bars from Polygon API with retry logic.
        
        Handles rate limiting, network errors, and 429 responses with exponential backoff.
        """
        def _fetch_bars():
            """Inner function to fetch and process aggregated bars."""
            it = self.client.list_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                adjusted=True,
                sort="asc",
                limit=50_000,
            )
            rows: List[Dict] = []
            for agg in it:
                ts = datetime.fromtimestamp(agg.timestamp / 1000.0)
                row = {
                    "ticker": ticker,
                    "timestamp": ts,
                    "open": agg.open,
                    "high": agg.high,
                    "low": agg.low,
                    "close": agg.close,
                    "volume": agg.volume,
                    "transactions": getattr(agg, "transactions", None),
                    "volume_weighted_avg_price": getattr(agg, "vwap", None),
                }
                if include_date_field:
                    row["date"] = ts.date()
                rows.append(row)
            
            print(f"Fetched {len(rows)} {label} bars for {ticker}")
            return rows
        
        return self._retry_api_call(_fetch_bars, label, ticker)
