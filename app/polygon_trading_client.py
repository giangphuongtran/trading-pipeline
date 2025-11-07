"""Polygon.io API client using official polygon-api-client library."""

import time
from datetime import datetime
from typing import Dict, List, Optional

from requests.exceptions import HTTPError
from polygon import RESTClient

from .config import POLYGON_API_KEY

class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""
    pass

class PolygonTradingClient:
    """Client for fetching trading data from Polygon.io."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_attempts: int = 3,
        rate_limit_delay: float = 13.0,
    ):
        """Initialize the PolygonTradingClient."""
        self.api_key = api_key or POLYGON_API_KEY
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY is not set in the environment variables.")

        self.client = RESTClient(self.api_key, timeout=60, connect_timeout=10)

        # Sleep between calls (13s for 5 calls/min on free tier)
        self.rate_limit_delay = rate_limit_delay
        self.last_call_time = 0
        self.max_attempts = max_attempts

    def _sleep_if_needed(self):
        """Sleep if needed to respect rate limits."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_call_time = time.time()

    def get_daily_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        timespan: str = "day",
        multiplier: int = 1
    ) -> List[Dict]:
        """
        Fetch daily bars for a given ticker within a date range.
        Uses Polygon's official API client library: client.list_aggs()
        
        Args:
            ticker: The ticker symbol to fetch data for (e.g., "AAPL")
            start_date: The start date of the date range (YYYY-MM-DD)
            end_date: The end date of the date range (YYYY-MM-DD)
            timespan: The time span of the bars (day, week, month)
            multiplier: The multiplier for the timespan (1, 2, 5, 10, 15, 30)

        Returns:
            List of daily bars
        """

        self._sleep_if_needed()

        bars = []
        attempt = 1

        while True:
            try:
                for agg in self.client.list_aggs(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=start_date,
                    to=end_date,
                    adjusted=True,
                    sort="asc",
                    limit=50000
                ):
                    # Convert timestamp from milliseconds to datetime
                    timestamp = datetime.fromtimestamp(agg.timestamp / 1000)

                    bars.append({
                        "ticker": ticker,
                        "date": timestamp.date(),
                        "timestamp": timestamp,
                        "open": agg.open,
                        "high": agg.high,
                        "low": agg.low,
                        "close": agg.close,
                        "volume": agg.volume,
                        "transactions": agg.transactions,
                        "volume_weighted_avg_price": agg.vwap
                    })

                print(f"Fetched {len(bars)} daily bars for {ticker}")
                return bars
            
            except HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status == 429:
                    retry_after = e.response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else self.rate_limit_delay
                    attempt += 1
                    if attempt > self.max_attempts:
                        raise RateLimitError(f"429 after {self.max_attempts} attempts for {ticker}") from e
                    time.sleep(delay)
                    continue
                if status == 404:
                    raise ValueError(f"Ticker {ticker} not found") from e
                raise
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                print(error_msg)
                return []

    def get_intraday_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        multiplier: int = 5,
        timespan: str = "minute"
    ) -> List[Dict]:
        """
        Fetch intraday bars for a given ticker within a date range.
        Uses Polygon's official API client library: client.list_aggs()
        
        Args:
            ticker: The ticker symbol to fetch data for (e.g., "AAPL")
            start_date: The start date of the date range (YYYY-MM-DD)
            end_date: The end date of the date range (YYYY-MM-DD)
            multiplier: The multiplier for the timespan (5, 15, 30, 60)
            timespan: The time span of the bars (minute, hour, day)

        Returns:
            List of intraday bars
        """

        self._sleep_if_needed()

        bars = []
        attempt = 1

        while True:
            try:
                for agg in self.client.list_aggs(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=start_date,
                    to=end_date,
                    adjusted=True,
                    sort="asc",
                    limit=50000
                ):
                    # Convert timestamp from milliseconds to datetime
                    timestamp = datetime.fromtimestamp(agg.timestamp / 1000)
                    
                    bars.append({
                        "ticker": ticker,
                        "timestamp": timestamp,
                        "open": agg.open,
                        "high": agg.high,
                        "low": agg.low,
                        "close": agg.close,
                        "volume": agg.volume,
                        "transactions": agg.transactions,
                        "volume_weighted_avg_price": agg.vwap
                    })
                
                print(f"Fetched {len(bars)} intraday bars for {ticker}")
                return bars

            except HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status == 429:
                    retry_after = e.response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else self.rate_limit_delay
                    attempt += 1
                    if attempt > self.max_attempts:
                        raise RateLimitError(f"429 after {self.max_attempts} attempts for {ticker}") from e
                    time.sleep(delay)
                    continue
                if status == 404:
                    raise ValueError(f"Ticker {ticker} not found") from e
                raise
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                print(error_msg)
                return []

    def get_news(
        self,
        ticker: str = None,
        limit: int = 100,
        order: str = "desc",
        published_utc_gte: str = None,
        published_utc_lte: str = None,
        only_with_insights: bool = True
    ) -> List[Dict]:
        """
        Fetch news for a given ticker.
        Uses Polygon's official API client library: client.list_news()
        
        Args:
            ticker: The ticker symbol to fetch news for (e.g., "AAPL")
            limit: The number of news items to fetch
            order: The order of the news items (desc, asc)
            published_utc_gte: The start date of the news items (YYYY-MM-DD)
            published_utc_lte: The end date of the news items (YYYY-MM-DD)
            only_with_insights: Whether to only fetch news items with insights
        
        Returns:
            List of news items
        """

        self._sleep_if_needed()

        items: List[Dict] = []
        attempt = 1

        while True:
            try:
                params = {
                    "limit": limit,
                    "order": order,
                    "sort": "published_utc",
                    "published_utc_gte": published_utc_gte,
                    "published_utc_lte": published_utc_lte,
                    "only_with_insights": only_with_insights
                }

                for item in self.client.list_news(**params):
                    # Skip news without insights
                    if hasattr(item, "insights") and item.insights:
                        # Get first insight (usually for the main ticker)
                        first_insight = item.insights[0]
                        sentiment_label = getattr(first_insight, "sentiment", None)
                        sentiment_reasoning = getattr(first_insight, "sentiment_reasoning", None)
                        sentiment_score = getattr(first_insight, "score", None)

                        items.append({
                            "id": item.id,
                            "ticker": ticker,
                            "published_at": item.published_utc,
                            "title": getattr(item, "title", ""),
                            "description": getattr(item, "description", ""),
                            "url": getattr(item, "article_url", ""),
                            "author": getattr(item, "author", ""),
                            "type": "article",
                            "sentiment_score": sentiment_score,
                            "sentiment_label": sentiment_label,
                            "sentiment_reasoning": sentiment_reasoning,
                            "tickers": getattr(item, "tickers", [ticker] if ticker else []),
                            "keywords": getattr(item, "keywords", []),
                        })
                
                print(f"Fetched {len(items)} news items for {ticker}")
                return items

            except HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status == 429:
                    retry_after = e.response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else self.rate_limit_delay
                    attempt += 1
                    if attempt > self.max_attempts:
                        raise RateLimitError(f"429 after {self.max_attempts} attempts for {ticker}") from e
                    time.sleep(delay)
                    continue
                if status == 404:
                    raise ValueError(f"Ticker {ticker} not found") from e
                raise
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                print(error_msg)
                return []