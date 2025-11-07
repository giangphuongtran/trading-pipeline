# Polygon API Examples (Python SDK)

Below are minimal examples using the official `polygon-api-client`.

> Prereq: `pip install polygon-api-client`

## Setup

```python
from polygon import RESTClient
import os

API_KEY = os.getenv("POLYGON_API_KEY") or "YOUR_API_KEY"
client = RESTClient(API_KEY)
```

## 1) Daily Bars (range over dates)

Fetch daily aggregates for a symbol in ascending order.

```python
from datetime import date

symbol = "AAPL"
start = "2024-01-01"
end = "2024-01-10"

bars = []
for agg in client.list_aggs(
    symbol,
    1,               # multiplier
    "day",          # timespan
    start,
    end,
    adjusted=True,
    sort="asc",
    limit=50000,
):
    bars.append({
        "ticker": symbol,
        "date": date.fromtimestamp(agg.timestamp/1000).isoformat(),
        "open": agg.open,
        "high": agg.high,
        "low": agg.low,
        "close": agg.close,
        "volume": agg.volume,
        "transactions": getattr(agg, "transactions", None),
        "volume_weighted_avg_price": getattr(agg, "vwap", None),
    })

print(len(bars), "daily bars")
```

Sample response (single bar):

```json
{
  "ticker": "AAPL",
  "date": "2024-01-02",
  "open": 187.15,
  "high": 188.44,
  "low": 185.83,
  "close": 187.65,
  "volume": 58432412,
  "transactions": 0,
  "volume_weighted_avg_price": 187.06
}
```

## 2) Intraday 5-minute Bars (range fetch)

Fetch 5-minute aggregates over a date range in one call.

```python
symbol = "AAPL"
start = "2024-01-02"
end = "2024-01-03"

bars_5m = []
for agg in client.list_aggs(
    symbol,
    5,               # 5-minute bars
    "minute",
    start,
    end,
    adjusted=True,
    sort="asc",
    limit=50000,
):
    bars_5m.append({
        "ticker": symbol,
        "timestamp": agg.timestamp,  # ms since epoch
        "open": agg.open,
        "high": agg.high,
        "low": agg.low,
        "close": agg.close,
        "volume": agg.volume,
        "transactions": getattr(agg, "transactions", None),
        "volume_weighted_avg_price": getattr(agg, "vwap", None),
    })

print(len(bars_5m), "5m bars")
```

Sample response (single bar):

```json
{
  "ticker": "AAPL",
  "timestamp": 1704191100000,
  "open": 187.40,
  "high": 187.55,
  "low": 187.30,
  "close": 187.52,
  "volume": 312345,
  "transactions": 0,
  "volume_weighted_avg_price": 187.46
}
```

## 3) News with Insights (only)

Fetch news filtered by date range and only those with Polygon AI insights.

```python
symbol = "AAPL"
start = "2024-06-01"
end = "2024-06-07"

news_items = []
for n in client.list_ticker_news(
    ticker=symbol,
    published_utc_gte=start,
    published_utc_lte=end,
    order="asc",
    sort="published_utc",
    limit=100,
    only_with_insights=True,
):
    # insights shape may vary; guard with getattr
    insights = getattr(n, "insights", None) or {}
    sentiment = insights.get("sentiment") or {}
    news_items.append({
        "id": n.id,
        "ticker": symbol,
        "published_at": n.published_utc,
        "title": n.title,
        "description": n.description,
        "url": n.article_url,
        "author": n.author,
        "type": getattr(n, "type", "article"),
        "sentiment_score": sentiment.get("score"),
        "sentiment_label": sentiment.get("label"),
        "sentiment_reasoning": sentiment.get("reasoning"),
        "keywords": insights.get("keywords") or [],
        "tickers": n.tickers or [symbol]
    })

print(len(news_items), "news with insights")
```

Sample response (single item):

```json
{
  "id": "8ec638777ca03b553ae516761c2a22ba2fdd2f37befae3ab6fdab74e9e5193eb",
  "ticker": "AAPL",
  "published_at": "2024-06-24T18:33:53Z",
  "title": "Markets are underestimating Fed cuts: UBS",
  "description": "UBS analysts warn ...",
  "url": "https://uk.investing.com/news/stock-market-news/...",
  "author": "Sam Boughedda",
  "type": "article",
  "sentiment_score": 0.72,
  "sentiment_label": "positive",
  "sentiment_reasoning": "UBS analysts are providing a bullish outlook ...",
  "keywords": ["Federal Reserve", "interest rates", "economic data"],
  "tickers": ["UBS", "AAPL"]
}
```

## Notes
- For heavy ranges, handle pagination or narrow the window (e.g., 7-day chunks for news).
- Detect 429 rate limits and back off; the SDK raises on repeated 429s.
- Use `sort="asc"` for oldest-first backfills.
