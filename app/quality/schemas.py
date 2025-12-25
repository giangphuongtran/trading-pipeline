"""Pandera schemas for raw/ingested entities."""

from __future__ import annotations


def _require_pandera():
    """Import pandera lazily so quality checks can be disabled without the dep."""
    try:
        import pandera as pa
        from pandera import Check, Column
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Data quality checks are enabled but pandera is not installed. "
            "Install it (pip install pandera) or set ENABLE_DATA_QUALITY_CHECKS=0."
        ) from exc
    return pa, Column, Check


def daily_bars_schema():
    pa, Column, Check = _require_pandera()

    return pa.DataFrameSchema(
        {
            "ticker": Column(str, nullable=False),
            "date": Column(object, nullable=False),
            "open": Column(float, nullable=False, checks=Check.ge(0)),
            "high": Column(float, nullable=False, checks=Check.ge(0)),
            "low": Column(float, nullable=False, checks=Check.ge(0)),
            "close": Column(float, nullable=False, checks=Check.ge(0)),
            "volume": Column(object, nullable=True),
            "transactions": Column(object, nullable=True),
            "volume_weighted_avg_price": Column(object, nullable=True),
        },
        checks=[
            # high >= low
            Check(lambda df: (df["high"] >= df["low"]).all(), error="high must be >= low"),
            # high >= open/close and low <= open/close
            Check(
                lambda df: (df["high"] >= df[["open", "close"]].max(axis=1)).all(),
                error="high must be >= max(open, close)",
            ),
            Check(
                lambda df: (df["low"] <= df[["open", "close"]].min(axis=1)).all(),
                error="low must be <= min(open, close)",
            ),
        ],
        strict=False,  # allow extra fields from upstream
        coerce=False,
    )


def intraday_bars_schema():
    pa, Column, Check = _require_pandera()

    return pa.DataFrameSchema(
        {
            "ticker": Column(str, nullable=False),
            "timestamp": Column(object, nullable=False),
            "open": Column(float, nullable=False, checks=Check.ge(0)),
            "high": Column(float, nullable=False, checks=Check.ge(0)),
            "low": Column(float, nullable=False, checks=Check.ge(0)),
            "close": Column(float, nullable=False, checks=Check.ge(0)),
            "volume": Column(object, nullable=True),
            "transactions": Column(object, nullable=True),
            "volume_weighted_avg_price": Column(object, nullable=True),
        },
        checks=[
            Check(lambda df: (df["high"] >= df["low"]).all(), error="high must be >= low"),
            Check(
                lambda df: (df["high"] >= df[["open", "close"]].max(axis=1)).all(),
                error="high must be >= max(open, close)",
            ),
            Check(
                lambda df: (df["low"] <= df[["open", "close"]].min(axis=1)).all(),
                error="low must be <= min(open, close)",
            ),
        ],
        strict=False,
        coerce=False,
    )


def news_articles_schema():
    pa, Column, Check = _require_pandera()

    return pa.DataFrameSchema(
        {
            "id": Column(str, nullable=False),
            "ticker": Column(object, nullable=True),
            "published_at": Column(object, nullable=False),
            "title": Column(str, nullable=False),
            "description": Column(str, nullable=False),
            "url": Column(object, nullable=True),
            "author": Column(object, nullable=True),
            "type": Column(object, nullable=True),
            "sentiment_score": Column(object, nullable=True),
            "sentiment_label": Column(object, nullable=True),
            "sentiment_reasoning": Column(object, nullable=True),
            "keywords": Column(object, nullable=True),
            "tickers": Column(object, nullable=True),
        },
        checks=[
            Check(lambda df: (df["id"].astype(str).str.len() > 0).all(), error="id must be non-empty"),
        ],
        strict=False,
        coerce=False,
    )


