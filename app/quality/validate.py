"""Validation entry points used by ingestion/backfill steps."""

from __future__ import annotations

from typing import Any, Dict, List

from app.config import ENABLE_DATA_QUALITY_CHECKS
from app.quality.schemas import daily_bars_schema, intraday_bars_schema, news_articles_schema


class DataQualityError(ValueError):
    """Raised when a data quality gate fails."""


def _require_pandas():
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Data quality checks are enabled but pandas is not installed. "
            "Install it (pip install pandas) or set ENABLE_DATA_QUALITY_CHECKS=0."
        ) from exc
    return pd


def maybe_validate_daily_bars(bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ENABLE_DATA_QUALITY_CHECKS:
        return bars
    return validate_daily_bars(bars)


def maybe_validate_intraday_bars(bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ENABLE_DATA_QUALITY_CHECKS:
        return bars
    return validate_intraday_bars(bars)


def maybe_validate_news_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ENABLE_DATA_QUALITY_CHECKS:
        return articles
    return validate_news_articles(articles)


def validate_daily_bars(bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not bars:
        return bars
    pd = _require_pandas()
    df = pd.DataFrame(bars)
    schema = daily_bars_schema()
    try:
        schema.validate(df, lazy=True)
    except Exception as exc:
        raise DataQualityError(f"Daily bars failed validation: {exc}") from exc
    return bars


def validate_intraday_bars(bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not bars:
        return bars
    pd = _require_pandas()
    df = pd.DataFrame(bars)
    schema = intraday_bars_schema()
    try:
        schema.validate(df, lazy=True)
    except Exception as exc:
        raise DataQualityError(f"Intraday bars failed validation: {exc}") from exc
    return bars


def validate_news_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not articles:
        return articles
    pd = _require_pandas()
    df = pd.DataFrame(articles)
    schema = news_articles_schema()
    try:
        schema.validate(df, lazy=True)
    except Exception as exc:
        raise DataQualityError(f"News articles failed validation: {exc}") from exc
    return articles


