"""Freshness checks for backfilled data."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional


def check_data_freshness(
    conn,
    data_type: str,
    ticker: str,
    expected_end_date: str,
    *,
    logger=None,
) -> bool:
    """
    Verify the latest record for a ticker reaches the expected end date.
    """
    table_by_type = {
        "daily": ("daily_bars", "date"),
        "intraday": ("intraday_bars", "timestamp"),
        "news": ("news_articles", "published_at"),
    }

    if data_type not in table_by_type:
        if logger:
            logger.warning("Freshness check skipped: unknown data_type=%s", data_type)
        return False

    table, column = table_by_type[data_type]
    cur = conn.cursor()
    try:
        cur.execute(
            f"SELECT MAX({column}) FROM {table} WHERE ticker = %s",
            (ticker,),
        )
        row = cur.fetchone()
    finally:
        cur.close()

    max_value = row[0] if row else None
    if max_value is None:
        if logger:
            logger.warning("Freshness check failed: no data for %s (%s)", ticker, data_type)
        return False

    if isinstance(max_value, datetime):
        max_date = max_value.date()
    elif isinstance(max_value, date):
        max_date = max_value
    else:
        max_date = date.fromisoformat(str(max_value))

    expected = date.fromisoformat(expected_end_date)
    is_fresh = max_date >= expected
    if logger:
        if is_fresh:
            logger.info("Freshness check OK for %s (%s): %s", ticker, data_type, max_date)
        else:
            logger.warning(
                "Freshness check lagging for %s (%s): latest=%s expected>=%s",
                ticker,
                data_type,
                max_date,
                expected,
            )
    return is_fresh
