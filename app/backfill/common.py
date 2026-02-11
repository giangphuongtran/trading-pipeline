"""Shared backfill execution helpers."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from app.config import update_metadata
from app.quality.freshness import check_data_freshness


def run_backfill(
    *,
    client,
    conn,
    ticker: str,
    start_date: str,
    end_date: str,
    data_type: str,
    fetch_fn: Callable[..., list[dict]],
    insert_fn: Callable[..., int],
    validate_fn: Optional[Callable[[list[dict]], Iterable[dict]]] = None,
    logger=None,
) -> int:
    """
    Execute a backfill for a single ticker/date range and record metadata.
    """
    if logger:
        logger.info("Processing %s from %s to %s", ticker, start_date, end_date)

    try:
        rows = fetch_fn(ticker, start_date, end_date)
        if rows:
            if validate_fn is not None:
                validate_fn(rows)
            rows_inserted = insert_fn(conn, rows)
            update_metadata(conn, data_type, ticker, start_date, end_date, rows_inserted)
            check_data_freshness(conn, data_type, ticker, end_date, logger=logger)
            return rows_inserted

        if logger:
            logger.info("No data found for %s from %s to %s", ticker, start_date, end_date)
        update_metadata(conn, data_type, ticker, start_date, end_date, 0, status="completed")
        check_data_freshness(conn, data_type, ticker, end_date, logger=logger)
        return 0
    except Exception as exc:
        if logger:
            logger.exception("Error processing %s from %s to %s", ticker, start_date, end_date)
        update_metadata(
            conn,
            data_type,
            ticker,
            start_date,
            end_date,
            0,
            status="failed",
            error_message=str(exc),
        )
        return 0
