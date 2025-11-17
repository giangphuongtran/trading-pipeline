"""Shared CLI utilities for backfill scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Tuple

from app.config import get_last_fetch_date


@dataclass
class BackfillConfig:
    """Represents a resolved backfill configuration for a ticker."""

    ticker: str
    start_date: str
    end_date: str


def build_arg_parser(dataset: str) -> argparse.ArgumentParser:
    """
    Build CLI argument parser for backfill scripts.
    
    Args:
        dataset: Dataset name ("daily", "intraday", "news")
        
    Returns:
        Configured ArgumentParser instance
    """

    dataset_title = dataset.capitalize()
    parser = argparse.ArgumentParser(
        description=f"Backfill {dataset_title} data from Polygon.io",
    )

    parser.add_argument(
        "--mode",
        choices=("resume", "full"),
        default="resume",
        help="resume = continue from metadata, full = explicit start/end dates",
    )

    parser.add_argument(
        "--start-date",
        help="YYYY-MM-DD start date (required with --mode full unless metadata exists)",
    )

    parser.add_argument(
        "--end-date",
        help="YYYY-MM-DD end date (defaults to yesterday)",
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=730,
        help="Fallback lookback window used when metadata is absent (resume mode)",
    )

    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Optional subset of tickers to backfill",
    )

    parser.add_argument(
        "--use-docker-db",
        action="store_true",
        help="Connect using DATABASE_URL_DOCKER (inside Docker network)",
    )

    return parser


def parse_args(dataset: str) -> argparse.Namespace:
    """
    Parse command-line arguments for backfill script.
    
    Args:
        dataset: Dataset name ("daily", "intraday", "news")
        
    Returns:
        Parsed arguments namespace
    """
    parser = build_arg_parser(dataset)
    return parser.parse_args()


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse YYYY-MM-DD date string to date object."""
    if not date_str:
        return None
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def resolve_date_range(
    conn,
    *,
    ticker: str,
    data_type: str,
    mode: str,
    explicit_start: Optional[str],
    explicit_end: Optional[str],
    lookback_days: int,
) -> Tuple[str, str]:
    """
    Determine date range for backfill based on mode and metadata.
    
    In "resume" mode: continues from last fetch date or uses lookback window.
    In "full" mode: uses explicit start/end dates.
    
    Args:
        conn: Database connection
        ticker: Stock symbol
        data_type: Type of data ("daily", "intraday", "news")
        mode: "resume" or "full"
        explicit_start: Explicit start date (YYYY-MM-DD) for "full" mode
        explicit_end: Explicit end date (YYYY-MM-DD), defaults to yesterday
        lookback_days: Days to look back if no metadata exists (resume mode)
        
    Returns:
        Tuple of (start_date, end_date) as ISO format strings
    """

    assert mode in {"resume", "full"}, "mode must be 'resume' or 'full'"

    today = date.today()
    default_end = today - timedelta(days=1)
    end_date = _parse_date(explicit_end) or default_end

    if end_date >= today:
        end_date = default_end

    if mode == "full":
        if explicit_start:
            start_date = _parse_date(explicit_start)
        else:
            raise ValueError("--start-date is required when --mode full")
        return start_date.isoformat(), end_date.isoformat()

    last_fetch = get_last_fetch_date(conn, ticker, data_type)

    if last_fetch:
        start_date = last_fetch + timedelta(days=1)
    else:
        start_date = end_date - timedelta(days=lookback_days)

    if start_date > end_date:
        start_date = end_date

    return start_date.isoformat(), end_date.isoformat()


def compute_backfill_plan(
    conn,
    *,
    tickers: Iterable[str],
    data_type: str,
    mode: str,
    start_date: Optional[str],
    end_date: Optional[str],
    lookback_days: int,
) -> Iterable[BackfillConfig]:
    """
    Generate backfill plan with date ranges and chunking for each ticker.
    
    Chunks large date ranges for API efficiency (e.g., news in 7-day chunks).
    In resume mode, limits initial chunk size to avoid long-running jobs.
    
    Args:
        conn: Database connection
        tickers: Iterable of stock symbols
        data_type: Type of data ("daily", "intraday", "news")
        mode: "resume" or "full"
        start_date: Optional explicit start date
        end_date: Optional explicit end date
        lookback_days: Fallback lookback window
        
    Yields:
        BackfillConfig instances for each ticker/date chunk
    """

    resume_chunk_days = {
        "daily": 30,
        "intraday": 30,
        "news": 7,
    }
    api_chunk_days = {
        "news": 7,
    }

    for ticker in tickers:
        start, end = resolve_date_range(
            conn,
            ticker=ticker,
            data_type=data_type,
            mode=mode,
            explicit_start=start_date,
            explicit_end=end_date,
            lookback_days=lookback_days,
        )
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()

        resume_chunk = resume_chunk_days.get(data_type)
        if mode == "resume" and resume_chunk is not None:
            max_chunk_end = start_dt + timedelta(days=resume_chunk - 1)
            if max_chunk_end < end_dt:
                end_dt = max_chunk_end

        api_chunk = api_chunk_days.get(data_type)
        if api_chunk:
            window = timedelta(days=api_chunk - 1)
            current = start_dt
            while current <= end_dt:
                chunk_end = min(current + window, end_dt)
                yield BackfillConfig(
                    ticker=ticker,
                    start_date=current.isoformat(),
                    end_date=chunk_end.isoformat(),
                )
                current = chunk_end + timedelta(days=1)
        else:
            yield BackfillConfig(
                ticker=ticker,
                start_date=start_dt.isoformat(),
                end_date=end_dt.isoformat(),
            )

