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
    """Construct an argument parser with shared backfill options."""

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
        default=30,
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
    """Parse CLI arguments for a given dataset."""
    parser = build_arg_parser(dataset)
    return parser.parse_args()


def _parse_date(date_str: Optional[str]) -> Optional[date]:
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
    """Derive the start/end window for a backfill run in ISO format."""

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
    """Yield `BackfillConfig` instances for each ticker."""

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
        yield BackfillConfig(ticker=ticker, start_date=start, end_date=end)

