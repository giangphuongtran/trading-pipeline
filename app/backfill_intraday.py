"""Backfill Polygon intraday bars for configured tickers.

This is the canonical implementation module used by Airflow and local runs:
`python -m app.backfill_intraday`.
"""

from __future__ import annotations

from app.backfill.cli import compute_backfill_plan, parse_args
from app.backfill.common import run_backfill
from app.config import connect_db, insert_intraday_bars
from app.observability.logging import get_logger
from app.polygon_trading_client import PolygonTradingClient
from app.quality.validate import maybe_validate_intraday_bars
from app.symbols import INTRADAY_BAR_SYMBOLS

logger = get_logger(__name__)


def backfill_intraday_bars(
    client: PolygonTradingClient,
    conn,
    ticker: str,
    start_date: str,
    end_date: str,
) -> int:
    """
    Fetch intraday bars from API and insert into database.

    Args:
        client: Polygon API client
        conn: Database connection
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Number of rows inserted
    """
    return run_backfill(
        client=client,
        conn=conn,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        data_type="intraday",
        fetch_fn=client.get_intraday_bars,
        insert_fn=insert_intraday_bars,
        validate_fn=maybe_validate_intraday_bars,
        logger=logger,
    )


def main() -> None:
    """Main entry point for intraday bars backfill script."""
    args = parse_args("intraday")
    tickers = args.tickers or INTRADAY_BAR_SYMBOLS

    client = PolygonTradingClient()
    conn = connect_db(use_docker=args.use_docker_db)

    try:
        plans = compute_backfill_plan(
            conn,
            tickers=tickers,
            data_type="intraday",
            mode=args.mode,
            start_date=args.start_date,
            end_date=args.end_date,
            lookback_days=args.lookback_days,
        )

        for plan in plans:
            backfill_intraday_bars(
                client,
                conn,
                plan.ticker,
                plan.start_date,
                plan.end_date,
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()

