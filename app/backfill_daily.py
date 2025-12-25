"""Backfill Polygon daily bars for configured tickers.

This is the canonical implementation module used by Airflow and local runs:
`python -m app.backfill_daily`.
"""

from __future__ import annotations

from app.backfill.cli import compute_backfill_plan, parse_args
from app.config import connect_db, insert_daily_bars, update_metadata
from app.observability.logging import get_logger
from app.polygon_trading_client import PolygonTradingClient
from app.quality.validate import maybe_validate_daily_bars
from app.symbols import DAILY_BAR_SYMBOLS, MARKET_INDICES

logger = get_logger(__name__)


def backfill_daily_bars(
    client: PolygonTradingClient,
    conn,
    ticker: str,
    start_date: str,
    end_date: str,
) -> int:
    """
    Fetch daily bars from API and insert into database.

    Args:
        client: Polygon API client
        conn: Database connection
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Number of rows inserted
    """
    logger.info("Processing %s from %s to %s", ticker, start_date, end_date)

    try:
        bars = client.get_daily_bars(ticker, start_date, end_date)
        if bars:
            maybe_validate_daily_bars(bars)
            rows_inserted = insert_daily_bars(conn, bars)
            update_metadata(conn, "daily", ticker, start_date, end_date, rows_inserted)
            return rows_inserted

        logger.info("No bars found for %s from %s to %s", ticker, start_date, end_date)
        update_metadata(conn, "daily", ticker, start_date, end_date, 0, status="completed")
        return 0
    except Exception as exc:
        logger.exception("Error processing %s from %s to %s", ticker, start_date, end_date)
        update_metadata(
            conn,
            "daily",
            ticker,
            start_date,
            end_date,
            0,
            status="failed",
            error_message=str(exc),
        )
        return 0


def main() -> None:
    """Main entry point for daily bars backfill script."""
    args = parse_args("daily")
    # Include market indices (e.g., SPY) by default along with regular stocks
    default_tickers = DAILY_BAR_SYMBOLS + [MARKET_INDICES["US"]]
    tickers = args.tickers or default_tickers

    client = PolygonTradingClient()
    conn = connect_db(use_docker=args.use_docker_db)

    try:
        plans = compute_backfill_plan(
            conn,
            tickers=tickers,
            data_type="daily",
            mode=args.mode,
            start_date=args.start_date,
            end_date=args.end_date,
            lookback_days=args.lookback_days,
        )

        for plan in plans:
            backfill_daily_bars(
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


