"""Backfill Polygon daily bars for configured tickers."""

from app.config import (
    connect_db,
    insert_daily_bars,
    update_metadata,
)
from app.backfill.cli import compute_backfill_plan, parse_args
from app.polygon_trading_client import PolygonTradingClient
from app.symbols import DAILY_BAR_SYMBOLS

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
    print(f"Processing {ticker} from {start_date} to {end_date}")

    try:
        bars = client.get_daily_bars(ticker, start_date, end_date)
        if bars:
            rows_inserted = insert_daily_bars(conn, bars)
            update_metadata(conn, "daily", ticker, start_date, end_date, rows_inserted)
            return rows_inserted

        print(f"No bars found for {ticker} from {start_date} to {end_date}")
        update_metadata(conn, "daily", ticker, start_date, end_date, 0, status="completed")
        return 0
    except Exception as exc:
        print(f"Error processing {ticker} from {start_date} to {end_date}: {exc}")
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
    tickers = args.tickers or DAILY_BAR_SYMBOLS

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