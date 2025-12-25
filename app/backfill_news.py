"""Backfill Polygon news articles for configured tickers.

This is the canonical implementation module used by Airflow and local runs:
`python -m app.backfill_news`.
"""

from __future__ import annotations

from app.backfill.cli import compute_backfill_plan, parse_args
from app.config import connect_db, insert_news_articles, update_metadata
from app.observability.logging import get_logger
from app.polygon_trading_client import PolygonTradingClient
from app.quality.validate import maybe_validate_news_articles
from app.symbols import NEWS_SYMBOLS

logger = get_logger(__name__)


def backfill_news_articles(
    client: PolygonTradingClient,
    conn,
    ticker: str,
    start_date: str,
    end_date: str,
) -> int:
    """
    Fetch news articles from API and insert into database.

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
        articles = client.get_news(
            ticker=ticker,
            published_utc_gte=start_date,
            published_utc_lte=end_date,
        )
        if articles:
            maybe_validate_news_articles(articles)
            rows_inserted = insert_news_articles(conn, articles)
            update_metadata(conn, "news", ticker, start_date, end_date, rows_inserted)
            return rows_inserted

        logger.info("No news found for %s from %s to %s", ticker, start_date, end_date)
        update_metadata(conn, "news", ticker, start_date, end_date, 0, status="completed")
        return 0
    except Exception as exc:
        logger.exception("Error processing %s from %s to %s", ticker, start_date, end_date)
        update_metadata(
            conn,
            "news",
            ticker,
            start_date,
            end_date,
            0,
            status="failed",
            error_message=str(exc),
        )
        return 0


def main() -> None:
    """Main entry point for news articles backfill script."""
    args = parse_args("news")
    tickers = args.tickers or NEWS_SYMBOLS

    client = PolygonTradingClient()
    conn = connect_db(use_docker=args.use_docker_db)

    try:
        plans = compute_backfill_plan(
            conn,
            tickers=tickers,
            data_type="news",
            mode=args.mode,
            start_date=args.start_date,
            end_date=args.end_date,
            lookback_days=args.lookback_days,
        )

        for plan in plans:
            backfill_news_articles(
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


