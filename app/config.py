import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # Fallback to default .env resolution (current working directory / env vars)
    load_dotenv()

# ---------------------------------------------------------------------------
# Configuration values
# ---------------------------------------------------------------------------

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "trading_data")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT_HOST = os.getenv("POSTGRES_PORT_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL_HOST = os.getenv("DATABASE_URL_HOST")
DATABASE_URL_DOCKER = os.getenv("DATABASE_URL_DOCKER")


def _resolve_port() -> str:
    """Return the port that should be used for PostgreSQL connections."""
    return POSTGRES_PORT_HOST or POSTGRES_PORT or "5432"


def build_database_url(
    *,
    use_docker: bool = False,
    overrides: Optional[dict] = None,
) -> str:
    """
    Build a PostgreSQL connection string.

    Precedence:
        1. overrides["url"]
        2. DATABASE_URL (generic)
        3. DATABASE_URL_DOCKER (when use_docker=True)
        4. DATABASE_URL_HOST (when use_docker=False)
        5. Constructed from individual components.
    """

    overrides = overrides or {}

    if "url" in overrides and overrides["url"]:
        return overrides["url"]

    if DATABASE_URL:
        return DATABASE_URL

    if use_docker and DATABASE_URL_DOCKER:
        return DATABASE_URL_DOCKER

    if not use_docker and DATABASE_URL_HOST:
        return DATABASE_URL_HOST

    user = overrides.get("user", POSTGRES_USER)
    password = overrides.get("password", POSTGRES_PASSWORD)
    host = overrides.get("host", POSTGRES_HOST)
    port = overrides.get("port", _resolve_port())
    database = overrides.get("database", POSTGRES_DB)

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def connect_db(*, use_docker: bool = False, overrides: Optional[dict] = None):
    """Return a psycopg2 connection using the configured settings."""
    url = build_database_url(use_docker=use_docker, overrides=overrides)
    return psycopg2.connect(url)


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------

def _execute_with_transaction(
    conn: psycopg2.extensions.connection,
    query: str,
    values: Sequence[Tuple],
) -> int:
    """Execute a batch query with transaction safety."""
    if not values:
        return 0

    cursor = conn.cursor()
    try:
        execute_batch(cursor, query, values)
        conn.commit()
        return len(values)
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def insert_daily_bars(conn, bars: Sequence[dict]) -> int:
    """Insert or update daily bars."""
    query = """
        INSERT INTO daily_bars (
            ticker, date, open, high, low, close, volume,
            transactions, volume_weighted_avg_price
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            transactions = EXCLUDED.transactions,
            volume_weighted_avg_price = EXCLUDED.volume_weighted_avg_price,
            updated_at = CURRENT_TIMESTAMP
    """

    values = [
        (
            bar["ticker"],
            bar["date"],
            bar["open"],
            bar["high"],
            bar["low"],
            bar["close"],
            bar.get("volume"),
            bar.get("transactions"),
            bar.get("volume_weighted_avg_price"),
        )
        for bar in bars
    ]

    rows = _execute_with_transaction(conn, query, values)
    if rows:
        print(f"Inserted/updated {rows} daily bars")
    return rows


def insert_intraday_bars(conn, bars: Sequence[dict]) -> int:
    """Insert or update intraday bars."""
    query = """
        INSERT INTO intraday_bars (
            ticker, timestamp, open, high, low, close, volume,
            transactions, volume_weighted_avg_price
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, timestamp) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            transactions = EXCLUDED.transactions,
            volume_weighted_avg_price = EXCLUDED.volume_weighted_avg_price,
            updated_at = CURRENT_TIMESTAMP
    """

    values = [
        (
            bar["ticker"],
            bar["timestamp"],
            bar["open"],
            bar["high"],
            bar["low"],
            bar["close"],
            bar.get("volume"),
            bar.get("transactions"),
            bar.get("volume_weighted_avg_price"),
        )
        for bar in bars
    ]

    rows = _execute_with_transaction(conn, query, values)
    if rows:
        print(f"Inserted/updated {rows} intraday bars")
    return rows


def insert_news_articles(conn, articles: Sequence[dict]) -> int:
    """Insert or update Polygon news articles."""
    query = """
        INSERT INTO news_articles (
            id,
            ticker,
            published_at,
            title,
            description,
            url,
            author,
            type,
            sentiment_score,
            sentiment_label,
            sentiment_reasoning,
            keywords,
            tickers
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            sentiment_score = EXCLUDED.sentiment_score,
            sentiment_label = EXCLUDED.sentiment_label,
            sentiment_reasoning = EXCLUDED.sentiment_reasoning,
            keywords = EXCLUDED.keywords,
            tickers = EXCLUDED.tickers,
            updated_at = CURRENT_TIMESTAMP
    """

    values = [
        (
            article["id"],
            article["ticker"],
            article["published_at"],
            article.get("title", ""),
            article.get("description") or article.get("text", ""),
            article.get("url", ""),
            article.get("author", ""),
            article.get("type", "article"),
            article.get("sentiment_score"),
            article.get("sentiment_label"),
            article.get("sentiment_reasoning"),
            article.get("keywords", []),
            article.get("tickers", []),
        )
        for article in articles
    ]

    rows = _execute_with_transaction(conn, query, values)
    if rows:
        print(f"Inserted/updated {rows} news articles")
    return rows


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def update_metadata(
    conn,
    data_type: str,
    ticker: str,
    start_date: str,
    end_date: str,
    rows_inserted: int,
    *,
    status: str = "completed",
    error_message: Optional[str] = None,
) -> None:
    """Insert or update metadata for a backfill run."""
    query = """
        INSERT INTO api_metadata (
            ticker,
            data_type,
            date_range_start,
            date_range_end,
            last_fetch_date,
            last_success_date,
            status,
            rows_inserted,
            error_message,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
    """

    values = (
        ticker,
        data_type,
        start_date,
        end_date,
        end_date,
        end_date,
        status,
        rows_inserted,
        error_message,
    )

    _execute_with_transaction(conn, query, [values])


def get_last_fetch_date(conn, ticker: str, data_type: str) -> Optional[str]:
    """
    Return the last fetch date recorded for a ticker/data_type combination.

    Returns the most recent `last_fetch_date`.
    """
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT last_fetch_date
            FROM api_metadata
            WHERE ticker = %s AND data_type = %s
            AND status = 'completed'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (ticker, data_type),
        )
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        cursor.close()
