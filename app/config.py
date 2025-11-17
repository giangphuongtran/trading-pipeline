# app/config.py  — minimal, .env-driven

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env (project root)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

# ---------------------------------------------------------------------------
# Env values
# ---------------------------------------------------------------------------
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "trading_data")

POSTGRES_PORT_LOCAL = os.getenv("POSTGRES_PORT_HOST")

POSTGRES_PORT_DOCKER = os.getenv("POSTGRES_PORT")

# Optional full URLs (preferred if set)
DATABASE_URL_HOST   = os.getenv("DATABASE_URL_HOST")   # e.g. postgresql://user:pass@localhost:5440/trading_data
DATABASE_URL_DOCKER = os.getenv("DATABASE_URL_DOCKER") # e.g. postgresql://airflow:supersecret@postgres:5432/trading_data


def build_database_url(
    *,
    use_docker: bool = False,
    overrides: Optional[dict] = None,
) -> str:
    """
    Build PostgreSQL connection URL from environment variables or overrides.
    
    Precedence: 1) overrides['url'], 2) DATABASE_URL_DOCKER/HOST env vars, 3) compose from POSTGRES_* vars.
    
    Why use_docker? When running inside Docker, the database hostname is "postgres" (Docker service name).
    When running locally, it's "localhost". This flag switches between network contexts.
    
    What are overrides? Custom values that replace defaults. Useful for testing or special cases.
    Example: build_database_url(use_docker=True, overrides={"host": "custom-host"})
    
    Args:
        use_docker: If True, use Docker network hostname "postgres" (default: False, uses "localhost")
        overrides: Optional dict to override specific parts:
            - 'url': Full connection URL (highest priority)
            - 'host', 'port', 'user', 'password', 'database': Individual connection parts
        
    Returns:
        PostgreSQL connection URL string
    """
    overrides = overrides or {}

    # 1) Explicit override URL
    if overrides.get("url"):
        return overrides["url"]

    # 2) Full URLs from env (.env you provided sets both)
    if use_docker and DATABASE_URL_DOCKER:
        return DATABASE_URL_DOCKER
    if not use_docker and DATABASE_URL_HOST:
        return DATABASE_URL_HOST

    # 3) Compose a URL from parts
    user     = overrides.get("user", POSTGRES_USER)
    password = overrides.get("password", POSTGRES_PASSWORD)
    database = overrides.get("database", POSTGRES_DB)

    if use_docker:
        host = overrides.get("host", "postgres")
        port = overrides.get("port", POSTGRES_PORT_DOCKER)
    else:
        host = overrides.get("host", "localhost")
        port = overrides.get("port", POSTGRES_PORT_LOCAL)

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def connect_db(*, use_docker: bool = False, overrides: Optional[dict] = None):
    """
    Create and return a PostgreSQL database connection.
    
    Args:
        use_docker: If True, connect to Docker network database (default: False)
        overrides: Optional connection parameter overrides
        
    Returns:
        psycopg2 connection object
    """
    url = build_database_url(use_docker=use_docker, overrides=overrides)
    # Optional: print masked URL for debugging
    try:
        if "@" in url and ":" in url.split("@")[0]:
            userinfo, rest = url.split("@", 1)
            scheme, creds = userinfo.split("://", 1)
            if ":" in creds:
                u, p = creds.split(":", 1)
                url_masked = f"{scheme}://{u}:*****@{rest}"
                print(f"[DB] Connecting to {url_masked}")
            else:
                print(f"[DB] Connecting to {url}")
        else:
            print(f"[DB] Connecting to {url}")
    except Exception:
        pass
    return psycopg2.connect(url)

# ---------------------------------------------------------------------------
# Insert helpers (unchanged)
# ---------------------------------------------------------------------------

def _execute_with_transaction(
    conn: psycopg2.extensions.connection,
    query: str,
    values: Sequence[Tuple],
) -> int:
    """
    Execute a batch SQL query within a transaction with rollback on error.
    
    Args:
        conn: Database connection
        query: SQL query string with placeholders
        values: Sequence of parameter tuples
        
    Returns:
        Number of rows affected
    """
    if not values:
        return 0
    cur = conn.cursor()
    try:
        execute_batch(cur, query, values)
        conn.commit()
        return len(values)
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def insert_daily_bars(conn, bars: Sequence[dict]) -> int:
    """
    Insert or update daily bars in the database (UPSERT).
    
    Args:
        conn: Database connection
        bars: List of bar dicts with ticker, date, open, high, low, close, volume, etc.
        
    Returns:
        Number of rows inserted/updated
    """
    query = """
        INSERT INTO daily_bars (
            ticker, date, open, high, low, close, volume,
            transactions, volume_weighted_avg_price
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
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
    vals = [
        (
            b["ticker"], b["date"], b["open"], b["high"], b["low"], b["close"],
            b.get("volume"), b.get("transactions"), b.get("volume_weighted_avg_price"),
        )
        for b in bars
    ]
    rows = _execute_with_transaction(conn, query, vals)
    if rows:
        print(f"Inserted/updated {rows} daily bars")
    return rows


def insert_intraday_bars(conn, bars: Sequence[dict]) -> int:
    """
    Insert or update intraday bars in the database (UPSERT).
    
    Args:
        conn: Database connection
        bars: List of bar dicts with ticker, timestamp, open, high, low, close, volume, etc.
        
    Returns:
        Number of rows inserted/updated
    """
    query = """
        INSERT INTO intraday_bars (
            ticker, timestamp, open, high, low, close, volume,
            transactions, volume_weighted_avg_price
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
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
    vals = [
        (
            b["ticker"], b["timestamp"], b["open"], b["high"], b["low"], b["close"],
            b.get("volume"), b.get("transactions"), b.get("volume_weighted_avg_price"),
        )
        for b in bars
    ]
    rows = _execute_with_transaction(conn, query, vals)
    if rows:
        print(f"Inserted/updated {rows} intraday bars")
    return rows


def insert_news_articles(conn, articles: Sequence[dict]) -> int:
    """
    Insert or update news articles in the database (UPSERT).
    
    Args:
        conn: Database connection
        articles: List of article dicts with id, ticker, published_at, title, sentiment, etc.
        
    Returns:
        Number of rows inserted/updated
    """
    query = """
        INSERT INTO news_articles (
            id, ticker, published_at, title, description, url, author, type,
            sentiment_score, sentiment_label, sentiment_reasoning, keywords, tickers
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
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
    vals = [
        (
            a["id"], a["ticker"], a["published_at"],
            a.get("title", ""), a.get("description") or a.get("text", ""),
            a.get("url", ""), a.get("author", ""), a.get("type", "article"),
            a.get("sentiment_score"), a.get("sentiment_label"),
            a.get("sentiment_reasoning"), a.get("keywords", []), a.get("tickers", []),
        )
        for a in articles
    ]
    rows = _execute_with_transaction(conn, query, vals)
    if rows:
        print(f"Inserted/updated {rows} news articles")
    return rows


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
    """
    Record backfill metadata in api_metadata table.
    
    Args:
        conn: Database connection
        data_type: Type of data ("daily", "intraday", "news")
        ticker: Stock symbol
        start_date: Start date of backfill (YYYY-MM-DD)
        end_date: End date of backfill (YYYY-MM-DD)
        rows_inserted: Number of rows inserted
        status: Status ("completed", "failed")
        error_message: Error message if status is "failed"
    """
    query = """
        INSERT INTO api_metadata (
            ticker, data_type, date_range_start, date_range_end,
            last_fetch_date, last_success_date, status,
            rows_inserted, error_message, updated_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,CURRENT_TIMESTAMP)
    """
    vals = (
        ticker, data_type, start_date, end_date, end_date, end_date,
        status, rows_inserted, error_message,
    )
    _execute_with_transaction(conn, query, [vals])


def get_last_fetch_date(conn, ticker: str, data_type: str) -> Optional[str]:
    """
    Get the last successful fetch date for a ticker and data type.
    
    Args:
        conn: Database connection
        ticker: Stock symbol
        data_type: Type of data ("daily", "intraday", "news")
        
    Returns:
        Last fetch date as YYYY-MM-DD string, or None if not found
    """
    cur = conn.cursor()
    try:
        cur.execute(
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
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        cur.close()