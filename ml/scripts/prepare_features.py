"""Feature preparation script for training models."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import psycopg2

# Add project root to PYTHONPATH so `ml.features` resolves when executed as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.features import (  # noqa: E402
    PriceFeatureEngineer,
    VolumeFeatureEngineer,
    TechnicalIndicatorsFeatureEngineer,
    NewsFeatureEngineer,
    TimeFeatureEngineer,
    CandlestickFeatureEngineer,
    ConfluenceFeatureEngineer,
    PriceFeatureConfig,
    VolumeFeatureConfig,
    TechnicalIndicatorConfig,
    NewsFeatureConfig,
    TimeFeatureConfig,
    ConfluenceConfig,
    RuleBasedSentimentModel,
    LLMSentimentModel,
    combine_sentiment_scores,
)

from app.polygon_trading_client import PolygonTradingClient  # noqa: E402
from app.config import insert_intraday_bars  # noqa: E402


def _connect_db():
    """
    Connect to PostgreSQL database using DATABASE_URL or DATABASE_URL_HOST env var.
    
    Returns:
        psycopg2.connection: Database connection
        
    Raises:
        RuntimeError: If neither env var is set
    """
    database_url = os.getenv("DATABASE_URL") or os.getenv("DATABASE_URL_HOST")
    if not database_url:
        raise RuntimeError("DATABASE_URL or DATABASE_URL_HOST must be set")
    return psycopg2.connect(database_url)

def _remove_non_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove weekend rows from daily bar data (keeps Monday-Friday only).
    
    Args:
        df: DataFrame with 'date' column
        
    Returns:
        DataFrame with only weekday rows
    """
    mask_weekday = df["date"].dt.dayofweek < 5
    filtered = df[mask_weekday].copy()
    return filtered

def _restrict_to_session(df: pd.DataFrame, cfg: TimeFeatureConfig) -> pd.DataFrame:
    """
    Filter intraday bars to only include market trading hours (excludes pre-market/after-hours).
    
    Args:
        df: DataFrame with 'timestamp' column
        cfg: TimeFeatureConfig with market hours
            
    Returns:
        DataFrame filtered to weekdays within market hours
    """
    if df.empty:
        return df

    minutes_open = cfg.market_open_hour * 60 + cfg.market_open_minute
    minutes_close = cfg.market_close_hour * 60 + cfg.market_close_minute

    minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    mask_weekday = df["timestamp"].dt.dayofweek < 5
    within_session = (minutes >= minutes_open) & (minutes < minutes_close)

    return df[mask_weekday & within_session].copy()


def _warn_if_date_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect gaps >3 days in daily bar data (may include holidays).
    
    Args:
        df: DataFrame with 'ticker' and 'date' columns (sorted by ticker, date)
            
    Returns:
        DataFrame with gap details (ticker, row_index, previous_date, current_date, gap_days).
        Empty if no gaps found.
        
    Note: Flags all gaps >3 days including holidays. Filter known holidays manually.
    """
    issues: list[pd.DataFrame] = []
    grouped = df.groupby("ticker")
    for ticker, group in grouped:
        gap_days = group["date"].diff().dt.days
        mask = gap_days > 3
        if mask.any():
            details = pd.DataFrame(
                {
                    "ticker": ticker,
                    "row_index": group.index[mask],
                    "previous_date": group["date"].shift(1)[mask].values,
                    "current_date": group["date"][mask].values,
                    "gap_days": gap_days[mask].values,
                }
            )
            issues.append(details)
            print(
                f"⚠️  {ticker}: detected {len(details)} daily gaps (>3 days) at row(s) "
                f"{details['row_index'].tolist()}"
            )
    if issues:
        return pd.concat(issues, ignore_index=True)
    return pd.DataFrame(columns=["ticker", "row_index", "previous_date", "current_date", "gap_days"])

def _warn_if_timestamp_gaps(df: pd.DataFrame, expected_interval_minutes: int = 5) -> pd.DataFrame:
    """
    Detect missing intraday bars and return exact missing timestamps.
    
    Excludes weekend/holiday gaps (201900s, 29100s). Flags gaps >1.5x expected interval.
    
    Args:
        df: DataFrame with 'ticker' and 'timestamp' columns (sorted)
        expected_interval_minutes: Expected bar interval (default: 5)
    
    Returns:
        DataFrame with gap details including missing_timestamps list. Empty if no gaps.
        
    Note: Use missing_timestamps with backfill_missing_intraday_timestamps() to auto-fill.
    """
    issues: list[pd.DataFrame] = []
    grouped = df.groupby("ticker")
    
    # Weekend/holiday gap thresholds (in seconds)
    WEEKEND_GAP_SECONDS = 201900.0  # ~56 hours (Friday close to Monday open)
    MARKET_CLOSED_GAP_SECONDS = 29100.0  # ~8 hours (market closed hours)
    
    for ticker, group in grouped:
        group = group.sort_values("timestamp").copy()
        # Preserve original index for reporting
        group["_original_index"] = group.index
        group = group.reset_index(drop=True)
        
        deltas_seconds = group["timestamp"].diff().dt.total_seconds()
        deltas_minutes = deltas_seconds / 60  # Convert to minutes
        
        # Flag gaps that are more than 1.5x the expected interval (allows small tolerance)
        # But exclude weekend/holiday gaps
        mask = (deltas_minutes > expected_interval_minutes * 1.5) & (
            deltas_seconds != WEEKEND_GAP_SECONDS
        ) & (deltas_seconds != MARKET_CLOSED_GAP_SECONDS)
        
        if mask.any():
            gap_details = []
            for idx in group.index[mask]:
                prev_ts = group.loc[idx - 1, "timestamp"] if idx > 0 else None
                curr_ts = group.loc[idx, "timestamp"]
                gap_minutes = deltas_minutes.loc[idx]
                gap_seconds = deltas_seconds.loc[idx]
                original_idx = group.loc[idx, "_original_index"]
                
                if prev_ts is None:
                    continue
                
                # Generate expected timestamps in the gap
                missing_timestamps = []
                expected = prev_ts + pd.Timedelta(minutes=expected_interval_minutes)
                while expected < curr_ts:
                    missing_timestamps.append(expected)
                    expected += pd.Timedelta(minutes=expected_interval_minutes)
                
                gap_details.append({
                    "ticker": ticker,
                    "row_index": original_idx,
                    "previous_timestamp": prev_ts,
                    "current_timestamp": curr_ts,
                    "gap_minutes": gap_minutes,
                    "gap_seconds": gap_seconds,
                    "missing_timestamps": missing_timestamps,
                    "missing_count": len(missing_timestamps),
                })
            
            if gap_details:
                details_df = pd.DataFrame(gap_details)
                issues.append(details_df)
                
                # Print summary with missing timestamps
                total_missing = sum(d["missing_count"] for d in gap_details)
                all_missing = [ts for d in gap_details for ts in d["missing_timestamps"]]
                threshold_min = expected_interval_minutes * 1.5
                print(
                    f"⚠️  {ticker}: detected {len(gap_details)} intraday gaps (>{threshold_min:.1f} minutes, excluding weekends/holidays) "
                    f"with {total_missing} missing bars. "
                    f"Missing timestamps: {all_missing[:10]}{'...' if len(all_missing) > 10 else ''}"
                )
    
    if issues:
        result = pd.concat(issues, ignore_index=True)
        return result
    return pd.DataFrame(
        columns=[
            "ticker",
            "row_index",
            "previous_timestamp",
            "current_timestamp",
            "gap_minutes",
            "gap_seconds",
            "missing_timestamps",
            "missing_count",
        ]
    )


def _flag_missing_intraday_neighbors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flag bars at sequence boundaries (start/end of day or dataset).
    
    Args:
        df: DataFrame with 'ticker' and 'timestamp' columns
        
    Returns:
        Tuple of (enhanced_df, missing_info_df):
        - enhanced_df: Adds missing_previous_bar, missing_next_bar flags
        - missing_info_df: Summary of rows with missing neighbors
    """
    features = df.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    features.sort_values(["ticker", "timestamp"], inplace=True)

    shifted_next = features.groupby("ticker")["timestamp"].shift(-1)
    shifted_prev = features.groupby("ticker")["timestamp"].shift(1)

    features["missing_next_bar"] = shifted_next.isna().astype(int)
    features["missing_previous_bar"] = shifted_prev.isna().astype(int)

    missing_mask = (features["missing_next_bar"] == 1) | (features["missing_previous_bar"] == 1)
    missing_info = (
        features.loc[
            missing_mask, ["ticker", "timestamp", "missing_previous_bar", "missing_next_bar"]
        ]
        .reset_index()
        .rename(columns={"index": "row_index"})
    )

    return features, missing_info

def _build_ticker_filter(ticker: Optional[Sequence[str] | str]) -> tuple[str, Optional[tuple]]:
    """
    Build SQL WHERE clause for ticker filtering (handles None, single, or multiple tickers).
    
    Args:
        ticker: None (all), str (single), or Sequence[str] (multiple)
            
    Returns:
        Tuple of (WHERE_clause, params) for parameterized queries
        
    Raises:
        TypeError: If ticker type is invalid
    """
    if ticker is None:
        return "", None
    if isinstance(ticker, str):
        return "WHERE ticker = %s", (ticker,)
    if isinstance(ticker, Iterable):
        tickers = list(ticker)
        if not tickers:
            return "", None
        return "WHERE ticker = ANY(%s)", (tickers,)
    raise TypeError("ticker must be None, a string, or an iterable of strings")


def _load_daily_bars(conn, ticker: Optional[Sequence[str] | str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load daily bars from database with data quality checks.
    
    Args:
        conn: Database connection
        ticker: Optional filter (None, str, or Sequence[str])
            
    Returns:
        Tuple of (daily_bars_df, gap_warnings_df):
        - daily_bars_df: OHLCV data with is_monday, is_friday, days_since_prev_close
        - gap_warnings_df: Gaps >3 days (empty if none)
    """
    where_clause, params = _build_ticker_filter(ticker)

    query = f"""
        SELECT ticker, date, open, high, low, close, volume, transactions, volume_weighted_avg_price
        FROM daily_bars
        {where_clause}
        ORDER BY ticker, date
    """
    df = pd.read_sql(query, conn, params=params)
    if df.empty:
        return df, pd.DataFrame(columns=["ticker", "row_index", "previous_date", "current_date", "gap_days"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["is_monday"] = (df["date"].dt.dayofweek == 0).astype(int)
    df["is_friday"] = (df["date"].dt.dayofweek == 4).astype(int)
    df["days_since_prev_close"] = (
        df.groupby("ticker")["date"].diff().dt.days.fillna(0).astype(int)
    )
    gap_warnings = _warn_if_date_gaps(df)
    return df, gap_warnings


def _load_news(conn, ticker: Optional[Sequence[str] | str]) -> pd.DataFrame:
    """
    Load news articles with sentiment scores from database.
    
    Args:
        conn: Database connection
        ticker: Optional filter (None, str, or Sequence[str])
            
    Returns:
        DataFrame with ticker, published_at, sentiment_score, sentiment_label.
        Empty if no news found.
        
    Note: Use _enrich_news_with_sentiment() to combine multiple sentiment sources.
    """
    where_clause, params = _build_ticker_filter(ticker)

    query = f"""
        SELECT ticker, published_at, sentiment_score, sentiment_label
        FROM news_articles
        {where_clause}
        ORDER BY ticker, published_at
    """
    df = pd.read_sql(query, conn, params=params)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def _load_intraday_bars(
    conn, ticker: Optional[Sequence[str] | str], time_cfg: TimeFeatureConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load intraday bars from database with gap detection.
    
    Args:
        conn: Database connection
        ticker: Optional filter (None, str, or Sequence[str])
        time_cfg: TimeFeatureConfig (required for API consistency)
                 
    Returns:
        Tuple of (intraday_bars_df, gap_warnings_df):
        - intraday_bars_df: OHLCV data with date, is_monday, is_friday, seconds_since_prev_bar
        - gap_warnings_df: Missing bars with missing_timestamps list (empty if none)
        
    Note: Use backfill_missing_intraday_timestamps() to fill gaps.
    """
    where_clause, params = _build_ticker_filter(ticker)

    query = f"""
        SELECT ticker, timestamp, open, high, low, close, volume, transactions, volume_weighted_avg_price
        FROM intraday_bars
        {where_clause}
        ORDER BY ticker, timestamp
    """
    df = pd.read_sql(query, conn, params=params)
    if df.empty:
        return df, pd.DataFrame(
            columns=[
                "ticker",
                "row_index",
                "previous_timestamp",
                "current_timestamp",
                "gap_minutes",
                "gap_seconds",
                "missing_timestamps",
                "missing_count",
            ]
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    df["date"] = df["timestamp"].dt.normalize()
    df["is_monday"] = (df["timestamp"].dt.dayofweek == 0).astype(int)
    df["is_friday"] = (df["timestamp"].dt.dayofweek == 4).astype(int)
    df["seconds_since_prev_bar"] = (
        df.groupby("ticker")["timestamp"].diff().dt.total_seconds().fillna(0).astype(int)
    )
    gap_warnings = _warn_if_timestamp_gaps(df)
    return df, gap_warnings


def _enrich_news_with_sentiment(
    news_df: pd.DataFrame,
    *,
    llm_model_name: Optional[str] = None,
    use_rule_sentiment: bool = True,
) -> pd.DataFrame:
    """
    Combine sentiment scores from vendor, rule-based, and LLM models.
    
    Args:
        news_df: DataFrame with 'description' or 'title' column
        llm_model_name: Optional HuggingFace model (e.g., "ProsusAI/finbert")
        use_rule_sentiment: Enable rule-based sentiment (default: True)
            
    Returns:
        Enriched DataFrame with sentiment_rule_score, sentiment_llm_score, 
        sentiment_score (combined), and sentiment_label.
        
    Note: Requires 'transformers' library for LLM sentiment.
    """
    working = news_df.copy()
    if "description" not in working.columns:
        working["description"] = working.get("title", "")

    rule_model = RuleBasedSentimentModel() if use_rule_sentiment else None
    llm_model = None
    if llm_model_name:
        try:
            llm_model = LLMSentimentModel(llm_model_name)
        except ImportError as exc:  # pragma: no cover - optional dependency
            warnings.warn(str(exc))
            llm_model = None

    enriched = combine_sentiment_scores(working, llm_model=llm_model, rule_model=rule_model)
    return enriched


def prepare_daily_features(
    ticker: Optional[str] = None,
    save_path: Optional[str] = None,
    *,
    price_config: PriceFeatureConfig | None = None,
    volume_config: VolumeFeatureConfig | None = None,
    technical_config: TechnicalIndicatorConfig | None = None,
    news_config: NewsFeatureConfig | None = None,
    confluence_config: ConfluenceConfig | None = None,
    llm_model_name: Optional[str] = None,
    use_rule_sentiment: bool = True,
) -> pd.DataFrame:
    """
    Complete pipeline to prepare daily features for ML models.
    
    Orchestrates: data loading, gap detection, price/volume/technical/news features,
    and confluence combination. Saves to Parquet if save_path provided.
    
    Args:
        ticker: Optional ticker filter (None = all tickers)
        save_path: Optional Parquet file path
        price_config, volume_config, technical_config, news_config, confluence_config:
            Optional configs (None = defaults)
        llm_model_name: Optional HuggingFace model for LLM sentiment
        use_rule_sentiment: Enable rule-based sentiment (default: True)
            
    Returns:
        DataFrame with all engineered features. Warnings in df.attrs["warnings"].
    """
    conn = _connect_db()
    try:
        daily_df, daily_gap_warnings = _load_daily_bars(conn, ticker)
        news_df = _load_news(conn, ticker)
    finally:
        conn.close()

    if daily_df.empty:
        _persist_if_requested(pd.DataFrame(), save_path)
        return pd.DataFrame()

    metadata: dict[str, pd.DataFrame] = {}
    if not daily_gap_warnings.empty:
        metadata["daily_gap_warnings"] = daily_gap_warnings

    price_engineer = PriceFeatureEngineer(price_config)
    volume_engineer = VolumeFeatureEngineer(volume_config)
    technical_engineer = TechnicalIndicatorsFeatureEngineer(technical_config)
    news_engineer = NewsFeatureEngineer(news_config)
    confluence_engineer = ConfluenceFeatureEngineer(
        price_config=price_config,
        volume_config=volume_config,
        technical_config=technical_config,
        news_config=news_config,
        confluence_config=confluence_config,
    )

    features = price_engineer.create_features(daily_df)
    features = volume_engineer.create_features(features)
    features = technical_engineer.create_features(features)
    if not news_df.empty:
        enriched_news = _enrich_news_with_sentiment(
            news_df,
            llm_model_name=llm_model_name,
            use_rule_sentiment=use_rule_sentiment,
        )
        features = news_engineer.create_features(enriched_news, features[["ticker", "date"]])
        article_counts = enriched_news.groupby("ticker").size().rename("news_article_count")
        features = features.merge(article_counts, on="ticker", how="left")
    else:
        features = news_engineer.create_features(pd.DataFrame(), features[["ticker", "date"]])

    confluence = confluence_engineer.create_features(features)
    confluence = confluence.dropna().reset_index(drop=True)

    if metadata:
        confluence.attrs["warnings"] = metadata

    _persist_if_requested(confluence, save_path)
    return confluence

def prepare_intraday_features(
    ticker: Optional[str] = None,
    save_path: Optional[str] = None,
    *,
    volume_config: VolumeFeatureConfig | None = None,
    time_config: TimeFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Complete pipeline to prepare intraday features for ML models.
    
    Generates: candlestick patterns, time-of-day features, volume features, and
    missing neighbor flags. Saves to Parquet if save_path provided.
    
    Args:
        ticker: Optional ticker filter (None = all tickers)
        save_path: Optional Parquet file path
        volume_config: Optional volume config (None = defaults)
        time_config: Optional time config (None = defaults)
            
    Returns:
        DataFrame with all engineered intraday features. Warnings in df.attrs["warnings"].
        Warnings include missing_timestamps for backfilling.
    """
    time_cfg = time_config or TimeFeatureConfig()
    conn = _connect_db()
    try:
        intraday_df, intraday_gap_warnings = _load_intraday_bars(conn, ticker, time_cfg)
    finally:
        conn.close()

    if intraday_df.empty:
        _persist_if_requested(pd.DataFrame(), save_path)
        return pd.DataFrame()

    metadata: dict[str, pd.DataFrame] = {}
    if not intraday_gap_warnings.empty:
        metadata["intraday_gap_warnings"] = intraday_gap_warnings

    candlestick_engineer = CandlestickFeatureEngineer()
    time_engineer = TimeFeatureEngineer(time_cfg)
    volume_engineer = VolumeFeatureEngineer(volume_config)

    features = candlestick_engineer.create_features(intraday_df)

    volume_input = features.copy()
    volume_input["date"] = pd.to_datetime(volume_input["timestamp"].dt.normalize())
    volume_features = volume_engineer.create_features(volume_input)
    volume_columns = [col for col in volume_features.columns if col.startswith("volume_") or col.startswith("price_volume")]
    features[volume_columns] = volume_features[volume_columns]

    features = time_engineer.create_features(features, timestamp_col="timestamp")
    features, missing_neighbor_info = _flag_missing_intraday_neighbors(features)
    if not missing_neighbor_info.empty:
        print(
            "⚠️  Missing intraday neighbor bars at row(s) "
            f"{missing_neighbor_info['row_index'].tolist()}"
        )
        metadata["missing_intraday_neighbors"] = missing_neighbor_info
    features = features.dropna().reset_index(drop=True)

    if metadata:
        features.attrs["warnings"] = metadata

    _persist_if_requested(features, save_path)
    return features

def _persist_if_requested(df: pd.DataFrame, save_path: Optional[str]) -> None:
    """
    Save DataFrame to Parquet file if save_path provided (creates parent dirs if needed).
    
    Args:
        df: DataFrame to save
        save_path: Optional file path (None = no save)
    """
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def backfill_missing_intraday_timestamps(
    gap_warnings: pd.DataFrame,
    conn,
    polygon_client: PolygonTradingClient,
    expected_interval_minutes: int = 5,
) -> dict[str, int]:
    """
    Automatically backfill missing intraday bars by fetching from Polygon API.
    
    Fetches full day's data for each date with missing bars, filters to missing timestamps,
    and inserts using UPSERT. Safe to run multiple times.
    
    Args:
        gap_warnings: DataFrame from _warn_if_timestamp_gaps() with missing_timestamps
        conn: Database connection
        polygon_client: Initialized PolygonTradingClient
        expected_interval_minutes: Expected bar interval (default: 5)
    
    Returns:
        Dict with dates_processed, bars_inserted, errors
        
    Note: Makes API calls (rate limits apply). Check gap_warnings first.
    """
    if gap_warnings.empty or "missing_timestamps" not in gap_warnings.columns:
        print("No gap warnings provided or missing_timestamps column not found")
        return {"dates_processed": 0, "bars_inserted": 0, "errors": 0}
    
    # Extract all missing timestamps with their tickers
    missing_records = []
    for _, row in gap_warnings.iterrows():
        ticker = row["ticker"]
        if isinstance(row["missing_timestamps"], list):
            for ts in row["missing_timestamps"]:
                missing_records.append({"ticker": ticker, "timestamp": ts})
        elif pd.notna(row["missing_timestamps"]):
            missing_records.append({"ticker": ticker, "timestamp": row["missing_timestamps"]})
    
    if not missing_records:
        print("No missing timestamps found in gap warnings")
        return {"dates_processed": 0, "bars_inserted": 0, "errors": 0}
    
    # Convert to DataFrame and group by date and ticker
    missing_df = pd.DataFrame(missing_records)
    missing_df["timestamp"] = pd.to_datetime(missing_df["timestamp"])
    missing_df["date"] = missing_df["timestamp"].dt.date
    
    # Get unique (ticker, date) combinations to fetch
    ticker_dates = missing_df[["ticker", "date"]].drop_duplicates().sort_values(["ticker", "date"])
    print(f"Found {len(ticker_dates)} ticker-date combinations with missing timestamps")
    
    dates_processed = 0
    bars_inserted = 0
    errors = 0
    
    for _, row in ticker_dates.iterrows():
        try:
            ticker = row["ticker"]
            date = row["date"]
            date_str = date.strftime("%Y-%m-%d")
            
            # Fetch full day's data from API
            print(f"Fetching {ticker} intraday data for {date_str}...")
            bars = polygon_client.get_intraday_bars(
                ticker=ticker,
                start_date=date_str,
                end_date=date_str,
                multiplier=expected_interval_minutes,
                timespan="minute",
            )
            
            if not bars:
                print(f"  ⚠️  No data returned from API for {date_str}")
                errors += 1
                continue
            
            # Convert API timestamps to datetime for comparison
            bars_df = pd.DataFrame(bars)
            bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], unit="ms", utc=True)
            
            # Get missing timestamps for this ticker and date
            date_missing = missing_df[
                (missing_df["ticker"] == ticker) & (missing_df["date"] == date)
            ]["timestamp"].tolist()
            date_missing_set = set(pd.to_datetime(date_missing))
            
            # Filter bars to only those that are missing
            bars_df["is_missing"] = bars_df["timestamp"].isin(date_missing_set)
            missing_bars = bars_df[bars_df["is_missing"]].copy()
            
            if missing_bars.empty:
                print(f"  ℹ️  No missing bars found in API response for {date_str} (may have been filled)")
                dates_processed += 1
                continue
            
            # Convert back to dict format for insert
            bars_to_insert = []
            for _, bar in missing_bars.iterrows():
                bars_to_insert.append({
                    "ticker": ticker,
                    "timestamp": bar["timestamp"],
                    "open": bar["open"],
                    "high": bar["high"],
                    "low": bar["low"],
                    "close": bar["close"],
                    "volume": bar.get("volume"),
                    "transactions": bar.get("transactions"),
                    "volume_weighted_avg_price": bar.get("volume_weighted_avg_price"),
                })
            
            # Insert into database
            inserted = insert_intraday_bars(conn, bars_to_insert)
            bars_inserted += inserted
            dates_processed += 1
            print(f"  ✅ Inserted {inserted} missing bars for {date_str}")
            
        except Exception as exc:
            print(f"  ❌ Error processing {date_str}: {exc}")
            errors += 1
            continue
    
    summary = {
        "dates_processed": dates_processed,
        "bars_inserted": bars_inserted,
        "errors": errors,
    }
    print(f"\n📊 Backfill summary: {summary}")
    return summary


def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for feature preparation script.
    
    Returns:
        Namespace with: model, ticker, save, llm_model, disable_rule_sentiment
    """
    parser = argparse.ArgumentParser(description="Prepare features for ML models")
    parser.add_argument("--model", choices=["daily", "intraday"], required=True, help="Model type")
    parser.add_argument("--ticker", default=None, help="Specific ticker (optional)")
    parser.add_argument("--save", default=None, help="Path to save features")
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional HuggingFace model name for sentiment scoring (e.g. ProsusAI/finbert)",
    )
    parser.add_argument(
        "--disable-rule-sentiment",
        action="store_true",
        help="Skip rule-based sentiment scoring (uses vendor/LLM only)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for CLI execution (used by Airflow DAGs and automation).
    
    For interactive use, call prepare_daily_features() or prepare_intraday_features() directly.
    """
    args = _parse_args()
    if args.model == "daily":
        df = prepare_daily_features(
            args.ticker,
            args.save,
            llm_model_name=args.llm_model,
            use_rule_sentiment=not args.disable_rule_sentiment,
        )
        print(f"✅ Prepared {len(df)} rows of daily confluence features")
    else:
        df = prepare_intraday_features(args.ticker, args.save)
        print(f"✅ Prepared {len(df)} rows of intraday features")


if __name__ == "__main__":
    main()



