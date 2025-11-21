"""Feature preparation script for training models."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta
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
    MarketFeatureEngineer,
    PriceFeatureConfig,
    VolumeFeatureConfig,
    TechnicalIndicatorConfig,
    NewsFeatureConfig,
    TimeFeatureConfig,
    ConfluenceConfig,
)
from app.symbols import get_market_index  # noqa: E402

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

def _is_holiday_gap(previous_date: pd.Timestamp, current_date: pd.Timestamp) -> bool:
    """
    Check if a date gap is due to a known market holiday.
    
    Args:
        previous_date: Previous trading date
        current_date: Current trading date
        
    Returns:
        True if the gap is due to a known market holiday, False otherwise
    """
    # Known market holidays that cause gaps
    # Format: (year, month, day) for the holiday date
    known_holidays = {
        # 2023
        (2023, 12, 25),  # Christmas Day
        (2024, 1, 1),    # New Year's Day
        # 2024
        (2024, 1, 15),   # Martin Luther King, Jr. Day
        (2024, 2, 19),   # Presidents' Day
        (2024, 3, 29),   # Good Friday
        (2024, 5, 27),   # Memorial Day
        (2024, 9, 2),    # Labor Day
        # 2025
        (2025, 1, 20),   # Martin Luther King, Jr. Day
        (2025, 2, 17),   # Presidents' Day
        (2025, 4, 18),   # Good Friday
        (2025, 5, 26),   # Memorial Day
        (2025, 7, 4),    # Independence Day
        (2025, 9, 1),    # Labor Day
    }
    
    gap_days = (current_date - previous_date).days
    
    # Check if gap is 4 days (typical Friday → Tuesday holiday pattern)
    # or longer (multiple holidays or extended weekends)
    if gap_days < 4:
        return False
    
    # Check if any date in the gap (excluding weekends) matches a known holiday
    # Iterate through dates in the gap, skipping weekends
    check_date = previous_date + pd.Timedelta(days=1)
    while check_date < current_date:
        # Skip weekends (Saturday=5, Sunday=6)
        if check_date.dayofweek < 5:  # Monday=0 to Friday=4
            date_tuple = (check_date.year, check_date.month, check_date.day)
            if date_tuple in known_holidays:
                return True
        check_date += pd.Timedelta(days=1)
    
    return False


def _warn_if_date_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect gaps >3 days in daily bar data, excluding known market holidays.
    
    Args:
        df: DataFrame with 'ticker' and 'date' columns (sorted by ticker, date)
            
    Returns:
        DataFrame with gap details (ticker, row_index, previous_date, current_date, gap_days).
        Empty if no gaps found. Holiday gaps are excluded.
        
    Note: Automatically filters out known market holidays (4-day gaps due to holidays).
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
            
            # Filter out holiday gaps
            holiday_mask = details.apply(
                lambda row: _is_holiday_gap(row["previous_date"], row["current_date"]),
                axis=1
            )
            holiday_count = holiday_mask.sum()
            details = details[~holiday_mask].copy()
            
            if holiday_count > 0:
                print(
                    f"{ticker}: filtered out {holiday_count} holiday gap(s) "
                    f"(legitimate market closures)"
                )
            
            if len(details) > 0:
                issues.append(details)
                print(
                    f"{ticker}: detected {len(details)} daily gaps (>3 days) at row(s) "
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
                    f"{ticker}: detected {len(gap_details)} intraday gaps (>{threshold_min:.1f} minutes, excluding weekends/holidays) "
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
    Load raw news articles from database (all fields from Polygon API including sentiment).
    
    Args:
        conn: Database connection
        ticker: Optional filter (None, str, or Sequence[str])
            
    Returns:
        DataFrame with all raw news fields: ticker, published_at, title, description,
        url, author, type, keywords, tickers, sentiment_score, sentiment_label, sentiment_reasoning.
        Empty if no news found.
    """
    where_clause, params = _build_ticker_filter(ticker)

    query = f"""
        SELECT 
            ticker, 
            published_at, 
            title, 
            description, 
            url, 
            author, 
            type, 
            keywords, 
            tickers,
            sentiment_score,
            sentiment_label,
            sentiment_reasoning
        FROM news_articles
        {where_clause}
        ORDER BY ticker, published_at
    """
    df = pd.read_sql(query, conn, params=params)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def _load_market_index_data(conn, market_index: str) -> pd.DataFrame:
    """
    Load market index daily bars from database.
    
    Args:
        conn: Database connection
        market_index: Market index ticker (e.g., "I:SPX" for S&P 500)
            
    Returns:
        DataFrame with date and close columns for the market index.
        Empty if no data found.
    """
    query = """
        SELECT date, close
        FROM daily_bars
        WHERE ticker = %s
        ORDER BY date
    """
    df = pd.read_sql(query, conn, params=(market_index,))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
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




def prepare_daily_features(
    ticker: Optional[str] = None,
    save_path: Optional[str] = None,
    *,
    price_config: PriceFeatureConfig | None = None,
    volume_config: VolumeFeatureConfig | None = None,
    technical_config: TechnicalIndicatorConfig | None = None,
    news_config: NewsFeatureConfig | None = None,
    confluence_config: ConfluenceConfig | None = None,
    # News features are optional and disabled by default
    include_news_features: bool = False,
) -> pd.DataFrame:
    """
    Complete pipeline to prepare daily features for ML models.
    
    Orchestrates: data loading, gap detection, price/volume/technical/market features,
    and confluence combination. News features are optional. Saves to Parquet if save_path provided.
    
    Args:
        ticker: Optional ticker filter (None = all tickers)
        save_path: Optional Parquet file path
        price_config, volume_config, technical_config, news_config, confluence_config:
            Optional configs (None = defaults)
        include_news_features: Include raw news features (default: False, disabled)
            
    Returns:
        DataFrame with all engineered features. Warnings in df.attrs["warnings"].
    """
    conn = _connect_db()
    try:
        daily_df, daily_gap_warnings = _load_daily_bars(conn, ticker)
        
        # Only load news data if news features are requested
        news_df = _load_news(conn, ticker) if include_news_features else pd.DataFrame()
        
        # Determine market index to use
        # If single ticker, use its market index; otherwise use S&P 500 as default
        if ticker and isinstance(ticker, str):
            market_index = get_market_index(ticker)
        else:
            # For multiple tickers or None, use S&P 500 (most common)
            from app.symbols import MARKET_INDICES
            market_index = MARKET_INDICES["US"]
        
        # Load market index data
        market_df = _load_market_index_data(conn, market_index)
    finally:
        conn.close()

    if daily_df.empty:
        _persist_if_requested(pd.DataFrame(), save_path)
        return pd.DataFrame()

    metadata: dict[str, pd.DataFrame] = {}
    if not daily_gap_warnings.empty:
        metadata["daily_gap_warnings"] = daily_gap_warnings

    # Initialize market feature engineer if market data is available
    market_engineer = None
    if not market_df.empty:
        try:
            market_engineer = MarketFeatureEngineer(market_df)
        except Exception as e:
            warnings.warn(f"Failed to initialize market feature engineer: {e}. Continuing without market features.")
            market_engineer = None
    
    # Initialize confluence engineer - it will create all features internally
    confluence_engineer = ConfluenceFeatureEngineer(
        price_config=price_config,
        volume_config=volume_config,
        technical_config=technical_config,
        news_config=news_config if include_news_features else None,
        confluence_config=confluence_config,
    )

    # Confluence engineer creates all features from raw daily_df
    # It handles price, volume, technical features internally
    confluence = confluence_engineer.create_features(
        daily_ohlcv=daily_df,
        news_df=news_df if include_news_features else None
    )
    
    # Add market features (market index close price, market returns, etc.) after confluence
    if market_engineer is not None:
        confluence = market_engineer.create_features(confluence)
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
    auto_backfill: bool = True,
    max_years_back: int = 2,
) -> pd.DataFrame:
    """
    Complete pipeline to prepare intraday features for ML models.
    
    Generates: candlestick patterns, time-of-day features, volume features, and
    missing neighbor flags. Automatically backfills missing bars if auto_backfill=True.
    Saves to Parquet if save_path provided.
    
    Args:
        ticker: Optional ticker filter (None = all tickers)
        save_path: Optional Parquet file path
        volume_config: Optional volume config (None = defaults)
        time_config: Optional time config (None = defaults)
        auto_backfill: Automatically backfill missing bars (default: True)
        max_years_back: Maximum years back to backfill (default: 2). Older gaps are filtered out.
            
    Returns:
        DataFrame with all engineered intraday features. Warnings in df.attrs["warnings"].
        Old gaps (>2 years) are filtered out and not included in warnings.
    """
    time_cfg = time_config or TimeFeatureConfig()
    conn = _connect_db()
    try:
        intraday_df, intraday_gap_warnings = _load_intraday_bars(conn, ticker, time_cfg)
        
        # Filter out gaps older than max_years_back and auto-backfill if enabled
        cutoff_date = (datetime.now() - timedelta(days=365 * max_years_back)).date()
        
        if not intraday_gap_warnings.empty:
            # Filter out old gaps (convert timestamps to dates for comparison)
            intraday_gap_warnings = intraday_gap_warnings.copy()
            intraday_gap_warnings["gap_date"] = pd.to_datetime(
                intraday_gap_warnings["previous_timestamp"]
            ).dt.date
            
            old_gaps = intraday_gap_warnings[intraday_gap_warnings["gap_date"] < cutoff_date]
            intraday_gap_warnings = intraday_gap_warnings[
                intraday_gap_warnings["gap_date"] >= cutoff_date
            ].copy()
            
            if not old_gaps.empty:
                print(
                    f"Filtered out {len(old_gaps)} gap(s) older than {cutoff_date} "
                    f"(not available in API plan, will impact analysis)"
                )
            
            # Auto-backfill if enabled and gaps exist
            if auto_backfill and not intraday_gap_warnings.empty:
                print(f"\n🔄 Auto-backfilling missing intraday bars...")
                from app.polygon_trading_client import PolygonTradingClient
                
                polygon_client = PolygonTradingClient()
                summary = backfill_missing_intraday_timestamps(
                    intraday_gap_warnings,
                    conn,
                    polygon_client,
                    expected_interval_minutes=5,
                    max_years_back=max_years_back,
                )
                
                if summary["bars_inserted"] > 0:
                    print(f"Auto-backfilled {summary['bars_inserted']} bars")
                    # Reload data after backfill to get updated dataset
                    intraday_df, intraday_gap_warnings = _load_intraday_bars(conn, ticker, time_cfg)
                    # Re-filter old gaps after reload
                    if not intraday_gap_warnings.empty:
                        intraday_gap_warnings["gap_date"] = pd.to_datetime(
                            intraday_gap_warnings["previous_timestamp"]
                        ).dt.date
                        intraday_gap_warnings = intraday_gap_warnings[
                            intraday_gap_warnings["gap_date"] >= cutoff_date
                        ].copy()
    finally:
        conn.close()

    if intraday_df.empty:
        _persist_if_requested(pd.DataFrame(), save_path)
        return pd.DataFrame()

    # Remove all data older than cutoff_date (can't be backfilled, will impact analysis)
    if "timestamp" in intraday_df.columns:
        intraday_df["date"] = pd.to_datetime(intraday_df["timestamp"]).dt.date
        rows_before = len(intraday_df)
        intraday_df = intraday_df[intraday_df["date"] >= cutoff_date].copy()
        rows_removed_old = rows_before - len(intraday_df)
        if rows_removed_old > 0:
            print(
                f"Removed {rows_removed_old} row(s) with dates before {cutoff_date} "
                f"(not available in API plan, will impact analysis)"
            )
            if "date" in intraday_df.columns:
                intraday_df = intraday_df.drop(columns=["date"])
        
        # Check if dataset is empty after removal
        if intraday_df.empty:
            print("Dataset is empty after removing old data")
            _persist_if_requested(pd.DataFrame(), save_path)
            return pd.DataFrame()

    metadata: dict[str, pd.DataFrame] = {}
    if not intraday_gap_warnings.empty:
        metadata["intraday_gap_warnings"] = intraday_gap_warnings

    candlestick_engineer = CandlestickFeatureEngineer()
    time_engineer = TimeFeatureEngineer(time_cfg)
    volume_engineer = VolumeFeatureEngineer(volume_config)

    features = candlestick_engineer.create_features(intraday_df)

    # Filter to market trading hours (9:30 AM - 4:00 PM ET) after timezone conversion
    # This removes pre-market and after-hours data
    features = time_engineer.create_features(features, timestamp_col="timestamp")
    
    # Filter to market hours using the timezone-aware timestamp
    if "timestamp" in features.columns and time_cfg.session_timezone:
        # Ensure timestamp is timezone-aware (should be after time_engineer conversion)
        if features["timestamp"].dt.tz is None:
            features["timestamp"] = pd.to_datetime(features["timestamp"]).dt.tz_localize("UTC").dt.tz_convert(time_cfg.session_timezone)
        
        # Convert market hours to minutes for comparison
        market_open_minutes = time_cfg.market_open_hour * 60 + time_cfg.market_open_minute  # 9*60 + 30 = 570
        market_close_minutes = time_cfg.market_close_hour * 60 + time_cfg.market_close_minute  # 16*60 + 0 = 960
        
        # Extract hour and minute from timezone-converted timestamp
        bar_minutes = features["hour"] * 60 + features["minute"]
        
        # Filter to market hours only
        market_hours_mask = (bar_minutes >= market_open_minutes) & (bar_minutes < market_close_minutes)
        rows_before = len(features)
        features = features[market_hours_mask].copy()
        rows_removed = rows_before - len(features)
        
        if rows_removed > 0:
            print(f"Removed {rows_removed} row(s) outside market hours ({time_cfg.market_open_hour}:{time_cfg.market_open_minute:02d} - {time_cfg.market_close_hour}:{time_cfg.market_close_minute:02d} {time_cfg.session_timezone})")

    volume_input = features.copy()
    volume_input["date"] = pd.to_datetime(volume_input["timestamp"].dt.normalize())
    volume_features = volume_engineer.create_features(volume_input)
    volume_columns = [col for col in volume_features.columns if col.startswith("volume_") or col.startswith("price_volume")]
    features[volume_columns] = volume_features[volume_columns]
    features, missing_neighbor_info = _flag_missing_intraday_neighbors(features)
    if not missing_neighbor_info.empty:
        print(
            "Missing intraday neighbor bars at row(s) "
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
    max_years_back: int = 2,
) -> dict[str, int]:
    """
    Automatically backfill missing intraday bars by fetching from Polygon API.
    
    Fetches full day's data for each date with missing bars, filters to missing timestamps,
    and inserts using UPSERT. Safe to run multiple times. Only backfills data within
    max_years_back (default: 2 years) due to API plan limitations.
    
    Args:
        gap_warnings: DataFrame from _warn_if_timestamp_gaps() with missing_timestamps
        conn: Database connection (must be open and valid)
        polygon_client: Initialized PolygonTradingClient
        expected_interval_minutes: Expected bar interval (default: 5)
        max_years_back: Maximum years back to backfill (default: 2). Older gaps are skipped.
    
    Returns:
        Dict with dates_processed, bars_inserted, errors, skipped_old
        
    Note: Makes API calls (rate limits apply). Filters out dates older than max_years_back.
    """
    if gap_warnings.empty or "missing_timestamps" not in gap_warnings.columns:
        print("No gap warnings provided or missing_timestamps column not found")
        return {"dates_processed": 0, "bars_inserted": 0, "errors": 0, "skipped_old": 0}
    
    # Calculate cutoff date (2 years ago)
    cutoff_date = (datetime.now() - timedelta(days=365 * max_years_back)).date()
    print(f"📅 Only backfilling data from {cutoff_date} onwards (API plan limitation)")
    
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
        return {"dates_processed": 0, "bars_inserted": 0, "errors": 0, "skipped_old": 0}
    
    # Convert to DataFrame and group by date and ticker
    missing_df = pd.DataFrame(missing_records)
    missing_df["timestamp"] = pd.to_datetime(missing_df["timestamp"])
    missing_df["date"] = missing_df["timestamp"].dt.date
    
    # Filter out dates older than cutoff
    total_dates = len(missing_df["date"].unique())
    missing_df = missing_df[missing_df["date"] >= cutoff_date].copy()
    skipped_old = total_dates - len(missing_df["date"].unique())
    
    if skipped_old > 0:
        print(f"Skipping {skipped_old} date(s) older than {cutoff_date} (not available in API plan)")
    
    if missing_df.empty:
        print("No missing timestamps within backfill window (last 2 years)")
        return {"dates_processed": 0, "bars_inserted": 0, "errors": 0, "skipped_old": skipped_old}
    
    # Group by ticker and create date ranges for batch fetching
    # This reduces API calls by fetching multiple days at once
    ticker_date_groups = missing_df.groupby("ticker")["date"].apply(lambda x: sorted(set(x))).to_dict()
    print(f"Found {len(ticker_date_groups)} ticker(s) with missing timestamps (within backfill window)")
    
    dates_processed = 0
    bars_inserted = 0
    errors = 0
    
    for ticker, dates in ticker_date_groups.items():
        try:
            # Check if connection is still open, reconnect if needed
            if conn.closed:
                print("  Connection closed, reconnecting...")
                conn = _connect_db()
            
            # Batch dates: fetch date ranges instead of individual days
            # Group consecutive dates into ranges to minimize API calls
            date_ranges = []
            if not dates:
                continue
            
            # Sort dates and group consecutive dates
            sorted_dates = sorted(dates)
            range_start = sorted_dates[0]
            range_end = sorted_dates[0]
            
            for i in range(1, len(sorted_dates)):
                # If dates are consecutive (within 7 days), include in same range
                # Otherwise, start a new range
                days_diff = (sorted_dates[i] - range_end).days
                if days_diff <= 7:
                    range_end = sorted_dates[i]
                else:
                    date_ranges.append((range_start, range_end))
                    range_start = sorted_dates[i]
                    range_end = sorted_dates[i]
            date_ranges.append((range_start, range_end))
            
            print(f"Fetching {ticker} intraday data for {len(date_ranges)} date range(s)...")
            
            # Get all missing timestamps for this ticker
            ticker_missing = missing_df[missing_df["ticker"] == ticker]["timestamp"].tolist()
            ticker_missing_set = set(pd.to_datetime(ticker_missing))
            
            # Fetch data for each date range
            all_missing_bars = []
            for range_start, range_end in date_ranges:
                start_str = range_start.strftime("%Y-%m-%d")
                end_str = range_end.strftime("%Y-%m-%d")
                
                try:
                    # Fetch date range from API (much more efficient than per-day calls)
                    if range_start == range_end:
                        print(f"  Fetching {ticker} for {start_str}...")
                    else:
                        print(f"  Fetching {ticker} for {start_str} to {end_str}...")
                    
                    bars = polygon_client.get_intraday_bars(
                        ticker=ticker,
                        start_date=start_str,
                        end_date=end_str,
                        multiplier=expected_interval_minutes,
                        timespan="minute",
                    )
                    
                    if not bars:
                        if range_start == range_end:
                            print(f"    No data returned from API for {start_str}")
                        else:
                            print(f"    No data returned from API for {start_str} to {end_str}")
                        errors += 1
                        continue
                    
                    # Convert API timestamps to datetime for comparison
                    bars_df = pd.DataFrame(bars)
                    bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], unit="ms", utc=True)
                    
                    # Filter bars to only those that are missing
                    bars_df["is_missing"] = bars_df["timestamp"].isin(ticker_missing_set)
                    range_missing_bars = bars_df[bars_df["is_missing"]].copy()
                    
                    if not range_missing_bars.empty:
                        all_missing_bars.append(range_missing_bars)
                    
                except Exception as exc:
                    error_msg = str(exc)
                    # Check for API authorization errors (old data not available)
                    if "NOT_AUTHORIZED" in error_msg or "doesn't include this data timeframe" in error_msg:
                        if range_start == range_end:
                            print(f"    Skipping {start_str}: Not available in API plan (too old)")
                        else:
                            print(f"    Skipping {start_str} to {end_str}: Not available in API plan (too old)")
                        skipped_old += 1
                    else:
                        if range_start == range_end:
                            print(f"    Error processing {start_str}: {exc}")
                        else:
                            print(f"    Error processing {start_str} to {end_str}: {exc}")
                        errors += 1
                    continue
            
            if not all_missing_bars:
                print(f"  No missing bars found in API response for {ticker} (may have been filled)")
                dates_processed += len(dates)
                continue
            
            # Combine all missing bars from all date ranges
            combined_missing = pd.concat(all_missing_bars, ignore_index=True)
            
            # Convert back to dict format for insert
            bars_to_insert = []
            for _, bar in combined_missing.iterrows():
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
            
            # Check connection again before insert
            if conn.closed:
                conn = _connect_db()
            
            # Insert into database
            inserted = insert_intraday_bars(conn, bars_to_insert)
            bars_inserted += inserted
            dates_processed += len(dates)
            print(f"  Inserted {inserted} missing bars for {ticker} ({len(dates)} date(s))")
            
        except Exception as exc:
            print(f"  Error processing {ticker}: {exc}")
            errors += 1
            continue
    
    summary = {
        "dates_processed": dates_processed,
        "bars_inserted": bars_inserted,
        "errors": errors,
        "skipped_old": skipped_old,
    }
    print(f"\nBackfill summary: {summary}")
    return summary


def export_features_to_parquet(
    data_type: str,
    ticker: Optional[str] = None,
    output_path: Optional[str] = None,
    *,
    price_config: PriceFeatureConfig | None = None,
    volume_config: VolumeFeatureConfig | None = None,
    technical_config: TechnicalIndicatorConfig | None = None,
    news_config: NewsFeatureConfig | None = None,
    confluence_config: ConfluenceConfig | None = None,
    time_config: TimeFeatureConfig | None = None,
    auto_backfill: bool = True,
    max_years_back: int = 2,
    include_news_features: bool = False,
) -> str:
    """
    Export engineered features to Parquet file.
    
    Args:
        data_type: Type of data to export ("daily", "intraday", or "news")
        ticker: Optional ticker filter (None = all tickers)
        output_path: Output file path (default: auto-generated based on data_type and ticker)
        price_config, volume_config, technical_config, news_config, confluence_config, time_config:
            Optional configs (None = defaults)
        auto_backfill: Automatically backfill missing intraday bars (default: True, intraday only)
        max_years_back: Maximum years back to backfill (default: 2, intraday only)
        include_news_features: Include raw news features (default: False, disabled)
        
    Returns:
        Path to the exported Parquet file
        
    Raises:
        ValueError: If data_type is not "daily", "intraday", or "news"
    """
    if data_type not in ("daily", "intraday", "news"):
        raise ValueError(f"data_type must be 'daily', 'intraday', or 'news', got '{data_type}'")
    
    # Generate output path if not provided
    if output_path is None:
        from pathlib import Path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker_suffix = f"_{ticker}" if ticker else "_all"
        output_path = f"ml/data/{data_type}_features{ticker_suffix}_{timestamp}.parquet"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting {data_type} features to {output_path}...")
    
    if data_type == "daily":
        df = prepare_daily_features(
            ticker=ticker,
            save_path=None,  # We'll save manually after
            price_config=price_config,
            volume_config=volume_config,
            technical_config=technical_config,
            news_config=news_config,
            confluence_config=confluence_config,
            include_news_features=include_news_features,
        )
    elif data_type == "intraday":
        df = prepare_intraday_features(
            ticker=ticker,
            save_path=None,  # We'll save manually after
            volume_config=volume_config,
            time_config=time_config,
            auto_backfill=auto_backfill,
            max_years_back=max_years_back,
        )
    else:  # news
        conn = _connect_db()
        try:
            news_df = _load_news(conn, ticker)
            if news_df.empty:
                print("No news data found")
                df = pd.DataFrame()
            else:
                # Return raw news data
                df = news_df
        finally:
            conn.close()
    
    # Save to parquet
    if df.empty:
        print(f"No data to export for {data_type}")
        return output_path
    
    _persist_if_requested(df, output_path)
    print(f"Exported {len(df)} rows to {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for feature preparation script.
    
    Returns:
        Namespace with: model, ticker, save, export, data_type, include_news
    """
    parser = argparse.ArgumentParser(description="Prepare features for ML models")
    parser.add_argument("--model", choices=["daily", "intraday"], required=False, help="Model type (deprecated: use --export with --data-type)")
    parser.add_argument("--ticker", default=None, help="Specific ticker (optional)")
    parser.add_argument("--save", default=None, help="Path to save features (deprecated: use --export)")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export features to Parquet file",
    )
    parser.add_argument(
        "--data-type",
        choices=["daily", "intraday", "news"],
        default=None,
        help="Data type to export (required if --export is used)",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output Parquet file path (auto-generated if not provided)",
    )
    parser.add_argument(
        "--include-news",
        action="store_true",
        help="Include raw news features (default: False)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for CLI execution (used by Airflow DAGs and automation).
    
    For interactive use, call prepare_daily_features(), prepare_intraday_features(), or export_features_to_parquet() directly.
    """
    args = _parse_args()
    
    # New export mode (preferred)
    if args.export:
        if not args.data_type:
            print("Error: --data-type is required when using --export")
            print("   Example: python -m ml.scripts.prepare_features --export --data-type daily --ticker AAPL")
            return
        
        output_path = export_features_to_parquet(
            data_type=args.data_type,
            ticker=args.ticker,
            output_path=args.output_path,
            include_news_features=args.include_news,
        )
        print(f"Export complete: {output_path}")
        return
    
    # Legacy mode (for backward compatibility with Airflow DAGs)
    if args.model:
        if args.model == "daily":
            df = prepare_daily_features(
                args.ticker,
                args.save,
                include_news_features=args.include_news,
            )
            print(f"Prepared {len(df)} rows of daily confluence features")
        else:
            df = prepare_intraday_features(args.ticker, args.save)
            print(f"Prepared {len(df)} rows of intraday features")
    else:
        print("Error: Either --export with --data-type or --model is required")
        print("   Example: python -m ml.scripts.prepare_features --export --data-type daily --ticker AAPL")


if __name__ == "__main__":
    main()



