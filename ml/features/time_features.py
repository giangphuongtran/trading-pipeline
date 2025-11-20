"""Configurable time-based feature engineering for intraday series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .indicator_config import TimeFeatureConfig


@dataclass(slots=True)
class TimeFeatureSummary:
    ticker: str | None
    rows: int


class TimeFeatureEngineer:
    """Create time-of-day features with configurable session windows."""

    def __init__(self, config: TimeFeatureConfig | None = None):
        """
        Initialize time feature engineer.
        
        Args:
            config: Configuration for market hours and session windows (defaults to TimeFeatureConfig())
        """
        self.config = config or TimeFeatureConfig()
        self._summaries: List[TimeFeatureSummary] = []

    @property
    def summaries(self) -> List[TimeFeatureSummary]:
        """Get summary statistics for processed tickers."""
        return self._summaries

    def create_features(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Create time-of-day features: hour, minute, session flags, cyclical encodings.
        
        Converts timestamps to market timezone (ET) before extracting time components
        to ensure hours/minutes reflect actual trading hours.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column (default: "timestamp")
            
        Returns:
            DataFrame with added time feature columns
        """
        if df.empty:
            return df.copy()

        data = df.copy()
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        
        # Convert to market timezone (ET) if timestamp is timezone-aware
        # This ensures hours/minutes reflect actual trading hours, not UTC
        if data[timestamp_col].dt.tz is not None:
            data[timestamp_col] = data[timestamp_col].dt.tz_convert(self.config.session_timezone)
        elif self.config.session_timezone:
            # If naive datetime, assume UTC and convert to market timezone
            data[timestamp_col] = data[timestamp_col].dt.tz_localize("UTC").dt.tz_convert(self.config.session_timezone)

        data = self._extract_time_components(data, timestamp_col)
        data = self._create_session_flags(data)
        data = self._create_cyclical_encoding(data)
        data = self._create_time_of_day_features(data)

        self._summaries = [
            TimeFeatureSummary(ticker=data.get("ticker", pd.Series([None])).iloc[0] if "ticker" in data.columns else None, rows=len(data))
        ]
        return data

    # ------------------------------------------------------------------
    def _extract_time_components(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        df["hour"] = df[timestamp_col].dt.hour
        df["minute"] = df[timestamp_col].dt.minute
        df["day_of_week"] = df[timestamp_col].dt.dayofweek
        return df

    def _create_session_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config

        open_start = cfg.market_open_hour * 60 + cfg.market_open_minute
        open_end = open_start + cfg.opening_window_minutes
        close_end = cfg.market_close_hour * 60 + cfg.market_close_minute
        close_start = max(close_end - cfg.closing_window_minutes, 0)

        minutes = df["hour"] * 60 + df["minute"]

        df["is_opening_window"] = ((minutes >= open_start) & (minutes < open_end)).astype(int)
        df["is_closing_window"] = ((minutes >= close_start) & (minutes < close_end)).astype(int)

        lunch_start, lunch_end = cfg.lunch_hours
        df["is_lunch_hour"] = ((df["hour"] >= lunch_start) & (df["hour"] < lunch_end)).astype(int)
        return df

    def _create_cyclical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60.0)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60.0)
        return df

    def _create_time_of_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        morning_start, morning_end = cfg.morning_hours
        afternoon_start, afternoon_end = cfg.afternoon_hours

        df["is_morning"] = ((df["hour"] >= morning_start) & (df["hour"] < morning_end)).astype(int)
        df["is_afternoon"] = ((df["hour"] >= afternoon_start) & (df["hour"] < afternoon_end)).astype(int)
        df["is_first_hour"] = (df["hour"] == cfg.market_open_hour).astype(int)
        last_hour = cfg.market_close_hour if cfg.market_close_minute == 0 else cfg.market_close_hour
        df["is_last_hour"] = (df["hour"] == (last_hour - 1)).astype(int)
        return df

