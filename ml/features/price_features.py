"""Configurable price-based feature engineering for daily bars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from .indicator_config import PriceFeatureConfig

_REQUIRED_COLUMNS = ("ticker", "date", "open", "high", "low", "close", "volume")


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    """Validate that DataFrame contains required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} requires columns {missing}, but DataFrame has {list(df.columns)}")


@dataclass(slots=True)
class PriceFeatureSummary:
    ticker: str
    rows: int


class PriceFeatureEngineer:
    """Engineer price-based features: returns, volatility, moving averages, price position."""

    def __init__(self, config: PriceFeatureConfig | None = None):
        """
        Initialize price feature engineer.
        
        Args:
            config: Configuration for feature windows (defaults to PriceFeatureConfig())
        """
        self.config = config or PriceFeatureConfig()
        self._summaries: List[PriceFeatureSummary] = []

    @property
    def summaries(self) -> List[PriceFeatureSummary]:
        """Get summary statistics for processed tickers."""
        return self._summaries

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price features: returns, volatility, SMAs, EMAs, price position metrics.
        
        Args:
            df: DataFrame with ticker, date, open, high, low, close, volume columns
            
        Returns:
            DataFrame with added price feature columns
        """
        if df.empty:
            return df.copy()

        _ensure_columns(df, _REQUIRED_COLUMNS, "PriceFeatureEngineer")
        cfg = self.config

        data = df.sort_values(["ticker", "date"]).copy()
        grouped = data.groupby("ticker", group_keys=False)

        # Returns and price change
        data["price_change"] = grouped["close"].diff()
        data["daily_return"] = grouped["close"].pct_change()
        for window in cfg.return_windows:
            data[f"return_{window}d"] = grouped["close"].pct_change(window)

        # Volatility
        for window in cfg.volatility_windows:
            data[f"return_volatility_{window}d"] = (
                grouped["daily_return"].rolling(window).std().reset_index(level=0, drop=True)
            )

        # Moving averages
        for window in cfg.sma_windows:
            sma_col = f"sma_{window}"
            data[sma_col] = grouped["close"].rolling(window).mean().reset_index(level=0, drop=True)
            data[f"close_vs_sma{window}"] = (data["close"] - data[sma_col]) / (data[sma_col] + 1e-10)

        for window in cfg.ema_windows:
            ema_col = f"ema_{window}"
            data[ema_col] = grouped["close"].transform(lambda s: s.ewm(span=window, adjust=False).mean())
            data[f"close_vs_ema{window}"] = (data["close"] - data[ema_col]) / (data[ema_col] + 1e-10)

        # Price position metrics
        for window in cfg.price_position_windows:
            high_col = grouped["high"].rolling(window).max().reset_index(level=0, drop=True)
            low_col = grouped["low"].rolling(window).min().reset_index(level=0, drop=True)
            data[f"price_vs_{window}d_range"] = (data["close"] - low_col) / (high_col - low_col + 1e-10)

        # True range style features
        prev_close = grouped["close"].shift(1)
        tr_components = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        )
        data["true_range"] = tr_components.max(axis=1)
        tr_window = cfg.true_range_window
        data[f"avg_true_range_{tr_window}"] = (
            grouped["true_range"].rolling(tr_window).mean().reset_index(level=0, drop=True)
        )
        data["tr_percent"] = data["true_range"] / (data["close"] + 1e-10)

        if not cfg.keep_na:
            data = data.dropna(subset=["close"]).reset_index(drop=True)

        self._summaries = [
            PriceFeatureSummary(ticker=t, rows=len(group)) for t, group in data.groupby("ticker")
        ]
        return data

    # Backwards compatibility ---------------------------------------------------------
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.create_features(df)

