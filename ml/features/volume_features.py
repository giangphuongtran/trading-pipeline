"""Configurable volume-based feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from .indicator_config import VolumeFeatureConfig

_REQUIRED_COLUMNS = ("ticker", "date", "close", "volume")


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    """Validate that DataFrame contains required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} requires columns {missing}, but DataFrame has {list(df.columns)}")


@dataclass(slots=True)
class VolumeFeatureSummary:
    ticker: str
    rows: int


class VolumeFeatureEngineer:
    """Engineer volume-based features: ratios, spikes, trends, price-volume interactions."""

    def __init__(self, config: VolumeFeatureConfig | None = None):
        """
        Initialize volume feature engineer.
        
        Args:
            config: Configuration for feature windows and thresholds (defaults to VolumeFeatureConfig())
        """
        self.config = config or VolumeFeatureConfig()
        self._summaries: List[VolumeFeatureSummary] = []

    @property
    def summaries(self) -> List[VolumeFeatureSummary]:
        """Get summary statistics for processed tickers."""
        return self._summaries

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume features: moving averages, ratios, spikes, trends, price-volume metrics.
        
        Args:
            df: DataFrame with ticker, date, close, volume columns
            
        Returns:
            DataFrame with added volume feature columns
        """
        if df.empty:
            return df.copy()

        _ensure_columns(df, _REQUIRED_COLUMNS, "VolumeFeatureEngineer")
        cfg = self.config

        data = df.sort_values(["ticker", "date"]).copy()
        grouped = data.groupby("ticker", group_keys=False)

        # Moving averages of volume
        for window in cfg.volume_windows:
            data[f"volume_sma_{window}"] = grouped["volume"].rolling(window).mean().reset_index(level=0, drop=True)

        baseline_col = f"volume_sma_{cfg.ratio_baseline_window}"
        if baseline_col not in data.columns:
            data[baseline_col] = grouped["volume"].rolling(cfg.ratio_baseline_window).mean().reset_index(level=0, drop=True)

        data["volume_ratio"] = data["volume"] / (data[baseline_col] + 1e-10)
        data["volume_spike"] = (data["volume_ratio"] > cfg.spike_threshold).astype(int)
        data["volume_dry"] = (data["volume_ratio"] < cfg.dry_threshold).astype(int)

        # Volume trends
        def _slope(series: pd.Series) -> float:
            if series.isna().any() or len(series) < 2:
                return 0.0
            x = np.arange(len(series))
            try:
                slope, _ = np.polyfit(x, series, 1)
            except np.linalg.LinAlgError:
                slope = 0.0
            return float(slope)

        for window in cfg.trend_windows:
            data[f"volume_trend_{window}d"] = (
                grouped["volume"].rolling(window).apply(lambda x: _slope(pd.Series(x)), raw=False).reset_index(level=0, drop=True)
            )

        # Price/volume interaction
        data["price_volume"] = data["close"] * data["volume"]
        data["price_volume_sma_20"] = (
            grouped["price_volume"].rolling(20).mean().reset_index(level=0, drop=True)
        )
        
        # Liquidity: average dollar volume = average price * average daily volume
        for window in cfg.volume_windows:
            avg_price = grouped["close"].rolling(window).mean().reset_index(level=0, drop=True)
            avg_volume = grouped["volume"].rolling(window).mean().reset_index(level=0, drop=True)
            data[f"liquidity_{window}d"] = avg_price * avg_volume
            # Also add as average of price_volume directly (equivalent)
            data[f"avg_dollar_volume_{window}d"] = (
                grouped["price_volume"].rolling(window).mean().reset_index(level=0, drop=True)
            )

        self._summaries = [
            VolumeFeatureSummary(ticker=t, rows=len(group)) for t, group in data.groupby("ticker")
        ]
        return data

