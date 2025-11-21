"""Market-related feature engineering: market index values and market return features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = ("ticker", "date", "close")


def _ensure_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    """Validate that DataFrame contains required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} requires columns {missing}, but DataFrame has {list(df.columns)}")


@dataclass(slots=True)
class MarketFeatureSummary:
    ticker: str
    rows: int


class MarketFeatureEngineer:
    """
    Engineer market-related features: market index close price and market returns.
    """

    def __init__(
        self,
        market_bars: pd.DataFrame,
    ):
        """
        Initialize market feature engineer.
        
        Args:
            market_bars: DataFrame with market index bars (must have 'date' and 'close' columns)
        """
        if market_bars.empty:
            raise ValueError("market_bars cannot be empty")
        
        _ensure_columns(market_bars, ("date", "close"), "MarketFeatureEngineer market_bars")
        
        # Prepare market data: ensure date is datetime, sort, calculate returns
        market_df = market_bars[["date", "close"]].copy()
        if not pd.api.types.is_datetime64_any_dtype(market_df["date"]):
            market_df["date"] = pd.to_datetime(market_df["date"])
        market_df = market_df.sort_values("date").reset_index(drop=True)
        market_df["market_return"] = market_df["close"].pct_change()
        
        # Rename close to market_close for clarity
        market_df = market_df.rename(columns={"close": "market_close"})
        
        self.market_data = market_df.set_index("date")[["market_close", "market_return"]]
        self._summaries: List[MarketFeatureSummary] = []

    @property
    def summaries(self) -> List[MarketFeatureSummary]:
        """Get summary statistics for processed tickers."""
        return self._summaries

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market-related features: market index close price and market returns.
        
        Args:
            df: DataFrame with ticker, date, close columns (and optionally daily_return)
            
        Returns:
            DataFrame with added market feature columns:
            - market_close: Market index close price for the date
            - market_return: Market index return for the date
            - stock_vs_market_return: Stock return minus market return
            - relative_return: Stock return / market return (when market return != 0)
        """
        if df.empty:
            return df.copy()

        _ensure_columns(df, _REQUIRED_COLUMNS, "MarketFeatureEngineer")
        
        data = df.sort_values(["ticker", "date"]).copy()
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(data["date"]):
            data["date"] = pd.to_datetime(data["date"])
        
        # Prepare market data for merge
        market_df_for_merge = self.market_data.reset_index().copy()
        
        # Align timezones for merge: convert both to same timezone format
        # If one has timezone and other doesn't, convert timezone-aware to timezone-naive
        # by extracting just the date part (since we're merging on dates, not timestamps)
        data_tz = data["date"].dt.tz
        market_tz = market_df_for_merge["date"].dt.tz
        
        if data_tz is not None and market_tz is None:
            # Data has timezone, market doesn't - convert data to naive
            data["date"] = data["date"].dt.tz_localize(None)
        elif data_tz is None and market_tz is not None:
            # Market has timezone, data doesn't - convert market to naive
            market_df_for_merge["date"] = market_df_for_merge["date"].dt.tz_localize(None)
        elif data_tz is not None and market_tz is not None:
            # Both have timezone - convert both to UTC then remove timezone for date comparison
            data["date"] = data["date"].dt.tz_convert("UTC").dt.tz_localize(None)
            market_df_for_merge["date"] = market_df_for_merge["date"].dt.tz_convert("UTC").dt.tz_localize(None)
        
        # Merge market data - only select market feature columns (exclude date which is the merge key)
        market_feature_cols = [col for col in market_df_for_merge.columns if col != "date"]
        market_to_merge = market_df_for_merge[["date"] + market_feature_cols].copy()
        
        data = data.merge(
            market_to_merge,
            on="date",
            how="left",
        )
        
        # Calculate stock returns if not present
        if "daily_return" not in data.columns:
            grouped = data.groupby("ticker", group_keys=False)
            data["daily_return"] = grouped["close"].pct_change()
        
        # Calculate relative returns
        data["stock_vs_market_return"] = data["daily_return"] - data["market_return"]
        data["relative_return"] = np.where(
            data["market_return"] != 0,
            data["daily_return"] / data["market_return"],
            np.nan
        )
        
        # Track summaries
        self._summaries = []
        for ticker, group in data.groupby("ticker"):
            self._summaries.append(
                MarketFeatureSummary(
                    ticker=ticker,
                    rows=len(group)
                )
            )
        
        return data

    # Backwards compatibility
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.create_features(df)

