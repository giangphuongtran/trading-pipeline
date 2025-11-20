"""Market-related feature engineering: beta calculation and market return features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from app.symbols import get_market_index


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
    beta_calculated: bool


class MarketFeatureEngineer:
    """
    Engineer market-related features: beta (slope of stock return vs market return),
    market returns, and relative performance metrics.
    """

    def __init__(
        self,
        market_bars: pd.DataFrame,
        *,
        beta_window: int = 252,  # 1 year of trading days
        min_periods: Optional[int] = None,
    ):
        """
        Initialize market feature engineer.
        
        Args:
            market_bars: DataFrame with market index bars (must have 'date' and 'close' columns)
            beta_window: Rolling window size for beta calculation (default: 252 trading days)
            min_periods: Minimum periods required for beta calculation (default: beta_window // 2)
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
        
        self.market_data = market_df.set_index("date")[["close", "market_return"]]
        self.beta_window = beta_window
        self.min_periods = min_periods or (beta_window // 2)
        self._summaries: List[MarketFeatureSummary] = []

    @property
    def summaries(self) -> List[MarketFeatureSummary]:
        """Get summary statistics for processed tickers."""
        return self._summaries

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market-related features: beta, market returns, relative returns.
        
        Args:
            df: DataFrame with ticker, date, close columns (and optionally daily_return)
            
        Returns:
            DataFrame with added market feature columns:
            - market_return: Market index return for the date
            - beta_{window}d: Rolling beta coefficient
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
        
        # Calculate rolling beta for each ticker
        grouped = data.groupby("ticker", group_keys=False)
        beta_col = f"beta_{self.beta_window}d"
        data[beta_col] = grouped.apply(
            lambda group: self._calculate_rolling_beta(
                group["daily_return"].values,
                group["market_return"].values,
                self.beta_window,
                self.min_periods
            )
        ).reset_index(level=0, drop=True)
        
        # Drop market close column (keep only market_return)
        if "close_market" in data.columns:
            data = data.drop(columns=["close_market"])
        
        # Track summaries
        self._summaries = []
        for ticker, group in data.groupby("ticker"):
            beta_calculated = group[beta_col].notna().any()
            self._summaries.append(
                MarketFeatureSummary(
                    ticker=ticker,
                    rows=len(group),
                    beta_calculated=beta_calculated
                )
            )
        
        return data

    @staticmethod
    def _calculate_rolling_beta(
        stock_returns: np.ndarray,
        market_returns: np.ndarray,
        window: int,
        min_periods: int,
    ) -> pd.Series:
        """
        Calculate rolling beta using linear regression.
        
        Beta is the slope of: stock_return = alpha + beta * market_return
        
        Args:
            stock_returns: Array of stock returns
            market_returns: Array of market returns
            window: Rolling window size
            min_periods: Minimum periods required
            
        Returns:
            Series of beta values
        """
        n = len(stock_returns)
        betas = np.full(n, np.nan)
        
        for i in range(min_periods - 1, n):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            stock_window = stock_returns[start_idx:end_idx]
            market_window = market_returns[start_idx:end_idx]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(stock_window) | np.isnan(market_window))
            if valid_mask.sum() < min_periods:
                continue
            
            stock_valid = stock_window[valid_mask]
            market_valid = market_window[valid_mask]
            
            # Calculate beta using linear regression
            if len(stock_valid) >= 2 and market_valid.std() > 1e-10:
                slope, _, _, _, _ = stats.linregress(market_valid, stock_valid)
                betas[i] = slope
        
        return pd.Series(betas, index=range(n))

    # Backwards compatibility
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.create_features(df)

