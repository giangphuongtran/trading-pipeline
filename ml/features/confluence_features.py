"""Confluence feature engineering across price, volume, time, technical and news signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .candlestick_features import CandlestickFeatureEngineer
from .price_features import PriceFeatureEngineer
from .volume_features import VolumeFeatureEngineer
from .time_features import TimeFeatureEngineer
from .news_features import NewsFeatureEngineer
from .technical_indicators import TechnicalIndicatorsFeatureEngineer
from .indicator_config import (
    TechnicalIndicatorConfig,
    PriceFeatureConfig,
    VolumeFeatureConfig,
    TimeFeatureConfig,
    NewsFeatureConfig,
    ConfluenceConfig,
)


@dataclass
class ConfluenceComponents:
    price: PriceFeatureEngineer
    volume: VolumeFeatureEngineer
    candlestick: CandlestickFeatureEngineer
    technical: TechnicalIndicatorsFeatureEngineer
    time: TimeFeatureEngineer
    news: NewsFeatureEngineer | None


class ConfluenceFeatureEngineer:
    """
    Combine individual feature sets and construct confirmation / confluence metrics.
    """

    def __init__(
        self,
        price_config: PriceFeatureConfig | None = None,
        volume_config: VolumeFeatureConfig | None = None,
        technical_config: TechnicalIndicatorConfig | None = None,
        time_config: TimeFeatureConfig | None = None,
        news_config: NewsFeatureConfig | None = None,
        confluence_config: ConfluenceConfig | None = None,
        components: ConfluenceComponents | None = None,
    ):
        """
        Initialize confluence feature engineer.
        
        Args:
            price_config: Price feature configuration
            volume_config: Volume feature configuration
            technical_config: Technical indicators configuration
            time_config: Time feature configuration
            news_config: News feature configuration
            confluence_config: Confluence combination configuration
            components: Pre-initialized feature engineers (optional)
        """
        self.confluence_config = confluence_config or ConfluenceConfig()
        if components:
            self.components = components
        else:
            self.components = ConfluenceComponents(
                price=PriceFeatureEngineer(price_config),
                volume=VolumeFeatureEngineer(volume_config),
                candlestick=CandlestickFeatureEngineer(),
                technical=TechnicalIndicatorsFeatureEngineer(technical_config),
                time=TimeFeatureEngineer(time_config),
                news=NewsFeatureEngineer(news_config) if news_config is not None else None,
            )

    def create_features(
        self,
        daily_ohlcv: pd.DataFrame,
        intraday_ohlcv: Optional[pd.DataFrame] = None,
        news_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build combined feature set with confluence scores.
        """
        if daily_ohlcv.empty:
            return daily_ohlcv.copy()

        # Base columns that all feature sets need
        base_cols = ["ticker", "date"]
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        
        # Get price features - keep base + OHLCV + all feature columns
        price = self.components.price.create_features(daily_ohlcv)
        price_feature_cols = [col for col in price.columns if col not in base_cols + ohlcv_cols]
        price_to_merge = price[base_cols + ohlcv_cols + price_feature_cols].copy()
        
        # Get technical features - only select feature columns (exclude base and OHLCV)
        technical = self.components.technical.create_features(daily_ohlcv)
        technical_feature_cols = [col for col in technical.columns if col not in base_cols + ohlcv_cols]
        # Remove any columns that already exist in price to avoid duplicates
        technical_feature_cols = [col for col in technical_feature_cols if col not in price_to_merge.columns]
        technical_to_merge = technical[base_cols + technical_feature_cols].copy()

        # Merge price and technical on ticker and date
        if "ticker" in price_to_merge.columns and "date" in price_to_merge.columns:
            merged = price_to_merge.merge(
                technical_to_merge,
                on=base_cols,
                how="left",
            )
        else:
            # Fallback to index merge if columns not available
            merged = price_to_merge.merge(
                technical_to_merge,
                left_index=True,
                right_index=True,
                how="left",
            )

        # Get volume features - only select feature columns
        volume = self.components.volume.create_features(daily_ohlcv)
        volume_feature_cols = [col for col in volume.columns if col not in base_cols + ohlcv_cols]
        # Remove any columns that already exist in merged to avoid duplicates
        volume_feature_cols = [col for col in volume_feature_cols if col not in merged.columns]
        volume_to_merge = volume[base_cols + volume_feature_cols].copy()
        
        if "ticker" in merged.columns and "date" in merged.columns:
            merged = merged.merge(volume_to_merge, on=base_cols, how="left")
        else:
            merged = merged.merge(volume_to_merge, left_index=True, right_index=True, how="left")

        if news_df is not None and self.components.news is not None:
            news_features = self.components.news.create_features(news_df, daily_ohlcv[base_cols])
            # Only select feature columns (exclude base columns that are already in merged)
            news_feature_cols = [col for col in news_features.columns if col not in base_cols]
            # Remove any columns that already exist in merged to avoid duplicates
            news_feature_cols = [col for col in news_feature_cols if col not in merged.columns]
            news_to_merge = news_features[base_cols + news_feature_cols].copy()
            
            if "ticker" in merged.columns and "date" in merged.columns:
                merged = merged.merge(news_to_merge, on=base_cols, how="left")
            else:
                merged = merged.merge(news_to_merge, left_index=True, right_index=True, how="left")

        if intraday_ohlcv is not None and not intraday_ohlcv.empty:
            candles = self.components.candlestick.create_features(intraday_ohlcv)
            timed = self.components.time.create_features(candles, timestamp_col="timestamp")
            intraday_daily = self._aggregate_intraday(timed)
            # Only select feature columns (exclude base columns that are already in merged)
            intraday_feature_cols = [col for col in intraday_daily.columns if col not in base_cols]
            # Remove any columns that already exist in merged to avoid duplicates
            intraday_feature_cols = [col for col in intraday_feature_cols if col not in merged.columns]
            intraday_to_merge = intraday_daily[base_cols + intraday_feature_cols].copy()
            
            if "ticker" in merged.columns and "date" in merged.columns:
                merged = merged.merge(intraday_to_merge, on=base_cols, how="left")
            else:
                merged = merged.merge(intraday_to_merge, left_index=True, right_index=True, how="left")

        merged = self._create_confluence(merged)
        
        # Sort by ticker and date if they exist, otherwise just reset index
        if "ticker" in merged.columns and "date" in merged.columns:
            merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        else:
            merged = merged.reset_index(drop=True)
        return merged

    # ------------------------------------------------------------------
    def _aggregate_intraday(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby(["ticker", df["timestamp"].dt.normalize()])
        agg = grouped.agg(
            hammer_count=("is_hammer", "sum"),
            bullish_engulf_count=("is_bullish_engulfing", "sum"),
            bearish_engulf_count=("is_bearish_engulfing", "sum"),
            opening_activity=("is_opening_window", "sum"),
            closing_activity=("is_closing_window", "sum"),
            avg_body_size=("body_size_pct", "mean"),
            max_range_pct=("price_range_pct", "max"),
        ).reset_index().rename(columns={"timestamp": "date"})
        agg["date"] = pd.to_datetime(agg["date"])
        return agg

    def _get_column_or_default(self, data: pd.DataFrame, col_name: str, default: float = 0.0) -> pd.Series:
        """Get column from DataFrame or return Series of default values."""
        if col_name in data.columns:
            return data[col_name]
        return pd.Series(default, index=data.index)

    def _create_confluence(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Volume confirmation
        cfg = self.confluence_config

        data["vol_confirm_primary"] = (self._get_column_or_default(data, "volume_ratio", 0) >= cfg.volume_ratio_confirm).astype(int)
        data["vol_confirm_secondary"] = (self._get_column_or_default(data, "volume_ratio", 0) >= cfg.volume_ratio_high).astype(int)
        data["vol_trend_up"] = (self._get_column_or_default(data, "volume_trend_5d", 0) > 0).astype(int)

        # Momentum alignment
        data["momentum_rsi_confirm"] = (
            ((self._get_column_or_default(data, "rsi_14", 0) < cfg.rsi_oversold) & (self._get_column_or_default(data, "price_change", 0) > 0))
            | ((self._get_column_or_default(data, "rsi_14", 0) > cfg.rsi_overbought) & (self._get_column_or_default(data, "price_change", 0) < 0))
        ).astype(int)

        data["momentum_macd_cross"] = (
            (self._get_column_or_default(data, "macd", 0) > self._get_column_or_default(data, "macd_signal", 0))
            & (self._get_column_or_default(data, "macd_histogram", 0) > 0)
        ).astype(int)

        # Trend context
        data["trend_price_above_sma20"] = (self._get_column_or_default(data, "close_vs_sma20", 0) > 0).astype(int)
        data["trend_price_above_sma50"] = (self._get_column_or_default(data, "close_vs_sma50", 0) > 0).astype(int)
        data["trend_strong_adx"] = (self._get_column_or_default(data, "adx", 0) > cfg.adx_trend_threshold).astype(int)

        # Volatility context
        # Find bb_pct column dynamically (e.g., bb_pct_20_2.0)
        bb_pct_col = None
        for col in data.columns:
            if col.startswith("bb_pct_"):
                bb_pct_col = col
                break
        
        if bb_pct_col:
            data["bb_reentry_signal"] = (
                (self._get_column_or_default(data, bb_pct_col, 0) < 0)
                & (self._get_column_or_default(data, "close_vs_sma20", 0) > 0)
            ).astype(int)
        else:
            data["bb_reentry_signal"] = 0
        data["atr_breakout"] = (
            self._get_column_or_default(data, "true_range", 0) >= cfg.atr_breakout_multiplier * self._get_column_or_default(data, "avg_true_range_14", np.nan)
        ).astype(int)

        # Support/resistance proximity
        price_range = self._get_column_or_default(data, "price_vs_20d_range", np.nan)
        data["near_high_volume_node"] = (
            price_range.between(cfg.price_range_midband_low, cfg.price_range_midband_high, inclusive="both")
            & (self._get_column_or_default(data, "volume_ratio", 0) > 1.0)
        ).astype(int)
        data["near_range_extreme"] = (
            (price_range <= cfg.price_range_extreme_low) | (price_range >= cfg.price_range_extreme_high)
        ).astype(int)

        # News alignment
        data["news_sentiment_positive"] = (self._get_column_or_default(data, "sentiment_mean", 0) > 0).astype(int)
        data["news_sentiment_trending_up"] = (self._get_column_or_default(data, "sentiment_trend", 0) > 0).astype(int)

        # Pattern strength
        pattern_strength = (
            self._get_column_or_default(data, "hammer_count", 0)
            + self._get_column_or_default(data, "bullish_engulf_count", 0)
            - self._get_column_or_default(data, "bearish_engulf_count", 0)
        )
        data["pattern_strength"] = pattern_strength

        # Composite scores
        bull_w = cfg.bullish_weights
        bear_w = cfg.bearish_weights

        data["bullish_confluence_score"] = (
            bull_w.get("volume", 0) * data["vol_confirm_primary"]
            + bull_w.get("macd", 0) * data["momentum_macd_cross"]
            + bull_w.get("trend", 0) * data["trend_price_above_sma20"]
            + bull_w.get("rsi", 0) * data["momentum_rsi_confirm"]
            + bull_w.get("bollinger", 0) * data["bb_reentry_signal"]
            + bull_w.get("news", 0) * data["news_sentiment_positive"]
            + bull_w.get("patterns", 0) * pattern_strength.clip(lower=0)
        )

        data["bearish_confluence_score"] = (
            bear_w.get("volume", 0) * data["vol_confirm_primary"]
            + bear_w.get("trend", 0) * (1 - data["trend_price_above_sma20"])
            + bear_w.get("rsi", 0) * data["momentum_rsi_confirm"]
            + bear_w.get("atr", 0) * data["atr_breakout"]
            + bear_w.get("news", 0) * (1 - data["news_sentiment_positive"])
            + bear_w.get("patterns", 0) * (-pattern_strength).clip(lower=0)
        )

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        return data

