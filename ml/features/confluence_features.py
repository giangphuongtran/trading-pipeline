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
    news: NewsFeatureEngineer


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
                news=NewsFeatureEngineer(news_config),
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

        price = self.components.price.create_features(daily_ohlcv)
        technical = self.components.technical.create_features(daily_ohlcv)
        technical = technical.drop(columns=["open", "high", "low", "close", "volume"], errors="ignore")

        merged = price.merge(
            technical,
            left_index=True,
            right_index=True,
            how="left",
            suffixes=("", "_tech"),
        )

        volume = self.components.volume.create_features(daily_ohlcv)
        volume = volume.drop(columns=["open", "high", "low", "close"], errors="ignore")
        merged = merged.merge(volume, left_index=True, right_index=True, how="left")

        if news_df is not None:
            news_features = self.components.news.create_features(news_df, daily_ohlcv[["ticker", "date"]])
            news_features = news_features.drop(columns=["ticker", "date"], errors="ignore")
            merged = merged.merge(news_features, left_index=True, right_index=True, how="left")

        if intraday_ohlcv is not None and not intraday_ohlcv.empty:
            candles = self.components.candlestick.create_features(intraday_ohlcv)
            timed = self.components.time.create_features(candles, timestamp_col="timestamp")
            intraday_daily = self._aggregate_intraday(timed)
            merged = merged.merge(
                intraday_daily,
                left_on=["ticker", "date"],
                right_on=["ticker", "date"],
                how="left",
            )

        merged = self._create_confluence(merged)
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
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

    def _create_confluence(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Volume confirmation
        cfg = self.confluence_config

        data["vol_confirm_primary"] = (data.get("volume_ratio", 0) >= cfg.volume_ratio_confirm).astype(int)
        data["vol_confirm_secondary"] = (data.get("volume_ratio", 0) >= cfg.volume_ratio_high).astype(int)
        data["vol_trend_up"] = (data.get("volume_trend_5d", 0) > 0).astype(int)

        # Momentum alignment
        data["momentum_rsi_confirm"] = (
            ((data.get("rsi_14") < cfg.rsi_oversold) & (data.get("price_change", 0) > 0))
            | ((data.get("rsi_14") > cfg.rsi_overbought) & (data.get("price_change", 0) < 0))
        ).astype(int)

        data["momentum_macd_cross"] = (
            (data.get("macd", 0) > data.get("macd_signal", 0))
            & (data.get("macd_histogram", 0) > 0)
        ).astype(int)

        # Trend context
        data["trend_price_above_sma20"] = (data.get("close_vs_sma20", 0) > 0).astype(int)
        data["trend_price_above_sma50"] = (data.get("close_vs_sma50", 0) > 0).astype(int)
        data["trend_strong_adx"] = (data.get("adx", 0) > cfg.adx_trend_threshold).astype(int)

        # Volatility context
        data["bb_reentry_signal"] = (
            (data.get("bb_position", 0) < 0)
            & (data.get("close_vs_sma20", 0) > 0)
        ).astype(int)
        data["atr_breakout"] = (
            data.get("true_range", 0) >= cfg.atr_breakout_multiplier * data.get("avg_true_range_14", np.nan)
        ).astype(int)

        # Support/resistance proximity
        price_range = data.get("price_vs_20d_range", np.nan)
        data["near_high_volume_node"] = (
            price_range.between(cfg.price_range_midband_low, cfg.price_range_midband_high, inclusive="both")
            & (data.get("volume_ratio", 0) > 1.0)
        ).astype(int)
        data["near_range_extreme"] = (
            (price_range <= cfg.price_range_extreme_low) | (price_range >= cfg.price_range_extreme_high)
        ).astype(int)

        # News alignment
        data["news_sentiment_positive"] = (data.get("sentiment_mean", 0) > 0).astype(int)
        data["news_sentiment_trending_up"] = (data.get("sentiment_trend", 0) > 0).astype(int)

        # Pattern strength
        pattern_strength = (
            data.get("hammer_count", 0)
            + data.get("bullish_engulf_count", 0)
            - data.get("bearish_engulf_count", 0)
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

