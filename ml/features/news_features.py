"""Configurable news sentiment feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .indicator_config import NewsFeatureConfig


@dataclass(slots=True)
class NewsFeatureSummary:
    """Metadata about generated news features (useful for debugging)."""

    ticker: str | None
    rows: int


class NewsFeatureEngineer:
    """Aggregate news sentiment into daily features with configurable lookback."""

    def __init__(self, config: NewsFeatureConfig | None = None):
        """
        Initialize news feature engineer.
        
        Args:
            config: Configuration for aggregation windows (defaults to NewsFeatureConfig())
        """
        self.config = config or NewsFeatureConfig()
        self._summaries: List[NewsFeatureSummary] = []

    @property
    def summaries(self) -> List[NewsFeatureSummary]:
        """Get summary statistics for processed tickers."""
        return self._summaries

    def create_features(self, news_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate news sentiment into daily features.
        
        Args:
            news_df: DataFrame with ticker, published_at, sentiment_score, sentiment_label
            daily_df: DataFrame with ticker, date columns to attach features to
            
        Returns:
            DataFrame with added news sentiment feature columns
        """
        if daily_df.empty:
            return daily_df.copy()

        merged = daily_df.copy()
        merged["date"] = pd.to_datetime(merged["date"])

        if news_df.empty:
            return self._attach_blank_features(merged)

        clean_news = news_df.copy()
        clean_news["published_at"] = pd.to_datetime(clean_news.get("published_at", clean_news.get("date")))
        clean_news["date"] = clean_news["published_at"].dt.normalize()

        sentiment_daily = self._aggregate_sentiment(clean_news)
        label_counts = self._count_labels(clean_news)
        extra_scores = self._aggregate_additional_scores(clean_news)

        features = sentiment_daily.merge(label_counts, on=["ticker", "date"], how="left")
        features = features.merge(extra_scores, on=["ticker", "date"], how="left")
        features["news_count"] = features["sentiment_count"]
        features = self._create_rolling(features)

        # Align timezones for merge: ensure both date columns have same timezone format
        merged_tz = merged["date"].dt.tz
        features_tz = features["date"].dt.tz
        
        if merged_tz is not None and features_tz is None:
            # merged has timezone, features doesn't - convert merged to naive
            merged["date"] = merged["date"].dt.tz_localize(None)
        elif merged_tz is None and features_tz is not None:
            # features has timezone, merged doesn't - convert features to naive
            features["date"] = features["date"].dt.tz_localize(None)
        elif merged_tz is not None and features_tz is not None:
            # Both have timezone - convert both to UTC then remove timezone for date comparison
            merged["date"] = merged["date"].dt.tz_convert("UTC").dt.tz_localize(None)
            features["date"] = features["date"].dt.tz_convert("UTC").dt.tz_localize(None)

        merged = merged.merge(features, on=["ticker", "date"], how="left")
        merged = self._fill_missing(merged)

        self._summaries = [
            NewsFeatureSummary(ticker=t, rows=len(group))
            for t, group in merged.groupby("ticker")
        ]
        return merged

    # ------------------------------------------------------------------
    def _attach_blank_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        return df.assign(
            sentiment_mean=cfg.fill_numeric,
            sentiment_std=cfg.fill_numeric,
            sentiment_count=cfg.fill_count,
            positive_count=cfg.fill_count,
            negative_count=cfg.fill_count,
            neutral_count=cfg.fill_count,
            news_count=cfg.fill_count,
            sentiment_rolling_avg=cfg.fill_numeric,
            news_rolling_count=cfg.fill_count,
            sentiment_trend=cfg.fill_numeric,
            sentiment_llm_mean=cfg.fill_numeric,
            sentiment_rule_mean=cfg.fill_numeric,
        )

    def _aggregate_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        agg = (
            news_df.groupby(["ticker", "date"])
            .agg(sentiment_mean=("sentiment_score", "mean"), sentiment_std=("sentiment_score", "std"), sentiment_count=("sentiment_score", "count"))
            .reset_index()
        )
        agg["sentiment_std"] = agg["sentiment_std"].fillna(self.config.fill_numeric)
        return agg

    def _aggregate_additional_scores(self, news_df: pd.DataFrame) -> pd.DataFrame:
        aggregations = {}
        if "sentiment_llm_score" in news_df.columns:
            aggregations["sentiment_llm_mean"] = ("sentiment_llm_score", "mean")
        if "sentiment_rule_score" in news_df.columns:
            aggregations["sentiment_rule_mean"] = ("sentiment_rule_score", "mean")
        if not aggregations:
            return pd.DataFrame(columns=["ticker", "date"])

        extra = news_df.groupby(["ticker", "date"]).agg(**aggregations).reset_index()
        return extra

    def _count_labels(self, news_df: pd.DataFrame) -> pd.DataFrame:
        df = news_df.copy()
        df["is_positive"] = (df["sentiment_label"] == "positive").astype(int)
        df["is_negative"] = (df["sentiment_label"] == "negative").astype(int)
        df["is_neutral"] = (df["sentiment_label"] == "neutral").astype(int)

        counts = (
            df.groupby(["ticker", "date"])
            .agg(
                positive_count=("is_positive", "sum"),
                negative_count=("is_negative", "sum"),
                neutral_count=("is_neutral", "sum"),
            )
            .reset_index()
        )
        return counts

    def _create_rolling(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        sentiment_df = sentiment_df.sort_values(["ticker", "date"]).copy()
        grouped = sentiment_df.groupby("ticker", group_keys=False)

        sentiment_df["sentiment_rolling_avg"] = (
            grouped["sentiment_mean"]
            .rolling(cfg.lookback_days, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        sentiment_df["news_rolling_count"] = (
            grouped["news_count"]
            .rolling(cfg.lookback_days, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        def _trend(group: pd.DataFrame) -> pd.Series:
            return (
                group["sentiment_mean"]
                .rolling(cfg.lookback_days, min_periods=cfg.trend_min_periods)
                .apply(self._calculate_trend, raw=False)
            )

        sentiment_df["sentiment_trend"] = (
            grouped.apply(_trend)
            .reset_index(level=0, drop=True)
            .fillna(cfg.fill_numeric)
        )
        return sentiment_df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        numeric_defaults = {
            "sentiment_mean": cfg.fill_numeric,
            "sentiment_std": cfg.fill_numeric,
            "sentiment_count": cfg.fill_count,
            "positive_count": cfg.fill_count,
            "negative_count": cfg.fill_count,
            "neutral_count": cfg.fill_count,
            "news_count": cfg.fill_count,
            "sentiment_rolling_avg": cfg.fill_numeric,
            "news_rolling_count": cfg.fill_count,
            "sentiment_trend": cfg.fill_numeric,
            "sentiment_llm_mean": cfg.fill_numeric,
            "sentiment_rule_mean": cfg.fill_numeric,
        }
        for column, value in numeric_defaults.items():
            if column not in df.columns:
                df[column] = value
            else:
                df[column] = df[column].fillna(value)
        return df

    @staticmethod
    def _calculate_trend(values: pd.Series) -> float:
        cleaned = values.dropna()
        if len(cleaned) < 2:
            return 0.0
        x = np.arange(len(cleaned))
        try:
            slope, _ = np.polyfit(x, cleaned, 1)
        except np.linalg.LinAlgError:
            slope = 0.0
        return float(slope)

