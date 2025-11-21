"""Configurable news feature engineering from raw news data."""

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
    """Aggregate raw news data into daily features with configurable lookback."""

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
        Aggregate raw news data into daily features including Polygon sentiment data.
        
        Args:
            news_df: DataFrame with ticker, published_at, title, description, url, author, type, 
                    keywords, tickers, sentiment_score, sentiment_label, sentiment_reasoning
            daily_df: DataFrame with ticker, date columns to attach features to
            
        Returns:
            DataFrame with added news feature columns including Polygon sentiment aggregations
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

        # Aggregate news features
        features = self._aggregate_news_features(clean_news)
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
        """Attach blank features when no news data is available."""
        cfg = self.config
        return df.assign(
            news_count=cfg.fill_count,
            news_rolling_count=cfg.fill_count,
            avg_title_length=cfg.fill_numeric,
            avg_description_length=cfg.fill_numeric,
            unique_authors=cfg.fill_count,
            unique_types=cfg.fill_count,
            total_keywords=cfg.fill_count,
            total_related_tickers=cfg.fill_count,
            # Polygon sentiment features
            sentiment_score_mean=cfg.fill_numeric,
            sentiment_score_std=cfg.fill_numeric,
            positive_count=cfg.fill_count,
            negative_count=cfg.fill_count,
            neutral_count=cfg.fill_count,
            sentiment_rolling_avg=cfg.fill_numeric,
            sentiment_trend=cfg.fill_numeric,
        )

    def _aggregate_news_features(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate raw news data into daily features including Polygon sentiment."""
        # Calculate text lengths
        news_df = news_df.copy()
        news_df["title_length"] = news_df["title"].fillna("").astype(str).str.len()
        news_df["description_length"] = news_df["description"].fillna("").astype(str).str.len()
        
        # Count keywords and related tickers (arrays)
        news_df["keywords_count"] = news_df["keywords"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        news_df["related_tickers_count"] = news_df["tickers"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        # Count sentiment labels (from Polygon API)
        if "sentiment_label" in news_df.columns:
            news_df["is_positive"] = (news_df["sentiment_label"] == "positive").astype(int)
            news_df["is_negative"] = (news_df["sentiment_label"] == "negative").astype(int)
            news_df["is_neutral"] = (news_df["sentiment_label"] == "neutral").astype(int)

        aggregations = {
            "news_count": ("ticker", "count"),
            "avg_title_length": ("title_length", "mean"),
            "avg_description_length": ("description_length", "mean"),
            "unique_authors": ("author", "nunique"),
            "unique_types": ("type", "nunique"),
            "total_keywords": ("keywords_count", "sum"),
            "total_related_tickers": ("related_tickers_count", "sum"),
        }
        
        # Add Polygon sentiment aggregations if available
        if "sentiment_score" in news_df.columns:
            aggregations["sentiment_score_mean"] = ("sentiment_score", "mean")
            aggregations["sentiment_score_std"] = ("sentiment_score", "std")
        
        if "is_positive" in news_df.columns:
            aggregations["positive_count"] = ("is_positive", "sum")
            aggregations["negative_count"] = ("is_negative", "sum")
            aggregations["neutral_count"] = ("is_neutral", "sum")
        
        features = news_df.groupby(["ticker", "date"]).agg(**aggregations).reset_index()
        
        # Fill NaN std with 0 (happens when only one article per day)
        if "sentiment_score_std" in features.columns:
            features["sentiment_score_std"] = features["sentiment_score_std"].fillna(self.config.fill_numeric)
        
        return features

    def _create_rolling(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features including sentiment rolling averages."""
        cfg = self.config
        features_df = features_df.sort_values(["ticker", "date"]).copy()
        grouped = features_df.groupby("ticker", group_keys=False)

        features_df["news_rolling_count"] = (
            grouped["news_count"]
            .rolling(cfg.lookback_days, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        # Rolling sentiment average (from Polygon API)
        if "sentiment_score_mean" in features_df.columns:
            features_df["sentiment_rolling_avg"] = (
                grouped["sentiment_score_mean"]
                .rolling(cfg.lookback_days, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Sentiment trend (slope of sentiment over time)
            def _trend(group: pd.DataFrame) -> pd.Series:
                return (
                    group["sentiment_score_mean"]
                    .rolling(cfg.lookback_days, min_periods=cfg.trend_min_periods)
                    .apply(self._calculate_trend, raw=False)
                )

            features_df["sentiment_trend"] = (
                grouped.apply(_trend)
                .reset_index(level=0, drop=True)
                .fillna(cfg.fill_numeric)
            )

        return features_df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with defaults."""
        cfg = self.config
        numeric_defaults = {
            "news_count": cfg.fill_count,
            "news_rolling_count": cfg.fill_count,
            "avg_title_length": cfg.fill_numeric,
            "avg_description_length": cfg.fill_numeric,
            "unique_authors": cfg.fill_count,
            "unique_types": cfg.fill_count,
            "total_keywords": cfg.fill_count,
            "total_related_tickers": cfg.fill_count,
            # Polygon sentiment features
            "sentiment_score_mean": cfg.fill_numeric,
            "sentiment_score_std": cfg.fill_numeric,
            "positive_count": cfg.fill_count,
            "negative_count": cfg.fill_count,
            "neutral_count": cfg.fill_count,
            "sentiment_rolling_avg": cfg.fill_numeric,
            "sentiment_trend": cfg.fill_numeric,
        }
        for column, value in numeric_defaults.items():
            if column not in df.columns:
                df[column] = value
            else:
                df[column] = df[column].fillna(value)
        return df

    @staticmethod
    def _calculate_trend(values: pd.Series) -> float:
        """Calculate linear trend (slope) of sentiment over time."""
        cleaned = values.dropna()
        if len(cleaned) < 2:
            return 0.0
        x = np.arange(len(cleaned))
        try:
            slope, _ = np.polyfit(x, cleaned, 1)
        except np.linalg.LinAlgError:
            slope = 0.0
        return float(slope)
