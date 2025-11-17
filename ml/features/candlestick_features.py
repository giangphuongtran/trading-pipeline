"""
Candlestick pattern feature engineering for intraday model.
"""

import pandas as pd
import numpy as np

class CandlestickFeatureEngineer:
    """
    Creates candlestick pattern features.

    Features:
    - Body size, shadows
    - Candlestick type (bullish, bearish, neutral)
    - Common patterns (hammer, engulfing, etc.)
    - Price momentum
    """

    def __init__(self):
        """Initialize the candlestick feature engineer."""
        pass

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all candlestick features from OHLCV data.

        Args:
        - df: pd.DataFrame with OHLCV data

        Returns:
        - pd.DataFrame with candlestick features
        """
        df = df.sort_values(['ticker', 'timestamp']).copy()
        df = self._create_basic_features(df)
        df = self._create_candlestick_type(df)
        df = self._create_price_momentum(df)
        df = self._detect_patterns(df)
        df = self._calculate_price_range(df)
        return df

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic candlestick features.
        
        Returns:
            DataFrame with columns:
            - body_size: |close - open|
            - body_size_pct: body_size / (high - low)
            - shadow_upper: high - max(close, open)
            - shadow_lower: min(close, open) - low
            - shadow_upper_pct: shadow_upper / (high - low)
            - shadow_lower_pct: shadow_lower / (high - low)
        """
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['body_size_pct'] = df['body_size'] / (df['high'] - df['low'])
        df['shadow_upper'] = df['high'] - np.maximum(df['close'], df['open'])
        df['shadow_upper_pct'] = df['shadow_upper'] / (df['high'] - df['low'])
        df['shadow_lower'] = np.minimum(df['close'], df['open']) - df['low']
        df['shadow_lower_pct'] = df['shadow_lower'] / (df['high'] - df['low'])
        return df

    def _create_candlestick_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create candlestick type flags.

        Returns:
            DataFrame with columns:
            - bullish: 1 if close > open, 0 otherwise
            - bearish: 1 if close < open, 0 otherwise
        """
        df['bullish'] = np.where(df['close'] > df['open'], 1, 0)
        df['bearish'] = np.where(df['close'] < df['open'], 1, 0)
        return df

    def _create_price_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price momentum features.

        Returns:
            DataFrame with columns:
            - price_change: close - open
            - price_change_5m: close - open for the last 5 candles
            - price_change_10m: close - open for the last 10 candles
        """
        grouped = df.groupby('ticker', group_keys=False)
        df['price_change'] = grouped['close'].pct_change()
        df['price_change_5m'] = grouped['close'].pct_change(5)
        df['price_change_10m'] = grouped['close'].pct_change(10)
        return df

    def _detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect common candlestick patterns.

        Returns:
            DataFrame with pattern flags:
            - is_bullish_engulfing: 1 if bullish engulfing pattern, 0 otherwise
            - is_bearish_engulfing: 1 if bearish engulfing pattern, 0 otherwise
            - is_hammer: 1 if hammer pattern, 0 otherwise
            - is_inverted_hammer: 1 if inverted hammer pattern, 0 otherwise
            - is_shooting_star: 1 if shooting star pattern, 0 otherwise
            - is_hanging_man: 1 if hanging man pattern, 0 otherwise
            - is_bullish_morning_star: 1 if bullish morning star pattern, 0 otherwise
            - is_bearish_evening_star: 1 if bearish evening star pattern, 0 otherwise
            - is_dragonfly_doji: 1 if dragonfly doji pattern, 0 otherwise
            - is_gravestone_doji: 1 if gravestone doji pattern, 0 otherwise
            - is_long_legged_doji: 1 if long legged doji pattern, 0 otherwise
            - is_piercing_pattern: 1 if piercing pattern, 0 otherwise
            - is_dark_cloud_cover: 1 if dark cloud cover pattern, 0 otherwise
            - is_three_black_crows: 1 if three black crows pattern, 0 otherwise
            - is_three_white_soldiers: 1 if three white soldiers pattern, 0 otherwise
        """
        df['is_bullish_engulfing'] = np.where(df['close'] > df['open'] and df['close'].shift(1) < df['open'].shift(1), 1, 0)
        df['is_bearish_engulfing'] = np.where(df['close'] < df['open'] and df['close'].shift(1) > df['open'].shift(1), 1, 0)
        df['is_hammer'] = np.where(df['close'] < df['open'] and df['high'] - df['close'] < 0.05 * (df['high'] - df['low']), 1, 0)
        df['is_inverted_hammer'] = np.where(df['close'] > df['open'] and df['close'] - df['low'] < 0.05 * (df['high'] - df['low']), 1, 0)
        df['is_shooting_star'] = np.where(df['close'] < df['open'] and df['close'] - df['low'] < 0.05 * (df['high'] - df['low']), 1, 0)
        df['is_hanging_man'] = np.where(df['close'] < df['open'] and df['close'] - df['low'] < 0.05 * (df['high'] - df['low']), 1, 0)
        df['is_bullish_morning_star'] = np.where(df['close'] > df['open'] and df['close'].shift(1) < df['open'].shift(1) and df['close'].shift(2) > df['open'].shift(2), 1, 0)
        df['is_bearish_evening_star'] = np.where(df['close'] < df['open'] and df['close'].shift(1) > df['open'].shift(1) and df['close'].shift(2) < df['open'].shift(2), 1, 0)
        df['is_dragonfly_doji'] = np.where(df['close'] == df['open'] and df['high'] - df['close'] < 0.05 * (df['high'] - df['low']), 1, 0)
        df['is_gravestone_doji'] = np.where(df['close'] == df['open'] and df['close'] - df['low'] < 0.05 * (df['high'] - df['low']), 1, 0)
        df['is_long_legged_doji'] = np.where(df['close'] == df['open'] and df['high'] - df['close'] < 0.05 * (df['high'] - df['low']), 1, 0)
        df['is_piercing_pattern'] = np.where(df['close'] < df['open'] and df['close'].shift(1) > df['open'].shift(1) and df['close'] > df['open'].shift(1), 1, 0)
        df['is_dark_cloud_cover'] = np.where(df['close'] > df['open'] and df['close'].shift(1) < df['open'].shift(1) and df['close'] < df['open'].shift(1), 1, 0)
        df['is_three_black_crows'] = np.where(df['close'] < df['open'] and df['close'].shift(1) < df['open'].shift(1) and df['close'].shift(2) < df['open'].shift(2), 1, 0)
        df['is_three_white_soldiers'] = np.where(df['close'] > df['open'] and df['close'].shift(1) > df['open'].shift(1) and df['close'].shift(2) > df['open'].shift(2), 1, 0)
        return df

    def _calculate_price_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price range features.

        Returns:
            DataFrame with columns:
            - price_range: high - low
            - price_range_pct: price_range / (high - low)
        """
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = df['price_range'] / (df['high'] - df['low'])
        return df