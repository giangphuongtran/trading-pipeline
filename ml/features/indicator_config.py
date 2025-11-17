"""Configuration helpers for feature engineering modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TechnicalIndicatorConfig:
    """Tunable parameters for the technical indicator feature engineer."""

    rsi_length: int = 14

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    stochastic_k: int = 14
    stochastic_d: int = 3
    stochastic_smooth: int = 3

    bollinger_length: int = 20
    bollinger_std: float = 2.0

    atr_length: int = 14
    adx_length: int = 14

    ema_length: int = 14
    sma_length: int = 14
    wma_length: int = 14

    def as_dict(self) -> dict[str, int | float]:
        """Convenience helper for serialization or logging."""
        return {
            "rsi_length": self.rsi_length,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "stochastic_k": self.stochastic_k,
            "stochastic_d": self.stochastic_d,
            "stochastic_smooth": self.stochastic_smooth,
            "bollinger_length": self.bollinger_length,
            "bollinger_std": self.bollinger_std,
            "atr_length": self.atr_length,
            "adx_length": self.adx_length,
            "ema_length": self.ema_length,
            "sma_length": self.sma_length,
            "wma_length": self.wma_length,
        }


@dataclass
class PriceFeatureConfig:
    """Parameters controlling price-based feature engineering."""

    return_windows: Tuple[int, ...] = (5, 10, 20)
    sma_windows: Tuple[int, ...] = (5, 10, 20, 50, 200)
    ema_windows: Tuple[int, ...] = (12, 26)
    volatility_windows: Tuple[int, ...] = (5, 20)
    price_position_windows: Tuple[int, ...] = (5, 20)
    true_range_window: int = 14
    keep_na: bool = False


@dataclass
class VolumeFeatureConfig:
    """Parameters for volume feature engineering."""

    volume_windows: Tuple[int, ...] = (5, 20)
    trend_windows: Tuple[int, ...] = (5, 20)
    ratio_baseline_window: int = 20
    spike_threshold: float = 2.0
    dry_threshold: float = 0.5


@dataclass
class TimeFeatureConfig:
    """Session boundary parameters for time-based features."""

    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0
    opening_window_minutes: int = 30
    closing_window_minutes: int = 30
    lunch_hours: Tuple[int, int] = (11, 14)
    morning_hours: Tuple[int, int] = (9, 12)
    afternoon_hours: Tuple[int, int] = (12, 16)
    session_timezone: str = "America/New_York"


@dataclass
class NewsFeatureConfig:
    """Parameters that govern news sentiment aggregation."""

    lookback_days: int = 7
    trend_min_periods: int = 2
    fill_numeric: float = 0.0
    fill_count: int = 0


@dataclass
class ConfluenceConfig:
    """Weights and thresholds used when composing confluence signals."""

    volume_ratio_confirm: float = 1.2
    volume_ratio_high: float = 1.5

    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    adx_trend_threshold: float = 20.0
    atr_breakout_multiplier: float = 1.2

    price_range_midband_low: float = 0.4
    price_range_midband_high: float = 0.6
    price_range_extreme_low: float = 0.1
    price_range_extreme_high: float = 0.9

    bullish_weights: dict[str, float] = field(
        default_factory=lambda: {
            "volume": 0.3,
            "macd": 0.2,
            "trend": 0.2,
            "rsi": 0.1,
            "bollinger": 0.1,
            "news": 0.1,
            "patterns": 0.1,
        }
    )

    bearish_weights: dict[str, float] = field(
        default_factory=lambda: {
            "volume": 0.3,
            "trend": 0.2,
            "rsi": 0.2,
            "atr": 0.1,
            "news": 0.1,
            "patterns": 0.1,
        }
    )

