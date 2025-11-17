"""Technical indicators feature engineering with configurable parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import pandas_ta as ta

from .indicator_config import TechnicalIndicatorConfig

_REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], indicator: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{indicator} requires columns {missing}, but DataFrame has {list(df.columns)}")


@dataclass(slots=True)
class IndicatorResult:
    """Container for debugging/logging indicator calculations."""

    name: str
    columns_added: list[str]


class TechnicalIndicatorsFeatureEngineer:
    """Compute a configurable set of technical indicators for OHLCV data."""

    def __init__(self, config: TechnicalIndicatorConfig | None = None):
        """
        Initialize technical indicators engineer.
        
        Args:
            config: Configuration for indicators (defaults to TechnicalIndicatorConfig())
        """
        self.config = config or TechnicalIndicatorConfig()
        self._results: list[IndicatorResult] = []

    @property
    def results(self) -> list[IndicatorResult]:
        """Get metadata about the most recent feature generation."""
        return self._results

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators: RSI, MACD, Stochastic, Bollinger Bands, ATR, ADX.
        
        Args:
            df: DataFrame with ticker, date, open, high, low, close, volume columns
            
        Returns:
            DataFrame with added indicator columns
        """
        if df.empty:
            return df.copy()

        _ensure_columns(df, _REQUIRED_OHLCV_COLUMNS, "technical indicators")
        features = df.copy()
        self._results = []

        self._create_rsi(features)
        self._create_macd(features)
        self._create_stochastic(features)
        self._create_bollinger_bands(features)
        self._create_atr(features)
        self._create_adx(features)
        self._create_ema(features)
        self._create_sma(features)
        self._create_wma(features)

        return features

    # --- individual indicator helpers -------------------------------------------------

    def _create_rsi(self, df: pd.DataFrame) -> None:
        length = self.config.rsi_length
        series = ta.rsi(df["close"], length=length)
        if series is None:
            return
        column = f"rsi_{length}"
        df[column] = series
        df["rsi"] = series  # legacy name
        self._results.append(IndicatorResult("RSI", [column, "rsi"]))

    def _create_macd(self, df: pd.DataFrame) -> None:
        cfg = self.config
        macd = ta.macd(
            df["close"],
            fast=cfg.macd_fast,
            slow=cfg.macd_slow,
            signal=cfg.macd_signal,
        )
        if macd is None or macd.empty:
            return
        fast, slow, signal = cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
        base = f"{fast}_{slow}_{signal}"

        macd_col = f"MACD_{base}"
        sig_col = f"MACDs_{base}"
        hist_col = f"MACDh_{base}"

        mappings = []
        if macd_col in macd.columns:
            df[f"macd_{base}"] = macd[macd_col]
            df["macd"] = macd[macd_col]
            mappings.append(f"macd_{base}")
        if sig_col in macd.columns:
            df[f"macd_signal_{base}"] = macd[sig_col]
            df["macd_signal"] = macd[sig_col]
            mappings.append(f"macd_signal_{base}")
        if hist_col in macd.columns:
            df[f"macd_hist_{base}"] = macd[hist_col]
            df["macd_histogram"] = macd[hist_col]
            mappings.append(f"macd_hist_{base}")
        if mappings:
            self._results.append(IndicatorResult("MACD", mappings + ["macd", "macd_signal", "macd_histogram"]))

    def _create_stochastic(self, df: pd.DataFrame) -> None:
        cfg = self.config
        stoch = ta.stoch(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            k=cfg.stochastic_k,
            d=cfg.stochastic_d,
            smooth_k=cfg.stochastic_smooth,
        )
        if stoch is None or stoch.empty:
            return
        base = f"{cfg.stochastic_k}_{cfg.stochastic_d}_{cfg.stochastic_smooth}"

        k_col = stoch.columns[0]
        d_col = stoch.columns[1] if len(stoch.columns) > 1 else None

        df[f"stoch_k_{base}"] = stoch[k_col]
        df["stoch_k"] = stoch[k_col]
        added = [f"stoch_k_{base}", "stoch_k"]

        if d_col:
            df[f"stoch_d_{base}"] = stoch[d_col]
            df["stoch_d"] = stoch[d_col]
            added.extend([f"stoch_d_{base}", "stoch_d"])

        self._results.append(IndicatorResult("Stochastic", added))

    def _create_bollinger_bands(self, df: pd.DataFrame) -> None:
        cfg = self.config
        bbands = ta.bbands(df["close"], length=cfg.bollinger_length, std=cfg.bollinger_std)
        if bbands is None or bbands.empty:
            return
        base = f"{cfg.bollinger_length}_{cfg.bollinger_std}"

        lower_col = bbands.columns[0]
        middle_col = bbands.columns[1]
        upper_col = bbands.columns[2]
        width_col = bbands.columns[3]
        pct_col = bbands.columns[4] if len(bbands.columns) > 4 else None

        df[f"bb_lower_{base}"] = bbands[lower_col]
        df[f"bb_middle_{base}"] = bbands[middle_col]
        df[f"bb_upper_{base}"] = bbands[upper_col]
        df[f"bb_bandwidth_{base}"] = bbands[width_col]

        df["bbands_lower"] = bbands[lower_col]
        df["bbands_middle"] = bbands[middle_col]
        df["bbands_upper"] = bbands[upper_col]
        df["bbands_width"] = (df["bbands_upper"] - df["bbands_lower"]) / (df["bbands_middle"])

        if pct_col:
            df[f"bb_pct_{base}"] = bbands[pct_col]
            df["bb_position"] = bbands[pct_col]
            added_pct = [f"bb_pct_{base}", "bb_position"]
        else:
            # Compute pct position manually
            position = (df["close"] - df["bbands_lower"]) / (df["bbands_upper"] - df["bbands_lower"])
            df["bb_position"] = position
            df[f"bb_pct_{base}"] = position
            added_pct = [f"bb_pct_{base}", "bb_position"]

        self._results.append(
            IndicatorResult(
                "BollingerBands",
                [
                    f"bb_lower_{base}",
                    f"bb_middle_{base}",
                    f"bb_upper_{base}",
                    f"bb_bandwidth_{base}",
                    "bbands_lower",
                    "bbands_middle",
                    "bbands_upper",
                    "bbands_width",
                ]
                + added_pct,
            )
        )

    def _create_atr(self, df: pd.DataFrame) -> None:
        length = self.config.atr_length
        atr = ta.atr(df["high"], df["low"], df["close"], length=length)
        if atr is None:
            return
        col = f"atr_{length}"
        if isinstance(atr, pd.DataFrame):
            atr_series = atr.iloc[:, 0]
        else:
            atr_series = atr
        df[col] = atr_series
        df[f"atr_pct_{length}"] = atr_series / df["close"]
        df["atr"] = atr_series
        df["atr_pct"] = df[f"atr_pct_{length}"]
        self._results.append(IndicatorResult("ATR", [col, f"atr_pct_{length}", "atr", "atr_pct"]))

    def _create_adx(self, df: pd.DataFrame) -> None:
        length = self.config.adx_length
        adx = ta.adx(df["high"], df["low"], df["close"], length=length)
        if adx is None or adx.empty:
            return
        adx_col = f"ADX_{length}"
        if adx_col in adx.columns:
            df[f"adx_{length}"] = adx[adx_col]
            df["adx"] = adx[adx_col]
            self._results.append(IndicatorResult("ADX", [f"adx_{length}", "adx"]))

    def _create_ema(self, df: pd.DataFrame) -> None:
        length = self.config.ema_length
        ema = ta.ema(df["close"], length=length)
        if ema is None:
            return
        col = f"ema_{length}"
        df[col] = ema
        df["ema"] = ema
        self._results.append(IndicatorResult("EMA", [col, "ema"]))

    def _create_sma(self, df: pd.DataFrame) -> None:
        length = self.config.sma_length
        sma = ta.sma(df["close"], length=length)
        if sma is None:
            return
        col = f"sma_{length}"
        df[col] = sma
        df["sma"] = sma
        self._results.append(IndicatorResult("SMA", [col, "sma"]))

    def _create_wma(self, df: pd.DataFrame) -> None:
        length = self.config.wma_length
        wma = ta.wma(df["close"], length=length)
        if wma is None:
            return
        col = f"wma_{length}"
        df[col] = wma
        df["wma"] = wma
        self._results.append(IndicatorResult("WMA", [col, "wma"]))

    # Backwards compatibility helper ---------------------------------------------------

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for :meth:`create_features` to keep the previous API stable."""
        return self.create_features(df)