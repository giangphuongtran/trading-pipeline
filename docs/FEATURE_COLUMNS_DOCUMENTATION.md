# Feature Columns Documentation

This document describes all 103 columns in the exported feature dataset, including their calculation methods, data types, and usage recommendations.

## Table of Contents
- [Base Columns](#base-columns)
- [Price Features](#price-features)
- [Volume Features](#volume-features)
- [Technical Indicators](#technical-indicators)
- [Market Features](#market-features)
- [News Features](#news-features)
- [Confluence Features](#confluence-features)

---

## Base Columns

| Column Name | Data Type | Description | Calculation | Usage |
|------------|-----------|-------------|-------------|-------|
| `ticker` | string | Stock ticker symbol | From raw data | Primary identifier for grouping and filtering |
| `date` | datetime | Trading date | From raw data | Time index for time-series analysis |
| `open` | float64 | Opening price | From raw data | OHLCV base data |
| `high` | float64 | Highest price of the day | From raw data | OHLCV base data |
| `low` | float64 | Lowest price of the day | From raw data | OHLCV base data |
| `close` | float64 | Closing price | From raw data | OHLCV base data, primary price reference |
| `volume` | int64 | Trading volume (shares) | From raw data | OHLCV base data |

---

## Price Features

| Column Name | Data Type | Description | Calculation | Usage |
|------------|-----------|-------------|-------------|-------|
| `price_change` | float64 | Absolute price change | `close - close.shift(1)` | Measure day-to-day price movement |
| `daily_return` | float64 | Daily percentage return | `close.pct_change()` | Primary return metric for analysis |
| `return_5d` | float64 | 5-day return | `close.pct_change(5)` | Short-term momentum indicator |
| `return_10d` | float64 | 10-day return | `close.pct_change(10)` | Medium-term momentum indicator |
| `return_20d` | float64 | 20-day return | `close.pct_change(20)` | Long-term momentum indicator |
| `return_volatility_5d` | float64 | 5-day rolling volatility | `daily_return.rolling(5).std()` | Short-term risk measure |
| `return_volatility_20d` | float64 | 20-day rolling volatility | `daily_return.rolling(20).std()` | Long-term risk measure |
| `mean_daily_return_5d` | float64 | 5-day mean return | `daily_return.rolling(5).mean()` | Short-term expected return |
| `mean_daily_return_20d` | float64 | 20-day mean return | `daily_return.rolling(20).mean()` | Long-term expected return |
| `sharpe_ratio_5d` | float64 | 5-day Sharpe ratio (annualized) | `(mean_return / volatility) * sqrt(252)` | Risk-adjusted return (5-day) |
| `sharpe_ratio_20d` | float64 | 20-day Sharpe ratio (annualized) | `(mean_return / volatility) * sqrt(252)` | Risk-adjusted return (20-day) |
| `max_drawdown_5d` | float64 | 5-day maximum drawdown | Rolling min of `(cum_returns - peak) / peak` | Worst loss in 5-day window |
| `max_drawdown_20d` | float64 | 20-day maximum drawdown | Rolling min of `(cum_returns - peak) / peak` | Worst loss in 20-day window |
| `momentum_5d` | float64 | 5-day momentum | `close.pct_change(5)` | Same as `return_5d` |
| `momentum_10d` | float64 | 10-day momentum | `close.pct_change(10)` | Same as `return_10d` |
| `momentum_20d` | float64 | 20-day momentum | `close.pct_change(20)` | Same as `return_20d` |
| `sma_5` | float64 | 5-day Simple Moving Average | `close.rolling(5).mean()` | Short-term trend indicator |
| `sma_10` | float64 | 10-day Simple Moving Average | `close.rolling(10).mean()` | Medium-term trend indicator |
| `sma_20` | float64 | 20-day Simple Moving Average | `close.rolling(20).mean()` | Standard trend indicator |
| `sma_50` | float64 | 50-day Simple Moving Average | `close.rolling(50).mean()` | Long-term trend indicator |
| `sma_200` | float64 | 200-day Simple Moving Average | `close.rolling(200).mean()` | Very long-term trend indicator |
| `close_vs_sma5` | float64 | Price vs 5-day SMA ratio | `(close - sma_5) / sma_5` | Deviation from short-term trend |
| `close_vs_sma10` | float64 | Price vs 10-day SMA ratio | `(close - sma_10) / sma_10` | Deviation from medium-term trend |
| `close_vs_sma20` | float64 | Price vs 20-day SMA ratio | `(close - sma_20) / sma_20` | Deviation from standard trend |
| `close_vs_sma50` | float64 | Price vs 50-day SMA ratio | `(close - sma_50) / sma_50` | Deviation from long-term trend |
| `close_vs_sma200` | float64 | Price vs 200-day SMA ratio | `(close - sma_200) / sma_200` | Deviation from very long-term trend |
| `ema_12` | float64 | 12-day Exponential Moving Average | `close.ewm(span=12).mean()` | Short-term EMA trend |
| `ema_26` | float64 | 26-day Exponential Moving Average | `close.ewm(span=26).mean()` | Medium-term EMA trend |
| `close_vs_ema12` | float64 | Price vs 12-day EMA ratio | `(close - ema_12) / ema_12` | Deviation from short EMA |
| `close_vs_ema26` | float64 | Price vs 26-day EMA ratio | `(close - ema_26) / ema_26` | Deviation from medium EMA |
| `price_vs_5d_range` | float64 | Price position in 5-day range | `(close - min_low) / (max_high - min_low)` | Relative position (0-1) in short range |
| `price_vs_20d_range` | float64 | Price position in 20-day range | `(close - min_low) / (max_high - min_low)` | Relative position (0-1) in long range |
| `true_range` | float64 | True Range (ATR component) | `max(high-low, |high-prev_close|, |low-prev_close|)` | Daily volatility measure |
| `avg_true_range_14` | float64 | 14-day Average True Range | `true_range.rolling(14).mean()` | Standard volatility indicator |
| `tr_percent` | float64 | True Range as % of price | `true_range / close` | Normalized volatility |

---

## Volume Features

| Column Name | Data Type | Description | Calculation | Usage |
|------------|-----------|-------------|-------------|-------|
| `volume_sma_5` | float64 | 5-day volume moving average | `volume.rolling(5).mean()` | Short-term volume trend |
| `volume_sma_20` | float64 | 20-day volume moving average | `volume.rolling(20).mean()` | Long-term volume trend |
| `volume_ratio` | float64 | Current volume vs 20-day average | `volume / volume_sma_20` | Volume anomaly indicator (>1 = high, <1 = low) |
| `volume_spike` | int64 | Volume spike flag | `1 if volume_ratio > 2.0 else 0` | Binary: unusual volume activity |
| `volume_dry` | int64 | Low volume flag | `1 if volume_ratio < 0.5 else 0` | Binary: unusually low volume |
| `volume_trend_5d` | float64 | 5-day volume trend slope | Linear regression slope of volume | Volume trend direction |
| `volume_trend_20d` | float64 | 20-day volume trend slope | Linear regression slope of volume | Long-term volume trend |
| `price_volume` | float64 | Dollar volume | `close * volume` | Total trading value |
| `price_volume_sma_20` | float64 | 20-day average dollar volume | `price_volume.rolling(20).mean()` | Average trading value |
| `liquidity_5d` | float64 | 5-day average dollar volume | `avg_price_5d * avg_volume_5d` | Short-term liquidity measure |
| `liquidity_20d` | float64 | 20-day average dollar volume | `avg_price_20d * avg_volume_20d` | Long-term liquidity measure |
| `avg_dollar_volume_5d` | float64 | 5-day average dollar volume (alternative) | `price_volume.rolling(5).mean()` | Same as `liquidity_5d` |
| `avg_dollar_volume_20d` | float64 | 20-day average dollar volume (alternative) | `price_volume.rolling(20).mean()` | Same as `liquidity_20d` |

---

## Technical Indicators

| Column Name | Data Type | Description | Calculation | Usage |
|------------|-----------|-------------|-------------|-------|
| `rsi_14` | float64 | Relative Strength Index (14-day) | RSI formula using 14-period | Momentum oscillator (0-100, >70 overbought, <30 oversold) |
| `macd_12_26_9` | float64 | MACD line (12,26,9) | EMA(12) - EMA(26) | Trend-following momentum indicator |
| `macd_signal_12_26_9` | float64 | MACD signal line | EMA(9) of MACD line | Signal line for MACD crossovers |
| `macd_hist_12_26_9` | float64 | MACD histogram | `MACD - Signal` | Momentum strength indicator |
| `stoch_k_14_3_3` | float64 | Stochastic %K (14,3,3) | `(close - low_14) / (high_14 - low_14) * 100` | Momentum oscillator (0-100) |
| `stoch_d_14_3_3` | float64 | Stochastic %D (14,3,3) | 3-period SMA of %K | Smoothed %K, signal line |
| `bb_lower_20_2.0` | float64 | Bollinger Lower Band | `SMA(20) - 2*std(20)` | Lower volatility band |
| `bb_middle_20_2.0` | float64 | Bollinger Middle Band | `SMA(20)` | 20-day moving average |
| `bb_upper_20_2.0` | float64 | Bollinger Upper Band | `SMA(20) + 2*std(20)` | Upper volatility band |
| `bb_bandwidth_20_2.0` | float64 | Bollinger Bandwidth | `(upper - lower) / middle` | Volatility expansion/contraction |
| `bb_pct_20_2.0` | float64 | Bollinger %B | `(close - lower) / (upper - lower)` | Price position in bands (0-1) |
| `atr_14` | float64 | Average True Range (14-day) | `true_range.rolling(14).mean()` | Volatility measure (same as `avg_true_range_14`) |
| `atr_pct_14` | float64 | ATR as % of price | `atr_14 / close` | Normalized volatility |
| `adx_14` | float64 | Average Directional Index (14-day) | ADX formula using 14-period | Trend strength (0-100, >20 = strong trend) |
| `ema_14` | float64 | 14-day EMA (technical) | `close.ewm(span=14).mean()` | Technical indicator EMA |
| `sma_14` | float64 | 14-day SMA (technical) | `close.rolling(14).mean()` | Technical indicator SMA |
| `wma_14` | float64 | 14-day Weighted Moving Average | WMA formula with 14-period | Weighted trend indicator |

---

## Market Features

| Column Name | Data Type | Description | Calculation | Usage |
|------------|-----------|-------------|-------------|-------|
| `market_return` | float64 | Market index daily return | Market index `close.pct_change()` | Benchmark return (e.g., S&P 500) |
| `beta_20d` | float64 | 20-day rolling beta | `Cov(stock_return, market_return) / Var(market_return)` | Stock sensitivity to market (1.0 = moves with market) |
| `stock_vs_market_return` | float64 | Excess return vs market | `daily_return - market_return` | Relative performance |
| `relative_return` | float64 | Relative return ratio | `daily_return / market_return` | Performance ratio (when market_return ≠ 0) |

---

## News Features

| Column Name | Data Type | Description | Calculation | Usage |
|------------|-----------|-------------|-------------|-------|
| `sentiment_mean` | float64 | Average sentiment score | `mean(sentiment_score)` per day | Overall sentiment (-1 to 1, >0 positive) |
| `sentiment_std` | float64 | Sentiment score std dev | `std(sentiment_score)` per day | Sentiment consistency |
| `sentiment_count` | int64 | Number of news articles | `count(articles)` per day | News volume |
| `positive_count` | int64 | Positive articles count | Count where `sentiment_score > 0` | Positive news volume |
| `negative_count` | int64 | Negative articles count | Count where `sentiment_score < 0` | Negative news volume |
| `neutral_count` | int64 | Neutral articles count | Count where `sentiment_score = 0` | Neutral news volume |
| `news_count` | int64 | Total news articles | Same as `sentiment_count` | News volume (duplicate) |
| `sentiment_rolling_avg` | float64 | 7-day rolling sentiment | `sentiment_mean.rolling(7).mean()` | Smoothed sentiment trend |
| `news_rolling_count` | int64 | 7-day rolling news count | `sentiment_count.rolling(7).sum()` | Smoothed news volume |
| `sentiment_trend` | float64 | Sentiment trend slope | Linear regression slope of 7-day sentiment | Sentiment direction |
| `sentiment_llm_mean` | float64 | LLM-based sentiment mean | `mean(sentiment_llm_score)` per day | LLM sentiment (if enabled) |
| `sentiment_rule_mean` | float64 | Rule-based sentiment mean | `mean(sentiment_rule_score)` per day | Rule-based sentiment |
| `news_article_count` | int64 | Total articles per ticker | `count(articles)` per ticker | Aggregate news volume |

---

## Confluence Features

| Column Name | Data Type | Description | Calculation | Usage |
|------------|-----------|-------------|-------------|-------|
| `vol_confirm_primary` | int64 | Volume confirmation flag | `1 if volume_ratio >= 1.2 else 0` | Binary: volume supports move |
| `vol_confirm_secondary` | int64 | High volume confirmation | `1 if volume_ratio >= 1.5 else 0` | Binary: strong volume support |
| `vol_trend_up` | int64 | Volume trend up flag | `1 if volume_trend_5d > 0 else 0` | Binary: increasing volume |
| `momentum_rsi_confirm` | int64 | RSI momentum confirmation | `1 if (RSI<30 & price_up) or (RSI>70 & price_down) else 0` | Binary: RSI confirms price move |
| `momentum_macd_cross` | int64 | MACD bullish crossover | `1 if MACD > Signal & Histogram > 0 else 0` | Binary: bullish MACD signal |
| `trend_price_above_sma20` | int64 | Price above 20-day SMA | `1 if close > sma_20 else 0` | Binary: uptrend indicator |
| `trend_price_above_sma50` | int64 | Price above 50-day SMA | `1 if close > sma_50 else 0` | Binary: long-term uptrend |
| `trend_strong_adx` | int64 | Strong trend flag | `1 if adx_14 > 20 else 0` | Binary: strong directional trend |
| `bb_reentry_signal` | int64 | Bollinger reentry signal | `1 if bb_pct < 0 & price > sma_20 else 0` | Binary: oversold bounce signal |
| `atr_breakout` | int64 | ATR breakout flag | `1 if true_range >= 1.2 * avg_true_range_14 else 0` | Binary: volatility expansion |
| `near_high_volume_node` | int64 | Near high volume price level | `1 if price in mid-range & volume_ratio > 1.0 else 0` | Binary: support/resistance area |
| `near_range_extreme` | int64 | Near price range extreme | `1 if price in top/bottom 10% of range else 0` | Binary: extreme price position |
| `news_sentiment_positive` | int64 | Positive news sentiment | `1 if sentiment_mean > 0 else 0` | Binary: positive news |
| `news_sentiment_trending_up` | int64 | Sentiment improving | `1 if sentiment_trend > 0 else 0` | Binary: improving sentiment |
| `pattern_strength` | int64 | Candlestick pattern strength | `hammer_count + bullish_engulf - bearish_engulf` | Integer: net bullish patterns |
| `bullish_confluence_score` | float64 | Composite bullish score | Weighted sum of bullish signals | Score: 0-1, higher = more bullish |
| `bearish_confluence_score` | float64 | Composite bearish score | Weighted sum of bearish signals | Score: 0-1, higher = more bearish |

---

## Summary Statistics

- **Total Columns**: 103
- **Base Columns**: 7 (ticker, date, OHLCV)
- **Price Features**: 33
- **Volume Features**: 13
- **Technical Indicators**: 17
- **Market Features**: 4
- **News Features**: 13 (if news data included)
- **Confluence Features**: 16

## Notes

1. **Data Types**: 
   - `float64`: Continuous numerical values
   - `int64`: Integer values (counts, flags)
   - `string`: Text identifiers
   - `datetime`: Date/time values

2. **Missing Values**: 
   - Early rows may have NaN for rolling window features
   - News features default to 0.0/0 if no news data
   - Use `.dropna()` or forward-fill for time-series models

3. **Usage Recommendations**:
   - **Price features**: Use for trend and momentum analysis
   - **Volume features**: Use for confirmation and liquidity analysis
   - **Technical indicators**: Use for entry/exit signals
   - **Market features**: Use for relative performance and risk analysis
   - **News features**: Use for sentiment-driven strategies
   - **Confluence features**: Use for multi-factor signal confirmation

4. **Feature Scaling**: 
   - Consider standardizing features before ML models
   - Binary features (0/1) don't need scaling
   - Ratio features (e.g., `close_vs_sma20`) are already normalized

5. **Time-Series Considerations**:
   - All features are calculated per-ticker (grouped)
   - Rolling windows use historical data only (no lookahead bias)
   - Features are suitable for time-series cross-validation

