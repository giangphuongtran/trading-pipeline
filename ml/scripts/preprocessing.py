import pandas as pd
import numpy as np
from scipy import stats
import argparse
from pathlib import Path

def load_and_prep_data(file_path):
    
    print(f"Loading raw data from {file_path}")
    df = pd.read_parquet(file_path)
    print(f"Number of rows before drop NaN: {df.shape[0]}")
    df = df.dropna()
    print(f"Number of rows after drop NaN: {df.shape[0]}")
    df = df.sort_values(['ticker', 'date'])
    
    return df

def get_distribution_stats(series):
    clean_series = series.dropna()
    if len(clean_series) < 3:
        return 0.0, 0.0
    return stats.skew(clean_series), stats.kurtosis(clean_series)

def calc_rolling_beta_vectorized(df, window=20):
    """
    Calculate Rolling Beta efficiently using vectorized operations.
    
    Formula: Beta = Cov(Stock, Market) / Var(Market)
    Using: Cov(X,Y) = E[XY] - E[X]E[Y] and Var(Y) = E[Y²] - E[Y]²
    
    This is much faster than rolling().cov() which is O(n*w) per group.
    """
    beta_list = []
    tickers = df['ticker'].unique()
    total_tickers = len(tickers)
    
    # Process each ticker group separately
    for idx, (ticker, group) in enumerate(df.groupby('ticker'), 1):
        if idx % 10 == 0 or idx == total_tickers:
            print(f"  Processing beta for ticker {idx}/{total_tickers} ({ticker})...")
        stock_ret = group['daily_return'].values
        market_ret = group['market_return'].values
        n = len(group)
        
        # Pre-allocate beta array
        beta = np.full(n, np.nan)
        
        # Calculate rolling beta for each window
        for i in range(window - 1, n):
            # Get window data
            stock_window = stock_ret[i - window + 1:i + 1]
            market_window = market_ret[i - window + 1:i + 1]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(stock_window) | np.isnan(market_window))
            if valid_mask.sum() < 2:  # Need at least 2 points
                continue
            
            stock_valid = stock_window[valid_mask]
            market_valid = market_window[valid_mask]
            
            # Calculate covariance and variance
            if len(stock_valid) >= 2 and market_valid.std() > 1e-10:
                # Covariance = E[XY] - E[X]E[Y]
                mean_product = np.mean(stock_valid * market_valid)
                mean_stock = np.mean(stock_valid)
                mean_market = np.mean(market_valid)
                cov = mean_product - (mean_stock * mean_market)
                
                # Variance = E[Y²] - E[Y]²
                var_market = np.var(market_valid, ddof=0)  # Population variance
                
                if var_market > 1e-10:
                    beta[i] = cov / var_market
        
        # Create Series with same index as group
        beta_series = pd.Series(beta, index=group.index)
        beta_list.append(beta_series)
    
    # Concatenate all beta series and align with original dataframe index
    beta_series = pd.concat(beta_list)
    return beta_series.reindex(df.index)

def calc_rolling_features_vectorized(df):
    """
    Calculate Rolling Beta 20d and Volatility Ratio efficiently.
    
    This version uses vectorized operations instead of groupby().apply()
    which is much faster for large datasets.
    """
    print("Calculating rolling beta (this may take a moment)...")
    
    # Calculate beta using optimized method
    df['beta_20d'] = calc_rolling_beta_vectorized(df, window=20)
    
    # Volatility ratio: short-term / long-term volatility
    # Use np.where to avoid replace() on entire series
    df['volatility_ratio'] = np.where(
        df['return_volatility_20d'] != 0,
        df['return_volatility_5d'] / df['return_volatility_20d'],
        np.nan
    )
    
    return df

def global_dataset(df):
    
    df = df.copy()
    agg_results = []

    for ticker, group in df.groupby('ticker'):
        market_var = group['market_return'].var()
        if market_var > 0:
            global_beta = group['daily_return'].cov(group['market_return']) / market_var
        else:
            global_beta = np.nan
        ret_skew, ret_kurt = get_distribution_stats(group['daily_return'])
        
        record = {
            'ticker': ticker,
            # Price & Returns 
            'mean_daily_return': group['daily_return'].mean(),
            'mean_momentum_20d': group['momentum_20d'].mean(),
            'mean_close_vs_sma200': group['close_vs_sma200'].mean(),
            
            # Market Context 
            'mean_stock_vs_market': group['stock_vs_market_return'].mean(),
            'beta_global': global_beta,
            
            # Volatility & Risk 
            'mean_volatility_20d': group['return_volatility_20d'].mean(),
            'mean_sharpe_20d': group['sharpe_ratio_20d'].mean(),
            'worst_drawdown': group['max_drawdown_20d'].min(),
            'mean_atr_14': group['avg_true_range_14'].mean(),
            'mean_volatility_ratio': group['volatility_ratio'].mean(),
            
            # Technicals 
            'mean_rsi_14': group['rsi_14'].mean(),
            'mean_adx_14': group['adx_14'].mean(),
            'mean_macd_hist': group['macd_hist_12_26_9'].mean(),
            'mean_stoch_k': group['stoch_k_14_3_3'].mean(),
            'mean_bb_width': group['bb_bandwidth_20_2.0'].mean(),
            
            # Volume & Liquidity 
            'mean_liquidity_20d': group['liquidity_20d'].mean(),
            'mean_volume_ratio': group['volume_ratio'].mean(),
            'mean_price_pos_20d': group['price_vs_20d_range'].mean(),
            
            # Distribution 
            'return_skewness': ret_skew,
            'return_kurtosis': ret_kurt
        }
        agg_results.append(record)
    return pd.DataFrame(agg_results)

def quarterly_dataset(df):
    df = df.copy()
    df['quarter'] = df['date'].dt.to_period('Q').astype(str)
    
    agg_results = []
    
    for (ticker, quarter), group in df.groupby(['ticker', 'quarter']):
        ret_skew, ret_kurt = get_distribution_stats(group['daily_return'])
        
        record = {
            'ticker': ticker,
            'quarter': quarter,
            
            # Price & Returns
            'mean_daily_return': group['daily_return'].mean(),
            'mean_momentum_20d': group['momentum_20d'].mean(),
            'mean_close_vs_sma200': group['close_vs_sma200'].mean(),
            
            # Market Context
            'mean_stock_vs_market': group['stock_vs_market_return'].mean(),
            'beta_20d_mean': group['beta_20d'].mean(),
            'beta_20d_std': group['beta_20d'].std(),
            
            # Volatility & Risk
            'mean_volatility_20d': group['return_volatility_20d'].mean(),
            'mean_sharpe_20d': group['sharpe_ratio_20d'].mean(),
            'worst_drawdown': group['max_drawdown_20d'].min(),
            'mean_atr_14': group['avg_true_range_14'].mean(),
            'mean_volatility_ratio': group['volatility_ratio'].mean(),
            
            # Technicals
            'mean_rsi_14': group['rsi_14'].mean(),
            'mean_adx_14': group['adx_14'].mean(),
            'mean_macd_hist': group['macd_hist_12_26_9'].mean(),
            'mean_stoch_k': group['stoch_k_14_3_3'].mean(),
            'mean_bb_width': group['bb_bandwidth_20_2.0'].mean(),
            
            # Volume & Liquidity
            'mean_liquidity_20d': group['liquidity_20d'].mean(),
            'mean_volume_ratio': group['volume_ratio'].mean(),
            'mean_price_pos_20d': group['price_vs_20d_range'].mean(),
            
            # Distribution
            'qtr_return_skewness': ret_skew,
            'qtr_return_kurtosis': ret_kurt
        }
        agg_results.append(record)
        
    return pd.DataFrame(agg_results)

def main():
    
    parser = argparse.ArgumentParser(description="Create Clustering Datasets")
    parser.add_argument("--input-path", required=True, help="Path to your parquet file")
    parser.add_argument("--output-path", default="ml/data/clustering", help="Output directory")
    args = parser.parse_args()
    
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    
    df = load_and_prep_data(args.input_path)
    df = calc_rolling_features_vectorized(df)
    
    print("Creating global dataset...")
    df_global = global_dataset(df)
    global_path = f"{args.output_path}/ds1_global_data.parquet"
    df_global.to_parquet(global_path)
    
    print("Creating quarterly dataset...")
    df_quarter = quarterly_dataset(df)
    quarter_path = f"{args.output_path}/ds2_quarter_data.parquet"
    df_quarter.to_parquet(quarter_path)
    
    return df_global, df_quarter

if __name__ == "__main__":
    main()