# Stock Clustering Visualization App

A Streamlit application for visualizing stock clustering segmentation with interactive charts and statistics.

## Features

- **Clustering Analysis**: Perform K-Means, PAM, or Hierarchical clustering on stock features
- **Cluster Visualization**: 2D PCA visualization and cluster distribution charts
- **Sharpe Ratio Statistics**: Calculate and compare Sharpe ratios for different periods:
  - Recent (last 30 days)
  - 3 months (last 90 days)
  - 6 months (last 180 days)
  - Overall (all available data)
- **Candlestick Charts**: Interactive candlestick charts for stocks in each cluster
- **Cluster Statistics**: Detailed statistics and feature comparisons by cluster

## Prerequisites

1. Ensure the clustering data file exists: `ml/data/ds1_global_data.parquet`
2. Database connection configured (for loading daily price data for Sharpe ratios and candlestick charts)
3. Required Python packages installed (see `requirements.txt`)

## Running the App

From the project root directory:

```bash
streamlit run streamlit/clustering_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

1. **Configure Clustering**:
   - Select clustering method (K-Means, PAM, or Hierarchical)
   - Choose number of clusters (2-10)
   - Click "Run Clustering"

2. **Explore Results**:
   - **Cluster Overview**: View cluster distribution and 2D PCA visualization
   - **Cluster Statistics**: See Sharpe ratios and other metrics by cluster
   - **Candlestick Charts**: View price charts for stocks in each cluster
   - **Detailed View**: Examine individual cluster details and feature comparisons

## Data Requirements

- **Clustering Data**: `ml/data/ds1_global_data.parquet` - Contains aggregated features per stock
- **Daily Bars**: Database table `daily_bars` - Contains OHLCV data for Sharpe ratio calculations and candlestick charts

## Notes

- If daily price data is not available in the database, the app will still work but Sharpe ratios and candlestick charts will not be available
- The app uses caching to improve performance when reloading data
- Clustering results are stored in session state and persist during your session

