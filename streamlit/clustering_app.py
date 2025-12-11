"""
Streamlit App for Stock Clustering Visualization

This app visualizes stock clustering segmentation with:
- Cluster assignments and visualizations
- Sharpe ratio statistics (recent, 3m, 6m)
- Candlestick charts for stocks in each cluster
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Clustering libraries
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False

# Database connection
from app.config import connect_db

# Configuration
DATA_PATH = project_root / 'ml' / 'data' / 'ds1_global_data.parquet'
RANDOM_STATE = 42

# Page config
st.set_page_config(
    page_title="Stock Clustering Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_clustering_data():
    """Load the global clustering dataset."""
    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()
    return pd.read_parquet(DATA_PATH)

@st.cache_data
def load_daily_bars(tickers):
    """Load daily OHLCV data from database or parquet file for given tickers."""
    # Try database first
    try:
        conn = connect_db(use_docker=False)
        ticker_list = "', '".join(tickers)
        query = f"""
            SELECT ticker, date, open, high, low, close, volume
            FROM daily_bars
            WHERE ticker IN ('{ticker_list}')
            ORDER BY ticker, date
        """
        df = pd.read_sql(query, conn)
        conn.close()
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception as e:
        st.warning(f"Could not load daily bars from database: {e}")
    
    # Fallback to parquet file
    project_root = Path(__file__).resolve().parents[1]
    parquet_path = project_root / 'ml' / 'data' / 'daily_data.parquet'
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            # Filter for requested tickers
            df = df[df['ticker'].isin(tickers)]
            # Ensure we have required columns
            required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
            if all(col in df.columns for col in required_cols):
                df = df[required_cols].copy()
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                return df.sort_values(['ticker', 'date'])
        except Exception as e:
            st.warning(f"Could not load daily bars from parquet file: {e}")
    
    return pd.DataFrame()

def calculate_sharpe_ratio(returns, periods=252):
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(periods)

def calculate_period_sharpe_ratios(daily_bars_df):
    """Calculate Sharpe ratios for different periods."""
    if daily_bars_df.empty:
        return pd.DataFrame()
    
    results = []
    today = datetime.now()
    
    for ticker in daily_bars_df['ticker'].unique():
        ticker_data = daily_bars_df[daily_bars_df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date')
        ticker_data['daily_return'] = ticker_data['close'].pct_change()
        
        # Recent (last 30 days)
        recent_data = ticker_data[ticker_data['date'] >= (ticker_data['date'].max() - timedelta(days=30))]
        sharpe_recent = calculate_sharpe_ratio(recent_data['daily_return'].dropna(), periods=252)
        
        # 3 months (last 90 days)
        data_3m = ticker_data[ticker_data['date'] >= (ticker_data['date'].max() - timedelta(days=90))]
        sharpe_3m = calculate_sharpe_ratio(data_3m['daily_return'].dropna(), periods=252)
        
        # 6 months (last 180 days)
        data_6m = ticker_data[ticker_data['date'] >= (ticker_data['date'].max() - timedelta(days=180))]
        sharpe_6m = calculate_sharpe_ratio(data_6m['daily_return'].dropna(), periods=252)
        
        # Overall (all available data)
        sharpe_overall = calculate_sharpe_ratio(ticker_data['daily_return'].dropna(), periods=252)
        
        results.append({
            'ticker': ticker,
            'sharpe_recent': sharpe_recent,
            'sharpe_3m': sharpe_3m,
            'sharpe_6m': sharpe_6m,
            'sharpe_overall': sharpe_overall
        })
    
    return pd.DataFrame(results)

def prepare_features_for_clustering(df):
    """Prepare features for clustering (matching notebook methodology)."""
    # Exclude non-feature columns
    exclude_cols = ['ticker']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    features = df[feature_cols].copy()
    tickers = df['ticker'].values
    
    # Handle missing values
    features = features.fillna(features.median())
    
    # Identify and transform skewed features (threshold = 2.0)
    SKEW_THRESHOLD = 2.0
    skewness_vals = features.skew()
    skewed_features = skewness_vals[abs(skewness_vals) > SKEW_THRESHOLD].index.tolist()
    
    # Apply Yeo-Johnson transformation to skewed features
    features_transformed = features.copy()
    for feat in skewed_features:
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        features_transformed[feat] = pt.fit_transform(features[[feat]]).flatten()
    
    # Standardize all features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features_transformed),
        columns=features_transformed.columns,
        index=features_transformed.index
    )
    
    return features_scaled, tickers, feature_cols

def perform_clustering(features_scaled, n_clusters, method='K-Means'):
    """Perform clustering using specified method."""
    if method == 'K-Means':
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(features_scaled)
        return labels, model
    
    elif method == 'PAM' and HAS_KMEDOIDS:
        model = KMedoids(n_clusters=n_clusters, metric='euclidean', random_state=RANDOM_STATE)
        labels = model.fit_predict(features_scaled)
        return labels, model
    
    elif method == 'Hierarchical':
        # Use correlation-based distance
        distance_matrix = squareform(pdist(features_scaled.values, metric='euclidean'))
        linkage_matrix = linkage(distance_matrix, method='ward')
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1  # 0-indexed
        return labels, {'linkage_matrix': linkage_matrix}
    
    else:
        st.error(f"Clustering method '{method}' not available")
        return None, None

def create_candlestick_chart(ticker_data, ticker):
    """Create a candlestick chart for a stock."""
    fig = go.Figure(data=[go.Candlestick(
        x=ticker_data['date'],
        open=ticker_data['open'],
        high=ticker_data['high'],
        low=ticker_data['low'],
        close=ticker_data['close'],
        name=ticker
    )])
    
    fig.update_layout(
        title=f'{ticker} - Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=400,
        template='plotly_white'
    )
    
    return fig

def main():
    st.title("📊 Stock Clustering Segmentation Analysis")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Load data
    with st.spinner("Loading clustering data..."):
        global_data = load_clustering_data()
    
    st.sidebar.info(f"Loaded {len(global_data)} stocks")
    
    # Clustering parameters
    clustering_method = st.sidebar.selectbox(
        "Clustering Method",
        ['K-Means', 'PAM', 'Hierarchical'],
        index=0
    )
    
    n_clusters = st.sidebar.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=5,
        step=1
    )
    
    # Prepare features
    with st.spinner("Preparing features for clustering..."):
        features_scaled, tickers, feature_cols = prepare_features_for_clustering(global_data)
    
    # Perform clustering
    if st.sidebar.button("Run Clustering", type="primary"):
        with st.spinner(f"Running {clustering_method} clustering..."):
            labels, model = perform_clustering(features_scaled, n_clusters, clustering_method)
            
            if labels is not None:
                # Create results dataframe
                results_df = pd.DataFrame({
                    'ticker': tickers,
                    'cluster': labels + 1  # 1-indexed for display
                })
                
                # Merge with original data
                results_df = results_df.merge(global_data, on='ticker', how='left')
                
                st.session_state['clustering_results'] = results_df
                st.session_state['clustering_method'] = clustering_method
                st.session_state['n_clusters'] = n_clusters
                st.session_state['model'] = model
                st.success(f"Clustering completed! Created {n_clusters} clusters using {clustering_method}")
    
    # Display results if available
    if 'clustering_results' in st.session_state:
        results_df = st.session_state['clustering_results']
        clustering_method = st.session_state['clustering_method']
        n_clusters = st.session_state['n_clusters']
        
        st.header(f"Clustering Results: {clustering_method} ({n_clusters} clusters)")
        
        # Cluster distribution
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks", len(results_df))
        with col2:
            st.metric("Number of Clusters", n_clusters)
        with col3:
            avg_cluster_size = len(results_df) / n_clusters
            st.metric("Avg Cluster Size", f"{avg_cluster_size:.1f}")
        
        # Cluster visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Cluster Overview",
            "📊 Cluster Statistics",
            "🕯️ Candlestick Charts",
            "📋 Detailed View"
        ])
        
        with tab1:
            st.subheader("Cluster Distribution")
            
            # Cluster counts
            cluster_counts = results_df['cluster'].value_counts().sort_index()
            fig_counts = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                labels={'x': 'Cluster', 'y': 'Number of Stocks'},
                title="Stocks per Cluster"
            )
            st.plotly_chart(fig_counts, use_container_width=True)
            
            # PCA visualization (2D projection)
            st.subheader("2D Cluster Visualization (PCA)")
            pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
            features_2d = pca_2d.fit_transform(features_scaled)
            
            pca_df = pd.DataFrame({
                'PC1': features_2d[:, 0],
                'PC2': features_2d[:, 1],
                'cluster': results_df['cluster'].values,
                'ticker': results_df['ticker'].values
            })
            
            fig_pca = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='cluster',
                hover_data=['ticker'],
                title=f"PCA Visualization (Explained Variance: {pca_2d.explained_variance_ratio_.sum():.2%})",
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # Cluster assignments table
            st.subheader("Cluster Assignments")
            display_cols = ['ticker', 'cluster'] + [col for col in global_data.columns if col != 'ticker'][:5]
            st.dataframe(
                results_df[display_cols].sort_values(['cluster', 'ticker']),
                use_container_width=True,
                height=400
            )
        
        with tab2:
            st.subheader("Cluster Statistics & Sharpe Ratios")
            
            # Load daily bars for Sharpe calculation
            all_tickers = results_df['ticker'].unique().tolist()
            with st.spinner("Loading daily price data for Sharpe ratio calculation..."):
                daily_bars = load_daily_bars(all_tickers)
            
            if not daily_bars.empty:
                # Calculate Sharpe ratios
                sharpe_df = calculate_period_sharpe_ratios(daily_bars)
                
                # Merge with clustering results
                stats_df = results_df.merge(sharpe_df, on='ticker', how='left')
                
                # Cluster-level statistics
                cluster_stats = stats_df.groupby('cluster').agg({
                    'sharpe_recent': 'mean',
                    'sharpe_3m': 'mean',
                    'sharpe_6m': 'mean',
                    'sharpe_overall': 'mean',
                    'mean_daily_return': 'mean',
                    'mean_volatility_20d': 'mean',
                    'mean_sharpe_20d': 'mean',
                    'ticker': 'count'
                }).round(4)
                
                cluster_stats.columns = [
                    'Avg Sharpe (Recent)',
                    'Avg Sharpe (3M)',
                    'Avg Sharpe (6M)',
                    'Avg Sharpe (Overall)',
                    'Avg Daily Return',
                    'Avg Volatility',
                    'Avg Sharpe (20d)',
                    'Stock Count'
                ]
                
                st.dataframe(cluster_stats, use_container_width=True)
                
                # Sharpe ratio comparison chart
                st.subheader("Sharpe Ratio Comparison by Cluster")
                sharpe_cols = ['sharpe_recent', 'sharpe_3m', 'sharpe_6m', 'sharpe_overall']
                sharpe_melted = stats_df.melt(
                    id_vars=['cluster', 'ticker'],
                    value_vars=sharpe_cols,
                    var_name='Period',
                    value_name='Sharpe Ratio'
                )
                sharpe_melted['Period'] = sharpe_melted['Period'].str.replace('sharpe_', '').str.title()
                
                fig_sharpe = px.box(
                    sharpe_melted,
                    x='cluster',
                    y='Sharpe Ratio',
                    color='Period',
                    title="Sharpe Ratio Distribution by Cluster and Period"
                )
                st.plotly_chart(fig_sharpe, use_container_width=True)
                
                # Individual stock Sharpe ratios
                st.subheader("Individual Stock Sharpe Ratios")
                sharpe_display = stats_df[['ticker', 'cluster'] + sharpe_cols].copy()
                sharpe_display.columns = ['Ticker', 'Cluster', 'Recent', '3M', '6M', 'Overall']
                st.dataframe(
                    sharpe_display.sort_values(['Cluster', 'Ticker']),
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("Daily price data not available. Sharpe ratios cannot be calculated.")
                # Show basic statistics from global data
                cluster_stats_basic = results_df.groupby('cluster').agg({
                    'mean_daily_return': 'mean',
                    'mean_volatility_20d': 'mean',
                    'mean_sharpe_20d': 'mean',
                    'ticker': 'count'
                }).round(4)
                cluster_stats_basic.columns = [
                    'Avg Daily Return',
                    'Avg Volatility',
                    'Avg Sharpe (20d)',
                    'Stock Count'
                ]
                st.dataframe(cluster_stats_basic, use_container_width=True)
        
        with tab3:
            st.subheader("Candlestick Charts by Cluster")
            
            # Load daily bars if not already loaded
            if 'daily_bars' not in locals() or daily_bars.empty:
                all_tickers = results_df['ticker'].unique().tolist()
                with st.spinner("Loading daily price data..."):
                    daily_bars = load_daily_bars(all_tickers)
            
            if daily_bars.empty:
                st.warning("Daily price data not available for candlestick charts.")
            else:
                # Select cluster
                selected_cluster = st.selectbox(
                    "Select Cluster",
                    sorted(results_df['cluster'].unique()),
                    key='candlestick_cluster'
                )
                
                cluster_tickers = results_df[results_df['cluster'] == selected_cluster]['ticker'].tolist()
                
                # Select ticker
                selected_ticker = st.selectbox(
                    "Select Stock",
                    cluster_tickers,
                    key='candlestick_ticker'
                )
                
                # Date range filter
                ticker_data = daily_bars[daily_bars['ticker'] == selected_ticker].copy()
                if not ticker_data.empty:
                    min_date = ticker_data['date'].min()
                    max_date = ticker_data['date'].max()
                    
                    date_range = st.date_input(
                        "Date Range",
                        value=(max_date - timedelta(days=180), max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key='candlestick_dates'
                    )
                    
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        ticker_data = ticker_data[
                            (ticker_data['date'] >= pd.Timestamp(start_date)) &
                            (ticker_data['date'] <= pd.Timestamp(end_date))
                        ]
                    
                    if not ticker_data.empty:
                        fig_candle = create_candlestick_chart(ticker_data, selected_ticker)
                        st.plotly_chart(fig_candle, use_container_width=True)
                        
                        # Show additional metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Price", f"${ticker_data['close'].iloc[-1]:.2f}")
                        with col2:
                            price_change = ticker_data['close'].iloc[-1] - ticker_data['close'].iloc[0]
                            st.metric("Price Change", f"${price_change:.2f}")
                        with col3:
                            pct_change = (price_change / ticker_data['close'].iloc[0]) * 100
                            st.metric("Return", f"{pct_change:.2f}%")
                        with col4:
                            avg_volume = ticker_data['volume'].mean()
                            st.metric("Avg Volume", f"{avg_volume:,.0f}")
                    else:
                        st.warning("No data available for selected date range.")
                else:
                    st.warning(f"No price data available for {selected_ticker}")
        
        with tab4:
            st.subheader("Detailed Cluster View")
            
            # Select cluster
            selected_cluster = st.selectbox(
                "Select Cluster",
                sorted(results_df['cluster'].unique()),
                key='detail_cluster'
            )
            
            cluster_data = results_df[results_df['cluster'] == selected_cluster].copy()
            
            st.write(f"**Cluster {selected_cluster}** - {len(cluster_data)} stocks")
            
            # Display all features for this cluster
            st.dataframe(
                cluster_data.sort_values('ticker'),
                use_container_width=True,
                height=600
            )
            
            # Feature comparison
            st.subheader("Feature Comparison")
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns.tolist()
            if 'cluster' in numeric_cols:
                numeric_cols.remove('cluster')
            
            selected_features = st.multiselect(
                "Select Features to Compare",
                numeric_cols,
                default=['mean_daily_return', 'mean_volatility_20d', 'mean_sharpe_20d', 'beta_global'],
                key='feature_comparison'
            )
            
            if selected_features:
                comparison_df = cluster_data[['ticker'] + selected_features].set_index('ticker')
                fig_heatmap = px.imshow(
                    comparison_df.T,
                    labels=dict(x="Stock", y="Feature", color="Value"),
                    title=f"Feature Heatmap - Cluster {selected_cluster}",
                    color_continuous_scale='RdYlGn',
                    aspect='auto'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
    
    else:
        st.info("👈 Configure clustering parameters in the sidebar and click 'Run Clustering' to start.")

if __name__ == "__main__":
    main()

