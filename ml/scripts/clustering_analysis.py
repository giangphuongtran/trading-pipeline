"""
Portfolio Clustering Analysis for Diversification and Risk Management

Three-phase approach:
1. Correlation-based clustering (hierarchical + PAM)
2. Feature-based clustering (k-means + PAM)
3. Comparison and interpretation
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Clustering libraries
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# PAM (Partitioning Around Medoids) - using sklearn_extra if available, otherwise fallback
try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False
    print("Warning: sklearn_extra not available. PAM clustering will use KMeans as approximation.")


def load_clustering_data(global_path: str, quarter_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed clustering datasets.
    
    Args:
        global_path: Path to global dataset parquet file
        quarter_path: Optional path to quarterly dataset
        
    Returns:
        Tuple of (global_df, quarter_df or None)
    """
    print(f"Loading global dataset from {global_path}")
    global_df = pd.read_parquet(global_path)
    print(f"Loaded {len(global_df)} stocks")
    
    quarter_df = None
    if quarter_path and Path(quarter_path).exists():
        print(f"Loading quarterly dataset from {quarter_path}")
        quarter_df = pd.read_parquet(quarter_path)
        print(f"Loaded {len(quarter_df)} stock-quarter records")
    
    return global_df, quarter_df


def prepare_correlation_matrix(df: pd.DataFrame, return_col: str = 'mean_daily_return') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create correlation matrix from returns data.
    
    For global dataset, we need to pivot returns by ticker.
    If we have time series data, we can calculate correlations directly.
    For aggregated data, we'll need to use a different approach.
    
    Args:
        df: DataFrame with ticker and return columns
        return_col: Column name for returns
        
    Returns:
        Tuple of (correlation_matrix, distance_matrix)
    """
    # For global aggregated data, we can't calculate correlation directly
    # We'll use feature similarity instead
    # If you have time series data, use that instead
    
    print("Note: Using feature-based correlation for global dataset.")
    print("For true return correlations, use time series data.")
    
    # Use multiple features to create similarity matrix
    feature_cols = [
        'mean_daily_return', 'mean_momentum_20d', 'mean_volatility_20d',
        'mean_sharpe_20d', 'mean_rsi_14', 'mean_adx_14'
    ]
    
    # Filter to available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    if len(available_cols) < 2:
        raise ValueError("Not enough feature columns available for correlation")
    
    # Create feature matrix
    feature_matrix = df.set_index('ticker')[available_cols].fillna(0)
    
    # Calculate correlation matrix (how similar are stocks based on features)
    correlation_matrix = feature_matrix.T.corr()  # Transpose to get stock-stock correlations
    
    # Convert correlation to distance: distance = 1 - correlation
    # (higher correlation = smaller distance)
    distance_matrix = 1 - correlation_matrix
    distance_matrix = distance_matrix.fillna(1.0)  # NaN correlations = maximum distance
    
    return correlation_matrix, distance_matrix


def phase1_correlation_clustering(
    df: pd.DataFrame,
    n_clusters: int = 5,
    linkage_method: str = 'ward',
    output_dir: str = 'ml/data/clustering'
) -> Dict:
    """
    Phase 1: Correlation-based clustering using hierarchical and PAM methods.
    
    Args:
        df: DataFrame with ticker and feature columns
        n_clusters: Number of clusters to create
        linkage_method: Linkage method for hierarchical ('ward', 'complete', 'average')
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with clustering results
    """
    print("\n" + "="*60)
    print("PHASE 1: Correlation-Based Clustering")
    print("="*60)
    
    # Prepare correlation and distance matrices
    correlation_matrix, distance_matrix = prepare_correlation_matrix(df)
    
    tickers = correlation_matrix.index.tolist()
    print(f"Clustering {len(tickers)} stocks into {n_clusters} clusters")
    
    # Convert distance matrix to condensed form for scipy
    condensed_distances = squareform(distance_matrix.values, checks=False)
    
    # Method 1: Hierarchical Clustering
    print("\n1. Hierarchical Clustering...")
    linkage_matrix = linkage(condensed_distances, method=linkage_method)
    
    # Get cluster assignments
    hierarchical_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    hierarchical_clusters = pd.DataFrame({
        'ticker': tickers,
        'cluster': hierarchical_labels,
        'method': 'hierarchical'
    })
    
    # Method 2: PAM (Partitioning Around Medoids)
    print("2. PAM Clustering...")
    if HAS_KMEDOIDS:
        # Use distance matrix directly
        pam_model = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
        pam_labels = pam_model.fit_predict(distance_matrix.values)
        pam_clusters = pd.DataFrame({
            'ticker': tickers,
            'cluster': pam_labels + 1,  # Make 1-indexed
            'method': 'pam'
        })
    else:
        print("   Using AgglomerativeClustering as PAM approximation...")
        # Approximate PAM with AgglomerativeClustering using precomputed distances
        pam_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric='precomputed'
        )
        pam_labels = pam_model.fit_predict(distance_matrix.values)
        pam_clusters = pd.DataFrame({
            'ticker': tickers,
            'cluster': pam_labels + 1,
            'method': 'pam_approx'
        })
    
    # Create dendrogram
    print("3. Creating dendrogram...")
    plt.figure(figsize=(15, 8))
    dendrogram(
        linkage_matrix,
        labels=tickers,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=linkage_matrix[-n_clusters+1, 2] if len(linkage_matrix) >= n_clusters else None
    )
    plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage)')
    plt.xlabel('Stock Ticker')
    plt.ylabel('Distance')
    plt.tight_layout()
    
    dendrogram_path = f"{output_dir}/phase1_dendrogram.png"
    plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved dendrogram to {dendrogram_path}")
    
    # Visualize correlation matrix with clusters
    print("4. Creating correlation heatmap with clusters...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Sort by hierarchical clusters
    hierarchical_sorted = hierarchical_clusters.sort_values('cluster')
    sorted_tickers = hierarchical_sorted['ticker'].tolist()
    corr_sorted = correlation_matrix.loc[sorted_tickers, sorted_tickers]
    
    sns.heatmap(
        corr_sorted,
        cmap='coolwarm',
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=axes[0]
    )
    axes[0].set_title('Correlation Matrix (Hierarchical Clusters)')
    
    # Sort by PAM clusters
    pam_sorted = pam_clusters.sort_values('cluster')
    sorted_tickers_pam = pam_sorted['ticker'].tolist()
    corr_sorted_pam = correlation_matrix.loc[sorted_tickers_pam, sorted_tickers_pam]
    
    sns.heatmap(
        corr_sorted_pam,
        cmap='coolwarm',
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=axes[1]
    )
    axes[1].set_title('Correlation Matrix (PAM Clusters)')
    
    plt.tight_layout()
    heatmap_path = f"{output_dir}/phase1_correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved heatmap to {heatmap_path}")
    
    # Calculate cluster statistics
    print("5. Calculating cluster statistics...")
    hierarchical_stats = hierarchical_clusters.groupby('cluster').size().to_dict()
    pam_stats = pam_clusters.groupby('cluster').size().to_dict()
    
    print(f"\nHierarchical Clustering Results:")
    for cluster, count in sorted(hierarchical_stats.items()):
        print(f"  Cluster {cluster}: {count} stocks")
    
    print(f"\nPAM Clustering Results:")
    for cluster, count in sorted(pam_stats.items()):
        print(f"  Cluster {cluster}: {count} stocks")
    
    return {
        'hierarchical_clusters': hierarchical_clusters,
        'pam_clusters': pam_clusters,
        'correlation_matrix': correlation_matrix,
        'distance_matrix': distance_matrix,
        'linkage_matrix': linkage_matrix,
        'n_clusters': n_clusters
    }


def phase2_feature_clustering(
    df: pd.DataFrame,
    n_clusters: int = 5,
    output_dir: str = 'ml/data/clustering'
) -> Dict:
    """
    Phase 2: Feature-based clustering using k-means and PAM.
    
    Args:
        df: DataFrame with ticker and feature columns
        n_clusters: Number of clusters to create
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with clustering results
    """
    print("\n" + "="*60)
    print("PHASE 2: Feature-Based Clustering")
    print("="*60)
    
    # Select feature columns (exclude ticker and identifier columns)
    exclude_cols = ['ticker', 'quarter', 'beta_global', 'return_skewness', 'return_kurtosis',
                   'qtr_return_skewness', 'qtr_return_kurtosis']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features for clustering")
    
    # Prepare feature matrix
    feature_matrix = df.set_index('ticker')[feature_cols].fillna(0)
    tickers = feature_matrix.index.tolist()
    
    print(f"Clustering {len(tickers)} stocks into {n_clusters} clusters")
    
    # Standardize features
    print("1. Standardizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)
    features_scaled_df = pd.DataFrame(
        features_scaled,
        index=tickers,
        columns=feature_cols
    )
    
    # Method 1: K-Means
    print("2. K-Means Clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(features_scaled)
    kmeans_clusters = pd.DataFrame({
        'ticker': tickers,
        'cluster': kmeans_labels + 1,  # Make 1-indexed
        'method': 'kmeans'
    })
    
    # Calculate K-Means metrics
    kmeans_silhouette = silhouette_score(features_scaled, kmeans_labels)
    kmeans_db = davies_bouldin_score(features_scaled, kmeans_labels)
    print(f"   Silhouette Score: {kmeans_silhouette:.3f}")
    print(f"   Davies-Bouldin Score: {kmeans_db:.3f} (lower is better)")
    
    # Method 2: PAM on features
    print("3. PAM Clustering on features...")
    if HAS_KMEDOIDS:
        pam_model = KMedoids(n_clusters=n_clusters, metric='euclidean', random_state=42)
        pam_labels = pam_model.fit_predict(features_scaled)
        pam_clusters = pd.DataFrame({
            'ticker': tickers,
            'cluster': pam_labels + 1,
            'method': 'pam'
        })
        
        # Calculate PAM metrics
        pam_silhouette = silhouette_score(features_scaled, pam_labels)
        pam_db = davies_bouldin_score(features_scaled, pam_labels)
        print(f"   Silhouette Score: {pam_silhouette:.3f}")
        print(f"   Davies-Bouldin Score: {pam_db:.3f} (lower is better)")
    else:
        print("   Using AgglomerativeClustering as PAM approximation...")
        pam_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        pam_labels = pam_model.fit_predict(features_scaled)
        pam_clusters = pd.DataFrame({
            'ticker': tickers,
            'cluster': pam_labels + 1,
            'method': 'pam_approx'
        })
        
        pam_silhouette = silhouette_score(features_scaled, pam_labels)
        pam_db = davies_bouldin_score(features_scaled, pam_labels)
        print(f"   Silhouette Score: {pam_silhouette:.3f}")
        print(f"   Davies-Bouldin Score: {pam_db:.3f} (lower is better)")
    
    # Visualize clusters in 2D using PCA
    print("4. Creating 2D visualization (PCA)...")
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # K-Means visualization
    scatter1 = axes[0].scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=kmeans_labels,
        cmap='tab10',
        alpha=0.6,
        s=50
    )
    axes[0].set_title(f'K-Means Clusters (PCA)\nSilhouette: {kmeans_silhouette:.3f}')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter1, ax=axes[0])
    
    # PAM visualization
    scatter2 = axes[1].scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=pam_labels,
        cmap='tab10',
        alpha=0.6,
        s=50
    )
    axes[1].set_title(f'PAM Clusters (PCA)\nSilhouette: {pam_silhouette:.3f}')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    pca_path = f"{output_dir}/phase2_pca_visualization.png"
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved PCA visualization to {pca_path}")
    
    # Analyze cluster characteristics
    print("5. Analyzing cluster characteristics...")
    kmeans_stats = kmeans_clusters.groupby('cluster').size().to_dict()
    pam_stats = pam_clusters.groupby('cluster').size().to_dict()
    
    print(f"\nK-Means Clustering Results:")
    for cluster, count in sorted(kmeans_stats.items()):
        print(f"  Cluster {cluster}: {count} stocks")
    
    print(f"\nPAM Clustering Results:")
    for cluster, count in sorted(pam_stats.items()):
        print(f"  Cluster {cluster}: {count} stocks")
    
    # Calculate cluster centroids (average features per cluster)
    kmeans_centroids = []
    for cluster_id in sorted(kmeans_clusters['cluster'].unique()):
        cluster_tickers = kmeans_clusters[kmeans_clusters['cluster'] == cluster_id]['ticker']
        cluster_features = feature_matrix.loc[cluster_tickers].mean()
        kmeans_centroids.append({
            'cluster': cluster_id,
            **cluster_features.to_dict()
        })
    kmeans_centroids_df = pd.DataFrame(kmeans_centroids)
    
    return {
        'kmeans_clusters': kmeans_clusters,
        'pam_clusters': pam_clusters,
        'feature_matrix': feature_matrix,
        'features_scaled': features_scaled_df,
        'scaler': scaler,
        'kmeans_model': kmeans,
        'pam_model': pam_model if HAS_KMEDOIDS else None,
        'kmeans_centroids': kmeans_centroids_df,
        'kmeans_silhouette': kmeans_silhouette,
        'pam_silhouette': pam_silhouette,
        'kmeans_db': kmeans_db,
        'pam_db': pam_db,
        'n_clusters': n_clusters
    }


def phase3_comparison(
    phase1_results: Dict,
    phase2_results: Dict,
    df: pd.DataFrame,
    output_dir: str = 'ml/data/clustering'
) -> Dict:
    """
    Phase 3: Compare correlation-based vs feature-based clusters.
    
    Args:
        phase1_results: Results from Phase 1
        phase2_results: Results from Phase 2
        df: Original dataframe
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*60)
    print("PHASE 3: Comparison and Interpretation")
    print("="*60)
    
    # Merge all cluster assignments
    hierarchical = phase1_results['hierarchical_clusters'].set_index('ticker')['cluster']
    pam_corr = phase1_results['pam_clusters'].set_index('ticker')['cluster']
    kmeans = phase2_results['kmeans_clusters'].set_index('ticker')['cluster']
    pam_feat = phase2_results['pam_clusters'].set_index('ticker')['cluster']
    
    comparison_df = pd.DataFrame({
        'hierarchical_corr': hierarchical,
        'pam_corr': pam_corr,
        'kmeans_feat': kmeans,
        'pam_feat': pam_feat
    })
    
    # Calculate agreement between methods
    print("1. Calculating cluster agreement...")
    
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    methods = {
        'Hierarchical vs K-Means': (hierarchical, kmeans),
        'Hierarchical vs PAM (Feature)': (hierarchical, pam_feat),
        'PAM (Correlation) vs K-Means': (pam_corr, kmeans),
        'PAM (Correlation) vs PAM (Feature)': (pam_corr, pam_feat),
        'K-Means vs PAM (Feature)': (kmeans, pam_feat)
    }
    
    agreement_results = []
    for method_pair, (labels1, labels2) in methods.items():
        ari = adjusted_rand_score(labels1, labels2)
        nmi = normalized_mutual_info_score(labels1, labels2)
        agreement_results.append({
            'comparison': method_pair,
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi
        })
        print(f"  {method_pair}:")
        print(f"    Adjusted Rand Index: {ari:.3f} (1.0 = perfect match, 0.0 = random)")
        print(f"    Normalized Mutual Info: {nmi:.3f} (1.0 = perfect match, 0.0 = independent)")
    
    agreement_df = pd.DataFrame(agreement_results)
    
    # Create comparison heatmap
    print("2. Creating comparison heatmap...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Hierarchical vs K-Means
    comparison_matrix_1 = pd.crosstab(hierarchical, kmeans)
    sns.heatmap(comparison_matrix_1, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Hierarchical (Corr) vs K-Means (Feature)')
    axes[0, 0].set_xlabel('K-Means Cluster')
    axes[0, 0].set_ylabel('Hierarchical Cluster')
    
    # PAM Correlation vs PAM Feature
    comparison_matrix_2 = pd.crosstab(pam_corr, pam_feat)
    sns.heatmap(comparison_matrix_2, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('PAM (Corr) vs PAM (Feature)')
    axes[0, 1].set_xlabel('PAM Feature Cluster')
    axes[0, 1].set_ylabel('PAM Correlation Cluster')
    
    # Hierarchical vs PAM Feature
    comparison_matrix_3 = pd.crosstab(hierarchical, pam_feat)
    sns.heatmap(comparison_matrix_3, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Hierarchical (Corr) vs PAM (Feature)')
    axes[1, 0].set_xlabel('PAM Feature Cluster')
    axes[1, 0].set_ylabel('Hierarchical Cluster')
    
    # K-Means vs PAM Feature
    comparison_matrix_4 = pd.crosstab(kmeans, pam_feat)
    sns.heatmap(comparison_matrix_4, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title('K-Means vs PAM (Feature)')
    axes[1, 1].set_xlabel('PAM Feature Cluster')
    axes[1, 1].set_ylabel('K-Means Cluster')
    
    plt.tight_layout()
    comparison_path = f"{output_dir}/phase3_comparison_heatmap.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved comparison heatmap to {comparison_path}")
    
    # Analyze cluster characteristics
    print("3. Analyzing cluster characteristics...")
    
    # Merge with original data
    df_with_clusters = df.set_index('ticker').join(comparison_df, how='inner')
    
    # Create summary statistics per cluster for each method
    cluster_summaries = {}
    
    for method in ['hierarchical_corr', 'pam_corr', 'kmeans_feat', 'pam_feat']:
        method_summaries = []
        for cluster_id in sorted(df_with_clusters[method].unique()):
            cluster_data = df_with_clusters[df_with_clusters[method] == cluster_id]
            
            # Calculate average characteristics
            summary = {
                'method': method,
                'cluster': cluster_id,
                'n_stocks': len(cluster_data),
                'avg_return': cluster_data.get('mean_daily_return', pd.Series()).mean(),
                'avg_volatility': cluster_data.get('mean_volatility_20d', pd.Series()).mean(),
                'avg_sharpe': cluster_data.get('mean_sharpe_20d', pd.Series()).mean(),
                'avg_beta': cluster_data.get('beta_global', pd.Series()).mean(),
            }
            method_summaries.append(summary)
        
        cluster_summaries[method] = pd.DataFrame(method_summaries)
    
    # Create summary report
    print("\n4. Cluster Summary Statistics:")
    for method, summary_df in cluster_summaries.items():
        print(f"\n{method.upper()}:")
        print(summary_df.to_string(index=False))
    
    # Save results
    comparison_df.to_parquet(f"{output_dir}/phase3_cluster_comparison.parquet")
    agreement_df.to_parquet(f"{output_dir}/phase3_agreement_metrics.parquet")
    
    # Create final summary report
    summary_report = {
        'agreement_metrics': agreement_df,
        'cluster_assignments': comparison_df,
        'cluster_summaries': cluster_summaries,
        'interpretation': _generate_interpretation(agreement_df, cluster_summaries)
    }
    
    return summary_report


def _generate_interpretation(agreement_df: pd.DataFrame, cluster_summaries: Dict) -> str:
    """Generate text interpretation of results."""
    interpretation = []
    interpretation.append("\n" + "="*60)
    interpretation.append("INTERPRETATION")
    interpretation.append("="*60)
    
    # Find best agreement
    best_agreement = agreement_df.loc[agreement_df['adjusted_rand_index'].idxmax()]
    interpretation.append(f"\nBest Agreement: {best_agreement['comparison']}")
    interpretation.append(f"  Adjusted Rand Index: {best_agreement['adjusted_rand_index']:.3f}")
    
    # Interpretation guidelines
    interpretation.append("\nInterpretation Guidelines:")
    interpretation.append("  - ARI > 0.5: Strong agreement between methods")
    interpretation.append("  - ARI 0.3-0.5: Moderate agreement")
    interpretation.append("  - ARI < 0.3: Weak agreement (methods find different patterns)")
    
    interpretation.append("\nRecommendations:")
    interpretation.append("  1. For portfolio diversification: Use feature-based clusters")
    interpretation.append("     (they capture risk/return profiles better)")
    interpretation.append("  2. For pairs trading: Use correlation-based clusters")
    interpretation.append("     (they identify stocks that move together)")
    interpretation.append("  3. For sector identification: Compare clusters with known sectors")
    interpretation.append("  4. For risk management: Use feature clusters with high Sharpe ratio")
    
    return "\n".join(interpretation)


def main():
    parser = argparse.ArgumentParser(description="Portfolio Clustering Analysis")
    parser.add_argument(
        "--global-path",
        required=True,
        help="Path to global dataset parquet file (from preprocessing.py)"
    )
    parser.add_argument(
        "--quarter-path",
        default=None,
        help="Optional path to quarterly dataset parquet file"
    )
    parser.add_argument(
        "--output-dir",
        default="ml/data/clustering",
        help="Output directory for results"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters to create (default: 5)"
    )
    parser.add_argument(
        "--linkage-method",
        default="ward",
        choices=["ward", "complete", "average"],
        help="Linkage method for hierarchical clustering (default: ward)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    global_df, quarter_df = load_clustering_data(args.global_path, args.quarter_path)
    
    # Phase 1: Correlation-based clustering
    phase1_results = phase1_correlation_clustering(
        global_df,
        n_clusters=args.n_clusters,
        linkage_method=args.linkage_method,
        output_dir=args.output_dir
    )
    
    # Save Phase 1 results
    phase1_results['hierarchical_clusters'].to_parquet(
        f"{args.output_dir}/phase1_hierarchical_clusters.parquet"
    )
    phase1_results['pam_clusters'].to_parquet(
        f"{args.output_dir}/phase1_pam_clusters.parquet"
    )
    phase1_results['correlation_matrix'].to_parquet(
        f"{args.output_dir}/phase1_correlation_matrix.parquet"
    )
    
    # Phase 2: Feature-based clustering
    phase2_results = phase2_feature_clustering(
        global_df,
        n_clusters=args.n_clusters,
        output_dir=args.output_dir
    )
    
    # Save Phase 2 results
    phase2_results['kmeans_clusters'].to_parquet(
        f"{args.output_dir}/phase2_kmeans_clusters.parquet"
    )
    phase2_results['pam_clusters'].to_parquet(
        f"{args.output_dir}/phase2_pam_clusters.parquet"
    )
    phase2_results['kmeans_centroids'].to_parquet(
        f"{args.output_dir}/phase2_kmeans_centroids.parquet"
    )
    
    # Phase 3: Comparison
    phase3_results = phase3_comparison(
        phase1_results,
        phase2_results,
        global_df,
        output_dir=args.output_dir
    )
    
    # Print interpretation
    print(phase3_results['interpretation'])
    
    # Save final summary
    with open(f"{args.output_dir}/clustering_summary.txt", "w") as f:
        f.write(phase3_results['interpretation'])
        f.write("\n\nAgreement Metrics:\n")
        f.write(phase3_results['agreement_metrics'].to_string())
    
    print(f"\n\nAll results saved to {args.output_dir}/")
    print("Files created:")
    print("  - phase1_hierarchical_clusters.parquet")
    print("  - phase1_pam_clusters.parquet")
    print("  - phase1_correlation_matrix.parquet")
    print("  - phase1_dendrogram.png")
    print("  - phase1_correlation_heatmap.png")
    print("  - phase2_kmeans_clusters.parquet")
    print("  - phase2_pam_clusters.parquet")
    print("  - phase2_kmeans_centroids.parquet")
    print("  - phase2_pca_visualization.png")
    print("  - phase3_cluster_comparison.parquet")
    print("  - phase3_agreement_metrics.parquet")
    print("  - phase3_comparison_heatmap.png")
    print("  - clustering_summary.txt")


if __name__ == "__main__":
    main()

