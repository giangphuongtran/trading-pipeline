"""
Streamlit App for Stock Clustering Visualization (Model 1)

This app visualizes stock clustering segmentation with:
- Dataset overview + basic data analysis (correlation, skewness, PCA variance)
- Clustering with PCA reduction (K-Means / PAM / Hierarchical)
- Correct silhouette scoring (precomputed distances for hierarchical)
- Dendrogram (hierarchical), silhouette plot, PCA 2D/3D visualization
- Optional Sharpe ratios + candlestick charts from Postgres (or parquet fallback)
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import PowerTransformer, StandardScaler

try:
    from sklearn_extra.cluster import KMedoids

    HAS_KMEDOIDS = True
except Exception:
    HAS_KMEDOIDS = False

import os
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.config import connect_db

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATHS = [
    PROJECT_ROOT / "ml" / "data" / "ds1_global_data.parquet",
    PROJECT_ROOT / "ml" / "data" / "clustering" / "ds1_global_data.parquet",
]

RANDOM_STATE = 42

FEATURE_GROUPS: dict[str, list[str]] = {
    "Risk/Return": [
        "mean_daily_return",
        "mean_stock_vs_market",
        "beta_global",
        "mean_sharpe_20d",
        "worst_drawdown",
    ],
    "Momentum": [
        "mean_momentum_20d",
        "mean_close_vs_sma200",
        "mean_adx_14",
        "mean_price_pos_20d",
    ],
    "Volatility": [
        "mean_volatility_20d",
        "mean_atr_14",
        "mean_volatility_ratio",
        "mean_bb_width",
    ],
    "Liquidity": [
        "mean_liquidity_20d",
        "mean_volume_ratio",
    ],
    "Technical": [
        "mean_rsi_14",
        "mean_macd_hist",
        "mean_stoch_k",
    ],
    "Distribution": [
        "return_skewness",
        "return_kurtosis",
    ],
}


@st.cache_data
def load_clustering_data() -> pd.DataFrame:
    """Load the global clustering dataset from parquet."""
    for path in DEFAULT_DATA_PATHS:
        if path.exists():
            df = pd.read_parquet(path)
            return df
    st.error(
        "Clustering dataset not found. Expected one of:\n- "
        + "\n- ".join(str(p) for p in DEFAULT_DATA_PATHS)
    )
    st.stop()


@st.cache_data
def load_daily_bars(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Load daily OHLCV from Postgres; fallback to parquet if DB unavailable."""
    if not tickers:
        return pd.DataFrame()

    # 1) Try Postgres (parameterized query)
    try:
        conn = connect_db(use_docker=False)
        query = """
            SELECT ticker, date, open, high, low, close, volume
            FROM daily_bars
            WHERE ticker = ANY(%s)
            ORDER BY ticker, date
        """
        df = pd.read_sql(query, conn, params=(list(tickers),))
        conn.close()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            return df
    except Exception as exc:
        st.warning(f"Could not load daily bars from Postgres (fallback to parquet): {exc}")

    # 2) Fallback parquet
    parquet_path = PROJECT_ROOT / "ml" / "data" / "daily_data.parquet"
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            required_cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                return pd.DataFrame()
            df = df[df["ticker"].isin(tickers)][required_cols].copy()
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values(["ticker", "date"])
        except Exception as exc:
            st.warning(f"Could not load daily bars from parquet: {exc}")

    return pd.DataFrame()


def prepare_features_for_clustering(global_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Transform + scale features for clustering."""
    if "ticker" not in global_df.columns:
        raise ValueError("Expected `ticker` column in clustering dataset.")

    feature_cols = [c for c in global_df.columns if c != "ticker"]
    features = global_df[feature_cols].copy()
    tickers = global_df["ticker"].astype(str).tolist()

    # Ensure numeric-only and handle invalid values
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(features.median(numeric_only=True))

    # Transform skewed features (Yeo-Johnson)
    skew = features.skew(numeric_only=True)
    skewed_cols = skew[skew.abs() > 2.0].index.tolist()
    features_transformed = features.copy()
    for col in skewed_cols:
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        features_transformed[col] = pt.fit_transform(features[[col]]).flatten()

    # Standardize with sklearn StandardScaler (population std, ddof=0)
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features_transformed),
        columns=features_transformed.columns,
        index=pd.Index(tickers, name="ticker"),
    )

    return features_scaled, feature_cols


@st.cache_data
def run_clustering(
    method: str,
    n_clusters: int,
    features_scaled: pd.DataFrame,
    pca_components: int,
) -> tuple[pd.DataFrame, dict, object | None, np.ndarray | None, pd.DataFrame]:
    """
    Run clustering with PCA reduction.

    Notes:
    - KMeans/PAM: silhouette uses euclidean distance on PCA space
    - Hierarchical: builds correlation-distance between stocks in PCA space and uses precomputed distances for silhouette
    """
    tickers = features_scaled.index.tolist()

    pca = PCA(n_components=int(pca_components), random_state=RANDOM_STATE)
    features_pca = pd.DataFrame(
        pca.fit_transform(features_scaled),
        index=features_scaled.index,
    )

    linkage_matrix = None
    model = None

    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(features_pca)
        sil_samples = silhouette_samples(features_pca, labels, metric="euclidean")
        sil = silhouette_score(features_pca, labels, metric="euclidean")
    elif method == "PAM":
        if not HAS_KMEDOIDS:
            raise RuntimeError("PAM selected but sklearn-extra is not installed.")
        model = KMedoids(n_clusters=n_clusters, metric="euclidean", random_state=RANDOM_STATE)
        labels = model.fit_predict(features_pca)
        sil_samples = silhouette_samples(features_pca, labels, metric="euclidean")
        sil = silhouette_score(features_pca, labels, metric="euclidean")
    elif method == "Hierarchical":
        X = features_pca.values
        corr = np.corrcoef(X)
        dist_sq = 1.0 - corr
        np.fill_diagonal(dist_sq, 0.0)

        condensed = squareform(dist_sq, checks=False)
        linkage_matrix = linkage(condensed, method="average")
        labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust") - 1

        sil_samples = silhouette_samples(dist_sq, labels, metric="precomputed")
        sil = silhouette_score(dist_sq, labels, metric="precomputed")
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    db = davies_bouldin_score(features_pca, labels)
    ch = calinski_harabasz_score(features_pca, labels)

    clusters_df = pd.DataFrame(
        {
            "ticker": tickers,
            "cluster": labels + 1,  # 1-indexed for UI
            "silhouette_width": sil_samples,
        }
    )

    metrics = {
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
        "pca_components": int(features_pca.shape[1]),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }

    return clusters_df, metrics, model, linkage_matrix, features_pca


def calculate_sharpe_ratio(returns: pd.Series, periods: int = 252) -> float:
    """Annualized Sharpe ratio (no risk-free rate)."""
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    return float((returns.mean() / returns.std()) * np.sqrt(periods))


def calculate_period_sharpe_ratios(daily_bars_df: pd.DataFrame) -> pd.DataFrame:
    if daily_bars_df.empty:
        return pd.DataFrame()

    results: list[dict] = []
    for ticker in daily_bars_df["ticker"].unique():
        d = daily_bars_df[daily_bars_df["ticker"] == ticker].copy().sort_values("date")
        d["daily_return"] = d["close"].pct_change()

        recent = d[d["date"] >= (d["date"].max() - timedelta(days=30))]
        data_3m = d[d["date"] >= (d["date"].max() - timedelta(days=90))]
        data_6m = d[d["date"] >= (d["date"].max() - timedelta(days=180))]

        results.append(
            {
                "ticker": ticker,
                "sharpe_recent": calculate_sharpe_ratio(recent["daily_return"]),
                "sharpe_3m": calculate_sharpe_ratio(data_3m["daily_return"]),
                "sharpe_6m": calculate_sharpe_ratio(data_6m["daily_return"]),
                "sharpe_overall": calculate_sharpe_ratio(d["daily_return"]),
            }
        )

    return pd.DataFrame(results)


def create_candlestick_chart(ticker_data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=ticker_data["date"],
                open=ticker_data["open"],
                high=ticker_data["high"],
                low=ticker_data["low"],
                close=ticker_data["close"],
                name=ticker,
            )
        ]
    )
    fig.update_layout(
        title=f"{ticker} - Candlestick",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=420,
        template="plotly_white",
    )
    return fig


def _plot_dendrogram(linkage_matrix: np.ndarray, labels: list[str]) -> go.Figure:
    # Build dendrogram segments using scipy's coordinates (no matplotlib dependency)
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    d = scipy_dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        no_plot=True,
    )
    icoord = d["icoord"]
    dcoord = d["dcoord"]
    color_list = d["color_list"]
    ivl = d["ivl"]

    fig = go.Figure()
    # Map matplotlib "C0..C9" to plotly-ish colors
    cmap = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    def to_color(c: str) -> str:
        if c.startswith("C") and c[1:].isdigit():
            idx = int(c[1:])
            return cmap[idx % len(cmap)]
        return "#7f7f7f"

    for xs, ys, c in zip(icoord, dcoord, color_list):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=to_color(c), width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    leaf_positions = [5 + 10 * i for i in range(len(ivl))]
    fig.update_layout(
        title="Hierarchical Dendrogram",
        template="plotly_white",
        height=600,
        margin=dict(b=140),
        xaxis=dict(
            tickmode="array",
            tickvals=leaf_positions,
            ticktext=ivl,
            tickangle=90,
            tickfont=dict(size=9),
        ),
        yaxis_title="Distance",
    )
    return fig


def _silhouette_barplot(df: pd.DataFrame, avg: float, *, show_labels: bool) -> go.Figure:
    sdf = df.sort_values(["cluster", "silhouette_width"], ascending=[True, False]).reset_index(drop=True)
    sdf["y"] = np.arange(len(sdf))

    fig = go.Figure()
    for cluster in sorted(sdf["cluster"].unique()):
        seg = sdf[sdf["cluster"] == cluster]
        fig.add_trace(
            go.Bar(
                x=seg["silhouette_width"],
                y=seg["y"],
                orientation="h",
                name=f"Cluster {cluster}",
                text=seg["ticker"],
                textposition="outside" if show_labels else "none",
                hovertemplate="<b>%{text}</b><br>Silhouette: %{x:.3f}<extra></extra>",
            )
        )

    fig.add_vline(
        x=avg,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {avg:.2f}",
        annotation_position="top",
    )
    fig.update_layout(
        title="Silhouette Analysis",
        template="plotly_white",
        height=max(420, len(sdf) * 14),
        yaxis=dict(showticklabels=show_labels),
        xaxis_title="Silhouette width",
    )
    if show_labels:
        fig.update_yaxes(
            tickmode="array",
            tickvals=sdf["y"],
            ticktext=sdf["ticker"],
        )
    return fig


def _compute_cluster_profiles(
    global_df: pd.DataFrame, clusters_df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    base = global_df.set_index("ticker")
    joined = clusters_df.set_index("ticker").join(base, how="left")
    numeric_cols = [c for c in feature_cols if c in joined.columns]
    if not numeric_cols:
        return pd.DataFrame()
    cluster_means = joined.groupby("cluster")[numeric_cols].mean(numeric_only=True)
    cluster_means = cluster_means.replace([np.inf, -np.inf], np.nan).fillna(0)
    # z-score across clusters for comparability
    cluster_means_z = cluster_means.apply(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0),
        axis=0,
    )
    return cluster_means_z


def _radar_chart_for_cluster(
    cluster_means_z: pd.DataFrame, cluster_id: int, color: str
) -> go.Figure:
    labels = []
    values = []
    for group_name, feats in FEATURE_GROUPS.items():
        available = [f for f in feats if f in cluster_means_z.columns]
        if not available:
            continue
        labels.append(group_name)
        values.append(float(cluster_means_z.loc[cluster_id, available].mean()))

    if not labels:
        return go.Figure()

    labels_plot = labels + [labels[0]]
    values_plot = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_plot,
            theta=labels_plot,
            fill="toself",
            line=dict(color=color, width=2),
            name=f"Cluster {cluster_id}",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(showticklabels=False, ticks="")),
        showlegend=False,
        template="plotly_white",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Stock Clustering Analysis", page_icon="📊", layout="wide")
    st.title("📊 Stock Clustering Segmentation (Model 1)")

    global_data = load_clustering_data()
    features_scaled, feature_cols = prepare_features_for_clustering(global_data)

    # Sidebar controls
    st.sidebar.header("Clustering Parameters")
    method = st.sidebar.selectbox("Method", ["KMeans", "PAM", "Hierarchical"], index=0)
    if method == "PAM" and not HAS_KMEDOIDS:
        st.sidebar.warning("PAM requires `scikit-learn-extra`. Install it to enable PAM.")
    n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=10, value=5, step=1)
    pca_target = 0.65

    run = st.sidebar.button("Run clustering", type="primary")

    tab_overview, tab_analysis, tab_results = st.tabs(["Overview", "Data Analysis", "Clustering Results"])

    with tab_overview:
        st.subheader("Dataset")
        st.write(
            "This dataset is an aggregated feature table where **each row is one stock**. "
            "It is typically generated by `ml/scripts/preprocessing.py` and saved as parquet."
        )
        st.write(f"Rows: **{len(global_data)}**, Features: **{len(feature_cols)}**")
        st.dataframe(global_data.head(10), use_container_width=True)

        st.subheader("Candlestick + Sharpe (optional)")
        st.caption("Candlesticks use Postgres `daily_bars` when available; otherwise it falls back to parquet.")
        tickers = tuple(sorted(global_data["ticker"].astype(str).unique().tolist()))
        daily_bars = load_daily_bars(tickers)
        if daily_bars.empty:
            st.warning("No daily bars found (DB/parquet). Candlestick + Sharpe stats are unavailable.")
        else:
            sharpe_df = calculate_period_sharpe_ratios(daily_bars)
            st.dataframe(sharpe_df.sort_values("sharpe_overall", ascending=False).head(20), use_container_width=True)

            selected = st.selectbox("Select ticker", list(tickers))
            td = daily_bars[daily_bars["ticker"] == selected].copy().sort_values("date")
            if not td.empty:
                max_date = td["date"].max()
                start, end = st.date_input(
                    "Date range",
                    value=(max_date.date() - timedelta(days=180), max_date.date()),
                    min_value=td["date"].min().date(),
                    max_value=max_date.date(),
                )
                td = td[(td["date"] >= pd.Timestamp(start)) & (td["date"] <= pd.Timestamp(end))]
                st.plotly_chart(create_candlestick_chart(td, selected), use_container_width=True)

    with tab_analysis:
        st.subheader("Correlation Matrix (features)")
        feats = features_scaled.copy()
        corr = feats.corr()
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="corr"),
            )
        )
        fig_corr.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Feature Skewness (before PCA)")
        skew = feats.skew(numeric_only=True).sort_values()
        skew_df = pd.DataFrame({"feature": skew.index, "skewness": skew.values})
        fig_skew = px.bar(
            skew_df,
            x="feature",
            y="skewness",
            color="skewness",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Skewness by feature (|skew|>2 is often worth transforming)",
        )
        fig_skew.add_hline(y=2, line_dash="dash", line_color="red")
        fig_skew.add_hline(y=-2, line_dash="dash", line_color="red")
        fig_skew.update_layout(height=450, template="plotly_white")
        st.plotly_chart(fig_skew, use_container_width=True)

        st.subheader("PCA variance (scree + cumulative)")
        pca_full = PCA(random_state=RANDOM_STATE)
        pca_full.fit(features_scaled)
        n_show = min(10, len(features_scaled.columns))
        explained = (pca_full.explained_variance_ratio_[:n_show] * 100).tolist()
        cumulative = np.cumsum(explained).tolist()
        cumulative_ratio = np.cumsum(pca_full.explained_variance_ratio_)
        pca_components = int(np.where(cumulative_ratio >= pca_target)[0][0] + 1)

        fig_pca = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pca.add_trace(go.Bar(x=list(range(1, n_show + 1)), y=explained, name="Explained %"), secondary_y=False)
        fig_pca.add_trace(
            go.Scatter(x=list(range(1, n_show + 1)), y=cumulative, mode="lines+markers", name="Cumulative %"),
            secondary_y=True,
        )
        fig_pca.update_layout(height=450, template="plotly_white")
        fig_pca.update_xaxes(title_text="Component")
        fig_pca.update_yaxes(title_text="Explained variance (%)", secondary_y=False)
        fig_pca.update_yaxes(title_text="Cumulative (%)", secondary_y=True)
        st.plotly_chart(fig_pca, use_container_width=True)
        st.caption(f"PCA target: **{pca_target:.2f}** variance → **{pca_components}** components.")

    with tab_results:
        if not run:
            st.info("Configure parameters in the sidebar and click **Run clustering**.")
            return

        if method == "PAM" and not HAS_KMEDOIDS:
            st.error("PAM requires `scikit-learn-extra`. Choose another method or install the dependency.")
            return

        pca_full = PCA(random_state=RANDOM_STATE)
        pca_full.fit(features_scaled)
        cumulative_ratio = np.cumsum(pca_full.explained_variance_ratio_)
        pca_components = int(np.where(cumulative_ratio >= pca_target)[0][0] + 1)

        clusters_df, metrics, _, linkage_matrix, features_pca = run_clustering(
            method=method,
            n_clusters=n_clusters,
            features_scaled=features_scaled,
            pca_components=pca_components,
        )

        cluster_palette = px.colors.qualitative.Set2
        cluster_color_map = {
            str(c): cluster_palette[(c - 1) % len(cluster_palette)] for c in sorted(clusters_df["cluster"].unique())
        }

        st.subheader(f"Results: {method} ({n_clusters} clusters)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Silhouette", f"{metrics['silhouette']:.4f}")
        c2.metric("Davies-Bouldin (↓)", f"{metrics['davies_bouldin']:.4f}")
        c3.metric("Calinski-Harabasz (↑)", f"{metrics['calinski_harabasz']:.2f}")
        c4.metric("PCA dims", f"{metrics['pca_components']} ({metrics['pca_explained_variance']:.1%})")

        if method == "Hierarchical" and linkage_matrix is not None:
            st.subheader("Dendrogram")
            st.plotly_chart(_plot_dendrogram(linkage_matrix, clusters_df["ticker"].tolist()), use_container_width=True)

        st.subheader("Cluster distribution")
        counts = clusters_df["cluster"].value_counts().sort_index()
        st.plotly_chart(
            px.bar(x=counts.index.astype(str), y=counts.values, labels={"x": "cluster", "y": "stocks"}),
            use_container_width=True,
        )
        st.dataframe(clusters_df.sort_values(["cluster", "ticker"]), use_container_width=True, height=420)

        st.subheader("Silhouette analysis")
        st.plotly_chart(
            _silhouette_barplot(clusters_df, metrics["silhouette"], show_labels=True),
            use_container_width=True,
        )
        bad = clusters_df[clusters_df["silhouette_width"] < 0]
        if not bad.empty:
            st.warning(f"{len(bad)} stock(s) have negative silhouette scores (potentially misclustered).")
            st.dataframe(bad.sort_values("silhouette_width"), use_container_width=True)

        st.subheader("Cluster profiles (radar)")
        cluster_means_z = _compute_cluster_profiles(global_data, clusters_df, feature_cols)
        if cluster_means_z.empty:
            st.info("No numeric features available for cluster profiling.")
        else:
            cols = st.columns(min(2, len(cluster_means_z)))
            for idx, cluster_id in enumerate(sorted(cluster_means_z.index)):
                col = cols[idx % len(cols)]
                with col:
                    fig = _radar_chart_for_cluster(
                        cluster_means_z,
                        int(cluster_id),
                        cluster_color_map.get(str(cluster_id), "#1f77b4"),
                    )
                    if fig.data:
                        st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Cluster {cluster_id} profile (z-scored feature groups)")

        st.subheader("PCA visualization (2D)")
        p2 = PCA(n_components=2, random_state=RANDOM_STATE)
        xy = p2.fit_transform(features_scaled)
        pca_df = pd.DataFrame(
            {"PC1": xy[:, 0], "PC2": xy[:, 1], "cluster": clusters_df["cluster"].astype(str), "ticker": clusters_df["ticker"]}
        )
        fig2d = px.scatter(pca_df, x="PC1", y="PC2", color="cluster", hover_data=["ticker"], title="2D PCA projection")
        st.plotly_chart(fig2d, use_container_width=True)

        if features_pca.shape[1] >= 3:
            st.subheader("PCA visualization (3D)")
            z = features_pca.iloc[:, :3].copy()
            z["cluster"] = clusters_df["cluster"].astype(str).values
            z["ticker"] = clusters_df["ticker"].values
            fig3d = px.scatter_3d(z, x=z.columns[0], y=z.columns[1], z=z.columns[2], color="cluster", hover_name="ticker")
            st.plotly_chart(fig3d, use_container_width=True)

        st.subheader("Conclusion / takeaways")
        largest_cluster = counts.idxmax()
        smallest_cluster = counts.idxmin()
        st.markdown(
            "\n".join(
                [
                    f"- **Cluster balance**: largest cluster is **{largest_cluster}** ({counts[largest_cluster]} stocks), "
                    f"smallest is **{smallest_cluster}** ({counts[smallest_cluster]} stocks).",
                    f"- **Separation quality**: silhouette score is **{metrics['silhouette']:.2f}** "
                    f"({'good' if metrics['silhouette'] >= 0.25 else 'weak'} separation).",
                    f"- **Potential misclusters**: **{len(bad)}** stock(s) have negative silhouette width.",
                    "- Use the radar charts to interpret each cluster’s dominant traits (risk/return, momentum, "
                    "volatility, liquidity).",
                ]
            )
        )

        st.subheader("Compare methods (same k)")
        with st.expander("Run comparison across KMeans / PAM / Hierarchical", expanded=False):
            rows = []
            for m in ["KMeans", "PAM", "Hierarchical"]:
                if m == "PAM" and not HAS_KMEDOIDS:
                    continue
                cdf, mtx, _, _, _ = run_clustering(m, n_clusters, features_scaled, pca_components)
                rows.append(
                    {
                        "method": m,
                        "silhouette": mtx["silhouette"],
                        "davies_bouldin": mtx["davies_bouldin"],
                        "calinski_harabasz": mtx["calinski_harabasz"],
                    }
                )
            comp = pd.DataFrame(rows)
            st.dataframe(comp, use_container_width=True)
            if not comp.empty:
                fig = make_subplots(rows=1, cols=3, subplot_titles=("Silhouette (↑)", "Davies-Bouldin (↓)", "Calinski-Harabasz (↑)"))
                fig.add_trace(go.Bar(x=comp["method"], y=comp["silhouette"], name="sil"), row=1, col=1)
                fig.add_trace(go.Bar(x=comp["method"], y=comp["davies_bouldin"], name="db"), row=1, col=2)
                fig.add_trace(go.Bar(x=comp["method"], y=comp["calinski_harabasz"], name="ch"), row=1, col=3)
                fig.update_layout(height=420, template="plotly_white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
