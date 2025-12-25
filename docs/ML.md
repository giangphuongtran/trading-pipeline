## Trading Pipeline – ML / Modeling

### Where we are now
- **Data ingestion**: daily/intraday/news backfills into Postgres (Airflow scheduled).
- **Feature engineering**: `ml/scripts/prepare_features.py` + `ml/features/*`.
- **Model 1 (done)**: clustering/segmentation (unsupervised) for portfolio grouping:
  - Script: `ml/scripts/clustering_analysis.py`
  - Streamlit visualization: `streamlit/clustering_app.py`

### How to run clustering (repro)

```bash
# Create clustering datasets (global + quarterly aggregates)
python -m ml.scripts.preprocessing \
  --input-path ml/data/daily_data_with_news.parquet \
  --out-path ml/data/clustering

# Run clustering analysis
python -m ml.scripts.clustering_analysis \
  --global-path ml/data/clustering/ds1_global_data.parquet \
  --quarter-path ml/data/clustering/ds1_quarter_data.parquet \
  --output-dir ml/data/clustering/results \
  --n-clusters 5 \
  --linkage-method ward
```

### How to run the dashboard (clustering)

```bash
streamlit run streamlit/clustering_app.py
```

### What’s next (to “complete” the ML side)

1) **Define target(s)** (one clear supervised task)
- Example: next-day return sign (classification) or next-day return magnitude (regression).

2) **Build a time-series-safe training pipeline**
- Train/val split by date (no leakage).
- Baselines: logistic regression / random forest / XGBoost (if allowed).

3) **Evaluation + reporting**
- Metrics: AUC/accuracy (classification), RMSE/MAE (regression)
- Trading-aware: hit-rate, turnover, simple long/short backtest with transaction costs.

4) **Productionization**
- Persist features and predictions back to DB (add tables like `daily_features`, `predictions`).
- Add model/version metadata (`model_version`, `feature_version`).
- Add drift checks on feature distributions (optional).


