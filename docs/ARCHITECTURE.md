## Trading Pipeline – Architecture

### Purpose
Batch-first trading data pipeline that ingests market data (OHLCV + news sentiment) from Polygon.io, stores it in Postgres, and produces ML-ready features for research and modeling.

### High-level components
- **Orchestration**: Airflow (`airflow/dags/`) schedules and runs backfill modules.
- **Ingestion / ETL**: Python modules in `app/`:
  - `app.backfill_daily`, `app.backfill_intraday`, `app.backfill_news`
  - `app.polygon_trading_client.PolygonTradingClient` (rate limiting + retries)
  - `app.config` (DB connection + inserts + metadata)
- **Storage**: Postgres (Docker Compose) initialized by `db/00_init.sql` and `db/01_create_tables.sql`
- **ML Feature Engineering**: `ml/scripts/prepare_features.py` and `ml/features/*`
- **UI (optional)**: Streamlit clustering app in `streamlit/`

### Data flow (batch)
1. Airflow triggers a backfill task (daily / intraday / news).
2. Backfill module calls Polygon API, applies optional data-quality validation, and UPSERTs into Postgres.
3. `api_metadata` is updated for observability (rows inserted + status + error).
4. Feature pipeline reads from Postgres / parquet, engineers features, and saves datasets for modeling.

### Data model (tables)
- **`daily_bars`**: daily OHLCV per (ticker, date)
- **`intraday_bars`**: 5-min OHLCV per (ticker, timestamp)
- **`news_articles`**: article metadata + sentiment fields
- **`api_metadata`**: ingestion/backfill runs, status, and row counts

See `docs/PIPELINE_DIAGRAM.md` for an ER diagram and end-to-end flow charts.

### Reliability & quality gates
- **Optional schema/invariant checks**: Pandera validation in `app/quality/*`
  - Toggle with `ENABLE_DATA_QUALITY_CHECKS=1`
  - Fails fast if data violates core assumptions (e.g., negative prices, `high < low`)
- **CI**: `.github/workflows/ci.yml` runs `ruff` + `pytest` on push/PR.

### Observability
- **Logs**: structured logging supported via `LOG_FORMAT=json` (default `plain`)
- **Operational state**: `api_metadata.status` + `rows_inserted` enables “did we ingest?” queries.

### Key design decisions (interview-friendly)
- **Batch-first**: simpler, cheaper, and sufficient for backtesting + research; streaming can be added later.
- **Postgres**: reliable single-store for ingestion + features at this scale; can evolve to lake/warehouse later.
- **Airflow**: standard orchestration for scheduling, retries, and operational control.

### Extension points (what’s next in production)
- Add “bronze/silver/gold” layers (raw snapshots → cleaned tables → feature/prediction tables).
- Add alerting (Airflow callbacks / Slack/email) + freshness SLAs.
- Add model versioning + prediction persistence (MLflow or lightweight registry).


