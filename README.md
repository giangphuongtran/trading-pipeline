# Trading Pipeline

Batch-first trading data pipeline that ingests market data (OHLCV + news sentiment) from Polygon.io, stores it in Postgres, and produces ML-ready features for research + modeling. Orchestrated with Airflow and runnable locally with Docker Compose.

## What this project does

- **Ingests**: daily bars, intraday (5-min) bars, and news from Polygon.io
- **Stores**: normalized raw tables in Postgres (`daily_bars`, `intraday_bars`, `news_articles`) + ingestion metadata (`api_metadata`)
- **Orchestrates**: Airflow DAGs schedule the batch ingestion
- **Builds features**: feature engineering code under `ml/` for modeling workflows
- **Includes Model 1**: unsupervised clustering + Streamlit visualization (`streamlit/clustering_app.py`)

## What this pipeline has

- **Orchestration**: Airflow DAGs in `airflow/dags/`
- **Storage**: Postgres via Docker Compose (`docker-compose.yml`, `db/*.sql`)
- **Batch ingestion**: Backfill CLIs (daily/intraday/news) + metadata tracking (`api_metadata`)
- **Quality gates (optional)**: Pandera-based schema + invariant checks (toggle with `ENABLE_DATA_QUALITY_CHECKS=1`)
- **Observability (baseline)**: Structured logging option (`LOG_FORMAT=json`) + failure status recorded in `api_metadata`
- **Reproducibility**: `Makefile`, `project_setup.sh`, `env.example`
- **Tests**: Unit tests for backfill + CLI planning (`tests/`)
- **CI**: GitHub Actions workflow at `.github/workflows/ci.yml` (ruff + pytest)

## Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- PostgreSQL database
- Polygon.io API key

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd trading-pipeline
   ```

2. **Configure environment**
   - Copy `env.example` to `.env`
   - Set at least `POLYGON_API_KEY=...` (required)
   - Optional:
     - Set `ENABLE_DATA_QUALITY_CHECKS=1` to fail fast on bad data
     - Set `LOG_FORMAT=json` for structured logs

3. **Run setup script**
   ```bash
   ./project_setup.sh
   ```

4. **Start services** (if not already started)
   ```bash
   docker compose up -d
   ```

5. **Verify installation**
   ```bash
   # Test database connection (host context)
   python -c "from app.config import connect_db; conn = connect_db(use_docker=False); conn.close(); print('✅ DB connected')"
   ```

## One-minute demo (interview-friendly)

```bash
# 1) Start services
make up

# 2) Follow Airflow logs
make logs

# 3) (Optional) run a backfill locally (host DB URL)
make backfill-daily

# 4) Run Model 1 clustering dashboard
streamlit run streamlit/clustering_app.py
```

## Project Structure

```
trading-pipeline/
├── app/                    # Application code
│   ├── backfill/          # Data backfill modules
│   ├── config.py          # Configuration management
│   └── polygon_trading_client.py  # API client
├── ml/                     # Machine learning code
│   ├── features/          # Feature engineering modules
│   ├── models/            # ML model definitions and training
│   ├── scripts/           # Feature preparation scripts
│   └── notebooks/         # Jupyter notebooks
├── airflow/               # Airflow DAGs
│   └── dags/             # Scheduled tasks
├── db/                    # Database scripts
│   ├── 00_init.sql       # Database initialization
│   └── 01_create_tables.sql  # Table definitions
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md    # System design + decisions
│   ├── RUNBOOK.md         # Ops / troubleshooting / manual backfills
│   ├── PIPELINE_DIAGRAM.md # Mermaid diagrams
│   ├── FEATURES.md        # Feature catalog
│   └── ML.md              # Modeling status + how to run clustering
├── tests/                 # Test files
└── docker-compose.yml     # Infrastructure setup
```

## Documentation

### Essential Reading
- **[Architecture](./docs/ARCHITECTURE.md)**: System design, data flow, and production-ready decisions
- **[Runbook](./docs/RUNBOOK.md)**: How to run, operate, and troubleshoot the pipeline
- **[Pipeline Diagram](./docs/PIPELINE_DIAGRAM.md)**: Detailed architecture + DAG + data model diagrams (Mermaid)
- **[Feature Catalog](./docs/FEATURES.md)**: Feature definitions and modeling notes
- **[ML / Modeling](./docs/ML.md)**: Current model status + how to reproduce clustering
- **[API Examples](./misc/API_EXAMPLE.md)**: Polygon.io API usage examples

### Project Planning
- **[Project Thinking](./PROJECT_THINKING.md)**: Early design notes (optional reading)

## How it works

- Airflow runs `python -m app.backfill_daily`, `app.backfill_intraday`, `app.backfill_news` on schedule.
- Each backfill fetches from Polygon, optionally validates via Pandera, UPSERTs to Postgres, and writes run status into `api_metadata`.
- Feature engineering lives under `ml/` (see `docs/FEATURES.md`).

## Usage Examples

### Feature Engineering Pipeline

```python
from ml.scripts.prepare_features import prepare_daily_features

# Generate daily features
features = prepare_daily_features(
    ticker="AAPL",
    save_path="data/daily_features.parquet"
)
```

### Data Quality Check

```python
from ml.scripts.prepare_features import _load_daily_bars, _load_intraday_bars
from app.config import connect_db

conn = connect_db(use_docker=False)
daily_bars, daily_gaps = _load_daily_bars(conn, "AAPL")
intraday_bars, intraday_gaps = _load_intraday_bars(conn, "AAPL", time_config)

# Check for gaps
if not intraday_gaps.empty:
    print(f"Found {len(intraday_gaps)} gaps")
    print(intraday_gaps[['missing_timestamps', 'missing_count']])
```

### Notebook Workflow

See `ml/notebooks/full_feature_pipeline.ipynb` for an interactive feature engineering workflow.

## Quick Reference

### Common Commands

```bash
# Backfill data (market indices like SPY are included automatically)
python -m app.backfill_daily --mode resume
python -m app.backfill_intraday --mode resume --tickers AAPL MSFT
python -m app.backfill_news --mode full --start-date 2024-01-01

# Generate features
python -m ml.scripts.prepare_features --model daily --ticker AAPL --save data/features.parquet
python -m ml.scripts.prepare_features --model intraday --ticker AAPL

# Start services
docker compose up -d
docker compose logs -f airflow-webserver airflow-scheduler

# Run tests
pytest tests/ -v
```

### Common Issues

| Issue | Solution |
|-------|----------|
| `ImportError: cannot import name 'X' from 'ml.features'` | Check `ml/features/__init__.py` - ensure all classes are exported |
| `ValueError: mutable default for field` | Use `field(default_factory=...)` instead of mutable defaults in dataclasses |
| `NameError: name '__file__' is not defined` | Use try/except block or `Path.cwd()` fallback in notebooks |
| Database connection fails with `local` hostname | Use `localhost` or actual IP address in `.env` file |
| Gap detection shows false positives for holidays | Manually filter known holidays (see list above) |

### Key File Locations

- **Feature Engineering**: `ml/scripts/prepare_features.py`
- **Gap Detection**: `ml/scripts/prepare_features.py::_warn_if_timestamp_gaps()`
- **Feature Classes**: `ml/features/`
- **Notebooks**: `ml/notebooks/full_feature_pipeline.ipynb`
- **Database Schema**: `db/01_create_tables.sql`
- **API Client**: `app/polygon_trading_client.py`

## Known Issues and Limitations

### Data Quality
- **Holiday Gaps**: 4-day gaps (Friday to Tuesday) are often market holidays, not data issues (see list above).
- **Missing Data**: Gap detection identifies missing timestamps, but automatic backfill is not yet implemented.

### Technical Debt
- Holiday calendar integration for gap detection (planned)
- Automatic backfill mechanism for missing data (planned)

For ops/troubleshooting, see the [Runbook](./docs/RUNBOOK.md).

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
ruff check app tests
```

### Adding New Features
1. Create feature branch
2. Implement feature with tests
3. Update documentation
4. Submit pull request

## Contributing

1. Use feature branches + PRs (even if solo, it demonstrates professional workflow)
2. Update documentation when behavior changes (`docs/ARCHITECTURE.md`, `docs/RUNBOOK.md`, `docs/ML.md`)
3. Add/adjust tests for new behavior (`pytest`)
4. Keep CI green (ruff + pytest)

## Support

For issues and questions:
- Start with the [Runbook](./docs/RUNBOOK.md) (common failure modes + fixes)
- Review [Architecture](./docs/ARCHITECTURE.md) for system context
- Open an issue on GitHub
