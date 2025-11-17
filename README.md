# Trading Pipeline Project

A comprehensive data pipeline for collecting, processing, and engineering features from financial market data using Polygon.io API.

## Overview

This project provides:
- **Data Collection**: Automated backfilling of daily bars, intraday bars, and news data
- **Feature Engineering**: Comprehensive feature sets for machine learning models
- **Data Quality**: Gap detection and data quality monitoring
- **Orchestration**: Airflow DAGs for scheduled data updates

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

2. **Run setup script**
   ```bash
   ./project_setup.sh
   ```

3. **Configure environment**
   - Copy `.env.example` to `.env`
   - Fill in your credentials (database, API keys)

4. **Start services**
   ```bash
   docker compose up -d
   ```

5. **Verify installation**
   ```bash
   # Test database connection
   python -c "from app.config import get_db_connection; print('✅ DB connected')"
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
│   ├── BUGS_AND_FIXES.md # Bug tracking
│   ├── DOCUMENTATION_TEMPLATE.md  # Documentation guide
│   └── manual_backfill_commands.md  # CLI usage
├── tests/                 # Test files
└── docker-compose.yml     # Infrastructure setup
```

## Documentation

### Essential Reading
- **[Bugs and Fixes](./docs/BUGS_AND_FIXES.md)**: Complete list of bugs discovered and fixed, including data quality issues
- **[Project Overview](./docs/PROJECT_OVERVIEW.md)**: Comprehensive project documentation (architecture, data schema, deployment, maintenance)
- **[Documentation Template](./docs/DOCUMENTATION_TEMPLATE.md)**: Comprehensive guide for documenting data science/engineering projects
- **[Manual Backfill Commands](./docs/manual_backfill_commands.md)**: How to manually trigger data backfills
- **[API Examples](./misc/API_EXAMPLE.md)**: Polygon.io API usage examples

### Project Planning
- **[Project Thinking](./PROJECT_THINKING.md)**: Original project plan and architecture decisions

## Key Features

### Data Collection
- **Daily Bars**: End-of-day OHLCV data
- **Intraday Bars**: 5-minute aggregated bars
- **News Data**: Market news with sentiment analysis

### Feature Engineering
- **Price Features**: Returns, volatility, moving averages
- **Volume Features**: Volume ratios, trends, spikes
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX
- **Time Features**: Session windows, cyclical encoding
- **News Features**: Sentiment aggregation, rolling averages
- **Candlestick Patterns**: Pattern detection for intraday data
- **Confluence Features**: Combined signals across all feature sets

### Data Quality
- **Gap Detection**: Identifies missing daily and intraday bars
- **Missing Timestamp Identification**: Lists exact missing timestamps for backfilling
- **Holiday Awareness**: (Planned) Distinguish holidays from data gaps

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
from ml.scripts.prepare_features import _load_daily_bars, _load_intraday_bars, _connect_db

conn = _connect_db()
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
# Backfill data
python -m app.backfill.backfill_daily --mode resume
python -m app.backfill.backfill_intraday --mode resume --tickers AAPL MSFT
python -m app.backfill.backfill_news --mode full --start-date 2024-01-01

# Generate features
python -m ml.scripts.prepare_features --model daily --ticker AAPL --save data/features.parquet
python -m ml.scripts.prepare_features --model intraday --ticker AAPL

# Start services
docker compose up -d
docker compose logs -f airflow

# Run tests
pytest tests/ -v
```

### Market Holidays (2023-2025)

These holidays cause legitimate 4-day gaps (Friday → Tuesday) in daily data:

**2023**: Christmas (Dec 25), New Year (Jan 1, 2024)  
**2024**: MLK Day (Jan 15), Presidents' Day (Feb 19), Good Friday (Mar 29), Memorial Day (May 27), Labor Day (Sep 2)  
**2025**: MLK Day (Jan 20), Presidents' Day (Feb 17), Good Friday (Apr 18), Memorial Day (May 26), Independence Day (Jul 4), Labor Day (Sep 1)

See [Bugs and Fixes](./docs/BUGS_AND_FIXES.md#market-holidays) for complete holiday list.

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
- **Holiday Gaps**: 4-day gaps (Friday to Tuesday) are often market holidays, not data issues. See [Bugs and Fixes](./docs/BUGS_AND_FIXES.md#market-holidays) for complete holiday list.
- **Missing Data**: Gap detection identifies missing timestamps, but automatic backfill is not yet implemented.

### Technical Debt
- Holiday calendar integration for gap detection (planned)
- Automatic backfill mechanism for missing data (planned)

See [Bugs and Fixes](./docs/BUGS_AND_FIXES.md) for complete list of known issues.

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black .
flake8 .
mypy .
```

### Adding New Features
1. Create feature branch
2. Implement feature with tests
3. Update documentation
4. Submit pull request

## Contributing

1. Read the [Documentation Template](./docs/DOCUMENTATION_TEMPLATE.md) for best practices
2. Document all bugs in [Bugs and Fixes](./docs/BUGS_AND_FIXES.md)
3. Update relevant documentation when making changes
4. Add tests for new features
5. Follow existing code patterns

## License

[Add your license here]

## Support

For issues and questions:
- Check [Bugs and Fixes](./docs/BUGS_AND_FIXES.md) for known issues
- Review [Documentation Template](./docs/DOCUMENTATION_TEMPLATE.md) for project structure
- Open an issue on GitHub

---

**Last Updated**: 2024-01-XX
**Maintained By**: [Your Name/Team]
