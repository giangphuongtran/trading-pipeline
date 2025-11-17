# Data Science & Engineering Project

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Setup and Installation](#2-setup-and-installation)
3. [Architecture and Design](#3-architecture-and-design)
4. [Data Documentation](#4-data-documentation)
5. [Code Documentation](#5-code-documentation)
6. [Bug Tracking](#6-bug-tracking)
7. [Testing](#7-testing)
8. [Deployment](#8-deployment)
9. [Maintenance](#9-maintenance)
10. [Lessons Learned](#10-lessons-learned)

---

## 1. Project Overview

### 1.1 Purpose
A comprehensive data pipeline for collecting, processing, and engineering features from financial market data using Polygon.io API. The system builds machine learning-ready feature sets for predicting trading trends (daily/intraday) based on OHLCV data and news sentiment.

### 1.2 Scope
**In-scope:**
- Automated data collection (daily bars, intraday bars, news)
- Feature engineering pipeline (price, volume, technical indicators, sentiment, time-based)
- Data quality monitoring (gap detection, missing data identification)
- Airflow orchestration for scheduled updates
- Interactive Jupyter notebook for feature exploration

**Out-of-scope (future work):**
- Real-time trading execution
- Model deployment infrastructure
- Web dashboard/UI
- Multi-exchange support

**Assumptions:**
- PostgreSQL database available
- Polygon.io API access
- Python 3.9+ environment
- Market hours: 9:30 AM - 4:00 PM ET (configurable)

### 1.3 Timeline
- Project start date: August 2025
- Current status: Active development

---

## 2. Setup and Installation

### 2.1 Prerequisites
- Python 3.9+
- Docker and Docker Compose
- PostgreSQL database (local or remote)
- Polygon.io API key

### 2.2 Installation Steps
```bash
# 1. Clone repository
git clone <repository-url>
cd trading-pipeline

# 2. Run setup script (creates database, initializes schema)
./project_setup.sh
```

### 2.3 Environment Setup
**Already done with ./project_setup.sh**

### 2.4 Verification
```bash
# Test database connection
python -c "from app.config import connect_db; conn = connect_db(); print('✅ Connected')"

# Test API client
python -c "from app.polygon_trading_client import PolygonTradingClient; client = PolygonTradingClient(); print('✅ API client initialized')"

# Run tests
pytest tests/ -v
```

---

## 3. Architecture and Design

### 3.1 System Architecture
```
┌─────────────┐
│ Polygon.io  │───API Calls───┐
│    API      │                │
└─────────────┘                │
                               ▼
                    ┌──────────────────┐
                    │  Backfill Scripts│
                    │  (Python)        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   PostgreSQL     │
                    │   Database       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Feature Pipeline │
                    │  (prepare_features)│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  ML Models       │
                    │  (Training/Inference)│
                    └──────────────────┘
```

### 3.2 Data Flow
1. **Data Collection**: Polygon.io API → Backfill scripts (`app/backfill/`) → PostgreSQL
2. **Feature Engineering**: PostgreSQL → `ml/scripts/prepare_features.py` → Feature DataFrames
3. **Model Training**: Features → `ml/models/` → Trained models
4. **Orchestration**: Airflow DAGs schedule daily/intraday/news backfills

### 3.3 Key Design Decisions
- **PostgreSQL**: Reliable, ACID-compliant, good for time-series with proper indexing
- **pandas-ta**: Standard technical indicators library, well-maintained
- **Airflow**: Industry-standard orchestration, good for scheduled data pipelines
- **Modular feature engineering**: Separate classes for each feature type (price, volume, etc.) for maintainability
- **Configuration dataclasses**: Type-safe, easy to customize feature windows

### 3.4 Directory Structure
```
trading-pipeline/
├── app/                    # Application code
│   ├── backfill/          # Data backfill modules
│   │   ├── backfill_daily.py
│   │   ├── backfill_intraday.py
│   │   ├── backfill_news.py
│   │   └── cli.py         # Shared CLI utilities
│   ├── config.py          # Database connection & insert helpers
│   ├── polygon_trading_client.py  # API client with retry logic
│   └── symbols.py         # Ticker symbol lists
├── ml/                     # Machine learning code
│   ├── features/          # Feature engineering modules
│   │   ├── price_features.py
│   │   ├── volume_features.py
│   │   ├── technical_indicators.py
│   │   ├── news_features.py
│   │   ├── time_features.py
│   │   ├── candlestick_features.py
│   │   ├── confluence_features.py
│   │   └── sentiment_models.py
│   ├── models/            # ML model definitions
│   ├── scripts/           # Feature preparation scripts
│   │   └── prepare_features.py  # Main feature pipeline
│   └── notebooks/         # Jupyter notebooks
│       └── full_feature_pipeline.ipynb
├── airflow/               # Airflow DAGs
│   └── dags/             # Scheduled tasks
├── db/                    # Database scripts
│   ├── 00_init.sql       # Database initialization
│   └── 01_create_tables.sql  # Table definitions
├── docs/                  # Documentation
├── tests/                 # Test files
└── docker-compose.yml     # Infrastructure setup
```

---

## 4. Data Documentation

### 4.1 Data Sources
- **Polygon.io API**: Primary data source
  - Daily bars: End-of-day OHLCV data
  - Intraday bars: 5-minute aggregated bars
  - News articles: Market news with vendor sentiment scores

### 4.2 Data Schema

**daily_bars**
- `ticker` (VARCHAR): Stock symbol
- `date` (DATE): Trading date
- `open, high, low, close` (DECIMAL): OHLC prices
- `volume` (BIGINT): Trading volume
- `transactions` (BIGINT): Number of transactions
- `volume_weighted_avg_price` (DECIMAL): VWAP
- Unique constraint: `(ticker, date)`

**intraday_bars**
- `ticker` (VARCHAR): Stock symbol
- `timestamp` (TIMESTAMPTZ): Bar timestamp (5-minute intervals)
- `open, high, low, close` (DECIMAL): OHLC prices
- `volume` (BIGINT): Trading volume
- `transactions` (BIGINT): Number of transactions
- `volume_weighted_avg_price` (DECIMAL): VWAP
- Unique constraint: `(ticker, timestamp)`

**news_articles**
- `id` (VARCHAR): Article ID (primary key)
- `ticker` (VARCHAR): Related stock symbol
- `published_at` (TIMESTAMPTZ): Publication timestamp
- `title, description` (TEXT): Article content
- `sentiment_score` (DECIMAL): Vendor sentiment (-1 to 1)
- `sentiment_label` (VARCHAR): "positive", "negative", "neutral"
- `keywords, tickers` (TEXT[]): Related keywords and tickers

**api_metadata**
- Tracks backfill progress and status per ticker/data_type

### 4.3 Data Quality
- **Gap Detection**: Automatically flags missing daily bars (>3 days) and intraday bars (>7.5 minutes)
- **Missing Timestamps**: Identifies exact missing timestamps for backfilling
- **Holiday Awareness**: Weekend/holiday gaps (201900s, 29100s) are excluded from intraday gap detection
- **Data Validation**: Unique constraints prevent duplicates, UPSERT handles updates

### 4.4 Data Dependencies
- **Upstream**: Polygon.io API (rate limits: ~5 req/min free tier)
- **Downstream**: Feature engineering pipeline → ML models
- **Data Lineage**: `api_metadata` table tracks when data was fetched

### 4.5 Known Issues and Limitations
- **Holiday Gaps**: 4-day daily gaps often indicate market holidays (not data issues). See [BUGS_AND_FIXES.md](./BUGS_AND_FIXES.md) for holiday list.
- **API Rate Limits**: Free tier limited to ~5 requests/minute (20s delay enforced)
- **Missing Data**: Gap detection identifies issues but requires manual backfill (automatic backfill function available)
- **Timezone**: All timestamps stored in UTC, market hours assume ET timezone

---

## 5. Code Documentation

### 5.1 Module Overview

**app/polygon_trading_client.py**
- Purpose: Polygon.io API client with rate limiting and retry logic
- Key classes: `PolygonTradingClient`
- Methods: `get_daily_bars()`, `get_intraday_bars()`, `get_news()`
- Handles: Rate limiting, network errors, 429 responses

**app/config.py**
- Purpose: Database connection and data insertion utilities
- Key functions: `connect_db()`, `insert_daily_bars()`, `insert_intraday_bars()`, `insert_news_articles()`
- Handles: Connection pooling, UPSERT operations, transaction management

**ml/scripts/prepare_features.py**
- Purpose: Main feature engineering pipeline
- Key functions: `prepare_daily_features()`, `prepare_intraday_features()`, `backfill_missing_intraday_timestamps()`
- Data loading: `_load_daily_bars()`, `_load_intraday_bars()`, `_load_news()`
- Quality checks: `_warn_if_date_gaps()`, `_warn_if_timestamp_gaps()`

**ml/features/** (Feature Engineering Modules)
- `price_features.py`: Returns, volatility, moving averages
- `volume_features.py`: Volume ratios, trends, spikes
- `technical_indicators.py`: RSI, MACD, Bollinger Bands, ATR, ADX
- `news_features.py`: Sentiment aggregation, rolling averages
- `time_features.py`: Session windows, cyclical encoding
- `candlestick_features.py`: Pattern detection
- `confluence_features.py`: Combined signals
- `sentiment_models.py`: Rule-based and LLM sentiment analysis

### 5.2 API Documentation
All functions have comprehensive docstrings. Key entry points:

```python
# Feature engineering
from ml.scripts.prepare_features import prepare_daily_features, prepare_intraday_features
features = prepare_daily_features(ticker="AAPL", save_path="data/features.parquet")

# Data loading with quality checks
from ml.scripts.prepare_features import _load_daily_bars, _connect_db
conn = _connect_db()
bars, gaps = _load_daily_bars(conn, "AAPL")

# Backfill missing data
from ml.scripts.prepare_features import backfill_missing_intraday_timestamps
from app.polygon_trading_client import PolygonTradingClient
client = PolygonTradingClient()
summary = backfill_missing_intraday_timestamps(gaps, conn, client)
```

### 5.3 Configuration
- **Environment Variables**: `.env` file (see Setup section)
- **Feature Configs**: Dataclasses in `ml/features/indicator_config.py`
  - `PriceFeatureConfig`: Return windows, SMA/EMA periods, volatility windows
  - `VolumeFeatureConfig`: Volume windows, spike thresholds
  - `TechnicalIndicatorConfig`: RSI length, MACD parameters, etc.
  - `TimeFeatureConfig`: Market hours, session windows
  - `NewsFeatureConfig`: Lookback windows, aggregation settings
- **Defaults**: All configs have sensible defaults, override in code

### 5.4 Common Patterns
- **Feature Engineers**: All follow `create_features(df) -> df` pattern
- **Configuration**: Dataclasses with type hints for all feature configs
- **Error Handling**: Try-except with meaningful error messages
- **Data Quality**: Gap warnings stored in DataFrame `.attrs["warnings"]`
- **Naming**: snake_case for functions, PascalCase for classes
- **Documentation**: All public functions have docstrings

---

## 6. Bug Tracking

### 6.1 Bug Log
See [BUGS_AND_FIXES.md](./BUGS_AND_FIXES.md) for complete bug tracking.

**Summary of Fixed Bugs:**
1. Intraday gap detection now identifies specific missing timestamps
2. Dataclass mutable default argument error fixed
3. Missing imports in `ml.features` fixed
4. `__file__` NameError in notebooks fixed
5. Database query parameter error for multiple tickers fixed
6. Environment variable loading in notebooks fixed
7. Database hostname resolution error documented

### 6.2 Known Issues
**Open Issues:**
- **Holiday Detection**: Daily gaps don't distinguish holidays from real data gaps (planned: integrate `pandas_market_calendars`)
- **Automatic Backfill**: Manual process required, though `backfill_missing_intraday_timestamps()` function exists

**Limitations:**
- Free tier API rate limits (~5 req/min)
- No real-time data streaming
- Single exchange support (NYSE/NASDAQ)

**Technical Debt:**
- Add unit tests for gap detection edge cases
- Integrate market calendar for holiday detection
- Add data quality dashboard

---

## 7. Testing

### 7.1 Test Strategy
- **Unit Tests**: Feature engineering functions, data loading utilities
- **Integration Tests**: Backfill scripts, database operations
- **Data Quality Tests**: Gap detection, missing data identification

### 7.2 Test Coverage
- Current coverage: Basic tests for backfill modules
- Critical paths: Database operations, API client retry logic
- Gaps: Feature engineering edge cases, gap detection edge cases

### 7.3 Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_backfill_daily.py -v

# Run with coverage
pytest tests/ --cov=app --cov=ml --cov-report=html
```

### 7.4 Test Data
- Test data: Uses real Polygon.io API (requires API key)
- Mock data: Can be generated using sample tickers
- Test database: Use separate test database or Docker container

---

## 8. Deployment

### 8.1 Deployment Process
**Local Development:**
```bash
# Start services
docker compose up -d

# Run backfills manually
python -m app.backfill.backfill_daily --mode resume
```

**Production (Airflow):**
- DAGs in `airflow/dags/` run on schedule
- `trading_data_backfill.py`: Master DAG orchestrating all backfills
- Individual DAGs: `daily_bars_backfill.py`, `intraday_bars_backfill.py`, `news_backfill.py`

**Environments:**
- Development: Local Docker setup
- Production: Airflow scheduler with production database

### 8.2 Monitoring
- **Database**: Monitor table sizes, query performance
- **API Usage**: Track Polygon.io API calls and rate limits
- **Data Quality**: Review gap warnings regularly
- **Airflow**: Monitor DAG run status, task failures

### 8.3 Troubleshooting
**Common Issues:**
- **Database connection errors**: Check `DATABASE_URL_HOST` in `.env`
- **API rate limit errors**: Increase `rate_limit_delay` in `PolygonTradingClient`
- **Missing data**: Check gap warnings, run backfill for specific dates
- **Import errors**: Ensure project root is in `PYTHONPATH`

**Debug Commands:**
```bash
# Check database connection
python -c "from app.config import connect_db; connect_db()"

# Test API client
python -c "from app.polygon_trading_client import PolygonTradingClient; PolygonTradingClient()"

# Check for gaps
python -c "from ml.scripts.prepare_features import _load_daily_bars, _connect_db; conn = _connect_db(); bars, gaps = _load_daily_bars(conn, 'AAPL'); print(f'Gaps: {len(gaps)}')"
```

---

## 9. Maintenance

### 9.1 Regular Tasks
- **Daily**: Airflow DAGs automatically backfill latest data
- **Weekly**: Review gap warnings, backfill missing data
- **Monthly**: Update dependencies, review feature performance
- **Quarterly**: Database maintenance (vacuum, analyze), review technical debt

### 9.2 Change Log
**Recent Changes:**
- Enhanced intraday gap detection with missing timestamp identification
- Added automatic backfill function for missing intraday bars
- Improved documentation with comprehensive docstrings
- Removed unused dependencies (streamlit, plotly, etc.)
- Fixed dataclass mutable default arguments
- Added support for multiple tickers in database queries

---

## 10. Lessons Learned

### 10.1 What Went Well
- **Modular feature engineering**: Separate classes for each feature type made code maintainable
- **Configuration dataclasses**: Type-safe configs with defaults reduced errors
- **Comprehensive docstrings**: Made codebase easy to understand for new developers
- **Gap detection enhancement**: Identifying specific missing timestamps was crucial for data quality

### 10.2 What Could Be Improved
- **Holiday detection**: Should have integrated market calendar from the start
- **Testing**: More unit tests for edge cases would have caught bugs earlier
- **Dependencies**: Removed unused libraries earlier to reduce bloat
- **Error handling**: More specific error messages would improve debugging

### 10.3 Recommendations for Future Projects
- **Test edge cases**: Especially for time-series data (gaps, holidays, timezones)
- **Keep dependencies minimal**: Regularly audit and remove unused packages
- **Use type hints**: Catches errors early, improves IDE support
- **Centralize configuration**: Environment variables and dataclasses work well

---

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

### Important Links
- **Documentation**: `docs/` directory
- **Bug Tracking**: [BUGS_AND_FIXES.md](./BUGS_AND_FIXES.md)
- **Manual Backfill**: [manual_backfill_commands.md](./manual_backfill_commands.md)
- **Quick Reference**: See [README.md](../README.md#quick-reference) for common commands and quick lookup

### Key Files
- **Feature Pipeline**: `ml/scripts/prepare_features.py`
- **API Client**: `app/polygon_trading_client.py`
- **Database Config**: `app/config.py`
- **Notebook**: `ml/notebooks/full_feature_pipeline.ipynb`

---

## Tips for Good Documentation

1. **Write for my future self**: Assume I'll forget everything in 6 months
2. **Keep it updated**: Outdated docs are worse than no docs
3. **Use examples**: Show, don't just tell
4. **Include context**: Explain why, not just what
5. **Make it searchable**: Use clear headings and structure
6. **Version control**: Keep docs in git, track changes
7. **Review regularly**: Schedule doc review sessions
8. **Get feedback**: Have others read your docs
9. **Link everything**: Cross-reference related docs
10. **Keep it concise**: Long docs are hard to maintain

---

---

## Additional Resources

- **Full Bug Log**: See [BUGS_AND_FIXES.md](./BUGS_AND_FIXES.md) for detailed bug entries
- **Quick Reference**: See [README.md](../README.md#quick-reference) for common commands and information

