# Trading Pipeline Project Plan

## Day 1: Infrastructure Setup

### 1. requirements.txt
- Python dependencies for the project

### 2. docker-compose.yml
- Infrastructure services:
  - **PostgreSQL**: Store historical data for training and reporting
  - **Airflow**: Schedule backfill jobs

### 3. .env
- Project credentials (database, Airflow, API keys)
- Template: `.env.example`

### 4. /db folder
- SQL scripts for database initialization:
  - `00_init.sql`: Create databases
  - `01_create_tables.sql`: Create tables, indexes and triggers

### 5. project_setup.sh
**Goal**: Automated setup script for new users (they get latest 6 months data; full historical data available separately)

**Tasks**:
- Create `.env` file from `.env.example`
- Create virtual environment (`.venv`)
- Install Python dependencies
- Start Docker services
- Verify PostgreSQL connection
- Generate Fernet key for Airflow
- Add Fernet key to `.env` file
- Guide user to fill in credentials

---

## Day 2: Application Code

### /app folder structure

#### /app/data
- `symbols.py`: Trading symbols used across the project
- `config.py`: Configuration parameters (no hardcoding)
- `polygon_trading_client.py`: Custom API client using Polygon RESTClient

#### Backfill modules
- `backfill_daily.py`: Daily price data backfill
- `backfill_intraday.py`: Intraday price data backfill
- `backfill_news.py`: News data backfill

**Requirements**:
- Use custom API from `polygon_trading_client.py`
- Loop to fetch data and insert into database
- Include error handling and logging

#### /app/utils
- Shared helpers:
  - Logging utilities
  - Database connection handling
  - API error handling

### Unit tests
- API response handling
- Database integration
- Individual function tests

---

## Day 3: Airflow DAGs

### Airflow tasks for backfilling
- 3 tasks for 3 data types (daily, intraday, news)
- Sequential execution to avoid rate limits
- Retry logic and alerting
- DAG factory pattern for scalability

---

## Day 4: Model
