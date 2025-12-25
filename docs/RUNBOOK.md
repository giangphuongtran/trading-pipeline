## Trading Pipeline – Runbook

### First clone

```bash
git clone <repo>
cd trading-pipeline

cp env.example .env
# edit .env and set POLYGON_API_KEY=...

./project_setup.sh
```

What this does:
- creates `.env` (if missing), creates `.venv`, installs deps
- generates `AIRFLOW_FERNET_KEY`
- starts Docker services (Postgres + Airflow)

### Common day-to-day commands

```bash
# Start/stop Docker services
make up
make down

# Follow Airflow logs (scheduler + webserver)
make logs

# Run unit tests (same as CI)
make test
```

### Manual backfill

Run inside the Airflow container (runs when using Docker DB networking):

```bash
# Daily
docker compose run --rm airflow-scheduler \
  python -m app.backfill_daily --mode full --tickers AAPL --start-date 2024-01-01 --end-date 2024-01-31 --use-docker-db

# Intraday
docker compose run --rm airflow-scheduler \
  python -m app.backfill_intraday --mode full --tickers AAPL --start-date 2024-01-01 --end-date 2024-01-31 --use-docker-db

# News
docker compose run --rm airflow-scheduler \
  python -m app.backfill_news --mode full --tickers AAPL --start-date 2024-01-01 --end-date 2024-01-31 --use-docker-db
```

Run locally (connects to host DB URL):

```bash
python -m app.backfill_daily --mode resume
python -m app.backfill_intraday --mode resume --tickers AAPL MSFT
python -m app.backfill_news --mode full --start-date 2024-01-01
```