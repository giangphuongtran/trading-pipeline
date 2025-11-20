# Manual Backfill Commands

These commands run the same modules the Airflow DAGs use, but let you trigger a single ticker/date range by hand. Run them from the project root (`/Users/mac/learning/trading-pipeline`).

## Intraday Bars (30-day slice)
```
docker compose run --rm airflow-scheduler \
  python -m app.backfill.backfill_intraday \
    --mode full \
    --tickers AAPL \
    --start-date 2023-11-08 \
    --end-date 2023-12-07 \
    --use-docker-db
```

## Daily Bars (30-day slice)
```
docker compose run --rm airflow-scheduler \
  python -m app.backfill.backfill_daily \
    --mode full \
    --tickers AAPL \
    --start-date 2023-11-08 \
    --end-date 2023-12-07 \
    --use-docker-db
```

**Note:** Market indices (e.g., SPY) are automatically included in the default ticker list. To backfill only market indices, use:
```
docker compose run --rm airflow-scheduler \
  python -m app.backfill.backfill_daily \
    --mode full \
    --tickers SPY \
    --start-date 2023-11-08 \
    --end-date 2023-12-07 \
    --use-docker-db
```

## News (automatically chunked into 7-day spans)
```
docker compose run --rm airflow-scheduler \
  python -m app.backfill.backfill_news \
    --mode full \
    --tickers AAPL \
    --start-date 2023-11-08 \
    --end-date 2023-12-07 \
    --use-docker-db
```

Tip: swap `AAPL` or the dates as needed, and drop `--use-docker-db` if you want to target your host database instead.
