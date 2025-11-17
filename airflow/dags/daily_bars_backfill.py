"""Standalone DAG for daily bars backfill."""

from airflow import DAG

from backfill_common import create_backfill_operator, dag_defaults

with DAG(**dag_defaults("daily_bars_backfill")) as dag:
    backfill_daily = create_backfill_operator(
        task_id="backfill_daily_bars",
        module="app.backfill.backfill_daily",
    )

