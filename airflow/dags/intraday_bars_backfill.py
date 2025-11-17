"""Standalone DAG for intraday bars backfill."""

from airflow import DAG

from backfill_common import create_backfill_operator, dag_defaults

with DAG(**dag_defaults("intraday_bars_backfill")) as dag:
    backfill_intraday = create_backfill_operator(
        task_id="backfill_intraday_bars",
        module="app.backfill.backfill_intraday",
    )

