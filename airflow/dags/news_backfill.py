"""Standalone DAG for news backfill."""

from airflow import DAG

from backfill_common import create_backfill_operator, dag_defaults

with DAG(**dag_defaults("news_backfill")) as dag:
    backfill_news = create_backfill_operator(
        task_id="backfill_news_articles",
        module="app.backfill.backfill_news",
    )
