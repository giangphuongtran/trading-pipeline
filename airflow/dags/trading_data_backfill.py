from airflow import DAG

from backfill_common import create_backfill_operator, dag_defaults

with DAG(**dag_defaults("trading_data_backfill")) as dag:
    backfill_daily = create_backfill_operator(
        task_id="backfill_daily_bars",
        module="app.backfill.backfill_daily",
    )

    backfill_intraday = create_backfill_operator(
        task_id="backfill_intraday_bars",
        module="app.backfill.backfill_intraday",
    )

    backfill_news = create_backfill_operator(
        task_id="backfill_news_articles",
        module="app.backfill.backfill_news",
    )

    backfill_daily >> backfill_intraday >> backfill_news