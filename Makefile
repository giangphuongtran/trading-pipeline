.PHONY: help venv install up down logs ps test fmt

PYTHON ?= python3

help:
	@echo "Common commands:"
	@echo "  make venv      - create .venv"
	@echo "  make install   - install python deps into .venv"
	@echo "  make up        - start docker compose services"
	@echo "  make down      - stop docker compose services"
	@echo "  make logs      - tail airflow logs"
	@echo "  make test      - run unit tests"
	@echo "  make backfill-daily     - run daily backfill (local DB)"
	@echo "  make backfill-intraday  - run intraday backfill (local DB)"
	@echo "  make backfill-news      - run news backfill (local DB)"

venv:
	@test -d .venv || $(PYTHON) -m venv .venv

install: venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f airflow-webserver airflow-scheduler

test:
	pytest -q

backfill-daily:
	$(PYTHON) -m app.backfill_daily --mode resume

backfill-intraday:
	$(PYTHON) -m app.backfill_intraday --mode resume

backfill-news:
	$(PYTHON) -m app.backfill_news --mode resume


