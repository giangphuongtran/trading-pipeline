"""Shared helpers for trading backfill DAGs (simple, Docker-first)."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from typing import Iterable, Optional

from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}


def _normalize_tickers(value: Optional[Iterable[str] | str]) -> list[str]:
    """
    Normalize ticker input to a list of strings.
    
    Args:
        value: Can be None, a single string, or an iterable of strings
        
    Returns:
        List of ticker strings (empty list if None)
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _docker_env(env: dict) -> dict:
    """
    Prepare environment for subprocess execution in Docker.
    
    Uses existing environment variables from .env file (DATABASE_URL_DOCKER, DATABASE_URL_HOST).
    Only falls back to constructing URL from parts if no URL env vars are set.
    
    Args:
        env: Current environment dictionary
        
    Returns:
        Environment dict with DATABASE_URL set and Docker-specific flags
    """
    e = env.copy()

    # Prefer DATABASE_URL_DOCKER (for Docker network) or DATABASE_URL (if already set)
    url = e.get("DATABASE_URL") or e.get("DATABASE_URL_DOCKER")
    
    # Only construct URL from parts if no URL env vars exist
    if not url:
        host = env.get("POSTGRES_HOST", "postgres")
        port = env.get("POSTGRES_PORT", "5432")
        db = env.get("POSTGRES_DB", "trading_data")
        usr = env.get("POSTGRES_USER", "postgres")
        pwd = env.get("POSTGRES_PASSWORD", "")
        url = f"postgresql://{usr}:{pwd}@{host}:{port}/{db}"

    e["DATABASE_URL"] = url
    e.setdefault("USE_DOCKER_DB", "1")
    e.setdefault("PYTHONPATH", "/opt/airflow")
    return e



def run_backfill(module: str, extra_args: Optional[list[str]] = None, **context) -> None:
    """
    Execute a backfill module as a subprocess with proper environment setup.
    
    Reads DAG run configuration for mode, dates, and tickers. Always uses Docker DB connection.
    Streams output to Airflow logs in real-time.
    
    Args:
        module: Python module to run (e.g., "app.backfill.backfill_daily")
        extra_args: Additional command-line arguments to pass
        **context: Airflow task context (contains dag_run with configuration)
    """
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run else {}

    cmd = ["python", "-u", "-m", module]

    # DAG run params (all optional)
    mode = conf.get("mode", "resume")
    start_date = conf.get("start_date")
    end_date = conf.get("end_date")
    tickers = _normalize_tickers(conf.get("tickers"))

    cmd.extend(["--mode", mode])
    if start_date:
        cmd.extend(["--start-date", start_date])
    if end_date:
        cmd.extend(["--end-date", end_date])
    if tickers:
        cmd.append("--tickers")
        cmd.extend(tickers)

    # Always prefer docker DB for tasks; allow caller to add more flags
    cmd.append("--use-docker-db")
    if extra_args:
        cmd.extend(extra_args)

    env = _docker_env(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    # Log command + masked DB URL
    safe_db = env.get("DATABASE_URL", "")
    if safe_db and "@" in safe_db and ":" in safe_db.split("@")[0]:
        try:
            userinfo, rest = safe_db.split("@", 1)
            scheme, creds = userinfo.split("://", 1)
            if ":" in creds:
                username, _ = creds.split(":", 1)
                safe_db = f"{scheme}://{username}:*****@{rest}"
        except Exception:
            pass
    print(f"[run_backfill] exec: {' '.join(cmd)}")
    if safe_db:
        print(f"[run_backfill] DATABASE_URL: {safe_db}")

    # Stream outputs into Airflow logs; raise on failure
    proc: Optional[subprocess.Popen[str]] = None
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                sys.stdout.write(line)
                sys.stdout.flush()

        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"{module} exited with code {return_code}")

    except KeyboardInterrupt:
        sys.stderr.write(f"[run_backfill] Received interrupt, terminating {module}\n")
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
        raise


def create_backfill_operator(task_id: str, module: str) -> PythonOperator:
    """
    Create a PythonOperator for running a backfill module.
    
    Args:
        task_id: Airflow task identifier
        module: Python module to run (e.g., "app.backfill.backfill_daily")
        
    Returns:
        Configured PythonOperator instance
    """
    return PythonOperator(
        task_id=task_id,
        python_callable=run_backfill,
        op_kwargs={"module": module, "extra_args": []},
    )


def dag_defaults(dag_id: str) -> dict:
    """
    Get standard DAG configuration dictionary.
    
    Args:
        dag_id: Unique DAG identifier
        
    Returns:
        Dictionary with DAG configuration (schedule, start_date, etc.)
    """
    return {
        "dag_id": dag_id,
        "description": "Trading data backfill pipeline",
        "default_args": DEFAULT_ARGS,
        "schedule_interval": "@daily",
        "start_date": datetime(2023, 11, 1),
        "catchup": False,
        "max_active_runs": 1,
        "tags": ["trading", "backfill"],
    }
