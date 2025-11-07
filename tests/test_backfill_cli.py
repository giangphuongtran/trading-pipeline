from datetime import date

import pytest

from app.backfill import cli


class FixedDate(date):
    """datetime.date subclass with controllable today()."""

    @classmethod
    def today(cls) -> "FixedDate":  # type: ignore[override]
        return cls(2025, 11, 7)


def set_fixed_today(monkeypatch):
    """Ensure cli.date.today() returns a deterministic value."""

    monkeypatch.setattr(cli, "date", FixedDate)


def test_build_arg_parser_defaults():
    parser = cli.build_arg_parser("daily")
    args = parser.parse_args([])
    assert args.mode == "resume"
    assert args.lookback_days == 30
    assert args.tickers is None


def test_build_arg_parser_accepts_custom_values():
    parser = cli.build_arg_parser("daily")
    args = parser.parse_args([
        "--mode",
        "full",
        "--start-date",
        "2024-01-01",
        "--end-date",
        "2024-01-03",
        "--lookback-days",
        "1",
        "--tickers",
        "AAPL",
        "--use-docker-db",
    ])

    assert args.mode == "full"
    assert args.start_date == "2024-01-01"
    assert args.end_date == "2024-01-03"
    assert args.lookback_days == 1
    assert args.tickers == ["AAPL"]
    assert args.use_docker_db is True


def test_resolve_date_range_full_mode_requires_start(monkeypatch):
    set_fixed_today(monkeypatch)

    with pytest.raises(ValueError):
        cli.resolve_date_range(
            conn=None,
            ticker="AAPL",
            data_type="daily",
            mode="full",
            explicit_start=None,
            explicit_end=None,
            lookback_days=1,
        )


def test_resolve_date_range_full_mode_defaults_end_to_yesterday(monkeypatch):
    set_fixed_today(monkeypatch)

    start, end = cli.resolve_date_range(
        conn=None,
        ticker="AAPL",
        data_type="daily",
        mode="full",
        explicit_start="2025-11-04",
        explicit_end=None,
        lookback_days=1,
    )

    assert start == "2025-11-04"
    assert end == "2025-11-06"  # yesterday relative to FixedDate.today()


def test_resolve_date_range_resume_with_metadata(monkeypatch):
    set_fixed_today(monkeypatch)

    def fake_last_fetch(conn, ticker, data_type):  # pragma: no cover - simple stub
        return FixedDate(2025, 11, 6)

    monkeypatch.setattr(cli, "get_last_fetch_date", fake_last_fetch)

    start, end = cli.resolve_date_range(
        conn=None,
        ticker="AAPL",
        data_type="daily",
        mode="resume",
        explicit_start=None,
        explicit_end=None,
        lookback_days=1,
    )

    assert start == "2025-11-06"
    assert end == "2025-11-06"


def test_resolve_date_range_resume_without_metadata(monkeypatch):
    set_fixed_today(monkeypatch)
    monkeypatch.setattr(cli, "get_last_fetch_date", lambda *args, **kwargs: None)

    start, end = cli.resolve_date_range(
        conn=None,
        ticker="AAPL",
        data_type="daily",
        mode="resume",
        explicit_start=None,
        explicit_end=None,
        lookback_days=5,
    )

    assert start == "2025-11-01"
    assert end == "2025-11-06"


def test_compute_backfill_plan_generates_config_per_ticker(monkeypatch):
    set_fixed_today(monkeypatch)

    def fake_resolve(conn, ticker, data_type, mode, explicit_start, explicit_end, lookback_days):
        return ("2025-11-04", "2025-11-06")

    monkeypatch.setattr(cli, "resolve_date_range", fake_resolve)

    plans = list(
        cli.compute_backfill_plan(
            conn=None,
            tickers=["AAPL", "MSFT"],
            data_type="intraday",
            mode="resume",
            start_date=None,
            end_date=None,
            lookback_days=1,
        )
    )

    assert len(plans) == 2
    assert plans[0] == cli.BackfillConfig("AAPL", "2025-11-04", "2025-11-06")
    assert plans[1] == cli.BackfillConfig("MSFT", "2025-11-04", "2025-11-06")

