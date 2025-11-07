from types import SimpleNamespace

import pytest

from app import backfill_daily


class DummyConn:
    """Small sentinel object used to assert we pass the same connection through."""


def test_backfill_daily_success(monkeypatch):
    bars = [{"ticker": "AAPL", "date": "2024-01-01"}]
    client = SimpleNamespace(get_daily_bars=lambda *args, **kwargs: bars)

    inserted = {}

    def fake_insert(conn, payload):
        inserted["value"] = payload
        return len(payload)

    metadata_calls = []

    def fake_update(conn, data_type, ticker, start, end, rows, *, status="completed", error_message=None):
        metadata_calls.append(
            {
                "data_type": data_type,
                "ticker": ticker,
                "start": start,
                "end": end,
                "rows": rows,
                "status": status,
                "error": error_message,
            }
        )

    monkeypatch.setattr(backfill_daily, "insert_daily_bars", fake_insert)
    monkeypatch.setattr(backfill_daily, "update_metadata", fake_update)

    result = backfill_daily.backfill_daily_bars(client, DummyConn(), "AAPL", "2024-01-01", "2024-01-02")

    assert result == 1
    assert inserted["value"] == bars
    assert metadata_calls == [
        {
            "data_type": "daily",
            "ticker": "AAPL",
            "start": "2024-01-01",
            "end": "2024-01-02",
            "rows": 1,
            "status": "completed",
            "error": None,
        }
    ]


def test_backfill_daily_no_results(monkeypatch):
    client = SimpleNamespace(get_daily_bars=lambda *args, **kwargs: [])

    def fail_insert(conn, payload):  # pragma: no cover - should not be called
        raise AssertionError("insert_daily_bars should not run when no bars are returned")

    metadata_calls = []

    def fake_update(conn, data_type, ticker, start, end, rows, *, status="completed", error_message=None):
        metadata_calls.append((data_type, ticker, start, end, rows, status, error_message))

    monkeypatch.setattr(backfill_daily, "insert_daily_bars", fail_insert)
    monkeypatch.setattr(backfill_daily, "update_metadata", fake_update)

    result = backfill_daily.backfill_daily_bars(client, DummyConn(), "AAPL", "2024-01-01", "2024-01-02")

    assert result == 0
    assert metadata_calls == [("daily", "AAPL", "2024-01-01", "2024-01-02", 0, "completed", None)]


def test_backfill_daily_error(monkeypatch):
    class ErrorClient:
        def get_daily_bars(self, *args, **kwargs):
            raise RuntimeError("boom")

    metadata_calls = []

    def fake_update(conn, data_type, ticker, start, end, rows, *, status="completed", error_message=None):
        metadata_calls.append((status, error_message))

    monkeypatch.setattr(backfill_daily, "update_metadata", fake_update)

    result = backfill_daily.backfill_daily_bars(ErrorClient(), DummyConn(), "AAPL", "2024-01-01", "2024-01-02")

    assert result == 0
    assert metadata_calls == [("failed", "boom")]

