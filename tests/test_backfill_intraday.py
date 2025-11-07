from types import SimpleNamespace

from app import backfill_intraday


class DummyConn:
    pass


def test_backfill_intraday_success(monkeypatch):
    bars = [{"ticker": "AAPL", "timestamp": "2024-01-01T09:30:00"}]
    client = SimpleNamespace(get_intraday_bars=lambda *args, **kwargs: bars)

    def fake_insert(conn, payload):
        assert payload == bars
        return len(payload)

    metadata_calls = []

    def fake_update(conn, data_type, ticker, start, end, rows, *, status="completed", error_message=None):
        metadata_calls.append((data_type, ticker, start, end, rows, status, error_message))

    monkeypatch.setattr(backfill_intraday, "insert_intraday_bars", fake_insert)
    monkeypatch.setattr(backfill_intraday, "update_metadata", fake_update)

    result = backfill_intraday.backfill_intraday_bars(
        client,
        DummyConn(),
        "AAPL",
        "2024-01-01",
        "2024-01-02",
    )

    assert result == 1
    assert metadata_calls == [
        ("intraday", "AAPL", "2024-01-01", "2024-01-02", 1, "completed", None)
    ]


def test_backfill_intraday_no_results(monkeypatch):
    client = SimpleNamespace(get_intraday_bars=lambda *args, **kwargs: [])

    def fail_insert(conn, payload):  # pragma: no cover
        raise AssertionError("insert_intraday_bars should not be called")

    metadata_calls = []

    def fake_update(conn, data_type, ticker, start, end, rows, *, status="completed", error_message=None):
        metadata_calls.append((data_type, ticker, start, end, rows, status, error_message))

    monkeypatch.setattr(backfill_intraday, "insert_intraday_bars", fail_insert)
    monkeypatch.setattr(backfill_intraday, "update_metadata", fake_update)

    result = backfill_intraday.backfill_intraday_bars(
        client,
        DummyConn(),
        "AAPL",
        "2024-01-01",
        "2024-01-02",
    )

    assert result == 0
    assert metadata_calls == [
        ("intraday", "AAPL", "2024-01-01", "2024-01-02", 0, "completed", None)
    ]


def test_backfill_intraday_error(monkeypatch):
    class ErrorClient:
        def get_intraday_bars(self, *args, **kwargs):
            raise RuntimeError("boom")

    metadata_calls = []

    def fake_update(conn, data_type, ticker, start, end, rows, *, status="completed", error_message=None):
        metadata_calls.append((status, error_message))

    monkeypatch.setattr(backfill_intraday, "update_metadata", fake_update)

    result = backfill_intraday.backfill_intraday_bars(
        ErrorClient(),
        DummyConn(),
        "AAPL",
        "2024-01-01",
        "2024-01-02",
    )

    assert result == 0
    assert metadata_calls == [("failed", "boom")]

