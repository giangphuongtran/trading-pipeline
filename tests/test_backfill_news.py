from types import SimpleNamespace

from app import backfill_news


class DummyConn:
    pass


def test_backfill_news_success(monkeypatch):
    articles = [{"id": "news-1", "ticker": "AAPL"}]

    def fake_get_news(*, ticker, published_utc_gte, published_utc_lte, **kwargs):
        assert ticker == "AAPL"
        assert published_utc_gte == "2024-01-01"
        assert published_utc_lte == "2024-01-02"
        return articles

    client = SimpleNamespace(get_news=fake_get_news)

    def fake_insert(conn, payload):
        assert payload == articles
        return len(payload)

    metadata_calls = []

    def fake_update(conn, data_type, ticker, start, end, rows, *, status="completed", error_message=None):
        metadata_calls.append((data_type, ticker, start, end, rows, status, error_message))

    monkeypatch.setattr(backfill_news, "insert_news_articles", fake_insert)
    monkeypatch.setattr(backfill_news, "update_metadata", fake_update)

    result = backfill_news.backfill_news_articles(
        client,
        DummyConn(),
        "AAPL",
        "2024-01-01",
        "2024-01-02",
    )

    assert result == 1
    assert metadata_calls == [
        ("news", "AAPL", "2024-01-01", "2024-01-02", 1, "completed", None)
    ]


def test_backfill_news_no_results(monkeypatch):
    client = SimpleNamespace(get_news=lambda **kwargs: [])

    def fail_insert(conn, payload):  # pragma: no cover
        raise AssertionError("insert_news_articles should not run when no articles")

    metadata_calls = []

    def fake_update(conn, data_type, ticker, start, end, rows, *, status="completed", error_message=None):
        metadata_calls.append((data_type, ticker, start, end, rows, status, error_message))

    monkeypatch.setattr(backfill_news, "insert_news_articles", fail_insert)
    monkeypatch.setattr(backfill_news, "update_metadata", fake_update)

    result = backfill_news.backfill_news_articles(
        client,
        DummyConn(),
        "AAPL",
        "2024-01-01",
        "2024-01-02",
    )

    assert result == 0
    assert metadata_calls == [
        ("news", "AAPL", "2024-01-01", "2024-01-02", 0, "completed", None)
    ]


def test_backfill_news_error(monkeypatch):
    class ErrorClient:
        def get_news(self, **kwargs):
            raise RuntimeError("boom")

    metadata_calls = []

    def fake_update(conn, data_type, ticker, start, end, rows, *, status="completed", error_message=None):
        metadata_calls.append((status, error_message))

    monkeypatch.setattr(backfill_news, "update_metadata", fake_update)

    result = backfill_news.backfill_news_articles(
        ErrorClient(),
        DummyConn(),
        "AAPL",
        "2024-01-01",
        "2024-01-02",
    )

    assert result == 0
    assert metadata_calls == [("failed", "boom")]

