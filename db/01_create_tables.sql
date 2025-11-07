-- Daily bar table
CREATE TABLE IF NOT EXISTS daily_bars (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(12, 4) NOT NULL,
    high DECIMAL(12, 4) NOT NULL,
    low DECIMAL(12, 4) NOT NULL,
    close DECIMAL(12, 4) NOT NULL,
    volume BIGINT,
    transactions BIGINT,
    volume_weighted_avg_price DECIMAL(12, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_daily_bars UNIQUE (ticker, date)
);

-- Daily bar indexes
CREATE INDEX idx_daily_bars_ticker ON daily_bars(ticker);
CREATE INDEX idx_daily_bars_date ON daily_bars(date);
CREATE INDEX idx_daily_bars_ticker_date ON daily_bars(ticker, date);

-- Intraday bar table (5-minute bars)
CREATE TABLE IF NOT EXISTS intraday_bars (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(12,4) NOT NULL,
    high DECIMAL(12,4) NOT NULL,
    low DECIMAL(12,4) NOT NULL,
    close DECIMAL(12,4) NOT NULL,
    volume BIGINT,
    transactions BIGINT,
    volume_weighted_avg_price DECIMAL(12,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_intraday_bars UNIQUE (ticker, timestamp)
);

CREATE INDEX idx_intraday_bars_ticker ON intraday_bars(ticker);
CREATE INDEX idx_intraday_bars_timestamp ON intraday_bars(timestamp);
CREATE INDEX idx_intraday_bars_ticker_timestamp ON intraday_bars(ticker, timestamp);

-- News article table
CREATE TABLE IF NOT EXISTS news_articles (
    id VARCHAR(100) PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    url TEXT,
    author VARCHAR(200),
    type VARCHAR(50),
    sentiment_score DECIMAL(3,2),  -- -1.0 to 1.0
    sentiment_label VARCHAR(20),   -- positive, negative, neutral
    sentiment_reasoning TEXT,      -- Polygon's reasoning for the sentiment
    keywords TEXT[],               -- Array of keywords
    tickers TEXT[],                -- Array of related tickers
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_news_articles_ticker ON news_articles(ticker);
CREATE INDEX idx_news_articles_published_at ON news_articles(published_at);
CREATE INDEX idx_news_articles_ticker_date ON news_articles(ticker, published_at);
CREATE INDEX idx_news_articles_sentiment ON news_articles(sentiment_label);
CREATE INDEX idx_news_articles_sentiment_score ON news_articles(sentiment_score);

COMMENT ON COLUMN news_articles.sentiment_score IS 'Sentiment score: -1.0 (negative) to 1.0 (positive)';
COMMENT ON COLUMN news_articles.sentiment_label IS 'Sentiment label: positive, negative, neutral';
COMMENT ON COLUMN news_articles.sentiment_reasoning IS 'Polygon AI reasoning for the sentiment';

-- API Metadata configuration
-- Enums for strict data quality
DO $$
BEGIN
    CREATE TYPE data_type_enum AS ENUM ('daily', 'intraday', 'news');
EXCEPTION
    WHEN duplicate_object THEN
        NULL; -- Type already exists
END $$;

DO $$
BEGIN
    CREATE TYPE status_enum AS ENUM ('pending', 'in_progress', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN
        NULL; -- Type already exists
END $$;

-- API Metadata table
CREATE TABLE IF NOT EXISTS api_metadata (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    data_type data_type_enum NOT NULL,
    date_range_start DATE,
    date_range_end DATE,
    last_fetch_date DATE,
    last_success_date DATE,
    status status_enum DEFAULT 'pending', 
    error_message TEXT,
    rows_inserted INTEGER DEFAULT 0,  -- Number of rows/data inserted in this run
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_api_metadata_ticker ON api_metadata(ticker);
CREATE INDEX idx_api_metadata_data_type ON api_metadata(data_type);
CREATE INDEX idx_api_metadata_status ON api_metadata(status);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_daily_bars_updated_at') THEN
        CREATE TRIGGER update_daily_bars_updated_at
        BEFORE UPDATE ON daily_bars 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_intraday_bars_updated_at') THEN
        CREATE TRIGGER update_intraday_bars_updated_at
        BEFORE UPDATE ON intraday_bars
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_news_articles_updated_at') THEN
        CREATE TRIGGER update_news_articles_updated_at
        BEFORE UPDATE ON news_articles
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_api_metadata_updated_at') THEN
        CREATE TRIGGER update_api_metadata_updated_at
        BEFORE UPDATE ON api_metadata
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;