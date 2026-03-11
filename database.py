"""
SentimentAlpha — Database Layer
================================
SQLite-backed storage for news articles, sentiment scores, price data,
factor scores, and trading signals.
"""

import sqlite3
import json
from datetime import datetime, date
from typing import Optional
from contextlib import contextmanager

from config import DATABASE_PATH


# ── Schema ─────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Raw news articles
CREATE TABLE IF NOT EXISTS news_articles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,                -- moneycontrol / economic_times / livemint / reddit / twitter
    url             TEXT UNIQUE,
    title           TEXT NOT NULL,
    summary         TEXT,
    full_text       TEXT,
    published_at    TIMESTAMP,
    scraped_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    category        TEXT                           -- market / stock / sector / macro
);

-- LLM sentiment analysis results
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id      INTEGER NOT NULL REFERENCES news_articles(id),
    provider        TEXT NOT NULL,                  -- anthropic / openai / finbert
    sentiment       TEXT NOT NULL,                  -- bullish / bearish / neutral
    confidence      REAL NOT NULL,                  -- 0.0 - 1.0
    signal_type     TEXT,                           -- earnings_beat, management_change, etc.
    affected_stocks TEXT,                           -- JSON list of ticker symbols
    affected_sectors TEXT,                          -- JSON list of sectors
    raw_response    TEXT,                           -- full LLM response for debugging
    analyzed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(article_id, provider)
);

-- Research briefs (LLM-generated summaries per stock)
CREATE TABLE IF NOT EXISTS research_briefs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    brief           TEXT NOT NULL,
    article_ids     TEXT NOT NULL,                  -- JSON list of article IDs used
    generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Historical stock prices
CREATE TABLE IF NOT EXISTS stock_prices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    trade_date      DATE NOT NULL,
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL,
    adj_close       REAL,
    volume          INTEGER,
    UNIQUE(ticker, trade_date)
);

-- Daily sentiment factor per stock
CREATE TABLE IF NOT EXISTS daily_sentiment (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    factor_date     DATE NOT NULL,
    raw_score       REAL NOT NULL,                 -- single-day sentiment (-1 to +1)
    rolling_7d      REAL,                          -- 7-day rolling average
    rolling_14d     REAL,                          -- 14-day rolling average
    article_count   INTEGER DEFAULT 0,
    avg_confidence  REAL,
    UNIQUE(ticker, factor_date)
);

-- Traditional factor scores
CREATE TABLE IF NOT EXISTS factor_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    factor_date     DATE NOT NULL,
    momentum_1m     REAL,
    momentum_3m     REAL,
    momentum_6m     REAL,
    volatility_20d  REAL,
    value_pe        REAL,
    value_pb        REAL,
    UNIQUE(ticker, factor_date)
);

-- Backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name   TEXT NOT NULL,
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    total_return    REAL,
    annualized_return REAL,
    sharpe_ratio    REAL,
    max_drawdown    REAL,
    win_rate        REAL,
    num_trades      INTEGER,
    params          TEXT,                           -- JSON dict of parameters
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ML model predictions
CREATE TABLE IF NOT EXISTS ml_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    prediction_date DATE NOT NULL,
    predicted_return REAL,
    actual_return   REAL,
    features_json   TEXT,                          -- JSON of feature values used
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, ticker, prediction_date)
);

-- Alerts (sudden sentiment shifts)
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    alert_type      TEXT NOT NULL,                 -- sentiment_spike / sentiment_drop / volume_anomaly
    message         TEXT NOT NULL,
    severity        TEXT DEFAULT 'medium',         -- low / medium / high
    is_read         INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_news_source ON news_articles(source);
CREATE INDEX IF NOT EXISTS idx_sentiment_article ON sentiment_scores(article_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_stocks ON sentiment_scores(affected_stocks);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON stock_prices(ticker, trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_sent_ticker ON daily_sentiment(ticker, factor_date);
CREATE INDEX IF NOT EXISTS idx_factor_ticker ON factor_scores(ticker, factor_date);
CREATE INDEX IF NOT EXISTS idx_alerts_ticker ON alerts(ticker, created_at);
"""


# ── Database Manager ───────────────────────────────────────────────────────

class DatabaseManager:
    """Thread-safe SQLite database manager for SentimentAlpha."""

    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or DATABASE_PATH)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── News Articles ──────────────────────────────────────────────────

    def insert_article(self, source: str, url: str, title: str,
                       summary: str = None, full_text: str = None,
                       published_at: datetime = None, category: str = None) -> Optional[int]:
        """Insert a news article. Returns article ID or None if duplicate."""
        with self.connect() as conn:
            try:
                cursor = conn.execute(
                    """INSERT INTO news_articles (source, url, title, summary, full_text, published_at, category)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (source, url, title, summary, full_text, published_at, category)
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None  # duplicate URL

    def get_unanalyzed_articles(self, provider: str, limit: int = 50) -> list:
        """Get articles that haven't been analyzed by a specific provider."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT n.* FROM news_articles n
                   LEFT JOIN sentiment_scores s ON n.id = s.article_id AND s.provider = ?
                   WHERE s.id IS NULL
                   ORDER BY n.published_at DESC
                   LIMIT ?""",
                (provider, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_articles_for_stock(self, ticker: str, days: int = 30) -> list:
        """Get recent articles mentioning a stock."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT n.*, s.sentiment, s.confidence, s.signal_type
                   FROM news_articles n
                   JOIN sentiment_scores s ON n.id = s.article_id
                   WHERE s.affected_stocks LIKE ?
                     AND n.published_at >= datetime('now', ?)
                   ORDER BY n.published_at DESC""",
                (f'%{ticker}%', f'-{days} days')
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_articles(self, limit: int = 100) -> list:
        """Get the most recent articles with sentiment."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT n.*, s.sentiment, s.confidence, s.signal_type, s.affected_stocks
                   FROM news_articles n
                   LEFT JOIN sentiment_scores s ON n.id = s.article_id
                   ORDER BY n.published_at DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Sentiment Scores ───────────────────────────────────────────────

    def insert_sentiment(self, article_id: int, provider: str, sentiment: str,
                         confidence: float, signal_type: str = None,
                         affected_stocks: list = None, affected_sectors: list = None,
                         raw_response: str = None) -> Optional[int]:
        """Insert sentiment analysis result."""
        with self.connect() as conn:
            try:
                cursor = conn.execute(
                    """INSERT INTO sentiment_scores
                       (article_id, provider, sentiment, confidence, signal_type,
                        affected_stocks, affected_sectors, raw_response)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (article_id, provider, sentiment, confidence, signal_type,
                     json.dumps(affected_stocks or []),
                     json.dumps(affected_sectors or []),
                     raw_response)
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None

    # ── Research Briefs ────────────────────────────────────────────────

    def insert_brief(self, ticker: str, brief: str, article_ids: list) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """INSERT INTO research_briefs (ticker, brief, article_ids) VALUES (?, ?, ?)""",
                (ticker, brief, json.dumps(article_ids))
            )
            return cursor.lastrowid

    def get_latest_brief(self, ticker: str) -> Optional[dict]:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT * FROM research_briefs WHERE ticker = ? ORDER BY generated_at DESC LIMIT 1""",
                (ticker,)
            ).fetchone()
            return dict(row) if row else None

    # ── Stock Prices ───────────────────────────────────────────────────

    def insert_prices(self, records: list[dict]):
        """Bulk insert price records. Each dict: ticker, trade_date, open, high, low, close, adj_close, volume."""
        with self.connect() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO stock_prices
                   (ticker, trade_date, open, high, low, close, adj_close, volume)
                   VALUES (:ticker, :trade_date, :open, :high, :low, :close, :adj_close, :volume)""",
                records
            )

    def get_prices(self, ticker: str, start_date: str = None, end_date: str = None) -> list:
        with self.connect() as conn:
            query = "SELECT * FROM stock_prices WHERE ticker = ?"
            params = [ticker]
            if start_date:
                query += " AND trade_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND trade_date <= ?"
                params.append(end_date)
            query += " ORDER BY trade_date"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_latest_price_date(self, ticker: str) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT MAX(trade_date) as max_date FROM stock_prices WHERE ticker = ?",
                (ticker,)
            ).fetchone()
            return row["max_date"] if row and row["max_date"] else None

    # ── Daily Sentiment ────────────────────────────────────────────────

    def upsert_daily_sentiment(self, ticker: str, factor_date: str,
                               raw_score: float, rolling_7d: float = None,
                               rolling_14d: float = None, article_count: int = 0,
                               avg_confidence: float = None):
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO daily_sentiment
                   (ticker, factor_date, raw_score, rolling_7d, rolling_14d, article_count, avg_confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, factor_date) DO UPDATE SET
                     raw_score=excluded.raw_score, rolling_7d=excluded.rolling_7d,
                     rolling_14d=excluded.rolling_14d, article_count=excluded.article_count,
                     avg_confidence=excluded.avg_confidence""",
                (ticker, factor_date, raw_score, rolling_7d, rolling_14d, article_count, avg_confidence)
            )

    def get_daily_sentiment(self, ticker: str, days: int = 90) -> list:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM daily_sentiment WHERE ticker = ?
                   AND factor_date >= date('now', ?)
                   ORDER BY factor_date""",
                (ticker, f'-{days} days')
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_daily_sentiment(self, start_date: str = None, end_date: str = None) -> list:
        with self.connect() as conn:
            query = "SELECT * FROM daily_sentiment WHERE 1=1"
            params = []
            if start_date:
                query += " AND factor_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND factor_date <= ?"
                params.append(end_date)
            query += " ORDER BY factor_date, ticker"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ── Factor Scores ──────────────────────────────────────────────────

    def upsert_factor_scores(self, ticker: str, factor_date: str, **kwargs):
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO factor_scores
                   (ticker, factor_date, momentum_1m, momentum_3m, momentum_6m,
                    volatility_20d, value_pe, value_pb)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, factor_date) DO UPDATE SET
                     momentum_1m=excluded.momentum_1m, momentum_3m=excluded.momentum_3m,
                     momentum_6m=excluded.momentum_6m, volatility_20d=excluded.volatility_20d,
                     value_pe=excluded.value_pe, value_pb=excluded.value_pb""",
                (ticker, factor_date,
                 kwargs.get("momentum_1m"), kwargs.get("momentum_3m"), kwargs.get("momentum_6m"),
                 kwargs.get("volatility_20d"), kwargs.get("value_pe"), kwargs.get("value_pb"))
            )

    def get_factor_scores(self, ticker: str = None, start_date: str = None, end_date: str = None) -> list:
        with self.connect() as conn:
            query = "SELECT * FROM factor_scores WHERE 1=1"
            params = []
            if ticker:
                query += " AND ticker = ?"
                params.append(ticker)
            if start_date:
                query += " AND factor_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND factor_date <= ?"
                params.append(end_date)
            query += " ORDER BY factor_date, ticker"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ── Backtest Results ───────────────────────────────────────────────

    def insert_backtest(self, strategy_name: str, start_date: str, end_date: str,
                        total_return: float, annualized_return: float = None,
                        sharpe_ratio: float = None, max_drawdown: float = None,
                        win_rate: float = None, num_trades: int = None,
                        params: dict = None) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """INSERT INTO backtest_results
                   (strategy_name, start_date, end_date, total_return, annualized_return,
                    sharpe_ratio, max_drawdown, win_rate, num_trades, params)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (strategy_name, start_date, end_date, total_return, annualized_return,
                 sharpe_ratio, max_drawdown, win_rate, num_trades, json.dumps(params or {}))
            )
            return cursor.lastrowid

    def get_all_backtests(self) -> list:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM backtest_results ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    # ── ML Predictions ─────────────────────────────────────────────────

    def insert_prediction(self, model_name: str, ticker: str, prediction_date: str,
                          predicted_return: float, features: dict = None):
        with self.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ml_predictions
                   (model_name, ticker, prediction_date, predicted_return, features_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (model_name, ticker, prediction_date, predicted_return, json.dumps(features or {}))
            )

    def update_actual_return(self, model_name: str, ticker: str, prediction_date: str,
                             actual_return: float):
        with self.connect() as conn:
            conn.execute(
                """UPDATE ml_predictions SET actual_return = ?
                   WHERE model_name = ? AND ticker = ? AND prediction_date = ?""",
                (actual_return, model_name, ticker, prediction_date)
            )

    def get_predictions(self, model_name: str = None) -> list:
        with self.connect() as conn:
            if model_name:
                rows = conn.execute(
                    "SELECT * FROM ml_predictions WHERE model_name = ? ORDER BY prediction_date",
                    (model_name,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ml_predictions ORDER BY prediction_date"
                ).fetchall()
            return [dict(r) for r in rows]

    # ── Alerts ─────────────────────────────────────────────────────────

    def insert_alert(self, ticker: str, alert_type: str, message: str,
                     severity: str = "medium") -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """INSERT INTO alerts (ticker, alert_type, message, severity)
                   VALUES (?, ?, ?, ?)""",
                (ticker, alert_type, message, severity)
            )
            return cursor.lastrowid

    def get_unread_alerts(self, limit: int = 50) -> list:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM alerts WHERE is_read = 0
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_alert_read(self, alert_id: int):
        with self.connect() as conn:
            conn.execute("UPDATE alerts SET is_read = 1 WHERE id = ?", (alert_id,))

    # ── Utility ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get summary statistics for the dashboard."""
        with self.connect() as conn:
            stats = {}
            for table in ["news_articles", "sentiment_scores", "stock_prices",
                          "daily_sentiment", "alerts"]:
                row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
                stats[table] = row["cnt"]
            return stats


# Module-level singleton
db = DatabaseManager()
