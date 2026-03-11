"""
SentimentAlpha — Price Data Module
====================================
Fetches and manages historical stock price data from yfinance for
Nifty 50 stocks, alongside traditional factor computation.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config import NIFTY50_STOCKS, STOCK_TICKERS
from database import db

logger = logging.getLogger("sentimentalpha.price_data")


# ── Price Fetcher ──────────────────────────────────────────────────────────

class PriceDataFetcher:
    """Fetches and stores historical prices via yfinance."""

    def __init__(self, lookback_days: int = 365):
        self.lookback_days = lookback_days

    def fetch_all_prices(self, tickers: list = None) -> int:
        """Fetch prices for all (or specified) Nifty 50 stocks."""
        tickers = tickers or STOCK_TICKERS
        total_inserted = 0

        for ticker in tickers:
            try:
                count = self._fetch_ticker(ticker)
                total_inserted += count
                logger.info(f"[{ticker}] Inserted {count} price records")
            except Exception as e:
                logger.error(f"[{ticker}] Price fetch failed: {e}")

        logger.info(f"Total new price records: {total_inserted}")
        return total_inserted

    def _fetch_ticker(self, ticker: str) -> int:
        """Fetch prices for a single ticker, starting from last available date."""
        latest_date = db.get_latest_price_date(ticker)
        if latest_date:
            start_date = (
                datetime.strptime(latest_date, "%Y-%m-%d") + timedelta(days=1)
            ).strftime("%Y-%m-%d")
        else:
            start_date = (
                datetime.now() - timedelta(days=self.lookback_days)
            ).strftime("%Y-%m-%d")

        end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date >= end_date:
            return 0

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            return 0

        records = []
        for idx, row in df.iterrows():
            records.append({
                "ticker": ticker,
                "trade_date": idx.strftime("%Y-%m-%d"),
                "open": round(float(row.get("Open", 0)), 2),
                "high": round(float(row.get("High", 0)), 2),
                "low": round(float(row.get("Low", 0)), 2),
                "close": round(float(row.get("Close", 0)), 2),
                "adj_close": round(float(row.get("Adj Close", row.get("Close", 0))), 2),
                "volume": int(row.get("Volume", 0)),
            })

        if records:
            db.insert_prices(records)

        return len(records)

    def get_price_dataframe(self, ticker: str, start_date: str = None,
                            end_date: str = None) -> pd.DataFrame:
        """Get prices from DB as a pandas DataFrame."""
        rows = db.get_prices(ticker, start_date, end_date)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()
        return df

    def get_multi_stock_prices(self, tickers: list = None,
                               start_date: str = None,
                               end_date: str = None) -> pd.DataFrame:
        """Get adj_close for multiple stocks pivoted into columns."""
        tickers = tickers or STOCK_TICKERS
        frames = []
        for ticker in tickers:
            df = self.get_price_dataframe(ticker, start_date, end_date)
            if not df.empty:
                frames.append(df[["adj_close"]].rename(columns={"adj_close": ticker}))

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=1).sort_index()


# ── Traditional Factor Calculator ──────────────────────────────────────────

class FactorCalculator:
    """Computes traditional quant factors (momentum, volatility, value)."""

    def __init__(self, price_fetcher: PriceDataFetcher):
        self.price_fetcher = price_fetcher

    def compute_all_factors(self, tickers: list = None,
                            as_of_date: str = None) -> pd.DataFrame:
        """Compute factor scores for all stocks as of a given date."""
        tickers = tickers or STOCK_TICKERS
        if as_of_date is None:
            as_of_date = datetime.now().strftime("%Y-%m-%d")

        all_factors = []
        for ticker in tickers:
            try:
                factors = self._compute_ticker_factors(ticker, as_of_date)
                if factors:
                    all_factors.append(factors)
                    # Store in database
                    db.upsert_factor_scores(
                        ticker=ticker,
                        factor_date=as_of_date,
                        **{k: v for k, v in factors.items() if k not in ("ticker", "factor_date")}
                    )
            except Exception as e:
                logger.warning(f"[{ticker}] Factor computation failed: {e}")

        return pd.DataFrame(all_factors) if all_factors else pd.DataFrame()

    def _compute_ticker_factors(self, ticker: str, as_of_date: str) -> Optional[dict]:
        """Compute factors for a single stock."""
        # Need at least 6 months of data
        start = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=200)).strftime("%Y-%m-%d")
        df = self.price_fetcher.get_price_dataframe(ticker, start, as_of_date)

        if len(df) < 30:
            return None

        close = df["adj_close"]

        # Momentum factors (% return over period)
        mom_1m = self._momentum(close, 21)
        mom_3m = self._momentum(close, 63)
        mom_6m = self._momentum(close, 126)

        # Volatility (20-day annualized)
        returns = close.pct_change().dropna()
        vol_20d = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else None

        # Value factors — fetch from yfinance info
        pe_ratio, pb_ratio = self._get_valuation(ticker)

        return {
            "ticker": ticker,
            "factor_date": as_of_date,
            "momentum_1m": mom_1m,
            "momentum_3m": mom_3m,
            "momentum_6m": mom_6m,
            "volatility_20d": vol_20d,
            "value_pe": pe_ratio,
            "value_pb": pb_ratio,
        }

    @staticmethod
    def _momentum(series: pd.Series, lookback: int) -> Optional[float]:
        """Compute simple momentum (% return) over lookback days."""
        if len(series) < lookback + 1:
            return None
        current = float(series.iloc[-1])
        past = float(series.iloc[-lookback - 1])
        if past == 0:
            return None
        return round((current - past) / past, 4)

    @staticmethod
    def _get_valuation(ticker: str) -> tuple:
        """Get PE and PB ratios from yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            pe = info.get("trailingPE") or info.get("forwardPE")
            pb = info.get("priceToBook")
            return (
                round(float(pe), 2) if pe else None,
                round(float(pb), 2) if pb else None,
            )
        except Exception:
            return None, None

    def compute_factor_history(self, tickers: list = None,
                               start_date: str = None,
                               end_date: str = None,
                               freq: str = "W") -> pd.DataFrame:
        """Compute factors at regular intervals for backtesting.

        Args:
            freq: 'D' for daily, 'W' for weekly, 'M' for monthly
        """
        tickers = tickers or STOCK_TICKERS
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        all_rows = []

        for dt in dates:
            date_str = dt.strftime("%Y-%m-%d")
            for ticker in tickers:
                try:
                    factors = self._compute_ticker_factors(ticker, date_str)
                    if factors:
                        all_rows.append(factors)
                except Exception:
                    continue

        return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


# ── CLI Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    fetcher = PriceDataFetcher(lookback_days=365)
    print("Fetching price data for Nifty 50...")
    count = fetcher.fetch_all_prices()
    print(f"Inserted {count} price records")

    calc = FactorCalculator(fetcher)
    print("\nComputing factors...")
    factors_df = calc.compute_all_factors()
    if not factors_df.empty:
        print(factors_df.to_string(index=False))
    print("Factor computation complete")
