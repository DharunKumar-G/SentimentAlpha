"""
SentimentAlpha — Backtesting Engine
=====================================
Backtests sentiment-based trading strategies against traditional
factor strategies and combined approaches. Includes lag analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import NIFTY50_STOCKS, STOCK_TICKERS
from database import db
from price_data import PriceDataFetcher

logger = logging.getLogger("sentimentalpha.backtest")


# ── Performance Metrics ────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.06) -> dict:
    """Compute standard performance metrics from a return series."""
    if returns.empty or returns.isna().all():
        return {
            "total_return": 0, "annualized_return": 0, "sharpe_ratio": 0,
            "max_drawdown": 0, "volatility": 0, "win_rate": 0,
            "avg_win": 0, "avg_loss": 0, "num_trades": 0,
        }

    cumulative = (1 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1) if len(cumulative) > 0 else 0

    # Annualized return (assume ~252 trading days)
    n_days = len(returns)
    if n_days > 0 and total_return > -1:
        annualized = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    else:
        annualized = 0

    # Sharpe ratio
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = returns - daily_rf
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    # Max drawdown
    cummax = cumulative.cummax()
    drawdown = (cumulative - cummax) / cummax
    max_dd = float(drawdown.min())

    # Win rate
    wins = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = float(wins / total_trades) if total_trades > 0 else 0

    # Average win / loss
    pos_returns = returns[returns > 0]
    neg_returns = returns[returns < 0]

    return {
        "total_return": round(total_return, 4),
        "annualized_return": round(annualized, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "volatility": round(float(returns.std() * np.sqrt(252)), 4),
        "win_rate": round(win_rate, 4),
        "avg_win": round(float(pos_returns.mean()), 4) if len(pos_returns) > 0 else 0,
        "avg_loss": round(float(neg_returns.mean()), 4) if len(neg_returns) > 0 else 0,
        "num_trades": int(total_trades),
    }


# ── Base Strategy ──────────────────────────────────────────────────────────

class BaseStrategy:
    """Base class for backtesting strategies."""

    def __init__(self, name: str, top_n: int = 10, rebalance_freq: str = "W"):
        self.name = name
        self.top_n = top_n
        self.rebalance_freq = rebalance_freq
        self.price_fetcher = PriceDataFetcher()

    def get_signals(self, as_of_date: str) -> pd.DataFrame:
        """
        Return a DataFrame with columns ['ticker', 'signal_score']
        sorted by signal_score descending. Override in subclasses.
        """
        raise NotImplementedError

    def backtest(self, start_date: str, end_date: str) -> dict:
        """Run the backtest and return results."""
        logger.info(f"Backtesting [{self.name}] from {start_date} to {end_date}")

        # Get price data
        prices_pivot = self.price_fetcher.get_multi_stock_prices(
            start_date=start_date, end_date=end_date
        )
        if prices_pivot.empty:
            logger.warning("No price data available for backtesting")
            return {}

        # Daily returns
        daily_returns = prices_pivot.pct_change().dropna()

        # Generate rebalance dates
        rebalance_dates = pd.date_range(
            start=start_date, end=end_date, freq=self.rebalance_freq
        )

        # Track portfolio returns
        portfolio_returns = []
        holdings = {}  # current portfolio

        for date in daily_returns.index:
            # Rebalance if needed
            date_str = date.strftime("%Y-%m-%d")
            if any(abs((date - rd).days) <= 1 for rd in rebalance_dates):
                try:
                    signals = self.get_signals(date_str)
                    if not signals.empty:
                        # Go long top_n stocks with highest signal
                        top_stocks = signals.nlargest(self.top_n, "signal_score")["ticker"].tolist()
                        # Equal weight
                        available = [t for t in top_stocks if t in daily_returns.columns]
                        if available:
                            holdings = {t: 1.0 / len(available) for t in available}
                except Exception as e:
                    logger.debug(f"Signal generation failed for {date_str}: {e}")

            # Compute portfolio return for this day
            if holdings:
                day_return = sum(
                    weight * daily_returns.loc[date].get(ticker, 0)
                    for ticker, weight in holdings.items()
                    if ticker in daily_returns.columns
                )
                portfolio_returns.append({"date": date, "return": day_return})

        if not portfolio_returns:
            return {}

        returns_df = pd.DataFrame(portfolio_returns).set_index("date")
        returns_series = returns_df["return"]

        # Compute metrics
        metrics = compute_metrics(returns_series)
        metrics["strategy_name"] = self.name
        metrics["start_date"] = start_date
        metrics["end_date"] = end_date

        # Cumulative returns for plotting
        metrics["cumulative_returns"] = ((1 + returns_series).cumprod()).to_dict()
        metrics["daily_returns"] = returns_series.to_dict()

        # Store in database
        db.insert_backtest(
            strategy_name=self.name,
            start_date=start_date,
            end_date=end_date,
            total_return=metrics["total_return"],
            annualized_return=metrics["annualized_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            num_trades=metrics["num_trades"],
            params={"top_n": self.top_n, "rebalance_freq": self.rebalance_freq},
        )

        logger.info(
            f"[{self.name}] Return: {metrics['total_return']:.2%}, "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
            f"MaxDD: {metrics['max_drawdown']:.2%}"
        )
        return metrics


# ── Sentiment-Only Strategy ────────────────────────────────────────────────

class SentimentStrategy(BaseStrategy):
    """
    Go long on stocks with highest positive sentiment shift.
    Avoid stocks with negative sentiment.
    """

    def __init__(self, top_n: int = 10, rebalance_freq: str = "W",
                 score_col: str = "rolling_7d"):
        super().__init__(
            name=f"Sentiment-Only (top{top_n}, {score_col})",
            top_n=top_n, rebalance_freq=rebalance_freq,
        )
        self.score_col = score_col

    def get_signals(self, as_of_date: str) -> pd.DataFrame:
        records = db.get_all_daily_sentiment(end_date=as_of_date)
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["factor_date"] = pd.to_datetime(df["factor_date"])

        # Get most recent score per stock
        latest = df.groupby("ticker").last().reset_index()

        return latest[["ticker", self.score_col]].rename(
            columns={self.score_col: "signal_score"}
        ).dropna()


# ── Momentum Strategy ─────────────────────────────────────────────────────

class MomentumStrategy(BaseStrategy):
    """Traditional momentum strategy — buy highest 3-month momentum."""

    def __init__(self, top_n: int = 10, rebalance_freq: str = "W"):
        super().__init__(
            name=f"Momentum-Only (top{top_n})",
            top_n=top_n, rebalance_freq=rebalance_freq,
        )

    def get_signals(self, as_of_date: str) -> pd.DataFrame:
        records = db.get_factor_scores(end_date=as_of_date)
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["factor_date"] = pd.to_datetime(df["factor_date"])

        latest = df.groupby("ticker").last().reset_index()

        if "momentum_3m" not in latest.columns:
            return pd.DataFrame()

        return latest[["ticker", "momentum_3m"]].rename(
            columns={"momentum_3m": "signal_score"}
        ).dropna()


# ── Combined Strategy ─────────────────────────────────────────────────────

class CombinedStrategy(BaseStrategy):
    """
    Combines sentiment + momentum + value factors.
    Z-score normalization, then equal-weight combination.
    """

    def __init__(self, top_n: int = 10, rebalance_freq: str = "W",
                 weights: dict = None):
        super().__init__(
            name=f"Combined (top{top_n})",
            top_n=top_n, rebalance_freq=rebalance_freq,
        )
        self.weights = weights or {
            "sentiment": 0.4,
            "momentum": 0.35,
            "value": 0.25,
        }

    def get_signals(self, as_of_date: str) -> pd.DataFrame:
        # Get sentiment scores
        sent_records = db.get_all_daily_sentiment(end_date=as_of_date)
        factor_records = db.get_factor_scores(end_date=as_of_date)

        if not sent_records:
            return pd.DataFrame()

        # Latest sentiment per stock
        sent_df = pd.DataFrame(sent_records)
        sent_df["factor_date"] = pd.to_datetime(sent_df["factor_date"])
        sent_latest = sent_df.groupby("ticker")["rolling_7d"].last().reset_index()
        sent_latest.columns = ["ticker", "sentiment_raw"]

        # Z-score normalize
        sent_latest["sentiment_z"] = self._zscore(sent_latest["sentiment_raw"])

        signals = sent_latest[["ticker", "sentiment_z"]].copy()

        if factor_records:
            factor_df = pd.DataFrame(factor_records)
            factor_df["factor_date"] = pd.to_datetime(factor_df["factor_date"])
            factor_latest = factor_df.groupby("ticker").last().reset_index()

            if "momentum_3m" in factor_latest.columns:
                factor_latest["momentum_z"] = self._zscore(factor_latest["momentum_3m"])
                signals = signals.merge(
                    factor_latest[["ticker", "momentum_z"]], on="ticker", how="left"
                )

            if "value_pe" in factor_latest.columns:
                # Lower PE is better, so negate
                factor_latest["value_z"] = -self._zscore(factor_latest["value_pe"])
                signals = signals.merge(
                    factor_latest[["ticker", "value_z"]], on="ticker", how="left"
                )

        # Compute combined score
        signals["signal_score"] = 0.0
        if "sentiment_z" in signals.columns:
            signals["signal_score"] += self.weights["sentiment"] * signals["sentiment_z"].fillna(0)
        if "momentum_z" in signals.columns:
            signals["signal_score"] += self.weights["momentum"] * signals["momentum_z"].fillna(0)
        if "value_z" in signals.columns:
            signals["signal_score"] += self.weights["value"] * signals["value_z"].fillna(0)

        return signals[["ticker", "signal_score"]].dropna()

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """Z-score normalization."""
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0, index=series.index)
        return (series - series.mean()) / std


# ── Lag Analysis ───────────────────────────────────────────────────────────

class LagAnalyzer:
    """
    Analyze predictive power of sentiment at different time horizons.
    Tests if sentiment predicts returns 1 day, 1 week, or 1 month ahead.
    """

    def __init__(self):
        self.price_fetcher = PriceDataFetcher()

    def run_lag_analysis(self, lags: list = None) -> pd.DataFrame:
        """
        For each lag period, compute:
        - Average correlation between sentiment and forward returns
        - Information coefficient (IC)
        - Hit rate (% of times sentiment correctly predicts direction)
        """
        if lags is None:
            lags = [1, 3, 5, 10, 21]  # days ahead

        results = []

        for ticker in STOCK_TICKERS:
            sent_data = db.get_daily_sentiment(ticker, days=365)
            price_data = db.get_prices(ticker)

            if not sent_data or not price_data:
                continue

            sent_df = pd.DataFrame(sent_data)
            sent_df["factor_date"] = pd.to_datetime(sent_df["factor_date"])
            sent_df = sent_df.set_index("factor_date").sort_index()

            price_df = pd.DataFrame(price_data)
            price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
            price_df = price_df.set_index("trade_date").sort_index()

            for lag in lags:
                # Forward return
                price_df[f"fwd_{lag}d"] = price_df["adj_close"].pct_change(lag).shift(-lag)

                merged = sent_df[["rolling_7d"]].join(price_df[f"fwd_{lag}d"], how="inner").dropna()

                if len(merged) < 20:
                    continue

                # Correlation (Information Coefficient)
                ic = merged["rolling_7d"].corr(merged[f"fwd_{lag}d"])

                # Hit rate: sentiment direction matches return direction
                merged["sent_dir"] = np.sign(merged["rolling_7d"])
                merged["ret_dir"] = np.sign(merged[f"fwd_{lag}d"])
                hit_rate = (merged["sent_dir"] == merged["ret_dir"]).mean()

                # Average return when sentiment is positive vs negative
                pos_mask = merged["rolling_7d"] > 0
                neg_mask = merged["rolling_7d"] < 0

                avg_ret_pos = float(merged.loc[pos_mask, f"fwd_{lag}d"].mean()) if pos_mask.any() else 0
                avg_ret_neg = float(merged.loc[neg_mask, f"fwd_{lag}d"].mean()) if neg_mask.any() else 0

                results.append({
                    "ticker": ticker,
                    "lag_days": lag,
                    "ic": round(ic, 4) if not np.isnan(ic) else 0,
                    "hit_rate": round(hit_rate, 4),
                    "avg_return_pos_sentiment": round(avg_ret_pos, 4),
                    "avg_return_neg_sentiment": round(avg_ret_neg, 4),
                    "n_observations": len(merged),
                    "spread": round(avg_ret_pos - avg_ret_neg, 4),
                })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Aggregate across stocks
        agg = df.groupby("lag_days").agg({
            "ic": "mean",
            "hit_rate": "mean",
            "avg_return_pos_sentiment": "mean",
            "avg_return_neg_sentiment": "mean",
            "spread": "mean",
            "n_observations": "sum",
        }).round(4).reset_index()

        return agg


# ── Backtest Runner ────────────────────────────────────────────────────────

class BacktestRunner:
    """Runs all strategies and compares performance."""

    def run_comparison(self, start_date: str = None, end_date: str = None,
                       top_n: int = 10) -> dict:
        """Run all strategies and return comparison results."""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        strategies = [
            SentimentStrategy(top_n=top_n, score_col="rolling_7d"),
            SentimentStrategy(top_n=top_n, score_col="raw_score"),
            MomentumStrategy(top_n=top_n),
            CombinedStrategy(top_n=top_n),
        ]

        results = {}
        for strategy in strategies:
            try:
                result = strategy.backtest(start_date, end_date)
                results[strategy.name] = result
            except Exception as e:
                logger.error(f"Strategy [{strategy.name}] failed: {e}")
                results[strategy.name] = {"error": str(e)}

        # Summary comparison
        comparison = []
        for name, result in results.items():
            if "error" not in result and result:
                comparison.append({
                    "Strategy": name,
                    "Total Return": f"{result.get('total_return', 0):.2%}",
                    "Annualized": f"{result.get('annualized_return', 0):.2%}",
                    "Sharpe": f"{result.get('sharpe_ratio', 0):.2f}",
                    "Max DD": f"{result.get('max_drawdown', 0):.2%}",
                    "Win Rate": f"{result.get('win_rate', 0):.2%}",
                    "Trades": result.get("num_trades", 0),
                })

        if comparison:
            comp_df = pd.DataFrame(comparison)
            logger.info("\n" + comp_df.to_string(index=False))

        # Lag analysis
        logger.info("\nRunning lag analysis...")
        lag_analyzer = LagAnalyzer()
        lag_results = lag_analyzer.run_lag_analysis()

        return {
            "strategy_results": results,
            "comparison": comparison,
            "lag_analysis": lag_results.to_dict("records") if not lag_results.empty else [],
        }


# ── CLI Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    runner = BacktestRunner()
    results = runner.run_comparison()

    print("\n" + "=" * 70)
    print("BACKTEST COMPARISON")
    print("=" * 70)

    if results["comparison"]:
        comp_df = pd.DataFrame(results["comparison"])
        print(comp_df.to_string(index=False))
    else:
        print("No backtest results (need sentiment + price data first)")

    if results["lag_analysis"]:
        print("\n" + "=" * 70)
        print("LAG ANALYSIS")
        print("=" * 70)
        lag_df = pd.DataFrame(results["lag_analysis"])
        print(lag_df.to_string(index=False))

    print("\nBacktesting complete")
