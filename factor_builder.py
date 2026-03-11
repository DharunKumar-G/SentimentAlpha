"""
SentimentAlpha — Sentiment Factor Construction
================================================
Builds daily sentiment scores per stock, rolling averages,
and combines with traditional quant factors for alpha generation.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import NIFTY50_STOCKS, STOCK_TICKERS
from database import db

logger = logging.getLogger("sentimentalpha.factors")


class SentimentFactorBuilder:
    """
    Constructs daily sentiment factor scores from raw article-level sentiments.

    Pipeline:
        1. Aggregate article sentiments → daily raw score per stock
        2. Compute rolling averages (7-day, 14-day)
        3. Detect sudden sentiment shifts for alerts
        4. Merge with traditional factors for combined factor matrix
    """

    SENTIMENT_MAP = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}

    def build_daily_scores(self, days: int = 90) -> pd.DataFrame:
        """
        Build daily sentiment scores for all stocks.
        Aggregates article-level sentiments into a single daily score per stock.
        """
        logger.info(f"Building daily sentiment scores for past {days} days...")

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        all_records = []

        for ticker in STOCK_TICKERS:
            articles = db.get_articles_for_stock(ticker, days=days)
            if not articles:
                continue

            # Group articles by date
            daily_data = {}
            for article in articles:
                pub_date = article.get("published_at")
                if not pub_date:
                    continue
                if isinstance(pub_date, str):
                    try:
                        pub_date = datetime.fromisoformat(pub_date)
                    except (ValueError, TypeError):
                        continue

                date_key = pub_date.strftime("%Y-%m-%d") if isinstance(pub_date, datetime) else str(pub_date)[:10]

                if date_key not in daily_data:
                    daily_data[date_key] = {"scores": [], "confidences": []}

                sentiment = article.get("sentiment", "neutral")
                confidence = article.get("confidence", 0.5)
                score = self.SENTIMENT_MAP.get(sentiment, 0.0) * confidence

                daily_data[date_key]["scores"].append(score)
                daily_data[date_key]["confidences"].append(confidence)

            # Convert to records
            for date_key, data in sorted(daily_data.items()):
                raw_score = np.mean(data["scores"]) if data["scores"] else 0.0
                avg_confidence = np.mean(data["confidences"]) if data["confidences"] else 0.0

                all_records.append({
                    "ticker": ticker,
                    "factor_date": date_key,
                    "raw_score": round(float(raw_score), 4),
                    "article_count": len(data["scores"]),
                    "avg_confidence": round(float(avg_confidence), 3),
                })

        if not all_records:
            logger.warning("No sentiment data found for any stocks")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df["factor_date"] = pd.to_datetime(df["factor_date"])
        df = df.sort_values(["ticker", "factor_date"])

        # Compute rolling averages per stock
        for ticker in df["ticker"].unique():
            mask = df["ticker"] == ticker
            df.loc[mask, "rolling_7d"] = (
                df.loc[mask, "raw_score"]
                .rolling(window=7, min_periods=1)
                .mean()
                .round(4)
            )
            df.loc[mask, "rolling_14d"] = (
                df.loc[mask, "raw_score"]
                .rolling(window=14, min_periods=1)
                .mean()
                .round(4)
            )

        # Store in database
        for _, row in df.iterrows():
            db.upsert_daily_sentiment(
                ticker=row["ticker"],
                factor_date=row["factor_date"].strftime("%Y-%m-%d"),
                raw_score=row["raw_score"],
                rolling_7d=row.get("rolling_7d"),
                rolling_14d=row.get("rolling_14d"),
                article_count=row["article_count"],
                avg_confidence=row.get("avg_confidence"),
            )

        logger.info(f"Built {len(df)} daily sentiment records for {df['ticker'].nunique()} stocks")
        return df

    def get_sentiment_matrix(self, start_date: str = None,
                             end_date: str = None,
                             score_col: str = "rolling_7d") -> pd.DataFrame:
        """
        Get a pivoted sentiment matrix: dates × stocks.
        Suitable for backtesting and factor analysis.
        """
        records = db.get_all_daily_sentiment(start_date, end_date)
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["factor_date"] = pd.to_datetime(df["factor_date"])

        pivot = df.pivot_table(
            index="factor_date", columns="ticker",
            values=score_col, aggfunc="mean"
        )
        return pivot.sort_index()

    def detect_sentiment_shifts(self, threshold: float = 0.5,
                                window: int = 3) -> list:
        """
        Detect sudden sentiment shifts (potential trading signals).
        A shift is when rolling_7d moves by more than `threshold` over `window` days.
        """
        alerts = []
        records = db.get_all_daily_sentiment()
        if not records:
            return alerts

        df = pd.DataFrame(records)
        df["factor_date"] = pd.to_datetime(df["factor_date"])
        df = df.sort_values(["ticker", "factor_date"])

        for ticker in df["ticker"].unique():
            stock_df = df[df["ticker"] == ticker].copy()
            if len(stock_df) < window + 1:
                continue

            stock_df["shift"] = stock_df["rolling_7d"].diff(window)
            recent = stock_df.tail(3)

            for _, row in recent.iterrows():
                shift = row.get("shift", 0) or 0
                if abs(shift) >= threshold:
                    alert_type = "sentiment_spike" if shift > 0 else "sentiment_drop"
                    severity = "high" if abs(shift) >= threshold * 1.5 else "medium"
                    stock_name = NIFTY50_STOCKS.get(ticker, {}).get("name", ticker)

                    message = (
                        f"{stock_name} ({ticker}): Sentiment shifted by "
                        f"{shift:+.2f} over {window} days "
                        f"(current rolling_7d: {row.get('rolling_7d', 0):.2f})"
                    )

                    alert_id = db.insert_alert(
                        ticker=ticker,
                        alert_type=alert_type,
                        message=message,
                        severity=severity,
                    )
                    alerts.append({
                        "ticker": ticker, "type": alert_type,
                        "shift": shift, "message": message,
                    })

        logger.info(f"Detected {len(alerts)} sentiment shift alerts")
        return alerts

    def get_combined_factor_matrix(self, start_date: str = None,
                                   end_date: str = None) -> pd.DataFrame:
        """
        Build a combined matrix of sentiment + traditional factors.
        Returns a DataFrame with MultiIndex (date, ticker) and factor columns.
        """
        # Get sentiment data
        sent_records = db.get_all_daily_sentiment(start_date, end_date)
        factor_records = db.get_factor_scores(start_date=start_date, end_date=end_date)

        if not sent_records:
            logger.warning("No sentiment data available")
            return pd.DataFrame()

        sent_df = pd.DataFrame(sent_records)
        sent_df["factor_date"] = pd.to_datetime(sent_df["factor_date"])
        sent_df = sent_df.set_index(["factor_date", "ticker"])

        if factor_records:
            factor_df = pd.DataFrame(factor_records)
            factor_df["factor_date"] = pd.to_datetime(factor_df["factor_date"])
            factor_df = factor_df.set_index(["factor_date", "ticker"])

            # Merge on nearest date (factor scores may be weekly)
            combined = sent_df.join(factor_df, how="left", rsuffix="_trad")
            # Forward-fill traditional factors
            for col in ["momentum_1m", "momentum_3m", "momentum_6m",
                         "volatility_20d", "value_pe", "value_pb"]:
                if col in combined.columns:
                    combined[col] = combined.groupby(level="ticker")[col].ffill()
        else:
            combined = sent_df

        return combined.reset_index()

    def compute_sentiment_returns_correlation(self, lag_days: list = None) -> pd.DataFrame:
        """
        Test correlation between sentiment scores and forward returns
        at different lag periods.
        """
        if lag_days is None:
            lag_days = [1, 5, 10, 21]  # 1 day, 1 week, 2 weeks, 1 month

        results = []
        for ticker in STOCK_TICKERS:
            # Get sentiment
            sent_data = db.get_daily_sentiment(ticker, days=365)
            price_data = db.get_prices(ticker)

            if not sent_data or not price_data:
                continue

            sent_df = pd.DataFrame(sent_data)
            sent_df["factor_date"] = pd.to_datetime(sent_df["factor_date"])
            sent_df = sent_df.set_index("factor_date")

            price_df = pd.DataFrame(price_data)
            price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
            price_df = price_df.set_index("trade_date")

            if price_df.empty or sent_df.empty:
                continue

            for lag in lag_days:
                # Forward return = price(t+lag) / price(t) - 1
                price_df[f"fwd_return_{lag}d"] = (
                    price_df["adj_close"].shift(-lag) / price_df["adj_close"] - 1
                )

                # Merge sentiment with returns
                merged = sent_df[["rolling_7d"]].join(
                    price_df[f"fwd_return_{lag}d"], how="inner"
                )
                merged = merged.dropna()

                if len(merged) < 10:
                    continue

                corr = merged["rolling_7d"].corr(merged[f"fwd_return_{lag}d"])
                results.append({
                    "ticker": ticker,
                    "lag_days": lag,
                    "correlation": round(corr, 4) if not np.isnan(corr) else 0,
                    "n_observations": len(merged),
                })

        return pd.DataFrame(results) if results else pd.DataFrame()


# ── CLI Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    builder = SentimentFactorBuilder()

    print("Building daily sentiment scores...")
    df = builder.build_daily_scores(days=90)
    if not df.empty:
        print(f"\nSentiment matrix shape: {df.shape}")
        print(df.head(20).to_string(index=False))

    print("\nDetecting sentiment shifts...")
    alerts = builder.detect_sentiment_shifts()
    for alert in alerts:
        print(f"  [!] {alert['message']}")

    print("\nComputing sentiment-return correlations...")
    corr_df = builder.compute_sentiment_returns_correlation()
    if not corr_df.empty:
        print(corr_df.to_string(index=False))

    print("\nFactor construction complete")
