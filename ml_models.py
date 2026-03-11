"""
SentimentAlpha — ML Signal Combination
========================================
Trains ML models (Random Forest, XGBoost) that combine sentiment
and traditional factor scores to predict next-month returns.
Includes walk-forward validation and feature importance analysis.
"""

import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, classification_report,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from config import MODELS_DIR, STOCK_TICKERS
from database import db
from price_data import PriceDataFetcher

logger = logging.getLogger("sentimentalpha.ml_models")


# ── Feature Engineering ────────────────────────────────────────────────────

class FeatureEngineer:
    """Builds feature matrix from sentiment + factor data + price-derived features."""

    FEATURE_COLS = [
        "raw_score", "rolling_7d", "rolling_14d", "avg_confidence", "article_count",
        "momentum_1m", "momentum_3m", "momentum_6m", "volatility_20d",
        "value_pe", "value_pb",
        "sent_momentum",         # change in rolling_7d over past 5 days
        "sent_vol",              # std of raw_score over past 14 days
        "sent_surprise",         # current raw_score vs rolling_14d
    ]

    def __init__(self):
        self.price_fetcher = PriceDataFetcher()

    def build_feature_matrix(self, lookback_months: int = 12,
                             forward_days: int = 21) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build features (X) and target (y) for ML training.

        Target: forward N-day return (continuous or binary)
        Features: sentiment scores + traditional factors

        Returns:
            X: feature matrix (date, ticker, features)
            y: forward return series
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_months * 30)).strftime("%Y-%m-%d")

        # Get sentiment data
        sent_records = db.get_all_daily_sentiment(start_date, end_date)
        if not sent_records:
            logger.warning("No sentiment data for feature engineering")
            return pd.DataFrame(), pd.Series()

        sent_df = pd.DataFrame(sent_records)
        sent_df["factor_date"] = pd.to_datetime(sent_df["factor_date"])
        sent_df = sent_df.sort_values(["ticker", "factor_date"])

        # Add derived sentiment features
        for ticker in sent_df["ticker"].unique():
            mask = sent_df["ticker"] == ticker
            # Sentiment momentum (5-day change in rolling_7d)
            sent_df.loc[mask, "sent_momentum"] = sent_df.loc[mask, "rolling_7d"].diff(5)
            # Sentiment volatility (14-day std of raw_score)
            sent_df.loc[mask, "sent_vol"] = (
                sent_df.loc[mask, "raw_score"].rolling(14, min_periods=3).std()
            )
            # Sentiment surprise (current vs 14d average)
            sent_df.loc[mask, "sent_surprise"] = (
                sent_df.loc[mask, "raw_score"] - sent_df.loc[mask, "rolling_14d"]
            )

        # Get traditional factor data
        factor_records = db.get_factor_scores(start_date=start_date, end_date=end_date)
        if factor_records:
            factor_df = pd.DataFrame(factor_records)
            factor_df["factor_date"] = pd.to_datetime(factor_df["factor_date"])

            # Merge on (ticker, factor_date) — use nearest date join
            sent_df = pd.merge_asof(
                sent_df.sort_values("factor_date"),
                factor_df[["ticker", "factor_date", "momentum_1m", "momentum_3m",
                            "momentum_6m", "volatility_20d", "value_pe", "value_pb"]]
                .sort_values("factor_date"),
                on="factor_date", by="ticker", direction="backward",
            )

        # Compute forward returns (target variable)
        all_targets = []
        for ticker in sent_df["ticker"].unique():
            prices = db.get_prices(ticker, start_date, end_date)
            if not prices:
                continue

            price_df = pd.DataFrame(prices)
            price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
            price_df = price_df.set_index("trade_date").sort_index()
            price_df[f"fwd_return_{forward_days}d"] = (
                price_df["adj_close"].pct_change(forward_days).shift(-forward_days)
            )

            ticker_sent = sent_df[sent_df["ticker"] == ticker].copy()
            ticker_sent = ticker_sent.set_index("factor_date")

            merged = ticker_sent.join(
                price_df[[f"fwd_return_{forward_days}d"]], how="inner"
            )
            merged["ticker"] = ticker
            all_targets.append(merged)

        if not all_targets:
            return pd.DataFrame(), pd.Series()

        combined = pd.concat(all_targets).reset_index()
        combined = combined.rename(columns={"index": "date", "factor_date": "date"})

        # Select features that exist
        available_features = [c for c in self.FEATURE_COLS if c in combined.columns]
        X = combined[["date", "ticker"] + available_features].copy()
        y = combined[f"fwd_return_{forward_days}d"]

        # Drop rows with missing target
        valid = y.notna()
        X = X[valid].reset_index(drop=True)
        y = y[valid].reset_index(drop=True)

        logger.info(
            f"Feature matrix built: {X.shape[0]} samples, "
            f"{len(available_features)} features, "
            f"{X['ticker'].nunique()} stocks"
        )
        return X, y


# ── ML Model Trainer ───────────────────────────────────────────────────────

class MLModelTrainer:
    """
    Trains and evaluates ML models for return prediction.
    Supports walk-forward validation to prevent lookahead bias.
    """

    def __init__(self, forward_days: int = 21):
        self.forward_days = forward_days
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.scalers = {}

    def train_all_models(self) -> dict:
        """Train all models and return evaluation results."""
        X, y = self.feature_engineer.build_feature_matrix(
            forward_days=self.forward_days
        )

        if X.empty:
            logger.warning("No data available for model training")
            return {}

        results = {}

        # 1. Random Forest
        logger.info("Training Random Forest...")
        rf_results = self._train_random_forest(X, y)
        results["random_forest"] = rf_results

        # 2. XGBoost (if available)
        if HAS_XGBOOST:
            logger.info("Training XGBoost...")
            xgb_results = self._train_xgboost(X, y)
            results["xgboost"] = xgb_results
        else:
            logger.warning("XGBoost not installed, skipping")

        # 3. Random Forest Classifier (direction prediction)
        logger.info("Training RF Classifier (direction)...")
        rfc_results = self._train_direction_classifier(X, y)
        results["rf_classifier"] = rfc_results

        return results

    def walk_forward_validation(self, n_splits: int = 5,
                                train_months: int = 6,
                                test_months: int = 1) -> dict:
        """
        Walk-forward validation: train on past, test on future.
        No lookahead bias.
        """
        X, y = self.feature_engineer.build_feature_matrix(
            lookback_months=train_months + test_months * n_splits + 3,
            forward_days=self.forward_days,
        )

        if X.empty or len(X) < 100:
            logger.warning("Insufficient data for walk-forward validation")
            return {}

        feature_cols = [c for c in X.columns if c not in ["date", "ticker"]]
        dates = X["date"].sort_values().unique()

        if len(dates) < 60:
            logger.warning("Not enough date range for walk-forward")
            return {}

        # Create time-based splits
        split_results = []
        step = len(dates) // (n_splits + 1)

        for i in range(n_splits):
            train_end_idx = step * (i + 1)
            test_end_idx = min(train_end_idx + step, len(dates) - 1)

            train_end = dates[train_end_idx]
            test_start = dates[train_end_idx + 1] if train_end_idx + 1 < len(dates) else dates[-1]
            test_end = dates[test_end_idx]

            train_mask = X["date"] <= train_end
            test_mask = (X["date"] > train_end) & (X["date"] <= test_end)

            X_train = X.loc[train_mask, feature_cols].fillna(0)
            y_train = y[train_mask]
            X_test = X.loc[test_mask, feature_cols].fillna(0)
            y_test = y[test_mask]

            if len(X_train) < 20 or len(X_test) < 5:
                continue

            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Train Random Forest
            rf = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
            )
            rf.fit(X_train_s, y_train)
            pred = rf.predict(X_test_s)

            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred) if len(y_test) > 1 else 0

            # Direction accuracy
            dir_accuracy = ((np.sign(pred) == np.sign(y_test)).mean()
                           if len(y_test) > 0 else 0)

            # IC (information coefficient)
            ic = np.corrcoef(pred, y_test)[0, 1] if len(pred) > 1 else 0

            split_results.append({
                "split": i + 1,
                "train_end": str(train_end)[:10],
                "test_start": str(test_start)[:10],
                "test_end": str(test_end)[:10],
                "train_size": len(X_train),
                "test_size": len(X_test),
                "rmse": round(rmse, 6),
                "r2": round(r2, 4),
                "direction_accuracy": round(dir_accuracy, 4),
                "ic": round(ic, 4) if not np.isnan(ic) else 0,
            })

        if not split_results:
            return {}

        results_df = pd.DataFrame(split_results)

        # Average metrics
        avg_metrics = {
            "avg_rmse": round(results_df["rmse"].mean(), 6),
            "avg_r2": round(results_df["r2"].mean(), 4),
            "avg_direction_accuracy": round(results_df["direction_accuracy"].mean(), 4),
            "avg_ic": round(results_df["ic"].mean(), 4),
            "n_splits": n_splits,
            "splits": split_results,
        }

        logger.info(
            f"Walk-Forward: Avg RMSE={avg_metrics['avg_rmse']:.6f}, "
            f"Avg IC={avg_metrics['avg_ic']:.4f}, "
            f"Avg Dir Accuracy={avg_metrics['avg_direction_accuracy']:.2%}"
        )

        return avg_metrics

    def get_feature_importance(self, model_name: str = "random_forest") -> pd.DataFrame:
        """Get feature importance from a trained model."""
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found. Train first.")
            return pd.DataFrame()

        model = self.models[model_name]
        feature_names = model._feature_names if hasattr(model, '_feature_names') else []

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            df = pd.DataFrame({
                "feature": feature_names[:len(importances)],
                "importance": importances,
            }).sort_values("importance", ascending=False)
            return df.round(4)

        return pd.DataFrame()

    def predict(self, model_name: str = "random_forest") -> pd.DataFrame:
        """Generate predictions for current data using a trained model."""
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not trained")
            return pd.DataFrame()

        X, _ = self.feature_engineer.build_feature_matrix(
            lookback_months=3, forward_days=self.forward_days
        )

        if X.empty:
            return pd.DataFrame()

        feature_cols = [c for c in X.columns if c not in ["date", "ticker"]]

        # Get latest data per stock
        latest = X.groupby("ticker").last().reset_index()
        X_pred = latest[feature_cols].fillna(0)

        scaler = self.scalers.get(model_name)
        if scaler:
            X_pred = scaler.transform(X_pred)

        model = self.models[model_name]
        predictions = model.predict(X_pred)

        results = pd.DataFrame({
            "ticker": latest["ticker"],
            "predicted_return": np.round(predictions, 4),
        }).sort_values("predicted_return", ascending=False)

        # Store predictions
        for _, row in results.iterrows():
            db.insert_prediction(
                model_name=model_name,
                ticker=row["ticker"],
                prediction_date=datetime.now().strftime("%Y-%m-%d"),
                predicted_return=float(row["predicted_return"]),
            )

        return results

    def save_model(self, model_name: str):
        """Save a trained model to disk."""
        if model_name not in self.models:
            return
        path = MODELS_DIR / f"{model_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.models[model_name],
                "scaler": self.scalers.get(model_name),
            }, f)
        logger.info(f"Model saved: {path}")

    def load_model(self, model_name: str) -> bool:
        """Load a trained model from disk."""
        path = MODELS_DIR / f"{model_name}.pkl"
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.models[model_name] = data["model"]
            self.scalers[model_name] = data.get("scaler")
        logger.info(f"Model loaded: {path}")
        return True

    # ── Private Training Methods ───────────────────────────────────────

    def _train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train a Random Forest regressor."""
        feature_cols = [c for c in X.columns if c not in ["date", "ticker"]]
        X_feat = X[feature_cols].fillna(0)

        # Train/test split (time-based: last 20% for test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_feat[:split_idx], X_feat[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train
        model = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train_s, y_train)
        model._feature_names = feature_cols  # Store for importance

        # Evaluate
        train_pred = model.predict(X_train_s)
        test_pred = model.predict(X_test_s)

        self.models["random_forest"] = model
        self.scalers["random_forest"] = scaler

        results = {
            "model": "Random Forest",
            "train_rmse": round(np.sqrt(mean_squared_error(y_train, train_pred)), 6),
            "test_rmse": round(np.sqrt(mean_squared_error(y_test, test_pred)), 6),
            "train_r2": round(r2_score(y_train, train_pred), 4),
            "test_r2": round(r2_score(y_test, test_pred), 4) if len(y_test) > 1 else 0,
            "test_direction_accuracy": round(
                (np.sign(test_pred) == np.sign(y_test)).mean(), 4
            ) if len(y_test) > 0 else 0,
            "feature_importance": self.get_feature_importance("random_forest").to_dict("records"),
        }

        self.save_model("random_forest")
        return results

    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train an XGBoost regressor."""
        feature_cols = [c for c in X.columns if c not in ["date", "ticker"]]
        X_feat = X[feature_cols].fillna(0)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_feat[:split_idx], X_feat[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            verbose=False,
        )
        model._feature_names = feature_cols

        train_pred = model.predict(X_train_s)
        test_pred = model.predict(X_test_s)

        self.models["xgboost"] = model
        self.scalers["xgboost"] = scaler

        results = {
            "model": "XGBoost",
            "train_rmse": round(np.sqrt(mean_squared_error(y_train, train_pred)), 6),
            "test_rmse": round(np.sqrt(mean_squared_error(y_test, test_pred)), 6),
            "train_r2": round(r2_score(y_train, train_pred), 4),
            "test_r2": round(r2_score(y_test, test_pred), 4) if len(y_test) > 1 else 0,
            "test_direction_accuracy": round(
                (np.sign(test_pred) == np.sign(y_test)).mean(), 4
            ) if len(y_test) > 0 else 0,
            "feature_importance": self.get_feature_importance("xgboost").to_dict("records"),
        }

        self.save_model("xgboost")
        return results

    def _train_direction_classifier(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train a classifier predicting return direction (+/-)."""
        feature_cols = [c for c in X.columns if c not in ["date", "ticker"]]
        X_feat = X[feature_cols].fillna(0)
        y_binary = (y > 0).astype(int)  # 1 = positive return, 0 = negative

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_feat[:split_idx], X_feat[split_idx:]
        y_train, y_test = y_binary[:split_idx], y_binary[split_idx:]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train_s, y_train)
        model._feature_names = feature_cols

        train_pred = model.predict(X_train_s)
        test_pred = model.predict(X_test_s)

        self.models["rf_classifier"] = model
        self.scalers["rf_classifier"] = scaler

        results = {
            "model": "RF Direction Classifier",
            "train_accuracy": round(accuracy_score(y_train, train_pred), 4),
            "test_accuracy": round(accuracy_score(y_test, test_pred), 4) if len(y_test) > 0 else 0,
            "test_precision": round(
                precision_score(y_test, test_pred, zero_division=0), 4
            ) if len(y_test) > 0 else 0,
            "test_recall": round(
                recall_score(y_test, test_pred, zero_division=0), 4
            ) if len(y_test) > 0 else 0,
            "feature_importance": self.get_feature_importance("rf_classifier").to_dict("records"),
        }

        self.save_model("rf_classifier")
        return results


# ── CLI Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    trainer = MLModelTrainer(forward_days=21)

    print("Training ML models...")
    results = trainer.train_all_models()

    for model_name, metrics in results.items():
        print(f"\n{'='*50}")
        print(f"Model: {metrics.get('model', model_name)}")
        print(f"{'='*50}")
        for k, v in metrics.items():
            if k not in ("feature_importance", "model"):
                print(f"  {k}: {v}")

        if "feature_importance" in metrics and metrics["feature_importance"]:
            print("\n  Feature Importance:")
            for fi in metrics["feature_importance"][:10]:
                print(f"    {fi['feature']:25s} {fi['importance']:.4f}")

    print("\n\nRunning walk-forward validation...")
    wf_results = trainer.walk_forward_validation(n_splits=5)
    if wf_results:
        print(f"\nWalk-Forward Summary:")
        for k, v in wf_results.items():
            if k != "splits":
                print(f"  {k}: {v}")

    print("\n\nGenerating predictions...")
    predictions = trainer.predict("random_forest")
    if not predictions.empty:
        print("\nTop 10 predicted returns:")
        print(predictions.head(10).to_string(index=False))

    print("\nML model training complete")
