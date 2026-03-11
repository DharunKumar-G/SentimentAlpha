"""
SentimentAlpha — Main Pipeline Orchestrator
=============================================
Entry point for running the full sentiment trading pipeline
or individual components via CLI flags.
"""

import argparse
import logging
import sys
import time
from datetime import datetime

from config import LLM_PROVIDER, SCRAPE_INTERVAL_MINUTES

# ── Logging Setup ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/sentimentalpha.log", mode="a"),
    ],
)
logger = logging.getLogger("sentimentalpha.main")


# ── Pipeline Steps ─────────────────────────────────────────────────────────

def step_ingest():
    """Step 1: Scrape news from all sources."""
    logger.info("=" * 60)
    logger.info("STEP 1: News Ingestion")
    logger.info("=" * 60)
    from ingestion import NewsIngestionPipeline
    pipeline = NewsIngestionPipeline()
    results = pipeline.run_full_ingestion()
    total = sum(results.values())
    logger.info(f"Ingestion complete: {total} new articles ({results})")
    return results


def step_prices():
    """Step 2: Fetch latest stock prices."""
    logger.info("=" * 60)
    logger.info("STEP 2: Price Data Fetch")
    logger.info("=" * 60)
    from price_data import PriceDataFetcher
    fetcher = PriceDataFetcher(lookback_days=365)
    count = fetcher.fetch_all_prices()
    logger.info(f"Price fetch complete: {count} new records")
    return count


def step_sentiment():
    """Step 3: Run LLM sentiment analysis on new articles."""
    logger.info("=" * 60)
    logger.info(f"STEP 3: Sentiment Analysis (provider={LLM_PROVIDER})")
    logger.info("=" * 60)
    from sentiment_analyzer import SentimentPipeline
    pipeline = SentimentPipeline()
    count = pipeline.analyze_pending_articles(limit=100)
    logger.info(f"Sentiment analysis complete: {count} articles processed")
    return count


def step_factors():
    """Step 4: Build sentiment factors and detect shifts."""
    logger.info("=" * 60)
    logger.info("STEP 4: Factor Construction")
    logger.info("=" * 60)
    from factor_builder import SentimentFactorBuilder
    from price_data import PriceDataFetcher, FactorCalculator

    # Build sentiment factors
    builder = SentimentFactorBuilder()
    sent_df = builder.build_daily_scores(days=90)
    logger.info(f"Built {len(sent_df)} daily sentiment records")

    # Compute traditional factors
    fetcher = PriceDataFetcher()
    calc = FactorCalculator(fetcher)
    factor_df = calc.compute_all_factors()
    logger.info(f"Computed traditional factors for {len(factor_df)} stocks")

    # Detect sentiment shifts
    alerts = builder.detect_sentiment_shifts(threshold=0.4)
    logger.info(f"Detected {len(alerts)} sentiment shift alerts")

    return {"sentiment_records": len(sent_df), "factor_records": len(factor_df), "alerts": len(alerts)}


def step_backtest():
    """Step 5: Run backtesting."""
    logger.info("=" * 60)
    logger.info("STEP 5: Backtesting")
    logger.info("=" * 60)
    from backtester import BacktestRunner
    runner = BacktestRunner()
    results = runner.run_comparison()
    logger.info("Backtesting complete")
    return results.get("comparison", [])


def step_ml():
    """Step 6: Train ML models."""
    logger.info("=" * 60)
    logger.info("STEP 6: ML Model Training")
    logger.info("=" * 60)
    from ml_models import MLModelTrainer
    trainer = MLModelTrainer(forward_days=21)

    # Train models
    train_results = trainer.train_all_models()

    # Walk-forward validation
    wf_results = trainer.walk_forward_validation(n_splits=5)

    # Generate predictions
    predictions = trainer.predict("random_forest")

    logger.info("ML training complete")
    return {
        "train_results": {k: {kk: vv for kk, vv in v.items() if kk != "feature_importance"}
                          for k, v in train_results.items()},
        "walk_forward": {k: v for k, v in wf_results.items() if k != "splits"} if wf_results else {},
        "n_predictions": len(predictions),
    }


def step_briefs():
    """Step 7: Generate research briefs for top stocks."""
    logger.info("=" * 60)
    logger.info("STEP 7: Research Brief Generation")
    logger.info("=" * 60)
    from sentiment_analyzer import SentimentPipeline
    from config import STOCK_TICKERS

    pipeline = SentimentPipeline()
    count = 0
    for ticker in STOCK_TICKERS[:20]:  # Top 20 stocks
        try:
            brief = pipeline.generate_research_brief(ticker, days=30)
            if brief and "No recent news" not in brief:
                count += 1
                logger.info(f"Generated brief for {ticker}")
        except Exception as e:
            logger.warning(f"Brief generation failed for {ticker}: {e}")

    logger.info(f"Generated {count} research briefs")
    return count


# ── Full Pipeline ──────────────────────────────────────────────────────────

def run_full_pipeline():
    """Run the complete SentimentAlpha pipeline end-to-end."""
    start_time = time.time()

    logger.info("SentimentAlpha — Full Pipeline Starting")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"LLM Provider: {LLM_PROVIDER}")

    results = {}

    try:
        results["ingestion"] = step_ingest()
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        results["ingestion"] = {"error": str(e)}

    try:
        results["prices"] = step_prices()
    except Exception as e:
        logger.error(f"Price fetch failed: {e}")
        results["prices"] = {"error": str(e)}

    try:
        results["sentiment"] = step_sentiment()
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        results["sentiment"] = {"error": str(e)}

    try:
        results["factors"] = step_factors()
    except Exception as e:
        logger.error(f"Factor construction failed: {e}")
        results["factors"] = {"error": str(e)}

    try:
        results["backtests"] = step_backtest()
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        results["backtests"] = {"error": str(e)}

    try:
        results["ml"] = step_ml()
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        results["ml"] = {"error": str(e)}

    try:
        results["briefs"] = step_briefs()
    except Exception as e:
        logger.error(f"Brief generation failed: {e}")
        results["briefs"] = {"error": str(e)}

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"{'='*60}")

    return results


# ── Scheduled Runner ───────────────────────────────────────────────────────

def run_scheduled():
    """Run the pipeline on a schedule."""
    import schedule

    logger.info(f"Starting scheduled pipeline (every {SCRAPE_INTERVAL_MINUTES} min)")

    def job():
        try:
            logger.info("Scheduled run starting...")
            step_ingest()
            step_sentiment()
            step_factors()
            logger.info("Scheduled run complete")
        except Exception as e:
            logger.error(f"Scheduled run failed: {e}")

    schedule.every(SCRAPE_INTERVAL_MINUTES).minutes.do(job)

    # Run once immediately
    job()

    while True:
        schedule.run_pending()
        time.sleep(60)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SentimentAlpha — AI-Powered News Sentiment Trading Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full              # Run complete pipeline
  python main.py --ingest            # Scrape news only
  python main.py --prices            # Fetch prices only
  python main.py --sentiment         # Run sentiment analysis
  python main.py --factors           # Build factors
  python main.py --backtest          # Run backtests
  python main.py --ml                # Train ML models
  python main.py --briefs            # Generate research briefs
  python main.py --dashboard         # Launch Streamlit dashboard
  python main.py --schedule          # Run on schedule
        """
    )

    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--ingest", action="store_true", help="Scrape news")
    parser.add_argument("--prices", action="store_true", help="Fetch stock prices")
    parser.add_argument("--sentiment", action="store_true", help="Run sentiment analysis")
    parser.add_argument("--factors", action="store_true", help="Build factor scores")
    parser.add_argument("--backtest", action="store_true", help="Run backtests")
    parser.add_argument("--ml", action="store_true", help="Train ML models")
    parser.add_argument("--briefs", action="store_true", help="Generate research briefs")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--schedule", action="store_true", help="Run on schedule")

    args = parser.parse_args()

    if args.full:
        run_full_pipeline()
    elif args.ingest:
        step_ingest()
    elif args.prices:
        step_prices()
    elif args.sentiment:
        step_sentiment()
    elif args.factors:
        step_factors()
    elif args.backtest:
        step_backtest()
    elif args.ml:
        step_ml()
    elif args.briefs:
        step_briefs()
    elif args.dashboard:
        import subprocess
        subprocess.run(["streamlit", "run", "dashboard.py", "--server.port=8501"])
    elif args.schedule:
        run_scheduled()
    else:
        parser.print_help()
        print("\nQuick start: python main.py --full")


if __name__ == "__main__":
    main()
