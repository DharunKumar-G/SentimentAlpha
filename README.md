# SentimentAlpha — AI-Powered News Sentiment Trading Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen)

> An end-to-end system that uses LLMs to extract trading signals from financial news for Nifty 50 stocks — combining AI sentiment analysis with traditional quant factors for alpha generation.

---

## Core Features

### 1. News & Data Ingestion
- **RSS Feeds**: Scrapes Moneycontrol, Economic Times, LiveMint financial news
- **Reddit**: Pulls sentiment from r/IndianStreetBets, r/IndiaInvestments
- **Earnings Transcripts**: Fetches quarterly results from Screener.in
- **Price Data**: Historical OHLCV via yfinance for all Nifty 50 stocks

### 2. LLM-Powered Sentiment Analysis
- **Multi-provider**: Claude (Anthropic), GPT (OpenAI), or FinBERT (free, local)
- **Beyond basic sentiment**: Extracts specific signals — earnings beat, management change, sector tailwind, regulatory risk
- **Confidence scoring**: 0-100% confidence on each classification
- **Entity extraction**: Identifies which stocks and sectors each article affects
- **Research briefs**: Summarizes 20+ articles into a one-paragraph brief per stock

### 3. Sentiment Factor Construction
- Daily sentiment score per stock (rolling 7-day and 14-day averages)
- Combined with traditional factors (momentum, volatility, value)
- Correlation analysis: tests if sentiment predicts returns at different lags

### 4. Backtesting Engine
- **Sentiment-Only Strategy**: Long highest positive sentiment stocks
- **Momentum Strategy**: Traditional price momentum
- **Combined Strategy**: Z-score normalized multi-factor
- **Lag Analysis**: Tests predictive power at 1-day, 1-week, 1-month horizons
- Full performance metrics: Sharpe ratio, max drawdown, win rate

### 5. ML Models for Signal Combination
- **Random Forest** & **XGBoost** regressors for return prediction
- **Direction Classifier**: Predicts positive/negative return
- **Walk-forward validation**: No lookahead bias
- **Feature importance**: Which signal matters most

### 6. Real-Time Dashboard (Streamlit)
- Live news feed with color-coded sentiment tags
- Sentiment heatmap (treemap) across Nifty 50
- Alert system for sudden sentiment shifts
- "Ask the AI" chatbot — type a stock name, get a research summary
- Backtesting results viewer
- ML model predictions & feature importance charts

---

## Architecture

```
main.py                     ← Pipeline orchestrator (CLI entry point)
├── ingestion.py            ← News scraping (RSS, Reddit, Earnings)
├── price_data.py           ← yfinance price fetcher + factor calculator
├── sentiment_analyzer.py   ← LLM sentiment (Claude/OpenAI/FinBERT)
├── factor_builder.py       ← Daily sentiment scores + rolling averages
├── backtester.py           ← Strategy backtesting + lag analysis
├── ml_models.py            ← Random Forest / XGBoost training
├── dashboard.py            ← Streamlit real-time dashboard
├── database.py             ← SQLite storage layer
└── config.py               ← Configuration & Nifty 50 universe
```

---

## Quick Start

### 1. Clone & Install

```bash
cd sentimentbasedanalysisDS-1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

```bash
cp .env.example .env
# Edit .env with your API keys (or use FinBERT for free)
```

**LLM Provider Options:**
| Provider | Cost | Quality | Setup |
|----------|------|---------|-------|
| `finbert` | Free | Good | No API key needed |
| `openai` | Pay-per-use | Excellent | Set `OPENAI_API_KEY` |
| `anthropic` | Pay-per-use | Excellent | Set `ANTHROPIC_API_KEY` |

### 3. Run the Full Pipeline

```bash
# Run everything: ingest → analyze → factors → backtest → ML
python main.py --full
```

### 4. Launch Dashboard

```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

---

## CLI Reference

```bash
python main.py --full        # Run complete pipeline end-to-end
python main.py --ingest      # Scrape news from all sources
python main.py --prices      # Fetch latest stock prices
python main.py --sentiment   # Run LLM sentiment analysis
python main.py --factors     # Build daily sentiment factors
python main.py --backtest    # Run backtesting strategies
python main.py --ml          # Train ML models
python main.py --briefs      # Generate AI research briefs
python main.py --dashboard   # Launch Streamlit dashboard
python main.py --schedule    # Run pipeline on schedule (every 30 min)
```

Or run individual modules directly:

```bash
python ingestion.py            # Just news scraping
python price_data.py           # Just price fetching
python sentiment_analyzer.py   # Just sentiment analysis
python factor_builder.py       # Just factor construction
python backtester.py           # Just backtesting
python ml_models.py            # Just ML training
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Core** | Python 3.10+, pandas, numpy |
| **LLM** | Anthropic Claude API, OpenAI GPT API |
| **NLP** | HuggingFace Transformers (ProsusAI/FinBERT) |
| **ML** | scikit-learn, XGBoost |
| **Data** | yfinance, feedparser, beautifulsoup4 |
| **Visualization** | Plotly, Streamlit |
| **Database** | SQLite3 (WAL mode) |

---

## Database Schema

The SQLite database stores:
- **news_articles** — Raw headlines, summaries, full text
- **sentiment_scores** — LLM classifications per article
- **research_briefs** — AI-generated stock summaries
- **stock_prices** — Historical OHLCV data
- **daily_sentiment** — Aggregated daily scores per stock
- **factor_scores** — Traditional quant factors
- **backtest_results** — Strategy performance metrics
- **ml_predictions** — Model predictions vs actuals
- **alerts** — Sentiment shift notifications

---

## Trading Strategies

### Sentiment-Only
- Long top-N stocks with highest 7-day rolling sentiment
- Rebalance weekly
- Tests pure alpha from news sentiment

### Momentum-Only
- Long top-N stocks with highest 3-month price momentum
- Traditional quant factor baseline

### Combined (Multi-Factor)
- Z-score normalized combination:
  - 40% Sentiment
  - 35% Momentum
  - 25% Value (inverse PE)
- Tests whether sentiment adds incremental alpha

### ML-Enhanced
- Random Forest / XGBoost predict next-month returns
- Features: sentiment scores + traditional factors
- Walk-forward validated to prevent overfitting

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and open a Pull Request

---

*Built for Wright Research's AI-first approach to quantitative investing.*
