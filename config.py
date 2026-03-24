"""
SentimentAlpha — AI-Powered News Sentiment Trading Engine
=========================================================
Global configuration constants and Nifty 50 stock universe.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Streamlit Cloud secrets support ────────────────────────────────────────
def _secret(key: str, default: str = "") -> str:
    """Read from st.secrets (Streamlit Cloud) with fallback to env vars."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
LOGS_DIR = PROJECT_ROOT / "logs"

for _dir in [DATA_DIR, MODELS_DIR, LOGS_DIR, DATA_DIR / "cache"]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Database ───────────────────────────────────────────────────────────────
DATABASE_PATH = PROJECT_ROOT / _secret("DATABASE_PATH", "data/sentimentalpha.db")

# ── LLM ────────────────────────────────────────────────────────────────────
LLM_PROVIDER = _secret("LLM_PROVIDER", "finbert")  # anthropic | openai | finbert
ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = _secret("OPENAI_API_KEY", "")
ANTHROPIC_MODEL = _secret("ANTHROPIC_MODEL", "claude-sonnet-4-5")
OPENAI_MODEL = _secret("OPENAI_MODEL", "gpt-4o-mini")

# ── Scraping ───────────────────────────────────────────────────────────────
SCRAPE_INTERVAL_MINUTES = int(os.getenv("SCRAPE_INTERVAL_MINUTES", "30"))

# ── Dashboard ──────────────────────────────────────────────────────────────
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))

# ── Nifty 50 Universe ─────────────────────────────────────────────────────
NIFTY50_STOCKS = {
    "RELIANCE.NS":  {"name": "Reliance Industries", "sector": "Energy"},
    "TCS.NS":       {"name": "Tata Consultancy Services", "sector": "IT"},
    "HDFCBANK.NS":  {"name": "HDFC Bank", "sector": "Banking"},
    "INFY.NS":      {"name": "Infosys", "sector": "IT"},
    "ICICIBANK.NS": {"name": "ICICI Bank", "sector": "Banking"},
    "HINDUNILVR.NS":{"name": "Hindustan Unilever", "sector": "FMCG"},
    "ITC.NS":       {"name": "ITC", "sector": "FMCG"},
    "SBIN.NS":      {"name": "State Bank of India", "sector": "Banking"},
    "BHARTIARTL.NS":{"name": "Bharti Airtel", "sector": "Telecom"},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra Bank", "sector": "Banking"},
    "LT.NS":        {"name": "Larsen & Toubro", "sector": "Infrastructure"},
    "AXISBANK.NS":  {"name": "Axis Bank", "sector": "Banking"},
    "ASIANPAINT.NS":{"name": "Asian Paints", "sector": "Consumer"},
    "MARUTI.NS":    {"name": "Maruti Suzuki", "sector": "Automobile"},
    "HCLTECH.NS":   {"name": "HCL Technologies", "sector": "IT"},
    "SUNPHARMA.NS": {"name": "Sun Pharma", "sector": "Pharma"},
    "TITAN.NS":     {"name": "Titan Company", "sector": "Consumer"},
    "BAJFINANCE.NS":{"name": "Bajaj Finance", "sector": "Finance"},
    "WIPRO.NS":     {"name": "Wipro", "sector": "IT"},
    "ULTRACEMCO.NS":{"name": "UltraTech Cement", "sector": "Cement"},
    "NESTLEIND.NS": {"name": "Nestle India", "sector": "FMCG"},
    "NTPC.NS":      {"name": "NTPC", "sector": "Power"},
    "TATAMOTORS.NS":{"name": "Tata Motors", "sector": "Automobile"},
    "M&M.NS":       {"name": "Mahindra & Mahindra", "sector": "Automobile"},
    "POWERGRID.NS": {"name": "Power Grid Corp", "sector": "Power"},
    "ONGC.NS":      {"name": "ONGC", "sector": "Energy"},
    "TATASTEEL.NS": {"name": "Tata Steel", "sector": "Metals"},
    "JSWSTEEL.NS":  {"name": "JSW Steel", "sector": "Metals"},
    "ADANIENT.NS":  {"name": "Adani Enterprises", "sector": "Conglomerate"},
    "ADANIPORTS.NS":{"name": "Adani Ports", "sector": "Infrastructure"},
    "TECHM.NS":     {"name": "Tech Mahindra", "sector": "IT"},
    "BAJAJFINSV.NS":{"name": "Bajaj Finserv", "sector": "Finance"},
    "COALINDIA.NS": {"name": "Coal India", "sector": "Mining"},
    "HDFCLIFE.NS":  {"name": "HDFC Life", "sector": "Insurance"},
    "SBILIFE.NS":   {"name": "SBI Life Insurance", "sector": "Insurance"},
    "GRASIM.NS":    {"name": "Grasim Industries", "sector": "Cement"},
    "DIVISLAB.NS":  {"name": "Divi's Laboratories", "sector": "Pharma"},
    "DRREDDY.NS":   {"name": "Dr Reddy's Labs", "sector": "Pharma"},
    "CIPLA.NS":     {"name": "Cipla", "sector": "Pharma"},
    "BRITANNIA.NS": {"name": "Britannia", "sector": "FMCG"},
    "EICHERMOT.NS": {"name": "Eicher Motors", "sector": "Automobile"},
    "APOLLOHOSP.NS":{"name": "Apollo Hospitals", "sector": "Healthcare"},
    "INDUSINDBK.NS":{"name": "IndusInd Bank", "sector": "Banking"},
    "HEROMOTOCO.NS":{"name": "Hero MotoCorp", "sector": "Automobile"},
    "BPCL.NS":      {"name": "BPCL", "sector": "Energy"},
    "TATACONSUM.NS":{"name": "Tata Consumer", "sector": "FMCG"},
    "HINDALCO.NS":  {"name": "Hindalco", "sector": "Metals"},
    "BAJAJ-AUTO.NS":{"name": "Bajaj Auto", "sector": "Automobile"},
    "WIPRO.NS":     {"name": "Wipro", "sector": "IT"},
    "UPL.NS":       {"name": "UPL", "sector": "Chemicals"},
}

# Helper maps
STOCK_NAMES = {v["name"].lower(): k for k, v in NIFTY50_STOCKS.items()}
STOCK_TICKERS = list(NIFTY50_STOCKS.keys())
SECTORS = list(set(v["sector"] for v in NIFTY50_STOCKS.values()))

# ── RSS Feed Sources ───────────────────────────────────────────────────────
RSS_FEEDS = {
    "moneycontrol": [
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://www.moneycontrol.com/rss/marketreports.xml",
        "https://www.moneycontrol.com/rss/stocksnews.xml",
    ],
    "economic_times": [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    ],
    "livemint": [
        "https://www.livemint.com/rss/markets",
        "https://www.livemint.com/rss/companies",
    ],
}

# ── Sentiment Labels ──────────────────────────────────────────────────────
SENTIMENT_LABELS = ["bullish", "bearish", "neutral"]
SIGNAL_TYPES = [
    "earnings_beat", "earnings_miss",
    "management_change", "sector_tailwind", "sector_headwind",
    "regulatory_risk", "regulatory_positive",
    "expansion", "cost_cutting",
    "upgrade", "downgrade",
    "general_positive", "general_negative", "neutral",
]
