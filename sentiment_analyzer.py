"""
SentimentAlpha — LLM-Powered Sentiment Analysis
=================================================
Multi-provider sentiment engine supporting:
  • Anthropic Claude API
  • OpenAI GPT API
  • HuggingFace FinBERT (free, local)

Extracts: sentiment, confidence, signal type, affected entities,
and generates research briefs.
"""

import json
import logging
import re
from typing import Optional

from config import (
    LLM_PROVIDER, ANTHROPIC_API_KEY, OPENAI_API_KEY,
    ANTHROPIC_MODEL, OPENAI_MODEL,
    NIFTY50_STOCKS, STOCK_NAMES, SIGNAL_TYPES, SENTIMENT_LABELS,
)
from database import db

logger = logging.getLogger("sentimentalpha.sentiment")


# ── Prompt Templates ───────────────────────────────────────────────────────

SENTIMENT_SYSTEM_PROMPT = """You are a financial sentiment analysis expert specializing in Indian equity markets (Nifty 50).

Analyze the given news headline/article and return a JSON object with these fields:
{
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": 0.0 to 1.0,
  "signal_type": one of [""" + ", ".join(f'"{s}"' for s in SIGNAL_TYPES) + """],
  "affected_stocks": ["TICKER.NS", ...],
  "affected_sectors": ["sector_name", ...],
  "reasoning": "one sentence explanation"
}

Rules:
- Only output valid JSON, no markdown or extra text
- Use NSE ticker format (e.g., "RELIANCE.NS", "TCS.NS")
- Confidence should reflect how clearly the news maps to a directional signal
- If the article is about the broad market, list the most affected stocks/sectors
- signal_type should capture the PRIMARY signal from the article

Known Nifty 50 tickers and sectors:
""" + "\n".join(f"  {k}: {v['name']} ({v['sector']})" for k, v in list(NIFTY50_STOCKS.items())[:20]) + "\n  ... and more"

RESEARCH_BRIEF_PROMPT = """You are a senior equity research analyst covering Indian markets.

Given the following {count} recent news articles about {stock_name} ({ticker}), write a single concise research brief paragraph (150-200 words) that:
1. Summarizes the key developments
2. Identifies the dominant sentiment trend
3. Highlights key risks and catalysts
4. Gives a short-term outlook

Articles:
{articles}

Write ONLY the research brief paragraph. No headers, no bullet points."""


# ── Base Analyzer ──────────────────────────────────────────────────────────

class BaseSentimentAnalyzer:
    """Base class for sentiment analyzers."""

    def analyze_article(self, title: str, summary: str = "",
                        full_text: str = "") -> dict:
        raise NotImplementedError

    def generate_brief(self, ticker: str, articles: list) -> str:
        raise NotImplementedError

    def _parse_response(self, response_text: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = response_text.strip()
        # Remove markdown code blocks
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM response: {text[:200]}")
                    return self._default_result()
            else:
                return self._default_result()

        # Validate and normalize
        return {
            "sentiment": data.get("sentiment", "neutral") if data.get("sentiment") in SENTIMENT_LABELS else "neutral",
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            "signal_type": data.get("signal_type", "neutral") if data.get("signal_type") in SIGNAL_TYPES else "neutral",
            "affected_stocks": data.get("affected_stocks", []),
            "affected_sectors": data.get("affected_sectors", []),
            "reasoning": data.get("reasoning", ""),
        }

    @staticmethod
    def _default_result() -> dict:
        return {
            "sentiment": "neutral",
            "confidence": 0.3,
            "signal_type": "neutral",
            "affected_stocks": [],
            "affected_sectors": [],
            "reasoning": "Failed to parse LLM response",
        }


# ── Anthropic Claude Analyzer ─────────────────────────────────────────────

class AnthropicAnalyzer(BaseSentimentAnalyzer):
    """Sentiment analysis via Anthropic Claude API."""

    def __init__(self):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        except ImportError:
            raise ImportError("pip install anthropic")

    def analyze_article(self, title: str, summary: str = "",
                        full_text: str = "") -> dict:
        content = f"Headline: {title}"
        if summary:
            content += f"\nSummary: {summary}"
        if full_text:
            content += f"\nFull text: {full_text[:2000]}"

        try:
            message = self.client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=500,
                system=SENTIMENT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            response_text = message.content[0].text
            result = self._parse_response(response_text)
            result["raw_response"] = response_text
            return result
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return self._default_result()

    def generate_brief(self, ticker: str, articles: list) -> str:
        stock_info = NIFTY50_STOCKS.get(ticker, {})
        stock_name = stock_info.get("name", ticker)

        articles_text = "\n\n".join(
            f"[{i+1}] {a.get('title', 'No title')}\n{a.get('summary', '')}"
            for i, a in enumerate(articles[:20])
        )

        prompt = RESEARCH_BRIEF_PROMPT.format(
            count=len(articles), stock_name=stock_name,
            ticker=ticker, articles=articles_text,
        )

        try:
            message = self.client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic brief generation error: {e}")
            return f"Unable to generate brief for {stock_name}."


# ── OpenAI Analyzer ───────────────────────────────────────────────────────

class OpenAIAnalyzer(BaseSentimentAnalyzer):
    """Sentiment analysis via OpenAI GPT API."""

    def __init__(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            raise ImportError("pip install openai")

    def analyze_article(self, title: str, summary: str = "",
                        full_text: str = "") -> dict:
        content = f"Headline: {title}"
        if summary:
            content += f"\nSummary: {summary}"
        if full_text:
            content += f"\nFull text: {full_text[:2000]}"

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.1,
            )
            response_text = response.choices[0].message.content
            result = self._parse_response(response_text)
            result["raw_response"] = response_text
            return result
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._default_result()

    def generate_brief(self, ticker: str, articles: list) -> str:
        stock_info = NIFTY50_STOCKS.get(ticker, {})
        stock_name = stock_info.get("name", ticker)

        articles_text = "\n\n".join(
            f"[{i+1}] {a.get('title', 'No title')}\n{a.get('summary', '')}"
            for i, a in enumerate(articles[:20])
        )

        prompt = RESEARCH_BRIEF_PROMPT.format(
            count=len(articles), stock_name=stock_name,
            ticker=ticker, articles=articles_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI brief generation error: {e}")
            return f"Unable to generate brief for {stock_name}."


# ── FinBERT Analyzer (Free, Local) ────────────────────────────────────────

class FinBERTAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analysis using HuggingFace FinBERT — free, no API key needed.
    Uses ProsusAI/finbert for financial sentiment classification.
    """

    def __init__(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            raise ImportError("pip install transformers torch")

        logger.info("Loading FinBERT model (first time may download ~440MB)...")
        model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self._label_map = {0: "bullish", 1: "bearish", 2: "neutral"}
        logger.info("FinBERT model loaded successfully")

    def analyze_article(self, title: str, summary: str = "",
                        full_text: str = "") -> dict:
        import torch

        text = title
        if summary:
            text += ". " + summary[:500]

        try:
            inputs = self.tokenizer(text, return_tensors="pt",
                                    truncation=True, max_length=512, padding=True)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]

            # FinBERT labels: positive(0), negative(1), neutral(2)
            pos_score = float(probs[0])
            neg_score = float(probs[1])
            neu_score = float(probs[2])

            if pos_score > neg_score and pos_score > neu_score:
                sentiment = "bullish"
                confidence = pos_score
            elif neg_score > pos_score and neg_score > neu_score:
                sentiment = "bearish"
                confidence = neg_score
            else:
                sentiment = "neutral"
                confidence = neu_score

            # Simple entity extraction from text
            affected_stocks = self._extract_stocks(text)
            affected_sectors = self._extract_sectors(text)
            signal_type = self._infer_signal_type(text, sentiment)

            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 3),
                "signal_type": signal_type,
                "affected_stocks": affected_stocks,
                "affected_sectors": affected_sectors,
                "reasoning": f"FinBERT scores: pos={pos_score:.3f}, neg={neg_score:.3f}, neu={neu_score:.3f}",
                "raw_response": json.dumps({
                    "positive": pos_score, "negative": neg_score, "neutral": neu_score
                }),
            }
        except Exception as e:
            logger.error(f"FinBERT analysis error: {e}")
            return self._default_result()

    def generate_brief(self, ticker: str, articles: list) -> str:
        """Generate a simple template-based brief (no LLM API available)."""
        stock_info = NIFTY50_STOCKS.get(ticker, {})
        stock_name = stock_info.get("name", ticker)

        if not articles:
            return f"No recent news available for {stock_name}."

        sentiments = []
        for a in articles:
            result = self.analyze_article(a.get("title", ""), a.get("summary", ""))
            sentiments.append(result["sentiment"])

        bullish_pct = sentiments.count("bullish") / len(sentiments) * 100
        bearish_pct = sentiments.count("bearish") / len(sentiments) * 100

        if bullish_pct > 60:
            outlook = "predominantly positive"
        elif bearish_pct > 60:
            outlook = "predominantly negative"
        else:
            outlook = "mixed"

        titles = [a.get("title", "") for a in articles[:5]]
        key_headlines = "; ".join(titles)

        return (
            f"{stock_name} ({ticker}) — Based on {len(articles)} recent articles, "
            f"sentiment is {outlook} with {bullish_pct:.0f}% bullish and "
            f"{bearish_pct:.0f}% bearish signals. "
            f"Key headlines: {key_headlines}. "
            f"Investors should monitor upcoming catalysts and sector trends."
        )

    @staticmethod
    def _extract_stocks(text: str) -> list:
        """Extract mentioned Nifty 50 stocks from text."""
        text_lower = text.lower()
        found = []
        for name_lower, ticker in STOCK_NAMES.items():
            if name_lower in text_lower:
                found.append(ticker)
        # Also check ticker symbols
        for ticker in NIFTY50_STOCKS:
            symbol = ticker.replace(".NS", "").lower()
            if symbol in text_lower:
                if ticker not in found:
                    found.append(ticker)
        return found

    @staticmethod
    def _extract_sectors(text: str) -> list:
        """Extract mentioned sectors from text."""
        text_lower = text.lower()
        sector_keywords = {
            "Banking": ["bank", "banking", "npa", "credit"],
            "IT": ["it ", "software", "tech", "digital"],
            "Pharma": ["pharma", "drug", "healthcare", "hospital"],
            "FMCG": ["fmcg", "consumer", "food", "beverage"],
            "Automobile": ["auto", "car", "vehicle", "ev "],
            "Energy": ["oil", "gas", "energy", "petrol", "refiner"],
            "Metals": ["steel", "metal", "aluminum", "copper", "mining"],
            "Infrastructure": ["infra", "construction", "cement", "road"],
            "Telecom": ["telecom", "5g", "spectrum", "mobile"],
            "Finance": ["nbfc", "finance", "insurance", "mutual fund"],
        }
        found = []
        for sector, keywords in sector_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found.append(sector)
        return found

    @staticmethod
    def _infer_signal_type(text: str, sentiment: str) -> str:
        """Rule-based signal type inference."""
        text_lower = text.lower()
        signal_keywords = {
            "earnings_beat": ["beat", "strong results", "profit up", "revenue beat", "record profit"],
            "earnings_miss": ["miss", "weak results", "profit down", "revenue miss", "loss"],
            "management_change": ["ceo", "appoint", "resign", "management change", "new chairman"],
            "sector_tailwind": ["sector rally", "sector up", "industry growth", "demand surge"],
            "sector_headwind": ["sector down", "industry slowdown", "demand weak"],
            "regulatory_risk": ["sebi", "penalty", "regulatory", "compliance", "ban"],
            "regulatory_positive": ["approval", "license", "clearance", "reform"],
            "expansion": ["expand", "new plant", "acquisition", "merger", "capacity"],
            "cost_cutting": ["cost cut", "restructur", "layoff", "efficiency"],
            "upgrade": ["upgrade", "target raised", "buy rating", "outperform"],
            "downgrade": ["downgrade", "target cut", "sell rating", "underperform"],
        }

        for signal, keywords in signal_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return signal

        if sentiment == "bullish":
            return "general_positive"
        elif sentiment == "bearish":
            return "general_negative"
        return "neutral"


# ── Analyzer Factory ───────────────────────────────────────────────────────

def get_analyzer(provider: str = None) -> BaseSentimentAnalyzer:
    """Factory function to get the configured sentiment analyzer."""
    provider = provider or LLM_PROVIDER

    if provider == "anthropic":
        return AnthropicAnalyzer()
    elif provider == "openai":
        return OpenAIAnalyzer()
    elif provider == "finbert":
        return FinBERTAnalyzer()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'anthropic', 'openai', or 'finbert'.")


# ── Sentiment Pipeline ─────────────────────────────────────────────────────

class SentimentPipeline:
    """Orchestrates sentiment analysis across all unanalyzed articles."""

    def __init__(self, provider: str = None):
        self.provider = provider or LLM_PROVIDER
        self.analyzer = get_analyzer(self.provider)

    def analyze_pending_articles(self, limit: int = 50) -> int:
        """Analyze articles that haven't been processed yet."""
        articles = db.get_unanalyzed_articles(self.provider, limit)
        if not articles:
            logger.info("No pending articles to analyze")
            return 0

        logger.info(f"Analyzing {len(articles)} articles with {self.provider}...")
        count = 0

        for article in articles:
            try:
                result = self.analyzer.analyze_article(
                    title=article["title"],
                    summary=article.get("summary", ""),
                    full_text=article.get("full_text", ""),
                )

                db.insert_sentiment(
                    article_id=article["id"],
                    provider=self.provider,
                    sentiment=result["sentiment"],
                    confidence=result["confidence"],
                    signal_type=result.get("signal_type"),
                    affected_stocks=result.get("affected_stocks", []),
                    affected_sectors=result.get("affected_sectors", []),
                    raw_response=result.get("raw_response"),
                )
                count += 1

                if count % 10 == 0:
                    logger.info(f"Analyzed {count}/{len(articles)} articles")

            except Exception as e:
                logger.error(f"Failed to analyze article {article['id']}: {e}")

        logger.info(f"Sentiment analysis complete: {count}/{len(articles)} articles processed")
        return count

    def generate_research_brief(self, ticker: str, days: int = 30) -> str:
        """Generate a research brief for a stock from recent news."""
        articles = db.get_articles_for_stock(ticker, days)
        if not articles:
            stock_name = NIFTY50_STOCKS.get(ticker, {}).get("name", ticker)
            return f"No recent news found for {stock_name} ({ticker}) in the last {days} days."

        brief = self.analyzer.generate_brief(ticker, articles)

        # Store the brief
        article_ids = [a["id"] for a in articles]
        db.insert_brief(ticker, brief, article_ids)

        return brief

    def get_stock_summary(self, ticker: str) -> dict:
        """Get a comprehensive summary for a stock."""
        articles = db.get_articles_for_stock(ticker, days=30)
        brief = db.get_latest_brief(ticker)

        sentiments = [a.get("sentiment") for a in articles if a.get("sentiment")]
        total = len(sentiments) or 1

        return {
            "ticker": ticker,
            "name": NIFTY50_STOCKS.get(ticker, {}).get("name", ticker),
            "sector": NIFTY50_STOCKS.get(ticker, {}).get("sector", "Unknown"),
            "article_count": len(articles),
            "bullish_pct": round(sentiments.count("bullish") / total * 100, 1),
            "bearish_pct": round(sentiments.count("bearish") / total * 100, 1),
            "neutral_pct": round(sentiments.count("neutral") / total * 100, 1),
            "latest_brief": brief.get("brief") if brief else None,
            "recent_articles": articles[:5],
        }


# ── CLI Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print(f"Using LLM provider: {LLM_PROVIDER}")
    pipeline = SentimentPipeline()

    # Analyze pending articles
    count = pipeline.analyze_pending_articles(limit=50)
    print(f"Analyzed {count} articles")

    # Generate a sample research brief
    sample_ticker = "RELIANCE.NS"
    print(f"\nGenerating research brief for {sample_ticker}...")
    brief = pipeline.generate_research_brief(sample_ticker)
    print(f"\nResearch Brief:\n{brief}")
