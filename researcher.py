#!/usr/bin/env python3
"""
Stock Ticker Research & Evaluation — Agentic Pipeline
======================================================
Multi-agent CLI tool that researches a stock ticker and produces
a structured research report. Uses yfinance (no API key required).

Usage:
    python stock_research.py NVDA
    python stock_research.py AAPL --peers MSFT GOOGL AMZN
    python stock_research.py TSLA --output report.md
    python stock_research.py NVDA --json results.json

Agents:
    1. Company Profile    — identity, sector, description
    2. Price & Volume     — current price, 52-week, volume trends
    3. Fundamentals       — valuation, profitability, balance sheet
    4. Technical Analysis — SMA, RSI, MACD, Bollinger Bands
    5. Risk Assessment    — beta, volatility, max drawdown
    6. Peer Comparison    — benchmark against sector peers
    7. SEC EDGAR Filings  — 10-K, 10-Q, 8-K filings + XBRL financials (free, no key)
    8. Finnhub Sentiment  — analyst recs, insider activity, price targets (optional)
    9. EODHD Sentiment    — news sentiment scores, trend analysis (optional)
   10. LLM Synthesis      — AI-generated narrative analysis (optional)
   11. Report Assembly    — synthesize into structured research report

LLM Synthesis (Agent 7) supports OpenAI and Anthropic:
    python stock_research.py NVDA --llm openai --llm-key sk-...
    python stock_research.py NVDA --llm anthropic --llm-key sk-ant-...

    Or set environment variables (or use a .env file):
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export FINNHUB_API_KEY=your_key
    export EODHD_API_KEY=your_key
    export SEC_EMAIL=you@email.com
    python stock_research.py NVDA --llm openai

    Or create a .env file in the same directory:
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    FINNHUB_API_KEY=your_key
    EODHD_API_KEY=your_key
    SEC_EMAIL=you@email.com
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Any

import requests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# .ENV FILE LOADER — load API keys and config from .env file
# ---------------------------------------------------------------------------

from pathlib import Path

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and "=" in _line and not _line.startswith("#"):
            _key, _val = _line.split("=", 1)
            _key = _key.strip()
            _val = _val.strip().strip('"').strip("'")
            os.environ.setdefault(_key, _val)

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)


# ---------------------------------------------------------------------------
# SHARED STATE
# ---------------------------------------------------------------------------

def create_state(ticker: str, peers: list[str] | None = None) -> dict:
    """Initialize the shared state dict that flows through all agents."""
    return {
        "ticker": ticker.upper(),
        "peers": [p.upper() for p in (peers or [])],
        "timestamp": datetime.now().isoformat(),
        "profile": {},
        "price": {},
        "fundamentals": {},
        "technical": {},
        "risk": {},
        "peer_comparison": {},
        "sec_filings": {},
        "synthesis": {},
        "eodhd_sentiment": {},
        "agent_log": [],
        "errors": [],
    }


def log_agent(state: dict, agent: str, status: str, detail: str = ""):
    """Log agent execution to state."""
    state["agent_log"].append({
        "agent": agent,
        "status": status,
        "detail": detail,
        "time": datetime.now().isoformat(),
    })


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def safe_get(info: dict, key: str, default=None):
    """Safely get a value from yfinance info dict."""
    val = info.get(key, default)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


def fmt_large_number(n) -> str:
    """Format large numbers: 1.5T, 230B, 45M."""
    if n is None:
        return "N/A"
    if isinstance(n, str):
        return n
    if n >= 1e12:
        return f"${n/1e12:.2f}T"
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    if n >= 1e6:
        return f"${n/1e6:.1f}M"
    return f"${n:,.0f}"


def fmt_pct(val) -> str:
    """Format as percentage."""
    if val is None:
        return "N/A"
    return f"{val:.2%}" if isinstance(val, float) else str(val)


def fmt_price(val) -> str:
    if val is None:
        return "N/A"
    return f"${val:.2f}"


def compute_sma(closes: list[float], period: int) -> list[float | None]:
    """Simple Moving Average."""
    result = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        result[i] = sum(closes[i - period + 1:i + 1]) / period
    return result


def compute_ema(closes: list[float], period: int) -> list[float | None]:
    """Exponential Moving Average."""
    result = [None] * len(closes)
    k = 2 / (period + 1)
    result[period - 1] = sum(closes[:period]) / period
    for i in range(period, len(closes)):
        result[i] = closes[i] * k + result[i - 1] * (1 - k)
    return result


def compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """RSI from closing prices."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(closes: list[float]) -> dict:
    """MACD (12, 26, 9)."""
    if len(closes) < 35:
        return {"line": None, "signal": None, "histogram": None}
    ema12 = compute_ema(closes, 12)
    ema26 = compute_ema(closes, 26)

    macd_line = []
    for a, b in zip(ema12, ema26):
        if a is not None and b is not None:
            macd_line.append(a - b)
        else:
            macd_line.append(None)

    valid = [x for x in macd_line if x is not None]
    if len(valid) < 9:
        return {"line": valid[-1] if valid else None, "signal": None, "histogram": None}

    signal = compute_ema(valid, 9)
    latest_macd = valid[-1]
    latest_signal = signal[-1]
    histogram = latest_macd - latest_signal if latest_signal else None

    return {"line": latest_macd, "signal": latest_signal, "histogram": histogram}


def compute_bollinger(closes: list[float], period: int = 20) -> dict:
    """Bollinger Bands (20, 2)."""
    if len(closes) < period:
        return {"upper": None, "middle": None, "lower": None, "pct_b": None}
    recent = closes[-period:]
    middle = sum(recent) / period
    std = (sum((x - middle) ** 2 for x in recent) / period) ** 0.5
    upper = middle + 2 * std
    lower = middle - 2 * std
    pct_b = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
    return {"upper": upper, "middle": middle, "lower": lower, "pct_b": pct_b}


# ---------------------------------------------------------------------------
# AGENT 1: COMPANY PROFILE
# ---------------------------------------------------------------------------

def agent_company_profile(state: dict) -> dict:
    """Fetch company identity, sector, description."""
    ticker = state["ticker"]
    print(f"  [1/11] Company profile...", end=" ", flush=True)

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            # Try fast_info as fallback
            pass

        state["profile"] = {
            "name": safe_get(info, "longName", ticker),
            "sector": safe_get(info, "sector", "N/A"),
            "industry": safe_get(info, "industry", "N/A"),
            "country": safe_get(info, "country", "N/A"),
            "exchange": safe_get(info, "exchange", "N/A"),
            "currency": safe_get(info, "currency", "USD"),
            "employees": safe_get(info, "fullTimeEmployees"),
            "website": safe_get(info, "website"),
            "description": safe_get(info, "longBusinessSummary", ""),
        }

        # Auto-detect peers if none provided
        if not state["peers"]:
            state["peers"] = _detect_peers(info, ticker)

        state["_yf_info"] = info  # cache for other agents
        state["_yf_stock"] = stock
        log_agent(state, "company_profile", "ok")
        print("OK")

    except Exception as e:
        state["errors"].append(f"Profile: {e}")
        log_agent(state, "company_profile", "error", str(e))
        print(f"ERROR: {e}")

    return state


def _detect_peers(info: dict, ticker: str) -> list[str]:
    """Auto-detect peer tickers from sector/industry."""
    sector_peers = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "CRM", "ORCL"],
        "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT"],
        "Financial Services": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW"],
        "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW"],
        "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"],
        "Industrials": ["CAT", "GE", "HON", "UNP", "RTX", "BA", "LMT", "DE"],
    }
    sector = safe_get(info, "sector", "")
    peers = sector_peers.get(sector, [])
    return [p for p in peers if p != ticker.upper()][:4]


# ---------------------------------------------------------------------------
# AGENT 2: PRICE & VOLUME
# ---------------------------------------------------------------------------

def agent_price_volume(state: dict) -> dict:
    """Current price, 52-week range, volume analysis."""
    ticker = state["ticker"]
    print(f"  [2/11] Price & volume...", end=" ", flush=True)

    try:
        info = state.get("_yf_info", {})
        stock = state.get("_yf_stock") or yf.Ticker(ticker)

        # Get historical data for analysis
        hist = stock.history(period="1y")
        if hist.empty:
            raise ValueError(f"No historical data for {ticker}")

        closes = hist["Close"].tolist()
        volumes = hist["Volume"].tolist()
        dates = hist.index.tolist()

        current_price = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else current_price

        # 52-week high/low
        high_52w = max(closes)
        low_52w = min(closes)
        high_52w_date = dates[closes.index(high_52w)].strftime("%Y-%m-%d")
        low_52w_date = dates[closes.index(low_52w)].strftime("%Y-%m-%d")

        # Volume analysis
        avg_vol_30 = sum(volumes[-30:]) / min(30, len(volumes))
        avg_vol_90 = sum(volumes[-90:]) / min(90, len(volumes))
        latest_vol = volumes[-1]
        vol_vs_avg = latest_vol / avg_vol_30 if avg_vol_30 > 0 else 1.0

        # Performance periods
        def pct_change(start, end):
            return (end - start) / start if start != 0 else 0

        perf_1d = pct_change(prev_close, current_price)
        perf_5d = pct_change(closes[-6], current_price) if len(closes) >= 6 else None
        perf_1m = pct_change(closes[-22], current_price) if len(closes) >= 22 else None
        perf_3m = pct_change(closes[-66], current_price) if len(closes) >= 66 else None
        perf_6m = pct_change(closes[-132], current_price) if len(closes) >= 132 else None
        perf_1y = pct_change(closes[0], current_price)

        # Where in 52-week range
        range_position = (current_price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5

        state["price"] = {
            "current": current_price,
            "prev_close": prev_close,
            "change_1d": perf_1d,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "high_52w_date": high_52w_date,
            "low_52w_date": low_52w_date,
            "range_position": range_position,
            "performance": {
                "1d": perf_1d,
                "5d": perf_5d,
                "1m": perf_1m,
                "3m": perf_3m,
                "6m": perf_6m,
                "1y": perf_1y,
            },
            "volume_latest": latest_vol,
            "volume_avg_30d": avg_vol_30,
            "volume_avg_90d": avg_vol_90,
            "volume_vs_avg": vol_vs_avg,
        }

        state["_closes"] = closes
        state["_volumes"] = volumes
        state["_hist"] = hist
        log_agent(state, "price_volume", "ok")
        print("OK")

    except Exception as e:
        state["errors"].append(f"Price: {e}")
        log_agent(state, "price_volume", "error", str(e))
        print(f"ERROR: {e}")

    return state


# ---------------------------------------------------------------------------
# AGENT 3: FUNDAMENTALS
# ---------------------------------------------------------------------------

def agent_fundamentals(state: dict) -> dict:
    """Valuation, profitability, growth, balance sheet."""
    ticker = state["ticker"]
    print(f"  [3/11] Fundamentals...", end=" ", flush=True)

    try:
        info = state.get("_yf_info", {})
        stock = state.get("_yf_stock") or yf.Ticker(ticker)

        # Valuation
        valuation = {
            "market_cap": safe_get(info, "marketCap"),
            "enterprise_value": safe_get(info, "enterpriseValue"),
            "pe_trailing": safe_get(info, "trailingPE"),
            "pe_forward": safe_get(info, "forwardPE"),
            "peg_ratio": safe_get(info, "pegRatio"),
            "ps_ratio": safe_get(info, "priceToSalesTrailing12Months"),
            "pb_ratio": safe_get(info, "priceToBook"),
            "ev_ebitda": safe_get(info, "enterpriseToEbitda"),
            "ev_revenue": safe_get(info, "enterpriseToRevenue"),
        }

        # Profitability
        profitability = {
            "gross_margin": safe_get(info, "grossMargins"),
            "operating_margin": safe_get(info, "operatingMargins"),
            "profit_margin": safe_get(info, "profitMargins"),
            "roe": safe_get(info, "returnOnEquity"),
            "roa": safe_get(info, "returnOnAssets"),
            "eps_trailing": safe_get(info, "trailingEps"),
            "eps_forward": safe_get(info, "forwardEps"),
        }

        # Growth
        growth = {
            "revenue_growth": safe_get(info, "revenueGrowth"),
            "earnings_growth": safe_get(info, "earningsGrowth"),
            "earnings_quarterly_growth": safe_get(info, "earningsQuarterlyGrowth"),
        }

        # Balance sheet
        balance_sheet = {
            "total_cash": safe_get(info, "totalCash"),
            "total_debt": safe_get(info, "totalDebt"),
            "debt_to_equity": safe_get(info, "debtToEquity"),
            "current_ratio": safe_get(info, "currentRatio"),
            "book_value": safe_get(info, "bookValue"),
            "free_cash_flow": safe_get(info, "freeCashflow"),
            "operating_cash_flow": safe_get(info, "operatingCashflow"),
        }

        # Dividends
        dividends = {
            "dividend_yield": safe_get(info, "dividendYield"),
            "dividend_rate": safe_get(info, "dividendRate"),
            "payout_ratio": safe_get(info, "payoutRatio"),
            "ex_dividend_date": safe_get(info, "exDividendDate"),
        }

        # Valuation signals
        signals = []
        pe = valuation["pe_trailing"]
        if pe:
            if pe < 15:
                signals.append("LOW_PE (potentially undervalued)")
            elif pe > 40:
                signals.append("HIGH_PE (growth priced in or overvalued)")

        peg = valuation["peg_ratio"]
        if peg:
            if peg < 1:
                signals.append("PEG_BELOW_1 (growth at reasonable price)")
            elif peg > 2:
                signals.append("PEG_ABOVE_2 (expensive relative to growth)")

        de = balance_sheet["debt_to_equity"]
        if de:
            if de > 200:
                signals.append("HIGH_LEVERAGE (debt/equity > 2x)")
            elif de < 50:
                signals.append("LOW_LEVERAGE (conservative balance sheet)")

        margin = profitability["profit_margin"]
        if margin:
            if margin > 0.20:
                signals.append("STRONG_MARGINS (>20% net)")
            elif margin < 0.05:
                signals.append("THIN_MARGINS (<5% net)")

        state["fundamentals"] = {
            "valuation": valuation,
            "profitability": profitability,
            "growth": growth,
            "balance_sheet": balance_sheet,
            "dividends": dividends,
            "signals": signals,
        }
        log_agent(state, "fundamentals", "ok")
        print("OK")

    except Exception as e:
        state["errors"].append(f"Fundamentals: {e}")
        log_agent(state, "fundamentals", "error", str(e))
        print(f"ERROR: {e}")

    return state


# ---------------------------------------------------------------------------
# AGENT 4: TECHNICAL ANALYSIS
# ---------------------------------------------------------------------------

def agent_technical(state: dict) -> dict:
    """SMA, RSI, MACD, Bollinger Bands, trend analysis."""
    ticker = state["ticker"]
    print(f"  [4/11] Technical analysis...", end=" ", flush=True)

    try:
        closes = state.get("_closes", [])
        if not closes or len(closes) < 50:
            raise ValueError("Insufficient price history for technical analysis")

        current = closes[-1]

        # Moving averages
        sma_20 = compute_sma(closes, 20)[-1]
        sma_50 = compute_sma(closes, 50)[-1]
        sma_200 = compute_sma(closes, 200)[-1] if len(closes) >= 200 else None

        # RSI
        rsi_14 = compute_rsi(closes, 14)

        # MACD
        macd = compute_macd(closes)

        # Bollinger Bands
        bbands = compute_bollinger(closes, 20)

        # Trend determination
        trend_signals = []

        # SMA trend
        if sma_50 and sma_200:
            if sma_50 > sma_200:
                trend_signals.append("GOLDEN_CROSS (SMA50 > SMA200, bullish)")
            else:
                trend_signals.append("DEATH_CROSS (SMA50 < SMA200, bearish)")

        if sma_50:
            if current > sma_50:
                trend_signals.append("ABOVE_SMA50 (bullish)")
            else:
                trend_signals.append("BELOW_SMA50 (bearish)")

        # RSI signals
        if rsi_14:
            if rsi_14 > 70:
                trend_signals.append(f"RSI_OVERBOUGHT ({rsi_14:.1f})")
            elif rsi_14 < 30:
                trend_signals.append(f"RSI_OVERSOLD ({rsi_14:.1f})")
            else:
                trend_signals.append(f"RSI_NEUTRAL ({rsi_14:.1f})")

        # MACD signals
        if macd["histogram"] is not None:
            if macd["histogram"] > 0:
                trend_signals.append("MACD_BULLISH (histogram positive)")
            else:
                trend_signals.append("MACD_BEARISH (histogram negative)")

        # Bollinger Band signals
        if bbands["pct_b"] is not None:
            if bbands["pct_b"] > 1.0:
                trend_signals.append("ABOVE_UPPER_BAND (potential overbought)")
            elif bbands["pct_b"] < 0.0:
                trend_signals.append("BELOW_LOWER_BAND (potential oversold)")

        # Overall trend
        bullish = sum(1 for s in trend_signals if "bullish" in s.lower() or "oversold" in s.lower())
        bearish = sum(1 for s in trend_signals if "bearish" in s.lower() or "overbought" in s.lower())
        if bullish > bearish:
            overall = "BULLISH"
        elif bearish > bullish:
            overall = "BEARISH"
        else:
            overall = "NEUTRAL"

        state["technical"] = {
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "rsi_14": rsi_14,
            "macd": macd,
            "bollinger": bbands,
            "signals": trend_signals,
            "overall_trend": overall,
            "bullish_count": bullish,
            "bearish_count": bearish,
        }
        log_agent(state, "technical", "ok")
        print("OK")

    except Exception as e:
        state["errors"].append(f"Technical: {e}")
        log_agent(state, "technical", "error", str(e))
        print(f"ERROR: {e}")

    return state


# ---------------------------------------------------------------------------
# AGENT 5: RISK ASSESSMENT
# ---------------------------------------------------------------------------

def agent_risk(state: dict) -> dict:
    """Beta, volatility, max drawdown, Sharpe ratio."""
    ticker = state["ticker"]
    print(f"  [5/11] Risk assessment...", end=" ", flush=True)

    try:
        info = state.get("_yf_info", {})
        closes = state.get("_closes", [])
        if not closes or len(closes) < 30:
            raise ValueError("Insufficient data for risk analysis")

        # Daily returns
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]

        # Volatility
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        daily_vol = variance ** 0.5
        annual_vol = daily_vol * (252 ** 0.5)

        # Max drawdown
        peak = closes[0]
        max_dd = 0
        max_dd_date = ""
        for i, price in enumerate(closes):
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (assuming risk-free rate of 4.5%)
        annual_return = (closes[-1] / closes[0]) ** (252 / len(closes)) - 1
        risk_free = 0.045
        sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0

        # Sortino ratio (downside deviation only)
        neg_returns = [r for r in returns if r < 0]
        if neg_returns:
            downside_var = sum(r ** 2 for r in neg_returns) / len(returns)
            downside_dev = (downside_var ** 0.5) * (252 ** 0.5)
            sortino = (annual_return - risk_free) / downside_dev if downside_dev > 0 else 0
        else:
            sortino = None

        # Value at Risk (95% historical)
        sorted_returns = sorted(returns)
        var_95_idx = int(len(sorted_returns) * 0.05)
        var_95 = sorted_returns[var_95_idx] if var_95_idx < len(sorted_returns) else 0

        # Risk signals
        risk_signals = []
        if annual_vol > 0.40:
            risk_signals.append("HIGH_VOLATILITY (annualized > 40%)")
        elif annual_vol < 0.15:
            risk_signals.append("LOW_VOLATILITY (annualized < 15%)")

        if max_dd > 0.30:
            risk_signals.append(f"LARGE_DRAWDOWN (max {max_dd:.1%})")

        beta = safe_get(info, "beta")
        if beta:
            if beta > 1.5:
                risk_signals.append(f"HIGH_BETA ({beta:.2f}, amplifies market moves)")
            elif beta < 0.7:
                risk_signals.append(f"LOW_BETA ({beta:.2f}, defensive)")

        if sharpe > 1.0:
            risk_signals.append(f"STRONG_SHARPE ({sharpe:.2f})")
        elif sharpe < 0:
            risk_signals.append(f"NEGATIVE_SHARPE ({sharpe:.2f}, underperforming risk-free)")

        state["risk"] = {
            "beta": beta,
            "daily_volatility": round(daily_vol, 6),
            "annualized_volatility": round(annual_vol, 4),
            "max_drawdown": round(max_dd, 4),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2) if sortino else None,
            "var_95": round(var_95, 4),
            "annualized_return": round(annual_return, 4),
            "signals": risk_signals,
        }
        log_agent(state, "risk", "ok")
        print("OK")

    except Exception as e:
        state["errors"].append(f"Risk: {e}")
        log_agent(state, "risk", "error", str(e))
        print(f"ERROR: {e}")

    return state


# ---------------------------------------------------------------------------
# AGENT 6: PEER COMPARISON
# ---------------------------------------------------------------------------

def agent_peer_comparison(state: dict) -> dict:
    """Compare against peer tickers on key metrics."""
    ticker = state["ticker"]
    peers = state.get("peers", [])
    print(f"  [6/11] Peer comparison...", end=" ", flush=True)

    if not peers:
        print("SKIPPED (no peers)")
        log_agent(state, "peer_comparison", "skipped", "no peers provided")
        return state

    try:
        # Metrics to compare
        all_tickers = [ticker] + peers[:4]
        comparison = {}

        for t in all_tickers:
            try:
                stock = yf.Ticker(t)
                info = stock.info
                hist = stock.history(period="1y")

                closes = hist["Close"].tolist() if not hist.empty else []
                perf_1y = (closes[-1] / closes[0] - 1) if len(closes) > 1 else None

                # Volatility
                if len(closes) > 30:
                    rets = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                    mean_r = sum(rets) / len(rets)
                    vol = ((sum((r - mean_r)**2 for r in rets) / len(rets)) ** 0.5) * (252 ** 0.5)
                else:
                    vol = None

                comparison[t] = {
                    "name": safe_get(info, "longName", t),
                    "market_cap": safe_get(info, "marketCap"),
                    "pe_trailing": safe_get(info, "trailingPE"),
                    "pe_forward": safe_get(info, "forwardPE"),
                    "profit_margin": safe_get(info, "profitMargins"),
                    "revenue_growth": safe_get(info, "revenueGrowth"),
                    "roe": safe_get(info, "returnOnEquity"),
                    "debt_to_equity": safe_get(info, "debtToEquity"),
                    "beta": safe_get(info, "beta"),
                    "perf_1y": perf_1y,
                    "volatility": round(vol, 4) if vol else None,
                }
            except Exception:
                comparison[t] = {"name": t, "error": "data unavailable"}

        # Rank the subject ticker
        rankings = {}
        metrics_to_rank = [
            ("pe_trailing", "lower_better"),
            ("profit_margin", "higher_better"),
            ("revenue_growth", "higher_better"),
            ("roe", "higher_better"),
            ("perf_1y", "higher_better"),
        ]

        for metric, direction in metrics_to_rank:
            values = [(t, comparison[t].get(metric))
                      for t in all_tickers if comparison[t].get(metric) is not None]
            if values:
                reverse = direction == "higher_better"
                sorted_vals = sorted(values, key=lambda x: x[1], reverse=reverse)
                rank = next((i + 1 for i, (t, _) in enumerate(sorted_vals) if t == ticker), None)
                rankings[metric] = {"rank": rank, "of": len(sorted_vals)}

        state["peer_comparison"] = {
            "peers": comparison,
            "rankings": rankings,
        }
        log_agent(state, "peer_comparison", "ok")
        print(f"OK ({len(peers)} peers)")

    except Exception as e:
        state["errors"].append(f"Peers: {e}")
        log_agent(state, "peer_comparison", "error", str(e))
        print(f"ERROR: {e}")

    return state


# ---------------------------------------------------------------------------
# AGENT 7: SEC EDGAR FILINGS (completely free, no API key)
# ---------------------------------------------------------------------------

SEC_EDGAR_BASE = "https://data.sec.gov"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_USER_AGENT = "StockResearchAgent/1.0 (research@example.com)"


def _sec_get(url: str) -> dict | list:
    """Make a SEC EDGAR API call. Free, no auth — just User-Agent required."""
    resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _ticker_to_cik(ticker: str) -> str | None:
    """Map a stock ticker to its SEC CIK number."""
    try:
        data = _sec_get(SEC_TICKERS_URL)
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass
    return None


def agent_sec_filings(state: dict, sec_email: str | None = None) -> dict:
    """
    SEC EDGAR filings agent — pulls 10-K, 10-Q, 8-K filing metadata
    and XBRL financial facts. Completely free, no API key.

    Uses: data.sec.gov (SEC official API)
    Requires: User-Agent header with name/email (SEC policy)
    Rate limit: 10 requests/second (we stay well under)
    """
    global SEC_USER_AGENT
    ticker = state["ticker"]

    # Allow custom email for User-Agent
    if sec_email:
        SEC_USER_AGENT = f"StockResearchAgent/1.0 ({sec_email})"
    elif os.environ.get("SEC_EMAIL"):
        SEC_USER_AGENT = f"StockResearchAgent/1.0 ({os.environ['SEC_EMAIL']})"

    print(f"  [7/11] SEC EDGAR filings...", end=" ", flush=True)

    try:
        # 1. Map ticker to CIK
        cik = _ticker_to_cik(ticker)
        if not cik:
            print(f"SKIPPED (could not resolve CIK for {ticker})")
            log_agent(state, "sec_filings", "error", f"CIK not found for {ticker}")
            return state

        # 2. Get submission history
        submissions_url = f"{SEC_EDGAR_BASE}/submissions/CIK{cik}.json"
        submissions = _sec_get(submissions_url)

        company_name = submissions.get("name", ticker)
        sic = submissions.get("sic", "")
        sic_desc = submissions.get("sicDescription", "")
        state_of_inc = submissions.get("stateOfIncorporation", "")
        fiscal_year_end = submissions.get("fiscalYearEnd", "")

        # 3. Extract recent filings by type
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        def _extract_filings(form_type: str, limit: int = 5) -> list[dict]:
            """Extract filings of a specific type."""
            results = []
            for i, form in enumerate(forms):
                if form == form_type and i < len(dates):
                    acc = accessions[i].replace("-", "") if i < len(accessions) else ""
                    doc = primary_docs[i] if i < len(primary_docs) else ""
                    desc = descriptions[i] if i < len(descriptions) else ""
                    filing = {
                        "form": form,
                        "date": dates[i],
                        "accession": accessions[i] if i < len(accessions) else "",
                        "description": desc,
                    }
                    if acc and doc:
                        filing["url"] = (f"https://www.sec.gov/Archives/edgar/data/"
                                         f"{cik.lstrip('0')}/{acc}/{doc}")
                    results.append(filing)
                    if len(results) >= limit:
                        break
            return results

        filings_10k = _extract_filings("10-K", 3)
        filings_10q = _extract_filings("10-Q", 4)
        filings_8k = _extract_filings("8-K", 5)

        # 4. Try XBRL companyfacts for key financial metrics
        xbrl_facts = {}
        try:
            facts_url = f"{SEC_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
            facts_data = _sec_get(facts_url)

            us_gaap = facts_data.get("facts", {}).get("us-gaap", {})

            def _get_latest_annual(concept: str) -> dict | None:
                """Get the most recent annual (10-K) value for a concept."""
                tag = us_gaap.get(concept, {})
                units = tag.get("units", {})
                # Try USD first, then shares
                for unit_type in ["USD", "shares", "pure"]:
                    values = units.get(unit_type, [])
                    # Filter to 10-K filings
                    annual = [v for v in values if v.get("form") == "10-K"]
                    if annual:
                        latest = sorted(annual, key=lambda x: x.get("end", ""))[-1]
                        return {
                            "value": latest.get("val"),
                            "end_date": latest.get("end"),
                            "filed": latest.get("filed"),
                        }
                return None

            # Pull key metrics
            metrics_to_fetch = {
                "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                            "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
                "net_income": ["NetIncomeLoss", "ProfitLoss"],
                "total_assets": ["Assets"],
                "total_liabilities": ["Liabilities"],
                "stockholders_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
                "eps_basic": ["EarningsPerShareBasic"],
                "eps_diluted": ["EarningsPerShareDiluted"],
                "operating_income": ["OperatingIncomeLoss"],
                "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
                "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
            }

            for key, concepts in metrics_to_fetch.items():
                for concept in concepts:
                    result = _get_latest_annual(concept)
                    if result and result["value"] is not None:
                        xbrl_facts[key] = result
                        break

        except Exception:
            # XBRL data not available for all companies — not fatal
            pass

        # 5. Signals
        sec_signals = []

        if filings_10k:
            latest_10k = filings_10k[0]
            sec_signals.append(f"LATEST_10K: {latest_10k['date']}")

        if filings_8k:
            recent_8k_count = len(filings_8k)
            sec_signals.append(f"RECENT_8K_COUNT: {recent_8k_count} filings")
            # Many 8-Ks in short period can signal material events
            if recent_8k_count >= 4:
                sec_signals.append("ELEVATED_8K_ACTIVITY (potential material events)")

        if xbrl_facts.get("revenue") and xbrl_facts.get("net_income"):
            rev = xbrl_facts["revenue"]["value"]
            ni = xbrl_facts["net_income"]["value"]
            if rev and ni and rev > 0:
                margin = ni / rev
                sec_signals.append(f"SEC_NET_MARGIN: {margin:.1%} (from 10-K XBRL)")

        state["sec_filings"] = {
            "cik": cik,
            "company_name": company_name,
            "sic": sic,
            "sic_description": sic_desc,
            "state_of_incorporation": state_of_inc,
            "fiscal_year_end": fiscal_year_end,
            "filings": {
                "10-K": filings_10k,
                "10-Q": filings_10q,
                "8-K": filings_8k,
            },
            "xbrl_financials": xbrl_facts,
            "signals": sec_signals,
            "source": "SEC EDGAR (free, no API key)",
        }

        total_filings = len(filings_10k) + len(filings_10q) + len(filings_8k)
        xbrl_count = len(xbrl_facts)
        log_agent(state, "sec_filings", "ok",
                  f"{total_filings} filings, {xbrl_count} XBRL metrics")
        print(f"OK ({total_filings} filings, {xbrl_count} XBRL metrics)")

    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        state["errors"].append(f"SEC EDGAR: {error_msg}")
        log_agent(state, "sec_filings", "error", error_msg)
        print(f"ERROR: {error_msg}")

    except Exception as e:
        state["errors"].append(f"SEC EDGAR: {e}")
        log_agent(state, "sec_filings", "error", str(e))
        print(f"ERROR: {e}")

    return state


# ---------------------------------------------------------------------------
# AGENT 8: FINNHUB SENTIMENT (free tier endpoints)
# ---------------------------------------------------------------------------

FINNHUB_BASE = "https://finnhub.io/api/v1"


def _finnhub_get(endpoint: str, params: dict, api_key: str) -> dict | list:
    """Make a Finnhub API call."""
    params["token"] = api_key
    resp = requests.get(f"{FINNHUB_BASE}/{endpoint}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _fetch_recommendation_trends(ticker: str, api_key: str) -> list[dict]:
    """
    /stock/recommendation [FREE]
    Returns monthly analyst recommendation trends:
    strongBuy, buy, hold, sell, strongSell counts.
    """
    data = _finnhub_get("stock/recommendation", {"symbol": ticker}, api_key)
    if not data or not isinstance(data, list):
        return []
    # Return latest 6 months
    return data[:6]


def _fetch_insider_sentiment(ticker: str, api_key: str) -> list[dict]:
    """
    /stock/insider-sentiment [FREE]
    Returns Monthly Share Purchase Ratio (MSPR).
    Positive MSPR = net buying, negative = net selling.
    """
    from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")
    data = _finnhub_get("stock/insider-sentiment", {
        "symbol": ticker, "from": from_date, "to": to_date,
    }, api_key)
    return data.get("data", []) if isinstance(data, dict) else []


def _fetch_company_news(ticker: str, api_key: str, days: int = 7) -> list[dict]:
    """
    /company-news [FREE]
    Returns recent news articles for the ticker.
    """
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    data = _finnhub_get("company-news", {
        "symbol": ticker, "from": from_date, "to": to_date,
    }, api_key)
    if not data or not isinstance(data, list):
        return []
    # Return latest 20 articles
    return data[:20]


def _fetch_price_target(ticker: str, api_key: str) -> dict:
    """
    /stock/price-target [FREE]
    Returns analyst price target consensus.
    """
    data = _finnhub_get("stock/price-target", {"symbol": ticker}, api_key)
    if not data or not isinstance(data, dict):
        return {}
    return data


def agent_finnhub_sentiment(state: dict, finnhub_key: str | None = None) -> dict:
    """
    Finnhub sentiment agent — pulls analyst recommendations, insider
    sentiment, company news, and price targets. All free-tier endpoints.

    Requires: Finnhub API key (free at https://finnhub.io)
    API calls per run: 4 (60/min limit, no daily cap)
    """
    ticker = state["ticker"]

    # Resolve key
    if not finnhub_key:
        finnhub_key = os.environ.get("FINNHUB_API_KEY")
    if not finnhub_key:
        print(f"  [8/11] Finnhub sentiment... SKIPPED (set FINNHUB_API_KEY or --finnhub-key)")
        log_agent(state, "finnhub_sentiment", "skipped", "no API key")
        return state

    print(f"  [8/11] Finnhub sentiment...", end=" ", flush=True)

    endpoint_status = {}

    def _try_finnhub(name, fn):
        """Try a single Finnhub endpoint, skip if premium/error."""
        try:
            result = fn()
            endpoint_status[name] = "ok"
            return result
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else 0
            if code in (401, 403):
                endpoint_status[name] = "premium"
            else:
                endpoint_status[name] = "error"
            return None
        except Exception:
            endpoint_status[name] = "error"
            return None

    # Try each endpoint independently
    recommendations = _try_finnhub(
        "recommendation", lambda: _fetch_recommendation_trends(ticker, finnhub_key))
    insider = _try_finnhub(
        "insider-sentiment", lambda: _fetch_insider_sentiment(ticker, finnhub_key))
    news = _try_finnhub(
        "company-news", lambda: _fetch_company_news(ticker, finnhub_key, days=7))
    price_target = _try_finnhub(
        "price-target", lambda: _fetch_price_target(ticker, finnhub_key))

    free_count = sum(1 for v in endpoint_status.values() if v == "ok")
    premium_count = sum(1 for v in endpoint_status.values() if v == "premium")

    if free_count == 0:
        status_str = ", ".join(f"{k}={v}" for k, v in endpoint_status.items())
        print(f"NO FREE DATA ({status_str})")
        state["errors"].append(f"Finnhub: no free endpoints ({status_str})")
        log_agent(state, "finnhub_sentiment", "error",
                  f"0/4 endpoints free, {premium_count} premium")
        return state

    try:

        # --- Process recommendations ---
        rec_summary = {}
        if recommendations:
            latest = recommendations[0]
            total = (latest.get("strongBuy", 0) + latest.get("buy", 0) +
                     latest.get("hold", 0) + latest.get("sell", 0) +
                     latest.get("strongSell", 0))
            if total > 0:
                bullish = latest.get("strongBuy", 0) + latest.get("buy", 0)
                bearish = latest.get("sell", 0) + latest.get("strongSell", 0)
                rec_summary = {
                    "period": latest.get("period"),
                    "strong_buy": latest.get("strongBuy", 0),
                    "buy": latest.get("buy", 0),
                    "hold": latest.get("hold", 0),
                    "sell": latest.get("sell", 0),
                    "strong_sell": latest.get("strongSell", 0),
                    "total_analysts": total,
                    "bullish_pct": round(bullish / total, 2),
                    "bearish_pct": round(bearish / total, 2),
                    "consensus": ("strong buy" if bullish / total > 0.7
                                  else "buy" if bullish / total > 0.5
                                  else "hold" if bearish / total < 0.3
                                  else "sell"),
                }

            # Trend: compare latest vs 3 months ago
            if len(recommendations) >= 3:
                prev = recommendations[2]
                prev_bullish = prev.get("strongBuy", 0) + prev.get("buy", 0)
                prev_total = sum(prev.get(k, 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"])
                if prev_total > 0 and total > 0:
                    shift = (bullish / total) - (prev_bullish / prev_total)
                    rec_summary["3m_shift"] = round(shift, 2)
                    rec_summary["trend"] = ("upgrading" if shift > 0.05
                                            else "downgrading" if shift < -0.05
                                            else "stable")

        # --- Process insider sentiment ---
        insider_summary = {}
        if insider:
            recent = insider[-3:] if len(insider) >= 3 else insider
            avg_mspr = sum(i.get("mspr", 0) for i in recent) / len(recent)
            total_change = sum(i.get("change", 0) for i in recent)
            insider_summary = {
                "months_analyzed": len(recent),
                "avg_mspr": round(avg_mspr, 2),
                "net_shares_change": total_change,
                "signal": ("NET_BUYING" if avg_mspr > 5
                           else "NET_SELLING" if avg_mspr < -5
                           else "MIXED"),
            }

        # --- Process news ---
        news_summary = {}
        if news:
            headlines = [{"headline": a.get("headline", ""),
                          "source": a.get("source", ""),
                          "datetime": a.get("datetime", 0)}
                         for a in news[:10]]
            news_summary = {
                "articles_last_7d": len(news),
                "top_headlines": headlines,
            }

        # --- Process price targets ---
        pt_summary = {}
        if price_target and price_target.get("targetMean"):
            current_price = state.get("price", {}).get("current")
            target_mean = price_target.get("targetMean")
            pt_summary = {
                "target_high": price_target.get("targetHigh"),
                "target_low": price_target.get("targetLow"),
                "target_mean": target_mean,
                "target_median": price_target.get("targetMedian"),
                "num_analysts": price_target.get("lastUpdated"),
            }
            if current_price and target_mean:
                upside = (target_mean - current_price) / current_price
                pt_summary["implied_upside"] = round(upside, 4)
                pt_summary["signal"] = ("UPSIDE" if upside > 0.10
                                        else "DOWNSIDE" if upside < -0.10
                                        else "NEAR_TARGET")

        # --- Sentiment signals ---
        sentiment_signals = []

        if rec_summary.get("consensus"):
            sentiment_signals.append(
                f"ANALYST_CONSENSUS: {rec_summary['consensus'].upper()} "
                f"({rec_summary.get('bullish_pct', 0):.0%} bullish, "
                f"{rec_summary['total_analysts']} analysts)")

        if rec_summary.get("trend"):
            sentiment_signals.append(
                f"ANALYST_TREND: {rec_summary['trend'].upper()} "
                f"({rec_summary.get('3m_shift', 0):+.0%} shift over 3 months)")

        if insider_summary.get("signal"):
            sentiment_signals.append(
                f"INSIDER_{insider_summary['signal']} "
                f"(MSPR: {insider_summary['avg_mspr']:+.1f}, "
                f"net shares: {insider_summary['net_shares_change']:+,})")

        if pt_summary.get("signal"):
            sentiment_signals.append(
                f"PRICE_TARGET_{pt_summary['signal']} "
                f"(mean: ${pt_summary['target_mean']:.2f}, "
                f"implied: {pt_summary.get('implied_upside', 0):+.1%})")

        if news_summary.get("articles_last_7d", 0) > 30:
            sentiment_signals.append(f"HIGH_NEWS_VOLUME ({news_summary['articles_last_7d']} articles/week)")
        elif news_summary.get("articles_last_7d", 0) < 5:
            sentiment_signals.append(f"LOW_NEWS_VOLUME ({news_summary['articles_last_7d']} articles/week)")

        state["sentiment"] = {
            "recommendations": rec_summary,
            "insider": insider_summary,
            "news": news_summary,
            "price_target": pt_summary,
            "signals": sentiment_signals,
            "endpoint_status": endpoint_status,
            "source": "finnhub (free tier)",
            "api_calls_used": free_count,
        }

        status_parts = [f"{k}={'OK' if v == 'ok' else '$$' if v == 'premium' else '!!'}"
                        for k, v in endpoint_status.items()]
        status_label = "OK" if premium_count == 0 else f"PARTIAL ({free_count}/4 free)"
        log_agent(state, "finnhub_sentiment", "ok",
                  f"{free_count}/4 free, {premium_count} premium")
        print(f"{status_label} [{', '.join(status_parts)}]")

    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        try:
            error_msg = e.response.json().get("error", str(e))
        except Exception:
            pass
        state["errors"].append(f"Finnhub: {error_msg}")
        log_agent(state, "finnhub_sentiment", "error", error_msg)
        print(f"ERROR: {error_msg}")

    except Exception as e:
        state["errors"].append(f"Finnhub: {e}")
        log_agent(state, "finnhub_sentiment", "error", str(e))
        print(f"ERROR: {e}")

    return state


# ---------------------------------------------------------------------------
# AGENT 9: EODHD SENTIMENT (free tier: 20 API calls/day)
# ---------------------------------------------------------------------------

EODHD_BASE = "https://eodhd.com/api"


def _eodhd_get(endpoint: str, params: dict, api_key: str) -> Any:
    """Make an EODHD API call."""
    params["api_token"] = api_key
    params["fmt"] = "json"
    resp = requests.get(f"{EODHD_BASE}/{endpoint}", params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def agent_eodhd_sentiment(state: dict, eodhd_key: str | None = None) -> dict:
    """
    EODHD sentiment agent — pulls daily sentiment scores and news.
    Sentiment is normalized from -1 (very negative) to +1 (very positive).

    Free tier: 20 API calls/day. This agent uses ~10 calls (5 per endpoint).
    Get a free key: https://eodhd.com/register
    """
    ticker = state["ticker"]

    # Resolve key
    if not eodhd_key:
        eodhd_key = os.environ.get("EODHD_API_KEY")
    if not eodhd_key:
        print(f"  [9/11] EODHD sentiment... SKIPPED (set EODHD_API_KEY or --eodhd-key)")
        log_agent(state, "eodhd_sentiment", "skipped", "no API key")
        return state

    print(f"  [9/11] EODHD sentiment...", end=" ", flush=True)

    try:
        eodhd_ticker = f"{ticker}.US"
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # 1. Sentiment scores (last 30 days) — 5 API calls
        sentiment_data = _eodhd_get("sentiments", {
            "s": eodhd_ticker,
            "from": from_date,
            "to": to_date,
        }, eodhd_key)

        # 2. News headlines (latest 10) — 5 API calls
        news_data = _eodhd_get("news", {
            "s": eodhd_ticker,
            "offset": "0",
            "limit": "10",
        }, eodhd_key)

        # --- Process sentiment scores ---
        sentiment_summary = {}
        # Response format: {"TICKER.US": [{"date": "...", "count": N, "normalized": 0.xx}, ...]}
        ticker_key = None
        for key in sentiment_data:
            if ticker.lower() in key.lower():
                ticker_key = key
                break

        daily_scores = []
        if ticker_key and isinstance(sentiment_data[ticker_key], list):
            raw = sentiment_data[ticker_key]
            for entry in raw:
                daily_scores.append({
                    "date": entry.get("date"),
                    "score": entry.get("normalized", 0),
                    "article_count": entry.get("count", 0),
                })

        if daily_scores:
            scores = [d["score"] for d in daily_scores]
            counts = [d["article_count"] for d in daily_scores]
            avg_score = sum(scores) / len(scores)
            latest_score = scores[-1] if scores else 0
            total_articles = sum(counts)

            # Trend: compare last 7 days vs prior 7 days
            if len(scores) >= 14:
                recent_avg = sum(scores[-7:]) / 7
                prior_avg = sum(scores[-14:-7]) / 7
                trend_shift = recent_avg - prior_avg
            else:
                recent_avg = avg_score
                trend_shift = 0

            # Classify
            if avg_score > 0.3:
                overall = "POSITIVE"
            elif avg_score < -0.3:
                overall = "NEGATIVE"
            elif avg_score > 0.1:
                overall = "SLIGHTLY_POSITIVE"
            elif avg_score < -0.1:
                overall = "SLIGHTLY_NEGATIVE"
            else:
                overall = "NEUTRAL"

            sentiment_summary = {
                "avg_score_30d": round(avg_score, 4),
                "latest_score": round(latest_score, 4),
                "latest_date": daily_scores[-1]["date"] if daily_scores else None,
                "total_articles_30d": total_articles,
                "days_with_data": len(daily_scores),
                "overall": overall,
                "trend_shift_7d": round(trend_shift, 4),
                "trend": ("improving" if trend_shift > 0.05
                          else "declining" if trend_shift < -0.05
                          else "stable"),
            }

        # --- Process news ---
        eodhd_news = []
        if isinstance(news_data, list):
            for article in news_data[:5]:
                eodhd_news.append({
                    "title": article.get("title", ""),
                    "date": article.get("date", ""),
                    "link": article.get("link", ""),
                })

        # --- Signals ---
        eodhd_signals = []
        if sentiment_summary:
            eodhd_signals.append(
                f"EODHD_SENTIMENT_{sentiment_summary['overall']} "
                f"(30d avg: {sentiment_summary['avg_score_30d']:+.2f}, "
                f"latest: {sentiment_summary['latest_score']:+.2f})")

            if sentiment_summary.get("trend") != "stable":
                eodhd_signals.append(
                    f"EODHD_TREND_{sentiment_summary['trend'].upper()} "
                    f"(7d shift: {sentiment_summary['trend_shift_7d']:+.2f})")

            if sentiment_summary["total_articles_30d"] > 100:
                eodhd_signals.append(
                    f"HIGH_MEDIA_COVERAGE ({sentiment_summary['total_articles_30d']} articles/30d)")

        # Store in a separate key to avoid overwriting Finnhub sentiment
        state.setdefault("eodhd_sentiment", {})
        state["eodhd_sentiment"] = {
            "scores": sentiment_summary,
            "news": eodhd_news,
            "signals": eodhd_signals,
            "source": "EODHD (free tier)",
            "api_calls_used": "~10 of 20 daily",
        }

        log_agent(state, "eodhd_sentiment", "ok")
        print("OK")

    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        try:
            error_msg = e.response.text[:200]
        except Exception:
            pass
        state["errors"].append(f"EODHD: {error_msg}")
        log_agent(state, "eodhd_sentiment", "error", error_msg)
        print(f"ERROR: {error_msg}")

    except Exception as e:
        state["errors"].append(f"EODHD: {e}")
        log_agent(state, "eodhd_sentiment", "error", str(e))
        print(f"ERROR: {e}")

    return state


# ---------------------------------------------------------------------------
# AGENT 10: LLM SYNTHESIS
# ---------------------------------------------------------------------------

def _build_llm_context(state: dict) -> str:
    """Build a structured data summary for the LLM prompt."""
    ticker = state["ticker"]
    profile = state.get("profile", {})
    price = state.get("price", {})
    fund = state.get("fundamentals", {})
    tech = state.get("technical", {})
    risk = state.get("risk", {})
    peers = state.get("peer_comparison", {})

    sections = []

    sections.append(f"TICKER: {ticker}")
    sections.append(f"COMPANY: {profile.get('name', ticker)}")
    sections.append(f"SECTOR: {profile.get('sector', 'N/A')} | INDUSTRY: {profile.get('industry', 'N/A')}")

    if price:
        perf = price.get("performance", {})
        sections.append(f"\nPRICE & PERFORMANCE:")
        sections.append(f"  Current: ${price.get('current', 0):.2f}")
        sections.append(f"  52-week range: ${price.get('low_52w', 0):.2f} - ${price.get('high_52w', 0):.2f}")
        sections.append(f"  Position in range: {price.get('range_position', 0):.0%}")
        for period in ["1d", "1m", "3m", "6m", "1y"]:
            v = perf.get(period)
            if v is not None:
                sections.append(f"  {period} return: {v:+.2%}")
        sections.append(f"  Volume vs 30d avg: {price.get('volume_vs_avg', 1):.1f}x")

    if fund:
        val = fund.get("valuation", {})
        prof = fund.get("profitability", {})
        growth = fund.get("growth", {})
        bs = fund.get("balance_sheet", {})

        sections.append(f"\nVALUATION:")
        sections.append(f"  Market cap: {fmt_large_number(val.get('market_cap'))}")
        sections.append(f"  P/E trailing: {val.get('pe_trailing') or 'N/A'} | Forward: {val.get('pe_forward') or 'N/A'}")
        sections.append(f"  PEG: {val.get('peg_ratio') or 'N/A'} | P/B: {val.get('pb_ratio') or 'N/A'} | EV/EBITDA: {val.get('ev_ebitda') or 'N/A'}")

        sections.append(f"\nPROFITABILITY:")
        sections.append(f"  Gross margin: {fmt_pct(prof.get('gross_margin'))} | Net margin: {fmt_pct(prof.get('profit_margin'))}")
        sections.append(f"  ROE: {fmt_pct(prof.get('roe'))} | ROA: {fmt_pct(prof.get('roa'))}")

        sections.append(f"\nGROWTH:")
        sections.append(f"  Revenue growth: {fmt_pct(growth.get('revenue_growth'))}")
        sections.append(f"  Earnings growth: {fmt_pct(growth.get('earnings_growth'))}")

        sections.append(f"\nBALANCE SHEET:")
        sections.append(f"  Cash: {fmt_large_number(bs.get('total_cash'))} | Debt: {fmt_large_number(bs.get('total_debt'))}")
        sections.append(f"  D/E: {bs.get('debt_to_equity') or 'N/A'} | Current ratio: {bs.get('current_ratio') or 'N/A'}")
        sections.append(f"  FCF: {fmt_large_number(bs.get('free_cash_flow'))}")

        if fund.get("signals"):
            sections.append(f"\nFUNDAMENTAL SIGNALS: {', '.join(fund['signals'])}")

    if tech:
        sections.append(f"\nTECHNICAL ANALYSIS:")
        sections.append(f"  Overall trend: {tech.get('overall_trend') or 'N/A'}")
        sections.append(f"  SMA(50): {fmt_price(tech.get('sma_50'))} | SMA(200): {fmt_price(tech.get('sma_200'))}")
        sections.append(f"  RSI(14): {tech.get('rsi_14') or 'N/A'}")
        macd = tech.get("macd", {})
        if macd.get("histogram") is not None:
            sections.append(f"  MACD histogram: {macd['histogram']:.4f}")
        if tech.get("signals"):
            sections.append(f"  SIGNALS: {', '.join(tech['signals'])}")

    if risk:
        sections.append(f"\nRISK PROFILE:")
        sections.append(f"  Beta: {risk.get('beta') or 'N/A'} | Volatility: {risk.get('annualized_volatility', 0):.2%}")
        sections.append(f"  Max drawdown: {risk.get('max_drawdown', 0):.2%}")
        sections.append(f"  Sharpe: {risk.get('sharpe_ratio') or 'N/A'} | Sortino: {risk.get('sortino_ratio') or 'N/A'}")
        if risk.get("signals"):
            sections.append(f"  SIGNALS: {', '.join(risk['signals'])}")

    if peers.get("peers"):
        sections.append(f"\nPEER COMPARISON:")
        for t, c in peers["peers"].items():
            if not c.get("error"):
                sections.append(f"  {t}: P/E={c.get('pe_trailing') or 'N/A'} Margin={fmt_pct(c.get('profit_margin'))} 1Y={fmt_pct(c.get('perf_1y'))}")
        rankings = peers.get("rankings", {})
        if rankings:
            for metric, r in rankings.items():
                sections.append(f"  Rank {metric}: #{r['rank']} of {r['of']}")

    sec = state.get("sec_filings", {})
    if sec:
        sections.append(f"\nSEC EDGAR FILINGS:")
        sections.append(f"  CIK: {sec.get('cik')} | SIC: {sec.get('sic')} ({sec.get('sic_description', '')})")
        filings = sec.get("filings", {})
        if filings.get("10-K"):
            sections.append(f"  Latest 10-K: {filings['10-K'][0]['date']}")
        if filings.get("10-Q"):
            sections.append(f"  Latest 10-Q: {filings['10-Q'][0]['date']}")
        if filings.get("8-K"):
            sections.append(f"  Recent 8-K filings: {len(filings['8-K'])}")
        xbrl = sec.get("xbrl_financials", {})
        if xbrl:
            sections.append(f"  XBRL financial data (from latest 10-K):")
            for key, data in xbrl.items():
                val = data.get("value")
                if val is not None:
                    if abs(val) >= 1e9:
                        sections.append(f"    {key}: ${val/1e9:.2f}B ({data.get('end_date', '')})")
                    elif abs(val) >= 1e6:
                        sections.append(f"    {key}: ${val/1e6:.1f}M ({data.get('end_date', '')})")
                    else:
                        sections.append(f"    {key}: {val} ({data.get('end_date', '')})")
        if sec.get("signals"):
            sections.append(f"  SIGNALS: {', '.join(sec['signals'])}")

    sentiment = state.get("sentiment", {})
    if sentiment:
        sections.append(f"\nSENTIMENT (Finnhub):")
        rec = sentiment.get("recommendations", {})
        if rec:
            sections.append(f"  Analyst consensus: {rec.get('consensus', 'N/A')} "
                            f"({rec.get('bullish_pct', 0):.0%} bullish, {rec.get('total_analysts', 0)} analysts)")
            if rec.get("trend"):
                sections.append(f"  Trend (3m): {rec['trend']} ({rec.get('3m_shift', 0):+.0%})")
        ins = sentiment.get("insider", {})
        if ins:
            sections.append(f"  Insider sentiment: {ins.get('signal', 'N/A')} "
                            f"(MSPR: {ins.get('avg_mspr', 0):+.1f}, net shares: {ins.get('net_shares_change', 0):+,})")
        pt = sentiment.get("price_target", {})
        if pt.get("target_mean"):
            sections.append(f"  Price target (mean): ${pt['target_mean']:.2f} "
                            f"(implied upside: {pt.get('implied_upside', 0):+.1%})")
            sections.append(f"  Target range: ${pt.get('target_low', 0):.2f} - ${pt.get('target_high', 0):.2f}")
        if sentiment.get("signals"):
            sections.append(f"  SIGNALS: {', '.join(sentiment['signals'])}")

    eodhd = state.get("eodhd_sentiment", {})
    if eodhd.get("scores"):
        sc = eodhd["scores"]
        sections.append(f"\nNEWS SENTIMENT (EODHD):")
        sections.append(f"  30-day avg score: {(sc.get('avg_score_30d') or 0):+.4f} (scale: -1 to +1)")
        sections.append(f"  Latest score: {(sc.get('latest_score') or 0):+.4f} ({sc.get('latest_date') or 'N/A'})")
        sections.append(f"  Overall: {sc.get('overall') or 'N/A'} | Trend: {sc.get('trend') or 'N/A'}")
        sections.append(f"  Articles (30d): {sc.get('total_articles_30d') or 0}")
        if eodhd.get("signals"):
            sections.append(f"  SIGNALS: {', '.join(eodhd['signals'])}")

    return "\n".join(sections)


LLM_SYSTEM_PROMPT = """You are a senior equity research analyst writing for institutional investors.
You produce clear, data-driven analysis. You do not give buy/sell recommendations —
you present the investment case (bull and bear) and let the reader decide.

Your writing style:
- Lead with the most important insight, not background
- Use specific numbers from the data to support every claim
- Be direct and concise — no filler, no caveats about "not being financial advice"
- Structure: Investment thesis, Key strengths, Key risks, What to watch, Conclusion
- Total length: 400-600 words"""

LLM_USER_PROMPT = """Analyze the following stock data and write a research synthesis.

{context}

Write your analysis with these sections:
1. INVESTMENT THESIS (2-3 sentences — the core argument for/against this stock right now)
2. KEY STRENGTHS (3-4 bullet points with specific data points)
3. KEY RISKS (3-4 bullet points with specific data points)
4. WHAT TO WATCH (2-3 upcoming catalysts or inflection points)
5. CONCLUSION (2-3 sentences — balanced summary of the risk/reward)

Use the actual numbers from the data. Be specific, not generic."""


def _call_openai(context: str, api_key: str, model: str) -> str:
    """Call OpenAI API for synthesis."""
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": LLM_USER_PROMPT.format(context=context)},
            ],
            "temperature": 0.3,
            "max_tokens": 1500,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_anthropic(context: str, api_key: str, model: str) -> str:
    """Call Anthropic API for synthesis."""
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1500,
            "system": LLM_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": LLM_USER_PROMPT.format(context=context)},
            ],
            "temperature": 0.3,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
}


def agent_llm_synthesis(state: dict, llm_provider: str | None = None,
                        llm_key: str | None = None, llm_model: str | None = None) -> dict:
    """
    LLM-powered synthesis agent. Feeds all structured data to an LLM
    and gets back a narrative research analysis.

    Supports: openai, anthropic
    Graceful skip if no provider/key configured.
    """
    print(f"  [10/11] LLM synthesis...", end=" ", flush=True)

    # Resolve provider and key
    if not llm_provider:
        # Auto-detect from environment
        if os.environ.get("OPENAI_API_KEY"):
            llm_provider = "openai"
            llm_key = llm_key or os.environ["OPENAI_API_KEY"]
        elif os.environ.get("ANTHROPIC_API_KEY"):
            llm_provider = "anthropic"
            llm_key = llm_key or os.environ["ANTHROPIC_API_KEY"]
        else:
            print("SKIPPED (no LLM configured, use --llm flag)")
            log_agent(state, "llm_synthesis", "skipped", "no LLM provider configured")
            return state

    if not llm_key:
        env_var = "OPENAI_API_KEY" if llm_provider == "openai" else "ANTHROPIC_API_KEY"
        llm_key = os.environ.get(env_var)
        if not llm_key:
            print(f"SKIPPED (set {env_var} or use --llm-key)")
            log_agent(state, "llm_synthesis", "skipped", f"no API key for {llm_provider}")
            return state

    model = llm_model or DEFAULT_MODELS.get(llm_provider, "gpt-4o")

    try:
        # Build context from all agent data
        context = _build_llm_context(state)

        # Call LLM
        print(f"{llm_provider}/{model}...", end=" ", flush=True)

        if llm_provider == "openai":
            narrative = _call_openai(context, llm_key, model)
        elif llm_provider == "anthropic":
            narrative = _call_anthropic(context, llm_key, model)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        state["synthesis"] = {
            "narrative": narrative,
            "provider": llm_provider,
            "model": model,
            "context_length": len(context),
        }

        log_agent(state, "llm_synthesis", "ok", f"{llm_provider}/{model}")
        print("OK")

    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        try:
            error_body = e.response.json()
            error_msg = error_body.get("error", {}).get("message", str(e))
        except Exception:
            pass
        state["errors"].append(f"LLM synthesis: {error_msg}")
        log_agent(state, "llm_synthesis", "error", error_msg)
        print(f"ERROR: {error_msg}")

    except Exception as e:
        state["errors"].append(f"LLM synthesis: {e}")
        log_agent(state, "llm_synthesis", "error", str(e))
        print(f"ERROR: {e}")

    return state


# ---------------------------------------------------------------------------
# AGENT 11: REPORT ASSEMBLY
# ---------------------------------------------------------------------------

def agent_report_assembly(state: dict) -> str:
    """Synthesize all agent outputs into a structured research report."""
    print(f"  [11/11] Assembling report...", end=" ", flush=True)

    ticker = state["ticker"]
    profile = state.get("profile", {})
    price = state.get("price", {})
    fund = state.get("fundamentals", {})
    tech = state.get("technical", {})
    risk = state.get("risk", {})
    peers = state.get("peer_comparison", {})
    synthesis = state.get("synthesis", {})

    lines = []
    def w(text=""):
        lines.append(text)

    # --- Header ---
    w(f"# {profile.get('name', ticker)} ({ticker}) — Research Report")
    w(f"Generated: {state['timestamp']}")
    w(f"Data source: yfinance (delayed)")
    w()

    # --- Executive Summary ---
    w("## Executive summary")
    w()
    current = price.get("current")
    if current:
        w(f"{profile.get('name', ticker)} is trading at {fmt_price(current)}, "
          f"{fmt_pct(price.get('change_1d'))} on the day. "
          f"The stock is at {price.get('range_position', 0):.0%} of its 52-week range "
          f"({fmt_price(price.get('low_52w'))} — {fmt_price(price.get('high_52w'))}).")
    w()

    # Key metrics summary
    val = fund.get("valuation", {})
    prof = fund.get("profitability", {})
    if val.get("pe_trailing") or val.get("market_cap"):
        w(f"Market cap: {fmt_large_number(val.get('market_cap'))} | "
          f"P/E: {val.get('pe_trailing') or 'N/A'} | "
          f"Fwd P/E: {val.get('pe_forward') or 'N/A'} | "
          f"Profit margin: {fmt_pct(prof.get('profit_margin'))}")
    w()

    # Technical stance
    if tech.get("overall_trend"):
        w(f"Technical outlook: **{tech['overall_trend']}** "
          f"({tech.get('bullish_count', 0)} bullish / {tech.get('bearish_count', 0)} bearish signals)")
    w()

    # --- LLM Synthesis ---
    if synthesis.get("narrative"):
        w("## AI research synthesis")
        w(f"*Generated by {synthesis.get('provider', 'LLM')}/{synthesis.get('model', 'unknown')}*")
        w()
        w(synthesis["narrative"])
        w()

    # --- Company Profile ---
    w("## Company profile")
    w()
    w(f"Sector: {profile.get('sector', 'N/A')} | Industry: {profile.get('industry', 'N/A')}")
    w(f"Country: {profile.get('country', 'N/A')} | Employees: {profile.get('employees', 'N/A'):,}" if profile.get('employees') else f"Country: {profile.get('country', 'N/A')}")
    w()
    desc = profile.get("description", "")
    if desc:
        # Truncate to ~300 chars
        if len(desc) > 300:
            desc = desc[:297] + "..."
        w(desc)
    w()

    # --- Price & Performance ---
    w("## Price & performance")
    w()
    if current:
        w(f"Current price: {fmt_price(current)}")
        w(f"52-week high: {fmt_price(price.get('high_52w'))} ({price.get('high_52w_date', '')})")
        w(f"52-week low: {fmt_price(price.get('low_52w'))} ({price.get('low_52w_date', '')})")
        w()

        perf = price.get("performance", {})
        w("Performance:")
        periods = [("1d", "1 Day"), ("5d", "5 Day"), ("1m", "1 Month"),
                   ("3m", "3 Month"), ("6m", "6 Month"), ("1y", "1 Year")]
        perf_parts = []
        for key, label in periods:
            v = perf.get(key)
            if v is not None:
                perf_parts.append(f"{label}: {v:+.2%}")
        w("  " + " | ".join(perf_parts))
        w()

        w(f"Volume (latest): {price.get('volume_latest', 0):,.0f}")
        w(f"Volume (30d avg): {price.get('volume_avg_30d', 0):,.0f}")
        vol_vs = price.get("volume_vs_avg", 1)
        if vol_vs > 1.5:
            w(f"Volume is {vol_vs:.1f}x above average (elevated activity)")
        elif vol_vs < 0.5:
            w(f"Volume is {vol_vs:.1f}x below average (low activity)")
    w()

    # --- Fundamentals ---
    w("## Fundamental analysis")
    w()
    if val:
        w("Valuation:")
        w(f"  Market cap: {fmt_large_number(val.get('market_cap'))} | "
          f"EV: {fmt_large_number(val.get('enterprise_value'))}")
        w(f"  P/E (trailing): {val.get('pe_trailing') or 'N/A'} | "
          f"P/E (forward): {val.get('pe_forward') or 'N/A'} | "
          f"PEG: {val.get('peg_ratio') or 'N/A'}")
        w(f"  P/S: {val.get('ps_ratio') or 'N/A'} | "
          f"P/B: {val.get('pb_ratio') or 'N/A'} | "
          f"EV/EBITDA: {val.get('ev_ebitda') or 'N/A'}")
        w()

    if prof:
        w("Profitability:")
        w(f"  Gross margin: {fmt_pct(prof.get('gross_margin'))} | "
          f"Operating margin: {fmt_pct(prof.get('operating_margin'))} | "
          f"Net margin: {fmt_pct(prof.get('profit_margin'))}")
        w(f"  ROE: {fmt_pct(prof.get('roe'))} | "
          f"ROA: {fmt_pct(prof.get('roa'))}")
        w(f"  EPS (trailing): {prof.get('eps_trailing') or 'N/A'} | "
          f"EPS (forward): {prof.get('eps_forward') or 'N/A'}")
        w()

    growth = fund.get("growth", {})
    if growth.get("revenue_growth") is not None:
        w("Growth:")
        w(f"  Revenue growth: {fmt_pct(growth.get('revenue_growth'))} | "
          f"Earnings growth: {fmt_pct(growth.get('earnings_growth'))}")
        w()

    bs = fund.get("balance_sheet", {})
    if bs:
        w("Balance sheet:")
        w(f"  Cash: {fmt_large_number(bs.get('total_cash'))} | "
          f"Debt: {fmt_large_number(bs.get('total_debt'))} | "
          f"D/E: {bs.get('debt_to_equity') or 'N/A'}")
        w(f"  Current ratio: {bs.get('current_ratio') or 'N/A'} | "
          f"FCF: {fmt_large_number(bs.get('free_cash_flow'))}")
        w()

    div = fund.get("dividends", {})
    if div.get("dividend_yield"):
        w(f"Dividends: Yield {fmt_pct(div.get('dividend_yield'))} | "
          f"Rate ${div.get('dividend_rate') or 'N/A'} | "
          f"Payout {fmt_pct(div.get('payout_ratio'))}")
        w()

    if fund.get("signals"):
        w("Fundamental signals:")
        for sig in fund["signals"]:
            w(f"  -> {sig}")
        w()

    # --- Technical Analysis ---
    w("## Technical analysis")
    w()
    if tech:
        w(f"Overall trend: {tech.get('overall_trend') or 'N/A'}")
        w()
        w("Moving averages:")
        w(f"  SMA(20): {fmt_price(tech.get('sma_20'))} | "
          f"SMA(50): {fmt_price(tech.get('sma_50'))} | "
          f"SMA(200): {fmt_price(tech.get('sma_200'))}")
        w()

        w(f"RSI(14): {tech.get('rsi_14', 'N/A'):.1f}" if tech.get('rsi_14') else "RSI(14): N/A")
        w()

        macd = tech.get("macd", {})
        if macd.get("line") is not None:
            w(f"MACD: line {macd['line']:.4f} | signal {macd['signal']:.4f} | histogram {macd['histogram']:.4f}")
            w()

        bb = tech.get("bollinger", {})
        if bb.get("upper") is not None:
            w(f"Bollinger Bands: upper {fmt_price(bb['upper'])} | "
              f"middle {fmt_price(bb['middle'])} | lower {fmt_price(bb['lower'])} | "
              f"%B {bb['pct_b']:.2f}")
            w()

        if tech.get("signals"):
            w("Technical signals:")
            for sig in tech["signals"]:
                w(f"  -> {sig}")
            w()

    # --- Risk Assessment ---
    w("## Risk assessment")
    w()
    if risk:
        w(f"Beta: {risk.get('beta') or 'N/A'} | "
          f"Annualized volatility: {risk.get('annualized_volatility', 0):.2%} | "
          f"Max drawdown: {risk.get('max_drawdown', 0):.2%}")
        w(f"Sharpe ratio: {risk.get('sharpe_ratio') or 'N/A'} | "
          f"Sortino ratio: {risk.get('sortino_ratio') or 'N/A'} | "
          f"VaR (95%): {risk.get('var_95', 0):.2%}")
        w(f"Annualized return: {risk.get('annualized_return', 0):.2%}")
        w()

        if risk.get("signals"):
            w("Risk signals:")
            for sig in risk["signals"]:
                w(f"  -> {sig}")
            w()

    # --- Peer Comparison ---
    if peers.get("peers"):
        w("## Peer comparison")
        w()

        # Table header
        comp = peers["peers"]
        tickers_list = list(comp.keys())
        w(f"{'Ticker':<8} {'Name':<25} {'Mkt Cap':>12} {'P/E':>8} {'Margin':>8} {'Rev Gr':>8} {'1Y Perf':>8} {'Beta':>6}")
        w("-" * 95)

        for t in tickers_list:
            c = comp[t]
            if c.get("error"):
                w(f"{t:<8} {'(data unavailable)':<25}")
                continue
            pe_str = f"{c['pe_trailing']:.1f}" if c.get('pe_trailing') is not None else "N/A"
            beta_str = f"{c['beta']:.2f}" if c.get('beta') is not None else "N/A"
            w(f"{t:<8} "
              f"{(c.get('name', t) or t)[:24]:<25} "
              f"{fmt_large_number(c.get('market_cap')):>12} "
              f"{pe_str:>8} "
              f"{fmt_pct(c.get('profit_margin')):>8} "
              f"{fmt_pct(c.get('revenue_growth')):>8} "
              f"{fmt_pct(c.get('perf_1y')):>8} "
              f"{beta_str:>6}")
        w()

        # Rankings
        rankings = peers.get("rankings", {})
        if rankings:
            w(f"{ticker} peer ranking:")
            rank_labels = {
                "pe_trailing": "P/E (lower=better)",
                "profit_margin": "Profit margin",
                "revenue_growth": "Revenue growth",
                "roe": "Return on equity",
                "perf_1y": "1-year performance",
            }
            for metric, r in rankings.items():
                label = rank_labels.get(metric, metric)
                w(f"  {label}: #{r['rank']} of {r['of']}")
            w()

    # --- SEC EDGAR Filings ---
    sec = state.get("sec_filings", {})
    if sec:
        w("## SEC EDGAR filings")
        w(f"*Source: {sec.get('source', 'SEC EDGAR')}*")
        w()
        w(f"CIK: {sec.get('cik')} | SIC: {sec.get('sic')} ({sec.get('sic_description', '')})")
        w(f"State of incorporation: {sec.get('state_of_incorporation', 'N/A')} | "
          f"Fiscal year end: {sec.get('fiscal_year_end', 'N/A')}")
        w()

        filings = sec.get("filings", {})
        for form_type in ["10-K", "10-Q", "8-K"]:
            form_list = filings.get(form_type, [])
            if form_list:
                w(f"Recent {form_type} filings:")
                for f in form_list:
                    desc = f" — {f['description']}" if f.get("description") else ""
                    url_str = f" [{f['url']}]" if f.get("url") else ""
                    w(f"  {f['date']}  {f['form']}{desc}")
                w()

        xbrl = sec.get("xbrl_financials", {})
        if xbrl:
            w("Key financials (XBRL, from latest 10-K):")
            for key, data in xbrl.items():
                val = data.get("value")
                end = data.get("end_date", "")
                if val is not None:
                    if key in ("eps_basic", "eps_diluted"):
                        w(f"  {key}: ${val:.2f} ({end})")
                    elif abs(val) >= 1e12:
                        w(f"  {key}: ${val/1e12:.2f}T ({end})")
                    elif abs(val) >= 1e9:
                        w(f"  {key}: ${val/1e9:.2f}B ({end})")
                    elif abs(val) >= 1e6:
                        w(f"  {key}: ${val/1e6:.1f}M ({end})")
                    else:
                        w(f"  {key}: {val:,.0f} ({end})")
            w()

        if sec.get("signals"):
            w("SEC signals:")
            for sig in sec["signals"]:
                w(f"  -> {sig}")
            w()

    # --- Sentiment (Finnhub) ---
    sentiment = state.get("sentiment", {})
    if sentiment:
        w("## Investor sentiment")
        w(f"*Source: {sentiment.get('source', 'Finnhub')}*")
        w()

        rec = sentiment.get("recommendations", {})
        if rec:
            w("Analyst recommendations:")
            w(f"  Consensus: {(rec.get('consensus') or 'N/A').upper()} "
              f"({rec.get('total_analysts', 0)} analysts)")
            w(f"  Strong Buy: {rec.get('strong_buy', 0)} | "
              f"Buy: {rec.get('buy', 0)} | "
              f"Hold: {rec.get('hold', 0)} | "
              f"Sell: {rec.get('sell', 0)} | "
              f"Strong Sell: {rec.get('strong_sell', 0)}")
            w(f"  Bullish: {rec.get('bullish_pct', 0):.0%} | "
              f"Bearish: {rec.get('bearish_pct', 0):.0%}")
            if rec.get("trend"):
                w(f"  3-month trend: {rec['trend']} "
                  f"({rec.get('3m_shift', 0):+.0%} shift)")
            w()

        ins = sentiment.get("insider", {})
        if ins:
            w("Insider activity:")
            w(f"  Signal: {ins.get('signal', 'N/A')} "
              f"(MSPR: {ins.get('avg_mspr', 0):+.1f})")
            w(f"  Net shares change (3m): {ins.get('net_shares_change', 0):+,}")
            w()

        pt = sentiment.get("price_target", {})
        if pt.get("target_mean"):
            w("Analyst price targets:")
            w(f"  Mean: ${pt['target_mean']:.2f} | "
              f"Median: ${pt.get('target_median', 0):.2f}")
            w(f"  Range: ${pt.get('target_low', 0):.2f} - "
              f"${pt.get('target_high', 0):.2f}")
            if pt.get("implied_upside") is not None:
                w(f"  Implied upside/downside: {pt['implied_upside']:+.1%}")
            w()

        news = sentiment.get("news", {})
        if news.get("articles_last_7d"):
            w(f"News activity: {news['articles_last_7d']} articles in last 7 days")
            if news.get("top_headlines"):
                w("Recent headlines:")
                for h in news["top_headlines"][:5]:
                    w(f"  - {h.get('headline', '')} ({h.get('source', '')})")
            w()

        if sentiment.get("signals"):
            w("Sentiment signals:")
            for sig in sentiment["signals"]:
                w(f"  -> {sig}")
            w()

    # --- EODHD Sentiment ---
    eodhd = state.get("eodhd_sentiment", {})
    if eodhd:
        w("## News sentiment (EODHD)")
        w(f"*Source: {eodhd.get('source', 'EODHD')}*")
        w()

        sc = eodhd.get("scores", {})
        if sc:
            w(f"Overall sentiment (30d): {sc.get('overall') or 'N/A'}")
            w(f"  Average score: {(sc.get('avg_score_30d') or 0):+.4f} "
              f"(scale: -1.0 negative to +1.0 positive)")
            w(f"  Latest score: {(sc.get('latest_score') or 0):+.4f} "
              f"({sc.get('latest_date') or 'N/A'})")
            w(f"  7-day trend: {sc.get('trend') or 'N/A'} "
              f"(shift: {(sc.get('trend_shift_7d') or 0):+.4f})")
            w(f"  Articles analyzed (30d): {sc.get('total_articles_30d') or 0} "
              f"across {sc.get('days_with_data') or 0} days")
            w()

        news = eodhd.get("news", [])
        if news:
            w("Recent headlines (EODHD):")
            for article in news[:5]:
                w(f"  - {article.get('title', '')} ({article.get('date', '')[:10]})")
            w()

        if eodhd.get("signals"):
            w("EODHD signals:")
            for sig in eodhd["signals"]:
                w(f"  -> {sig}")
            w()

    # --- Agent Execution Log ---
    w("## Agent execution log")
    w()
    for entry in state.get("agent_log", []):
        icon = "OK" if entry["status"] == "ok" else entry["status"].upper()
        w(f"  [{icon}] {entry['agent']}")
    w()

    if state.get("errors"):
        w("Errors encountered:")
        for err in state["errors"]:
            w(f"  [!] {err}")
        w()

    w("---")
    w(f"Disclaimer: This report is generated from publicly available data via yfinance, SEC EDGAR, "
      f"Finnhub, and EODHD. It is not financial advice. Data may be delayed or incomplete. "
      f"Always conduct your own due diligence before making investment decisions.")

    report = "\n".join(lines)
    log_agent(state, "report_assembly", "ok")
    print("OK")
    return report


# ---------------------------------------------------------------------------
# PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_pipeline(ticker: str, peers: list[str] | None = None,
                 llm_provider: str | None = None, llm_key: str | None = None,
                 llm_model: str | None = None,
                 finnhub_key: str | None = None,
                 eodhd_key: str | None = None,
                 sec_email: str | None = None) -> tuple[dict, str]:
    """Execute the full agent pipeline and return (state, report)."""

    print(f"\n{'='*60}")
    print(f"  STOCK RESEARCH PIPELINE — {ticker.upper()}")
    print(f"  Agents: 11 | Source: yfinance + SEC EDGAR + Finnhub + EODHD")
    if llm_provider:
        print(f"  LLM: {llm_provider}/{llm_model or DEFAULT_MODELS.get(llm_provider, '?')}")
    else:
        print(f"  LLM: not configured (use --llm to enable)")
    fh_status = "configured" if (finnhub_key or os.environ.get("FINNHUB_API_KEY")) else "not configured (use --finnhub-key)"
    eo_status = "configured" if (eodhd_key or os.environ.get("EODHD_API_KEY")) else "not configured (use --eodhd-key)"
    print(f"  SEC EDGAR: always on (free, no key)")
    print(f"  Finnhub: {fh_status}")
    print(f"  EODHD: {eo_status}")
    print(f"{'='*60}\n")

    state = create_state(ticker, peers)

    # Execute agents in sequence
    state = agent_company_profile(state)
    state = agent_price_volume(state)
    state = agent_fundamentals(state)
    state = agent_technical(state)
    state = agent_risk(state)
    state = agent_peer_comparison(state)
    state = agent_sec_filings(state, sec_email)
    state = agent_finnhub_sentiment(state, finnhub_key)
    state = agent_eodhd_sentiment(state, eodhd_key)
    state = agent_llm_synthesis(state, llm_provider, llm_key, llm_model)
    report = agent_report_assembly(state)

    # Summary
    ok_count = sum(1 for a in state["agent_log"] if a["status"] == "ok")
    total = len(state["agent_log"])

    print(f"\n{'='*60}")
    print(f"  COMPLETE — {ok_count}/{total} agents succeeded")
    if state["errors"]:
        print(f"  Errors: {len(state['errors'])}")
    print(f"{'='*60}\n")

    return state, report


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stock Ticker Research & Evaluation — Agentic Pipeline",
        epilog="Example: python stock_research.py NVDA --llm openai --peers AMD INTC AVGO",
    )
    parser.add_argument("ticker", help="Stock ticker to research (e.g. NVDA)")
    parser.add_argument("--peers", nargs="*", default=[],
                        help="Peer tickers for comparison (auto-detected if omitted)")
    parser.add_argument("--output", "-o", help="Save report to markdown file")
    parser.add_argument("--json", help="Save raw state to JSON file")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress report output to terminal")

    # LLM options
    llm_group = parser.add_argument_group("LLM synthesis (optional)")
    llm_group.add_argument("--llm", choices=["openai", "anthropic"],
                           help="LLM provider for AI synthesis agent")
    llm_group.add_argument("--llm-key",
                           help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)")
    llm_group.add_argument("--llm-model",
                           help="Model override (default: gpt-4o / claude-sonnet-4-20250514)")

    # Finnhub options
    fh_group = parser.add_argument_group("Finnhub sentiment (optional)")
    fh_group.add_argument("--finnhub-key",
                          help="Finnhub API key (or set FINNHUB_API_KEY env var). "
                               "Free at https://finnhub.io")

    # EODHD options
    eo_group = parser.add_argument_group("EODHD sentiment (optional)")
    eo_group.add_argument("--eodhd-key",
                          help="EODHD API key (or set EODHD_API_KEY env var). "
                               "Free at https://eodhd.com/register")

    # SEC options
    sec_group = parser.add_argument_group("SEC EDGAR (always free)")
    sec_group.add_argument("--sec-email",
                           help="Your email for SEC User-Agent header (or set SEC_EMAIL env var). "
                                "SEC requires identification but no API key.")

    args = parser.parse_args()

    state, report = run_pipeline(
        args.ticker,
        args.peers,
        llm_provider=args.llm,
        llm_key=args.llm_key,
        llm_model=args.llm_model,
        finnhub_key=args.finnhub_key,
        eodhd_key=args.eodhd_key,
        sec_email=args.sec_email,
    )

    # Print report
    if not args.quiet:
        print(report)

    # Save markdown
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\n  Report saved: {args.output}")

    # Save JSON
    if args.json:
        # Remove non-serializable cached objects
        export = {k: v for k, v in state.items()
                  if not k.startswith("_")}
        with open(args.json, "w") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"  State saved: {args.json}")


if __name__ == "__main__":
    main()