#!/usr/bin/env python3
"""
Value Stock Scanner — Graham/Buffett Methodology
==================================================
Screens a universe of stocks (S&P 500 or custom list) against
classic value investing criteria and ranks by composite score.

Usage:
    python value_scanner.py                          # Scan S&P 500
    python value_scanner.py --top 20                 # Show top 20
    python value_scanner.py --universe custom.txt    # Custom ticker list
    python value_scanner.py --sector Technology      # Filter by sector
    python value_scanner.py --output results.csv     # Export CSV
    python value_scanner.py --deep-dive              # Auto-run pipeline on top picks

Scoring (100 points max):
    Valuation (0-25)      — P/E, P/B, Graham Number, earnings yield
    Financial Strength    — current ratio, D/E, interest coverage
    Profitability (0-25)  — margins, ROE, earnings consistency
    Income & Safety (0-25) — dividend yield, payout ratio, FCF yield

Requires: pip install yfinance
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# Load .env if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and "=" in _line and not _line.startswith("#"):
            _key, _val = _line.split("=", 1)
            os.environ.setdefault(_key.strip(), _val.strip().strip('"').strip("'"))

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)


# ---------------------------------------------------------------------------
# S&P 500 UNIVERSE
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    try:
        import io
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = __import__("requests").get(url, timeout=15)
        # Parse first table — ticker is in column 0
        lines = resp.text
        tickers = []
        in_table = False
        for line in lines.split("\n"):
            if '<table class="wikitable sortable"' in line:
                in_table = True
            if in_table and '<td>' in line and '<a' in line:
                # Extract ticker from link text
                start = line.find('</a>')
                if start > 0:
                    # Look for ticker pattern
                    pass
            if in_table and 'NYSE:' in line or 'NASDAQ:' in line:
                pass
        # Fallback: use a requests + simple parse approach
        import re
        pattern = r'<td[^>]*><a[^>]*>([A-Z.]+)</a>'
        matches = re.findall(pattern, lines)
        # Filter to likely tickers (1-5 uppercase chars, maybe with .)
        tickers = [m for m in matches if 1 <= len(m) <= 5 and m.isalpha()]
        if len(tickers) > 400:
            return tickers[:505]
    except Exception:
        pass

    # Hardcoded fallback: major S&P 500 tickers (top ~100 by weight)
    return _SP500_FALLBACK


_SP500_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "V", "XOM", "MA", "JNJ", "PG", "HD", "COST", "MRK",
    "ABBV", "NFLX", "CRM", "AMD", "BAC", "CVX", "KO", "PEP", "WMT", "TMO",
    "LIN", "CSCO", "ACN", "MCD", "ABT", "DHR", "ADBE", "WFC", "TXN", "PM",
    "CMCSA", "VZ", "NEE", "INTC", "IBM", "RTX", "LOW", "AMGN", "HON", "QCOM",
    "UNP", "CAT", "SPGI", "GE", "BA", "ISRG", "INTU", "ELV", "BKNG", "AMAT",
    "DE", "T", "ADP", "MDLZ", "SYK", "TJX", "PLD", "MMC", "GS", "GILD",
    "REGN", "CI", "ADI", "VRTX", "BDX", "CB", "SCHW", "ETN", "PGR", "LRCX",
    "SO", "DUK", "MO", "ZTS", "CME", "CL", "ITW", "MMM", "FDX", "USB",
    "TGT", "NOC", "EMR", "APD", "PXD", "SLB", "F", "GM", "DAL", "DOW",
]


def load_custom_universe(filepath: str) -> list[str]:
    """Load tickers from a text file (one per line)."""
    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: Universe file not found: {filepath}")
        sys.exit(1)
    tickers = []
    for line in path.read_text().splitlines():
        t = line.strip().upper()
        if t and not t.startswith("#"):
            tickers.append(t)
    return tickers


# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_stock_data(ticker: str) -> dict | None:
    """Fetch fundamental data for a single ticker via yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or not info.get("marketCap"):
            return None

        def safe(key, default=None):
            val = info.get(key, default)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return val

        return {
            "ticker": ticker,
            "name": safe("longName", ticker),
            "sector": safe("sector", "N/A"),
            "industry": safe("industry", "N/A"),
            "market_cap": safe("marketCap"),
            "price": safe("currentPrice") or safe("regularMarketPrice"),
            # Valuation
            "pe_trailing": safe("trailingPE"),
            "pe_forward": safe("forwardPE"),
            "pb_ratio": safe("priceToBook"),
            "ps_ratio": safe("priceToSalesTrailing12Months"),
            "peg_ratio": safe("pegRatio"),
            "ev_ebitda": safe("enterpriseToEbitda"),
            # Per share
            "eps_trailing": safe("trailingEps"),
            "eps_forward": safe("forwardEps"),
            "book_value": safe("bookValue"),
            # Profitability
            "profit_margin": safe("profitMargins"),
            "operating_margin": safe("operatingMargins"),
            "gross_margin": safe("grossMargins"),
            "roe": safe("returnOnEquity"),
            "roa": safe("returnOnAssets"),
            # Growth
            "revenue_growth": safe("revenueGrowth"),
            "earnings_growth": safe("earningsGrowth"),
            # Balance sheet
            "debt_to_equity": safe("debtToEquity"),
            "current_ratio": safe("currentRatio"),
            "total_cash": safe("totalCash"),
            "total_debt": safe("totalDebt"),
            # Income
            "dividend_yield": safe("dividendYield"),
            "payout_ratio": safe("payoutRatio"),
            "free_cash_flow": safe("freeCashflow"),
            # Risk
            "beta": safe("beta"),
            "52_week_high": safe("fiftyTwoWeekHigh"),
            "52_week_low": safe("fiftyTwoWeekLow"),
        }

    except Exception:
        return None


# ---------------------------------------------------------------------------
# GRAHAM/BUFFETT SCORING MODEL
# ---------------------------------------------------------------------------

def compute_graham_number(eps: float | None, book_value: float | None) -> float | None:
    """Graham Number = sqrt(22.5 × EPS × Book Value)."""
    if eps is None or book_value is None or eps <= 0 or book_value <= 0:
        return None
    return (22.5 * eps * book_value) ** 0.5


def score_valuation(data: dict) -> tuple[float, list[str]]:
    """Score 0-25 on valuation metrics. Lower = more undervalued = higher score."""
    score = 0.0
    reasons = []

    # P/E ratio (0-8 points)
    pe = data.get("pe_trailing")
    if pe is not None and pe > 0:
        if pe < 10:
            score += 8; reasons.append(f"P/E {pe:.1f} (deeply undervalued)")
        elif pe < 15:
            score += 6; reasons.append(f"P/E {pe:.1f} (undervalued)")
        elif pe < 20:
            score += 3; reasons.append(f"P/E {pe:.1f} (fair)")
        elif pe < 30:
            score += 1; reasons.append(f"P/E {pe:.1f} (elevated)")
        else:
            reasons.append(f"P/E {pe:.1f} (expensive)")
    else:
        reasons.append("P/E N/A or negative")

    # P/B ratio (0-7 points)
    pb = data.get("pb_ratio")
    if pb is not None and pb > 0:
        if pb < 1.0:
            score += 7; reasons.append(f"P/B {pb:.2f} (below book value)")
        elif pb < 1.5:
            score += 5; reasons.append(f"P/B {pb:.2f} (near book value)")
        elif pb < 3.0:
            score += 2; reasons.append(f"P/B {pb:.2f} (fair)")
        else:
            reasons.append(f"P/B {pb:.2f} (expensive)")

    # Graham Number — margin of safety (0-5 points)
    graham = compute_graham_number(data.get("eps_trailing"), data.get("book_value"))
    price = data.get("price")
    if graham and price and price > 0:
        discount = (graham - price) / graham
        if discount > 0.3:
            score += 5; reasons.append(f"Graham #{graham:.0f} ({discount:.0%} margin of safety)")
        elif discount > 0.1:
            score += 3; reasons.append(f"Graham #{graham:.0f} ({discount:.0%} margin)")
        elif discount > 0:
            score += 1; reasons.append(f"Graham #{graham:.0f} (slim margin)")
        else:
            reasons.append(f"Graham #{graham:.0f} (above intrinsic, {discount:.0%})")

    # Earnings yield vs risk-free rate (0-5 points)
    if pe and pe > 0:
        earnings_yield = 1 / pe
        risk_free = 0.045  # ~4.5% 10Y Treasury
        spread = earnings_yield - risk_free
        if spread > 0.04:
            score += 5; reasons.append(f"Earnings yield {earnings_yield:.1%} (+{spread:.1%} vs T-bill)")
        elif spread > 0.02:
            score += 3; reasons.append(f"Earnings yield {earnings_yield:.1%}")
        elif spread > 0:
            score += 1; reasons.append(f"Earnings yield {earnings_yield:.1%} (thin spread)")

    return min(score, 25), reasons


def score_financial_strength(data: dict) -> tuple[float, list[str]]:
    """Score 0-25 on balance sheet health."""
    score = 0.0
    reasons = []

    # Current ratio (0-8 points)
    cr = data.get("current_ratio")
    if cr is not None:
        if cr >= 2.0:
            score += 8; reasons.append(f"Current ratio {cr:.2f} (strong)")
        elif cr >= 1.5:
            score += 6; reasons.append(f"Current ratio {cr:.2f} (healthy)")
        elif cr >= 1.0:
            score += 3; reasons.append(f"Current ratio {cr:.2f} (adequate)")
        else:
            reasons.append(f"Current ratio {cr:.2f} (liquidity risk)")

    # Debt/Equity (0-8 points)
    de = data.get("debt_to_equity")
    if de is not None:
        if de < 30:
            score += 8; reasons.append(f"D/E {de:.0f}% (minimal debt)")
        elif de < 50:
            score += 6; reasons.append(f"D/E {de:.0f}% (conservative)")
        elif de < 100:
            score += 3; reasons.append(f"D/E {de:.0f}% (moderate)")
        elif de < 200:
            score += 1; reasons.append(f"D/E {de:.0f}% (leveraged)")
        else:
            reasons.append(f"D/E {de:.0f}% (highly leveraged)")

    # Cash vs debt (0-5 points)
    cash = data.get("total_cash")
    debt = data.get("total_debt")
    if cash and debt and debt > 0:
        ratio = cash / debt
        if ratio > 1.0:
            score += 5; reasons.append(f"Cash/Debt {ratio:.2f}x (net cash)")
        elif ratio > 0.5:
            score += 3; reasons.append(f"Cash/Debt {ratio:.2f}x")
        elif ratio > 0.2:
            score += 1

    # Beta — stability (0-4 points)
    beta = data.get("beta")
    if beta is not None:
        if 0.5 <= beta <= 1.0:
            score += 4; reasons.append(f"Beta {beta:.2f} (defensive)")
        elif 1.0 < beta <= 1.3:
            score += 2; reasons.append(f"Beta {beta:.2f} (moderate)")
        elif beta < 0.5 or beta > 1.5:
            reasons.append(f"Beta {beta:.2f} (volatile)")

    return min(score, 25), reasons


def score_profitability(data: dict) -> tuple[float, list[str]]:
    """Score 0-25 on earnings quality and margins."""
    score = 0.0
    reasons = []

    # Profit margin (0-7 points)
    margin = data.get("profit_margin")
    if margin is not None:
        if margin > 0.20:
            score += 7; reasons.append(f"Net margin {margin:.1%} (excellent)")
        elif margin > 0.10:
            score += 5; reasons.append(f"Net margin {margin:.1%} (strong)")
        elif margin > 0.05:
            score += 3; reasons.append(f"Net margin {margin:.1%} (adequate)")
        elif margin > 0:
            score += 1; reasons.append(f"Net margin {margin:.1%} (thin)")
        else:
            reasons.append(f"Net margin {margin:.1%} (negative)")

    # ROE — Buffett's favorite (0-7 points)
    roe = data.get("roe")
    if roe is not None:
        if roe > 0.20:
            score += 7; reasons.append(f"ROE {roe:.1%} (exceptional)")
        elif roe > 0.15:
            score += 5; reasons.append(f"ROE {roe:.1%} (strong)")
        elif roe > 0.10:
            score += 3; reasons.append(f"ROE {roe:.1%} (fair)")
        elif roe > 0:
            score += 1; reasons.append(f"ROE {roe:.1%} (weak)")
        else:
            reasons.append(f"ROE {roe:.1%} (negative)")

    # Operating margin (0-5 points)
    op_margin = data.get("operating_margin")
    if op_margin is not None:
        if op_margin > 0.25:
            score += 5
        elif op_margin > 0.15:
            score += 3
        elif op_margin > 0.05:
            score += 1

    # Earnings consistency — positive EPS is baseline (0-3 points)
    eps = data.get("eps_trailing")
    fwd_eps = data.get("eps_forward")
    if eps and eps > 0:
        score += 2; reasons.append(f"EPS ${eps:.2f} (positive)")
        if fwd_eps and fwd_eps > eps:
            score += 1; reasons.append("Forward EPS growing")

    # Revenue growth — Buffett wants steady (0-3 points)
    rev_growth = data.get("revenue_growth")
    if rev_growth is not None:
        if 0.03 <= rev_growth <= 0.25:
            score += 3; reasons.append(f"Revenue growth {rev_growth:.1%} (steady)")
        elif rev_growth > 0.25:
            score += 1; reasons.append(f"Revenue growth {rev_growth:.1%} (high but sustainable?)")
        elif rev_growth > 0:
            score += 1
        else:
            reasons.append(f"Revenue growth {rev_growth:.1%} (declining)")

    return min(score, 25), reasons


def score_income_safety(data: dict) -> tuple[float, list[str]]:
    """Score 0-25 on dividend and cash flow safety."""
    score = 0.0
    reasons = []

    # Dividend yield (0-8 points)
    div_yield = data.get("dividend_yield")
    if div_yield is not None and div_yield > 0:
        if 0.02 <= div_yield <= 0.06:
            score += 8; reasons.append(f"Dividend yield {div_yield:.2%} (sweet spot)")
        elif div_yield > 0.06:
            score += 4; reasons.append(f"Dividend yield {div_yield:.2%} (high — check sustainability)")
        elif div_yield > 0.01:
            score += 5; reasons.append(f"Dividend yield {div_yield:.2%}")
        else:
            score += 2; reasons.append(f"Dividend yield {div_yield:.2%} (token)")
    else:
        reasons.append("No dividend")

    # Payout ratio — sustainable? (0-7 points)
    payout = data.get("payout_ratio")
    if payout is not None and div_yield and div_yield > 0:
        if 0.2 <= payout <= 0.6:
            score += 7; reasons.append(f"Payout ratio {payout:.0%} (sustainable)")
        elif payout < 0.2:
            score += 5; reasons.append(f"Payout ratio {payout:.0%} (room to grow)")
        elif payout <= 0.8:
            score += 3; reasons.append(f"Payout ratio {payout:.0%} (stretched)")
        else:
            score += 0; reasons.append(f"Payout ratio {payout:.0%} (unsustainable)")

    # Free cash flow yield (0-7 points)
    fcf = data.get("free_cash_flow")
    mcap = data.get("market_cap")
    if fcf and mcap and mcap > 0:
        fcf_yield = fcf / mcap
        if fcf_yield > 0.08:
            score += 7; reasons.append(f"FCF yield {fcf_yield:.2%} (excellent)")
        elif fcf_yield > 0.05:
            score += 5; reasons.append(f"FCF yield {fcf_yield:.2%} (strong)")
        elif fcf_yield > 0.03:
            score += 3; reasons.append(f"FCF yield {fcf_yield:.2%} (fair)")
        elif fcf_yield > 0:
            score += 1; reasons.append(f"FCF yield {fcf_yield:.2%} (thin)")
        else:
            reasons.append(f"FCF yield {fcf_yield:.2%} (negative FCF)")

    # 52-week position — buying low (0-3 points)
    price = data.get("price")
    high = data.get("52_week_high")
    low = data.get("52_week_low")
    if price and high and low and high > low:
        position = (price - low) / (high - low)
        if position < 0.3:
            score += 3; reasons.append(f"Near 52-week low ({position:.0%} of range)")
        elif position < 0.5:
            score += 1; reasons.append(f"Lower half of 52-week range")

    return min(score, 25), reasons


def score_stock(data: dict) -> dict:
    """Compute composite Graham/Buffett value score (0-100)."""
    v_score, v_reasons = score_valuation(data)
    f_score, f_reasons = score_financial_strength(data)
    p_score, p_reasons = score_profitability(data)
    i_score, i_reasons = score_income_safety(data)

    total = v_score + f_score + p_score + i_score

    # Graham Number
    graham = compute_graham_number(data.get("eps_trailing"), data.get("book_value"))
    price = data.get("price")
    margin_of_safety = None
    if graham and price and price > 0:
        margin_of_safety = (graham - price) / graham

    # Grade
    if total >= 75:
        grade = "A"
        label = "Strong value candidate"
    elif total >= 60:
        grade = "B"
        label = "Good value characteristics"
    elif total >= 45:
        grade = "C"
        label = "Mixed — some value, some concerns"
    elif total >= 30:
        grade = "D"
        label = "Weak value case"
    else:
        grade = "F"
        label = "Not a value stock"

    return {
        "ticker": data["ticker"],
        "name": data.get("name", data["ticker"]),
        "sector": data.get("sector", "N/A"),
        "price": price,
        "total_score": round(total, 1),
        "grade": grade,
        "label": label,
        "valuation_score": round(v_score, 1),
        "financial_score": round(f_score, 1),
        "profitability_score": round(p_score, 1),
        "income_score": round(i_score, 1),
        "graham_number": round(graham, 2) if graham else None,
        "margin_of_safety": round(margin_of_safety, 4) if margin_of_safety is not None else None,
        "pe_trailing": data.get("pe_trailing"),
        "pb_ratio": data.get("pb_ratio"),
        "dividend_yield": data.get("dividend_yield"),
        "debt_to_equity": data.get("debt_to_equity"),
        "roe": data.get("roe"),
        "details": {
            "valuation": v_reasons,
            "financial_strength": f_reasons,
            "profitability": p_reasons,
            "income_safety": i_reasons,
        },
    }


# ---------------------------------------------------------------------------
# SCANNER ENGINE
# ---------------------------------------------------------------------------

def scan_universe(tickers: list[str], sector_filter: str | None = None,
                  min_market_cap: float = 5e8,
                  show_progress: bool = True) -> list[dict]:
    """Scan a list of tickers and return scored results."""

    print(f"\n{'='*60}")
    print(f"  GRAHAM/BUFFETT VALUE SCANNER")
    print(f"  Universe: {len(tickers)} tickers")
    if sector_filter:
        print(f"  Sector filter: {sector_filter}")
    print(f"  Min market cap: ${min_market_cap/1e9:.0f}B")
    print(f"  Methodology: Graham Number, P/E, P/B, D/E, ROE, FCF yield")
    print(f"{'='*60}\n")

    results = []
    errors = 0
    skipped = 0

    for i, ticker in enumerate(tickers):
        if show_progress and (i + 1) % 10 == 0:
            print(f"  Scanning... {i+1}/{len(tickers)} "
                  f"({len(results)} scored, {skipped} skipped, {errors} errors)",
                  flush=True)

        data = fetch_stock_data(ticker)
        if data is None:
            errors += 1
            continue

        # Apply filters
        mcap = data.get("market_cap")
        if mcap and mcap < min_market_cap:
            skipped += 1
            continue

        if sector_filter and data.get("sector", "").lower() != sector_filter.lower():
            skipped += 1
            continue

        # Must have basic data to score
        if not data.get("pe_trailing") and not data.get("pb_ratio"):
            skipped += 1
            continue

        scored = score_stock(data)
        results.append(scored)

        # Small delay to avoid hammering yfinance
        if (i + 1) % 5 == 0:
            time.sleep(0.5)

    # Sort by total score descending
    results.sort(key=lambda x: x["total_score"], reverse=True)

    print(f"\n  Scan complete: {len(results)} scored, {skipped} filtered, {errors} errors")
    return results


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def print_results(results: list[dict], top_n: int = 25):
    """Print ranked results as a formatted table."""

    print(f"\n{'='*110}")
    print(f"  TOP {min(top_n, len(results))} VALUE STOCKS — Graham/Buffett Score")
    print(f"{'='*110}")
    print(f"{'#':>3}  {'Ticker':<7} {'Name':<22} {'Sector':<16} "
          f"{'Score':>5} {'Grd':>3}  {'P/E':>6} {'P/B':>6} {'D/E':>6} "
          f"{'ROE':>7} {'DivY':>6} {'Graham#':>8} {'Margin':>7}")
    print("-" * 110)

    for i, r in enumerate(results[:top_n]):
        pe = f"{r['pe_trailing']:.1f}" if r.get("pe_trailing") else "N/A"
        pb = f"{r['pb_ratio']:.2f}" if r.get("pb_ratio") else "N/A"
        de = f"{r['debt_to_equity']:.0f}%" if r.get("debt_to_equity") is not None else "N/A"
        roe = f"{r['roe']:.1%}" if r.get("roe") is not None else "N/A"
        div = f"{r['dividend_yield']:.2%}" if r.get("dividend_yield") else "—"
        graham = f"${r['graham_number']:.0f}" if r.get("graham_number") else "N/A"
        mos = f"{r['margin_of_safety']:.0%}" if r.get("margin_of_safety") is not None else "N/A"

        print(f"{i+1:>3}  {r['ticker']:<7} {r['name'][:21]:<22} {r['sector'][:15]:<16} "
              f"{r['total_score']:>5.1f} {r['grade']:>3}  {pe:>6} {pb:>6} {de:>6} "
              f"{roe:>7} {div:>6} {graham:>8} {mos:>7}")

    print(f"\n{'='*110}")
    print(f"  Scoring: Valuation (25) + Financial Strength (25) + Profitability (25) + Income & Safety (25) = 100")
    print(f"  A = 75+  |  B = 60-74  |  C = 45-59  |  D = 30-44  |  F = <30")
    print(f"  Margin = margin of safety (positive = price below Graham Number)")
    print(f"{'='*110}\n")


def print_detail(result: dict):
    """Print detailed scoring breakdown for a single stock."""
    print(f"\n  {result['ticker']} — {result['name']} ({result['sector']})")
    print(f"  Total: {result['total_score']}/100 ({result['grade']}) — {result['label']}")
    print(f"  Price: ${result['price']:.2f}" if result.get("price") else "")
    if result.get("graham_number"):
        print(f"  Graham Number: ${result['graham_number']:.2f} "
              f"(margin of safety: {result.get('margin_of_safety', 0):.1%})")
    print()

    for section, label in [("valuation", "Valuation"),
                           ("financial_strength", "Financial Strength"),
                           ("profitability", "Profitability"),
                           ("income_safety", "Income & Safety")]:
        score_key = section.split("_")[0] + "_score" if "_" not in section else section + "_score"
        # Map to actual keys
        key_map = {"valuation": "valuation_score", "financial_strength": "financial_score",
                   "profitability": "profitability_score", "income_safety": "income_score"}
        s = result.get(key_map.get(section, section + "_score"), 0)
        print(f"  {label}: {s}/25")
        for reason in result["details"].get(section, []):
            print(f"    - {reason}")
        print()


def export_csv(results: list[dict], filepath: str):
    """Export results to CSV."""
    if not results:
        return

    fields = ["ticker", "name", "sector", "price", "total_score", "grade",
              "valuation_score", "financial_score", "profitability_score", "income_score",
              "graham_number", "margin_of_safety",
              "pe_trailing", "pb_ratio", "dividend_yield", "debt_to_equity", "roe"]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"  Exported {len(results)} results to {filepath}")


def export_json(results: list[dict], filepath: str):
    """Export results with full detail to JSON."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Exported {len(results)} results to {filepath}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Graham/Buffett Value Stock Scanner",
        epilog="Example: python value_scanner.py --top 20 --sector Technology --output results.csv",
    )
    parser.add_argument("--universe", help="Path to text file with tickers (one per line). "
                        "Default: S&P 500")
    parser.add_argument("--top", type=int, default=25, help="Show top N results (default: 25)")
    parser.add_argument("--sector", help="Filter by sector (e.g. Technology, Healthcare)")
    parser.add_argument("--min-cap", type=float, default=1e9,
                        help="Minimum market cap in dollars (default: 1B)")
    parser.add_argument("--output", "-o", help="Export results to CSV file")
    parser.add_argument("--json", help="Export results with detail to JSON file")
    parser.add_argument("--detail", type=int, default=5,
                        help="Show detailed breakdown for top N (default: 5)")
    parser.add_argument("--deep-dive", action="store_true",
                        help="Run stock_research.py on top 3 picks (if available)")
    args = parser.parse_args()

    # Load universe
    if args.universe:
        tickers = load_custom_universe(args.universe)
    else:
        print("  Loading S&P 500 universe...")
        tickers = get_sp500_tickers()

    # Scan
    results = scan_universe(tickers, sector_filter=args.sector,
                            min_market_cap=args.min_cap)

    if not results:
        print("  No stocks matched the criteria.")
        return

    # Print summary table
    print_results(results, top_n=args.top)

    # Print detailed breakdown for top picks
    if args.detail and args.detail > 0:
        print(f"  DETAILED BREAKDOWN — Top {min(args.detail, len(results))}")
        print(f"  {'='*50}")
        for r in results[:args.detail]:
            print_detail(r)

    # Export
    if args.output:
        export_csv(results, args.output)
    if args.json:
        export_json(results, args.json)

    # Deep-dive integration
    if args.deep_dive:
        script_dir = Path(__file__).parent
        research_script = script_dir / "stock_research.py"
        if research_script.exists():
            top_tickers = [r["ticker"] for r in results[:3]]
            print(f"\n  Running deep-dive research on top picks: {', '.join(top_tickers)}")
            print(f"  {'='*50}\n")
            for t in top_tickers:
                os.system(f"python {research_script} {t} -o {t.lower()}_value_report.md")
        else:
            print(f"  stock_research.py not found in {script_dir}")
            print(f"  Run manually: python stock_research.py TICKER")


if __name__ == "__main__":
    main()