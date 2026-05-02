"""Shared utilities for the quant pipeline.

Anything that would otherwise be duplicated across `value.py`, `researcher.py`,
or new pipeline modules lives here: env loading, formatters, state helpers,
and the yfinance entry point.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any

import yfinance as yf

from config import ENV_FILE


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def load_env(path: Any = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (setdefault).

    Idempotent and silent if the file is missing.
    """
    env_path = path if path is not None else ENV_FILE
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


# Auto-load on import so callers don't have to remember.
load_env()


# ---------------------------------------------------------------------------
# yfinance entry point
# ---------------------------------------------------------------------------

def get_yfinance_ticker(ticker: str) -> yf.Ticker:
    """Return a yfinance Ticker. Single chokepoint for future rate limiting."""
    return yf.Ticker(ticker)


# ---------------------------------------------------------------------------
# Safe accessors
# ---------------------------------------------------------------------------

def safe_get(info: dict, key: str, default: Any = None) -> Any:
    """Get a value from a yfinance .info dict, treating None/NaN as missing."""
    val = info.get(key, default)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def fmt_large_number(n: Any) -> str:
    """Format large numbers as $1.5T / $230B / $45M / $1,234."""
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


def fmt_pct(val: Any) -> str:
    """Format float as percentage (0.123 -> 12.30%)."""
    if val is None:
        return "N/A"
    return f"{val:.2%}" if isinstance(val, float) else str(val)


def fmt_price(val: Any) -> str:
    if val is None:
        return "N/A"
    return f"${val:.2f}"


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

def create_pipeline_state(
    ticker: str | None = None,
    tickers: list[str] | None = None,
) -> dict:
    """Initialize the state dict that flows through pipeline modules.

    Matches the contract in CLAUDE.md: ticker, tickers, timestamp, agent_log,
    errors. Module-specific keys (`screen`, `research`, `strategy`,
    `backtest`, `decision`) are added as the pipeline progresses.
    """
    return {
        "ticker": ticker.upper() if ticker else None,
        "tickers": [t.upper() for t in (tickers or [])],
        "timestamp": datetime.now().isoformat(),
        "agent_log": [],
        "errors": [],
    }


def log_agent(state: dict, agent: str, status: str, detail: str = "") -> None:
    """Append an agent execution record to state['agent_log']."""
    state["agent_log"].append({
        "agent": agent,
        "status": status,
        "detail": detail,
        "time": datetime.now().isoformat(),
    })
