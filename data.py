"""Historical price data layer with local parquet cache.

All price fetching in the pipeline goes through this module.
Never call yfinance history methods directly elsewhere.

Public API
----------
get_price_history(ticker, years, refresh)  -> pd.DataFrame
get_price_histories(tickers, years, refresh) -> dict[str, pd.DataFrame]
load_price_data(state, years, refresh)     -> dict  (pipeline integration)
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from config import CACHE_DIR, YFINANCE_RATE_LIMIT_EVERY, YFINANCE_SLEEP_SECONDS
from shared import get_yfinance_ticker, log_agent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Re-fetch if the newest cached row is older than this many days.
_STALE_DAYS = 2

# Columns we keep from yfinance — normalised to consistent names.
_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_ADJ_CLOSE_CANDIDATES = ["Adj Close", "Adj. Close", "adjclose"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.parquet"


def _is_fresh(df: pd.DataFrame, start: date) -> bool:
    """Return True if df covers the required window and isn't stale."""
    if df.empty:
        return False
    idx = df.index
    # Normalise index to dates for comparison
    if hasattr(idx, "date"):
        newest = idx[-1].date() if hasattr(idx[-1], "date") else idx[-1]
        oldest = idx[0].date() if hasattr(idx[0], "date") else idx[0]
    else:
        newest = idx[-1]
        oldest = idx[0]

    today = date.today()
    stale_cutoff = today - timedelta(days=_STALE_DAYS)
    required_start = start

    return (oldest <= required_start) and (newest >= stale_cutoff)


def _load_cache(ticker: str) -> pd.DataFrame | None:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    try:
        _cache_path(ticker).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_cache_path(ticker))
    except Exception:
        pass  # cache write failure is non-fatal


def _fetch_from_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch OHLCV + Adj Close from yfinance. Returns empty df on any error."""
    try:
        stock = get_yfinance_ticker(ticker)
        raw = stock.history(
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        raw.index.name = "Date"

        # Keep standard columns
        keep = [c for c in _COLUMNS if c in raw.columns]

        # Find adjusted close under any of the known names
        adj_col = next((c for c in _ADJ_CLOSE_CANDIDATES if c in raw.columns), None)
        if adj_col:
            keep.append(adj_col)
            raw = raw[keep].rename(columns={adj_col: "Adj Close"})
        else:
            # yfinance sometimes auto-adjusts even with auto_adjust=False;
            # fall back to Close as Adj Close so downstream always has the column.
            raw = raw[keep].copy()
            raw["Adj Close"] = raw["Close"]

        return raw.sort_index()

    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_price_history(
    ticker: str,
    years: int = 10,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV + Adj Close for *ticker* covering the last *years*.

    Loads from the local parquet cache when available and fresh; fetches from
    yfinance and updates the cache otherwise. Returns an empty DataFrame (never
    raises) if the data cannot be obtained.

    Parameters
    ----------
    ticker:
        Stock ticker symbol (case-insensitive).
    years:
        How many calendar years of history to retrieve.
    refresh:
        Force a re-fetch even if the cache is valid.
    """
    ticker = ticker.upper()
    end = date.today()
    start = date(end.year - years, end.month, end.day)

    # --- cache read ---
    if not refresh:
        cached = _load_cache(ticker)
        if cached is not None and _is_fresh(cached, start):
            # Slice to the requested window in case cache covers more history.
            mask = cached.index >= pd.Timestamp(start)
            return cached.loc[mask]

    # --- network fetch ---
    df = _fetch_from_yfinance(ticker, start, end)
    if not df.empty:
        _save_cache(ticker, df)

    return df


def get_price_histories(
    tickers: list[str],
    years: int = 10,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch price history for multiple tickers, returning {ticker: df}.

    Tickers that fail are omitted from the result. Applies the same rate-limit
    pause used in value.py (sleep every N calls).
    """
    results: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        df = get_price_history(ticker, years=years, refresh=refresh)
        if not df.empty:
            results[ticker.upper()] = df
        # Rate-limit courtesy pause on every Nth ticker (only for network hits)
        if (i + 1) % YFINANCE_RATE_LIMIT_EVERY == 0:
            time.sleep(YFINANCE_SLEEP_SECONDS)
    return results


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def load_price_data(
    state: dict,
    years: int = 10,
    refresh: bool = False,
) -> dict:
    """Populate state['price_data'] with {ticker: df} for state['tickers'].

    Follows the pipeline convention: errors go to state['errors'] and
    state['agent_log'], never raised to the caller.
    """
    tickers: list[str] = state.get("tickers") or []
    if not tickers:
        log_agent(state, "load_price_data", "skip", "no tickers in state")
        return state

    price_data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for i, ticker in enumerate(tickers):
        try:
            df = get_price_history(ticker, years=years, refresh=refresh)
            if df.empty:
                failed.append(ticker)
                state["errors"].append(f"load_price_data: no data for {ticker}")
            else:
                price_data[ticker] = df
        except Exception as exc:
            failed.append(ticker)
            state["errors"].append(f"load_price_data: {ticker} error: {exc}")

        if (i + 1) % YFINANCE_RATE_LIMIT_EVERY == 0:
            time.sleep(YFINANCE_SLEEP_SECONDS)

    state["price_data"] = price_data

    detail = (
        f"{len(price_data)}/{len(tickers)} loaded"
        + (f"; failed: {failed}" if failed else "")
    )
    status = "ok" if not failed else ("partial" if price_data else "error")
    log_agent(state, "load_price_data", status, detail)

    return state
