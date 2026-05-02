"""Backtest engine wrapping backtrader.

All backtesting goes through this module. The core function backtest_tickers()
runs a periodic-rebalance buy-and-hold strategy and returns a standard stats
dict. run_backtest() integrates with the pipeline state.

Backtest fidelity note (v1 Path 1)
-----------------------------------
The universe is today's screened tickers (survivorship bias). Fundamentals
used for screening are current snapshots (lookahead bias). Results are
directionally indicative, not rigorous. See CLAUDE.md for rationale.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import backtrader as bt
import numpy as np
import pandas as pd

from config import RISK_FREE_RATE
from data import get_price_history
from shared import log_agent

_INITIAL_CASH = 100_000.0
_MAX_TICKERS = 25           # hard cap per CLAUDE.md
_MAX_YEARS = 10             # hard cap per CLAUDE.md


# ---------------------------------------------------------------------------
# Rebalance date helpers
# ---------------------------------------------------------------------------

def _target_dates(start: date, end: date, rebalance: str) -> list[date]:
    """Calendar anchor dates for the chosen rebalance frequency."""
    month_starts = {"annual": [1], "semiannual": [1, 7], "quarterly": [1, 4, 7, 10]}
    months = month_starts[rebalance]
    out: list[date] = []
    year = start.year
    while True:
        for m in months:
            d = date(year, m, 1)
            if d > end:
                return out
            if d >= start:
                out.append(d)
        year += 1


def _to_trading_date(target: date, trading_days: set[date]) -> date | None:
    """First actual trading day >= target, within 15 calendar days."""
    for offset in range(15):
        candidate = target + timedelta(days=offset)
        if candidate in trading_days:
            return candidate
    return None


def _rebalance_schedule(
    start: date, end: date, rebalance: str, trading_days: set[date]
) -> list[date]:
    """Return the actual trading-day rebalance dates for this backtest window."""
    targets = _target_dates(start, end, rebalance)
    seen: set[date] = set()
    result: list[date] = []
    for t in targets:
        actual = _to_trading_date(t, trading_days)
        if actual and actual not in seen:
            seen.add(actual)
            result.append(actual)
    return result


# ---------------------------------------------------------------------------
# Backtrader strategy
# ---------------------------------------------------------------------------

class _BtRebalanceStrategy(bt.Strategy):
    """Periodic equal-weight rebalancer used internally by backtest_tickers."""

    params = (
        ("rebalance_set", set()),   # set of date objects
        ("per_ticker_pct", 0.065),  # target weight per ticker (0-1)
    )

    def __init__(self):
        self._dates: list[date] = []
        self._values: list[float] = []
        self._period_values: list[float] = []   # value at each rebalance (for win rate)

    def next(self):
        dt: date = self.data.datetime.date(0)
        val: float = self.broker.getvalue()
        self._dates.append(dt)
        self._values.append(val)

        if dt in self.p.rebalance_set:
            self._period_values.append(val)
            for d in self.datas:
                self.order_target_percent(data=d, target=self.p.per_ticker_pct)


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def _compute_stats(
    dates: list[date],
    values: list[float],
    period_values: list[float],
    spy_cagr: float | None,
    n_tickers: int,
    years: int,
) -> dict:
    if len(values) < 2:
        return {"error": "insufficient equity curve data"}

    initial = values[0]
    final = values[-1]
    actual_years = (dates[-1] - dates[0]).days / 365.25

    total_return = (final / initial) - 1.0
    cagr = ((final / initial) ** (1.0 / actual_years) - 1.0) if actual_years > 0 else 0.0

    arr = np.array(values, dtype=float)
    daily_returns = np.diff(arr) / arr[:-1]
    daily_rfr = RISK_FREE_RATE / 252.0
    excess = daily_returns - daily_rfr

    sharpe: float | None = None
    sortino: float | None = None
    if len(daily_returns) >= 60:
        std = float(daily_returns.std())
        if std > 0:
            sharpe = round(float(excess.mean() / std * 252 ** 0.5), 4)
        downside = daily_returns[daily_returns < daily_rfr]
        if len(downside) > 1:
            dd_std = float(downside.std())
            if dd_std > 0:
                sortino = round(float(excess.mean() / dd_std * 252 ** 0.5), 4)

    # Max drawdown (peak-to-trough, expressed as a negative fraction)
    peak = arr[0]
    max_dd = 0.0
    for v in arr:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Win rate: fraction of rebalance periods where ending value > starting value
    win_rate: float | None = None
    if len(period_values) >= 2:
        wins = sum(b > a for a, b in zip(period_values[:-1], period_values[1:]))
        win_rate = round(wins / (len(period_values) - 1), 4)

    vs_spy = round(cagr - spy_cagr, 6) if spy_cagr is not None else None

    return {
        "cagr":         round(cagr, 6),
        "total_return": round(total_return, 6),
        "sharpe":       sharpe,
        "sortino":      sortino,
        "max_drawdown": round(max_dd, 6),
        "win_rate":     win_rate,
        "spy_cagr":     round(spy_cagr, 6) if spy_cagr is not None else None,
        "vs_spy":       vs_spy,
        "n_tickers":    n_tickers,
        "years":        years,
        "start_date":   dates[0].isoformat(),
        "end_date":     dates[-1].isoformat(),
        "note": (
            "Survivorship bias: universe is current screener output. "
            "Lookahead bias: fundamentals are today's snapshots. "
            "Results are directionally indicative only."
        ),
    }


# ---------------------------------------------------------------------------
# SPY benchmark
# ---------------------------------------------------------------------------

def _get_spy_cagr(start: date, end: date, years: int) -> float | None:
    try:
        df = get_price_history("SPY", years=min(years + 1, _MAX_YEARS))
        if df.empty:
            return None
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        sliced = df.loc[mask]
        if len(sliced) < 2:
            return None
        p0 = float(sliced["Adj Close"].iloc[0])
        p1 = float(sliced["Adj Close"].iloc[-1])
        actual_years = (end - start).days / 365.25
        if actual_years <= 0 or p0 <= 0:
            return None
        return (p1 / p0) ** (1.0 / actual_years) - 1.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core backtest function
# ---------------------------------------------------------------------------

def backtest_tickers(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    strategy_cfg: dict,
    dropped: list[str] | None = None,
) -> dict:
    """Run the backtest and return a stats dict.

    Parameters
    ----------
    tickers:
        Ordered list of candidate tickers (screener output).
    price_data:
        {ticker: df} mapping from data.py.
    strategy_cfg:
        Validated strategy dict from strategy.py.
    dropped:
        If supplied, tickers excluded for insufficient history are appended.
    """
    bt_cfg = strategy_cfg["backtest"]
    port_cfg = strategy_cfg["portfolio"]

    years: int = min(bt_cfg["years"], _MAX_YEARS)
    rebalance: str = bt_cfg["rebalance"]
    commission: float = bt_cfg["commission"]
    cash_pct: float = bt_cfg["cash_pct"]
    max_positions: int = port_cfg["max_positions"]

    end = date.today()
    start = date(end.year - years, end.month, end.day)
    # Require at least 80 % of expected trading days so thin data is excluded.
    min_bars = int(years * 252 * 0.8)

    # --- select valid tickers ---
    valid: list[str] = []
    for t in tickers[:_MAX_TICKERS]:
        df = price_data.get(t)
        if df is None or df.empty:
            if dropped is not None:
                dropped.append(t)
            continue
        bars_in_window = int((df.index >= pd.Timestamp(start)).sum())
        if bars_in_window < min_bars:
            if dropped is not None:
                dropped.append(t)
            continue
        valid.append(t)
        if len(valid) >= max_positions:
            break

    if not valid:
        return {"error": "no tickers with sufficient history for backtest"}

    # --- build trading-day set from first valid ticker ---
    ref_df = price_data[valid[0]]
    trading_days: set[date] = {
        ts.date() for ts in ref_df.index if ts.date() >= start
    }

    # Rebalance schedule; always open on the first available trading day.
    schedule = _rebalance_schedule(start, end, rebalance, trading_days)
    first_day = min(trading_days)
    schedule_set: set[date] = set(schedule)
    schedule_set.add(first_day)

    per_ticker_pct = (1.0 - cash_pct) / len(valid)

    # --- build cerebro ---
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(_INITIAL_CASH)
    cerebro.broker.setcommission(commission=commission)

    added: list[str] = []
    for ticker in valid:
        df = price_data[ticker]
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        df_slice = df.loc[mask][["Open", "High", "Low", "Close", "Volume"]].copy()
        if df_slice.empty:
            continue
        feed = bt.feeds.PandasData(dataname=df_slice, openinterest=-1)
        cerebro.adddata(feed, name=ticker)
        added.append(ticker)

    if not added:
        return {"error": "no data feeds could be constructed"}

    cerebro.addstrategy(
        _BtRebalanceStrategy,
        rebalance_set=schedule_set,
        per_ticker_pct=per_ticker_pct,
    )

    # --- run ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = cerebro.run(runonce=False)

    strat = results[0]

    spy_cagr = _get_spy_cagr(start, end, years)

    return _compute_stats(
        strat._dates,
        strat._values,
        strat._period_values,
        spy_cagr,
        n_tickers=len(added),
        years=years,
    )


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def run_backtest(state: dict) -> dict:
    """Run the backtest and populate state['backtest'].

    Reads state['tickers'], state['price_data'], state['strategy'].
    Errors are captured in state['errors'] and state['agent_log'].
    """
    tickers: list[str] = state.get("tickers") or []
    price_data: dict = state.get("price_data") or {}
    strategy_cfg: dict = state.get("strategy") or {}

    state.setdefault("backtest", {})

    for key, label in [
        (tickers, "tickers"),
        (price_data, "price_data"),
        (strategy_cfg, "strategy"),
    ]:
        if not key:
            msg = f"run_backtest: {label} missing from state"
            state["errors"].append(msg)
            log_agent(state, "run_backtest", "error", msg)
            return state

    dropped: list[str] = []
    try:
        stats = backtest_tickers(tickers, price_data, strategy_cfg, dropped=dropped)
    except Exception as exc:
        msg = f"run_backtest: {exc}"
        state["errors"].append(msg)
        log_agent(state, "run_backtest", "error", msg)
        state.setdefault("backtest", {})
        return state

    for t in dropped:
        state["errors"].append(f"run_backtest: dropped {t} — insufficient history")

    state["backtest"] = stats

    if "error" in stats:
        log_agent(state, "run_backtest", "error", stats["error"])
    else:
        log_agent(
            state,
            "run_backtest",
            "ok",
            f"CAGR {stats['cagr']:.2%}  Sharpe {stats['sharpe']}  "
            f"MaxDD {stats['max_drawdown']:.2%}  vs SPY {stats.get('vs_spy', 'N/A')}",
        )

    return state
