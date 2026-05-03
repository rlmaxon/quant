"""Strategy loader and validator.

Strategies are YAML files in strategies/. This module loads, validates, and
fills defaults so downstream modules (backtest.py, decide.py) always receive
a fully-specified strategy dict with known types.

Public API
----------
load_strategy(name_or_path)           -> dict
list_strategies()                     -> list[str]
validate_strategy(raw)                -> dict   # raises ValueError on bad input
load_strategy_into_state(state, name) -> dict   # pipeline integration
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from config import STRATEGIES_DIR
from shared import log_agent


# ---------------------------------------------------------------------------
# Schema: defaults and validation rules
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    # Top-level identity
    "name": None,           # required — no default
    "description": "",

    # Screening thresholds applied to value.py output before backtesting.
    # null means "no limit" for upper/lower bounds.
    "screen": {
        "min_score": 50.0,          # minimum Graham/Buffett composite score
        "max_pe": None,             # max trailing P/E  (null = no limit)
        "max_pb": None,             # max P/B
        "min_roe": None,            # min ROE (fraction, e.g. 0.10 = 10%)
        "max_debt_to_equity": None, # max D/E (percent as reported by yfinance)
        "min_market_cap": 1.0e9,    # min market cap in dollars
    },

    # Portfolio construction rules.
    "portfolio": {
        "max_positions": 15,
        "position_size": "equal",   # equal | cap_weighted
        "max_position_pct": 0.15,   # ignored when position_size=equal
    },

    # Backtest parameters.
    "backtest": {
        "years": 10,
        "rebalance": "annual",      # annual | semiannual | quarterly
        "commission": 0.001,        # fraction per trade
        "cash_pct": 0.02,           # idle cash buffer fraction
    },
}

_REBALANCE_VALUES = {"annual", "semiannual", "quarterly"}
_POSITION_SIZE_VALUES = {"equal", "cap_weighted"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _err(field: str, msg: str) -> ValueError:
    return ValueError(f"strategy[{field}]: {msg}")


def _coerce_float_or_none(val: Any, field: str) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        raise _err(field, f"expected a number or null, got {val!r}")


def _coerce_float(val: Any, field: str) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        raise _err(field, f"expected a number, got {val!r}")


def _coerce_int(val: Any, field: str) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        raise _err(field, f"expected an integer, got {val!r}")


def validate_strategy(raw: dict) -> dict:
    """Validate *raw* (parsed YAML dict), fill defaults, and return clean dict.

    Raises ValueError with a human-readable message on the first problem found.
    Unknown top-level keys are preserved (forward compatibility) but unknown
    keys inside `screen`, `portfolio`, and `backtest` are silently dropped to
    keep the output shape stable.
    """
    raw = copy.deepcopy(raw)

    # --- name (required) ---
    name = raw.get("name")
    if not name or not str(name).strip():
        raise _err("name", "required and must be a non-empty string")

    out: dict[str, Any] = {
        "name": str(name).strip(),
        "description": str(raw.get("description", _DEFAULTS["description"])),
    }

    # --- screen ---
    raw_screen = raw.get("screen", {}) or {}
    d_screen = _DEFAULTS["screen"]

    min_score = _coerce_float(raw_screen.get("min_score", d_screen["min_score"]), "screen.min_score")
    if not (0.0 <= min_score <= 100.0):
        raise _err("screen.min_score", f"must be between 0 and 100, got {min_score}")

    out["screen"] = {
        "min_score": min_score,
        "max_pe": _coerce_float_or_none(raw_screen.get("max_pe", d_screen["max_pe"]), "screen.max_pe"),
        "max_pb": _coerce_float_or_none(raw_screen.get("max_pb", d_screen["max_pb"]), "screen.max_pb"),
        "min_roe": _coerce_float_or_none(raw_screen.get("min_roe", d_screen["min_roe"]), "screen.min_roe"),
        "max_debt_to_equity": _coerce_float_or_none(
            raw_screen.get("max_debt_to_equity", d_screen["max_debt_to_equity"]),
            "screen.max_debt_to_equity",
        ),
        "min_market_cap": _coerce_float(
            raw_screen.get("min_market_cap", d_screen["min_market_cap"]),
            "screen.min_market_cap",
        ),
    }

    # --- portfolio ---
    raw_port = raw.get("portfolio", {}) or {}
    d_port = _DEFAULTS["portfolio"]

    max_positions = _coerce_int(raw_port.get("max_positions", d_port["max_positions"]), "portfolio.max_positions")
    if not (1 <= max_positions <= 25):
        raise _err("portfolio.max_positions", f"must be between 1 and 25, got {max_positions}")

    position_size = str(raw_port.get("position_size", d_port["position_size"])).strip().lower()
    if position_size not in _POSITION_SIZE_VALUES:
        raise _err("portfolio.position_size", f"must be one of {sorted(_POSITION_SIZE_VALUES)}, got {position_size!r}")

    max_position_pct = _coerce_float(
        raw_port.get("max_position_pct", d_port["max_position_pct"]),
        "portfolio.max_position_pct",
    )
    if not (0.0 < max_position_pct <= 1.0):
        raise _err("portfolio.max_position_pct", f"must be between 0 (exclusive) and 1, got {max_position_pct}")

    out["portfolio"] = {
        "max_positions": max_positions,
        "position_size": position_size,
        "max_position_pct": max_position_pct,
    }

    # --- backtest ---
    raw_bt = raw.get("backtest", {}) or {}
    d_bt = _DEFAULTS["backtest"]

    years = _coerce_int(raw_bt.get("years", d_bt["years"]), "backtest.years")
    if not (1 <= years <= 10):
        raise _err("backtest.years", f"must be between 1 and 10, got {years}")

    rebalance = str(raw_bt.get("rebalance", d_bt["rebalance"])).strip().lower()
    if rebalance not in _REBALANCE_VALUES:
        raise _err("backtest.rebalance", f"must be one of {sorted(_REBALANCE_VALUES)}, got {rebalance!r}")

    commission = _coerce_float(raw_bt.get("commission", d_bt["commission"]), "backtest.commission")
    if not (0.0 <= commission <= 0.05):
        raise _err("backtest.commission", f"must be between 0 and 0.05, got {commission}")

    cash_pct = _coerce_float(raw_bt.get("cash_pct", d_bt["cash_pct"]), "backtest.cash_pct")
    if not (0.0 <= cash_pct <= 0.5):
        raise _err("backtest.cash_pct", f"must be between 0 and 0.5, got {cash_pct}")

    out["backtest"] = {
        "years": years,
        "rebalance": rebalance,
        "commission": commission,
        "cash_pct": cash_pct,
    }

    return out


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _resolve_path(name_or_path: str) -> Path:
    """Return the YAML file path for a strategy name or direct file path."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml"):
        # Treat as a path (absolute or relative to cwd)
        return p if p.is_absolute() else Path.cwd() / p
    # Treat as a bare name — look in strategies/
    for suffix in (".yaml", ".yml"):
        candidate = STRATEGIES_DIR / f"{name_or_path}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Strategy {name_or_path!r} not found in {STRATEGIES_DIR}. "
        f"Available: {list_strategies() or ['(none)']}"
    )


def load_strategy(name_or_path: str) -> dict:
    """Load, parse, and validate a strategy YAML.

    Parameters
    ----------
    name_or_path:
        Either a bare strategy name (e.g. ``"graham_value"``) resolved against
        ``strategies/``, or a path to a ``.yaml`` / ``.yml`` file.

    Returns
    -------
    dict
        Fully validated strategy dict with all defaults filled in.

    Raises
    ------
    FileNotFoundError
        If the YAML file cannot be located.
    ValueError
        If the YAML content fails validation.
    """
    path = _resolve_path(name_or_path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return validate_strategy(raw)


def list_strategies() -> list[str]:
    """Return bare names (no extension) of strategy YAML files in strategies/."""
    names = []
    for p in sorted(STRATEGIES_DIR.iterdir()):
        if p.suffix in (".yaml", ".yml"):
            names.append(p.stem)
    return names


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def load_strategy_into_state(state: dict, name_or_path: str) -> dict:
    """Load and validate a strategy, storing it in state['strategy'].

    Errors are caught and written to state['errors'] / state['agent_log'].
    """
    try:
        strategy = load_strategy(name_or_path)
        state["strategy"] = strategy
        log_agent(
            state, "load_strategy", "ok",
            f"{strategy['name']} — {strategy['backtest']['years']}yr "
            f"{strategy['backtest']['rebalance']} rebalance, "
            f"top {strategy['portfolio']['max_positions']} positions",
        )
    except (FileNotFoundError, ValueError) as exc:
        state.setdefault("strategy", {})
        state["errors"].append(f"load_strategy: {exc}")
        log_agent(state, "load_strategy", "error", str(exc))
    return state
