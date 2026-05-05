"""Project paths, defaults, and tunable constants.

Centralizes anything that other modules might want to tweak. Importing this
module is side-effect-free except for ensuring output directories exist.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
ENV_FILE: Path = PROJECT_ROOT / ".env"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CACHE_DIR: Path = PROJECT_ROOT / "cache"
STRATEGIES_DIR: Path = PROJECT_ROOT / "strategies"

for _d in (REPORTS_DIR, CACHE_DIR, STRATEGIES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM defaults
# ---------------------------------------------------------------------------

DEFAULT_LLM_PROVIDER: str = "anthropic"
DEFAULT_LLM_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}


# ---------------------------------------------------------------------------
# Scoring / valuation constants
# ---------------------------------------------------------------------------

# 10-year Treasury yield used as risk-free baseline in earnings-yield scoring.
RISK_FREE_RATE: float = 0.045


# ---------------------------------------------------------------------------
# yfinance throttling
# ---------------------------------------------------------------------------

# When iterating across many tickers, sleep briefly every N calls.
YFINANCE_RATE_LIMIT_EVERY: int = 5
YFINANCE_SLEEP_SECONDS: float = 0.5
