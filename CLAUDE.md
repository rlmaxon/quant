# CLAUDE.md

Context file for Claude Code. Read this at the start of every session before making changes.

## Project goal

An agentic quantitative analyst for U.S. public equities. The system gathers data, screens for value opportunities, performs deep per-ticker research, formulates a strategy from a thesis, backtests it on historical price data, and produces a structured buy/hold/avoid recommendation with explicit pros, cons, and risks. Investment style is long-term value (hybrid Graham/Buffett bias). The user runs this locally via CLI; output is markdown reports plus JSON state.

## Architecture: sequential pipeline (Option A)

The system is a sequential pipeline, **not** an LLM-orchestrated tool-use agent. Each module runs in fixed order and passes a state dict to the next. The LLM is called at specific synthesis points (research narrative, final recommendation), not as a top-level orchestrator. Do not refactor toward LLM-driven control flow without an explicit conversation about it.

**Flow:** `value.py` (screener) → `researcher.py` (deep dive on top N) → `strategy.py` (thesis → rules) → `backtest.py` (rules → performance stats) → `decide.py` (LLM synthesis) → combined report.

## Key design decisions (locked)

These were chosen deliberately. Don't relitigate without asking:

- **Backtest fidelity: Path 1.** Run today's screener, take top N, backtest a price-only buy-and-hold or annual-rebalance over 5–10 years. Has lookahead/survivorship bias. Acceptable for v1 because v1 priority is end-to-end plumbing, not backtest rigor. Path 2 (point-in-time fundamentals reconstruction) is a v1.5 concern.
- **Strategy parameterization: YAML.** Strategies live as YAML configs in `strategies/`, not Python classes. Lets the user iterate on thresholds without code changes.
- **Backtest engine: backtrader.** Event-driven, realistic enough, slower than vectorized alternatives. Cap usage at ~25 tickers and ~10 years for v1 to keep runtime manageable.
- **Time horizon: long-term value.** Daily bars, annual or semi-annual rebalance. Don't add intraday or high-frequency features.
- **"Directionally correct" is good enough for v1.** Don't gold-plate the backtest engine, transaction cost model, or risk analytics. Get the full pipeline working end-to-end first.

## Existing modules (do not break)

- **`value.py`** — Graham/Buffett screener over S&P 500 or custom universe. 100-point composite score across valuation, financial strength, profitability, income/safety. Outputs ranked list + optional CSV/JSON. Has `--deep-dive` flag that pipes top 3 into `researcher.py`.
- **`researcher.py`** — Per-ticker deep research pipeline with 11 sequential agents (profile, price, fundamentals, technical, risk, peer comparison, SEC filings, Finnhub sentiment, EODHD sentiment, LLM synthesis, report assembly). Outputs markdown report + JSON state. Uses yfinance + SEC EDGAR (free) + Finnhub + EODHD + OpenAI/Anthropic.

Both scripts read API keys from `.env` (loader is duplicated — Phase 0 fixes this).

## Target module layout

```
quant_analyst/
├── .env                    # API keys
├── CLAUDE.md               # this file
├── requirements.txt        # pinned deps
├── config.py               # NEW — paths, thresholds, defaults
├── shared.py               # NEW — extracted utils, state helpers
├── data.py                 # NEW — cached historical price wrapper
├── value.py                # EXISTING — screener
├── researcher.py           # EXISTING — per-ticker research
├── strategy.py             # NEW — YAML strategy loader + validator
├── backtest.py             # NEW — backtrader runner
├── decide.py               # NEW — LLM synthesis to recommendation
├── quant.py                # NEW — orchestrator CLI (main entry point)
├── strategies/             # NEW — YAML strategy configs
├── cache/                  # local price cache (parquet)
└── reports/                # output markdown + JSON + plots
```

## Conventions

- Python 3.10+. Type hints on all function signatures.
- All new modules import shared utilities from `shared.py`. Never duplicate `.env` loaders, `safe_get`, formatters, or state helpers.
- State flows as plain dicts, matching the pattern in `researcher.py` (`create_state` + `log_agent`). Each module mutates and returns the state dict.
- Errors are caught and logged into `state["errors"]` and `state["agent_log"]`. Do not raise to top level except in the orchestrator (`quant.py`).
- All file paths are relative to `PROJECT_ROOT` defined in `config.py`. Auto-create output dirs (`reports/`, `cache/`, `strategies/`) if missing.
- yfinance calls go through `shared.get_yfinance_ticker()` which handles rate limiting. Don't call `yf.Ticker()` directly in new modules.
- Historical price data goes through `data.py`. Don't fetch prices anywhere else.
- LLM calls use the Anthropic API (already wired in `researcher.py`). Default model: `claude-sonnet-4-20250514`. The user already has the key in `.env`.
- Markdown reports use the same section style as `researcher.py` (h2 headers, plain prose, no excessive tables).
- No new top-level dependencies without updating `requirements.txt` with a pinned minimum version.

## Pipeline state contract

All modules receive and return a state dict. Minimum required keys (`shared.create_pipeline_state` initializes these):

```python
{
    "ticker": str,           # primary ticker, if applicable
    "tickers": list[str],    # universe or top picks
    "timestamp": str,        # ISO format
    "agent_log": list[dict], # execution log entries
    "errors": list[str],     # accumulated errors
    # module-specific keys added as pipeline progresses:
    # "screen": {...}, "research": {...}, "strategy": {...},
    # "backtest": {...}, "decision": {...}
}
```

## Phased build plan

Build in this order. Don't start a phase until the prior is verified.

- **Phase 0** — Refactor: extract shared utilities to `shared.py` + `config.py`. No behavior change.
- **Phase 1** — Data layer: `data.py` with cached parquet historical prices.
- **Phase 2** — Strategy module: YAML schema + loader/validator in `strategy.py`.
- **Phase 3** — Backtest engine: `backtest.py` wrapping backtrader, returns standard stats dict (CAGR, Sharpe, Sortino, max DD, win rate, vs SPY).
- **Phase 4** — Decision module: `decide.py` LLM call producing structured recommendation.
- **Phase 5** — Orchestrator: `quant.py` CLI tying everything together.
- **Phase 6** — Combined markdown report stitching all outputs.

Detailed briefs for each phase are provided per session. Do not skip ahead.

## Constraints and known gotchas

- **yfinance is not point-in-time.** `.info` returns current snapshot fundamentals. Historical financials (`.financials`, `.balance_sheet`) only go back ~4 years and are jumpy. This is why v1 uses Path 1 (price-only backtest on currently-cheap names) — accept this and document it in reports, do not pretend the backtest is more rigorous than it is.
- **Survivorship bias.** The S&P 500 ticker list from Wikipedia is today's constituents. Backtests on this list inherit survivorship bias. Note this in any output that reports backtest results.
- **Backtrader is slow.** Event-driven simulation across many tickers gets expensive. Hard cap v1 at 25 tickers × 10 years.
- **yfinance rate limits.** Heavy scans can throttle. The existing `value.py` sleeps every 5 calls — preserve that pattern when batching.
- **API keys are optional except for LLM synthesis.** Finnhub, EODHD, and SEC email all degrade gracefully when missing. Maintain that — don't make any new module hard-fail on a missing optional key.

## What this system is not

To prevent scope creep, explicitly out of scope for v1:

- Real-time data, intraday bars, or live trading
- Order execution, broker integration, paper trading
- Multi-asset (no bonds, FX, crypto, options)
- Portfolio optimization (mean-variance, Black-Litterman, risk parity)
- ML-derived signals or factor models beyond Graham/Buffett scoring
- True point-in-time backtesting
- Web UI, dashboards, or scheduled jobs
- Multi-agent LLM orchestration

If the user asks for any of the above, flag it as a v2 conversation rather than silently expanding the build.

## Working with the user

- The user prefers a plan before code. When given a phase brief, propose the plan first, get approval, then implement.
- Acceptance criteria in each phase brief are non-negotiable — verify them before declaring done.
- The existing scripts (`value.py`, `researcher.py`) are the user's work and currently functional. Refactor carefully and verify identical output after Phase 0.
- This is a personal research tool, not production software. Don't over-engineer (no abstract base classes, no plugin systems, no DI containers). Plain functions and dicts are the default.
