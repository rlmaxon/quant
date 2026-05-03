# Quant Value Pipeline — User Guide

A personal CLI tool for long-term value stock analysis using the Graham/Buffett
methodology. Screens a universe of equities, backtests a configurable strategy,
and produces a structured markdown report with an optional LLM recommendation.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [API Keys](#api-keys)
4. [Quick Start](#quick-start)
5. [The Pipeline](#the-pipeline)
6. [CLI Reference — quant.py](#cli-reference--quantpy)
7. [Strategy YAML Configuration](#strategy-yaml-configuration)
8. [Output Files](#output-files)
9. [Individual Modules](#individual-modules)
10. [Known Limitations](#known-limitations)

---

## Overview

The pipeline runs six sequential steps and writes two output files — a JSON
state snapshot and a markdown report — to the `reports/` directory.

```
Strategy YAML
     │
     ▼
[1] Load strategy      — validate YAML, apply overrides
[2] Screen universe    — Graham/Buffett composite score (or use --tickers)
[3] Fetch price data   — yfinance, cached locally as parquet
[4] Backtest           — backtrader equal-weight periodic rebalance
[5] LLM recommendation — Anthropic API → BUY / HOLD / AVOID
[6] Write report       — markdown + JSON to reports/
```

Investment style is **long-term value** (Graham/Buffett). Daily bars, annual
or semi-annual rebalance, 5–10 year window.

---

## Installation

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd quant

# 2. Install dependencies
pip install -r requirements.txt
```

**Required Python:** 3.10 or later.

**Dependencies installed by `requirements.txt`:**

| Package | Purpose |
|---|---|
| `yfinance` | Market data (price history, fundamentals) |
| `requests` | HTTP calls (SEC EDGAR, Anthropic API) |
| `pandas` | DataFrame operations |
| `pyarrow` | Parquet cache read/write |
| `pyyaml` | Strategy YAML parsing |
| `backtrader` | Event-driven backtest engine |
| `numpy` | Stats computation (Sharpe, Sortino, drawdown) |

---

## API Keys

Place API keys in a `.env` file in the project root. The pipeline loads it
automatically on startup.

```bash
# .env — copy this template, fill in your keys
ANTHROPIC_API_KEY=sk-ant-...       # Required for LLM recommendation (Step 5)
FINNHUB_API_KEY=...                # Optional — used by researcher.py only
EODHD_API_KEY=...                  # Optional — used by researcher.py only
SEC_EMAIL=you@example.com          # Optional — used by researcher.py only
```

**Only `ANTHROPIC_API_KEY` is required for the main pipeline.** Without it,
Step 5 is silently skipped and the report's Recommendation section will say
"not generated." All other steps run normally.

Free keys:
- **Anthropic:** https://console.anthropic.com
- **Finnhub:** https://finnhub.io (researcher.py only)
- **EODHD:** https://eodhd.com/register (researcher.py only)

---

## Quick Start

### Fastest path — explicit tickers, no LLM

```bash
python quant.py --tickers MSFT AAPL KO JNJ --years 3 --no-llm
```

Skips the screener, fetches 3 years of price history, runs the backtest,
and writes `reports/quant_graham_value_<timestamp>.{md,json}`.

### With LLM recommendation

```bash
# Make sure ANTHROPIC_API_KEY is in .env, then:
python quant.py --tickers MSFT AAPL KO JNJ --years 3
```

### Full screener run (takes ~10 minutes)

```bash
python quant.py --strategy graham_value
```

Screens the S&P 500, selects the top 15 value stocks, fetches 10 years of
price data, backtests, and generates a recommendation.

### Concentrated quality screen

```bash
python quant.py --strategy buffett_quality --top 8
```

---

## The Pipeline

### Step 1 — Load Strategy

Reads a YAML file from `strategies/`, validates all fields, and fills in
defaults. If `--years` is passed on the CLI, it overrides the value in the
YAML.

```
[1/6] Loading strategy: graham_value
      graham_value | 10yr annual | top 15 equal-weight
```

### Step 2 — Screen Universe

Runs the Graham/Buffett composite screener (100-point scale) across the
chosen universe, then applies the strategy's screen thresholds to filter
the results.

```
[2/6] Screening universe… 412 scored → 28 passed filters → top 15 selected
      BRK-B, CVX, JNJ, KO, MO, …
```

**Bypassing the screener:** pass `--tickers` to provide an explicit list.
The screener step is skipped entirely.

**Custom universe:** pass `--universe my_tickers.txt` (one ticker per line)
to screen a list other than the S&P 500.

The composite score covers four categories (25 points each):

| Category | Key metrics |
|---|---|
| Valuation | P/E, P/B, Graham Number, earnings yield vs T-bill |
| Financial Strength | Current ratio, D/E, cash/debt ratio, beta |
| Profitability | Net margin, ROE, operating margin, EPS growth |
| Income & Safety | Dividend yield, payout ratio, FCF yield, 52-week position |

### Step 3 — Fetch Price Data

Downloads daily OHLCV + adjusted-close history for each ticker via yfinance
and caches it as a parquet file in `cache/`. Subsequent runs load from cache
(typically <0.1 s per ticker) unless `--refresh` is passed.

```
[3/6] Fetching price data (15 tickers, 10yr window)… 15/15 loaded (1.2s)
```

Cache files live at `cache/<TICKER>.parquet`. The cache is refreshed
automatically if the newest row is more than 2 days old.

### Step 4 — Backtest

Runs an equal-weight, periodic-rebalance buy-and-hold strategy using
backtrader. On each rebalance date, all positions are reset to
`(1 − cash_pct) / N` of portfolio value.

```
[4/6] Running backtest (10yr annual)… CAGR 11.3%  Sharpe 0.82  MaxDD -24.1%  (4.1s)
```

**Stats returned:**

| Metric | Description |
|---|---|
| CAGR | Compound annual growth rate |
| Total return | Gross return over the full window |
| Sharpe ratio | Annualised excess return / volatility |
| Sortino ratio | Annualised excess return / downside volatility |
| Max drawdown | Largest peak-to-trough decline |
| Win rate | Fraction of rebalance periods with positive return |
| SPY CAGR | Benchmark buy-and-hold return over same window |
| vs SPY | CAGR − SPY CAGR (positive = outperformed) |

Hard limits (v1): 25 tickers maximum, 10 years maximum.

### Step 5 — LLM Recommendation

Assembles the strategy config and backtest stats into a prompt and calls
the Anthropic API. The model returns a structured JSON object which is
parsed into the decision dict.

```
[5/6] Generating recommendation… HOLD (MEDIUM)
```

**Skipped when:**
- `--no-llm` flag is passed
- `ANTHROPIC_API_KEY` is not set (logs a skip, no error)

**Decision fields:** `verdict` (BUY / HOLD / AVOID), `conviction`
(HIGH / MEDIUM / LOW), `summary`, `pros`, `cons`, `risks`,
`position_guidance`.

### Step 6 — Write Report

Saves two files to `reports/` (or `--output` directory):

- `quant_<strategy>_<timestamp>.md` — formatted markdown report
- `quant_<strategy>_<timestamp>.json` — full pipeline state

```
[6/6] Writing report… quant_graham_value_20260503_121857.md
```

---

## CLI Reference — quant.py

```
python quant.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--strategy NAME` | `graham_value` | Strategy name (in `strategies/`) or path to a `.yaml` file |
| `--tickers T [T ...]` | *(screener)* | Explicit ticker list — bypasses the screener entirely |
| `--top N` | strategy `max_positions` | Number of screener results to take |
| `--universe FILE` | S&P 500 | Custom universe file for the screener (one ticker per line) |
| `--years N` | strategy YAML | Override backtest window in years (1–10) |
| `--refresh` | off | Force re-fetch price data, ignoring the local cache |
| `--no-llm` | off | Skip the Anthropic API call (Step 5) |
| `--output DIR` | `reports/` | Directory where `.md` and `.json` are saved |

### Common recipes

```bash
# Screen S&P 500, full 10yr backtest, with LLM (slow — ~10 min)
python quant.py

# Same but skip LLM
python quant.py --no-llm

# Concentrated quality screen, 5yr window
python quant.py --strategy buffett_quality --years 5

# Specific tickers, quick 2yr test
python quant.py --tickers AAPL MSFT KO --years 2 --no-llm

# Custom universe from a file
python quant.py --universe my_watchlist.txt --strategy graham_value

# Force fresh price data
python quant.py --tickers MSFT --years 3 --refresh

# Save output to a custom directory
python quant.py --tickers MSFT --no-llm --output ~/Desktop/quant_reports
```

---

## Strategy YAML Configuration

Strategies live in `strategies/*.yaml`. Two are included; copy and modify
to create your own.

### Full schema with defaults

```yaml
name: my_strategy           # required, must be unique
description: "..."          # optional, shown in reports

screen:
  min_score: 50.0           # min Graham/Buffett composite score (0-100)
  max_pe: null              # max trailing P/E  (null = no limit)
  max_pb: null              # max P/B
  min_roe: null             # min ROE as a fraction (0.10 = 10%)
  max_debt_to_equity: null  # max D/E as reported by yfinance (percent)
  min_market_cap: 1.0e9     # min market cap in dollars

portfolio:
  max_positions: 15         # 1–25 positions
  position_size: equal      # equal | cap_weighted
  max_position_pct: 0.15    # ignored when position_size = equal

backtest:
  years: 10                 # 1–10
  rebalance: annual         # annual | semiannual | quarterly
  commission: 0.001         # fraction per trade (0.001 = 0.1%)
  cash_pct: 0.02            # idle cash buffer (0.02 = 2%)
```

### Included strategies

**`graham_value`** — The default. Broad value screen with a composite score
floor of 60, P/E ≤ 20, P/B ≤ 3, ROE ≥ 10%. 15 equal-weight positions,
annual rebalance, 10-year window.

**`buffett_quality`** — Tighter quality focus. Score ≥ 65, ROE ≥ 15%, D/E
≤ 80%, accepts higher P/E (≤ 30) for moat businesses. 10 positions, semi-annual
rebalance, 5-year window.

### Creating a custom strategy

```bash
cp strategies/graham_value.yaml strategies/my_screen.yaml
# Edit my_screen.yaml, then:
python quant.py --strategy my_screen --tickers MSFT AAPL --years 3 --no-llm
```

Validation runs on load. If a value is out of range the pipeline exits with a
clear error message pointing to the offending field.

---

## Output Files

Both files share the same timestamp suffix so they are always paired.

### Markdown report (`quant_<strategy>_<timestamp>.md`)

Six sections:

| Section | Contents |
|---|---|
| Executive Summary | Verdict + thesis (or CAGR summary if no LLM) |
| Strategy Configuration | Screen thresholds and portfolio rules |
| Universe & Screener Results | Scored table of top picks |
| Backtest Performance | Stats table + bias disclaimer |
| Investment Recommendation | Verdict, pros, cons, risks, guidance |
| Pipeline Log | Agent step log + warnings |

### JSON state (`quant_<strategy>_<timestamp>.json`)

Full pipeline state dict (price DataFrames excluded). Useful for
post-processing or feeding into custom analysis. Top-level keys:

```json
{
  "tickers":   [...],
  "timestamp": "2026-05-03T12:18:56",
  "strategy":  { "name": ..., "screen": {...}, "portfolio": {...}, "backtest": {...} },
  "screen":    { "top_picks": [...], "total_scored": 412, "filtered": 28 },
  "backtest":  { "cagr": 0.113, "sharpe": 0.82, ... },
  "decision":  { "verdict": "HOLD", "conviction": "MEDIUM", "pros": [...], ... },
  "agent_log": [...],
  "errors":    [...]
}
```

---

## Individual Modules

The pipeline modules can also be imported and called directly.

### `value.py` — Screener

```bash
# Standalone screener (original interface unchanged)
python value.py --top 20
python value.py --sector Healthcare --output results.csv
python value.py --deep-dive          # runs researcher.py on top 3
```

### `researcher.py` — Deep-dive research

```bash
# Per-ticker research report with 11 agents
python researcher.py NVDA
python researcher.py AAPL --peers MSFT GOOGL --llm anthropic
python researcher.py TSLA --output tsla_report.md --json tsla_state.json
```

### `data.py` — Price data cache

```python
from data import get_price_history, get_price_histories

df = get_price_history("MSFT", years=5)          # single ticker
prices = get_price_histories(["MSFT","AAPL"], years=10)  # batch
```

### `strategy.py` — Strategy loader

```python
from strategy import load_strategy, list_strategies

print(list_strategies())           # ['buffett_quality', 'graham_value']
cfg = load_strategy("graham_value")
```

### `backtest.py` — Backtest engine

```python
from backtest import backtest_tickers
from data import get_price_histories
from strategy import load_strategy

price_data = get_price_histories(["MSFT", "AAPL", "KO"], years=5)
stats = backtest_tickers(["MSFT","AAPL","KO"], price_data, load_strategy("graham_value"))
print(stats["cagr"], stats["sharpe"])
```

### `decide.py` — LLM recommendation

```python
from decide import make_decision
from shared import create_pipeline_state

state = create_pipeline_state(tickers=["MSFT"])
state["strategy"] = ...
state["backtest"] = ...
state = make_decision(state)   # reads ANTHROPIC_API_KEY from env
print(state["decision"]["verdict"])
```

### `report.py` — Report generator

```python
from report import build_report, save_report
from pathlib import Path

md = build_report(state)              # returns markdown string
path = save_report(state, Path("reports/"), "my_strategy")
```

---

## Known Limitations

These are accepted v1 constraints, documented here for transparency.

### Survivorship bias

The S&P 500 ticker list is fetched from Wikipedia and reflects **today's
constituents**. Stocks that were in the index during the backtest window but
have since been removed (bankruptcies, acquisitions, de-listings) are not
included. This inflates backtest returns.

### Lookahead bias

The screener uses **current fundamentals** (today's P/E, ROE, etc.) to select
tickers, then backtests those same tickers over historical prices. In a real
backtest you would only know the fundamentals available at each historical date.
This also inflates returns for stocks whose fundamentals look good *now*
because they did well *then*.

### yfinance data quality

- `.info` returns a **current snapshot** — not point-in-time.
- Historical financials (`.financials`, `.balance_sheet`) only go back ~4 years
  and can be inconsistent around restatements.
- Price history is adjusted for splits and dividends but the adjustment
  methodology can change retroactively.

### Backtest engine

- **No short selling.** Long-only equal-weight only.
- **No slippage model.** Commission is applied but bid/ask spread is not.
- **Market orders at next-bar open.** Rebalance orders execute at the open
  of the day after the rebalance date.
- **Hard caps:** 25 tickers maximum, 10-year window maximum.

### What this tool is not

- Not a live trading system. No order execution or broker integration.
- Not a multi-asset tool. U.S. equities only.
- Not a rigorous research platform. Results are directionally indicative.

> **Bottom line:** treat backtest results as a rough signal that the strategy
> has historically selected stocks with certain return characteristics — not as
> a forecast of future performance or a precise measure of historical alpha.
