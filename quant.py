#!/usr/bin/env python3
"""quant.py — end-to-end quantitative value pipeline.

Runs the full pipeline: strategy load → screener → price data → backtest →
LLM recommendation, then prints a summary and saves a JSON state file.

Usage
-----
  python quant.py --strategy graham_value --no-llm
  python quant.py --tickers MSFT AAPL KO JNJ --years 5
  python quant.py --universe my_tickers.txt --strategy buffett_quality
  python quant.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from config import REPORTS_DIR
from shared import create_pipeline_state, log_agent


# ---------------------------------------------------------------------------
# Step helpers (each prints its own progress line)
# ---------------------------------------------------------------------------

def _step_load_strategy(state: dict, strategy_name: str, years_override: int | None) -> dict:
    print(f"  [1/6] Loading strategy: {strategy_name}")
    from strategy import load_strategy_into_state
    state = load_strategy_into_state(state, strategy_name)
    if not state.get("strategy"):
        print(f"        ERROR: {state['errors'][-1]}", file=sys.stderr)
        sys.exit(1)
    if years_override is not None:
        state["strategy"]["backtest"]["years"] = years_override
        print(f"        (years overridden → {years_override})")
    s = state["strategy"]
    bt = s["backtest"]
    po = s["portfolio"]
    print(
        f"        {s['name']} | "
        f"{bt['years']}yr {bt['rebalance']} | "
        f"top {po['max_positions']} {po['position_size']}-weight"
    )
    return state


def _step_screen(
    state: dict,
    tickers_override: list[str] | None,
    universe_file: str | None,
    top_n: int,
) -> dict:
    if tickers_override:
        state["tickers"] = [t.upper() for t in tickers_override]
        print(f"  [2/6] Using provided tickers ({len(state['tickers'])}): "
              f"{', '.join(state['tickers'])}")
        # Populate a minimal screen section so decide.py has ticker context
        state["screen"] = {"top_picks": [{"ticker": t} for t in state["tickers"]]}
        log_agent(state, "screen", "skip", f"explicit tickers: {state['tickers']}")
        return state

    print(f"  [2/6] Screening universe…", end=" ", flush=True)
    from value import get_sp500_tickers, load_custom_universe, scan_universe

    if universe_file:
        raw_tickers = load_custom_universe(universe_file)
    else:
        raw_tickers = get_sp500_tickers()

    strategy_cfg = state.get("strategy", {})
    sc = strategy_cfg.get("screen", {})

    # Run the screener (suppresses its own header via show_progress=False for clean output)
    results = scan_universe(
        raw_tickers,
        min_market_cap=sc.get("min_market_cap", 1e9),
        show_progress=False,
    )

    # Apply strategy screen filters on top of the composite score
    min_score = sc.get("min_score", 0.0)
    max_pe    = sc.get("max_pe")
    max_pb    = sc.get("max_pb")
    min_roe   = sc.get("min_roe")
    max_de    = sc.get("max_debt_to_equity")

    filtered = []
    for r in results:
        if r["total_score"] < min_score:
            continue
        if max_pe  is not None and r.get("pe_trailing") and r["pe_trailing"] > max_pe:
            continue
        if max_pb  is not None and r.get("pb_ratio")    and r["pb_ratio"]    > max_pb:
            continue
        if min_roe is not None and r.get("roe")         and r["roe"]         < min_roe:
            continue
        if max_de  is not None and r.get("debt_to_equity") is not None \
                and r["debt_to_equity"] > max_de:
            continue
        filtered.append(r)

    top = filtered[:top_n]
    state["tickers"] = [r["ticker"] for r in top]
    state["screen"]  = {"top_picks": top, "total_scored": len(results), "filtered": len(filtered)}

    print(f"{len(results)} scored → {len(filtered)} passed filters → top {len(top)} selected")
    if top:
        preview = ", ".join(r["ticker"] for r in top[:8])
        suffix = "…" if len(top) > 8 else ""
        print(f"        {preview}{suffix}")

    log_agent(state, "screen", "ok",
              f"{len(results)} scored, {len(filtered)} filtered, {len(top)} selected")
    return state


def _step_price_data(state: dict, refresh: bool) -> dict:
    years = state.get("strategy", {}).get("backtest", {}).get("years", 10)
    n = len(state.get("tickers", []))
    print(f"  [3/6] Fetching price data ({n} tickers, {years}yr window)…", end=" ", flush=True)
    t0 = time.time()
    from data import load_price_data
    state = load_price_data(state, years=years + 1, refresh=refresh)
    loaded = len(state.get("price_data", {}))
    elapsed = time.time() - t0
    print(f"{loaded}/{n} loaded ({elapsed:.1f}s)")
    return state


def _step_backtest(state: dict) -> dict:
    strategy_cfg = state.get("strategy", {})
    bt = strategy_cfg.get("backtest", {})
    print(
        f"  [4/6] Running backtest "
        f"({bt.get('years')}yr {bt.get('rebalance')})…",
        end=" ", flush=True,
    )
    t0 = time.time()
    from backtest import run_backtest
    state = run_backtest(state)
    elapsed = time.time() - t0
    stats = state.get("backtest", {})
    if "error" in stats:
        print(f"ERROR — {stats['error']}")
    else:
        print(
            f"CAGR {stats.get('cagr', 0):.1%}  "
            f"Sharpe {stats.get('sharpe')}  "
            f"MaxDD {stats.get('max_drawdown', 0):.1%}  "
            f"({elapsed:.1f}s)"
        )
    return state


def _step_decision(state: dict, skip: bool) -> dict:
    if skip:
        print("  [5/6] Recommendation: skipped (--no-llm)")
        log_agent(state, "make_decision", "skip", "--no-llm flag")
        return state

    print("  [5/6] Generating recommendation…", end=" ", flush=True)
    from decide import make_decision
    state = make_decision(state)
    d = state.get("decision", {})
    if d.get("verdict"):
        print(f"{d['verdict']} ({d.get('conviction', '?')})")
    elif state["agent_log"] and state["agent_log"][-1]["status"] == "skip":
        print("skipped (ANTHROPIC_API_KEY not set)")
    else:
        print("error (see warnings below)")
    return state


def _step_report(state: dict, output_dir: Path, strategy_name: str) -> Path:
    print(f"  [6/6] Writing report…", end=" ", flush=True)
    from report import save_report
    path = save_report(state, output_dir, strategy_name)
    print(path.name)
    return path


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(state: dict, output_path: Path | None, report_path: Path | None = None) -> None:
    W = 54
    bar = "═" * W
    print(f"\n  {bar}")
    print(f"  {'QUANT PIPELINE SUMMARY':^{W}}")
    print(f"  {bar}")

    s = state.get("strategy", {})
    print(f"  {'Strategy':<14}: {s.get('name', 'N/A')}")
    print(f"  {'Tickers':<14}: {len(state.get('tickers', []))} in backtest universe")

    bt = state.get("backtest", {})
    if bt and "error" not in bt:
        print(f"\n  BACKTEST")
        print(f"  {'Period':<14}: {bt.get('start_date')} → {bt.get('end_date')}")
        print(f"  {'Tickers used':<14}: {bt.get('n_tickers')}")
        print(f"  {'CAGR':<14}: {bt.get('cagr', 0):.2%}")
        print(f"  {'Total return':<14}: {bt.get('total_return', 0):.2%}")
        print(f"  {'Sharpe':<14}: {bt.get('sharpe')}")
        print(f"  {'Sortino':<14}: {bt.get('sortino')}")
        print(f"  {'Max drawdown':<14}: {bt.get('max_drawdown', 0):.2%}")
        print(f"  {'Win rate':<14}: "
              f"{bt.get('win_rate', 0):.1%}" if bt.get('win_rate') is not None else "  Win rate       : N/A")
        spy_cagr = bt.get('spy_cagr')
        vs_spy   = bt.get('vs_spy')
        if spy_cagr is not None:
            sign = "+" if (vs_spy or 0) >= 0 else ""
            print(f"  {'SPY CAGR':<14}: {spy_cagr:.2%}  (vs SPY: {sign}{vs_spy:.2%})")
    elif bt.get("error"):
        print(f"\n  BACKTEST ERROR: {bt['error']}")

    d = state.get("decision", {})
    if d.get("verdict"):
        print(f"\n  RECOMMENDATION")
        print(f"  {'Verdict':<14}: {d['verdict']}  [{d.get('conviction','?')} conviction]")
        summary = d.get("summary", "")
        if summary:
            # Word-wrap to fit terminal width
            words = summary.split()
            line, lines_out = [], []
            for w in words:
                if sum(len(x) + 1 for x in line) + len(w) > W - 2:
                    lines_out.append(" ".join(line))
                    line = [w]
                else:
                    line.append(w)
            if line:
                lines_out.append(" ".join(line))
            print(f"  {'Summary':<14}: {lines_out[0]}")
            for l in lines_out[1:]:
                print(f"  {'':<16}{l}")
        if d.get("pros"):
            print(f"  Pros:")
            for p in d["pros"]:
                print(f"    + {p}")
        if d.get("cons"):
            print(f"  Cons:")
            for c in d["cons"]:
                print(f"    - {c}")
        if d.get("risks"):
            print(f"  Risks:")
            for r in d["risks"]:
                print(f"    ! {r}")
        if d.get("position_guidance"):
            print(f"  {'Guidance':<14}: {d['position_guidance']}")

    errors = [e for e in state.get("errors", []) if not e.startswith("run_backtest: dropped")]
    if errors:
        print(f"\n  WARNINGS ({len(errors)})")
        for e in errors[:5]:
            print(f"    ⚠  {e[:W]}")

    if output_path:
        print(f"\n  JSON  : {output_path}")
    if report_path:
        print(f"  Report: {report_path}")
    print(f"  {bar}\n")


# ---------------------------------------------------------------------------
# State serialisation
# ---------------------------------------------------------------------------

def _save_state(state: dict, output_dir: Path, strategy_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quant_{strategy_name}_{ts}.json"
    path = output_dir / filename
    # Drop non-serialisable price DataFrames before saving
    export = {k: v for k, v in state.items() if k != "price_data"}
    path.write_text(json.dumps(export, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quant.py",
        description="End-to-end quantitative value pipeline (Graham/Buffett).",
        epilog=(
            "Examples:\n"
            "  python quant.py --strategy graham_value --no-llm\n"
            "  python quant.py --tickers MSFT AAPL KO JNJ --years 5\n"
            "  python quant.py --universe tickers.txt --strategy buffett_quality"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--strategy", default="graham_value", metavar="NAME",
        help="Strategy name (in strategies/) or path to YAML  [default: graham_value]",
    )
    p.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Explicit tickers — bypasses the screener",
    )
    p.add_argument(
        "--top", type=int, default=None, metavar="N",
        help="Take top N screener results  [default: strategy max_positions]",
    )
    p.add_argument(
        "--universe", metavar="FILE",
        help="Custom ticker universe file (one ticker per line)",
    )
    p.add_argument(
        "--years", type=int, default=None, metavar="N",
        help="Override strategy backtest window (1–10)",
    )
    p.add_argument(
        "--refresh", action="store_true",
        help="Force re-fetch price data (ignore cache)",
    )
    p.add_argument(
        "--no-llm", action="store_true",
        help="Skip the LLM recommendation step",
    )
    p.add_argument(
        "--output", default=str(REPORTS_DIR), metavar="DIR",
        help=f"Directory for JSON output  [default: {REPORTS_DIR}]",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_name = args.strategy

    print(f"\n  {'='*54}")
    print(f"  {'QUANT VALUE PIPELINE':^54}")
    print(f"  {'='*54}\n")

    state = create_pipeline_state()

    # --- 1. Strategy ---
    state = _step_load_strategy(state, strategy_name, args.years)

    # Determine top_n (CLI flag overrides strategy default)
    top_n = args.top or state["strategy"]["portfolio"]["max_positions"]

    # --- 2. Screener / tickers ---
    state = _step_screen(state, args.tickers, args.universe, top_n)

    if not state.get("tickers"):
        print("\n  No tickers selected — exiting.", file=sys.stderr)
        sys.exit(1)

    # --- 3. Price data ---
    state = _step_price_data(state, refresh=args.refresh)

    # --- 4. Backtest ---
    state = _step_backtest(state)

    # --- 5. Decision ---
    state = _step_decision(state, skip=args.no_llm)

    # --- 6. Report ---
    report_path = _step_report(state, output_dir, strategy_name)

    # --- Save JSON + summarise ---
    output_path = _save_state(state, output_dir, strategy_name)
    _print_summary(state, output_path, report_path)


if __name__ == "__main__":
    main()
