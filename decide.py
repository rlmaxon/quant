"""LLM-powered decision synthesis.

Takes the assembled pipeline state (strategy config, backtest stats, screener
top picks, optional research narratives) and calls the Anthropic API to produce
a structured BUY / HOLD / AVOID recommendation.

Gracefully skips if ANTHROPIC_API_KEY is absent — the rest of the pipeline
continues unaffected.

Public API
----------
make_decision(state, api_key, model) -> dict   # pipeline entry point
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

import requests

from config import DEFAULT_LLM_MODELS
from shared import log_agent

_DEFAULT_MODEL = DEFAULT_LLM_MODELS["anthropic"]
_MAX_TOKENS = 1200
_TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior long-term value investment analyst with expertise in the
Graham/Buffett methodology. You evaluate quantitative screening results and
backtested strategy performance to produce clear, actionable recommendations.

Your output must be a single JSON object with exactly these keys:
  verdict           — "BUY", "HOLD", or "AVOID"
  conviction        — "HIGH", "MEDIUM", or "LOW"
  summary           — 2-3 sentence investment thesis (plain prose)
  pros              — list of 3-5 concise strings, strongest positives
  cons              — list of 3-5 concise strings, key weaknesses
  risks             — list of 3-5 concise strings, material risks
  position_guidance — 1-2 sentence note on sizing, timing, or conditions

Rules:
- Output ONLY the JSON object, no markdown fences, no extra text.
- Be direct. Avoid hedging language that says nothing.
- Ground every point in the data provided; do not invent metrics.
- Flag the survivorship/lookahead bias in risks if backtest results appear.
"""

_USER_PROMPT = """\
Evaluate this value strategy and produce a structured recommendation.

{context}

Respond with the JSON object as specified in your instructions.
"""

# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _fmt_pct(v: Any) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.1%}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_f(v: Any, decimals: int = 2) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _build_context(state: dict) -> str:
    """Assemble a compact, token-efficient context string from pipeline state."""
    parts: list[str] = []

    # --- strategy ---
    strategy = state.get("strategy") or {}
    if strategy:
        bt = strategy.get("backtest", {})
        sc = strategy.get("screen", {})
        po = strategy.get("portfolio", {})
        parts.append(
            f"STRATEGY: {strategy.get('name', 'unnamed')}\n"
            f"  Description: {strategy.get('description', '').strip()}\n"
            f"  Backtest window: {bt.get('years', '?')} years, "
            f"{bt.get('rebalance', '?')} rebalance\n"
            f"  Screen: min_score={sc.get('min_score')}, "
            f"max_pe={sc.get('max_pe')}, max_pb={sc.get('max_pb')}, "
            f"min_roe={_fmt_pct(sc.get('min_roe'))}\n"
            f"  Portfolio: {po.get('max_positions')} positions, "
            f"{po.get('position_size')} weight"
        )

    # --- backtest stats ---
    bt_stats = state.get("backtest") or {}
    if bt_stats and "error" not in bt_stats:
        parts.append(
            f"BACKTEST RESULTS ({bt_stats.get('start_date')} to {bt_stats.get('end_date')}):\n"
            f"  CAGR:          {_fmt_pct(bt_stats.get('cagr'))}\n"
            f"  Total return:  {_fmt_pct(bt_stats.get('total_return'))}\n"
            f"  Sharpe ratio:  {_fmt_f(bt_stats.get('sharpe'))}\n"
            f"  Sortino ratio: {_fmt_f(bt_stats.get('sortino'))}\n"
            f"  Max drawdown:  {_fmt_pct(bt_stats.get('max_drawdown'))}\n"
            f"  Win rate:      {_fmt_pct(bt_stats.get('win_rate'))}\n"
            f"  SPY CAGR:      {_fmt_pct(bt_stats.get('spy_cagr'))}\n"
            f"  vs SPY:        {_fmt_pct(bt_stats.get('vs_spy'))}\n"
            f"  Tickers used:  {bt_stats.get('n_tickers')}\n"
            f"  Note: {bt_stats.get('note', '')}"
        )
    elif bt_stats.get("error"):
        parts.append(f"BACKTEST: failed — {bt_stats['error']}")

    # --- top screener picks ---
    screen = state.get("screen") or {}
    top_picks = screen.get("top_picks") or []
    if top_picks:
        lines = ["TOP SCREENER PICKS (Graham/Buffett composite score):"]
        for r in top_picks[:10]:
            lines.append(
                f"  {r.get('ticker', '?'):<7} score={r.get('total_score', '?'):>5}  "
                f"grade={r.get('grade', '?')}  "
                f"P/E={r.get('pe_trailing') or 'N/A'}  "
                f"P/B={r.get('pb_ratio') or 'N/A'}  "
                f"ROE={_fmt_pct(r.get('roe'))}  "
                f"{r.get('name', '')[:30]}"
            )
        parts.append("\n".join(lines))

    # --- tickers list (fallback if no screen section) ---
    elif state.get("tickers"):
        parts.append(f"TICKERS IN BACKTEST: {', '.join(state['tickers'])}")

    # --- research summaries (truncated) ---
    research = state.get("research") or {}
    if research:
        summaries: list[str] = []
        for ticker, rdata in list(research.items())[:3]:
            narrative = (
                rdata.get("synthesis", {}).get("narrative")
                or rdata.get("narrative")
                or ""
            )
            if narrative:
                # Keep first 400 chars to stay within token budget
                snippet = narrative[:400].rsplit(" ", 1)[0] + "…"
                summaries.append(f"  {ticker}: {snippet}")
        if summaries:
            parts.append("RESEARCH SUMMARIES:\n" + "\n".join(summaries))

    # --- accumulated errors / warnings ---
    errors = state.get("errors") or []
    if errors:
        parts.append(
            "PIPELINE WARNINGS:\n"
            + "\n".join(f"  - {e}" for e in errors[:5])
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Anthropic API call
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": _MAX_TOKENS,
            "temperature": _TEMPERATURE,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "verdict", "conviction", "summary", "pros", "cons", "risks",
    "position_guidance",
}
_VALID_VERDICTS = {"BUY", "HOLD", "AVOID"}
_VALID_CONVICTIONS = {"HIGH", "MEDIUM", "LOW"}


def _parse_response(text: str) -> dict:
    """Extract and validate the JSON object from the LLM response.

    Returns a clean decision dict, or raises ValueError with a description
    of what went wrong so the caller can capture it gracefully.
    """
    # Strip any accidental markdown fences the model adds despite instructions
    stripped = re.sub(r"```(?:json)?|```", "", text).strip()

    # Find the outermost {...} block
    start = stripped.find("{")
    end = stripped.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")

    raw = json.loads(stripped[start:end])

    # Normalise and validate
    verdict = str(raw.get("verdict", "")).upper().strip()
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"verdict {verdict!r} not in {_VALID_VERDICTS}")

    conviction = str(raw.get("conviction", "")).upper().strip()
    if conviction not in _VALID_CONVICTIONS:
        raise ValueError(f"conviction {conviction!r} not in {_VALID_CONVICTIONS}")

    def _as_list(v: Any) -> list[str]:
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            return [v]
        return []

    return {
        "verdict":           verdict,
        "conviction":        conviction,
        "summary":           str(raw.get("summary", "")).strip(),
        "pros":              _as_list(raw.get("pros")),
        "cons":              _as_list(raw.get("cons")),
        "risks":             _as_list(raw.get("risks")),
        "position_guidance": str(raw.get("position_guidance", "")).strip(),
    }


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def make_decision(
    state: dict,
    api_key: str | None = None,
    model: str | None = None,
) -> dict:
    """Call the Anthropic API and populate state['decision'].

    Parameters
    ----------
    state:
        Pipeline state dict. Reads strategy, backtest, screen, research.
    api_key:
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    model:
        Model override. Defaults to config.DEFAULT_LLM_MODELS["anthropic"].

    The module degrades gracefully: if the API key is absent it logs a skip
    and returns state unchanged. All other failures go to state['errors'].
    """
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    resolved_model = model or _DEFAULT_MODEL

    if not resolved_key:
        log_agent(state, "make_decision", "skip", "ANTHROPIC_API_KEY not set")
        return state

    context = _build_context(state)
    prompt = _USER_PROMPT.format(context=context)

    try:
        raw_text = _call_anthropic(prompt, resolved_key, resolved_model)
    except requests.HTTPError as exc:
        msg = f"make_decision: API error {exc.response.status_code}: {exc.response.text[:200]}"
        state["errors"].append(msg)
        log_agent(state, "make_decision", "error", msg)
        return state
    except Exception as exc:
        msg = f"make_decision: {exc}"
        state["errors"].append(msg)
        log_agent(state, "make_decision", "error", msg)
        return state

    try:
        decision = _parse_response(raw_text)
    except (ValueError, json.JSONDecodeError) as exc:
        msg = f"make_decision: parse error — {exc}"
        state["errors"].append(msg)
        log_agent(state, "make_decision", "error", msg)
        # Store raw text so it's not lost
        state["decision"] = {"raw": raw_text}
        return state

    decision["model"] = resolved_model
    decision["timestamp"] = datetime.now().isoformat()
    state["decision"] = decision

    log_agent(
        state,
        "make_decision",
        "ok",
        f"{decision['verdict']} ({decision['conviction']}) — {decision['summary'][:80]}…",
    )
    return state
