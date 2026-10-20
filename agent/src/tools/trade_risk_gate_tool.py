"""Agent tool: pre-trade risk gate (advisory, read-only).

Wraps ``backtest.trade_risk_policy.evaluate_trade``. The tool only reports a
verdict (``go`` / ``no_go`` / ``wait_pullback``) with the checks behind it. It
never places, cancels or approves an order and does no network or file I/O;
live order placement remains behind the existing mandate gate.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backtest.trade_risk_policy import (
    AccountState,
    RiskPolicy,
    TradeProposal,
    evaluate_trade,
)
from src.agent.tools import BaseTool

logger = logging.getLogger(__name__)


class TradeRiskGateTool(BaseTool):
    """Check a proposed long equity/ETF trade against a written risk policy."""

    name = "trade_risk_gate"
    description = (
        "Pre-trade risk gate for a proposed LONG stock/ETF trade. Give the "
        "proposal (symbol, entry, stop, target, shares, thesis, optional market "
        "context such as ma10/rsi6/day_change_pct/high_52w/spread_pct) and the "
        "account state (nav, day_pnl, holdings, ...). Returns go / no_go / "
        "wait_pullback with the setup table, a scale-out ladder and every check "
        "(per-trade risk, daily loss halt, stop-out cooldown, concentration, "
        "reward:risk, chase/dip-buy/52-week-high/spread/cost vetoes, intraday "
        "quota and PDT). Checks lacking input are listed under 'unchecked'. "
        "Advisory only: it never places or approves orders."
    )
    parameters = {
        "type": "object",
        "properties": {
            "proposal": {
                "type": "object",
                "description": (
                    "Required: symbol, entry, stop (below entry), target, shares. "
                    "Optional: thesis, chain, instrument (stock|etf), horizon "
                    "(swing|intraday), ma10, rsi6, day_change_pct, high_52w, "
                    "new_catalyst, extended_hours, spread_pct, "
                    "est_round_trip_cost, benchmark_broken, et_time (HH:MM)."
                ),
            },
            "account": {
                "type": "object",
                "description": (
                    "Required: nav. Optional: day_pnl, consecutive_stops, "
                    "stopped_today (symbols), holdings "
                    "[{symbol, market_value, chain}], intraday_trades_today, "
                    "intraday_round_trips_5d."
                ),
            },
            "policy": {
                "type": "object",
                "description": (
                    'Optional limit overrides, e.g. {"max_risk_per_trade_pct": 1}. '
                    "Defaults apply to omitted keys; unknown keys are rejected."
                ),
            },
        },
        "required": ["proposal", "account"],
    }
    is_readonly = True
    deterministic = True  # pure computation; identical args -> identical result
    repeatable = True  # a session reviews many candidate trades

    def execute(self, **kwargs: Any) -> str:
        """Evaluate one proposal and return a JSON envelope.

        Args:
            **kwargs: ``proposal``, ``account`` and optional ``policy`` objects.

        Returns:
            A strict-JSON string; ``{"status": "error", ...}`` on bad input.
        """
        try:
            proposal = TradeProposal.from_mapping(kwargs.get("proposal"))
            account = AccountState.from_mapping(kwargs.get("account"))
            policy_raw = kwargs.get("policy")
            policy = RiskPolicy.from_mapping(policy_raw)
            data = evaluate_trade(proposal, account, policy)
            envelope = {
                "status": "ok",
                "data": data,
                "meta": {"policy_overridden": bool(policy_raw)},
            }
            return json.dumps(envelope, ensure_ascii=False, indent=2, allow_nan=False)
        except Exception as exc:  # noqa: BLE001 — tool must always return JSON
            logger.warning("trade_risk_gate failed: %s", exc)
            return json.dumps(
                {"status": "error", "error": str(exc)},
                ensure_ascii=False,
                allow_nan=False,
            )
