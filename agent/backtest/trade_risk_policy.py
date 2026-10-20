"""Pre-trade risk policy: hard limits, setup quality gate and automatic vetoes.

A rule set for short-horizon (1-5 day swing and intraday) long equity/ETF
trades, expressed as pure functions of numbers. It answers one question for a
proposed trade -- ``go``, ``no_go`` or ``wait_pullback`` -- and shows which
checks produced that answer. Nothing here touches a broker, the network, the
filesystem or the clock, so it can never place or cancel an order; it only
advises. Order placement stays behind the existing mandate gate.

Conventions:
- Long-only: a stop above the entry is rejected rather than reinterpreted.
- Every limit is a field of :class:`RiskPolicy`; nothing is hardcoded in the
  checks. The defaults are one reasonable desk policy, not a recommendation.
- Optional market context (MA10, RSI, spreads, ...) that is missing makes the
  matching check ``skipped`` and lists it under ``unchecked`` instead of
  silently passing it, so a ``go`` verdict states how complete the review was.
- Non-finite numbers are rejected on input and never emitted, so results
  survive ``json.dumps(..., allow_nan=False)``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

VERDICT_GO = "go"
VERDICT_NO_GO = "no_go"
VERDICT_WAIT = "wait_pullback"

ALLOWED_INSTRUMENTS = frozenset({"stock", "etf"})
HORIZONS = frozenset({"swing", "intraday"})

# A failed check in this set alone means "right idea, wrong price": the
# verdict becomes ``wait_pullback`` instead of ``no_go``.
_CHASE_CHECKS = frozenset({"veto_chase"})


def _finite(value: Any, name: str) -> float:
    """Return ``value`` as a finite float or raise ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _optional(raw: Mapping[str, Any], name: str) -> float | None:
    value = raw.get(name)
    return None if value is None else _finite(value, name)


def _hhmm_to_minutes(text: str, name: str) -> int:
    try:
        hours, minutes = text.split(":")
        total = int(hours) * 60 + int(minutes)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{name} must look like HH:MM") from exc
    if not 0 <= total < 24 * 60:
        raise ValueError(f"{name} must look like HH:MM")
    return total


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


@dataclass(frozen=True)
class RiskPolicy:
    """Tunable limits. Percentages are percent numbers (``2.0`` means 2%)."""

    max_risk_per_trade_pct: float = 2.0
    daily_loss_limit_pct: float = 1.5
    max_symbol_weight_pct: float = 40.0
    max_positions_per_chain: int = 2
    min_reward_risk: float = 2.0
    consecutive_stops_halt: int = 2
    max_ma10_deviation_pct: float = 8.0
    rsi6_overbought: float = 80.0
    dip_buy_veto_drop_pct: float = 3.0
    min_distance_to_52w_high_pct: float = 3.0
    max_extended_hours_spread_pct: float = 0.5
    max_cost_fraction_of_profit: float = 0.2
    max_intraday_trades_per_day: int = 2
    pdt_nav_threshold: float = 25000.0
    pdt_max_round_trips_5d: int = 3
    no_new_intraday_entry_after: str = "15:30"

    def __post_init__(self) -> None:
        _hhmm_to_minutes(
            self.no_new_intraday_entry_after, "no_new_intraday_entry_after"
        )
        for spec in fields(self):
            value = getattr(self, spec.name)
            if isinstance(value, str):
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"policy.{spec.name} must be a positive finite number")
        for name in (
            "max_risk_per_trade_pct",
            "daily_loss_limit_pct",
            "max_symbol_weight_pct",
        ):
            if getattr(self, name) > 100:
                raise ValueError(f"policy.{name} cannot exceed 100")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "RiskPolicy":
        """Build a policy from overrides, rejecting unknown or bad values."""
        if not raw:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("policy must be an object")
        defaults = cls()
        unknown = sorted(set(raw) - {spec.name for spec in fields(cls)})
        if unknown:
            raise ValueError(f"unknown policy keys: {unknown}")
        values: dict[str, Any] = {}
        for key, value in raw.items():
            kind = type(getattr(defaults, key))
            if kind is str:
                values[key] = str(value)
            elif kind is int:
                number = _finite(value, f"policy.{key}")
                if number != int(number):
                    raise ValueError(f"policy.{key} must be a whole number")
                values[key] = int(number)
            else:
                values[key] = _finite(value, f"policy.{key}")
        return cls(**values)


@dataclass(frozen=True)
class Holding:
    """One open position as far as concentration limits are concerned."""

    symbol: str
    market_value: float
    chain: str | None = None


@dataclass(frozen=True)
class AccountState:
    """Account facts the limits are measured against."""

    nav: float
    day_pnl: float = 0.0
    consecutive_stops: int = 0
    stopped_today: frozenset[str] = frozenset()
    holdings: tuple[Holding, ...] = ()
    intraday_trades_today: int = 0
    intraday_round_trips_5d: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.nav) or self.nav <= 0:
            raise ValueError("account.nav must be a positive finite number")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AccountState":
        """Validate a plain mapping (for example tool arguments)."""
        if not isinstance(raw, Mapping):
            raise ValueError("account must be an object")
        holdings = []
        for item in raw.get("holdings") or []:
            if not isinstance(item, Mapping) or not str(item.get("symbol", "")).strip():
                raise ValueError("each holding needs a symbol")
            chain = item.get("chain")
            holdings.append(
                Holding(
                    symbol=str(item["symbol"]).strip().upper(),
                    market_value=_finite(
                        item.get("market_value"), "holding.market_value"
                    ),
                    chain=str(chain).strip().lower() if chain else None,
                )
            )
        return cls(
            nav=_finite(raw.get("nav"), "account.nav"),
            day_pnl=_finite(raw.get("day_pnl", 0.0), "account.day_pnl"),
            consecutive_stops=int(
                _finite(raw.get("consecutive_stops", 0), "account.consecutive_stops")
            ),
            stopped_today=frozenset(
                str(s).strip().upper() for s in raw.get("stopped_today") or []
            ),
            holdings=tuple(holdings),
            intraday_trades_today=int(
                _finite(
                    raw.get("intraday_trades_today", 0), "account.intraday_trades_today"
                )
            ),
            intraday_round_trips_5d=int(
                _finite(
                    raw.get("intraday_round_trips_5d", 0),
                    "account.intraday_round_trips_5d",
                )
            ),
        )


@dataclass(frozen=True)
class TradeProposal:
    """A proposed long entry plus whatever market context is known."""

    symbol: str
    entry: float
    stop: float
    target: float
    shares: float
    thesis: str = ""
    chain: str | None = None
    instrument: str = "stock"
    horizon: str = "swing"
    ma10: float | None = None
    rsi6: float | None = None
    day_change_pct: float | None = None
    high_52w: float | None = None
    new_catalyst: bool = False
    extended_hours: bool = False
    spread_pct: float | None = None
    est_round_trip_cost: float | None = None
    benchmark_broken: bool | None = None
    et_time: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("proposal.symbol is required")
        for name in ("entry", "stop", "target", "shares"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"proposal.{name} must be a positive finite number")
        if self.stop >= self.entry:
            raise ValueError(
                "long stop must be below entry; name a stop or there is no trade"
            )
        if self.horizon not in HORIZONS:
            raise ValueError(f"proposal.horizon must be one of {sorted(HORIZONS)}")
        if self.et_time is not None:
            _hhmm_to_minutes(self.et_time, "proposal.et_time")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TradeProposal":
        """Validate a plain mapping (for example tool arguments)."""
        if not isinstance(raw, Mapping):
            raise ValueError("proposal must be an object")
        chain = raw.get("chain")
        benchmark = raw.get("benchmark_broken")
        return cls(
            symbol=str(raw.get("symbol", "")).strip().upper(),
            entry=_finite(raw.get("entry"), "proposal.entry"),
            stop=_finite(raw.get("stop"), "proposal.stop"),
            target=_finite(raw.get("target"), "proposal.target"),
            shares=_finite(raw.get("shares"), "proposal.shares"),
            thesis=str(raw.get("thesis") or "").strip(),
            chain=str(chain).strip().lower() if chain else None,
            instrument=str(raw.get("instrument") or "stock").strip().lower(),
            horizon=str(raw.get("horizon") or "swing").strip().lower(),
            ma10=_optional(raw, "ma10"),
            rsi6=_optional(raw, "rsi6"),
            day_change_pct=_optional(raw, "day_change_pct"),
            high_52w=_optional(raw, "high_52w"),
            new_catalyst=bool(raw.get("new_catalyst", False)),
            extended_hours=bool(raw.get("extended_hours", False)),
            spread_pct=_optional(raw, "spread_pct"),
            est_round_trip_cost=_optional(raw, "est_round_trip_cost"),
            benchmark_broken=None if benchmark is None else bool(benchmark),
            et_time=raw.get("et_time"),
        )


def max_shares_for_risk(
    nav: float, entry: float, stop: float, policy: RiskPolicy | None = None
) -> float:
    """Largest share count whose entry-to-stop loss fits the per-trade budget."""
    policy = policy or RiskPolicy()
    if stop >= entry:
        raise ValueError("long stop must be below entry")
    return (nav * policy.max_risk_per_trade_pct / 100.0) / (entry - stop)


def ratchet_stop(current_stop: float, proposed_stop: float) -> tuple[float, bool]:
    """Apply the one-way stop rule for a long position.

    Returns the stop to keep and whether the proposal was accepted. A stop may
    only move up; lowering it is refused and the current stop is returned.
    """
    current = _finite(current_stop, "current_stop")
    proposed = _finite(proposed_stop, "proposed_stop")
    return (proposed, True) if proposed >= current else (current, False)


def reduction_ladder(entry: float, shares: float) -> list[dict[str, Any]]:
    """Scale-out plan: trim into strength while the stop ratchets up."""
    third = 1.0 / 3.0
    steps = (
        (0.05, "sell 1/3, move stop to cost", third, 0.0),
        (0.10, "sell another 1/3, move stop to +5%", third, 0.05),
        (
            0.15,
            "close the rest, or keep at most 1/4 of the original size; stop to +10%",
            third,
            0.10,
        ),
    )
    ladder = []
    for gain, action, fraction, stop_gain in steps:
        ladder.append(
            {
                "trigger_gain_pct": round(gain * 100, 2),
                "trigger_price": _round(entry * (1 + gain)),
                "action": action,
                "shares_to_sell": _round(shares * fraction, 4),
                "stop_after": _round(entry * (1 + stop_gain)),
            }
        )
    return ladder


def evaluate_trade(
    proposal: TradeProposal,
    account: AccountState,
    policy: RiskPolicy | None = None,
) -> dict[str, Any]:
    """Run every limit, gate and veto against a proposed long entry.

    Args:
        proposal: The trade under review.
        account: Account facts the limits are measured against.
        policy: Limits to apply; defaults to :class:`RiskPolicy`.

    Returns:
        A JSON-safe dict with the verdict, the setup table, the scale-out
        ladder and the full list of checks (each ``pass``, ``fail`` or
        ``skipped``).

    Raises:
        ValueError: If the proposal or account is structurally invalid.
    """
    policy = policy or RiskPolicy()
    p, a = proposal, account
    checks: list[dict[str, str]] = []

    def record(check_id: str, category: str, passed: bool | None, detail: str) -> None:
        status = "skipped" if passed is None else ("pass" if passed else "fail")
        checks.append(
            {"id": check_id, "category": category, "status": status, "detail": detail}
        )

    risk_per_share = p.entry - p.stop
    risk_amount = risk_per_share * p.shares
    risk_pct = risk_amount / a.nav * 100.0
    reward_risk = (p.target - p.entry) / risk_per_share
    expected_profit = (p.target - p.entry) * p.shares
    held_same = sum(h.market_value for h in a.holdings if h.symbol == p.symbol)
    weight_pct = (held_same + p.entry * p.shares) / a.nav * 100.0

    # --- hard limits -------------------------------------------------
    loss_floor = -a.nav * policy.daily_loss_limit_pct / 100.0
    record(
        "daily_loss_halt",
        "hard_limit",
        a.day_pnl > loss_floor,
        f"day P&L {a.day_pnl:.2f} vs halt at {loss_floor:.2f} ({policy.daily_loss_limit_pct}% of NAV)",
    )
    record(
        "consecutive_stop_halt",
        "hard_limit",
        a.consecutive_stops < policy.consecutive_stops_halt,
        f"{a.consecutive_stops} consecutive stop-outs; halt at {policy.consecutive_stops_halt}",
    )
    record(
        "stopped_out_today",
        "hard_limit",
        p.symbol not in a.stopped_today,
        (
            f"{p.symbol} was stopped out today"
            if p.symbol in a.stopped_today
            else "no stop-out today"
        ),
    )
    record(
        "risk_per_trade",
        "hard_limit",
        risk_pct <= policy.max_risk_per_trade_pct,
        f"risk {risk_amount:.2f} = {risk_pct:.2f}% of NAV; cap {policy.max_risk_per_trade_pct}%",
    )
    record(
        "symbol_concentration",
        "hard_limit",
        weight_pct <= policy.max_symbol_weight_pct,
        f"{p.symbol} would be {weight_pct:.1f}% of NAV; cap {policy.max_symbol_weight_pct}%",
    )
    if p.chain:
        chain_symbols = {h.symbol for h in a.holdings if h.chain == p.chain} | {
            p.symbol
        }
        record(
            "chain_concentration",
            "hard_limit",
            len(chain_symbols) <= policy.max_positions_per_chain,
            f"{len(chain_symbols)} positions in chain '{p.chain}'; cap {policy.max_positions_per_chain}",
        )
    else:
        record("chain_concentration", "hard_limit", None, "no chain given")

    # --- setup quality gate ------------------------------------------
    record(
        "reward_risk",
        "quality_gate",
        reward_risk >= policy.min_reward_risk,
        f"reward:risk {reward_risk:.2f}; minimum {policy.min_reward_risk}",
    )
    record(
        "thesis",
        "quality_gate",
        bool(p.thesis),
        "one-line thesis given" if p.thesis else "no thesis",
    )

    # --- automatic vetoes --------------------------------------------
    record(
        "veto_instrument",
        "veto",
        p.instrument in ALLOWED_INSTRUMENTS,
        f"instrument '{p.instrument}'; allowed {sorted(ALLOWED_INSTRUMENTS)}",
    )
    if p.ma10 is not None and p.rsi6 is not None and p.ma10 > 0:
        deviation = (p.entry / p.ma10 - 1.0) * 100.0
        chasing = (
            deviation > policy.max_ma10_deviation_pct
            and p.rsi6 > policy.rsi6_overbought
        )
        record(
            "veto_chase",
            "veto",
            not chasing,
            f"{deviation:.1f}% above MA10, RSI(6) {p.rsi6:.0f}; "
            f"veto above {policy.max_ma10_deviation_pct}% and RSI {policy.rsi6_overbought:.0f}",
        )
    else:
        record("veto_chase", "veto", None, "needs ma10 and rsi6")
    if p.day_change_pct is not None:
        record(
            "veto_dip_buy",
            "veto",
            p.day_change_pct >= -policy.dip_buy_veto_drop_pct,
            f"down {p.day_change_pct:.1f}% today; no dip buys beyond -{policy.dip_buy_veto_drop_pct}%",
        )
    else:
        record("veto_dip_buy", "veto", None, "needs day_change_pct")
    if p.high_52w is not None and p.high_52w > 0:
        distance = (p.high_52w - p.entry) / p.high_52w * 100.0
        record(
            "veto_near_52w_high",
            "veto",
            distance >= policy.min_distance_to_52w_high_pct or p.new_catalyst,
            f"{distance:.1f}% below the 52-week high; "
            f"{'catalyst present' if p.new_catalyst else 'no new catalyst'}",
        )
    else:
        record("veto_near_52w_high", "veto", None, "needs high_52w")
    if not p.extended_hours:
        record("veto_extended_spread", "veto", True, "regular session")
    elif p.spread_pct is None:
        record("veto_extended_spread", "veto", None, "extended hours; needs spread_pct")
    else:
        record(
            "veto_extended_spread",
            "veto",
            p.spread_pct <= policy.max_extended_hours_spread_pct,
            f"non-regular session; spread {p.spread_pct:.2f}% vs cap {policy.max_extended_hours_spread_pct}%",
        )
    if p.benchmark_broken is None:
        record("veto_benchmark_break", "veto", None, "needs benchmark_broken")
    else:
        record(
            "veto_benchmark_break",
            "veto",
            not p.benchmark_broken,
            "benchmark broke down today" if p.benchmark_broken else "benchmark intact",
        )
    if p.est_round_trip_cost is not None:
        record(
            "veto_cost_discipline",
            "veto",
            p.est_round_trip_cost
            <= expected_profit * policy.max_cost_fraction_of_profit,
            f"round-trip cost {p.est_round_trip_cost:.2f} vs expected profit {expected_profit:.2f}; "
            f"cap {policy.max_cost_fraction_of_profit:.0%} of profit",
        )
    else:
        record("veto_cost_discipline", "veto", None, "needs est_round_trip_cost")

    # --- intraday overlay --------------------------------------------
    if p.horizon == "intraday":
        record(
            "intraday_daily_quota",
            "intraday",
            a.intraday_trades_today < policy.max_intraday_trades_per_day,
            f"{a.intraday_trades_today} intraday trades today; cap {policy.max_intraday_trades_per_day}",
        )
        under_pdt = a.nav < policy.pdt_nav_threshold
        record(
            "intraday_pdt",
            "intraday",
            (not under_pdt)
            or a.intraday_round_trips_5d < policy.pdt_max_round_trips_5d,
            f"{a.intraday_round_trips_5d} round trips in 5 days; "
            + (
                f"cap {policy.pdt_max_round_trips_5d} below NAV {policy.pdt_nav_threshold:.0f}"
                if under_pdt
                else "PDT not binding"
            ),
        )
        if p.et_time is None:
            record("intraday_entry_cutoff", "intraday", None, "needs et_time")
        else:
            cutoff = _hhmm_to_minutes(policy.no_new_intraday_entry_after, "cutoff")
            record(
                "intraday_entry_cutoff",
                "intraday",
                _hhmm_to_minutes(p.et_time, "et_time") < cutoff,
                f"{p.et_time} ET; no new entries from {policy.no_new_intraday_entry_after}",
            )

    failed = [c["id"] for c in checks if c["status"] == "fail"]
    unchecked = [c["id"] for c in checks if c["status"] == "skipped"]
    if not failed:
        verdict = VERDICT_GO
    elif set(failed) <= _CHASE_CHECKS:
        verdict = VERDICT_WAIT
    else:
        verdict = VERDICT_NO_GO
    suggested_limit = _round(p.ma10) if verdict == VERDICT_WAIT and p.ma10 else None
    headline = {
        VERDICT_GO: f"GO: {p.symbol} passes every check that could be run",
        VERDICT_WAIT: f"WAIT: {p.symbol} is extended; rest a limit near MA10 instead of chasing",
        VERDICT_NO_GO: f"NO GO: {p.symbol} fails {', '.join(failed)}",
    }[verdict]

    return {
        "verdict": verdict,
        "headline": headline,
        "setup": {
            "symbol": p.symbol,
            "horizon": p.horizon,
            "entry": _round(p.entry),
            "stop": _round(p.stop),
            "target": _round(p.target),
            "shares": _round(p.shares),
            "reward_risk": _round(reward_risk, 2),
            "thesis": p.thesis,
        },
        "metrics": {
            "risk_amount": _round(risk_amount, 2),
            "risk_pct_of_nav": _round(risk_pct, 3),
            "expected_profit": _round(expected_profit, 2),
            "symbol_weight_pct": _round(weight_pct, 2),
            "max_shares_for_risk": _round(
                max_shares_for_risk(a.nav, p.entry, p.stop, policy), 4
            ),
        },
        "suggested_limit": suggested_limit,
        "reduction_ladder": reduction_ladder(p.entry, p.shares),
        "checks": checks,
        "failed": failed,
        "unchecked": unchecked,
        "complete": not unchecked,
        "policy": asdict(policy),
    }
