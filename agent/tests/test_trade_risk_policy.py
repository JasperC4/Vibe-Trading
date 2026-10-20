"""Tests for the pre-trade risk policy core and its agent tool."""

from __future__ import annotations

import json

import pytest

from backtest.trade_risk_policy import (
    VERDICT_GO,
    VERDICT_NO_GO,
    VERDICT_WAIT,
    AccountState,
    RiskPolicy,
    TradeProposal,
    evaluate_trade,
    max_shares_for_risk,
    ratchet_stop,
    reduction_ladder,
)
from src.tools.trade_risk_gate_tool import TradeRiskGateTool


def _proposal(**overrides) -> dict:
    base = {
        "symbol": "abcd",
        "entry": 50.0,
        "stop": 48.0,
        "target": 55.0,
        "shares": 10,
        "thesis": "Earnings gap held above the prior base",
    }
    base.update(overrides)
    return base


def _account(**overrides) -> dict:
    base = {"nav": 10_000.0}
    base.update(overrides)
    return base


def _run(proposal=None, account=None, policy=None) -> dict:
    return evaluate_trade(
        TradeProposal.from_mapping(proposal or _proposal()),
        AccountState.from_mapping(account or _account()),
        RiskPolicy.from_mapping(policy),
    )


def _status(result: dict, check_id: str) -> str:
    return next(c["status"] for c in result["checks"] if c["id"] == check_id)


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def test_clean_setup_is_go_and_reports_unchecked_context():
    result = _run()
    assert result["verdict"] == VERDICT_GO
    assert result["failed"] == []
    # Reward 5 / risk 2 = 2.5; $20 risk on a $10k account is 0.2%.
    assert result["setup"]["reward_risk"] == 2.5
    assert result["metrics"]["risk_pct_of_nav"] == 0.2
    # No market context was supplied, so the review is not complete.
    assert result["complete"] is False
    assert "veto_chase" in result["unchecked"]
    json.dumps(result, allow_nan=False)


def test_symbol_is_normalised_to_upper_case():
    assert _run()["setup"]["symbol"] == "ABCD"


def test_low_reward_risk_is_no_go():
    result = _run(_proposal(target=52.0))  # 2 reward / 2 risk = 1.0
    assert result["verdict"] == VERDICT_NO_GO
    assert "reward_risk" in result["failed"]


def test_missing_thesis_is_no_go():
    result = _run(_proposal(thesis=""))
    assert result["verdict"] == VERDICT_NO_GO
    assert "thesis" in result["failed"]


# ---------------------------------------------------------------------------
# hard limits
# ---------------------------------------------------------------------------


def test_risk_per_trade_cap():
    # 2% of 10k = $200; 150 shares * $2 risk = $300.
    result = _run(_proposal(shares=150, target=60.0))
    assert "risk_per_trade" in result["failed"]
    ok = _run(_proposal(shares=100, target=60.0))
    assert _status(ok, "risk_per_trade") == "pass"


def test_max_shares_for_risk_matches_the_cap():
    assert max_shares_for_risk(10_000.0, 50.0, 48.0) == pytest.approx(100.0)


def test_daily_loss_halt_blocks_new_entries():
    # 1.5% of 10k = $150 loss.
    assert "daily_loss_halt" in _run(account=_account(day_pnl=-150.01))["failed"]
    assert _status(_run(account=_account(day_pnl=-149.0)), "daily_loss_halt") == "pass"


def test_two_consecutive_stops_halt_trading():
    assert (
        "consecutive_stop_halt" in _run(account=_account(consecutive_stops=2))["failed"]
    )
    assert (
        _status(_run(account=_account(consecutive_stops=1)), "consecutive_stop_halt")
        == "pass"
    )


def test_no_reentry_after_stop_out_today():
    result = _run(account=_account(stopped_today=["abcd"]))
    assert "stopped_out_today" in result["failed"]


def test_single_symbol_concentration_includes_existing_holding():
    account = _account(holdings=[{"symbol": "ABCD", "market_value": 3_900.0}])
    # 3,900 held + 500 new = 44% of NAV.
    assert "symbol_concentration" in _run(account=account)["failed"]


def test_chain_concentration_counts_distinct_symbols():
    holdings = [
        {"symbol": "AAA", "market_value": 500.0, "chain": "Semis"},
        {"symbol": "BBB", "market_value": 500.0, "chain": "semis"},
    ]
    blocked = _run(_proposal(chain="semis"), _account(holdings=holdings))
    assert "chain_concentration" in blocked["failed"]
    # Adding to a name already held does not add a position to the chain.
    adding = _run(_proposal(symbol="aaa", chain="semis"), _account(holdings=holdings))
    assert _status(adding, "chain_concentration") == "pass"


# ---------------------------------------------------------------------------
# vetoes
# ---------------------------------------------------------------------------


def test_chasing_alone_means_wait_for_pullback():
    result = _run(_proposal(ma10=45.0, rsi6=85.0))  # entry 11% above MA10
    assert result["verdict"] == VERDICT_WAIT
    assert result["suggested_limit"] == 45.0


def test_chasing_with_another_failure_is_no_go():
    result = _run(_proposal(ma10=45.0, rsi6=85.0, target=52.0))
    assert result["verdict"] == VERDICT_NO_GO
    assert result["suggested_limit"] is None


def test_extended_but_not_overbought_is_not_chasing():
    assert _status(_run(_proposal(ma10=45.0, rsi6=60.0)), "veto_chase") == "pass"


def test_no_dip_buying_beyond_three_percent_down():
    assert "veto_dip_buy" in _run(_proposal(day_change_pct=-3.5))["failed"]
    assert _status(_run(_proposal(day_change_pct=-2.0)), "veto_dip_buy") == "pass"


def test_near_52_week_high_needs_a_catalyst():
    near = _run(_proposal(high_52w=51.0))  # entry is 2% below the high
    assert "veto_near_52w_high" in near["failed"]
    with_catalyst = _run(_proposal(high_52w=51.0, new_catalyst=True))
    assert _status(with_catalyst, "veto_near_52w_high") == "pass"


def test_extended_hours_spread_veto():
    wide = _run(_proposal(extended_hours=True, spread_pct=0.8))
    assert "veto_extended_spread" in wide["failed"]
    unknown = _run(_proposal(extended_hours=True))
    assert _status(unknown, "veto_extended_spread") == "skipped"


def test_benchmark_breakdown_veto():
    assert "veto_benchmark_break" in _run(_proposal(benchmark_broken=True))["failed"]


def test_cost_must_stay_under_a_fifth_of_expected_profit():
    # Expected profit = 5 * 10 = 50, so the cap is 10.
    assert "veto_cost_discipline" in _run(_proposal(est_round_trip_cost=11.0))["failed"]
    assert (
        _status(_run(_proposal(est_round_trip_cost=9.0)), "veto_cost_discipline")
        == "pass"
    )


@pytest.mark.parametrize("instrument", ["leveraged_etf", "option", "crypto"])
def test_only_plain_stocks_and_etfs_are_allowed(instrument):
    assert "veto_instrument" in _run(_proposal(instrument=instrument))["failed"]


# ---------------------------------------------------------------------------
# intraday overlay
# ---------------------------------------------------------------------------


def test_intraday_checks_only_run_for_intraday_trades():
    swing = _run(_proposal(horizon="swing", et_time="15:45"))
    assert all(c["category"] != "intraday" for c in swing["checks"])


def test_intraday_cutoff_quota_and_pdt():
    late = _run(_proposal(horizon="intraday", et_time="15:30"))
    assert "intraday_entry_cutoff" in late["failed"]
    early = _run(_proposal(horizon="intraday", et_time="10:15"))
    assert _status(early, "intraday_entry_cutoff") == "pass"

    quota = _run(_proposal(horizon="intraday"), _account(intraday_trades_today=2))
    assert "intraday_daily_quota" in quota["failed"]

    pdt = _run(_proposal(horizon="intraday"), _account(intraday_round_trips_5d=3))
    assert "intraday_pdt" in pdt["failed"]
    rich = _run(
        _proposal(horizon="intraday"), _account(nav=30_000.0, intraday_round_trips_5d=3)
    )
    assert _status(rich, "intraday_pdt") == "pass"


# ---------------------------------------------------------------------------
# stop ratchet, ladder, policy
# ---------------------------------------------------------------------------


def test_stop_only_moves_up():
    assert ratchet_stop(48.0, 49.0) == (49.0, True)
    assert ratchet_stop(48.0, 47.0) == (48.0, False)
    assert ratchet_stop(48.0, 48.0) == (48.0, True)


def test_reduction_ladder_scales_out_and_ratchets_the_stop():
    ladder = reduction_ladder(100.0, 30)
    assert [step["trigger_price"] for step in ladder] == [105.0, 110.0, 115.0]
    assert [step["stop_after"] for step in ladder] == [100.0, 105.0, 110.0]
    assert sum(step["shares_to_sell"] for step in ladder) == pytest.approx(30.0)


def test_policy_overrides_change_the_limits():
    loose = _run(
        _proposal(shares=150, target=60.0), policy={"max_risk_per_trade_pct": 5}
    )
    assert _status(loose, "risk_per_trade") == "pass"
    assert loose["policy"]["max_risk_per_trade_pct"] == 5.0


@pytest.mark.parametrize(
    "policy",
    [
        {"nonsense": 1},
        {"max_risk_per_trade_pct": 0},
        {"max_symbol_weight_pct": 150},
        {"max_positions_per_chain": 1.5},
        {"no_new_intraday_entry_after": "25:99"},
    ],
)
def test_bad_policy_is_rejected(policy):
    with pytest.raises(ValueError):
        RiskPolicy.from_mapping(policy)


@pytest.mark.parametrize(
    "overrides",
    [
        {"stop": 50.0},
        {"stop": 51.0},
        {"entry": float("nan")},
        {"shares": 0},
        {"horizon": "weekly"},
    ],
)
def test_structurally_invalid_proposals_raise(overrides):
    with pytest.raises(ValueError):
        TradeProposal.from_mapping(_proposal(**overrides))


def test_invalid_account_raises():
    with pytest.raises(ValueError):
        AccountState.from_mapping({"nav": 0})


# ---------------------------------------------------------------------------
# agent tool
# ---------------------------------------------------------------------------


def test_tool_returns_strict_json_envelope():
    payload = json.loads(
        TradeRiskGateTool().execute(proposal=_proposal(), account=_account())
    )
    assert payload["status"] == "ok"
    assert payload["data"]["verdict"] == VERDICT_GO
    assert payload["meta"]["policy_overridden"] is False


def test_tool_reports_overridden_policy():
    payload = json.loads(
        TradeRiskGateTool().execute(
            proposal=_proposal(), account=_account(), policy={"min_reward_risk": 1.5}
        )
    )
    assert payload["meta"]["policy_overridden"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"proposal": _proposal(stop=60.0), "account": _account()},
        {"proposal": _proposal(), "account": {"nav": "lots"}},
        {"proposal": _proposal(), "account": _account(), "policy": {"bogus": 1}},
    ],
)
def test_tool_never_raises_on_bad_input(kwargs):
    payload = json.loads(TradeRiskGateTool().execute(**kwargs))
    assert payload["status"] == "error"
    assert payload["error"]


def test_tool_is_readonly_and_deterministic():
    tool = TradeRiskGateTool()
    assert tool.is_readonly and tool.deterministic
    assert tool.to_openai_schema()["function"]["name"] == "trade_risk_gate"
