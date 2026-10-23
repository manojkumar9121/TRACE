"""Tests for :mod:`trace_core.gating` — the freshness gate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trace_core.gating import FreshnessGate, FreshnessVerdict, TemporalValidator
from trace_core.temporal import FreshnessState, TemporalScope

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def gate() -> FreshnessGate:
    return FreshnessGate(fresh_days=45, stale_days=180)


def verdict(state: FreshnessState) -> FreshnessVerdict:
    return FreshnessVerdict(
        state=state,
        age_days=60.0,
        evidence_through=NOW - timedelta(days=60),
        window_start=None,
        explanation="",
    )


# -- state computation -------------------------------------------------------


def test_no_evidence_is_absent():
    result = gate().evaluate(None, now=NOW)
    assert result.state is FreshnessState.ABSENT
    assert result.age_days is None
    assert not result.allows_present_tense


def test_recent_evidence_is_fresh():
    result = gate().evaluate(NOW - timedelta(days=10), now=NOW)
    assert result.state is FreshnessState.FRESH
    assert result.allows_present_tense
    assert result.permitted_scope is TemporalScope.PRESENT
    assert result.window_start is None


def test_middle_aged_evidence_is_aging():
    result = gate().evaluate(NOW - timedelta(days=60), now=NOW)
    assert result.state is FreshnessState.AGING
    assert not result.allows_present_tense
    assert result.permitted_scope is TemporalScope.HISTORICAL
    assert result.window_start is not None


def test_old_evidence_is_stale():
    result = gate().evaluate(NOW - timedelta(days=400), now=NOW)
    assert result.state is FreshnessState.STALE
    assert result.permitted_scope is TemporalScope.UNKNOWN
    assert result.window_start is not None


def test_archived_forces_stale():
    result = gate().evaluate(NOW - timedelta(days=1), now=NOW, archived=True)
    assert result.state is FreshnessState.STALE


def test_a_future_timestamp_clamps_to_zero():
    result = gate().evaluate(NOW + timedelta(days=10), now=NOW)
    assert result.age_days == 0.0
    assert result.state is FreshnessState.FRESH


def test_the_clock_is_injected_not_read():
    """A gate must never read the wall clock."""
    result = gate().evaluate(NOW - timedelta(days=10), now=NOW)
    assert result.age_days == pytest.approx(10.0)


def test_a_naive_now_is_rejected():
    with pytest.raises(ValueError):
        gate().evaluate(NOW, now=datetime(2026, 10, 20))


def test_a_naive_evidence_timestamp_is_read_as_utc():
    """A chat export writes naive timestamps constantly; rejecting them would
    reject most real corpora. A naive timestamp is read as UTC."""
    result = gate().evaluate(datetime(2026, 10, 19), now=NOW)
    # NOW is 2026-10-20 12:00 and the evidence is 2026-10-19 00:00.
    assert result.age_days == pytest.approx(1.5)
    assert result.state is FreshnessState.FRESH


def test_a_naive_now_is_still_rejected():
    """``now`` is the caller's own clock reading, so it must be unambiguous."""
    with pytest.raises(ValueError):
        gate().evaluate(NOW, now=datetime(2026, 10, 20))


def test_the_thresholds_are_validated():
    with pytest.raises(ValueError):
        FreshnessGate(fresh_days=-1)
    with pytest.raises(ValueError):
        FreshnessGate(fresh_days=100, stale_days=50)


# -- what each state permits -------------------------------------------------


def test_only_fresh_permits_present_tense():
    assert verdict(FreshnessState.FRESH).allows_present_tense
    assert not verdict(FreshnessState.AGING).allows_present_tense
    assert not verdict(FreshnessState.STALE).allows_present_tense


def test_aging_permits_simulation_but_stale_does_not():
    assert verdict(FreshnessState.AGING).allows_simulation
    assert not verdict(FreshnessState.STALE).allows_simulation
    assert not verdict(FreshnessState.ABSENT).allows_simulation


def test_every_state_allows_some_analysis():
    for state in FreshnessState:
        assert verdict(state).allows_analysis


# -- schema constraint -------------------------------------------------------


def test_the_schema_is_constrained_to_the_permitted_scope():
    schema = {
        "type": "object",
        "properties": {
            "temporal_scope": {"type": "string", "enum": ["present", "historical", "unknown"]}
        },
    }
    constrained = gate().constrain_schema(schema, verdict(FreshnessState.STALE))
    enum = constrained["properties"]["temporal_scope"]["enum"]
    assert "present" not in enum
    assert "unknown" in enum
    # the original is untouched
    assert "present" in schema["properties"]["temporal_scope"]["enum"]


def test_the_constraint_recurses_into_nested_claims():
    schema = {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "object",
                "properties": {
                    "interpretations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "temporal_scope": {
                                    "type": "string",
                                    "enum": ["present", "historical", "unknown"],
                                }
                            },
                        },
                    }
                },
            }
        },
    }
    constrained = gate().constrain_schema(schema, verdict(FreshnessState.STALE))
    enum = constrained["properties"]["analysis"]["properties"]["interpretations"]["items"][
        "properties"
    ]["temporal_scope"]["enum"]
    assert "present" not in enum


def test_a_fresh_verdict_does_not_constrain_the_schema():
    schema = {
        "type": "object",
        "properties": {
            "temporal_scope": {"type": "string", "enum": ["present", "historical", "unknown"]}
        },
    }
    constrained = gate().constrain_schema(schema, verdict(FreshnessState.FRESH))
    assert "present" in constrained["properties"]["temporal_scope"]["enum"]


def test_a_malformed_schema_does_not_hang():
    node: dict = {"m": 1}
    for _ in range(14):
        node = {"child": node}
    schema: dict = node
    constrained = gate().constrain_schema(schema, verdict(FreshnessState.STALE))
    assert isinstance(constrained, dict)


# -- the instruction ---------------------------------------------------------


def test_a_fresh_verdict_needs_no_instruction():
    assert gate().constraining_instruction(verdict(FreshnessState.FRESH)) is None


def test_an_aging_verdict_instructs_historical():
    text = gate().constraining_instruction(verdict(FreshnessState.AGING))
    assert text is not None
    assert "historical" in text.lower()


def test_a_stale_verdict_says_current_is_unknown():
    text = gate().constraining_instruction(verdict(FreshnessState.STALE))
    assert text is not None
    assert "unknown" in text.lower()


# -- the output validator ----------------------------------------------------


def test_a_present_tense_claim_is_downgraded():
    payload = {
        "interpretations": [
            {"id": "i1", "temporal_scope": "present"},
            {"id": "i2", "temporal_scope": "historical"},
        ]
    }
    validator = TemporalValidator()
    result = validator.validate(payload, verdict(FreshnessState.STALE))
    scopes = {c["id"]: c["temporal_scope"] for c in result["interpretations"]}
    assert scopes["i1"] == "historical"
    assert scopes["i2"] == "historical"
    assert result["interpretations"][0]["downgraded_by_freshness_gate"] is True
    assert validator.violation_count == 1


def test_a_fresh_verdict_leaves_the_payload_alone():
    payload = {"interpretations": [{"id": "i1", "temporal_scope": "present"}]}
    validator = TemporalValidator()
    result = validator.validate(payload, verdict(FreshnessState.FRESH))
    assert result["interpretations"][0]["temporal_scope"] == "present"
    assert validator.violation_count == 0


def test_nested_claims_are_validated():
    payload = {
        "analysis": {
            "interpretations": [{"id": "i1", "temporal_scope": "present"}],
        }
    }
    validator = TemporalValidator()
    result = validator.validate(payload, verdict(FreshnessState.AGING))
    assert result["analysis"]["interpretations"][0]["temporal_scope"] == "historical"
    assert validator.violation_count == 1


def test_a_claim_is_not_downgraded_twice():
    payload = {"interpretations": [{"id": "i1", "temporal_scope": "present"}]}
    validator = TemporalValidator()
    validator.validate(payload, verdict(FreshnessState.STALE))
    assert validator.violation_count == 1


def test_the_validator_can_be_reset():
    validator = TemporalValidator()
    validator.validate(
        {"interpretations": [{"id": "i1", "temporal_scope": "present"}]},
        verdict(FreshnessState.STALE),
    )
    assert validator.violation_count == 1
    validator.reset()
    assert validator.violation_count == 0
