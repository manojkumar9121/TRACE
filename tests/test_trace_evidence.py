"""Tests for :mod:`trace_core.evidence` — the evidence packet."""

from __future__ import annotations

from datetime import UTC, datetime

from trace_core.evidence import (
    PacketCandidate,
    ProvenanceRef,
    build_packet,
    empty_packet,
)
from trace_core.temporal import FreshnessState, TemporalScope

NOW = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)


def candidate(cid: str, text: str, score: float) -> PacketCandidate:
    return PacketCandidate(
        citation_id=cid, text=text, score=score, sent_at=NOW, conversation_id="c1"
    )


def build(candidates, **overrides):
    kwargs = {
        "freshness_state": FreshnessState.FRESH,
        "temporal_scope": TemporalScope.PRESENT,
        "window_start": None,
        "max_tokens": 1000,
    }
    kwargs.update(overrides)
    return build_packet(candidates, **kwargs)


# -- the allow-list ----------------------------------------------------------


def test_the_packet_is_the_allow_list():
    packet = build([candidate("e1", "hello", 0.9), candidate("e2", "world", 0.5)])
    assert packet.citation_ids == frozenset({"e1", "e2"})


def test_an_empty_packet_has_no_citations():
    packet = empty_packet(
        freshness_state=FreshnessState.ABSENT, temporal_scope=TemporalScope.UNKNOWN
    )
    assert packet.citation_ids == frozenset()
    assert packet.is_empty


def test_freshness_always_travels_with_the_packet():
    packet = build([candidate("e1", "hi", 0.5)], freshness_state=FreshnessState.STALE)
    assert packet.freshness_state is FreshnessState.STALE


def test_an_empty_packet_cannot_disagree_with_the_gate():
    """'No evidence' has exactly one representation."""
    packet = empty_packet(
        freshness_state=FreshnessState.ABSENT, temporal_scope=TemporalScope.HISTORICAL
    )
    assert packet.freshness_state is FreshnessState.ABSENT
    assert packet.temporal_scope is TemporalScope.HISTORICAL
    assert packet.truncated is False


# -- ordering and the budget -------------------------------------------------


def test_passages_are_ordered_by_score():
    packet = build(
        [candidate("e1", "low", 0.1), candidate("e2", "high", 0.9), candidate("e3", "mid", 0.5)]
    )
    assert [p.citation_id for p in packet.passages] == ["e2", "e3", "e1"]


def test_ties_break_on_citation_id():
    packet = build([candidate("b", "x", 0.5), candidate("a", "x", 0.5)])
    assert [p.citation_id for p in packet.passages] == ["a", "b"]


def test_the_budget_drops_the_lowest_scoring_and_says_so():
    packet = build(
        [candidate("e1", "x" * 100, 0.9), candidate("e2", "y" * 100, 0.5)],
        max_tokens=10,
        chars_per_token=4,
    )
    # budget = 40 chars; the first passage costs 101, so it is kept alone
    assert [p.citation_id for p in packet.passages] == ["e1"]
    assert packet.truncated is True


def test_the_best_passage_is_kept_even_if_it_exceeds_the_budget():
    """An empty packet is a less useful answer than one oversized passage."""
    packet = build([candidate("e1", "x" * 5000, 0.9)], max_tokens=1, chars_per_token=4)
    assert len(packet.passages) == 1
    assert packet.truncated is False


def test_a_packet_that_fits_is_not_truncated():
    packet = build([candidate("e1", "short", 0.9)], max_tokens=1000)
    assert packet.truncated is False


def test_the_budget_is_never_zero():
    packet = build([candidate("e1", "hello", 0.9)], max_tokens=0)
    assert len(packet.passages) == 1


# -- provenance ---------------------------------------------------------------


def test_provenance_is_attached_when_supplied():
    packet = build(
        [candidate("e1", "hello", 0.9)],
        provenance={"e1": ProvenanceRef(upload_id="u1", region_id="r1", bbox=(1, 2, 3, 4))},
    )
    passage = packet.passages[0]
    assert passage.upload_id == "u1"
    assert passage.region_id == "r1"
    assert passage.bbox == (1, 2, 3, 4)


def test_missing_provenance_does_not_drop_the_passage():
    """Dropping it would silently shrink the evidence set."""
    packet = build([candidate("e1", "hello", 0.9)])
    assert len(packet.passages) == 1
    assert packet.passages[0].bbox is None


# -- citations ----------------------------------------------------------------


def test_citations_carry_the_text_and_the_date():
    packet = build([candidate("e1", "hello there", 0.9)])
    refs = packet.citations()
    assert refs[0]["id"] == "e1"
    assert refs[0]["label"] == "hello there"
    assert refs[0]["sent_at"] is not None


def test_by_id_indexes_the_passages():
    packet = build([candidate("e1", "hello", 0.9)])
    assert set(packet.by_id()) == {"e1"}


def test_a_packet_is_immutable():
    packet = build([candidate("e1", "hello", 0.9)])
    try:
        packet.passages = ()  # type: ignore[misc]
    except (AttributeError, TypeError):
        return
    raise AssertionError("a packet should not be mutable")


# -- the link to the validators ----------------------------------------------


def test_the_packet_allow_list_is_what_the_validator_accepts():
    """The packet and the reference validator must agree on the allow-list."""
    from trace_core.reference import citation_validator

    packet = build([candidate("e1", "hello", 0.9)])
    payload = {
        "interpretations": [
            {"id": "i1", "epistemic_type": "observation", "evidence_refs": ["e1"]},
            {"id": "i2", "epistemic_type": "observation", "evidence_refs": ["GHOST"]},
        ]
    }
    report = citation_validator(payload, allowed_ids=packet.citation_ids)
    assert [c["id"] for c in report.payload["interpretations"]] == ["i1"]
