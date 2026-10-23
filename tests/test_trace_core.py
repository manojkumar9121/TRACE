"""Tests for :mod:`trace_core` — the contract, the formulas, the invariants."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trace_core import (
    Citation,
    CitationError,
    Claim,
    Evidence,
    Link,
    PolicyError,
    RetrievalCandidate,
    Scorer,
    TableConfidencePolicy,
    TableRetrievalPolicy,
    TemporalScope,
    Verdict,
    fingerprint,
    score_reconstructs,
)

NOW = datetime(2026, 10, 20, tzinfo=UTC)


def policy(**overrides):
    base = {
        "source_weights": {
            "user_stated": 1.00,
            "manual_edit": 1.00,
            "feedback": 0.85,
            "message": 0.70,
            "screenshot": 0.65,
            "system_inference": 0.55,
        },
        "half_lives": {"behavioral": 270, "tone": 180, "style": 540, "factual": 1095},
        "corroboration_floor": 2,
        "coverage_denominator": 3,
    }
    base.update(overrides)
    return TableConfidencePolicy(**base)


def retrieval(conf):
    return TableRetrievalPolicy(
        weights={"vector": 0.55, "recency": 0.20, "source": 0.15, "confidence": 0.10},
        source_weights=conf.source_weights,
    )


def evidence(eid, *, lineage="L1", source_type="message", days=0, speaker="them", text="hi"):
    return Evidence(
        evidence_id=eid,
        segment_id=lineage,
        lineage_id=lineage,
        source_id=eid,
        text=text,
        source_type=source_type,
        modality="chat",
        occurred_at=NOW - timedelta(days=days),
        speaker=speaker,
    )


def claim(**overrides):
    base = {
        "claim_id": "c1",
        "text": "replies briefly",
        "subject": "them",
        "kind": "behavioral",
        "scope": TemporalScope.HISTORICAL,
    }
    base.update(overrides)
    return Claim(**base)


# -- contract ---------------------------------------------------------------


def test_evidence_metadata_is_frozen():
    item = evidence("e1")
    with pytest.raises(TypeError):
        item.metadata["x"] = 1  # type: ignore[index]


def test_evidence_metadata_copies_the_mapping():
    source = {"a": 1}
    item = Evidence(
        evidence_id="e1",
        segment_id="s",
        lineage_id="l",
        source_id="e1",
        text="t",
        source_type="message",
        modality="chat",
        occurred_at=NOW,
        speaker="them",
        metadata=source,
    )
    source["a"] = 2
    assert item.metadata["a"] == 1


def test_fingerprint_is_stable_across_punctuation_and_case():
    assert fingerprint("Hello, world!") == fingerprint("hello world")
    assert fingerprint("hello world") != fingerprint("goodbye world")


# -- policy validation (PLAN.md section 4.3) --------------------------------


def test_policy_rejects_a_corroboration_floor_below_two():
    with pytest.raises(PolicyError):
        policy(corroboration_floor=1)


def test_policy_rejects_a_non_finite_weight():
    with pytest.raises(PolicyError):
        policy(source_weights={"message": float("nan")})
    with pytest.raises(PolicyError):
        policy(source_weights={"message": float("inf")})


def test_policy_rejects_a_weight_outside_the_unit_interval():
    with pytest.raises(PolicyError):
        policy(source_weights={"message": 1.4})
    with pytest.raises(PolicyError):
        policy(source_weights={"message": -0.1})


def test_policy_rejects_a_non_positive_half_life():
    with pytest.raises(PolicyError):
        policy(half_lives={"behavioral": 0})


def test_retrieval_weights_may_be_unnormalised_but_not_negative():
    TableRetrievalPolicy(weights={"vector": 3.0}, source_weights={"message": 0.7})
    with pytest.raises(PolicyError):
        TableRetrievalPolicy(weights={"vector": -1.0}, source_weights={"message": 0.7})


def test_unknown_source_takes_the_lowest_known_weight():
    conf = policy()
    assert conf.source_weight("not_a_known_source") == 0.55


# -- confidence formula (PLAN.md section 4.1) --------------------------------


def test_confidence_is_the_product_of_the_four_terms():
    conf = policy()
    items = [
        evidence("e1", lineage="L1"),
        evidence("e2", lineage="L2"),
        evidence("e3", lineage="L3"),
    ]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    links = [Link(evidence_id=item.evidence_id, stance="supports") for item in items]
    scored = scorer.score(claim(), links, now=NOW)
    assert scored.value == pytest.approx(0.70)
    assert scored.terms["src_weight"] == pytest.approx(0.70)
    assert scored.terms["coverage"] == pytest.approx(1.0)
    assert scored.terms["freshness"] == pytest.approx(1.0)
    assert scored.terms["contradiction"] == pytest.approx(1.0)


def test_a_weak_source_caps_the_whole_claim():
    conf = policy()
    items = [
        evidence("e1", lineage="L1", source_type="user_stated"),
        evidence("e2", lineage="L2", source_type="screenshot"),
    ]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    links = [Link(evidence_id=item.evidence_id, stance="supports") for item in items]
    scored = scorer.score(claim(), links, now=NOW)
    assert scored.terms["src_weight"] == pytest.approx(0.65)


def test_freshness_halves_at_the_half_life():
    conf = policy()
    items = [evidence("e1", lineage="L1", days=270)]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    scored = scorer.score(claim(kind="behavioral"), [Link("e1", "supports")], now=NOW)
    assert scored.terms["freshness"] == pytest.approx(0.5)


def test_undated_evidence_has_no_freshness():
    conf = policy()
    item = Evidence(
        evidence_id="e1",
        segment_id="s",
        lineage_id="L1",
        source_id="e1",
        text="t",
        source_type="message",
        modality="chat",
        occurred_at=None,
        speaker="them",
    )
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=[item])
    scored = scorer.score(claim(), [Link("e1", "supports")], now=NOW)
    assert scored.terms["freshness"] == pytest.approx(0.0)
    assert scored.evidence.n_undated == 1


def test_contradiction_reduces_the_score():
    """The pre-port shipped formula: the denominator is the *supporting* count.

    PLAN.md §4.1 writes "n_contradicting / max(1, n_citations)", which reads
    either way. The pre-port code (``confidence.py``) divides by
    ``corroboration_count`` — the supporting rows — and §4.1 says the formulas
    are "exactly as ported", so the supporting count is what is implemented.
    The ambiguity is recorded in PLAN.md §13.
    """
    conf = policy()
    items = [
        evidence("e1", lineage="L1"),
        evidence("e2", lineage="L2"),
        evidence("e3", lineage="L3"),
        evidence("c1", lineage="L4"),
    ]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    links = [
        Link("e1", "supports"),
        Link("e2", "supports"),
        Link("e3", "supports"),
        Link("c1", "contradicts"),
    ]
    scored = scorer.score(claim(), links, now=NOW)
    assert scored.terms["contradiction"] == pytest.approx(1 - 1 / 3)


def test_no_citation_scores_zero_with_a_named_cap():
    conf = policy()
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf))
    scored = scorer.score(claim(), [], now=NOW)
    assert scored.value == 0.0
    assert [cap.code for cap in scored.caps_applied] == ["no_evidence"]


def test_one_lineage_is_capped_by_the_corroboration_floor():
    conf = policy()
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=[evidence("e1")])
    scored = scorer.score(claim(), [Link("e1", "supports")], now=NOW)
    assert [cap.code for cap in scored.caps_applied] == ["insufficient_corroboration"]
    assert scored.value <= 0.25


def test_coverage_and_the_acceptance_floor_are_separate_knobs():
    """Accepts at 2 conversations but saturates coverage at 3 — the two-knob
    behaviour the pre-port system verified."""
    conf = policy(corroboration_floor=2, coverage_denominator=3)
    items = [evidence("e1", lineage="L1"), evidence("e2", lineage="L2")]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    links = [Link("e1", "supports"), Link("e2", "supports")]
    scored = scorer.score(claim(), links, now=NOW)
    assert scored.terms["coverage"] == pytest.approx(2 / 3)
    decision = scorer.decide(
        claim(),
        [Citation("e1", "supports"), Citation("e2", "supports")],
        verdict=Verdict("c1", "supported"),
        now=NOW,
    )
    assert decision.state == "accepted"


def test_every_score_reconstructs_from_its_terms_and_caps():
    conf = policy()
    items = [
        evidence("e1", lineage="L1"),
        evidence("e2", lineage="L2"),
        evidence("e3", lineage="L3", days=40),
    ]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    links = [Link("e1", "supports"), Link("e2", "supports"), Link("e3", "contradicts")]
    assert score_reconstructs(scorer.score(claim(), links, now=NOW))


# -- decisions (PLAN.md section 3.1) ----------------------------------------


def test_absent_verdict_is_not_acceptance():
    conf = policy()
    items = [evidence("e1", lineage="L1"), evidence("e2", lineage="L2")]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    decision = scorer.decide(
        claim(),
        [Citation("e1", "supports"), Citation("e2", "supports")],
        verdict=None,
        now=NOW,
    )
    assert decision.state == "candidate"


def test_unsupported_is_rejected_with_unknown_scope():
    conf = policy()
    items = [evidence("e1", lineage="L1")]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    decision = scorer.decide(
        claim(), [Citation("e1", "supports")], verdict=Verdict("c1", "unsupported"), now=NOW
    )
    assert decision.state == "rejected"
    assert decision.effective_scope is TemporalScope.UNKNOWN


def test_insufficient_is_a_candidate_at_best_historical():
    conf = policy()
    items = [evidence("e1", lineage="L1")]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    decision = scorer.decide(
        claim(), [Citation("e1", "supports")], verdict=Verdict("c1", "insufficient"), now=NOW
    )
    assert decision.state == "candidate"
    assert decision.effective_scope is TemporalScope.HISTORICAL


def test_human_asserted_evidence_bypasses_corroboration():
    conf = policy()
    item = evidence("e1", lineage="L1", source_type="user_stated")
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=[item])
    decision = scorer.decide(claim(), [Citation("e1", "supports")], verdict=None, now=NOW)
    assert decision.state == "accepted"


def test_a_citation_naming_a_missing_evidence_id_raises():
    conf = policy()
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf))
    with pytest.raises(CitationError):
        scorer.decide(claim(), [Citation("nope", "supports")], verdict=None, now=NOW)


def test_a_citation_speaking_for_someone_else_raises():
    conf = policy()
    item = evidence("e1", lineage="L1", speaker="someone-else")
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=[item])
    with pytest.raises(CitationError):
        scorer.decide(claim(), [Citation("e1", "supports")], verdict=None, now=NOW)


def test_present_scope_is_downgraded_on_stale_evidence():
    conf = policy()
    items = [evidence("e1", lineage="L1", days=400), evidence("e2", lineage="L2", days=400)]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    decision = scorer.decide(
        claim(scope=TemporalScope.PRESENT),
        [Citation("e1", "supports"), Citation("e2", "supports")],
        verdict=Verdict("c1", "supported"),
        now=NOW,
    )
    assert decision.effective_scope is TemporalScope.HISTORICAL


def test_quarantine_detector_produces_the_quarantined_state():
    conf = policy()
    scorer = Scorer(
        confidence=conf,
        retrieval=retrieval(conf),
        evidence=[evidence("e1")],
        quarantine_detector=lambda c, links: True,
    )
    decision = scorer.decide(
        claim(), [Citation("e1", "supports")], verdict=Verdict("c1", "supported"), now=NOW
    )
    assert decision.state == "quarantined"
    assert decision.effective_scope is TemporalScope.UNKNOWN


# -- retrieval ---------------------------------------------------------------


def test_ranking_is_deterministic_and_ordered():
    """The vector term dominates, which is what §3.8's weighting intends.

    A stale passage with high similarity outranks a fresh one with lower
    similarity. That is the documented behaviour of the four-term sum, not a
    bug — the recency term is only 0.20 of the weight.
    """
    conf = policy()
    items = [evidence("e1", lineage="L1"), evidence("e2", lineage="L2", days=200)]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    candidates = [
        RetrievalCandidate(evidence_id="e2", similarity=0.9, confidence=0.5),
        RetrievalCandidate(evidence_id="e1", similarity=0.4, confidence=0.9),
    ]
    ranked = scorer.rank(candidates, now=NOW)
    assert [item.evidence_id for item in ranked] == ["e2", "e1"]
    assert ranked[0].value > ranked[1].value
    # and the same inputs give the same order every time
    assert [item.evidence_id for item in scorer.rank(candidates, now=NOW)] == ["e2", "e1"]


def test_ranking_puts_a_fresher_passage_first_at_equal_similarity():
    conf = policy()
    items = [evidence("e1", lineage="L1"), evidence("e2", lineage="L2", days=200)]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    candidates = [
        RetrievalCandidate(evidence_id="e2", similarity=0.5, confidence=0.5),
        RetrievalCandidate(evidence_id="e1", similarity=0.5, confidence=0.5),
    ]
    ranked = scorer.rank(candidates, now=NOW)
    assert [item.evidence_id for item in ranked] == ["e1", "e2"]


def test_undated_evidence_gets_no_recency():
    conf = policy()
    item = Evidence(
        evidence_id="e1",
        segment_id="s",
        lineage_id="L1",
        source_id="e1",
        text="t",
        source_type="message",
        modality="chat",
        occurred_at=None,
        speaker="them",
    )
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=[item])
    ranked = scorer.rank([RetrievalCandidate("e1", 1.0, 1.0)], now=NOW)
    assert ranked[0].terms["recency"] == 0.0


# -- time is injected --------------------------------------------------------


def test_naive_now_is_rejected():
    conf = policy()
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf))
    with pytest.raises(ValueError):
        scorer.score(claim(), [], now=datetime(2026, 10, 20))


def test_two_runs_over_identical_evidence_agree():
    conf = policy()
    items = [evidence("e1", lineage="L1", days=17)]
    scorer = Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)
    first = scorer.score(claim(), [Link("e1", "supports")], now=NOW)
    second = scorer.score(claim(), [Link("e1", "supports")], now=NOW)
    assert first.value == second.value
    assert dict(first.terms) == dict(second.terms)
