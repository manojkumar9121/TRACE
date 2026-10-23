"""Regression tests for the defects found by the independent audit.

Each test here corresponds to a bug that shipped in an earlier revision and
was caught by an adversarial audit pass rather than by the test suite. They are
kept together, and named for what they prevent, because a bug that was found
once by luck will be found again by accident.

The bugs, in the order they were found:

**H3 — an unattributed row satisfied the corroboration floor.** INV-1 counts
independent *attributed* lineages. An anonymous row was counted, so one
attributed message plus one anonymous message corroborated a claim about a
named person.

**H4 — a neutral citation counted as support.** A citation is never neutral
(CDI-003). The neutral one was skipped by the stance filter and the *other*
citation carried the claim, so a neutral citation was silently ignored rather
than rejected — and a caller could pass only neutral citations and get
through the floor.

**H5 — an undefined verdict value was treated as "supported".** The verdict
was compared against `"unsupported"` and `"insufficient"` and fell through to
the supported branch. A typo — or a forged verdict — read as support.

**H6 — the speaker check skipped contradicting citations.** The stance filter
ran *before* the attribution check, so a contradicting citation from the wrong
speaker never raised.

**H9 — `BY_FINGERPRINT` gave every distinct message its own lineage.** Fourteen
messages in one conversation counted as fourteen independent sources, which is
exactly the corroboration inflation the lineage mechanism exists to prevent.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trace_core import (
    Citation,
    CitationError,
    Claim,
    Evidence,
    Scorer,
    TableConfidencePolicy,
    TableRetrievalPolicy,
    Verdict,
)
from trace_traits import LineageStrategy, Message, Person, corpus_from_messages

NOW = datetime(2026, 10, 20, tzinfo=UTC)


def policy() -> TableConfidencePolicy:
    return TableConfidencePolicy(
        source_weights={"user_stated": 1.0, "message": 0.70},
        half_lives={"behavioral": 270},
        corroboration_floor=2,
        coverage_denominator=3,
    )


def retrieval(conf: TableConfidencePolicy) -> TableRetrievalPolicy:
    return TableRetrievalPolicy(
        weights={"vector": 0.55, "recency": 0.20, "source": 0.15, "confidence": 0.10},
        source_weights=conf.source_weights,
    )


def evidence(eid: str, lineage: str, speaker: str | None = "them") -> Evidence:
    return Evidence(
        evidence_id=eid,
        segment_id="s",
        lineage_id=lineage,
        source_id=eid,
        text="hi",
        source_type="message",
        modality="chat",
        occurred_at=NOW,
        speaker=speaker,
    )


def scorer(items: list[Evidence]) -> Scorer:
    conf = policy()
    return Scorer(confidence=conf, retrieval=retrieval(conf), evidence=items)


def claim() -> Claim:
    return Claim(claim_id="c1", text="x", subject="them", kind="behavioral")


# -- H3 ----------------------------------------------------------------------


def test_an_unattributed_row_does_not_satisfy_the_corroboration_floor():
    items = [evidence("e1", "L1", speaker=None), evidence("e2", "L2")]
    decision = scorer(items).decide(
        claim(),
        [Citation("e1", "supports"), Citation("e2", "supports")],
        verdict=Verdict("c1", "supported"),
        now=NOW,
    )
    assert decision.state == "candidate"


def test_two_attributed_lineages_still_satisfy_the_floor():
    items = [evidence("e1", "L1"), evidence("e2", "L2")]
    decision = scorer(items).decide(
        claim(),
        [Citation("e1", "supports"), Citation("e2", "supports")],
        verdict=Verdict("c1", "supported"),
        now=NOW,
    )
    assert decision.state == "accepted"


# -- H4 ----------------------------------------------------------------------


def test_a_neutral_citation_is_rejected():
    items = [evidence("e1", "L1"), evidence("e2", "L2")]
    with pytest.raises(CitationError):
        scorer(items).decide(
            claim(),
            [Citation("e1", "supports"), Citation("e2", "neutral")],  # type: ignore[arg-type]
            verdict=Verdict("c1", "supported"),
            now=NOW,
        )


def test_only_neutral_citations_cannot_pass_the_floor():
    """The original bug: the neutral citation was skipped, so the *other*
    citation carried the claim and the floor was met with one lineage."""
    items = [evidence("e1", "L1"), evidence("e2", "L2")]
    with pytest.raises(CitationError):
        scorer(items).decide(
            claim(),
            [Citation("e1", "neutral"), Citation("e2", "neutral")],  # type: ignore[arg-type]
            verdict=Verdict("c1", "supported"),
            now=NOW,
        )


# -- H5 ----------------------------------------------------------------------


def test_an_undefined_verdict_is_not_treated_as_supported():
    items = [evidence("e1", "L1"), evidence("e2", "L2")]
    decision = scorer(items).decide(
        claim(),
        [Citation("e1", "supports"), Citation("e2", "supports")],
        verdict=Verdict("c1", "SUPPRTED"),  # type: ignore[arg-type]
        now=NOW,
    )
    assert decision.state == "candidate"
    assert [cap.code for cap in decision.caps_applied] == ["invalid_verdict"]


def test_a_forged_verdict_cannot_authorise_a_promotion():
    items = [evidence("e1", "L1"), evidence("e2", "L2")]
    decision = scorer(items).decide(
        claim(),
        [Citation("e1", "supports"), Citation("e2", "supports")],
        verdict=Verdict("c1", "true"),  # type: ignore[arg-type]
        now=NOW,
    )
    assert decision.state == "candidate"


# -- H6 ----------------------------------------------------------------------


def test_a_contradicting_citation_from_the_wrong_speaker_raises():
    items = [evidence("e1", "L1"), evidence("e2", "L2", speaker="someone-else")]
    with pytest.raises(CitationError):
        scorer(items).decide(
            claim(),
            [Citation("e1", "supports"), Citation("e2", "contradicts")],
            verdict=Verdict("c1", "supported"),
            now=NOW,
        )


# -- H9 ----------------------------------------------------------------------


def test_the_fingerprint_strategy_treats_one_conversation_as_one_lineage():
    messages = [
        Message(
            message_id=f"m{i}",
            conversation_id="conv1",
            speaker="them",
            text=f"a distinct message number {i}",
            sent_at=NOW,
        )
        for i in range(14)
    ]
    corpus = corpus_from_messages(
        Person(subject="them"), messages, lineage_strategy=LineageStrategy.BY_FINGERPRINT
    )
    assert len({corpus.lineage_of(item) for item in corpus.messages}) == 1


def test_the_fingerprint_strategy_separates_distinct_conversations():
    messages = [
        Message(
            message_id=f"m{i}",
            conversation_id=f"conv{i}",
            speaker="them",
            text=f"conversation {i}",
            sent_at=NOW,
        )
        for i in range(5)
    ]
    corpus = corpus_from_messages(
        Person(subject="them"), messages, lineage_strategy=LineageStrategy.BY_FINGERPRINT
    )
    assert len({corpus.lineage_of(item) for item in corpus.messages}) == 5


def test_a_re_uploaded_conversation_is_one_lineage():
    messages = [
        Message(
            message_id=f"m{i}",
            conversation_id="convX",
            speaker="them",
            text="hello",
            sent_at=NOW,
        )
        for i in range(3)
    ]
    corpus = corpus_from_messages(
        Person(subject="them"), messages, lineage_strategy=LineageStrategy.BY_FINGERPRINT
    )
    assert len({corpus.lineage_of(item) for item in corpus.messages}) == 1


def test_one_conversation_cannot_corroborate_itself():
    """The end-to-end consequence: 14 messages in one conversation is one source."""
    items = [evidence(f"e{i}", "L1") for i in range(14)]
    decision = scorer(items).decide(
        claim(),
        [Citation(f"e{i}", "supports") for i in range(14)],
        verdict=Verdict("c1", "supported"),
        now=NOW,
    )
    assert decision.state == "candidate"
