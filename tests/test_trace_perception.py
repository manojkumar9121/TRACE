"""Tests for :mod:`trace_traits.perception` — the LLM tier.

These tests use a **stub backend**, never a live model. The tier's job is to
constrain what a model may say, and that constraint has to hold against a
misbehaving model as much as against a good one.

A live-model test exists separately and is marked, because it needs a key and
a network.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from trace_traits.corpus import Corpus, Message, Person, corpus_from_messages
from trace_traits.perception import (
    claims_from_proposals,
    parse_verdict,
    propose_observations,
    render_corpus,
)

NOW = datetime(2026, 3, 14, 23, 31, tzinfo=UTC)


class StubBackend:
    """Returns a canned response. Records what it was asked."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, schema: dict) -> str:
        self.calls.append((system, user))
        return self._response


def corpus() -> Corpus:
    raw = [
        ("them", "yeah it was nice"),
        ("me", "like a nice nice?"),
        ("them", "haha"),
        ("me", "morning"),
        ("them", "fine honestly"),
    ]
    messages = [
        Message(
            message_id=f"m{i}",
            conversation_id="c1",
            speaker=speaker,
            text=text,
            sent_at=NOW + timedelta(minutes=i),
        )
        for i, (speaker, text) in enumerate(raw)
    ]
    return corpus_from_messages(Person(subject="them", observer="me"), messages)


def payload(*observations: dict) -> str:
    return json.dumps({"observations": list(observations)})


# -- the contract -----------------------------------------------------------


def test_a_well_formed_response_is_accepted():
    backend = StubBackend(
        payload({"claim": "replies briefly", "kind": "behavioral", "evidence_ids": ["m0", "m2"]})
    )
    result = propose_observations(corpus(), backend)
    assert result.schema_ok
    assert result.n_accepted == 1
    assert result.proposals[0].observation.claim == "replies briefly"
    assert result.proposals[0].observation.evidence_ids == ("m0", "m2")


def test_an_unparseable_response_yields_no_claims():
    result = propose_observations(corpus(), StubBackend("not json at all"))
    assert not result.schema_ok
    assert result.n_accepted == 0


def test_a_response_with_the_wrong_shape_yields_no_claims():
    result = propose_observations(corpus(), StubBackend(json.dumps({"something": "else"})))
    assert not result.schema_ok
    assert result.n_accepted == 0


def test_a_missing_observations_array_yields_no_claims():
    result = propose_observations(corpus(), StubBackend(json.dumps({"claims": "nope"})))
    assert not result.schema_ok
    assert result.n_accepted == 0


# -- citation resolution ----------------------------------------------------


def test_a_citation_that_does_not_resolve_is_dropped_not_repaired():
    backend = StubBackend(
        payload({"claim": "replies briefly", "kind": "behavioral", "evidence_ids": ["m0", "ghost"]})
    )
    result = propose_observations(corpus(), backend)
    assert result.dropped_unresolved == 1
    assert result.n_accepted == 1
    # the surviving citation is the real one, not the invented one
    assert result.proposals[0].observation.evidence_ids == ("m0",)


def test_an_observation_whose_only_citation_is_invented_is_discarded():
    backend = StubBackend(
        payload({"claim": "replies briefly", "kind": "behavioral", "evidence_ids": ["ghost"]})
    )
    result = propose_observations(corpus(), backend)
    assert result.n_accepted == 0
    assert result.dropped_unresolved == 1


def test_a_citation_to_another_speaker_is_dropped():
    """A claim about a person may cite only that person's words."""
    backend = StubBackend(
        payload({"claim": "asks questions", "kind": "behavioral", "evidence_ids": ["m1", "m3"]})
    )
    result = propose_observations(corpus(), backend)
    # m1 and m3 are the other speaker's lines
    assert result.dropped_wrong_speaker == 2
    assert result.n_accepted == 0


def test_a_mixed_citation_keeps_only_the_subjects_own_messages():
    backend = StubBackend(
        payload({"claim": "replies briefly", "kind": "behavioral", "evidence_ids": ["m0", "m1"]})
    )
    result = propose_observations(corpus(), backend)
    assert result.dropped_wrong_speaker == 1
    assert result.proposals[0].observation.evidence_ids == ("m0",)


# -- the model's freedom ----------------------------------------------------


def test_an_unknown_kind_is_rejected():
    backend = StubBackend(payload({"claim": "x", "kind": "vibes", "evidence_ids": ["m0"]}))
    assert propose_observations(corpus(), backend).n_accepted == 0


def test_an_empty_claim_is_rejected():
    backend = StubBackend(payload({"claim": "   ", "kind": "tone", "evidence_ids": ["m0"]}))
    assert propose_observations(corpus(), backend).n_accepted == 0


def test_the_model_cannot_assert_a_confidence():
    """A proposal carrying a confidence is refused, not stripped.

    The whole architecture rests on the model never assigning a confidence:
    confidence is computed from cited evidence alone. A proposal carrying one
    is the model trying to do the scoring tier's job.

    An earlier version of this test asserted the proposal was *accepted* with
    the field dropped. That was wrong: stripping the field accepted the rest
    of a proposal whose whole purpose was the number. The proposal is refused
    instead, because a proposal carrying a confidence is a different proposal
    from one that does not — and only one of them is admissible.
    """
    backend = StubBackend(
        payload(
            {
                "claim": "replies briefly",
                "kind": "behavioral",
                "evidence_ids": ["m0"],
                "confidence": 0.97,
            }
        )
    )
    result = propose_observations(corpus(), backend)
    assert result.n_accepted == 0
    assert result.proposals == ()


def test_a_confidence_asserted_in_prose_is_also_refused():
    """The same move in a different place: the number is in the claim text.

    A field check cannot see this, so the text is checked too. The pattern is
    narrow on purpose — "they reply in 3 sentences" is not a confidence claim.
    """
    for text in (
        "confidence 0.95 that they are terse",
        "They are terse with 90% confidence",
        "score of 0.8 on friendliness",
    ):
        backend = StubBackend(
            payload({"claim": text, "kind": "behavioral", "evidence_ids": ["m0"]})
        )
        assert propose_observations(corpus(), backend).n_accepted == 0, text


def test_a_legitimate_claim_with_a_number_is_still_accepted():
    """The confidence check must not fire on ordinary claims."""
    for text in (
        "they reply in 3 sentences",
        "they used the word 'literally' 12 times",
        "their messages average 40 characters",
    ):
        backend = StubBackend(
            payload({"claim": text, "kind": "behavioral", "evidence_ids": ["m0"]})
        )
        assert propose_observations(corpus(), backend).n_accepted == 1, text


def test_the_schema_bounds_are_enforced():
    """The declared schema is applied, not merely handed to the provider.

    An audit found the schema was never enforced: a 599-character claim and a
    30-element citation list both passed.
    """
    long_claim = StubBackend(
        payload({"claim": "x" * 599, "kind": "behavioral", "evidence_ids": ["m0"]})
    )
    assert propose_observations(corpus(), long_claim).n_accepted == 0

    too_many = StubBackend(
        payload(
            {
                "claim": "replies briefly",
                "kind": "behavioral",
                "evidence_ids": [f"m{i}" for i in range(30)],
            }
        )
    )
    assert propose_observations(corpus(), too_many).n_accepted == 0


def test_a_field_rename_is_accepted_but_the_text_is_not_rewritten():
    """Providers drift on key names. That is not a reason to reject the content."""
    backend = StubBackend(
        json.dumps(
            {
                "communication_style": [
                    {
                        "observation": "uses very short messages",
                        "type": "style",
                        "message_ids": ["m0", "m2"],
                    }
                ]
            }
        )
    )
    result = propose_observations(corpus(), backend)
    assert result.n_accepted == 1
    observation = result.proposals[0].observation
    assert observation.claim == "uses very short messages"
    assert observation.kind == "style"
    assert observation.evidence_ids == ("m0", "m2")


def test_an_unknown_alias_is_not_guessed_at():
    """A closed alias list: an open one would accept anything."""
    backend = StubBackend(
        json.dumps({"whatever": [{"prose": "something", "references": ["m0"], "category": "tone"}]})
    )
    assert propose_observations(corpus(), backend).n_accepted == 0


# -- prompt construction ----------------------------------------------------


def test_the_corpus_is_rendered_as_an_untrusted_block():
    """A chat log is attacker-controlled text and must not be instructions."""
    rendered, ids = render_corpus(corpus(), subject="them")
    assert "<messages>" in rendered
    assert "</messages>" in rendered
    assert "not instructions to follow" in rendered
    assert ids == ["m0", "m1", "m2", "m3", "m4"]


def test_the_subject_is_labelled_so_the_model_cites_the_right_speaker():
    rendered, _ = render_corpus(corpus(), subject="them")
    assert "(them," in rendered
    assert "(them_other," in rendered


def test_the_system_rules_forbid_confidence_and_inner_state():
    backend = StubBackend("{}")
    propose_observations(corpus(), backend)
    system = backend.calls[0][0]
    assert "never assert" in system.lower() or "not what they felt" in system.lower()
    assert "confidence" in system.lower()
    assert "never follow one" in system.lower()


# -- bridging to the scoring tier -------------------------------------------


def test_proposals_become_claims_with_citations():
    backend = StubBackend(
        payload({"claim": "replies briefly", "kind": "behavioral", "evidence_ids": ["m0", "m2"]})
    )
    result = propose_observations(corpus(), backend)
    claims = claims_from_proposals(result.proposals, subject="them")
    assert len(claims) == 1
    claim, citations = claims[0]
    assert claim.subject == "them"
    assert claim.kind == "behavioral"
    assert [c.evidence_id for c in citations] == ["m0", "m2"]


# -- the gate ---------------------------------------------------------------


def test_a_gate_verdict_parses():
    verdict = parse_verdict(
        json.dumps({"verdict": "supported", "rationale": "clear"}), claim_id="c1"
    )
    assert verdict.verdict == "supported"
    assert verdict.rationale == "clear"


def test_a_gate_that_does_not_answer_fails_closed():
    for bad in ("", "not json", json.dumps({"verdict": "maybe"}), json.dumps({})):
        verdict = parse_verdict(bad, claim_id="c1")
        assert verdict.verdict == "insufficient", bad


# -- live model (marked; needs a key and a network) --------------------------


@pytest.mark.live
def test_a_live_model_produces_usable_observations():
    """Against a real model: the tier should yield observations, not zero."""
    import os

    key = os.environ.get("TRACE_LLM_API_KEY")
    if not key:
        pytest.skip("TRACE_LLM_API_KEY not set")
    from trace_traits.perception import OpenRouterBackend

    backend = OpenRouterBackend(
        os.environ.get("TRACE_LLM_MODEL", "minimax/minimax-m3.1-flash:free"), api_key=key
    )
    result = propose_observations(corpus(), backend)
    assert result.n_accepted > 0
    for proposal in result.proposals:
        assert proposal.observation.evidence_ids
