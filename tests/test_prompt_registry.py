"""Tests for the prompt-version registry (PLAN.md D9, IF-002.4).

The registry exists so a finding can name the words that were asked: a prompt
that can change under a result is a prompt whose results cannot be re-derived.
Every prompt the perception tier sends is therefore a closed-registry entry
with a version, and the proposing result carries the entry it actually used.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from trace_traits.corpus import Corpus, Message, Person, corpus_from_messages
from trace_traits.perception import (
    OBSERVATIONS_PROMPT_V1,
    PROMPT_VERSIONS,
    SYSTEM_RULES,
    PerceptionResult,
    PerceptionTier,
    prompt_version,
    propose_observations,
    verification_prompt,
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


def response_with_claims() -> str:
    return '{"observations": [{"claim": "they sign off politely", "kind": "tone", "evidence_ids": ["m0", "m2"]}]}'


# -- the registry itself ------------------------------------------------------


def test_the_registry_holds_a_closed_set_of_prompts():
    """Two roles, each with exactly one shipped version, and nothing else."""
    assert sorted(PROMPT_VERSIONS) == ["propose/1", "verify/1"]


def test_propose_v1_is_the_words_the_call_actually_sent():
    """The entry *is* the system rules, not a copy that can drift."""
    assert prompt_version("propose", "1").render() is SYSTEM_RULES


def test_verify_v1_is_the_verification_prompt_function_itself():
    """Editing the prompt is editing the entry: identity, not duplication."""
    assert prompt_version("verify", "1").render is verification_prompt


def test_rendering_verify_v1_by_the_registry_spells_the_same_text():
    body = prompt_version("verify", "1").render(
        corpus(), claim_text="they sign off politely", evidence_ids=["m0", "m2"]
    )
    assert body == verification_prompt(corpus(), "they sign off politely", ["m0", "m2"])


def test_an_unknown_prompt_refuses_and_names_what_exists():
    with pytest.raises(KeyError, match="propose/2"):
        prompt_version("propose", "2")
    with pytest.raises(KeyError, match="propose/1, verify/1"):
        prompt_version("verdict", "1")


def test_a_prompt_version_is_frozen_for_its_whole_life():
    entry = OBSERVATIONS_PROMPT_V1
    with pytest.raises(FrozenInstanceError):
        entry.version = "2"  # type: ignore[misc]


# -- what a proposing call records --------------------------------------------


def test_a_proposing_result_names_its_prompt_and_its_model():
    result = propose_observations(corpus(), StubBackend(response_with_claims()), model="stub-1")
    assert result.prompt is OBSERVATIONS_PROMPT_V1
    assert result.prompt_id == "propose/1"
    assert result.model == "stub-1"


def test_the_prompt_name_survives_a_schema_failure():
    """The version is provenance for the *attempt*, not just the success."""
    result = propose_observations(corpus(), StubBackend("not json at all"), model="stub-1")
    assert result.schema_ok is False
    assert result.prompt_id == "propose/1"
    assert result.model == "stub-1"


def test_an_undeclared_model_says_so():
    result = propose_observations(corpus(), StubBackend(response_with_claims()))
    assert result.prompt_id == "propose/1"
    assert result.model is None
    assert "model" not in result.as_lines()[0]
    assert result.as_lines()[0] == "prompt propose/1"


def test_the_output_names_the_words_it_was_asked_with():
    result = propose_observations(corpus(), StubBackend(response_with_claims()), model="stub-1")
    first, second = result.as_lines()
    assert first == "prompt propose/1 · model stub-1"
    assert "sign off politely" in second


def test_the_tier_carries_the_model_and_a_call_may_name_a_better_one():
    base = PerceptionTier(StubBackend(response_with_claims()), subject="them", model="a")
    overtaken = PerceptionTier(StubBackend(response_with_claims()), model="b")
    assert base.propose(corpus()).model == "a"
    assert baseline_and_caller_agree(base, "b")
    assert overtaken.propose(corpus()).model == "b"


def baseline_and_caller_agree(tier: PerceptionTier, model: str) -> bool:
    """A per-call name wins over the tier's constructor, as the specific one."""
    return tier.propose(corpus(), model=model).model == model


# -- the record is load-bearing, not decorative --------------------------------


def test_a_hand_built_result_reports_an_undeclared_prompt():
    """Only results from real calls may claim a version."""
    bare = PerceptionResult(
        proposals=(), dropped_unresolved=0, dropped_wrong_speaker=0, raw="", schema_ok=False
    )
    assert bare.prompt_id == "(undeclared)"
    assert bare.prompt is None
