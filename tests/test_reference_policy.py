"""Tests for the reference confidence policy (PLAN.md D5, M−1).

Core ships no default policy; ``trace_traits`` declares one example so the
CLI, the docs and the tests cannot drift apart. These tests pin the example
where it matters: one table, one assembly, and the human-over-model shape
that any adopted policy preserves on its own responsibility.
"""

from __future__ import annotations

import pytest

from trace_traits import (
    REFERENCE_HALF_LIVES,
    REFERENCE_POLICY,
    REFERENCE_SOURCE_WEIGHTS,
    reference_scorer,
)


def test_the_policy_is_assembled_from_the_declared_tables() -> None:
    """One table, one assembly: editing a copy must not diverge silently."""
    assert REFERENCE_POLICY.source_weights == REFERENCE_SOURCE_WEIGHTS
    assert REFERENCE_POLICY.half_lives == REFERENCE_HALF_LIVES


def test_a_human_asserted_source_is_worth_a_model_inference() -> None:
    """The shape an example policy must keep: humans over models."""
    weights = REFERENCE_SOURCE_WEIGHTS
    assert weights["user_stated"] == pytest.approx(1.0)
    assert weights["manual_edit"] == pytest.approx(1.0)
    assert weights["system_inference"] < weights["message"]
    assert weights["message"] < weights["feedback"] < weights["user_stated"]


def test_the_reference_scorer_wires_the_reference_policy() -> None:
    scorer = reference_scorer()
    # A private read on purpose: the pin is that the convenience factory
    # wired the declared policy, not a copy of it.
    assert scorer._confidence is REFERENCE_POLICY  # noqa: SLF001


def test_an_unknown_source_reads_the_lowest_known_weight() -> None:
    """Unnameable provenance is treated as the least trustworthy kind: the
    policy's own declared rule, applied to the example table."""
    scorer = reference_scorer()
    # A private read on purpose: the pin is the wired policy, not a copy.
    confidence = scorer._confidence  # noqa: SLF001
    assert confidence.source_weight("carrier_pigeon") == min(REFERENCE_SOURCE_WEIGHTS.values())
