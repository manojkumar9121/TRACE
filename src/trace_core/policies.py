"""Policy — integrator-owned, validated, and unable to raise confidence.

PLAN.md §4 splits policy into two protocols because the formulas genuinely
differ; conflating them is a bug waiting to happen.

**Two thresholds, not one.** The pre-port implementation used two distinct
numbers: ``MIN_INDEPENDENT_FOR_PROMOTION = 2`` (a claim needs two independent
conversations to be *accepted*) and ``TARGET_INDEPENDENT = 3`` (the
denominator that makes coverage reach 1.0). PLAN.md §4.1 collapses these into
one ``corroboration_threshold()``, which cannot represent the verified
behaviour: raising it to 3 would reject claims the working system accepts, and
leaving it at 2 would saturate coverage two conversations early. This module
therefore exposes them separately — ``corroboration_threshold()`` for the INV-1
floor and ``coverage_target()`` for the denominator — and validates both.

Every weight is validated at construction (INV-5, PLAN.md §4.3): finite, in
range, and the core's own caps are applied *after* the policy returns, so a
policy can lower a value or add a ceiling but never restore what an invariant
removed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Re-exported for compatibility: the INV-2 mechanic was defined here before it
# moved to :mod:`trace_core.attribution`, whose names this module's ``__all__``
# still lists.
from trace_core.attribution import MIN_CORROBORATION, claim_scope_is_current
from trace_core.errors import PolicyError
from trace_core.types import EvidenceSummary

__all__ = [
    "ConfidencePolicy",
    "RetrievalPolicy",
    "TableConfidencePolicy",
    "TableRetrievalPolicy",
    "claim_scope_is_current",
    "cited_lineages",
    "evidence_summary_terms",
]


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise PolicyError(f"{name} must be finite, got {value!r}")
    return number


def _unit(name: str, value: float) -> float:
    number = _finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise PolicyError(f"{name} must be in [0, 1], got {number}")
    return number


def _validate_confidence_tables(
    *,
    source_weights: Mapping[str, float],
    half_lives: Mapping[str, float],
    corroboration_floor: int,
    coverage_denominator: int,
) -> None:
    for name, weight in source_weights.items():
        _unit(f"source_weight[{name!r}]", weight)
    for kind, half_life in half_lives.items():
        value = _finite(f"half_life_days[{kind!r}]", half_life)
        if value <= 0:
            raise PolicyError(f"half_life_days[{kind!r}] must be positive, got {value}")
    if corroboration_floor < MIN_CORROBORATION:
        # The INV-1 floor, single-sourced from the decision layer. A policy
        # may raise it, never lower it.
        raise PolicyError(
            f"corroboration_floor must be >= {MIN_CORROBORATION} (the INV-1 floor), "
            f"got {corroboration_floor}"
        )
    if coverage_denominator < 1:
        raise PolicyError(f"coverage_denominator must be >= 1, got {coverage_denominator}")


def _validate_retrieval_weights(weights: Mapping[str, float]) -> None:
    for name, weight in weights.items():
        value = _finite(f"retrieval weight[{name!r}]", weight)
        if value < 0:
            # Non-negative, but not required to sum to 1: the original did not
            # require it, and inventing the constraint would be a silent
            # behaviour change.
            raise PolicyError(f"retrieval weight[{name!r}] must be non-negative, got {value}")


@runtime_checkable
class ConfidencePolicy(Protocol):
    """The multiplicative confidence formula's inputs (PLAN.md §4.1).

    Implementations supply *tables*, not arithmetic. The formula itself is the
    core's, so two integrators cannot compute "confidence" two different ways.
    """

    def source_weight(self, source_type: str) -> float:
        """``W[source_type]`` — how much a source can be trusted at all.

        The **minimum** across cited evidence is what enters the formula, so one
        weak source caps the whole claim.
        """

    def half_life_days(self, kind: str) -> float:
        """Days for a claim of this ``kind`` to lose half its weight."""

    def corroboration_threshold(self) -> int:
        """How many independent lineages a claim needs to be accepted (INV-1).

        Must be ``>= 2``. Core enforces the floor; a policy may raise it.
        """

    def coverage_target(self) -> int:
        """The denominator that makes coverage reach 1.0.

        Distinct from :meth:`corroboration_threshold` because the pre-port
        system used two numbers and one cannot express both.
        """

    def sample_adequacy(self, n_evidence: int) -> float:
        """How adequate the sample is, as a factor in ``[0, 1]``.

        ``coverage`` counts *lineages*, so it cannot distinguish a claim
        measured from eight messages from one measured from two hundred. This
        is where quantity enters.

        The default is ``1.0`` — no effect — so a policy that does not care
        about sample size reproduces the ported arithmetic exactly. A policy
        that does care supplies a target and gets a factor of
        ``min(1, n_evidence / target)``.
        """


@runtime_checkable
class RetrievalPolicy(Protocol):
    """The additive retrieval formula's inputs (PLAN.md §4.1)."""

    def retrieval_weights(self) -> Mapping[str, float]:
        """The four coefficients, keyed by term name."""

    def source_weight(self, source_type: str) -> float:
        """``W[source_type]`` — the same table the confidence policy uses."""


@dataclass(frozen=True, slots=True)
class TableConfidencePolicy:
    """A confidence policy expressed as tables.

    The dataclass fields are named apart from the protocol methods on purpose:
    a field named ``corroboration_threshold`` would shadow the method that
    satisfies the protocol, and the structural check would then pass for the
    wrong reason.
    """

    source_weights: Mapping[str, float]
    half_lives: Mapping[str, float]
    #: How many independent lineages a claim needs to be accepted (INV-1).
    corroboration_floor: int = 2
    #: The denominator that makes coverage reach 1.0.
    coverage_denominator: int = 3
    #: The weight for a source type the table does not name. ``None`` means
    #: "use the lowest known weight", because a default that ranked unknown
    #: evidence as authoritative would let a mislabelled row outrank a
    #: user-stated one.
    unknown_source_weight: float | None = None
    #: The number of evidence rows a claim wants before sample size stops
    #: counting against it. ``None`` means sample size does not enter the
    #: score at all, which is the ported behaviour.
    sample_target: int | None = None

    def __post_init__(self) -> None:
        _validate_confidence_tables(
            source_weights=self.source_weights,
            half_lives=self.half_lives,
            corroboration_floor=self.corroboration_floor,
            coverage_denominator=self.coverage_denominator,
        )
        if self.unknown_source_weight is not None:
            _unit("unknown_source_weight", self.unknown_source_weight)
        if self.sample_target is not None and self.sample_target <= 0:
            # IF-001.9: positive when supplied. Checked at construction so a
            # bad policy is rejected where it was written, not at the first
            # call that happens to use it.
            raise PolicyError(
                f"sample_target must be positive when supplied, got {self.sample_target}"
            )

    def source_weight(self, source_type: str) -> float:
        weights = self.source_weights
        if source_type in weights:
            return _unit(f"source_weight[{source_type!r}]", weights[source_type])
        if not weights:
            raise PolicyError("source_weights is empty, so no source can be weighted")
        conservative = min(
            _unit(f"source_weight[{key!r}]", value) for key, value in weights.items()
        )
        if self.unknown_source_weight is not None:
            override = _unit("unknown_source_weight", self.unknown_source_weight)
            # IF-001.6: an unrecognised source takes the MOST CONSERVATIVE
            # known weight, never an optimistic one. An override may make it
            # *more* conservative, never less — a policy that could raise the
            # weight of an unknown source could promote a mislabelled row
            # above a user-stated one.
            if override > conservative:
                raise PolicyError(
                    f"unknown_source_weight ({override}) may not exceed the most "
                    f"conservative known weight ({conservative}); an unrecognised "
                    "source must never be trusted more than a known one"
                )
            return override
        return conservative

    def half_life_days(self, kind: str) -> float:
        if kind in self.half_lives:
            return _finite(f"half_life_days[{kind!r}]", self.half_lives[kind])
        if not self.half_lives:
            raise PolicyError("half_lives is empty, so no claim kind can decay")
        # An unknown kind takes the *shortest* half-life, not an optimistic
        # default: a mislabelled claim must not look more durable than it is.
        return min(
            _finite(f"half_life_days[{key!r}]", value) for key, value in self.half_lives.items()
        )

    def corroboration_threshold(self) -> int:
        return self.corroboration_floor

    def coverage_target(self) -> int:
        return self.coverage_denominator

    def sample_adequacy(self, n_evidence: int) -> float:
        if self.sample_target is None:
            return 1.0
        if self.sample_target <= 0:
            raise PolicyError(f"sample_target must be positive, got {self.sample_target}")
        if n_evidence <= 0:
            return 0.0
        return min(1.0, n_evidence / self.sample_target)


@dataclass(frozen=True, slots=True)
class TableRetrievalPolicy:
    """A retrieval policy expressed as tables."""

    #: The four coefficients. Named apart from :meth:`retrieval_weights` so the
    #: field does not shadow the method that satisfies the protocol.
    weights: Mapping[str, float]
    source_weights: Mapping[str, float]
    unknown_source_weight: float | None = None

    def __post_init__(self) -> None:
        _validate_retrieval_weights(self.weights)
        for name, weight in self.source_weights.items():
            _unit(f"source_weight[{name!r}]", weight)
        if self.unknown_source_weight is not None:
            _unit("unknown_source_weight", self.unknown_source_weight)

    def retrieval_weights(self) -> Mapping[str, float]:
        return dict(self.weights)

    def source_weight(self, source_type: str) -> float:
        if source_type in self.source_weights:
            return _unit(f"source_weight[{source_type!r}]", self.source_weights[source_type])
        if self.unknown_source_weight is not None:
            return _unit("unknown_source_weight", self.unknown_source_weight)
        if not self.source_weights:
            raise PolicyError("source_weights is empty, so no source can be weighted")
        return min(
            _unit(f"source_weight[{key!r}]", value) for key, value in self.source_weights.items()
        )


def evidence_summary_terms(summary: EvidenceSummary) -> dict[str, float]:
    """The summary, as flat terms. Informational, not multiplicative."""
    return {
        "n_evidence": float(summary.n_evidence),
        "n_segments": float(summary.n_segments),
        "n_lineages": float(summary.n_lineages),
        "n_undated": float(summary.n_undated),
        "n_unattributed": float(summary.n_unattributed),
        "n_contradicting": float(summary.n_contradicting),
    }


def cited_lineages(citations: Sequence[str], lineage_of: Mapping[str, str]) -> set[str]:
    """Distinct lineages among the given evidence ids."""
    return {lineage_of[cid] for cid in citations if cid in lineage_of}
