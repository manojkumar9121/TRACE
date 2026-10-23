"""The reference confidence policy (PLAN.md Decision D5, milestones M−1).

Core ships **no default policy** (§6 A3): what a source type is worth is the
*integrator's* judgment about the risk of each provenance, and a default would
be exactly the silent assumption this project exists to remove. But a policy
with nothing to compare against is a policy nobody can sanity-check, so
``trace_traits`` declares **one example** — the table this package's CLI and
its verifier documentation were written against. It is a reference, with two
properties worth stating:

1. **It is an example, not a default.** Nothing in ``trace_core`` or in the
   reading path looks these numbers up. An integrator may adopt them, change
   them, or ignore them, and the arithmetic is the arithmetic either way.
2. **Human-derived sources outrank model-derived sources.** ``system_inference``
   is capped below ``message``, which is capped below what a person wrote down
   about themselves. If adopting a different table changes that *shape*, the
   thing to check is not the table but the claim about what deserves trust.

The half-lives below assume the human recency conversation: a behavioural trait
decays fastest, a factual statement slowest. Changing them changes *when* a
claim gets old; changing the weights changes *how much a claim is worth*.
"""

from __future__ import annotations

from trace_core import Scorer, TableConfidencePolicy, TableRetrievalPolicy
from trace_core.errors import PolicyError

__all__ = [
    "REFERENCE_POLICY",
    "REFERENCE_SOURCE_WEIGHTS",
    "reference_scorer",
]

#: Source-type weights, as one example of a policy (D5). The table is declared
#: here in one place; nothing else re-types it.
REFERENCE_SOURCE_WEIGHTS: dict[str, float] = {
    "user_stated": 1.00,
    "manual_edit": 1.00,
    "feedback": 0.85,
    "message": 0.70,
    "screenshot": 0.65,
    "system_inference": 0.55,
}

#: Half-lives in days, by claim kind: how fast each class of trait goes stale.
REFERENCE_HALF_LIVES: dict[str, float] = {
    "behavioral": 270.0,
    "tone": 180.0,
    "style": 540.0,
    "factual": 1095.0,
    "distributional": 540.0,
}

#: The example policy assembled from the two tables.
REFERENCE_POLICY: TableConfidencePolicy = TableConfidencePolicy(
    source_weights=REFERENCE_SOURCE_WEIGHTS,
    half_lives=REFERENCE_HALF_LIVES,
    corroboration_floor=2,
    coverage_denominator=3,
)


def reference_scorer() -> Scorer:
    """A scorer assembled from the reference policy.

    A convenience — a caller with its own should build theirs directly, and
    the retrieval weights here are the same declared defaults the CLI ships.
    """
    return Scorer(
        confidence=REFERENCE_POLICY,
        retrieval=TableRetrievalPolicy(
            weights={
                "vector": 0.55,
                "recency": 0.20,
                "source": 0.15,
                "confidence": 0.10,
            },
            source_weights=REFERENCE_POLICY.source_weights,
        ),
    )


def _check_tables_agree() -> None:
    """The policy is assembled from the tables; guard the assembly."""
    if set(REFERENCE_SOURCE_WEIGHTS) != set(REFERENCE_POLICY.source_weights):
        raise PolicyError("reference tables and policy disagree on keys")


_check_tables_agree()
