"""The scoring engine: two formulas, five invariants, no clock (PLAN.md §3, §4).

**Confidence is multiplicative.** A product of four factors, so any one of them
approaching zero collapses the result. This is why one weak term cannot be
argued away by three strong ones.

**Retrieval is additive.** Four weighted terms, each in ``[0, 1]``.

The two are separate because the formulas genuinely differ, and conflating them
is a bug waiting to happen (PLAN.md §4.1).

**No operation reads the wall clock.** ``now`` is injected everywhere, so two
runs over identical evidence cannot disagree for no stated reason, and the
golden fixtures are deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

# Re-exported for compatibility: the decision layer's constants were defined
# here before the decision half moved to :mod:`trace_core.attribution`, whose
# names this module's ``__all__`` still lists.
from trace_core.attribution import (
    HUMAN_ASSERTED_SOURCES,
    MIN_CORROBORATION,
    reject_mixed_timestamps,
    summarise_evidence,
)
from trace_core.attribution import (
    decide as _decide,
)
from trace_core.errors import CitationError
from trace_core.policies import ConfidencePolicy, RetrievalPolicy, evidence_summary_terms
from trace_core.temporal import age_days, decay
from trace_core.types import (
    Cap,
    Citation,
    Claim,
    Decision,
    Evidence,
    Link,
    RetrievalCandidate,
    Scored,
    Verdict,
)

__all__ = [
    "CONFIDENCE_TERMS",
    "HUMAN_ASSERTED_SOURCES",
    "RETRIEVAL_TERMS",
    "Scorer",
]

#: The four multiplicative factors, in the order they are reported. A score
#: reconstructs from exactly these, so they are named once.
CONFIDENCE_TERMS: tuple[str, ...] = (
    "src_weight",
    "coverage",
    "freshness",
    "contradiction",
    "sample_adequacy",
)

#: The four additive retrieval terms.
RETRIEVAL_TERMS: tuple[str, ...] = ("vector", "recency", "source", "confidence")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class Scorer:
    """Computes and explains. Decides nothing about what the numbers mean.

    The scorer is handed an evidence set and asked about claims over it. It
    never reads the clock, never embeds text, and never generates candidates.
    """

    def __init__(
        self,
        *,
        confidence: ConfidencePolicy,
        retrieval: RetrievalPolicy,
        evidence: Sequence[Evidence] = (),
        quarantine_detector: Callable[[Claim, Sequence[Link]], bool] | None = None,
    ) -> None:
        self._confidence = confidence
        self._retrieval = retrieval
        self._evidence: dict[str, Evidence] = {item.evidence_id: item for item in evidence}
        self._quarantine_detector = quarantine_detector

    # -- evidence set ------------------------------------------------------

    @property
    def evidence(self) -> Mapping[str, Evidence]:
        """The evidence set, by id. Immutable from the caller's side."""
        return dict(self._evidence)

    def with_evidence(self, evidence: Sequence[Evidence]) -> Scorer:
        """A scorer over a different evidence set, with the same policies.

        Returns a new scorer rather than mutating this one: a scorer is a
        statement about a fixed evidence set, and two claims scored against
        two corpora must not share one.
        """
        return Scorer(
            confidence=self._confidence,
            retrieval=self._retrieval,
            evidence=evidence,
            quarantine_detector=self._quarantine_detector,
        )

    def _require(self, evidence_id: str) -> Evidence:
        try:
            return self._evidence[evidence_id]
        except KeyError:
            raise CitationError(f"no evidence with id {evidence_id!r}") from None

    # -- confidence --------------------------------------------------------

    def score(self, claim: Claim, links: Sequence[Link], *, now: datetime) -> Scored:
        """Score one claim over the given evidence links.

        The four factors multiply; each is reported by name. Nothing is folded
        into the score that a reader cannot reconstruct.
        """
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")

        cited: list[Evidence] = []
        contradicting: list[Evidence] = []
        for link in links:
            item = self._require(link.evidence_id)
            if link.stance == "contradicts":
                contradicting.append(item)
            elif link.stance == "supports":
                cited.append(item)

        if not cited:
            # No citation at all: value 0.0 with a named cap. Absence is an
            # answer, not a failure.
            summary = summarise_evidence(cited, contradicting)
            return Scored(
                claim_id=claim.claim_id,
                evidence_id=None,
                value=0.0,
                terms={name: 0.0 for name in CONFIDENCE_TERMS},
                evidence=summary,
                caps_applied=(Cap("no_evidence", "Nothing supports this claim."),),
            )

        reject_mixed_timestamps(cited + contradicting)
        summary = summarise_evidence(cited, contradicting)

        src_weight = min(self._confidence.source_weight(item.source_type) for item in cited)
        coverage = min(1.0, summary.n_lineages / max(1, self._confidence.coverage_target()))
        age = self._age_of(cited, now=now)
        freshness = decay(age, self._confidence.half_life_days(claim.kind))
        denominator = max(1, len(cited))
        contradiction = 1.0 - min(1.0, len(contradicting) / denominator)
        # Sample size enters here. `coverage` counts lineages, so it cannot
        # distinguish a claim measured from eight messages from one measured
        # from two hundred. The default factor is 1.0, so a policy that does
        # not care reproduces the ported arithmetic exactly.
        adequacy = _clamp01(self._confidence.sample_adequacy(summary.n_evidence))

        value = _clamp01(src_weight * coverage * freshness * contradiction * adequacy)

        caps: list[Cap] = []
        floor = self._confidence.corroboration_threshold()
        # The floor is checked against the **attributed** lineages: only
        # evidence that can be traced to the claim's subject can corroborate a
        # claim about them. An unattributed row is still evidence and still
        # counts toward coverage, but it cannot stand in for a second source
        # about the person.
        if summary.n_attributed_lineages < max(MIN_CORROBORATION, floor):
            # INV-1's floor, applied as a ceiling on the *value* so a thin claim
            # cannot read as strong however well it is otherwise supported.
            caps.append(
                Cap(
                    "insufficient_corroboration",
                    f"Only {summary.n_attributed_lineages} attributed source"
                    f"{'' if summary.n_attributed_lineages == 1 else 's'}; "
                    f"at least {max(MIN_CORROBORATION, floor)} are required.",
                    ceiling=0.25,
                )
            )
            value = min(value, 0.25)

        terms: dict[str, float] = dict(
            zip(
                CONFIDENCE_TERMS,
                (src_weight, coverage, freshness, contradiction, adequacy),
                strict=True,
            )
        )
        terms["age_days"] = age
        terms.update(evidence_summary_terms(summary))

        return Scored(
            claim_id=claim.claim_id,
            evidence_id=None,
            value=_clamp01(value),
            terms=terms,
            evidence=summary,
            caps_applied=tuple(caps),
        )

    # -- retrieval ---------------------------------------------------------

    def rank(self, candidates: Sequence[RetrievalCandidate], *, now: datetime) -> list[Scored]:
        """Rank candidates the caller has already scored for similarity.

        Core embeds nothing and generates no candidates; it does the
        deterministic ranking and shows its work.
        """
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        weights = self._retrieval.retrieval_weights()
        total_weight = sum(weights.get(name, 0.0) for name in RETRIEVAL_TERMS)

        scored: list[Scored] = []
        for candidate in candidates:
            item = self._require(candidate.evidence_id)
            recency = self._recency_of(item, now=now)
            source = self._retrieval.source_weight(item.source_type)
            terms = {
                "vector": _clamp01(candidate.similarity),
                "recency": recency,
                "source": _clamp01(source),
                "confidence": _clamp01(candidate.confidence),
            }
            if total_weight > 0:
                value = (
                    sum(weights.get(name, 0.0) * terms[name] for name in RETRIEVAL_TERMS)
                    / total_weight
                )
            else:
                value = 0.0
            scored.append(
                Scored(
                    claim_id=None,
                    evidence_id=candidate.evidence_id,
                    value=_clamp01(value),
                    terms=terms,
                    evidence=summarise_evidence([item]),
                )
            )
        scored.sort(key=lambda item: (-item.value, item.evidence_id or ""))
        return scored

    # -- decisions ---------------------------------------------------------

    def decide(
        self,
        claim: Claim,
        citations: Sequence[Citation],
        *,
        verdict: Verdict | None,
        now: datetime,
    ) -> Decision:
        """Decide a claim's state. Every input combination is specified.

        The table is PLAN.md §5 PM-003.1, and there is exactly one
        implementation of it: :func:`trace_core.attribution.decide`. This
        method only carries the scorer's evidence set, policy and quarantine
        detector to that function, so there cannot be a second answer to "may
        this be promoted", and the decision stays callable with no
        application present — the shape M4 requires.
        """
        return _decide(
            claim,
            citations,
            evidence=self._evidence,
            verdict=verdict,
            now=now,
            policy=self._confidence,
            quarantine_detector=self._quarantine_detector,
        )

    # -- internals ---------------------------------------------------------

    def _age_of(self, cited: Sequence[Evidence], *, now: datetime) -> float:
        """Days from ``now`` to the newest dated cited evidence; infinity if none."""
        dated = [item.occurred_at for item in cited if item.occurred_at is not None]
        if not dated:
            return float("inf")
        return age_days(now, max(dated))

    def _recency_of(self, item: Evidence, *, now: datetime) -> float:
        """Undated evidence scores 0.0, not 1.0 or 0.5.

        The only honest recency for something undated is none. Scoring it
        neutral would let a screenshot with no visible timestamp outrank one
        that has one — which is precisely the evidence a user cannot check.
        """
        if item.occurred_at is None:
            return 0.0
        return decay(age_days(now, item.occurred_at), self._confidence.half_life_days("behavioral"))


def score_reconstructs(scored: Scored, *, tolerance: float = 1e-9) -> bool:
    """Whether ``value`` reconstructs from ``terms`` and the recorded caps.

    This is the property test PLAN.md §8 requires in M2, as a function so it
    can be asserted from anywhere rather than only from the test file.
    """
    product = 1.0
    for name in CONFIDENCE_TERMS:
        product *= scored.terms.get(name, 0.0)
    value = _clamp01(product)
    for cap in scored.caps_applied:
        if cap.ceiling is not None:
            value = min(value, cap.ceiling)
    return math.isclose(value, scored.value, rel_tol=tolerance, abs_tol=tolerance)
