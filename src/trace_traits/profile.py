"""Turning measurements into claims, and claims into a trait profile.

This is the join between :mod:`trace_traits.features` (what the messages show)
and :mod:`trace_core` (how much that is worth). Every trait reading becomes a
:class:`~trace_core.Claim` with evidence links, so every number a user sees
has a path back to the messages that produced it.

The claims produced here are ``kind="distributional"`` — a statement about a
distribution over a corpus, not about one moment. They are also
``source_type="message"``, because that is what they are: derived from stored
messages. A derived claim is not a human assertion and must not borrow the
weight of one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from trace_core import Citation, Claim, Decision, Link, Scored, Scorer, Verdict
from trace_core.temporal import TemporalScope
from trace_traits.corpus import Corpus, evidence_from_corpus
from trace_traits.features import FeatureReading, FeatureVector, extract_features

__all__ = [
    "TraitProfile",
    "TraitReading",
    "Verifier",
    "build_profile",
    "claims_for_features",
    "measurement_verifier",
]

#: The caller's entailment check: does the cited evidence support the claim,
#: about the stated subject? Absent, the claim stays a candidate — a gate that
#: cannot be reached fails closed.
Verifier = Callable[[Claim, Corpus], Verdict | None]


@dataclass(frozen=True, slots=True)
class TraitReading:
    """One trait, with the confidence machinery's verdict on it."""

    dimension: str
    band: str
    detail: str
    n_evidence: int
    n_conversations: int
    score: float
    decision: Decision
    #: The terms behind ``score``, so a reader can see why it is what it is.
    terms: dict[str, float]
    caps_applied: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.decision.state == "accepted"

    def as_line(self) -> str:
        return (
            f"{self.dimension:<22} {self.band:<22} "
            f"confidence {self.score:.2f}  ({self.n_evidence} messages, "
            f"{self.n_conversations} conversations)"
        )


@dataclass(frozen=True, slots=True)
class TraitProfile:
    """Everything known about how one person communicated."""

    subject: str
    readings: tuple[TraitReading, ...]
    corpus_size: int
    n_conversations: int

    def get(self, dimension: str) -> TraitReading | None:
        for reading in self.readings:
            if reading.dimension.lower() == dimension.lower():
                return reading
        return None

    @property
    def accepted(self) -> tuple[TraitReading, ...]:
        return tuple(r for r in self.readings if r.accepted)

    def as_lines(self) -> tuple[str, ...]:
        return tuple(reading.as_line() for reading in self.readings)


@dataclass(frozen=True, slots=True)
class _ClaimSpec:
    """A claim, its links, and the reading it came from."""

    claim: Claim
    links: tuple[Link, ...]
    reading: FeatureReading


def claims_for_features(
    features: FeatureVector,
    *,
    subject: str,
    prefix: str = "trait",
) -> tuple[_ClaimSpec, ...]:
    """One claim per feature reading, with its evidence links.

    A reading that does not have enough data produces **no claim** rather
    than a claim about nothing: "not enough messages" is an absence, and the
    profile reports it as such.

    The links cite exactly the messages the reading was computed over — not the
    whole corpus. A claim about 14 conversation openers cites those 14
    openers, so its corroboration count means what it says.
    """
    out: list[_ClaimSpec] = []
    for reading in features.readings:
        if reading.band == "insufficient_evidence":
            continue
        slug = reading.dimension.lower().replace(" ", "-")
        claim = Claim(
            claim_id=f"{prefix}:{slug}",
            text=f"{reading.dimension}: {reading.band} ({reading.detail})",
            subject=subject,
            kind="distributional",
            scope=TemporalScope.HISTORICAL,
        )
        links = tuple(Link(evidence_id=eid, stance="supports") for eid in reading.evidence_ids)
        out.append(_ClaimSpec(claim=claim, links=links, reading=reading))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class MeasurementClaim:
    """A claim's parsed content: the dimension, the band, and the detail.

    A measurement claim is a structured statement rendered as text, so the
    verifier parses it rather than pattern-matching a prefix. A prefix check
    would accept a claim whose *detail* had been altered while its band
    stayed the same, which is exactly the tampering this is meant to catch.
    """

    dimension: str
    band: str
    detail: str


def parse_measurement_claim(text: str) -> MeasurementClaim | None:
    """Parse ``"{dimension}: {band} ({detail})"``, or ``None`` if it is not one.

    The split is on the **first** ``": "`` after the dimension and the
    **first** opening parenthesis, and the detail is everything up to the
    final ``")"``.

    Using the *first* parenthesis rather than the last is deliberate: a detail
    containing its own parentheses — "44 distinct words in 77 (38-word
    windows)" — would otherwise be split at the wrong one, and the verifier
    would reject a claim it had just produced. Taking the first ``": "`` and
    the first ``"("`` makes the parse independent of the detail's contents.
    """
    colon = text.find(": ")
    if colon == -1:
        return None
    dimension = text[:colon].strip()
    rest = text[colon + 2 :]

    open_paren = rest.find("(")
    if open_paren == -1:
        return None
    band = rest[:open_paren].strip()
    detail = rest[open_paren + 1 :]
    if not detail.endswith(")"):
        return None
    detail = detail[:-1].strip()

    if not dimension or not band:
        return None
    return MeasurementClaim(dimension=dimension, band=band, detail=detail)


def measurement_verifier(
    claim: Claim, corpus: Corpus, *, readings: FeatureVector | None = None
) -> Verdict:
    """Verify a measurement claim by re-deriving it from the corpus.

    A distributional claim is not a matter of opinion: it says a statistic
    computed over messages has a value, and that statistic can be recomputed.
    This verifier does exactly that — it recomputes the feature over the
    subject's messages and checks the band *and* the detail still hold.

    This is the difference between a trait layer and a claim generator. A model
    asked "is this person terse?" produces a judgement; a measurement
    produces a number that either reproduces or does not.
    """
    features = readings if readings is not None else extract_features(corpus)
    parsed = parse_measurement_claim(claim.text)
    if parsed is None:
        return Verdict(
            claim_id=claim.claim_id,
            verdict="unsupported",
            rationale="The claim is not a measurement this corpus can produce.",
        )
    reading = features.get(parsed.dimension)
    if reading is None:
        return Verdict(
            claim_id=claim.claim_id,
            verdict="unsupported",
            rationale="No such dimension is measurable over this corpus.",
        )
    if reading.band != parsed.band:
        return Verdict(
            claim_id=claim.claim_id,
            verdict="unsupported",
            rationale=(
                f"Recomputing gives {reading.dimension}: {reading.band}, not {parsed.band}."
            ),
        )
    if reading.detail != parsed.detail:
        return Verdict(
            claim_id=claim.claim_id,
            verdict="unsupported",
            rationale=(f"Recomputing gives '{reading.detail}', not '{parsed.detail}'."),
        )
    return Verdict(
        claim_id=claim.claim_id,
        verdict="supported",
        rationale=(
            f"Recomputed from {reading.n_evidence} messages across "
            f"{reading.n_conversations} conversations: {reading.detail}."
        ),
    )


def build_profile(
    corpus: Corpus,
    *,
    scorer: Scorer,
    now: datetime,
    source_type: str = "message",
    verifier: Verifier | None = None,
) -> TraitProfile:
    """Measure a corpus, score every reading, and decide each one.

    ``verifier`` defaults to :func:`measurement_verifier`, because a
    distributional claim is checkable by recomputation. Pass ``None``
    explicitly to model a pipeline with no entailment gate — every claim then
    stays a candidate, which is the fail-closed behaviour.
    """
    features = extract_features(corpus)
    evidence = evidence_from_corpus(corpus, source_type=source_type)
    working = scorer.with_evidence(evidence)

    check = verifier or (
        lambda claim, target: measurement_verifier(claim, target, readings=features)
    )

    readings: list[TraitReading] = []
    for spec in claims_for_features(features, subject=corpus.person.subject):
        scored: Scored = working.score(spec.claim, spec.links, now=now)
        verdict = check(spec.claim, corpus)
        decision = working.decide(
            spec.claim,
            [Citation(evidence_id=link.evidence_id, stance="supports") for link in spec.links],
            verdict=verdict,
            now=now,
        )
        readings.append(
            TraitReading(
                dimension=spec.reading.dimension,
                band=spec.reading.band,
                detail=spec.reading.detail,
                n_evidence=scored.evidence.n_evidence,
                # The number of *conversations* the reading rests on, not the
                # number of lineages. They are different quantities: three
                # uploads of one conversation are three segments and one
                # lineage, and reporting one under the other's name is how a
                # reader ends up believing a claim was corroborated across
                # conversations it was never seen in (DM-005.1).
                n_conversations=spec.reading.n_conversations,
                score=scored.value,
                decision=decision,
                terms=dict(scored.terms),
                caps_applied=tuple(cap.code for cap in scored.caps_applied),
            )
        )

    return TraitProfile(
        subject=corpus.person.subject,
        readings=tuple(readings),
        corpus_size=len(corpus),
        n_conversations=len(corpus.conversations),
    )


def citations_for(links: Sequence[Link]) -> list[Citation]:
    """Links as citations, for a caller driving :meth:`Scorer.decide` directly."""
    return [Citation(evidence_id=link.evidence_id, stance="supports") for link in links]
