"""The public contract — PLAN.md §2, complete and frozen.

Every dataclass here is frozen and slotted. A slot is not a micro-optimisation:
it is the reason a field cannot be added by accident, and the reason
``dataclasses.replace`` on a ``slots=True`` dataclass cannot silently widen the
record that a stored score was computed from.

The one deviation from the plan's text is documented at ``Cap.ceiling`` and
``Evidence.metadata``; both are noted in ``PLAN.md`` §13.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

from trace_core.temporal import TemporalScope

__all__ = [
    "Cap",
    "Citation",
    "Claim",
    "Decision",
    "Evidence",
    "EvidenceSummary",
    "Link",
    "RetrievalCandidate",
    "Scored",
    "Verdict",
]


def _freeze_metadata(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Wrap a mapping immutably. Nested values remain the caller's.

    PLAN.md §2.6: "Immutability is shallow — a `MappingProxyType` wraps the
    mapping; nested values are the caller's." Copying the mapping means a later
    mutation of the caller's dict cannot change a score that was already
    computed from it.
    """
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True, slots=True)
class Evidence:
    """One observed thing. Claim-independent by construction (PLAN.md §2.1).

    ``stance`` deliberately does not live here: the same message supports
    "replies briefly" and contradicts "writes long replies" at the same time,
    and a field on the evidence cannot express both.
    """

    evidence_id: str
    #: Corroboration unit: a thread, a document, a session.
    segment_id: str
    #: Independence group — copies and derivatives share one. Asserted by the
    #: adapter, never verified by core (PLAN.md §6.1).
    lineage_id: str
    #: Provenance: bbox, URL, id. Free-form.
    source_id: str
    text: str
    #: What scoring weights key on: "user_stated", "message", "screenshot", …
    source_type: str
    #: Provenance only: "screenshot", "mbox", "transcript", … Never an input to
    #: scoring.
    modality: str
    occurred_at: datetime | None
    speaker: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Link:
    """The claim↔evidence relationship (PLAN.md §2.2).

    ``stance`` may be neutral here; a :class:`Citation` may not, because citing
    something you have no position on is not a citation.
    """

    evidence_id: str
    stance: Literal["supports", "contradicts", "neutral"]


@dataclass(frozen=True, slots=True)
class Claim:
    """One assertion about a subject, in the integrator's taxonomy."""

    claim_id: str
    text: str
    #: The speaker this claim is about. ``None`` means unattributed.
    subject: str | None
    #: Integrator taxonomy: "behavioral", "tone", "distributional", …
    kind: str
    scope: TemporalScope = TemporalScope.UNKNOWN


@dataclass(frozen=True, slots=True)
class Citation:
    """A claim's link to evidence, as a caller states it (PLAN.md §2.2)."""

    evidence_id: str
    stance: Literal["supports", "contradicts"]


@dataclass(frozen=True, slots=True)
class Verdict:
    """Supplied by the caller; core never invents one (PLAN.md §2.2)."""

    claim_id: str
    verdict: Literal["supported", "insufficient", "unsupported"]
    rationale: str | None = None


@dataclass(frozen=True, slots=True)
class Cap:
    """A ceiling the core applied, and why.

    ``ceiling`` carries the numeric bound that was applied so a score
    reconstructs from its terms and its caps without re-deriving policy. It is
    ``None`` for caps that are not numeric bounds (a quarantine, say).

    PLAN.md §2.3 lists ``code`` and ``reason`` only; the third field is
    additive and is recorded in PLAN.md §13.
    """

    code: str
    reason: str
    ceiling: float | None = None


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    """The auditable inputs to a score (PLAN.md §2.3)."""

    n_evidence: int
    n_segments: int
    n_lineages: int
    #: How many of those lineages can be traced to the claim's subject.
    #: Evidence with no speaker is still a source, but it cannot corroborate
    #: a claim *about a person*.
    n_attributed_lineages: int
    n_undated: int
    n_unattributed: int
    n_contradicting: int
    newest_at: datetime | None
    oldest_at: datetime | None


@dataclass(frozen=True, slots=True)
class Scored:
    """A score with every contributing term named (PLAN.md §2.3)."""

    claim_id: str | None
    evidence_id: str | None
    value: float
    #: Every contribution, named. The four keys in
    #: :data:`trace_core.scoring.CONFIDENCE_TERMS` multiply; anything else in
    #: this mapping is informational.
    terms: Mapping[str, float]
    evidence: EvidenceSummary
    caps_applied: tuple[Cap, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", _freeze_metadata(self.terms))


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of a decision, in structured form (PLAN.md §2.3)."""

    state: Literal["accepted", "candidate", "rejected", "quarantined"]
    #: Always renderable to a human.
    reason: str
    #: The scope actually permitted, post-INV-2. Structured data, not something
    #: a caller parses out of ``reason``.
    effective_scope: TemporalScope
    caps_applied: tuple[Cap, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """A retrieval candidate, already scored for similarity by the caller."""

    evidence_id: str
    #: Precomputed by the caller; core embeds nothing.
    similarity: float
    #: Precomputed; may be a prior score, may be 1.0.
    confidence: float
