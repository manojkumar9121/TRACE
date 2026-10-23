"""The evidence packet: what one turn is allowed to reason from.

Three properties the rest of the system depends on, all enforced here rather
than by callers:

1. **Freshness travels with the packet.** A packet that could be built without
   one would let a caller assemble a prompt with evidence and no idea how old
   it is — which is exactly the answer the freshness gate exists to prevent.
2. **Only packet ids may be cited.** ``EvidencePacket.citation_ids`` is the
   allow-list, and it is the same set :mod:`trace_core.reference` validates
   against. A citation to an id outside the packet is a validator failure, not
   a rendering bug.
3. **The packet is bounded.** The budget is real, not advice. Overflowing it
   drops the *lowest-scoring* passages and records that in ``truncated`` —
   silently truncating would let the model answer from a fraction of the
   evidence it was shown, without recording that it happened.

The budget is counted in characters rather than tokens because this module has
no tokenizer and does not want one: ``chars_per_token`` is an approximation,
and an approximate budget is only a problem if it is presented as exact. What
it does guarantee is that a packet cannot grow without bound as a person's
history does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from trace_core.temporal import FreshnessState, TemporalScope

__all__ = [
    "EvidencePacket",
    "PacketCandidate",
    "PacketPassage",
    "ProvenanceRef",
    "build_packet",
    "empty_packet",
]


@dataclass(frozen=True, slots=True)
class ProvenanceRef:
    """Region + image reference for one passage.

    Structurally similar to an ingestion provenance record minus the message id,
    because a packet passage may also be a *claim* — and a claim has no region
    of its own. Keeping this local is what lets a claim passage carry ``None``
    fields without pretending to be a message provenance record.
    """

    upload_id: str | None = None
    region_id: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    image_url: str | None = None


@dataclass(frozen=True, slots=True)
class PacketPassage:
    """One citable passage with the provenance a chip needs to render.

    ``bbox`` and ``image_url`` are carried here rather than resolved at render
    time because a citation chip that opens to nothing is worse than no chip:
    the evidence thread exists so provenance can be *checked*, and a chip whose
    region is missing has quietly become decorative.
    """

    citation_id: str
    reference_type: str
    text: str
    score: float
    sent_at: datetime | None = None
    bbox: tuple[int, int, int, int] | None = None
    image_url: str | None = None
    region_id: str | None = None
    upload_id: str | None = None
    epistemic_type: str = "fact"
    #: The independence unit, carried so a claim can be banded from its cited
    #: passages. ``None`` for a claim passage, which belongs to no single
    #: conversation.
    conversation_id: str | None = None
    #: The ``src_weight`` key for this passage. A message passage is
    #: ``message``; a claim passage carries whatever the stored claim was
    #: recorded with, because the gap between "you told us" and "we inferred
    #: this" is the largest single term in the confidence product.
    source_type: str = "message"
    #: The contradiction term, carried from the stored claim. Always 0 for a
    #: message: a message is evidence, and only a claim about behaviour can be
    #: argued against.
    contradiction_count: int = 0

    def as_ref(self) -> dict[str, Any]:
        """The evidence-reference shape, for a client's citation chips.

        Carries the passage text and its date alongside the id, because an id
        alone leaves the client with nothing to put on the chip except the id.
        """
        return {
            "type": self.reference_type,
            "id": self.citation_id,
            "label": self.text[:200] if self.text else None,
            "sent_at": self.sent_at.isoformat() if self.sent_at is not None else None,
        }


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    """What one turn is allowed to reason from."""

    passages: tuple[PacketPassage, ...] = ()
    #: Required, not optional. See the module docstring.
    freshness_state: FreshnessState = FreshnessState.ABSENT
    #: The retrieval window, or ``None`` while the gate is fresh.
    window_start: datetime | None = None
    temporal_scope: TemporalScope = TemporalScope.UNKNOWN
    #: Passages dropped to stay inside the budget.
    truncated: bool = False
    #: The scores of the passages that survived, for debug output.
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def citation_ids(self) -> frozenset[str]:
        """The allow-list. The only ids that may be cited back."""
        return frozenset(passage.citation_id for passage in self.passages)

    @property
    def is_empty(self) -> bool:
        return not self.passages

    def citations(self) -> list[dict[str, Any]]:
        return [passage.as_ref() for passage in self.passages]

    def by_id(self) -> dict[str, PacketPassage]:
        return {passage.citation_id: passage for passage in self.passages}


@dataclass(frozen=True, slots=True)
class PacketCandidate:
    """One scored thing that may enter the packet.

    Separate from :class:`~trace_core.types.Scored` because a score does not
    know what it scored: the score is arithmetic over evidence, and the text
    lives on the evidence. The packet budget is a budget on *text*, so the
    candidate has to carry it.
    """

    citation_id: str
    text: str
    score: float
    reference_type: str = "evidence"
    sent_at: datetime | None = None
    epistemic_type: str = "observation"
    conversation_id: str | None = None
    source_type: str = "message"
    contradiction_count: int = 0


def build_packet(
    candidates: Sequence[PacketCandidate],
    *,
    freshness_state: FreshnessState,
    temporal_scope: TemporalScope,
    window_start: datetime | None,
    max_tokens: int,
    provenance: Mapping[str, ProvenanceRef] | None = None,
    chars_per_token: int = 4,
) -> EvidencePacket:
    """Assemble a packet from scored candidates, highest score first.

    A candidate carries its own text, because the budget is a budget on *text*
    and a score that does not know what it scored cannot be costed.

    ``provenance`` is keyed by citation id and supplied by the caller because
    resolving it needs a session. Missing provenance does **not** drop a
    passage — dropping it would silently shrink the evidence set — but the
    passage keeps ``bbox=None``, and a render will show it without a region
    rather than inventing one.
    """
    ordered = sorted(candidates, key=lambda item: (-item.score, item.citation_id))
    refs = provenance or {}

    kept: list[PacketPassage] = []
    spent = 0
    budget = max(1, max_tokens) * chars_per_token
    for candidate in ordered:
        cost = len(candidate.text) + len(candidate.citation_id)
        # The best-scoring passage is always kept, even if it alone exceeds
        # the budget: an empty packet is a less useful answer than one
        # oversized passage, and the budget exists to bound a prompt, not to
        # silence it.
        if kept and spent + cost > budget:
            break
        ref = refs.get(candidate.citation_id)
        kept.append(
            PacketPassage(
                citation_id=candidate.citation_id,
                reference_type=candidate.reference_type,
                text=candidate.text,
                score=candidate.score,
                sent_at=candidate.sent_at,
                bbox=ref.bbox if ref is not None else None,
                image_url=ref.image_url if ref is not None else None,
                region_id=ref.region_id if ref is not None else None,
                upload_id=ref.upload_id if ref is not None else None,
                epistemic_type=candidate.epistemic_type,
                conversation_id=candidate.conversation_id,
                source_type=candidate.source_type,
                contradiction_count=candidate.contradiction_count,
            )
        )
        spent += cost

    truncated = len(kept) < len(ordered)
    return EvidencePacket(
        passages=tuple(kept),
        freshness_state=freshness_state,
        window_start=window_start,
        temporal_scope=temporal_scope,
        truncated=truncated,
        scores={passage.citation_id: passage.score for passage in kept},
    )


def empty_packet(
    *,
    freshness_state: FreshnessState,
    temporal_scope: TemporalScope,
    window_start: datetime | None = None,
) -> EvidencePacket:
    """Absence is an answer, not a failure.

    Built here so "no evidence" has exactly one representation. A caller
    inventing its own empty packet could set a ``freshness_state`` that
    disagrees with the gate, and the client would then have no state to render.
    """
    return EvidencePacket(
        passages=(),
        freshness_state=freshness_state,
        window_start=window_start,
        temporal_scope=temporal_scope,
    )
