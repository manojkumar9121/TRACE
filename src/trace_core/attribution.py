"""The decision layer: may this claim be promoted, and at what scope?

Scoring answers *how strong* a claim is (:mod:`trace_core.scoring`); this
module answers the different question the write path actually gates on: *may
it move at all*, and with it a :class:`~trace_core.types.Decision` carrying
state, a reason a human can read, the **effective scope** the claim may assert
and the caps that were applied. The outcome table is PLAN.md §5 PM-003.1, and
it is complete: every input combination yields exactly one state.

M4's exit condition is that this is callable with **no application present**:
the evidence set, the verdict, the clock and the policy are all parameters of
:func:`decide`, and a hypothetical integrator supplies everything. Nothing
reads a wall clock, touches a database, or asks a model a question. The
quarantine detector is a caller-supplied callable; without one the
``quarantined`` state is unreachable (PM-003.3).

Three properties of the table are the load-bearing ones:

1. **Everything model-generated stays a candidate until the evidence says
   otherwise.** A missing verdict is not acceptance (INV-3), a verdict value
   the contract does not define fails closed to ``candidate`` with the cap
   ``invalid_verdict``, and a ``rejected`` one drops the claim with its scope
   set to ``unknown`` — never the scope the caller claimed.
2. **Promotion requires corroboration across attributed lineages** (INV-1):
   the hard floor is 2 in core, a policy may only raise it, and an
   unattributed row may establish recency but can never act as the second
   source about a named person (PM-003.7).
3. **Human-asserted evidence bypasses corroboration and the verdict both**
   (PM-003.2): a person who said something is a different kind of evidence
   from an inference about them.

Scorer delegates here in full, so there is exactly one implementation of "may
this be promoted"; the other ported piece is the pre-port specification's
§7.3 detector, shipped
as a ready-made quarantine detector and defence in depth only — the claim it
watches is stuck behind the corroboration gate either way.

What this module deliberately does not do:

- **Bands.** The band vocabulary is policy, not core: the decision reports
  state and scope, never a number about a person.
- **Folding contradictions.** A contradicting citation lowers the score in
  :mod:`trace_core.scoring`; nothing here averages two claims into a middle
  neither conversation supports.
- **Repairing citations.** An id outside the evidence set raises — INV-4 —
  it is never dropped, and a neutral stance is a broken citation, not weak
  support.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from trace_core.errors import CitationError
from trace_core.temporal import TemporalScope, age_days, decay
from trace_core.types import (
    Cap,
    Citation,
    Claim,
    Decision,
    Evidence,
    EvidenceSummary,
    Link,
    Verdict,
)

__all__ = [
    "AttributionPolicy",
    "HUMAN_ASSERTED_SOURCES",
    "INSTRUCTION_SHAPED",
    "MIN_CORROBORATION",
    "attributed_lineages",
    "claim_scope_is_current",
    "decide",
    "effective_scope",
    "instruction_detector",
    "is_instruction_shaped",
    "reject_mixed_timestamps",
    "summarise_evidence",
]

#: Sources that are a person saying something, rather than an inference about
#: them (PM-003.2). These bypass corroboration and the verdict, because an
#: explicit human action is a different kind of evidence; a message inheriting
#: this set is the port of the pre-port specification's §7.4 rule 2.
HUMAN_ASSERTED_SOURCES: frozenset[str] = frozenset({"user_stated", "manual_edit", "feedback"})

#: The hard floor of attributed lineages for acceptance (INV-1). It lives in
#: core, not policy: a policy may raise it, never lower it, and the clamp in
#: :func:`decide` is what enforces that for a policy that forgot.
MIN_CORROBORATION = 2

#: Instruction-shaped claim patterns, verbatim from the pre-port
#: specification's §7.3 list.
#: Detection only — :func:`instruction_detector` is defence in depth, not the
#: defence. That is why it is matched against the *whole* claim rather than
#: any substring: the benign "forget what I said earlier, I was drunk" is not
#: an attack, and a filter that eats it would work equally on real messages.
INSTRUCTION_SHAPED: tuple[str, ...] = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "you are now",
    "disregard the above",
    "new instructions",
)


@runtime_checkable
class AttributionPolicy(Protocol):
    """The two policy knobs the decision reads (PM-003, INV-1, INV-2).

    Deliberately narrower than :class:`~trace_core.policies.ConfidencePolicy`:
    the decision is not the scoring formula, and it must not grow a dependency
    on the weights the score uses.
    :class:`~trace_core.policies.TableConfidencePolicy` satisfies this protocol
    structurally, and so does any hand-rolled implementation.
    """

    def corroboration_threshold(self) -> int:
        """How many attributed lineages an accepted claim needs (INV-1).

        Must be ``>= 2``: that is the core's hard floor, and :func:`decide`
        clamps to it rather than asking for it politely. A policy may raise
        the number; it shall never lower it.
        """

    def half_life_days(self, kind: str) -> float:
        """Days for a claim of this ``kind`` to lose half its weight.

        Read only for INV-2 — deciding whether the support is recent enough
        to assert as present. It never enters a score from here.
        """


def is_instruction_shaped(text: str) -> bool:
    """Whether a claim text reads as an instruction rather than a fact.

    Case-insensitive whole-claim substring match, deliberately narrow: this
    quarantines a small, visible set, it does not decide what is true.
    """
    lowered = text.casefold()
    return any(pattern in lowered for pattern in INSTRUCTION_SHAPED)


def instruction_detector(claim: Claim, links: Sequence[Link]) -> bool:
    """A ready-made quarantine detector over :func:`is_instruction_shaped`.

    The ``links`` are ignored: §7.3's rule is about the *shape of the claim*,
    not about the citations under it. The parameter exists so this matches the
    :func:`decide` detector contract exactly, and so an integrator composing
    detectors sees the citations it would otherwise miss.
    """
    return is_instruction_shaped(claim.text)


def attributed_lineages(items: Sequence[Evidence]) -> int:
    """INV-1's count: distinct lineages among these rows that carry a speaker.

    An unattributed row still counts toward ``n_evidence`` and may establish
    recency (CDI-009), but it shall never satisfy the corroboration floor for
    a claim about a named person — otherwise one attributed row plus one
    anonymous row would be corroboration for what *they* do.
    """
    return len({item.lineage_id for item in items if item.speaker is not None})


def summarise_evidence(
    cited: Sequence[Evidence], contradicting: Sequence[Evidence] = ()
) -> EvidenceSummary:
    """DM-002 built once: the auditable inputs to both the score and the decision.

    Two lineage counts are carried because they answer different questions:
    ``n_lineages`` counts independent sources; ``n_attributed_lineages`` counts
    those that can be traced to a speaker and therefore to the claim's
    subject. An audit found the two conflated, and a claim citing two
    unattributed rows was accepted while the floor check read a different
    number — the value and the decision disagreed about the same evidence.
    Computing the count through :func:`attributed_lineages` is what keeps them
    provably the same number everywhere both layers consume it.
    """
    dated = [item.occurred_at for item in cited if item.occurred_at is not None]
    return EvidenceSummary(
        n_evidence=len(cited),
        n_segments=len({item.segment_id for item in cited}),
        n_lineages=len({item.lineage_id for item in cited}),
        n_attributed_lineages=attributed_lineages(cited),
        n_undated=sum(1 for item in cited if item.occurred_at is None),
        n_unattributed=sum(1 for item in cited if item.speaker is None),
        n_contradicting=len(contradicting),
        newest_at=max(dated) if dated else None,
        oldest_at=min(dated) if dated else None,
    )


def reject_mixed_timestamps(items: Sequence[Evidence]) -> None:
    """Reject a *mixture* of naive and aware timestamps before any comparison.

    A naive timestamp mixed with an aware one makes a later ``max()`` raise a
    TypeError from inside a comparison — an opaque failure that names neither
    the problem nor the evidence. It is checked here, where the message can
    say what is wrong.

    Naive timestamps on their own are not an error: a naive timestamp is
    interpreted as UTC, which is what a chat export means by it. Only the
    mixture is ambiguous.
    """
    aware = [
        item
        for item in items
        if item.occurred_at is not None and item.occurred_at.tzinfo is not None
    ]
    naive = [
        item for item in items if item.occurred_at is not None and item.occurred_at.tzinfo is None
    ]
    if aware and naive:
        raise ValueError(
            f"evidence mixes {len(aware)} timezone-aware timestamps with "
            f"{len(naive)} naive ones; every timestamp must be the same kind"
        )


def claim_scope_is_current(claim: Claim, summary: EvidenceSummary) -> bool:
    """Whether a claim's support is current enough to assert as present.

    A dated, attributed, non-stale observation. Undated or unattributed
    evidence cannot make a claim current, because "current" is a claim about
    *when*, and neither null answers that question.
    """
    if summary.newest_at is None:
        return False
    return not (summary.n_evidence > 0 and summary.n_unattributed >= summary.n_evidence)


def effective_scope(
    claim: Claim,
    summary: EvidenceSummary,
    *,
    now: datetime,
    half_life_days: float,
) -> TemporalScope:
    """INV-2: ``present`` scope requires dated, attributed, non-stale support.

    A claim asking for ``present`` on undated, unattributed or older evidence
    gets ``historical``. ``Claim.scope`` is never silently honoured, and the
    returned value is structured data (CDI-007.1): a caller may never need to
    parse :attr:`Decision.reason` to learn what scope was permitted.
    """
    if claim.scope is not TemporalScope.PRESENT:
        return claim.scope
    if not claim_scope_is_current(claim, summary):
        return TemporalScope.HISTORICAL
    age = float("inf") if summary.newest_at is None else age_days(now, summary.newest_at)
    if decay(age, half_life_days) < 0.5:
        # Older than one half-life: not current enough to assert as present.
        return TemporalScope.HISTORICAL
    return TemporalScope.PRESENT


def decide(
    claim: Claim,
    citations: Sequence[Citation],
    *,
    evidence: Mapping[str, Evidence],
    verdict: Verdict | None,
    now: datetime,
    policy: AttributionPolicy,
    quarantine_detector: Callable[[Claim, Sequence[Link]], bool] | None = None,
) -> Decision:
    """Decide a claim's state. Every input combination is specified.

    The arguments are everything an integrator has and nothing that needs an
    application: the evidence set keyed by id (DM-001.1), the citations as the
    caller states them, the verdict produced elsewhere or ``None`` when the
    gate was never reached, the injected ``now``, and the integrator's own
    policy.

    ``quarantine_detector`` is a callable ``(claim, links) -> bool``; ``links``
    are the *supporting* citations as :class:`~trace_core.types.Link`. When it
    is supplied it runs before every other branch — an injected claim with a
    mountain of fake support is still an injection — and it is the only thing
    that can reach ``quarantined`` (PM-003.3). Absent, the row falls through
    to the corroboration and verdict rules, which is deliberate: the detector
    is an optional, visible layer, not a secret fallback.

    The order is the security-relevant part (the pre-port specification
    §7.4's one door, carried
    through PM-003):

    1. **Citations are validated before anything is decided.** An id outside
       the evidence set raises (INV-4), a neutral stance raises (PM-003.6),
       and a citation spoken by someone other than ``Claim.subject`` raises
       (PM-003.5) — checked on *every* citation, contradicting ones included.
    2. **Instruction-shaped → quarantined**, scope ``unknown``.
    3. **Human-asserted → accepted** (PM-003.2), skipping corroboration and
       the verdict entirely.
    4. **An unreachable check never promotes** (INV-3): ``verdict is None``
       stays ``candidate``; an unrecognised verdict value fails closed to
       ``candidate`` with the cap ``invalid_verdict``.
    5. **Corroboration across attributed lineages** (PM-003.7): fewer than
       the floor and even a supported claim stays ``candidate``, historical
       at best.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    cited: list[Evidence] = []
    contradicting: list[Evidence] = []
    for citation in citations:
        # INV-4 first: a citation naming an id outside the evidence set is a
        # failure, not a dropped citation.
        try:
            item = evidence[citation.evidence_id]
        except KeyError:
            raise CitationError(f"no evidence with id {citation.evidence_id!r}") from None
        if citation.stance not in ("supports", "contradicts"):
            # A Citation is never neutral (CDI-003): citing something you
            # have no position on is not a citation. Accepting a neutral one
            # and counting it as support would let a caller smuggle an
            # uncorroborated claim past the floor.
            raise CitationError(
                f"citation {item.evidence_id!r} has stance "
                f"{citation.stance!r}; a citation is supports or contradicts"
            )
        # The speaker check applies to EVERY citation, contradicting ones
        # included. A claim about someone that cites other people's words
        # is not evidence about them, whichever way it is pointed.
        if claim.subject is not None and item.speaker is not None and item.speaker != claim.subject:
            raise CitationError(
                f"evidence {item.evidence_id!r} is spoken by {item.speaker!r}, "
                f"but the claim is about {claim.subject!r}"
            )
        if citation.stance == "contradicts":
            contradicting.append(item)
        else:
            cited.append(item)

    reject_mixed_timestamps(cited + contradicting)
    summary = summarise_evidence(cited, contradicting)
    floor = max(MIN_CORROBORATION, policy.corroboration_threshold())
    # INV-1 counts *attributed* lineages. Unattributed evidence may establish
    # recency and may be cited for an unattributed claim, but it shall never
    # satisfy the corroboration floor (CDI-009) — otherwise one attributed
    # row plus one anonymous row would corroborate a claim about a named
    # person. ``attributed`` and ``summary.n_attributed_lineages`` are the
    # same number on purpose: the check and the reason string must never be
    # able to disagree about it.
    attributed = summary.n_attributed_lineages
    human_asserted = any(item.source_type in HUMAN_ASSERTED_SOURCES for item in cited)

    if quarantine_detector is not None:
        links = [Link(evidence_id=item.evidence_id, stance="supports") for item in cited]
        if quarantine_detector(claim, links):
            return Decision(
                state="quarantined",
                reason=(
                    "This was quarantined rather than scored, so it can be "
                    "reviewed rather than silently used."
                ),
                effective_scope=TemporalScope.UNKNOWN,
                caps_applied=(Cap("quarantined", "A quarantine detector flagged this claim."),),
            )

    if human_asserted:
        return Decision(
            state="accepted",
            reason="A person stated this directly, so it counts on its own.",
            effective_scope=effective_scope(
                claim, summary, now=now, half_life_days=policy.half_life_days(claim.kind)
            ),
        )

    if verdict is None:
        return Decision(
            state="candidate",
            reason=(
                "Nothing checked this claim against the evidence it came from, "
                "so it stays a possibility rather than an observation."
            ),
            effective_scope=effective_scope(
                claim, summary, now=now, half_life_days=policy.half_life_days(claim.kind)
            ),
        )

    if verdict.verdict not in ("supported", "insufficient", "unsupported"):
        # A verdict the contract does not define is not a verdict. Failing
        # closed is the only safe reading: a gate that cannot be reached must
        # not authorise anything (INV-3), and a mistyped verdict must not
        # read as support.
        return Decision(
            state="candidate",
            reason=(
                "The check against the evidence did not return a usable "
                "answer, so this stays a possibility rather than an observation."
            ),
            effective_scope=effective_scope(
                claim, summary, now=now, half_life_days=policy.half_life_days(claim.kind)
            ),
            caps_applied=(
                Cap("invalid_verdict", "The verdict was not one of the defined outcomes."),
            ),
        )

    if verdict.verdict == "unsupported":
        # A dropped claim asserts no time at all: its scope is unknown, never
        # the one the caller claimed for it.
        return Decision(
            state="rejected",
            reason="The evidence does not support this claim, so it was dropped.",
            effective_scope=TemporalScope.UNKNOWN,
        )

    if verdict.verdict == "insufficient":
        return Decision(
            state="candidate",
            reason=(
                "The evidence touches on this but does not settle it. "
                "More examples would tell us whether it holds."
            ),
            # "historical at best": the gate declined to conclude, so a
            # present-tense reading is not on offer either way.
            effective_scope=TemporalScope.HISTORICAL,
            caps_applied=(Cap("insufficient_verdict", "The gate declined to conclude."),),
        )

    # verdict == "supported"
    if attributed < floor:
        return Decision(
            state="candidate",
            reason=(
                f"Only {attributed} attributed source"
                f"{'' if attributed == 1 else 's'}. "
                f"At least {floor} are needed before this counts as a pattern."
            ),
            effective_scope=TemporalScope.HISTORICAL,
            caps_applied=(
                Cap(
                    "insufficient_corroboration",
                    f"{attributed} attributed sources, {floor} required.",
                ),
            ),
        )

    return Decision(
        state="accepted",
        reason=(
            f"Specific to this person, and seen in "
            f"{attributed} independent source"
            f"{'' if attributed == 1 else 's'}."
        ),
        effective_scope=effective_scope(
            claim, summary, now=now, half_life_days=policy.half_life_days(claim.kind)
        ),
    )
