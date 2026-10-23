"""The freshness gate: what a corpus may be used to assert, and when.

The gate implements all four mechanisms rather than the one that is easy:

1. :meth:`FreshnessGate.constrain_schema` runs *before* the prompt and narrows
   ``temporal_scope`` in the response schema, so the decoder cannot emit a
   present-tense claim for a stale persona.
2. ``window_start`` bounds retrieval.
3. :class:`TemporalValidator` runs on the output and *downgrades* rather than
   erroring, because the answer is still useful — just not as a claim about
   now.
4. ``freshness_state`` is a required field of every packet, so no consumer has
   a code path that renders a current-behaviour card without it.

Two things this module deliberately refuses to do:

- **Per-corpus freshness.** Freshness is measured from the newest accepted
  evidence for the whole corpus. Per-conversation freshness would make a
  persona's confidence a function of which thread a question happened to
  surface in, which is the instability the band model exists to prevent.
- **Borrowing ``stale`` for capability loss.** Freshness measures evidence
  age; a paused analysis pipeline is not evidence age, and conflating them
  would tell a user their evidence is old when it is fine.

**The clock is injected.** ``now`` is a parameter, never read from the wall
clock — the same rule the rest of the package obeys.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from trace_core.temporal import FreshnessState, TemporalScope, age_days

__all__ = [
    "FreshnessGate",
    "FreshnessVerdict",
    "TemporalValidator",
]

#: There is deliberately **no depth cap** on the schema walk.
#:
#: A depth cap is a hang guard that behaves like a scope guard: an audit found
#: that stopping the walk at the cap left the ``temporal_scope`` below it with
#: its original enum, so a deeply-nested claim could still be emitted with
#: ``present`` scope by a gate that believed it had constrained it. A silently
#: bypassed gate is worse than a truncated schema.
#:
#: The hang guard is the **identity check** in :func:`_constrain_node`: a
#: self-referential schema is detected by object identity and the walk stops
#: there. A schema that references itself cannot be rendered, so it cannot
#: leak a claim either.


@dataclass(frozen=True, slots=True)
class FreshnessVerdict:
    """The gate's decision, plus what it is based on."""

    state: FreshnessState
    #: Age in days of the newest accepted evidence. ``None`` when there is none.
    age_days: float | None
    #: Newest accepted evidence, or ``None`` for ``absent``.
    evidence_through: datetime | None
    #: Start of the retrieval window. ``None`` when the gate imposes no window,
    #: i.e. when fresh.
    window_start: datetime | None
    #: Why the state is what it is, for the banner copy. The threshold is always
    #: shown to the user rather than hidden.
    explanation: str

    @property
    def is_fresh(self) -> bool:
        return self.state is FreshnessState.FRESH

    @property
    def allows_present_tense(self) -> bool:
        """Only ``fresh`` may assert present tense."""
        return self.state is FreshnessState.FRESH

    @property
    def allows_simulation(self) -> bool:
        """``aging`` allows simulation labelled historical; ``stale`` and
        ``absent`` block it."""
        return self.state in (FreshnessState.FRESH, FreshnessState.AGING)

    @property
    def allows_analysis(self) -> bool:
        """Every state allows *some* analysis. ``absent`` returns ``unknown``,
        which is a useful result rather than a failure."""
        return True

    @property
    def permitted_scope(self) -> TemporalScope:
        """The only ``temporal_scope`` a response may carry."""
        if self.state is FreshnessState.FRESH:
            return TemporalScope.PRESENT
        if self.state is FreshnessState.AGING:
            return TemporalScope.HISTORICAL
        return TemporalScope.UNKNOWN


class FreshnessGate:
    """Computes freshness state and constrains responses accordingly.

    The thresholds are the integrator's — ``fresh_days`` and ``stale_days`` are
    constructor arguments, not library constants, because a threshold is a
    claim about how long a behaviour persists and that claim belongs to whoever
    is making it.
    """

    def __init__(self, *, fresh_days: int = 45, stale_days: int = 180) -> None:
        if fresh_days < 0:
            raise ValueError(f"fresh_days must be >= 0, got {fresh_days}")
        if stale_days < fresh_days:
            raise ValueError(f"stale_days ({stale_days}) must be >= fresh_days ({fresh_days})")
        self._fresh_days = fresh_days
        self._stale_days = stale_days

    @property
    def fresh_days(self) -> int:
        return self._fresh_days

    @property
    def stale_days(self) -> int:
        return self._stale_days

    def evaluate(
        self,
        evidence_through: datetime | None,
        *,
        now: datetime,
        archived: bool = False,
    ) -> FreshnessVerdict:
        """State for one corpus.

        ``evidence_through`` is the newest **accepted** evidence. Candidates do
        not count: a claim that has not been corroborated is not evidence.

        ``archived`` forces ``stale`` by definition: freezing a version means
        the user has stopped supplying evidence, and the product should say so
        rather than pretending the last-known state is current.
        """
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")

        if evidence_through is None:
            return FreshnessVerdict(
                state=FreshnessState.ABSENT,
                age_days=None,
                evidence_through=None,
                window_start=None,
                explanation=(
                    "No accepted evidence yet. Current communication is unknown; "
                    "upload a few screenshots to build a profile."
                ),
            )

        # A naive timestamp is read as UTC, not rejected: chat exports write
        # naive timestamps constantly, and `now` has already been required to
        # be aware, so the comparison is unambiguous.

        age = max(0.0, age_days(now, evidence_through))

        if archived or age > self._stale_days:
            state = FreshnessState.STALE
            explanation = (
                f"Your latest evidence is {int(age)} days old, so we can't model "
                "their current style. You can still ask how things went historically."
            )
        elif age > self._fresh_days:
            state = FreshnessState.AGING
            explanation = (
                f"Your latest evidence is {int(age)} days old. We can describe how "
                "they communicated up to then, not how they communicate now."
            )
        else:
            state = FreshnessState.FRESH
            explanation = f"Latest evidence is {int(age)} days old."

        window_start = None
        if state in (FreshnessState.AGING, FreshnessState.STALE):
            # Bounding retrieval is what stops newer-sounding material outside
            # the window leaking into a historical answer.
            window_start = evidence_through

        return FreshnessVerdict(
            state=state,
            age_days=age,
            evidence_through=evidence_through,
            window_start=window_start,
            explanation=explanation,
        )

    def constrain_schema(
        self, schema: Mapping[str, Any], verdict: FreshnessVerdict
    ) -> dict[str, Any]:
        """Narrow ``temporal_scope`` in the response schema.

        This runs *before* the prompt and before the decoder, so a constrained
        endpoint cannot emit a present-tense claim for a stale corpus at all.
        Doing it only as a post-response check would mean generating the claim
        and then discarding it, which is both slower and less safe: a
        constrained decode cannot produce the forbidden value.

        **Recurses into nested properties.** Claims live at different depths by
        mode — a chat response has none, an analysis puts them under
        ``analysis.interpretations[]``, a simulation under ``simulations[]`` —
        and a top-level-only constraint would silently leave the whole nested
        claim array unconstrained while appearing to have applied the gate.
        """
        return _constrain_temporal_scope(schema, permitted=verdict.permitted_scope.value, depth=0)

    def constraining_instruction(self, verdict: FreshnessVerdict) -> str | None:
        """Extra system rule for endpoints where the schema is advisory.

        Where the schema does the work this is unnecessary. Where the output is
        validated rather than constrained, the instruction matters — and
        :class:`TemporalValidator` still backstops it either way.
        """
        if verdict.state is FreshnessState.FRESH:
            return None
        if verdict.state is FreshnessState.AGING:
            return (
                "All claims must be historical. Do not describe how this person "
                "communicates now, and do not use the present tense about their "
                "current state. Their latest evidence is "
                f"{int(verdict.age_days or 0)} days old."
            )
        return (
            "Describe only what is supported by the historical evidence provided. "
            "Current communication is unknown. Do not assert anything about the "
            "present tense, and do not fill the gap with a guess."
        )


class TemporalValidator:
    """Post-response validator: downgrades rather than errors.

    A present-tense claim made on a stale corpus is a real bug in the model's
    reasoning, but throwing the whole answer away would be worse than making it
    honest: the user still gets the historical reading, just not as a claim
    about now.
    """

    def __init__(self) -> None:
        self.violations: list[str] = []

    def validate(self, payload: dict[str, Any], verdict: FreshnessVerdict) -> dict[str, Any]:
        """Return the payload with forbidden present-tense claims downgraded."""
        if verdict.allows_present_tense:
            return payload

        for claim in _iter_claims(payload):
            scope = claim.get("temporal_scope")
            if scope == TemporalScope.PRESENT.value:
                claim["temporal_scope"] = TemporalScope.HISTORICAL.value
                claim["downgraded_by_freshness_gate"] = True
                self.violations.append(
                    f"present-tense claim downgraded from {claim.get('id', '<unnamed>')}"
                )
        return payload

    def reset(self) -> None:
        self.violations.clear()

    @property
    def violation_count(self) -> int:
        return len(self.violations)


def _constrain_temporal_scope(
    schema: Mapping[str, Any], *, permitted: str, depth: int
) -> dict[str, Any]:
    """Narrow every ``temporal_scope`` in ``schema``, at any depth.

    A copy is returned rather than the input mutated, because the same
    module-level schema constant is rendered for many turns and narrowing it
    once would leave every later turn — including a *fresh* one — constrained
    by the stalest verdict the process has ever seen.
    """
    constrained = _constrain_node(schema, permitted=permitted, depth=depth, name=None)
    return constrained if isinstance(constrained, dict) else dict(schema)


def _constrain_node(
    node: Any,
    *,
    permitted: str,
    depth: int,
    name: str | None,
    seen: frozenset[int] = frozenset(),
) -> Any:
    """Walk one schema node, narrowing the enum of a node *named* ``temporal_scope``.

    The depth cap stops the *recursion*, not the constraining: past the cap the
    walk stops descending but the node it lands on is still constrained, so a
    ``temporal_scope`` at any depth is narrowed. A ``temporal_scope`` the walk
    never reaches — because the schema is self-referential — is a schema that
    cannot be rendered at all, and the identity check returns it unchanged
    rather than looping forever.
    """
    if id(node) in seen:
        # A self-referential schema. Stopping here is the hang guard; it
        # cannot be rendered, so it cannot leak a claim either.
        return node
    if isinstance(node, list):
        return [
            _constrain_node(
                item,
                permitted=permitted,
                depth=depth + 1,
                name=name,
                seen=seen | {id(node)},
            )
            for item in node
        ]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "temporal_scope" and isinstance(value, dict):
            enum = value.get("enum")
            if isinstance(enum, list):
                # The gate *permits* one scope. Every other scope in the enum
                # is removed, so the decoder cannot emit it at all. The
                # permitted scope is kept even if it was absent from the enum
                # — otherwise the node would end up with an empty enum, which
                # is a schema no decoder can satisfy.
                narrowed = [item for item in enum if item == permitted]
                if not narrowed:
                    narrowed = [permitted]
                out[key] = {**value, "enum": narrowed}
                continue
        child_depth = depth + 1
        out[key] = _constrain_node(
            value,
            permitted=permitted,
            depth=child_depth,
            name=key,
            seen=seen | {id(node)},
        )
    return out


def _iter_claims(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every claim-bearing object in a payload, at any depth.

    Walked generically rather than by a fixed list of container names, because
    a claim's depth varies by mode and a fixed list would leave the whole
    nested claim array unvalidated while appearing to have been checked.
    """
    from trace_core.reference import CLAIM_CONTAINERS

    found: list[dict[str, Any]] = []

    def walk(node: Any, depth: int, seen: frozenset[int]) -> None:
        # The walk does not stop at a depth limit. A limit here is a limit on
        # how deep a present-tense claim can hide from the downgrade — an
        # audit found one at depth 7 that was never touched, so the gate
        # reported itself as having validated the payload while leaving a
        # current-tense claim in it. The hang guard is the identity check, not
        # a depth.
        if id(node) in seen:
            # A self-referential payload. Stopping here is the hang guard.
            return
        if isinstance(node, dict):
            if "temporal_scope" in node:
                found.append(node)
            for value in node.values():
                walk(value, depth + 1, seen | {id(node)})
        elif isinstance(node, list):
            for item in node:
                walk(item, depth + 1, seen | {id(node)})

    for container in CLAIM_CONTAINERS:
        claims = payload.get(container)
        if isinstance(claims, list):
            for claim in claims:
                walk(claim, 0, frozenset())
    walk(payload, 0, frozenset())

    # A claim reached twice — once by container, once by the generic walk —
    # must not be downgraded twice.
    seen: list[dict[str, Any]] = []
    for claim in found:
        if not any(claim is prior for prior in seen):
            seen.append(claim)
    return seen
