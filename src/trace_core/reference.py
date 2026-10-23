"""Reference validation: what may be shown, given what can be checked.

A model returns a payload. Some of it cites evidence that exists; some of it
does not. The rule here is the same one the rest of the package obeys:

**A citation to an unknown id is not repaired.** The id is dropped, and a
claim left with no citation is dropped with it. "Repairing" would mean picking
a plausible id, which produces a claim citing evidence nobody checked — the
failure this whole package exists to catch, manufactured by the defence
itself.

**A failed validator downgrades the payload; it does not fail the turn.**
Discarding a response containing one uncited claim loses a correct answer
because of one bad part. Stripping the bad part keeps the answer and removes
the part that cannot be trusted.

**A `fact` without a resolvable citation is dropped, whatever its other
refs.** Not downgraded to `observation`. The epistemic type is a column and
not a description, so silently re-typing a fact is the same coercion as
turning a `simulation` into a `fact`.

This module is the "R" of the module map: the check between what a model said
and what may be shown.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CLAIM_CONTAINERS",
    "KNOWN_CLAIM_SHAPES",
    "RETIRED_CLAIM_CONTAINERS",
    "ValidationReport",
    "citation_validator",
    "claim_type_validator",
    "contact_gate_validator",
    "resolve_citations",
    "run_validators",
]

#: Containers a claim-bearing object can appear in, per mode.
#:
#: Enumerated rather than descended, so a new surface has to be *added* to this
#: tuple before its claims can appear at all. A validator that walked the
#: payload generically would silently skip a shape nobody thought to name.
CLAIM_CONTAINERS: tuple[str, ...] = (
    "interpretations",
    "simulations",
    "reply_suggestions",
    "claims",
)

#: Every name a claim-bearing container could plausibly carry, **including ones
#: no schema uses today**.
#:
#: Deliberately broader than :data:`CLAIM_CONTAINERS`. A renamed composer
#: container has to land inside the intersection of the two to be caught, and a
#: test that intersects the producer's names against the validator's own list
#: cannot catch a rename — the renamed name simply stops appearing in the
#: result.
#:
#: The superset lives here rather than in a test because it is a claim about the
#: *namespace* claims can occupy, which is a product fact: adding a composer
#: container should force someone to ask whether it belongs in
#: :data:`CLAIM_CONTAINERS`.
KNOWN_CLAIM_SHAPES: frozenset[str] = frozenset(
    {
        # Written out rather than splatted from CLAIM_CONTAINERS. Deriving the
        # superset from the list it is supposed to check makes the two move
        # together: deleting "simulations" from CLAIM_CONTAINERS would delete it
        # from the superset in the same edit, and a cross-check would see an
        # empty difference in *both* directions and pass.
        "interpretations",
        "simulations",
        "reply_suggestions",
        "claims",
        # Near-synonyms, kept deliberately — "scenarios" is the exact confusion
        # that shipped once, and it is the name a future composer is most likely
        # to reach for. It has to be *caught* rather than assumed absent.
        "scenarios",
        "scenario",
        "suggestions",
    }
)

#: Names the validator walks that no composer declares today.
#:
#: "claims" is *reserved*, not retired: it is the name any future composer is
#: expected to use, and having the validator already walk it means a new
#: surface gets citation checking by construction.
RETIRED_CLAIM_CONTAINERS: frozenset[str] = frozenset({"claims"})

#: Strip reasons → the bounded label set a metric is partitioned by.
#:
#: A closed map rather than the free-text reason, because a metric label
#: carrying claim text would put model output about a third party into an
#: observability backend. An unmapped reason is counted as "unknown" rather
#: than dropped, because a drop would make the count read as "no violations".
VIOLATION_TYPES: Mapping[str, str] = {
    "uncited_claim": "uncited_claim",
    "citation_not_in_packet": "unresolvable_citation",
    "unresolvable_citation": "unresolvable_citation",
    "model_output_rejected": "model_output_rejected",
    "forbidden_epistemic_type": "forbidden_epistemic_type",
    "mode_shape_mismatch": "mode_shape_mismatch",
    "fact_without_resolvable_evidence": "uncited_claim",
}


def violation_type(reason: str) -> str:
    """The bounded metric label for a strip reason."""
    return VIOLATION_TYPES.get(reason, "unknown")


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """What the validators did to one payload."""

    payload: dict[str, Any]
    #: Claims removed, each with the validator that removed it.
    stripped: tuple[tuple[str, str], ...] = ()
    #: Claims downgraded by the temporal gate.
    temporal_violations: tuple[str, ...] = ()
    #: Free-form notes, for the integrator.
    notes: tuple[str, ...] = field(default=())

    @property
    def reasons_by_claim(self) -> dict[str, tuple[str, ...]]:
        """Strip reasons grouped by claim id.

        A claim can be stripped for more than one reason — a simulation with no
        citations is both uncited and forbidden — and the reasons are kept
        separate so a reader can tell "this was refused" from "this was
        rejected twice".
        """
        grouped: dict[str, list[str]] = {}
        for claim_id, reason in self.stripped:
            grouped.setdefault(claim_id, []).append(reason)
        return {key: tuple(value) for key, value in grouped.items()}

    @property
    def ok(self) -> bool:
        """Whether the structured payload may be shown as-is.

        False whenever anything was stripped or downgraded: a failed validator
        downgrades the turn, and "the answer minus its bad parts" is still a
        validator failure as far as the contract is concerned.
        """
        return not self.stripped and not self.temporal_violations

    @property
    def degraded(self) -> bool:
        """Whether the payload is usable but marked degraded."""
        return not self.ok

    def reason(self) -> str:
        """The user-visible explanation for a degraded payload.

        Names the *kind* of problem, never the claim's content: the stripped
        text is model output about a third party, and an error payload is safe
        to show.
        """
        if self.ok:
            return ""
        reasons = sorted({reason for _id, reason in self.stripped})
        if self.temporal_violations:
            reasons.append("freshness_gate")
        return ", ".join(reasons)

    def violation_counts(self) -> dict[str, int]:
        """Stripped claims by bounded metric label."""
        counts: dict[str, int] = {}
        for _id, reason in self.stripped:
            label = violation_type(reason)
            counts[label] = counts.get(label, 0) + 1
        return counts


def _ref_id(ref: Any) -> str | None:
    """The id of a reference, whether it is a bare string or a mapping."""
    if isinstance(ref, str):
        return ref
    if isinstance(ref, Mapping):
        value = ref.get("id")
        if isinstance(value, str):
            return value
    return None


def citation_validator(
    payload: Mapping[str, Any], *, allowed_ids: frozenset[str]
) -> ValidationReport:
    """Every citation must resolve to an id the caller allows.

    ``allowed_ids`` is the union of the turn's evidence packet ids and the
    persona snapshot's pinned claim ids — both, and only both. A claim citing
    neither is not "probably right", it is uncheckable, which in a system whose
    entire claim is that you can check things is the same as wrong.

    A claim whose refs resolve to *some* allowed ids keeps the resolvable ones
    and loses the rest. A claim left with none is dropped.
    """
    updated = dict(payload)
    stripped: list[tuple[str, str]] = []

    cited = updated.get("cited_ids")
    if isinstance(cited, list):
        resolved_ids = [str(value) for value in cited if str(value) in allowed_ids]
        if len(resolved_ids) != len(cited):
            stripped.append(("<cited_ids>", "citation_not_in_packet"))
        updated["cited_ids"] = resolved_ids

    for container in CLAIM_CONTAINERS:
        claims = updated.get(container)
        if not isinstance(claims, list):
            continue
        kept: list[Any] = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            refs = claim.get("evidence_refs")
            if not isinstance(refs, list) or not refs:
                # No refs is not this validator's failure to strip — the
                # epistemic-type rule owns it — but a claim with no refs at all
                # cannot be kept either.
                stripped.append((str(claim.get("id", "<unnamed>")), "uncited_claim"))
                continue
            resolved = [ref for ref in refs if _ref_id(ref) in allowed_ids]
            if not resolved:
                stripped.append((str(claim.get("id", "<unnamed>")), "citation_not_in_packet"))
                continue
            if len(resolved) != len(refs):
                claim = {**claim, "evidence_refs": resolved}
            kept.append(claim)
        updated[container] = kept

    return ValidationReport(payload=updated, stripped=tuple(stripped))


def claim_type_validator(
    payload: Mapping[str, Any], *, citation_ids: frozenset[str]
) -> ValidationReport:
    """No ``fact`` without a resolvable ``evidence_ref``.

    A ``fact`` that reaches this point already has surviving refs, so what is
    being checked is whether those refs *resolve*. A fact whose refs are
    present but all point outside the allowed set is dropped rather than
    downgraded: the epistemic type is a column and coercion is how a simulation
    becomes a fact.
    """
    updated = dict(payload)
    stripped: list[tuple[str, str]] = []

    for container in CLAIM_CONTAINERS:
        claims = updated.get(container)
        if not isinstance(claims, list):
            continue
        kept: list[Any] = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            epistemic = claim.get("epistemic_type")
            refs = claim.get("evidence_refs") or []
            resolved = [ref for ref in refs if _ref_id(ref) in citation_ids]
            if epistemic == "fact" and not resolved:
                stripped.append(
                    (str(claim.get("id", "<unnamed>")), "fact_without_resolvable_evidence")
                )
                continue
            kept.append(claim)
        updated[container] = kept

    return ValidationReport(payload=updated, stripped=tuple(stripped))


def contact_gate_validator(
    payload: Mapping[str, Any], *, simulation_allowed: bool
) -> ValidationReport:
    """No ``simulation`` for a gated person.

    The second door, not the first. A person-level gate refuses the capability
    before the model is called; this catches the case where a capability was
    routed without passing the gate, which is exactly the kind of divergence
    this module exists to make visible rather than silently tolerate.
    """
    if simulation_allowed:
        return ValidationReport(payload=dict(payload))

    updated = dict(payload)
    stripped: list[tuple[str, str]] = []
    claims = updated.get("simulations")
    if isinstance(claims, list):
        kept: list[Any] = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            if claim.get("epistemic_type") == "simulation":
                stripped.append((str(claim.get("id", "<unnamed>")), "forbidden_epistemic_type"))
                continue
            kept.append(claim)
        updated["simulations"] = kept

    return ValidationReport(payload=updated, stripped=tuple(stripped))


def run_validators(
    payload: Mapping[str, Any],
    *,
    allowed_ids: frozenset[str],
    citation_ids: frozenset[str] | None = None,
    simulation_allowed: bool = True,
) -> ValidationReport:
    """Run every validator, in order, and merge the results.

    Order matters: the citation validator runs first, so the epistemic-type
    validator sees only claims whose refs survived. A fact that would have been
    dropped is not re-typed on the way past.
    """
    # Each validator runs on the *previous* validator's output, so a claim
    # dropped by the citation validator is not seen by the next one and cannot
    # be re-reported as stripped twice. Running them all against the original
    # payload would make each compute its own "kept" list from the same
    # starting point, and merging those lists would resurrect a stripped
    # claim.
    citation_ids = citation_ids if citation_ids is not None else allowed_ids

    first = citation_validator(payload, allowed_ids=allowed_ids)
    second = claim_type_validator(first.payload, citation_ids=citation_ids)
    third = contact_gate_validator(second.payload, simulation_allowed=simulation_allowed)

    return ValidationReport(
        payload=third.payload,
        stripped=(*first.stripped, *second.stripped, *third.stripped),
        temporal_violations=(),
    )


def resolve_citations(
    citations: Sequence[Any], *, allowed_ids: frozenset[str]
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """Split citations into those that resolve and those that do not.

    Returns ``(resolved, dropped_ids)``. A citation is never repaired: an
    unresolvable id is reported, not replaced with a plausible one.

    This is the native-typed counterpart to :func:`citation_validator`, for a
    caller holding CDI-003 citations rather than a raw payload.
    """
    resolved: list[Any] = []
    dropped: list[str] = []
    for citation in citations:
        evidence_id = getattr(citation, "evidence_id", None)
        if evidence_id is None and isinstance(citation, Mapping):
            value = citation.get("evidence_id")
            evidence_id = value if isinstance(value, str) else None
        if evidence_id is None:
            dropped.append("<malformed>")
            continue
        if evidence_id in allowed_ids:
            resolved.append(citation)
        else:
            dropped.append(evidence_id)
    return tuple(resolved), tuple(dropped)


def unknown_containers(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Claim-shaped containers present in a payload that the validator does not walk.

    A composer that invents a new container name gets its claims checked only
    if the validator knows the name. This reports the ones it does not, so a
    rename is caught rather than silently skipping citation checking.
    """
    present = {
        key
        for key in payload
        if isinstance(payload.get(key), list)
        and (key in KNOWN_CLAIM_SHAPES or key.endswith("claims") or key.endswith("suggestions"))
    }
    walked = set(CLAIM_CONTAINERS)
    return tuple(sorted(present - walked))


def iter_claim_ids(payload: Mapping[str, Any]) -> Iterable[str]:
    """Every claim id in a payload, across the walked containers."""
    for container in CLAIM_CONTAINERS:
        claims = payload.get(container)
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if isinstance(claim, dict):
                identifier = claim.get("id")
                if isinstance(identifier, str):
                    yield identifier
