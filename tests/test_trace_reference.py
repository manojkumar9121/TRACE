"""Tests for :mod:`trace_core.reference` — the reference validators.

These are the checks between what a model said and what may be shown. The
rule that matters most: **a citation to an unknown id is not repaired.**
"""

from __future__ import annotations

from typing import Any

from trace_core.reference import (
    CLAIM_CONTAINERS,
    KNOWN_CLAIM_SHAPES,
    RETIRED_CLAIM_CONTAINERS,
    citation_validator,
    claim_type_validator,
    contact_gate_validator,
    resolve_citations,
    run_validators,
    unknown_containers,
    violation_type,
)


def payload() -> dict[str, Any]:
    return {
        "interpretations": [
            {
                "id": "i1",
                "epistemic_type": "observation",
                "evidence_refs": [{"id": "e1"}, {"id": "GHOST"}],
            },
            {"id": "i2", "epistemic_type": "fact", "evidence_refs": [{"id": "GHOST"}]},
            {"id": "i3", "epistemic_type": "observation", "evidence_refs": []},
        ],
        "simulations": [{"id": "s1", "epistemic_type": "simulation"}],
    }


ALLOWED = frozenset({"e1", "e2"})


# -- citation resolution -----------------------------------------------------


def test_a_resolvable_citation_survives():
    report = citation_validator(payload(), allowed_ids=ALLOWED)
    kept = report.payload["interpretations"]
    assert [claim["id"] for claim in kept] == ["i1"]


def test_an_unresolvable_citation_is_dropped_not_repaired():
    report = citation_validator(payload(), allowed_ids=ALLOWED)
    ids = {claim_id for claim_id, _ in report.stripped}
    assert "i2" in ids
    # and the surviving claim kept only the ref that resolves
    kept = report.payload["interpretations"][0]
    assert kept["evidence_refs"] == [{"id": "e1"}]


def test_a_claim_with_no_citations_is_stripped():
    report = citation_validator(payload(), allowed_ids=ALLOWED)
    reasons = dict(report.stripped)
    assert reasons.get("i3") == "uncited_claim"


def test_a_cited_ids_list_is_filtered():
    report = citation_validator({"cited_ids": ["e1", "GHOST"]}, allowed_ids=ALLOWED)
    assert report.payload["cited_ids"] == ["e1"]
    assert any(reason == "citation_not_in_packet" for _id, reason in report.stripped)


def test_a_bare_string_ref_resolves():
    report = citation_validator(
        {"interpretations": [{"id": "x", "evidence_refs": ["e1"]}]}, allowed_ids=ALLOWED
    )
    assert report.payload["interpretations"][0]["id"] == "x"
    assert report.stripped == ()


# -- epistemic type ----------------------------------------------------------


def test_a_fact_without_resolvable_evidence_is_dropped_not_downgraded():
    """The epistemic type is a column, and coercion is how a simulation becomes a fact."""
    report = claim_type_validator(
        {"interpretations": [{"id": "f1", "epistemic_type": "fact", "evidence_refs": []}]},
        citation_ids=ALLOWED,
    )
    assert report.payload["interpretations"] == []
    assert any(reason == "fact_without_resolvable_evidence" for _id, reason in report.stripped)


def test_a_fact_with_resolvable_evidence_survives():
    report = claim_type_validator(
        {"interpretations": [{"id": "f1", "epistemic_type": "fact", "evidence_refs": ["e1"]}]},
        citation_ids=ALLOWED,
    )
    assert len(report.payload["interpretations"]) == 1


def test_an_observation_without_evidence_is_not_dropped_by_the_type_rule():
    report = claim_type_validator(
        {"interpretations": [{"id": "o1", "epistemic_type": "observation", "evidence_refs": []}]},
        citation_ids=ALLOWED,
    )
    assert len(report.payload["interpretations"]) == 1


# -- the contact gate --------------------------------------------------------


def test_a_simulation_is_dropped_when_not_allowed():
    report = contact_gate_validator(payload(), simulation_allowed=False)
    assert report.payload["simulations"] == []
    assert any(reason == "forbidden_epistemic_type" for _id, reason in report.stripped)


def test_a_simulation_survives_when_allowed():
    report = contact_gate_validator(payload(), simulation_allowed=True)
    assert len(report.payload["simulations"]) == 1
    assert report.stripped == ()


# -- the chain ---------------------------------------------------------------


def test_the_chain_strips_a_claim_once():
    """Each validator runs on the previous one's output, so a claim is not
    stripped twice and a stripped claim is not resurrected by a later merge."""
    report = run_validators(payload(), allowed_ids=ALLOWED, simulation_allowed=False)
    by_claim = report.reasons_by_claim
    assert len(by_claim["i2"]) == 1
    assert [claim["id"] for claim in report.payload["interpretations"]] == ["i1"]
    assert report.payload["simulations"] == []


def test_the_chain_degrades_rather_than_fails():
    report = run_validators(payload(), allowed_ids=ALLOWED, simulation_allowed=False)
    assert not report.ok
    assert report.degraded
    # the reason names the kind of problem, never the claim's content
    assert "citation_not_in_packet" in report.reason()
    assert "i1" not in report.reason()


def test_a_clean_payload_is_ok():
    clean = {
        "interpretations": [{"id": "i1", "epistemic_type": "observation", "evidence_refs": ["e1"]}]
    }
    report = run_validators(clean, allowed_ids=ALLOWED)
    assert report.ok
    assert report.reason() == ""


# -- naming ------------------------------------------------------------------


def test_a_renamed_container_is_reported():
    """A composer that invents a container name gets caught, not skipped."""
    found = unknown_containers({"interpretations": [], "scenarios": []})
    assert found == ("scenarios",)


def test_the_known_shapes_superset_is_wider_than_the_walked_containers():
    assert set(CLAIM_CONTAINERS) < KNOWN_CLAIM_SHAPES
    assert "scenarios" in KNOWN_CLAIM_SHAPES
    assert set(CLAIM_CONTAINERS) >= RETIRED_CLAIM_CONTAINERS


def test_an_unmapped_reason_becomes_unknown_not_a_drop():
    """A dropped count must never read as 'no violations'."""
    assert violation_type("a_reason_nobody_mapped") == "unknown"


# -- the native-typed counterpart --------------------------------------------


def test_resolve_citations_splits_resolved_from_dropped():
    resolved, dropped = resolve_citations(
        [{"evidence_id": "e1"}, {"evidence_id": "GHOST"}], allowed_ids=ALLOWED
    )
    assert len(resolved) == 1
    assert dropped == ("GHOST",)


def test_resolve_citations_never_repairs():
    resolved, dropped = resolve_citations([{"evidence_id": "GHOST"}], allowed_ids=ALLOWED)
    assert resolved == ()
    assert dropped == ("GHOST",)
