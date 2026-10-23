"""TRACE — Time-aware, Referenced, Attributed, Corroborated Evidence.

A general-purpose library that turns heterogeneous evidence — chat exports,
email threads, transcripts, documents — into a parameterised catalogue of
observed behaviour, with every value traceable to the evidence that produced
it.

TRACE computes and explains. It does not decide what the numbers mean: the
integrator owns the thresholds, the vocabulary and the names.

See ``PLAN.md`` for the contract, the module map and the milestones, and
``README.md`` for what TRACE does not claim.
"""

from __future__ import annotations

from trace_core.attribution import (
    INSTRUCTION_SHAPED,
    AttributionPolicy,
    decide,
    instruction_detector,
    is_instruction_shaped,
)
from trace_core.errors import CitationError, PolicyError
from trace_core.evidence import (
    EvidencePacket,
    PacketCandidate,
    PacketPassage,
    ProvenanceRef,
    build_packet,
    empty_packet,
)
from trace_core.gating import FreshnessGate, FreshnessVerdict, TemporalValidator
from trace_core.lineage import fingerprint
from trace_core.policies import (
    ConfidencePolicy,
    RetrievalPolicy,
    TableConfidencePolicy,
    TableRetrievalPolicy,
)
from trace_core.reference import (
    ValidationReport,
    citation_validator,
    claim_type_validator,
    contact_gate_validator,
    resolve_citations,
    run_validators,
)
from trace_core.scoring import CONFIDENCE_TERMS, RETRIEVAL_TERMS, Scorer, score_reconstructs
from trace_core.temporal import Freshness, FreshnessState, TemporalScope, age_days, decay
from trace_core.types import (
    Cap,
    Citation,
    Claim,
    Decision,
    Evidence,
    EvidenceSummary,
    Link,
    RetrievalCandidate,
    Scored,
    Verdict,
)

__version__ = "0.2.0"

__all__ = [
    # evidence
    "Evidence",
    "Link",
    "Claim",
    "Citation",
    "Verdict",
    # results
    "EvidenceSummary",
    "Scored",
    "Decision",
    "Cap",
    "RetrievalCandidate",
    # attribution
    "AttributionPolicy",
    "decide",
    "INSTRUCTION_SHAPED",
    "instruction_detector",
    "is_instruction_shaped",
    # entry points
    "Scorer",
    # gating
    "FreshnessGate",
    "FreshnessVerdict",
    "TemporalValidator",
    # evidence
    "EvidencePacket",
    "PacketCandidate",
    "PacketPassage",
    "ProvenanceRef",
    "build_packet",
    "empty_packet",
    # reference
    "ValidationReport",
    "citation_validator",
    "claim_type_validator",
    "contact_gate_validator",
    "resolve_citations",
    "run_validators",
    # policy
    "ConfidencePolicy",
    "RetrievalPolicy",
    "TableConfidencePolicy",
    "TableRetrievalPolicy",
    # temporal
    "Freshness",
    "FreshnessState",
    "TemporalScope",
    "age_days",
    "decay",
    # lineage
    "fingerprint",
    # errors
    "CitationError",
    "PolicyError",
    # scoring helpers
    "CONFIDENCE_TERMS",
    "RETRIEVAL_TERMS",
    "score_reconstructs",
    "__version__",
]
