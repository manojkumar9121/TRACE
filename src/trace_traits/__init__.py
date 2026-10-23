"""TRACE Traits — measuring and predicting how a person communicates.

Where :mod:`trace_core` scores claims, this package produces them.

It is a separate package on purpose. PLAN.md §6 A4 says core defines no
features, and that is the right call: a feature is a claim about what a
*person* is like, and the vocabulary of traits is the integrator's, not the
library's. :mod:`trace_traits` is one such integrator.

The pieces:

``corpus``
    Messages, and the evidence they become. This is where lineage is decided,
    and where the assumption that makes corroboration mean anything lives.

``features``
    Measurable signals — counts and ratios over stored messages — banded into
    words. Deterministic: the same corpus always produces the same reading.

``profile``
    The join. Every feature becomes a claim with evidence links, scored and
    decided by :mod:`trace_core`.

``predict``
    What this person is likely to do next, as a distribution with an interval,
    derived from their own history.

``temporal``
    When they write, and what that means. The model labels each message's
    communicative purpose; the arithmetic asks whether each purpose clusters
    at a particular hour; a reading of the strongest pattern is checked
    against the cited text.

``evaluate``
    Whether any of it works. Feature recovery and prediction accuracy are
    measured against a synthetic corpus with planted parameters, so a claim
    about the extractor can be checked rather than believed.
"""

from __future__ import annotations

from trace_traits.corpus import (
    Corpus,
    LineageStrategy,
    Message,
    Person,
    corpus_from_messages,
    corpus_lineages,
    count_conversations,
    evidence_from_corpus,
)
from trace_traits.evaluate import (
    FeatureEval,
    Plant,
    PredictionEval,
    evaluate_features,
    evaluate_prediction,
    synthetic_corpus,
)
from trace_traits.features import (
    MIN_CONVERSATIONS,
    MIN_SAMPLE,
    MIN_VOCABULARY_WORDS,
    MSTTR_WINDOW,
    DimensionBand,
    FeatureReading,
    FeatureVector,
    extract_features,
)
from trace_traits.perception import (
    LLMBackend,
    Observation,
    OpenRouterBackend,
    PerceptionResult,
    PerceptionTier,
    Proposal,
    ProposalError,
    claims_from_proposals,
    propose_observations,
)
from trace_traits.predict import (
    ContextualReplyPredictor,
    ContextualSkill,
    Prediction,
    ReplyContext,
    ReplyPredictor,
    build_contextual_predictor,
    build_predictor,
)
from trace_traits.profile import TraitProfile, TraitReading, build_profile, claims_for_features
from trace_traits.reference_policy import (
    REFERENCE_HALF_LIVES,
    REFERENCE_POLICY,
    REFERENCE_SOURCE_WEIGHTS,
    reference_scorer,
)
from trace_traits.respond import (
    Margin,
    ReactivityReading,
    Stimulus,
    StimulusKind,
    candidate_stimuli,
    reactivity,
)
from trace_traits.temporal import (
    PURPOSES,
    LabelResult,
    MessageLabel,
    SilenceInterval,
    SilenceReading,
    TemporalInterpretation,
    TemporalPattern,
    TemporalReading,
    TemporalTier,
    WeekdayPattern,
    interpret_pattern,
    label_messages,
    measure_activity,
    measure_patterns,
    measure_silences,
    measure_weekday,
)

__all__ = [
    # corpus
    "Corpus",
    "LineageStrategy",
    "Message",
    "Person",
    "corpus_from_messages",
    "corpus_lineages",
    "count_conversations",
    "evidence_from_corpus",
    # features
    "DimensionBand",
    "FeatureReading",
    "FeatureVector",
    "MIN_CONVERSATIONS",
    "MIN_SAMPLE",
    "MIN_VOCABULARY_WORDS",
    "MSTTR_WINDOW",
    "extract_features",
    # profile
    "TraitProfile",
    "TraitReading",
    "REFERENCE_HALF_LIVES",
    "REFERENCE_POLICY",
    "REFERENCE_SOURCE_WEIGHTS",
    "build_profile",
    "claims_for_features",
    "reference_scorer",
    # perception (LLM tier)
    "LLMBackend",
    "Observation",
    "OpenRouterBackend",
    "PerceptionResult",
    "PerceptionTier",
    "Proposal",
    "ProposalError",
    "claims_from_proposals",
    "propose_observations",
    # prediction
    "Prediction",
    "ReplyPredictor",
    "ReplyContext",
    "ContextualReplyPredictor",
    "ContextualSkill",
    "build_contextual_predictor",
    "build_predictor",
    # response contrast (T3)
    "Margin",
    "ReactivityReading",
    "Stimulus",
    "StimulusKind",
    "candidate_stimuli",
    "reactivity",
    # temporal (T4)
    "PURPOSES",
    "LabelResult",
    "MessageLabel",
    "SilenceInterval",
    "SilenceReading",
    "TemporalInterpretation",
    "TemporalPattern",
    "TemporalReading",
    "TemporalTier",
    "WeekdayPattern",
    "interpret_pattern",
    "label_messages",
    "measure_activity",
    "measure_patterns",
    "measure_silences",
    "measure_weekday",
    # evaluation
    "FeatureEval",
    "Plant",
    "PredictionEval",
    "evaluate_features",
    "evaluate_prediction",
    "synthetic_corpus",
]
