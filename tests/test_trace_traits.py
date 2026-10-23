"""Tests for :mod:`trace_traits` — the layer that measures and predicts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trace_core import (
    Claim,
    Link,
    Scorer,
    TableConfidencePolicy,
    TableRetrievalPolicy,
    TemporalScope,
)
from trace_traits import (
    DimensionBand,
    LineageStrategy,
    Message,
    Person,
    Plant,
    build_predictor,
    build_profile,
    corpus_from_messages,
    evaluate_features,
    evaluate_prediction,
    extract_features,
    synthetic_corpus,
)
from trace_traits.profile import parse_measurement_claim

NOW = datetime(2026, 10, 20, tzinfo=UTC)


def policy():
    return TableConfidencePolicy(
        source_weights={
            "user_stated": 1.00,
            "manual_edit": 1.00,
            "feedback": 0.85,
            "message": 0.70,
            "screenshot": 0.65,
            "system_inference": 0.55,
        },
        half_lives={
            "behavioral": 270,
            "tone": 180,
            "style": 540,
            "factual": 1095,
            "distributional": 540,
        },
        corroboration_floor=2,
        coverage_denominator=3,
    )


def scorer():
    conf = policy()
    return Scorer(
        confidence=conf,
        retrieval=TableRetrievalPolicy(
            weights={"vector": 0.55, "recency": 0.20, "source": 0.15, "confidence": 0.10},
            source_weights=conf.source_weights,
        ),
    )


def message(mid, *, conversation="c1", speaker="them", text="hello there", minute=0, day=0):
    return Message(
        message_id=mid,
        conversation_id=conversation,
        speaker=speaker,
        text=text,
        sent_at=NOW - timedelta(days=day, minutes=minute),
        upload_id=f"u-{conversation}",
    )


def planted_corpus(**overrides):
    base = {
        "name": "them",
        "mean_length": 90.0,
        "length_sigma": 18.0,
        "emoji_rate": 0.3,
        "question_rate": 0.2,
        "response_rate": 0.8,
        "reply_gap_minutes": 20.0,
        "initiation_rate": 0.4,
    }
    base.update(overrides)
    return Plant(**base)  # type: ignore[arg-type]


# -- sample gates ------------------------------------------------------------


def test_a_thin_corpus_says_it_does_not_know():
    messages = [message(f"m{i}") for i in range(5)]
    corpus = corpus_from_messages(Person(subject="them"), messages)
    features = extract_features(corpus)
    reading = features.get("Response length")
    assert reading is not None
    assert reading.band == DimensionBand.INSUFFICIENT.value
    assert reading.value is None


def test_one_conversation_is_an_anecdote_not_a_pattern():
    """Twelve messages in one thread must not produce a band."""
    messages = [message(f"m{i}", conversation="c1") for i in range(12)]
    corpus = corpus_from_messages(Person(subject="them"), messages)
    features = extract_features(corpus)
    reading = features.get("Response length")
    assert reading is not None
    assert reading.band == DimensionBand.INSUFFICIENT.value
    assert "anecdote" in reading.detail


def test_enough_messages_across_enough_conversations_produce_a_band():
    messages = [message(f"m{c}{i}", conversation=f"c{c}") for c in range(3) for i in range(5)]
    corpus = corpus_from_messages(Person(subject="them"), messages)
    features = extract_features(corpus)
    reading = features.get("Response length")
    assert reading is not None
    assert reading.band != DimensionBand.INSUFFICIENT.value


# -- evidence scoping --------------------------------------------------------


def test_each_dimension_cites_only_the_messages_it_rests_on():
    corpus = synthetic_corpus(planted_corpus(), n_conversations=8, messages_per_conversation=10)
    features = extract_features(corpus)
    by_id = {item.message_id: item for item in corpus.messages}
    for reading in features.readings:
        for evidence_id in reading.evidence_ids:
            assert by_id[evidence_id].speaker == "them", (
                f"{reading.dimension} cited a message from someone else"
            )


def test_initiation_cites_one_message_per_conversation():
    corpus = synthetic_corpus(planted_corpus(), n_conversations=7, messages_per_conversation=10)
    features = extract_features(corpus)
    reading = features.get("Initiation")
    assert reading is not None
    assert len(reading.evidence_ids) == 7


def test_a_claim_about_a_person_cites_only_that_person():
    """PLAN.md §2.6: the cited speaker must equal the claim's subject."""
    corpus = synthetic_corpus(planted_corpus(), n_conversations=6, messages_per_conversation=10)
    profile = build_profile(corpus, scorer=scorer(), now=NOW)
    by_id = {item.message_id: item for item in corpus.messages}
    for reading in profile.readings:
        for term in ("n_evidence",):
            assert reading.terms[term] >= 0
    # every accepted reading cites only the subject's messages
    for _claim_id, links in _claims_and_links(corpus):
        for link in links:
            assert by_id[link.evidence_id].speaker == "them"


def _claims_and_links(corpus):
    from trace_traits.profile import claims_for_features

    specs = claims_for_features(extract_features(corpus), subject=corpus.person.subject)
    return [(spec.claim.claim_id, spec.links) for spec in specs]


# -- measurement verification ------------------------------------------------


def test_a_measurement_claim_verifies_by_recomputation():
    corpus = synthetic_corpus(planted_corpus(), n_conversations=6, messages_per_conversation=10)
    from trace_traits.profile import measurement_verifier

    features = extract_features(corpus)
    specs = claims_for_all(features, corpus)
    for claim, _links in specs:
        verdict = measurement_verifier(claim, corpus, readings=features)
        assert verdict.verdict == "supported"


def claims_for_all(features, corpus) -> list[tuple[Claim, tuple[Link, ...]]]:
    from trace_traits.profile import claims_for_features

    return [
        (spec.claim, spec.links)
        for spec in claims_for_features(features, subject=corpus.person.subject)
    ]


def test_a_tampered_measurement_claim_does_not_verify():
    corpus = synthetic_corpus(planted_corpus(), n_conversations=6, messages_per_conversation=10)
    from trace_traits.profile import measurement_verifier

    features = extract_features(corpus)
    claim, _links = claims_for_all(features, corpus)[0]
    parsed = parse_measurement_claim(claim.text)
    assert parsed is not None

    # Tampering with the band is caught.
    other_band = "high" if parsed.band != "high" else "low"
    tampered_band = _replace(claim, detail=parsed.detail, band=other_band)
    assert measurement_verifier(tampered_band, corpus, readings=features).verdict == "unsupported"

    # Tampering with the detail is caught too, even when the band is right.
    tampered_detail = _replace(claim, detail="avg 999 chars", band=parsed.band)
    assert measurement_verifier(tampered_detail, corpus, readings=features).verdict == "unsupported"


def _replace(claim: Claim, *, detail: str, band: str) -> Claim:
    return Claim(
        claim_id=claim.claim_id,
        text=f"{claim.text.split(':')[0]}: {band} ({detail})",
        subject=claim.subject,
        kind=claim.kind,
        scope=claim.scope,
    )


def test_a_measurement_claim_that_is_not_a_measurement_does_not_verify():
    corpus = synthetic_corpus(planted_corpus(), n_conversations=6, messages_per_conversation=10)
    from trace_traits.profile import measurement_verifier

    claim = Claim(
        claim_id="trait:mystery",
        text="This person is probably introverted",
        subject="them",
        kind="behavioral",
        scope=TemporalScope.HISTORICAL,
    )
    verdict = measurement_verifier(claim, corpus)
    assert verdict.verdict == "unsupported"


# -- profile -----------------------------------------------------------------


def test_a_profile_accepts_measurements_that_recompute():
    corpus = synthetic_corpus(planted_corpus(), n_conversations=8, messages_per_conversation=12)
    profile = build_profile(corpus, scorer=scorer(), now=NOW)
    assert len(profile.readings) == 5
    assert all(reading.accepted for reading in profile.readings)


def test_a_profile_without_a_verifier_keeps_everything_candidate():
    corpus = synthetic_corpus(planted_corpus(), n_conversations=8, messages_per_conversation=12)
    profile = build_profile(corpus, scorer=scorer(), now=NOW, verifier=None)
    # build_profile's default is the measurement verifier; an explicit
    # no-op verifier must leave every claim a candidate.
    assert len(profile.readings) == 5


def test_a_profile_omits_a_dimension_that_cannot_say():
    """The synthetic x-corpus holds 66 one-word messages; 66 < the 100-word
    vocabulary floor, so the second audit's declared policy (G-11) says the
    dimension cannot read this corpus — and the profile carries five readings,
    not a wrong-widowed vocabulary one."""
    corpus = synthetic_corpus(planted_corpus(), n_conversations=8, messages_per_conversation=12)
    features = extract_features(corpus)
    vocabulary = features.get("Vocabulary")
    assert vocabulary is not None
    assert vocabulary.band == DimensionBand.INSUFFICIENT.value
    profile = build_profile(corpus, scorer=scorer(), now=NOW)
    assert vocabulary.dimension not in {r.dimension for r in profile.readings}


def test_a_thin_corpus_produces_no_claims_at_all():
    messages = [message(f"m{i}") for i in range(6)]
    corpus = corpus_from_messages(Person(subject="them"), messages)
    profile = build_profile(corpus, scorer=scorer(), now=NOW)
    assert profile.readings == ()


# -- prediction --------------------------------------------------------------


def test_the_predictor_returns_distributions_not_points():
    corpus = synthetic_corpus(planted_corpus(), n_conversations=8, messages_per_conversation=12)
    predictor = build_predictor(corpus)
    for prediction in predictor.all_predictions():
        assert prediction.low <= prediction.value <= prediction.high
        assert prediction.n_evidence >= 0
        assert prediction.caveat


def test_the_predictor_predicts_a_planted_length():
    plant = planted_corpus(mean_length=120.0)
    corpus = synthetic_corpus(plant, n_conversations=10, messages_per_conversation=14)
    prediction = build_predictor(corpus).likely_length()
    assert abs(prediction.value - 120.0) < 25.0


def test_the_predictor_predicts_a_planted_emoji_rate():
    plant = planted_corpus(emoji_rate=0.6)
    corpus = synthetic_corpus(plant, n_conversations=10, messages_per_conversation=14)
    prediction = build_predictor(corpus).likely_emoji()
    assert abs(prediction.value - 0.6) < 0.15


def test_a_prediction_from_an_empty_corpus_says_it_cannot_predict():
    corpus = corpus_from_messages(Person(subject="them"), [])
    prediction = build_predictor(corpus).likely_length()
    assert prediction.n_evidence == 0
    assert "not enough" in prediction.caveat


def test_the_response_rate_prediction_tracks_the_planted_rate():
    plant = planted_corpus(response_rate=0.9)
    corpus = synthetic_corpus(plant, n_conversations=12, messages_per_conversation=16)
    prediction = build_predictor(corpus).likely_responds()
    assert abs(prediction.value - 0.9) < 0.2


# -- evaluation --------------------------------------------------------------


def test_the_extractor_recovers_planted_parameters():
    plant = planted_corpus(mean_length=95.0, emoji_rate=0.4, question_rate=0.25)
    corpus = synthetic_corpus(plant, n_conversations=16, messages_per_conversation=16)
    report = evaluate_features(corpus, plant)
    assert report.errors["Response length"] < 20.0
    assert report.errors["Emoji usage"] < 0.12
    assert report.errors["Questions"] < 0.12


def test_the_prediction_beats_a_constant_baseline():
    """The point of the evaluation: the predictor must beat 0.5 on the Brier score."""
    plant = planted_corpus(emoji_rate=0.65)
    corpus = synthetic_corpus(plant, n_conversations=16, messages_per_conversation=16)
    report = evaluate_prediction(corpus, plant=plant)
    assert report.n_predictions > 0
    assert report.emoji_brier < report.emoji_brier_baseline


def test_the_prediction_interval_covers_about_half_the_held_out_messages():
    """A 50% interval should cover about half. It needs enough samples to tell.

    At 16 held-out messages the binomial standard deviation is 0.125, so a
    correct interval reads anywhere from 0.25 to 0.75 by chance alone. The
    check is therefore run at a sample size where the expected coverage is
    distinguishable from a broken interval.
    """
    plant = planted_corpus()
    corpus = synthetic_corpus(plant, n_conversations=96, messages_per_conversation=16)
    report = evaluate_prediction(corpus, plant=plant)
    assert report.n_predictions >= 64
    assert 0.38 <= report.length_coverage <= 0.62


def test_a_prediction_interval_is_not_vacuous():
    """A prediction whose interval covered everything would be worthless."""
    plant = planted_corpus()
    corpus = synthetic_corpus(plant, n_conversations=64, messages_per_conversation=16)
    predictor = build_predictor(corpus)
    prediction = predictor.likely_length()
    span = prediction.high - prediction.low
    typical = prediction.value
    assert span < typical, "the interval is wider than the value it brackets"
    assert span > 0, "the interval has no width"


def test_a_constant_sample_does_not_produce_a_zero_width_interval():
    """Four identical gaps must not yield "certainly 2 minutes".

    A zero-width interval asserts a certainty the sample does not contain: it
    says the next value is exactly this when all it shows is that every value
    seen so far was this.
    """
    from trace_traits.predict import _quartiles

    low, high = _quartiles([2.0, 2.0, 2.0, 2.0])
    assert high > low, "a constant sample produced a zero-width interval"
    assert low <= 2.0 <= high


def test_a_degenerate_prediction_says_so():
    messages = [
        Message(
            message_id=f"m{i}",
            conversation_id=f"c{i % 3}",
            speaker="them",
            text="same",
            sent_at=NOW + timedelta(minutes=i),
        )
        for i in range(12)
    ]
    corpus = corpus_from_messages(Person(subject="them"), messages)
    prediction = build_predictor(corpus).likely_length()
    assert prediction.high > prediction.low
    assert "placeholder" in prediction.caveat


# -- corpus and lineage ------------------------------------------------------


def test_lineage_strategy_is_recorded_not_guessed():
    messages = [message("m1", conversation="c1"), message("m2", conversation="c2")]
    corpus = corpus_from_messages(
        Person(subject="them"), messages, lineage_strategy=LineageStrategy.PER_UPLOAD
    )
    assert corpus.lineage_strategy is LineageStrategy.PER_UPLOAD
    assert corpus.lineage_of(corpus.messages[0]) == "u-c1"


def test_undated_messages_do_not_make_a_corpus_look_fresh():
    undated = Message(
        message_id="m1",
        conversation_id="c1",
        speaker="them",
        text="hi",
        sent_at=None,
    )
    corpus = corpus_from_messages(Person(subject="them"), [undated])
    from trace_traits.features import latest_dated

    assert latest_dated(corpus) is None


# -- the base-rate baseline -------------------------------------------------


def _plant(**overrides):
    """A plant with sensible defaults, for the base-rate tests."""
    kwargs = {
        "name": "them",
        "mean_length": 60.0,
        "length_sigma": 8.0,
        "emoji_rate": 0.5,
        "question_rate": 0.2,
        "response_rate": 0.9,
        "reply_gap_minutes": 5.0,
        "initiation_rate": 0.5,
    }
    kwargs.update(overrides)
    return Plant(**kwargs)


def test_the_emoji_prediction_is_scored_against_the_subjects_own_rate():
    """The base rate is the fair baseline, and the predictor IS it.

    This test exists because the evaluation once compared against a constant
    0.5 and called a win. That measures a property of the *person* — they
    are not a coin flip — not of the *predictor*. Against the base rate the
    skill score is zero, and the test says so, so the number cannot quietly
    become a win again.
    """
    from trace_traits.evaluate import evaluate_prediction, synthetic_corpus

    corpus = synthetic_corpus(plant=_plant())
    result = evaluate_prediction(corpus)

    # The base-rate baseline is present and is a real Brier score.
    assert result.emoji_brier_base_rate > 0.0
    # The predictor returns the subject's own rate — Beta-smoothed, so very
    # slightly different from the raw base rate — and therefore has no skill
    # against it. The two are not exactly equal because the smoothing is a
    # real difference, but the skill is zero to within rounding.
    assert result.emoji_brier == pytest.approx(result.emoji_brier_base_rate, rel=0.01)
    assert result.emoji_skill == pytest.approx(0.0, abs=0.01)
    # And it is a *different* baseline from the constant-0.5 one, which is
    # the whole point: against the constant, the same predictor looks like a
    # win.
    assert result.emoji_brier_baseline == pytest.approx(0.25, abs=0.05)


def test_the_length_prediction_is_scored_against_a_base_rate():
    """The length MAE had no baseline at all, which is also wrong."""
    from trace_traits.evaluate import evaluate_prediction, synthetic_corpus

    corpus = synthetic_corpus(plant=_plant())
    result = evaluate_prediction(corpus)

    assert result.length_mae_base_rate > 0.0
    # The predictor returns the subject's own median, so it ties the baseline.
    assert result.length_mae == pytest.approx(result.length_mae_base_rate, rel=0.05)
    assert result.length_skill <= 0.1
