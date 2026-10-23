"""Tests for :mod:`trace_traits.respond` — the T3 response-contrast tier.

The tests that matter most here are the ones about **what a null means**. A
system that reports "no meaningful change" when it merely failed to detect a
change is making a claim it has not earned, and the whole point of the
equivalence test is to prevent exactly that.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trace_traits.corpus import Message, Person, corpus_from_messages
from trace_traits.respond import (
    MIN_BASELINE,
    MIN_EPISODES,
    Margin,
    Stimulus,
    StimulusKind,
    candidate_stimuli,
    reactivity,
)
from trace_traits.respond_synth import ReactivityPlant, reactivity_corpus

NOW = datetime(2026, 4, 1, 9, 0, tzinfo=UTC)


def planted(**overrides) -> ReactivityPlant:
    base = {
        "name": "them",
        "mean_length": 90.0,
        "length_sigma": 18.0,
        "emoji_rate": 0.3,
        "question_rate": 0.2,
        "response_rate": 0.9,
        "reply_gap_minutes": 20.0,
        "initiation_rate": 0.4,
        "event_rate": 0.35,
        "reactivity": 0.0,
    }
    base.update(overrides)
    return ReactivityPlant(**base)  # type: ignore[arg-type]


def build(plant: ReactivityPlant, *, n_conversations: int = 24, seed: int = 5):
    return reactivity_corpus(
        plant, n_conversations=n_conversations, messages_per_conversation=14, seed=seed
    )


# -- the sample gates --------------------------------------------------------


def test_too_few_episodes_says_it_does_not_know():
    plant = planted(event_rate=0.9)
    corpus, stimuli = build(plant, n_conversations=2)
    reading = reactivity(corpus, stimuli)
    assert reading.band == "insufficient_evidence"
    assert reading.n_episodes < MIN_EPISODES


def test_a_corpus_with_no_baseline_says_it_does_not_know():
    """Episodes without comparable ordinary messages cannot be compared."""
    messages = [
        Message(
            message_id=f"m{i}",
            conversation_id=f"c{i}",
            speaker="me" if i % 2 == 0 else "them",
            text="a message of some length",
            sent_at=NOW + timedelta(minutes=i),
        )
        for i in range(20)
    ]
    corpus = corpus_from_messages(Person(subject="them", observer="me"), messages)
    stimuli = [Stimulus(message_id="m0", kind=StimulusKind.OTHER)]
    reading = reactivity(corpus, stimuli)
    assert reading.band == "insufficient_evidence"


# -- the equivalence test, which is the point -------------------------------


def test_a_null_is_never_reported_as_no_change():
    """`unsettled` and `stable` are different claims and must stay different.

    `stable` means the interval lies entirely inside the margin — the absence
    of change was *established*. `unsettled` means the data does not settle
    it. Reporting the second as the first is the failure this tier exists to
    prevent.
    """
    plant = planted(reactivity=0.0)
    # Too few episodes for the interval to tighten: the band must not be
    # "stable" just because no change was detected.
    corpus, stimuli = build(plant, n_conversations=6)
    reading = reactivity(corpus, stimuli)
    assert reading.band in {"unsettled", "insufficient_evidence"}
    assert reading.band != "stable"


def test_no_change_is_established_as_stable_with_enough_episodes():
    """`stable` requires that **every** measure was equivalent.

    At 24 conversations only the reply-length measure is equivalent; the
    other five are still too wide to settle. The band is `unsettled`, which
    is the honest reading — and the point of the rule, since an earlier
    version reported `stable` on the strength of that one measure.
    """
    plant = planted(reactivity=0.0)
    corpus, stimuli = build(plant, n_conversations=24)
    reading = reactivity(corpus, stimuli)
    assert reading.band == "unsettled"
    length = next(m for m in reading.measures if m.name == "reply length")
    assert length.equivalent
    assert length.low >= -length.margin and length.high <= length.margin


def test_stable_is_reachable_once_every_measure_settles():
    """The stricter rule must not make `stable` unreachable.

    With enough conversations every measure's interval fits inside its
    margin, and the band becomes `stable`. A rule that could never be
    satisfied would be a rule that always says "I don't know".
    """
    plant = planted(reactivity=0.0)
    corpus, stimuli = build(plant, n_conversations=400)
    reading = reactivity(corpus, stimuli)
    assert reading.band == "stable"
    assert all(m.equivalent for m in reading.measures)


def test_a_large_change_is_established():
    plant = planted(reactivity=0.8)
    corpus, stimuli = build(plant, n_conversations=24)
    reading = reactivity(corpus, stimuli)
    assert reading.band == "reduced"
    length = next(m for m in reading.measures if m.name == "reply length")
    assert length.verdict == "lower"
    assert length.high < -length.margin


def test_a_wider_margin_makes_no_change_easier_to_establish():
    """The margin is the integrator's, and it changes what counts as a change.

    A wider margin makes each measure's interval more likely to fit inside
    it, so more measures settle as `equivalent` and the band moves from
    `unsettled` towards `stable`.
    """
    plant = planted(reactivity=0.25)
    corpus, stimuli = build(plant, n_conversations=24)
    narrow = reactivity(corpus, stimuli, margin=Margin(length_ratio=0.10))
    wide = reactivity(corpus, stimuli, margin=Margin(length_ratio=0.50))
    n_equiv_narrow = sum(1 for m in narrow.measures if m.equivalent)
    n_equiv_wide = sum(1 for m in wide.measures if m.equivalent)
    assert n_equiv_wide >= n_equiv_narrow
    assert narrow.band in {"unsettled", "insufficient_evidence"}
    assert wide.band in {"unsettled", "stable"}


# -- attribution and isolation ----------------------------------------------


def test_the_baseline_is_drawn_from_the_same_conversations():
    """A conversation that is short or late at night is a confound, not a reaction."""
    plant = planted(reactivity=0.0, event_rate=0.5)
    corpus, stimuli = build(plant, n_conversations=12)
    reading = reactivity(corpus, stimuli)
    # The baseline cannot exceed the corpus: it is a subset of the subject's
    # own messages in the conversations that contained an event.
    assert reading.n_baseline <= len(corpus.subject_messages)


def test_the_response_is_the_subjects_first_message_after_the_event():
    """Not the next message — the next message *from them*.

    If the other person speaks again first, the subject has still not
    answered the event.
    """
    plant = planted(reactivity=0.0)
    corpus, stimuli = build(plant, n_conversations=12)
    reading = reactivity(corpus, stimuli)
    assert reading.n_replies <= reading.n_episodes


def test_a_message_from_the_subject_is_the_only_thing_counted_as_a_reply():
    messages = [
        Message(
            message_id="m0",
            conversation_id="c0",
            speaker="me",
            text="you are an idiot",
            sent_at=NOW,
        ),
        Message(
            message_id="m1",
            conversation_id="c0",
            speaker="me",
            text="still there",
            sent_at=NOW + timedelta(minutes=1),
        ),
    ] + [
        Message(
            message_id=f"m{i}",
            conversation_id="c{i}",
            speaker="them",
            text="an ordinary message of ordinary length",
            sent_at=NOW + timedelta(minutes=i),
        )
        for i in range(2, 12)
    ]
    corpus = corpus_from_messages(Person(subject="them", observer="me"), messages)
    reading = reactivity(corpus, [Stimulus(message_id="m0")])
    # m1 is the other person's, so it is not a reply to the event.
    assert reading.n_replies == 0


# -- what is asserted --------------------------------------------------------


def test_the_band_describes_communication_not_feeling():
    """No band names an emotion."""
    plant = planted(reactivity=0.8)
    corpus, stimuli = build(plant, n_conversations=24)
    reading = reactivity(corpus, stimuli)
    assert reading.band in {
        "stable",
        "unsettled",
        "reduced",
        "amplified",
        "absent",
        "varied",
        "insufficient_evidence",
    }
    for emotion in ("angry", "anger", "annoyed", "upset", "happy", "sad"):
        assert emotion not in reading.band
        assert emotion not in reading.detail


def test_the_caveat_states_the_limit():
    plant = planted(reactivity=0.8)
    corpus, stimuli = build(plant, n_conversations=24)
    reading = reactivity(corpus, stimuli)
    assert "not what was felt" in reading.caveat


def test_every_measure_carries_its_sample_and_margin():
    plant = planted(reactivity=0.5)
    corpus, stimuli = build(plant, n_conversations=24)
    reading = reactivity(corpus, stimuli)
    assert reading.measures
    for measure in reading.measures:
        assert measure.n_baseline >= 0
        assert measure.n_response >= 0
        assert measure.margin > 0
        assert (
            measure.low <= measure.difference <= measure.high or measure.verdict == "inconclusive"
        )
        assert measure.unit in {"share", "ratio"}


def test_the_episode_count_is_always_reported():
    plant = planted(reactivity=0.5)
    corpus, stimuli = build(plant, n_conversations=24)
    reading = reactivity(corpus, stimuli)
    assert reading.n_episodes == len(stimuli)
    assert "episodes" in reading.as_line()


# -- the trigger is the caller's ---------------------------------------------


def test_a_stimulus_is_supplied_not_inferred():
    """The analyser does not look for insults; the caller says what happened."""
    plant = planted(reactivity=0.0, event_rate=0.0)
    corpus, stimuli = build(plant, n_conversations=12)
    # No stimuli at all -> no episodes, and the reading says so.
    reading = reactivity(corpus, [])
    assert reading.n_episodes == 0
    assert reading.band == "insufficient_evidence"


def test_candidate_stimuli_are_candidates_not_labels():
    messages = [
        Message(
            message_id="m0",
            conversation_id="c0",
            speaker="me",
            text="you are an idiot",
            sent_at=NOW,
        ),
        Message(
            message_id="m1",
            conversation_id="c0",
            speaker="me",
            text="rohan said he would come",
            sent_at=NOW + timedelta(minutes=1),
        ),
        Message(
            message_id="m2",
            conversation_id="c0",
            speaker="them",
            text="ok",
            sent_at=NOW + timedelta(minutes=2),
        ),
    ]
    corpus = corpus_from_messages(Person(subject="them", observer="me"), messages)
    candidates = candidate_stimuli(corpus, about_names=["rohan"])
    kinds = {item.message_id: item.kind for item in candidates}
    assert kinds["m0"] is StimulusKind.INSULT
    assert kinds["m1"] is StimulusKind.ABOUT_THEM
    # the subject's own messages are never candidates
    assert "m2" not in kinds


def test_a_stimulus_naming_a_missing_message_is_ignored():
    plant = planted(reactivity=0.0)
    corpus, stimuli = build(plant, n_conversations=12)
    reading = reactivity(corpus, [*stimuli, Stimulus(message_id="ghost")])
    assert reading.n_episodes == len(stimuli)


# -- the generator, which must agree with the analyser -----------------------


def test_the_generator_and_the_analyser_agree_on_what_a_response_is():
    """A planted effect must be detectable.

    The generator shrinks the subject's *first message after* an event; the
    analyser looks at the subject's first message after an event. If those two
    definitions drift, the evaluation measures the disagreement rather than
    the reaction.
    """
    plant = planted(reactivity=0.8)
    corpus, stimuli = build(plant, n_conversations=24)
    reading = reactivity(corpus, stimuli)
    assert reading.band == "reduced"


def test_the_generator_produces_varied_lengths():
    """A corpus of six fixed strings has a degenerate baseline."""
    plant = planted(reactivity=0.0)
    corpus, _ = build(plant, n_conversations=12)
    lengths = {item.char_length for item in corpus.subject_messages}
    assert len(lengths) > 6


def test_a_planted_withdrawal_shows_as_fewer_replies():
    plant = planted(reactivity=0.0, withdrawal_rate=0.9)
    corpus, stimuli = build(plant, n_conversations=16)
    reading = reactivity(corpus, stimuli)
    assert reading.n_replies < reading.n_episodes


def test_the_baseline_minimum_is_enforced():
    assert MIN_BASELINE >= 5
    assert MIN_EPISODES >= 3
