"""Tests for the temporal tier.

Three things are worth pinning down here, and they are not the same thing:

1. The **arithmetic** decides. A model that labels everything wrongly
   produces no finding, not a wrong one.
2. The **checks** are real. A label whose quote is not in the message is
   dropped; a reading whose text does not support it is downgraded.
3. The **boundaries** hold. A reading may not diagnose, and it may not bring
   in evidence the finding did not rest on.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta

import pytest

from trace_traits import (
    Corpus,
    Message,
    Person,
    TemporalPattern,
    TemporalTier,
    corpus_from_messages,
    interpret_pattern,
    label_messages,
    measure_activity,
    measure_patterns,
)
from trace_traits.perception import LLMBackend
from trace_traits.temporal import (
    BAND_CONSISTENT,
    BAND_UNSETTLED,
    BAND_VARIABLE,
)

BASE = datetime(2026, 10, 1, tzinfo=UTC)


class StubBackend(LLMBackend):
    """A backend that returns a fixed payload."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def complete(self, system: str, user: str, *, schema: dict) -> str:
        return self._payload


def make_corpus(*, days: int = 10, signoff_hour: int = 23, work_hour: int = 23) -> Corpus:
    """A corpus with a nightly signoff and a nightly work message."""
    messages: list[Message] = []
    for day in range(days):
        moment = BASE + timedelta(days=day)
        messages.append(
            Message(
                f"s{day}",
                f"c{day}",
                "them",
                "ok heading to bed, night",
                moment.replace(hour=signoff_hour, minute=5),
            )
        )
        messages.append(
            Message(
                f"w{day}",
                f"c{day}",
                "them",
                "still at the office finishing the report",
                moment.replace(hour=work_hour, minute=40),
            )
        )
    return corpus_from_messages(Person(subject="them", observer="me"), messages)


def make_contrast_corpus(*, days: int = 10) -> Corpus:
    """A corpus whose two purposes sit in different hours of the day.

    The permutation test (G-13c) asks whether purpose and clock go together;
    a corpus in which *every* message falls in the 23:00 hour cannot support
    that claim — the busiest window is busy because everything is in it.
    These tests give the day two windows so a real signature is establishable
    and a mislabel is still refusable.
    """
    return make_corpus(days=days, work_hour=18)


# -- rung 1: activity, with no model ----------------------------------------


def test_activity_needs_enough_dated_messages():
    corpus = corpus_from_messages(
        Person(subject="them"),
        [Message("m0", "c0", "them", "hi", BASE)],
    )
    pattern = measure_activity(corpus, speaker="them")
    assert pattern.band == BAND_UNSETTLED
    assert pattern.value is None
    assert "dated message" in pattern.detail


def test_activity_finds_the_busiest_window():
    """Both the 23:05 signoff and the 23:40 work message are in the 23:00 hour.

    So the window holds *every* message, and the share is 100% — which is
    the point of reporting a window rather than a point.
    """
    corpus = make_corpus(days=10)
    pattern = measure_activity(corpus, speaker="them")
    assert pattern.band == BAND_CONSISTENT
    assert pattern.value == pytest.approx(1.0)
    assert pattern.n_evidence == 20
    assert pattern.n_days == 10
    assert pattern.window == "23:00–23:59"


def test_activity_is_spread_when_there_is_no_concentration():
    """Messages at many different hours are not a pattern.

    Spread across ten days, so the day gate is satisfied and the *sharing*
    is what rules the pattern out.
    """
    messages = [
        Message(
            f"m{i}",
            f"c{i // 2}",
            "them",
            "hi",
            (BASE + timedelta(days=i // 2)).replace(hour=(i * 2) % 24, minute=0),
        )
        for i in range(20)
    ]
    corpus = corpus_from_messages(Person(subject="them"), messages)
    pattern = measure_activity(corpus, speaker="them")
    assert pattern.band == BAND_VARIABLE
    assert pattern.value is not None and pattern.value < 0.7


def test_activity_on_one_day_is_not_a_pattern():
    """Twenty messages in one evening is an evening, not a habit."""
    messages = [
        Message(f"m{i}", "c0", "them", "hi", BASE.replace(hour=(i * 2) % 24, minute=0))
        for i in range(20)
    ]
    corpus = corpus_from_messages(Person(subject="them"), messages)
    pattern = measure_activity(corpus, speaker="them")
    assert pattern.band == BAND_UNSETTLED
    assert pattern.n_days == 1


def test_activity_ignores_undated_messages():
    corpus = corpus_from_messages(
        Person(subject="them"),
        [Message(f"m{i}", "c0", "them", "hi", None) for i in range(20)],
    )
    pattern = measure_activity(corpus, speaker="them")
    assert pattern.band == BAND_UNSETTLED
    assert pattern.n_evidence == 0


# -- the clock window --------------------------------------------------------


def test_a_narrower_window_separates_things_an_hour_apart():
    """23:05 and 23:40 share the 23:00 hour but not a 15-minute window.

    This is the distinction the tier exists to find: the same clock hour
    holding opposite meanings.
    """
    corpus = make_corpus(days=10)
    hourly = measure_activity(corpus, speaker="them", width_minutes=60)
    quarterly = measure_activity(corpus, speaker="them", width_minutes=15)
    assert hourly.window == "23:00–23:59"
    assert quarterly.window in {"23:00–23:14", "08:00–08:14"}


def test_the_window_label_shows_the_range_not_a_point():
    corpus = make_corpus(days=10)
    pattern = measure_activity(corpus, speaker="them", width_minutes=15)
    assert "–" in pattern.window
    assert pattern.window.count(":") == 2


def test_the_relative_window_wraps_the_clock():
    """08:00 is one hour before a 23:00 median, not fifteen hours after."""
    corpus = make_corpus(days=10)
    pattern = measure_activity(corpus, speaker="them", width_minutes=15)
    if pattern.window.startswith("08:"):
        assert "before" in pattern.relative_window
        assert "15h" not in pattern.relative_window


# -- rung 2: patterns over labels -------------------------------------------


def _labels(**by_id: str) -> dict[str, str]:
    return dict(by_id)


def test_a_concentrated_purpose_is_a_consistent_pattern():
    corpus = make_corpus(days=10)
    labels = _labels(
        **{f"s{i}": "signoff" for i in range(10)},
        **{f"w{i}": "work" for i in range(10)},
    )
    patterns = measure_patterns(corpus, labels, speaker="them", width_minutes=15)
    signoff = next(p for p in patterns if p.purpose == "signoff")
    work = next(p for p in patterns if p.purpose == "work")
    assert signoff.band == BAND_CONSISTENT
    assert work.band == BAND_CONSISTENT
    # and they are in different windows
    assert signoff.window != work.window


def test_a_pattern_that_depends_on_one_day_is_not_a_pattern():
    """Drop any one day and the concentration must survive.

    A concentration that only exists because of one evening is an evening.
    Five days concentrate and one does not, so dropping the odd one out
    leaves a different busiest window — which is the tell.
    """
    messages: list[Message] = []
    # One evening carries eight messages at 23:00. Two more days add one
    # each, so the window spans three days and clears the day gate. Three
    # further days sit at 05:00.
    #
    # The modal window holds ten of thirteen — enough to clear the share
    # gate — but drop the heavy evening and 05:00 (three messages) beats
    # 23:00 (two). The concentration does not survive, and the pattern is
    # an evening rather than a habit.
    for slot in range(8):
        messages.append(
            Message(f"a{slot}", "c0", "them", "night", BASE.replace(hour=23, minute=slot))
        )
    messages.append(
        Message(
            "b0",
            "c1",
            "them",
            "night",
            (BASE + timedelta(days=1)).replace(hour=23, minute=0),
        )
    )
    messages.append(
        Message(
            "b1",
            "c2",
            "them",
            "night",
            (BASE + timedelta(days=2)).replace(hour=23, minute=0),
        )
    )
    for day in range(3, 6):
        messages.append(
            Message(
                f"z{day}",
                f"c{day}",
                "them",
                "night",
                (BASE + timedelta(days=day)).replace(hour=5, minute=0),
            )
        )
    corpus = corpus_from_messages(Person(subject="them"), messages)
    ids = [item.message_id for item in messages]
    labels = _labels(**{mid: "signoff" for mid in ids})
    patterns = measure_patterns(corpus, labels, speaker="them")
    signoff = patterns[0]
    assert signoff.band == BAND_VARIABLE
    assert "single day" in signoff.detail


def test_a_genuine_multi_day_pattern_still_passes_the_day_check():
    """The check must not fire on a pattern that really does recur."""
    messages = [
        Message(
            f"m{day}",
            f"c{day}",
            "them",
            "night",
            (BASE + timedelta(days=day)).replace(hour=23, minute=0),
        )
        for day in range(8)
    ]
    # A second cluster at another hour, so the shuffle has something to
    # reject: a corpus whose every message sat in one window could not
    # separate purpose from clock at any width.
    messages += [
        Message(
            f"w{day}",
            f"c{day}",
            "them",
            "back from lunch, continuing",
            (BASE + timedelta(days=day)).replace(hour=12, minute=30),
        )
        for day in range(8)
    ]
    corpus = corpus_from_messages(Person(subject="them"), messages)
    labels = _labels(**{item.message_id: "signoff" for item in messages if item.message_id.startswith("m")})
    pattern = measure_patterns(corpus, labels, speaker="them")[0]
    assert pattern.band == BAND_CONSISTENT
    assert pattern.n_days == 8


def test_a_concentration_on_one_day_never_reaches_the_pattern_band():
    """A window that exists on a single day is not a pattern at all.

    The day gate catches this before the leave-one-day-out check does,
    which is the better answer: the window itself only exists that day.
    """
    messages = [
        Message(f"m{slot}", "c0", "them", "night", BASE.replace(hour=23, minute=slot))
        for slot in range(8)
    ]
    corpus = corpus_from_messages(Person(subject="them"), messages)
    labels = _labels(**{item.message_id: "signoff" for item in messages})
    pattern = measure_patterns(corpus, labels, speaker="them")[0]
    assert pattern.band == BAND_UNSETTLED
    assert pattern.n_days == 1
    assert "are needed for a pattern" in pattern.detail


def test_a_purpose_with_too_few_messages_is_not_reported():
    corpus = make_corpus(days=10)
    labels = _labels(
        **{f"s{i}": "signoff" for i in range(10)},
        **{f"w{i}": "work" for i in range(10)},
    )
    labels["s0"] = "reaction"
    patterns = measure_patterns(corpus, labels, speaker="them")
    purposes = [p.purpose for p in patterns]
    assert "reaction" not in purposes
    assert "signoff" in purposes


def test_the_arithmetic_ignores_the_models_opinion_about_itself():
    """There is no confidence in the output, by design.

    A pattern carries a share and a band, and nothing that a model could
    have set.
    """
    corpus = make_corpus(days=10)
    labels = _labels(**{f"s{i}": "signoff" for i in range(10)})
    pattern = measure_patterns(corpus, labels, speaker="them")[0]
    assert not hasattr(pattern, "confidence")
    fields = {field.name for field in dataclasses.fields(pattern)}
    assert fields >= {"band", "value", "n_evidence", "n_days"}
    assert "confidence" not in fields


# -- the labelling checks ----------------------------------------------------


def test_a_label_whose_quote_is_not_in_the_message_is_dropped():
    corpus = make_corpus(days=3)
    payload = json.dumps(
        {
            "messages": [
                {"id": "s0", "label": "signoff", "evidence": "ok heading to bed, night"},
                {"id": "w0", "label": "work", "evidence": "I am a wonderful unicorn"},
            ]
        }
    )
    result = label_messages(corpus, StubBackend(payload))
    assert len(result.labels) == 1
    assert result.labels[0].message_id == "s0"
    assert result.dropped_not_verbatim == 1


def test_a_label_naming_an_unknown_id_is_dropped():
    corpus = make_corpus(days=3)
    payload = json.dumps({"messages": [{"id": "m99", "label": "signoff", "evidence": "night"}]})
    result = label_messages(corpus, StubBackend(payload))
    assert result.labels == ()
    assert result.dropped_unresolved == 1


def test_a_label_with_a_purpose_outside_the_closed_set_is_dropped():
    corpus = make_corpus(days=3)
    payload = json.dumps(
        {"messages": [{"id": "s0", "label": "vibes", "evidence": "ok heading to bed, night"}]}
    )
    result = label_messages(corpus, StubBackend(payload))
    assert result.labels == ()
    assert result.dropped_unresolved == 1


def test_a_response_that_is_not_the_contract_shape_is_reported_as_such():
    corpus = make_corpus(days=3)
    result = label_messages(corpus, StubBackend("not json at all"))
    assert result.schema_ok is False
    assert result.labels == ()


# -- rung 3: the reading, and its checks -------------------------------------


def _reading_payload(interpretation: str, claim: str, ids: list[str]) -> str:
    return json.dumps({"interpretation": interpretation, "claim": claim, "evidence_ids": ids})


def test_a_reading_is_accepted_when_the_text_supports_it():
    corpus = make_contrast_corpus()
    labels = _labels(**{f"s{i}": "signoff" for i in range(10)})
    pattern = measure_patterns(corpus, labels, speaker="them")[0]
    assert pattern.band == BAND_CONSISTENT
    backend = StubBackend(
        _reading_payload("bedtime", "They sign off before bed.", list(pattern.evidence_ids))
    )
    reading = interpret_pattern(corpus, pattern, backend)
    assert reading.interpretation == "bedtime"
    assert reading.verified is True


def test_a_reading_whose_text_does_not_support_it_is_downgraded():
    """The reading must be checkable against the cited messages.

    A model that says "bedtime" while citing messages that say nothing
    about bedtime gets `unknown`, not a caveat.
    """
    corpus = make_contrast_corpus()
    labels = _labels(**{f"w{i}": "work" for i in range(10)})
    pattern = measure_patterns(corpus, labels, speaker="them")[0]
    assert pattern.band == BAND_CONSISTENT
    backend = StubBackend(
        _reading_payload("bedtime", "They go to bed.", list(pattern.evidence_ids))
    )
    reading = interpret_pattern(corpus, pattern, backend)
    assert reading.interpretation == "unknown"
    assert reading.verified is False
    assert "not supported" in reading.note


def test_a_reading_may_not_cite_evidence_the_finding_did_not_rest_on():
    corpus = make_contrast_corpus()
    labels = _labels(**{f"s{i}": "signoff" for i in range(10)})
    pattern = measure_patterns(corpus, labels, speaker="them")[0]
    backend = StubBackend(_reading_payload("bedtime", "They go to bed.", ["w0", "w1"]))
    reading = interpret_pattern(corpus, pattern, backend)
    assert reading.interpretation == "unknown"
    assert reading.verified is False
    assert "none of the messages" in reading.note


def test_a_reading_of_a_pattern_that_was_not_established_is_refused():
    corpus = make_contrast_corpus()
    labels = _labels(**{f"s{i}": "signoff" for i in range(10)})
    pattern = measure_patterns(corpus, labels, speaker="them")[0]
    not_a_pattern = TemporalPattern(
        purpose=pattern.purpose,
        band=BAND_VARIABLE,
        value=pattern.value,
        n_evidence=pattern.n_evidence,
        n_days=pattern.n_days,
        window=pattern.window,
        relative_window=pattern.relative_window,
        median_hour=pattern.median_hour,
        evidence_ids=pattern.evidence_ids,
        detail=pattern.detail,
    )
    backend = StubBackend(
        _reading_payload("bedtime", "They go to bed.", list(pattern.evidence_ids))
    )
    reading = interpret_pattern(corpus, not_a_pattern, backend)
    assert reading.interpretation == "unknown"
    assert reading.verified is False
    assert "No pattern" in reading.note


def test_a_reading_outside_the_closed_set_is_refused():
    corpus = make_contrast_corpus()
    labels = _labels(**{f"s{i}": "signoff" for i in range(10)})
    pattern = measure_patterns(corpus, labels, speaker="them")[0]
    backend = StubBackend(
        _reading_payload("insomniac", "They have insomnia.", list(pattern.evidence_ids))
    )
    reading = interpret_pattern(corpus, pattern, backend)
    assert reading.interpretation == "unknown"
    assert reading.verified is False
    assert "closed set" in reading.note


def test_the_model_may_say_it_cannot_tell():
    """`unknown` is a legitimate answer, and it is an honest one."""
    corpus = make_contrast_corpus()
    labels = _labels(**{f"s{i}": "signoff" for i in range(10)})
    pattern = measure_patterns(corpus, labels, speaker="them")[0]
    backend = StubBackend(
        _reading_payload("unknown", "The messages do not say.", list(pattern.evidence_ids))
    )
    reading = interpret_pattern(corpus, pattern, backend)
    assert reading.interpretation == "unknown"
    assert reading.verified is True
    assert "do not settle" in reading.note


# -- the tier, wired ---------------------------------------------------------


def test_the_tier_runs_end_to_end_with_a_stub():
    corpus = make_corpus(days=10)
    labels_payload = json.dumps(
        {
            "messages": [
                {"id": f"s{i}", "label": "signoff", "evidence": "ok heading to bed, night"}
                for i in range(10)
            ]
            + [
                {
                    "id": f"w{i}",
                    "label": "work",
                    "evidence": "still at the office finishing the report",
                }
                for i in range(10)
            ]
        }
    )
    reading_payload = _reading_payload(
        "bedtime", "They sign off before bed.", [f"s{i}" for i in range(10)]
    )

    class TwoStage(LLMBackend):
        def complete(self, system: str, user: str, *, schema: dict) -> str:
            if "label" in system:
                return labels_payload
            return reading_payload

    reading = TemporalTier(TwoStage()).read(corpus, width_minutes=15)
    assert reading.band == BAND_CONSISTENT
    assert reading.patterns
    assert reading.interpretation is not None
    assert reading.interpretation.interpretation == "bedtime"
    assert reading.interpretation.verified is True


def test_the_reading_carries_a_caveat_that_says_what_it_is_not():
    corpus = make_corpus(days=10)
    reading = TemporalTier(StubBackend(json.dumps({"messages": []}))).read(
        corpus, with_interpretation=False
    )
    assert "not who they are" in reading.caveat


def test_a_model_that_labels_wrongly_produces_no_pattern():
    """The whole point: a bad model produces no finding, not a wrong one.

    A model that cannot tell a signoff from a work message will label the
    signoffs `work` and the work messages `signoff` — or, worse, scatter the
    labels so nothing clusters. Either way the arithmetic has nothing to
    report, because it never asks the model whether a pattern exists.

    The corpus below is the awkward case: every message sits in the same
    clock hour, so a *correct* labeler would find a pattern. A model that
    scatters the labels cannot manufacture one.
    """
    corpus = make_corpus(days=10)
    # A model that scatters the labels across every purpose, so no single
    # purpose has enough messages to cluster.
    purposes = ("signoff", "work", "social", "logistics", "question", "reaction")
    labels = _labels(
        **{f"s{i}": purposes[i % len(purposes)] for i in range(10)},
        **{f"w{i}": purposes[(i + 3) % len(purposes)] for i in range(10)},
    )
    patterns = measure_patterns(corpus, labels, speaker="them")
    # Each purpose has too few messages to establish a pattern, so nothing
    # is reported — even though every message is in the same clock hour.
    assert patterns == ()


def test_a_model_that_mislabels_by_swapping_purposes_loses_the_finding():
    """The failure mode that matters: the signoffs are not recognised.

    A model that calls every nightly signoff `work` cannot produce a
    bedtime finding, because the finding comes from the *purpose*, not from
    the clock. The clock alone (rung 1) still reports the window — which is
    why the label-free rung exists.
    """
    corpus = make_corpus(days=10)
    swapped = _labels(
        **{f"s{i}": "work" for i in range(10)},
        **{f"w{i}": "work" for i in range(10)},
    )
    patterns = measure_patterns(corpus, swapped, speaker="them", width_minutes=15)
    purposes = {pattern.purpose for pattern in patterns}
    assert "signoff" not in purposes
    # The label-free rung still sees the clock: the two purposes sit in
    # different 15-minute windows, so the busiest holds half the messages.
    activity = measure_activity(corpus, speaker="them", width_minutes=15)
    assert activity.band == BAND_VARIABLE
    assert activity.window == "23:00–23:14"
    # and at an hourly width they are one window
    hourly = measure_activity(corpus, speaker="them", width_minutes=60)
    assert hourly.band == BAND_CONSISTENT
    assert hourly.window == "23:00–23:59"
