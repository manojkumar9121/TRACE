"""Tests for the statistics the audit asked for (G-13), and the two
clock-time instruments it shares a habit with (G-9, G-10).

Numbers here are not thresholds — the thresholds live, declared, in the
modules. What these tests pin is the *shape* of the honest answer:

1. A clustered corpus gets clustered intervals (G-13a), and a corpus too
   small to cluster says so instead of pretending.
2. The family level is what the family promises (G-13b), and a one-window
   corpus cannot support a signature no matter how tall its share.
3. Discoveries have to survive a shuffled null, and the null is seeded, so
   the number reproduces (G-13c).
4. The zone a window is read in is a declared policy, visibly labelled
   (G-9). Weekday and silence gates are declared for the instrument they
   measure, not inherited (G-10).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from trace_traits.corpus import Message, Person, corpus_from_messages
from trace_traits.respond import Margin, Stimulus, StimulusKind, reactivity
from trace_traits.respond_synth import ReactivityPlant, reactivity_corpus
from trace_traits.stats import (
    cluster_median_ratio_ci,
    cluster_proportion_difference_ci,
    iid_bootstrap_median_ratio_ci,
    percentile_ci,
    probit,
)
from trace_traits.temporal import (
    BAND_CONSISTENT,
    BAND_UNSETTLED,
    BAND_VARIABLE,
    MIN_OBSERVATIONS,
    MIN_PERMUTATIONS,
    measure_activity,
    measure_patterns,
    measure_silences,
    measure_weekday,
)

BASE = datetime(2026, 10, 1, tzinfo=UTC)  # a Thursday


# -- stats: the normal quantile and the intervals -----------------------------


def test_probit_returns_the_standard_normal_quantile():
    assert probit(0.5) == pytest.approx(0.0, abs=1e-9)
    assert probit(0.975) == pytest.approx(1.959964, abs=1e-4)
    # Six measures at the 5% family level: two-sided alpha = 1/240.
    assert probit(1.0 - 0.05 / 12.0) == pytest.approx(2.6383, abs=1e-3)


def test_probit_refuses_a_probability_outside_the_open_interval():
    with pytest.raises(ValueError):
        probit(0.0)
    with pytest.raises(ValueError):
        probit(1.0)
    with pytest.raises(ValueError):
        probit(-0.1)


def test_the_percentile_interval_takes_order_statistics_at_the_declared_level():
    ordered = list(range(100))
    low, high = percentile_ci(ordered, level=0.05)
    assert low == 2
    assert high == 96
    assert percentile_ci((), level=0.05) == (0.0, 0.0)


def test_one_conversation_is_no_cluster_interval():
    """Resampling one conversation reproduces it exactly: no interval exists."""
    single = {"c0": [1.0, 2.0, 3.0]}
    rates = {"c0": (2, 10)}
    assert cluster_median_ratio_ci(single, {**single}, seed=1) is None
    assert cluster_proportion_difference_ci(rates, rates, seed=1) is None


def test_one_sided_data_is_no_cluster_interval():
    both = {
        "a": [1.0, 2.0],
        "b": [3.0, 4.0],
    }
    assert cluster_median_ratio_ci(both, {}, seed=1) is None
    assert cluster_proportion_difference_ci(
        {"a": (1, 2), "b": (0, 2)}, {"a": (0, 0), "b": (0, 0)}, seed=1
    ) is None


def test_the_cluster_mixture_interval_spans_the_two_clusters():
    """Two conversations with different ratios produce an interval that
    cannot shrink below the difference between them."""
    baseline = {"conv_a": [10.0] * 10, "conv_b": [40.0] * 10}
    response = {"conv_a": [20.0] * 10, "conv_b": [120.0] * 10}
    point, low, high = cluster_median_ratio_ci(baseline, response, seed=7)
    assert low < point < high
    assert high - low > 0.2
    again = cluster_median_ratio_ci(baseline, response, seed=7)
    assert (point, low, high) == again


def test_the_iid_fallback_matches_the_median_ratio():
    point, _low, _high = iid_bootstrap_median_ratio_ci([10.0, 40.0], [20.0, 120.0])
    assert point == pytest.approx(70.0 / 25.0)


def test_the_difference_interval_point_is_the_pooled_difference():
    baseline = {"a": (1, 10), "b": (0, 10)}
    response = {"a": (8, 10), "b": (7, 10)}
    point, low, high = cluster_proportion_difference_ci(baseline, response, seed=7)
    assert point == pytest.approx(0.75 - 0.05)
    assert low <= point <= high


# -- T3: what a corpus that cannot cluster says -------------------------------

NOW = datetime(2026, 4, 1, 9, 0, tzinfo=UTC)


def _one_conversation_corpus() -> tuple:
    messages = []
    minute = 0
    for index in range(8):
        messages.append(
            Message(
                f"o{index}", "c0", "me", "an everyday message of ordinary length",
                NOW + timedelta(minutes=minute),
            )
        )
        minute += 7
        messages.append(
            Message(
                f"s{index}", "c0", "them", "the subject's ordinary reply, likewise",
                NOW + timedelta(minutes=minute),
            )
        )
        minute += 3
    for episode in range(3):
        messages.append(
            Message(
                f"u{episode}", "c0", "me", "the everyday trigger from the observer",
                NOW + timedelta(minutes=minute),
            )
        )
        minute += 8
        messages.append(
            Message(
                f"p{episode}", "c0", "them", "the subject's answer to this trigger",
                NOW + timedelta(minutes=minute),
            )
        )
        minute += 2
    corpus = corpus_from_messages(Person(subject="them", observer="me"), messages)
    stimuli = [
        Stimulus(message_id=f"u{i}", kind=StimulusKind.OTHER) for i in range(3)
    ]
    return corpus, stimuli


def test_a_one_conversation_corpus_says_its_intervals_assume_independence():
    """The fallback runs, but it is stated as a fallback."""
    corpus, stimuli = _one_conversation_corpus()
    reading = reactivity(corpus, stimuli, margin=Margin(length_ratio=0.25))
    assert reading.measures, "the measures should still be computed"
    assert len(reading.notes) == 1, "the same caveat says itself once"
    assert "fewer than two conversations" in reading.notes[0]


def test_the_note_reaches_the_reading_output():
    corpus, stimuli = _one_conversation_corpus()
    reading = reactivity(corpus, stimuli, margin=Margin(length_ratio=0.25))
    lines = reading.as_lines()
    assert any(line.startswith("  note: fewer than two conversations") for line in lines)


def test_a_multi_conversation_corpus_carries_no_cluster_note():
    plant = ReactivityPlant(
        name="them", mean_length=90.0, length_sigma=18.0, emoji_rate=0.3,
        question_rate=0.2, response_rate=0.9, reply_gap_minutes=20.0,
        initiation_rate=0.4, event_rate=0.35, reactivity=0.0,
    )
    corpus, stimuli = reactivity_corpus(
        plant, n_conversations=6, messages_per_conversation=14, seed=5
    )
    reading = reactivity(corpus, stimuli)
    assert reading.notes == ()


# -- T4: the shuffled null ----------------------------------------------------


def _signature_corpus(*, days: int = 12) -> tuple:
    """Sign-offs nightly at 23:05, work nightly at 18:40 — a real signature."""
    messages = []
    for day in range(days):
        moment = BASE + timedelta(days=day)
        messages.append(
            Message(
                f"sig{day}", f"conv{day}", "them", "ok heading to bed, night",
                moment.replace(hour=23, minute=5),
            )
        )
        messages.append(
            Message(
                f"work{day}", f"conv{day}", "them",
                "wrapping up today's report before dinner",
                moment.replace(hour=18, minute=40),
            )
        )
    corpus = corpus_from_messages(Person(subject="them", observer="me"), messages)
    labels = {f"sig{day}": "signoff" for day in range(days)}
    return corpus, labels


def test_a_signature_that_beats_the_shuffle_keeps_its_band():
    corpus, labels = _signature_corpus()
    pattern = measure_patterns(corpus, labels, speaker="them")[0]
    assert pattern.band == BAND_CONSISTENT
    assert pattern.permutation_p is not None
    assert pattern.permutation_p <= 0.05


def test_the_shuffle_is_reproducible():
    corpus, labels = _signature_corpus()
    first = measure_patterns(corpus, labels, speaker="them")[0]
    second = measure_patterns(corpus, labels, speaker="them")[0]
    assert first.permutation_p == second.permutation_p


def test_a_corpus_of_one_window_cannot_separate_purpose_from_clock():
    """Every message the person wrote sits in one window.

    No relabelling can change the share, so the null is vacuous — the
    downgrade may not claim the window was "selected"; the truth is that
    this corpus cannot separate purpose from clock at this width.
    """
    messages = [
        Message(
            f"m{day}", f"c{day}", "them", "night",
            (BASE + timedelta(days=day)).replace(hour=23, minute=0),
        )
        for day in range(8)
    ]
    corpus = corpus_from_messages(Person(subject="them"), messages)
    labels = {f"m{day}": "signoff" for day in range(8)}
    pattern = measure_patterns(corpus, labels, speaker="them")[0]
    assert pattern.band == BAND_VARIABLE
    assert "not separable" in pattern.detail
    assert "selected for being busiest" not in pattern.detail


def test_a_best_window_chance_can_reach_is_downgraded():
    """Five of seven labels land in the busiest bucket of a busy pool.

    The gates all pass, and so does a random relabelling about a quarter of
    the time — the window was chosen for being the busiest, and chance does
    the rest. The band must be `variable`, with the selection wording.
    """
    messages = []
    labels: dict[str, str] = {}
    # Five days, three 09:00 messages each; one per day labelled signoff.
    for day in range(5):
        moment = BASE + timedelta(days=day)
        for extra in range(3):
            message_id = f"m{day}{extra}"
            messages.append(
                Message(message_id, f"c{day}{extra}", "them", "morning note",
                        moment.replace(hour=9, minute=0))
            )
            if extra == 0:
                labels[message_id] = "signoff"
    for day in range(5, 10):
        messages.append(
            Message(f"b{day}", f"c{day}", "them", "midday errand",
                    (BASE + timedelta(days=day)).replace(hour=12, minute=40))
        )
    labels["b5"] = "signoff"
    messages.append(
        Message("c17", "c17", "them", "evening wrap-up",
                (BASE + timedelta(days=10)).replace(hour=17, minute=0))
    )
    labels["c17"] = "signoff"
    corpus = corpus_from_messages(Person(subject="them"), messages)
    pattern = measure_patterns(corpus, labels, speaker="them")[0]
    assert pattern.band == BAND_VARIABLE
    assert "does not beat" in pattern.detail
    assert "selected for" in pattern.detail


def test_fewer_than_the_minimum_permutations_is_refused():
    corpus, labels = _signature_corpus()
    with pytest.raises(ValueError):
        measure_patterns(corpus, labels, speaker="them", permutations=10)


def test_the_minimum_permutation_count_is_accepted():
    corpus, labels = _signature_corpus()
    pattern = measure_patterns(corpus, labels, speaker="them", permutations=MIN_PERMUTATIONS)[0]
    assert pattern.band == BAND_CONSISTENT


# -- T4, rung 1: days inside the window ---------------------------------------


def test_activity_a_habit_needs_the_days_inside_the_window():
    """A tall share carried by a single day is an evening, not a habit."""
    messages = [
        Message(
            f"m{index}", f"c{index}", "them", "night", BASE.replace(hour=23, minute=15)
        )
        for index in range(16)
    ]
    for index in range(3):
        day = BASE + timedelta(days=index + 1)
        messages.append(
            Message(
                f"x{index}", f"c{index}", "them", "an afternoon thought",
                day.replace(hour=9 + index, minute=0),
            )
        )
        messages.append(
            Message(
                f"y{index}", f"c{index}", "them", "another afternoon thought",
                day.replace(hour=13 + index, minute=0),
            )
        )
    corpus = corpus_from_messages(Person(subject="them"), messages)
    pattern = measure_activity(corpus, speaker="them")
    assert pattern.band == BAND_UNSETTLED
    assert "days inside the window" in pattern.detail
    assert pattern.value is None


def test_activity_concentration_resting_on_one_day_is_variable():
    """The in-window day gate passes; the leave-one-day-out check must fail."""
    messages = []
    for index in range(9):
        messages.append(
            Message(
                f"b{index}", f"c{index}", "them", "night",
                (BASE + timedelta(days=1)).replace(hour=23, minute=20),
            )
        )
    for index in range(3):
        messages.append(
            Message(
                f"d{index}", f"d{index}", "them", "night",
                (BASE + timedelta(days=2)).replace(hour=23, minute=45),
            )
        )
    for index in range(2):
        messages.append(
            Message(
                f"f{index}", f"f{index}", "them", "night",
                (BASE + timedelta(days=3)).replace(hour=23, minute=50),
            )
        )
    for index in range(6):
        messages.append(
            Message(
                f"z{index}", f"z{index}", "them", "evening, after work",
                (BASE + timedelta(days=0)).replace(hour=22, minute=10),
            )
        )
    corpus = corpus_from_messages(Person(subject="them"), messages)
    pattern = measure_activity(corpus, speaker="them")
    assert pattern.band == BAND_VARIABLE
    assert "single day" in pattern.detail


# -- G-9: the declared zone ---------------------------------------------------


def _dated_messages(tzinfo_aware: bool) -> list[Message]:
    stamps = []
    for day in range(MIN_OBSERVATIONS):
        moment = datetime(2026, 6, 1, 23, 5) + timedelta(days=day)
        stamps.append(
            Message(
                f"n{day}", f"c{day}", "them", "night",
                moment if not tzinfo_aware else moment.replace(tzinfo=UTC),
            )
        )
    return stamps


def test_the_window_is_labelled_as_stored_when_no_zone_is_declared():
    corpus = corpus_from_messages(Person(subject="them"), _dated_messages(tzinfo_aware=True))
    pattern = measure_activity(corpus, speaker="them")
    assert pattern.window_basis == "as-stored"
    assert pattern.window == "23:00–23:59"


def test_a_naive_timestamp_without_a_declared_zone_is_used_as_stored():
    corpus = corpus_from_messages(Person(subject="them"), _dated_messages(tzinfo_aware=False))
    pattern = measure_activity(corpus, speaker="them")
    assert pattern.window_basis == "as-stored"
    assert pattern.window == "23:00–23:59"


def test_a_declared_zone_shifts_the_window_and_is_labelled():
    """A naive 23:05 local (+02:00) read in UTC is 21:05, and says so."""
    corpus = corpus_from_messages(Person(subject="them"), _dated_messages(tzinfo_aware=False))
    pattern = measure_activity(
        corpus,
        speaker="them",
        assume_tz=timezone(timedelta(hours=2)),
        reference=UTC,
    )
    assert pattern.window_basis == "declared-zone"
    assert pattern.window == "21:00–21:59"


# -- G-10: weekdays -----------------------------------------------------------

FRIDAYS = (datetime(2026, 10, 2), datetime(2026, 10, 9), datetime(2026, 10, 16), datetime(2026, 10, 23))
MONDAYS = (datetime(2026, 10, 5), datetime(2026, 10, 12), datetime(2026, 10, 19))
TUESDAYS = (datetime(2026, 10, 6), datetime(2026, 10, 13))
WEDNESDAYS = (datetime(2026, 10, 7), datetime(2026, 10, 14))
THURSDAYS = (datetime(2026, 10, 8),)


def _weekday_messages(dates: list[datetime], *, hour: int = 10) -> list[Message]:
    """One message per date, at ``hour`` (offset by a minute each)."""
    rebuilt: list[Message] = []
    for index, date in enumerate(dates):
        rebuilt.append(
            Message(
                f"d{index}", f"c{index}", "them", "a message that day",
                date.replace(hour=hour, minute=index % 60),
            )
        )
    return rebuilt


def test_a_weekday_habit_needs_the_share_and_enough_dates():
    dates = list(FRIDAYS) + list(MONDAYS[:2]) + list(TUESDAYS) + list(WEDNESDAYS)
    corpus = corpus_from_messages(Person(subject="them"), _weekday_messages(dates))
    pattern = measure_weekday(corpus, speaker="them")
    assert pattern.weekday == "Friday"
    assert pattern.band == BAND_CONSISTENT
    assert pattern.value == pytest.approx(4 / 10)
    assert pattern.n_evidence == 10
    assert pattern.n_weeks == 4
    assert pattern.is_pattern is True


def test_a_weekday_with_too_few_dates_does_not_reach_consistent():
    """Three Friday messages over only two dates are not three Fridays."""
    dates = [FRIDAYS[1], FRIDAYS[1].replace(hour=11), FRIDAYS[2]]
    dates += list(MONDAYS[:2]) + list(TUESDAYS) + [WEDNESDAYS[0]]
    corpus = corpus_from_messages(Person(subject="them"), _weekday_messages(dates))
    pattern = measure_weekday(corpus, speaker="them")
    assert pattern.weekday == "Friday"
    assert pattern.band == BAND_UNSETTLED
    assert "2 date(s)" in pattern.detail


def test_tied_weekdays_after_dropping_one_date_are_variable():
    """Four Fridays, three Mondays: losing a Friday contests the crown."""
    dates = list(FRIDAYS) + list(MONDAYS)
    corpus = corpus_from_messages(Person(subject="them"), _weekday_messages(dates))
    pattern = measure_weekday(corpus, speaker="them")
    assert pattern.weekday == "Friday"
    assert pattern.band == BAND_VARIABLE
    assert "depends on" in pattern.detail


def test_a_modest_weekday_base_rate_is_not_a_habit():
    dates = list(FRIDAYS[:3]) + list(MONDAYS[:2]) + list(TUESDAYS) + list(WEDNESDAYS) + list(THURSDAYS)
    corpus = corpus_from_messages(Person(subject="them"), _weekday_messages(dates))
    pattern = measure_weekday(corpus, speaker="them")
    assert pattern.band == BAND_VARIABLE
    assert "below the" in pattern.detail


def test_too_few_messages_cannot_even_ask_the_weekday_question():
    corpus = corpus_from_messages(
        Person(subject="them"), _weekday_messages(list(FRIDAYS[:3]), hour=12)
    )
    assert len(FRIDAYS[:3]) < 6
    pattern = measure_weekday(corpus, speaker="them", )
    assert pattern.band == BAND_UNSETTLED
    assert pattern.value is None


# -- G-10: silences -----------------------------------------------------------


def _gapped_corpus() -> tuple:
    start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    messages = [
        Message("a1", "c0", "them", "a note", start),
        Message("a2", "c0", "them", "another note", start + timedelta(days=1)),
        # the nine-day gap
        Message("a3", "c0", "them", "back", start + timedelta(days=10)),
        Message("a4", "c0", "them", "and again", start + timedelta(days=11)),
    ]
    corpus = corpus_from_messages(Person(subject="them"), messages)
    return corpus, start


def test_a_long_gap_between_two_messages_is_a_silence():
    corpus, _start = _gapped_corpus()
    reading = measure_silences(corpus)
    assert len(reading.intervals) == 1
    gap = reading.intervals[0]
    assert gap.days == pytest.approx(9.0)
    assert gap.start_message_id == "a2"
    assert gap.end_message_id == "a3"
    assert gap.n_messages_during == 0
    assert reading.longest is gap
    assert reading.threshold_days == 3.0


def test_what_happened_during_the_gap_is_counted():
    start = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
    messages = [
        Message("a1", "c0", "them", "a note", start),
        Message("a2", "c0", "them", "another note", start + timedelta(days=1)),
        Message("l1", "c0", "me", "the other side writes once", start + timedelta(days=5)),
        Message("l2", "c0", "me", "and again", start + timedelta(days=6)),
        Message("a3", "c0", "them", "back", start + timedelta(days=10)),
    ]
    corpus = corpus_from_messages(Person(subject="them"), messages)
    reading = measure_silences(corpus)
    assert len(reading.intervals) == 1
    assert reading.intervals[0].n_messages_during == 2


def test_the_threshold_is_the_caller_s_knob():
    corpus, _start = _gapped_corpus()
    reading = measure_silences(corpus, min_gap_days=10.0)
    assert reading.intervals == ()
    assert reading.threshold_days == 10.0
    assert "No silence" in reading.as_lines()[0]


def test_one_dated_message_has_no_silences():
    corpus = corpus_from_messages(
        Person(subject="them"),
        [Message("a1", "c0", "them", "a note", BASE)],
    )
    reading = measure_silences(corpus)
    assert reading.intervals == ()
    assert "not a mood" in reading.caveat


def test_the_silence_reading_claims_no_meaning():
    corpus, _start = _gapped_corpus()
    reading = measure_silences(corpus)
    assert "not a mood" in reading.caveat
    lines = reading.as_lines()
    assert any("not a mood" in line for line in lines)
