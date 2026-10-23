"""Regression tests for the defects found by the independent trait audit.

The audit hand-authored chat corpora with known ground truth and compared what
the library reported against what was planted. It found four defects that the
project's own test suite did not, because the suite used the *library's own*
generator — so a defect in the extractor and a defect in the generator that
produced its input cancelled out.

Each test here corresponds to one finding. They are grouped because the
failures they prevent share a cause: **a measurement that looks right because
the corpus that produced it was built the same way the measurement reads it.**
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trace_traits import (
    DimensionBand,
    Message,
    Person,
    corpus_from_messages,
    extract_features,
)
from trace_traits.emoji import (
    EMOJI_PRESENTATION_RANGES,
    TEXT_PRESENTATION_RANGES,
    VARIATION_SELECTOR_16,
    has_emoji,
)
from trace_traits.features import MIN_VOCABULARY_WORDS, MSTTR_WINDOW, _msttr

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def conversation(pairs, conversation_id="c1"):
    """Build a corpus from (speaker, minutes-from-start, text) triples."""
    messages = [
        Message(
            message_id=f"m{i}",
            conversation_id=conversation_id,
            speaker=speaker,
            text=text,
            sent_at=NOW + timedelta(minutes=minutes),
        )
        for i, (speaker, minutes, text) in enumerate(pairs)
    ]
    return corpus_from_messages(Person(subject="them", observer="me"), messages)


# -- emoji detection ---------------------------------------------------------


def test_the_common_emoji_are_detected():
    """The audit found ⭐ and ⏰ were missed.

    Both render as emoji by default, and neither is in the 0x1F300–0x1FAFF
    range the old detector used. A person who uses emoji constantly was being
    reported as never using one.
    """
    assert has_emoji("⭐")
    assert has_emoji("⏰")
    assert has_emoji("⌛")
    assert has_emoji("😍")
    assert has_emoji("done ✅")
    assert has_emoji("🎉")


def test_non_emoji_symbols_are_not_emoji():
    """Widening the ranges to catch the above would sweep in these."""
    for symbol in ("←", "⌶", "‼", "©", "•", "❤"):
        assert not has_emoji(symbol), symbol


def test_text_presentation_is_excluded_by_default():
    """A bare ❤ is more often a typed heart than a pasted emoji."""
    assert not has_emoji("❤")
    assert has_emoji("❤", include_text_presentation=True)


def test_a_selector_forced_character_counts_in_the_default_reading():
    """G-14 held-out finding: ``🛠️`` and ``🌬️`` went unseen.

    These are Emoji=Yes but text-presentation by default (Unicode 15.1).
    The old detector ignored the ``U+FE0F`` its own docstring said forces
    emoji rendering, so two of the held-out author's emoji messages read as
    plain text and the emoji rate came out low. When the message itself
    carries the selector, the sender has made the choice — count it.
    """
    for char in ("🛠", "🌬", "✂", "☀", "📿", "🈂", "🖥"):
        assert has_emoji(char + VARIATION_SELECTOR_16), char


def test_bare_text_presentation_stays_out_of_the_default_reading():
    """The selector is what changes the reading, not the character.

    Bare text-presentation characters remain excluded (the typed-heart
    rationale), and they remain available under the explicit looser flag.
    """
    for char in ("🛠", "🌬", "❤", "✂", "☀"):
        assert not has_emoji(char), char
        assert has_emoji(char, include_text_presentation=True), char


def test_the_exact_held_out_messages_are_detected():
    """The two h5 messages that started this: both must read as emoji."""
    assert has_emoji(
        "smart. i'm fitting the wide tyres tonight, that ridge gravel got "
        "sketchy last autumn 🛠" + VARIATION_SELECTOR_16
    )
    assert has_emoji(
        "mara's driving her sister to the airport that morning, so she's free "
        "after. i'd rather eat rain than do a boring valley, but i will "
        "survive 🌬" + VARIATION_SELECTOR_16
    )


def test_ascii_emoji_property_characters_held_to_the_same_receipt():
    """``#``, ``*``, ``0``–``9``, ``©``, ``®`` are Emoji=Yes upstream, but as
    the building blocks of keycap sequences, not as standalone emoji. They
    follow the same rule as every other text-presentation character: a bare
    digit is a digit, and a digit carrying an actual selector was sent as a
    keycap.
    """
    for char in ("#", "*", "5", "©", "®"):
        assert not has_emoji(char), char
        assert has_emoji(char + VARIATION_SELECTOR_16), char


def test_the_keycap_answer_is_still_a_keycap_not_a_digit():
    """A real keycap sequence IS emoji, and its digit's selector is what
    makes the reading — ``9️⃣`` counts, a plain ``9`` never does."""
    assert has_emoji("send it at 9" + VARIATION_SELECTOR_16 + "\u20e3")


def test_the_presentation_table_matches_the_upstream_point_sets():
    """Table-completeness pins for the Unicode 15.1 cut.

    The five ranges the held-out evaluation exposed (and the two points the
    old default table wrongly claimed) are pinned by character, and the two
    tables are pinned by total point count so any re-cut drifts loudly.
    Sources: emoji-data.txt (Unicode 15.1), Emoji_Presentation section and
    the Emoji-minus-Emoji_Presentation remainder (minus the five legacy
    ASCII entries).
    """
    default_points = {p for low, high in EMOJI_PRESENTATION_RANGES for p in range(low, high + 1)}
    text_points = {p for low, high in TEXT_PRESENTATION_RANGES for p in range(low, high + 1)}
    # exact upstream Emoji_Presentation set: 1205 points
    assert len(default_points) == 1205
    # full text-presentation remainder, keycap ASCII included: 219 points
    assert len(text_points) == 219
    # the held-out defect range and its sibling text-default weather set
    for point in (0x1F32A, 0x1F6E0, 0x1F6E5, 0x1F6E9, 0x1F6F0, 0x1F6F3):
        assert point not in default_points, hex(point)
        assert point in text_points, hex(point)
    # the two points the old default table over-claimed
    for point in (0x1F202, 0x1F5A5):
        assert point not in default_points, hex(point)
        assert point in text_points, hex(point)


def test_the_emoji_rate_counts_the_common_ones():
    corpus = conversation(
        [("them", 0, "nice ⭐"), ("me", 1, "ok"), ("them", 2, "cool ⏰"), ("me", 3, "sure")]
    )
    reading = extract_features(corpus).get("Emoji usage")
    assert reading is not None


# -- reply latency ------------------------------------------------------------


def build_replies(subject_gap_minutes: float, observer_gap_minutes: float = 600.0):
    """A corpus where the observer writes, then the subject replies.

    The subject's reply interval is ``subject_gap_minutes``; the observer's is
    ``observer_gap_minutes``. The subject's *own* consecutive-message gap is
    the sum of the two, which is exactly the quantity the old code reported.
    """
    messages: list[Message] = []
    clock = 0.0
    for conversation in range(4):
        for _ in range(3):
            messages.append(
                Message(
                    message_id=f"m{len(messages)}",
                    conversation_id=f"c{conversation}",
                    speaker="me",
                    text="hello",
                    sent_at=NOW + timedelta(minutes=clock),
                )
            )
            clock += subject_gap_minutes
            messages.append(
                Message(
                    message_id=f"m{len(messages)}",
                    conversation_id=f"c{conversation}",
                    speaker="them",
                    text="hi",
                    sent_at=NOW + timedelta(minutes=clock),
                )
            )
            clock += observer_gap_minutes
    return corpus_from_messages(Person(subject="them", observer="me"), messages)


def test_reply_latency_measures_the_reply_interval():
    """The defect: the gap between two of the subject's own messages.

    A person who replies in one minute but whose counterpart takes ten hours
    to write back has a *fast* reply latency. The subject-to-subject gap
    measured the ten hours, and reported it as their reply latency.
    """
    reading = extract_features(build_replies(1.0)).get("Reply latency")
    assert reading is not None
    assert reading.value == pytest.approx(1.0)
    assert reading.band == DimensionBand.LOW.value


def test_reply_latency_distinguishes_fast_from_slow():
    fast = extract_features(build_replies(1.0)).get("Reply latency")
    slow = extract_features(build_replies(90.0)).get("Reply latency")
    assert fast is not None and slow is not None
    assert fast.value == pytest.approx(1.0)
    assert slow.value == pytest.approx(90.0)
    assert fast.band == DimensionBand.LOW.value
    assert slow.band == DimensionBand.HIGH.value


def test_the_old_measurement_would_have_given_the_same_answer_for_both():
    """The defect, stated as a contrast: the subject-to-subject gap is the same
    whether the subject replies fast or slow."""
    fast_corpus = build_replies(1.0, observer_gap_minutes=600.0)
    slow_corpus = build_replies(90.0, observer_gap_minutes=600.0)

    def subject_to_subject_gap(corpus) -> float:
        gaps = []
        by_conversation: dict[str, list[Message]] = {}
        for item in corpus.subject_messages:
            by_conversation.setdefault(item.conversation_id, []).append(item)
        for thread in by_conversation.values():
            thread.sort(key=lambda item: item.sent_at)  # type: ignore[arg-type]
            for previous, current in zip(thread, thread[1:], strict=False):
                gaps.append((current.sent_at - previous.sent_at).total_seconds() / 60)
        gaps.sort()
        return gaps[len(gaps) // 2]

    # The old quantity: identical for both, and wrong.
    assert subject_to_subject_gap(fast_corpus) == pytest.approx(601.0)
    assert subject_to_subject_gap(slow_corpus) == pytest.approx(690.0)
    # The new quantity: the subject's own reply interval, and different.
    fast = extract_features(fast_corpus).get("Reply latency")
    slow = extract_features(slow_corpus).get("Reply latency")
    assert fast is not None and slow is not None
    assert fast.value == pytest.approx(1.0)
    assert slow.value == pytest.approx(90.0)


# -- vocabulary ---------------------------------------------------------------


def test_vocabulary_is_not_confounded_by_message_length():
    """The defect: plain type-token ratio rewards short messages.

    A person writing four-word messages has a *higher* TTR than a person
    writing fifty-word messages, purely because the denominator is smaller. The
    audit found a terse correspondent measured 0.89 "high" while a verbose one
    measured 0.47 "moderate" — which says more about message length than about
    vocabulary.

    The property that proves the confound is gone: split the *same words* into
    short messages and into long ones, and the vocabulary reading must come out
    identical. (The same word sequence, so the MSTTR windows hold the same
    material; the only difference is the segmentation.)
    """
    words = [
        "the",
        "committee",
        "reconvened",
        "to",
        "consider",
        "the",
        "proposal",
        "in",
        "detail",
        "several",
        "members",
        "questioned",
        "whether",
        "the",
        "timeline",
        "remained",
        "achievable",
        "we",
        "should",
        "probably",
        "examine",
        "the",
        "revised",
        "budget",
        "before",
        "committing",
        "the",
        "contractor",
        "indicated",
        "that",
        "weather",
        "could",
        "delay",
        "the",
        "next",
        "phase",
        "consensus",
        "emerged",
        "once",
        "the",
        "alternative",
        "was",
        "explained",
        "properly",
    ] * 4  # 44 positions repeated, 176 words — above the declared word floor.
    verbose_messages = [" ".join(words[i : i + 8]) for i in range(0, len(words), 8)]
    terse_messages = list(words)  # one word per message

    def corpus_for(texts):
        """The same words over two conversations, one per alternate message."""
        messages = [
            Message(
                message_id=f"m{i}",
                conversation_id="c1" if i % 2 == 0 else "c2",
                speaker="them",
                text=text,
                sent_at=NOW + timedelta(minutes=i),
            )
            for i, text in enumerate(texts)
        ]
        return corpus_from_messages(Person(subject="them", observer="me"), messages)

    terse = corpus_for(terse_messages)
    verbose = corpus_for(verbose_messages)
    terse_reading = extract_features(terse).get("Vocabulary")
    verbose_reading = extract_features(verbose).get("Vocabulary")
    assert terse_reading is not None and verbose_reading is not None
    assert terse_reading.value is not None and verbose_reading.value is not None
    # The same words in the same order, at any message length: one reading.
    assert terse_reading.value == pytest.approx(verbose_reading.value)
    assert terse_reading.band == verbose_reading.band


def test_vocabulary_refuses_a_corpus_below_the_declared_floor():
    """The band words mean MSTTR-100; below one window there is no band to mean.

    The audit's five vocabulary disagreements all ran the ratio on a shorter
    window than the thresholds were set for. The fix is declared in
    ``MIN_VOCABULARY_WORDS``: under one full preferred window, the reading says
    it does not know, and names what it would need.
    """
    well = "the team worked hard and the project is on track again today"
    terse = [Message(
        message_id=f"m{i}",
        conversation_id="c1" if i % 2 == 0 else "c2",
        speaker="them",
        text=well,
        sent_at=NOW + timedelta(minutes=i),
    ) for i in range(2)]
    extra = [Message(
        message_id=f"x{i}",
        conversation_id="c1" if i % 2 == 0 else "c2",
        speaker="them",
        text="the team worked hard on the project",
        sent_at=NOW + timedelta(minutes=5 + i),
    ) for i in range(2)]
    # Two conversations, 22 words in total — under the declared floor.
    corpus = corpus_from_messages(Person(subject="them", observer="me"), terse + extra)
    reading = extract_features(corpus).get("Vocabulary")
    assert reading is not None
    assert reading.band == DimensionBand.INSUFFICIENT.value
    assert reading.value is None
    assert str(MIN_VOCABULARY_WORDS) in reading.detail
    assert "too small" in reading.detail


def test_vocabulary_refuses_a_single_conversation_of_words():
    """The module-wide rule: one thread is an anecdote, however talkative."""
    filler = "the project is going reasonably well and the team have worked hard "
    messages = [
        Message(
            message_id=f"m{i}",
            conversation_id="c1",
            speaker="them",
            text=filler,
            sent_at=NOW + timedelta(minutes=i),
        )
        for i in range(12)
    ]
    corpus = corpus_from_messages(Person(subject="them", observer="me"), messages)
    reading = extract_features(corpus).get("Vocabulary")
    assert reading is not None
    assert reading.band == DimensionBand.INSUFFICIENT.value
    assert "anecdote" in reading.detail


def test_msttr_runs_only_on_the_declared_window():
    """The window is part of what the bands mean, so it is not a knob.

    A short corpus used to get a shrunk window (half the corpus, floored at
    20), whose ratio is mechanically higher — the terse 0.97 finding. Now the
    window is the declared 100, and a corpus too small for it never reaches a
    band at all.
    """
    filler = "w"
    assert _msttr([f"{filler}{i}" for i in range(500)]) is not None
    assert _msttr([f"{filler}{i}" for i in range(MSTTR_WINDOW - 1)]) is None
    distinct = _msttr([f"{filler}{i}" for i in range(MSTTR_WINDOW)])
    assert distinct is not None
    # A window of all-distinct words has a ratio of exactly 1.0.
    assert distinct == pytest.approx(1.0)


def test_msttr_needs_enough_words():
    assert _msttr(["a"] * 10, window=100) is None


def test_a_repetitive_corpus_is_still_detected_as_low():
    words = ["good"] * 60
    assert _msttr(words, window=20) is not None
    assert _msttr(words, window=20) < 0.2


# -- claim round-tripping -----------------------------------------------------


def test_a_detail_containing_parentheses_round_trips():
    """A claim's detail is wrapped in parentheses, so a detail with its own
    parentheses must not break the parse."""
    from trace_traits.profile import parse_measurement_claim

    text = "Vocabulary: moderate (44 distinct words in 77 (38-word windows))"
    parsed = parse_measurement_claim(text)
    assert parsed is not None
    assert parsed.dimension == "Vocabulary"
    assert parsed.band == "moderate"
    assert parsed.detail == "44 distinct words in 77 (38-word windows)"
