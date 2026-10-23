"""Feature extraction: measurable signals, banded, never metered.

Every trait here is a **count or a ratio over stored messages**, computed
without calling a model, so the same corpus always produces the same reading.
That determinism is what makes the readings testable and what lets the
confidence machinery score them at all.

Four rules the whole module obeys:

1. **A dimension below ``MIN_SAMPLE`` says it does not know.** Absence is an
   answer, not a failure.
2. **A band is a word, never a number.** "high" is a description of what the
   evidence showed; 0.83 would invite a reader to treat a communication
   pattern as a quantity, and to extrapolate from it as though it were a
   measurement of a person.
3. **Every reading carries the sample it rests on**, so a reader cannot see a
   band without also seeing how much is behind it.
4. **Every reading carries the exact messages it was computed over.** This is
   not bookkeeping: the cited evidence is what the confidence machinery scores
   and what the verifier re-derives the band from. A reading that cited every
   message in the corpus would claim the whole corpus supports it, and a
   reading over 14 conversation openers must cite those 14 openers — not the
   165 messages that happen to be in the same corpus.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from trace_traits.corpus import Corpus, Message
from trace_traits.emoji import has_emoji

__all__ = [
    "MIN_CONVERSATIONS",
    "MIN_SAMPLE",
    "DimensionBand",
    "FeatureReading",
    "FeatureVector",
    "extract_features",
    "initiation_rate",
    "mean_message_length",
    "question_rate",
    "reply_latency",
    "response_length",
    "vocabulary_richness",
]

#: Below this many messages a reading states that it does not know. The count
#: is of *messages*, and it is checked against a conversation floor too, so a
#: burst of chatter in one thread cannot masquerade as a broad pattern.
MIN_SAMPLE = 8

#: Messages shorter than this are treated as acknowledgements ("ok", "haha").
#: They are real messages — they still count toward the sample — but averaging
#: them in produces a length figure that describes the shortest thing a person
#: can send rather than how they write.
ACK_MIN_CHARS = 4

#: A reading needs evidence from at least this many conversations. This is the
#: dimension-level counterpart to INV-1: one thread is an anecdote.
MIN_CONVERSATIONS = 2

#: Reply latency is a median over a small number of rich observations, so it
#: carries its own gate rather than borrowing the message-count one.
MIN_LATENCY_SAMPLES = 5

#: Emoji detection lives in one place, ``trace_traits.emoji``, because three
#: modules needed it and each had its own copy of the ranges — which is how
#: they drifted apart.


class DimensionBand(StrEnum):
    """The closed band vocabulary.

    Ordered weakest-first so a caller can compare bands, which is what the
    "no meters" rule needs to be checkable: the bands form a scale that is
    never rendered as one.
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    INSUFFICIENT = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class FeatureReading:
    """One measurable signal about how the subject communicated.

    ``band`` is a word. ``value`` is the underlying statistic, kept for
    auditing and for comparison between people — never rendered as the reading
    itself, and never as a confidence.
    """

    dimension: str
    band: str
    #: The statistic the band was derived from. ``None`` when there is not
    #: enough data to compute one.
    value: float | None
    n_evidence: int
    n_conversations: int
    #: A short phrase, e.g. "avg 24 chars vs your 88". Descriptive, never
    #: numeric as a confidence.
    detail: str
    #: The exact messages this reading was computed over. These are what a
    #: claim about it cites.
    evidence_ids: tuple[str, ...] = ()

    def as_line(self) -> str:
        """The rendered form: band and sample size on one line, never split."""
        return (
            f"{self.dimension:<22} {self.band:<22} "
            f"({self.n_evidence} messages, {self.n_conversations} conversations)"
        )


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Every reading for one corpus, plus the evidence that backs them."""

    readings: tuple[FeatureReading, ...]

    def get(self, dimension: str) -> FeatureReading | None:
        for reading in self.readings:
            if reading.dimension.lower() == dimension.lower():
                return reading
        return None

    @property
    def sufficient(self) -> tuple[FeatureReading, ...]:
        return tuple(r for r in self.readings if r.band != DimensionBand.INSUFFICIENT.value)

    def __iter__(self) -> Iterator[FeatureReading]:
        return iter(self.readings)


def _has_emoji(text: str) -> bool:
    """Whether the text contains an emoji, by the Unicode property."""
    return has_emoji(text)


def _substantive(messages: Sequence[Message]) -> tuple[Message, ...]:
    """Messages long enough to say something about length."""
    return tuple(item for item in messages if item.char_length >= ACK_MIN_CHARS)


def _band(value: float, *, low: float, high: float) -> str:
    if value < low:
        return DimensionBand.LOW.value
    if value > high:
        return DimensionBand.HIGH.value
    return DimensionBand.MODERATE.value


def _ids(messages: Sequence[Message]) -> tuple[str, ...]:
    return tuple(item.message_id for item in messages)


def _insufficient(dimension: str, messages: Sequence[Message], why: str) -> FeatureReading:
    return FeatureReading(
        dimension=dimension,
        band=DimensionBand.INSUFFICIENT.value,
        value=None,
        n_evidence=len(messages),
        n_conversations=len({item.conversation_id for item in messages}),
        detail=why,
        evidence_ids=(),
    )


def _gate(dimension: str, messages: Sequence[Message], why: str) -> FeatureReading | None:
    """The shared sample gate. ``None`` means the reading may proceed."""
    n_messages = len(messages)
    n_conversations = len({item.conversation_id for item in messages})
    if n_messages < MIN_SAMPLE:
        return _insufficient(
            dimension,
            messages,
            f"Not enough messages yet to say anything about {why}.",
        )
    if n_conversations < MIN_CONVERSATIONS:
        return _insufficient(
            dimension,
            messages,
            (
                f"Only {n_conversations} conversation"
                f"{'' if n_conversations == 1 else 's'} so far — that is an anecdote, "
                "not a pattern."
            ),
        )
    return None


# -- individual features ---------------------------------------------------


def mean_message_length(messages: Sequence[Message]) -> float | None:
    """Mean length of substantive messages, in characters."""
    lengths = [item.char_length for item in _substantive(messages)]
    if not lengths:
        return None
    return sum(lengths) / len(lengths)


def response_length(theirs: Sequence[Message], *, yours: Sequence[Message] = ()) -> FeatureReading:
    """How long their replies are, banded.

    Banded against the **observer's own** average when one is supplied, so
    "short" means short relative to this pairing rather than short in the
    abstract — a person who writes one-word replies should not have every
    conversation reported as long. The observer's figure is a comparison point
    in the detail text, not a score.

    With no observer baseline, the absolute bands apply. The two are never
    silently mixed: a reading records which basis produced it.
    """
    blocked = _gate("Response length", theirs, "their reply length")
    if blocked is not None:
        return blocked

    measured = _substantive(theirs)
    theirs_mean = mean_message_length(theirs)
    if theirs_mean is None:
        return _insufficient("Response length", theirs, "No messages long enough to measure yet.")

    yours_mean = mean_message_length(yours) if yours else None
    if yours_mean:
        ratio = theirs_mean / yours_mean
        band = _band(ratio, low=0.75, high=1.5)
        detail = f"avg {int(theirs_mean)} chars vs your {int(yours_mean)}"
    else:
        ratio = theirs_mean / 40.0
        band = _band(ratio, low=0.75, high=1.5)
        detail = f"avg {int(theirs_mean)} chars"

    return FeatureReading(
        dimension="Response length",
        band=band,
        value=theirs_mean,
        n_evidence=len(measured),
        n_conversations=len({item.conversation_id for item in measured}),
        detail=detail,
        evidence_ids=_ids(measured),
    )


def emoji_usage(messages: Sequence[Message]) -> FeatureReading:
    """Share of their messages containing at least one emoji, banded.

    A per-message rate rather than a per-character count: a message either
    communicates with emoji or does not.
    """
    blocked = _gate("Emoji usage", messages, "how often they use emoji")
    if blocked is not None:
        return blocked

    with_emoji = sum(1 for item in messages if _has_emoji(item.text))
    rate = with_emoji / len(messages)
    return FeatureReading(
        dimension="Emoji usage",
        band=_band(rate, low=0.15, high=0.5),
        value=rate,
        n_evidence=len(messages),
        n_conversations=len({item.conversation_id for item in messages}),
        detail=f"in {with_emoji} of {len(messages)} messages",
        evidence_ids=_ids(messages),
    )


def question_rate(messages: Sequence[Message]) -> FeatureReading:
    """How often they ask something, as a band.

    Included because it is one of the few dimensions derivable without a model
    and without collapsing into tone: whether a message contains a question is
    a property of the text.
    """
    blocked = _gate("Questions", messages, "how often they ask things")
    if blocked is not None:
        return blocked

    asking = sum(1 for item in messages if "?" in item.text)
    rate = asking / len(messages)
    return FeatureReading(
        dimension="Questions",
        band=_band(rate, low=0.08, high=0.35),
        value=rate,
        n_evidence=len(messages),
        n_conversations=len({item.conversation_id for item in messages}),
        detail=f"asked something in {asking} of {len(messages)} messages",
        evidence_ids=_ids(messages),
    )


def initiation_rate(
    messages: Sequence[Message],
    *,
    subject: str,
    all_messages: Sequence[Message] = (),
) -> FeatureReading:
    """How often they start a conversation, as a band.

    Computed over conversations, not messages: "do they open" is a property of
    a thread's first turn. ``all_messages`` supplies the full interleaving so
    a thread's opener can be identified; ``subject`` says whose opener counts.

    **The evidence is the subject's own first message in each conversation.**
    In a conversation they opened, that is the opener. In one they did not, it
    is their reply — which is still their own message, and still evidence of
    how they enter a thread somebody else began.

    This matters for more than tidiness. A claim about a person may only cite
    that person's own words (PLAN.md §2.6), so citing the *other* person's
    openers as support would break the attribution rule that stops a claim
    about someone being propped up by someone else's behaviour.
    """
    openers = _conversation_openers(all_messages) if all_messages else ()
    if not openers:
        return _insufficient(
            "Initiation",
            messages,
            "Not enough conversations yet to see who starts them.",
        )
    if len(openers) < MIN_CONVERSATIONS:
        return _insufficient(
            "Initiation",
            messages,
            f"Only {len(openers)} conversation so far — that is an anecdote, not a pattern.",
        )

    subject_openers = sum(1 for opener in openers if opener.speaker == subject)
    rate = subject_openers / len(openers)

    # The subject's own messages are what the dimension is about. If they sent
    # none, "they never open" is not a finding about them — it is an absence
    # of evidence about them, and the two must not read the same.
    own = [item for item in messages if item.speaker == subject]
    if not own:
        return _insufficient(
            "Initiation",
            messages,
            "They have not sent anything, so there is nothing to say about "
            "whether they start conversations.",
        )

    # The subject's own first message in each conversation.
    own_first = _subject_first_per_conversation(all_messages, subject=subject)
    return FeatureReading(
        dimension="Initiation",
        band=_band(rate, low=0.25, high=0.6),
        value=rate,
        n_evidence=len(own_first),
        n_conversations=len(openers),
        detail=f"opened {subject_openers} of {len(openers)} conversations",
        evidence_ids=_ids(own_first),
    )


def _conversation_openers(messages: Sequence[Message]) -> tuple[Message, ...]:
    """The first message of each conversation, in order."""
    seen: dict[str, Message] = {}
    for item in messages:
        seen.setdefault(item.conversation_id, item)
    return tuple(seen.values())


def _subject_first_per_conversation(
    messages: Sequence[Message], *, subject: str
) -> tuple[Message, ...]:
    """The subject's own first message in each conversation."""
    seen: dict[str, Message] = {}
    for item in messages:
        if item.speaker != subject:
            continue
        seen.setdefault(item.conversation_id, item)
    return tuple(seen.values())


def reply_latency(
    messages: Sequence[Message],
    *,
    all_messages: Sequence[Message] = (),
    subject: str | None = None,
) -> FeatureReading:
    """How long they take to reply, as a band.

    The interval measured is **the other person's message to this person's
    reply** — that is what "reply latency" means. It is not the gap between
    two of the subject's own messages, which is a different quantity entirely:
    a person who replies in thirty seconds but whose counterpart takes an hour
    to write back has a *fast* reply latency, and a subject-to-subject gap
    would report the hour.

    ``all_messages`` supplies the full interleaving so the other person's
    messages are visible. Without it the measurement cannot be made at all,
    and the reading says so rather than reporting the wrong interval.

    ``subject`` names who "they" are. The corpus knows exactly; it is passed
    rather than inferred, because inferring it from message counts would pick
    the wrong speaker in a corpus where the other person talks more.

    Undated messages contribute nothing rather than a guess — a message with no
    visible timestamp cannot contribute a duration.

    The evidence is the subject's own replies, which are the messages the
    duration is attributed to.
    """
    dated = [item for item in messages if item.sent_at is not None]
    gaps: list[float] = []
    supporting: list[Message] = []

    if all_messages:
        by_conversation: dict[str, list[Message]] = {}
        for item in all_messages:
            if item.sent_at is None:
                continue
            by_conversation.setdefault(item.conversation_id, []).append(item)
        for thread in by_conversation.values():
            thread.sort(key=lambda item: item.sent_at)  # type: ignore[arg-type,return-value]
            for previous, current in zip(thread, thread[1:], strict=False):
                # The reply interval: the OTHER person wrote, then THIS person
                # replied. A subject-to-subject gap is not a reply latency.
                if previous.speaker == current.speaker:
                    continue
                if subject is not None:
                    if current.speaker != subject:
                        continue
                elif current.speaker != _subject_of(thread):
                    continue
                assert previous.sent_at is not None and current.sent_at is not None
                gap = (current.sent_at - previous.sent_at).total_seconds() / 60.0
                if gap >= 0:
                    gaps.append(gap)
                    supporting.append(current)

    if len(gaps) < MIN_LATENCY_SAMPLES:
        # The gate is per-dimension. Reply latency is a *median* over
        # observations that each carry a whole interval of meaning — the time
        # from being spoken to to answering — so fewer observations support a
        # usable median than a rate over many messages does. Eight replies is
        # the convention for a stable median; the message-count gate is for
        # proportions.
        return _insufficient(
            "Reply latency",
            dated,
            (
                f"Only {len(gaps)} timed repl"
                f"{'' if len(gaps) == 1 else 'ies'} yet; "
                f"{MIN_LATENCY_SAMPLES} are needed to say how quickly they reply."
                if all_messages
                else "Reply latency needs the other person's messages to measure."
            ),
        )

    median = statistics.median(gaps)
    if median < 5:
        band = DimensionBand.LOW.value
    elif median < 60:
        band = DimensionBand.MODERATE.value
    else:
        band = DimensionBand.HIGH.value
    return FeatureReading(
        dimension="Reply latency",
        band=band,
        value=median,
        n_evidence=len(gaps),
        n_conversations=len(by_conversation),
        detail=f"median {int(median)} minutes between messages",
        evidence_ids=_ids(supporting),
    )


#: The preferred window for MSTTR, in words.
#:
#: Type-token ratio is confounded by sample length: a person who writes
#: four-word messages has a higher TTR than a person who writes fifty-word
#: messages, purely because the denominator is smaller. An audit found exactly
#: this — a terse correspondent measured 0.89 "high" while a verbose one
#: measured 0.47 "moderate", which says more about message length than about
#: vocabulary.
#:
#: MSTTR (McCarthy, 2005) fixes it by computing TTR over a sliding window and
#: averaging. 100 words is the conventional size: long enough that the ratio is
#: stable, short enough to fit most people's messages.
MSTTR_WINDOW = 100

#: The smallest word supply over which the vocabulary dimension claims a band
#: (PLAN.md G-11).
#:
#: The band thresholds below (0.45 / 0.80) mean MSTTR **at the preferred
#: window**. A short corpus used to get a shorter window — and type-token
#: ratio over a short window is mechanically higher — so a terse correspondent
#: measured 0.97 and banded ``high`` where a human reading the transcript said
#: the vocabulary was limited. Five of the audit's five vocabulary
#: disagreements were of exactly this kind, and in each the measured value was
#: right and the *scale* was wrong.
#:
#: The fix is not a moved threshold. The band words are declared to mean
#: MSTTR-at-the-preferred-window, so a band is only claimed once the corpus can
#: supply a full window at that declared size. Under it, the dimension says it
#: does not know, naming how many words it would need — in the same spirit as
#: every other sample gate in this module. The audit corpora exercised here
#: were *re-measured* against this floor after it was set from the window size
#: alone, not set so that they would pass.
MIN_VOCABULARY_WORDS = MSTTR_WINDOW


def _msttr(words: Sequence[str], window: int = MSTTR_WINDOW) -> float | None:
    """Mean type-token ratio over sliding windows of the declared size.

    Returns ``None`` when there are fewer words than one window — which the
    vocabulary reading treats as "cannot say", never as a band.
    """
    if len(words) < window:
        return None
    total = 0.0
    count = 0
    for start in range(0, len(words) - window + 1):
        chunk = words[start : start + window]
        total += len(set(chunk)) / window
        count += 1
    return total / count if count else None


def vocabulary_richness(messages: Sequence[Message]) -> FeatureReading:
    """How varied their vocabulary is, as a band — or nothing, when too small.

    Measured as **MSTTR** — the type-token ratio over a fixed 100-word
    sliding window, averaged — rather than a plain TTR over the whole corpus.
    A plain TTR is confounded by how much a person writes: short messages
    produce a high ratio for reasons that have nothing to do with vocabulary.
    MSTTR removes that confound, so the band compares people rather than
    message lengths.

    Two gates, both declared (PLAN.md G-11):

    1. At least :data:`MIN_VOCABULARY_WORDS` words. The band words mean
       MSTTR-at-the-100-word-window; a shorter corpus used to get a shorter
       window, whose ratio is mechanically high — the terse correspondent
       that measured ``high`` at 0.97, where the transcript's own words said
       the vocabulary was limited. Below one full window at the claimed
       scale the reading refuses rather than banded on a different scale.
    2. At least :data:`MIN_CONVERSATIONS` conversations, the module-wide
       "one thread is an anecdote" rule — a chatter-heavy single thread
       cannot masquerade as a vocabulary finding.

    It is still a crude measure and is banded as one — the bands describe this
    corpus's spread, not a property of the person.
    """
    words: list[str] = []
    supporting: list[Message] = []
    for item in messages:
        item_words = item.words
        if not item_words:
            continue
        words.extend(item_words)
        supporting.append(item)

    if len(words) < MIN_VOCABULARY_WORDS:
        return _insufficient(
            "Vocabulary",
            messages,
            f"Only {len(words)} words yet; bands for vocabulary are read on "
            f"{MIN_VOCABULARY_WORDS}-word windows, so this corpus is too small "
            "to say how varied their vocabulary is.",
        )
    if len({item.conversation_id for item in supporting}) < MIN_CONVERSATIONS:
        return _insufficient(
            "Vocabulary",
            messages,
            "Their words sit in one conversation so far — that is an anecdote, "
            "not a pattern.",
        )

    # Past the floor, one full window at the declared size is guaranteed.
    ratio = _msttr(words, MSTTR_WINDOW)
    assert ratio is not None, "the word floor guarantees at least one window"
    # No parentheses in the detail: the claim wraps it in "(...)", and a
    # detail containing its own parentheses would make the claim
    # ambiguous to parse back.
    detail = (
        f"{len(set(words))} distinct words in {len(words)} "
        f"across {MSTTR_WINDOW}-word windows"
    )

    return FeatureReading(
        dimension="Vocabulary",
        band=_band(ratio, low=0.45, high=0.80),
        value=ratio,
        n_evidence=len(supporting),
        n_conversations=len({item.conversation_id for item in supporting}),
        detail=detail,
        evidence_ids=_ids(supporting),
    )


def _subject_of(messages: Sequence[Message]) -> str:
    """The speaker who sends the most messages — the subject of the corpus.

    A proxy, and the caller knows the answer exactly. It is only used to
    decide which side of a gap is "the reply", so it is used together with
    the corpus's own speaker labels where those are available.
    """
    counts: dict[str, int] = {}
    for item in messages:
        counts[item.speaker] = counts.get(item.speaker, 0) + 1
    return max(counts, key=lambda key: (counts[key], key)) if counts else ""


def extract_features(corpus: Corpus) -> FeatureVector:
    """Every measurable reading for a corpus.

    The observer's own messages are the baseline when the corpus carries one,
    so "short" is relative to the pairing rather than absolute.
    """
    theirs = corpus.subject_messages
    yours = corpus.observer_messages

    readings = (
        response_length(theirs, yours=yours),
        emoji_usage(theirs),
        question_rate(theirs),
        initiation_rate(
            theirs,
            subject=corpus.person.subject,
            all_messages=corpus.messages,
        ),
        reply_latency(
            theirs,
            all_messages=corpus.messages,
            subject=corpus.person.subject,
        ),
        vocabulary_richness(theirs),
    )

    return FeatureVector(readings=readings)


def latest_dated(corpus: Corpus) -> datetime | None:
    """The newest dated message, for a freshness input.

    Timestamped rows only: a message with no visible timestamp cannot be
    placed in time, so it cannot make a persona look fresher than it is.
    """
    dated = [item.sent_at for item in corpus.messages if item.sent_at is not None]
    return max(dated) if dated else None
