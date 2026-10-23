"""T3 — response contrast: how a person's communication changes after an event.

This is the tier that answers *"does this person get angry when insulted"* and
*"do they get annoyed when something is discussed about them"* — and the
tier a forensic use-case actually needs.

**It measures communication. It does not measure emotion.** Those are different
things and only one of them is observable in a chat log. A reply that is
shorter than baseline is a fact about the reply. "They were angry" is a fact
about the person, and no chat log settles it — a person who was angry and hid
it looks exactly like a person who was not angry.

So every output here is a statement about *what was written and when*. The
readers who want to say "they did not get angry" are making that inference
themselves, from evidence that is auditable.

Three things this tier requires that the others do not:

1. **A trigger.** The caller declares which messages were the stimulus. This
   module does not infer one: "is this an insult" is a judgement about
   meaning, and a judgement about meaning belongs to whoever reads the
   conversation. A *candidate* helper is offered and is clearly marked as
   needing confirmation.
2. **A matched baseline.** The comparison is drawn from the **same
   conversations** as the triggers, excluding the responses themselves. Time
   of day, thread length and topic are confounders that would otherwise be
   read as reaction.
3. **An equivalence test.** To say "no meaningful change" — the "doesn't get
   angry" case — the confidence interval must lie **entirely inside** a
   pre-declared margin. ``p > 0.05`` means *"no evidence of change"*, which is
   not the same claim. This is the distinction most systems get wrong, and it
   is the whole reason the absence of a finding is more expensive to establish
   than the presence of one.

**Many episodes are needed.** One calm reply to one insult is an episode, not a
disposition. The reading reports how many episodes it rests on, and a corpus
with one trigger reports that it does not know.

**The intervals are clustered, not independent (G-13a).** Messages within one
conversation are not independent draws — they share a topic, a day, and often
the argument the measures are about. Every interval here therefore resamples
**conversations** with replacement and pools the messages inside whatever it
drew, so the within-thread dependence travels with its cluster instead of
being resampled away. A corpus of one conversation cannot support a clustered
interval; the reading falls back, and says so in its notes, rather than
shipping a one-point interval as if it were 95% coverage.

**The family is multiplicity-adjusted (G-13b).** Six measures judged at 95%
each inflate the chance of one false finding to about 1-in-4 — that is also
why a null felt like a verdict. Each interval here is widened to the
family-wide level: the two-sided alpha is split evenly across the measures
that were actually computed, which makes both a false *change* and a false
*equivalence* less likely at exactly the rate the family promises. Holm on
per-measure p-values is the less conservative alternative and is the next
step if the Bonferroni widening proves too blunt; the widening was chosen
because it needs no p-value machinery and never makes an error claim easier.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from trace_traits.corpus import Corpus, Message
from trace_traits.emoji import has_emoji
from trace_traits.stats import (
    cluster_median_ratio_ci,
    cluster_proportion_difference_ci,
    iid_bootstrap_median_ratio_ci,
    probit,
)

__all__ = [
    "MEASURES",
    "Episode",
    "Margin",
    "ReactivityReading",
    "Stimulus",
    "StimulusKind",
    "candidate_stimuli",
    "reactivity",
]

#: Below this many episodes a reading states that it does not know. One
#: reaction is an episode; a disposition needs a pattern.
MIN_EPISODES = 3

#: Below this many baseline messages a comparison is not worth making.
MIN_BASELINE = 8

#: The note attached to a reading when the comparison had nothing to cluster
#: on. A one-conversation corpus cannot support a clustered interval; the
#: library falls back to the independent-message bootstrap and *states that*
#: in the notes rather than shipping an interval whose only honest peer is
#: an assumption the data does not meet.
_NO_CLUSTER_NOTE = (
    "fewer than two conversations took part in this comparison; the interval "
    "resamples messages independently, which widens nothing and assumes a "
    "within-thread independence the conversation may not have"
)

#: The family-wide error rate the measures may claim, total.
#:
#: Six measures each judged at 95% would let roughly one corpus in four
#: report a change that did not happen. The two-sided alpha is split evenly
#: across the measures that were actually computed, so the *family* keeps
#: its 95% and every interval is what that implies. (G-13b. Holm's step-down
#: would spend less width; Bonferroni was chosen because it needs no p-value
#: machinery, cannot be mis-ordered by a tie, and errs in the conservative
#: direction — the same reason the equivalence test uses the 95% interval
#: rather than the conventional 90%.)
FAMILY_ALPHA = 0.05


class StimulusKind(StrEnum):
    """What kind of event a stimulus was.

    The vocabulary is closed because the *band* vocabulary is closed, but the
    meaning is the integrator's: this module never interprets a kind. It
    measures what happened after the marked message, and nothing else.
    """

    #: A message hostile toward the subject.
    INSULT = "insult"
    #: A message that discusses the subject.
    ABOUT_THEM = "about_them"
    #: Anything else the caller wants to measure a response to.
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Stimulus:
    """A message the caller has identified as an event of interest.

    The caller decides what counts. That is not a limitation of the library —
    it is the correct division of labour, because deciding what an insult is
    requires reading the conversation, and a similarity heuristic would be
    both wrong and biased.
    """

    message_id: str
    kind: StimulusKind = StimulusKind.OTHER
    #: Which person the stimulus is about, when it is about someone other than
    #: the subject.
    about: str | None = None


@dataclass(frozen=True, slots=True)
class Margin:
    """The pre-declared bound inside which a difference is not meaningful.

    One margin per measure, because the measures are not on the same scale: a
    15-percentage-point swing in reply rate and a 15% swing in message length
    are different amounts of evidence.

    The margin is **declared before the data is examined**. That is what stops
    a large corpus from reporting a 2% difference as a finding, and it is why
    the margin lives here — in the integrator's hands — rather than in the
    library.
    """

    reply_rate: float = 0.15
    reply_latency_ratio: float = 0.30
    length_ratio: float = 0.25
    emoji_rate: float = 0.15
    question_rate: float = 0.15
    exclamation_rate: float = 0.10

    def for_measure(self, name: str) -> float:
        return float(getattr(self, name))


#: The measures taken, and what each one means.
#:
#: ``ratio`` measures are scale-free — a person who writes 20 characters and
#: one who writes 200 are comparable — so a change is reported as a ratio
#: against their own baseline rather than as a difference in characters.
MEASURES: tuple[str, ...] = (
    "reply_rate",
    "reply_latency_ratio",
    "length_ratio",
    "emoji_rate",
    "question_rate",
    "exclamation_rate",
)


@dataclass(frozen=True, slots=True)
class Episode:
    """One stimulus and the subject's response to it, if any."""

    stimulus: Stimulus
    #: The subject's reply, or ``None`` when they did not answer.
    reply: Message | None
    #: Seconds between the stimulus and the reply, when both are dated.
    latency: float | None


@dataclass(frozen=True, slots=True)
class Measure:
    """One comparison, with the margin it was judged against.

    ``verdict`` is one of:

    - ``equivalent`` — the interval lies entirely inside the margin, so there
      is no *meaningful* change. This is the interesting one: "p > 0.05" only
      says the test failed to find a difference, which is a weaker claim.
    - ``higher`` / ``lower`` — a real change, in that direction.
    - ``inconclusive`` — the evidence does not settle it either way.
    """

    name: str
    baseline: float
    response: float
    #: The difference (response − baseline) and its 95% interval.
    difference: float
    low: float
    high: float
    #: The pre-declared margin within which a difference is not meaningful.
    margin: float
    verdict: str
    n_baseline: int
    n_response: int
    #: How the numbers are read: "share", "ratio", "seconds", "characters".
    unit: str

    @property
    def meaningful(self) -> bool:
        """Whether a real change was established."""
        return self.verdict in {"higher", "lower"}

    @property
    def equivalent(self) -> bool:
        """Whether the change was shown to be negligible."""
        return self.verdict == "equivalent"

    def describe(self) -> str:
        if self.verdict == "equivalent":
            return (
                f"{self.name}: no meaningful change "
                f"({self.baseline:.2f} → {self.response:.2f} {self.unit}, "
                f"within ±{self.margin:.2f})"
            )
        if self.verdict == "inconclusive":
            return (
                f"{self.name}: not settled ({self.baseline:.2f} → {self.response:.2f} {self.unit})"
            )
        direction = "more" if self.verdict == "higher" else "less"
        return (
            f"{self.name}: {direction} "
            f"({self.baseline:.2f} → {self.response:.2f} {self.unit}, "
            f"outside ±{self.margin:.2f})"
        )


@dataclass(frozen=True, slots=True)
class ReactivityReading:
    """What a person's communication did after a marked event.

    ``band`` describes communication, never feeling. See the module docstring
    for why, and read ``caveat`` — it is the honest limit of what this says.
    """

    band: str
    #: One measure per comparison, each with its own verdict and interval.
    measures: tuple[Measure, ...]
    n_episodes: int
    n_replies: int
    n_baseline: int
    #: The share of episodes whose response moved in the dominant direction.
    consistency: float
    detail: str
    caveat: str
    #: Statistical caveats a caller must know before resting anything on the
    #: intervals above — the cluster fallback, for one. Empty when none apply.
    notes: tuple[str, ...] = ()

    @property
    def established_change(self) -> bool:
        """Whether any measure established a real change."""
        return any(measure.meaningful for measure in self.measures)

    def as_line(self) -> str:
        return (
            f"Response to marked events: {self.band:<12} "
            f"({self.n_episodes} episodes, {self.n_replies} replies)"
        )

    def as_lines(self) -> tuple[str, ...]:
        return (
            f"{self.band} — {self.n_episodes} episodes, {self.n_replies} replies, "
            f"{self.consistency:.0%} consistent",
            *(f"  {measure.describe()}" for measure in self.measures),
            f"  {self.caveat}",
            *(f"  note: {note}" for note in self.notes),
        )


# -- statistics --------------------------------------------------------------


def _wilson(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for one proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _proportion_difference_ci(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    *,
    z: float = 1.96,
) -> tuple[float, float, float]:
    """Newcombe's hybrid-score interval for ``p_b - p_a``.

    Used instead of the normal approximation because at these sample sizes the
    normal approximation produces intervals that run past 0 and 1, which is not
    a number. ``z`` carries the multiplicity-adjusted level (1.96 is only the
    95% default).
    """
    if n_a == 0 or n_b == 0:
        return 0.0, 0.0, 0.0
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    point = p_b - p_a
    l1, u1 = _wilson(successes_a, n_a, z=z)
    l2, u2 = _wilson(successes_b, n_b, z=z)
    low = point - math.sqrt((p_b - l2) ** 2 + (u1 - p_a) ** 2)
    high = point + math.sqrt((u2 - p_b) ** 2 + (p_a - l1) ** 2)
    return point, max(-1.0, low), min(1.0, high)


def _verdict(point: float, low: float, high: float, margin: float) -> str:
    """Classify a difference against a pre-declared margin.

    The order matters. ``equivalent`` is checked first because it is the
    strongest statement available: the interval lies *entirely* inside the
    margin, so the data positively supports "no meaningful change" rather than
    merely failing to find one. **That is the equivalence test (PM-007.3).**
    """
    if low >= -margin and high <= margin:
        return "equivalent"
    if low > margin:
        return "higher"
    if high < -margin:
        return "lower"
    return "inconclusive"


# -- measurement -------------------------------------------------------------


def _has_emoji_flag(text: str) -> bool:
    return has_emoji(text)


def _asks_question(text: str) -> bool:
    return "?" in text


def _exclaims(text: str) -> bool:
    return "!" in text


def _share(messages: Sequence[Message], flag: Callable[[str], bool]) -> float:
    """The share of messages matching a per-text predicate."""
    if not messages:
        return 0.0
    return sum(1 for item in messages if flag(item.text)) / len(messages)


def _by_conversation(messages: Sequence[Message]) -> dict[str, list[Message]]:
    grouped: dict[str, list[Message]] = {}
    for item in messages:
        grouped.setdefault(item.conversation_id, []).append(item)
    for thread in grouped.values():
        thread.sort(key=lambda item: (item.sent_at is None, item.sent_at or datetime.min))
    return grouped


def _split(
    corpus: Corpus, stimuli: Sequence[Stimulus]
) -> tuple[list[Episode], list[Message], set[str], dict[str, list[float]]]:
    """Split the corpus into episodes, response messages, baseline, and gaps.

    The baseline is drawn from the **same conversations** as the stimuli and
    excludes the responses themselves, so a conversation that happens to be
    short or late at night does not become a confound. Matching within
    conversation is the cheapest control available and it removes most of the
    variation that has nothing to do with the stimulus.

    The ordinary gaps are returned **per conversation** (G-13a): the cluster
    bootstrap resamples whole conversations, so the gap observations must
    carry their conversation with them.
    """
    by_id = {item.message_id: item for item in corpus.messages}
    stimulus_ids = {item.message_id for item in stimuli}
    conversations = {by_id[sid].conversation_id for sid in stimulus_ids if sid in by_id}

    threads = _by_conversation(corpus.messages)
    episodes: list[Episode] = []
    response_ids: set[str] = set()

    for stimulus in stimuli:
        message = by_id.get(stimulus.message_id)
        if message is None:
            continue
        thread = threads.get(message.conversation_id, [])
        index = next(
            (i for i, item in enumerate(thread) if item.message_id == message.message_id), None
        )
        if index is None:
            continue
        # The subject's next message in this conversation is their response to
        # the stimulus. If the other person speaks again first, the subject has
        # still not answered it, so the reply is the first *subject* message
        # after the stimulus — not merely the next message.
        reply = next(
            (item for item in thread[index + 1 :] if item.speaker == corpus.person.subject),
            None,
        )
        latency = None
        if reply is not None and reply.sent_at is not None and message.sent_at is not None:
            latency = (reply.sent_at - message.sent_at).total_seconds()
        episodes.append(Episode(stimulus=stimulus, reply=reply, latency=latency))
        if reply is not None:
            response_ids.add(reply.message_id)

    baseline_messages = [
        item
        for item in corpus.messages
        if item.speaker == corpus.person.subject
        and item.message_id not in response_ids
        and item.message_id not in stimulus_ids
        and item.conversation_id in conversations
    ]

    # The subject's reply latency to *ordinary* messages: the same measurement
    # as the stimulus responses, taken on messages that are not stimuli.
    # Keyed by conversation so the bootstrap can resample clusters.
    ordinary_gaps: dict[str, list[float]] = {}
    for conversation_id, thread in threads.items():
        for previous, current in zip(thread, thread[1:], strict=False):
            if previous.speaker == corpus.person.subject:
                continue
            if current.speaker != corpus.person.subject:
                continue
            if previous.message_id in stimulus_ids or current.message_id in response_ids:
                continue
            if previous.sent_at is None or current.sent_at is None:
                continue
            gap = (current.sent_at - previous.sent_at).total_seconds()
            if gap >= 0:
                ordinary_gaps.setdefault(conversation_id, []).append(gap)

    return episodes, baseline_messages, response_ids, ordinary_gaps


def _opportunities(corpus: Corpus, exclude: set[str]) -> dict[str, tuple[int, int]]:
    """``(conversation -> (answered, opportunities))`` for ordinary messages.

    The denominator for the baseline reply rate: every message from the other
    person that was not a stimulus. The numerator is how many of those the
    subject answered. **The denominators must match** — comparing a reply rate
    computed over one population against a rate computed over another produces
    a confident, meaningless difference.

    The counts are kept **per conversation** (G-13a) so the cluster bootstrap
    can resample whole conversations; totals are summed where a point
    estimate needs them.
    """
    per_conversation: dict[str, tuple[int, int]] = {}
    for conversation_id, thread in _by_conversation(corpus.messages).items():
        for index, item in enumerate(thread):
            if item.speaker == corpus.person.subject:
                continue
            if item.message_id in exclude:
                continue
            answered, total = per_conversation.get(conversation_id, (0, 0))
            opportunities = total + 1
            following = thread[index + 1 :]
            if following and following[0].speaker == corpus.person.subject:
                answered += 1
            per_conversation[conversation_id] = (answered, opportunities)
    return per_conversation


def reactivity(
    corpus: Corpus,
    stimuli: Sequence[Stimulus],
    *,
    margin: Margin | None = None,
) -> ReactivityReading:
    """Measure how the subject's communication changes after a stimulus.

    The returned band describes **communication**, not feeling. See the module
    docstring.
    """
    bounds = margin if margin is not None else Margin()

    episodes, baseline_messages, response_ids, ordinary_gaps = _split(corpus, stimuli)
    response_id_set: set[str] = set(response_ids)
    replies = [episode.reply for episode in episodes if episode.reply is not None]
    n_episodes = len(episodes)
    n_replies = len(replies)

    caveat = (
        "Describes what was written, not what was felt. A reply that reads as "
        "calm and a reply that hides anger look the same here, so this cannot "
        "show that anyone was or was not upset."
    )

    if n_episodes < MIN_EPISODES:
        return ReactivityReading(
            band="insufficient_evidence",
            measures=(),
            n_episodes=n_episodes,
            n_replies=n_replies,
            n_baseline=len(baseline_messages),
            consistency=0.0,
            detail=(
                f"{n_episodes} marked event{'' if n_episodes == 1 else 's'} so far. "
                "One reaction is an episode, not a pattern."
            ),
            caveat=caveat,
        )

    if len(baseline_messages) < MIN_BASELINE:
        return ReactivityReading(
            band="insufficient_evidence",
            measures=(),
            n_episodes=n_episodes,
            n_replies=n_replies,
            n_baseline=len(baseline_messages),
            consistency=0.0,
            detail=(
                f"Only {len(baseline_messages)} ordinary messages to compare against. "
                "There is not enough of their normal writing to tell a change "
                "from their habit."
            ),
            caveat=caveat,
        )

    measures: list[Measure] = []
    notes: list[str] = []

    # -- the per-conversation pools (G-13a) -------------------------------
    #
    # Every interval below resamples *conversations*, so each measure's raw
    # material is grouped by the conversation it came from. The point
    # estimates are unchanged — pooled exactly as before.
    by_message_id = {item.message_id: item for item in corpus.messages}

    episode_counts: dict[str, tuple[int, int]] = {}
    response_latencies: dict[str, list[float]] = {}
    response_lengths: dict[str, list[float]] = {}
    reply_by_conv: dict[str, list[Message]] = {}
    for episode in episodes:
        conversation_id = (
            episode.reply.conversation_id
            if episode.reply is not None
            else by_message_id[episode.stimulus.message_id].conversation_id
        )
        answered, seen = episode_counts.get(conversation_id, (0, 0))
        episode_counts[conversation_id] = (
            answered + (1 if episode.reply is not None else 0),
            seen + 1,
        )
        if episode.reply is None:
            continue
        reply_by_conv.setdefault(conversation_id, []).append(episode.reply)
        if episode.latency is not None:
            response_latencies.setdefault(conversation_id, []).append(float(episode.latency))
        response_lengths.setdefault(conversation_id, []).append(
            float(episode.reply.char_length)
        )

    baseline_by_conv: dict[str, list[Message]] = {}
    for item in baseline_messages:
        baseline_by_conv.setdefault(item.conversation_id, []).append(item)
    baseline_lengths: dict[str, list[float]] = {
        conversation_id: [float(item.char_length) for item in items]
        for conversation_id, items in baseline_by_conv.items()
    }
    opportunity_counts = _opportunities(corpus, response_id_set)

    # -- the family level (G-13b) -----------------------------------------
    #
    # Which measures are computable is decided *before* any of them is
    # computed, so the family size is a function of the corpus, not of the
    # first interval's outcome.
    flat_baseline_gaps = [gap for gaps in ordinary_gaps.values() for gap in gaps]
    flat_response_latencies = [
        latency for latencies in response_latencies.values() for latency in latencies
    ]
    flat_baseline_lengths = [
        length for lengths in baseline_lengths.values() for length in lengths
    ]
    flat_response_lengths = [
        length for lengths in response_lengths.values() for length in lengths
    ]
    total_opportunities = sum(n for _, n in opportunity_counts.values())
    total_replied = sum(answered for answered, _ in opportunity_counts.values())

    family_size = (
        (1 if total_opportunities else 0)
        + (1 if flat_baseline_gaps and flat_response_latencies else 0)
        + (1 if flat_baseline_lengths and flat_response_lengths else 0)
        + 3  # the three share measures; they survive the MIN_BASELINE gate
    )
    level = FAMILY_ALPHA / family_size

    # -- reply rate ------------------------------------------------------
    if total_opportunities:
        cluster = cluster_proportion_difference_ci(
            opportunity_counts, episode_counts, level=level
        )
        if cluster is not None:
            point, low, high = cluster
        else:
            point, low, high = _proportion_difference_ci(
                total_replied,
                total_opportunities,
                n_replies,
                n_episodes,
                z=probit(1.0 - level / 2.0),
            )
            notes.append(_NO_CLUSTER_NOTE)
        bound = bounds.reply_rate
        measures.append(
            Measure(
                name="reply rate",
                baseline=total_replied / total_opportunities,
                response=n_replies / n_episodes,
                difference=point,
                low=low,
                high=high,
                margin=bound,
                verdict=_verdict(point, low, high, bound),
                n_baseline=total_opportunities,
                n_response=n_episodes,
                unit="share",
            )
        )

    # -- reply latency ratio ---------------------------------------------
    if flat_baseline_gaps and flat_response_latencies:
        cluster = cluster_median_ratio_ci(ordinary_gaps, response_latencies, level=level)
        if cluster is not None:
            point, low, high = cluster
        else:
            point, low, high = iid_bootstrap_median_ratio_ci(
                flat_baseline_gaps, flat_response_latencies, level=level
            )
            notes.append(_NO_CLUSTER_NOTE)
        bound = bounds.reply_latency_ratio
        measures.append(
            Measure(
                name="reply latency",
                baseline=statistics.median(flat_baseline_gaps),
                response=statistics.median(flat_response_latencies),
                difference=point - 1.0,
                low=low - 1.0,
                high=high - 1.0,
                margin=bound,
                verdict=_verdict(point - 1.0, low - 1.0, high - 1.0, bound),
                n_baseline=len(flat_baseline_gaps),
                n_response=len(flat_response_latencies),
                unit="ratio",
            )
        )

    # -- length ratio ----------------------------------------------------
    if flat_baseline_lengths and flat_response_lengths:
        cluster = cluster_median_ratio_ci(baseline_lengths, response_lengths, level=level)
        if cluster is not None:
            point, low, high = cluster
        else:
            point, low, high = iid_bootstrap_median_ratio_ci(
                flat_baseline_lengths, flat_response_lengths, level=level
            )
            notes.append(_NO_CLUSTER_NOTE)
        bound = bounds.length_ratio
        measures.append(
            Measure(
                name="reply length",
                baseline=statistics.median(flat_baseline_lengths),
                response=statistics.median(flat_response_lengths),
                difference=point - 1.0,
                low=low - 1.0,
                high=high - 1.0,
                margin=bound,
                verdict=_verdict(point - 1.0, low - 1.0, high - 1.0, bound),
                n_baseline=len(flat_baseline_lengths),
                n_response=len(flat_response_lengths),
                unit="ratio",
            )
        )

    # -- rate measures ----------------------------------------------------
    for key, label, flag in (
        ("emoji_rate", "emoji use", _has_emoji_flag),
        ("question_rate", "question use", _asks_question),
        ("exclamation_rate", "exclamation use", _exclaims),
    ):
        bound = bounds.for_measure(key)
        base_rate = _share(baseline_messages, flag)
        reply_rate = _share(replies, flag)
        baseline_counts = {
            conversation_id: (
                sum(1 for item in items if flag(item.text)),
                len(items),
            )
            for conversation_id, items in baseline_by_conv.items()
        }
        response_counts = {
            conversation_id: (
                sum(1 for item in items if flag(item.text)),
                len(items),
            )
            for conversation_id, items in reply_by_conv.items()
        }
        cluster = cluster_proportion_difference_ci(baseline_counts, response_counts, level=level)
        if cluster is not None:
            point, low, high = cluster
        else:
            point, low, high = _proportion_difference_ci(
                int(round(base_rate * len(baseline_messages))),
                len(baseline_messages),
                int(round(reply_rate * len(replies))),
                len(replies),
                z=probit(1.0 - level / 2.0),
            )
            notes.append(_NO_CLUSTER_NOTE)
        measures.append(
            Measure(
                name=label,
                baseline=base_rate,
                response=reply_rate,
                difference=point,
                low=low,
                high=high,
                margin=bound,
                verdict=_verdict(point, low, high, bound),
                n_baseline=len(baseline_messages),
                n_response=len(replies),
                unit="share",
            )
        )

    band, consistency = _band(
        measures,
        episodes=episodes,
        baseline_median_length=_median(flat_baseline_lengths),
        n_replies=n_replies,
        n_episodes=n_episodes,
    )
    return ReactivityReading(
        band=band,
        measures=tuple(measures),
        n_episodes=n_episodes,
        n_replies=n_replies,
        n_baseline=len(baseline_messages),
        consistency=consistency,
        detail=_detail(band, n_episodes, n_replies),
        caveat=caveat,
        notes=tuple(dict.fromkeys(notes)),
    )


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def _band(
    measures: Sequence[Measure],
    *,
    episodes: Sequence[Episode],
    baseline_median_length: float,
    n_replies: int = 0,
    n_episodes: int = 0,
) -> tuple[str, float]:
    """Band the communication change, and measure how consistent it is.

    The bands describe communication only:

    ``stable``
        **Every** measure was equivalent to baseline within its margin — the
        absence of change was *established*, not merely unrefuted.
    ``unsettled``
        No measure established a change, but at least one was left unsettled
        by the data. The data does not settle it. **This is not "stable".**
    ``reduced``
        The replies got measurably shorter, or fewer of them.
    ``amplified``
        The replies got measurably longer, or more of them.
    ``absent``
        Most stimuli went unanswered.
    ``varied``
        The measures disagree, or the responses did not point one way.
    """
    if not measures:
        return "insufficient_evidence", 0.0

    # A subject who mostly did not answer is a finding in itself, and it is
    # checkable before any other measure: the reply rate says so directly.
    # Without this branch the "absent" band was unreachable (DM-008.3).
    if n_episodes > 0 and n_replies / n_episodes < 0.5:
        return "absent", n_replies / n_episodes

    meaningful = [measure for measure in measures if measure.meaningful]
    if not meaningful:
        # No measure established a change. That is NOT the same as having
        # established that there was no change, and the difference is the whole
        # point of the equivalence test (PM-007.3).
        #
        # `equivalent` means the interval lies entirely inside the margin —
        # positive evidence that nothing meaningful happened.
        # `inconclusive` means the interval straddles the margin — the data
        # does not settle it.
        #
        # Reporting the second as the first is the failure this tier exists to
        # avoid: a failure-to-reject null is not evidence of no change.
        # `stable` requires that **every** measure was equivalent.
        #
        # An earlier version reported "stable" when *any* one measure was
        # equivalent, and an audit found the consequence: in a corpus where
        # reply length was equivalent but reply rate, latency, emoji use,
        # question use and exclamation use were all unsettled, the band came
        # back "stable". That is the null-as-evidence failure this tier
        # exists to prevent, wearing a different name.
        #
        # The distinction is not academic. "stable" tells a caller it can
        # stop collecting; "unsettled" tells it the data has not settled it
        # and more is needed. Reporting the first when the truth is the
        # second is how a person gets told "they did not react" on the
        # strength of one lucky measure.
        if all(measure.equivalent for measure in measures):
            return "stable", 1.0
        return "unsettled", 0.0

    directions = {measure.verdict for measure in meaningful}
    if len(directions) > 1:
        # Measures disagree. A shorter reply that came faster is not one thing,
        # and calling it a single direction would be a fiction.
        return "varied", 0.5

    direction = next(iter(directions))

    consistency = _consistency(episodes, baseline_median_length, direction)
    if episodes and consistency < 0.6:
        # The veto needs episodes to disagree about. With no episodes there is
        # nothing to disagree, and reporting "varied" for a measure that
        # clearly says "higher" would be a second way of losing the signal.
        return "varied", consistency

    if direction == "lower":
        return "reduced", consistency
    return "amplified", consistency


def _consistency(
    episodes: Sequence[Episode], baseline_median_length: float, direction: str
) -> float:
    """The share of episodes that moved in the dominant direction.

    Computed over the per-episode deltas rather than the averages, because the
    averages can agree while the individual episodes disagree — and a claim
    about "how they respond" is a claim about the episodes.
    """
    if not episodes:
        return 0.0
    agreeing = 0
    considered = 0
    for episode in episodes:
        if episode.reply is None:
            # No reply. That agrees with "reduced" and disagrees with
            # "amplified", but only if there is a reply-rate measure to carry
            # it — a silent subject is a signal, not a missing datum.
            considered += 1
            if direction == "lower":
                agreeing += 1
            continue
        if baseline_median_length <= 0:
            continue
        considered += 1
        ratio = episode.reply.char_length / baseline_median_length
        if direction == "lower" and ratio < 1.0 or direction == "higher" and ratio > 1.0:
            agreeing += 1
    return agreeing / considered if considered else 0.0


def _detail(band: str, n_episodes: int, n_replies: int) -> str:
    if band == "stable":
        return f"Across {n_episodes} marked events, their writing was measurably the same as usual."
    if band == "absent":
        return f"Across {n_episodes} marked events, they replied to {n_replies}."
    if band == "reduced":
        return (
            f"Across {n_episodes} marked events, their replies were measurably "
            "smaller or shorter than usual."
        )
    if band == "amplified":
        return (
            f"Across {n_episodes} marked events, their replies were measurably "
            "larger or longer than usual."
        )
    if band == "varied":
        return f"Across {n_episodes} marked events, their replies did not move in one direction."
    if band == "unsettled":
        return (
            f"Across {n_episodes} marked events, the evidence does not settle "
            "whether their writing changed."
        )
    return "Not enough evidence."


# -- candidate detection -----------------------------------------------------

#: Patterns that mark a message as *possibly* hostile. Deliberately small and
#: crude: this is a triage list for a human to work through, not a detector.
#:
#: The list is visibly biased towards English and towards a particular
#: register. That is a reason to treat its output as candidates and not as
#: labels, and it is why the API takes stimuli from the caller rather than
#: inferring them.
_HOSTILE_PATTERNS = (
    "you're an idiot",
    "you are an idiot",
    "you're pathetic",
    "you are pathetic",
    "you're useless",
    "you are useless",
    "shut up",
    "you're wrong",
    "you are wrong",
    "you never listen",
    "you always do this",
    "what's wrong with you",
    "nobody likes you",
)


def candidate_stimuli(corpus: Corpus, *, about_names: Sequence[str] = ()) -> tuple[Stimulus, ...]:
    """Messages that *might* be insults, or might be about the subject.

    A triage helper. Every one of these needs confirming by a person before it
    is used as a stimulus, and the confidence in a stimulus is not the same as
    the confidence in the measurement that follows it.

    ``about_names`` supplies the names that count as "about the subject" — a
    message mentioning one of them is a candidate ``ABOUT_THEM`` stimulus.
    """
    names = tuple(name.casefold() for name in about_names if name)
    out: list[Stimulus] = []
    for item in corpus.messages:
        if item.speaker == corpus.person.subject:
            continue
        text = item.text.casefold()
        if any(pattern in text for pattern in _HOSTILE_PATTERNS):
            out.append(Stimulus(message_id=item.message_id, kind=StimulusKind.INSULT))
            continue
        if names and any(name in text for name in names):
            out.append(Stimulus(message_id=item.message_id, kind=StimulusKind.ABOUT_THEM))
    return tuple(out)
