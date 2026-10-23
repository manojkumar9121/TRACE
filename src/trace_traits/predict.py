"""Prediction: what this person is likely to do next.

**This is not a claim about a person.** It is a statement about a
*distribution over their next message*, derived from how they have written
before. The difference matters and the output is shaped to keep it: every
prediction carries a probability, a basis in the evidence, and an explicit
statement of what would have to be true for it to land.

Three kinds of prediction are supported, in increasing order of how much they
claim:

``likely_length``
    How long their next message is likely to be. A distribution over a
    length, derived from their own length distribution.

``likely_emoji``
    Whether their next message is likely to contain an emoji.

``likely_responds``
    Whether they are likely to reply at all, and how quickly — derived from
    their observed response rate and latency.

Every prediction is *evaluated* by :mod:`trace_traits.evaluate` against a
held-out corpus, and the evaluation is what says whether any of it is worth
believing. A predictor that is not measured is a guess with extra steps.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from trace_traits.corpus import Corpus, Message
from trace_traits.emoji import has_emoji
from trace_traits.features import ACK_MIN_CHARS

__all__ = [
    "ContextualReplyPredictor",
    "ContextualSkill",
    "Prediction",
    "ReplyContext",
    "ReplyPredictor",
    "build_contextual_predictor",
    "build_predictor",
]


@dataclass(frozen=True, slots=True)
class Prediction:
    """One prediction, with the evidence it rests on and its limits."""

    kind: str
    #: A short label for the predicted quantity, e.g. "reply length".
    label: str
    #: The point prediction. For a probability this is the probability; for a
    #: length it is the expected characters.
    value: float
    #: The interval the prediction is actually asserting, as a low/high pair.
    #: A prediction without an interval is a point estimate wearing a
    #: prediction's clothes.
    low: float
    high: float
    #: How many observations the estimate rests on.
    n_evidence: int
    #: What would have to be true for this to be right.
    caveat: str

    @property
    def unit(self) -> str:
        if self.kind == "likely_emoji":
            return "probability"
        if self.kind == "likely_responds":
            return "probability"
        if self.kind == "reply_latency_minutes":
            return "minutes"
        return "characters"

    def describe(self) -> str:
        return (
            f"{self.label}: {self.value:.2f}{_unit_suffix(self.unit)} "
            f"[{self.low:.2f}, {self.high:.2f}]  "
            f"({self.n_evidence} observations; {self.caveat})"
        )


def _unit_suffix(unit: str) -> str:
    return {"probability": "", "characters": " chars", "minutes": " min"}.get(unit, "")


class ReplyPredictor:
    """Predicts a person's next message from their own history.

    The predictor is deliberately simple and inspectable: every prediction is a
    summary statistic over the person's own messages, and every one carries the
    count it rests on. There is no model, no embedding, and no fitting — which
    means a prediction can be checked by hand, and a bad corpus produces a bad
    prediction rather than a confidently wrong one.
    """

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus
        self._theirs = corpus.subject_messages
        self._lengths = [
            item.char_length for item in self._theirs if item.char_length >= ACK_MIN_CHARS
        ]
        self._emoji = [1 if _has_emoji(item.text) else 0 for item in self._theirs]
        self._latencies = _latencies(self._theirs)

    # -- predictions -------------------------------------------------------

    def likely_length(self) -> Prediction:
        """How long their next message is likely to be.

        The median of their own substantive messages, with an interval from
        the interquartile range. The median rather than the mean because a
        distribution of message lengths is right-skewed and one very long
        message should not drag the estimate of "typical".
        """
        if len(self._lengths) < 3:
            return Prediction(
                kind="likely_length",
                label="Reply length",
                value=0.0,
                low=0.0,
                high=0.0,
                n_evidence=len(self._lengths),
                caveat="not enough messages to predict from",
            )
        median = statistics.median(self._lengths)
        low, high = _quartiles(self._lengths)
        caveat = "their own typical length; a long day or a short reply would fall outside it"
        if _is_degenerate(self._lengths):
            caveat = (
                "every message seen so far was this length, so the range is a "
                "narrow placeholder rather than a measured spread"
            )
        return Prediction(
            kind="likely_length",
            label="Reply length",
            value=float(median),
            low=float(low),
            high=float(high),
            n_evidence=len(self._lengths),
            caveat=caveat,
        )

    def likely_emoji(self) -> Prediction:
        """Whether their next message is likely to contain an emoji.

        A Beta-smoothed rate: the observed proportion, pulled toward 0.5 by
        half an observation either way. Smoothing matters here because an
        unsmoothed rate from six messages is either 0.0 or 1.0, and both are
        confident nonsense.
        """
        n = len(self._emoji)
        if n == 0:
            return Prediction(
                kind="likely_emoji",
                label="Uses emoji",
                value=0.5,
                low=0.0,
                high=1.0,
                n_evidence=0,
                caveat="no messages to learn from",
            )
        successes = sum(self._emoji)
        alpha = 1.0
        beta = 1.0
        posterior = (successes + alpha) / (n + alpha + beta)
        low, high = _wilson_interval(successes, n)
        return Prediction(
            kind="likely_emoji",
            label="Uses emoji",
            value=posterior,
            low=low,
            high=high,
            n_evidence=n,
            caveat="their own rate across past messages, not a personality trait",
        )

    def likely_responds(self) -> Prediction:
        """Whether they are likely to reply, and how quickly.

        The response rate is the share of the other person's messages that
        they answered. It is the most useful signal in the whole predictor and
        the easiest to get wrong: a person who leaves many messages unanswered
        will look identical to someone who replied, if the comparison is made
        against the wrong denominator.
        """
        opportunities = _opportunities(self._corpus.messages, self._corpus.person.subject)
        if opportunities == 0:
            return Prediction(
                kind="likely_responds",
                label="Replies",
                value=0.5,
                low=0.0,
                high=1.0,
                n_evidence=0,
                caveat="no one has written to them in this corpus",
            )
        replied = _replied_count(self._corpus.messages, self._corpus.person.subject)
        # Beta-smoothed: an unsmoothed rate from two opportunities is either
        # 0.0 or 1.0, and both are confident nonsense (PM-005.3).
        rate = (replied + 1.0) / (opportunities + 2.0)
        low, high = _wilson_interval(replied, opportunities)
        return Prediction(
            kind="likely_responds",
            label="Replies",
            value=rate,
            low=low,
            high=high,
            n_evidence=opportunities,
            caveat="share of the other's messages they answered",
        )

    def reply_latency_minutes(self) -> Prediction:
        """How long they typically take to reply."""
        if len(self._latencies) < 3:
            return Prediction(
                kind="reply_latency_minutes",
                label="Reply latency",
                value=0.0,
                low=0.0,
                high=0.0,
                n_evidence=len(self._latencies),
                caveat="not enough timed replies to predict from",
            )
        median = statistics.median(self._latencies)
        low, high = _quartiles(self._latencies)
        caveat = "their own median gap between messages"
        if _is_degenerate(self._latencies):
            caveat = (
                "every timed reply so far took exactly this long, so the range "
                "is a narrow placeholder rather than a measured spread"
            )
        return Prediction(
            kind="reply_latency_minutes",
            label="Reply latency",
            value=float(median),
            low=float(low),
            high=float(high),
            n_evidence=len(self._latencies),
            caveat=caveat,
        )

    def all_predictions(self) -> tuple[Prediction, ...]:
        return (
            self.likely_length(),
            self.likely_emoji(),
            self.likely_responds(),
            self.reply_latency_minutes(),
        )


def build_predictor(corpus: Corpus) -> ReplyPredictor:
    """A predictor over a corpus."""
    return ReplyPredictor(corpus)


# -- the contextual predictor -------------------------------------------------
#
# The base-rate predictor above reports *the person's rate with an interval*,
# not a prediction with skill (PLAN.md G-12): the same number is returned
# whatever the next message looks like. These conditions — the other person's
# last message, the time of day, and the thread position — are the smallest
# conditioning that changes that. Each feature is inspectable, the table is
# reported with its counts, and thin cells shrink to the base rate rather than
# inventing a rate from one observation.


#: Shrinkage strength, in pseudo-observations.
#:
#: A single conversation primes a cell with one 👍 whose rate is 1.0; without
#: shrinkage, one lucky reply would move a cell's prediction by the full
#: distance between it and the base rate. A cell is therefore smoothed toward
#: the base rate with the weight of this many observations of it, and thin
#: cells lean toward what the person does overall rather than what happened
#: once. Smoothing is a term whose value is declared, and a knob the caller
#: may set; it may not be tuned against an audit corpus (same house rule as
#: every threshold).
DEFAULT_SHRINKAGE = 8.0

#: Half-width, in clock hours, of the subject's own reply window.
#:
#: The window centres on the median clock hour of the subject's own replies —
#: *their* habitual hour — and no zone is assumed: hours are read as stored, so
#: a corpus of naive timestamps is one clock and a corpus of aware timestamps
#: is another (G-9). A half-day-wide window would not discriminate anything.
WINDOW_HALF_HOURS = 3.0

#: How many observations a cell needs before its caveat says it is informed.
MIN_INFORMED_CELL = 8


@dataclass(frozen=True, slots=True)
class ReplyContext:
    """The situation a would-be reply would land in.

    Three features, and no thresholds:

    ``question``
        The other person's last message asked something — a "?" in its text
        is the whole test. Two people who reply at the same overall rate
        often differ enormously by whether they were asked anything.
    ``opening``
        The other person's message opened the conversation it is in. An
        opening is answered by a different person-behaviour than a
        continuation in most histories.
    ``in_window``
        The message arrived within :data:`WINDOW_HALF_HOURS` clock hours of
        the subject's own median reply hour.
    """

    question: bool
    opening: bool
    in_window: bool

    def describe(self) -> str:
        """A compact, human-readable description of the context."""
        return "+".join(
            label
            for flag, label in (
                (self.question, "question"),
                (self.opening, "opening"),
                (self.in_window, "in-window"),
            )
            if flag
        ) or "background"


@dataclass(frozen=True, slots=True)
class ContextualSkill:
    """The skill the context actually adds, measured by leaving one out.

    For every observed reply opportunity, the predictions of the contextual
    predictor and of the plain base-rate predictor are each rebuilt *without*
    that opportunity's own vote, and both are scored by Brier against what
    happened. Skill is ``1 - contextual/base``: zero says the context added
    nothing this corpus measures, negative says it hurt. It is a corpus-local
    measurement — the honest statement about a *particular* person's history,
    not a claim that the conditioning helps everywhere.
    """

    #: Reply opportunities scored.
    n: int
    #: Brier score of the contextual predictions.
    brier_contextual: float
    #: Brier score of the base-rate predictions on the same outcomes.
    brier_base: float
    #: ``1 - contextual/base``; 0.0 when it could not be measured.
    skill: float
    #: What a zero skill means when there was nothing to score against.
    note: str = ""

    def as_lines(self) -> tuple[str, ...]:
        if self.n == 0:
            return (self.note or "no reply opportunities to score",)
        lines = [
            f"contextual reply skill (leave-one-out): {self.skill:+.3f} "
            f"over {self.n} opportunities",
            f"  Brier contextual vs base rate: {self.brier_contextual:.4f} "
            f"vs {self.brier_base:.4f}",
        ]
        if self.note:
            lines.append(f"  note: {self.note}")
        return tuple(lines)


class ContextualReplyPredictor:
    """Predicts a reply conditioned on the situation it would land in.

    Built from the corpus's own reply opportunities (PLAN.md R-f's one-per-turn
    definition). Every opportunity is tagged with a :class:`ReplyContext`, and
    the predictor keeps the reply counts per cell of that context. A prediction
    for a fresh context is the cell's observed rate **shrunk toward the base
    rate** — a cell with two observations cannot command their 0-or-1 — and the
    prediction always names which cell produced it and how many observations
    that cell holds.

    Hours are read as stored, with no zone assumed (G-9): the window is *this
    person's* habitual reply hour on whatever clock the corpus carries, not a
    wall-clock claim about their life.
    """

    def __init__(
        self,
        corpus: Corpus,
        *,
        shrinkage: float = DEFAULT_SHRINKAGE,
        window_half_hours: float = WINDOW_HALF_HOURS,
    ) -> None:
        if shrinkage < 0:
            raise ValueError(f"shrinkage must be >= 0, got {shrinkage}")
        self._corpus = corpus
        self._shrinkage = shrinkage
        self._window_half_hours = window_half_hours
        self._pairs = _opportunity_pairs(corpus)
        self._reply_hours = _reply_hours(corpus)
        self._median_reply_hour = (
            statistics.median(self._reply_hours) if self._reply_hours else None
        )
        self._opening_ids = frozenset(
            item.message_id
            for conversation in _by_conversation(corpus.messages).values()
            for item in conversation[:1]
        )
        cells: dict[ReplyContext, list[int]] = {}
        for message, replied in self._pairs:
            context = self._context_of(message)
            cells.setdefault(context, []).append(1 if replied else 0)
        self._cells: dict[ReplyContext, tuple[int, int]] = {
            context: (len(votes), sum(votes)) for context, votes in cells.items()
        }

    # -- reading the table -------------------------------------------------

    @property
    def n_opportunities(self) -> int:
        """How many reply opportunities the table was built from."""
        return len(self._pairs)

    def cell(self, context: ReplyContext) -> tuple[int, int]:
        """The (observations, replies) count for one context cell.

        A context never observed maps to ``(0, 0)`` — an honest zero, not an
        invented rate.
        """
        return self._cells.get(context, (0, 0))

    def base_cell(self) -> tuple[int, int]:
        """The (observations, replies) counts over every opportunity."""
        n = sum(count for count, _ in self._cells.values())
        replied = sum(value for _, value in self._cells.values())
        return n, replied

    def context_for(self, message: Message) -> ReplyContext:
        """The context a would-be reply to ``message`` would land in.

        ``question`` and ``in_window`` come from the message itself; ``opening``
        comes from whether it opens its conversation in this corpus. A message
        from outside the corpus is context-read by its text and clock, as a
        live caller would do — and it is never an opening, because nothing in
        this corpus says it is.
        """
        return self._context_of(message)

    # -- the prediction ------------------------------------------------------

    def base_rate(self) -> Prediction:
        """The reply probability without conditioning — the fallback, named."""
        n, replied = self.base_cell()
        if n == 0:
            return Prediction(
                kind="likely_responds",
                label="Replies (base rate)",
                value=0.5,
                low=0.0,
                high=1.0,
                n_evidence=0,
                caveat="no reply opportunities in this corpus",
            )
        value = (replied + 1.0) / (n + 2.0)
        low, high = _wilson_interval(replied, n)
        return Prediction(
            kind="likely_responds",
            label="Replies (base rate)",
            value=value,
            low=low,
            high=high,
            n_evidence=n,
            caveat="their own reply share across every opportunity",
        )

    def likely_responds(self, context: ReplyContext) -> Prediction:
        """The reply probability for one context.

        The value is the cell's rate shrunk toward the base rate: full cell
        weight when the cell holds many observations, almost entirely the base
        rate when it holds none. The interval covers **both** the cell's own
        interval and the base rate's, because the number is a blend and an
        interval narrower than either of its ingredients would be a
        measurement the blend did not make.
        """
        n, replied = self.base_cell()
        cell_n, cell_replied = self.cell(context)
        if n == 0:
            return Prediction(
                kind="likely_responds",
                label="Replies",
                value=0.5,
                low=0.0,
                high=1.0,
                n_evidence=0,
                caveat="no reply opportunities in this corpus",
            )
        base = (replied + 1.0) / (n + 2.0)
        weight = self._shrinkage
        value = (cell_replied + weight * base) / (cell_n + weight)
        base_low, base_high = _wilson_interval(replied, n)
        if cell_n == 0:
            low, high = base_low, base_high
        else:
            cell_low, cell_high = _wilson_interval(cell_replied, cell_n)
            low, high = min(base_low, cell_low), max(base_high, cell_high)
        # The point is a blend of the cell and the base rate; neither
        # interval alone is claimed to hold it.
        value = min(high, max(low, value))
        base_rate = self.base_rate()
        if cell_n >= MIN_INFORMED_CELL:
            caveat = (
                f"{context.describe()}: {cell_n} observed, {cell_replied} replied; "
                f"shrunk toward their reply rate of {base_rate.value:.2f}"
            )
        elif cell_n == 0:
            caveat = (
                f"unobserved context ({context.describe()}); "
                f"their reply rate of {base_rate.value:.2f} is the whole story"
            )
        else:
            caveat = (
                f"thin context ({cell_n} of {n} opportunities, {context.describe()}), "
                f"leaning on their reply rate of {base_rate.value:.2f}"
            )
        return Prediction(
            kind="likely_responds",
            label="Replies (context)",
            value=value,
            low=low,
            high=high,
            n_evidence=cell_n or n,
            caveat=caveat,
        )

    # -- the measurement -----------------------------------------------------

    def skill(self) -> ContextualSkill:
        """How much the context adds, scored by leaving one opportunity out.

        The base-rate predictor is scored on the same held-out outcomes, so the
        comparison is exactly the one PLAN.md G-12 asks for. Leave-one-out
        rather than a single split so a 40-opportunity corpus still gets a
        number, and because there is no seed: the number reproduces.
        """
        total = len(self._pairs)
        if total == 0:
            return ContextualSkill(
                n=0,
                brier_contextual=0.0,
                brier_base=0.0,
                skill=0.0,
                note="no reply opportunities in this corpus",
            )
        global_n, global_replied = self.base_cell()
        brier_ctx = 0.0
        brier_base = 0.0
        for message, replied in self._pairs:
            truth = 1.0 if replied else 0.0
            context = self._context_of(message)
            cell_n, cell_replied = self.cell(context)
            # The corpus's smoothed base rate without this opportunity.
            base = (global_replied - (1 if replied else 0) + 1.0) / (global_n - 1 + 2.0)
            # The contextual prediction without this opportunity's own vote.
            value = (
                cell_replied - (1 if replied else 0) + self._shrinkage * base
            ) / (cell_n - 1 + self._shrinkage)
            brier_ctx += (value - truth) ** 2
            brier_base += (base - truth) ** 2
        brier_ctx /= total
        brier_base /= total
        # The base prediction is smoothed (+1,+2), so it never lands exactly
        # on the truth: the denominator is strictly positive, and skill is a
        # plain ratio, with no perfect-baseline special case to guard.
        return ContextualSkill(
            n=total,
            brier_contextual=brier_ctx,
            brier_base=brier_base,
            skill=1.0 - (brier_ctx / brier_base),
        )

    # -- internals -----------------------------------------------------------

    def _context_of(self, message: Message) -> ReplyContext:
        question = "?" in message.text
        opening = message.message_id in self._opening_ids
        in_window = self._in_window(message)
        return ReplyContext(question=question, opening=opening, in_window=in_window)

    def _in_window(self, message: Message) -> bool:
        if self._median_reply_hour is None or message.sent_at is None:
            # No replies measured, or no clock to read: the window says
            # nothing, and "False" by convention rather than by claim.
            return False
        hour = message.sent_at.hour + message.sent_at.minute / 60.0
        offset = abs((hour - self._median_reply_hour + 12.0) % 24.0 - 12.0)
        return offset <= self._window_half_hours


def _opportunity_pairs(corpus: Corpus) -> list[tuple[Message, bool]]:
    """Every reply opportunity as (the other's message, whether it was answered).

    One opportunity per turn (PLAN.md R-f): an other-person message is an
    opportunity, and it counts as answered exactly when the subject's message
    immediately follows it in the same conversation. A trailing unanswered
    message is a real unanswered opportunity — the outcome that was observed.
    """
    pairs: list[tuple[Message, bool]] = []
    subject = corpus.person.subject
    for conversation in _by_conversation(corpus.messages).values():
        for index, item in enumerate(conversation):
            if item.speaker == subject:
                continue
            following = conversation[index + 1 :]
            if following and following[0].speaker == subject:
                pairs.append((item, True))
            else:
                pairs.append((item, False))
    return pairs


def _reply_hours(corpus: Corpus) -> list[float]:
    """Hours, as stored, of the subject's messages that answered someone.

    The subject's *reply* hour, not their overall writing hour: a person who
    receives prompts all morning and writes their own things at night has a
    reply window in the morning, and the window exists to capture that.
    """
    subject = corpus.person.subject
    hours: list[float] = []
    for conversation in _by_conversation(corpus.messages).values():
        for previous, current in zip(conversation, conversation[1:], strict=False):
            if (
                previous.speaker != subject
                and current.speaker == subject
                and current.sent_at is not None
            ):
                hours.append(current.sent_at.hour + current.sent_at.minute / 60.0)
    return hours


def build_contextual_predictor(
    corpus: Corpus, *, shrinkage: float = DEFAULT_SHRINKAGE
) -> ContextualReplyPredictor:
    """A contextual predictor over a corpus."""
    return ContextualReplyPredictor(corpus, shrinkage=shrinkage)


# -- helpers ---------------------------------------------------------------


def _has_emoji(text: str) -> bool:
    """Whether the text contains an emoji, by the Unicode property."""
    return has_emoji(text)


def _quartiles(values: Sequence[float]) -> tuple[float, float]:
    """The 25th and 75th percentiles, by nearest-rank.

    Nearest-rank rather than an interpolating method so the interval is always
    an observed value — a reader can point at a real message at each end.

    **A degenerate result is widened, not returned.** When every observation
    is the same value — four gaps of two minutes, say — the percentile
    interval is ``[2, 2]``, which asserts a certainty the sample does not
    contain: it says the next gap is certainly two minutes when in fact all it
    shows is that every gap seen so far was two minutes. So a zero-width
    interval is widened by the sample's own scale of variation, which for a
    constant sample is zero, and the interval is instead widened to the
    nearest observed neighbour.

    In practice this means the interval reports the range of what was seen,
    and a corpus with one distinct value reports that value as a point with
    the caveat that it was the only one seen.
    """
    ordered = sorted(values)
    if not ordered:
        return 0.0, 0.0
    n = len(ordered)
    low_index = max(0, int(math.ceil(0.25 * n)) - 1)
    high_index = min(n - 1, int(math.ceil(0.75 * n)) - 1)
    low = float(ordered[low_index])
    high = float(ordered[high_index])
    if low == high and n >= 1:
        # A zero-width interval. Two cases, and they must not be confused:
        #
        # - The sample is **constant** (every observation the same). Then the
        #   percentile interval is legitimately zero-width, and widening it by
        #   the sample's spread is impossible — there is no spread. A
        #   placeholder is used instead, and the caveat says so.
        # - The sample is **not constant** but the 25th and 75th percentiles
        #   land on the same value. That happens with a skewed sample, and a
        #   zero-width interval would then assert a certainty the data does
        #   not contain. The interval is widened to the observed range.
        distinct = sorted(set(ordered))
        if len(distinct) == 1:
            # No variation at all. Report a symmetric interval of one unit
            # either side: zero would claim certainty, and anything wider
            # would be invented.
            margin = 1.0 if high > 0 else 0.5
            return max(0.0, high - margin), high + margin
        # Not constant: report the range that was actually observed.
        return float(distinct[0]), float(distinct[-1])
    return low, high


def _is_degenerate(values: Sequence[float]) -> bool:
    """Whether the sample itself has no variation.

    This is a property of the *sample*, not of the interval: an interval that
    has been widened from a constant sample is no longer zero-width, but the
    caveat still has to say the underlying data was constant — otherwise a
    reader takes the widened range for a measured spread.
    """
    if not values:
        return True
    return len(set(values)) <= 1


def _wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for a proportion.

    Used instead of the normal approximation because the normal approximation
    is meaningless at the rates that matter here — a 0/6 record would produce a
    degenerate interval that claims certainty about nothing.
    """
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _latencies(messages: Sequence[Message]) -> list[float]:
    """Minutes between consecutive messages in the same conversation."""
    by_conversation: dict[str, list[Message]] = {}
    for item in messages:
        if item.sent_at is None:
            continue
        by_conversation.setdefault(item.conversation_id, []).append(item)
    gaps: list[float] = []
    for thread in by_conversation.values():
        thread.sort(key=lambda item: item.sent_at)  # type: ignore[arg-type,return-value]
        for previous, current in zip(thread, thread[1:], strict=False):
            assert previous.sent_at is not None and current.sent_at is not None
            gap = (current.sent_at - previous.sent_at).total_seconds() / 60.0
            if gap >= 0:
                gaps.append(gap)
    return gaps


def _opportunities(messages: Sequence[Message], subject: str) -> int:
    """How many times the other person wrote something the subject could answer.

    Every message from the other person is an opportunity: it was addressed to
    the subject, and the subject either answered it or did not.
    """
    count = 0
    for conversation in _by_conversation(messages).values():
        count += sum(1 for item in conversation if item.speaker != subject)
    return count


def _replied_count(messages: Sequence[Message], subject: str) -> int:
    """How many of those opportunities the subject actually answered.

    An opportunity counts as answered when the subject's message immediately
    follows it in the same conversation. The denominator and the numerator must
    use the *same* definition of an opportunity, or the rate is a ratio of two
    different populations.
    """
    count = 0
    for conversation in _by_conversation(messages).values():
        for index, item in enumerate(conversation):
            if item.speaker == subject:
                continue
            following = conversation[index + 1 :]
            if following and following[0].speaker != subject:
                # Not an opportunity: another message from the other person
                # follows.
                continue
            if following and following[0].speaker == subject:
                count += 1
    return count


def _by_conversation(messages: Sequence[Message]) -> dict[str, list[Message]]:
    grouped: dict[str, list[Message]] = {}
    for item in messages:
        grouped.setdefault(item.conversation_id, []).append(item)
    for thread in grouped.values():
        thread.sort(key=lambda item: (item.sent_at is None, item.sent_at or datetime.min))
    return grouped
