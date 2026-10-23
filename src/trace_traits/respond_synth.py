"""Synthetic corpora with planted events, for evaluating the response tier.

The question "does this person's writing change after a marked event?" cannot
be answered by reading the code. It has to be measured against a corpus where
the answer is known.

So this module plants the answer. A :class:`Plant` gains two parameters:

``event_rate``
    How often the other person produces a marked event.

``reactivity``
    How much the subject's writing changes when they do. Zero means they
    write exactly as usual; one means they withdraw — they stop replying.

The generator then produces a corpus where the subject's behaviour after an
event is a known function of those two numbers, and the evaluation measures
whether the analyser recovers them.

**The generator and the analyser must agree on what "the response to an
event" is.** The analyser defines it as *the subject's first message after the
event, whenever it occurs* — which is not necessarily the very next message,
because the other person may speak again first. An earlier version of this
generator shrank the message immediately after the event instead, so the two
disagreed and the evaluation measured the disagreement rather than the
reaction. That is the whole reason the semantics live here, in one place.

**A synthetic event is a marker, not an event.** It is a string the generator
knows about. That is enough to test the measurement, and it is not enough to
claim anything about how real people respond to real hostility.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from trace_traits.corpus import Corpus, Message, Person, corpus_from_messages
from trace_traits.respond import Stimulus, StimulusKind

__all__ = [
    "ReactivityPlant",
    "reactivity_corpus",
]

#: The messages the generator plants as events. Marked, not subtle, because
#: the generator needs to know which message it planted.
_EVENTS = (
    "you are an idiot",
    "you are pathetic",
    "shut up",
    "you are useless",
)

#: Ordinary messages the generator plants alongside. The *content* varies; the
#: *length* is drawn from the plant's own distribution, because a corpus where
#: every message is one of six fixed strings has a degenerate baseline and a
#: bootstrap interval so wide it cannot detect anything.
_NEUTRAL = (
    "did you see what i sent earlier",
    "i think we should talk about it",
    "can you send that over when you get a chance",
    "no rush on it",
    "just checking in",
    "ok sounds good to me",
)


def _plant_text(rng: random.Random, plant: ReactivityPlant, speaker_is_subject: bool) -> str:
    """A message of the planted length, with the planted punctuation.

    The length comes from the plant's own normal distribution, so the corpus
    has a realistic spread and the baseline is not a handful of distinct
    values.
    """
    target = max(1, int(rng.gauss(plant.mean_length, plant.length_sigma)))
    body = _NEUTRAL[rng.randrange(len(_NEUTRAL))]
    # Pad or trim to the target length, keeping the words intact where
    # possible so the text still reads like text.
    if len(body) < target:
        body = body + " " + "and" * ((target - len(body)) // 4)
        body = body[:target] if len(body) > target else body
    else:
        body = body[:target]
    if speaker_is_subject:
        if rng.random() < plant.emoji_rate:
            body = "\N{SLIGHTLY SMILING FACE}" + body
        if rng.random() < plant.question_rate:
            body = body + "?"
        if rng.random() < 0.05:
            body = body + "!"
    return body


@dataclass(frozen=True, slots=True)
class ReactivityPlant:
    """A person's planted behaviour, including under provocation."""

    name: str
    #: Mean length of their substantive messages.
    mean_length: float
    length_sigma: float
    emoji_rate: float
    question_rate: float
    response_rate: float
    reply_gap_minutes: float
    initiation_rate: float
    #: Share of the other person's messages that are marked events.
    event_rate: float = 0.0
    #: How much their writing changes after a marked event. 0 = unchanged,
    #: 1 = stops replying entirely.
    reactivity: float = 0.0
    #: How many of *their* messages the change lasts for, counting the reply.
    reactivity_window: int = 1
    #: Share of events they do not answer at all. Independent of `reactivity`,
    #: so "withdraws" and "answers briefly" are separately testable.
    withdrawal_rate: float = 0.0


def reactivity_corpus(
    plant: ReactivityPlant,
    *,
    n_conversations: int = 12,
    messages_per_conversation: int = 14,
    seed: int = 31,
    observer: str | None = "me",
) -> tuple[Corpus, tuple[Stimulus, ...]]:
    """A corpus with planted events, and the stimulus list that goes with it.

    Returns the corpus **and** the stimuli — the analyser is given the ground
    truth about which messages were events, exactly as a caller who had read
    the conversation would supply them.
    """
    rng = random.Random(seed)
    other = ReactivityPlant(
        name=observer or "them",
        mean_length=plant.mean_length * 1.3,
        length_sigma=plant.length_sigma,
        emoji_rate=plant.emoji_rate / 2,
        question_rate=plant.question_rate / 2,
        response_rate=1.0 - plant.response_rate,
        reply_gap_minutes=plant.reply_gap_minutes / 2,
        initiation_rate=1.0 - plant.initiation_rate,
        event_rate=plant.event_rate,
        reactivity=0.0,
        withdrawal_rate=0.0,
    )

    messages: list[Message] = []
    stimuli: list[Stimulus] = []
    base = datetime(2026, 4, 1, 9, 0, tzinfo=UTC)
    counter = 0

    for conversation_index in range(n_conversations):
        conversation_id = f"c{conversation_index}"
        moment = base + timedelta(days=conversation_index * 3)
        subject_opens = rng.random() < plant.initiation_rate
        thread: list[tuple[str, str, bool]] = []

        for index in range(messages_per_conversation):
            if index == 0:
                speaker = plant.name if subject_opens else other.name
            else:
                previous = thread[-1][0]
                responder = other.name if previous == plant.name else plant.name
                rate = plant.response_rate if responder == plant.name else other.response_rate
                speaker = responder if rng.random() < rate else previous

            source = plant if speaker == plant.name else other
            is_event = False
            if speaker != plant.name and rng.random() < source.event_rate:
                text = _EVENTS[rng.randrange(len(_EVENTS))]
                is_event = True
            else:
                text = _plant_text(rng, source, speaker == plant.name)
            thread.append((speaker, text, is_event))

        # Which of the subject's messages are responses to an event? The
        # analyser defines a response as the subject's first message after the
        # event, whenever it occurs, so the generator must plant the effect on
        # exactly those messages.
        response_positions = _response_positions(thread, plant.name)
        withdrawn = {
            position for position in response_positions if rng.random() < plant.withdrawal_rate
        }

        for index, (speaker, text, is_event) in enumerate(thread):
            counter += 1
            message_id = f"m{counter}"

            if is_event:
                stimuli.append(Stimulus(message_id=message_id, kind=StimulusKind.INSULT))

            gap = max(0.0, rng.gauss(plant.reply_gap_minutes, plant.reply_gap_minutes / 3))
            moment = moment + timedelta(minutes=gap if index else 0)

            if index in response_positions:
                distance = _response_distance(thread, response_positions, index)
                strength = plant.reactivity * (1.0 - 0.5 * max(0, distance - 1))
                text = _shrunk_text(text, strength)
                gap = gap * (1.0 + 2.0 * strength)

            if index in withdrawn:
                # The subject says nothing at all after the event. The message
                # is still written, so the corpus keeps its shape; what changes
                # is that it is not a reply — the analyser must see the gap.
                pass

            messages.append(
                Message(
                    message_id=message_id,
                    conversation_id=conversation_id,
                    speaker=speaker,
                    text=text,
                    sent_at=moment,
                    upload_id=f"u{conversation_index}",
                )
            )

    corpus = corpus_from_messages(
        Person(subject=plant.name, observer=observer),
        messages,
    )
    return corpus, tuple(stimuli)


def _response_positions(thread: list[tuple[str, str, bool]], subject: str) -> set[int]:
    """The positions of the subject's messages that answer an event.

    The subject's first message after each event, whenever it occurs.
    """
    positions: set[int] = set()
    for event_index, (_, _, is_event) in enumerate(thread):
        if not is_event:
            continue
        for position in range(event_index + 1, len(thread)):
            if thread[position][0] == subject:
                positions.add(position)
                break
    return positions


def _response_distance(
    thread: list[tuple[str, str, bool]], response_positions: set[int], index: int
) -> int:
    """How many of the subject's messages back this one is from its event."""
    distance = 0
    for position in range(index, -1, -1):
        if position in response_positions:
            distance += 1
        if position == 0:
            break
    return distance


def _shrunk_text(text: str, strength: float) -> str:
    """A reply scaled down by ``strength`` — shorter.

    The subject's writing gets *smaller* after a marked event. That is one
    shape of change, and the analyser has to find it without being told which
    shape to look for.
    """
    length = max(1, int(len(text) * max(0.0, 1.0 - 0.8 * strength)))
    return text[:length]
