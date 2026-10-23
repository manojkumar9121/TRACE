"""Evaluation: does any of this actually work?

A trait analyser that reports "high emoji usage, 0.87 confidence" and is never
checked against reality is a very confident way to be wrong. This module is
the check.

Two evaluations, for two different claims:

**Feature recovery** — do the measured features match the truth? Only
measurable against a corpus whose ground truth is known, so the harness
generates one: a synthetic corpus with planted parameters. That measures the
*extractor*, not the model, which is the part that can be wrong silently.

**Prediction accuracy** — do the predictions match what actually happened? Also
measured against the synthetic corpus, by holding out the final message of
each conversation and predicting it from the rest.

Every number this module produces is a measurement. None of them is a claim
about how good the system is in general, because a synthetic corpus is not a
distribution of real people.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from trace_traits.corpus import Corpus, Message, Person, corpus_from_messages
from trace_traits.features import extract_features
from trace_traits.predict import build_contextual_predictor, build_predictor

__all__ = [
    "FeatureEval",
    "PredictionEval",
    "Plant",
    "evaluate_features",
    "evaluate_prediction",
    "synthetic_corpus",
]


@dataclass(frozen=True, slots=True)
class Plant:
    """A person's planted parameters — the ground truth of a synthetic corpus."""

    name: str
    #: Mean length of their substantive messages, in characters.
    mean_length: float
    #: Standard deviation of message length.
    length_sigma: float
    #: Probability a message contains an emoji.
    emoji_rate: float
    #: Probability a message contains a question mark.
    question_rate: float
    #: Share of the other person's messages they answer.
    response_rate: float
    #: Mean minutes between their message and the previous one.
    reply_gap_minutes: float
    #: Share of conversations they open.
    initiation_rate: float

    def expected_length(self) -> float:
        return self.mean_length


@dataclass(frozen=True, slots=True)
class FeatureEval:
    """How well the extractor recovered the planted parameters."""

    plant: Plant
    n_messages: int
    n_conversations: int
    #: ``{dimension: (measured, planted)}`` where both are comparable.
    comparisons: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: ``{dimension: absolute error}``.
    errors: dict[str, float] = field(default_factory=dict)

    @property
    def mean_absolute_error(self) -> float:
        if not self.errors:
            return 0.0
        return sum(self.errors.values()) / len(self.errors)

    def as_lines(self) -> tuple[str, ...]:
        return tuple(
            f"{dim:<20} measured={measured:>8.3f}  planted={planted:>8.3f}  "
            f"err={self.errors[dim]:>7.3f}"
            for dim, (measured, planted) in self.comparisons.items()
        )


@dataclass(frozen=True, slots=True)
class PredictionEval:
    """How well the predictor predicted held-out messages."""

    n_predictions: int
    #: Mean absolute error on predicted message length, in characters.
    length_mae: float
    #: Mean absolute error of the *median* predictor against the median actual,
    #: which is the honest comparison for a distribution-valued prediction.
    length_median_mae: float
    #: Fraction of held-out messages whose length fell inside the predicted
    #: interquartile range.
    length_coverage: float
    #: Brier score for the emoji prediction. Lower is better; 0.25 is the
    #: score of a constant 0.5 prediction.
    emoji_brier: float
    #: Brier score of a constant-0.5 baseline, for comparison.
    emoji_brier_baseline: float
    #: Brier score of the **base-rate** baseline: predict every message to
    #: have the subject's own training-set emoji rate.
    #:
    #: This is the fair comparison, and the constant-0.5 baseline is not. A
    #: constant-0.5 prediction scores 0.25 for *any* subject, so beating it
    #: says only that the subject is not a coin flip — a persona whose emoji
    #: rate is near 0.5 has almost no skill over it. An audit caught a
    #: persona scoring 0.2491 against 0.2500 and calling it a win.
    emoji_brier_base_rate: float
    #: The Brier **skill score** against the base-rate baseline:
    #: ``1 - brier / brier_base_rate``. Zero means no better than knowing the
    #: subject's rate; negative means worse.
    emoji_skill: float
    #: Mean absolute error of the base-rate length baseline: predict every
    #: held-out message to be the subject's own median training length.
    #: The length MAE had no baseline at all, which is also wrong.
    length_mae_base_rate: float
    #: The length skill score against that baseline: ``1 - mae / mae_base``.
    length_skill: float
    #: Share of the other's messages the subject answered, predicted vs actual.
    response_rate_predicted: float
    response_rate_actual: float
    #: The **contextual** reply predictor's leave-one-out Brier skill against
    #: the plain base-rate predictor, over this corpus's own reply
    #: opportunities (PLAN.md G-12). Zero means the conditioning added nothing
    #: this corpus measures; negative means it hurt.
    contextual_skill: float = 0.0
    #: The opportunities the contextual skill was measured over.
    contextual_n: int = 0

    def as_lines(self) -> tuple[str, ...]:
        return (
            f"held-out predictions: {self.n_predictions}",
            f"length MAE:           {self.length_mae:.1f} chars",
            f"length median MAE:    {self.length_median_mae:.1f} chars",
            f"length IQR coverage:  {self.length_coverage:.2f} (target ~0.50)",
            f"emoji Brier:          {self.emoji_brier:.4f} "
            f"(const-0.5 {self.emoji_brier_baseline:.4f}, "
            f"base rate {self.emoji_brier_base_rate:.4f})",
            f"emoji skill vs base:  {self.emoji_skill:+.3f} (0 = no better than their own rate)",
            f"length MAE vs base:   {self.length_mae_base_rate:.1f} chars "
            f"(skill {self.length_skill:+.3f})",
            f"response rate:        predicted {self.response_rate_predicted:.3f} "
            f"vs actual {self.response_rate_actual:.3f}",
            f"contextual reply skill vs base rate: {self.contextual_skill:+.3f} "
            f"over {self.contextual_n} opportunities (0 = no better)",
        )


def synthetic_corpus(
    plant: Plant,
    *,
    n_conversations: int = 12,
    messages_per_conversation: int = 14,
    seed: int = 7,
    observer: str | None = "me",
    observer_plant: Plant | None = None,
) -> Corpus:
    """A corpus generated from a known set of parameters.

    The generator is deliberately simple and the parameters are the ground
    truth: the evaluator's job is to check that the extractor recovers them,
    not to model a realistic chat.

    The observer gets their own parameters by default — a different message
    length and emoji rate from the subject. Without that, the two speakers are
    indistinguishable and the corpus cannot tell a trait of the subject from a
    trait of everybody.
    """
    rng = random.Random(seed)
    other = observer_plant or Plant(
        name=observer or "them",
        mean_length=plant.mean_length * 1.4,
        length_sigma=plant.length_sigma,
        emoji_rate=plant.emoji_rate / 2,
        question_rate=plant.question_rate / 2,
        response_rate=1.0 - plant.response_rate,
        reply_gap_minutes=plant.reply_gap_minutes / 2,
        initiation_rate=1.0 - plant.initiation_rate,
    )
    messages: list[Message] = []
    base = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
    counter = 0

    for conversation_index in range(n_conversations):
        conversation_id = f"c{conversation_index}"
        moment = base + timedelta(days=conversation_index * 3)
        subject_opens = rng.random() < plant.initiation_rate
        thread: list[tuple[str, str]] = []

        for index in range(messages_per_conversation):
            if index == 0:
                speaker = plant.name if subject_opens else other.name
            else:
                previous = thread[-1][0]
                # The speaker who replies is the one who did not speak last.
                responder = other.name if previous == plant.name else plant.name
                rate = plant.response_rate if responder == plant.name else other.response_rate
                speaker = responder if rng.random() < rate else previous
            source = plant if speaker == plant.name else other
            thread.append((speaker, _plant_text(rng, source, conversation_index, index)))

        for index, (speaker, text) in enumerate(thread):
            counter += 1
            gap = max(0.0, rng.gauss(plant.reply_gap_minutes, plant.reply_gap_minutes / 3))
            moment = moment + timedelta(minutes=gap if index else 0)
            messages.append(
                Message(
                    message_id=f"m{counter}",
                    conversation_id=conversation_id,
                    speaker=speaker,
                    text=text,
                    sent_at=moment,
                    upload_id=f"u{conversation_index}",
                )
            )

    return corpus_from_messages(
        Person(subject=plant.name, observer=observer),
        messages,
    )


def _plant_text(rng: random.Random, plant: Plant, conversation: int, index: int) -> str:
    """A message whose properties match the planted parameters.

    Each message gets its **own** random stream, seeded from its position.
    That is not fastidiousness: ``random.gauss`` caches the second of its pair
    of normal deviates, so it consumes a random number alternately — two, then
    zero — and a shared stream therefore aligns the draw that decides "emoji
    or not" with the message's index. The visible symptom was a corpus whose
    final message in every thread carried far fewer emoji than every other
    message, which the prediction evaluation then measured as a failure of the
    predictor. It was a failure of the generator.

    Independent streams per message make the properties of each message
    independent of its position, which is what a generator of synthetic people
    has to guarantee for an evaluation of them to mean anything.
    """
    local = random.Random((conversation << 16) ^ (index * 2654435761) ^ 0x5EED)
    length = max(1, int(local.gauss(plant.mean_length, plant.length_sigma)))
    body = "😀" + "x" * max(0, length - 1) if local.random() < plant.emoji_rate else "x" * length
    if local.random() < plant.question_rate:
        body += "?"
    return body


def evaluate_features(corpus: Corpus, plant: Plant) -> FeatureEval:
    """Compare the measured features against the planted parameters."""
    features = extract_features(corpus)
    comparisons: dict[str, tuple[float, float]] = {}
    errors: dict[str, float] = {}

    def record(dimension: str, measured: float | None, planted: float) -> None:
        if measured is None:
            return
        comparisons[dimension] = (measured, planted)
        errors[dimension] = abs(measured - planted)

    length = features.get("Response length")
    if length is not None and length.value is not None:
        record("Response length", length.value, plant.mean_length)
    emoji = features.get("Emoji usage")
    if emoji is not None and emoji.value is not None:
        record("Emoji usage", emoji.value, plant.emoji_rate)
    questions = features.get("Questions")
    if questions is not None and questions.value is not None:
        record("Questions", questions.value, plant.question_rate)
    initiation = features.get("Initiation")
    if initiation is not None and initiation.value is not None:
        record("Initiation", initiation.value, plant.initiation_rate)

    return FeatureEval(
        plant=plant,
        n_messages=len(corpus.subject_messages),
        n_conversations=len(corpus.conversations),
        comparisons=comparisons,
        errors=errors,
    )


def evaluate_prediction(
    corpus: Corpus,
    *,
    plant: Plant | None = None,
) -> PredictionEval:
    """Predict the subject's last message in each conversation and score it.

    The **subject's** final message in each conversation is held out; the
    predictor sees everything else. This is the honest test: the prediction is
    made from what the person has actually written before, and scored against
    what they wrote next.

    Only the subject's messages are held out and scored. Holding out the last
    message of a conversation regardless of who wrote it would score the
    predictor against the *other* person's writing — and the predictor is
    about the subject, so that measures nothing.
    """
    subject = corpus.person.subject
    held_out: list[Message] = []
    train: list[Message] = []
    by_conversation: dict[str, list[Message]] = {}
    for item in corpus.messages:
        by_conversation.setdefault(item.conversation_id, []).append(item)
    for thread in by_conversation.values():
        own = [item for item in thread if item.speaker == subject]
        if len(own) < 2:
            train.extend(thread)
            continue
        # Hold out the subject's last message, and everything after it (which
        # is the other person's reply to it).
        last_own_index = max(index for index, item in enumerate(thread) if item.speaker == subject)
        held_out.append(thread[last_own_index])
        train.extend(thread[:last_own_index])

    if not held_out:
        return PredictionEval(
            n_predictions=0,
            length_mae=0.0,
            length_median_mae=0.0,
            length_coverage=0.0,
            emoji_brier=0.0,
            emoji_brier_baseline=0.0,
            emoji_brier_base_rate=0.0,
            emoji_skill=0.0,
            length_mae_base_rate=0.0,
            length_skill=0.0,
            response_rate_predicted=0.0,
            response_rate_actual=0.0,
        )

    train_corpus = corpus_from_messages(
        corpus.person, train, lineage_strategy=corpus.lineage_strategy
    )
    predictor = build_predictor(train_corpus)

    length_prediction = predictor.likely_length()
    emoji_prediction = predictor.likely_emoji()
    response_prediction = predictor.likely_responds()

    actual_lengths = [item.char_length for item in held_out]
    length_mae = sum(abs(length_prediction.value - value) for value in actual_lengths) / len(
        actual_lengths
    )
    length_median_mae = abs(
        statistics.median(actual_lengths)
        - statistics.median(
            [m.char_length for m in train_corpus.subject_messages if m.char_length >= 4]
            or [length_prediction.value]
        )
    )
    coverage = sum(
        1 for value in actual_lengths if length_prediction.low <= value <= length_prediction.high
    ) / len(actual_lengths)

    actual_emoji = [1 if _has_emoji(item.text) else 0 for item in held_out]
    emoji_brier = sum((emoji_prediction.value - truth) ** 2 for truth in actual_emoji) / len(
        actual_emoji
    )
    emoji_brier_baseline = sum((0.5 - truth) ** 2 for truth in actual_emoji) / len(actual_emoji)

    # The **base-rate** baseline: predict every held-out message to have the
    # subject's own training-set emoji rate. This is the fair comparison,
    # because it is what you would achieve by knowing only how often this
    # person uses emoji and nothing else. The constant-0.5 baseline is a
    # property of the coin, not of the person.
    train_subject = train_corpus.subject_messages
    train_emoji_rate = (
        sum(1 for item in train_subject if _has_emoji(item.text)) / len(train_subject)
        if train_subject
        else 0.0
    )
    emoji_brier_base_rate = sum((train_emoji_rate - truth) ** 2 for truth in actual_emoji) / len(
        actual_emoji
    )

    # The length baseline, which there was not: predict every held-out
    # message to be the subject's own median training length.
    train_lengths = [item.char_length for item in train_subject]
    base_length = statistics.median(train_lengths) if train_lengths else 0.0
    length_mae_base_rate = (
        sum(abs(base_length - value) for value in actual_lengths) / len(actual_lengths)
        if actual_lengths
        else 0.0
    )

    def _skill(value: float, baseline: float) -> float:
        """Skill against a baseline: 0 is no better, 1 is perfect."""
        if baseline <= 0:
            return 0.0
        return 1.0 - (value / baseline)

    actual_response = _actual_response_rate(corpus)
    # The contextual reply predictor is measured on this corpus's own reply
    # opportunities, by leaving each one out and scoring both it and the
    # base-rate predictor against what happened (PLAN.md G-12). This is a
    # corpus-local measurement of whether the conditioning helps *this*
    # person's history, not a claim about conditioning in general.
    context_skill = build_contextual_predictor(corpus).skill()
    return PredictionEval(
        n_predictions=len(held_out),
        length_mae=length_mae,
        length_median_mae=length_median_mae,
        length_coverage=coverage,
        emoji_brier=emoji_brier,
        emoji_brier_baseline=emoji_brier_baseline,
        emoji_brier_base_rate=emoji_brier_base_rate,
        emoji_skill=_skill(emoji_brier, emoji_brier_base_rate),
        length_mae_base_rate=length_mae_base_rate,
        length_skill=_skill(length_mae, length_mae_base_rate),
        response_rate_predicted=response_prediction.value,
        response_rate_actual=actual_response,
        contextual_skill=context_skill.skill,
        contextual_n=context_skill.n,
    )


def _has_emoji(text: str) -> bool:
    ranges = ((0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x1F000, 0x1F2FF))
    return any(any(low <= ord(char) <= high for low, high in ranges) for char in text)


def _actual_response_rate(corpus: Corpus) -> float:
    from trace_traits.predict import _opportunities, _replied_count

    opportunities = _opportunities(corpus.messages, corpus.person.subject)
    if opportunities == 0:
        return 0.0
    return _replied_count(corpus.messages, corpus.person.subject) / opportunities


def run_suite(plants: Sequence[Plant], *, seed: int = 7) -> tuple[FeatureEval, ...]:
    """Evaluate a set of plants."""
    out: list[FeatureEval] = []
    for index, plant in enumerate(plants):
        corpus = synthetic_corpus(plant, seed=seed + index)
        out.append(evaluate_features(corpus, plant))
    return tuple(out)
