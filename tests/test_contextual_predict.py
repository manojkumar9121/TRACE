"""Tests for the contextual reply predictor (PLAN.md G-12).

The audit's finding was precise: the plain T2 predictor *is* the base rate, so
its skill against the base rate is zero by construction. What closes the gap
is conditioning — the same person at their question-answering best behaves
differently from the same person at midnight — and a skill score that is
measured, not asserted. These tests plant known conditional behaviour into
hand-built corpora and check the predictor both reads it and is scored
honestly when it holds nothing.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from trace_traits.corpus import Corpus, Message, Person, corpus_from_messages
from trace_traits.evaluate import evaluate_prediction
from trace_traits.predict import (
    ContextualReplyPredictor,
    ReplyContext,
    build_contextual_predictor,
)

BASE = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
QUESTION = "where are we meeting tonight?"
STATEMENT = "heading out now"

# The eight context cells, for naming tests.
EVERY = ReplyContext(question=False, opening=False, in_window=False)


def _explicit_conditioned_corpus(
    *,
    n_conversations: int = 24,
    question_rate: float = 0.9,
    statement_rate: float = 0.3,
    opening_rate: float = 0.5,
    seed: int = 7,
) -> Corpus:
    rng = random.Random(seed)
    messages: list[Message] = []
    counter = 0
    for conversation in range(n_conversations):
        discussion_id = f"c{conversation}"
        # Half the conversations live in the reply window, half outside it,
        # so the question axis stays decorrelated from the hour axis: the /
        # question rate is the same in both halves.
        hour = 9 if conversation % 2 == 0 else 21
        rows = (
            (False, True, opening_rate),
            (True, False, question_rate),
            (False, False, statement_rate),
        )
        for offset, (question, _opening, rate) in enumerate(rows):
            counter += 1
            sent = datetime(
                2026, 6, 1, hour, 5 * offset + (counter % 3) + 1, tzinfo=UTC
            ) + timedelta(days=conversation)
            text = QUESTION if question else STATEMENT
            messages.append(Message(f"m{counter}", discussion_id, "me", text, sent))
            if rng.random() < rate:
                counter += 1
                messages.append(
                    Message(
                        f"m{counter}",
                        discussion_id,
                        "them",
                        "back home now, reply lands",
                        sent + timedelta(minutes=1),
                    )
                )
    return corpus_from_messages(Person(subject="them", observer="me"), messages)


def _window_corpus() -> Corpus:
    """A person who answers at their own hour and never otherwise."""
    messages: list[Message] = []
    counter = 0
    for conversation in range(12):
        conversation_id = f"c{conversation}"
        early = datetime(
            2026, 6, 1, 10, counter % 50, tzinfo=UTC
        ) + timedelta(days=conversation)
        late = datetime(
            2026, 6, 1, 22, counter % 50, tzinfo=UTC
        ) + timedelta(days=conversation)
        counter += 1
        # An in-window prompt, always answered.
        messages.append(
            Message(f"m{counter}", conversation_id, "them", "morning ping", early)
        )
        counter += 1
        messages.append(
            Message(f"m{counter}", conversation_id, "me", "here, answering", early)
        )
        counter += 1
        # An out-of-window prompt, never answered.
        messages.append(
            Message(f"m{counter}", conversation_id, "them", "late ping", late)
        )
    return corpus_from_messages(Person(subject="me", observer="them"), messages)


# -- the conditioning reads what was planted ----------------------------------


@pytest.fixture()
def conditioned() -> ContextualReplyPredictor:
    return build_contextual_predictor(_explicit_conditioned_corpus())


def test_the_question_cell_holds_what_was_planted(
    conditioned: ContextualReplyPredictor,
) -> None:
    question_cell = conditioned.cell(
        ReplyContext(question=True, opening=False, in_window=True)
    )
    assert question_cell[0] == 12, "one question turn per in-window conversation"
    assert question_cell[1] / question_cell[0] >= 0.8
    statement_cell = conditioned.cell(
        ReplyContext(question=False, opening=False, in_window=True)
    )
    assert statement_cell[0] == 12
    assert statement_cell[1] / statement_cell[0] < 0.5
    question = conditioned.likely_responds(
        ReplyContext(question=True, opening=False, in_window=True)
    )
    statement = conditioned.likely_responds(
        ReplyContext(question=False, opening=False, in_window=True)
    )
    assert question.value > statement.value


def test_an_unobserved_context_reports_the_base_rate_by_name(
    conditioned: ContextualReplyPredictor,
) -> None:
    """Questions at an opening are not planted; the answer is the fallback."""
    context = ReplyContext(question=True, opening=True, in_window=True)
    assert conditioned.cell(context) == (0, 0)
    prediction = conditioned.likely_responds(context)
    plain = conditioned.base_rate()
    assert prediction.value == pytest.approx(plain.value)
    assert prediction.caveat.startswith("unobserved context")


def test_a_thin_cell_says_it_is_leaning_on_the_base_rate(
    conditioned: ContextualReplyPredictor,
) -> None:
    """Three conversations put three observations in every cell — not enough."""
    small = build_contextual_predictor(_explicit_conditioned_corpus(n_conversations=3))
    context = ReplyContext(question=True, opening=False, in_window=True)
    assert small.cell(context)[0] == 2
    prediction = small.likely_responds(context)
    assert prediction.caveat.startswith("thin context")


def test_the_table_counts_every_opportunity(
    conditioned: ContextualReplyPredictor,
) -> None:
    """No opportunity is silently unaccounted for."""
    cells_total = sum(
        count for context in _all_cells() for count, _ in (conditioned.cell(context),)
    )
    assert cells_total == conditioned.n_opportunities
    assert conditioned.n_opportunities == conditioned.base_cell()[0] == 72


def _all_cells() -> tuple[ReplyContext, ...]:
    return tuple(
        ReplyContext(question=question, opening=opening, in_window=in_window)
        for question in (False, True)
        for opening in (False, True)
        for in_window in (False, True)
    )


def test_a_planted_context_scores_skill_above_the_base_rate(
    conditioned: ContextualReplyPredictor,
) -> None:
    measured = conditioned.skill()
    assert measured.n == conditioned.n_opportunities > 0
    assert measured.brier_contextual < measured.brier_base
    assert measured.skill > 0.10


def test_the_time_window_reads_the_planted_hour() -> None:
    corpus = _window_corpus()
    predictor = build_contextual_predictor(corpus)
    in_window = predictor.likely_responds(
        ReplyContext(question=False, opening=False, in_window=True)
    )
    outside = predictor.likely_responds(
        ReplyContext(question=False, opening=False, in_window=False)
    )
    assert in_window.value > outside.value
    # The window is *their* reply hour, measured from their own messages.
    assert predictor.context_for(
        Message("fresh", "c0", "them", "late ping", datetime(2026, 6, 9, 22, 30, tzinfo=UTC))
    ).in_window is False
    assert predictor.context_for(
        Message("fresh", "c0", "them", "morning ping", datetime(2026, 6, 9, 10, 5, tzinfo=UTC))
    ).in_window is True


def test_a_corpus_without_opportunities_refuses_to_predict() -> None:
    corpus = corpus_from_messages(
        Person(subject="them"),
        [Message("m1", "c0", "them", "just me here", BASE)],
    )
    predictor = build_contextual_predictor(corpus)
    measured = predictor.skill()
    assert measured.n == 0
    assert measured.skill == 0.0
    assert "no reply opportunities" in measured.note
    base = predictor.base_rate()
    assert base.n_evidence == 0
    assert "no reply opportunities" in base.caveat


def test_a_base_rate_cannot_be_perfect_and_the_context_still_scores(
) -> None:
    """Every other-person message answered, in one uniform context.

    The base rate is smoothed, so a perfect record never predicts 1.0 — and
    the contextual predictor, reading the only cell there is, gets closer to
    the truth the smoothing of the pooled rate cannot.
    """
    messages: list[Message] = []
    for conversation in range(6):
        conversation_id = f"c{conversation}"
        moment = BASE + timedelta(days=conversation)
        messages.append(
            Message(f"o{conversation}", conversation_id, "me", "any news?", moment)
        )
        messages.append(
            Message(
                f"r{conversation}",
                conversation_id,
                "them",
                "all good here",
                moment + timedelta(minutes=5),
            )
        )
    corpus = corpus_from_messages(Person(subject="them", observer="me"), messages)
    measured = build_contextual_predictor(corpus).skill()
    assert measured.n == 6
    assert measured.brier_base > 0.0
    assert measured.brier_contextual < measured.brier_base
    assert measured.note == ""


def test_negative_shrinkage_is_refused() -> None:
    corpus = _explicit_conditioned_corpus()
    with pytest.raises(ValueError, match="shrinkage"):
        build_contextual_predictor(corpus, shrinkage=-1.0)


def test_the_evaluation_record_carries_the_contextual_skill() -> None:
    """The skill measurement reaches the evaluation ledger, not just the API."""
    corpus = _explicit_conditioned_corpus()
    evaluation = evaluate_prediction(corpus)
    assert evaluation.contextual_n == 72
    assert evaluation.contextual_skill != 0.0
    assert any(
        "contextual reply skill" in line for line in evaluation.as_lines()
    )
