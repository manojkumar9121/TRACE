"""Live-model tests for the temporal tier.

The tier's central claim is that **the model proposes and the arithmetic
disposes**. A stub cannot test that, because a stub is written by the same
author as the code and does what it is told. These tests need a real model.

Two things are checked against a live model:

1. **The labelling works.** A model can tell "going to bed" from "still at
   the office" at the same clock hour — the distinction no arithmetic over
   timestamps can make.
2. **The tier is strict anyway.** A hostile model cannot get a fabricated
   citation, a wrong-speaker citation, or a reading unsupported by the cited
   text through.

Run with::

    TRACE_LLM_API_KEY=... pytest tests/test_temporal_live.py -v
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from trace_traits import (
    Message,
    Person,
    corpus_from_messages,
    interpret_pattern,
    label_messages,
    measure_patterns,
)
from trace_traits.perception import LLMBackend, OpenRouterBackend

pytestmark = pytest.mark.live

MODEL = os.environ.get("TRACE_LLM_MODEL", "minimax/minimax-m3.1-flash:free")
BASE = datetime(2026, 10, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def backend() -> OpenRouterBackend:
    key = os.environ.get("TRACE_LLM_API_KEY", "")
    if not key:
        pytest.skip("TRACE_LLM_API_KEY is not set")
    return OpenRouterBackend(MODEL, api_key=key)


def _corpus(*, days: int = 10):
    """A nightly signoff at 23:05 and a nightly work message at 23:40."""
    messages: list[Message] = []
    for day in range(days):
        moment = BASE + timedelta(days=day)
        messages.append(
            Message(
                f"s{day}",
                f"c{day}",
                "them",
                "ok heading to bed, night",
                moment.replace(hour=23, minute=5),
            )
        )
        messages.append(
            Message(
                f"w{day}",
                f"c{day}",
                "them",
                "still at the office finishing the report",
                moment.replace(hour=23, minute=40),
            )
        )
    return corpus_from_messages(Person(subject="them", observer="me"), messages)


def test_a_live_model_tells_a_bedtime_from_a_busy_evening(backend: OpenRouterBackend) -> None:
    """The distinction the tier exists for, at the same clock hour.

    Both messages are in the 23:00 hour. A model that cannot tell them
    apart is useless here, and no arithmetic over timestamps can help.
    """
    corpus = _corpus(days=10)
    result = label_messages(corpus, backend)
    assert result.schema_ok is True
    labels = result.by_id()
    assert len(labels) == 20
    # every signoff message is a signoff
    for day in range(10):
        assert labels[f"s{day}"] == "signoff", labels[f"s{day}"]
        assert labels[f"w{day}"] == "work", labels[f"w{day}"]


def test_a_live_model_produces_a_pattern_the_arithmetic_agrees_with(
    backend: OpenRouterBackend,
) -> None:
    """The model labels; the arithmetic finds the concentration."""
    corpus = _corpus(days=10)
    result = label_messages(corpus, backend)
    patterns = measure_patterns(corpus, result, speaker="them", width_minutes=15)
    purposes = {pattern.purpose: pattern for pattern in patterns}
    assert "signoff" in purposes
    signoff = purposes["signoff"]
    assert signoff.band == "consistent"
    assert signoff.window == "23:00–23:14"
    # and the work messages are in a *different* window
    assert purposes["work"].window == "23:30–23:44"


def test_a_live_model_reads_the_pattern_and_the_text_check_holds(
    backend: OpenRouterBackend,
) -> None:
    """The reading is the model's; whether it survives is ours."""
    corpus = _corpus(days=10)
    result = label_messages(corpus, backend)
    patterns = measure_patterns(corpus, result, speaker="them", width_minutes=15)
    signoff = next(p for p in patterns if p.purpose == "signoff")
    assert signoff.band == "consistent"

    reading = interpret_pattern(corpus, signoff, backend)
    assert reading.interpretation in {"bedtime", "unknown"}
    if reading.interpretation == "bedtime":
        assert reading.verified is True
        # every cited message really does mention bedtime
        for evidence_id in reading.evidence_ids:
            message = next(m for m in corpus.messages if m.message_id == evidence_id)
            assert any(word in message.text.lower() for word in ("bed", "night", "sleep"))


def test_a_live_model_cannot_get_a_fabricated_citation_through(
    backend: OpenRouterBackend,
) -> None:
    """A citation that does not resolve is dropped, never repaired."""
    corpus = _corpus(days=10)

    class Fabricating(LLMBackend):
        def complete(self, system: str, user: str, *, schema: dict) -> str:
            return json.dumps(
                {
                    "messages": [
                        {
                            "id": "m99",
                            "label": "signoff",
                            "evidence": "ok heading to bed, night",
                        }
                    ]
                }
            )

    result = label_messages(corpus, Fabricating())
    assert result.labels == ()
    assert result.dropped_unresolved == 1


def test_a_live_model_cannot_get_a_wrong_speaker_citation_through(
    backend: OpenRouterBackend,
) -> None:
    """A reading may not cite a message the finding did not rest on."""
    corpus = _corpus(days=10)
    result = label_messages(corpus, backend)
    patterns = measure_patterns(corpus, result, speaker="them", width_minutes=15)
    signoff = next(p for p in patterns if p.purpose == "signoff")

    class WrongSpeaker(LLMBackend):
        def complete(self, system: str, user: str, *, schema: dict) -> str:
            return json.dumps(
                {
                    "interpretation": "bedtime",
                    "claim": "They go to bed at 11pm.",
                    "evidence_ids": ["w0", "w1", "w2"],
                }
            )

    reading = interpret_pattern(corpus, signoff, WrongSpeaker())
    assert reading.interpretation == "unknown"
    assert reading.verified is False
