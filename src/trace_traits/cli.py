"""Command-line entry point: measure a corpus, predict, and report.

Two modes:

``profile``
    Read a JSON export of messages, produce the trait profile and the
    predictions, and print them.

``evaluate``
    Run the synthetic-corpus evaluation and print the measured numbers. This
    is the mode that says whether any of it works.

The JSON shape is deliberately boring — a list of message objects with
``id``, ``conversation_id``, ``speaker``, ``text`` and an optional
``sent_at`` — because the point of this file is to be the seam between "a
source of chat messages" and "a corpus TRACE can measure".
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from trace_core import Scorer, TableConfidencePolicy, TableRetrievalPolicy
from trace_traits import (
    Corpus,
    LineageStrategy,
    Message,
    Person,
    build_predictor,
    build_profile,
    corpus_from_messages,
    evaluate_features,
    evaluate_prediction,
    synthetic_corpus,
)
from trace_traits.evaluate import Plant
from trace_traits.reference_policy import REFERENCE_POLICY
from trace_traits.respond import reactivity
from trace_traits.respond_synth import ReactivityPlant, reactivity_corpus

__all__ = ["main"]


def _policy() -> TableConfidencePolicy:
    # The example weights, declared once in reference_policy (D5) so the CLI
    # and the docs cannot drift apart.
    return REFERENCE_POLICY


def _scorer() -> Scorer:
    conf = _policy()
    return Scorer(
        confidence=conf,
        retrieval=TableRetrievalPolicy(
            weights={"vector": 0.55, "recency": 0.20, "source": 0.15, "confidence": 0.10},
            source_weights=conf.source_weights,
        ),
    )


def _load_corpus(path: Path, *, subject: str, observer: str | None) -> Corpus:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["messages"] if isinstance(payload, dict) else payload
    messages = []
    for index, item in enumerate(raw):
        sent_at = item.get("sent_at")
        messages.append(
            Message(
                message_id=str(item.get("id") or f"m{index}"),
                conversation_id=str(item.get("conversation_id") or "c0"),
                speaker=str(item.get("speaker") or subject),
                text=str(item.get("text") or ""),
                sent_at=datetime.fromisoformat(sent_at) if sent_at else None,
                upload_id=item.get("upload_id"),
                source_id=item.get("source_id"),
            )
        )
    strategy = LineageStrategy.CONVERSATION
    if isinstance(payload, dict):
        raw_strategy = payload.get("lineage_strategy")
        if raw_strategy:
            strategy = LineageStrategy(raw_strategy)
    return corpus_from_messages(
        Person(subject=subject, observer=observer), messages, lineage_strategy=strategy
    )


def _cmd_profile(args: argparse.Namespace) -> int:
    corpus = _load_corpus(Path(args.path), subject=args.subject, observer=args.observer)
    now = datetime.now().astimezone()

    print(
        f"Corpus: {len(corpus)} messages, {len(corpus.conversations)} conversations, "
        f"subject={corpus.person.subject}"
    )
    print()

    profile = build_profile(corpus, scorer=_scorer(), now=now)
    print("TRAIT PROFILE")
    print("-" * 78)
    for reading in profile.readings:
        print(
            f"  {reading.dimension:<20} {reading.band:<10} "
            f"confidence={reading.score:.2f}  state={reading.decision.state}"
        )
        print(f"    {reading.detail}")
        print(f"    {reading.n_evidence} messages across {reading.n_conversations} conversations")
    print()

    print("PREDICTIONS")
    print("-" * 78)
    for prediction in build_predictor(corpus).all_predictions():
        print(f"  {prediction.describe()}")
    print()
    print("  These are distributions over what this person is likely to do next,")
    print("  derived from their own messages. They are not claims about a person.")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    print("SYNTHETIC EVALUATION")
    print("=" * 78)
    print("A corpus with known parameters, measured. This checks the extractor")
    print("and the predictor against ground truth — not real people.")
    print()

    plants = [
        Plant("terse", 12.0, 4.0, 0.05, 0.03, 0.9, 3.0, 0.6),
        Plant("average", 60.0, 20.0, 0.25, 0.15, 0.8, 25.0, 0.4),
        Plant("verbose", 180.0, 45.0, 0.45, 0.35, 0.7, 90.0, 0.25),
    ]

    print("FEATURE RECOVERY — measured against planted parameters")
    print("-" * 78)
    for index, plant in enumerate(plants):
        corpus = synthetic_corpus(
            plant, n_conversations=24, messages_per_conversation=16, seed=101 + index
        )
        report = evaluate_features(corpus, plant)
        print(f"  {plant.name}:")
        for line in report.as_lines():
            print(f"    {line}")
        print()

    print("PREDICTION ACCURACY — held-out messages")
    print("-" * 78)
    wins = 0
    for index, plant in enumerate(plants):
        corpus = synthetic_corpus(
            plant, n_conversations=96, messages_per_conversation=16, seed=201 + index
        )
        prediction_report = evaluate_prediction(corpus, plant=plant)
        beats = prediction_report.emoji_brier < prediction_report.emoji_brier_baseline
        wins += int(beats)
        print(f"  {plant.name}: {'beats' if beats else 'LOSES TO'} a constant-0.5 baseline")
        for line in prediction_report.as_lines():
            print(f"    {line}")
        print()
    print(
        f"  {wins}/{len(plants)} personas beat a constant-0.5 prediction on emoji.\n"
        "  Against the subject's OWN base rate, the length and emoji\n"
        "  predictions are that rate by construction (skill 0.000): they\n"
        "  return a smoothed historical rate, and cannot beat it on the\n"
        "  data that produced it. The contextual reply predictor is NOT\n"
        "  that predictor: it is conditioned on the other side's last\n"
        "  message, the thread position, and the hour, and its\n"
        "  leave-one-out Brier skill is printed above. On these generated\n"
        "  personas that skill is near zero — the honest reading, because\n"
        "  this generator plants no conditional structure for the context\n"
        "  to find. Where the structure exists (a corpus whose questions\n"
        "  really are answered more), the same measurement reports a\n"
        "  positive skill; the planted-case numbers live in\n"
        "  tests/test_contextual_predict.py and the ledger, not here."
    )
    print()
    print("  A synthetic corpus is not a distribution of real people. These numbers")
    print("  say the machinery works on data with known ground truth; they say")
    print("  nothing about how it behaves on real conversations.")
    return 0


def _cmd_response(args: argparse.Namespace) -> int:
    print("RESPONSE-CONTRAST EVALUATION (T3)")
    print("=" * 78)
    print("A corpus with planted events, measured. Does the analyser recover")
    print("a planted change in how someone writes?")
    print()

    print("RECOVERY — planted reactivity vs the band the analyser reported")
    print("-" * 78)
    correct = 0
    total = 0
    for reactivity_planted in (0.0, 0.25, 0.5, 0.8, 1.0):
        plant = ReactivityPlant(
            name="them",
            mean_length=90.0,
            length_sigma=18.0,
            emoji_rate=0.3,
            question_rate=0.2,
            response_rate=0.9,
            reply_gap_minutes=20.0,
            initiation_rate=0.4,
            event_rate=0.35,
            reactivity=reactivity_planted,
        )
        corpus, stimuli = reactivity_corpus(
            plant, n_conversations=24, messages_per_conversation=14, seed=5
        )
        reading = reactivity(corpus, stimuli)
        # A plant below 0.35 means "the subject did not change". The band
        # that honestly reports that is `unsettled` when the data does not
        # settle it and `stable` when it does — both are a correct recovery
        # of "no change". `stable` alone is not required, because claiming
        # that the absence of change was *established* needs every measure
        # to have been equivalent, and the synthetic corpus does not always
        # give it that.
        expected = "reduced" if reactivity_planted >= 0.35 else "no_change"
        hit = (
            reading.band == expected
            if expected != "no_change"
            else reading.band in {"stable", "unsettled"}
        )
        correct += int(hit)
        total += 1
        print(
            f"  planted reactivity={reactivity_planted:<5} -> band={reading.band:<12} "
            f"expected={expected:<10} {'OK' if hit else 'MISS'}"
        )
        for measure in reading.measures:
            print(f"      {measure.describe()}")
        print()
    print(f"  {correct}/{total} planted cases recovered correctly.")
    print()
    print("  The 'unsettled' band is the important one: it means the evidence")
    print("  does not settle whether anything changed. It is NOT reported as")
    print("  'no change' — that is the equivalence test, and a null is not a")
    print("  finding.")
    return 0


def _cmd_temporal(args: argparse.Namespace) -> int:
    """Measure when this person writes, and what that pattern means."""
    import os

    from trace_traits import OpenRouterBackend, TemporalTier

    corpus = _load_corpus(Path(args.path), subject=args.subject, observer=args.observer)

    api_key = os.environ.get("TRACE_LLM_API_KEY", "")
    backend = OpenRouterBackend(
        os.environ.get("TRACE_LLM_MODEL", "minimax/minimax-m3.1-flash:free"),
        api_key=api_key,
        enabled=bool(api_key) and not args.no_labels,
    )
    tier = TemporalTier(backend, speaker=args.subject)
    reading = tier.read(corpus, width_minutes=args.width)

    print("TEMPORAL READING (T4)")
    print("=" * 78)
    print("When they write, and what that pattern means. The clock is a")
    print("measurement; the meaning is a reading, and it is marked as one.")
    print()
    for line in reading.as_lines():
        print(f"  {line}")
    print()
    print(f"  band: {reading.band}")
    print()
    print(f"  {reading.caveat}")
    if not args.no_labels and not os.environ.get("TRACE_LLM_API_KEY", ""):
        print()
        print("  (no TRACE_LLM_API_KEY set, so the model was not consulted and")
        print("   only the label-free activity measurement was made)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trace-traits")
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile", help="measure a corpus and predict")
    profile.add_argument("path", help="JSON file of messages")
    profile.add_argument("--subject", default="them", help="who the traits are about")
    profile.add_argument("--observer", default="me", help="who the messages are read by")
    profile.set_defaults(func=_cmd_profile)

    evaluate = sub.add_parser("evaluate", help="run the synthetic evaluation")
    evaluate.set_defaults(func=_cmd_evaluate)

    response = sub.add_parser("response", help="run the T3 response-contrast evaluation")
    response.set_defaults(func=_cmd_response)

    temporal = sub.add_parser("temporal", help="measure when they write, and what that means")
    temporal.add_argument("path", help="JSON file of messages")
    temporal.add_argument("--subject", default="them", help="who the traits are about")
    temporal.add_argument("--observer", default="me", help="who the messages are read by")
    temporal.add_argument(
        "--width",
        type=int,
        default=15,
        help="clock window in minutes (default 15; 60 is coarser)",
    )
    temporal.add_argument(
        "--no-labels", action="store_true", help="skip the model, report activity only"
    )
    temporal.set_defaults(func=_cmd_temporal)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
