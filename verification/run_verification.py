"""Run TRACE's T1/T2 against the hand-authored corpora and build the ledger.

Reads every corpus under `verification/corpus/` and `verification/probes/`,
computes the ground truth independently (via `ground_truth.py`, which does not
import trace_traits), runs `extract_features` and `build_predictor` from the
library, and compares every dimension.

The **bands** are hand judgements: what a human reading the transcript would
say. The **values** are computed by the independent calculator, so a value
mismatch is a real disagreement rather than a stale number in this file.

Output: a JSON ledger at `verification/ledger.json` and a summary on stdout.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ROOT))

from ground_truth import load, stats, to_messages  # noqa: E402

from trace_traits import (  # noqa: E402
    Corpus,
    Message,
    Person,
    build_predictor,
    corpus_from_messages,
    extract_features,
)

# ---------------------------------------------------------------------------
# The planted truth, stated BY HAND for each corpus.
#
# `band` is what a human reading the transcript would say. `why` is the
# reasoning. The numeric value is taken from the independent calculator, so it
# is never a number typed into this file and left to rot.
#
# A `band` of `insufficient_evidence` is also a hand judgement: the corpus is
# declared too small for the dimension to say anything honest, and the library
# is expected to agree. Reporting a band where the planted truth says
# insufficient is an OVER-CLAIM, and counts against the tally.
#
# `gt_key` names the ground-truth statistic the library's `value` should
# match. For reply latency it is deliberately the HUMAN reading of the term
# (time from the other person's message to the subject's reply), not the
# subject-to-subject gap the library actually computes — that difference is
# one of the findings.
# ---------------------------------------------------------------------------

PLANTED: dict[str, dict[str, dict]] = {
    "p1_verbose_formal": {
        "response_length": {
            "band": "high",
            "gt_key": "subject_mean_length",
            "why": "13 substantive messages averaging 275 chars; observer averages 20. "
            "Ratio ~13.7x the observer, so 'high' on any basis.",
        },
        "emoji": {
            "band": "low",
            "gt_key": "emoji_rate",
            "why": "no emoji in any of the 13 messages",
        },
        "questions": {
            "band": "low",
            "gt_key": "question_rate",
            "why": "0 of 13 messages contain '?'",
        },
        "initiation": {
            "band": "moderate",
            "gt_key": "initiation_rate",
            "why": "subject opened 2 of 4 conversations (c2, c4) = 0.50",
        },
        "reply_latency": {
            "band": "moderate",
            "gt_key": "reply_latency_median",
            "why": "median 12 min from the other person's message to the subject's reply",
        },
        "vocabulary": {
            "band": "moderate",
            "gt_key": "vocabulary_msttr",
            "why": "310 distinct / 666 words = 0.466 TTR; long formal prose, repeats "
            "function words but no distinctive vocabulary",
        },
    },
    "p2_terse_lowercase": {
        "response_length": {
            "band": "low",
            "gt_key": "subject_mean_length",
            "why": "16 substantive messages averaging 7.4 chars; observer averages 43.8. "
            "Ratio ~0.17x, so 'low' on any basis.",
        },
        "emoji": {
            "band": "low",
            "gt_key": "emoji_rate",
            "why": "no emoji in any of the 17 messages",
        },
        "questions": {
            "band": "low",
            "gt_key": "question_rate",
            "why": "0 of 17 messages contain '?'",
        },
        "initiation": {
            "band": "low",
            "gt_key": "initiation_rate",
            "why": "subject opened 1 of 5 conversations (c2) = 0.20",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median 2 min from the other person's message to the subject's reply",
        },
        "vocabulary": {
            "band": "insufficient_evidence",
            "gt_key": "vocabulary_msttr",
            "why": "the person uses ~28 words total. The TTR is high (0.89) only "
            "because the corpus is tiny — every word in it is distinct — so a "
            "ratio cannot carry the band. Declared too small under G-11's "
            "floor: bands mean MSTTR-100, and this corpus cannot supply one. "
            "The human reading ('very limited vocabulary') is carried by the "
            "response-length dimension instead.",
        },
    },
    "p3_emoji_heavy": {
        "response_length": {
            "band": "moderate",
            "gt_key": "subject_mean_length",
            "why": "17 messages averaging 24.6 chars; observer averages 31.7. "
            "Ratio ~0.78, so 'moderate' (0.75-1.5).",
        },
        "emoji": {
            "band": "high",
            "gt_key": "emoji_rate",
            "why": "emoji in 15 of 17 messages = 0.88",
        },
        "questions": {
            "band": "low",
            "gt_key": "question_rate",
            "why": "'?' in 1 of 17 messages (m7) = 0.059, below the 0.08 low threshold",
        },
        "initiation": {
            "band": "moderate",
            "gt_key": "initiation_rate",
            "why": "subject opened 2 of 4 conversations (c2, c3) = 0.50",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median 1 min from the other person's message to the subject's reply",
        },
        "vocabulary": {
            "band": "insufficient_evidence",
            "gt_key": "vocabulary_msttr",
            "why": "80 words in total: under the declared G-11 floor, where a band "
            "would mean reading the ratio on a scale other than the one the "
            "thresholds were set for. (The ratio 0.84 is what a small sample "
            "does to any text.)",
        },
    },
    "p3b_emoji_star_clock": {
        "response_length": {
            "band": "moderate",
            "gt_key": "subject_mean_length",
            "why": "13 messages averaging 18.7 chars; observer averages 20.9. Ratio ~0.90.",
        },
        "emoji": {
            "band": "high",
            "gt_key": "emoji_rate",
            "why": "emoji in ALL 13 messages (star, alarm clock, hourglass, heavy circle) "
            "= 1.00. A human counts every one of these as an emoji.",
        },
        "questions": {
            "band": "low",
            "gt_key": "question_rate",
            "why": "0 of 13 messages contain '?'",
        },
        "initiation": {
            "band": "moderate",
            "gt_key": "initiation_rate",
            "why": "subject opened 1 of 3 conversations (c2) = 0.33",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median 1 min from the other person's message to the subject's reply",
        },
        "vocabulary": {
            "band": "insufficient_evidence",
            "gt_key": "vocabulary_msttr",
            "why": "45 words in total — a bag of emoji and single words. Declared too "
            "small under G-11's floor: no band can mean the same thing here as "
            "it does for a full 100-word window.",
        },
    },
    "p4_no_questions": {
        "response_length": {
            "band": "high",
            "gt_key": "subject_mean_length",
            "why": "12 messages averaging 57.8 chars; observer averages 26.7. "
            "Ratio ~2.17, above the 1.5 high threshold.",
        },
        "emoji": {
            "band": "low",
            "gt_key": "emoji_rate",
            "why": "no emoji in any of the 12 messages",
        },
        "questions": {
            "band": "low",
            "gt_key": "question_rate",
            "why": "0 of 12 subject messages contain '?'; the observer asks in 8 of 12. "
            "This is the planted trait: the subject states, the observer asks.",
        },
        "initiation": {
            "band": "low",
            "gt_key": "initiation_rate",
            "why": "subject opened 0 of 3 conversations = 0.00",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median 3.5 min from the other person's message to the subject's reply",
        },
        "vocabulary": {
            "band": "moderate",
            "gt_key": "vocabulary_msttr",
            "why": "98 distinct / 147 words = 0.67 TTR; ordinary working vocabulary",
        },
    },
    "p5_initiator": {
        "response_length": {
            "band": "high",
            "gt_key": "subject_mean_length",
            "why": "12 messages averaging 193.8 chars; observer averages 29.2. Ratio ~6.6.",
        },
        "emoji": {
            "band": "low",
            "gt_key": "emoji_rate",
            "why": "no emoji in any of the 12 messages",
        },
        "questions": {
            "band": "moderate",
            "gt_key": "question_rate",
            "why": "'?' in 2 of 12 messages (m1, m5) = 0.167, inside 0.08-0.35",
        },
        "initiation": {
            "band": "high",
            "gt_key": "initiation_rate",
            "why": "subject opened 4 of 5 conversations (c1,c2,c3,c4) = 0.80",
        },
        "reply_latency": {
            "band": "moderate",
            "gt_key": "reply_latency_median",
            "why": "median 25 min from the other person's message to the subject's reply",
        },
        "vocabulary": {
            "band": "moderate",
            "gt_key": "vocabulary_msttr",
            "why": "224 distinct / 476 words = 0.47 TTR; long, ordinary prose",
        },
    },
    "p6_fast_replier": {
        "response_length": {
            "band": "high",
            "gt_key": "subject_mean_length",
            "why": "13 messages averaging 61.9 chars; observer averages 38.0. "
            "Ratio ~1.63, just above the 1.5 high threshold.",
        },
        "emoji": {
            "band": "low",
            "gt_key": "emoji_rate",
            "why": "no emoji in any of the 13 messages",
        },
        "questions": {
            "band": "low",
            "gt_key": "question_rate",
            "why": "0 of 13 messages contain '?'",
        },
        "initiation": {
            "band": "low",
            "gt_key": "initiation_rate",
            "why": "subject opened 0 of 4 conversations = 0.00",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median 1 min from the other person's message to the subject's reply. "
            "This is the planted trait: replies in 1-3 minutes every time.",
        },
        "vocabulary": {
            "band": "moderate",
            "gt_key": "vocabulary_msttr",
            "why": "102 distinct / 177 words = 0.58 TTR; ordinary working vocabulary",
        },
    },
    "p7_slow_replier": {
        "response_length": {
            "band": "high",
            "gt_key": "subject_mean_length",
            "why": "9 messages averaging 152.3 chars; observer averages 46.2. Ratio ~3.3.",
        },
        "emoji": {
            "band": "low",
            "gt_key": "emoji_rate",
            "why": "no emoji in any of the 9 messages",
        },
        "questions": {
            "band": "low",
            "gt_key": "question_rate",
            "why": "0 of 9 messages contain '?'",
        },
        "initiation": {
            "band": "low",
            "gt_key": "initiation_rate",
            "why": "subject opened 0 of 3 conversations = 0.00",
        },
        "reply_latency": {
            "band": "high",
            "gt_key": "reply_latency_median",
            "why": "median 918 min (15.3 h) from the other person's message to the "
            "subject's reply. Planted: replies after four to nine hours, every time.",
        },
        "vocabulary": {
            "band": "moderate",
            "gt_key": "vocabulary_msttr",
            "why": "150 distinct / 290 words = 0.52 TTR; long, ordinary prose",
        },
    },
    "p8_rich_vocab": {
        "response_length": {
            "band": "high",
            "gt_key": "subject_mean_length",
            "why": "9 messages averaging 241.6 chars; observer averages 30.9. Ratio ~7.8.",
        },
        "emoji": {
            "band": "low",
            "gt_key": "emoji_rate",
            "why": "no emoji in any of the 9 messages",
        },
        "questions": {
            "band": "low",
            "gt_key": "question_rate",
            "why": "0 of 9 messages contain '?'",
        },
        "initiation": {
            "band": "low",
            "gt_key": "initiation_rate",
            "why": "subject opened 0 of 3 conversations = 0.00",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median 4 min from the other person's message to the subject's reply",
        },
        "vocabulary": {
            "band": "high",
            "gt_key": "vocabulary_msttr",
            "why": "213 distinct / 304 words = 0.70 TTR, above the 0.62 high threshold. "
            "Planted: a large, varied, technical vocabulary.",
        },
    },
    "p9_repetitive": {
        "response_length": {
            "band": "low",
            "gt_key": "subject_mean_length",
            "why": "12 messages averaging 13.6 chars; observer averages 30.3. Ratio ~0.45.",
        },
        "emoji": {
            "band": "low",
            "gt_key": "emoji_rate",
            "why": "no emoji in any of the 12 messages",
        },
        "questions": {
            "band": "low",
            "gt_key": "question_rate",
            "why": "0 of 12 messages contain '?'",
        },
        "initiation": {
            "band": "low",
            "gt_key": "initiation_rate",
            "why": "subject opened 0 of 3 conversations = 0.00",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median 2 min from the other person's message to the subject's reply",
        },
        "vocabulary": {
            "band": "insufficient_evidence",
            "gt_key": "vocabulary_msttr",
            "why": "1 distinct word ('good') in 35 words. The repetition is real and a "
            "human could read it at a glance, but the ratio scale cannot carry the "
            "judgment under the declared G-11 floor; the trait shows in response "
            "length (planted low: 13.6 chars, all acknowledgements).",
        },
    },
    "p10_fast_subject_slow_observer": {
        "response_length": {
            "band": "moderate",
            "gt_key": "subject_mean_length",
            "why": "9 messages averaging 33.3 chars; observer averages 30.4. Ratio ~1.09.",
        },
        "emoji": {
            "band": "low",
            "gt_key": "emoji_rate",
            "why": "no emoji in any of the 9 messages",
        },
        "questions": {
            "band": "moderate",
            "gt_key": "question_rate",
            "why": "'?' in 2 of 9 messages (m2, m8) = 0.222, inside 0.08-0.35",
        },
        "initiation": {
            "band": "low",
            "gt_key": "initiation_rate",
            "why": "subject opened 0 of 3 conversations = 0.00",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median 20 SECONDS from the other person's message to the subject's "
            "reply. Planted: the subject replies in 9-35 seconds, every time. "
            "This corpus separates 'how long they take to reply' from 'the gap "
            "between two of their own messages'.",
        },
        "vocabulary": {
            "band": "insufficient_evidence",
            "gt_key": "vocabulary_msttr",
            "why": "66 words in total; the measured ratio at this size (0.82) runs "
            "high because the window is short — the exact mismatch G-11 declares. "
            "A band here would measure the sample, not the vocabulary.",
        },
    },
    "p11_fast_subject_slow_observer_padded": {
        "response_length": {
            "band": "moderate",
            "gt_key": "subject_mean_length",
            "why": "12 messages averaging 36.0 chars; observer averages 31.8. Ratio ~1.13.",
        },
        "emoji": {
            "band": "low",
            "gt_key": "emoji_rate",
            "why": "no emoji in any of the 12 messages",
        },
        "questions": {
            "band": "moderate",
            "gt_key": "question_rate",
            "why": "'?' in 2 of 12 messages = 0.167, inside 0.08-0.35",
        },
        "initiation": {
            "band": "low",
            "gt_key": "initiation_rate",
            "why": "subject opened 0 of 3 conversations = 0.00",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median ~19 SECONDS from the other person's message to the subject's "
            "reply. Planted: the subject replies in 10-35 seconds, every time. Same "
            "plant as p10 but sized so the library's gate passes and it must commit "
            "to a band. The gap between two of the SUBJECT's own messages is ~9.3 "
            "hours here, because the observer takes hours to answer.",
        },
        "vocabulary": {
            "band": "insufficient_evidence",
            "gt_key": "vocabulary_msttr",
            "why": "93 words in total — this corpus was sized to clear the older gate "
            "so the library would commit to a band; under the declared G-11 floor "
            "it stays insufficient, because every corpus that gets a band has the "
            "same scale to be compared on.",
        },
    },
    "p12_plain_typography_no_emoji": {
        "response_length": {
            "band": "high",
            "gt_key": "subject_mean_length",
            "why": "9 messages averaging 56.0 chars; observer averages 30.7. "
            "Ratio ~1.83, above the 1.5 high threshold.",
        },
        "emoji": {
            "band": "low",
            "gt_key": "emoji_rate",
            "why": "NO emoji. The subject types plain typographic characters — check "
            "marks, crosses, plain stars, arrows, a wavy dash — which are text, not "
            "emoji. A human reading this transcript says zero emoji.",
        },
        "questions": {
            "band": "low",
            "gt_key": "question_rate",
            "why": "0 of 9 subject messages contain '?'",
        },
        "initiation": {
            "band": "low",
            "gt_key": "initiation_rate",
            "why": "subject opened 0 of 3 conversations = 0.00",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median 2 min from the other person's message to the subject's reply",
        },
        "vocabulary": {
            "band": "insufficient_evidence",
            "gt_key": "vocabulary_msttr",
            "why": "91 words in total; measured ratio 0.84 runs high on a short sample "
            "— declared too small under G-11's floor rather than banded across a "
            "scale shift.",
        },
    },
    "r1_equal_lengths": {
        "response_length": {
            "band": "moderate",
            "gt_key": "subject_mean_length",
            "why": "9 messages averaging 134.6 chars; observer averages 115.9. "
            "Ratio ~1.16 => 'moderate'. The ABSOLUTE basis (mean/40) would say 'high'.",
        },
        "emoji": {"band": "low", "gt_key": "emoji_rate", "why": "no emoji"},
        "questions": {"band": "low", "gt_key": "question_rate", "why": "no '?'"},
        "initiation": {
            "band": "low",
            "gt_key": "initiation_rate",
            "why": "subject opened 0 of 3 conversations = 0.00",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median 7 min from the other person's message to the subject's reply",
        },
        "vocabulary": {
            "band": "moderate",
            "gt_key": "vocabulary_msttr",
            "why": "TTR over ~233 words; ordinary prose",
        },
    },
    "r2_verbose_subject_terse_observer": {
        "response_length": {
            "band": "high",
            "gt_key": "subject_mean_length",
            "why": "SAME subject messages as r1 (mean 134.6 chars); observer averages "
            "15.8. Ratio ~8.5 => 'high'. The absolute basis would also say 'high'. "
            "If r1 and r2 report the same band, the observer baseline is not applied.",
        },
        "emoji": {"band": "low", "gt_key": "emoji_rate", "why": "no emoji"},
        "questions": {"band": "low", "gt_key": "question_rate", "why": "no '?'"},
        "initiation": {
            "band": "low",
            "gt_key": "initiation_rate",
            "why": "subject opened 0 of 3 conversations = 0.00",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median 7 min from the other person's message to the subject's reply",
        },
        "vocabulary": {
            "band": "moderate",
            "gt_key": "vocabulary_msttr",
            "why": "TTR over ~233 words; ordinary prose",
        },
    },
    "r3_verbose_subject_very_verbose_observer": {
        "response_length": {
            "band": "moderate",
            "gt_key": "subject_mean_length",
            "why": "9 messages averaging 168.4 chars; observer averages 132.9. "
            "Ratio ~1.27 => 'moderate' (0.75-1.5). The ABSOLUTE basis would say 'high'.",
        },
        "emoji": {"band": "low", "gt_key": "emoji_rate", "why": "no emoji"},
        "questions": {"band": "low", "gt_key": "question_rate", "why": "no '?'"},
        "initiation": {
            "band": "low",
            "gt_key": "initiation_rate",
            "why": "subject opened 0 of 3 conversations = 0.00",
        },
        "reply_latency": {
            "band": "low",
            "gt_key": "reply_latency_median",
            "why": "median 7 min from the other person's message to the subject's reply",
        },
        "vocabulary": {
            "band": "moderate",
            "gt_key": "vocabulary_msttr",
            "why": "TTR over ~292 words; ordinary prose",
        },
    },
}

# Probe corpora: the expectation is a gate outcome, not a band.
PROBE_EXPECTATIONS: dict[str, dict] = {
    "g1_seven_messages": {
        "n_subject_messages": 7,
        "expect": "insufficient_evidence on every dimension (7 subject messages < "
        "MIN_SAMPLE 8; 74 words < the 100-word vocabulary floor)",
    },
    "g2_twenty_messages_one_conversation": {
        "n_subject_messages": 20,
        "expect": "insufficient_evidence on every dimension (1 conversation < "
        "MIN_CONVERSATIONS 2) despite 20 messages and 329 words — the vocabulary "
        "floor is word-based, but the conversation rule still refuses one thread",
    },
    "g3_three_long_messages": {
        "n_subject_messages": 3,
        "expect": "insufficient_evidence on every dimension (3 < MIN_SAMPLE 8; the "
        "6 messages sit in 1 conversation, and word count alone does not carry "
        "the vocabulary dimension)",
    },
}

DIMENSION_KEYS = {
    "Response length": "response_length",
    "Emoji usage": "emoji",
    "Questions": "questions",
    "Initiation": "initiation",
    "Reply latency": "reply_latency",
    "Vocabulary": "vocabulary",
}


def build_corpus(payload: dict) -> Corpus:
    messages = [
        Message(
            message_id=item["id"],
            conversation_id=item["conversation_id"],
            speaker=item["speaker"],
            text=item["text"],
            sent_at=datetime.fromisoformat(item["sent_at"]),
        )
        for item in payload["messages"]
    ]
    return corpus_from_messages(Person(subject="them", observer="me"), messages)


def verdict_for(
    planted: dict,
    expected_value: float | None,
    measured_band: str,
    measured_value: float | None,
    truth: dict,
) -> tuple[str, str]:
    """Compare a planted truth against what the library reported."""
    expected_band = planted["band"]

    if expected_band == "insufficient_evidence":
        if measured_band == "insufficient_evidence":
            return (
                "MATCH",
                "the planted truth also reads this corpus as too small for this "
                "dimension",
            )
        return (
            "OVER-CLAIM",
            f"the planted truth says this corpus is too small to carry this "
            f"dimension, but the library reported '{measured_band}'",
        )
    if measured_band == "insufficient_evidence":
        return (
            "NOT-DETECTED",
            f"planted '{expected_band}' but the library reported insufficient_evidence",
        )
    if measured_value is None:
        return "MISMATCH", f"band {measured_band} but value is None"

    value_ok = expected_value is None or abs(measured_value - expected_value) <= max(
        0.02 * abs(expected_value), 0.02
    )
    band_ok = measured_band == expected_band

    # For reply latency, also report whether the library's number matches the
    # OTHER quantity it could have computed (subject-to-subject gap).
    alt = ""
    if planted.get("gt_key") == "reply_latency_median":
        sub = truth.get("subject_to_subject_gap_median")
        if sub is not None and abs(measured_value - sub) <= 0.5:
            alt = (
                f" [the reported value equals the SUBJECT-TO-SUBJECT gap "
                f"({sub:.1f} min), not the reply latency]"
            )

    if band_ok and value_ok:
        return "MATCH", alt.strip()
    if not band_ok and value_ok:
        return (
            "WRONG-BAND",
            f"value {measured_value:.4f} matches the planted statistic, but the band "
            f"is '{measured_band}' where the planted truth is '{expected_band}'" + alt,
        )
    if band_ok and not value_ok:
        return (
            "MISMATCH",
            f"band '{measured_band}' matches, but the reported value {measured_value:.4f} "
            f"disagrees with the planted statistic {expected_value}" + alt,
        )
    return (
        "MISMATCH",
        f"band '{measured_band}' vs planted '{expected_band}'; value {measured_value:.4f} "
        f"vs planted {expected_value}" + alt,
    )


def main() -> None:
    paths = sorted(
        [p for p in (ROOT / "corpus").glob("*.json")]
        + [p for p in (ROOT / "probes").glob("*.json")]
    )
    ledger: dict[str, dict] = {}

    for path in paths:
        payload = load(path)
        name = payload["persona"]
        truth = stats(to_messages(payload))
        corpus = build_corpus(payload)
        features = extract_features(corpus)
        predictor = build_predictor(corpus)

        rows = []
        for reading in features.readings:
            key = DIMENSION_KEYS.get(reading.dimension, reading.dimension.lower())
            planted = PLANTED.get(name, {}).get(key)
            if planted is None:
                rows.append(
                    {
                        "dimension": reading.dimension,
                        "planted_band": None,
                        "planted_value": None,
                        "measured_band": reading.band,
                        "measured_value": reading.value,
                        "measured_detail": reading.detail,
                        "n_evidence": reading.n_evidence,
                        "n_conversations": reading.n_conversations,
                        "verdict": "NO-PLANTED-TRUTH",
                        "note": reading.detail,
                    }
                )
                continue
            expected_value = truth.get(planted["gt_key"])
            verdict, note = verdict_for(planted, expected_value, reading.band, reading.value, truth)
            rows.append(
                {
                    "dimension": reading.dimension,
                    "planted_band": planted["band"],
                    "planted_value": expected_value,
                    "planted_why": planted.get("why", ""),
                    "measured_band": reading.band,
                    "measured_value": reading.value,
                    "measured_detail": reading.detail,
                    "n_evidence": reading.n_evidence,
                    "n_conversations": reading.n_conversations,
                    "verdict": verdict,
                    "note": note,
                }
            )

        predictions = [
            {
                "kind": pred.kind,
                "label": pred.label,
                "value": pred.value,
                "low": pred.low,
                "high": pred.high,
                "n_evidence": pred.n_evidence,
                "caveat": pred.caveat,
            }
            for pred in predictor.all_predictions()
        ]

        ledger[name] = {
            "path": str(path.relative_to(ROOT)),
            "n_subject_messages": truth["n_subject_messages"],
            "n_observer_messages": truth["n_observer_messages"],
            "n_conversations": truth["n_conversations"],
            "ground_truth": {
                k: v for k, v in truth.items() if not k.endswith("_ids") and k != "message_lengths"
            },
            "readings": rows,
            "predictions": predictions,
            "probe_expectation": PROBE_EXPECTATIONS.get(name),
        }

    out = ROOT / "ledger.json"
    out.write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")

    tally: dict[str, int] = {}
    for entry in ledger.values():
        for row in entry["readings"]:
            tally[row["verdict"]] = tally.get(row["verdict"], 0) + 1
    print("VERDICT TALLY:", tally)
    print()
    for name, entry in ledger.items():
        print(f"=== {name} ===")
        for row in entry["readings"]:
            if row["verdict"] == "NO-PLANTED-TRUTH":
                print(f"  {row['dimension']:<16} {row['measured_band']:<22} (no planted truth)")
                continue
            flag = "" if row["verdict"] == "MATCH" else f"  <<< {row['verdict']}"
            mv = row["measured_value"]
            mvs = "None" if mv is None else f"{mv:.4f}"
            pv = row["planted_value"]
            pvs = "None" if pv is None else f"{pv:.4f}"
            print(
                f"  {row['dimension']:<16} planted={row['planted_band']:<10} "
                f"({pvs})  measured={row['measured_band']:<22} ({mvs})  "
                f"[{row['verdict']}]{flag}"
            )
            if row["note"]:
                print(f"      {row['note']}")
        print()


# ---------------------------------------------------------------------------
# G-14: a held-out run against a frozen, separately-authored corpus set.
#
# The discipline the gap demands: a second author writes NEW chat corpora and
# their hand truth, the labels are frozen (each corpus's SHA-256 recorded in
# the labels file), and only then does the library run. This runner refuses a
# corpus whose bytes changed after its labels were frozen — a label set that
# could be tuned against results is not a pre-registered one. The verdicts
# reuse `verdict_for`, so the held-out tally is the same vocabulary as the
# in-sample tally; the point is that these rows were never used to fix
# anything.
# ---------------------------------------------------------------------------

HOLDOUT_DIR = ROOT / "holdout"
HOLDOUT_LABELS = HOLDOUT_DIR / "labels.json"
HOLDOUT_LEDGER = ROOT / "holdout_ledger.json"

#: The runner owns the value-key vocabulary; the author never sees it. These
#: name the independent calculator's statistics, so a value can be compared
#: alongside the label band without the frozen labels naming internals.
HOLDOUT_GT_KEYS = {
    "Response length": "subject_mean_length",
    "Emoji usage": "emoji_rate",
    "Questions": "question_rate",
    "Initiation": "initiation_rate",
    "Reply latency": "reply_latency_median",
    "Vocabulary": "vocabulary_msttr",
}


def _labels_kinds(planned_dimensions: dict) -> list[str]:
    """Every label the author wrote, kept verbatim for the ledger."""
    return sorted(planned_dimensions)


def run_holdout(verbose: bool = True) -> int:
    """Measure the frozen holdout set and write `holdout_ledger.json`.

    Returns a shell exit code: 2 when the freeze contract is broken or the
    set is absent, 0 when the run completed — the run's results are recorded
    whatever they are, and Moving thresholds to change them is forbidden
    (the house rule).
    """
    if not HOLDOUT_DIR.is_dir() or not HOLDOUT_LABELS.is_file():
        print(f"G-14: no frozen holdout set at {HOLDOUT_DIR} (labels.json absent)")
        return 2
    labels = json.loads(HOLDOUT_LABELS.read_text(encoding="utf-8"))
    planned: dict = labels.get("corpora", {})
    corpus_paths = sorted(p for p in HOLDOUT_DIR.glob("*.json") if p.name != "labels.json")
    if not corpus_paths:
        print("G-14: a holdout directory exists but holds no corpora yet")
        return 2

    # -- the freeze contract ------------------------------------------------
    present = [p.stem for p in corpus_paths]
    hashes = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for p in corpus_paths}
    stale = []
    for stem in present:
        frozen_hash = planned.get(stem, {}).get("corpus_sha256")
        if frozen_hash != hashes[stem]:
            stale.append(
                f"{stem}: labels froze {str(frozen_hash)[:12]}… but the file now "
                f"hashes {hashes[stem][:12]}… — it changed after its labels "
                "were frozen"
            )
    unlabelled = [stem for stem in present if stem not in planned]
    ghost = [stem for stem in planned if stem not in present]
    if stale or unlabelled or ghost:
        print("G-14: REFUSING to run — the freeze contract is broken:")
        for line in stale:
            print(f"  stale corpus: {line}")
        for line in unlabelled:
            print(f"  unlabelled: {line} has no frozen labels")
        for line in ghost:
            print(f"  missing: {line} is labelled but has no corpus file")
        return 2

    # -- measure ------------------------------------------------------------
    ledger: dict[str, dict] = {}
    tally: dict[str, int] = {}
    for path in corpus_paths:
        stem = path.stem
        planned_entry = planned[stem]
        payload = load(path)
        truth = stats(to_messages(payload))
        corpus = build_corpus(payload)
        features = extract_features(corpus)
        reading_by_dimension = {r.dimension: r for r in features.readings}
        predictions = [
            {
                "kind": pred.kind,
                "label": pred.label,
                "value": pred.value,
                "low": pred.low,
                "high": pred.high,
                "n_evidence": pred.n_evidence,
                "caveat": pred.caveat,
            }
            for pred in build_predictor(corpus).all_predictions()
        ]
        rows = []
        for dimension, planted in planned_entry["dimensions"].items():
            reading = reading_by_dimension.get(dimension)
            if reading is None:
                rows.append(
                    {
                        "dimension": dimension,
                        "planted_band": planted["band"],
                        "planted_value": None,
                        "planted_why": planted.get("why", ""),
                        "measured_band": None,
                        "measured_value": None,
                        "measured_detail": "the library emitted no reading for "
                        "this dimension at all",
                        "n_evidence": None,
                        "n_conversations": None,
                        "verdict": "MISMATCH",
                        "note": "no reading emitted",
                    }
                )
                verdict = "MISMATCH"
            else:
                expected_value = truth.get(HOLDOUT_GT_KEYS[dimension])
                verdict, note = verdict_for(
                    planted, expected_value, reading.band, reading.value, truth
                )
                rows.append(
                    {
                        "dimension": dimension,
                        "planted_band": planted["band"],
                        "planted_value": expected_value,
                        "planted_why": planted.get("why", ""),
                        "measured_band": reading.band,
                        "measured_value": reading.value,
                        "measured_detail": reading.detail,
                        "n_evidence": reading.n_evidence,
                        "n_conversations": reading.n_conversations,
                        "verdict": verdict,
                        "note": note,
                    }
                )
            tally[verdict] = tally.get(verdict, 0) + 1
        for dimension in sorted(set(reading_by_dimension) - set(planned_entry["dimensions"])):
            reading = reading_by_dimension[dimension]
            rows.append(
                {
                    "dimension": dimension,
                    "planted_band": None,
                    "planted_value": None,
                    "measured_band": reading.band,
                    "measured_value": reading.value,
                    "measured_detail": reading.detail,
                    "n_evidence": reading.n_evidence,
                    "n_conversations": reading.n_conversations,
                    "verdict": "UNLABELLED",
                    "note": "the author froze no label for this dimension",
                }
            )
            tally["UNLABELLED"] = tally.get("UNLABELLED", 0) + 1

        ledger[stem] = {
            "path": str(path.relative_to(ROOT)),
            "corpus_sha256": hashes[stem],
            "persona": planned_entry.get("persona", stem),
            "n_subject_messages": truth["n_subject_messages"],
            "n_observer_messages": truth["n_observer_messages"],
            "n_conversations": truth["n_conversations"],
            "ground_truth": {
                k: v for k, v in truth.items() if not k.endswith("_ids") and k != "message_lengths"
            },
            "readings": rows,
            "predictions": predictions,
        }

    # -- the headline numbers, stated with their denominators ---------------
    labelled = [row for entry in ledger.values() for row in entry["readings"]]
    graded = [row for row in labelled if row["planted_band"] not in (None, "insufficient_evidence")]
    withheld = [row for row in labelled if row["planted_band"] == "insufficient_evidence"]
    matches = sum(1 for row in labelled if row["verdict"] == "MATCH")
    graded_matches = sum(1 for row in graded if row["verdict"] == "MATCH")
    withheld_matches = sum(1 for row in withheld if row["verdict"] == "MATCH")

    summary = {
        "run": "G-14 held-out evaluation (frozen, separately-authored labels)",
        "labels_file_sha256": hashlib.sha256(HOLDOUT_LABELS.read_bytes()).hexdigest(),
        "provenance": labels.get("provenance", {}),
        "n_corpora": len(ledger),
        "verdict_tally": tally,
        "labelled_rows": len(labelled),
        "graded_rows": len(graded),
        "graded_matches": graded_matches,
        "graded_agreement": round(graded_matches / len(graded), 4) if graded else None,
        "withheld_rows": len(withheld),
        "withheld_matches": withheld_matches,
        "withheld_agreement": round(withheld_matches / len(withheld), 4) if withheld else None,
        "overall_agreement": round(matches / len(labelled), 4) if labelled else None,
        "disclaimer": (
            "AI-authored corpora, disclosed as such. The labels were frozen "
            "before the library ran; the corpus hashes above prove it. None of "
            "these numbers is a claim about real people."
        ),
    }
    ledger_out = {"summary": summary, "corpora": ledger}
    HOLDOUT_LEDGER.write_text(json.dumps(ledger_out, indent=2, default=str), encoding="utf-8")

    if verbose:
        print("G-14 HELD-OUT RUN (frozen labels, AI-authored, disclosed)")
        print(f"  labels sha256: {summary['labels_file_sha256'][:16]}…")
        print(f"  verdict tally: {tally}")
        print(
            f"  graded rows ({len(graded)}): agreement "
            f"{summary['graded_agreement']}"
        )
        print(
            f"  withheld rows ({len(withheld)}): agreement "
            f"{summary['withheld_agreement']}"
        )
        print(f"  overall: {summary['overall_agreement']} over {len(labelled)} labelled rows")
        print()
        for stem, entry in ledger.items():
            print(f"=== {stem} ===")
            for row in entry["readings"]:
                if row["verdict"] == "UNLABELLED":
                    print(f"  {row['dimension']:<16} (no frozen label)")
                    continue
                mv = row["measured_value"]
                mvs = "None" if mv is None else f"{mv:.4f}"
                pv = row["planted_value"]
                pvs = "None" if pv is None else f"{pv:.4f}"
                flag = "" if row["verdict"] == "MATCH" else f"  <<< {row['verdict']}"
                print(
                    f"  {row['dimension']:<16} frozen={row['planted_band']:<10} ({pvs})  "
                    f"measured={row['measured_band']:<22} ({mvs})  "
                    f"[{row['verdict']}]{flag}"
                )
                if row["note"]:
                    print(f"      {row['note']}")
            print()
    return 0


if __name__ == "__main__":
    if "--holdout" in sys.argv:
        raise SystemExit(run_holdout())
    main()
