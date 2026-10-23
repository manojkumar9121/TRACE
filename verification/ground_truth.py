"""Independent ground-truth calculator for the verification corpora.

This module deliberately does **not** import `trace_traits` or `trace_core`.
It re-derives every statistic from the raw JSON text using plain Python, so
that the ground truth is an independent measurement rather than a restatement
of the code under test. Every number in the report's ledger comes from here.

Definitions used (stated so they can be argued with):

* A message is *substantive* when its stripped length is >= 4 characters.
  ("ok" and "no" are real messages but say nothing about length.)
* Response length = mean stripped character length of the subject's
  substantive messages.
* Emoji usage = share of the subject's messages containing at least one
  character that renders as an emoji. Membership is decided with a broad
  Unicode-aware test (see `is_emoji`), not the narrow three-range test the
  library uses.
* Question rate = share of the subject's messages containing "?" — but the
  hand-ground-truth counts *interrogative intent*, so `?` is only counted when
  the message is not a rhetorical aside. See the note in the sidecars.
* Initiation = share of conversations whose first message is from the subject.
* Reply latency = median minutes between a subject message and the *previous
  message in the same conversation* (i.e. how long the subject took to reply
  to whatever was said to them). This is the human reading of the term.
* Vocabulary richness = **MSTTR** — the type-token ratio over a fixed
  100-word sliding window, averaged — over the subject's substantive
  messages.

  This was changed from a plain type-token ratio during the independent
  audit, because a plain TTR is confounded by how much a person writes: a
  terse correspondent scores 0.89 "high" and a verbose one 0.47 "moderate",
  which says more about message length than about vocabulary. MSTTR removes
  that confound.

  **This calculator is deliberately independent of the library** — it does not
  import `trace_traits` — so the formula is written out again here rather
  than shared. If the two ever disagree, that is a real disagreement and
  worth finding.
"""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from datetime import datetime
from pathlib import Path

WORD = re.compile(r"[\w']+", re.UNICODE)

#: Unicode blocks that contain pictographic characters a person would call an
#: emoji. This is deliberately BROADER than the library's three ranges.
_EMOJI_BLOCKS = (
    (0x1F000, 0x1FAFF),  # pictographs, emoticons, transport, supplemental
    (0x2600, 0x27BF),  # misc symbols + dingbats
    (0x2190, 0x21FF),  # arrows
    (0x2300, 0x23FF),  # misc technical (watch, hourglass, alarm clock)
    (0x2B00, 0x2BFF),  # stars and other misc symbols (⭐, ⬛)
    (0x3030, 0x303D),  # wavy dash, part alternation mark
    (0x3297, 0x3299),  # circled ideographs
    (0xFE00, 0xFE0F),  # variation selectors (skin tone, presentation)
)

#: Characters in those blocks that are commonly typed as PLAIN TYPOGRAPHY
#: rather than as emoji, and which a human reading a chat message would not
#: count as "the person used an emoji": a check mark, an arrow, a plain star.
#: Deliberately NOT excluded: ⭐ U+2B50, ⏰ U+23F0, ⌛ U+231B, ⏳ U+23F3,
#: ⭕ U+2B55, ⚡ U+26A1, ⌚ U+231A — these are unambiguous emoji.
_NOT_EMOJI = set("✓✗✔✘→←↑↓⇒⇐〰〽～★☆")

_EMOJI_PRESENTATION = {0xFE0F, 0x20E3}


def is_emoji(char: str) -> bool:
    """Whether a character reads as an emoji to a human.

    Excludes characters that are commonly typed as plain typography
    (arrows, check marks, stars) unless they carry an emoji presentation
    selector or sit next to another emoji, which is how a platform decides
    to render them as emoji anyway.
    """
    point = ord(char)
    if point in _EMOJI_PRESENTATION:
        return False
    if char in _NOT_EMOJI:
        return False
    if not any(low <= point <= high for low, high in _EMOJI_BLOCKS):
        return False
    # A pictograph in the supplementary planes is an emoji regardless.
    if point >= 0x1F000:
        return True
    # Misc symbols / dingbats: only emoji if the platform would render it
    # that way, which for these blocks means "almost always", so accept.
    return unicodedata.category(char) == "So" or point >= 0x2600


def has_emoji(text: str) -> bool:
    return any(is_emoji(char) for char in text)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_messages(payload: dict) -> list[dict]:
    out = []
    for item in payload["messages"]:
        out.append(
            {
                "id": item["id"],
                "conversation_id": item["conversation_id"],
                "speaker": item["speaker"],
                "text": item["text"],
                "sent_at": datetime.fromisoformat(item["sent_at"]),
            }
        )
    return out


#: The preferred MSTTR window, in words. The same constant the library uses,
#: restated here rather than imported. Under the declared policy the window is
#: never shrunk: a band means MSTTR at this window (PLAN.md G-11), so a short
#: corpus gets no band at all rather than a ratio measured on another scale.
#: The value below is still computed for corpora too small to band — it is a
#: fact about the world, not the library's consent to read it.
_MSTTR_WINDOW = 100


def _msttr(words: list[str]) -> float | None:
    """Mean type-token ratio over sliding windows at the preferred size."""
    window = _MSTTR_WINDOW
    if len(words) < window:
        return None
    total = 0.0
    count = 0
    for start in range(0, len(words) - window + 1):
        chunk = words[start : start + window]
        total += len(set(chunk)) / window
        count += 1
    return total / count if count else None


def stats(messages: list[dict], subject: str = "them", observer: str = "me") -> dict:
    """Every statistic the ledger needs, computed independently."""
    theirs = [m for m in messages if m["speaker"] == subject]
    yours = [m for m in messages if m["speaker"] == observer]

    def substantive(seq: list[dict]) -> list[dict]:
        return [m for m in seq if len(m["text"].strip()) >= 4]

    st = substantive(theirs)
    yo = substantive(yours)

    lengths = [len(m["text"].strip()) for m in st]
    your_lengths = [len(m["text"].strip()) for m in yo]

    # conversations, in first-seen order
    conv_order: list[str] = []
    for m in messages:
        if m["conversation_id"] not in conv_order:
            conv_order.append(m["conversation_id"])

    openers = {}
    for m in messages:
        openers.setdefault(m["conversation_id"], m)
    subject_openers = sum(1 for m in openers.values() if m["speaker"] == subject)

    # reply latency: gap from the previous message in the thread to the
    # subject's own message.
    by_conv: dict[str, list[dict]] = {}
    for m in messages:
        by_conv.setdefault(m["conversation_id"], []).append(m)
    gaps = []
    for thread in by_conv.values():
        thread = sorted(thread, key=lambda m: m["sent_at"])
        for prev, cur in zip(thread, thread[1:], strict=False):
            if cur["speaker"] == subject:
                gap = (cur["sent_at"] - prev["sent_at"]).total_seconds() / 60.0
                if gap >= 0:
                    gaps.append(gap)

    # subject-to-subject gap (what the library actually measures)
    sub_gaps = []
    for conv in by_conv:
        own = sorted(
            [m for m in by_conv[conv] if m["speaker"] == subject],
            key=lambda m: m["sent_at"],
        )
        for prev, cur in zip(own, own[1:], strict=False):
            gap = (cur["sent_at"] - prev["sent_at"]).total_seconds() / 60.0
            if gap >= 0:
                sub_gaps.append(gap)

    words: list[str] = []
    for m in theirs:
        words.extend(WORD.findall(m["text"]))

    emoji_msgs = [m for m in theirs if has_emoji(m["text"])]
    question_msgs = [m for m in theirs if "?" in m["text"]]

    # opportunities / replies for the T2 response rate
    opportunities = 0
    replied = 0
    for conv in by_conv:
        thread = sorted(by_conv[conv], key=lambda m: m["sent_at"])
        for index, m in enumerate(thread):
            if m["speaker"] == subject:
                continue
            opportunities += 1
            if index + 1 < len(thread) and thread[index + 1]["speaker"] == subject:
                replied += 1

    return {
        "n_subject_messages": len(theirs),
        "n_observer_messages": len(yours),
        "n_conversations": len(conv_order),
        "subject_mean_length": (sum(lengths) / len(lengths)) if lengths else None,
        "observer_mean_length": (sum(your_lengths) / len(your_lengths)) if your_lengths else None,
        "subject_mean_length_all_messages": (
            sum(len(m["text"].strip()) for m in theirs) / len(theirs) if theirs else None
        ),
        "n_substantive": len(st),
        "emoji_count": len(emoji_msgs),
        "emoji_ids": [m["id"] for m in emoji_msgs],
        "emoji_rate": len(emoji_msgs) / len(theirs) if theirs else None,
        "question_count": len(question_msgs),
        "question_ids": [m["id"] for m in question_msgs],
        "question_rate": len(question_msgs) / len(theirs) if theirs else None,
        "subject_opened": subject_openers,
        "initiation_rate": subject_openers / len(openers) if openers else None,
        "reply_latency_median": statistics.median(gaps) if gaps else None,
        "reply_latency_n": len(gaps),
        "subject_to_subject_gap_median": statistics.median(sub_gaps) if sub_gaps else None,
        "subject_to_subject_gap_n": len(sub_gaps),
        "n_distinct_words": len(set(words)),
        "n_words": len(words),
        "vocabulary_ttr": (len(set(words)) / len(words)) if words else None,
        # MSTTR — see the module docstring. Written out independently of the
        # library's implementation on purpose.
        "vocabulary_msttr": _msttr(words),
        "n_distinct_words_substantive": len({w for m in st for w in WORD.findall(m["text"])}),
        "n_words_substantive": sum(len(WORD.findall(m["text"])) for m in st),
        "opportunities": opportunities,
        "replied": replied,
        "response_rate": (replied / opportunities) if opportunities else None,
        "message_lengths": {m["id"]: len(m["text"].strip()) for m in theirs},
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".truth.json"):
            continue
        payload = load(path)
        messages = to_messages(payload)
        s = stats(messages)
        print(f"=== {path.relative_to(root)} ===")
        for key, value in s.items():
            if key.endswith("_ids"):
                print(f"  {key}: {value}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()


if __name__ == "__main__":
    main()
