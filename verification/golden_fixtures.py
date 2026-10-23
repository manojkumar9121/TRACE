"""Freeze and verify golden fixtures for every dimension over every corpus.

PLAN.md milestone **M1**: current outputs snapshotted **per term**, before
anything moves. The snapshot lives in ``verification/golden.json`` and holds,
for each corpus and probe, every T1 reading (band, value, evidence counts,
detail) and every T2 prediction the library currently produces. The point is
the one the milestone states: when a threshold, a gate, or a scorer changes,
the fixture changes with it — and the diff is visible in review rather than
appearing silently in somebody else's numbers later.

The file carries its own SHA-256 over the readings section, so a hand-edited
"the audit passed, honestly" fixture is detectable rather than honoured:

    python3 verification/golden_fixtures.py            # verify against the snapshot
    python3 verification/golden_fixtures.py --freeze   # re-snapshot current outputs

A frozen snapshot is not an oracle — correctness lives with the audit, the
ground truth and the tests. It records *what the library said*, so any change
to what it says is a deliberate, reviewable act.
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

from ground_truth import load  # noqa: E402

from trace_traits import (  # noqa: E402
    Message,
    Person,
    build_predictor,
    corpus_from_messages,
    extract_features,
)

GOLDEN = ROOT / "golden.json"


def build_corpus(payload: dict) -> tuple:
    """A corpus from a payload — the same shape ``run_verification`` uses."""
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
    subject = payload.get("subject", "them")
    observer = payload.get("observer", "me")
    person = Person(subject=subject, observer=observer) if observer else Person(subject=subject)
    return corpus_from_messages(person, messages)


def snapshot() -> dict:
    """What the library currently says, per corpus, per dimension, per term."""
    paths = sorted(
        [p for p in (ROOT / "corpus").glob("*.json")]
        + [p for p in (ROOT / "probes").glob("*.json")]
    )
    output: dict[str, dict] = {}
    for path in paths:
        payload = load(path)
        name = payload["persona"]
        corpus = build_corpus(payload)
        readings = {}
        for reading in extract_features(corpus).readings:
            readings[reading.dimension] = {
                "band": reading.band,
                "value": reading.value,
                "n_evidence": reading.n_evidence,
                "n_conversations": reading.n_conversations,
                "detail": reading.detail,
            }
        predictions = {
            f"{prediction.kind}/{prediction.label}": {
                "value": prediction.value,
                "low": prediction.low,
                "high": prediction.high,
                "n_evidence": prediction.n_evidence,
                "caveat": prediction.caveat,
            }
            for prediction in build_predictor(corpus).all_predictions()
        }
        output[name] = {"readings": readings, "predictions": predictions, "path": str(path.relative_to(ROOT))}
    return output


def digest(snapshot_map: dict) -> str:
    """SHA-256 over the canonical form of the snapshot's numbers."""
    body = json.dumps(
        {name: entry for name, entry in snapshot_map.items()},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def freeze() -> None:
    snap = snapshot()
    payload = {
        "purpose": "golden fixture: what the library says today (PLAN.md M1)",
        "sha256": digest(snap),
        "snapshot": snap,
    }
    GOLDEN.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"frozen {len(snap)} corpora → {GOLDEN} (sha256 {payload['sha256'][:16]}…)")


def verify() -> int:
    """Recompute and compare. Returns a shell exit code."""
    frozen = json.loads(GOLDEN.read_text(encoding="utf-8"))
    recorded = frozen["sha256"]
    body = frozen["snapshot"]
    if digest(body) != recorded:
        print(f"REFUSING: {GOLDEN}'s checksum does not match its contents")
        return 2
    snap = snapshot()
    drift = []
    for name, current in snap.items():
        if name not in body:
            drift.append(f"{name}: not in the frozen snapshot")
            continue
        for section in ("readings", "predictions"):
            for term, value in current[section].items():
                if term not in body[name][section]:
                    drift.append(f"{name} {term}: new term")
                    continue
                past = body[name][section][term]
                for key, now_value in value.items():
                    if past.get(key) != now_value:
                        drift.append(
                            f"{name} {term} {key}: was {past.get(key)!r}, now {now_value!r}"
                        )
    if drift:
        print(f"DRIFT: {len(drift)} terms differ from the frozen snapshot:")
        for line in drift:
            print(f"  {line}")
        return 1
    print(f"GOLDEN OK: {len(snap)} corpora match the frozen snapshot")
    return 0


def main() -> int:
    if "--freeze" in sys.argv:
        freeze()
        return 0
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
