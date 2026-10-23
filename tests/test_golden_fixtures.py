"""PLAN.md milestone M1 — golden fixtures.

The milestone's point: current outputs snapshotted **per term** before
anything moves, so a threshold, gate or scorer change shows up as a visible,
reviewed diff rather than a silent numeric drift. The frozen snapshot lives in
``verification/golden.json`` (regenerate with
``python3 verification/golden_fixtures.py --freeze``); these tests pin that
what the library says today is exactly what was frozen, and that the snapshot
file has not been hand-edited since it was frozen (SHA-256 over its own
numbers).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from trace_traits import (
    Message,
    Person,
    build_predictor,
    corpus_from_messages,
    extract_features,
)

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "verification" / "golden.json"

CORPUS_DIR = ROOT / "verification" / "corpus"
PROBE_DIR = ROOT / "verification" / "probes"


def _payloads() -> dict[str, dict[str, Any]]:
    payloads = {}
    for path in sorted(CORPUS_DIR.glob("*.json")) + sorted(PROBE_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[payload["persona"]] = payload
    return payloads


def _corpus(payload: dict[str, Any]) -> Any:
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


def _computed() -> dict[str, dict[str, dict[str, Any]]]:
    """Every T1 reading and T2 prediction, as the library currently says them."""
    output = {}
    for name, payload in _payloads().items():
        corpus = _corpus(payload)
        readings = {
            reading.dimension: {
                "band": reading.band,
                "value": reading.value,
                "n_evidence": reading.n_evidence,
                "n_conversations": reading.n_conversations,
                "detail": reading.detail,
            }
            for reading in extract_features(corpus).readings
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
        output[name] = {"readings": readings, "predictions": predictions}
    return output


def _canonical(mapping: Any) -> str:
    return json.dumps(mapping, sort_keys=True, separators=(",", ":"), default=str)


@pytest.fixture(scope="module")
def frozen() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_the_snapshot_file_has_not_been_hand_edited(frozen: dict[str, Any]) -> None:
    """The checksum is recomputed from the file's own numbers."""
    recorded = frozen["sha256"]
    body = _canonical(frozen["snapshot"])
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == recorded


def test_every_corpus_matches_the_frozen_snapshot(frozen: dict[str, Any]) -> None:
    """Band, value and counts must reproduce exactly as frozen."""
    current = _computed()
    drift: list[str] = []
    for name, computed in current.items():
        assert name in frozen["snapshot"], f"{name} is missing from the snapshot"
        frozen_entry = frozen["snapshot"][name]
        for section in ("readings", "predictions"):
            for term, value in computed[section].items():
                assert term in frozen_entry[section], f"{name} {term} is new since the freeze"
                past = frozen_entry[section][term]
                for key, now_value in value.items():
                    if past.get(key) != now_value:
                        drift.append(
                            f"{name} {term} {key}: frozen {past.get(key)!r}, computed {now_value!r}"
                        )
    assert drift == [], "\n".join(drift)


def test_no_frozen_term_has_disappeared(frozen: dict[str, Any]) -> None:
    """Removal is also drift: a term must never be quietly dropped."""
    current = _computed()
    for name, frozen_entry in frozen["snapshot"].items():
        assert name in current, f"{name} no longer measured"
        for section in ("readings", "predictions"):
            assert section in frozen_entry, f"{name} {section} missing from the snapshot"
