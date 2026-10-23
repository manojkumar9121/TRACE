"""Time, and the vocabulary that goes with it (PLAN.md §2.6).

The three scope enums the original embedding imported from a shared package
are re-declared here rather than imported, so ``trace_core`` has no dependency
outside the standard library. That is PLAN.md's M0 requirement, and this
module is where it is met.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

__all__ = ["Freshness", "FreshnessState", "TemporalScope", "age_days", "decay"]


class TemporalScope(StrEnum):
    """How current a claim is allowed to be asserted as.

    ``PRESENT`` is the only value the freshness gate can forbid, and it is
    downgraded rather than dropped: an aged persona can still support a
    statement about how someone wrote last March.
    """

    PRESENT = "present"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


class FreshnessState(StrEnum):
    """Per-person freshness, measured from the newest accepted evidence."""

    FRESH = "fresh"
    AGING = "aging"
    STALE = "stale"
    ABSENT = "absent"


#: ``Freshness`` is the public name in PLAN.md §2's ``__all__``;
#: ``FreshnessState`` is the pre-port name for the same four states. Both are
#: exported so either spelling resolves to the same enum.
Freshness = FreshnessState


def age_days(now: datetime, observed: datetime) -> float:
    """Days from ``observed`` to ``now``, never negative.

    A naive timestamp is interpreted as **UTC**. A chat export writes naive
    timestamps constantly, and rejecting them would reject most real corpora.
    A *mixture* of naive and aware timestamps within one comparison is the
    ambiguous case, and the caller checks for it before getting here.

    Both sides are normalised to aware UTC before subtracting. A row read back
    from SQLite has no ``tzinfo`` while a row read from Postgres does, and
    subtracting the two raises — which would make decay depend on which
    database a deployment happens to run.
    """
    delta = _as_utc(now) - _as_utc(observed)
    return max(0.0, delta.total_seconds() / 86400.0)


def _as_utc(moment: datetime) -> datetime:
    """Normalise a datetime to aware UTC, reading a naive one as UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def decay(age: float, half_life_days: float) -> float:
    """``0.5 ** (age / half_life_days)`` — the freshness term.

    ``age`` of infinity yields 0.0: an undated observation has no freshness to
    offer, and 1.0 would be a claim about the present that the evidence cannot
    make.
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if age == float("inf"):
        return 0.0
    return float(0.5 ** (age / half_life_days))
