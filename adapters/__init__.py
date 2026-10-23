"""Reference adapters: source formats in, TRACE's public shapes out.

An adapter does three things, and nothing else:

1. **Normalises** a source format into :class:`trace_traits.Message` rows —
   the integration point the README's examples start from.
2. **Asserts lineage** — which rows were independent, by a strategy the
   adapter states outright (DM-003.1) rather than one the library guessed.
3. **States the clock** — what its timestamps mean, and what ``None`` means,
   in the source's own terms (DM-003.3, CDI-009).

Everything downstream — corroboration counting, the acceptance floor, band
vocabulary — belongs to TRACE and is deliberately not duplicated here.
See the module docstring of :mod:`adapters.mbox` for what an mbox export
asserts and what it declines to see.
"""

from __future__ import annotations

from adapters.mbox import (
    MBOX_LINEAGE_STRATEGY,
    mbox_to_corpus,
    mbox_to_evidence,
    mbox_to_messages,
)

__all__ = [
    "MBOX_LINEAGE_STRATEGY",
    "mbox_to_corpus",
    "mbox_to_evidence",
    "mbox_to_messages",
]
