"""Lineage: the trust assumption, made safe by default (PLAN.md §6.1).

Core cannot verify that two evidence rows are copies of the same conversation.
An adapter that mints a fresh ``lineage_id`` per row defeats the whole
mechanism, and the README's promise would be false.

So this module ships a *safe default* — a normalised-content digest — so an
adapter that has nothing better gets copy-detection for free rather than
silently opting out.

**What this does not solve.** Cross-modality lineage is not solved and is not
claimed. A screenshot and the ``.json`` export of the same conversation are
near-duplicates, not byte-identical, and detecting that is fuzzy matching. It
is the adapter's job; TRACE offers no heuristic and the README says so.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

__all__ = ["fingerprint"]

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def _normalise(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace.

    Deliberately lossy: the goal is that a message and a re-typed copy of it
    land on the same fingerprint, so the normalisation has to survive
    punctuation and spacing differences without collapsing distinct messages
    into one.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = _PUNCTUATION.sub(" ", folded)
    return _WHITESPACE.sub(" ", folded).strip()


def fingerprint(text: str) -> str:
    """A normalised-content digest of ``text``.

    Two evidence rows whose text normalises identically share a ``lineage_id``,
    so a re-uploaded conversation corroborates once rather than twice.

    Usage::

        lineage_id = lineage.fingerprint(evidence.text)
    """
    normalised = _normalise(text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]
