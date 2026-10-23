"""Errors that are failures, not answers (PLAN.md §3, INV-4)."""

from __future__ import annotations

__all__ = ["CitationError", "PolicyError"]


class CitationError(Exception):
    """A citation names something the evidence set does not contain.

    Raised rather than returned as a "candidate" decision: silently dropping a
    broken citation is how a broken pipeline becomes a quiet one.
    """


class PolicyError(Exception):
    """A policy returned something the contract forbids (PLAN.md §4.3)."""
