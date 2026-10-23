"""The corpus: messages, and the evidence they become.

This is the layer where a failure mode of the pre-port embedding is designed
out, and the lesson generalises.

That system derived a conversation's identity from an optional upload field
and defaulted it per person, so every upload for a person collapsed into a
single thread; corroboration then counted one lineage forever and no trait
claim could ever be promoted. The specifics are recorded in the private
integration notes; the general lesson is here.

The fix is not clever: **the corpus carries an explicit conversation key, and
lineage is derived from it.** Where the source does not supply one, the
strategy is declared rather than guessed — see :class:`LineageStrategy`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from trace_core import Evidence
from trace_core.lineage import fingerprint

__all__ = [
    "Corpus",
    "LineageStrategy",
    "Message",
    "Person",
    "corpus_from_messages",
    "evidence_from_corpus",
]

_WORD = re.compile(r"[\w']+", re.UNICODE)


class LineageStrategy(StrEnum):
    """How a message's independence group is decided.

    The choice is declared, never inferred silently, because lineage is the
    trust assumption the whole corroboration mechanism rests on (PLAN.md
    §6.1).
    """

    #: Use the message's ``conversation_id`` verbatim. Correct when the source
    #: knows which conversation a message belongs to.
    CONVERSATION = "conversation"
    #: Treat each upload as its own independent source. Correct for a source
    #: that ingests one conversation per upload and does not label them.
    PER_UPLOAD = "per_upload"
    #: Derive independence from content, so a re-uploaded conversation
    #: corroborates once. Safe but blind to near-duplicates.
    BY_FINGERPRINT = "by_fingerprint"


@dataclass(frozen=True, slots=True)
class Person:
    """Someone whose behaviour is being described.

    ``subject`` is the identity the claims are *about*; ``observer`` is the
    person the corpus is being read by, if there is one. The observer's own
    writing is the baseline a trait like "replies briefly" is measured
    against — "short" means short relative to this pairing, not short in the
    abstract.
    """

    subject: str
    observer: str | None = None


@dataclass(frozen=True, slots=True)
class Message:
    """One message, as a corpus reads it."""

    message_id: str
    conversation_id: str
    #: Whose message this is — the subject, the observer, or a third party.
    speaker: str
    text: str
    sent_at: datetime | None = None
    #: The upload this message came from, when the source has one.
    upload_id: str | None = None
    #: Provenance: bbox, URL, id. Free-form.
    source_id: str | None = None

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(_WORD.findall(self.text))

    @property
    def char_length(self) -> int:
        return len(self.text.strip())

    @property
    def word_count(self) -> int:
        return len(self.words)


@dataclass(frozen=True, slots=True)
class Corpus:
    """A person's messages, with the provenance needed to score them."""

    person: Person
    messages: tuple[Message, ...]
    #: How lineage was decided. Recorded so a score's independence assumption is
    #: readable, not just asserted.
    lineage_strategy: LineageStrategy = LineageStrategy.CONVERSATION
    #: Cache of conversation digests, for the fingerprint strategy.
    _fingerprints: dict[str, str] = field(default_factory=dict, compare=False, repr=False)
    #: Cache of per-conversation text, so the digest is not rebuilt per call.
    _conversation_texts: dict[str, tuple[str, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    #: The corpus grouped by conversation, computed at construction.
    _conversation_index: dict[str, tuple[Message, ...]] = field(
        default_factory=dict, compare=False, repr=False, init=False
    )

    def for_speaker(self, speaker: str) -> tuple[Message, ...]:
        return tuple(item for item in self.messages if item.speaker == speaker)

    @property
    def subject_messages(self) -> tuple[Message, ...]:
        return self.for_speaker(self.person.subject)

    @property
    def observer_messages(self) -> tuple[Message, ...]:
        if self.person.observer is None:
            return ()
        return self.for_speaker(self.person.observer)

    def __post_init__(self) -> None:
        grouped: dict[str, list[Message]] = {}
        for item in self.messages:
            grouped.setdefault(item.conversation_id, []).append(item)
        object.__setattr__(
            self, "_conversation_index", {key: tuple(value) for key, value in grouped.items()}
        )

    @property
    def conversations(self) -> tuple[str, ...]:
        """Distinct conversation keys, in first-seen order."""
        seen: dict[str, None] = {}
        for item in self.messages:
            seen.setdefault(item.conversation_id, None)
        return tuple(seen)

    def lineage_of(self, message: Message) -> str:
        """The independence group a message belongs to.

        ``BY_FINGERPRINT`` fingerprints the **conversation**, not the message.
        Fingerprinting the message would give every distinct message its own
        lineage, so fourteen messages in one conversation would count as
        fourteen independent sources — which is exactly the corroboration
        inflation the whole mechanism exists to prevent. A conversation-level
        digest catches a re-uploaded conversation while still treating one
        conversation as one source.
        """
        match self.lineage_strategy:
            case LineageStrategy.CONVERSATION:
                return message.conversation_id
            case LineageStrategy.PER_UPLOAD:
                return message.upload_id or message.conversation_id
            case LineageStrategy.BY_FINGERPRINT:
                return self._conversation_fingerprint(message.conversation_id)

    def __len__(self) -> int:
        return len(self.messages)

    @property
    def _by_conversation(self) -> dict[str, tuple[Message, ...]]:
        """Every conversation's messages, grouped once.

        Built lazily and cached. An audit measured the fingerprint strategy at
        13 seconds for 50,000 messages because each conversation's text was
        rebuilt by scanning the whole corpus — quadratic in the corpus size.
        """
        cached = self._conversation_index
        if cached is not None:
            return cached
        grouped: dict[str, list[Message]] = {}
        for item in self.messages:
            grouped.setdefault(item.conversation_id, []).append(item)
        index = {key: tuple(value) for key, value in grouped.items()}
        self._conversation_index = index
        return index

    def _conversation_fingerprint(self, conversation_id: str) -> str:
        """A digest of one conversation's text, computed once and cached.

        The digest covers every message in the conversation, so a re-upload of
        the same conversation lands on the same lineage even when individual
        messages differ slightly (a truncated screenshot, an OCR difference).

        The per-conversation text is also cached, because rebuilding it on every
        call made the whole thing quadratic: an audit measured 13 seconds for
        50,000 messages across 5,000 conversations.
        """
        cached = self._fingerprints.get(conversation_id)
        if cached is not None:
            return cached
        texts = self._conversation_texts.get(conversation_id)
        if texts is None:
            texts = tuple(
                sorted(item.text for item in self._by_conversation.get(conversation_id, ()))
            )
            self._conversation_texts[conversation_id] = texts
        digest = fingerprint("\n".join(texts))
        self._fingerprints[conversation_id] = digest
        return digest


def corpus_from_messages(
    person: Person,
    messages: Iterable[Message],
    *,
    lineage_strategy: LineageStrategy = LineageStrategy.CONVERSATION,
) -> Corpus:
    """Build a corpus, ordering messages by time where it is known.

    Dated messages sort first, by timestamp; undated messages follow, keeping
    their relative order. A message with no visible timestamp cannot be placed
    in time, so it takes no seat in the dated exchange — dropping it would
    silently shrink the sample every trait is measured from, and putting it
    first would let an unpositioned message speak for how the conversation
    opened. The measures that read timestamps simply never see it; the count
    and text measures still do.
    """
    # A *mixture* of naive and aware timestamps makes the sort below raise a
    # TypeError from deep inside the comparison — an opaque failure that names
    # neither the problem nor the corpus. It is checked here, where the message
    # can say what is wrong.
    #
    # Naive timestamps on their own are **not** an error. A naive timestamp is
    # interpreted as UTC, which is what a chat export means by it, and a
    # corpus of them is perfectly usable. Only the mixture is ambiguous.
    aware = [
        item for item in messages if item.sent_at is not None and item.sent_at.tzinfo is not None
    ]
    naive = [item for item in messages if item.sent_at is not None and item.sent_at.tzinfo is None]
    if aware and naive:
        raise ValueError(
            f"corpus mixes {len(aware)} timezone-aware timestamps with "
            f"{len(naive)} naive ones; every timestamp must be the same kind"
        )
    ordered = sorted(
        messages, key=lambda item: (item.sent_at is None, item.sent_at or datetime.min)
    )
    return Corpus(
        person=person,
        messages=tuple(ordered),
        lineage_strategy=lineage_strategy,
    )


def evidence_from_corpus(
    corpus: Corpus,
    *,
    source_type: str = "message",
    modality: str = "chat",
) -> tuple[Evidence, ...]:
    """Turn a corpus into TRACE evidence.

    ``lineage_id`` comes from the corpus's declared strategy, so the
    independence assumption behind every later score is inspectable on the
    evidence itself.
    """
    out: list[Evidence] = []
    for message in corpus.messages:
        out.append(
            Evidence(
                evidence_id=message.message_id,
                segment_id=message.conversation_id,
                lineage_id=corpus.lineage_of(message),
                source_id=message.source_id or message.message_id,
                text=message.text,
                source_type=source_type,
                modality=modality,
                occurred_at=message.sent_at,
                speaker=message.speaker,
                metadata={
                    "conversation_id": message.conversation_id,
                    "upload_id": message.upload_id,
                    "word_count": message.word_count,
                },
            )
        )
    return tuple(out)


def corpus_lineages(corpus: Corpus) -> Mapping[str, str]:
    """``{evidence_id: lineage_id}`` for the whole corpus."""
    return {item.message_id: corpus.lineage_of(item) for item in corpus.messages}


def count_conversations(messages: Sequence[Message]) -> int:
    """Distinct conversations among messages."""
    return len({item.conversation_id for item in messages})
