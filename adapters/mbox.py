"""The reference mbox adapter: an email mailbox in, TRACE messages out.

**mbox is the modality the contract names.** PLAN.md CDI-001.1 uses it as its
example: an mbox export is still ``message``-sourced, and ``mbox`` is
provenance — the ``modality`` — never a scoring key. Everything this adapter
emits carries ``source_type="message"`` and ``modality="mbox"``.

**What the adapter asserts, and what TRACE verifies.** An adapter is the one
place where lineage may be *asserted*. This one asserts three things:

1. **Threading** — the conversation a message belongs to, solved from
   ``In-Reply-To`` / ``References``, with a normalised-subject fallback when
   the header context is missing.
2. **Lineage** — the independence group, declared as
   :data:`MBOX_LINEAGE_STRATEGY`, never inferred.
3. **Clock** — a naive parse is a UTC reading (DM-003.3); a missing or
   unparseable ``Date`` header is an *undated* row (CDI-009), never a
   guessed time.

What TRACE verifies afterwards — that corroboration counts each lineage once,
that a claim that cannot reach two lineages stays a candidate — is the
library's, and none of it is re-implemented here.

**Why the fallback message ids are derivable.** A mail without a
``Message-ID`` is common — list digests, relay rewrites, damaged loaders. Its
id becomes a normalised-content digest, the same ``fingerprint`` the library
uses for lineage, over sender, date, subject and body. Two identical copies of
such a mail therefore collide by construction, and the collision is resolved
by keeping the first occurrence and dropping the rest: the same message stored
twice is one message, and letting two stored copies corroborate a claim
independently is exactly the inflation INV-1 forbids.

**What neither TRACE nor this adapter can see.** A mail whose body carries no
``text/plain`` part at all — a photo, an HTML-only mail, an attach-only
bundle — contributes no message: there is nothing citable to quote. A
cross-forward whose body was re-worded or re-quoted under new headers is a
near-duplicate, not a byte copy; no normalised digest can bind it without
fuzzy matching, which the README hands to the integrator. This adapter does
not pretend otherwise.
"""

from __future__ import annotations

import mailbox
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.header import decode_header, make_header
from email.message import Message as _StdMessage
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import cast

from trace_core import Evidence, fingerprint
from trace_traits import (
    Corpus,
    LineageStrategy,
    Message,
    Person,
    corpus_from_messages,
    evidence_from_corpus,
)

__all__ = [
    "MBOX_LINEAGE_STRATEGY",
    "mbox_to_corpus",
    "mbox_to_evidence",
    "mbox_to_messages",
]

#: The lineage strategy this adapter declares for every corpus it builds
#: (DM-003.1: declared, never inferred). Email is what makes the choice a
#: decision:
#:
#: - ``PER_UPLOAD`` is wrong immediately. A mailbox holds many threads, so one
#:   upload equals one lineage and every claim in the corpus scores against a
#:   single source — the collapse the corpus module's own commentary
#:   describes. Worse, a second upload of the same mailbox would count a
#:   second time: multiple deliveries of the same evidence multiply
#:   independence, which is the failure the mechanism exists to prevent.
#: - ``CONVERSATION`` is wrong because an mbox has no conversation key to use
#:   verbatim: the thread identity is *reconstructed* here, from reply
#:   headers, normalised subject tags and deterministic fallbacks. Taking the
#:   reconstruction's own label as the lineage would bless its mistakes as
#:   independence — and a cross-forward, the same message re-sent under a
#:   fresh thread root, would land in its own lineage: a copy of the same
#:   conversation counted as a separate source.
#: - ``BY_FINGERPRINT`` decides independence from what the messages say, not
#:   how they are labelled: a mailbox re-uploaded, re-split across files or
#:   re-tagged by a different subject heuristic digests identically and
#:   corroborates once. DM-003.2's shape is respected — the fingerprint
#:   covers the conversation's whole text, so a mailbox's threads remain
#:   distinct sources, and the copies inside one thread do not become
#:   fourteen independent messages.
MBOX_LINEAGE_STRATEGY = LineageStrategy.BY_FINGERPRINT

_SUBJECT_PREFIX = re.compile(
    r"^\s*(?:re\s*(?:\[\d+\])?\s*[:：]|fwd?\s*[:：]|\[\d+\])\s*", re.IGNORECASE
)
_ANGLE = re.compile(r"<([^<>]+)>")


def _bare_ids(raw: str | None) -> list[str]:
    """Message-ids carried by a header value: brackets gone, repeats kept."""
    if raw is None:
        return []
    return [hit.strip() for hit in _ANGLE.findall(raw) if hit.strip()]


def _normalise_subject(subject: str | None) -> str:
    """``"Re: Fwd: Re[2]: hello"`` -> ``"hello"``, casefolded, spaces collapsed.

    Strips repeatedly, because people pile prefixes. What survives the strip
    is the thread tag the fallback groups by.
    """
    folded = " ".join((subject or "").split())
    while True:
        stripped = _SUBJECT_PREFIX.sub("", folded).strip()
        if stripped == folded:
            break
        folded = stripped
    return folded.casefold()


def _decoded_header(raw: str | None) -> str:
    """RFC 2047-decoded header text, or the raw value if decoding fails.

    A header that will not decode is kept verbatim rather than dropped: a
    subject the decoder mangles still groups its thread deterministically.
    """
    if raw is None:
        return ""
    try:
        return str(make_header(decode_header(raw)))
    except Exception:
        return raw


def _sent_at(raw: str | None) -> datetime | None:
    """The ``Date`` header as a UTC reading, or ``None`` when undated.

    Three seams, each deliberate:

    - **Absent header** -> ``None``: an undated row (CDI-009). It carries no
      decay and no recency, and it never borrows a timestamp the corpus does
      not have.
    - **Unparseable value** -> ``None``: a broken clock is not a clock.
    - **Naive parse** — which includes the RFC ``-0000`` "zone unknown" case,
      which :func:`email.utils.parsedate_to_datetime` returns *without* a
      timezone — is read as UTC, exactly as DM-003.3 prescribes. Attaching
      the zone here is also why this adapter can never hand
      :func:`trace_traits.corpus_from_messages` a mixed naive/aware corpus:
      every timestamp it emits is aware, or undated, and never a mixture.
    """
    if raw is None:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _speaker(raw: str | None) -> str:
    """The sender's address, as the message's ``speaker``.

    Display names are dropped: a name is client-rendered and unstable, the
    address is what a reply answers to. A mail with no ``From`` header at all
    yields ``""`` — the corpus layer cannot express an *unattributed* row
    (``Message.speaker`` is required), and the empty string is the one value
    guaranteed not to collide with a real address. See the adapter report's
    contract findings.
    """
    if raw is None:
        return ""
    addresses = getaddresses([raw])
    for _name, addr in addresses:
        if addr.strip():
            return addr.strip()
    return raw.strip()


def _payload_text(part: _StdMessage) -> str:
    """One text/plain part's decoded body; an unreadable charset degrades."""
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return ""
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def _is_attachment(part: _StdMessage) -> bool:
    """An attachment is a named part, or one explicitly marked.

    The filename test is the load-bearing one: an attached ``.txt`` file is
    ``text/plain`` too, and skipping by content type alone would bury the
    attachment in the body.
    """
    if part.get_filename():
        return True
    return part.get_content_disposition() == "attachment"


def _plain_parts(part: _StdMessage) -> list[str]:
    """Every text/plain *body* part under this part, in header order.

    Multiparts are walked depth-first, in the order the composing client
    chose. Attachment parts are skipped — including ``.txt`` files, see
    :func:`_is_attachment` — and a ``message/rfc822`` subtree is pruned
    outright: an inline stored copy of a message is an attachment and would
    otherwise be read twice.
    """
    if _is_attachment(part) or part.get_content_type() == "message/rfc822":
        return []
    if part.is_multipart():
        payload = cast("str | list[_StdMessage] | None", part.get_payload())
        children: list[_StdMessage] = [kid for kid in payload] if isinstance(payload, list) else []
        return [text for kid in children for text in _plain_parts(kid)]
    if part.get_content_type() == "text/plain":
        return [_payload_text(part)]
    return []


@dataclass(frozen=True, slots=True)
class _RawMail:
    """One parseable mail: everything the adapter asserts about it."""

    message_id: str
    ancestors: tuple[str, ...]
    subject: str
    sender: str
    sent_at: datetime | None
    body: str

    @property
    def fallback_id(self) -> str:
        """The deterministic id a mail without a ``Message-ID`` receives."""
        when = self.sent_at.isoformat() if self.sent_at is not None else ""
        seed = f"{self.sender}|{when}|{self.subject}|{self.body}"
        return f"mbox+{fingerprint(seed)}"

    @property
    def id(self) -> str:
        return self.message_id or self.fallback_id


def _read_mails(path: str | Path) -> list[_RawMail]:
    """Parse the mailbox into raw mails, de-duplicated on the final id.

    The final id is the header's ``Message-ID`` when present, else the
    content digest. Either way, two mails sharing a final id are two stored
    copies of one message; the first occurrence is kept — mbox order, which
    is stable for the same bytes. A mail with no citable text at all (no
    ``text/plain`` body) is dropped here, before it can become an empty
    evidence row.
    """
    box = mailbox.mbox(str(path), create=False)
    raws: list[_RawMail] = []
    try:
        seen: set[str] = set()
        for msg in box:
            id_header = msg.get("Message-ID")
            from_header = msg.get("From")
            date_header = msg.get("Date")
            candidates = _RawMail(
                message_id=next(iter(_bare_ids(str(id_header) if id_header else None)), ""),
                ancestors=tuple(_bare_ids(" ".join(str(v) for v in msg.get_all("In-Reply-To", []))))
                + tuple(_bare_ids(" ".join(str(v) for v in msg.get_all("References", [])))),
                subject=_decoded_header(" ".join(str(v) for v in msg.get_all("Subject", []))),
                sender=_speaker(str(from_header) if from_header is not None else None),
                sent_at=_sent_at(str(date_header) if date_header is not None else None),
                body="\n\n".join(_plain_parts(msg)).strip(),
            )
            if not candidates.body.strip():
                continue
            if candidates.id in seen:
                continue
            seen.add(candidates.id)
            raws.append(candidates)
    finally:
        box.close()
    return raws


class _ThreadSolver:
    """Conversation ids from reply headers, with a subject-tag fallback.

    Two passes:

    1. **Reply edges.** A mail joins the thread of every ancestor id that is
       a ``Message-ID`` *present in this mailbox* — its ``In-Reply-To``
       target and every id in its ``References`` list. Ancestors outside the
       mailbox bind nothing: a reply whose parents were never delivered
       cannot cite a thread this corpus does not contain. Unioning with
       *every* present ancestor is deliberate — the RFC's ``References`` is
       an ancestry list, any of its members is as good as another as an
       edge, and a superfluous edge can only *merge* threads, which reduces
       independence: it can make corroboration harder, never easier.
    2. **Subject tags.** A mail with no resolved ancestor, and cited by
       nobody, joins the tag ``thread:subject:<normalised subject>``.
       Headers that exist but resolve to nothing count as missing — the
       usual case is a reply chain severed across mailboxes — so the severed
       reply falls back to the tag of the thread it plausibly continues. The
       fallback errs the safe way: joining threads that share a normalised
       subject *reduces* independence. It can make a claim harder to
       corroborate, never easier.

    A component formed by reply edges keeps the id of its root — the one
    member with no resolved ancestor — so every reply in the chain names the
    same conversation root. A component that yields several roots ties
    lexicographically over the whole group, which keeps the id a pure
    function of the mailbox bytes.
    """

    def __init__(self, mails: Sequence[_RawMail]) -> None:
        self._parent: dict[str, str] = {}
        self._tags: set[str] = set()
        self._resolved: dict[str, list[str]] = {}
        present = {mail.id for mail in mails}
        cited = {ancestor for mail in mails for ancestor in mail.ancestors}
        for mail in mails:
            self._parent[mail.id] = mail.id
            self._resolved[mail.id] = [
                target for target in mail.ancestors if target in present and target != mail.id
            ]
            for target in self._resolved[mail.id]:
                self._union(mail.id, target)
        for mail in mails:
            if self._resolved[mail.id] or mail.id in cited:
                continue
            tag = _normalise_subject(mail.subject)
            tag_node = f"thread:subject:{tag}" if tag else f"thread:anon:{fingerprint(mail.body)}"
            self._tags.add(tag_node)
            self._union(mail.id, tag_node)

    def _find(self, node: str) -> str:
        root = node
        while self._parent.get(root, root) != root:
            root = self._parent[root]
        while self._parent[node] != node:
            nth = self._parent[node]
            self._parent[node] = root
            node = nth
        return root

    def _union(self, a: str, b: str) -> None:
        self._parent.setdefault(a, a)
        self._parent.setdefault(b, b)
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)

    def conversations(self) -> dict[str, str]:
        """``final mail id -> conversation id`` for every mail."""
        component: dict[str, list[str]] = {}
        for mail_id in sorted(self._parent):
            if mail_id in self._tags:
                continue
            component.setdefault(self._find(mail_id), []).append(mail_id)
        tag_of_component: dict[str, str] = {}
        for tag in sorted(self._tags):
            tag_of_component.setdefault(self._find(tag), tag)
        threads: dict[str, str] = {}
        for group in component.values():
            component_tag = tag_of_component.get(self._find(group[0]))
            if component_tag is not None:
                conversation = component_tag
            else:
                roots = [mail_id for mail_id in group if not self._resolved[mail_id]]
                ordered = roots if len(roots) == 1 else sorted(group)
                conversation = f"thread:{sorted(ordered)[0]}"
            for mail_id in group:
                threads[mail_id] = conversation
        return threads


def mbox_to_messages(path: str | Path, *, upload_id: str | None = None) -> tuple[Message, ...]:
    """Every citable text/plain mail in the mailbox as a corpus message.

    - ``message_id`` — the mail's ``Message-ID``, or the content digest when
      it has none (see :func:`_read_mails`); unique across the mailbox.
    - ``conversation_id`` — the solved thread (see :class:`_ThreadSolver`).
    - ``speaker`` — the sender's address, verbatim.
    - ``text`` — the body's text/plain parts joined; attachments skipped, and
      the edge whitespace of the body (which the composing client leaned on
      its encoder to add) is stripped.
    - ``sent_at`` — the ``Date`` header as UTC, or ``None`` when the mail is
      undated (CDI-009).
    - ``upload_id`` — forwarded untouched: provenance, never identity.
    - ``source_id`` — the same id ``message_id`` carries.

    Deterministic: the same bytes parsed twice give the same messages.
    """
    raws = _read_mails(path)
    threads = _ThreadSolver(raws).conversations()
    return tuple(
        Message(
            message_id=raw.id,
            conversation_id=threads[raw.id],
            speaker=raw.sender,
            text=raw.body,
            sent_at=raw.sent_at,
            upload_id=upload_id,
            source_id=raw.id,
        )
        for raw in raws
    )


def mbox_to_corpus(
    path: str | Path,
    *,
    subject: str,
    observer: str | None = None,
    upload_id: str | None = None,
) -> Corpus:
    """The mailbox as a corpus for one subject, with its lineage declared.

    ``subject`` and ``observer`` are the *speaker strings* of the two people
    the analysis is about — sender addresses, exactly as
    :func:`mbox_to_messages` writes them. TRACE does not decide who matters;
    the integrator does. The strategy is :data:`MBOX_LINEAGE_STRATEGY`,
    passed explicitly so the independence assumption is on the corpus object
    for anyone to read.
    """
    return corpus_from_messages(
        Person(subject=subject, observer=observer),
        mbox_to_messages(path, upload_id=upload_id),
        lineage_strategy=MBOX_LINEAGE_STRATEGY,
    )


def mbox_to_evidence(
    path: str | Path,
    *,
    subject: str,
    observer: str | None = None,
    upload_id: str | None = None,
) -> tuple[Evidence, ...]:
    """The mailbox as TRACE evidence, through the library's own helper.

    ``source_type="message"``, ``modality="mbox"`` — CDI-001.1's example
    verbatim: provenance is provenance, scoring keys are scoring keys. The
    rows are built by :func:`trace_traits.evidence_from_corpus`; no evidence
    field — and no evidence *summary* field — is hand-derived here.
    """
    corpus = mbox_to_corpus(path, subject=subject, observer=observer, upload_id=upload_id)
    return evidence_from_corpus(corpus, source_type="message", modality="mbox")
