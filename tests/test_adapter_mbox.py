"""What the mbox adapter, and only the adapter, owes the contract.

These tests pin the *behaviours* the adapter asserts; nothing here reaches
into ``trace_traits`` or ``trace_core`` internals. Whatever TRACE verifies
afterwards — corroboration counting, acceptance, band vocabulary — is
exercised through its public API and assumed:

- a reply chain whose headers point at one another resolves to **one**
  conversation root (``In-Reply-To`` / ``References``);
- independent threads get their own roots, and threads that share a
  normalised subject merge (the safe direction: fewer lineages, never more);
- the **plain-text alternative** of an HTML mail is what a message becomes,
  never the markup;
- an attachment with a filename stays an attachment even when it is
  ``text/plain``; a mail that is *only* attachments leaves no message;
- a mail with no ``Date`` is undated (CDI-009: ``sent_at=None``), and a
  naive date is read as UTC (DM-003.3);
- the same bytes parse to the same messages; the same message recorded
  twice parses once;
- the corpus's lineage strategy is the adapter's declared one, and multiple
  uploads of the same conversation corroborate once under it;
- the *corpus builder* is what rejects a naive/aware mixture — the adapter
  never emits one, and this suite does not re-test the builder's rule, only
  affirms the error once (test
  ``test_the_corpus_builder_rejects_a_naive_aware_mixture``).

Fixtures are written on the fly with :mod:`email.message.EmailMessage`; no
binary fixture is committed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

import pytest

from adapters import (
    MBOX_LINEAGE_STRATEGY,
    mbox_to_corpus,
    mbox_to_evidence,
    mbox_to_messages,
)
from trace_core import Scorer, TableConfidencePolicy, TableRetrievalPolicy
from trace_traits import (
    DimensionBand,
    LineageStrategy,
    Message,
    Person,
    build_profile,
    corpus_from_messages,
    extract_features,
)

SUBJECT = "avery@trace.test"
OBSERVER = "sam@trace.test"

# Band vocabulary as a set of words, from the public enum — never a meter.
BAND_WORDS = frozenset({item.value for item in DimensionBand})


def _a_mail(
    *,
    sender: str,
    subject: str,
    body: str,
    date: str | None = None,
    message_id: str | None = None,
    in_reply_to: str | None = None,
    references: Sequence[str] | None = None,
    html_alternative: str | None = None,
    html_only: bool = False,
    text_attachment: str | None = None,
    binary_attachment: tuple[str, bytes] | None = None,
) -> EmailMessage:
    """One email message exactly as the fixture intends it."""
    mail = EmailMessage()
    mail["From"] = sender
    mail["To"] = "the-archive@trace.test"
    mail["Subject"] = subject
    if date is not None:
        mail["Date"] = date
    if message_id is not None:
        mail["Message-ID"] = message_id
    if in_reply_to is not None:
        mail["In-Reply-To"] = in_reply_to
    if references:
        mail["References"] = " ".join(references)
    if html_only:
        mail.set_content(body, subtype="html")
    else:
        mail.set_content(body)
    if html_alternative is not None:
        mail.add_alternative(html_alternative, subtype="html")
    if text_attachment is not None:
        # The text content manager decides its own maintype; passing one
        # raises TypeError in the stdlib dispatch.
        mail.add_attachment(text_attachment, subtype="plain", filename="notes.txt")
    if binary_attachment is not None:
        filename, blob = binary_attachment
        mail.add_attachment(blob, maintype="application", subtype="pdf", filename=filename)
    return mail


def _write_mailbox(
    tmp_path: Path, mails: Sequence[EmailMessage], name: str = "corpus.mbox"
) -> Path:
    """Write an mbox file; deterministically, with a fixed ``From `` line."""
    wire = bytearray()
    for mail in mails:
        chunk = mail.as_bytes().rstrip(b"\n")
        for line in chunk.split(b"\n"):
            assert not line.startswith(b"From "), "fixture body would confuse the mbox"
        wire += b"From - Mon Jan 05 09:00:00 2026\n" + chunk + b"\n"
    path = tmp_path / name
    path.write_bytes(bytes(wire))
    return path


def test_a_reply_chain_solves_to_one_conversation_root(tmp_path: Path) -> None:
    path = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=OBSERVER,
                subject="Weekly notes",
                body="Tuesday's sync, in writing this time.",
                message_id="<t1-root@trace.test>",
                date="Mon, 05 Jan 2026 09:15:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Weekly notes",
                body="Seen. I will trim the roadmap slide before Friday.",
                message_id="<t1-r1@trace.test>",
                in_reply_to="<t1-root@trace.test>",
                date="Mon, 05 Jan 2026 11:02:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Weekly notes",
                body="Adding the outage note so it stays on the record.",
                message_id="<t1-r2@trace.test>",
                in_reply_to="<t1-r1@trace.test>",
                references=["<t1-root@trace.test>", "<t1-r1@trace.test>"],
                date="Mon, 05 Jan 2026 12:40:00 +0100",
            ),
        ],
    )
    messages = mbox_to_messages(path)
    conversations = {m.message_id: m.conversation_id for m in messages}
    assert conversations["<t1-root@trace.test>".strip(" <>")] == "thread:t1-root@trace.test"
    assert (
        conversations["t1-r1@trace.test"]
        == conversations["t1-r2@trace.test"]
        == conversations["t1-root@trace.test"]
    )


def test_independent_threads_get_their_own_conversation_roots(tmp_path: Path) -> None:
    path = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=OBSERVER,
                subject="Lunch plans",
                body="Thursday, usual place?",
                message_id="<lunch-root@trace.test>",
                date="Mon, 05 Jan 2026 09:15:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Lunch plans",
                body="Works for me, noon it is.",
                message_id="<lunch-r1@trace.test>",
                in_reply_to="<lunch-root@trace.test>",
                date="Mon, 05 Jan 2026 10:00:00 +0100",
            ),
            _a_mail(
                sender=OBSERVER,
                subject="Roadmap call",
                body="Can we push the roadmap call to next week?",
                message_id="<roadmap-root@trace.test>",
                date="Mon, 05 Jan 2026 21:10:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Roadmap call",
                body="Pushing it, sending a new invite.",
                message_id="<roadmap-r1@trace.test>",
                in_reply_to="<roadmap-root@trace.test>",
                date="Mon, 05 Jan 2026 21:55:00 +0100",
            ),
        ],
    )
    messages = mbox_to_messages(path)
    conversations = {m.message_id: m.conversation_id for m in messages}
    lunch = conversations["lunch-root@trace.test"]
    roadmap = conversations["roadmap-root@trace.test"]
    assert lunch == conversations["lunch-r1@trace.test"]
    assert roadmap == conversations["roadmap-r1@trace.test"]
    assert lunch != roadmap


def test_the_plain_text_alternative_is_what_becomes_a_message(tmp_path: Path) -> None:
    body = "Lunch at the usual place, one o'clock."
    path = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=SUBJECT,
                subject="Lunch plans",
                body=body,
                html_alternative="<html><body><b>Lunch</b> at the usual place.</body></html>",
                message_id="<alt@trace.test>",
                date="Mon, 05 Jan 2026 09:15:00 +0100",
            ),
        ],
    )
    (messages,) = mbox_to_messages(path)
    assert messages.text == body
    assert "<b>" not in messages.text
    assert "<html" not in messages.text


def test_an_attachment_with_a_filename_is_skipped_but_the_body_stays(tmp_path: Path) -> None:
    body = "Here are the notes, summary below: two action items."
    path = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=SUBJECT,
                subject="Weekly notes",
                body=body,
                message_id="<with-attach@trace.test>",
                date="Mon, 05 Jan 2026 09:15:00 +0100",
                text_attachment="ATTACHMENT TEXT THE PARSER MUST NOT BURY",
            ),
        ],
    )
    (messages,) = mbox_to_messages(path)
    assert messages.text == body
    assert "ATTACHMENT" not in messages.text


def test_a_mail_that_is_attachment_only_leaves_no_message(tmp_path: Path) -> None:
    path = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=SUBJECT,
                subject="Q1 comp plan",
                body=" enclosed",
                html_only=True,
                message_id="<attach-only@trace.test>",
                date="Mon, 05 Jan 2026 09:15:00 +0100",
                binary_attachment=("plan.pdf", b"%PDF-1.4 mock"),
            ),
        ],
    )
    assert mbox_to_messages(path) == ()


def test_an_empty_mailbox_leaves_no_messages(tmp_path: Path) -> None:
    (tmp_path / "empty.mbox").write_bytes(b"")
    assert mbox_to_messages(tmp_path / "empty.mbox") == ()


def test_an_html_mail_without_a_text_alternative_leaves_no_message(tmp_path: Path) -> None:
    path = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=SUBJECT,
                subject="Pixels only",
                body="<html><body>No text part in here at all.</body></html>",
                html_only=True,
                message_id="<html-only@trace.test>",
                date="Mon, 05 Jan 2026 09:15:00 +0100",
            ),
        ],
    )
    assert mbox_to_messages(path) == ()


def test_an_undated_mail_arrives_undated_cdi_009(tmp_path: Path) -> None:
    path = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=OBSERVER,
                subject="Weekly notes",
                body="Tuesday's sync, in writing this time.",
                message_id="<u-root@trace.test>",
                date="Mon, 05 Jan 2026 09:15:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Weekly notes",
                body="Seen, no clock attached to this one.",
                message_id="<u-r1@trace.test>",
                in_reply_to="<u-root@trace.test>",
                date=None,
            ),
        ],
    )
    messages = mbox_to_messages(path)
    sent = {m.message_id: m.sent_at for m in messages}
    assert sent["u-r1@trace.test"] is None  # CDI-009: undated, not zero
    assert sent["u-root@trace.test"] is not None


def test_a_naive_date_is_read_as_utc(tmp_path: Path) -> None:
    path = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=SUBJECT,
                subject="Late check",
                body="This one says the zone is unknown (-0000).",
                message_id="<naive@trace.test>",
                date="Mon, 12 Jan 2026 23:40:00 -0000",
            ),
        ],
    )
    (messages,) = mbox_to_messages(path)
    assert messages.sent_at is not None
    assert messages.sent_at.tzinfo is not None  # never omissions: naive parse became UTC
    assert messages.sent_at.utcoffset() == timedelta(0)


def test_a_mailbox_parses_deterministically(tmp_path: Path) -> None:
    mails = [
        _a_mail(
            sender=OBSERVER,
            subject="Weekly notes",
            body="Tuesday's sync, in writing this time.",
            message_id="<d-root@trace.test>",
            date="Mon, 05 Jan 2026 09:15:00 +0100",
        ),
        _a_mail(
            sender=SUBJECT,
            subject="Re: Weekly notes",
            body="Seen.",
            message_id=None,
            in_reply_to="<d-root@trace.test>",
            date="Mon, 05 Jan 2026 11:02:00 +0100",
        ),
        _a_mail(
            sender=SUBJECT,
            subject="Re: Weekly notes",
            body="Seen, and raised twice.",
            message_id=None,
            in_reply_to="<d-root@trace.test>",
            date=None,
        ),
    ]
    first = _write_mailbox(tmp_path, mails, name="one.mbox")
    again = _write_mailbox(tmp_path, mails, name="two.mbox")
    assert mbox_to_messages(first) == mbox_to_messages(again)


def test_one_mail_recorded_twice_corroborates_once(tmp_path: Path) -> None:
    def the_same_mail() -> EmailMessage:
        """The byte-identical second copy, rebuilt rather than mutated."""
        return _a_mail(
            sender=SUBJECT,
            subject="Lunch plans",
            body="Works for me, noon it is.",
            message_id="<dup@trace.test>",
            date="Mon, 05 Jan 2026 10:00:00 +0100",
        )

    path = _write_mailbox(tmp_path, [the_same_mail(), the_same_mail()])
    messages = mbox_to_messages(path)
    assert len(messages) == 1
    assert messages[0].message_id == "dup@trace.test"


def test_headerless_mails_fall_back_to_a_deterministic_subject_tag(tmp_path: Path) -> None:
    path = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=SUBJECT,
                subject="Re: Tennis quick rematch",
                body="I am in, Saturday stands.",
                date="Sun, 11 Jan 2026 08:00:00 +0000",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: re: Fwd: Tennis quick rematch",
                body="Bringing the spare balls too.",
                date="Sun, 18 Jan 2026 07:30:00 +0000",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Tax docs",
                body="Filed these last night, finally.",
                date="Sun, 18 Jan 2026 22:00:00 +0000",
            ),
        ],
    )
    messages = mbox_to_messages(path)
    conversations = [m.conversation_id for m in messages]
    first, second, third = conversations
    assert first == second == "thread:subject:tennis quick rematch"
    assert third == "thread:subject:tax docs"
    assert first != third


def test_a_severed_reply_falls_back_to_a_subject_tag(tmp_path: Path) -> None:
    path = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=SUBJECT,
                subject="Tennis quick rematch",
                body="Saturday stands then.",
                date="Sun, 11 Jan 2026 08:00:00 +0000",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Tennis quick rematch",
                body="Same court, 10am.",
                in_reply_to="<moments-gone@elsewhere.test>",
                date="Sun, 11 Jan 2026 08:30:00 +0000",
            ),
        ],
    )
    (first, second) = mbox_to_messages(path)
    assert first.conversation_id == second.conversation_id
    assert first.conversation_id.startswith("thread:subject:")


def test_reuploading_one_conversation_does_not_multiply_independence(
    tmp_path: Path,
) -> None:
    texts = ["Seen, thanks, parking it here.", "Pushing the branch tonight instead."]
    first_mailbox = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=SUBJECT,
                subject="Branch plan",
                body=texts[0],
                message_id="<up1@one.test>",
                date="Mon, 05 Jan 2026 09:15:00 +0100",
            ),
            _a_mail(
                sender=OBSERVER,
                subject="Re: Branch plan",
                body=texts[1],
                message_id="<up2@one.test>",
                in_reply_to="<up1@one.test>",
                date="Mon, 05 Jan 2026 10:00:00 +0100",
            ),
        ],
        name="first.mbox",
    )
    second_mailbox = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=SUBJECT,
                subject="Branch plan, copy",
                body=texts[0],
                message_id="<up9@two.test>",
                date="Tue, 06 Jan 2026 09:15:00 +0100",
            ),
            _a_mail(
                sender=OBSERVER,
                subject="Re: Branch plan, copy",
                body=texts[1],
                message_id="<up8@two.test>",
                in_reply_to="<up9@two.test>",
                date="Tue, 06 Jan 2026 10:00:00 +0100",
            ),
        ],
        name="second.mbox",
    )
    first_corpus = mbox_to_corpus(first_mailbox, subject=SUBJECT, observer=OBSERVER)
    second_corpus = mbox_to_corpus(second_mailbox, subject=SUBJECT, observer=OBSERVER)
    assert first_corpus.lineage_strategy is LineageStrategy.BY_FINGERPRINT
    assert second_corpus.lineage_strategy is LineageStrategy.BY_FINGERPRINT
    (first_message,) = [m for m in first_corpus.messages if m.text == texts[0]]
    (second_message,) = [m for m in second_corpus.messages if m.text == texts[0]]
    assert first_corpus.lineage_of(first_message) == second_corpus.lineage_of(second_message)


def test_the_corpus_builder_rejects_a_naive_aware_mixture() -> None:
    aware = [
        Message(
            message_id="a",
            conversation_id="c",
            speaker=SUBJECT,
            text="aware",
            sent_at=datetime(2026, 1, 5, 12, 0, tzinfo=UTC),
        )
    ]
    naive = [
        Message(
            message_id="b",
            conversation_id="c",
            speaker=SUBJECT,
            text="naive",
            sent_at=datetime(2026, 1, 6, 12, 0),
        )
    ]
    with pytest.raises(ValueError, match="mixes"):
        corpus_from_messages(Person(subject=SUBJECT), aware + naive)


def test_evidence_keeps_mbox_as_provenance_not_a_scoring_key(tmp_path: Path) -> None:
    path = _write_mailbox(
        tmp_path,
        [
            _a_mail(
                sender=OBSERVER,
                subject="Lunch plans",
                body="Thursday, usual place?",
                message_id="<ev-root@trace.test>",
                date="Mon, 05 Jan 2026 09:15:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Lunch plans",
                body="Works for me, noon it is.",
                message_id="<ev-r1@trace.test>",
                in_reply_to="<ev-root@trace.test>",
                date="Mon, 05 Jan 2026 10:00:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Standup rota",
                body="Taking Thursday's standup if you want it.",
                message_id="<ev-root2@trace.test>",
                date="Mon, 05 Jan 2026 20:10:00 +0100",
            ),
        ],
    )
    evidence = mbox_to_evidence(path, subject=SUBJECT, observer=OBSERVER)
    assert all(row.source_type == "message" for row in evidence)  # CDI-001.1
    assert all(row.modality == "mbox" for row in evidence)
    assert all(row.lineage_id for row in evidence)  # the assumption is on the row
    assert len({row.lineage_id for row in evidence}) >= 2
    undated = [row for row in evidence if row.occurred_at is None]
    assert undated == []
    (root_row,) = [row for row in evidence if row.evidence_id == "ev-root@trace.test"]
    assert root_row.speaker == OBSERVER
    assert root_row.source_id == "ev-root@trace.test"


def _readme_scorer() -> Scorer:
    """The exact policy the README's worked example builds."""
    conf = TableConfidencePolicy(
        source_weights={"user_stated": 1.0, "message": 0.70, "system_inference": 0.55},
        half_lives={"behavioral": 270, "distributional": 540},
        corroboration_floor=2,
        coverage_denominator=3,
    )
    return Scorer(
        confidence=conf,
        retrieval=TableRetrievalPolicy(
            weights={"vector": 0.55, "recency": 0.20, "source": 0.15, "confidence": 0.10},
            source_weights=conf.source_weights,
        ),
    )


def test_the_full_pipeline_reads_a_multithreaded_mailbox(tmp_path: Path) -> None:
    """The README's own walk: messages -> corpus -> profile, on an mbox.

    Nine messages from the subject across four threads — the two messages of
    one thread are recorded twice (dedup), one mail is attach-only (no
    message), one reply is undated. Assertions stay on the public surface:
    readings are words from the closed set with evidence attached.
    """
    path = _write_mailbox(
        tmp_path,
        [
            # thread 1: notes — observer root, three subject replies (one undated)
            _a_mail(
                sender=OBSERVER,
                subject="Weekly notes",
                body="Tuesday's sync, in writing this time.",
                message_id="<f1-root@trace.test>",
                date="Mon, 05 Jan 2026 09:15:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Weekly notes",
                body="Seen. I will trim the roadmap slide before Friday.",
                message_id="<f1-r1@trace.test>",
                in_reply_to="<f1-root@trace.test>",
                date="Mon, 05 Jan 2026 11:02:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Weekly notes",
                body="Adding the outage note so it stays on the record.",
                message_id="<f1-r2@trace.test>",
                in_reply_to="<f1-r1@trace.test>",
                references=["<f1-root@trace.test>", "<f1-r1@trace.test>"],
                date="Mon, 05 Jan 2026 12:40:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Weekly notes",
                body="Done, review link sent.",
                message_id="<f1-r3@trace.test>",
                in_reply_to="<f1-r2@trace.test>",
                date=None,
            ),
            # thread 2: lunch — observer root, two subject replies, one HTML alt
            _a_mail(
                sender=OBSERVER,
                subject="Lunch plans",
                body="Thursday, usual place?",
                message_id="<f2-root@trace.test>",
                date="Tue, 06 Jan 2026 08:05:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Lunch plans",
                body="Works for me, noon it is.",
                message_id="<f2-r1@trace.test>",
                in_reply_to="<f2-root@trace.test>",
                date="Tue, 06 Jan 2026 08:50:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Lunch plans",
                body="Booked for 12:30, see you there.",
                html_alternative="<html><body><i>Booked</i> for 12:30.</body></html>",
                message_id="<f2-r2@trace.test>",
                in_reply_to="<f2-r1@trace.test>",
                date="Tue, 06 Jan 2026 09:30:00 +0100",
            ),
            # thread 3: tennis — subject only, headersless, one attachment mail
            _a_mail(
                sender=SUBJECT,
                subject="Tennis quick rematch",
                body="Saturday stands, same court at ten.",
                date="Sun, 11 Jan 2026 07:15:00 +0000",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Tennis quick rematch",
                body="Bringing the spare balls too.",
                date="Sun, 18 Jan 2026 07:30:00 +0000",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Tennis quick rematch",
                body="Court fee paid, receipts inside.",
                message_id="<f3-attach@trace.test>",
                date="Sun, 18 Jan 2026 08:10:00 +0000",
                text_attachment="COURT RECEIPT NOT A MESSAGE",
            ),
            # thread 4: rota — observer root, subject reply with a -0000 date
            _a_mail(
                sender=OBSERVER,
                subject="Standup rota",
                body="Can you take Thursday's standup?",
                message_id="<f4-root@trace.test>",
                date="Tue, 06 Jan 2026 20:10:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Re: Standup rota",
                body="Taking it, slides in the doc.",
                message_id="<f4-r1@trace.test>",
                in_reply_to="<f4-root@trace.test>",
                date="Tue, 06 Jan 2026 23:40:00 -0000",
            ),
            # the same lunch root re-recorded (dedup), and an attach-only mail
            _a_mail(
                sender=OBSERVER,
                subject="Lunch plans",
                body="Thursday, usual place?",
                message_id="<f2-root@trace.test>",
                date="Tue, 06 Jan 2026 08:05:00 +0100",
            ),
            _a_mail(
                sender=SUBJECT,
                subject="Q1 comp plan",
                body=" enclosed",
                html_only=True,
                message_id="<f5-attach@trace.test>",
                date="Sun, 18 Jan 2026 09:00:00 +0000",
                binary_attachment=("plan.pdf", b"%PDF-1.4 mock"),
            ),
        ],
    )
    messages = mbox_to_messages(path, upload_id="smoke-upload")
    # Fourteen mails written; the duplicate lunch root dedupes, the
    # attach-only bundle drops, leaving twelve — every one carries the
    # upload id verbatim.
    assert len(messages) == 12
    assert all(m.upload_id == "smoke-upload" for m in messages)
    undated = [m for m in messages if m.sent_at is None]
    assert len(undated) == 1

    corpus = mbox_to_corpus(path, subject=SUBJECT, observer=OBSERVER)
    assert corpus.lineage_strategy is MBOX_LINEAGE_STRATEGY

    # The adapter's wrapper routes through corpus_from_messages with the
    # declared strategy; a caller driving the primitives directly must get
    # an equivalent corpus, which pins the wrapper without re-testing it.
    direct = corpus_from_messages(
        Person(subject=SUBJECT, observer=OBSERVER), messages, lineage_strategy=MBOX_LINEAGE_STRATEGY
    )
    assert len(direct) == len(corpus) == len(messages)
    assert {m.message_id for m in direct.messages} == {m.message_id for m in corpus.messages}

    features = extract_features(corpus)
    for reading in features.readings:
        assert reading.band in BAND_WORDS, reading.band
        if reading.band != "insufficient_evidence":
            assert reading.evidence_ids

    profile = build_profile(corpus, scorer=_readme_scorer(), now=datetime(2026, 2, 1, tzinfo=UTC))
    assert profile.subject == SUBJECT
    assert profile.n_conversations >= 3
    for trait in profile.readings:
        assert trait.band in BAND_WORDS
        assert trait.n_evidence >= 1
        assert trait.n_conversations >= 1
        assert trait.decision.reason
        assert trait.decision.state in {"accepted", "candidate", "rejected", "quarantined"}
        assert dataclasses.is_dataclass(trait)
        assert trait.as_line()
    assert profile.readings
