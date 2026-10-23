"""Emoji detection, by the Unicode property rather than by guesswork.

The obvious implementation — "is this codepoint in one of the ranges emoji
live in?" — is wrong in both directions, and the audit found both errors:

- **False negatives.** ``⭐`` (U+2B50) and ``⏰`` (U+23F0) are among the most
  common emoji in chat, and neither is in the range 0x1F300–0x1FAFF. A person
  who uses emoji constantly could be reported as never using one.
- **False positives.** Widening the ranges to cover them sweeps in ``←→``
  (U+2194), APL symbols like ``⌶`` (U+2336), and typographic marks like
  ``‼`` (U+203C) — characters that are punctuation or mathematics, and that a
  reader would not call emoji.

What discriminates them is the Unicode ``Emoji_Presentation`` property: a
character has it when it *renders as an emoji by default*, without a
variation selector. That property is defined by an explicit list of codepoint
ranges, which this module encodes.

**Why not a dependency.** ``regex`` supports ``\\p{Emoji_Presentation}``
directly, but `trace_traits` ships no third-party dependencies — the same rule
that keeps `trace_core` on the standard library alone. The ranges are data,
not logic, so encoding them costs nothing but a table.

**What the counts decide, and what they do not.** ``❤`` (U+2764) and ``🌬``
(U+1F32A) are *text presentation* by default: they render as text unless a
variation selector (U+FE0F) follows. A **bare** such character is not counted:
in a chat message a bare ``❤`` is more often a typed heart than a pasted
emoji, and counting it would inflate the emoji rate. But when the message
itself carries the selector — ``❤️``, ``🌬️`` — the sender has made the choice,
and it is counted: the held-out evaluation (PLAN.md G-14) caught this hole
when an author's ``🛠️`` and ``🌬️`` went unseen and the emoji rate read low.
A caller who also wants the bare characters counted can pass
``include_text_presentation=True``.
"""

from __future__ import annotations

__all__ = [
    "EMOJI_PRESENTATION_RANGES",
    "TEXT_PRESENTATION_RANGES",
    "has_emoji",
    "is_emoji_presentation",
]

#: The Unicode ``Emoji_Presentation`` property (Unicode 15.1, emoji-data.txt).
#: A character in one of these ranges renders as an emoji by default.
EMOJI_PRESENTATION_RANGES: tuple[tuple[int, int], ...] = (
    (0x231A, 0x231B),  # watch, hourglass
    (0x23E9, 0x23EC),  # media control symbols
    (0x23F0, 0x23F0),  # alarm clock
    (0x23F3, 0x23F3),  # hourglass flowing sand
    (0x25FD, 0x25FE),  # small/medium white square
    (0x2614, 0x2615),  # umbrella, hot beverage
    (0x2648, 0x2653),  # zodiac
    (0x267F, 0x267F),  # wheelchair
    (0x2693, 0x2693),  # anchor
    (0x26A1, 0x26A1),  # high voltage
    (0x26AA, 0x26AB),  # white circle, white medium square
    (0x26BD, 0x26BE),  # soccer, baseball
    (0x26C4, 0x26C5),  # snowman, sun behind cloud
    (0x26CE, 0x26CE),  # ophiuchus
    (0x26D4, 0x26D4),  # no entry
    (0x26EA, 0x26EA),  # church
    (0x26F2, 0x26F3),  # fountain, golf
    (0x26F5, 0x26F5),  # sailboat
    (0x26FA, 0x26FA),  # tent
    (0x26FD, 0x26FD),  # fuel pump
    (0x2705, 0x2705),  # white heavy check mark
    (0x270A, 0x270B),  # fist, raised hand
    (0x2728, 0x2728),  # sparkles
    (0x274C, 0x274C),  # cross mark
    (0x274E, 0x274E),  # negative squared cross mark
    (0x2753, 0x2755),  # question, exclamation
    (0x2757, 0x2757),  # red exclamation
    (0x2795, 0x2797),  # heavy plus, heavy minus, heavy division sign
    (0x27B0, 0x27B0),  # curly loop
    (0x27BF, 0x27BF),  # double curly loop
    (0x2B1B, 0x2B1C),  # black square, white square
    (0x2B50, 0x2B50),  # white medium star
    (0x2B55, 0x2B55),  # heavy large circle
    (0x1F004, 0x1F004),  # mahjong red dragon
    (0x1F0CF, 0x1F0CF),  # joker
    (0x1F18E, 0x1F18E),  # AB button
    (0x1F191, 0x1F19A),  # squared latin letters
    (0x1F1E6, 0x1F1FF),  # regional indicators
    (0x1F201, 0x1F201),  # squared katakana "here" (🈂 is text-presentation)
    (0x1F21A, 0x1F21A),  # squared korean consonants
    (0x1F22F, 0x1F22F),  # squared russian
    (0x1F232, 0x1F236),  # squared cjk ideographs
    (0x1F238, 0x1F23A),  # squared japanese
    (0x1F250, 0x1F251),  # squared cjk unified ideographs
    (0x1F300, 0x1F320),  # weather, sky, animals
    (0x1F32D, 0x1F335),
    (0x1F337, 0x1F37C),
    (0x1F37E, 0x1F393),
    (0x1F3A0, 0x1F3CA),
    (0x1F3CF, 0x1F3D3),
    (0x1F3E0, 0x1F3F0),
    (0x1F3F4, 0x1F3F4),
    (0x1F3F8, 0x1F43E),
    (0x1F440, 0x1F440),
    (0x1F442, 0x1F4FC),
    (0x1F4FF, 0x1F53D),
    (0x1F54B, 0x1F54E),
    (0x1F550, 0x1F567),
    (0x1F57A, 0x1F57A),
    (0x1F595, 0x1F596),
    (0x1F5A4, 0x1F5A4),  # smiling face with horns (🖥 is text-presentation)
    (0x1F5FB, 0x1F64F),
    (0x1F680, 0x1F6C5),
    (0x1F6CC, 0x1F6CC),
    (0x1F6D0, 0x1F6D2),
    (0x1F6D5, 0x1F6D7),
    (0x1F6DC, 0x1F6DF),
    (0x1F6EB, 0x1F6EC),
    (0x1F6F4, 0x1F6FC),
    (0x1F7E0, 0x1F7EB),
    (0x1F7F0, 0x1F7F0),
    (0x1F90C, 0x1F93A),
    (0x1F93C, 0x1F945),
    (0x1F947, 0x1F9FF),
    (0x1FA70, 0x1FA7C),
    (0x1FA80, 0x1FA88),
    (0x1FA90, 0x1FABD),
    (0x1FABF, 0x1FAC5),
    (0x1FACE, 0x1FADB),
    (0x1FAE0, 0x1FAE8),
    (0x1FAF0, 0x1FAF8),
)

#: Characters with the Unicode ``Emoji`` property but *text* presentation by
#: default — they render as text unless something forces emoji presentation.
#:
#: **Complete at Unicode 15.1**, generated from ``emoji-data.txt`` as
#: (``Emoji`` properties) minus (``Emoji_Presentation``). The legacy ASCII
#: entries the file carries (``#``, ``*``, ``0``–``9``, ``©``, ``®``) are
#: here too, held to the same receipt rule as everything else: a bare digit
#: is a digit — these points exist upstream only so the *keycap* sequences
#: can be built from them — but a digit the sender actually followed with
#: ``U+FE0F`` was sent as a keycap.
#:
#: The counting rule is in :func:`has_emoji`: a character in this table counts
#: when the message **actually follows it with U+FE0F** — the sender has
#: explicitly forced emoji presentation, and whatever renders for them renders
#: for the reader. A bare character in this table is *not* counted by default
#: (a typed heart is not a pasted emoji); pass
#: ``include_text_presentation=True`` for the looser reading.
TEXT_PRESENTATION_RANGES: tuple[tuple[int, int], ...] = (
    (0x203C, 0x203C),  # U+203C double exclamation mark
    (0x2049, 0x2049),  # U+2049 exclamation question mark
    (0x2122, 0x2122),  # U+2122 trade mark
    (0x2139, 0x2139),  # U+2139 information
    (0x2194, 0x2199),  # U+2194..2199 left-right arrow..down-left arrow
    (0x21A9, 0x21AA),  # U+21A9..21AA right arrow curving left..left arrow curving right
    (0x2328, 0x2328),  # U+2328 keyboard
    (0x23CF, 0x23CF),  # U+23CF eject button
    (0x23ED, 0x23EF),  # U+23ED..23EF next track button..last track button
    (0x23F1, 0x23F2),  # U+23F1..23F2 stopwatch..timer clock
    (0x23F8, 0x23FA),  # U+23F8..23FA pause button..record button
    (0x24C2, 0x24C2),  # U+24C2 circled M
    (0x25AA, 0x25AB),  # U+25AA..25AB black small square..white small square
    (0x25B6, 0x25B6),  # U+25B6 play button
    (0x25C0, 0x25C0),  # U+25C0 reverse button
    (0x25FB, 0x25FC),  # U+25FB..25FC white medium square..black medium-small square
    (0x2600, 0x2604),  # U+2600..2604 sun..cloud
    (0x260E, 0x260E),  # U+260E telephone
    (0x2611, 0x2611),  # U+2611 check box with check
    (0x2618, 0x2618),  # U+2618 shamrock
    (0x261D, 0x261D),  # U+261D index pointing up
    (0x2620, 0x2620),  # U+2620 skull and crossbones
    (0x2622, 0x2623),  # U+2622..2623 radioactive..biohazard
    (0x2626, 0x2626),  # U+2626 orthodox cross
    (0x262A, 0x262A),  # U+262A star and crescent
    (0x262E, 0x262F),  # U+262E..262F peace symbol
    (0x2638, 0x263A),  # U+2638..263A wheel of dharma..smiling face
    (0x2640, 0x2640),  # U+2640 female sign
    (0x2642, 0x2642),  # U+2642 male sign
    (0x265F, 0x2660),  # U+265F..2660 chess pawn..spade suit
    (0x2663, 0x2663),  # U+2663 club suit
    (0x2665, 0x2666),  # U+2665..2666 heart suit..diamond suit
    (0x2668, 0x2668),  # U+2668 hot springs
    (0x267B, 0x267B),  # U+267B recycling symbol
    (0x267E, 0x267E),  # U+267E infinity
    (0x2692, 0x2692),  # U+2692 hammer and pick
    (0x2694, 0x2697),  # U+2694..2697 crossed swords..alembic
    (0x2699, 0x2699),  # U+2699 gear
    (0x269B, 0x269C),  # U+269B..269C atom symbol..fleur-de-lis
    (0x26A0, 0x26A0),  # U+26A0 warning
    (0x26A7, 0x26A7),  # U+26A7 transgender symbol
    (0x26B0, 0x26B1),  # U+26B0..26B1 coffin..funeral urn
    (0x26C8, 0x26C8),  # U+26C8 cloud with lightning and rain
    (0x26CF, 0x26CF),  # U+26CF pick
    (0x26D1, 0x26D1),  # U+26D1 rescue worker's helmet
    (0x26D3, 0x26D3),  # U+26D3 chains
    (0x26E9, 0x26E9),  # U+26E9 shinto shrine
    (0x26F0, 0x26F1),  # U+26F0..26F1 mountain..umbrella on ground
    (0x26F4, 0x26F4),  # U+26F4 ferry
    (0x26F7, 0x26F9),  # U+26F7..26F9 skier..person bouncing ball
    (0x2702, 0x2702),  # U+2702 scissors
    (0x2708, 0x2709),  # U+2708..2709 airplane..e-mail symbol
    (0x270C, 0x270D),  # U+270C..270D victory hand..writing hand
    (0x270F, 0x270F),  # U+270F pencil
    (0x2712, 0x2712),  # U+2712 black nib
    (0x2714, 0x2714),  # U+2714 check mark
    (0x2716, 0x2716),  # U+2716 multiply
    (0x271D, 0x271D),  # U+271D latin cross
    (0x2721, 0x2721),  # U+2721 star of David
    (0x2733, 0x2734),  # U+2733..2734 eight-spoked asterisk..eight-pointed star
    (0x2744, 0x2744),  # U+2744 snowflake
    (0x2747, 0x2747),  # U+2747 sparkle
    (0x2763, 0x2764),  # U+2763..2764 heart exclamation..heavy black heart
    (0x27A1, 0x27A1),  # U+27A1 right arrow
    (0x2934, 0x2935),  # U+2934..2935 right arrow curving up..right arrow curving down
    (0x2B05, 0x2B07),  # U+2B05..2B07 left arrow..down arrow
    (0x3030, 0x3030),  # U+3030 wavy dash
    (0x303D, 0x303D),  # U+303D part alternation mark
    (0x3297, 0x3297),  # U+3297 Japanese "congratulations" button
    (0x3299, 0x3299),  # U+3299 Japanese "secret" button
    (0x1F170, 0x1F171),  # U+1F170..1F171 A button (blood type)..B button (blood type)
    (0x1F17E, 0x1F17F),  # U+1F17E..1F17F O button (blood type)..P button
    (0x1F202, 0x1F202),  # U+1F202 Japanese "service charge" button
    (0x1F237, 0x1F237),  # U+1F237 Japanese "monthly amount" button
    (0x1F321, 0x1F321),  # U+1F321 thermometer
    (0x1F324, 0x1F32C),  # U+1F324..1F32C sun behind small cloud..wind face
    (0x1F336, 0x1F336),  # U+1F336 hot pepper
    (0x1F37D, 0x1F37D),  # U+1F37D fork and knife with plate
    (0x1F396, 0x1F397),  # U+1F396..1F397 military medal..reminder ribbon
    (0x1F399, 0x1F39B),  # U+1F399..1F39B studio microphone..control knobs
    (0x1F39E, 0x1F39F),  # U+1F39E..1F39F film frames..admission tickets
    (0x1F3CB, 0x1F3CE),  # U+1F3CB..1F3CE person lifting weights..racing car
    (0x1F3D4, 0x1F3DF),  # U+1F3D4..1F3DF snow-capped mountain..stadium
    (0x1F3F3, 0x1F3F3),  # U+1F3F3 white flag
    (0x1F3F5, 0x1F3F5),  # U+1F3F5 rosette
    (0x1F3F7, 0x1F3F7),  # U+1F3F7 label
    (0x1F43F, 0x1F43F),  # U+1F43F chipmunk
    (0x1F441, 0x1F441),  # U+1F441 eye
    (0x1F4FD, 0x1F4FD),  # U+1F4FD film projector
    (0x1F549, 0x1F54A),  # U+1F549..1F54A om..dove
    (0x1F56F, 0x1F570),  # U+1F56F..1F570 candle..mantelpiece clock
    (0x1F573, 0x1F579),  # U+1F573..1F579 hole..joystick
    (0x1F587, 0x1F587),  # U+1F587 linked paperclips
    (0x1F58A, 0x1F58D),  # U+1F58A..1F58D pen..crayon
    (0x1F590, 0x1F590),  # U+1F590 hand with fingers splayed
    (0x1F5A5, 0x1F5A5),  # U+1F5A5 desktop computer
    (0x1F5A8, 0x1F5A8),  # U+1F5A8 printer
    (0x1F5B1, 0x1F5B2),  # U+1F5B1..1F5B2 computer mouse..trackball
    (0x1F5BC, 0x1F5BC),  # U+1F5BC framed picture
    (0x1F5C2, 0x1F5C4),  # U+1F5C2..1F5C4 card index dividers..file cabinet
    (0x1F5D1, 0x1F5D3),  # U+1F5D1..1F5D3 wastebasket..spiral calendar
    (0x1F5DC, 0x1F5DE),  # U+1F5DC..1F5DE clamp..rolled-up newspaper
    (0x1F5E1, 0x1F5E1),  # U+1F5E1 dagger
    (0x1F5E3, 0x1F5E3),  # U+1F5E3 speaking head
    (0x1F5E8, 0x1F5E8),  # U+1F5E8 left speech bubble
    (0x1F5EF, 0x1F5EF),  # U+1F5EF right anger bubble
    (0x1F5F3, 0x1F5F3),  # U+1F5F3 ballot box with ballot
    (0x1F5FA, 0x1F5FA),  # U+1F5FA world map
    (0x1F6CB, 0x1F6CB),  # U+1F6CB couch and lamp
    (0x1F6CD, 0x1F6CF),  # U+1F6CD..1F6CF shopping bags..bed
    (0x1F6E0, 0x1F6E5),  # U+1F6E0..1F6E5 hammer and wrench..motor boat
    (0x1F6E9, 0x1F6E9),  # U+1F6E9 small airplane
    (0x1F6F0, 0x1F6F0),  # U+1F6F0 satellite
    (0x1F6F3, 0x1F6F3),  # U+1F6F3 passenger ship
    # Legacy ASCII entries (upstream Emoji, text presentation): the keycap
    # building blocks. Same receipt rule — bare means the character itself.
    (0x0023, 0x0023),  # U+0023 hash sign
    (0x002A, 0x002A),  # U+002A asterisk
    (0x0030, 0x0039),  # U+0030..0039 digit zero..nine
    (0x00A9, 0x00A9),  # U+00A9 copyright
    (0x00AE, 0x00AE),  # U+00AE registered
)


#: U+FE0F — the variation selector that forces emoji presentation.
VARIATION_SELECTOR_16: str = "\ufe0f"


def _in_ranges(point: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(low <= point <= high for low, high in ranges)


def is_emoji_presentation(char: str) -> bool:
    """Whether the character renders as an emoji by default."""
    if not char:
        return False
    return _in_ranges(ord(char[0]), EMOJI_PRESENTATION_RANGES)


def has_emoji(text: str, *, include_text_presentation: bool = False) -> bool:
    """Whether the text contains something that renders as an emoji.

    A character in the presentation table renders as an emoji **by default**
    and always counts. A character in the text-presentation table renders as
    an emoji only when something forces it, and this function counts it when
    the message itself follows it with U+FE0F — the sender explicitly asked
    for emoji presentation, and the transcript carries the receipt.

    A *bare* text-presentation character (``❤``, ``☀``) is not counted by
    default: a typed heart is more often a typed heart than a pasted emoji,
    and counting it would inflate the rate for exactly the people who type
    them most. Pass ``include_text_presentation=True`` for the looser
    reading, which counts it whether or not a selector follows.
    """
    count = len(text)
    for index, char in enumerate(text):
        point = ord(char)
        if _in_ranges(point, EMOJI_PRESENTATION_RANGES):
            return True
        if not _in_ranges(point, TEXT_PRESENTATION_RANGES):
            continue
        forced = index + 1 < count and text[index + 1] == VARIATION_SELECTOR_16
        if forced or include_text_presentation:
            return True
    return False

