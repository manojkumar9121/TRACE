# TRACE — Build Summary

**Spec:** `PLAN.md` (TRACE-SPEC-001, Rev 6) — 1,656 lines, S1000D structure
**Code:** 25 modules, 10,003 lines, stdlib-only in `trace_core`
**Tests:** 440 pass, 1 skipped (non-live), 14 live-model (keyed), 100 fuzz checks
**Date:** 2026-10-23

---

## 1. The headline

The plan called for an evidence-scoring library that a model cannot fool, with
four tiers of inference on top of it. **All four tiers are built and
measured**, two rounds of adversarial audit have been run against them, and —
the part Rev 6 adds — **an evaluation with labels frozen by an independent
second author has been run twice**, found a defect, and re-run after the fix.

**On "audited".** The auditors were AI subagents I dispatched, working from
the same repository and the same understanding of its intent. That is a real
test — it found 26 defects the project's own suite had missed, including a
working prompt-injection vector — but it is **not** independent review in the
sense a human auditor or a second team would provide, and this summary does
not claim it is. A CV line should say "audited by an AI subagent under
adversarial instructions", not "independently audited".

| | Before this work | Now |
|---|---|---|
| Tiers built | 0 | **4** (T1 trait, T2 behaviour, T3 response, T4 temporal) |
| T2 skill vs a *fair* baseline | — | base-rate layer **0.000 by construction** (stated, not hidden); contextual layer measured **+0.188** planted / **+0.823** window-conditional |
| **Held-out agreement (frozen labels)** | — | **0.617 graded (29/47), overall 30/48, zero value mismatches** |
| Gaps opened over the course | — | **14** — **13 closed or moved out; 1 open (G-3)** |
| Normative requirements implemented | — | **107** (28 DM · 54 PM · 13 IF · 7 INV · 5 CDI) |
| Milestones done | 3 of 14 | **11 of 14** (2 moved to the consumer's record, 1 open — the upload itself) |
| Trait recovery (planted corpora) | 71/96 (74%) | **93/96 graded (96.9%)** — the vocabulary disagreements went 8 → 0 (G-11); the 3 residual rows are recorded, not re-thresholded |
| T3 planted-reactivity recovery | — | **5/5** (2 of them `unsettled`) |
| T4 live-model separation | — | **20/20** (a smoke test, not a benchmark) |
| Tests | 165 | **440** + 14 live + 100 fuzz |
| Adversarial defects open | 6 | **0** — and the held-out run found one more (E-a), fixed and re-measured |

**The one number that matters most is not 96.9%, and it is not 0.617 either.**
It is that every claim resolves to cited evidence a reader can check by hand.
"Their replies got shorter" is checkable against messages. "They were angry"
is not checkable against anything in a chat log — and the library stops at the
first kind. The held-out number exists precisely so the *accuracy* claim is
not made on the same transcripts that tuned the code.

---

## 2. How much of the plan is done

### 2.1 Milestones — 11 of 14 done, 2 moved to the consumer, 1 open

| # | Milestone | Status | Note |
|---|---|---|---|
| M−1 | Decide the weights | **Done** | Decided as D5: the documented example weights are the **reference policy** of `trace_traits.reference_policy`; the core ships no default (`PLAN.md` §9.17). |
| M0 | Import-graph audit | **Done** | `trace_core` imports stdlib only. Mechanically verified. |
| M1 | Golden fixtures | **Done** | 20 corpora × every T1 reading and T2 prediction, self-checksummed, verify/freeze tooling (`verification/golden_fixtures.py`). Re-verified after the E-a repair: no drift (§9.18). |
| M2 | `trace_core.types` | **Done** | §4 complete, frozen, typed, tested. INV-1–6 hold. |
| M3 | Port `temporal` + `scoring` | **Done** | PM-001 as written. |
| M4 | Port `attribution` + `reference` | **Done** | `trace_core.attribution` is the PM-003 decision layer — state, reason, effective scope, caps, fail-closed verdict handling, INV-1's corroboration floor — with `Scorer` delegating to it (42 tests). `trace_core.reference` is the validators. |
| M5 | `trace_traits` | **Done** | T1 + T2 (base-rate *and* contextual) with a synthetic evaluation measuring both. |
| M5.5 | `trace_traits.perception` | **Done** | LLM proposes; every citation resolved and attributed; prompt-version registry (D9). |
| M6 | `trace_traits.respond` (T3) | **Done** | Matched baseline, equivalence testing, family-level multiplicity control. |
| **M6.5** | **`trace_traits.temporal` (T4)** | **Done** | Weekday and silence readings now included (G-10), with a permutation-controlled window search. |
| M7 | Screenshot adapter | **Moved out** | A record of the reference integration, not a TRACE deliverable — held in the private integration notes (M10). |
| M8 | Second author on the evidence | **Done** | Satisfied by the held-out authoring protocol: 8 corpora + labels written by an independent second author with no access to the implementation, frozen before any run (§9.19). |
| M9 | Consumer adopts the library | **Moved out** | The consumer's policy-equivalence record, held with the consumer (private notes). |
| M10 | Publish | **Open — user-side** | Publication-ready after Rev 6's sanitization; the PyPI/GitHub upload is the maintainer's action. |

### 2.2 Requirements — 107 normative statements

| Class | Count | Where |
|---|---|---|
| DM (data model) | 28 | §4 — evidence, summary, message/corpus, profile, reading |
| PM (process) | 54 | §5 — scoring, retrieval, perception, response, temporal |
| IF (interface) | 13 | §6 — policy interfaces, module contracts |
| INV (invariants) | 7 | §7 — the things that must never be violated |
| CDI (common data items) | 5 | §3 — the shared vocabulary |

### 2.3 Module map — the library as built

The map of *which integration files each module was ported from*, with the
originals' line counts, is part of the private integration notes (M10). What
ships is the library itself:

| Package | Lines | Contents |
|---|---|---|
| `trace_core` | 2,565 | Confidence scoring (PM-001), the decision layer (PM-003 — may a claim be promoted), retrieval ranking (PM-002), the reference validators, the evidence packet, the freshness gate, the temporal arithmetic, the data model, INV-1–6. |
| `trace_traits` | 7,438 | `features` (T1), `predict` (T2 base-rate **and** contextual), `profile` (the join), `perception` (the LLM tier), `respond` (T3), `respond_synth` (the T3 generator), `temporal` (T4), `emoji` (Unicode 15.1 tables), `stats` (clustered intervals, probit, permutation support), `reference_policy` (D5), `evaluate`, `cli`, `corpus`. |

`trace_core` imports nothing beyond the standard library — verified
mechanically (M0), not asserted.

---

## 3. What was added across the build

### 3.1 T4 — the temporal tier

**The question:** *this person usually goes to bed at this time, and that
other one is usually busy at this time.* The same clock hour meaning opposite
things for different people, which no arithmetic over timestamps can tell
apart.

**The design, decided before it was built.** A live-model probe was run
first: 9/9 clean messages classified correctly, 8 hard cases identical across
3 runs, no invented ids, and "can't sleep, up again" correctly *not* treated
as a bedtime signoff. The model can do the semantic part. The question was
then what stops it asserting the conclusion.

**The answer: the model proposes, the arithmetic disposes.** Three rungs:

| Rung | Needs a model? | What it does |
|---|---|---|
| `measure_activity` | **no** | When are their messages concentrated? |
| `measure_patterns` | yes (labels) | Does each *purpose* cluster at a clock window? |
| `interpret_pattern` | yes (reading) | What does the strongest pattern mean? |

**A model that labels everything wrongly produces no finding, not a wrong
one.** It can only fail to find a pattern, never assert one. Verified: a
model that calls every signoff `work` yields no bedtime finding, because the
finding comes from the *purpose*, not the clock.

**A reading must survive the text.** "Their sign-offs cluster at 23:00" is a
measurement, checkable against the messages. "They go to bed at 11pm" is a
*reading* of it, accepted only if the cited messages actually say something
about bedtime — otherwise downgraded to `unknown`, not accepted with a
caveat. The set of readings is closed (`bedtime`, `work_end`, `social_end`,
`unknown`), so a diagnosis is not expressible.

**The window is a range, and it must be finer than an hour.** Found while
building, not designed in — a sign-off at 23:05 and a work message at 23:40
share the 23:00 hour, hiding exactly the distinction the tier exists to find:

```
width 60:  signoff (23:00–23:59)   work (23:00–23:59)   <- same window
width 15:  signoff (23:00–23:14)   work (23:30–23:44)   <- separated
```

### 3.2 Second audit round — what it found, and what it cost

Two more adversarial audits were run against the finished library, with
frontier models given the documents and asked to find what was wrong. They
were right about most of it, and two findings were more serious than
anything the first round caught.

**The most serious: T2's predictor was the baseline.** The evaluation
compared the emoji prediction against a *constant-0.5* prediction and
reported "3/3 personas beat the constant baseline". The fair baseline is the
subject's **own base rate** — predict every message to have the emoji rate
they already have. Against that, the Brier skill score is **0.000** on all
three personas. The reason is simple and was hiding in the code:
`likely_emoji()` returns their smoothed historical rate and
`likely_length()` returns the median of their own lengths. **The predictor
*is* the baseline.** The old number measured a property of the *person* —
they are not a coin flip — and called it a property of the *predictor*.

**Second: the performance table was wrong.** The summary claimed
`extract_features` at 1.32s on 50,000 messages. Re-measured, it is **9.6s**;
`build_predictor` is 8.5s, not 1.15s. The code was never that fast — the
overstatement was in the document, and it is now corrected to the measured
figures.

**The rest of the findings, all accepted:**

| Finding | Action |
|---|---|
| The 91.7% gain is **in-sample** — the fixes were made against the same corpora used to measure them, and the 8 residual errors were reclassified *after* seeing the results | **G-14** was opened for exactly this — and **closed in Rev 6**: the held-out set exists, its labels were frozen before any run, and the out-of-sample number is measured (§3.6–§3.8). The in-sample caveat on the *planted* set still stands as stated. |
| T3's "5 of 5" is generous — two are `unsettled` not `stable`, and the 0.25 plant's 8–24% shortening falls inside the declared ±0.25 margin | Now stated in §4.3, with the margin named as a **policy** number. |
| The guard regexes are English-only and bypassable by paraphrase | Now documented in the README and §4.6, with the note that the *arithmetic* is what decides a pattern, so a bypassed guard yields a badly-worded reading, not a false one. |
| Intervals assume independence that does not hold; T3 has no multiplicity control; T4's window selection is biased | **G-13 opened — and closed in Rev 6** (§3.10): cluster bootstrap by conversation, family-level multiplicity control for T3, a permutation test with a vacuous-null guard for T4. |
| The contradiction denominator, the freshness rule and the TOST margin were underspecified | PM-001.5 and PM-007.4 now state the intended semantics and why they were chosen. |
| "Independently audited" overclaims — the auditors were AI subagents I dispatched | Now says exactly that, and says what a CV should write instead. |
| §1.3 said "three tiers" and T3 "Specified, not built"; Appendix A had no T4 entry; "all seven modules ported" contradicted M4 | Fixed: four tiers, Rev 5 in the appendix, and the count corrected. |
| The plan stated the project is **confidential** and recorded defects in the producer's own code | **Resolved in Rev 6 by removal, not clearance** — see §6.0. |

### 3.3 Six adversarial defects, all fixed

A three-way audit (trait recovery, robustness, adversarial/security) found
defects the project's own test suite had missed.

| # | Defect | Why it mattered |
|---|---|---|
| **S-a** | A chat message containing `</messages>` **escaped the untrusted block** into the instruction region. | A working prompt-injection vector, in the one place injections arrive. |
| **S-b** | A malformed gate response could reach `accepted`. | `{"verdict":"supported","verdict":"unsupported"}` was read as whichever key `json.loads` kept. |
| **S-c** | **The declared schema was never enforced.** | A 599-char claim and a 30-element citation list both passed. A schema that is only a suggestion to the model is not a schema. |
| **S-d** | A confidence asserted **in prose** was accepted. | "confidence 0.95 that they are terse" smuggled the number into the text. |
| **S-e** | A claim **judging the person** was accepted. | "they are violent and dangerous" — a conclusion about a person, not a communication. |
| **S-f** | **`stable` was reported on one measure** while five others were unsettled. | The null-as-evidence failure the tier exists to prevent. |

### 3.4 A methodological finding worth more than the fixes

> The test suite built its corpora with the library's **own generator**. A
> defect in the extractor and a defect in the generator cancelled out, and
> both looked green.

Hand-authored corpora exposed four measurement bugs immediately — reply
latency measured the wrong interval, emoji detection missed ⭐ and ⏰,
vocabulary confounded by message length, and a claim with parentheses that
could not be parsed back. The hand-written corpora are now checked in under
`verification/corpus/`, so the next person can re-run them.

### 3.5 A defect found by the tests while building T4

The leave-one-day-out check — "does this concentration survive dropping any
one day?" — skipped any day whose removal left fewer than `MIN_OBSERVATIONS`
messages. The day whose removal is the *whole test* is the heavy day, the one
with the **most** messages. So the check skipped precisely the case it
existed to catch, and the pattern it should have rejected passed every time.

### 3.6 Rev 6 — the held-out evaluation exists (G-14 / M8)

The sharpest caveat on every Rev 5 number was that the labels were not
frozen, the corpora were authored by the same person who fixed the defects,
and the residuals were reclassified *after* seeing the results. Rev 6 built
the instrument that removes the caveat, in this order:

1. **The runner was written before any held-out corpus existed.**
   `verification/run_verification.py --holdout` measures labelled, hashed,
   separately-authored corpora through the same verdict vocabulary the
   in-sample tally uses. Its freeze contract **refuses — exit 2, not a
   measurement** — any corpus whose bytes changed after its labels were
   frozen, any labelled corpus that is missing, and any present corpus
   without labels. The refusal path was exercised on a *copy*, so the real
   set stayed byte-untouched.
2. **A second author wrote the data.** A separate-session AI agent — no
   access to the implementation (`src/`), the tests, the spec, the existing
   corpora, the thresholds or any tooling — authored **8 new corpora** and a
   `labels.json` labelling every corpus on all six dimensions, with a
   provenance block disclosing exactly what the authoring brief contained
   (the payload schema, plain-language dimension meanings, the band
   vocabulary — nothing else). Per-corpus SHA-256s were computed by that
   author only after the files were final. This is M8's requirement —
   evidence written by someone who did not write the library.
3. **The labels were frozen before any run.** The labels file's SHA-256 was
   recorded before run #1 (`ff0f1164…`), and the same value appears inside
   both run records — so the labels measured in run #2 are the labels frozen
   before run #1.

**The eight corpora** (messages / subject words / the subject, as written):

| Corpus | Msgs | them-words | The subject, as written |
|---|---|---|---|
| `h1_launch_night` | 38 | 273 | chatty colleague; quick playful volleys; rarely asks |
| `h2_two_cities` | 39 | 564 | long reflective planner; question-rich; no emoji |
| `h3_burst_pipe` | 37 | 93 | terse flatmate; 1–6-word replies; hours of silence |
| `h4_career_crossroads` | 38 | 416 | thoughtful mentee; compact reflective messages |
| `h5_trail_weekend` | 42 | 428 | trip instigator; emoji-dense; minute-fast replies |
| `h6_mums_seventieth` | 40 | 246 | single-line sibling texts, hours apart |
| `h7_overdue_invoice` | 39 | 477 | courteous freelancer; businesslike paragraphs |
| `h8_book_club` | 38 | 524 | book-club neighbour; wide, adventurous vocabulary |

48 labelled rows; 47 graded. The withheld row is `h3`'s vocabulary — at 93
subject words the corpus genuinely cannot support the dimension, and both
sides (the frozen label and the frozen 100-word floor) say so independently:
**a withheld row counts as agreement, which is what a floor is for.**

### 3.7 The defect the held-out set found: E-a, the emoji detector

Run #1 against the pre-repair library graded **28/47 = 0.5957**, and produced
exactly one **value** mismatch: `h5`'s emoji rate measured **0.6364** (14 of
22 messages) against a frozen **0.7273** (16 of 22). A value mismatch is a
candidate defect, not a boundary opinion — two of the author's emoji messages
were simply not being seen.

They carried `🛠️` (U+1F6E0 + U+FE0F) and `🌬️` (U+1F32A + U+FE0F). Checked
against upstream (`emoji-data.txt`, Unicode 15.1): both base characters are
`Emoji=Yes` but **text-presentation by default**, so the annotated
default-presentation table was right to exclude them — but the messages
carried the variation selector that *forces* emoji presentation, and
`has_emoji` ignored it. The module's own docstring had already contradicted
itself: it said text-presentation characters "need a variation selector to
render as an emoji", then did not count them when the selector was actually
present.

**The repair**, all upstream-derived, in `trace_traits.emoji`:

- `TEXT_PRESENTATION_RANGES` is now **complete**, generated mechanically from
  `emoji-data.txt` as (`Emoji` − `Emoji_Presentation`) — 119 range rows over
  219 points, including the legacy ASCII keycap building blocks
  (`#`, `*`, `0–9`, `©`, `®`) held to the same rule.
- The **selector is the receipt**: a text-presentation character actually
  followed by `U+FE0F` *in the message* counts in the default reading. A bare
  one (a typed heart, a bare `9`) still does not;
  `include_text_presentation=True` keeps its meaning for the looser reading.
- The default-presentation table itself was re-diffed point-wise against
  upstream and corrected: it had **over-claimed** two points (`U+1F202`,
  `U+1F5A5`) and missed none. Both tables now differ from upstream in
  **zero** points.

Six regression tests pin the repair — the exact held-out messages, the
forced/bare pairs, the keycap reading, the two over-claimed points, and a
table-completeness pin (1205 + 219 points) so any later re-cut drifts loudly.

### 3.8 The two held-out runs, before and after

| | Run #1 (pre-repair) | Run #2 (post-repair) |
|---|---|---|
| Graded agreement | 28/47 = **0.5957** | 29/47 = **0.617** |
| Withheld row | 1/1 | 1/1 |
| Overall | 29/48 = 0.6042 | 30/48 = **0.625** |
| Tally | MATCH 29 · WRONG-BAND 18 · **MISMATCH 1** | MATCH 30 · WRONG-BAND 18 · **MISMATCH 0** |
| Labels SHA-256 | `ff0f1164…` (frozen pre-run) | **the same** — verified inside the record |
| Planted tally that day | unchanged | unchanged (`MATCH 93 · WRONG-BAND 3`) |
| Golden fixtures | — | **no drift** after the repair |

The emoji fix moved exactly the one row it should have, and nothing else —
which is itself a useful confirmation that the change did what it claimed:
the planted set contains no selector-forced text-presentation character, so
its readings were untouched and the golden snapshot still verifies.

**The 18 residual rows are all band-boundary disagreements, and every one has
a measured value that matches the frozen statistic.** They split:

| Class | Rows | Shape |
|---|---|---|
| Response length, absolute vs relative | 5 | The author judged length in absolute ordinary-terms; the library bands against the observer baseline exactly as declared (PM-004.3) — 74–125 chars reads "ordinary" to a human and `high` relative to a terse observer; one 59-char row reads the other way. |
| Vocabulary at the declared 0.80 cut | 6 | Values 0.79 and 0.81 land either side of the boundary, **split in both directions** — the signature of a boundary, not a bias. |
| Initiation, 3-conversation quantisation | 4 | With three conversations the statistic is quantised to thirds (0.333/0.667), and the moderate/high boundary between them is thinner than the sample's granularity. |
| Question-rate cut placement | 2 | 0.15 (low→moderate) and 0.318 (high→moderate) — the cut sits between the two readings in both directions. |
| Reply latency, `h3`'s burst shape | 1 | Hours of silence punctuated by burst replies; the median *reply* gap is 15.5 min (`moderate`) while the reader weighed the silences (`high`). The metric is declared and would be wrong to redesign against one corpus. |

**Protocol notes.** Run #1's ledger is archived as
`verification/holdout_ledger_pre_emoji_fix.json`; run #2's is the live
`verification/holdout_ledger.json`; both consoles are kept. **No band
threshold moved for any of this.** None of these numbers is a claim about
people: the corpora are AI-authored and disclosed as such in the runner's
output.

### 3.9 Rev 6 — the policy decisions that fell to it (G-11, D5/M−1, D9, G-9)

- **G-11, the vocabulary bands.** Decided: a band word means what it means
  **at the preferred 100-word window**, and the dimension produces a band
  only when the corpus supports it — **≥ 100 subject words and ≥ 2
  conversations**, else `insufficient_evidence` naming what is missing. The
  window-shrinkage machinery is deleted from the extractor *and* from the
  independent ground-truth calculator, so no two implementations of "how
  small is too small" can drift. **The cuts (0.45 / 0.80) did not move; the
  floor did the work.** Planted vocabulary disagreements went **8 → 0**.
- **D5 / M−1, the weights.** The documented example weights are now
  `trace_traits.reference_policy` — `REFERENCE_SOURCE_WEIGHTS`,
  `REFERENCE_HALF_LIVES`, `REFERENCE_POLICY`, `reference_scorer()` — with
  four dedicated tests; the CLI's default policy delegates to it. The core
  ships **no** default: scoring without a policy is a construction error,
  not a placeholder.
- **D9, the prompt-version registry.** `PROMPT_VERSIONS` in
  `trace_traits.perception` — frozen (`MappingProxyType`), keyed
  `name/version`, refusing an unknown key with the list of known ones. Every
  perception-stage finding records its prompt version.
- **G-9, the clock.** Implemented as decided: T4 reads the hour **relative
  to the person's own median message time** (wrapping the clock), which
  needs no timezone and is immune to DST and travel. An absolute-hour
  reading remains the integrator's declared policy input, never a default.

### 3.10 Rev 6 — the statistics landed (G-13/G-10), and the contextual predictor (G-12)

All three statistics gaps closed with a standard-library implementation
(`statistics.NormalDist` for the probit — no Decimal, no hand-rolled
inverse):

- **Independence:** clustered intervals where they belong —
  `cluster_median_ratio_ci` and `cluster_proportion_difference_ci` bootstrap
  **by conversation**, with `MIN_CLUSTERS = 2` below which the result is
  `None` rather than an over-confident interval.
- **Multiplicity:** T3 judges its six measures as a **family** with
  Bonferroni-adjusted level (`FAMILY_ALPHA = 0.05`, per-measure
  `FAMILY_ALPHA / family_size`) — six 95% tests were a ~26% family-wide
  false-positive rate, and that is what the family-wide level now declares.
- **Selection bias:** T4's best-window search runs a **permutation test**
  (`PERMUTATIONS = 200` under seed `20261020`, hole-clipped at
  `MIN_PERMUTATIONS = 100`, significance at 0.05) before a concentration can
  be reported — and it knows the difference between *unlucky* and
  **vacuous**: when every relabelling reproduces the share, the null cannot
  fail, and the band is downgraded with a detail saying so. The Rayleigh-test
  suggestion from the original gap text was **not** adopted; the recorded
  fix is the permutation test with the vacuous-null guard.
- **G-10, weekday and silence:** landed in `trace_traits.temporal` with
  gates of their own — `MIN_OBSERVATIONS = 6`, `MIN_WEEKS = 3`,
  `MIN_WEEKDAY_SHARE = 0.35`, and `DEFAULT_SILENCE_DAYS = 3.0` — because a
  weekday bucket needs a larger sample than a 15-minute window and must not
  inherit the window's gates.

**The contextual predictor** (`build_contextual_predictor`,
`ContextualReplyPredictor`, `ContextualSkill`) conditions the T2 reply
prediction on the observed context — the subject's own reply window, question
openings, message lengths — smoothed toward the base cell
(`DEFAULT_SHRINKAGE = 8.0`), and scores itself with a **leave-one-out Brier
skill** against the base rate over every observed reply opportunity:

| Corpus | Contextual skill (LOO, vs base rate) | Opportunities |
|---|---|---|
| planted, where context carries signal | **+0.188** | 72 |
| planted, window-conditional cell | **+0.823** | 24 |
| generated `terse` persona | −0.011 | 196 |
| generated `average` persona | +0.041 | 342 |
| generated `verbose` persona | −0.009 | 496 |

The near-zero rows are the honest result, not a failure: the generator plants
no conditional structure, so a contextual predictor has nothing to find
there — exactly what the gap text predicted it would report. The planted rows
show the skill is real where the context actually carries information. The
evaluation record carries the skill, so `trace-traits evaluate` reports a
distribution with a measured skill rather than a rate pretending to be a
prediction.

### 3.11 Rev 6 — publication hygiene (M1, M10, and the reclassification)

- **M1, the golden fixtures.** `verification/golden_fixtures.py` freezes, for
  **20 corpora**, every T1 reading (band, value, evidence counts, detail) and
  every T2 prediction into `verification/golden.json`, with its own SHA-256
  over the readings section (`b25141f0a9a0adb8…`) — a hand-edited snapshot is
  refused, not honoured. No argument verifies; `--freeze` re-snapshots, so
  any change to what the library says is a diff visible in review. After the
  E-a repair it re-verified with **no drift**.
- **M10, the sanitization.** Every consumer-specific item — file-level
  provenance, the six findings about the consumer's code (F-a…F-f), the
  consumer milestones and their gaps, the internal references — was moved to
  a **private document outside this repository**. The public spec now says
  "the reference integration" / "the consumer" and names none of it. M7
  (screenshot adapter) and M9 (consumer adoption) are thereby **reclassified
  as consumer records**, outside TRACE's definition of done, by owner
  decision.
- **The licence is MIT.** What remains before M10 is the upload itself —
  PyPI/GitHub — which is the maintainer's action, not engineering.

---

## 4. Measured results

### 4.1 Trait recovery — 93 of 96 graded (96.9%), after G-11

Against hand-authored corpora with known ground truth. The number comes from
`verification/run_verification.py`, which computes the ground truth with a
calculator that **does not import the library**:

| | Before this work | After the first audit round | Now |
|---|---|---|---|
| Match | 71 | 88 | **93** |
| Mismatch (value) | 10 | 0 | **0** |
| Wrong band | 7 | 8 | **3** |
| Not graded (`insufficient_evidence`) | 8 | 0 | 24 — insufficient on **both** sides |

**The three remaining disagreements are the reply-latency band rows** —
planted `low` at a 7.0-minute median against the declared `< 5 min` cut. The
vocabulary half — the 8 rows that motivated G-11 — went to zero when the band
words were tied to the 100-word window and the floors enforced. **No
threshold moved; the floor did the work.**

**The distinction still matters.** These are not measurement errors — the
values agree with an independent calculator. They are disagreements about
where to put a boundary, and a boundary is a **policy** number.

**Caveats — one is now answered, two stand.**

1. **The corpora are hand-authored by one auditor.** They test whether the
   *machinery* measures what it claims to measure; they do not establish how
   the algorithm behaves on real conversations, and 96.9% is not a claim
   about people.
2. **The planted-set gain is in-sample** — the fixes were made against these
   corpora and then measured on them. This is the caveat Rev 6 was built to
   answer: the **held-out set** (§3.6–§3.8) — 8 corpora authored by an
   independent second writer, labels frozen before any run — grades **0.617
   out-of-sample with zero value mismatches**, every residual a boundary
   disagreement against the frozen labels. The planted number still carries
   its in-sample caveat honestly; the out-of-sample claim now has a
   measurement.
3. **The 107 requirements are counted, not audited.** A count says the
   statements are present; it does not say each is implemented.

### 4.2 Prediction (T2) — the base-rate layer says what it is; the contextual layer is measured

| Persona | Length MAE | vs base rate | IQR coverage | Emoji Brier | vs base rate | Response rate (pred vs actual) |
|---|---|---|---|---|---|---|
| terse (12 chars) | 3.2 | 3.4 (skill **+0.055**) | 0.52 | 0.0677 | 0.0677 (skill **0.000**) | 0.852 vs 0.816 |
| average (60 chars) | 16.2 | 16.2 (skill **+0.003**) | 0.47 | 0.1878 | 0.1878 (skill **0.000**) | 0.766 vs 0.766 |
| verbose (180 chars) | 35.2 | 35.2 (skill **0.000**) | 0.45 | 0.2491 | 0.2491 (skill **0.000**) | 0.640 vs 0.661 |

96 held-out messages per persona.

**The honest reading, both halves.** The *base-rate* layer reports what it is
— the subject's own rates with intervals, and a skill of **0.000** against a
baseline that predicts exactly that. That was the Rev 5 audit's finding (the
evaluation had compared against a constant-0.5 prediction and called the
person's non-randomness a win), and the correction is not cosmetic.

The *contextual* layer is the answer to it, measured (§3.10): skill
**+0.188** over 72 planted opportunities and **+0.823** on the
window-conditional cell — where context carries signal — and ≈0 on the
generated personas, where there is no conditional structure to find. The
evaluation record carries the skill, so the CLI reports a distribution with a
measured skill rather than a rate pretending to be a prediction.

Coverage converges to 0.50 (0.25 → 0.375 → 0.453 → 0.539) as the sample
grows.

### 4.3 Response contrast (T3) — 5 of 5 recovered

| Planted reactivity | Band | Length measure | Interval |
|---|---|---|---|
| 0.00 | `unsettled` | `equivalent` | [-0.065, +0.125] |
| 0.25 | `unsettled` | `equivalent` | [-0.240, -0.080] |
| 0.50 | `reduced` | `lower` | [-0.427, -0.284] |
| 0.80 | `reduced` | `lower` | [-0.667, -0.568] |
| 1.00 | `reduced` | `lower` | [-0.816, -0.770] |

**The two no-change cases report `unsettled`, not `stable`.** That is
PM-007.3.2: `stable` requires that *every* measure was equivalent, and at 24
conversations only reply length is. Reporting `stable` on one measure was the
null-as-evidence failure — caught by the audit, and deliberately changed.

**"5 of 5 recovered" is generous, and should be qualified.** Two of the five
are "no change" plants and both come back **`unsettled`**, not `stable` — the
data did not settle them, and the tier said so rather than claiming an
absence it could not establish. And the 0.25 plant's length interval is
`[-0.240, -0.080]` — entirely **below** zero, so an 8–24% shortening — which
the declared ±0.25 margin treats as negligible. So that case is "recovered"
in the sense that the tier did *not* report a change; it is not a detection.
The margin is a **policy** number, and a different integrator would set it
differently. (The equivalence test uses a 95% interval, which is *stricter*
than conventional TOST — see PM-007.4.)

### 4.4 Temporal (T4) — verified against a live model

| Property | Result |
|---|---|
| Separates a bedtime signoff from a busy evening at the same hour | **20/20** (a smoke test, not a benchmark) |
| The arithmetic finds them in different windows | `23:00–23:14` vs `23:30–23:44` |
| The reading survives the text check | verified |
| A fabricated citation | dropped, counted |
| A wrong-speaker citation | refused |
| A reading outside the closed set | refused |
| No model at all | the label-free rung still reports the clock |

**What 20/20 is and is not.** It is a **smoke test**: one synthetic corpus,
one model, one run, twenty messages. It says the mechanism works end to end
and that the injection defences hold. It is **not** an accuracy measurement,
and it is not a claim about how well a model labels real chat — the labels
were checked against a hand-built set of 20 messages with obvious ground
truth. A real number needs a real corpus and an agreement statistic between
annotators.

**The interpretation vocabulary is a boundary.** `bedtime` labels a *cluster
of sign-off messages at a particular hour*. It is not a claim about when
someone sleeps, and the library cannot support that claim — the messages do
not say. The reading is checked against the cited text so at least the
*sign-off* claim is verifiable, and the caveat travels with every reading.

### 4.5 Performance

Measured on this machine (8 cores, Python 3.14), 50,000 messages across 5,000
conversations:

| Operation | Time |
|---|---|
| `corpus_from_messages` | 0.05s |
| `extract_features` | **9.6s** |
| `build_predictor` | **8.5s** |
| `evidence_from_corpus` (fingerprint) | **0.27s** (was 12.96s — a 48× improvement) |
| `reactivity` | 0.07s |
| `Scorer.score`, 10,000 links | 0.013s |

**These numbers were wrong in an earlier draft of this summary**, which
claimed 1.32s and 1.15s for the first two. An audit caught it and
re-measurement confirmed the larger figures. The overstatement was in the
document, not in the code: the library was never faster than this. The
fuzz harness's own threshold — "completes in seconds, not minutes" — passes
at these values.

`extract_features` and `build_predictor` are O(n) in messages but each does
per-message work (regex over the text, an emoji scan), so ~10s for 50,000
messages is the honest figure. A caller with a corpus of that size should
expect it, and the fuzz harness asserts it stays in seconds rather than
minutes.

### 4.6 What the guards do and do not catch

The perception tier refuses proposals that carry a confidence, that judge the
person rather than the writing, that cite an id which does not resolve, or
whose reading is unsupported by the cited text. Those checks are **regular
expressions**, and it is worth being exact about what that means:

- **They are English-only.** `_REQUIRED_MARKERS` looks for English words
  ("bed", "night", "work", "deadline"). A corpus in another language passes
  the text check trivially, so a reading in that corpus is *not* verified.
- **They are bypassable by paraphrase.** "I'm off to bed" and "time for my
  pillow" both read as bedtime; "not going to sleep over this" does not,
  even though it is about sleep. A phrase the regex does not know slips
  through, and one it mis-reads is rejected.
- **They are a floor, not a proof.** A passing check means "the cited text
  contains something of the expected kind", which is necessary and not
  sufficient for the reading to be right.

The design does not rely on them for safety. The **arithmetic** is what
decides whether a pattern exists, and a model cannot talk it into reporting
one. The guards are there to keep the *wording* honest; a bypassed guard
produces a badly-worded reading, not a false pattern.

### 4.7 Three statistical limitations — now fixed (G-13)

Rev 5 recorded three statistics gaps as open. **Rev 6 closed them** (§3.10);
this table is kept for the record, with what actually happened:

| # | Limitation (as recorded) | Effect | Now |
|---|---|---|---|
| 1 | Messages within a conversation are not independent | Every interval was optimistic about coverage. | **Fixed:** cluster bootstrap **by conversation** (`trace_traits.stats`), `MIN_CLUSTERS = 2` below which the result is `None`. |
| 2 | T3 computed six measures at 95% each (~26% family-wise) | T3 reported a change more often than it should. | **Fixed:** the family is judged at a Bonferroni-adjusted level (`FAMILY_ALPHA = 0.05` / family size). |
| 3 | T4 picked the best window and then tested it (selection bias) | T4 could report a pattern where there is none. | **Fixed:** a permutation test (`PERMUTATIONS = 200`, seed `20261020`, level 0.05) gates every concentration, with a **vacuous-null** guard — when the null *cannot* fail, the band is downgraded and the detail says which. The Rayleigh-test suggestion was not adopted; the permutation + vacuous-null path is the recorded one. |

---

## 5. The properties that make it trustworthy

These are not features. They are the reason the numbers mean something.

**The model proposes; the evidence disposes.** Confidence is computed from
cited evidence alone. A model never assigns one. A citation that does not
resolve is **dropped, never repaired**. A citation from another speaker is
dropped. A claim with no surviving citation is discarded.

**Every score reconstructs.** The value is a product of named terms, and the
terms are reported. A reader can say "0.55 because 0.70 × 1.0 × 0.79 × 1.0"
or check the arithmetic themselves.

**The null is never a finding.** To report `stable`, every interval must lie
entirely inside its margin. A failure to reject is reported as `unsettled`.
The same rule governs T3's `stable` and T4's pattern bands — and now T4's
window search too, where a null that *cannot* fail is reported as vacuous
rather than as a pattern.

**Bands, never meters.** A band describes communication, not feeling. The
underlying number is kept for auditing, not for display.

**`trace_core` imports the standard library only.** No web framework, no
database, no HTTP client, no tenant model, no third-party package of any
kind. It embeds nothing, calls no model, and reads no clock.

**Every figure states whether it is synthetic, hand-authored, or
independently authored.** The library distinguishes "no evidence" from
"evidence of nothing" everywhere, including in the naming of its own gaps —
and the held-out tier keeps its authoring provenance in the runner's output.

---

## 6. What is not done, and why

### 6.0 Publication state — the Rev 5 blocker is resolved, not waived

Rev 5 named the ownership question as the release blocker: the spec
documented ports from a commercial product, recorded findings about that
product's internal behaviour, and stated the project was confidential.

**Rev 6 resolved it by removal, not by clearance.** Every consumer-specific
item — the file-level provenance, the six findings (F-a…F-f), the consumer
milestones M7/M9 and their gaps, the internal normative references — now
lives in a private document outside this repository, and the public spec
refers to it only as "the private integration notes". TRACE reads as the
standalone library it is; the consumer is unnamed in every public document.

What remains before M10 is the **upload itself** (PyPI/GitHub) — the
maintainer's action — and the standing G-3 below.

### 6.1 The remaining open items

| Item | Status | Why |
|---|---|---|
| **G-3** Real-data validation | **Open** | Every synthetic, hand-authored and independently-authored number above verifies the *machinery*, not a claim about people. What this needs: **two trained annotators** and a labelled corpus set with **Krippendorff's α ≥ 0.67**. Every piece of tooling for it exists — the golden fixtures freeze readings, the G-14 runner measures frozen labels, the annotation path is the one G-14 exercised. What is missing is **the two people**, whose availability is the maintainer's decision (recorded privately). **This is the one that matters.** |
| **G-4, G-5** | Moved out | Consumer-side records; the structural fixes live in `trace_traits` (observer baselines, conversation gating) and the consumer's own backlog lives in the private notes. |
| **D6** `Corroborated` is also "attributed" | Open | A naming question for the next doc pass; nothing blocks on it. |
| **D7** T3's equivalence margin | Open | A policy number per integrator, with a sample-size cost attached. |
| **D8** Perception tier in-process | Open | Allowed (no shared state); a strict deployment boundary is the operator's choice. |
| **M10** Publish | Open — user-side | The upload (PyPI/GitHub) is the maintainer's action; everything publishable is publish-ready. |

**What TRACE does not claim.** It does not read minds. It cannot tell you
someone was angry, or that they sleep at eleven, or that a pattern is a
personality. It measures communication, and where it offers an
interpretation it marks it as one and shows the text it rests on —
*possibilities, not predictions.*

---

## 7. Where to look

| | |
|---|---|
| **PLAN.md** | The spec. 1,656 lines. §9 is the build ledger with every measured result — §9.16–§9.22 record Rev 6 (the held-out runs, E-a, the decisions, the statistics); §9.9–9.15 the audits; §10 the gaps; §11 the decisions; §12 the definition of done. |
| **README.md** | The seven runnable examples. Every one executes. |
| **verification/VERIFICATION_REPORT.md** | The audit findings, in full, with the reasoning. |
| **verification/corpus/** | The hand-authored corpora. |
| **verification/holdout/** | The frozen G-14 set — 8 corpora and `labels.json`, written by the independent second author. **Do not edit after the freeze; the runner refuses a byte change.** |
| **verification/holdout_ledger.json** + **holdout_ledger_pre_emoji_fix.json** | Run #2's record, and run #1 archived. |
| **verification/golden_fixtures.py** + **golden.json** | The M1 snapshot — 20 corpora, self-checksummed, `--freeze` to re-snapshot. |
| **verification/run_verification.py** | The planted ledger, and `--holdout` for the frozen evaluation. The ground truth is computed by `ground_truth.py`, which **does not import the library**, so a value mismatch is a real disagreement rather than a stale number. |
| **verification/fuzz_harness.py** | 100 checks against degenerate, malformed and hostile inputs. |
| **tests/test_regressions_audit.py** | 23 tests, each naming the defect it pins down. |
| **tests/test_regressions_traits.py** | The E-a pins — the exact held-out messages, the forced/bare pairs, the keycap rule, the table-completeness counts. |
| **tests/test_contextual_predict.py**, **tests/test_g13_statistics.py** | The G-12 and G-13 machinery, measured. |
| **tests/test_*_live.py** | The live-model tests. Skipped without `TRACE_LLM_API_KEY`. |

---

## 8. How to check this summary

```bash
cd TRACE    # the repository root
PYTHONPATH=src python3 -m pytest tests/ -q \
  --ignore=tests/test_temporal_live.py --ignore=tests/test_perception_live.py
                                                 # 440 passed, 1 skipped
PYTHONPATH=src python3 -m pytest tests/ -m live  # with a key set: 14 passed
PYTHONPATH=src python3 verification/fuzz_harness.py
                                                 # PASS=100 FAIL=0
PYTHONPATH=src python3 verification/run_verification.py
                                                 # MATCH 93, WRONG-BAND 3
PYTHONPATH=src python3 verification/run_verification.py --holdout
                                                 # graded 29/47 = 0.617; overall 30/48
python3 verification/golden_fixtures.py          # GOLDEN OK — 20 corpora match the snapshot
PYTHONPATH=src python3 -m trace_traits.cli evaluate
                                                 # per-persona skill vs the base rate,
                                                 # incl. the contextual layer
ruff check . && mypy src/ --strict               # both clean
```

**A note on the ledger's provenance.** `verification/ledger.json` was
regenerated during the audit work, because the independent ground-truth
calculator was still computing the *pre-audit* plain type-token ratio while
the library had moved to MSTTR. That staleness made the checked-in ledger
report 71 matches against a library that now measures 93. The fix was to the
**calculator**, not to the number: `ground_truth.py` computes MSTTR
independently — restated rather than imported, so the two implementations
can still disagree and the disagreement would be real. The house rule that
grew from this: never move a threshold to make tests pass; re-measure and
record the changed numbers as reviewed fixes.

The frozen holdout set is never touched while checking: the tamper-refusal
path was exercised on a copy, not on `verification/holdout/`.

With a key set, the live-model tests run too:

```bash
TRACE_LLM_API_KEY=... python3 -m pytest tests/ -m live   # 14 passed
```
