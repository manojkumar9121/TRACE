# TRACE

**T**ime-aware, **R**eferenced, **A**ttributed, **C**orroborated **E**vidence.

A general-purpose library that turns mixed evidence (chat exports, email
threads, transcripts, documents) into a parameterised catalogue of observed
behaviour, with every value traceable to the evidence that produced it.

TRACE computes and explains. It does not decide what the numbers mean: the
integrator owns the thresholds, the vocabulary and the names.

Two packages:

| | |
|---|---|
| **`trace_core`** | The scoring and attribution layer. Given evidence and a claim, it produces a score with every term named, and a decision with a stated reason. It embeds nothing, calls no model, and reads no clock. Also the reference validators, the evidence packet, and the freshness gate. |
| **`trace_traits`** | The trait layer. Measures how a person communicates, scores each measurement as a TRACE claim, and predicts what they are likely to do next, as a distribution with an interval and a caveat. |
| **`trace_traits.perception`** | The LLM tier. A model reads a corpus and proposes candidate observations with citations; every citation is resolved and attributed, and the model assigns no confidence. Proposals are judged by what they cite. |
| **`trace_traits.temporal`** | When they write, and what that means. The model labels each message's communicative purpose; the arithmetic asks whether each purpose clusters at a particular clock window; a reading of the strongest pattern is checked against the cited text. |

Four tiers of inference, in increasing difficulty: **T1 trait** (a level over
the corpus, built), **T2 behaviour** (a distribution over what happens next,
built), **T3 response** (a contrast against a matched baseline, built),
**T4 temporal** (when they write, and what that means, built).

> **What T2 claims.** Two layers, stated apart. The base-rate layer
> (`build_predictor`) returns the subject's own smoothed historical rates,
> a rate with an interval, sample size attached. Against a baseline that does
> the same thing, its skill is 0.000 by construction, and it does not claim
> otherwise. The contextual layer (`build_contextual_predictor`) conditions
> on the conversation: the subject's own reply window, question openings,
> message lengths. It is scored by a leave-one-out Brier skill against the
> base rate over every observed reply opportunity. Where context carries
> signal the skill is real: **+0.188** on a planted context-conditioned
> corpus (72 opportunities), **+0.823** on the window-conditional cell.
> Where a corpus has no conditional structure the measured skill is ≈0, and
> that is the number you get ([PLAN.md](PLAN.md) §9.20).

> **Status.** **440 tests pass** (plus 14 live-model tests when an API key
> is set), ruff clean, `mypy --strict` clean, and `trace_core` imports
> nothing beyond the standard library. All CLI entry points work, the docs
> are in place, and PyPI publishes remain manual until the first upload
> (see §9.x). The evaluation was **held out**: eight corpora with
> hash-frozen labels, authored independently of the implementation, grade
> **0.617** out-of-sample agreement (30/48 labelled rows, zero value
> mismatches). The specification, with the build ledger and every measured
> result, is in **[PLAN.md](PLAN.md)** (TRACE-SPEC-001, Rev 6).

## What it's for

Six uses this fits, and one that needs caution. All of them follow the same
rule: describe the communication, cite the messages it came from, and say
nothing when the evidence is thin.

1. **Reply suggestions.** Whoever's messages you hold (your own habits, or
   a contact's rhythm), T2 models reply timing, length, emoji use and
   questions: do they answer in minutes or hours, in one line or
   paragraphs. The predictors return a distribution, and the measured skill
   tells you when the context carries no signal, so the interface can hide
   suggestions there instead of guessing. Most suggestion bars show
   something whenever a model can produce it; here the decision to show is
   a data question. An unsettled reading is information too.
2. **Timing.** T4 measures when someone writes and how fast they reply,
   relative to their own median hour. Midnight, DST and travel are handled,
   and no location data is needed. There is also a long-silence reading
   from the same tier (gaps around three days). The shape of the product:
   "best hours to reach Rhea", "Priya takes ~2h on weekdays", "no reply
   from Sam in five days". Descriptive reporting about your own contacts,
   with their consent, and quite different from a last-seen indicator.
3. **Your archive.** Run a chat export you own through it and get a
   six-dimension profile, bands with sample sizes, no meters. Add T3 and
   you can check whether your replies actually got shorter after some
   incident, or whether it only feels that way, contrasted against your own
   matched baseline with the episodes and the margin stated.
4. **Assistant tone.** Use a recipient's measured profile to shape an
   assistant's outgoing messages: length, emoji use, question rate. The
   freshness half-lives keep the profile moving as habits drift, and every
   profile statement traces to the messages it was measured over.
5. **Chatbot memory gating.** The half with no chat traits in it is
   `trace_core`: score any model-made claim about a person by the evidence
   behind it. "She prefers weekends" and "he's frustrated with billing" are
   the same problem. One observation never becomes a known fact (INV-1
   requires two independent lineages), stale beliefs decay on half-lives
   instead of lingering, uncited output is dropped instead of repaired, and
   injected text cannot mint confidence. Any product with an LLM forming
   beliefs about users has this problem, and this is the piece worth
   borrowing. It also doesn't start from zero: an existing assistant memory
   (the kind ChatGPT or Claude keeps) can be imported as declared evidence.
   Memory the user wrote or confirmed enters at the human-asserted weight;
   memory the model inferred enters at the inference weight, and unknown
   ages are treated conservatively. Later conversations corroborate a seed
   up or let it decay.
6. **Stylometry.** Every number reconstructs from named terms, every figure
   declares whether it is synthetic, hand- or independently-authored, and
   the frozen-label evaluation protocol (labels locked by a second author
   before any run) is a method you can copy for your own feature's evals.

### Careful here: predicting what someone is about to send

Prompting ahead of the user ("watch them type, dispatch the model call
before they press send") is technically possible, and three brakes matter:

1. TRACE measures coarse habits, not content. The held-out grade is 62%
   agreement on six band words. Keystroke-level prediction is the LLM's
   job, and that is a different problem.
2. A wrong pre-suggested prompt costs more goodwill than a right one
   earns. This is where the intervals matter most: a wide interval should
   read as "don't".
3. Dispatching text the user hasn't sent raises consent questions.
   Prediction is not permission.

If you build it, treat the intervals as the per-person rule for when a
guess is worth spending on: high predicted probability, narrow interval,
low cost of being wrong. Where any of the three is missing, stay quiet.

Chat-derived habits cannot be conjured from nothing. Below the floors the
library declares (100 subject words, two conversations), the trait
dimensions return `insufficient_evidence` instead of a guess. The belief
layer from use 5 can start earlier, because a memory file you already hold
can be seeded in at the weight its provenance deserves; but no seed
substitutes for measuring a habit a chat never showed. The held-out grade
is band-word agreement on six coarse dimensions, so read every output as a
prior. If a use case needs consent you don't have, read
**[Intended use and non-goals](#intended-use-and-non-goals)** first.

## Measuring and predicting a person

```python
from datetime import datetime, timezone

from trace_core import Scorer, TableConfidencePolicy, TableRetrievalPolicy
from trace_traits import (
    Message, Person, build_profile, build_predictor, corpus_from_messages,
)

messages = [
    Message(message_id="m1", conversation_id="c1", speaker="them",
            text="yeah that works", sent_at=datetime.now(timezone.utc)),
    # … more messages, across more conversations
]
corpus = corpus_from_messages(Person(subject="them", observer="me"), messages)

conf = TableConfidencePolicy(
    source_weights={"user_stated": 1.0, "message": 0.70, "system_inference": 0.55},
    half_lives={"behavioral": 270, "distributional": 540},
    corroboration_floor=2,
    coverage_denominator=3,
)
scorer = Scorer(
    confidence=conf,
    retrieval=TableRetrievalPolicy(
        weights={"vector": 0.55, "recency": 0.20, "source": 0.15, "confidence": 0.10},
        source_weights=conf.source_weights,
    ),
)

profile = build_profile(corpus, scorer=scorer, now=datetime.now(timezone.utc))
for reading in profile.readings:
    print(reading.as_line())        # "Response length   moderate   confidence 0.55 …"

for prediction in build_predictor(corpus).all_predictions():
    print(prediction.describe())    # "Reply length: 42.00 chars [18.00, 91.00] (32 observations; …)"
```

From the command line:

```sh
trace-traits evaluate                       # the measured evaluation
trace-traits profile corpus.json --subject them --observer me
```

### Validators

A model returns a payload, and some of it cites evidence. The rule
everywhere else in the package applies here too: a citation to an unknown
id is not repaired.

```python
from trace_core import run_validators

# `payload` is a model's answer; `packet` is what the evidence set actually
# contains. The validators check the answer against the packet.
payload = {
    "interpretations": [
        {"id": "i1", "text": "they reply briefly", "temporal_scope": "present",
         "citations": ["m1"]}
    ]
}
packet = type("P", (), {"citation_ids": ("m1",)})()

report = run_validators(
    payload,
    allowed_ids=packet.citation_ids,   # what the packet actually contains
    simulation_allowed=False,
)
if not report.ok:
    print(report.reason())              # names the kind of problem, never the claim's content
```

A failed validator degrades the payload instead of failing the turn.
Throwing out an answer because one claim was uncited trades a correct
answer for one bad part. Stripping the bad part keeps the correct part and
removes what cannot be trusted.

### The LLM tier

Arithmetic cannot read meaning, and meaning is what a language model is
for. The perception tier asks a model for observations with citations, then
runs the answer through the same checks as everything else:

```python
import os

from trace_traits import OpenRouterBackend, Message, Person, propose_observations
from trace_traits import corpus_from_messages
from datetime import datetime, timezone

corpus = corpus_from_messages(
    Person(subject="them", observer="me"),
    [Message("m1", "c1", "them", "yeah that works", datetime.now(timezone.utc))],
)

# The model proposes; TRACE verifies. The key comes from the environment, and
# the example degrades to "no proposals" when it is not set, rather than
# raising: a missing key is not a reason to crash a program that was only
# demonstrating a shape.
api_key = os.environ.get("TRACE_LLM_API_KEY", "")
backend = OpenRouterBackend(
    "minimax/minimax-m3.1-flash:free",
    api_key=api_key,
    enabled=bool(api_key),          # no key -> a backend that proposes nothing
)
result = propose_observations(corpus, backend)

for proposal in result.proposals:
    print(proposal.observation.claim, proposal.observation.evidence_ids)

print(f"dropped: {result.dropped_unresolved} unresolved, "
      f"{result.dropped_wrong_speaker} wrong-speaker")
```

Three rules, each enforced in code:

- **The model never assigns a confidence.** Confidence is computed from
  the cited evidence alone.
- **The model never asserts a conclusion.** It is asked for observations
  ("their replies are usually under ten characters"), and verdicts like
  "they are terse" are the caller's to draw.
- **A citation that does not resolve is dropped, never repaired.** A
  repaired citation points at something nobody checked.

Measured against a live model: the schema is honoured about half the time
(providers rename fields), the parser accepts a closed alias list, and with
a single conversation in the corpus every proposed claim scores 0.23 and is
capped `insufficient_corroboration`. The model proposed six plausible
observations and the machinery refused all six on evidence grounds.

### T3: response contrast

The tier that answers "does this person change how they write after
something happens to them". The caller marks which messages were the
events; TRACE measures the difference against the subject's own baseline
in the same conversations, and reports it as a distribution.

```python
from datetime import datetime, timezone

from trace_traits import (
    Margin, Message, Person, Stimulus, StimulusKind, corpus_from_messages, reactivity,
)

corpus = corpus_from_messages(
    Person(subject="them", observer="me"),
    [Message("m42", "c1", "them", "yeah that works", datetime.now(timezone.utc))],
)

stimuli = [Stimulus(message_id="m42", kind=StimulusKind.INSULT)]
reading = reactivity(corpus, stimuli, margin=Margin(length_ratio=0.25))

print(reading.as_line())
for measure in reading.measures:
    print(" ", measure.describe())
```

Three things make this tier different from the other two:

- **The trigger is supplied.** "Is this an insult" is a judgement about
  meaning, and it belongs to whoever read the conversation.
- **The baseline is matched.** Time of day, thread length and topic are
  confounders; drawing the comparison from the same conversations removes
  most of them.
- **"No change" has to be established.** To report `stable`, the 95%
  interval must lie entirely inside the margin. Otherwise the band is
  `unsettled`, because "no evidence of change" is a different claim from
  "evidence of no change".

Measured on a corpus with planted events: **5 of 5 planted reactions
recovered**, with intervals tight enough to separate a 25% shortening from
a 50% one. With too few episodes the band is `unsettled`, never `stable`;
establishing the absence of a change takes more evidence than detecting
one. The band describes messages; a chat log does not settle what someone
felt.

## T4: when they write, and what that means

The same clock hour means different things for different people, and
timestamp arithmetic alone cannot tell those meanings apart. This tier
adds a model that labels what each message is for, then checks whether
each purpose clusters at a particular window. It works in three rungs,
each usable on its own:

```python
from datetime import datetime, timedelta, timezone

from trace_traits import Message, Person, corpus_from_messages, measure_activity

base = datetime(2026, 10, 1, tzinfo=timezone.utc)
messages = []
for day in range(14):
    moment = base + timedelta(days=day)
    messages.append(
        Message(f"s{day}", f"c{day}", "them", "ok, heading to bed. night",
                moment.replace(hour=23, minute=5))
    )
    messages.append(
        Message(f"w{day}", f"c{day}", "them", "still at the office finishing the report",
                moment.replace(hour=23, minute=40))
    )
corpus = corpus_from_messages(Person(subject="them", observer="me"), messages)

# Rung 1: no model at all, just the clock.
pattern = measure_activity(corpus, speaker="them", width_minutes=15)
print(pattern.as_line())
```

The full tier adds a model:

```python
import os
from datetime import datetime, timedelta, timezone

from trace_traits import (
    Message, OpenRouterBackend, Person, TemporalTier, corpus_from_messages,
)

base = datetime(2026, 10, 1, tzinfo=timezone.utc)
messages = []
for day in range(14):
    moment = base + timedelta(days=day)
    messages.append(
        Message(f"s{day}", f"c{day}", "them", "ok, heading to bed. night",
                moment.replace(hour=23, minute=5))
    )
    messages.append(
        Message(f"w{day}", f"c{day}", "them", "still at the office finishing the report",
                moment.replace(hour=23, minute=40))
    )
corpus = corpus_from_messages(Person(subject="them", observer="me"), messages)

backend = OpenRouterBackend(
    "minimax/minimax-m3.1-flash:free",
    api_key=os.environ.get("TRACE_LLM_API_KEY", ""),
    enabled=bool(os.environ.get("TRACE_LLM_API_KEY", "")),
)
reading = TemporalTier(backend).read(corpus, width_minutes=15)

for line in reading.as_lines():
    print(" ", line)
print(" ", reading.caveat)
```

Three things make the division of labour the way it is:

- **The model labels, the arithmetic decides.** The model says what each
  message is (`signoff`, `work`, `social`, `logistics`, `question`,
  `reaction`), and the arithmetic asks whether each purpose clusters at a
  window. A model that labels everything wrongly yields no finding; it can
  only fail to find a pattern, never assert one.
- **A reading must survive the text.** "Their sign-offs cluster at 23:00"
  is a measurement, checkable against the messages. "They go to bed at
  11pm" is a reading of it, and it is accepted only if the cited messages
  actually say something about bedtime. Otherwise it is downgraded to
  `unknown`, not accepted with a caveat.
- **The window is a range, and it wraps.** A sign-off at 23:05 and a work
  message at 23:40 share the 23:00 hour, which hides exactly the
  distinction the tier exists to find. So the window is 15 minutes wide by
  default, and reported as `23:00–23:14`.

Measured against a live model: the tier separates a bedtime sign-off from
a busy evening at the same clock hour 20/20, puts them in different
windows (`23:00–23:14` and `23:30–23:44`), and refuses a fabricated
citation and a wrong-speaker citation.

The band describes the clock, and the reading is labelled as a reading. A
message sent at 23:00 is a fact about a clock; what it means is an
interpretation, and it is marked as one.

**`bedclock` is a label for a cluster of sign-off messages at a particular
hour.** It is not a claim about when someone sleeps, and the library
cannot support such a claim: the messages do not say. The text check
exists so that the sign-off part, at least, is verifiable.

**The text check is a floor.** It is a regular expression, English-only,
and paraphrase bypasses it. A corpus in another language passes it
trivially. What a bypass gets you is a badly-worded reading: the
*arithmetic* decides whether a pattern exists, and no amount of prompt
engineering can talk it into reporting one.

### The traits it measures

Six signals, all counts and ratios over stored messages, with no model
call, so the same corpus always produces the same reading:

| Dimension | What it measures |
|---|---|
| Response length | How long their replies are, relative to the observer's own average when one is available. |
| Emoji usage | Share of their messages containing an emoji, detected against complete Unicode 15.1 emoji-property tables (a selector-forced text-presentation emoji such as 🛠️ or ❤️ counts, the bare form doesn't). |
| Questions | Share of their messages asking something. |
| Initiation | How often they open a conversation. |
| Reply latency | How long they take to reply. |
| Vocabulary | How varied their vocabulary is. |

Each is reported as a band word with the sample size attached. No meters
and no bars: a labelled bar invites people to read a communication habit
as a quantity.

### Why the measurements hold up

A measurement claim is verified by recomputation: the statistic is
recalculated from the cited messages, and the band and the detail both
have to still hold. That is the difference between a trait layer and a
claim generator. A model asked "is this person terse?" returns a
judgement; a measurement returns a number that reproduces or does not.

Every reading cites only the messages it was computed over. A reading over
14 conversation openers cites those 14 openers, and only the subject's own
messages. A claim about a person cannot borrow someone else's behaviour.

### Does it work?

Measured against a synthetic corpus with planted parameters:

| persona | length MAE | IQR coverage | emoji Brier | baseline |
|---|---|---|---|---|
| terse (12 chars) | 3.2 chars | 0.52 | 0.068 | 0.250 |
| average (60 chars) | 16.2 chars | 0.47 | 0.188 | 0.250 |
| verbose (180 chars) | 35.2 chars | 0.45 | 0.249 | 0.250 |

Three of three personas beat the constant-0.5 baseline. IQR coverage
converges to 0.50 as the sample grows, which is what a correctly
calibrated 50% interval does.

**On baselines.** The constant-0.5 comparison says the machinery works and
nothing more. The fair baseline is the subject's own historical rate,
which the base-rate layer scores 0.000 by construction (it *is* that
rate). The prediction claim rests on the contextual layer's measured
skill; see the T2 note at the top.

These are synthetic corpora with planted parameters, which is exactly what
makes them useful for checking machinery, and also what makes them
useless as a claim about real people. The real-conversation question is
open, and this README does not answer it.

## The scoring layer

Every number is a weighted sum or product of named terms, and every term
is visible. A property test enforces it: `value` must reconstruct from
`terms` and the recorded caps.

```python
from datetime import datetime, timezone

from trace_core import (
    Claim,
    Evidence,
    Link,
    Scorer,
    TableConfidencePolicy,
    TableRetrievalPolicy,
)

conf = TableConfidencePolicy(
    source_weights={"user_stated": 1.0, "message": 0.70, "system_inference": 0.55},
    half_lives={"behavioral": 270, "distributional": 540},
    corroboration_floor=2,
    coverage_denominator=3,
)
scorer = Scorer(
    confidence=conf,
    retrieval=TableRetrievalPolicy(
        weights={"vector": 0.55, "recency": 0.20, "source": 0.15, "confidence": 0.10},
        source_weights=conf.source_weights,
    ),
    evidence=[
        Evidence(evidence_id="e1", segment_id="s1", lineage_id="L1", source_id="e1",
                 text="ok", source_type="message", modality="chat",
                 occurred_at=datetime.now(timezone.utc), speaker="rohan"),
        Evidence(evidence_id="e2", segment_id="s2", lineage_id="L2", source_id="e2",
                 text="sure", source_type="message", modality="chat",
                 occurred_at=datetime.now(timezone.utc), speaker="rohan"),
    ],
)

claim = Claim(
    claim_id="c1",
    text="replies in one or two words",
    subject="rohan",
    kind="behavioral",
    scope="historical",
)
links = [Link(evidence_id="e1", stance="supports"), Link(evidence_id="e2", stance="supports")]

now = datetime.now(timezone.utc)  # injected: TRACE never reads a clock
result = scorer.score(claim, links, now=now)

print(result.value, result.terms, result.caps_applied)
```

Confidence is multiplicative, a product of four factors, so any one factor
near zero collapses the number:

```
value = clamp01( src_weight × coverage × freshness × contradiction )
```

Retrieval is additive, four weighted terms, each in `[0, 1]`.

Three details worth knowing before you design around them:

- **Evidence carries a lineage.** The same conversation present as a
  screenshot, a `.json` export and a pasted document corroborates a claim
  once. TRACE cannot verify a lineage; the adapter asserts it.
  `lineage.fingerprint()` is the safe default, so an adapter gets
  copy-detection rather than silently opting out. Cross-modality
  near-duplicates (a screenshot against an export of the same thread) are
  not detected by TRACE; that is fuzzy matching, and it is the adapter's
  job.
- **Stance belongs to the link.** One message can support "replies
  briefly" and contradict "writes long replies" at the same time.
- **Ranking takes candidates you have already scored for similarity.**
  TRACE embeds nothing and generates no candidates; it does the
  deterministic ranking and shows its work.

## What TRACE does not do

- **It does not claim calibration.** `value` is a policy-dependent score
  with auditable terms. It is not a probability, and no measurement here
  establishes that 0.8 means 80% of such claims are correct. The C stands
  for **Corroborated** (independent sources agreeing), not Calibrated.
- **It does not predict people.** `trace_traits` predicts behaviour: a
  distribution over a next message, derived from the person's own
  history. It does not predict personality, emotion, or what anyone feels.
- **It does not process raw media.** Images, audio and PDFs are the
  adapter's problem; an adapter normalises them to `Evidence(text=…)`.
- **It does not call models, embed text, or generate candidates.**
- **It does not ship thresholds.** Core has no default policy, because a
  default is a claim nobody measured.
- **It does not store anything.** No database, no HTTP, no tenant model.

## In this repository

| Path | What it is |
|---|---|
| `src/trace_core/` | The scoring, decision, evidence and gating layer. Standard library only. |
| `src/trace_traits/` | The trait, prediction, perception, response-contrast and temporal tiers. |
| `adapters/mbox.py` | A worked example adapter. Turns an mbox export into `Evidence` with hashed lineages; the reference for writing your own. |
| `tests/` | 440 tests, including the regression files that pin each audit defect by name. |
| `verification/` | The evaluation record: the frozen held-out corpora with their labels and SHA-256s, both run ledgers and consoles, the golden snapshot, the fuzz harness (100 hostile-input checks) and the full audit report. |

If you want the deep version of any claim on this page, `PLAN.md` is the
spec and `SUMMARY.md` links each number to how it was measured.

## Intended use and non-goals

**Advisory.** MIT cannot enforce this section; it is a statement of
intent, and the licence grants whatever MIT grants.

TRACE is for analysing evidence you are authorised to analyse: studying
writing style across a corpus you hold, auditing your own communication
patterns, research on stylistic attribution.

It is **not** intended for inferring the psychological state of people who
have not consented, ranking people, or any decision about a person:
hiring, credit, surveillance, discipline. A behavioural description drawn
from a chat log is an inference about a human being, and that weight
belongs to whoever runs the analysis.

The invariants in the library (`INV-1`–`INV-5`) protect data integrity.
They stop a single utterance, or three copies of one file, from passing as
corroboration. They do not make a prohibited use safe, and they are not a
substitute for consent.

## Naming

| | |
|---|---|
| Directory / project | `TRACE` |
| Distribution (PyPI) | `trace-core` |
| Import packages | `trace_core`, `trace_traits` |

The distribution is deliberately not `trace`: CPython resolves
`import trace` to the stdlib module of the same name, so a package called
`trace` would be unimportable. `trace-core` is free on PyPI; `tracecore`
is taken, so the two are confusable rather than conflicting. Rechecked
before release; availability changes, and this is not trademark
clearance.

## Licence

MIT. See the note above on what MIT does not constrain.
