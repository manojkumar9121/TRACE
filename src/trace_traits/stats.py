"""Cluster-aware statistics, and why they exist.

**G-13: the intervals in this library assumed independence that the data does
not have.** Messages within one conversation are not independent draws — they
share a topic, a mood, a day, and often an argument. Resampling *messages*
treats a fifty-message thread as fifty independent observations, which is the
reason T3's intervals were narrower than they should have been and its
"95% coverage" was optimistic.

The fix here is the one the audit asked for: **resample conversations, not
messages.** A cluster bootstrap draws conversations with replacement and pools
the messages inside whatever it drew, so the replicate reflects the
within-thread dependence the real corpus has. When a response exists in only
one conversation, the interval widens properly instead of pretending the other
conversations each contributed an exchangeable opinion.

What this module deliberately does *not* do:

- It does not smooth or shrink. It resamples the data the caller gave it,
  and the interval is the honest width of *that* data under exchangeability
  **at the conversation level**.
- It does not sort or validate. A garbage corpus produces a garbage
  replicate distribution, exactly as it should.
- It is not a model of a person. It is arithmetic over whatever was supplied.

Everything is deterministic for a fixed ``seed`` — the same corpus and seed
produce the same interval, every run.

Determinism is a property, not a preference: an interval that moved between
runs could not be cited twice. The default seeds across this library were all
fixed at one date (``20261020``) so a re-run reproduces the documented ledgers.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence
from statistics import NormalDist, StatisticsError

__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_SAMPLES",
    "MIN_CLUSTERS",
    "cluster_median_ratio_ci",
    "cluster_proportion_difference_ci",
    "iid_bootstrap_median_ratio_ci",
    "percentile_ci",
    "probit",
]

#: The seed every random process in the library shares.
DEFAULT_SEED = 20261020

#: The default number of bootstrap replicates. 2,000 was enough for a 95%
#: two-sided percentile interval; a Bonferroni-adjusted family level digs into
#: the outer 0.4% of tails, where order statistics from 2,000 replicates are
#: too jittery to reproduce, so the default is 4,000. It is a parameter, not a
#: promise: a caller who needs tighter tails passes more.
DEFAULT_SAMPLES = 4_000

#: Below this many clusters a conversation-level interval is meaningless:
#: resampling a single conversation with replacement produces the same pooled
#: sample every time, and the "interval" collapses to the point estimate. The
#: callers fall back to the declared within-IID behaviour and *say so*, rather
#: than shipping a one-point interval as if it were a 95% one.
MIN_CLUSTERS = 2


def probit(p: float) -> float:
    """The standard-normal quantile function, exact within the stdlib.

    Needed because a Bonferroni-adjusted family level is not a nice number:
    for six measures the two-sided level is 0.00833, and the corresponding
    z is not 1.96. This delegates to :meth:`statistics.NormalDist.inv_cdf`,
    which implements Wichura's AS241 algorithm — the same approach a
    hand-rolled rational approximation would imitate, without the
    hand-rolled constants. One function, one implementation, stdlib only.

    Raises :class:`ValueError` for a probability outside (0, 1) rather than
    returning a plausible-looking infinity for a typo.
    """
    try:
        return NormalDist().inv_cdf(p)
    except StatisticsError as error:
        raise ValueError(f"probit needs a probability strictly inside (0, 1), got {p!r}") from error


def percentile_ci(
    replicates: Sequence[float], *, level: float
) -> tuple[float, float]:
    """The two-sided percentile interval of a replicate distribution.

    ``level`` is the two-sided alpha, so the interval covers the central
    ``1 − level`` of the replicates. Percentile endpoints are taken at the
    order statistic, not interpolated: with a few thousand replicates the
    interpolation is noise, and the integer index keeps the interval
    reproducible bit for bit.
    """
    if not replicates:
        return 0.0, 0.0
    ordered = sorted(replicates)
    low = ordered[int((level / 2) * (len(ordered) - 1))]
    high = ordered[int((1 - level / 2) * (len(ordered) - 1))]
    return low, high


def _pool(values: Mapping[str, Sequence[float]], keys: Sequence[str]) -> list[float]:
    return [item for key in keys for item in values.get(key, ())]


def cluster_median_ratio_ci(
    baseline: Mapping[str, Sequence[float]],
    response: Mapping[str, Sequence[float]],
    *,
    seed: int = DEFAULT_SEED,
    samples: int = DEFAULT_SAMPLES,
    level: float = 0.05,
) -> tuple[float, float, float] | None:
    """The CI for ``median(response) / median(baseline)``, clustered by unit.

    Both mappings are keyed by the independence unit — for T3 that is the
    conversation. The bootstrap resamples **keys** with replacement, pools the
    values of the keys it drew, and takes the median ratio of the pooled
    samples. Within-conversation dependence therefore travels with its
    cluster instead of being resampled away.

    Returns ``None`` when the data cannot support it — fewer than
    ``MIN_CLUSTERS`` clusters, or one side empty over every cluster. ``None``
    is not an error: the caller decides what an interval not worth drawing
    means, and says so rather than shipping a fake one.

    A replicate whose pooled response side is empty has no median to take;
    it is discarded and another draw is attempted, up to a bounded number of
    tries. If whole-cluster exclusion is the only way to keep sampling, the
    data is telling the caller something; the bounded retry keeps it honest
    without looping forever.
    """
    keys = sorted(set(baseline) | set(response))
    if len(keys) < MIN_CLUSTERS:
        return None
    if not any(baseline.get(k) for k in keys) or not any(response.get(k) for k in keys):
        return None

    base_median = statistics.median(_pool(baseline, keys))
    response_median = statistics.median(_pool(response, keys))
    point = response_median / base_median if base_median else 0.0

    rng = random.Random(seed)
    replicates: list[float] = []
    attempts = 0
    while len(replicates) < samples and attempts < samples * 8:
        attempts += 1
        drawn = [keys[index] for index in rng.choices(range(len(keys)), k=len(keys))]
        pooled_base = _pool(baseline, drawn)
        pooled_response = _pool(response, drawn)
        if not pooled_base or not pooled_response:
            continue
        ratio = statistics.median(pooled_response) / statistics.median(pooled_base)
        replicates.append(ratio)
    if len(replicates) < DEFAULT_SAMPLES // 10:
        # So few usable replicates that the interval itself is an artefact.
        # The caller sees None and says why.
        return None
    low, high = percentile_ci(replicates, level=level)
    return point, low, high


def cluster_proportion_difference_ci(
    baseline: Mapping[str, tuple[int, int]],
    response: Mapping[str, tuple[int, int]],
    *,
    seed: int = DEFAULT_SEED,
    samples: int = DEFAULT_SAMPLES,
    level: float = 0.05,
) -> tuple[float, float, float] | None:
    """The CI for a difference of proportions, clustered by unit.

    Both maps take ``(successes, n)`` per cluster — for the reply-rate
    measure, the opportunities the subject answered and the opportunities
    they had, per conversation. The bootstrap resamples clusters, pools the
    successes and totals, and takes the difference of the pooled proportions.

    Empty pooled denominators on either side are skipped as above; the same
    ``None`` rules apply.
    """
    keys = sorted(set(baseline) | set(response))
    if len(keys) < MIN_CLUSTERS:
        return None
    if not any(baseline.get(k, (0, 0))[1] for k in keys) or not any(
        response.get(k, (0, 0))[1] for k in keys
    ):
        return None

    base_successes = sum(baseline.get(k, (0, 0))[0] for k in keys)
    base_n = sum(baseline.get(k, (0, 0))[1] for k in keys)
    response_successes = sum(response.get(k, (0, 0))[0] for k in keys)
    response_n = sum(response.get(k, (0, 0))[1] for k in keys)
    if base_n == 0 or response_n == 0:
        return None
    point = response_successes / response_n - base_successes / base_n

    rng = random.Random(seed)
    replicates: list[float] = []
    attempts = 0
    while len(replicates) < samples and attempts < samples * 8:
        attempts += 1
        drawn = [keys[index] for index in rng.choices(range(len(keys)), k=len(keys))]
        drawn_base = [baseline.get(k, (0, 0)) for k in drawn]
        drawn_response = [response.get(k, (0, 0)) for k in drawn]
        b_s = sum(pair[0] for pair in drawn_base)
        b_n = sum(pair[1] for pair in drawn_base)
        r_s = sum(pair[0] for pair in drawn_response)
        r_n = sum(pair[1] for pair in drawn_response)
        if b_n == 0 or r_n == 0:
            continue
        replicates.append(r_s / r_n - b_s / b_n)
    if len(replicates) < DEFAULT_SAMPLES // 10:
        return None
    low, high = percentile_ci(replicates, level=level)
    return point, low, high


def iid_bootstrap_median_ratio_ci(
    baseline: Sequence[float],
    response: Sequence[float],
    *,
    seed: int = DEFAULT_SEED,
    samples: int = 2_000,
    level: float = 0.05,
) -> tuple[float, float, float]:
    """The declared fallback: the within-IID bootstrap, at any level.

    Used only when there is nothing to cluster on — a one-conversation corpus
    — and only while *saying so* in the reading's notes. It resamples
    messages independently, which is exactly the assumption G-13 retired for
    clustered data; keeping it here, named, beats a caller quietly importing
    it or pretending the cluster case covers one cluster.
    """
    if not baseline or not response:
        return 0.0, 0.0, 0.0
    base_median = statistics.median(baseline)
    point = statistics.median(response) / base_median if base_median else 0.0
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(samples):
        b = statistics.median(rng.choices(baseline, k=len(baseline)))
        r = statistics.median(rng.choices(response, k=len(response)))
        ratios.append(r / b if b else 0.0)
    low, high = percentile_ci(ratios, level=level)
    return point, low, high
