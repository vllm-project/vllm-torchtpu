# SPDX-License-Identifier: Apache-2.0
"""The HOSTPHASES report as it was before q38 L6-INSTR T-1(ii) HP-REPORT
(runner/host_phases.py ``_report`` at q38-l6-next 5685da398, verbatim apart
from taking its state as arguments). The reference for the byte-identity test
and the CPU timing; not used by the runner."""

import numpy as np


def report_ref(samples, every, rank, steps, logger):
    parts = []
    total = 0.0
    for name, values in samples.items():
        v = np.asarray(values, dtype=np.float64) / 1e3
        mean = float(v.sum()) / every
        total += mean
        parts.append(
            f"{name}={np.percentile(v, 50):.0f}/{np.percentile(v, 90):.0f}/{mean:.0f}"
        )
    logger.info(
        "HOSTPHASES rank=%d steps=%d per-step host us (p50/p90/mean-per-step) "
        "total_mean=%.0f %s",
        rank,
        steps,
        total,
        " ".join(parts),
    )
    samples.clear()


def stats_ref(samples):
    """name -> (p50, p90, sum) floats, computed per phase as the old report did."""
    out = {}
    for name, values in samples.items():
        v = np.asarray(values, dtype=np.float64) / 1e3
        out[name] = (np.percentile(v, 50), np.percentile(v, 90), v.sum())
    return out


def window(rng, every=200, n_phases=45):
    """A ragged HOSTPHASES window like the served ones: most phases one sample
    per step, some twice per step or on a few steps only, some ties."""
    lengths = [every] * 34 + [2 * every] * 4 + [max(1, every // 2), 37, 3, 2, 1] + [4 * every, max(1, every - 1)]
    rng.shuffle(lengths)
    samples = {}
    for i, n in enumerate(lengths[:n_phases]):
        scale = float(rng.choice([2_000, 50_000, 300_000, 1_500_000]))
        vals = rng.lognormal(np.log(scale), 0.6, n).astype(np.int64)
        if i % 7 == 0:
            vals[: n // 3] = vals[0]  # ties
        samples[f"ph{i:02d}_{n}"] = [int(x) for x in vals]
    return samples
