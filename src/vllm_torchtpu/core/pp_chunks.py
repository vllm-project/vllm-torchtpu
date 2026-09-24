# SPDX-License-Identifier: Apache-2.0
"""Sizing prefill chunks so every pipeline step costs about the same time.

The stages of a pipeline advance in lockstep (``distributed/pp_wave.py``), so
a wave lasts as long as the slowest step in flight. Steps of equal token
count are not steps of equal time: a chunk attends to every token before it,
so a chunk late in a long prompt costs more than a first chunk. The
scheduler therefore gives each step a time budget and sizes prefill chunks
against it instead of filling a token budget.

Step time on one stage is modelled as

    t = linear(padded tokens) + attn * sum_i (chunk_i * prefix_i + chunk_i^2 / 2)

where ``attn`` (ms per token of chunk per token of context) is the median
over buckets of the slope between a bucket's two prefix points, and
``linear`` is the prefix-free time of each bucket with its own causal
attention taken out. Each
stage times its own forwards at startup (``TPUModelRunner.profile_pipeline_chunks``);
the slowest stage's ladder and slope form the model. The target is the time
of a full prefix-free step plus a slack, so a step that fills the bucket with
first chunks is never shortened.

Chunks are cut in granules: an eighth of a step, rounded up to whole KV
blocks when a mamba state cache needs block-aligned chunk ends. Buckets at
every granule multiple are compiled so a shortened step pads to its own size.

The batched attention kernel schedules one (query block, KV block) pair per
entry of an SMEM table of fixed capacity, so a step also holds at most that
many pairs: a chunk of ``c`` tokens behind ``p`` tokens of context needs
about ``c/bq * (p + c/2)/bkv`` of them, which is what bounds a chunk deep in
a very long prompt.
"""
import math
from dataclasses import dataclass
from statistics import median

import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)

# Buckets below a quarter of the largest one are timed without a prefix only.
_PREFIX_PROFILE_MIN_DIVISOR = 4


def chunk_granularity(block_size: int, max_tokens: int,
                      block_aligned: bool) -> int:
    """Token multiple that prefill chunks are cut at: an eighth of a step,
    rounded up to whole KV blocks when chunk ends must stay block aligned and
    a step spans at least one block. A block larger than a step leaves the
    scheduler's own split to stop a chunk at the block boundary."""
    granule = max(max_tokens // 8, 1)
    if block_aligned and block_size <= max_tokens:
        granule = -(-granule // block_size) * block_size
    return granule


def chunk_buckets(max_tokens: int, granularity: int) -> list[int]:
    """Token buckets at every granule multiple below ``max_tokens``."""
    return list(range(granularity, max_tokens, granularity))


def chunk_pairs(schedule: tuple[int, int, int] | None, chunk: int,
                prefix: int) -> int:
    """(query block, KV block) pairs the attention kernel schedules for a
    chunk behind a prefix with ``schedule`` = (capacity, query tile, KV
    tile): every query block attends to the KV blocks up to its own last
    token. 0 without a schedule."""
    if schedule is None or chunk <= 0:
        return 0
    _, bq, bkv = schedule
    total = 0
    for start in range(0, chunk, bq):
        total += -(-(prefix + min(start + bq, chunk)) // bkv)
    return total


def schedule_pairs(q_lens, kv_lens, bq: int, bkv: int) -> int:
    """(query block, KV block) pairs the attention kernel schedules for
    sequences with ``q_lens`` new tokens at the end of ``kv_lens`` tokens of
    context, with query tile ``bq`` and KV tile ``bkv``: every query block
    attends to the KV blocks up to its own last token. The multi-sequence
    form of `chunk_pairs`."""
    q = np.asarray(q_lens, dtype=np.int64)
    kv = np.asarray(kv_lens, dtype=np.int64)
    keep = q > 0
    q, kv = q[keep], kv[keep]
    if q.size == 0:
        return 0
    blocks = -(-q // bq)
    i = np.arange(int(blocks.max()))
    q_end = np.minimum((i[None, :] + 1) * bq, q[:, None])
    k_end = -(-(kv[:, None] - q[:, None] + q_end) // bkv)
    return int(np.where(i[None, :] < blocks[:, None], k_end, 0).sum())


def profile_points(
        buckets: list[int],
        max_tokens: int,
        max_model_len: int,
        schedule: tuple[int, int, int] | None = None) -> list[tuple[int, int]]:
    """(tokens, prefix) pairs to time as one request: every bucket without a
    prefix, and the larger buckets behind one and two steps of prefix,
    staying a step short of the model length. A prefixed point whose
    attention pairs exceed the ``schedule`` capacity is left out, since the
    kernel cannot run it."""
    points = []
    for bucket in buckets:
        points.append((bucket, 0))
        if bucket * _PREFIX_PROFILE_MIN_DIVISOR < max_tokens:
            continue
        for prefix in (max_tokens, 2 * max_tokens):
            prefix = min(prefix, max_model_len - max_tokens - bucket)
            if prefix <= 0 or (bucket, prefix) in points:
                continue
            if (schedule is not None
                    and chunk_pairs(schedule, bucket, prefix) > schedule[0]):
                continue
            points.append((bucket, prefix))
    return points


@dataclass
class StepCostModel:
    """Per-stage step time in ms as a function of the step's shape."""
    buckets: list[int]
    linear_ms: dict[int, float]
    attn_ms_per_token_pair: float
    target_ms: float
    granularity: int
    # (pairs, bq, bkv) the attention kernel schedules per step, or None.
    schedule: tuple[int, int, int] | None = None

    @classmethod
    def fit(cls,
            stage_samples: list[list[tuple[int, int, float]]],
            slack: float,
            granularity: int,
            schedule: tuple[int, int, int] | None = None) -> "StepCostModel":
        """``stage_samples`` holds one list of (tokens, prefix, ms) per
        stage. Every stage yields a ladder and a slope; the model keeps the
        slowest ladder entry per bucket and the largest slope."""
        if slack < 0:
            raise ValueError(f"the chunk slack must not be negative: {slack}")
        if not stage_samples:
            raise ValueError("no stage reported step timings")
        ladders: list[dict[int, float]] = []
        slopes: list[float] = []
        for samples in stage_samples:
            timed: dict[tuple[int, int], float] = {}
            for tokens, prefix, ms in samples:
                key = (int(tokens), int(prefix))
                timed[key] = min(timed.get(key, math.inf), float(ms))
            base = {t: ms for (t, p), ms in timed.items() if p == 0}
            # The slope of each bucket comes from its two prefix points;
            # a bucket with one prefix point uses its prefix-free point.
            stage_slopes = []
            for t in base:
                prefixed = sorted((p, ms) for (tt, p), ms in timed.items()
                                  if tt == t and p > 0)
                if len(prefixed) >= 2:
                    (p1, ms1), (p2, ms2) = prefixed[0], prefixed[-1]
                    stage_slopes.append(max((ms2 - ms1) / (t * (p2 - p1)),
                                            0.0))
                elif prefixed:
                    p1, ms1 = prefixed[0]
                    stage_slopes.append(max((ms1 - base[t]) / (t * p1), 0.0))
            attn = median(stage_slopes) if stage_slopes else 0.0
            ladders.append({
                t: max(ms - attn * t * t / 2, 0.0)
                for t, ms in base.items()
            })
            slopes.append(attn)
        buckets = sorted(ladders[0])
        if any(sorted(ladder) != buckets for ladder in ladders):
            raise ValueError("stages timed different token buckets: " +
                             ", ".join(
                                 str(sorted(ladder)) for ladder in ladders))
        linear = {b: max(ladder[b] for ladder in ladders) for b in buckets}
        attn = max(slopes)
        top = buckets[-1]
        full_step = linear[top] + attn * top * top / 2
        return cls(buckets, linear, attn, full_step * (1.0 + slack),
                   int(granularity), schedule)

    def pairs(self, chunk: int, prefix: int) -> int:
        """(query block, KV block) pairs the attention kernel schedules for
        a chunk behind a prefix: every query block attends to the KV blocks
        up to its own last token. 0 without a schedule capacity."""
        return chunk_pairs(self.schedule, chunk, prefix)

    def step_ms(self, padded_tokens: int, token_pairs: float,
                token_squares: float) -> float:
        return (self.linear_ms[padded_tokens] + self.attn_ms_per_token_pair *
                (token_pairs + token_squares / 2))

    def chunk(self,
              prefix: int,
              remaining: int,
              step_tokens: int,
              step_pairs: float,
              step_squares: float,
              step_limit: int | None = None,
              step_schedule: int = 0) -> int:
        """Largest chunk of a request with ``prefix`` computed tokens and
        ``remaining`` tokens to prefill that keeps a step already holding
        ``step_tokens`` tokens (``step_pairs`` chunk*prefix products and
        ``step_squares`` squared chunk lengths) within the target. A chunk
        that does not finish the request leaves the step total on a granule
        multiple and is at least a quarter of a step when the step has the
        room, since a run of tiny steps costs more than the balance it buys;
        ``step_limit`` keeps the step within a smaller bucket, and with a
        schedule capacity the chunk's attention pairs must fit beside the
        ``step_schedule`` pairs already in the step. 0 when nothing
        fits."""
        best = 0
        attn = self.attn_ms_per_token_pair
        floor = min(remaining, self.buckets[-1] // 4)
        for bucket in self.buckets:
            if step_limit is not None and bucket > step_limit:
                continue
            room = bucket - step_tokens
            if room <= 0:
                continue
            spare = self.target_ms - self.step_ms(bucket, step_pairs,
                                                  step_squares)
            if spare < 0:
                continue
            fit = min(remaining, room)
            if attn > 0:
                # attn * (x * prefix + x^2 / 2) <= spare
                fit = min(
                    fit,
                    int(-prefix +
                        math.sqrt(prefix * prefix + 2 * spare / attn)))
            if fit < remaining:
                fit = min(max(fit, floor), room)
            if fit < remaining:
                total = step_tokens + fit
                fit = total - total % self.granularity - step_tokens
            if self.schedule is not None:
                capacity = self.schedule[0] - step_schedule
                while fit > 0 and self.pairs(fit, prefix) > capacity:
                    # Back off a granule, keeping the step total aligned.
                    total = step_tokens + fit - 1
                    fit = total - total % self.granularity - step_tokens
            best = max(best, fit)
        return best

    def describe(self) -> str:
        ladder = ", ".join(f"{b}: {self.linear_ms[b]:.1f}"
                           for b in self.buckets)
        text = (f"linear ms per bucket {{{ladder}}}, attention "
                f"{self.attn_ms_per_token_pair:.3e} ms per token pair, "
                f"target {self.target_ms:.1f} ms, granularity "
                f"{self.granularity} tokens")
        if self.schedule is not None:
            pairs, bq, bkv = self.schedule
            text += f", at most {pairs} attention pairs ({bq}x{bkv}) per step"
        return text
