# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Microbenchmark for the DCP indexer top-k, `streamindex_topk_dcp`, on GLM-5.2.

The defaults are the GLM-5.2 AgentX recipe: DCP16 with interleave 4, 256-token
SEQ_ALONG_LANE pages, MTP with 3 draft tokens, `--max-num-seqs 2`,
`--max-num-batched-tokens 4096`, a 1M-token `max_model_len` and
`TPU_STREAMIDX_CHUNK_TOKENS=256`. The kernel runs on a `--dcp-size` mesh of this
host's devices, all-gather and SparseCore merge included, so the recipe's DCP16
needs its 16-device slice; one 8-device host runs `--dcp-size 8`.

A case is one engine step's batch, padded the way the runner pads it: tokens to
the next power of two from 16, sequences to `--max-num-seqs` but at least 8.
Cases with more sequences or tokens than the scheduler admits are skipped. The
page table is `max_model_len` wide like the runner's, and the kernel sizes its
score rows from that width rather than from the context, so `--max-model-len`
changes the top-k cost. Block sizes follow `_streamidx_block_sizes` in
`layers/adapter/custom_ops/mla_attention_op.py`; ops from before the
verify-window pass use `--kv-pages-per-block 2,2,2 --queries-per-block
1,128,128`.

`step us` is the per-call slope `(t(1 + n calls) - t(1 call)) / n` of calls
chained in one `fori_loop`, each waiting for every row of the previous output,
so host dispatch cancels. `recall` is the worst overlap of sampled rows,
gathered from every rank, with a dense f32 reference top-k over the whole
sequence. Over a million positions the kernel's rounding swaps a few near-ties
at the k-th score, so a case fails only when a row is not full or selects a
position scored more than 1% below the reference's k-th score.

Each case is also traced while the same loop runs 10 calls in one program, so
devices do not wait on each other's launches, and rank 0's device time per call
is split into the scoring kernel, the local SparseCore top-k, the all-gather,
the SparseCore merge and everything else; `--xprof DIR` keeps the traces for
TensorBoard or XProf. `jax.profiler` crashes on this stack, and torch_tpu's
profiler records JAX programs too once a torch_tpu op has run in its session.
The scoring roofline uses peaks measured at startup, HBM read bandwidth from an
f32 reduction and fp8 MXU throughput from a square matmul:

  pairs    = (query token, key) pairs rank 0 scores, causally visible
  bytes    = rank 0's visible keys * bytes per key + q + weights + 4 * pairs
  flops    = 2 * heads * head_dim * pairs
  roofline = max(bytes / HBM bandwidth, flops / MXU peak)

Each visible key is read once and each pair writes one f32 score. The indexer
contracts over only 128, so compute-bound cases may top out below the
square-matmul peak.

Usage:
  python scripts/benchmark_streamindex_topk.py --dcp-size 8
  python scripts/benchmark_streamindex_topk.py --list
  python scripts/benchmark_streamindex_topk.py --dcp-size 8 --cases decode \
      --xprof /tmp/xprof --json results.jsonl
  PYTHONPATH=<older checkout>/src python scripts/benchmark_streamindex_topk.py \
      --dcp-size 8 --kv-pages-per-block 2,2,2 --queries-per-block 1,128,128
"""

import argparse
import collections
import dataclasses
import functools
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    KVLayout,
    convert_cache_to_seq_along_lane,
    cp_global_to_local,
    cp_local_length,
    cp_local_to_global,
    cp_owner_rank,
    get_kv_cache_shape,
    streamindex_topk_dcp,
)

P = jax.sharding.PartitionSpec

_NUM_HEADS, _HEAD_DIM, _TOPK = 32, 128, 2048
_K, _M = 1 << 10, 1 << 20
_MIN_TOKEN_BUCKET = 16
_MIN_NUM_SEQS = 8
_POOL_PAGES = 512
_SCORE_TOLERANCE = 0.01
# No top-k slot ever holds this, so comparing against it is a dependency on the
# output that XLA cannot fold away.
_NEVER = -12345
_TARGET_SECONDS, _MIN_EXTRA_CALLS, _MAX_EXTRA_CALLS, _REPEATS = 0.2, 4, 200, 5
_PEAK_READ_BYTES, _PEAK_MATMUL_DIM = 2 << 30, 8192
_XPROF_CALLS = 10


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    q_lens: tuple[int, ...]
    seq_lens: tuple[int, ...]  # global, including this step's tokens


def _label(tokens: int) -> str:
    if tokens and tokens % _M == 0:
        return f"{tokens // _M}M"
    return f"{tokens // _K}K" if tokens % _K == 0 else str(tokens)


def _staggered(num_seqs: int, context: int) -> tuple[int, ...]:
    """Running requests sit at different offsets within their pages."""
    return tuple(context - (i * 61) % 4096 for i in range(num_seqs))


def _cases(window: int) -> list[Case]:
    """One engine step's batch per case; `window` is 1 + the draft tokens."""
    cases = []
    for context in (32 * _K, 128 * _K, 456 * _K, _M):
        for num_seqs in (1, 2, 4, 8, 16, 32, 64):
            cases.append(
                Case(
                    f"decode c={num_seqs} @{_label(context)}",
                    (window,) * num_seqs,
                    _staggered(num_seqs, context),
                )
            )
    for num_seqs, low, high in (
        (2, 16 * _K, _M),
        (8, 16 * _K, _M),
        (64, 8 * _K, 456 * _K),
    ):
        cases.append(
            Case(
                f"decode c={num_seqs} spread {_label(low)}-{_label(high)}",
                (window,) * num_seqs,
                tuple(int(x) for x in np.geomspace(low, high, num_seqs)),
            )
        )
    # Requests with fewer drafts than the rest take the 1-token decode pass.
    for num_seqs in (1, 4):
        cases.append(
            Case(
                f"decode {num_seqs}x1+{num_seqs}x{window} @128K",
                (1,) * num_seqs + (window,) * num_seqs,
                _staggered(2 * num_seqs, 128 * _K),
            )
        )
    # A prefill chunk fills the rest of the token budget next to running
    # requests: a new prompt, or a new turn on top of a cached long context.
    for num_seqs, context, budget, prefix in (
        (1, 128 * _K, 1024, 0),
        (1, 456 * _K, 4096, 456 * _K),
        (7, 128 * _K, 1024, 0),
        (7, 128 * _K, 4096, 128 * _K),
        (15, 456 * _K, 8192, 456 * _K),
        (63, 128 * _K, 8192, 128 * _K),
    ):
        chunk = budget - num_seqs * window
        running = f"{num_seqs}x{window} @{_label(context)}"
        cases.append(
            Case(
                f"mixed {running} + {chunk} @{_label(prefix)}",
                (window,) * num_seqs + (chunk,),
                _staggered(num_seqs, context) + (prefix + chunk,),
            )
        )
    for tokens in (1024, 4096, 8192):
        for prefix in (0, 128 * _K, 456 * _K):
            cases.append(
                Case(
                    f"prefill {tokens} @{_label(prefix)}", (tokens,), (prefix + tokens,)
                )
            )
    cases.append(Case("prefill 4x1024 @32K", (1024,) * 4, (33 * _K,) * 4))
    return cases


def _op_block_sizes(
    page_size: int, num_spec: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """What `_streamidx_block_sizes` in `mla_attention_op.py` hands the kernel."""
    if num_spec == 0:
        return (2, 2, 2), (1, 128, 128)
    return (2, max(1, 16384 // page_size), 2), (1, num_spec + 1, 128)


def _rank0_len(global_len: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Positions below `global_len` that rank 0, the rank holding the most, owns."""
    return np.asarray(
        cp_local_length(
            np.asarray(global_len, np.int32), 0, args.dcp_size, args.interleave
        )
    )


def _num_pages(case: Case, args: argparse.Namespace) -> np.ndarray:
    return -(-_rank0_len(np.asarray(case.seq_lens), args) // args.page_size)


def _skip_reason(case: Case, args: argparse.Namespace) -> str | None:
    if len(case.q_lens) > args.max_num_seqs:
        return f"{len(case.q_lens)} sequences > --max-num-seqs {args.max_num_seqs}"
    if sum(case.q_lens) > args.max_num_batched_tokens:
        return (
            f"{sum(case.q_lens)} tokens > --max-num-batched-tokens"
            f" {args.max_num_batched_tokens}"
        )
    if max(case.seq_lens) > args.max_model_len:
        return f"context {max(case.seq_lens)} > --max-model-len {args.max_model_len}"
    return None


def _key_pool(page_size: int) -> tuple[jax.Array, jax.Array]:
    """fp8 keys and UE8M0 exponents of `_POOL_PAGES` distinct pages."""
    key_values, key_exponents = jax.random.split(jax.random.key(0))
    keys = jax.random.normal(key_values, (_POOL_PAGES, page_size, _HEAD_DIM))
    exponents = jax.random.randint(key_exponents, (_POOL_PAGES, page_size), 124, 131)
    return keys.astype(jnp.float8_e4m3fn), exponents.astype(jnp.uint8)


def _cache(
    keys: jax.Array,
    exponents: jax.Array,
    num_pages: int,
    kv_layout: KVLayout,
    mesh: jax.sharding.Mesh,
) -> jax.Array:
    """`num_pages` pages per rank, sharded by rank; page `g` holds pool page `g % P`."""
    pool_pages, page_size, _ = keys.shape
    width = get_kv_cache_shape(1, page_size, _HEAD_DIM)[-1]
    record = jnp.concatenate(
        [lax.bitcast_convert_type(keys, jnp.uint8), exponents[..., None]], axis=-1
    )
    record = jnp.pad(record, ((0, 0), (0, 0), (0, width - record.shape[-1])))
    pool = record.reshape(pool_pages, page_size // 4, 4, width)
    if kv_layout == KVLayout.SEQ_ALONG_LANE:
        pool = convert_cache_to_seq_along_lane(pool, _HEAD_DIM)
    pages = jnp.arange(mesh.size * num_pages) % pool_pages
    return jax.device_put(
        jnp.take(pool, pages, axis=0),
        jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME)),
    )


@dataclasses.dataclass(frozen=True)
class Batch:
    # q, weights, seq_lens, page_indices, cu_q_lens, distribution; replicated
    kernel_args: tuple[jax.Array, ...]
    table: np.ndarray  # i32[seq_slots, table_width], page ids every rank shares
    row_seq: np.ndarray  # i32[num_tokens]
    row_pos: np.ndarray  # i32[num_tokens], global position of each query token
    num_tokens: int
    bucket: int
    kv_keys: int  # keys rank 0 holds for these sequences
    pairs: int  # (query token, key) pairs rank 0 scores


def _batch(
    case: Case,
    args: argparse.Namespace,
    num_pages: int,
    seed: int,
    prefill_pass: bool,
    mesh: jax.sharding.Mesh,
) -> Batch:
    rng = np.random.default_rng(seed)
    q_lens = np.asarray(case.q_lens, np.int32)
    seq_lens = np.asarray(case.seq_lens, np.int32)
    num_seqs, num_tokens = len(q_lens), int(q_lens.sum())
    bucket = max(_MIN_TOKEN_BUCKET, 1 << (num_tokens - 1).bit_length())

    # A page id names the same slot on every rank, which holds that rank's
    # positions of a `page_size * dcp_size`-position virtual page.
    seq_slots = max(args.max_num_seqs, _MIN_NUM_SEQS)
    table_width = -(-args.max_model_len // (args.page_size * args.dcp_size))
    table = np.zeros((seq_slots, table_width), np.int32)
    free = iter(rng.permutation(num_pages))
    for s, n in enumerate(_num_pages(case, args)):
        table[s, :n] = [next(free) for _ in range(n)]

    padded_seq_lens = np.zeros(seq_slots, np.int32)
    padded_seq_lens[:num_seqs] = seq_lens
    cu_q_lens = np.full(seq_slots + 1, num_tokens, np.int32)
    cu_q_lens[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    # The runner puts the 1-token decodes first. The op offers every later
    # sequence to the prefill pass; without one, `(d, d, n)` is the same thing
    # and is what kernels from before that pass need.
    num_decodes = next((s for s, q in enumerate(q_lens) if q != 1), num_seqs)
    distribution = [num_decodes, num_seqs if prefill_pass else num_decodes, num_seqs]

    row_seq = np.repeat(np.arange(num_seqs, dtype=np.int32), q_lens)
    row_pos = (
        seq_lens[row_seq] - q_lens[row_seq] + np.arange(num_tokens) - cu_q_lens[row_seq]
    )

    key_q, key_w = jax.random.split(jax.random.key(seed))
    q = jax.random.normal(key_q, (bucket, _NUM_HEADS, _HEAD_DIM))
    # Positive weights keep the reference free of ties at the k-th score.
    weights = jax.random.uniform(key_w, (bucket, _NUM_HEADS), minval=0.5, maxval=1.5)
    replicated = jax.sharding.NamedSharding(mesh, P())
    return Batch(
        kernel_args=jax.device_put(
            (
                q.astype(jnp.float8_e4m3fn),
                weights.astype(jnp.bfloat16),
                jnp.asarray(padded_seq_lens),
                jnp.asarray(table.reshape(-1)),
                jnp.asarray(cu_q_lens),
                jnp.asarray(distribution, jnp.int32),
            ),
            replicated,
        ),
        table=table,
        row_seq=row_seq,
        row_pos=row_pos,
        num_tokens=num_tokens,
        bucket=bucket,
        kv_keys=int(_rank0_len(seq_lens, args).sum()),
        pairs=int(_rank0_len(row_pos + 1, args).sum()),
    )


@functools.partial(jax.jit, static_argnames=("num_pages", "dcp_size", "interleave"))
def _reference_scores(
    q: jax.Array,  # f8[rows, heads, head_dim]
    weights: jax.Array,  # bf16[rows, heads]
    pages: jax.Array,  # i32[rows, table_width]
    visible: jax.Array,  # i32[rows], global positions each row sees
    keys: jax.Array,  # f8[pool_pages, page_size, head_dim]
    exponents: jax.Array,  # u8[pool_pages, page_size]
    *,
    num_pages: int,
    dcp_size: int,
    interleave: int,
) -> jax.Array:  # f32[rows, positions], -inf where a row sees nothing
    pool_pages, page_size, _ = keys.shape

    def row(args):
        q, weights, pages, visible = args
        position = jnp.arange(pages.shape[0] * page_size * dcp_size)
        local = cp_global_to_local(position, dcp_size, interleave)
        rank = cp_owner_rank(position, dcp_size, interleave)
        pool_page = (rank * num_pages + pages[local // page_size]) % pool_pages
        slot = local % page_size
        scale = jnp.exp2(exponents[pool_page, slot].astype(jnp.float32) - 127)
        row_keys = keys[pool_page, slot].astype(jnp.float32) * scale[:, None]
        logits = jnp.dot(
            row_keys, q.astype(jnp.float32).T, precision=lax.Precision.HIGHEST
        )
        scores = jnp.sum(jnp.maximum(logits, 0) * weights.astype(jnp.float32), axis=1)
        return jnp.where(position < visible, scores, -jnp.inf)

    return lax.map(row, (q, weights, pages, visible))


def _check(
    out: jax.Array,
    batch: Batch,
    keys: jax.Array,
    exponents: jax.Array,
    num_pages: int,
    args: argparse.Namespace,
) -> tuple[float, str | None]:
    """Worst recall of sampled real rows, from every rank, and what is wrong."""
    rows = np.unique(
        np.linspace(0, batch.num_tokens - 1, args.check_rows).astype(np.int32)
    )
    q, weights = batch.kernel_args[:2]
    scores = np.asarray(
        _reference_scores(
            np.asarray(q[rows]),
            np.asarray(weights[rows]),
            batch.table[batch.row_seq[rows]],
            batch.row_pos[rows] + 1,
            keys,
            exponents,
            num_pages=num_pages,
            dcp_size=args.dcp_size,
            interleave=args.interleave,
        )
    )
    # Rank r's rows of `out`, left-packed local indices, map back to positions.
    got = np.asarray(out.reshape(args.dcp_size, batch.bucket, _TOPK)[:, rows])
    worst, wrong = 1.0, None
    for i, row_scores in enumerate(scores):
        selected = np.concatenate(
            [
                np.asarray(
                    cp_local_to_global(
                        got[rank, i][got[rank, i] >= 0],
                        rank,
                        args.dcp_size,
                        args.interleave,
                    )
                )
                for rank in range(args.dcp_size)
            ]
        )
        visible = np.flatnonzero(row_scores > -np.inf)
        want = visible
        if len(visible) > _TOPK:
            want = np.argpartition(-row_scores, _TOPK - 1)[:_TOPK]
        worst = min(worst, len(np.intersect1d(selected, want)) / max(len(want), 1))
        kth = row_scores[want].min() if len(want) else 0.0
        if len(np.unique(selected)) != len(want):
            wrong = f"row {rows[i]} holds {len(np.unique(selected))} of {len(want)}"
        elif (row_scores[selected] < kth * (1 - _SCORE_TOLERANCE)).any():
            wrong = f"row {rows[i]} selects a position well below the k-th score"
    return worst, wrong


def _never(out: jax.Array, seq_lens: jax.Array) -> jax.Array:
    """`seq_lens`, after every row of this rank's `out`.

    With `chunk_tokens` each chunk of rows is its own scoring and top-k, and XLA
    drops the chunks whose rows nothing reads.
    """
    return seq_lens + (jnp.min(out[:, 0]) == _NEVER).astype(jnp.int32)


@functools.partial(jax.jit, static_argnames=("kernel_kwargs",))
def _chained(reps, kernel_args, cache, *, kernel_kwargs):
    kwargs = dict(kernel_kwargs)
    q, weights, seq_lens, page_indices, cu_q_lens, distribution = kernel_args

    def body(_, seq_lens):
        # Every input goes through the carry, or XLA hoists the per-call work
        # that only depends on them (q and metadata prep) out of the loop.
        args = lax.optimization_barrier(
            (q, weights, cache, seq_lens, page_indices, cu_q_lens, distribution)
        )
        out = streamindex_topk_dcp(*args, **kwargs)
        # Each rank waits for its own output, with no collective; the result is
        # the same `seq_lens` on every rank.
        return jax.shard_map(
            _never,
            mesh=kwargs["mesh"],
            in_specs=(P(DCP_AXIS_NAME), P()),
            out_specs=P(),
            check_vma=False,
        )(out, args[3])

    return lax.fori_loop(0, reps, body, seq_lens)


@jax.jit
def _read_loop(reps, x):
    def body(_, total):
        x_, total = lax.optimization_barrier((x, total))
        return total + jnp.sum(x_)

    return lax.fori_loop(0, reps, body, jnp.float32(0))


@jax.jit
def _matmul_loop(reps, lhs, rhs):
    def body(_, total):
        lhs_, rhs_, total = lax.optimization_barrier((lhs, rhs, total))
        return total + jnp.sum(jnp.dot(lhs_, rhs_, preferred_element_type=jnp.float32))

    return lax.fori_loop(0, reps, body, jnp.float32(0))


def _per_call_us(loop, *args) -> float:
    """Device time of one call: the slope between 1 and `1 + n` chained calls."""

    def seconds(reps):
        start = time.perf_counter()
        jax.block_until_ready(loop(reps, *args))
        return time.perf_counter() - start

    seconds(1)
    estimate = max((seconds(5) - seconds(1)) / 4, 1e-6)
    extra = int(np.clip(_TARGET_SECONDS / estimate, _MIN_EXTRA_CALLS, _MAX_EXTRA_CALLS))
    short = np.median([seconds(1) for _ in range(_REPEATS)])
    long = np.median([seconds(1 + extra) for _ in range(_REPEATS)])
    return float((long - short) / extra * 1e6)


@dataclasses.dataclass(frozen=True)
class Peaks:
    hbm_bytes_per_s: float
    mxu_flops_per_s: float

    def roofline_us(self, num_bytes: int, flops: int) -> tuple[float, str]:
        """Time at the peaks, and whether HBM or the MXU bounds it."""
        memory_s = num_bytes / self.hbm_bytes_per_s
        compute_s = flops / self.mxu_flops_per_s
        return max(memory_s, compute_s) * 1e6, "HBM" if memory_s >= compute_s else "MXU"


def _measure_peaks() -> Peaks:
    x = jnp.ones((_PEAK_READ_BYTES // 4 // 1024, 1024), jnp.float32)
    read_us = _per_call_us(_read_loop, x)
    del x
    operand = jnp.ones((_PEAK_MATMUL_DIM,) * 2, jnp.float8_e4m3fn)
    matmul_us = _per_call_us(_matmul_loop, operand, operand)
    return Peaks(
        _PEAK_READ_BYTES / read_us * 1e6, 2 * _PEAK_MATMUL_DIM**3 / matmul_us * 1e6
    )


def _xprof(run, trace_dir: str, merge_width: int) -> dict[str, float]:
    """Rank 0's device us per call, by part, tracing `run`'s `_chained` loop."""
    import torch
    from jax.profiler import ProfileData
    from torch_tpu._internal import sync
    from torch_tpu._internal.profiler import TpuProfilerConfig

    jax.block_until_ready(run())
    # The device tracer only records once torch_tpu is up before the session
    # starts and has launched something inside it.
    one = torch.ones(1, device="tpu")
    sync.synchronize([one], wait=True)
    config = TpuProfilerConfig(
        run_dir=trace_dir,
        host_tracer_level=0,
        device_tracer_level=1,
        python_tracer_level=0,
        experimental_options={
            "tpu_trace_mode": "TRACE_COMPUTE",
            "tpu_num_sparse_cores_to_trace": 4,
            "tpu_num_sparse_core_tiles_to_trace": 16,
        },
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.PrivateUse1],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        experimental_config=config,
    ):
        sync.synchronize([one + 1], wait=True)
        jax.block_until_ready(run())

    path = max(
        glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True),
        key=os.path.getmtime,
    )
    us = collections.defaultdict(float)
    for plane in ProfileData.from_file(path).planes:
        if plane.name != "/device:TPU:0":
            continue
        for line in plane.lines:
            for event in line.events:
                name, event_us = event.name, event.duration_ns / 1e3 / _XPROF_CALLS
                if line.name == "XLA Modules" and "_chained" in name:
                    us["program"] += event_us
                elif line.name != "XLA Ops":
                    continue
                # The scoring `pallas_call`s, then the SparseCore top-k, which
                # the merge runs too, over `dcp_size * k` candidates per row.
                elif "StreamIdxTC" in name:
                    us["scoring"] += event_us
                elif "sc_topk" in name:
                    us["merge" if f"_n{merge_width}_" in name else "topk"] += event_us
                elif "all-gather" in name:
                    us["all_gather"] += event_us
    us["other"] = us["program"] - sum(
        us[part] for part in ("scoring", "topk", "all_gather", "merge")
    )
    return dict(us)


def _triple(text: str) -> tuple[int, int, int]:
    values = [int(v) for v in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"expected decode,prefill,mixed; got {text!r}")
    return values[0], values[1], values[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", default="", help="comma-separated name substrings")
    parser.add_argument("--list", action="store_true", help="list the cases and exit")
    parser.add_argument("--layout", choices=("sal", "has"), default="sal")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--dcp-size", type=int, default=16)
    parser.add_argument("--interleave", type=int, default=4)
    parser.add_argument("--num-spec", type=int, default=3, help="MTP draft tokens")
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=_M)
    parser.add_argument(
        "--kv-pages-per-block", type=_triple, help="decode,prefill,mixed"
    )
    parser.add_argument(
        "--queries-per-block", type=_triple, help="decode,prefill,mixed"
    )
    parser.add_argument("--decode-req-batch-size", type=int, default=4)
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=256,
        help="TPU_STREAMIDX_CHUNK_TOKENS; 0 off",
    )
    parser.add_argument("--check-rows", type=int, default=4, help="0 skips the check")
    parser.add_argument("--json", help="also write one JSON line per case here")
    parser.add_argument("--xprof", help="keep each case's trace under this directory")
    args = parser.parse_args()
    if args.dcp_size < 2:
        parser.error("--dcp-size must be at least 2; this benchmarks the DCP kernel")
    return args


def _print_row(cells) -> None:
    print("| " + " | ".join(str(c) for c in cells) + " |", flush=True)


def main() -> None:
    args = _parse_args()
    kv_layout = (
        KVLayout.SEQ_ALONG_LANE if args.layout == "sal" else KVLayout.HEAD_ALONG_SUBLANE
    )
    kv_blocks, q_blocks = _op_block_sizes(args.page_size, args.num_spec)
    kv_blocks = args.kv_pages_per_block or kv_blocks
    q_blocks = args.queries_per_block or q_blocks
    prefill_pass = (kv_blocks[1], q_blocks[1]) != (kv_blocks[2], q_blocks[2])

    terms = [t for t in args.cases.split(",") if t]
    cases = [
        c
        for c in _cases(args.num_spec + 1)
        if not terms or any(t in c.name for t in terms)
    ]
    config = (
        f"layout={kv_layout.name} page={args.page_size} dcp={args.dcp_size}"
        f" interleave={args.interleave} drafts={args.num_spec}"
        f" max_num_seqs={args.max_num_seqs}"
        f" max_num_batched_tokens={args.max_num_batched_tokens}"
        f" max_model_len={args.max_model_len}"
        f" kv pages/block={kv_blocks} queries/block={q_blocks}"
        f" decode batch={args.decode_req_batch_size} chunk_tokens={args.chunk_tokens}"
    )
    if args.list:
        print(config)
        for case in cases:
            reason = _skip_reason(case, args)
            print(case.name + (f"  (skipped: {reason})" if reason else ""))
        return
    if jax.device_count() < args.dcp_size:
        sys.exit(
            f"--dcp-size {args.dcp_size} needs {args.dcp_size} devices and this host"
            f" has {jax.device_count()}; run on a slice that big or pass"
            f" --dcp-size {jax.device_count()}"
        )
    for case in cases:
        if reason := _skip_reason(case, args):
            print(f"skipping {case.name!r}: {reason}")
    cases = [c for c in cases if _skip_reason(c, args) is None]
    if not cases:
        sys.exit(f"no case matches --cases {args.cases!r}; see --list")

    mesh = jax.sharding.Mesh(np.array(jax.devices()[: args.dcp_size]), (DCP_AXIS_NAME,))
    kernel_kwargs = dict(
        mesh=mesh,
        k=_TOPK,
        compression_ratio=1,
        dcp_size=args.dcp_size,
        interleave_size=args.interleave,
        num_kv_pages_per_block=kv_blocks,
        num_queries_per_block=q_blocks,
        decode_req_batch_size=args.decode_req_batch_size,
        kv_layout=kv_layout,
        chunk_tokens=args.chunk_tokens or None,
    )
    frozen_kwargs = tuple(sorted(kernel_kwargs.items()))
    peaks = _measure_peaks()
    keys, exponents = _key_pool(args.page_size)
    num_pages = max(int(_num_pages(c, args).sum()) for c in cases)
    cache = jax.block_until_ready(_cache(keys, exponents, num_pages, kv_layout, mesh))
    bytes_per_key = int(np.prod(cache.shape[1:])) // args.page_size
    trace_root = args.xprof or tempfile.mkdtemp(prefix="streamindex_xprof_")

    print(
        f"GLM-5.2 indexer heads={_NUM_HEADS} head_dim={_HEAD_DIM} k={_TOPK}"
        f" on {jax.devices()[0].device_kind}, {num_pages} cache pages per rank"
    )
    print(config)
    print(
        f"measured peaks: HBM read {peaks.hbm_bytes_per_s / 1e12:.2f} TB/s,"
        f" fp8 MXU {peaks.mxu_flops_per_s / 1e12:.0f} TFLOP/s; per-part times"
        f" are rank 0's, from traces"
        + (f" kept under {trace_root}\n" if args.xprof else "\n")
    )
    header = ["case", "tokens", "step us", "program us", "scoring us"]
    header += ["scoring roofline us", "bound", "scoring eff", "top-k us"]
    header += ["all-gather us", "merge us", "other us", "recall"]
    _print_row(header)
    _print_row(["---"] + ["--:"] * (len(header) - 1))

    results, failed = [], []
    for seed, case in enumerate(cases):
        batch = _batch(case, args, num_pages, seed, prefill_pass, mesh)
        q, weights, *metadata = batch.kernel_args
        call = functools.partial(
            streamindex_topk_dcp, q, weights, cache, *metadata, **kernel_kwargs
        )
        out = call()
        recall, wrong = (
            _check(out, batch, keys, exponents, num_pages, args)
            if args.check_rows
            else (None, None)
        )
        if wrong:
            failed.append(f"{case.name}: {wrong}")
        chained = functools.partial(_chained, kernel_kwargs=frozen_kwargs)
        step_us = _per_call_us(chained, batch.kernel_args, cache)
        trace_dir = os.path.join(trace_root, re.sub(r"[^0-9A-Za-z]+", "_", case.name))
        parts = _xprof(
            functools.partial(chained, _XPROF_CALLS, batch.kernel_args, cache),
            trace_dir,
            args.dcp_size * _TOPK,
        )
        if not args.xprof:
            # A case's trace runs to gigabytes; keeping every one fills a disk.
            shutil.rmtree(trace_dir, ignore_errors=True)
        scoring_bytes = (
            batch.kv_keys * bytes_per_key + q.nbytes + weights.nbytes + 4 * batch.pairs
        )
        flops = 2 * _NUM_HEADS * _HEAD_DIM * batch.pairs
        roofline_us, bound = peaks.roofline_us(scoring_bytes, flops)
        if parts.get("scoring", 0.0) < roofline_us:
            failed.append(f"{case.name}: scoring kernel faster than its roofline")
        result = dict(
            case=case.name,
            tokens=batch.num_tokens,
            bucket=batch.bucket,
            seqs=len(case.q_lens),
            step_us=round(step_us, 1),
            **{f"{part}_us": round(us, 1) for part, us in parts.items()},
            scoring_roofline_us=round(roofline_us, 1),
            bound=bound,
            scoring_bytes=scoring_bytes,
            flops=flops,
            recall=recall,
        )
        results.append(result)
        _print_row(
            [
                case.name,
                f"{batch.num_tokens}/{batch.bucket}",
                result["step_us"],
                result.get("program_us", "-"),
                result.get("scoring_us", "-"),
                result["scoring_roofline_us"],
                bound,
                f"{roofline_us / parts['scoring']:.0%}" if "scoring" in parts else "-",
                result.get("topk_us", "-"),
                result.get("all_gather_us", "-"),
                result.get("merge_us", "-"),
                result["other_us"],
                "-" if recall is None else f"{recall:.4f}",
            ]
        )
        del batch, q, weights, metadata, call, out

    if not args.xprof:
        shutil.rmtree(trace_root, ignore_errors=True)
    if args.json:
        config_json = dict(
            vars(args),
            hbm_tb_s=round(peaks.hbm_bytes_per_s / 1e12, 3),
            mxu_tflop_s=round(peaks.mxu_flops_per_s / 1e12, 1),
        )
        with open(args.json, "w") as f:
            for result in results:
                f.write(json.dumps(dict(config=config_json, **result)) + "\n")
    if failed:
        sys.exit("check failed:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
