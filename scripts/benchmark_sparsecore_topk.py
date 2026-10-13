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
"""Microbenchmark for the SparseCore top-k of one DCP rank on GLM-5.2 AgentX.

Per indexer call, `streamindex_topk_dcp` runs one local `sparsecore_topk` per
`TPU_STREAMIDX_CHUNK_TOKENS` chunk of the token bucket, over score rows as wide
as the page table, then `sparsecore_topk_dcp_merge` over the `dcp_size * k`
candidates every rank gathered. This replays exactly those calls on one chip,
with rank 0's row lengths from the kernel's own `_effective_row_lengths`.

The defaults are the AgentX recipe: DCP16 with interleave 4, 256-token pages,
MTP with 3 draft tokens, `--max-num-seqs 2`, `--max-num-batched-tokens 4096`,
a 1M-token `max_model_len`, 256-token chunks and k = 2048. The page table is
`max_model_len` wide like the runner's, so the local score rows are
`max_model_len / dcp_size` = 64K wide whatever the context; how much of that
width a kernel reads is what separates versions. A case is one engine step's
batch, tokens padded to the next power of two from 16. Scores are standard
normal, which the selection cost depends on only weakly.

`--kernel NAME=SRC` loads `sparsecore_topk.py` from SRC, a file or a git ref
such as `origin/main`, so versions run side by side on the same inputs; without
it the checkout's kernel runs alone. Every version's indices must equal the
first version's as sets per row (`check`).

`local us` and `merge us` are per-call slopes `(t(1 + n) - t(1)) / n` of calls
chained in one `fori_loop` through the row lengths, so host dispatch cancels.
They include the TensorCore ops that prepare each call's lengths, which the
real step can overlap with scoring. `--device-time` also traces the chained
step and reports the SparseCore kernels' own device time per step (`sc local
us`, `sc merge us`);
traces go to a temporary directory that is deleted, or to `--xprof DIR`.
`--device-time` needs the local rows and the merge rows to differ in width,
which they do unless `max_model_len / dcp_size` equals `dcp_size * k`.

Usage:
  python scripts/benchmark_sparsecore_topk.py
  python scripts/benchmark_sparsecore_topk.py --list
  python scripts/benchmark_sparsecore_topk.py --cases prefill,mixed \\
      --kernel main=origin/main \\
      --kernel branch=src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py \\
      --device-time --json results.jsonl
"""

import argparse
import collections
import dataclasses
import functools
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as checkout_kernel
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
    _effective_row_lengths,
)

_K, _M = 1 << 10, 1 << 20
_MIN_TOKEN_BUCKET = 16
_MIN_NUM_SEQS = 8
# No top-k slot ever holds this, so comparing against it is a dependency on the
# output that XLA cannot fold away.
_NEVER = -12345
_TARGET_SECONDS, _MIN_EXTRA_CALLS, _MAX_EXTRA_CALLS, _REPEATS = 0.2, 4, 200, 5
_XPROF_CALLS = 10
_KERNEL_PATH = "src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py"


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
        for num_seqs in (1, 2):
            cases.append(
                Case(
                    f"decode c={num_seqs} @{_label(context)}",
                    (window,) * num_seqs,
                    _staggered(num_seqs, context),
                )
            )
    cases.append(Case("decode c=2 spread 16K-1M", (window,) * 2, (16 * _K, _M)))
    cases.append(
        Case(f"decode 1x1+1x{window} @128K", (1, window), _staggered(2, 128 * _K))
    )
    # A prefill chunk fills the rest of the token budget next to a running
    # request: a new prompt, or a new turn on top of a cached long context.
    for context, budget, prefix in (
        (128 * _K, 1024, 0),
        (128 * _K, 4096, 128 * _K),
        (456 * _K, 4096, 456 * _K),
    ):
        chunk = budget - window
        cases.append(
            Case(
                f"mixed 1x{window} @{_label(context)} + {chunk} @{_label(prefix)}",
                (window, chunk),
                (context, prefix + chunk),
            )
        )
    # AgentX turns land at 50-115K; 456K and the 1M edge are the long tail.
    for tokens in (1024, 4096):
        for prefix in (0, 64 * _K, 128 * _K, 456 * _K):
            cases.append(
                Case(
                    f"prefill {tokens} @{_label(prefix)}", (tokens,), (prefix + tokens,)
                )
            )
    cases.append(Case("prefill 4096 @1M-4K", (4096,), (_M,)))
    return cases


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


def _score_width(args: argparse.Namespace) -> int:
    """`num_positions` in `streamindex_topk`: the page table's width in keys.

    Block sizes follow `_streamidx_block_sizes` in `mla_attention_op.py`.
    """
    table_width = -(-args.max_model_len // (args.page_size * args.dcp_size))
    if args.num_spec == 0:
        kv_pages_per_block = (2, 2, 2)
    else:
        kv_pages_per_block = (2, max(1, 16384 // args.page_size), 2)
    sublanes = max(
        -(-table_width // bkv_p) * bkv_p * args.page_size // 128
        for bkv_p in kv_pages_per_block
    )
    return max(sublanes * 128, args.k)


@dataclasses.dataclass(frozen=True)
class Step:
    """One rank's SparseCore top-k work for one indexer call."""

    bucket: int
    chunk_len: int
    row_lengths: np.ndarray  # i32[bucket], rank 0's visible positions per row
    width: int


def _step(case: Case, args: argparse.Namespace) -> Step:
    q_lens = np.asarray(case.q_lens, np.int32)
    num_seqs, num_tokens = len(q_lens), int(q_lens.sum())
    bucket = max(_MIN_TOKEN_BUCKET, 1 << (num_tokens - 1).bit_length())
    seq_slots = max(args.max_num_seqs, _MIN_NUM_SEQS)
    seq_lens = np.zeros(seq_slots, np.int32)
    seq_lens[:num_seqs] = case.seq_lens
    cu_q_lens = np.full(seq_slots + 1, num_tokens, np.int32)
    cu_q_lens[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    width = _score_width(args)
    row_lengths = _effective_row_lengths(
        jnp.asarray(seq_lens),
        jnp.asarray(cu_q_lens),
        jnp.asarray([0, num_seqs, num_seqs], jnp.int32),
        num_tokens=bucket,
        num_positions=width,
        compression_ratio=1,
        cp_rank=0,
        cp_size=args.dcp_size,
        interleave_c=args.interleave,
    )
    chunk = args.chunk_tokens
    chunk_len = chunk if chunk and bucket % chunk == 0 else bucket
    return Step(bucket, chunk_len, np.asarray(row_lengths), width)


def _kernel_source(name: str, src: str) -> str:
    """`sparsecore_topk.py`'s source from a file, or else from a git ref."""
    if os.path.isfile(src):
        with open(src) as f:
            return f.read()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        ["git", "show", f"{src}:{_KERNEL_PATH}"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        sys.exit(
            f"--kernel {name}={src}: not a file, and git show {src}:{_KERNEL_PATH}"
            f" failed: {result.stderr.strip()}"
        )
    return result.stdout


def _load_kernels(specs: list[str]) -> dict:
    if not specs:
        return {"checkout": checkout_kernel}
    kernels = {}
    for spec in specs:
        name, _, src = spec.partition("=")
        with tempfile.NamedTemporaryFile("w", suffix=".py") as f:
            f.write(_kernel_source(name, src))
            f.flush()
            module_spec = importlib.util.spec_from_file_location(
                f"_sc_topk_{name}", f.name
            )
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = module
            module_spec.loader.exec_module(module)
        kernels[name] = module
    return kernels


def _local_calls(kernel, step: Step, scores, row_lengths, k: int):
    """The per-chunk local top-k calls of one indexer call."""
    outs = []
    for start in range(0, step.bucket, step.chunk_len):
        idx, _ = kernel.sparsecore_topk(
            scores,
            k,
            row_lengths=row_lengths[start : start + step.chunk_len],
            write_empty_rows=True,
            return_scores=True,
        )
        outs.append(idx)
    return outs


def _merge_call(kernel, merge_scores, local_idxs, k: int, dcp_size: int, rank):
    return kernel.sparsecore_topk_dcp_merge(
        merge_scores, local_idxs, k, dcp_rank=rank, dcp_size=dcp_size
    )


def _never(outs, row_lengths):
    """`row_lengths`, after every output row."""
    hit = sum((jnp.min(out[:, 0]) == _NEVER).astype(jnp.int32) for out in outs)
    return row_lengths + hit


def _loops(kernel, step: Step, args: argparse.Namespace):
    """Chained `fori_loop`s of the local calls, the merge, and both."""
    k, dcp = args.k, args.dcp_size

    def chain(body):
        return jax.jit(
            lambda reps, *xs: lax.fori_loop(
                0, reps, lambda _, ln: body(ln, *xs), xs[-1]
            )
        )

    def local(ln, scores, merge_scores, local_idxs, lengths):
        return _never(_local_calls(kernel, step, scores, ln, k), ln)

    def merge(ln, scores, merge_scores, local_idxs, lengths):
        # The merge reads no lengths; its rank, always 0, carries the chain.
        rank = (ln[0] == _NEVER).astype(jnp.int32)
        out = _merge_call(kernel, merge_scores, local_idxs, k, dcp, rank)
        return _never([out], ln)

    def both(ln, scores, merge_scores, local_idxs, lengths):
        outs = _local_calls(kernel, step, scores, ln, k)
        # Every chunk feeds the merge, or XLA drops the chunks nothing reads.
        rank = _never(outs, jnp.zeros((1,), jnp.int32))[0]
        out = _merge_call(kernel, merge_scores, local_idxs, k, dcp, rank)
        return _never([out], ln)

    return chain(local), chain(merge), chain(both)


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


def _sc_device_us(run, trace_dir: str, merge_width: int) -> tuple[float, float]:
    """(local, merge) SparseCore top-k device us per step on TPU 0, tracing
    `run`; a kernel `merge_width` wide is the merge."""
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
    # Each SparseCore's own op line: the TensorCore's `call-done` ops only time
    # how long the TensorCore waited, short of the kernel whenever it had other
    # work first. A kernel ends with the busier SparseCore.
    local = collections.defaultdict(float)
    merge = collections.defaultdict(float)
    for plane in ProfileData.from_file(path).planes:
        if not plane.name.startswith("/device:TPU:0 SparseCore"):
            continue
        for line in plane.lines:
            if line.name != "Sparse Core Ops":
                continue
            for event in line.events:
                if "sc_topk" not in event.name:
                    continue
                us = event.duration_ns / 1e3 / _XPROF_CALLS
                part = merge if f"_n{merge_width}_" in event.name else local
                part[plane.name] += us
    return max(local.values(), default=0.0), max(merge.values(), default=0.0)


def _check(outs: dict) -> str:
    """Whether every version selected the first version's indices per row,
    in every local call and in the merge."""
    names = list(outs)
    ref = [np.sort(o, axis=1) for o in outs[names[0]]]
    bad = []
    for name in names[1:]:
        got = [np.sort(o, axis=1) for o in outs[name]]
        if any(not np.array_equal(g, r) for g, r in zip(got, ref)):
            bad.append(name)
    return "ok" if not bad else "DIFF " + ",".join(bad)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", default="", help="comma-separated name substrings")
    parser.add_argument("--list", action="store_true", help="list the cases and exit")
    parser.add_argument(
        "--kernel",
        action="append",
        default=[],
        help="NAME=PATH or NAME=GIT_REF, repeatable",
    )
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--dcp-size", type=int, default=16)
    parser.add_argument("--interleave", type=int, default=4)
    parser.add_argument("--num-spec", type=int, default=3, help="MTP draft tokens")
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=_M)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--device-time", action="store_true")
    parser.add_argument("--xprof", help="keep the --device-time traces here")
    parser.add_argument("--json", help="also write one JSON line per case here")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = _cases(args.num_spec + 1)
    if args.cases:
        wanted = args.cases.split(",")
        cases = [c for c in cases if any(w in c.name for w in wanted)]
    if args.list:
        for case in cases:
            reason = _skip_reason(case, args)
            print(case.name + (f"  (skipped: {reason})" if reason else ""))
        return

    kernels = _load_kernels(args.kernel)
    width = _score_width(args)
    merge_width = args.dcp_size * args.k
    print(
        f"rank 0 of DCP{args.dcp_size}, local rows {width} wide, merge rows"
        f" {merge_width} wide, k={args.k}, chunks of {args.chunk_tokens}"
    )
    header = ["case", "rows", "len p50/max", "P"]
    for name in kernels:
        header += [f"{name} local us", f"{name} merge us"]
        if args.device_time:
            header += [f"{name} sc local us", f"{name} sc merge us"]
    header.append("check")
    print(" | ".join(header))

    rng = np.random.default_rng(0)
    for case in cases:
        reason = _skip_reason(case, args)
        if reason:
            print(f"{case.name} | skipped: {reason}")
            continue
        step = _step(case, args)
        scores = jnp.asarray(
            rng.standard_normal((step.chunk_len, width), dtype=np.float32)
        )
        merge_scores = jnp.asarray(
            rng.standard_normal((step.bucket, merge_width), dtype=np.float32)
        )
        local_idxs = jnp.asarray(
            rng.integers(0, width, (step.bucket, args.k), dtype=np.int32)
        )
        lengths = jnp.asarray(step.row_lengths)
        inputs = (scores, merge_scores, local_idxs, lengths)
        real = step.row_lengths[step.row_lengths > 0]
        row = {
            "case": case.name,
            "rows": f"{step.bucket // step.chunk_len}x{step.chunk_len}",
            "len p50/max": (
                f"{int(np.median(real))}/{int(real.max())}" if real.size else "0/0"
            ),
            "P": "/".join(
                str(m._pick_partition(step.chunk_len, width, args.k))
                for m in kernels.values()
            ),
        }
        outs = {}
        for name, kernel in kernels.items():
            local, merge, both = _loops(kernel, step, args)
            row[f"{name} local us"] = _per_call_us(local, *inputs)
            row[f"{name} merge us"] = _per_call_us(merge, *inputs)
            if args.device_time:
                # The profiler crashes on some characters in its run_dir.
                label = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{name}-{case.name}")
                trace_dir = (
                    os.path.join(args.xprof, label)
                    if args.xprof
                    else tempfile.mkdtemp(prefix="sc_topk_xprof_")
                )
                try:
                    (
                        row[f"{name} sc local us"],
                        row[f"{name} sc merge us"],
                    ) = _sc_device_us(
                        functools.partial(both, _XPROF_CALLS, *inputs),
                        trace_dir,
                        merge_width,
                    )
                finally:
                    if not args.xprof:
                        shutil.rmtree(trace_dir, ignore_errors=True)
            merged = _merge_call(
                kernel, merge_scores, local_idxs, args.k, args.dcp_size, jnp.int32(0)
            )
            outs[name] = [
                np.asarray(o)
                for o in [*_local_calls(kernel, step, scores, lengths, args.k), merged]
            ]
        row["check"] = _check(outs)
        print(
            " | ".join(
                f"{v:.1f}" if isinstance(v, float) else str(v) for v in row.values()
            ),
            flush=True,
        )
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
