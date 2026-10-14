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
"""Kernel-time benchmark for GDN v3 plain decode and speculative verify.

Times `fused_conv1d_gdn` at Qwen3.5-397B-A17B per-chip shapes at TP8 (2 KQ
heads, 8 V heads, head dim 128, conv kernel 4) for both state layouts:

* `dense`: the compact slot pool used with prefix caching off. Each request
  owns `num_spec_tokens + 1` consecutive slots.
* `pooled`: the unified KV pool used with prefix caching on. Each checkpoint
  is its own block of a 1-byte pool, like the FP8 KV cache.

Every sequence runs a full window (1 token for plain decode, 1 + drafts for
verify) and resumes from a random checkpoint. Each timed call runs `--iters`
dependent steps in one compiled program, so dispatch stays out of the
per-step time; compilation and warmup are excluded. Run it on the base and on
the change under test and compare the tables.

Run:  python scripts/benchmark_gdn_spec_verify.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vllm_torchtpu.kernels import pool_adapters  # noqa: E402
from vllm_torchtpu.kernels.gdn.v3 import wrapper  # noqa: E402

# Qwen3.5-397B-A17B per chip at TP8 (16 KQ and 64 V heads split 8 ways).
N_KQ, N_V, D_K, D_V = 2, 8, 128, 128
KERNEL_SIZE = 4
DIM = N_KQ * D_K * 2 + N_V * D_V

# Unified pool: 256-token kernel blocks of 256 lanes, 1-byte payload. The
# recurrent state takes two whole kernel blocks; the conv state 16 token rows
# at the start of the third.
KBS, LANES, PAYLOAD, SPLIT = 256, 256, (1, 4), 3
TOK_BYTES = int(np.prod(PAYLOAD)) * LANES
SSM_NTOK = N_V * D_K * D_V * 4 // TOK_BYTES
CONV_NTOK = 16
CONV_ROWS = (KERNEL_SIZE - 1) * DIM // LANES
CONV_SLOT_ROWS = CONV_NTOK * TOK_BYTES // (2 * LANES)


def make_case(layout: str, num_seqs: int, num_spec_tokens: int):
    """Returns (states, kwargs) for one decode/verify step."""
    window = num_spec_tokens + 1
    num_tokens = num_seqs * window
    num_states = 1 + num_seqs * window  # state 0 is the null block
    keys = iter(jax.random.split(jax.random.key(0), 10))

    def normal(shape, dtype, scale=1.0):
        return (jax.random.normal(next(keys), shape) * scale).astype(dtype)

    base = 1 + window * jnp.arange(num_seqs, dtype=jnp.int32)
    kwargs = dict(
        qkv=normal((num_tokens, DIM), jnp.bfloat16),
        b=normal((num_tokens, N_V), jnp.bfloat16),
        a=normal((num_tokens, N_V), jnp.bfloat16),
        conv_weight=normal((DIM, 1, KERNEL_SIZE), jnp.float32),
        conv_bias=normal((DIM,), jnp.float32),
        a_log=normal((N_V,), jnp.float32),
        dt_bias=normal((N_V,), jnp.float32),
        query_start_loc=jnp.arange(0, num_tokens + 1, window, dtype=jnp.int32),
        state_indices=base,
        distribution=jnp.full((3,), num_seqs, dtype=jnp.int32),
        seq_lens=jnp.full((num_seqs,), 64 + window, dtype=jnp.int32),
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
        num_spec_tokens=num_spec_tokens,
        batched_only=True,
    )
    if num_spec_tokens:
        kwargs["read_offsets"] = jax.random.randint(
            next(keys), (num_seqs,), 0, window, dtype=jnp.int32
        )
    recurrent = normal((num_states, N_V, D_K, D_V), jnp.float32, 0.1)
    conv = normal((num_states, KERNEL_SIZE - 1, DIM), jnp.bfloat16, 0.1)
    if layout == "dense":
        return (conv, recurrent), kwargs

    idx = jnp.arange(num_states, dtype=jnp.int32)
    pool = jnp.zeros((num_states * SPLIT, KBS) + PAYLOAD + (LANES,), jnp.int8)
    pool = pool_adapters.scatter_region(
        pool,
        recurrent.reshape(num_states, N_V * D_K, D_V),
        idx,
        tok0=0,
        ntok=SSM_NTOK,
        split=SPLIT,
    )
    rows = conv.reshape(num_states, CONV_ROWS, LANES)
    rows = jnp.pad(rows, ((0, 0), (0, CONV_SLOT_ROWS - CONV_ROWS), (0, 0)))
    pool = pool_adapters.scatter_region(
        pool, rows, idx, tok0=SSM_NTOK, ntok=CONV_NTOK, split=SPLIT
    )
    kwargs["state_plan"] = pool_adapters.v3_state_source(
        pool,
        split=SPLIT,
        ssm_ntok=SSM_NTOK,
        conv_tok0=SSM_NTOK,
        conv_ntok=CONV_NTOK,
        conv_dim=DIM,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
    )
    if num_spec_tokens:
        kwargs["ckpt_indices"] = base[:, None] + jnp.arange(window, dtype=jnp.int32)
    return pool, kwargs


def step(layout: str, states, kwargs):
    if layout == "dense":
        return wrapper.fused_conv1d_gdn(
            conv_state=states[0], recurrent_state=states[1], **kwargs
        )
    return wrapper.fused_conv1d_gdn(
        conv_state=None, recurrent_state=None, state_source=states, **kwargs
    )


def time_case(layout, num_seqs, num_spec_tokens, iters, repeats) -> list[float]:
    """Per-step microseconds, one sample per repeat."""
    states, kwargs = make_case(layout, num_seqs, num_spec_tokens)

    def body(carry, _):
        return step(layout, carry, kwargs)

    run = jax.jit(
        lambda s: jax.lax.scan(body, s, xs=None, length=iters), donate_argnums=0
    )

    def fresh():
        return jax.block_until_ready(jax.tree.map(jnp.copy, states))

    jax.block_until_ready(run(fresh()))
    samples = []
    for _ in range(repeats):
        s = fresh()
        start = time.perf_counter()
        jax.block_until_ready(run(s))
        samples.append((time.perf_counter() - start) / iters * 1e6)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--layouts", nargs="+", default=["dense", "pooled"], choices=["dense", "pooled"]
    )
    parser.add_argument(
        "--num-spec-tokens",
        nargs="+",
        type=int,
        default=[0, 3],
        help="0 is plain decode; N > 0 is a verify window of N + 1 tokens.",
    )
    parser.add_argument("--num-seqs", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--json", type=Path, help="Also write the rows here.")
    args = parser.parse_args()

    rows = []
    print(f"{'layout':<7} {'window':>6} {'seqs':>5} {'median_us':>10} {'min_us':>8}")
    for layout in args.layouts:
        for num_spec_tokens in args.num_spec_tokens:
            for num_seqs in args.num_seqs:
                samples = time_case(
                    layout, num_seqs, num_spec_tokens, args.iters, args.repeats
                )
                row = dict(
                    layout=layout,
                    window=num_spec_tokens + 1,
                    num_seqs=num_seqs,
                    median_us=statistics.median(samples),
                    min_us=min(samples),
                )
                rows.append(row)
                print(
                    f"{layout:<7} {row['window']:>6} {num_seqs:>5} "
                    f"{row['median_us']:>10.1f} {row['min_us']:>8.1f}",
                    flush=True,
                )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
