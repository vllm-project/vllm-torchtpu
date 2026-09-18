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
"""Offline flash-attention block-size tuner.

This is an adaptation of the offline tuner introduced in
https://github.com/vllm-project/tpu-inference/pull/3301.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from statistics import median

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.flash_attention.kernel import (
    BlockSizes,
    SegmentIds,
    flash_attention,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

VMEM_LIMIT_BYTES = 64 * 1024 * 1024
TUNING_GRIDS = (
    (
        "kimi-k3",
        12,
        128,
        (256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 67_328),
    ),
    (
        "qwen2.5-vl",
        16,
        80,
        (256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536),
    ),
)

_Q_BLOCKS = (128, 256, 512, 1_024, 2_048, 4_096, 8_192)
_K_MAJOR_BLOCKS = (128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384)
_K_BLOCKS = (128, 256, 512)
# Larger Q x K-major combinations exceeded TPU7x VMEM during the search.
_MAX_Q_K_MAJOR_ELEMENTS = 8 * 1_024 * 1_024


@dataclass(frozen=True)
class FlashAttentionTuningSpec:
    """One flash-attention workload and its candidate block sizes."""

    name: str
    batch_size: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int
    head_dim: int
    candidates: tuple[BlockSizes, ...]
    q_dtype: str = "bfloat16"
    kv_dtype: str = "bfloat16"
    v_dtype: str = "bfloat16"
    causal: bool = False
    has_segment_ids: bool = False
    has_attention_bias: bool = False
    sm_scale: float = 1.0
    vmem_limit_bytes: int | None = VMEM_LIMIT_BYTES


@dataclass(frozen=True)
class _TuningInputs:
    q: jax.Array
    k: jax.Array
    v: jax.Array
    segment_ids: SegmentIds | None


def _make_inputs(spec: FlashAttentionTuningSpec) -> _TuningInputs:
    if spec.has_attention_bias:
        raise NotImplementedError(
            "Flash-attention tuning does not support attention bias"
        )

    q_shape = (spec.batch_size, spec.num_heads, spec.q_seq_len, spec.head_dim)
    kv_shape = (spec.batch_size, spec.num_heads, spec.kv_seq_len, spec.head_dim)
    q = jnp.ones(q_shape, dtype=jnp.dtype(spec.q_dtype))
    if q_shape == kv_shape and spec.q_dtype == spec.kv_dtype:
        k = q
    else:
        k = jnp.ones(kv_shape, dtype=jnp.dtype(spec.kv_dtype))
    if spec.v_dtype == spec.kv_dtype:
        v = k
    else:
        v = jnp.ones(kv_shape, dtype=jnp.dtype(spec.v_dtype))

    segment_ids = None
    if spec.has_segment_ids:
        q_segment = jnp.zeros((spec.batch_size, spec.q_seq_len), dtype=jnp.int32)
        if spec.q_seq_len == spec.kv_seq_len:
            kv_segment = q_segment
        else:
            kv_segment = jnp.zeros((spec.batch_size, spec.kv_seq_len), dtype=jnp.int32)
        segment_ids = SegmentIds(q=q_segment, kv=kv_segment)
    return _TuningInputs(q=q, k=k, v=v, segment_ids=segment_ids)


def _benchmark_candidate(
    spec: FlashAttentionTuningSpec,
    inputs: _TuningInputs,
    block_sizes: BlockSizes,
    benchmark_iterations: int,
) -> tuple[float, float]:
    start = time.perf_counter()
    out = flash_attention(
        inputs.q,
        inputs.k,
        inputs.v,
        segment_ids=inputs.segment_ids,
        causal=spec.causal,
        sm_scale=spec.sm_scale,
        block_sizes=block_sizes,
        vmem_limit_bytes=spec.vmem_limit_bytes,
    )
    jax.block_until_ready(out)
    compile_and_warmup_seconds = time.perf_counter() - start

    samples_ns = []
    for _ in range(benchmark_iterations):
        start_ns = time.perf_counter_ns()
        out = flash_attention(
            inputs.q,
            inputs.k,
            inputs.v,
            segment_ids=inputs.segment_ids,
            causal=spec.causal,
            sm_scale=spec.sm_scale,
            block_sizes=block_sizes,
            vmem_limit_bytes=spec.vmem_limit_bytes,
        )
        jax.block_until_ready(out)
        samples_ns.append(time.perf_counter_ns() - start_ns)
    return median(samples_ns), compile_and_warmup_seconds


def tune_flash_attention(
    spec: FlashAttentionTuningSpec,
    *,
    benchmark_iterations: int = 3,
) -> BlockSizes | None:
    """Benchmark ``spec`` and return its fastest successful candidate.

    The result is intended to be checked into ``tuned_params.py``. Runtime
    mutation is intentionally unsupported because all enclosing JIT and Pallas
    caches must observe the selected parameters before their first trace.
    """
    if benchmark_iterations < 1:
        raise ValueError("benchmark_iterations must be at least 1")
    if not spec.candidates:
        logger.warning("No flash-attention candidates configured for %s", spec.name)
        return None

    started = time.perf_counter()
    inputs = None
    fastest: tuple[float, BlockSizes] | None = None
    try:
        inputs = _make_inputs(spec)
        for block_sizes in spec.candidates:
            try:
                latency_ns, compile_seconds = _benchmark_candidate(
                    spec, inputs, block_sizes, benchmark_iterations
                )
            except Exception as exc:
                error = str(exc).lower()
                failure = (
                    "VMEM/OOM"
                    if "vmem" in error or "out of memory" in error
                    else "error"
                )
                detail = str(exc).partition("\n")[0]
                logger.warning(
                    "Flash-attention autotune %s skipped %s after %s: %s",
                    spec.name,
                    block_sizes,
                    failure,
                    detail,
                )
                continue

            logger.info(
                "Flash-attention autotune %s candidate=%s "
                "compile_and_warmup_s=%.3f median_ms=%.3f",
                spec.name,
                block_sizes,
                compile_seconds,
                latency_ns / 1e6,
            )
            if fastest is None or latency_ns < fastest[0]:
                fastest = (latency_ns, block_sizes)

        if fastest is None:
            logger.warning(
                "Flash-attention autotune %s found no usable candidate; "
                "keeping the default block-size heuristic",
                spec.name,
            )
            return None

        _, winner = fastest
        logger.info(
            "Flash-attention autotune %s selected %s in %.3fs",
            spec.name,
            winner,
            time.perf_counter() - started,
        )
        return winner
    finally:
        # Candidate executables are only useful for measurement. Callers build
        # their serving executable after lookup, so do not retain every losing
        # candidate in the JIT cache.
        flash_attention.clear_cache()
        del inputs
        gc.collect()


def _candidate_block_sizes(seq_len: int) -> tuple[BlockSizes, ...]:
    candidates = []
    for block_q in _Q_BLOCKS:
        if block_q > seq_len:
            continue
        for block_k_major in _K_MAJOR_BLOCKS:
            if seq_len % block_k_major:
                continue
            if block_q * block_k_major > _MAX_Q_K_MAJOR_ELEMENTS:
                continue
            for block_k in _K_BLOCKS:
                if block_k > block_k_major or block_k_major % block_k:
                    continue
                candidates.append(BlockSizes(block_q, block_k_major, block_k, 1))
    return tuple(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the requested Kimi-K3 and Qwen2.5-VL grids."
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(grid[0] for grid in TUNING_GRIDS),
        help="model to tune; repeat to select both (default: both)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="timed executions per candidate (default: 5)",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="index in jax.local_devices() to benchmark on (default: 0)",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    devices = jax.local_devices()
    if not devices or devices[0].platform != "tpu":
        parser.error("run this script on a TPU node")
    if devices[0].device_kind != "TPU7x":
        parser.error(f"this search targets TPU7x, found {devices[0].device_kind}")
    if args.device_index < 0 or args.device_index >= len(devices):
        parser.error(f"--device-index must be between 0 and {len(devices) - 1}")

    selected_models = set(args.model or (grid[0] for grid in TUNING_GRIDS))
    results = []
    device = devices[args.device_index]
    with jax.default_device(device):
        for model, num_heads, head_dim, seq_lens in TUNING_GRIDS:
            if model not in selected_models:
                continue
            for seq_len in seq_lens:
                candidates = _candidate_block_sizes(seq_len)
                print(
                    f"model={model} seq_len={seq_len} heads={num_heads} "
                    f"head_dim={head_dim} candidates={len(candidates)} "
                    f"device={device}",
                    flush=True,
                )
                winner = tune_flash_attention(
                    FlashAttentionTuningSpec(
                        name=f"{model}-{seq_len}",
                        batch_size=1,
                        num_heads=num_heads,
                        q_seq_len=seq_len,
                        kv_seq_len=seq_len,
                        head_dim=head_dim,
                        candidates=candidates,
                        has_segment_ids=True,
                    ),
                    benchmark_iterations=args.iterations,
                )
                if winner is None:
                    raise SystemExit(f"No candidate compiled for {model} at {seq_len}")
                results.append((model, num_heads, head_dim, seq_len, winner))

    print("\nPaste into tuned_params_mapping:")
    previous_model = None
    for model, num_heads, head_dim, seq_len, winner in results:
        if model != previous_model:
            print(f"    # {model}.")
            previous_model = model
        print(
            f"    _vision_tuning_key(num_heads={num_heads}, "
            f"seq_len={seq_len:_}, head_dim={head_dim}):"
        )
        print(
            f"    TunableParams({winner.block_q:_}, "
            f"{winner.block_k_major:_}, {winner.block_k:_}, "
            f"{winner.block_b}),"
        )


if __name__ == "__main__":
    main()
