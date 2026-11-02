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
"""`skip_empty_launch`: a batched-RPA call whose batch segment is empty does no
work, and every output is identical to the call that runs zero steps.

The schedule half runs on CPU in Pallas TPU interpret mode; the kernel half
needs a TPU (interpret mode cannot run the kernel's ref bitcasts). The kernel
test runs with max_num_seqs equal to the token count (conditional launch) and
below it (unconditional launch with the in-kernel early exit).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as KBC
from vllm_torchtpu.kernels.experimental.batched_rpa import schedule as KBS
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as KB

_DECODE_BLOCKS = KBC.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8)
_PREFILL_BLOCKS = KBC.BlockSizes(bq_sz=64, bq_c_sz=16, bkv_sz=256, batch_size=2)
_NUM_Q_HEADS, _NUM_KV_HEADS, _HEAD_DIM = 2, 1, 256
_PAGE_SIZE, _PAGES_PER_SEQ, _MAX_SEQS, _TOKENS = 16, 8, 16, 16


def _configs(mode: KBC.RpaCase, skip_empty_launch: bool) -> KBC.RpaConfigs:
    return KBC.RpaConfigs(
        block=_DECODE_BLOCKS if mode == KBC.RpaCase.DECODE else _PREFILL_BLOCKS,
        model=KBC.ModelConfigs(
            num_q_heads=_NUM_Q_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            sliding_window=None,
            sm_scale=1.0,
            soft_cap=None,
            mask_value=float(jnp.finfo(jnp.bfloat16).min),
        ),
        serve=KBC.ServingConfigs(
            num_seqs=_MAX_SEQS,
            page_size=_PAGE_SIZE,
            total_q_tokens=_TOKENS,
            num_page_indices=_MAX_SEQS * _PAGES_PER_SEQ,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.float8_e4m3fn,
            dtype_out=jnp.bfloat16,
            skip_empty_launch=skip_empty_launch,
        ),
        mode=mode,
        vmem_limit_bytes=64 * 1024 * 1024,
    )


def _schedule(mode, skip_empty_launch, distribution, kv_lens, cu_q_lens):
    """Run the schedule kernel in Pallas TPU interpret mode on CPU."""
    orig = pl.pallas_call

    def interpreted(*args, **kwargs):
        kwargs.pop("interpret", None)
        return orig(*args, interpret=pltpu.InterpretParams(), **kwargs)

    tpu7x = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=tpu7x)
    with pytest.MonkeyPatch.context() as mp, jax.sharding.use_abstract_mesh(mesh):
        mp.setattr(pl, "pallas_call", interpreted)
        cfgs = _configs(mode, skip_empty_launch)
        q_lens = np.diff(cu_q_lens)
        return KBS.generate_rpa_metadata(
            jnp.asarray(cu_q_lens, jnp.int32),
            jnp.asarray(kv_lens, jnp.int32),  # q_offsets
            jnp.asarray(kv_lens, jnp.int32),  # kv_cache_lens
            jnp.asarray(q_lens, jnp.int32),  # kv_new_lens
            jnp.asarray(distribution, jnp.int32),
            jnp.arange(_MAX_SEQS * _PAGES_PER_SEQ, dtype=jnp.int32),
            cfgs=cfgs,
        )


# One decode sequence and no mixed one: the decode-only step of c=1 serving.
_DECODE_ONLY = dict(
    distribution=(1, 1, 1),
    kv_lens=(100,) + (0,) * (_MAX_SEQS - 1),
    cu_q_lens=(0,) + (1,) * _MAX_SEQS,
)


def test_nonempty_schedule_is_unchanged():
    ref = _schedule(KBC.RpaCase.DECODE, False, **_DECODE_ONLY)
    new = _schedule(KBC.RpaCase.DECODE, True, **_DECODE_ONLY)
    assert int(new.actual_steps[0]) == int(ref.actual_steps[0]) > 0
    for r, n in zip(jax.tree.leaves(ref), jax.tree.leaves(new), strict=True):
        np.testing.assert_array_equal(np.asarray(n), np.asarray(r))


@pytest.mark.parametrize("skip_empty_launch", [False, True])
def test_empty_schedule_has_zero_steps(skip_empty_launch):
    sched = _schedule(KBC.RpaCase.MIXED, skip_empty_launch, **_DECODE_ONLY)
    assert int(sched.actual_steps[0]) == 0


def _require_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip(f"Requires TPU backend, found {jax.default_backend()!r}.")


@pytest.mark.tpu_test
@pytest.mark.parametrize("max_seqs", [_MAX_SEQS, 4])
@pytest.mark.parametrize(
    "distribution, kv_lens, cu_q_lens",
    [
        ((1, 1, 1), (100,), (0, 1)),  # decode only: mixed empty
        ((0, 0, 1), (40,), (0, 16)),  # prefill only: decode empty
        ((1, 1, 2), (100, 40), (0, 1, 13)),  # both segments
    ],
)
def test_kernel_outputs_bitwise_equal(max_seqs, distribution, kv_lens, cu_q_lens):
    kv_lens = kv_lens + (0,) * (max_seqs - len(kv_lens))
    cu_q_lens = cu_q_lens + (cu_q_lens[-1],) * (max_seqs + 1 - len(cu_q_lens))
    _require_tpu()
    rng = np.random.default_rng(0)
    cache_shape = KB.get_kv_cache_shape(
        max_seqs * _PAGES_PER_SEQ,
        _PAGE_SIZE,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        jnp.float8_e4m3fn,
    )
    cache = rng.standard_normal(cache_shape).astype(np.float32)
    q = rng.standard_normal((_TOKENS, _NUM_Q_HEADS, _HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((_TOKENS, _NUM_KV_HEADS, _HEAD_DIM)).astype(np.float32)
    v = rng.standard_normal((_TOKENS, _NUM_KV_HEADS, _HEAD_DIM)).astype(np.float32)

    def run(skip_empty_launch):
        # The cache is donated, so each run gets a fresh copy.
        out, new_cache = KB.ragged_paged_attention(
            jnp.asarray(q, jnp.bfloat16),
            # New K/V arrive in the cache dtype, as the attention layer quantizes them.
            jnp.asarray(k, jnp.float8_e4m3fn),
            jnp.asarray(v, jnp.float8_e4m3fn),
            jnp.asarray(cache, jnp.float8_e4m3fn),
            jnp.asarray(kv_lens, jnp.int32),
            jnp.arange(max_seqs * _PAGES_PER_SEQ, dtype=jnp.int32),
            jnp.asarray(cu_q_lens, jnp.int32),
            jnp.asarray(distribution, jnp.int32),
            decode_block_sizes=_DECODE_BLOCKS,
            prefill_block_sizes=_PREFILL_BLOCKS,
            skip_empty_launch=skip_empty_launch,
        )
        return np.asarray(out.view(jnp.uint16)), np.asarray(new_cache.view(jnp.uint8))

    ref_out, ref_cache = run(False)
    new_out, new_cache = run(True)
    np.testing.assert_array_equal(new_out, ref_out)
    np.testing.assert_array_equal(new_cache, ref_cache)
