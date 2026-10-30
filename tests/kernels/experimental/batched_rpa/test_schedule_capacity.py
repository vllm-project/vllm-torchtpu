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
"""The HBM schedule is sized for the shape's worst case, and the schedule
kernel never writes past it (rpa_kernel reads it with bounds checks off)."""

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, schedule

_MULT = configs.RpaConfigs.max_schedule_size_multiplier


def _mult(mode, num_seqs, tokens, bq, bkv, batch, pages=2050, ub=128):
    serve = SimpleNamespace(
        num_seqs=num_seqs,
        total_q_tokens=tokens,
        pages_per_seq=pages,
        page_size=128,
        max_schedule_size_multiplier=16,
    )
    cfgs = SimpleNamespace(
        serve=serve, mode=mode, bq_sz=bq, bkv_sz=bkv, batch_size=batch, max_steps_ub=ub
    )
    return _MULT.fget(cfgs)


def test_multiplier_covers_the_worst_case_pairs():
    # DCP=1 MIXED at 32 seqs: (8192/1024 + 32) q-blocks x 257 KV blocks of a
    # 262400-token block table = 10280 pairs; 16 x 128 x 2 = 4096 overflowed.
    m = _mult(configs.RpaCase.MIXED, 32, 8192, 1024, 1024, 2)
    assert m == 41 and m * 128 * 2 >= (8 + 32) * 257
    # DECODE: one q-block per sequence; the serving value already covers it.
    assert _mult(configs.RpaCase.DECODE, 32, 8192, 4, 4096, 8) == 16
    # Capped; past the cap the kernel drops steps instead of writing OOB.
    assert _mult(configs.RpaCase.MIXED, 1024, 1 << 20, 16, 16, 1) == 128


@pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="Mosaic lowering (dynamic DMA sizes)"
)
def test_schedule_overflow_is_dropped_not_written_out_of_bounds(monkeypatch):
    monkeypatch.setattr(
        configs.RpaConfigs, "max_schedule_size_multiplier", property(lambda s: 1)
    )
    num_seqs, pages = 4, 8192
    serve = configs.ServingConfigs(
        num_seqs=num_seqs,
        page_size=128,
        total_q_tokens=128,
        num_page_indices=num_seqs * pages,
        dtype_q=jnp.dtype(jnp.bfloat16),
        dtype_kv=jnp.dtype(jnp.bfloat16),
        dtype_out=jnp.dtype(jnp.bfloat16),
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    cfgs = configs.RpaConfigs(
        block=configs.BlockSizes(
            bq_sz=128, bq_c_sz=128, bkv_sz=128, batch_size=2, n_buffer=2
        ),
        model=configs.ModelConfigs(
            num_q_heads=2,
            num_kv_heads=1,
            head_dim=128,
            mask_value=-1e30,
            sm_scale=1.0,
            soft_cap=None,
            sliding_window=None,
        ),
        serve=serve,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=64 << 20,
    )
    cfgs = dataclasses.replace(cfgs)
    ctx = pages * 128 - 128  # one 128-token q-block over ~8191 KV blocks
    capacity = cfgs.max_steps_ub
    if (ctx // 128 + 1) // 2 <= capacity:
        pytest.skip("SMEM large enough that this shape does not overflow")
    i32 = lambda x: jnp.asarray(x, jnp.int32)  # noqa: E731
    out = schedule.generate_rpa_metadata(
        i32([0, 128, 128, 128, 128]),
        i32([ctx, 0, 0, 0]),
        i32([ctx, 0, 0, 0]),
        i32([128, 0, 0, 0]),
        i32([0, 0, 1]),
        cfgs,
    )
    assert int(jax.device_get(out.actual_steps)[0]) == capacity
