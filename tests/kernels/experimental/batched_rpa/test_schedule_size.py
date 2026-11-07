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
"""The HBM schedule holds every step a call can produce."""

import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

pytestmark = pytest.mark.tpu_test


def _cfgs(mode, num_q_tokens, max_kv, bq=512, bkv=512, sliding_window=None):
    page_size = 256
    num_seqs = 1 if mode == configs.RpaCase.MIXED else 8
    return configs.RpaConfigs(
        block=configs.BlockSizes(bq_sz=bq, bq_c_sz=bq, bkv_sz=bkv, batch_size=1),
        model=configs.ModelConfigs(
            num_q_heads=16,
            num_kv_heads=2,
            head_dim=256,
            mask_value=-1e30,
            sliding_window=sliding_window,
        ),
        serve=configs.ServingConfigs(
            num_seqs=num_seqs,
            page_size=page_size,
            total_q_tokens=num_q_tokens,
            num_page_indices=num_seqs * pl.cdiv(max_kv, page_size),
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.bfloat16,
            dtype_out=jnp.bfloat16,
        ),
        mode=mode,
        vmem_limit_bytes=64 * 1024 * 1024,
    )


def _upper_bound_steps(cfgs, num_q_tokens):
    q_blocks = pl.cdiv(num_q_tokens, cfgs.bq_sz) + cfgs.serve.num_seqs
    kv_blocks = pl.cdiv(cfgs.serve.pages_per_seq * cfgs.serve.page_size, cfgs.bkv_sz)
    return pl.cdiv(q_blocks * kv_blocks, cfgs.batch_size)


def test_short_call_keeps_configured_multiplier():
    cfgs = _cfgs(configs.RpaCase.MIXED, num_q_tokens=8192, max_kv=8192)
    assert wrapper.schedule_size_multiplier(cfgs, 8192) == (
        cfgs.max_schedule_size_multiplier
    )


@pytest.mark.parametrize("num_q_tokens", [131072, 212992, 262144])
def test_long_prefill_grows_schedule(num_q_tokens):
    cfgs = _cfgs(configs.RpaCase.MIXED, num_q_tokens, max_kv=num_q_tokens)
    multiplier = wrapper.schedule_size_multiplier(cfgs, num_q_tokens)
    assert multiplier * cfgs.max_steps_ub >= _upper_bound_steps(cfgs, num_q_tokens)


def test_sliding_window_bounds_kv_blocks():
    num_q_tokens = 262144
    full = _cfgs(configs.RpaCase.MIXED, num_q_tokens, max_kv=num_q_tokens)
    windowed = _cfgs(
        configs.RpaCase.MIXED, num_q_tokens, max_kv=num_q_tokens, sliding_window=1024
    )
    assert wrapper.schedule_size_multiplier(
        windowed, num_q_tokens
    ) < wrapper.schedule_size_multiplier(full, num_q_tokens)


def test_decode_bounds_query_blocks():
    cfgs = _cfgs(configs.RpaCase.DECODE, num_q_tokens=4096, max_kv=262144, bq=8)
    assert wrapper.schedule_size_multiplier(cfgs, 4096) == (
        cfgs.max_schedule_size_multiplier
    )
