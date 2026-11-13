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
"""GDN v3 builds on the installed JAX with its DMA hints.

kernels/gdn/v3/memory_ref.py asks for DMA `priority` / `strict_ordering`
hints that only an internal Pallas build accepts. They go through
kernels/pallas_compat.py, so the kernel traces and lowers for TPU on the
pinned public JAX, and the qkv, b, a, out and state copies still issue at
priority 1 there (public Pallas takes `priority` on `start`).
"""
import collections
import contextlib
from unittest import mock

import jax
import jax.numpy as jnp
import pytest
from jax._src import core as jax_core
from jax._src.pallas.mosaic import tpu_info
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.gdn.v3 import memory_ref, wrapper

pytestmark = pytest.mark.cpu_test

N_KQ, N_V, D_K, D_V, KERNEL_SIZE = 2, 8, 128, 128, 4
DIM = N_KQ * D_K * 2 + N_V * D_V
# Pool geometry of tests/kernels/test_v3_pool_state_seam.py.
SPLIT, KBS, LANES = 3, 256, 256
POOL_TAIL = (KBS, 1, 2, LANES)
SSM_NTOK = N_V * D_K * D_V * 4 // (2 * LANES * 2)


@contextlib.contextmanager
def _tpu_info_on_cpu():
    """Answer v6e for the CPU device (as tests/kernels/test_pallas_compat.py
    does), so the wrapper can size its tiles."""

    def v6e():
        return tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V6E, 1)

    tpu_info.get_tpu_info.cache_clear()
    jax.clear_caches()
    try:
        with mock.patch.dict(tpu_info.registry, {"cpu": v6e}):
            yield
    finally:
        tpu_info.get_tpu_info.cache_clear()
        jax.clear_caches()


def _step_args(num_tokens, num_seqs):
    """Abstract inputs of one step of `num_tokens` rows over `num_seqs`
    sequences (how they split into decodes and prefills is data)."""
    f32, i32 = jnp.float32, jnp.int32
    sds = jax.ShapeDtypeStruct
    return dict(
        qkv=sds((num_tokens, DIM), f32),
        b=sds((num_tokens, N_V), f32),
        a=sds((num_tokens, N_V), f32),
        conv_weight=sds((DIM, 1, KERNEL_SIZE), f32),
        conv_bias=sds((DIM, ), f32),
        a_log=sds((N_V, ), f32),
        dt_bias=sds((N_V, ), f32),
        query_start_loc=sds((num_seqs + 1, ), i32),
        state_indices=sds((num_seqs, ), i32),
        distribution=sds((3, ), i32),
        seq_lens=sds((num_seqs, ), i32),
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
    )


def _dense(num_seqs):
    sds = jax.ShapeDtypeStruct
    return dict(conv_state=sds((num_seqs + 1, KERNEL_SIZE - 1, DIM),
                               jnp.float32),
                recurrent_state=sds((num_seqs + 1, N_V, D_K, D_V),
                                    jnp.float32))


def _pooled(num_seqs, gdn_form):
    pool = jax.ShapeDtypeStruct(((num_seqs + 2) * SPLIT, ) + POOL_TAIL,
                                jnp.bfloat16)
    plan = pool_adapters.v3_state_source(pool,
                                         split=SPLIT,
                                         ssm_ntok=SSM_NTOK,
                                         conv_tok0=SSM_NTOK,
                                         conv_ntok=16,
                                         conv_dim=DIM,
                                         n_v=N_V,
                                         d_k=D_K,
                                         d_v=D_V,
                                         kernel_size=KERNEL_SIZE)
    return dict(conv_state=None,
                recurrent_state=None,
                state_source=pool,
                state_plan=plan,
                gdn_form=gdn_form)


def _dma_start_priorities(jaxpr) -> collections.Counter:
    found = collections.Counter()
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dma_start":
            found[eqn.params["priority"]] += 1
        for sub in jax_core.jaxprs_in_params(eqn.params):
            found.update(_dma_start_priorities(sub))
    return found


def test_buffered_ref_keeps_its_priority():
    """BaseBufferedRef.create copies the BufferedRef fields over to the GDN
    subclass; the hints must survive that copy."""
    with _tpu_info_on_cpu():
        ref = memory_ref.InBufferedRef.create(
            spec=pl.BlockSpec((8, 128), lambda i: (i, 0)),
            dtype_or_type=jax.ShapeDtypeStruct((64, 128), jnp.float32),
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=2,
            use_lookahead=False,
            cfg=None,
            metadata_ref=None,
            priority=1)
    assert (ref.priority, ref.strict_ordering) == (1, False)


@pytest.mark.parametrize("state",
                         ["dense", "pooled_two_calls", "pooled_tables"])
@pytest.mark.parametrize("num_tokens, num_seqs", [(4, 4), (26, 4), (300, 3)])
def test_kernel_lowers_with_priority_dmas(state, num_tokens, num_seqs):
    args = _step_args(num_tokens, num_seqs)
    if state == "dense":
        args |= _dense(num_seqs)
    else:
        args |= _pooled(num_seqs, state.removeprefix("pooled_"))
    with _tpu_info_on_cpu():
        traced = wrapper.fused_conv1d_gdn.trace(**args)
        traced.lower(lowering_platforms=("tpu", ))
    priorities = _dma_start_priorities(traced.jaxpr.jaxpr)
    # memory_ref.create_allocs asks for priority 1 on every copy: qkv, b, a,
    # out and the states.
    assert priorities[1] > 0 and set(priorities) == {1}, priorities
