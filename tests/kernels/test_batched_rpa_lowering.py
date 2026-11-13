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
"""The batched RPA kernel (--attention-backend CUSTOM) on the installed JAX.

Both KV cache layouts trace and lower for TPU, and the SMEM schedule the
runner admits steps against keeps the whole SMEM budget, so long-context
decode steps still fit.
"""
import contextlib
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core as jax_core
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

pytestmark = pytest.mark.cpu_test

# Qwen3.5-397B attention on one TP8 shard: 4 query heads, 1 KV head, head
# dim 256, fp8 KV cache in 256-token pages.
QWEN = dict(num_q_heads=4,
            num_kv_heads=1,
            head_dim=256,
            page_size=256,
            dtype_q=jnp.bfloat16,
            dtype_kv=jnp.float8_e4m3fn)


@contextlib.contextmanager
def _tpu_info_on_cpu():
    """Answer TPU7x for the CPU device (as tests/kernels/test_pallas_compat.py
    does), so the wrapper can size its tiles and SMEM schedule."""

    def tpu7x():
        return tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)

    tpu_info.get_tpu_info.cache_clear()
    jax.clear_caches()
    try:
        with mock.patch.dict(tpu_info.registry, {"cpu": tpu7x}):
            yield
    finally:
        tpu_info.get_tpu_info.cache_clear()
        jax.clear_caches()


def _trace(kv_layout, num_seqs, pages_per_seq, num_tokens, *, page_size,
           num_q_heads, num_kv_heads, head_dim, dtype_q, dtype_kv):
    sds = jax.ShapeDtypeStruct
    kv_shape = wrapper.get_kv_cache_shape(num_seqs * pages_per_seq + 1,
                                          page_size, num_kv_heads, head_dim,
                                          dtype_kv, kv_layout)
    return wrapper.ragged_paged_attention.trace(
        sds((num_tokens, num_q_heads, head_dim), dtype_q),
        sds((num_tokens, num_kv_heads, head_dim), dtype_kv),
        sds((num_tokens, num_kv_heads, head_dim), dtype_kv),
        sds(kv_shape, dtype_kv),
        sds((num_seqs, ), jnp.int32),
        sds((num_seqs * pages_per_seq, ), jnp.int32),
        sds((num_seqs + 1, ), jnp.int32),
        sds((3, ), jnp.int32),
        sm_scale=head_dim**-0.5,
        kv_layout=kv_layout)


@pytest.mark.parametrize("dtype_kv", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize(
    "kv_layout",
    [configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE])
def test_kernel_lowers_for_tpu(kv_layout, dtype_kv):
    # SEQ_ALONG_LANE (VLLM_KV_CACHE_LAYOUT=HND) takes 128-token pages.
    page_size = (128 if kv_layout == configs.KVLayout.SEQ_ALONG_LANE else 256)
    with _tpu_info_on_cpu():
        traced = _trace(kv_layout,
                        num_seqs=8,
                        pages_per_seq=8,
                        num_tokens=512,
                        **QWEN | dict(page_size=page_size, dtype_kv=dtype_kv))
        traced.lower(lowering_platforms=("tpu", ))


@pytest.mark.parametrize(
    "num_seqs, context",
    [
        # scripts/vllm/benchmarking/configs/qwen3.5-397b-fp8-tp8-dp1-ep.sh:
        # --max-num-seqs 512 --max-model-len 10240.
        (512, 10240),
        (128, 65536),
        (256, 32768),
    ])
def test_schedule_capacity_admits_long_context_decode(num_seqs, context):
    """The runner refuses a step whose (query block, KV block) pairs exceed
    `schedule_capacity`; a decode step with every sequence at full context
    must stay within it."""
    with _tpu_info_on_cpu():
        pairs, bq, bkv = wrapper.schedule_capacity(
            mode=configs.RpaCase.DECODE,
            num_seqs=num_seqs,
            pages_per_seq=-(-context // QWEN["page_size"]),
            **QWEN)
    need = wrapper.schedule_pairs([1] * num_seqs, [context] * num_seqs, bq,
                                  bkv)
    assert need <= pairs


def _smem_bytes(kernel_jaxpr) -> int:
    """SMEM a Pallas TPU kernel allocates: its SMEM operands (the scalar
    prefetch) and its SMEM scratch."""

    def size(refs):
        return sum(
            int(np.prod(r.aval.shape)) * jnp.dtype(r.aval.dtype).itemsize
            for r in refs
            if getattr(r.aval, "memory_space", None) == pltpu.SMEM)

    def scoped(jaxpr):
        total = 0
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "run_scoped":
                total += size(eqn.params["jaxpr"].invars)
            total += sum(
                scoped(s) for s in jax_core.jaxprs_in_params(eqn.params))
        return total

    return size(kernel_jaxpr.invars) + scoped(kernel_jaxpr)


def _kernel_jaxprs(jaxpr):
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            yield eqn.params["jaxpr"]
        else:
            for sub in jax_core.jaxprs_in_params(eqn.params):
                yield from _kernel_jaxprs(sub)


@pytest.mark.parametrize("num_seqs, context, num_tokens", [
    (512, 10240, 512),
    (512, 10240, 8192),
    (128, 65536, 128),
])
def test_kernels_fit_in_smem(num_seqs, context, num_tokens):
    """`max_steps_ub` sizes the schedule from the SMEM capacity less a
    32 KiB reserve; every kernel the wrapper launches stays within that."""
    with _tpu_info_on_cpu():
        limit = pltpu.get_tpu_info().smem_capacity_bytes - 32 * 1024
        traced = _trace(configs.KVLayout.HEAD_ALONG_SUBLANE,
                        num_seqs=num_seqs,
                        pages_per_seq=-(-context // QWEN["page_size"]),
                        num_tokens=num_tokens,
                        **QWEN)
    used = [_smem_bytes(j) for j in _kernel_jaxprs(traced.jaxpr.jaxpr)]
    assert used and max(used) <= limit, used
