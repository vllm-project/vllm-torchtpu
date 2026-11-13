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
"""batched_rpa_longctx DMA priorities on the installed JAX, on CPU.

The kernel issues its KV-cache fetch and writeback, Q fetch and schedule
metadata DMAs at priority 1 and its O / LSE writes at priority 0, through
`kernels.pallas_compat`. Public JAX (the repo pins jax==0.10.2) has neither
`priority` / `strict_ordering` on `pltpu.BufferedRef.create` nor
`strict_ordering` on DMA `start` / `wait`, so passing them to `pltpu` directly
made every call of the kernel raise `TypeError` while tracing. These tests
trace the decode and prefill kernels and the scheduler for both KV layouts and
check the priority each `dma_start` gets.
"""
import contextlib
import functools
from unittest import mock

import jax
import jax.numpy as jnp
import pytest
from jax._src.pallas.mosaic import tpu_info

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import (configs,
                                                                    kernel,
                                                                    wrapper)

pytestmark = pytest.mark.cpu_test

FETCH = "HBM->VMEM (KV cache, new KV, Q)"
KV_WRITEBACK = "VMEM->HBM KV cache"
OUTPUT = "VMEM->HBM O / LSE"
SCHEDULE = "HBM<->SMEM schedule metadata"


@pytest.fixture(autouse=True)
def _on_the_host():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@contextlib.contextmanager
def _tpu_info_on_cpu():
    """The kernel asks for the TPU generation; answer TPU7x for the CPU
    device. Caches are cleared so every call traces afresh."""
    tpu_info.get_tpu_info.cache_clear()
    jax.clear_caches()
    try:
        with mock.patch.dict(
                tpu_info.registry, {
                    "cpu":
                    lambda: tpu_info.get_tpu_info_for_chip(
                        tpu_info.ChipVersion.TPU_7X, 1)
                }):
            yield
    finally:
        tpu_info.get_tpu_info.cache_clear()
        jax.clear_caches()


def _memory_space(ref):
    return str(getattr(ref, "ref", ref).aval.memory_space)


def _priorities_by_kind(jaxpr, kv_cache_shape):
    """{kind of DMA: the priorities its `dma_start`s get} over `jaxpr` and
    its sub-jaxprs."""
    found = {}

    def walk(value):
        if isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
        elif hasattr(value, "eqns"):
            for eqn in value.eqns:
                if eqn.primitive.name == "dma_start":
                    src, dst, *_ = eqn.params["tree"].unflatten(eqn.invars)
                    spaces = _memory_space(src), _memory_space(dst)
                    if "smem" in spaces:
                        kind = SCHEDULE
                    elif spaces == ("hbm", "vmem"):
                        kind = FETCH
                    elif spaces == ("vmem", "hbm"):
                        dst_shape = getattr(dst, "ref", dst).aval.shape
                        kind = (KV_WRITEBACK
                                if dst_shape == kv_cache_shape else OUTPUT)
                    else:
                        kind = spaces
                    found.setdefault(kind, set()).add(eqn.params["priority"])
                walk(list(eqn.params.values()))
        elif hasattr(value, "jaxpr"):
            walk(value.jaxpr)

    walk(jaxpr)
    return found


@pytest.mark.parametrize(
    "kv_layout, num_q_heads, num_kv_heads, head_dim, page_size, kv_dtype", [
        (configs.KVLayout.HEAD_ALONG_SUBLANE, 4, 2, 128, 128, jnp.bfloat16),
        (configs.KVLayout.SEQ_ALONG_LANE, 4, 2, 128, 128, jnp.bfloat16),
        (configs.KVLayout.HEAD_ALONG_SUBLANE, 32, 2, 256, 256,
         jnp.float8_e4m3fn),
    ],
    ids=["head_along_sublane", "seq_along_lane", "qwen3_5_fp8"])
def test_kv_q_and_schedule_dmas_are_issued_at_priority_1(
        monkeypatch, kv_layout, num_q_heads, num_kv_heads, head_dim, page_size,
        kv_dtype):
    allocs = []
    create_allocs = kernel.create_allocs

    def recording_create_allocs(*args, **kwargs):
        allocs.append(create_allocs(*args, **kwargs))
        return allocs[-1]

    monkeypatch.setattr(kernel, "create_allocs", recording_create_allocs)
    num_seqs, num_tokens, pages_per_seq = 4, 64, 4
    num_pages = num_seqs * pages_per_seq + 2
    attention = functools.partial(wrapper.ragged_paged_attention,
                                  sm_scale=head_dim**-0.5,
                                  kv_layout=kv_layout,
                                  return_lse=True)
    with _tpu_info_on_cpu():
        kv_cache_shape = wrapper.get_kv_cache_shape(num_pages,
                                                    page_size,
                                                    num_kv_heads,
                                                    head_dim,
                                                    kv_dtype,
                                                    kv_layout=kv_layout)
        jaxpr = jax.make_jaxpr(attention)(
            jax.ShapeDtypeStruct((num_tokens, num_q_heads, head_dim),
                                 jnp.bfloat16),
            jax.ShapeDtypeStruct((num_tokens, num_kv_heads, head_dim),
                                 kv_dtype),
            jax.ShapeDtypeStruct((num_tokens, num_kv_heads, head_dim),
                                 kv_dtype),
            jax.ShapeDtypeStruct(kv_cache_shape, kv_dtype),
            jax.ShapeDtypeStruct((num_seqs, ), jnp.int32),
            jax.ShapeDtypeStruct((num_seqs * pages_per_seq, ), jnp.int32),
            jax.ShapeDtypeStruct((num_seqs + 1, ), jnp.int32),
            jax.ShapeDtypeStruct((3, ), jnp.int32),
        )

    # One set of allocations per kernel (decode, prefill).
    assert len(allocs) == 2
    for q, kv, o, lse in allocs:
        assert [ref.priority for ref in (q, kv, o, lse)] == [1, 1, 0, 0]
        assert not any(ref.strict_ordering for ref in (q, kv, o, lse))
    assert _priorities_by_kind(jaxpr.jaxpr, kv_cache_shape) == {
        FETCH: {1},
        KV_WRITEBACK: {1},
        OUTPUT: {0},
        SCHEDULE: {1},
    }
