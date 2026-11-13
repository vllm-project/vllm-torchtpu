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
"""GDN v3 DMA hints on the installed JAX, on CPU.

memory_ref issues the qkv input and the conv / recurrent state copies at DMA
priority 1 (and carries `strict_ordering`) through `kernels.pallas_compat`.
Public JAX (the repo pins jax==0.10.2) has neither hint on
`pltpu.BufferedRef.create` nor `strict_ordering` on DMA waits, so passing them
to `pltpu` directly made every GDN v3 call raise `TypeError` while tracing.
These tests trace the dense and both pooled forms, check the priority each
`dma_start` gets, and run decode and zero-tile steps in TPU interpret mode
against the eager reference.
"""
import contextlib
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from test_gdn_attention_v3 import gdn_attention_ref

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.gdn.v3 import memory_ref, wrapper

pytestmark = pytest.mark.cpu_test

N_KQ, N_V, D_K, D_V, KERNEL_SIZE = 1, 2, 128, 128, 4
DIM = N_KQ * D_K * 2 + N_V * D_V
DIMS = dict(n_kq=N_KQ, n_v=N_V, d_k=D_K, d_v=D_V, kernel_size=KERNEL_SIZE)
# Pool geometry for the pooled forms: the SSM state is one whole kernel block
# per slot, the conv state 16 token rows of the next one.
SPLIT, KBS, LANES, NUM_MGR = 2, 128, 256, 6
POOL_SHAPE = (NUM_MGR * SPLIT, KBS, 1, 2, LANES)
SSM_NTOK = N_V * D_K * D_V * 4 // (2 * 2 * LANES)

INTERPRET = pltpu.InterpretParams(uninitialized_memory="nan",
                                  out_of_bounds_reads="raise")


@pytest.fixture(autouse=True)
def _on_the_host():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@contextlib.contextmanager
def _tpu_info_on_cpu():
    """The kernel and its pipeline ask for the TPU generation; answer TPU7x
    for the CPU device. Caches are cleared so every call traces afresh."""
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


def _step(lengths, context, max_reqs, num_decodes=None, seed=0):
    """A step of `lengths` tokens per sequence after `context` tokens, the
    first `num_decodes` (default: all) decodes, padded to `max_reqs` sequences
    (padding maps to the null state slot 0)."""
    num_seqs = len(lengths)
    if num_decodes is None:
        num_decodes = num_seqs
    num_tokens = max(sum(lengths), 1)
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    qsl = np.concatenate([[0], np.cumsum(lengths, dtype=np.int32)])
    pad = max_reqs - num_seqs
    inputs = dict(
        qkv=normal(num_tokens, DIM),
        b=normal(num_tokens, N_V),
        a=normal(num_tokens, N_V),
        conv_state=normal(max_reqs + 1, KERNEL_SIZE - 1, DIM),
        recurrent_state=normal(max_reqs + 1, N_V, D_K, D_V) * 0.1,
        conv_weight=normal(DIM, 1, KERNEL_SIZE),
        conv_bias=normal(DIM),
        a_log=normal(N_V),
        dt_bias=normal(N_V),
        query_start_loc=np.concatenate([qsl, np.full(pad, qsl[-1])]),
        state_indices=np.arange(1, max_reqs + 1) *
        (np.arange(max_reqs) < num_seqs),
        distribution=np.array([num_decodes, num_seqs, num_seqs]),
        seq_lens=np.concatenate([
            np.asarray(lengths, np.int64) + np.asarray(context, np.int64),
            np.zeros(pad, np.int64)
        ]),
    )
    return {
        name:
        jnp.asarray(value,
                    jnp.float32 if value.dtype.kind == "f" else jnp.int32)
        for name, value in inputs.items()
    }


def _call(form):
    """fused_conv1d_gdn on a mixed decode + prefill step, dense or pooled."""
    step = _step([1, 1, 1, 18, 7], [2, 0, 4, 0, 3], max_reqs=8, num_decodes=3)
    if form == "dense":
        return lambda: wrapper.fused_conv1d_gdn(**step, **DIMS)
    pool = jnp.zeros(POOL_SHAPE, jnp.bfloat16)
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
    step.update(conv_state=None,
                recurrent_state=None,
                state_indices=step["state_indices"] % NUM_MGR)
    return lambda: wrapper.fused_conv1d_gdn(**step,
                                            **DIMS,
                                            state_source=pool,
                                            state_plan=plan,
                                            gdn_form=form,
                                            mixed_tile_size=16)


def _dma_start_priorities(jaxpr):
    """The `priority` of every `dma_start` in `jaxpr` and its sub-jaxprs."""
    found = []

    def walk(value):
        if isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
        elif hasattr(value, "eqns"):
            for eqn in value.eqns:
                if eqn.primitive.name == "dma_start":
                    found.append(eqn.params["priority"])
                walk(list(eqn.params.values()))
        elif hasattr(value, "jaxpr"):
            walk(value.jaxpr)

    walk(jaxpr)
    return found


@pytest.mark.parametrize("form", ["dense", "two_calls", "tables"])
def test_qkv_and_state_dmas_are_issued_at_priority_1(monkeypatch, form):
    allocs = []
    create_allocs = memory_ref.create_allocs

    def recording_create_allocs(*args, **kwargs):
        allocs.append(create_allocs(*args, **kwargs))
        return allocs[-1]

    monkeypatch.setattr(memory_ref, "create_allocs", recording_create_allocs)
    with _tpu_info_on_cpu():
        jaxpr = jax.make_jaxpr(_call(form))()

    # One set of allocations per pass (batched decode, per-sequence prefill).
    assert len(allocs) == 2
    for refs in allocs:
        # qkv, b, a, conv state, recurrent state, out
        assert [ref.priority for ref in refs] == [1, 0, 0, 1, 1, 0]
        assert not any(ref.strict_ordering for ref in refs)
    priorities = _dma_start_priorities(jaxpr.jaxpr)
    assert sorted(set(priorities)) == [0, 1], priorities


@pytest.mark.parametrize(
    "lengths, context, max_reqs",
    [
        # Decode only: the per-sequence pass has no tiles.
        ([1] * 5, [0, 3, 0, 2, 1], 8),
        # Padded decode (a partial decode tile).
        ([1] * 3, [1, 1, 1], 6),
        # Empty step: neither pass has tiles.
        ([], [], 3),
    ],
    ids=["decode", "padded_decode", "empty"])
def test_decode_and_zero_tile_steps_match_reference(monkeypatch, lengths,
                                                    context, max_reqs):
    pallas_call = pl.pallas_call

    def interpreted(*args, **kwargs):
        kwargs["interpret"] = INTERPRET
        return pallas_call(*args, **kwargs)

    monkeypatch.setattr(pl, "pallas_call", interpreted)
    step = _step(lengths, context, max_reqs)
    # First: the kernel donates the state inputs.
    (conv_ref, recurrent_ref), out_ref = gdn_attention_ref(**step, **DIMS)
    with _tpu_info_on_cpu():
        (conv, recurrent), out = wrapper.fused_conv1d_gdn(**step, **DIMS)

    num_tokens = sum(lengths)
    np.testing.assert_allclose(np.asarray(out).reshape(out.shape[0],
                                                       -1)[:num_tokens],
                               np.asarray(out_ref)[:num_tokens],
                               rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_allclose(conv, conv_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(recurrent, recurrent_ref, rtol=1e-5, atol=1e-5)
