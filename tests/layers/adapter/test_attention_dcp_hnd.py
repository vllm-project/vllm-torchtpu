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
"""Verify DCP + HND behavior with real TPU kernels and a NumPy reference.

Run from the repository root with its dependencies active::

    PYTHONPATH=src python -m pytest -v tests/layers/adapter/test_attention_dcp_hnd.py

The adapter cases use the backend's cache allocation and initialize_kernel(),
including the real Pallas custom ops. The longctx cases pass the layout
explicitly to the underlying wrapper, to distinguish adapter wiring failures
from kernel failures. Neither path replaces or patches a kernel.

The TP8 matrix uses 16 global Q heads and (global KV heads, DCP size) of
(4, 2), (2, 4), and (1, 8). Every rank has two Q heads and one KV head.
The 64-global-Q-head DCP2/4 cases remain as alignment controls, with eight
local Q heads per rank.

Each configuration tests every rank of one representative DCP group, with
the ranks running independently on one TPU. The other groups in TP8 have
the same geometry. This exercises real kernels, not a distributed TP8 job:
AllGather and the final LSE merges are not covered by these pass tests.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.config import get_current_vllm_config

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import (configs,
                                                                    wrapper)
from vllm_torchtpu.layers.adapter import attention, cp_attention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context

_TP_SIZE = 8
_HEAD_DIM = 256
_PAGE_SIZE = 128
_PAGES_PER_SEQ = 2
_K_SCALE = 0.5
_V_SCALE = 0.25
_FP8 = torch.float8_e4m3fn

# (global Q heads, global KV heads, DCP size). All TP replicas of a KV head
# belong to the same DCP group, so every rank still holds ONE local KV head.
_PARALLEL_CASES = (
    # Eight local Q heads provide alignment controls.
    (64, 4, 2),
    (64, 2, 4),
    # Two local Q heads exercise LSE writeback with small groups.
    (16, 4, 2),
    (16, 2, 4),
    (16, 1, 8),
)


@pytest.fixture(autouse=True)
def _require_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("DCP Pallas kernels require a TPU")


def _write_slot(cache, layout, page, offset, key, value):
    """Pack one KV head without using the kernel's packing/writeback helpers."""
    if layout == "HND":
        cache[page, 0, :, :, offset] = key.reshape(_HEAD_DIM // 4, 4)
        cache[page, 1, :, :, offset] = value.reshape(_HEAD_DIM // 4, 4)
    else:
        cache[page, offset, 0, 0, :] = key
        cache[page, offset, 0, 1, :] = value


def _reference(query, keys, values):
    if len(keys) == 0:
        return (np.zeros_like(query),
                np.full(query.shape[0], -np.inf, np.float32))
    scores = query @ (keys * _K_SCALE).T * _HEAD_DIM**-0.5
    maximum = scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores - maximum)
    denominator = weights.sum(axis=-1, keepdims=True)
    return (weights @ (values * _V_SCALE) / denominator,
            (maximum + np.log(denominator))[:, 0])


def _make_case(layout, rank, scope, *, dcp_size, own_q_heads):
    # Identical logical data across layouts, ranks, and entry points. Quantize
    # before computing the reference so FP8 rounding is not a reference error.
    rng = np.random.default_rng(20260915)
    # Exercise both sides of every ownership boundary, including wraparound
    # into rank 0's second local page. Keeping only 127/128/129/255/256/257
    # would never test a live cache or a new-token write on ranks 3 through 7.
    kv_lens = tuple(page * _PAGE_SIZE + delta
                    for page in range(1, dcp_size + 1) for delta in (-1, 0, 1))
    num_seqs = len(kv_lens)

    def random_tensor(shape, dtype):
        data = rng.normal(0, 0.5, shape).astype(np.float32)
        return torch.from_numpy(data).to(dtype)

    queries = random_tensor((num_seqs, dcp_size * own_q_heads, _HEAD_DIM),
                            torch.bfloat16)
    keys = random_tensor((num_seqs, max(kv_lens), _HEAD_DIM), _FP8).float()
    values = random_tensor((num_seqs, max(kv_lens), _HEAD_DIM), _FP8).float()
    keys, values = keys.numpy(), values.numpy()
    if scope == configs.AttentionScope.NEW_TOKENS_ONLY:
        queries = queries[:, rank * own_q_heads:(rank + 1) * own_q_heads]
    queries = queries.contiguous()

    num_pages = num_seqs * _PAGES_PER_SEQ + 4  # Also check unused pages.
    page_table = rng.permutation(num_pages)[:num_seqs * _PAGES_PER_SEQ]
    page_table = page_table.reshape(num_seqs, _PAGES_PER_SEQ).astype(np.int32)
    shape = attention.PallasBatchedRPAAttentionBackend.get_kv_cache_shape(
        num_pages, _PAGE_SIZE, 1, _HEAD_DIM, _FP8)
    cache = np.zeros(shape, dtype=np.float32)
    # A physical HND page uses exactly half the bytes for a single FP8 KV head.
    assert np.prod(shape[1:]) // _PAGE_SIZE == (512
                                                if layout == "HND" else 1024)
    expected_out, expected_lse = [], []
    new_keys, new_values = [], []
    for seq, kv_len in enumerate(kv_lens):
        history = np.arange(kv_len - 1)
        owned = history[(history // _PAGE_SIZE) % dcp_size == rank]
        for pos in owned:
            page = page_table[seq, pos // (_PAGE_SIZE * dcp_size)]
            _write_slot(cache, layout, page, pos % _PAGE_SIZE, keys[seq, pos],
                        values[seq, pos])
        positions = (owned if scope == configs.AttentionScope.CACHE_ONLY else
                     np.array([kv_len - 1]))
        out, lse = _reference(queries[seq].float().numpy(),
                              keys[seq, positions], values[seq, positions])
        expected_out.append(out)
        expected_lse.append(lse)
        new_keys.append(keys[seq, kv_len - 1])
        new_values.append(values[seq, kv_len - 1])

    expected_cache = cache.copy()
    if scope == configs.AttentionScope.NEW_TOKENS_ONLY:
        for seq, kv_len in enumerate(kv_lens):
            pos = kv_len - 1
            if (pos // _PAGE_SIZE) % dcp_size == rank:
                page = page_table[seq, pos // (_PAGE_SIZE * dcp_size)]
                _write_slot(expected_cache, layout, page, pos % _PAGE_SIZE,
                            new_keys[seq], new_values[seq])

    args = (
        torch.from_numpy(cache).to(_FP8),
        queries,
        torch.from_numpy(np.stack(new_keys)[:, None, :]).to(_FP8),
        torch.from_numpy(np.stack(new_values)[:, None, :]).to(_FP8),
        torch.tensor(kv_lens, dtype=torch.int32),
        torch.from_numpy(page_table.reshape(-1)),
        torch.arange(num_seqs + 1, dtype=torch.int32),
        torch.full((3, ), num_seqs, dtype=torch.int32),
    )
    return args, np.stack(expected_out), np.stack(expected_lse), expected_cache


def _run_adapter(args, rank, scope, monkeypatch, *, dcp_size, own_q_heads):
    # Only supply process-group metadata; initialization and both custom ops
    # remain the production implementation. No collective is needed for a pass.
    monkeypatch.setattr(
        attention, "_get_dcp_group",
        lambda: SimpleNamespace(world_size=dcp_size, rank_in_group=rank))
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_REGISTRY", {})
    impl = attention.PallasBatchedRPAAttentionBackendImpl(
        num_heads=own_q_heads,
        head_size=_HEAD_DIM,
        scale=_HEAD_DIM**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=_TP_SIZE,
                                        prefill_context_parallel_size=1,
                                        decode_context_parallel_size=dcp_size))
    layer = SimpleNamespace(_k_scale_float=_K_SCALE, _v_scale_float=_V_SCALE)
    with set_vllm_model_wrapper_context(mesh=None, vllm_config=config):
        impl.initialize_kernel(layer)
    kernel = (impl.rpa_dcp_cache_kernel
              if scope == configs.AttentionScope.CACHE_ONLY else
              impl.rpa_dcp_new_kernel)
    device_args = tuple(tensor.to("tpu") for tensor in args)
    output, lse = kernel(*device_args, None, _K_SCALE, _V_SCALE)
    return (output.cpu().float().numpy(), lse.cpu().float().numpy(),
            device_args[0].cpu().float().numpy())


def _run_longctx(args, layout, rank, scope, *, dcp_size):
    """Control: call the unmodified lower wrapper with an explicit layout."""
    dtype_map = {
        _FP8: jnp.float8_e4m3fn,
        torch.bfloat16: jnp.bfloat16,
        torch.int32: jnp.int32,
    }
    cache, query, key, value, *metadata = (jnp.asarray(
        tensor.float().numpy(), dtype=dtype_map[tensor.dtype])
                                           for tensor in args)
    output, updated_cache, lse = wrapper.ragged_paged_attention(
        query,
        key,
        value,
        cache,
        *metadata,
        sm_scale=_HEAD_DIM**-0.5,
        k_scale=_K_SCALE,
        v_scale=_V_SCALE,
        cp_group_size=dcp_size,
        cp_rank=jnp.array([rank], jnp.int32),
        attention_scope=scope,
        return_lse=True,
        kv_layout=(configs.KVLayout.SEQ_ALONG_LANE if layout == "HND" else
                   configs.KVLayout.HEAD_ALONG_SUBLANE),
    )
    return tuple(
        np.asarray(tensor.astype(jnp.float32))
        for tensor in (output, lse, updated_cache))


@pytest.mark.parametrize("entry", ["adapter", "longctx"])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "global_q_heads,global_kv_heads,dcp_size,rank",
    [
        pytest.param(
            q_heads,
            kv_heads,
            dcp_size,
            rank,
            id=f"tp8-q{q_heads}-kv{kv_heads}-dcp{dcp_size}-rank{rank}")
        for q_heads, kv_heads, dcp_size in _PARALLEL_CASES
        for rank in range(dcp_size)
    ],
)
@pytest.mark.parametrize("scope", [
    configs.AttentionScope.CACHE_ONLY,
    configs.AttentionScope.NEW_TOKENS_ONLY,
],
                         ids=lambda scope: scope.name)
def test_dcp_fp8_pass_matches_reference(entry, layout, global_q_heads,
                                        global_kv_heads, dcp_size, rank, scope,
                                        monkeypatch, vllm_config_context):
    assert global_q_heads % _TP_SIZE == 0
    assert global_kv_heads * dcp_size == _TP_SIZE
    own_q_heads = global_q_heads // _TP_SIZE
    get_current_vllm_config().cache_config.kv_cache_layout = (
        "LBHNC" if layout == "HND" else "LBNHC")
    args, expected_out, expected_lse, expected_cache = _make_case(
        layout, rank, scope, dcp_size=dcp_size, own_q_heads=own_q_heads)
    if entry == "adapter":
        output, lse, cache = _run_adapter(args,
                                          rank,
                                          scope,
                                          monkeypatch,
                                          dcp_size=dcp_size,
                                          own_q_heads=own_q_heads)
    else:
        output, lse, cache = _run_longctx(args,
                                          layout,
                                          rank,
                                          scope,
                                          dcp_size=dcp_size)

    # BF16 output and LSE; the FP8 input rounding is already in the reference.
    # An empty shard has LSE=-inf and zero merge weight. Its output buffer is
    # unspecified (the kernel may leave the input Q there), so only compare
    # output rows with a nonempty history; still check every LSE below.
    nonempty = np.isfinite(expected_lse)
    np.testing.assert_allclose(output[nonempty],
                               expected_out[nonempty],
                               atol=3e-3,
                               rtol=1e-2)
    np.testing.assert_allclose(lse, expected_lse, atol=4e-2, rtol=1e-2)
    # Exact because writes only copy already-quantized values. Comparing the
    # entire pool also catches changes to old tokens, other owners and padding.
    np.testing.assert_array_equal(cache, expected_cache)
