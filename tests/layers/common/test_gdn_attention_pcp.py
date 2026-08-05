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

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    build_pcp_rank_major_token_order as _build_pcp_rank_major_token_order
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import \
    pcp_local_token_counts as _pcp_local_token_counts
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_v3_wrapper
from vllm_torchtpu.kernels.quantized_matmul import \
    util as quantized_matmul_util
from vllm_torchtpu.layers.common.gdn_attention import (
    _derive_pcp_ragged_exchange_descriptors,
    _derive_pcp_rank_major_reorder_indices,
    _exchange_pcp_token_shards_for_head_shards,
    _select_replicated_shard_for_pcp_rank,
    _validate_pcp_ragged_exchange_layout_support,
    run_jax_gdn_attention_pcp_tp_prefill,
    run_jax_gdn_attention_pooled_pcp_prefill,
    run_jax_gdn_attention_pooled_pcp_prefill_projection)
from vllm_torchtpu.layers.common.utils import (
    inverse_reorder_for_sharding, reorder_concatenated_tensor_for_sharding)

pytestmark = pytest.mark.multichip

_GDN_PCP_NUMERICAL_CASES = (
    pytest.param(2, (20, 20, 24), 4, None, id="pcp2-multi-request-padding"),
    pytest.param(4, (34, 30), 4, None, id="pcp4-uneven-rank-split"),
    pytest.param(4, (20, 20, 24), 4, (7, 22, 3), id="pcp4-chunk-continuation"),
)

_GDN_PCP_DESCRIPTOR_CASES = _GDN_PCP_NUMERICAL_CASES + (pytest.param(
    8, (5, 7, 3, 9), 2, (1, 6, 13, 29), id="pcp8-fragmented-descriptors"), )


def test_pcp_gdn_helpers_are_available():
    assert callable(_exchange_pcp_token_shards_for_head_shards)
    assert callable(run_jax_gdn_attention_pcp_tp_prefill)


def test_inverse_reorder_for_sharding_round_trips_concatenated_splits():
    tensor = jnp.arange(2 * 12).reshape(2, 12)
    split_sizes = [4, 4, 4]
    reordered = reorder_concatenated_tensor_for_sharding(
        tensor,
        split_sizes,
        n_shards=2,
        dim=-1,
    )
    restored = inverse_reorder_for_sharding(reordered, split_sizes, 2, -1)
    assert restored.tolist() == tensor.tolist()


def test_replicated_pcp_shard_selection_avoids_axis_index_partition_id(
        monkeypatch):
    tensor = jnp.arange(2 * 6).reshape(2, 6)
    calls = {}

    def fail_axis_index(axis_name):
        raise AssertionError(
            f"replicated shard selection must not use axis_index({axis_name})")

    original_dynamic_slice = jax.lax.dynamic_slice_in_dim

    def fake_dynamic_slice_in_dim(tensor_arg, start_index, slice_size, axis=0):
        calls["slice"] = (start_index, slice_size, axis)
        return original_dynamic_slice(tensor_arg,
                                      start_index,
                                      slice_size,
                                      axis=axis)

    def fake_all_to_all(tensor_arg, *, axis_name, split_axis, concat_axis,
                        tiled):
        calls["all_to_all"] = (axis_name, split_axis, concat_axis, tiled)
        rank_one_shard = original_dynamic_slice(tensor_arg, 2, 2, axis=1)
        return jnp.concatenate([rank_one_shard] * 3, axis=1)

    monkeypatch.setattr(jax.lax, "axis_index", fail_axis_index)
    monkeypatch.setattr(jax.lax, "dynamic_slice_in_dim",
                        fake_dynamic_slice_in_dim)
    monkeypatch.setattr(jax.lax, "all_to_all", fake_all_to_all)

    shard = _select_replicated_shard_for_pcp_rank(
        tensor,
        "pcp",
        3,
        axis=1,
    )

    assert calls["all_to_all"] == ("pcp", 1, 1, True)
    assert calls["slice"] == (0, 2, 1)
    assert shard.tolist() == [[2, 3], [8, 9]]


def test_derive_pcp_rank_major_reorder_indices_matches_host_order():
    pcp_size = 2
    interleave_size = 16
    lengths = np.array([32, 64], dtype=np.int32)
    padded_num_tokens = 96
    local_padded_num_tokens = padded_num_tokens // pcp_size
    query_start_loc = jnp.array([0, 32, 96], dtype=jnp.int32)

    expected, _ = _build_pcp_rank_major_token_order(
        lengths,
        pcp_size,
        interleave_size,
        padded_num_tokens,
    )
    actual = _derive_pcp_rank_major_reorder_indices(
        query_start_loc,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        local_padded_num_tokens=local_padded_num_tokens,
    )

    np.testing.assert_array_equal(np.array(actual), expected.astype(np.int32))


def test_derive_pcp_rank_major_reorder_indices_matches_unaligned_host_order():
    pcp_size = 4
    interleave_size = 4
    lengths = np.array([10, 10], dtype=np.int32)
    padded_num_tokens = int(
        _pcp_local_token_counts(lengths, pcp_size,
                                interleave_size).max()) * pcp_size
    local_padded_num_tokens = padded_num_tokens // pcp_size
    query_start_loc = jnp.array([0, 10, 20], dtype=jnp.int32)

    expected, _ = _build_pcp_rank_major_token_order(
        lengths,
        pcp_size,
        interleave_size,
        padded_num_tokens,
    )
    actual = _derive_pcp_rank_major_reorder_indices(
        query_start_loc,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        local_padded_num_tokens=local_padded_num_tokens,
    )

    np.testing.assert_array_equal(np.array(actual), expected.astype(np.int32))


def test_derive_pcp_rank_major_reorder_indices_uses_chunk_offsets():
    pcp_size = 4
    interleave_size = 4
    lengths = np.array([13, 11], dtype=np.int32)
    token_start_offsets = np.array([7, 22], dtype=np.int32)
    padded_num_tokens = int(
        _pcp_local_token_counts(
            lengths,
            pcp_size,
            interleave_size,
            token_start_offsets_per_req=token_start_offsets,
        ).max()) * pcp_size
    local_padded_num_tokens = padded_num_tokens // pcp_size
    query_start_loc = jnp.array([0, 13, 24], dtype=jnp.int32)
    seq_lens = jnp.array(token_start_offsets + lengths, dtype=jnp.int32)

    expected, _ = _build_pcp_rank_major_token_order(
        lengths,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_start_offsets_per_req=token_start_offsets,
    )
    actual = _derive_pcp_rank_major_reorder_indices(
        query_start_loc,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        local_padded_num_tokens=local_padded_num_tokens,
        seq_lens=seq_lens,
    )

    np.testing.assert_array_equal(np.array(actual), expected.astype(np.int32))


@pytest.mark.parametrize(
    ("pcp_size", "lengths", "interleave_size", "token_start_offsets"),
    _GDN_PCP_DESCRIPTOR_CASES,
)
def test_derive_pcp_ragged_exchange_descriptors_reconstructs_reorder(
        pcp_size, lengths, interleave_size, token_start_offsets):
    lengths = np.asarray(lengths, dtype=np.int32)
    offsets = (None if token_start_offsets is None else np.asarray(
        token_start_offsets, dtype=np.int32))
    local_counts = _pcp_local_token_counts(
        lengths,
        pcp_size,
        interleave_size,
        token_start_offsets_per_req=offsets,
    )
    local_padded_num_tokens = int(local_counts.max())
    query_start_loc = jnp.asarray(
        np.concatenate(([0], np.cumsum(lengths, dtype=np.int32))),
        dtype=jnp.int32,
    )
    seq_lens = None
    if offsets is not None:
        seq_lens = jnp.asarray(offsets + lengths, dtype=jnp.int32)

    reorder = _derive_pcp_rank_major_reorder_indices(
        query_start_loc,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        local_padded_num_tokens=local_padded_num_tokens,
        seq_lens=seq_lens,
    )
    input_starts, sizes, output_starts = (
        _derive_pcp_ragged_exchange_descriptors(
            reorder,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            local_padded_num_tokens=local_padded_num_tokens,
            max_num_requests=len(lengths),
        ))

    reconstructed = np.full(
        (pcp_size * local_padded_num_tokens, ),
        -1,
        dtype=np.int32,
    )
    input_starts = np.asarray(input_starts)
    sizes = np.asarray(sizes)
    output_starts = np.asarray(output_starts)
    for rank in range(pcp_size):
        rank_base = rank * local_padded_num_tokens
        for input_start, size, output_start in zip(input_starts[rank],
                                                   sizes[rank],
                                                   output_starts[rank]):
            size = int(size)
            if size == 0:
                continue
            dst_start = rank_base + int(input_start)
            reconstructed[dst_start:dst_start + size] = np.arange(
                int(output_start),
                int(output_start) + size,
                dtype=np.int32,
            )

    np.testing.assert_array_equal(reconstructed, np.asarray(reorder))
    assert int(sizes.sum()) == int(lengths.sum())


def test_pcp_ragged_exchange_layout_accepts_tpu_generation_7(monkeypatch):
    monkeypatch.setattr(
        "vllm_torchtpu.layers.common.gdn_attention.pltpu.get_tpu_info",
        lambda: SimpleNamespace(generation=7),
    )

    _validate_pcp_ragged_exchange_layout_support()


@pytest.mark.parametrize("generation", (4, 5, 6, 8))
def test_pcp_ragged_exchange_layout_rejects_unvalidated_tpu_generation(
        monkeypatch, generation):
    monkeypatch.setattr(
        "vllm_torchtpu.layers.common.gdn_attention.pltpu.get_tpu_info",
        lambda: SimpleNamespace(generation=generation),
    )

    with pytest.raises(
            NotImplementedError,
            match="only validated on TPU generation 7",
    ):
        _validate_pcp_ragged_exchange_layout_support()


def _require_tpu_devices(min_count, reason):
    devices = jax.local_devices()
    if len(devices) < min_count or devices[0].platform != 'tpu':
        pytest.skip(reason)
    return devices


@pytest.mark.parametrize(
    ("pcp_size", "lengths", "interleave_size", "token_start_offsets"),
    _GDN_PCP_NUMERICAL_CASES,
)
@pytest.mark.parametrize("state_layout", ("split", "unified_pool"))
def test_pcp_prefill_matches_non_pcp_baseline_with_raw_qkv_layout(
        pcp_size, lengths, interleave_size, token_start_offsets, state_layout):
    _require_tpu_devices(
        pcp_size,
        f"GDN PCP numerical test requires {pcp_size} TPU devices.",
    )
    lengths = np.asarray(lengths, dtype=np.int32)
    offsets = (None if token_start_offsets is None else np.asarray(
        token_start_offsets, dtype=np.int32))
    n_kq = 4
    n_v = 4
    d_k = 128
    d_v = 128
    kernel_size = 4
    num_tokens = int(lengths.sum())
    num_blocks = len(lengths) + 1
    dim = 2 * n_kq * d_k + n_v * d_v
    qkv_split_sizes = [n_kq * d_k, n_kq * d_k, n_v * d_v]

    rng = jax.random.key(0)
    keys = jax.random.split(rng, 8)
    mixed_qkv0 = jax.random.normal(keys[0], (num_tokens, dim),
                                   dtype=jnp.bfloat16)
    b0 = jax.random.normal(keys[1], (num_tokens, n_v), dtype=jnp.bfloat16)
    a0 = jax.random.normal(keys[2], (num_tokens, n_v), dtype=jnp.bfloat16)
    conv_state0 = jnp.zeros((num_blocks, kernel_size - 1, dim),
                            dtype=jnp.bfloat16)
    rec_state0 = jnp.zeros((num_blocks, n_v, d_k, d_v), dtype=jnp.float32)
    conv_weight0 = jax.random.normal(keys[3], (dim, 1, kernel_size),
                                     dtype=jnp.bfloat16)
    conv_bias0 = jax.random.normal(keys[4], (dim, ), dtype=jnp.bfloat16)
    A_log0 = jax.random.normal(keys[5], (n_v, ), dtype=jnp.float32)
    dt_bias0 = jax.random.normal(keys[6], (n_v, ), dtype=jnp.float32)

    query_start_loc = jnp.asarray(
        np.concatenate(([0], np.cumsum(lengths, dtype=np.int32))),
        dtype=jnp.int32,
    )
    state_indices = jnp.arange(1, len(lengths) + 1, dtype=jnp.int32)
    distribution = jnp.array([0, len(lengths), len(lengths)], dtype=jnp.int32)
    seq_lens = jnp.asarray(
        lengths if offsets is None else offsets + lengths,
        dtype=jnp.int32,
    )

    padded_num_tokens = int(
        _pcp_local_token_counts(
            lengths,
            pcp_size,
            interleave_size,
            token_start_offsets_per_req=offsets,
        ).max()) * pcp_size
    token_order, _ = _build_pcp_rank_major_token_order(
        lengths,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_start_offsets_per_req=offsets,
    )
    valid = token_order >= 0
    pad_tokens = padded_num_tokens - num_tokens
    mixed_qkv_padded = jnp.pad(mixed_qkv0, ((0, pad_tokens), (0, 0)))
    b_padded = jnp.pad(b0, ((0, pad_tokens), (0, 0)))
    a_padded = jnp.pad(a0, ((0, pad_tokens), (0, 0)))
    valid_indices = np.where(valid)[0]
    src_indices = token_order[valid]
    packed_qkv = jnp.zeros_like(mixed_qkv_padded).at[valid_indices].set(
        mixed_qkv_padded[src_indices])
    packed_b = jnp.zeros_like(b_padded).at[valid_indices].set(
        b_padded[src_indices])
    packed_a = jnp.zeros_like(a_padded).at[valid_indices].set(
        a_padded[src_indices])

    mixed_qkv = jnp.array(np.array(mixed_qkv0))
    b = jnp.array(np.array(b0))
    a = jnp.array(np.array(a0))
    conv_state = jnp.array(np.array(conv_state0))
    rec_state = jnp.array(np.array(rec_state0))
    conv_weight = jnp.array(np.array(conv_weight0))
    conv_bias = jnp.array(np.array(conv_bias0))
    A_log = jnp.array(np.array(A_log0))
    dt_bias = jnp.array(np.array(dt_bias0))

    (ref_conv, ref_rec), ref_output = gdn_v3_wrapper.fused_conv1d_gdn(
        mixed_qkv,
        b,
        a,
        conv_state,
        rec_state,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    mesh = Mesh(
        np.array(jax.devices()[:pcp_size]).reshape((pcp_size, )), ('pcp', ))

    def shard_tokens(x):
        return jax.device_put(x, NamedSharding(mesh, P('pcp', None)))

    def shard_conv_state(x):
        rank_major = reorder_concatenated_tensor_for_sharding(
            x, qkv_split_sizes, pcp_size, -1)
        return jax.device_put(rank_major,
                              NamedSharding(mesh, P(None, None, 'pcp')))

    def shard_rec_state(x):
        return jax.device_put(x, NamedSharding(mesh, P(None, 'pcp', None,
                                                       None)))

    def shard_pool(x):
        return jax.device_put(x, NamedSharding(mesh, P('pcp')))

    def replicate(x):
        return jax.device_put(x, NamedSharding(mesh, P()))

    common_pcp_args = (
        shard_tokens(packed_qkv),
        shard_tokens(packed_b),
        shard_tokens(packed_a),
    )
    common_pcp_kwargs = dict(
        j_conv_weight=replicate(conv_weight0),
        j_conv_bias=replicate(conv_bias0),
        j_A_log=replicate(A_log0),
        j_dt_bias=replicate(dt_bias0),
        state_indices=replicate(state_indices),
        query_start_loc=replicate(query_start_loc),
        distribution=replicate(distribution),
        seq_lens=replicate(seq_lens),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        mesh=mesh,
    )
    if state_layout == "split":
        (pcp_conv, pcp_rec), pcp_output = run_jax_gdn_attention_pcp_tp_prefill(
            *common_pcp_args,
            conv_state=shard_conv_state(conv_state0),
            recurrent_state=shard_rec_state(rec_state0),
            **common_pcp_kwargs,
        )
    else:
        pool_kernel_block_tokens = 256
        pool_block_tokens = 512
        pool_split = pool_block_tokens // pool_kernel_block_tokens
        local_pool_shape = (
            num_blocks * pool_split,
            pool_kernel_block_tokens,
            1,
            4,
            128,
        )
        global_pool_shape = (pcp_size * local_pool_shape[0],
                             *local_pool_shape[1:])
        pcp_pool, pcp_output = run_jax_gdn_attention_pooled_pcp_prefill(
            *common_pcp_args,
            recurrent_state=shard_pool(
                jnp.zeros(global_pool_shape, dtype=jnp.float8_e4m3fn)),
            pool_block_tokens=pool_block_tokens,
            **common_pcp_kwargs,
        )

    pcp_output_np = np.array(pcp_output).reshape(padded_num_tokens, -1)
    pcp_output_seq = np.zeros((padded_num_tokens, pcp_output_np.shape[1]),
                              dtype=pcp_output_np.dtype)
    pcp_output_seq[token_order[valid]] = pcp_output_np[valid]

    np.testing.assert_allclose(pcp_output_seq[:num_tokens],
                               np.array(ref_output),
                               rtol=5e-2,
                               atol=5e-2)
    if state_layout == "split":
        pcp_conv_raw = inverse_reorder_for_sharding(pcp_conv, qkv_split_sizes,
                                                    pcp_size, -1)
        np.testing.assert_allclose(np.array(pcp_conv_raw),
                                   np.array(ref_conv),
                                   rtol=5e-2,
                                   atol=5e-2)
        np.testing.assert_allclose(np.array(pcp_rec),
                                   np.array(ref_rec),
                                   rtol=5e-2,
                                   atol=5e-2)
    else:
        assert np.any(np.array(pcp_pool).view(np.uint8))


def test_pooled_pcp_prefill_fused_projection_matches_non_pcp_baseline():
    """Cover FP8 QKVZ projection, both PCP exchanges, and pooled GDN."""
    pcp_size = 2
    lengths = np.asarray((64, 64), dtype=np.int32)
    interleave_size = 16
    _require_tpu_devices(
        pcp_size,
        f"GDN PCP numerical test requires {pcp_size} TPU devices.",
    )

    n_kq = 2 * pcp_size
    n_v = 2 * pcp_size
    d_k = 128
    d_v = 128
    kernel_size = 4
    num_tokens = int(lengths.sum())
    num_blocks = len(lengths) + 1
    qkv_dim = 2 * n_kq * d_k + n_v * d_v
    hidden_size = qkv_dim + n_v * d_v

    keys = jax.random.split(jax.random.key(0), 9)
    hidden = jax.random.normal(keys[0], (num_tokens, hidden_size),
                               dtype=jnp.bfloat16)
    weight_f32 = 0.02 * jax.random.normal(
        keys[7],
        (hidden_size, hidden_size),
        dtype=jnp.float32,
    )
    qkvz_weight, qkvz_weight_scale = quantized_matmul_util.quantize_tensor(
        weight_f32,
        jnp.float8_e4m3fn,
    )
    qkvz_weight_scale = qkvz_weight_scale[:, 0]
    projected_qkvz = quantized_matmul_util.xla_quantized_matmul(
        hidden,
        qkvz_weight,
        qkvz_weight_scale,
    )
    mixed_qkv = projected_qkvz[:, :qkv_dim]
    ref_z = projected_qkvz[:, qkv_dim:]
    b = jax.random.normal(keys[1], (num_tokens, n_v), dtype=jnp.bfloat16)
    a = jax.random.normal(keys[2], (num_tokens, n_v), dtype=jnp.bfloat16)
    conv_state = jnp.zeros((num_blocks, kernel_size - 1, qkv_dim),
                           dtype=jnp.bfloat16)
    recurrent_state = jnp.zeros((num_blocks, n_v, d_k, d_v), dtype=jnp.float32)
    conv_weight = jax.random.normal(keys[3], (qkv_dim, 1, kernel_size),
                                    dtype=jnp.bfloat16)
    conv_bias = jax.random.normal(keys[4], (qkv_dim, ), dtype=jnp.bfloat16)
    A_log = jax.random.normal(keys[5], (n_v, ), dtype=jnp.float32)
    dt_bias = jax.random.normal(keys[6], (n_v, ), dtype=jnp.float32)

    query_start_loc = jnp.asarray((0, 64, 128), dtype=jnp.int32)
    state_indices = jnp.arange(1, len(lengths) + 1, dtype=jnp.int32)
    distribution = jnp.asarray((0, len(lengths), len(lengths)),
                               dtype=jnp.int32)
    seq_lens = jnp.asarray(lengths, dtype=jnp.int32)
    (_, _), ref_output = gdn_v3_wrapper.fused_conv1d_gdn(
        mixed_qkv,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    local_required_tokens = int(
        _pcp_local_token_counts(lengths, pcp_size, interleave_size).max())
    projection_token_block = 2 * interleave_size
    local_padded_tokens = max(
        8 * projection_token_block,
        (local_required_tokens + projection_token_block - 1) //
        projection_token_block * projection_token_block,
    )
    padded_num_tokens = local_padded_tokens * pcp_size
    token_order, _ = _build_pcp_rank_major_token_order(
        lengths,
        pcp_size,
        interleave_size,
        padded_num_tokens,
    )
    valid = token_order >= 0
    valid_rows = np.where(valid)[0]
    source_rows = token_order[valid]

    def pack_tokens(tensor):
        padded = jnp.pad(tensor, ((0, padded_num_tokens - num_tokens), (0, 0)))
        return jnp.zeros_like(padded).at[valid_rows].set(padded[source_rows])

    packed_hidden = pack_tokens(hidden)
    packed_b = pack_tokens(b)
    packed_a = pack_tokens(a)

    mesh = Mesh(np.asarray(jax.devices()[:pcp_size]), ("pcp", ))

    def shard_tokens(tensor):
        return jax.device_put(tensor, NamedSharding(mesh, P("pcp", None)))

    def replicate(tensor):
        return jax.device_put(tensor, NamedSharding(mesh, P()))

    pool_kernel_block_tokens = 256
    pool_block_tokens = 512
    pool_split = pool_block_tokens // pool_kernel_block_tokens
    local_pool_shape = (
        num_blocks * pool_split,
        pool_kernel_block_tokens,
        1,
        4,
        128,
    )
    global_pool_shape = (pcp_size * local_pool_shape[0], *local_pool_shape[1:])
    pool = jax.device_put(
        jnp.zeros(global_pool_shape, dtype=jnp.float8_e4m3fn),
        NamedSharding(mesh, P("pcp")),
    )

    new_pool, pcp_output, pcp_z = (
        run_jax_gdn_attention_pooled_pcp_prefill_projection(
            shard_tokens(packed_hidden),
            replicate(qkvz_weight),
            replicate(qkvz_weight_scale),
            shard_tokens(packed_b),
            shard_tokens(packed_a),
            pool,
            replicate(conv_weight),
            replicate(conv_bias),
            replicate(A_log),
            replicate(dt_bias),
            replicate(state_indices),
            replicate(query_start_loc),
            replicate(distribution),
            replicate(seq_lens),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            pool_block_tokens=pool_block_tokens,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            mesh=mesh,
        ))

    def restore_request_order(tensor):
        packed = np.asarray(tensor).reshape(padded_num_tokens, -1)
        assert np.isfinite(packed).all()
        np.testing.assert_array_equal(packed[~valid],
                                      np.zeros_like(packed[~valid]))
        restored = np.zeros_like(packed)
        restored[token_order[valid]] = packed[valid]
        return restored[:num_tokens]

    np.testing.assert_allclose(
        restore_request_order(pcp_output),
        np.asarray(ref_output).reshape(num_tokens, -1),
        rtol=5e-2,
        atol=5e-2,
    )
    np.testing.assert_allclose(
        restore_request_order(pcp_z),
        np.asarray(ref_z).reshape(num_tokens, -1),
        rtol=5e-2,
        atol=5e-2,
    )
    assert np.any(np.asarray(new_pool).view(np.uint8))
