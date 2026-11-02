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

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from kernels.gdn.v3.gdn_pool_state_utils import (
    GRANULE,
    WORDS_PER_GRANULE,
    read_states,
    recurrent_granules,
    write_states,
)

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order as _build_pcp_rank_major_token_order,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    pcp_local_token_counts as _pcp_local_token_counts,
)
from vllm_torchtpu.kernels.gdn.head_geometry import derive_gdn_head_geometry
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_v3_wrapper
from vllm_torchtpu.kernels.quantized_matmul import util as quantized_matmul_util
from vllm_torchtpu.layers.core import gdn_attention as gdn_attention_module
from vllm_torchtpu.layers.core.gdn_attention import (
    _derive_pcp_ragged_exchange_descriptors,
    _derive_pcp_rank_major_reorder_indices,
    _exchange_pcp_token_shards_for_head_shards,
    _select_replicated_shard_for_pcp_rank,
    _validate_pcp_ragged_exchange_layout_support,
    run_jax_gdn_attention_pcp_tp_prefill,
    run_jax_gdn_attention_pooled_pcp_prefill,
    run_jax_gdn_attention_pooled_pcp_prefill_projection,
)
from vllm_torchtpu.layers.core.utils import (
    inverse_reorder_for_sharding,
    reorder_concatenated_tensor_for_sharding,
)

pytestmark = pytest.mark.multichip

_GDN_PCP_NUMERICAL_CASES = (
    pytest.param(
        2, (20, 20, 24), 4, (0, 20, 40), None, id="pcp2-multi-request-padding"
    ),
    pytest.param(4, (34, 30), 4, (0, 34), None, id="pcp4-uneven-rank-split"),
    pytest.param(
        4, (20, 20, 24), 4, (0, 20, 40), (7, 22, 3), id="pcp4-chunk-continuation"
    ),
    pytest.param(
        4,
        (13, 11, 40),
        4,
        (0, 13, 24),
        (7, 22, 3),
        id="pcp4-batch-flat-owner-with-history",
    ),
)

_GDN_PCP_DESCRIPTOR_CASES = (
    pytest.param(2, (20, 20, 24), 4, None, id="pcp2-multi-request-padding"),
    pytest.param(4, (34, 30), 4, None, id="pcp4-uneven-rank-split"),
    pytest.param(4, (20, 20, 24), 4, (7, 22, 3), id="pcp4-chunk-continuation"),
    pytest.param(8, (5, 7, 3, 9), 2, (1, 6, 13, 29), id="pcp8-fragmented-descriptors"),
)

_GDN_PCP_FUSED_PROJECTION_CASES = (
    pytest.param(
        2,
        4,
        16,
        2048,
        (17, 15, 32),
        (0, 17, 32),
        (5, 41, 9),
        jnp.float32,
        1280,
        id="gqa-multiple-groups-per-shard",
    ),
    pytest.param(
        8,
        4,
        32,
        2048,
        (64, 64),
        (0, 64),
        None,
        jnp.float32,
        1024,
        id="qwen38-kq-replication-pcp8-spmd",
    ),
    pytest.param(
        2,
        4,
        4,
        2048,
        (1024, 1024),
        (0, 1024),
        None,
        jnp.float32,
        1024,
        id="multi-stage-fresh",
    ),
    pytest.param(
        2, 4, 4, 2048, (64, 64), (0, 64), None, jnp.float32, 512, id="batch-flat-fresh"
    ),
    pytest.param(
        2,
        4,
        4,
        2048,
        (17, 15, 32),
        (0, 17, 32),
        (5, 41, 9),
        jnp.float32,
        512,
        id="batch-flat-owner-with-history",
    ),
    pytest.param(
        2,
        4,
        4,
        2048,
        (17, 15, 32),
        (0, 17, 32),
        (5, 41, 9),
        jnp.bfloat16,
        512,
        id="batch-flat-owner-with-history-bf16-ssm-cache",
    ),
)
# HND (seq-on-lane) is the production layout. With 128 granules per pool
# block slice it is also what leaves non-state padding in the kernel's
# compact state updates, which the NHD shapes above never do.
_GDN_PCP_FUSED_PROJECTION_HND_CASE_IDS = (
    "gqa-multiple-groups-per-shard",
    "batch-flat-owner-with-history-bf16-ssm-cache",
)
_GDN_PCP_FUSED_PROJECTION_CASES = tuple(
    pytest.param(*case.values, "NHD", id=case.id)
    for case in _GDN_PCP_FUSED_PROJECTION_CASES
) + tuple(
    pytest.param(*case.values, "HND", id=f"{case.id}-hnd")
    for case in _GDN_PCP_FUSED_PROJECTION_CASES
    if case.id in _GDN_PCP_FUSED_PROJECTION_HND_CASE_IDS
)


def _query_start_loc_from_lengths(lengths):
    lengths = np.asarray(lengths, dtype=np.int32)
    return jnp.asarray(
        np.concatenate(
            (np.zeros((1,), dtype=np.int32), np.cumsum(lengths, dtype=np.int32))
        )
    )


def _pool_block_words(rank_pool_bytes, num_blocks):
    """uint8 rank pool -> `[num_blocks, words]` in the kernel's word order.

    NumPy twin of `gdn_pool_state_utils.state_words`, kept on host so raw
    bytes never pass through a TPU float path.
    """
    moved = np.ascontiguousarray(np.moveaxis(rank_pool_bytes, -2, -1))
    return moved.view(np.uint32).reshape(num_blocks, -1)


def _pool_from_block_words(words, pool_shape):
    """Inverse of `_pool_block_words`."""
    moved_shape = (*pool_shape[:-2], pool_shape[-1], pool_shape[-2])
    return np.moveaxis(
        np.ascontiguousarray(words).view(np.uint8).reshape(moved_shape), -1, -2
    )


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


def test_replicated_pcp_shard_selection_avoids_axis_index_partition_id(monkeypatch):
    tensor = jnp.arange(2 * 6).reshape(2, 6)
    calls = {}

    def fail_axis_index(axis_name):
        raise AssertionError(
            f"replicated shard selection must not use axis_index({axis_name})"
        )

    original_dynamic_slice = jax.lax.dynamic_slice_in_dim

    def fake_dynamic_slice_in_dim(tensor_arg, start_index, slice_size, axis=0):
        calls["slice"] = (start_index, slice_size, axis)
        return original_dynamic_slice(tensor_arg, start_index, slice_size, axis=axis)

    def fake_all_to_all(tensor_arg, *, axis_name, split_axis, concat_axis, tiled):
        calls["all_to_all"] = (axis_name, split_axis, concat_axis, tiled)
        rank_one_shard = original_dynamic_slice(tensor_arg, 2, 2, axis=1)
        return jnp.concatenate([rank_one_shard] * 3, axis=1)

    monkeypatch.setattr(jax.lax, "axis_index", fail_axis_index)
    monkeypatch.setattr(jax.lax, "dynamic_slice_in_dim", fake_dynamic_slice_in_dim)
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

    expected, _ = _build_pcp_rank_major_token_order(
        lengths,
        pcp_size,
        interleave_size,
        padded_num_tokens,
    )
    actual = _derive_pcp_rank_major_reorder_indices(
        _query_start_loc_from_lengths(lengths),
        jnp.zeros_like(jnp.asarray(lengths)),
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        local_padded_num_tokens=local_padded_num_tokens,
    )

    np.testing.assert_array_equal(np.array(actual), expected.astype(np.int32))


def test_derive_pcp_rank_major_reorder_indices_matches_unaligned_host_order():
    pcp_size = 4
    interleave_size = 4
    lengths = np.array([10, 10], dtype=np.int32)
    padded_num_tokens = (
        int(_pcp_local_token_counts(lengths, pcp_size, interleave_size).max())
        * pcp_size
    )
    local_padded_num_tokens = padded_num_tokens // pcp_size

    expected, _ = _build_pcp_rank_major_token_order(
        lengths,
        pcp_size,
        interleave_size,
        padded_num_tokens,
    )
    actual = _derive_pcp_rank_major_reorder_indices(
        _query_start_loc_from_lengths(lengths),
        jnp.zeros_like(jnp.asarray(lengths)),
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        local_padded_num_tokens=local_padded_num_tokens,
    )

    np.testing.assert_array_equal(np.array(actual), expected.astype(np.int32))


def test_derive_pcp_rank_major_reorder_indices_uses_token_owner_starts():
    pcp_size = 4
    interleave_size = 4
    lengths = np.array([13, 11], dtype=np.int32)
    token_owner_starts = np.array([7, 22], dtype=np.int32)
    padded_num_tokens = (
        int(
            _pcp_local_token_counts(
                lengths,
                pcp_size,
                interleave_size,
                token_owner_starts,
            ).max()
        )
        * pcp_size
    )
    local_padded_num_tokens = padded_num_tokens // pcp_size

    expected, _ = _build_pcp_rank_major_token_order(
        lengths,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_owner_starts,
    )
    actual = _derive_pcp_rank_major_reorder_indices(
        _query_start_loc_from_lengths(lengths),
        jnp.asarray(token_owner_starts),
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        local_padded_num_tokens=local_padded_num_tokens,
    )

    np.testing.assert_array_equal(np.array(actual), expected.astype(np.int32))


def test_derive_pcp_rank_major_reorder_indices_uses_only_owner_coordinates():
    pcp_size = 4
    interleave_size = 4
    q_lens = np.asarray([13, 11], dtype=np.int32)
    # Request 1 begins inside rank 3's [12, 16) interleave chunk, so the
    # batch-flat owner layout exercises a cross-request chunk boundary.
    token_owner_starts = np.asarray([0, 13], dtype=np.int32)
    request_absolute_starts = np.asarray([7, 22], dtype=np.int32)
    local_padded_num_tokens = int(
        _pcp_local_token_counts(
            q_lens,
            pcp_size,
            interleave_size,
            token_owner_starts,
        ).max()
    )
    padded_num_tokens = local_padded_num_tokens * pcp_size

    expected, _ = _build_pcp_rank_major_token_order(
        q_lens,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_owner_starts,
    )
    actual = _derive_pcp_rank_major_reorder_indices(
        _query_start_loc_from_lengths(q_lens),
        jnp.asarray(token_owner_starts),
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        local_padded_num_tokens=local_padded_num_tokens,
    )
    absolute_owned = _derive_pcp_rank_major_reorder_indices(
        _query_start_loc_from_lengths(q_lens),
        jnp.asarray(request_absolute_starts),
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        local_padded_num_tokens=local_padded_num_tokens,
    )

    np.testing.assert_array_equal(np.asarray(actual), expected.astype(np.int32))
    assert not np.array_equal(np.asarray(actual), np.asarray(absolute_owned))


@pytest.mark.parametrize(
    ("pcp_size", "lengths", "interleave_size", "token_start_offsets"),
    _GDN_PCP_DESCRIPTOR_CASES,
)
def test_derive_pcp_ragged_exchange_descriptors_reconstructs_reorder(
    pcp_size, lengths, interleave_size, token_start_offsets
):
    lengths = np.asarray(lengths, dtype=np.int32)
    offsets = (
        None
        if token_start_offsets is None
        else np.asarray(token_start_offsets, dtype=np.int32)
    )
    local_counts = _pcp_local_token_counts(
        lengths,
        pcp_size,
        interleave_size,
        token_start_offsets_per_req=offsets,
    )
    local_padded_num_tokens = int(local_counts.max())
    token_owner_starts = np.zeros_like(lengths) if offsets is None else offsets

    reorder = _derive_pcp_rank_major_reorder_indices(
        _query_start_loc_from_lengths(lengths),
        jnp.asarray(token_owner_starts),
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        local_padded_num_tokens=local_padded_num_tokens,
    )
    input_starts, sizes, output_starts = _derive_pcp_ragged_exchange_descriptors(
        reorder,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        local_padded_num_tokens=local_padded_num_tokens,
        max_num_requests=len(lengths),
    )

    reconstructed = np.full(
        (pcp_size * local_padded_num_tokens,),
        -1,
        dtype=np.int32,
    )
    input_starts = np.asarray(input_starts)
    sizes = np.asarray(sizes)
    output_starts = np.asarray(output_starts)
    for rank in range(pcp_size):
        rank_base = rank * local_padded_num_tokens
        for input_start, size, output_start in zip(
            input_starts[rank], sizes[rank], output_starts[rank]
        ):
            size = int(size)
            if size == 0:
                continue
            dst_start = rank_base + int(input_start)
            reconstructed[dst_start : dst_start + size] = np.arange(
                int(output_start),
                int(output_start) + size,
                dtype=np.int32,
            )

    np.testing.assert_array_equal(reconstructed, np.asarray(reorder))
    assert int(sizes.sum()) == int(lengths.sum())


def test_pcp_ragged_exchange_layout_accepts_tpu_generation_7(monkeypatch):
    monkeypatch.setattr(
        "vllm_torchtpu.layers.core.gdn_attention.pltpu.get_tpu_info",
        lambda: SimpleNamespace(generation=7),
    )

    _validate_pcp_ragged_exchange_layout_support()


@pytest.mark.parametrize("generation", (4, 5, 6, 8))
def test_pcp_ragged_exchange_layout_rejects_unvalidated_tpu_generation(
    monkeypatch, generation
):
    monkeypatch.setattr(
        "vllm_torchtpu.layers.core.gdn_attention.pltpu.get_tpu_info",
        lambda: SimpleNamespace(generation=generation),
    )

    with pytest.raises(
        NotImplementedError,
        match="only validated on TPU generation 7",
    ):
        _validate_pcp_ragged_exchange_layout_support()


def _require_tpu_devices(min_count, reason):
    devices = jax.local_devices()
    if len(devices) < min_count or devices[0].platform != "tpu":
        pytest.skip(reason)
    return devices


@pytest.mark.parametrize(
    (
        "pcp_size",
        "lengths",
        "interleave_size",
        "token_owner_starts",
        "request_absolute_starts",
    ),
    _GDN_PCP_NUMERICAL_CASES,
)
@pytest.mark.parametrize("state_layout", ("split", "unified_pool"))
def test_pcp_prefill_matches_non_pcp_baseline_with_raw_qkv_layout(
    pcp_size,
    lengths,
    interleave_size,
    token_owner_starts,
    request_absolute_starts,
    state_layout,
):
    _require_tpu_devices(
        pcp_size,
        f"GDN PCP numerical test requires {pcp_size} TPU devices.",
    )
    lengths = np.asarray(lengths, dtype=np.int32)
    token_owner_starts = (
        np.zeros_like(lengths)
        if token_owner_starts is None
        else np.asarray(token_owner_starts, dtype=np.int32)
    )
    request_absolute_starts = (
        np.zeros_like(lengths)
        if request_absolute_starts is None
        else np.asarray(request_absolute_starts, dtype=np.int32)
    )
    expected_owner_starts = np.concatenate(
        (np.zeros((1,), dtype=np.int32), np.cumsum(lengths[:-1], dtype=np.int32))
    )
    np.testing.assert_array_equal(token_owner_starts, expected_owner_starts)
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
    keys = jax.random.split(rng, 9)
    mixed_qkv0 = jax.random.normal(keys[0], (num_tokens, dim), dtype=jnp.bfloat16)
    b0 = jax.random.normal(keys[1], (num_tokens, n_v), dtype=jnp.bfloat16)
    a0 = jax.random.normal(keys[2], (num_tokens, n_v), dtype=jnp.bfloat16)
    if state_layout == "split" and np.any(request_absolute_starts):
        # Nonzero history makes treating an absolute continuation as a fresh
        # request numerically observable in both conv and recurrent state.
        conv_state0 = (
            jax.random.normal(
                keys[7], (num_blocks, kernel_size - 1, dim), dtype=jnp.float32
            )
            * 0.01
        )
        rec_state0 = (
            jax.random.normal(keys[8], (num_blocks, n_v, d_k, d_v), dtype=jnp.float32)
            * 0.01
        )
    else:
        conv_state0 = jnp.zeros((num_blocks, kernel_size - 1, dim), dtype=jnp.float32)
        rec_state0 = jnp.zeros((num_blocks, n_v, d_k, d_v), dtype=jnp.float32)
    conv_weight0 = jax.random.normal(keys[3], (dim, 1, kernel_size), dtype=jnp.bfloat16)
    conv_bias0 = jax.random.normal(keys[4], (dim,), dtype=jnp.bfloat16)
    A_log0 = jax.random.normal(keys[5], (n_v,), dtype=jnp.float32)
    dt_bias0 = jax.random.normal(keys[6], (n_v,), dtype=jnp.float32)

    query_start_loc = jnp.asarray(
        np.concatenate(([0], np.cumsum(lengths, dtype=np.int32))),
        dtype=jnp.int32,
    )
    state_indices = jnp.arange(1, len(lengths) + 1, dtype=jnp.int32)
    distribution = jnp.array([0, len(lengths), len(lengths)], dtype=jnp.int32)
    seq_lens = jnp.asarray(request_absolute_starts + lengths, dtype=jnp.int32)

    padded_num_tokens = (
        int(
            _pcp_local_token_counts(
                lengths,
                pcp_size,
                interleave_size,
                token_owner_start_offsets_per_req=token_owner_starts,
            ).max()
        )
        * pcp_size
    )
    token_order, _ = _build_pcp_rank_major_token_order(
        lengths,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_owner_start_offsets_per_req=token_owner_starts,
    )
    valid = token_order >= 0
    pad_tokens = padded_num_tokens - num_tokens
    mixed_qkv_padded = jnp.pad(mixed_qkv0, ((0, pad_tokens), (0, 0)))
    b_padded = jnp.pad(b0, ((0, pad_tokens), (0, 0)))
    a_padded = jnp.pad(a0, ((0, pad_tokens), (0, 0)))
    valid_indices = np.where(valid)[0]
    src_indices = token_order[valid]
    packed_qkv = (
        jnp.zeros_like(mixed_qkv_padded)
        .at[valid_indices]
        .set(mixed_qkv_padded[src_indices])
    )
    packed_b = jnp.zeros_like(b_padded).at[valid_indices].set(b_padded[src_indices])
    packed_a = jnp.zeros_like(a_padded).at[valid_indices].set(a_padded[src_indices])

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
        (conv_state, rec_state),
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

    mesh = Mesh(np.array(jax.devices()[:pcp_size]).reshape((pcp_size,)), ("pcp",))

    def shard_tokens(x):
        return jax.device_put(x, NamedSharding(mesh, P("pcp", None)))

    def shard_conv_state(x):
        rank_major = reorder_concatenated_tensor_for_sharding(
            x, qkv_split_sizes, pcp_size, -1
        )
        return jax.device_put(rank_major, NamedSharding(mesh, P(None, None, "pcp")))

    def shard_rec_state(x):
        return jax.device_put(x, NamedSharding(mesh, P(None, "pcp", None, None)))

    def shard_pool(x):
        return jax.device_put(x, NamedSharding(mesh, P("pcp")))

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
        global_pool_shape = (pcp_size * local_pool_shape[0], *local_pool_shape[1:])
        pcp_pool, pcp_output = run_jax_gdn_attention_pooled_pcp_prefill(
            *common_pcp_args,
            recurrent_state=shard_pool(
                jnp.zeros(global_pool_shape, dtype=jnp.float8_e4m3fn)
            ),
            pool_block_tokens=pool_block_tokens,
            **common_pcp_kwargs,
        )

    pcp_output_np = np.array(pcp_output).reshape(padded_num_tokens, -1)
    pcp_output_seq = np.zeros(
        (padded_num_tokens, pcp_output_np.shape[1]), dtype=pcp_output_np.dtype
    )
    pcp_output_seq[token_order[valid]] = pcp_output_np[valid]

    np.testing.assert_allclose(
        pcp_output_seq[:num_tokens], np.array(ref_output), rtol=5e-2, atol=5e-2
    )
    if state_layout == "split":
        pcp_conv_raw = inverse_reorder_for_sharding(
            pcp_conv, qkv_split_sizes, pcp_size, -1
        )
        np.testing.assert_allclose(
            np.array(pcp_conv_raw), np.array(ref_conv), rtol=5e-2, atol=5e-2
        )
        np.testing.assert_allclose(
            np.array(pcp_rec), np.array(ref_rec), rtol=5e-2, atol=5e-2
        )
    else:
        assert np.any(np.array(pcp_pool).view(np.uint8))


@pytest.mark.parametrize(
    "projection_dtype,weight_scheme",
    [
        pytest.param(jnp.bfloat16, "none", id="bf16"),
        pytest.param(jnp.float8_e4m3fn, "channel", id="fp8-channel"),
        pytest.param(jnp.float8_e4m3fn, "tensor", id="fp8-tensor"),
        pytest.param(jnp.float8_e4m3fn, "block_k", id="fp8-block-k128"),
        pytest.param(jnp.float8_e4m3fn, "block_nk", id="fp8-block-n512-k128"),
    ],
)
@pytest.mark.parametrize(
    (
        "pcp_size",
        "n_kq",
        "n_v",
        "hidden_size",
        "lengths",
        "token_owner_starts",
        "request_absolute_starts",
        "recurrent_state_dtype",
        "pool_block_tokens",
        "kv_layout",
    ),
    _GDN_PCP_FUSED_PROJECTION_CASES,
)
def test_pooled_pcp_prefill_fused_projection_matches_non_pcp_baseline(
    pcp_size,
    n_kq,
    n_v,
    hidden_size,
    lengths,
    token_owner_starts,
    request_absolute_starts,
    recurrent_state_dtype,
    pool_block_tokens,
    kv_layout,
    projection_dtype,
    weight_scheme,
    monkeypatch,
):
    """Cover PCP stage metadata, native BF16/FP8 projection, and pooled GDN."""
    lengths = np.asarray(lengths, dtype=np.int32)
    token_owner_starts = (
        np.zeros_like(lengths)
        if token_owner_starts is None
        else np.asarray(token_owner_starts, dtype=np.int32)
    )
    request_absolute_starts = (
        np.zeros_like(lengths)
        if request_absolute_starts is None
        else np.asarray(request_absolute_starts, dtype=np.int32)
    )
    expected_owner_starts = np.concatenate(
        (np.zeros((1,), dtype=np.int32), np.cumsum(lengths[:-1], dtype=np.int32))
    )
    np.testing.assert_array_equal(token_owner_starts, expected_owner_starts)
    interleave_size = 16
    if token_owner_starts[1] % interleave_size != 0:
        # Request 1 begins inside rank 1's owner chunk [16, 32), while its
        # absolute continuation position is unrelated to that boundary.
        assert token_owner_starts[1] == lengths[0]
        assert not np.array_equal(token_owner_starts, request_absolute_starts)
    _require_tpu_devices(
        pcp_size,
        f"GDN PCP numerical test requires {pcp_size} TPU devices.",
    )

    d_k = 128
    d_v = 128
    kernel_size = 4
    num_tokens = int(lengths.sum())
    num_blocks = len(lengths) + 1
    qkv_dim = 2 * n_kq * d_k + n_v * d_v
    qkvz_dim = qkv_dim + n_v * d_v

    keys = jax.random.split(jax.random.key(0), 9)
    hidden = jax.random.normal(keys[0], (num_tokens, hidden_size), dtype=jnp.bfloat16)
    weight_f32 = 0.02 * jax.random.normal(
        keys[7],
        (qkvz_dim, hidden_size),
        dtype=jnp.float32,
    )
    # `qkvz_weight` is N-major `[n_out, n_in]` to match the projection kernel;
    # `xla_quantized_matmul` takes the canonical (k, n) layout.
    if projection_dtype == jnp.bfloat16:
        qkvz_weight = weight_f32.astype(jnp.bfloat16)
        qkvz_weight_scale = None
        projected_qkvz = jnp.matmul(
            hidden, qkvz_weight.T, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
    elif weight_scheme == "channel":
        qkvz_weight, qkvz_weight_scale = quantized_matmul_util.quantize_tensor(
            weight_f32, projection_dtype
        )
        qkvz_weight_scale = qkvz_weight_scale[:, 0]
        projected_qkvz = quantized_matmul_util.xla_quantized_matmul(
            hidden, qkvz_weight.T, qkvz_weight_scale
        )
    else:
        if weight_scheme == "tensor":
            qkvz_weight_scale = jnp.max(jnp.abs(weight_f32)) / 448.0
            full_scale = jnp.broadcast_to(qkvz_weight_scale, weight_f32.shape)
        else:
            block_k = 128
            block_n = 1 if weight_scheme == "block_k" else 512
            # Strong variation across K blocks catches incorrect global scaling.
            weight_f32 *= jnp.repeat(
                jnp.geomspace(0.25, 4.0, hidden_size // block_k), block_k
            )[None, :]
            grid = weight_f32.reshape(
                qkvz_dim // block_n, block_n, hidden_size // block_k, block_k
            )
            scales = jnp.max(jnp.abs(grid), axis=(1, 3)) / 448.0
            full_scale = jnp.repeat(
                jnp.repeat(scales, block_n, axis=0), block_k, axis=1
            )
            qkvz_weight_scale = (
                scales.T[None, :, None, :] if weight_scheme == "block_k" else scales
            )
        qkvz_weight = jnp.clip(weight_f32 / full_scale, -448.0, 448.0).astype(
            projection_dtype
        )
        x_q, x_scale = quantized_matmul_util.quantize_tensor(hidden, projection_dtype)
        # Independent dense dequantized-weight reference, not the kernel's
        # block accumulation loop or its scale-normalization helper.
        projected_qkvz = (
            jnp.matmul(
                x_q.astype(jnp.float32),
                (qkvz_weight.astype(jnp.float32) * full_scale).T,
                precision=jax.lax.Precision.HIGHEST,
            )
            * x_scale
        ).astype(jnp.bfloat16)
    mixed_qkv = projected_qkvz[:, :qkv_dim]
    ref_z = projected_qkvz[:, qkv_dim:]
    b = jax.random.normal(keys[1], (num_tokens, n_v), dtype=jnp.bfloat16)
    a = jax.random.normal(keys[2], (num_tokens, n_v), dtype=jnp.bfloat16)
    conv_state = jnp.zeros((num_blocks, kernel_size - 1, qkv_dim), dtype=jnp.float32)
    recurrent_state = jnp.zeros((num_blocks, n_v, d_k, d_v), dtype=jnp.float32)
    conv_weight = jax.random.normal(
        keys[3], (qkv_dim, 1, kernel_size), dtype=jnp.bfloat16
    )
    conv_bias = jax.random.normal(keys[4], (qkv_dim,), dtype=jnp.bfloat16)
    A_log = jax.random.normal(keys[5], (n_v,), dtype=jnp.float32)
    dt_bias = jax.random.normal(keys[6], (n_v,), dtype=jnp.float32)

    query_start_loc = _query_start_loc_from_lengths(lengths)
    state_indices = jnp.arange(1, len(lengths) + 1, dtype=jnp.int32)
    distribution = jnp.asarray((0, len(lengths), len(lengths)), dtype=jnp.int32)
    seq_lens = jnp.asarray(request_absolute_starts + lengths, dtype=jnp.int32)
    (ref_conv, ref_recurrent), ref_output = gdn_v3_wrapper.fused_conv1d_gdn(
        mixed_qkv,
        b,
        a,
        (conv_state, recurrent_state),
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
        _pcp_local_token_counts(
            lengths,
            pcp_size,
            interleave_size,
            token_owner_start_offsets_per_req=token_owner_starts,
        ).max()
    )
    projection_token_block = 2 * interleave_size
    local_padded_tokens = max(
        8 * projection_token_block,
        (local_required_tokens + projection_token_block - 1)
        // projection_token_block
        * projection_token_block,
    )
    padded_num_tokens = local_padded_tokens * pcp_size
    token_order, _ = _build_pcp_rank_major_token_order(
        lengths,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_owner_start_offsets_per_req=token_owner_starts,
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

    mesh = Mesh(np.asarray(jax.devices()[:pcp_size]), ("pcp",))

    def shard_tokens(tensor):
        return jax.device_put(tensor, NamedSharding(mesh, P("pcp", None)))

    def replicate(tensor):
        return jax.device_put(tensor, NamedSharding(mesh, P()))

    pool_kernel_block_tokens = 256
    pool_split = pool_block_tokens // pool_kernel_block_tokens
    # The Qwen3.8 K=4/V=32 shape needs four local recurrent heads. Match its
    # production pool capacity while keeping the smaller PCP2 cases unchanged.
    pool_tail_lanes = max(4, 2 * (n_v // pcp_size))
    # The V3 kernel addresses the pool in its compact (packing, lanes) layout,
    # so fp8 rows keep four packed entries in the second-minor axis.
    pool_packing = 4
    # Two spare blocks after the last request block, plus null block 0, are
    # never addressed by state_indices and must come back bit-identical.
    pool_num_blocks = num_blocks + 2
    if kv_layout == "HND":
        monkeypatch.setattr(
            gdn_attention_module, "_is_seq_along_lane_pool", lambda: True
        )
        # (kernel blocks, Kx2, D // P, P, page): two KV heads of head_dim 256
        # in fp8 with the page on lanes, as the Qwen3.5 attention pool.
        local_pool_shape = (
            pool_num_blocks * pool_split,
            2,
            256 // pool_packing,
            pool_packing,
            pool_kernel_block_tokens,
        )
    else:
        local_pool_shape = (
            pool_num_blocks * pool_split,
            pool_kernel_block_tokens,
            pool_tail_lanes // pool_packing,
            pool_packing,
            128,
        )
    global_pool_shape = (pcp_size * local_pool_shape[0], *local_pool_shape[1:])
    touched_blocks = np.zeros((pcp_size, pool_num_blocks), dtype=bool)
    touched_blocks[:, np.asarray(state_indices)] = True
    touched_rows = np.repeat(touched_blocks.reshape(-1), pool_split)
    # Only the state granules at the front of an addressed block belong to
    # GDN; the kernel must leave the rest of the block alone. Keep the state
    # granules zero so the fused numerical case isolates owner-driven
    # stage/DMA geometry. Nonzero absolute-history state loading is covered by
    # the raw-QKV split-state case above; the metadata unit test separately
    # checks that these absolute starts set the initial-state flags.
    pool_geometry = derive_gdn_head_geometry(n_kq, n_v, pcp_size)
    conv_state_bytes = (kernel_size - 1) * pool_geometry.local_conv_dim(d_k, d_v) * 4
    state_words = WORDS_PER_GRANULE * (
        recurrent_granules(
            (pool_geometry.local_num_v_heads, d_k, d_v), recurrent_state_dtype
        )
        + (conv_state_bytes + GRANULE - 1) // GRANULE
    )
    initial_pool_bytes = np.random.default_rng(0).integers(
        0, 256, size=global_pool_shape, dtype=np.uint8
    )
    for rank_pool in initial_pool_bytes.reshape(pcp_size, *local_pool_shape):
        words = _pool_block_words(rank_pool, pool_num_blocks)
        words[np.asarray(state_indices), :state_words] = 0
        rank_pool[...] = _pool_from_block_words(words, local_pool_shape)
    pool = jax.device_put(
        jax.lax.bitcast_convert_type(
            jnp.asarray(initial_pool_bytes), jnp.float8_e4m3fn
        ),
        NamedSharding(mesh, P("pcp")),
    )

    new_pool, pcp_output, pcp_z = run_jax_gdn_attention_pooled_pcp_prefill_projection(
        shard_tokens(packed_hidden),
        replicate(qkvz_weight),
        replicate(qkvz_weight_scale) if qkvz_weight_scale is not None else None,
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
        recurrent_state_dtype=recurrent_state_dtype,
    )
    # Compare raw bytes: fp8 NaN patterns must not go through a float path.
    # Reinterpret on host: a standalone fp8 -> uint8 bitcast on TPU (JAX
    # 0.11.2 / libtpu 0.0.50) canonicalizes NaN 0xFF to 0x7F.
    new_pool_bytes = np.asarray(new_pool).view(np.uint8)
    np.testing.assert_array_equal(
        new_pool_bytes[~touched_rows], initial_pool_bytes[~touched_rows]
    )
    for new_rank_pool, initial_rank_pool in zip(
        new_pool_bytes.reshape(pcp_size, *local_pool_shape),
        initial_pool_bytes.reshape(pcp_size, *local_pool_shape),
        strict=True,
    ):
        addressed = np.asarray(state_indices)
        np.testing.assert_array_equal(
            _pool_block_words(new_rank_pool, pool_num_blocks)[addressed, state_words:],
            _pool_block_words(initial_rank_pool, pool_num_blocks)[
                addressed, state_words:
            ],
        )

    def restore_request_order(tensor):
        packed = np.asarray(tensor).reshape(padded_num_tokens, -1)
        assert np.isfinite(packed).all()
        np.testing.assert_array_equal(packed[~valid], np.zeros_like(packed[~valid]))
        restored = np.zeros_like(packed)
        restored[token_order[valid]] = packed[valid]
        return restored[:num_tokens]

    tolerance = 2e-2 if projection_dtype == jnp.bfloat16 else 5e-2
    np.testing.assert_allclose(
        restore_request_order(pcp_output),
        np.asarray(ref_output).reshape(num_tokens, -1),
        rtol=tolerance,
        atol=tolerance,
    )
    geometry = derive_gdn_head_geometry(n_kq, n_v, pcp_size)
    local_n_kq = geometry.local_num_kq_heads
    local_n_v = geometry.local_num_v_heads
    local_conv_dim = geometry.local_conv_dim(d_k, d_v)
    # Each rank's pool shard holds its local heads' states in the V3 layout.
    # Decode the raw bytes copied out above, so no float path sees them.
    local_states = [
        read_states(
            local_pool_bytes,
            state_indices,
            split=pool_split,
            recurrent_shape=(local_n_v, d_k, d_v),
            conv_shape=(kernel_size - 1, local_conv_dim),
            recurrent_dtype=recurrent_state_dtype,
        )
        for local_pool_bytes in new_pool_bytes.reshape(pcp_size, *local_pool_shape)
    ]
    pcp_recurrent = np.concatenate([np.asarray(ssm) for ssm, _ in local_states])
    pcp_conv = np.concatenate([np.asarray(conv) for _, conv in local_states])
    q, k, v = np.split(np.asarray(ref_conv)[1:], [n_kq * d_k, 2 * n_kq * d_k], -1)
    local_key_dim = local_n_kq * d_k
    local_value_dim = local_n_v * d_v
    expected_conv_by_rank = []
    for rank in range(pcp_size):
        kq_start = geometry.kq_shard_index(rank) * local_key_dim
        v_start = rank * local_value_dim
        expected_conv_by_rank.append(
            np.concatenate(
                (
                    q[..., kq_start : kq_start + local_key_dim],
                    k[..., kq_start : kq_start + local_key_dim],
                    v[..., v_start : v_start + local_value_dim],
                ),
                axis=-1,
            )
        )
    expected_conv = np.concatenate(expected_conv_by_rank)
    expected_recurrent = np.concatenate(
        np.split(np.asarray(ref_recurrent)[1:], pcp_size, axis=1)
    )
    np.testing.assert_allclose(
        np.asarray(pcp_conv), expected_conv, rtol=tolerance, atol=tolerance
    )
    if geometry.kq_replication_factor > 1:
        actual_conv_by_rank = np.asarray(pcp_conv).reshape(
            pcp_size, len(lengths), kernel_size - 1, local_conv_dim
        )
        for rank in range(0, pcp_size, geometry.kq_replication_factor):
            replicas = actual_conv_by_rank[
                rank : rank + geometry.kq_replication_factor, ..., : 2 * local_key_dim
            ]
            for replica in replicas[1:]:
                np.testing.assert_allclose(
                    replica, replicas[0], rtol=tolerance, atol=tolerance
                )
    np.testing.assert_allclose(
        np.asarray(pcp_recurrent), expected_recurrent, rtol=tolerance, atol=tolerance
    )
    actual_z = restore_request_order(pcp_z)
    expected_z = np.asarray(ref_z).reshape(num_tokens, -1)
    np.testing.assert_allclose(
        actual_z,
        expected_z,
        rtol=tolerance,
        atol=tolerance,
    )


@dataclasses.dataclass(frozen=True)
class _PrefillCkptPoolConfig:
    """Head counts, PCP chunk, pool and state layout of a checkpoint case."""

    n_kq: int
    n_v: int
    interleave_size: int
    pool_block_tokens: int
    # One kernel block (page) of a rank's pool, without the block dim.
    page_shape: tuple[int, ...]
    recurrent_state_dtype: jnp.dtype
    kv_layout: str

    @property
    def page_tokens(self) -> int:
        # HND keeps the page's tokens on the minormost (lane) axis.
        return self.page_shape[-1] if self.kv_layout == "HND" else self.page_shape[0]


_PREFILL_CKPT_NHD_POOL = _PrefillCkptPoolConfig(
    n_kq=4,
    n_v=4,
    interleave_size=64,
    pool_block_tokens=512,
    page_shape=(256, 1, 4, 128),
    recurrent_state_dtype=jnp.float32,
    kv_layout="NHD",
)
# Matches the prefill engine pool layout of qwen38-disagg-fp8-correctness
# (interleave_size=128, block_size=384, fp8 HND pages, bfloat16 SSM state).
# Scales Qwen3.5-397B's PCP8 head count (16 KQ, 64 V) down to PCP2 (4 KQ,
# 16 V) so each rank still holds 2 KQ and 8 V heads (one physical Z tile).
# Its 64-granule pool slices leave non-state padding in the compact final
# and checkpoint updates.
_PREFILL_CKPT_PRODUCTION_POOL = _PrefillCkptPoolConfig(
    n_kq=4,
    n_v=16,
    interleave_size=128,
    pool_block_tokens=384,
    page_shape=(8, 64, 4, 128),
    recurrent_state_dtype=jnp.bfloat16,
    kv_layout="HND",
)


@pytest.mark.parametrize(
    ("lengths", "ckpt_offsets", "request_absolute_starts", "pool_config"),
    [
        # Request starts are multiples of 64, so each checkpoint row ends a
        # tile and the export follows that tile.
        pytest.param(
            (192, 256, 40),
            (128, 192, 0),
            None,
            _PREFILL_CKPT_NHD_POOL,
            id="aligned-query-start",
        ),
        # Request 1 starts at row 100. Its checkpoint row 292 falls inside
        # the tile [256, 320), which is preceded by a shadow tile of 36 rows.
        pytest.param(
            (100, 300, 40),
            (64, 192, 0),
            None,
            _PREFILL_CKPT_NHD_POOL,
            id="unaligned-query-start",
        ),
        # Requests 1 and 2 continue earlier chunks and read their incoming
        # state from the pool. Request 1 exports after the tile ending at row
        # 384 and request 2 starts two tiles later, on the same pipeline slot.
        pytest.param(
            (192, 256, 40),
            (128, 192, 0),
            (0, 6144, 7),
            _PREFILL_CKPT_NHD_POOL,
            id="aligned-continuations",
        ),
        # Request 1 continues an earlier chunk and checkpoints on a shadow
        # tile before its last tile [256, 320). Request 2 continues another
        # chunk and starts right after it, on the shadow's pipeline slot.
        pytest.param(
            (100, 220, 60),
            (64, 192, 0),
            (0, 3072, 5),
            _PREFILL_CKPT_NHD_POOL,
            id="unaligned-continuations",
        ),
        # Production layout cases. PCP chunks are 128 rows per rank, and
        # tiles are 64 rows from the start of a request's part of a chunk.
        # Request 0 checkpoints at the end of its first tile [0, 64). Request
        # 1 starts at row 100, so its tiles start at rows 100, 128, 192, 256
        # and so on; its checkpoint row 292 falls inside the tile [256, 320),
        # which a shadow tile of 36 rows precedes.
        pytest.param(
            (100, 300, 40),
            (64, 192, 0),
            None,
            _PREFILL_CKPT_PRODUCTION_POOL,
            id="production-unaligned-query-start",
        ),
        # Every request continues an earlier chunk. Request 0 checkpoints at
        # row 16 inside its first tile, so its shadow tile and the real tile
        # both load the incoming state. Request 1 checkpoints on the shadow
        # tile [256, 292) as above.
        pytest.param(
            (100, 300, 40),
            (16, 192, 0),
            (2048, 3072, 5),
            _PREFILL_CKPT_PRODUCTION_POOL,
            id="production-unaligned-continuations",
        ),
        # Request 0 checkpoints at the end of its first tile [0, 64). Request 1
        # starts at row 100 and checkpoints at row 116 inside its first tile
        # [100, 128), so the shadow tile [100, 116) immediately follows request
        # 0's last tile, exercising back-to-back state copy-out and copy-in.
        pytest.param(
            (100, 300, 40),
            (64, 16, 0),
            (2048, 3072, 5),
            _PREFILL_CKPT_PRODUCTION_POOL,
            id="production-first-tile-shadow-after-request",
        ),
    ],
)
def test_pooled_pcp_prefill_fused_projection_writes_prefill_checkpoint(
    lengths, ckpt_offsets, request_absolute_starts, pool_config, monkeypatch
):
    """The checkpoint block matches a reference run truncated at the row."""
    pcp_size = 2
    n_kq = pool_config.n_kq
    n_v = pool_config.n_v
    hidden_size = 2048
    interleave_size = pool_config.interleave_size
    pool_block_tokens = pool_config.pool_block_tokens
    recurrent_state_dtype = pool_config.recurrent_state_dtype
    if pool_config.kv_layout == "HND":
        monkeypatch.setattr(
            gdn_attention_module, "_is_seq_along_lane_pool", lambda: True
        )
    _require_tpu_devices(
        pcp_size,
        f"GDN PCP numerical test requires {pcp_size} TPU devices.",
    )

    lengths = np.asarray(lengths, dtype=np.int32)
    ckpt_offsets = np.asarray(ckpt_offsets, dtype=np.int32)
    request_absolute_starts = (
        np.zeros_like(lengths)
        if request_absolute_starts is None
        else np.asarray(request_absolute_starts, dtype=np.int32)
    )
    is_continuation = request_absolute_starts > 0
    num_seqs = len(lengths)
    d_k = 128
    d_v = 128
    kernel_size = 4
    num_tokens = int(lengths.sum())
    state_indices = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)
    ckpt_indices = jnp.arange(num_seqs + 1, 2 * num_seqs + 1, dtype=jnp.int32)
    # Null block 0 and two spare blocks after the checkpoint blocks are never
    # addressed and must come back bit-identical.
    num_blocks = 2 * num_seqs + 3
    qkv_dim = 2 * n_kq * d_k + n_v * d_v
    qkvz_dim = qkv_dim + n_v * d_v

    geometry = derive_gdn_head_geometry(n_kq, n_v, pcp_size)
    local_key_dim = geometry.local_num_kq_heads * d_k
    local_value_dim = geometry.local_num_v_heads * d_v
    local_conv_dim = geometry.local_conv_dim(d_k, d_v)
    recurrent_shape = (geometry.local_num_v_heads, d_k, d_v)
    conv_shape = (kernel_size - 1, local_conv_dim)
    pool_split = pool_block_tokens // pool_config.page_tokens
    local_pool_shape = (num_blocks * pool_split, *pool_config.page_shape)
    global_pool_shape = (pcp_size * local_pool_shape[0], *local_pool_shape[1:])
    conv_state_bytes = (kernel_size - 1) * local_conv_dim * 4
    state_words = WORDS_PER_GRANULE * (
        recurrent_granules(recurrent_shape, recurrent_state_dtype)
        + (conv_state_bytes + GRANULE - 1) // GRANULE
    )

    keys = jax.random.split(jax.random.key(0), 10)
    hidden = jax.random.normal(keys[0], (num_tokens, hidden_size), dtype=jnp.bfloat16)
    qkvz_weight = (
        0.02 * jax.random.normal(keys[7], (qkvz_dim, hidden_size), dtype=jnp.float32)
    ).astype(jnp.bfloat16)
    projected_qkvz = jnp.matmul(
        hidden, qkvz_weight.T, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    mixed_qkv = projected_qkvz[:, :qkv_dim]
    b = jax.random.normal(keys[1], (num_tokens, n_v), dtype=jnp.bfloat16)
    a = jax.random.normal(keys[2], (num_tokens, n_v), dtype=jnp.bfloat16)
    conv_weight = jax.random.normal(
        keys[3], (qkv_dim, 1, kernel_size), dtype=jnp.bfloat16
    )
    conv_bias = jax.random.normal(keys[4], (qkv_dim,), dtype=jnp.bfloat16)
    # A slow decay keeps the incoming state visible in the checkpoints and
    # final states of the continued requests, not only in their first outputs.
    A_log = jax.random.normal(keys[5], (n_v,), dtype=jnp.float32) - 4.0
    dt_bias = jax.random.normal(keys[6], (n_v,), dtype=jnp.float32)
    # Every state block starts nonzero; only the continued requests read it.
    initial_conv = 0.5 * jax.random.normal(
        keys[8], (num_seqs, kernel_size - 1, qkv_dim), dtype=jnp.float32
    )
    # The pool stores the SSM state in its cache dtype, so the reference
    # starts from the same rounded values.
    initial_recurrent = (
        (0.1 * jax.random.normal(keys[9], (num_seqs, n_v, d_k, d_v), dtype=jnp.float32))
        .astype(recurrent_state_dtype)
        .astype(jnp.float32)
    )
    query_start_loc = _query_start_loc_from_lengths(lengths)
    distribution = jnp.asarray((0, num_seqs, num_seqs), dtype=jnp.int32)
    seq_lens = jnp.asarray(request_absolute_starts + lengths, dtype=jnp.int32)

    def non_pcp_reference(seqs, ends, conv, recurrent):
        """Final states and outputs of the given requests over [start, end)."""
        starts = np.asarray(query_start_loc)[seqs]
        rows = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
        ref_lengths = np.asarray(ends - starts, np.int32)
        num_ref_seqs = len(seqs)
        (conv, recurrent), output = gdn_v3_wrapper.fused_conv1d_gdn(
            mixed_qkv[rows],
            b[rows],
            a[rows],
            (
                jnp.concatenate((jnp.zeros_like(conv[:1]), conv)),
                jnp.concatenate((jnp.zeros_like(recurrent[:1]), recurrent)),
            ),
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            _query_start_loc_from_lengths(ref_lengths),
            jnp.arange(1, num_ref_seqs + 1, dtype=jnp.int32),
            jnp.asarray((0, num_ref_seqs, num_ref_seqs), dtype=jnp.int32),
            jnp.asarray(request_absolute_starts[seqs] + ref_lengths),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )
        return np.asarray(conv)[1:], np.asarray(recurrent)[1:], np.asarray(output)

    all_seqs = np.arange(num_seqs)
    seq_ends = np.asarray(query_start_loc)[1:]
    ckpt_rows = np.nonzero(ckpt_offsets)[0]
    ckpt_ends = np.asarray(query_start_loc)[ckpt_rows] + ckpt_offsets[ckpt_rows]
    ref_conv, ref_recurrent, ref_output = non_pcp_reference(
        all_seqs, seq_ends, initial_conv, initial_recurrent
    )
    ckpt_conv, ckpt_recurrent, _ = non_pcp_reference(
        ckpt_rows, ckpt_ends, initial_conv[ckpt_rows], initial_recurrent[ckpt_rows]
    )

    tolerance = 2e-2
    if np.any(is_continuation):
        # The incoming state must be observable, or reading the wrong one
        # would pass the comparisons below.
        _, fresh_recurrent, fresh_output = non_pcp_reference(
            all_seqs,
            seq_ends,
            jnp.zeros_like(initial_conv),
            jnp.zeros_like(initial_recurrent),
        )
        continued_rows = np.repeat(is_continuation, lengths)
        assert not np.allclose(
            fresh_output.reshape(num_tokens, -1)[continued_rows],
            ref_output.reshape(num_tokens, -1)[continued_rows],
            atol=tolerance,
        )
        assert not np.allclose(
            fresh_recurrent[is_continuation],
            ref_recurrent[is_continuation],
            atol=tolerance,
        )
        continued_ckpt = is_continuation[ckpt_rows]
        _, fresh_ckpt_recurrent, _ = non_pcp_reference(
            ckpt_rows[continued_ckpt],
            ckpt_ends[continued_ckpt],
            jnp.zeros_like(initial_conv[ckpt_rows[continued_ckpt]]),
            jnp.zeros_like(initial_recurrent[ckpt_rows[continued_ckpt]]),
        )
        assert not np.allclose(
            fresh_ckpt_recurrent, ckpt_recurrent[continued_ckpt], atol=tolerance
        )

    token_owner_starts = np.asarray(query_start_loc)[:-1]
    local_required_tokens = int(
        _pcp_local_token_counts(
            lengths,
            pcp_size,
            interleave_size,
            token_owner_start_offsets_per_req=token_owner_starts,
        ).max()
    )
    projection_token_block = 2 * interleave_size
    local_padded_tokens = max(
        8 * projection_token_block,
        (local_required_tokens + projection_token_block - 1)
        // projection_token_block
        * projection_token_block,
    )
    padded_num_tokens = local_padded_tokens * pcp_size
    token_order, _ = _build_pcp_rank_major_token_order(
        lengths,
        pcp_size,
        interleave_size,
        padded_num_tokens,
        token_owner_start_offsets_per_req=token_owner_starts,
    )
    valid = token_order >= 0
    valid_rows = np.where(valid)[0]
    source_rows = token_order[valid]

    def pack_tokens(tensor):
        padded = jnp.pad(tensor, ((0, padded_num_tokens - num_tokens), (0, 0)))
        return jnp.zeros_like(padded).at[valid_rows].set(padded[source_rows])

    mesh = Mesh(np.asarray(jax.devices()[:pcp_size]), ("pcp",))

    def shard_tokens(tensor):
        return jax.device_put(tensor, NamedSharding(mesh, P("pcp", None)))

    def replicate(tensor):
        return jax.device_put(tensor, NamedSharding(mesh, P()))

    def rank_local_reference(conv, recurrent):
        """Stacks each rank's conv channels and recurrent heads, rank-major."""
        q, k, v = np.split(np.asarray(conv), [n_kq * d_k, 2 * n_kq * d_k], -1)
        conv_by_rank = []
        for rank in range(pcp_size):
            kq_start = geometry.kq_shard_index(rank) * local_key_dim
            v_start = rank * local_value_dim
            conv_by_rank.append(
                np.concatenate(
                    (
                        q[..., kq_start : kq_start + local_key_dim],
                        k[..., kq_start : kq_start + local_key_dim],
                        v[..., v_start : v_start + local_value_dim],
                    ),
                    axis=-1,
                )
            )
        return (
            np.concatenate(conv_by_rank),
            np.concatenate(np.split(np.asarray(recurrent), pcp_size, axis=1)),
        )

    def rank_pools(pool_bytes):
        return pool_bytes.reshape(pcp_size, *local_pool_shape)

    def read_rank_states(pool_bytes, block_ids):
        """Rank-major (conv, recurrent) of `block_ids`, decoded from raw bytes."""
        states = [
            read_states(
                rank_pool,
                jnp.asarray(block_ids),
                split=pool_split,
                recurrent_shape=recurrent_shape,
                conv_shape=conv_shape,
                recurrent_dtype=recurrent_state_dtype,
            )
            for rank_pool in rank_pools(pool_bytes)
        ]
        return (
            np.concatenate([np.asarray(conv) for _, conv in states]),
            np.concatenate([np.asarray(ssm, dtype=np.float32) for ssm, _ in states]),
        )

    # Every pool byte starts as random garbage, so a stray write of any value
    # shows in the comparison of untouched bytes at the end. The state blocks
    # then get each rank's slice of the initial states.
    initial_pool_bytes = np.random.default_rng(0).integers(
        0, 256, size=global_pool_shape, dtype=np.uint8
    )
    seed_conv, seed_recurrent = rank_local_reference(initial_conv, initial_recurrent)
    for rank, rank_pool in enumerate(rank_pools(initial_pool_bytes)):
        seqs = slice(rank * num_seqs, (rank + 1) * num_seqs)
        rank_pool[...] = np.asarray(
            write_states(
                rank_pool,
                seed_recurrent[seqs],
                seed_conv[seqs],
                state_indices,
                split=pool_split,
                recurrent_dtype=recurrent_state_dtype,
            )
        )
    pool = jax.device_put(
        jax.lax.bitcast_convert_type(
            jnp.asarray(initial_pool_bytes), jnp.float8_e4m3fn
        ),
        NamedSharding(mesh, P("pcp")),
    )

    new_pool, pcp_output, _ = run_jax_gdn_attention_pooled_pcp_prefill_projection(
        shard_tokens(pack_tokens(hidden)),
        replicate(qkvz_weight),
        None,
        shard_tokens(pack_tokens(b)),
        shard_tokens(pack_tokens(a)),
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
        recurrent_state_dtype=recurrent_state_dtype,
        prefill_ckpt_offsets=replicate(jnp.asarray(ckpt_offsets)),
        prefill_ckpt_indices=replicate(ckpt_indices),
    )

    # A shadow tile must leave the real tile's output and carry untouched,
    # and an export must leave the next request's incoming state untouched.
    packed = np.asarray(pcp_output).reshape(padded_num_tokens, -1)
    restored = np.zeros_like(packed)
    restored[token_order[valid]] = packed[valid]
    np.testing.assert_allclose(
        restored[:num_tokens],
        ref_output.reshape(num_tokens, -1),
        rtol=tolerance,
        atol=tolerance,
    )

    # Reinterpret on host: raw fp8 bytes must not pass through a float path.
    new_pool_bytes = np.asarray(new_pool).view(np.uint8)
    final_conv, final_recurrent = read_rank_states(new_pool_bytes, state_indices)
    expected_conv, expected_recurrent = rank_local_reference(ref_conv, ref_recurrent)
    np.testing.assert_allclose(
        final_conv, expected_conv, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        final_recurrent, expected_recurrent, rtol=tolerance, atol=tolerance
    )

    ckpt_blocks = np.asarray(ckpt_indices)[ckpt_rows]
    ckpt_conv_actual, ckpt_recurrent_actual = read_rank_states(
        new_pool_bytes, ckpt_blocks
    )
    expected_conv, expected_recurrent = rank_local_reference(ckpt_conv, ckpt_recurrent)
    np.testing.assert_allclose(
        ckpt_conv_actual, expected_conv, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        ckpt_recurrent_actual, expected_recurrent, rtol=tolerance, atol=tolerance
    )
    # The checkpoint differs from the final state, so the comparison above
    # cannot pass by reading the wrong block.
    assert not np.allclose(ckpt_recurrent, ref_recurrent[ckpt_rows], atol=tolerance)

    # Only the state granules at the front of the final-state and checkpoint
    # blocks may change. Every other block, including the checkpoint blocks
    # of requests without a checkpoint, and every granule past the state in
    # a written block keep their initial bytes.
    written_blocks = np.concatenate((np.asarray(state_indices), ckpt_blocks))
    touched_blocks = np.zeros((pcp_size, num_blocks), dtype=bool)
    touched_blocks[:, written_blocks] = True
    touched_rows = np.repeat(touched_blocks.reshape(-1), pool_split)
    np.testing.assert_array_equal(
        new_pool_bytes[~touched_rows], initial_pool_bytes[~touched_rows]
    )
    for new_rank_pool, initial_rank_pool in zip(
        rank_pools(new_pool_bytes), rank_pools(initial_pool_bytes), strict=True
    ):
        np.testing.assert_array_equal(
            _pool_block_words(new_rank_pool, num_blocks)[written_blocks, state_words:],
            _pool_block_words(initial_rank_pool, num_blocks)[
                written_blocks, state_words:
            ],
        )
