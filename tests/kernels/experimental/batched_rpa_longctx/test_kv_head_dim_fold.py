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
"""Deviceless arithmetic + byte-placement checks for `TPU_RPA_FOLD_KV_HEAD_DIM`.

The fold reinterprets the packing slots that `num_kv_heads * 2` pads out as
extra head_dim segments. Nothing here needs a TPU: the shape/stride math is
integer arithmetic over `num_lanes`, and the pack/unpack round-trip is the
kernel's byte de-interleave replayed in numpy.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import (configs,
                                                                    utils,
                                                                    wrapper)

_NUM_LANES = 128
_NUM_SUBLANES = 8


class _FakeTpuInfo:
    num_lanes = _NUM_LANES
    num_sublanes = _NUM_SUBLANES
    mxu_column_size = 128
    vmem_capacity_bytes = 128 * 1024 * 1024
    smem_capacity_bytes = 1024 * 1024


@pytest.fixture(autouse=True)
def _deviceless_tpu_info(monkeypatch):
    """`num_lanes` is the only hardware fact the fold arithmetic reads."""
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: _FakeTpuInfo())


# num_kv_heads, head_dim, kv_packing, expected fold, why.
_FOLD_CASES = [
    (1, 256, 4, 2, "TP8 fp8: 2 live slots of 4, head_dim spans 2 registers"),
    (2, 256, 4, 1, "DP: 4 live slots of 4, nothing to reclaim"),
    (4, 256, 4, 1, "8 live slots of 4, nothing to reclaim"),
    (1, 256, 2, 1, "bf16: 2 live slots of 2, nothing to reclaim"),
    (1, 128, 4, 1, "head_dim is one register, nothing to cut"),
    (1, 512, 4, 1, "a word would hold K without its V"),
    (3, 256, 4, 2, "6 live slots of 4: folds to 3 exact words per token"),
]


@pytest.mark.parametrize("num_kv_heads, head_dim, kv_packing, expected, why",
                         _FOLD_CASES)
def test_fold_factor(num_kv_heads, head_dim, kv_packing, expected, why):
    assert utils.kv_head_dim_fold(num_kv_heads, head_dim, kv_packing,
                                  _NUM_LANES) == expected, why


def _shape(num_kv_heads, head_dim, kv_packing, fold_kv_head_dim):
    return wrapper.get_kv_cache_shape(
        8,
        256,
        num_kv_heads,
        head_dim,
        kv_packing=kv_packing,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        fold_kv_head_dim=fold_kv_head_dim,
    )


def test_cache_shape_halves_the_page_under_tp8():
    """1 KV head of an fp8 cache: 512 B of live K+V used to cost 1024 B."""
    unfolded = _shape(1, 256, 4, False)
    folded = _shape(1, 256, 4, True)
    assert unfolded == (8, 256, 1, 4, 256)
    assert folded == (8, 256, 1, 4, 128)

    live_bytes_per_token = 1 * 2 * 256
    assert np.prod(folded[2:]) == live_bytes_per_token
    assert np.prod(unfolded[2:]) == 2 * live_bytes_per_token
    # The token axis and the packed-word count are untouched, so everything
    # addressing the pool by token (the GDN pool adapters) and everything
    # sharding the cache by word (`P(None, None, "model", None, None)`) sees
    # the same geometry.
    assert folded[:3] == unfolded[:3]


@pytest.mark.parametrize("num_kv_heads, head_dim, kv_packing", [(2, 256, 4),
                                                                (4, 256, 4),
                                                                (1, 256, 2),
                                                                (1, 128, 4)])
def test_cache_shape_is_a_no_op_without_padding(num_kv_heads, head_dim,
                                                kv_packing):
    """No second flag for DP: a rank with no padding keeps its layout."""
    assert (_shape(num_kv_heads, head_dim, kv_packing,
                   True) == _shape(num_kv_heads, head_dim, kv_packing, False))


def _cfgs(num_kv_heads,
          head_dim,
          fold_kv_head_dim,
          kv_dtype=jnp.float8_e4m3fn):
    return configs.RpaConfigs(
        block=configs.BlockSizes(bq_sz=1,
                                 bq_c_sz=1,
                                 bkv_sz=256,
                                 batch_size=8,
                                 n_buffer=3),
        model=configs.ModelConfigs(num_q_heads=32,
                                   num_kv_heads=num_kv_heads,
                                   head_dim=head_dim,
                                   mask_value=-1e30),
        serve=configs.ServingConfigs(num_seqs=8,
                                     page_size=256,
                                     total_q_tokens=8,
                                     num_page_indices=64,
                                     dtype_q=jnp.dtype(jnp.bfloat16),
                                     dtype_kv=jnp.dtype(kv_dtype),
                                     dtype_out=jnp.dtype(jnp.bfloat16),
                                     fold_kv_head_dim=fold_kv_head_dim),
        mode=configs.RpaCase.DECODE,
        vmem_limit_bytes=64 * 1024 * 1024,
    )


def test_strides_stay_paired_with_the_lane_width():
    """The whole set moves together, or the kernel reads the wrong stride."""
    unfolded = _cfgs(1, 256, False)
    folded = _cfgs(1, 256, True)

    assert (unfolded.kv_head_dim_fold, folded.kv_head_dim_fold) == (1, 2)
    # One packed word per token either way -- the fold buys lane width, not
    # words -- so the VMEM tile and the HBM tile keep the same row count.
    assert folded.bkv_stride == unfolded.bkv_stride == 1
    assert folded.kv_hbm_stride == unfolded.kv_hbm_stride == 1
    assert folded.kv_cache_lane_dim == 128
    assert unfolded.kv_cache_lane_dim == 256
    # The compute-side head_dim is not folded: q is sliced to it and the
    # accumulator is that wide.
    assert folded.aligned_kv_head_dim == unfolded.aligned_kv_head_dim == 256
    assert folded.acc_scratch_shape[-1] == 256

    assert folded.kv_vmem_shape == (8, 256, 1, 4, 128)
    assert unfolded.kv_vmem_shape == (8, 256, 1, 4, 256)
    # Drives the schedule's DMA-completion lane counts; a stale value here
    # waits on the wrong number of 512-byte chunks.
    assert folded.kv_bytes_per_token == 512
    assert unfolded.kv_bytes_per_token == 1024

    # What `validate_inputs` will demand of the cache tensor, which is what
    # `wrapper.get_kv_cache_shape` hands vLLM.
    for cfgs, expected in ((folded, 128), (unfolded, 256)):
        assert (cfgs.serve.page_size,
                cfgs.aligned_num_kv_heads_x2 // cfgs.serve.packing_kv,
                cfgs.serve.packing_kv, cfgs.kv_cache_lane_dim) == (256, 1, 4,
                                                                   expected)


def test_dp_rank_strides_are_untouched():
    unfolded = _cfgs(2, 256, False)
    folded = _cfgs(2, 256, True)
    assert folded.kv_head_dim_fold == 1
    assert folded.kv_vmem_shape == unfolded.kv_vmem_shape
    assert folded.kv_bytes_per_token == unfolded.kv_bytes_per_token == 1024


# ---------------------------------------------------------------------------
# Byte placement: pack as `prepare_inputs` does, unpack as the kernel does.
# ---------------------------------------------------------------------------


def _pack_new_kv(k, v, kv_packing, fold):
    """`prepare_inputs`' HEAD_ALONG_SUBLANE branch, on uint8 payloads."""
    total_tokens, num_kv_heads, head_dim = k.shape
    heads_x2 = num_kv_heads * 2
    aligned_head_dim = -(-head_dim // _NUM_LANES) * _NUM_LANES
    rows = utils.align_to(heads_x2 * fold, kv_packing)
    packed = np.pad(
        np.concatenate([k, v], axis=-1).reshape(total_tokens, heads_x2,
                                                head_dim),
        ((0, 0), (0, rows // fold - heads_x2),
         (0, aligned_head_dim - head_dim)),
    )
    return packed.reshape(total_tokens, rows // kv_packing, kv_packing,
                          aligned_head_dim // fold)


def _unpack_cache(cache, fold):
    """`strided_load_bkv`, replayed: u32 view -> byte split -> re-pair.

    The u32 view and `convert_to_target_bitwidth`'s binary tree are spelled
    out rather than assumed, because the fold's whole correctness argument is
    that packing slot `j` is byte `j` of the little-endian word.
    """
    total_tokens, words, kv_packing, lane_dim = cache.shape
    assert kv_packing == 4, "this replay is written for the fp8 packing"
    word = np.zeros((total_tokens, words, lane_dim), np.uint32)
    for j in range(kv_packing):
        word |= cache[:, :, j, :].astype(np.uint32) << (8 * j)

    left = word.astype(np.uint16)
    right = (word >> 16).astype(np.uint16)
    slots = [
        left.astype(np.uint8),
        (left >> 8).astype(np.uint8),
        right.astype(np.uint8),
        (right >> 8).astype(np.uint8),
    ]

    # Per word: `fold` consecutive slots are one (K|V, head) vector.
    heads = []
    for w in range(words):
        segments = [s[:, w, :] for s in slots]
        heads += [
            np.concatenate(segments[i * fold:(i + 1) * fold], axis=-1)
            for i in range(kv_packing // fold)
        ]
    return np.stack(heads[0::2], axis=1), np.stack(heads[1::2], axis=1)


@pytest.mark.parametrize("num_kv_heads, fold", [(1, 2), (1, 1), (2, 1)])
def test_pack_unpack_round_trip_is_bit_exact(num_kv_heads, fold):
    """Same values, different byte placement -- no requantisation anywhere."""
    rng = np.random.default_rng(0)
    head_dim, total_tokens = 256, 7
    shape = (total_tokens, num_kv_heads, head_dim)
    k = rng.integers(0, 256, shape, dtype=np.uint8)
    v = rng.integers(0, 256, shape, dtype=np.uint8)

    cache = _pack_new_kv(k, v, 4, fold)
    assert cache.shape == (total_tokens,
                           utils.align_to(num_kv_heads * 2 * fold, 4) // 4, 4,
                           head_dim // fold)

    got_k, got_v = _unpack_cache(cache, fold)
    np.testing.assert_array_equal(got_k[:, :num_kv_heads], k)
    np.testing.assert_array_equal(got_v[:, :num_kv_heads], v)


def test_folded_and_unfolded_caches_hold_the_same_bytes():
    """The fold only moves the live bytes; the unfolded page pads the rest."""
    rng = np.random.default_rng(1)
    shape = (5, 1, 256)
    k = rng.integers(1, 256, shape, dtype=np.uint8)
    v = rng.integers(1, 256, shape, dtype=np.uint8)

    unfolded = _pack_new_kv(k, v, 4, 1)
    folded = _pack_new_kv(k, v, 4, 2)
    assert folded.size * 2 == unfolded.size
    # Everything the fold drops was a structural zero.
    assert unfolded[:, :, 2:, :].sum() == 0

    np.testing.assert_array_equal(
        _unpack_cache(folded, 2)[0][:, :1],
        _unpack_cache(unfolded, 1)[0][:, :1])
    np.testing.assert_array_equal(
        _unpack_cache(folded, 2)[1][:, :1],
        _unpack_cache(unfolded, 1)[1][:, :1])
