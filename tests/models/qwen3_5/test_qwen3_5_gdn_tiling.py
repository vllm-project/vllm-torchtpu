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
"""Tile selection, metadata bounds and the fused 1-dot recurrence of the
Qwen3.5 GDN v3 copy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.gdn.v3 import config as main_config
from vllm_torchtpu.kernels.gdn.v3 import metadata as main_metadata
from vllm_torchtpu.models.qwen3_5.kernels.gdn import (
    compute_gdn,
    config,
    metadata,
    tiling,
    wrapper,
)

D_K = D_V = 128
KERNEL_SIZE = 4
F32 = jnp.dtype(jnp.float32)
BF16 = jnp.dtype(jnp.bfloat16)


@pytest.fixture(autouse=True)
def _tpu7x_info():
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        yield


def _tiles(n_kq, n_v, b, n, w, state_dtype, **kwargs):
    limit = config.GDNConfig.get_vmem_limit_bytes(w)
    return tiling.get_tile_sizes(
        batch_size=b,
        num_seqs=n,
        padded_batch_size=b,
        n_kq=n_kq,
        n_v=n_v,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
        conv_state_dim_size=2 * n_kq * D_K + n_v * D_V,
        act_out_dtype=BF16,
        recurrent_state_dtype=state_dtype,
        num_lanes=128,
        decode_vmem_limit_bytes=limit,
        mixed_vmem_limit_bytes=kwargs.pop(
            "mixed_limit", config.GDNConfig.get_vmem_limit_bytes()
        ),
        window_size=w,
        **kwargs,
    )


# (label, n_kq, n_v, batch, num_seqs, window, state dtype, (T, C)) at the
# Qwen3.5-397B GDN head splits.
_CASES = [
    # TP8 + DCP=4 (n_kq=2, n_v=8, W=4 for MTP=3): warmup and verify buckets.
    ("tp8_dcp4_c16_b4", 2, 8, 4, 16, 4, F32, (2, 4)),
    ("tp8_dcp4_c16_b8", 2, 8, 8, 16, 4, F32, (4, 8)),
    ("tp8_dcp4_c16_b16", 2, 8, 16, 16, 4, F32, (4, 16)),
    ("tp8_dcp4_c16_b64", 2, 8, 64, 16, 4, F32, (4, 64)),
    ("tp8_dcp4_c32_b128", 2, 8, 128, 32, 4, F32, (4, 128)),
    ("tp8_dcp4_c64_b256", 2, 8, 256, 64, 4, F32, (4, 128)),
    # TP8 + DCP=1 (W=1 and W=4) and a padded slot table.
    ("tp8_dcp1_c8_mtp3_b32", 2, 8, 32, 8, 4, F32, (4, 32)),
    ("tp8_dcp1_c32_b16", 2, 8, 16, 32, 1, F32, (4, 16)),
    ("tp8_dcp1_c64_b64", 2, 8, 64, 64, 1, F32, (8, 64)),
    ("tp8_padded_n512_b64", 2, 8, 64, 512, 1, BF16, (8, 64)),
    # DP4 + TP2 (n_kq=8, n_v=32, W=1 and W=4).
    ("dp4tp2_c64_b8", 8, 32, 8, 32, 1, F32, (2, 8)),
    ("dp4tp2_c64_b32", 8, 32, 32, 32, 1, F32, (4, 32)),
    ("dp4tp2_c64_b2048", 8, 32, 2048, 32, 1, F32, (4, 128)),
    ("dp4tp2_mtp3_c16_b64_f32", 8, 32, 64, 16, 4, F32, (1, 64)),
    ("dp4tp2_mtp3_c32_b128_bf16", 8, 32, 128, 32, 4, BF16, (2, 128)),
    # DP8 + TP1 (n_kq=16, n_v=64, W=1 and W=4).
    ("dp8_c128_b16", 16, 64, 16, 16, 1, F32, (2, 16)),
    ("dp8_c256_b320_f32", 16, 64, 320, 32, 1, F32, (2, 64)),
    ("dp8_mtp3_c32_b128_bf16", 16, 64, 128, 32, 4, BF16, (1, 64)),
]


@pytest.mark.parametrize(
    ("n_kq", "n_v", "b", "n", "w", "state_dtype", "expected"),
    [c[1:] for c in _CASES],
    ids=[c[0] for c in _CASES],
)
def test_qwen3_5_397b_tile_selection(n_kq, n_v, b, n, w, state_dtype, expected):
    assert _tiles(n_kq, n_v, b, n, w, state_dtype) == expected


def test_w1_decode_tiles_count_an_input_and_an_output_state_slot():
    """One state slot per sequence would pick T=8 and T=4 here."""
    limit = config.GDNConfig.get_vmem_limit_bytes(1)
    assert _tiles(8, 32, 128, 128, 1, F32, mixed_limit=limit) == (4, 128)
    assert _tiles(16, 64, 64, 64, 1, F32, mixed_limit=limit) == (2, 64)


def test_a_caller_chunk_is_capped_at_the_vmem_fit():
    """A caller's prefill chunk stands unless it does not fit."""
    _, chunk = _tiles(2, 8, 8192, 16, 1, F32, mixed_tile_size=64)
    assert chunk == 64
    _, chunk = _tiles(16, 64, 8192, 1, 1, F32, mixed_tile_size=128)
    assert chunk < 128


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # Static path: the caller's tile, capped by the window cap.
        (dict(default_tile=8, dyn_tile=None, num_spec_tokens=0), 8),
        (dict(default_tile=8, dyn_tile=None, num_spec_tokens=0, spec_tile_cap=2), 2),
        # Verify windows without the tiler: the token-count heuristic.
        (dict(default_tile=8, dyn_tile=None, num_spec_tokens=3), 4),
        # The tiler's pick replaces the heuristic and is capped for windows.
        (dict(default_tile=4, dyn_tile=16, num_spec_tokens=3, spec_tile_cap=8), 8),
        # An explicit override wins, capped at the sequence count.
        (dict(default_tile=4, dyn_tile=16, num_spec_tokens=3, override=1), 1),
        (dict(default_tile=4, dyn_tile=4, num_spec_tokens=0, override=64), 40),
        # KDA verify windows stay at one sequence, override or not.
        (dict(default_tile=4, dyn_tile=None, num_spec_tokens=3, is_kda=True), 1),
    ],
)
def test_select_decode_tile_size(kwargs, expected):
    args = dict(
        batch_size=160,
        spec_tile_cap=64,
        is_kda=False,
        override=0,
        num_seqs=40,
    )
    args.update(kwargs)
    assert tiling.select_decode_tile_size(**args) == expected


def _cfg(mode, tile_size, window_size=1, batch_size=256, **kwargs):
    return config.GDNConfig(
        mode=mode,
        dtypes=config.Dtypes(
            act_in=F32, act_out=BF16, compute=F32, recurrent_state=F32, conv_state=F32
        ),
        batch_size=batch_size,
        dim_size=2 * 2 * D_K + 8 * D_V,
        kernel_size=KERNEL_SIZE,
        tile_size=tile_size,
        num_kq_heads=2,
        num_v_heads=8,
        kq_head_dim=D_K,
        v_head_dim=D_V,
        window_size=window_size,
        **kwargs,
    )


def test_batched_smem_tables_are_padded_to_whole_tiles():
    """5 sequences at T=4: the last tile reads padded, inert SMEM entries."""
    num_seqs = 5
    lens = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)
    q_loc = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(lens)])
    md = metadata.compute_batched_seq_metadata(
        cfg=_cfg(config.GDNMode.BATCHED, tile_size=4, window_size=4),
        seq_lens=lens + 3,
        query_start_loc=q_loc,
        state_indices=jnp.arange(num_seqs, dtype=jnp.int32),
        read_offsets=jnp.zeros((num_seqs,), jnp.int32),
        end_seq=jnp.int32(num_seqs),
    )
    assert int(md.num_tiles) == 2
    r_size = np.asarray(md.p_id_to_r_size.data)
    assert r_size.shape == (8,)
    np.testing.assert_array_equal(r_size, [1, 2, 3, 4, 5, 0, 0, 0])


@pytest.mark.parametrize(
    ("lens", "offsets"),
    [
        ([37, 128, 1, 70], None),
        ([37, 128, 1, 70], [20, 65, 0, 33]),
        # Every split leaves two partial tiles: 4 tiles per 66-token sequence,
        # past cdiv(batch, chunk) + num_seqs.
        ([66, 66, 66, 66], [33, 33, 33, 33]),
    ],
)
def test_per_seq_tables_cover_every_tile(lens, offsets):
    """The bounded PER_SEQ tables hold the same tiles as main's batch-sized
    ones, including the extra split at an internal prefill checkpoint."""
    num_seqs = len(lens)
    batch = sum(lens)
    q_loc = jnp.asarray(np.cumsum([0, *lens]), jnp.int32)
    kwargs = dict(
        seq_lens=jnp.asarray(lens, jnp.int32),
        query_start_loc=q_loc,
        state_indices=jnp.arange(num_seqs, dtype=jnp.int32),
        read_offsets=jnp.zeros((num_seqs,), jnp.int32),
        start_seq=jnp.int32(0),
        end_seq=jnp.int32(num_seqs),
    )
    if offsets is not None:
        kwargs.update(
            prefill_ckpt_offsets=jnp.asarray(offsets, jnp.int32),
            prefill_ckpt_indices=jnp.arange(7, 7 + num_seqs, dtype=jnp.int32),
        )
    chunk = 32
    cfg = _cfg(config.GDNMode.PER_SEQ, tile_size=chunk, batch_size=batch)
    got = metadata.compute_per_seq_metadata(cfg=cfg, **kwargs)
    ref = main_metadata.compute_per_seq_metadata(
        cfg=main_config.GDNConfig(
            mode=main_config.GDNMode.PER_SEQ,
            dtypes=main_config.Dtypes(
                act_in=F32,
                act_out=BF16,
                compute=F32,
                recurrent_state=F32,
                conv_state=F32,
            ),
            batch_size=batch,
            dim_size=cfg.dim_size,
            kernel_size=KERNEL_SIZE,
            tile_size=chunk,
            num_kq_heads=2,
            num_v_heads=8,
            kq_head_dim=D_K,
            v_head_dim=D_V,
        ),
        **kwargs,
    )
    n = int(ref.num_tiles)
    assert int(got.num_tiles) == n
    assert np.asarray(got.p_id_to_s_idx.data).size < batch
    for name in ("p_id_to_s_idx", "p_id_to_r_base", "p_id_to_r_size"):
        np.testing.assert_array_equal(
            np.asarray(getattr(got, name).data)[:n],
            np.asarray(getattr(ref, name).data)[:n],
            err_msg=name,
        )


def test_fused_1dot_matches_the_two_dot_recurrence():
    seq_tile, window, aligned_v = 2, 4, 128
    rngs = iter(jax.random.split(jax.random.key(5), 9))
    args = (
        jnp.array([window, 2], jnp.int32),
        jax.random.normal(next(rngs), (seq_tile, 2, window, 1, D_K), F32),
        jax.random.normal(next(rngs), (seq_tile, 2, window, 1, D_K), F32),
        jax.random.normal(next(rngs), (seq_tile, 8, window, 1, D_V), F32),
        jax.random.normal(next(rngs), (seq_tile, 1, window, 1, aligned_v), F32),
        jax.random.normal(next(rngs), (seq_tile, 1, window, 1, aligned_v), F32),
        0.1 * jax.random.normal(next(rngs), (seq_tile, 8, D_K, D_V), F32),
        jax.random.normal(next(rngs), (aligned_v,), F32),
        jax.random.normal(next(rngs), (aligned_v,), F32),
    )
    outs = {}
    for fused in (False, True):
        cfg = _cfg(
            config.GDNMode.BATCHED,
            tile_size=seq_tile,
            window_size=window,
            fused_1dot=fused,
        )
        outs[fused] = compute_gdn.recurrent_gdn(*args, cfg=cfg)
    for got, want in zip(
        jax.tree.leaves(outs[True]), jax.tree.leaves(outs[False]), strict=True
    ):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    # One dot per token instead of two.
    cfg = _cfg(
        config.GDNMode.BATCHED, tile_size=seq_tile, window_size=window, fused_1dot=True
    )
    jaxpr = jax.make_jaxpr(lambda *a: compute_gdn.recurrent_gdn(*a, cfg=cfg))(*args)
    dots = [e for e in jaxpr.jaxpr.eqns if e.primitive.name == "dot_general"]
    assert len(dots) == seq_tile * window


def _decode_call_params(n_kq=1, n_v=2, window=4, **kwargs):
    """Parameters of the decode pallas_call of a 16-sequence batch."""
    num_seqs = 16
    dim = 2 * n_kq * D_K + n_v * D_V
    num_tokens = num_seqs * window
    inputs = dict(
        qkv=jnp.zeros((num_tokens, dim), jnp.bfloat16),
        b=jnp.zeros((num_tokens, n_v), jnp.bfloat16),
        a=jnp.zeros((num_tokens, n_v), jnp.bfloat16),
        states=(
            jnp.zeros((num_seqs * window + window, KERNEL_SIZE - 1, dim)),
            jnp.zeros((num_seqs * window + window, n_v, D_K, D_V)),
        ),
        conv_weight=jnp.zeros((dim, 1, KERNEL_SIZE), jnp.bfloat16),
        conv_bias=None,
        a_log=jnp.zeros((n_v,), F32),
        dt_bias=jnp.zeros((n_v,), F32),
        query_start_loc=jnp.arange(num_seqs + 1, dtype=jnp.int32) * window,
        state_indices=jnp.arange(1, num_seqs + 1, dtype=jnp.int32) * window,
        distribution=jnp.array([num_seqs] * 3, jnp.int32),
        seq_lens=jnp.full((num_seqs,), 20, jnp.int32),
        read_offsets=jnp.zeros((num_seqs,), jnp.int32),
    )
    jaxpr = jax.make_jaxpr(
        lambda x: wrapper.fused_conv1d_gdn.__wrapped__(
            **x,
            n_kq=n_kq,
            n_v=n_v,
            d_k=D_K,
            d_v=D_V,
            kernel_size=KERNEL_SIZE,
            num_spec_tokens=window - 1,
            batched_only=True,
            **kwargs,
        )
    )(inputs)
    (call,) = [e for e in jaxpr.jaxpr.eqns if e.primitive.name == "pallas_call"]
    return str(call.params)


def test_the_decode_tile_override_reaches_the_kernel():
    """decode_tile_override sets the decode tile in place of the pick."""
    assert "_b1_c4_w4" not in _decode_call_params(decode_tile_size=4)
    assert "_b1_c4_w4" in _decode_call_params(
        decode_tile_size=4, decode_tile_override=1
    )


@pytest.mark.parametrize(
    ("n_kq", "n_v", "window", "fused"),
    [
        # Qwen3.5-397B per-rank heads: TP2 (the DP4 x TP2 decode) and TP1 verify
        # windows run fused; TP8 and non-speculative decode do not.
        (8, 32, 4, True),
        (16, 64, 2, True),
        (2, 8, 4, False),
        (8, 32, 1, False),
    ],
)
def test_the_fused_1dot_rule(n_kq, n_v, window, fused):
    assert wrapper.use_fused_1dot(window, n_v) is fused
    params = _decode_call_params(n_kq=n_kq, n_v=n_v, window=window)
    assert f"'fused_1dot': {fused}" in params
