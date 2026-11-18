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
Qwen3.5 GDN v3 copy. The tiler, the per-step trims and the fused 1-dot run only
at the Qwen3.5-397B shapes they were measured at; every other shape takes
main's tiles and main's per-step code."""

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import compute_conv1d as main_compute_conv1d
from vllm_torchtpu.kernels.gdn.v3 import compute_gdn as main_compute_gdn
from vllm_torchtpu.kernels.gdn.v3 import config as main_config
from vllm_torchtpu.kernels.gdn.v3 import metadata as main_metadata
from vllm_torchtpu.kernels.gdn.v3 import wrapper as main_wrapper
from vllm_torchtpu.models.qwen3_5.kernels.gdn import (
    compute_conv1d,
    compute_gdn,
    compute_kda,
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
def _tpu7x_info(request):
    # Host tests trace for a TPU7x device; a TPU test runs on its own device.
    if request.node.get_closest_marker("tpu_test"):
        yield
        return
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
        # ... and at both VMEM ceilings, while a smaller override stands.
        (
            dict(
                default_tile=4,
                dyn_tile=2,
                num_spec_tokens=3,
                spec_tile_cap=2,
                override=4,
            ),
            2,
        ),
        (
            dict(
                default_tile=4,
                dyn_tile=2,
                num_spec_tokens=3,
                vmem_tile_cap=2,
                override=4,
            ),
            2,
        ),
        (
            dict(
                default_tile=4,
                dyn_tile=None,
                num_spec_tokens=3,
                spec_tile_cap=2,
                vmem_tile_cap=2,
                override=1,
            ),
            1,
        ),
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


def _call_kernels(
    n_kq, n_v, window, num_seqs, num_tokens, state_dtype, module=wrapper, **kwargs
):
    """Kernel names of one fused_conv1d_gdn call (decode and prefill/mixed),
    traced from shapes only. `module` is the copy's wrapper or main's."""
    dim = 2 * n_kq * D_K + n_v * D_V
    num_slots = num_seqs * window + 1

    def sds(shape, dtype):
        return jax.ShapeDtypeStruct(shape, dtype)

    inputs = dict(
        qkv=sds((num_tokens, dim), BF16),
        b=sds((num_tokens, n_v), BF16),
        a=sds((num_tokens, n_v), BF16),
        states=(
            sds((num_slots, KERNEL_SIZE - 1, dim), F32),
            sds((num_slots, n_v, D_K, D_V), state_dtype),
        ),
        conv_weight=sds((dim, 1, KERNEL_SIZE), BF16),
        a_log=sds((n_v,), F32),
        dt_bias=sds((n_v,), F32),
        query_start_loc=sds((num_seqs + 1,), jnp.int32),
        state_indices=sds((num_seqs,), jnp.int32),
        distribution=sds((3,), jnp.int32),
        seq_lens=sds((num_seqs,), jnp.int32),
    )
    if window > 1:
        inputs["read_offsets"] = sds((num_seqs,), jnp.int32)
    jaxpr = jax.make_jaxpr(
        lambda x: module.fused_conv1d_gdn.__wrapped__(
            **x,
            conv_bias=None,
            n_kq=n_kq,
            n_v=n_v,
            d_k=D_K,
            d_v=D_V,
            kernel_size=KERNEL_SIZE,
            num_spec_tokens=window - 1,
            **kwargs,
        )
    )(inputs)
    names = re.findall(r"fused_conv1d_gdn_(\w+?_b\d+_c\d+(?:_w\d+)?)", str(jaxpr))
    return tuple(sorted(set(names)))


# (label, n_kq, n_v, window, sequences, tokens, SSM state, override, kernels)
# under dynamic tiling, which the served Qwen3.5-397B legs enable. The first is
# a TPU run of the copy's kernel at the DP4 x TP2 decode shape (64 verify
# windows, 250 tokens): its override T = 4 needed 61.56 MiB of scoped VMEM
# against the 57.60 MiB limit, so the override now stops at the ceilings' T =
# 2. The rest are the validation entries' shapes (max-num-seqs sequences, a
# compiled token bucket), unchanged by the cap: the DP4 x TP2 decode with its
# layout tile 1 and with the tiler's pick, the f32 state of the gates' off arm,
# the TP8 + DCP4 decodes at 16, 8, 3 and 1 sequences, and the PP8 x TP1 prefill
# leg with MTP 1. The last case is an override of 16 at TP8, where the VMEM fit
# (8) is below spec_tile_cap (10).
_CALLS = [
    ("tp2_override4_c64", 8, 32, 4, 64, 250, BF16, 4, ("b2_c4_w4", "c128")),
    ("dp4tp2_layout_tile", 8, 32, 4, 28, 112, BF16, 1, ("b1_c4_w4", "c112")),
    ("dp4tp2_pick", 8, 32, 4, 28, 112, BF16, 0, ("b2_c4_w4", "c112")),
    ("dp4tp2_f32_state", 8, 32, 4, 28, 112, F32, 1, ("b1_c4_w4", "c112")),
    ("tp8_dcp4_c16", 2, 8, 4, 16, 64, BF16, 0, ("b4_c4_w4", "c64")),
    ("tp8_dcp4_c8", 2, 8, 4, 8, 32, BF16, 0, ("b4_c4_w4", "c32")),
    ("tp8_dcp4_c3", 2, 8, 4, 3, 16, BF16, 0, ("b2_c4_w4", "c16")),
    ("tp8_dcp4_c1", 2, 8, 4, 1, 4, BF16, 0, ("b1_c4_w4", "c4")),
    ("pp8_tp1_mtp1_prefill", 16, 64, 2, 32, 8192, BF16, 0, ("b2_c2_w2", "c64")),
    ("tp8_override16", 2, 8, 4, 16, 64, BF16, 16, ("b8_c4_w4", "c64")),
]


@pytest.mark.parametrize(
    ("n_kq", "n_v", "window", "num_seqs", "num_tokens", "state", "override", "tiles"),
    [c[1:] for c in _CALLS],
    ids=[c[0] for c in _CALLS],
)
def test_decode_tile_per_call(
    monkeypatch, n_kq, n_v, window, num_seqs, num_tokens, state, override, tiles
):
    monkeypatch.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True)
    batched, mixed = tiles
    assert _call_kernels(
        n_kq,
        n_v,
        window,
        num_seqs,
        num_tokens,
        state,
        decode_tile_override=override,
    ) == (f"batched_{batched}", f"per_seq_b1_{mixed}")


def _cfg(
    mode,
    tile_size,
    window_size=1,
    batch_size=256,
    n_kq=2,
    n_v=8,
    cfg_module=config,
    **kwargs,
):
    return cfg_module.GDNConfig(
        mode=mode,
        dtypes=cfg_module.Dtypes(
            act_in=F32, act_out=BF16, compute=F32, recurrent_state=F32, conv_state=F32
        ),
        batch_size=batch_size,
        dim_size=2 * n_kq * D_K + n_v * D_V,
        kernel_size=KERNEL_SIZE,
        tile_size=tile_size,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
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


@pytest.mark.tpu_test
@pytest.mark.parametrize("state_dtype", [BF16, F32], ids=["bf16_state", "f32_state"])
def test_fused_1dot_kernel_matches_mains_two_dot_kernel_on_tpu(state_dtype):
    """The compiled copy at the Qwen3.5-397B DP4 x TP2 decode shape (8 kq and
    32 value heads per rank) with verify windows of 4, where the fused 1-dot
    recurrence runs, against main's v3 kernel (the two-dot recurrence) on the
    same inputs: the outputs and the conv and recurrent state checkpoints of
    every real token."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    n_kq, n_v, window = 8, 32, 4
    assert wrapper.use_fused_1dot(window, n_kq, n_v, D_K, D_V)
    # 24 verify windows: full, and with fewer drafts accepted (shorter
    # windows); every sequence continues a context and reads its state from
    # one of its window's checkpoint slots.
    lengths = [window, 3, 2, 1] * 6
    num_seqs = len(lengths)
    num_tokens = sum(lengths)
    dim = 2 * n_kq * D_K + n_v * D_V
    num_blocks = 1 + num_seqs * window
    rngs = iter(jax.random.split(jax.random.key(11), 8))
    kwargs = dict(
        qkv=jax.random.normal(next(rngs), (num_tokens, dim), BF16),
        b=jax.random.normal(next(rngs), (num_tokens, n_v), BF16),
        a=jax.random.normal(next(rngs), (num_tokens, n_v), BF16),
        states=(
            jax.random.normal(next(rngs), (num_blocks, KERNEL_SIZE - 1, dim), F32),
            (
                0.1 * jax.random.normal(next(rngs), (num_blocks, n_v, D_K, D_V), F32)
            ).astype(state_dtype),
        ),
        conv_weight=jax.random.normal(next(rngs), (dim, 1, KERNEL_SIZE), BF16),
        conv_bias=None,
        a_log=jax.random.normal(next(rngs), (n_v,), F32),
        dt_bias=jnp.zeros((n_v,), F32),
        query_start_loc=jnp.asarray(
            np.concatenate([[0], np.cumsum(lengths)]), jnp.int32
        ),
        state_indices=jnp.arange(num_seqs, dtype=jnp.int32) * window + 1,
        distribution=jnp.array([num_seqs] * 3, jnp.int32),
        seq_lens=jnp.asarray([32 + n for n in lengths], jnp.int32),
        read_offsets=jnp.asarray([s % window for s in range(num_seqs)], jnp.int32),
    )
    static = dict(
        n_kq=n_kq,
        n_v=n_v,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
        num_spec_tokens=window - 1,
    )
    got = jax.jit(lambda x: wrapper.fused_conv1d_gdn(**x, **static))(kwargs)
    want = jax.jit(lambda x: main_wrapper.fused_conv1d_gdn(**x, **static))(kwargs)
    (got_conv, got_rec), got_out = got
    (want_conv, want_rec), want_out = want
    # The checkpoint slots each window writes for its real tokens.
    slots = np.concatenate(
        [s * window + 1 + np.arange(n) for s, n in enumerate(lengths)]
    )
    # Conv states are input rows, moved, not computed.
    np.testing.assert_array_equal(got_conv[slots], want_conv[slots])
    for g, w in ((got_out, want_out), (got_rec[slots], want_rec[slots])):
        np.testing.assert_allclose(
            np.asarray(g, np.float32), np.asarray(w, np.float32), rtol=2e-2, atol=2e-2
        )


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
        # TP4 (16 value heads per rank) is not a fused shape.
        (4, 16, 4, False),
        # Qwen3.5-35B-A3B at TP1 (32 value heads per rank, as the 397B at TP2,
        # but 16 kq heads), TP2 and TP4 keeps main's two-dot form.
        (16, 32, 4, False),
        (8, 16, 4, False),
        (4, 8, 4, False),
    ],
)
def test_the_fused_1dot_rule(n_kq, n_v, window, fused):
    assert wrapper.use_fused_1dot(window, n_kq, n_v, D_K, D_V) is fused
    params = _decode_call_params(n_kq=n_kq, n_v=n_v, window=window)
    assert f"'fused_1dot': {fused}" in params


def test_off_the_fused_shapes_a_verify_window_computes_mains_recurrence():
    """Qwen3.5-35B-A3B at TP1 with MTP: its verify windows give main's
    recurrence bit for bit."""
    n_kq, n_v, seq_tile, window = 16, 32, 2, 4
    aligned_v = 128
    rngs = iter(jax.random.split(jax.random.key(7), 9))
    args = (
        jnp.array([window, 2], jnp.int32),
        jax.random.normal(next(rngs), (seq_tile, n_kq, window, 1, D_K), F32),
        jax.random.normal(next(rngs), (seq_tile, n_kq, window, 1, D_K), F32),
        jax.random.normal(next(rngs), (seq_tile, n_v, window, 1, D_V), F32),
        jax.random.normal(next(rngs), (seq_tile, 1, window, 1, aligned_v), F32),
        jax.random.normal(next(rngs), (seq_tile, 1, window, 1, aligned_v), F32),
        0.1 * jax.random.normal(next(rngs), (seq_tile, n_v, D_K, D_V), F32),
        jax.random.normal(next(rngs), (aligned_v,), F32),
        jax.random.normal(next(rngs), (aligned_v,), F32),
    )
    shape = dict(tile_size=seq_tile, window_size=window, n_kq=n_kq, n_v=n_v)
    cfg = _cfg(
        config.GDNMode.BATCHED,
        fused_1dot=wrapper.use_fused_1dot(window, n_kq, n_v, D_K, D_V),
        **shape,
    )
    ref_cfg = _cfg(main_config.GDNMode.BATCHED, cfg_module=main_config, **shape)
    got = jax.jit(lambda *a: compute_gdn.recurrent_gdn(*a, cfg=cfg))(*args)
    want = jax.jit(lambda *a: main_compute_gdn.recurrent_gdn(*a, cfg=ref_cfg))(*args)
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_array_equal(g, w)


def _per_seq_chunk(kernels):
    (chunk,) = [int(k.rsplit("_c", 1)[1]) for k in kernels if k.startswith("per_seq")]
    return chunk


# Per-rank GDN shapes outside config.TILER_SHAPES, with token counts at
# which the tiler's chunk would differ from main's: Qwen3.5-35B-A3B (16 kq and
# 32 value heads) at TP4, as main's CI serves it, at TP2 and at TP1, and the
# Qwen3.5-397B at TP4. A 96-token batch takes main's 64-token chunk (the VMEM
# fit is 64 as well), where the tiler would keep 96.
_MAIN_CHUNK_CASES = [
    (4, 8, 96),
    (4, 8, 8192),
    (8, 16, 96),
    (8, 16, 8192),
    (16, 32, 96),
    (16, 32, 8192),
    (4, 16, 96),
    (4, 16, 8192),
]


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(("n_kq", "n_v", "num_tokens"), _MAIN_CHUNK_CASES)
def test_off_the_tiler_shapes_the_prefill_chunk_is_mains(
    monkeypatch, n_kq, n_v, num_tokens, dynamic
):
    """Outside config.TILER_SHAPES the prefill/mixed kernel takes the
    chunk main's kernel takes: 64 tokens by default (the batch rounded down to
    a power of two, if smaller), or the VMEM fit under dynamic tiling."""
    monkeypatch.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", dynamic)
    got = _per_seq_chunk(_call_kernels(n_kq, n_v, 1, 1, num_tokens, F32))
    want = _per_seq_chunk(
        _call_kernels(n_kq, n_v, 1, 1, num_tokens, F32, module=main_wrapper)
    )
    assert got == want
    if not dynamic:
        assert got == tiling.clip_mixed_tile_size(
            wrapper.MAIN_MIXED_TILE_SIZE, num_tokens
        )


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    ("n_kq", "n_v"),
    # Qwen3.5-35B-A3B at TP4, TP2 and TP1 (triangular blocks of 16, 8 and 4
    # rows), and 24 value heads per rank, where the block's exact quotient
    # 16 // 3 = 5 divides no power of two.
    [(4, 8), (8, 16), (16, 32), (8, 24)],
)
def test_off_the_tiler_shapes_a_small_bucket_takes_a_power_of_two_chunk(
    monkeypatch, n_kq, n_v, dynamic
):
    """Outside config.TILER_SHAPES a token bucket that is not a power of two
    (TPU_TOKEN_BUCKET_EXTRA, a small max_num_batched_tokens) takes the largest
    power of two below it as the chunk, which the triangular block divides,
    as main's kernel does since #2438. A 24-token bucket at 8 value heads per
    rank (blocks of 16) failed to trace with a 24-token chunk."""
    monkeypatch.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", dynamic)
    for num_tokens, chunk in ((3, 2), (5, 4), (12, 8), (24, 16), (48, 32), (96, 64)):
        kernels = _call_kernels(n_kq, n_v, 1, 1, num_tokens, F32)
        assert _per_seq_chunk(kernels) == chunk, num_tokens


_GDN_BLOCKS = {8: 16, 16: 8, 24: 4, 32: 4, 40: 2, 48: 2, 64: 2}


def test_the_triangular_block_is_a_power_of_two_dividing_every_chunk():
    """B is the largest power of two within the head-group budget, for every
    head count, GDN and KDA (main's rule since #2438). The Qwen3.5-397B blocks
    (2 rows at 64 value heads per rank, 4 at 32, 16 at 8) are the old
    quotient's."""
    for n_v in (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 96, 128):
        for mode in config.AttentionMode:
            for chunk in tiling.MIXED_TILE_SIZES:
                cfg = _cfg(
                    config.GDNMode.PER_SEQ,
                    chunk,
                    n_kq=max(1, n_v // 4),
                    n_v=n_v,
                    attention_mode=mode,
                )
                b = cfg.triangular_block_size
                assert b >= 2 and b & (b - 1) == 0, (n_v, mode, b)
                assert cfg.chunk_size % min(b, cfg.chunk_size) == 0, (n_v, chunk)
        if n_v in _GDN_BLOCKS:
            assert config.gdn_triangular_block_size(n_v) == _GDN_BLOCKS[n_v]


def test_the_solves_reject_a_block_that_does_not_divide_the_chunk():
    """A non-dividing block fails at the solve with the divisibility message
    instead of dropping the remainder rows (main's check since #2438)."""
    t = jnp.tril(jnp.ones((2, 128, 128), F32))
    with pytest.raises(AssertionError, match="divisible"):
        compute_gdn.invert_triangular_matrix(t, block_size=5)
    for block_size in (16, 8, 4, 2):
        inv = compute_gdn.invert_triangular_matrix(t, block_size=block_size)
        assert inv.shape == t.shape

    def kda(chunk):
        cfg = _cfg(
            config.GDNMode.PER_SEQ,
            chunk,
            n_kq=2,
            n_v=8,
            attention_mode=config.AttentionMode.KDA,
        )
        assert cfg.triangular_block_size == 16
        return compute_kda.chunked_kda_per_seq(
            q_large=jnp.zeros((2, chunk, D_K), F32),
            k_large=jnp.zeros((2, chunk, D_K), F32),
            v_large=jnp.zeros((8, chunk, D_V), F32),
            gating_log=jnp.zeros((8, chunk, D_K), F32),
            beta=jnp.zeros((1, chunk, 8), F32),
            state_prev=jnp.zeros((8, D_K, D_V), F32),
            cfg=cfg,
        )

    with pytest.raises(AssertionError, match="divisible"):
        kda(24)
    out, state = kda(32)
    assert out.shape == (8, 32, D_V)
    assert state.shape == (8, D_K, D_V)


def test_every_power_of_two_block_inverts_the_same_matrix():
    """The block only changes the solve's schedule: blocks of 16, 8, 4 and 2
    rows give the same inverse of a unit lower-triangular matrix."""
    strict = jnp.tril(jax.random.normal(jax.random.key(7), (3, 128, 128)) * 0.1, k=-1)
    t = strict + jnp.eye(128)[None]
    with jax.default_matmul_precision("highest"):
        want = jnp.linalg.inv(t)
        for block_size in (16, 8, 4, 2):
            got = compute_gdn.invert_triangular_matrix(t, block_size=block_size)
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("n_kq", "n_v", "num_tokens", "chunk"),
    [
        # The PP8 x TP1 prefill legs: the VMEM fit, and a 40-token batch
        # rounded down to a multiple of 16.
        (16, 64, 8192, 64),
        (16, 64, 40, 32),
        # The DP4 x TP2 and TP8 decode legs: 128 fits.
        (8, 32, 8192, 128),
        (2, 8, 8192, 128),
        (8, 32, 40, 32),
        (2, 8, 40, 32),
        # Compiled token buckets of the decode legs whose chunk is a multiple
        # of 16 but not a power of two, which main's rule would round down to
        # 32 or 64.
        (8, 32, 48, 48),
        (8, 32, 80, 80),
        (8, 32, 112, 112),
        (2, 8, 48, 48),
        (2, 8, 96, 96),
    ],
)
def test_at_the_tiler_shapes_the_tiler_sizes_the_prefill_chunk(
    monkeypatch, n_kq, n_v, num_tokens, chunk
):
    assert (n_kq, n_v, D_K, D_V) in config.TILER_SHAPES
    for dynamic in (False, True):
        monkeypatch.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", dynamic)
        assert _per_seq_chunk(_call_kernels(n_kq, n_v, 1, 1, num_tokens, F32)) == (
            chunk
        )


@pytest.mark.parametrize(
    ("n_kq", "n_v", "num_tokens", "chunk"),
    [
        # Batches under 16 tokens at the tiler shapes keep the batch as the
        # chunk where the solve's triangular block (2 rows at 64 value heads, 4
        # at 32, 16 at 8) divides it or is longer than it ...
        (16, 64, 10, 10),
        (8, 32, 12, 12),
        (8, 32, 3, 3),
        (2, 8, 10, 10),
        # ... and otherwise round it down to whole blocks, where the chunked
        # solve used to fail to trace.
        (16, 64, 9, 8),
        (16, 64, 15, 14),
        (8, 32, 10, 8),
        (8, 32, 6, 4),
    ],
)
def test_at_the_tiler_shapes_a_small_chunk_holds_whole_triangular_blocks(
    monkeypatch, n_kq, n_v, num_tokens, chunk
):
    assert (n_kq, n_v, D_K, D_V) in config.TILER_SHAPES
    for dynamic in (False, True):
        monkeypatch.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", dynamic)
        assert _per_seq_chunk(_call_kernels(n_kq, n_v, 1, 1, num_tokens, F32)) == (
            chunk
        )


def _batched_tile(kernels):
    (tile,) = [
        int(re.search(r"_b(\d+)_", k).group(1))
        for k in kernels
        if k.startswith("batched")
    ]
    return tile


# (window, sequences, tokens): a one-sequence decode (main's CI serves
# max-num-seqs 1), a 32-sequence decode and 16 verify windows of 4.
_MAIN_DECODE_CALLS = [(1, 1, 16), (1, 32, 256), (4, 16, 64)]


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(("window", "num_seqs", "num_tokens"), _MAIN_DECODE_CALLS)
@pytest.mark.parametrize(("n_kq", "n_v"), [(4, 8), (8, 16), (16, 32)])
def test_off_the_tiler_shapes_the_decode_tile_is_mains(
    monkeypatch, n_kq, n_v, window, num_seqs, num_tokens, dynamic
):
    """Qwen3.5-35B-A3B at TP4, TP2 and TP1: the decode kernel takes the tile
    main's kernel takes, the caller's 4 under the window cap, or main's
    dynamic pick."""
    monkeypatch.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", dynamic)
    args = (n_kq, n_v, window, num_seqs, num_tokens, F32)
    got = _batched_tile(_call_kernels(*args))
    want = _batched_tile(_call_kernels(*args, module=main_wrapper))
    assert got == want


@pytest.mark.parametrize(
    ("n_kq", "n_v", "trims"),
    [
        (16, 64, True),
        (8, 32, True),
        (2, 8, True),
        (4, 8, False),
        (8, 16, False),
        (16, 32, False),
        (4, 16, False),
    ],
)
def test_the_step_trims_rule(n_kq, n_v, trims):
    assert ((n_kq, n_v, D_K, D_V) in config.TILER_SHAPES) is trims
    params = _decode_call_params(n_kq=n_kq, n_v=n_v, window=1)
    assert f"'step_trims': {trims}" in params


def test_without_the_step_trims_the_conv_taps_are_mains():
    """The conv program off the tiler shapes is main's, op for op; the trims'
    first-product start is a different program."""
    n_kq, n_v, seq_tile, chunk = 4, 8, 4, 1
    dim = 2 * n_kq * D_K + n_v * D_V
    rngs = iter(jax.random.split(jax.random.key(11), 2))
    args = (
        jnp.array([1, 1, 0, 1], jnp.int32),
        jax.random.normal(next(rngs), (seq_tile, chunk + KERNEL_SIZE - 1, 1, dim), F32),
        jax.random.normal(next(rngs), (KERNEL_SIZE, 1, dim), F32),
        None,
    )
    shape = dict(tile_size=seq_tile, n_kq=n_kq, n_v=n_v)
    cfg = _cfg(config.GDNMode.BATCHED, **shape)
    ref_cfg = _cfg(main_config.GDNMode.BATCHED, cfg_module=main_config, **shape)

    def program(fn, c):
        return str(jax.make_jaxpr(lambda *a: fn(*a, cfg=c))(*args))

    want = program(main_compute_conv1d.causal_conv1d, ref_cfg)
    assert not cfg.step_trims
    assert program(compute_conv1d.causal_conv1d, cfg) == want
    trims = dataclasses.replace(cfg, step_trims=True)
    assert program(compute_conv1d.causal_conv1d, trims) != want
