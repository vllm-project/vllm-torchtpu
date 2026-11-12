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
"""`TPU_MLA_SPARSE_FLAT_Q`: 2D flat `[T, H * 256]` q into `absorb_q` and sparse MLA.

Verifies on CPU (TPU Pallas interpret mode with race detection + TPU7x Mosaic
MLIR export) that:
  1. `sparse_attention_chunk` with 2D `q_pe` (`[T, H * 256]`) + `q_rope_table`
     matches 3D `q_pe` (`[T, H, 256]`) and XLA `rope_reference` bit-for-bit
     (0 ULP on both output and LSE) across empty rows (`kv_len == 0`), partial
     windows, full windows, capped overflow fix-up (`kv_len > topk_cap`),
     default `"pair"` / `"fxskip"` mixed body levers, and small-chunk decode
     fallback;
  2. unused `q_nope` lanes (`0:128`) poisoned with `NaN` in 2D `q_pe` do not
     contaminate `out` or `lse`, while positive-control faults (`skip_q_rope`,
     `rope_swap`) fail;
  3. TPU7x Mosaic MLIR lowering of the combined 4K step (`absorb_q` +
     `sparse_ragged_paged_attention`) consumes `%arg0` (`bf16[4096, 16384]`)
     directly with `dense<[1, 0]>` layout in both kernels without any 3D
     reshape/transpose on `%arg0`.
"""

from unittest import mock

import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.absorb_q import absorb_q
from vllm_torchtpu.kernels.mla.sparse import (
    kernel as sparse_mla,
)
from vllm_torchtpu.kernels.mla.sparse import (
    native_sc_cache_layout,
)


def _bits16(x: jax.Array) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint16))


def _bits32(x: jax.Array) -> np.ndarray:
    return np.asarray(x).view(np.uint32)


def _rope_tab_and_ref(q_raw: jax.Array, cs: jax.Array) -> tuple[jax.Array, jax.Array]:
    cos, sin = cs[:, :32], cs[:, 32:]
    tab = jnp.stack(
        [
            jnp.concatenate([cos, cos, cos, cos], -1).astype(jnp.float32),
            jnp.concatenate([-sin, sin, -sin, sin], -1).astype(jnp.float32),
        ],
        1,
    )
    x = q_raw[..., 192:]
    q_pe_roped = jnp.concatenate(
        [
            x[..., :32] * cos[:, None, :] - x[..., 32:] * sin[:, None, :],
            x[..., 32:] * cos[:, None, :] + x[..., :32] * sin[:, None, :],
        ],
        -1,
    )
    return tab, q_pe_roped


def _packed_gathered_kv(
    rng: np.random.Generator, windows: int, tokens: int, cap: int
) -> tuple[jax.Array, jax.Array]:
    def qbytes(rows: int, dim: int) -> np.ndarray:
        vals = rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(np.float32)
        fp8 = jnp.asarray(vals).astype(jnp.float8_e4m3fn)
        return np.asarray(jax.lax.bitcast_convert_type(fp8, jnp.uint8))

    nope_u8 = qbytes(windows * tokens * cap, 512).reshape(windows * tokens, cap, 4, 128)
    rope_u8 = np.pad(qbytes(windows * tokens * cap, 64), ((0, 0), (0, 64))).reshape(
        windows * tokens, cap // 4, 4, 128
    )
    return (
        jnp.asarray(native_sc_cache_layout.pack_nope(nope_u8)),
        jnp.asarray(native_sc_cache_layout.pack_rope_banded(rope_u8)),
    )


@pytest.mark.parametrize(
    ("heads", "use_live_group_levers"),
    [(16, False), (32, True)],
    ids=["plain_capped_h16", "live_group_pair_fxskip_h32_multichunk"],
)
def test_sparse_attention_chunk_flat_q_pe_cpu_interpret(
    heads: int,
    use_live_group_levers: bool,
):
    tokens, topk, cap = 32, 128, 64
    batch_size = 16
    windows = topk // cap
    rng = np.random.default_rng(100 + heads + int(use_live_group_levers))

    q_nope = jnp.asarray(
        rng.normal(size=(tokens, heads, 512)).astype(np.float32), jnp.bfloat16
    )
    q_raw = jnp.asarray(
        rng.normal(size=(tokens, heads, 256)).astype(np.float32), jnp.bfloat16
    )
    cs = jnp.asarray(rng.normal(size=(tokens, 64)).astype(np.float32), jnp.bfloat16)
    tab, q_pe_roped = _rope_tab_and_ref(q_raw, cs)
    gnope, grope = _packed_gathered_kv(rng, windows, tokens, cap)

    # Row 0: empty (0 -> -inf LSE); row 1: partial (17); row 2: full window (64);
    # row 3: overflow (95 > 64 -> triggers `fixup_capped_overflow`!).
    kv_lens = jnp.array([0, 17, 64, 95] + [32] * (tokens - 4), dtype=jnp.int32)
    lse_in = jnp.full((tokens, heads), -jnp.inf, dtype=jnp.float32)

    live_views = None
    group_tab = None
    group_levers: tuple[str, ...] = ()
    if use_live_group_levers:
        atom = sparse_mla.GATHER_ATOM_ROWS  # 64
        num_groups = tokens // batch_size  # 2
        # All atoms live at group offsets 0, 1 (window 0's two 8-token groups).
        group_tab = jnp.array([1] * num_groups + list(range(num_groups)), jnp.int32)
        block = batch_size * atom
        live_views = (
            gnope.reshape(tokens * topk // block, batch_size, atom, 128),
            grope.reshape(tokens * topk // block, batch_size, atom // 4, 128),
        )
        group_levers = sparse_mla.MIXED_BODY_DEFAULT + sparse_mla.MIXED_KERNEL_V2_LEVERS

    def run(*, q_pe_in, rope_tab=None, fault=None):
        return sparse_mla.sparse_attention_chunk(
            q_nope,
            gnope,
            grope,
            kv_lens,
            None,
            lse_in,
            start_pos=0,
            end_pos=tokens,
            token_end=jnp.int32(tokens),
            sm_scale=0.125,
            k_scale=0.75,
            attention_kernel_batch_size=batch_size,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            is_dcp_mode=True,
            topk_cap=cap,
            topk_override=topk,
            q_pe=q_pe_in,
            q_rope_table=rope_tab,
            prologue_fault=fault,
            lse_lane_dense=True,
            lse_mask_empty=True,
            group_tab=group_tab,
            group_levers=group_levers,
            pre_gathered=(gnope, grope, live_views),
        )

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        ref_o, ref_l = run(q_pe_in=q_pe_roped)
        o3d, l3d = run(q_pe_in=q_raw, rope_tab=tab)
        o2d, l2d = run(q_pe_in=q_raw.reshape(tokens, heads * 256), rope_tab=tab)

        # Poison unused q_nope lanes [0:128] with NaN in 2D q_pe.
        q_poison = q_raw.at[:, :, :128].set(jnp.nan).reshape(tokens, heads * 256)
        o2d_poison, l2d_poison = run(q_pe_in=q_poison, rope_tab=tab)

        # Positive controls: skipping or swapping RoPE must fail.
        for bad_fault in ("skip_q_rope", "rope_swap"):
            obad, _ = run(
                q_pe_in=q_raw.reshape(tokens, heads * 256),
                rope_tab=tab,
                fault=bad_fault,
            )
            assert not np.array_equal(_bits16(obad), _bits16(o2d)), bad_fault

    assert jnp.isfinite(o2d).all()
    assert np.isneginf(np.asarray(l2d[0])).all()
    assert np.isfinite(np.asarray(l2d[1:])).all()
    np.testing.assert_array_equal(_bits16(o3d), _bits16(ref_o))
    np.testing.assert_array_equal(_bits32(l3d), _bits32(ref_l))
    np.testing.assert_array_equal(_bits16(o2d), _bits16(o3d))
    np.testing.assert_array_equal(_bits32(l2d), _bits32(l3d))
    np.testing.assert_array_equal(_bits16(o2d_poison), _bits16(o3d))
    np.testing.assert_array_equal(_bits32(l2d_poison), _bits32(l3d))


def test_sparse_attention_chunk_small_chunk_decode_fallback_cpu_interpret():
    """Small decode chunks (`small_chunk=True`, `batch_size=4`) reshape 2D `q_pe`
    cleanly onto the 3D decode path with 0-ULP bit-identity."""
    tokens, heads, topk, cap = 8, 8, 128, 64
    rng = np.random.default_rng(77)
    q_nope = jnp.asarray(
        rng.normal(size=(tokens, heads, 512)).astype(np.float32), jnp.bfloat16
    )
    q_raw = jnp.asarray(
        rng.normal(size=(tokens, heads, 256)).astype(np.float32), jnp.bfloat16
    )
    cs = jnp.asarray(rng.normal(size=(tokens, 64)).astype(np.float32), jnp.bfloat16)
    tab, _ = _rope_tab_and_ref(q_raw, cs)
    gnope, grope = _packed_gathered_kv(rng, topk // cap, tokens, cap)
    kv_lens = jnp.array([0, 32, 64, 80, 16, 16, 0, 0], dtype=jnp.int32)

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        o3d, l3d = sparse_mla.sparse_attention_chunk(
            q_nope,
            gnope,
            grope,
            kv_lens,
            None,
            None,
            start_pos=0,
            end_pos=tokens,
            token_end=jnp.int32(6),
            sm_scale=0.125,
            k_scale=0.75,
            small_chunk_tokens=16,
            small_chunk_batch_size=4,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            is_dcp_mode=True,
            topk_cap=cap,
            topk_override=topk,
            q_pe=q_raw,
            q_rope_table=tab,
            lse_lane_dense=True,
            lse_mask_empty=True,
            lse_fill=True,
            pre_gathered=(gnope, grope, None),
        )
        o2d, l2d = sparse_mla.sparse_attention_chunk(
            q_nope,
            gnope,
            grope,
            kv_lens,
            None,
            None,
            start_pos=0,
            end_pos=tokens,
            token_end=jnp.int32(6),
            sm_scale=0.125,
            k_scale=0.75,
            small_chunk_tokens=16,
            small_chunk_batch_size=4,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            is_dcp_mode=True,
            topk_cap=cap,
            topk_override=topk,
            q_pe=q_raw.reshape(tokens, heads * 256),
            q_rope_table=tab,
            lse_lane_dense=True,
            lse_mask_empty=True,
            lse_fill=True,
            pre_gathered=(gnope, grope, None),
        )
    np.testing.assert_array_equal(_bits16(o2d), _bits16(o3d))
    np.testing.assert_array_equal(_bits32(l2d), _bits32(l3d))


def test_sparse_ragged_paged_attention_flat_q_pe_validation():
    q = jnp.zeros((16, 8, 512), jnp.bfloat16)
    nope = jnp.zeros((4, 256, 128), jnp.uint32)
    rope = jnp.zeros((4, 64, 128), jnp.uint32)
    topk = jnp.zeros((16, 128), jnp.int32)
    ptab = jnp.zeros((4,), jnp.int32)
    cu = jnp.array([0, 16], jnp.int32)
    dist = jnp.array([0, 0, 1], jnp.int32)
    tab = jnp.zeros((16, 2, 128), jnp.float32)

    # 2D q_pe without q_rope_table must raise.
    with pytest.raises(ValueError, match="q_pe"):
        sparse_mla.sparse_ragged_paged_attention(
            q,
            nope,
            rope,
            topk,
            ptab,
            cu,
            dist,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            q_pe=jnp.zeros((16, 8 * 64), jnp.bfloat16),
        )
    # 2D q_pe with wrong width must raise.
    with pytest.raises(ValueError, match="q_rope_table needs"):
        sparse_mla.sparse_ragged_paged_attention(
            q,
            nope,
            rope,
            topk,
            ptab,
            cu,
            dist,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            q_pe=jnp.zeros((16, 8 * 192), jnp.bfloat16),
            q_rope_table=tab,
        )


def test_combined_4k_tpu7x_mosaic_export_zero_q_relayout():
    """At 4K (`T = 4096, H = 64`), `%arg0` (`bf16[4096, 16384]`) feeds `absorb_q`
    and all 8 `MLA-sparsecore` chunk calls directly with `dense<[1, 0]>` layout
    and zero intermediate reshape/transpose on `%arg0`."""
    info_7x = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1)

    def combined_4k(
        q_flat, w_uk_t, nope_cache, rope_cache, topk_idx, page_tab, cu_q, dist, tab_4k
    ):
        ql_nope = absorb_q(q_flat, w_uk_t)
        out, lse = sparse_mla.sparse_ragged_paged_attention(
            ql_nope,
            nope_cache,
            rope_cache,
            topk_idx,
            page_tab,
            cu_q,
            dist,
            sm_scale=0.125,
            k_scale=0.75,
            return_lse=True,
            is_dcp_mode=True,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            gather_and_attention_chunk_size=512,
            topk_cap=512,
            small_chunk_batch_size=4,
            small_chunk_tokens=16,
            lse_lane_dense=True,
            lse_mask_empty=True,
            fused_window_prep=True,
            mixed_live_gather=True,
            mixed_live_widths=(256, 384),
            mixed_body_levers=(
                sparse_mla.MIXED_BODY_DEFAULT + sparse_mla.MIXED_KERNEL_V2_LEVERS
            ),
            mixed_fused_prep=True,
            q_pe=q_flat,
            q_rope_table=tab_4k,
        )
        return out, lse

    with (
        mock.patch("jax._src.tpu_info.get_tpu_info", return_value=info_7x),
        mock.patch.object(tpu_info, "get_tpu_info", return_value=info_7x),
        mock.patch.object(tpu_info, "is_tpu_device", return_value=True),
        mock.patch.object(pltpu, "get_tpu_info", return_value=info_7x),
    ):
        exp = jax.export.export(jax.jit(combined_4k), platforms=("tpu",))(
            jax.ShapeDtypeStruct((4096, 64 * 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((64, 192, 512), jnp.bfloat16),
            jax.ShapeDtypeStruct((64, 256, 128), jnp.uint32),
            jax.ShapeDtypeStruct((64, 64, 128), jnp.uint32),
            jax.ShapeDtypeStruct((4096, 2048), jnp.int32),
            jax.ShapeDtypeStruct((64,), jnp.int32),
            jax.ShapeDtypeStruct((9,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
            jax.ShapeDtypeStruct((4096, 2, 128), jnp.float32),
        )
    mlir_txt = exp.mlir_module()
    flat_q_lines = [
        ln.strip() for ln in mlir_txt.splitlines() if "tensor<4096x16384xbf16>" in ln
    ]
    custom_calls = [
        ln for ln in flat_q_lines if "stablehlo.custom_call @tpu_custom_call" in ln
    ]
    non_call_ops = [
        ln
        for ln in flat_q_lines
        if "stablehlo.custom_call @tpu_custom_call" not in ln
        and not ln.startswith("func.func ")
        and "call @sparse_ragged_paged_attention(" not in ln
        and "%arg" not in ln.split(":")[0]
    ]
    # 1 absorb_q (`_kernel_flat`) + 8 `MLA-sparsecore` chunk calls = 9 custom calls,
    # and zero reshape/transpose/bitcast ops on `tensor<4096x16384xbf16>`!
    assert len(custom_calls) == 9, len(custom_calls)
    assert non_call_ops == [], non_call_ops
    assert any('kernel_name = "_kernel_flat"' in ln for ln in custom_calls)
    for ln in custom_calls:
        assert "dense<[1, 0]> : tensor<2xindex>" in ln, ln
