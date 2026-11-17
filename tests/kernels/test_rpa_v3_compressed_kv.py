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
"""Validation, VMEM, tuned-cap, and TPU parity tests for compressed-KV RPA v3."""

from __future__ import annotations

from typing import NamedTuple
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.kernels.tpu_test_utils import require_tpu
from tests.layers.core.tq_test_utils import make_tq_layout_and_tables
from vllm_torchtpu.kernels.pallas_util import array_vmem_bytes, get_tpu_info_for_version
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import (
    kernel_compressed_kv,
    tuned_block_sizes_compressed_kv,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import RpaCase
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel_compressed_kv import (
    CompressedKvCodec,
    get_compressed_kv_vmem_estimate_bytes,
    prepare_compressed_kv_block_sizes,
    ragged_paged_attention_bundled,
    static_validate_inputs,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.tuned_block_sizes_compressed_kv import (  # noqa: E501
    get_tuned_max_bkv_csz,
)
from vllm_torchtpu.layers.core import turboquant as tq


def _dummy_dequant(refs, kv_dtype):
    del refs
    z = jnp.zeros((16, 128), dtype=kv_dtype)
    return z, z, None


def _dummy_codec(*, with_update: bool = True) -> CompressedKvCodec:
    return CompressedKvCodec(
        dequant_page_fn=_dummy_dequant,
        update_page_fn=(lambda old_p, *_a, **_kw: old_p) if with_update else None,
        page_size_from_words=lambda _pw: 16,
    )


@pytest.mark.parametrize(
    ("case_id", "exc", "match"),
    [
        ("int_q", ValueError, "floating-point"),
        ("int_kv_dtype", ValueError, "floating-point kv_dtype"),
        ("hd64", ValueError, "multiple of 128"),
        ("missing_new_kv", ValueError, "packed_new_kv and codec.update_page_fn"),
        ("bf16_cache", ValueError, "4D uint8 compressed kv_cache"),
        ("bad_block_sizes", ValueError, "divisible by page_size"),
    ],
)
def test_static_validate_inputs_errors(case_id: str, exc, match: str) -> None:
    s = jax.ShapeDtypeStruct
    base: dict[str, object] = dict(
        queries=s((8, 4, 128), jnp.bfloat16),
        kv_cache=s((16, 2, 24, 128), jnp.uint8),
        kv_lens=s((2,), jnp.int32),
        page_indices=s((4,), jnp.int32),
        cu_q_lens=s((3,), jnp.int32),
        distribution=s((3,), jnp.int32),
        codec=_dummy_codec(),
        kv_dtype=jnp.bfloat16,
        packed_new_kv=s((8, 2, 2, 128), jnp.uint8),
    )
    static_validate_inputs(**base)  # type: ignore[arg-type]
    overrides: dict[str, dict[str, object]] = {
        "int_q": {"queries": s((8, 4, 128), jnp.int32)},
        "int_kv_dtype": {"kv_dtype": jnp.int32},
        "hd64": {"queries": s((8, 4, 64), jnp.bfloat16)},
        "missing_new_kv": {"packed_new_kv": None},
        "bf16_cache": {"kv_cache": s((16, 2, 24, 128), jnp.bfloat16)},
        "bad_block_sizes": {"d_block_sizes": (1, 24, 1, 24)},
    }
    with pytest.raises(exc, match=match):
        static_validate_inputs(**(base | overrides[case_id]))  # type: ignore[arg-type]


def test_bundled_rpa_5d_validation() -> None:
    with pytest.raises(ValueError, match="5D uint8 compressed kv_cache_bundle"):
        ragged_paged_attention_bundled(
            jnp.zeros((4, 4, 128), jnp.bfloat16),
            jnp.zeros((4, 2, 2, 24, 128), jnp.bfloat16),
            jnp.int32(0),
            jnp.array([4], jnp.int32),
            jnp.array([0], jnp.int32),
            jnp.array([0, 4], jnp.int32),
            jnp.array([0, 1, 1], jnp.int32),
            codec=_dummy_codec(with_update=False),
            skip_kv_update=True,
        )


def test_get_tuned_max_bkv_csz_table_and_uncalibrated_logging() -> None:
    assert get_tuned_max_bkv_csz(128, 16, tpu_version=7) == 128
    assert get_tuned_max_bkv_csz(256, 32, tpu_version=7) == 128
    assert get_tuned_max_bkv_csz(256, 64, tpu_version=7) == 256

    with patch.object(
        tuned_block_sizes_compressed_kv.logger, "warning_once"
    ) as mock_warn:
        assert get_tuned_max_bkv_csz(512, 64, tpu_version=7) == 128
        assert get_tuned_max_bkv_csz(512, 16, tpu_version=7) == 64
        assert get_tuned_max_bkv_csz(256, 64, tpu_version=6) == 128
        assert get_tuned_max_bkv_csz(256, 16, tpu_version=6) == 64
        assert mock_warn.call_count == 4


def test_vmem_sublane_alignment_and_block_size_bounds() -> None:
    info = get_tpu_info_for_version(7)
    est_kw = dict(
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        actual_num_q_heads=4,
        actual_num_kv_heads=1,
        head_dim=128,
        page_size=16,
        page_words=24,
        num_new_words=2,
        case=RpaCase.MIXED,
        tpu_version=7,
    )
    with pytest.raises(ValueError, match="page_words"):
        get_compressed_kv_vmem_estimate_bytes(
            bq_sz=8, bkv_sz=64, **(est_kw | {"page_words": 0})
        )
    with pytest.raises(ValueError, match="num_new_words"):
        get_compressed_kv_vmem_estimate_bytes(
            bq_sz=8, bkv_sz=64, **(est_kw | {"num_new_words": 0})
        )

    est = get_compressed_kv_vmem_estimate_bytes(bq_sz=8, bkv_sz=64, **est_kw)
    want = (
        array_vmem_bytes((2, 4, 1, 24, 128), jnp.uint8, info)
        + 2 * array_vmem_bytes((2, 1, 8, 2, 2, 128), jnp.bfloat16, info)
        + 2 * array_vmem_bytes((1, 32, 128), jnp.bfloat16, info)
        + array_vmem_bytes((1, 32, 128), jnp.bfloat16, info)
        + array_vmem_bytes((2, 64, 2, 1, 128), jnp.uint8, info)
    )
    assert est == want

    hd256_est = est_kw | dict(
        actual_num_q_heads=32,
        actual_num_kv_heads=8,
        head_dim=256,
        page_size=64,
        page_words=136,
        num_new_words=5,
    )
    bs_kw = hd256_est | dict(max_num_tokens=2048, max_num_seqs=64, pages_per_seq=64)
    for bad_ver in (-1, 4):
        with pytest.raises(NotImplementedError, match="Unsupported tpu_version"):
            prepare_compressed_kv_block_sizes(
                None, vmem_limit_bytes=64 << 20, **(bs_kw | {"tpu_version": bad_ver})
            )

    bs_v7 = prepare_compressed_kv_block_sizes(
        None,
        vmem_limit_bytes=64 << 20,
        max_bkv_csz=get_tuned_max_bkv_csz(256, 64, tpu_version=7),
        **bs_kw,
    )
    assert bs_v7["bkv_csz"] == 256
    assert get_compressed_kv_vmem_estimate_bytes(
        bq_sz=bs_v7["bq_sz"], bkv_sz=bs_v7["bkv_sz"], **hd256_est
    ) <= int((64 << 20) * 0.75)

    bs_v6_p64 = prepare_compressed_kv_block_sizes(
        None,
        vmem_limit_bytes=64 << 20,
        max_bkv_csz=get_tuned_max_bkv_csz(256, 64, tpu_version=6),
        **(bs_kw | {"tpu_version": 6}),
    )
    assert bs_v6_p64["bkv_csz"] == 128

    bs_v6_p16 = prepare_compressed_kv_block_sizes(
        None,
        vmem_limit_bytes=64 << 20,
        max_bkv_csz=get_tuned_max_bkv_csz(256, 16, tpu_version=6),
        **(bs_kw | dict(page_size=16, page_words=40, pages_per_seq=256, tpu_version=6)),
    )
    assert bs_v6_p16["bkv_csz"] == 64

    with patch.object(kernel_compressed_kv.logger, "warning_once") as mock_warn:
        tiny = prepare_compressed_kv_block_sizes(None, vmem_limit_bytes=1024, **bs_kw)
        assert tiny == {"bq_sz": 1, "bkv_sz": 64, "bq_csz": 1, "bkv_csz": 64}
        mock_warn.assert_called_once()
        assert "tpu_v7" in str(mock_warn.call_args)


class _Case(NamedTuple):
    preset: str
    hd: int
    ps: int
    hkv: int
    gqa: int
    ql: list[int]
    kvl: list[int]
    dist: tuple[int, int, int]
    dtype: str = "bfloat16"
    sw: int | None = None
    cap: float | None = None
    skip: bool = False
    d_bs: tuple[int, int, int, int] | None = None
    m_bs: tuple[int, int, int, int] | None = None


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "c",
    [
        _Case("4bit_nc", 128, 16, 2, 1, [1, 1, 7, 19], [33, 18, 25, 40], (2, 2, 4)),
        _Case(
            "4bit_nc",
            128,
            16,
            2,
            4,
            [1, 9, 21, 5],
            [40, 29, 37, 16],
            (1, 3, 4),
            dtype="float32",
        ),
        _Case("k8v4", 128, 32, 2, 8, [1, 12, 18], [65, 44, 50], (1, 2, 3), cap=30.0),
        _Case("k8v4", 128, 16, 2, 4, [1, 9], [32, 28], (1, 2, 2), skip=True),
        # First-chunk prefill (`q_len == kv_len`, no prior cached pages).
        _Case("4bit_nc", 128, 16, 2, 4, [16, 20], [16, 20], (0, 2, 2)),
        # Multi-block DECODE: full unmasked tiles, partial tail tile, and >1 KV blocks.
        _Case(
            "4bit_nc",
            128,
            16,
            2,
            4,
            [1, 1],
            [97, 50],
            (2, 2, 2),
            d_bs=(1, 32, 1, 16),
        ),
        # Explicit 2-page (`bkv_csz=128`) MSE `page_size=64` case.
        _Case(
            "3bit_nc",
            256,
            64,
            4,
            4,
            [1, 20],
            [260, 280],
            (1, 2, 2),
            d_bs=(1, 256, 1, 128),
            m_bs=(16, 256, 8, 128),
        ),
        # Auto-sized `page_size=64` (`d_bs=None, m_bs=None` -> `bkv_csz=256` on v7).
        _Case("4bit_nc", 256, 64, 2, 4, [1, 24], [300, 320], (1, 2, 2)),
        # `skip_kv_update=True` MSE `page_size=64` case.
        _Case("4bit_nc", 256, 64, 2, 4, [1, 20], [260, 280], (1, 2, 2), skip=True),
        _Case(
            "k3v4_nc", 128, 16, 2, 4, [14, 9], [30, 25], (0, 2, 2), m_bs=(12, 32, 4, 16)
        ),
        _Case(
            "4bit_nc",
            128,
            16,
            2,
            4,
            [1, 80],
            [49, 96],
            (1, 1, 2),
            sw=32,
            m_bs=(16, 32, 8, 16),
        ),
        _Case("4bit_nc", 128, 16, 2, 8, [1, 9], [32, 28], (1, 2, 2), skip=True),
    ],
)
def test_compressed_kv_rpa_parity(c: _Case) -> None:
    require_tpu()
    layout, tables = make_tq_layout_and_tables(c.preset, c.hd)
    qkv_dtype = getattr(jnp, c.dtype)
    hq, n_seqs = c.hkv * c.gqa, len(c.kvl)
    pps = max((kl + c.ps - 1) // c.ps for kl in c.kvl) + 1
    tot_p, pw, sm = n_seqs * pps + 4, layout.page_words_128(c.ps), c.hd**-0.5
    rng = np.random.default_rng(0)
    pi = jnp.asarray(rng.permutation(tot_p)[: n_seqs * pps].astype(np.int32))

    # Start from random bytes so recycled pages and skipped pages are non-zero.
    init_c = jnp.asarray(
        rng.integers(0, 256, size=(tot_p, c.hkv, pw, 128), dtype=np.uint8)
    )
    hist = [kl if c.skip else (kl - ql) for ql, kl in zip(c.ql, c.kvl, strict=True)]
    act = [(i, hl) for i, hl in enumerate(hist[: c.dist[2]]) if hl > 0]
    if act:
        h_cu = jnp.array([0, *np.cumsum([hl for _, hl in act])], jnp.int32)
        ht = int(h_cu[-1])
        h_pi = jnp.concatenate([pi[i * pps : (i + 1) * pps] for i, _ in act])
        init_c, _ = tq.ref_tq_ragged_paged_attention(
            jnp.zeros((ht, hq, c.hd), qkv_dtype),
            jnp.asarray(rng.standard_normal((ht, c.hkv, c.hd)), qkv_dtype),
            jnp.asarray(rng.standard_normal((ht, c.hkv, c.hd)), qkv_dtype),
            init_c,
            jnp.array([hl for _, hl in act], jnp.int32),
            h_pi,
            h_cu,
            jnp.array([0, len(act), len(act)], jnp.int32),
            layout=layout,
            tables=tables,
            sm_scale=sm,
        )

    cu_q = np.array([0, *np.cumsum(c.ql)], np.int32)
    mt = max(int(cu_q[-1]) + 2, 4)
    q = jnp.asarray(rng.standard_normal((mt, hq, c.hd)), qkv_dtype)
    nan_kv = jnp.full((mt, c.hkv, c.hd), jnp.nan, qkv_dtype)
    k = (
        nan_kv
        if c.skip
        else jnp.asarray(rng.standard_normal((mt, c.hkv, c.hd)), qkv_dtype)
    )
    v = (
        nan_kv
        if c.skip
        else jnp.asarray(rng.standard_normal((mt, c.hkv, c.hd)), qkv_dtype)
    )
    meta = (
        jnp.array(c.kvl, jnp.int32),
        pi,
        jnp.asarray(cu_q),
        jnp.array(c.dist, jnp.int32),
    )
    kw = dict(
        layout=layout,
        tables=tables,
        sm_scale=sm,
        sliding_window=c.sw,
        soft_cap=c.cap,
        skip_kv_update=c.skip,
    )
    ref_c, ref_o = tq.ref_tq_ragged_paged_attention(q, k, v, init_c, *meta, **kw)
    got_c, got_o = tq.tq_ragged_paged_attention(
        jnp.copy(q),
        k,
        v,
        jnp.copy(init_c),
        *meta,
        d_block_sizes=c.d_bs,
        m_block_sizes=c.m_bs,
        **kw,
    )
    np.testing.assert_array_equal(np.asarray(got_c), np.asarray(ref_c))
    if c.skip:
        np.testing.assert_array_equal(np.asarray(got_c), np.asarray(init_c))
    nv = int(cu_q[c.dist[2]])
    assert not np.any(np.isnan(np.asarray(got_o[:nv])))
    # Pallas MXU matmul rounds float32 operands to bfloat16 in a single pass (~5e-3).
    tol = 1e-2 if c.dtype == "float32" else 2e-2
    np.testing.assert_allclose(
        np.asarray(got_o[:nv], np.float32),
        np.asarray(ref_o[:nv], np.float32),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("preset", ["4bit_nc", "k8v4"])
def test_decode_vs_spec_verify_parity(preset: str) -> None:
    """Verify sequential `DECODE` steps match a `MIXED` spec-decode verify step."""
    require_tpu()
    layout, tables = make_tq_layout_and_tables(preset, 128)
    ps, hkv, hq, sm = 16, 2, 8, 128**-0.5
    pw = layout.page_words_128(ps)
    rng = np.random.default_rng(19)
    pi = jnp.array([0, 1, 2], jnp.int32)
    init_c = jnp.zeros((4, hkv, pw, 128), jnp.uint8)
    init_c, _ = tq.tq_ragged_paged_attention(
        jnp.asarray(rng.standard_normal((14, hq, 128)), jnp.bfloat16),
        jnp.asarray(rng.standard_normal((14, hkv, 128)), jnp.bfloat16),
        jnp.asarray(rng.standard_normal((14, hkv, 128)), jnp.bfloat16),
        init_c,
        jnp.array([14], jnp.int32),
        pi,
        jnp.array([0, 14], jnp.int32),
        jnp.array([0, 1, 1], jnp.int32),
        layout=layout,
        tables=tables,
        sm_scale=sm,
    )

    q_step = jnp.asarray(rng.standard_normal((4, hq, 128)), jnp.bfloat16)
    k_step = jnp.asarray(rng.standard_normal((4, hkv, 128)), jnp.bfloat16)
    v_step = jnp.asarray(rng.standard_normal((4, hkv, 128)), jnp.bfloat16)

    verify_c, verify_o = tq.tq_ragged_paged_attention(
        jnp.copy(q_step),
        k_step,
        v_step,
        jnp.copy(init_c),
        jnp.array([18], jnp.int32),
        pi,
        jnp.array([0, 4], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
        layout=layout,
        tables=tables,
        sm_scale=sm,
    )

    dec_c = jnp.copy(init_c)
    dec_outs = []
    for step_i in range(4):
        dec_c, o_i = tq.tq_ragged_paged_attention(
            jnp.copy(q_step[step_i : step_i + 1]),
            k_step[step_i : step_i + 1],
            v_step[step_i : step_i + 1],
            dec_c,
            jnp.array([15 + step_i], jnp.int32),
            pi,
            jnp.array([0, 1], jnp.int32),
            jnp.array([1, 1, 1], jnp.int32),
            layout=layout,
            tables=tables,
            sm_scale=sm,
        )
        dec_outs.append(o_i)
    dec_o = jnp.concatenate(dec_outs, axis=0)

    np.testing.assert_array_equal(np.asarray(verify_c), np.asarray(dec_c))
    np.testing.assert_allclose(
        np.asarray(verify_o, np.float32),
        np.asarray(dec_o, np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.tpu_test
def test_tq_jit_cache_hit_without_retracing() -> None:
    """Verify repeat calls (and equal `TQTables`) execute under `jax.no_tracing`."""
    require_tpu()
    layout, tables = make_tq_layout_and_tables("4bit_nc", 128)
    tables_copy = tq.TQTables(
        hadamard=tables.hadamard.copy(),
        centroids=tables.centroids.copy(),
        midpoints=tables.midpoints.copy(),
    )
    npages, hkv, hq, ps = 8, 2, 8, 16
    pw, sm = layout.page_words_128(ps), 128**-0.5
    rng = np.random.default_rng(29)
    meta = (
        jnp.array([17, 14], jnp.int32),
        jnp.array([1, 4, 2, 7], jnp.int32),
        jnp.array([0, 17, 31], jnp.int32),
        jnp.array([0, 2, 2], jnp.int32),
    )
    q1 = jnp.asarray(rng.standard_normal((32, hq, 128)), jnp.bfloat16)
    q2 = jnp.copy(q1)
    k = jnp.asarray(rng.standard_normal((32, hkv, 128)), jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((32, hkv, 128)), jnp.bfloat16)
    c4 = jnp.zeros((npages, hkv, pw, 128), jnp.uint8)

    c4, o1 = tq.tq_ragged_paged_attention(
        q1, k, v, c4, *meta, layout=layout, tables=tables, sm_scale=sm
    )
    jax.block_until_ready(o1)
    with jax.no_tracing():
        c4, o2 = tq.tq_ragged_paged_attention(
            q2, k, v, c4, *meta, layout=layout, tables=tables_copy, sm_scale=sm
        )
    jax.block_until_ready(o2)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("preset", "expect_q_donated"),
    [
        ("turboquant_4bit_nc", False),
        ("turboquant_k8v4", True),
    ],
)
def test_tq_bundled_layers_and_buffer_donation(
    preset: str, expect_q_donated: bool
) -> None:
    """Verify KV-cache/queries buffer donation (`is_deleted()`) and 5D bundle parity."""
    require_tpu()
    layout, tables = make_tq_layout_and_tables(preset, 128)
    npages, nlayers, hkv, hq, ps = 12, 4, 2, 8, 16
    pw, sm = layout.page_words_128(ps), 128**-0.5
    rng = np.random.default_rng(31)
    meta = (
        jnp.array([17, 14], jnp.int32),
        jnp.array([1, 4, 2, 7], jnp.int32),
        jnp.array([0, 17, 31], jnp.int32),
        jnp.array([0, 2, 2], jnp.int32),
    )
    qs = jnp.asarray(rng.standard_normal((3, 32, hq, 128)), jnp.bfloat16)
    ks = jnp.asarray(rng.standard_normal((3, 32, hkv, 128)), jnp.bfloat16)
    vs = jnp.asarray(rng.standard_normal((3, 32, hkv, 128)), jnp.bfloat16)
    kw = dict(layout=layout, tables=tables, sm_scale=sm)

    @jax.jit(donate_argnums=(0,))
    def run_layers(b, q_a, k_a, v_a):
        outs = []
        for idx, lid in enumerate((0, 2, 3)):
            b, o = tq.tq_ragged_paged_attention(
                q_a[idx], k_a[idx], v_a[idx], b, *meta, layer_idx=jnp.int32(lid), **kw
            )
            outs.append(o)
        return b, outs

    b_donated = jnp.zeros((npages, nlayers, hkv, pw, 128), jnp.uint8)
    b_out, outs = run_layers(b_donated, qs, ks, vs)
    jax.block_until_ready(b_out)
    assert b_donated.is_deleted()
    assert not np.any(np.asarray(b_out[:, 1]))

    for idx, lid in enumerate((0, 2, 3)):
        c_single = jnp.zeros((npages, hkv, pw, 128), jnp.uint8)
        q_single = jnp.copy(qs[idx])
        sc, so = tq.tq_ragged_paged_attention(
            q_single, ks[idx], vs[idx], c_single, *meta, **kw
        )
        jax.block_until_ready(sc)
        assert c_single.is_deleted()
        assert q_single.is_deleted() is expect_q_donated
        np.testing.assert_array_equal(np.asarray(b_out[:, lid]), np.asarray(sc))
        np.testing.assert_array_equal(np.asarray(outs[idx][:31]), np.asarray(so[:31]))
