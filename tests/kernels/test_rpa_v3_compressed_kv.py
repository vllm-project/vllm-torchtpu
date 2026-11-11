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
"""Validation, VMEM, and TPU parity tests for compressed-KV RPA v3."""

from __future__ import annotations

import functools
from typing import NamedTuple
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel_compressed_kv
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    RpaCase,
    get_vmem_estimate_bytes,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel_compressed_kv import (
    align_second_minor_sublanes,
    get_compressed_kv_vmem_estimate_bytes,
    prepare_compressed_kv_block_sizes,
    ragged_paged_attention,
    ragged_paged_attention_bundled,
    static_validate_inputs,
)


def _layout_and_tables(preset: str, head_dim: int):
    from vllm.model_executor.layers.quantization.turboquant.centroids import (
        get_centroids,
    )
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

    from vllm_torchtpu.layers.core import turboquant as tq

    cfg = TurboQuantConfig.from_cache_dtype(f"turboquant_{preset}", head_dim)
    layout = tq.TQLayout.from_tq_config(cfg)
    return layout, tq.make_tables(
        layout, get_centroids(head_dim, cfg.centroid_bits).numpy()
    )


def _dummy_dequant(refs, kv_dtype, *, return_k_norm: bool = False):
    del refs
    z = jnp.zeros((16, 128), dtype=kv_dtype)
    return (z, z, None) if return_k_norm else (z, z)


@pytest.mark.parametrize(
    ("case_id", "exc", "match"),
    [
        ("fused_swap", NotImplementedError, "fuse_non_tiling_axis_swap"),
        ("int_q", ValueError, "floating-point"),
        ("kv_dtype_mismatch", ValueError, r"k\.dtype.*v\.dtype"),
        ("hd64", ValueError, "multiple of 128"),
        ("missing_new_kv", ValueError, "packed_new_kv and update_page_fn"),
        ("bf16_cache", ValueError, "4D uint8 compressed kv_cache"),
        ("bad_block_sizes", ValueError, "divisible by page_size"),
    ],
)
def test_static_validate_inputs_errors(case_id: str, exc, match: str) -> None:
    s = jax.ShapeDtypeStruct
    base: dict[str, object] = dict(
        queries=s((8, 4, 128), jnp.bfloat16),
        keys=s((8, 2, 128), jnp.bfloat16),
        values=s((8, 2, 128), jnp.bfloat16),
        kv_cache=s((16, 2, 24, 128), jnp.uint8),
        kv_lens=s((2,), jnp.int32),
        page_indices=s((4,), jnp.int32),
        cu_q_lens=s((3,), jnp.int32),
        distribution=s((3,), jnp.int32),
        page_size=16,
        dequant_page_fn=_dummy_dequant,
        packed_new_kv=s((8, 2, 2, 128), jnp.uint8),
        update_page_fn=lambda old_p, *_a, **_kw: old_p,
    )
    static_validate_inputs(**base)  # type: ignore[arg-type]
    q64, k64 = s((8, 4, 64), jnp.bfloat16), s((8, 2, 64), jnp.bfloat16)
    overrides: dict[str, dict[str, object]] = {
        "fused_swap": {"fuse_non_tiling_axis_swap": True},
        "int_q": {"queries": s((8, 4, 128), jnp.int32)},
        "kv_dtype_mismatch": {"values": s((8, 2, 128), jnp.float32)},
        "hd64": {"queries": q64, "keys": k64, "values": k64},
        "missing_new_kv": {"packed_new_kv": None},
        "bf16_cache": {"kv_cache": s((16, 2, 24, 128), jnp.bfloat16)},
        "bad_block_sizes": {"d_block_sizes": (1, 24, 1, 24)},
    }
    with pytest.raises(exc, match=match):
        static_validate_inputs(**(base | overrides[case_id]))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("rows", "dtype_name", "expected"),
    [
        (1, "uint8", 4),
        (12, "uint8", 16),
        (33, "uint8", 64),
        (3, "bfloat16", 4),
        (9, "float32", 16),
    ],
)
def test_align_second_minor_sublanes(rows: int, dtype_name: str, expected: int) -> None:
    assert align_second_minor_sublanes(rows, getattr(jnp, dtype_name)) == expected


def test_vmem_sublane_alignment_and_block_size_bounds() -> None:
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
    )
    est = get_compressed_kv_vmem_estimate_bytes(bq_sz=8, bkv_sz=64, **est_kw)
    base = get_vmem_estimate_bytes(
        1, 4, 128, 8, 64, jnp.bfloat16, jnp.bfloat16, out_dtype=jnp.bfloat16
    )
    assert est == base + 2 * 4 * 1 * 32 * 128 + 2 * 64 * 2 * 4 * 128

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
                None, vmem_limit_bytes=64 << 20, tpu_version=bad_ver, **bs_kw
            )
    bs = prepare_compressed_kv_block_sizes(
        None, vmem_limit_bytes=64 << 20, tpu_version=7, **bs_kw
    )
    assert get_compressed_kv_vmem_estimate_bytes(
        bq_sz=bs["bq_sz"], bkv_sz=bs["bkv_sz"], **hd256_est
    ) <= int((64 << 20) * 0.75)

    with patch.object(kernel_compressed_kv.logger, "warning_once") as mock_warn:
        tiny = prepare_compressed_kv_block_sizes(
            None, vmem_limit_bytes=1024, tpu_version=7, **bs_kw
        )
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
    sw: int | None = None
    cap: float | None = None
    skip: bool = False
    m_bs: tuple[int, int, int, int] | None = None


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "c",
    [
        _Case("4bit_nc", 128, 16, 2, 1, [1, 1, 7, 19], [33, 18, 25, 40], (2, 2, 4)),
        _Case("4bit_nc", 128, 16, 2, 4, [1, 9, 21, 5], [40, 29, 37, 16], (1, 3, 4)),
        _Case("k8v4", 128, 32, 2, 8, [1, 12, 18], [65, 44, 50], (1, 2, 3), cap=30.0),
        _Case("3bit_nc", 256, 64, 8, 4, [1, 20], [65, 84], (1, 2, 2)),
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
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected TPU backend, got {jax.default_backend()}.")
    from vllm_torchtpu.layers.core import turboquant as tq

    layout, tables = _layout_and_tables(c.preset, c.hd)
    hq, n_seqs = c.hkv * c.gqa, len(c.kvl)
    pps = max((kl + c.ps - 1) // c.ps for kl in c.kvl) + 1
    tot_p, pw, sm = n_seqs * pps + 4, layout.page_words_128(c.ps), c.hd**-0.5
    rng = np.random.default_rng(0)
    pi = jnp.asarray(rng.permutation(tot_p)[: n_seqs * pps].astype(np.int32))

    hist = [kl if c.skip else (kl - ql) for ql, kl in zip(c.ql, c.kvl, strict=True)]
    init_c = jnp.zeros((tot_p, c.hkv, pw, 128), jnp.uint8)
    act = [(i, hl) for i, hl in enumerate(hist[: c.dist[2]]) if hl > 0]
    if act:
        h_cu = jnp.array([0, *np.cumsum([hl for _, hl in act])], jnp.int32)
        ht = int(h_cu[-1])
        h_pi = jnp.concatenate([pi[i * pps : (i + 1) * pps] for i, _ in act])
        init_c, _ = tq.ref_tq_ragged_paged_attention(
            jnp.zeros((ht, hq, c.hd), jnp.bfloat16),
            jnp.asarray(rng.standard_normal((ht, c.hkv, c.hd)), jnp.bfloat16),
            jnp.asarray(rng.standard_normal((ht, c.hkv, c.hd)), jnp.bfloat16),
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
    q = jnp.asarray(rng.standard_normal((mt, hq, c.hd)), jnp.bfloat16)
    nan_kv = jnp.full((mt, c.hkv, c.hd), jnp.nan, jnp.bfloat16)
    k = (
        nan_kv
        if c.skip
        else jnp.asarray(rng.standard_normal((mt, c.hkv, c.hd)), jnp.bfloat16)
    )
    v = (
        nan_kv
        if c.skip
        else jnp.asarray(rng.standard_normal((mt, c.hkv, c.hd)), jnp.bfloat16)
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
    rpa_fn = (
        functools.partial(ragged_paged_attention, m_block_sizes=c.m_bs)
        if c.m_bs
        else None
    )
    got_c, got_o = tq.tq_ragged_paged_attention(
        jnp.copy(q), k, v, jnp.copy(init_c), *meta, rpa_fn=rpa_fn, **kw
    )
    np.testing.assert_array_equal(np.asarray(got_c), np.asarray(ref_c))
    if c.skip:
        np.testing.assert_array_equal(np.asarray(got_c), np.asarray(init_c))
    nv = int(cu_q[c.dist[2]])
    assert not np.any(np.isnan(np.asarray(got_o[:nv])))
    np.testing.assert_allclose(
        np.asarray(got_o[:nv], np.float32),
        np.asarray(ref_o[:nv], np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.tpu_test
def test_jit_cache_bundled_donation_and_shard_map() -> None:
    """Verify JIT cache hit on repeat calls, multi-layer donation, and `shard_map`."""
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected TPU backend, got {jax.default_backend()}.")
    from vllm_torchtpu.layers.core import turboquant as tq

    layout, tables = _layout_and_tables("4bit_nc", 128)
    npages, nlayers, hkv, hq, ps = 12, 4, 2, 8, 16
    pw, sm = layout.page_words_128(ps), 128**-0.5
    rng = np.random.default_rng(29)
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

    c4 = jnp.zeros((npages, hkv, pw, 128), jnp.uint8)
    c4, o1 = tq.tq_ragged_paged_attention(
        jnp.copy(qs[0]), ks[0], vs[0], c4, *meta, **kw
    )
    jax.block_until_ready(o1)
    n_cached = ragged_paged_attention._cache_size()
    c4, o2 = tq.tq_ragged_paged_attention(
        jnp.copy(qs[0]), ks[0], vs[0], c4, *meta, **kw
    )
    jax.block_until_ready(o2)
    assert ragged_paged_attention._cache_size() == n_cached

    b_init = jnp.zeros((npages, nlayers, hkv, pw, 128), jnp.uint8)

    @jax.jit(donate_argnums=(0,))
    def run_layers(b, q_a, k_a, v_a):
        outs = []
        for idx, lid in enumerate((0, 2, 3)):
            b, o = tq.tq_ragged_paged_attention(
                q_a[idx], k_a[idx], v_a[idx], b, *meta, layer_idx=jnp.int32(lid), **kw
            )
            outs.append(o)
        return b, outs

    b_out, outs = run_layers(jnp.copy(b_init), qs, ks, vs)
    assert not np.any(np.asarray(b_out[:, 1]))
    for idx, lid in enumerate((0, 2, 3)):
        sc, so = tq.tq_ragged_paged_attention(
            jnp.copy(qs[idx]), ks[idx], vs[idx], jnp.zeros_like(c4), *meta, **kw
        )
        np.testing.assert_array_equal(np.asarray(b_out[:, lid]), np.asarray(sc))
        np.testing.assert_array_equal(np.asarray(outs[idx][:31]), np.asarray(so[:31]))

    tp = min(len(jax.devices()), hkv)
    mesh = Mesh(np.array(jax.devices()[:tp]), ("tp",))
    q_in, k_in, v_in, deq_fn, p_new, upd_fn = tq._prepare_tq_attention_inputs(
        qs[0],
        ks[0],
        vs[0],
        layout=layout,
        tables=tables,
        page_size=ps,
        skip_kv_update=False,
    )
    assert p_new is not None and upd_fn is not None
    sharded = jax.shard_map(
        lambda q_s, k_s, v_s, b_s, p_s, l_s, kl_s, pi_s, cql_s, d_s: (
            ragged_paged_attention_bundled(
                q_s,
                k_s,
                v_s,
                b_s,
                l_s,
                kl_s,
                pi_s,
                cql_s,
                d_s,
                page_size=ps,
                dequant_page_fn=deq_fn,
                packed_new_kv=p_s,
                update_page_fn=upd_fn,
                sm_scale=sm,
            )
        ),
        mesh=mesh,
        in_specs=(
            P(None, "tp", None),
            P(None, "tp", None),
            P(None, "tp", None),
            P(None, None, "tp", None, None),
            P(None, None, "tp", None),
            P(),
            P(None),
            P(None),
            P(None),
            P(None),
        ),
        out_specs=(P(None, "tp", None), P(None, None, "tp", None, None)),
        check_vma=False,
    )
    sm_o, sm_b = sharded(
        jnp.copy(q_in), k_in, v_in, jnp.copy(b_init), p_new, jnp.int32(1), *meta
    )
    np.testing.assert_array_equal(np.asarray(sm_b[:, 1]), np.asarray(b_out[:, 0]))
    np.testing.assert_array_equal(np.asarray(sm_o[:31]), np.asarray(outs[0][:31]))
