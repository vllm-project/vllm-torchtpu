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
"""Unit and TPU parity tests for RPA v3 staging, pipeline, launch, and tile helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.kernels.tpu_test_utils import require_tpu
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import (
    flash_attention,
    kernel,
    kernel_hd64,
    launch,
    pipeline,
    staging,
    util,
)


@pytest.mark.parametrize(
    ("dtype_name", "pages_per_seq", "page_size", "tpu_version", "want_i16"),
    [
        ("bfloat16", 2047, 16, 6, True),
        # 2048 * 16 == 32768 > int16.max (32767) -> must select int32.
        ("bfloat16", 2048, 16, 6, False),
        ("bfloat16", 2049, 16, 7, False),
        ("float32", 64, 16, 7, False),
        ("bfloat16", 64, 16, 5, False),
        ("bfloat16", 64, 16, -1, False),
    ],
)
def test_get_span_int_dtype(
    dtype_name: str,
    pages_per_seq: int,
    page_size: int,
    tpu_version: int,
    want_i16: bool,
) -> None:
    q_dtype = getattr(jnp, dtype_name)
    got = util.get_span_int_dtype(
        q_dtype, pages_per_seq, page_size, tpu_version=tpu_version
    )
    assert got == (jnp.int16 if want_i16 else jnp.int32)


@pytest.mark.parametrize(
    ("case_name", "expected"),
    [
        ("DECODE", (0, 2)),
        ("PREFILL", (2, 5)),
        ("MIXED", (5, 7)),
    ],
)
def test_rpa_case_get_range(case_name: str, expected: tuple[int, int]) -> None:
    dist = jnp.array([2, 5, 7], dtype=jnp.int32)
    lo, hi = getattr(util.RpaCase, case_name).get_range(dist)
    assert (int(lo), int(hi)) == expected


def test_vmem_estimates_match_scratch_shapes() -> None:
    n_kv, n_q_per_kv, head_dim, bq_sz, bkv_sz = 4, 4, 128, 16, 64
    got_f32 = kernel.get_vmem_estimate_bytes(
        n_kv,
        n_q_per_kv,
        head_dim,
        bq_sz,
        bkv_sz,
        jnp.bfloat16,
        jnp.bfloat16,
        out_dtype=jnp.float32,
    )
    # Default for bf16 queries is out_dtype=jnp.bfloat16 (matching kernel launch).
    got_bf16_default = kernel.get_vmem_estimate_bytes(
        n_kv, n_q_per_kv, head_dim, bq_sz, bkv_sz, jnp.bfloat16, jnp.bfloat16
    )
    got_bf16 = kernel.get_vmem_estimate_bytes(
        n_kv,
        n_q_per_kv,
        head_dim,
        bq_sz,
        bkv_sz,
        jnp.bfloat16,
        jnp.bfloat16,
        out_dtype=jnp.bfloat16,
    )
    assert got_bf16_default == got_bf16

    # n_kv=4 -> bkv_stride = 4 + 1 = 5 (bank-conflict padded).
    want_f32 = (
        2 * bkv_sz * 5 * 2 * head_dim * 2
        + 2 * (2 * n_kv * bq_sz * n_q_per_kv * head_dim * 2)
        + 2 * (n_kv * bq_sz * n_q_per_kv * 128 * 4)
        + (n_kv * bq_sz * n_q_per_kv * head_dim * 4)
    )
    scratch_elems = n_kv * bq_sz * n_q_per_kv * (2 * 128 + head_dim)
    assert got_f32 == want_f32
    assert got_bf16 == want_f32 - scratch_elems * 2

    for out_dt, got in ((jnp.float32, got_f32), (jnp.bfloat16, got_bf16)):
        shapes = launch.make_base_rpa_scratch_shapes(
            q_tail_shape=(n_q_per_kv // 2, 2, head_dim),
            q_dtype=jnp.bfloat16,
            kv_tail_shape=(2, head_dim),
            kv_dtype=jnp.bfloat16,
            out_dtype=out_dt,
            actual_num_kv_heads=n_kv,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            kv_rows_per_token=n_kv,
        )
        assert got == launch.get_scratch_vmem_bytes(shapes)

    got_hd64 = kernel_hd64.get_vmem_estimate_bytes(
        n_kv, n_q_per_kv, 64, bq_sz, bkv_sz, jnp.bfloat16, jnp.bfloat16
    )
    want_hd64 = (
        2 * bkv_sz * n_kv * 128 * 2
        + 2 * (2 * n_kv * bq_sz * n_q_per_kv * 128 * 2)
        + 2 * (n_kv * bq_sz * n_q_per_kv * 128 * 4)
        + (n_kv * bq_sz * n_q_per_kv * 128 * 4)
    )
    assert got_hd64 == want_hd64


@pytest.mark.parametrize(
    ("acc_dtype_name", "out_dtype_name"),
    [
        ("float32", "float32"),
        ("bfloat16", "bfloat16"),
        ("float32", "bfloat16"),
    ],
)
def test_normalize_bo(
    acc_dtype_name: str, out_dtype_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: SimpleNamespace(num_lanes=128))
    used_approx_recip: list[bool] = []
    monkeypatch.setattr(
        pltpu,
        "reciprocal",
        lambda x, *, approx=False: (
            used_approx_recip.append(approx),
            jnp.reciprocal(x),
        )[1],
    )
    acc_dt = getattr(jnp, acc_dtype_name)
    out_dt = getattr(jnp, out_dtype_name)
    acc = jnp.arange(1, 2 * 4 * 256 + 1, dtype=jnp.float32).reshape(2, 4, 256) * 0.01
    denom = jnp.full((2, 4, 128), 2.0, dtype=acc_dt)
    acc_ref = jax.new_ref(acc.astype(acc_dt))
    l_ref = jax.new_ref(denom)
    got = staging.normalize_bo(acc_ref, l_ref, out_dtype=out_dt)
    assert got.dtype == out_dt
    assert used_approx_recip == (
        [True] if (acc_dt == jnp.float32 and out_dt != jnp.float32) else []
    )
    want = (acc.astype(acc_dt) / 2.0).astype(out_dt)
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float32),
        np.asarray(want, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("tpu_version", [5, 6, 7, -1, 4])
def test_get_default_block_sizes(tpu_version: int) -> None:
    if tpu_version in (5, 6, 7):
        bs = kernel.get_default_block_sizes(
            jnp.bfloat16,
            jnp.bfloat16,
            8,
            2,
            128,
            16,
            64,
            8,
            8,
            tpu_version=tpu_version,
        )
        assert bs["bkv_sz"] % 16 == 0 and bs["bkv_csz"] % 16 == 0
        assert 1 <= bs["bq_csz"] <= bs["bq_sz"]
    else:
        with pytest.raises(NotImplementedError, match="Unsupported tpu_version"):
            kernel.get_default_block_sizes(
                jnp.bfloat16,
                jnp.bfloat16,
                8,
                2,
                128,
                16,
                64,
                8,
                8,
                tpu_version=tpu_version,
            )


@pytest.mark.parametrize(
    ("override", "err_match"),
    [
        ({}, None),
        ({"q_shape": (8, 4)}, "Expected 3D array"),
        ({"q_dtype": "int32"}, "floating-point"),
        ({"k_dtype": "float32"}, r"k\.dtype.*v\.dtype"),
        ({"q_shape": (8, 5, 128)}, "divisible by"),
        ({"block_sizes": (("m", (8, 24, 8, 24)),)}, "divisible by page_size"),
        ({"sliding_window": 0}, "sliding_window"),
    ],
)
def test_validate_common_inputs(
    override: dict[str, object], err_match: str | None
) -> None:
    s = jax.ShapeDtypeStruct
    q_dt = getattr(jnp, str(override.get("q_dtype", "bfloat16")))
    k_dt = getattr(jnp, str(override.get("k_dtype", "bfloat16")))
    q = s(override.get("q_shape", (8, 4, 128)), q_dt)  # type: ignore[arg-type]
    k = s((8, 2, 128), k_dt)
    v = s((8, 2, 128), jnp.bfloat16)
    meta = (
        s((2,), jnp.int32),
        s((4,), jnp.int32),
        s((3,), jnp.int32),
        s((3,), jnp.int32),
    )
    kwargs: dict[str, object] = dict(
        page_size=16,
        kv_packing=2,
        sliding_window=override.get("sliding_window"),
        block_sizes_by_name=override.get("block_sizes", ()),
    )
    if err_match is None:
        util.validate_common_inputs(q, k, v, *meta, **kwargs)  # type: ignore[arg-type]
    else:
        with pytest.raises(ValueError, match=err_match):
            util.validate_common_inputs(q, k, v, *meta, **kwargs)  # type: ignore[arg-type]


def test_validate_dynamic_common_inputs() -> None:
    q = jnp.zeros((8, 4, 128), jnp.bfloat16)
    kv_cache = jnp.zeros((4, 16, 2, 2, 128), jnp.bfloat16)
    kv_lens = jnp.array([16, 20], jnp.int32)
    page_indices = jnp.array([0, 1, 2, 3], jnp.int32)
    cu_q_lens = jnp.array([0, 1, 5], jnp.int32)
    util.validate_dynamic_common_inputs(
        q, kv_cache, kv_lens, page_indices, cu_q_lens, jnp.array([1, 1, 2], jnp.int32)
    )
    with pytest.raises(ValueError, match="Invalid distribution"):
        util.validate_dynamic_common_inputs(
            q,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([2, 1, 2], jnp.int32),
        )


@pytest.mark.parametrize(
    ("bq_idx", "bq_sem", "num_bq", "expected"),
    [(0, 0, 3, (2, 1, 1)), (2, 1, 3, (3, 0, 0))],
)
def test_get_next_bq_ids(
    bq_idx: int, bq_sem: int, num_bq: int, expected: tuple[int, int, int]
) -> None:
    got = pipeline.get_next_bq_ids(
        jnp.int32(2), jnp.int32(bq_idx), jnp.int32(bq_sem), num_bq=num_bq
    )
    assert tuple(int(x) for x in got) == expected


@pytest.mark.parametrize(
    ("bq_idx", "bkv_idx", "sliding_window", "expected"),
    [
        (0, 1, None, (4, 0, 2, 1)),
        # End of bkv for bq_idx=0 with sliding_window=32: next bq starts at
        # (48 - 32) // 16 = 1.
        (0, 2, 32, (4, 1, 1, 1)),
        # End of last bq (bq_idx=1 of 2): advances to seq 5 at
        # next_seq_start_bkv_idx = 3.
        (1, 2, 32, (5, 0, 3, 1)),
    ],
)
def test_get_next_bkv_ids(
    bq_idx: int,
    bkv_idx: int,
    sliding_window: int | None,
    expected: tuple[int, int, int, int],
) -> None:
    got = pipeline.get_next_bkv_ids(
        jnp.int32(4),
        jnp.int32(bq_idx),
        jnp.int32(bkv_idx),
        jnp.int32(0),
        num_bq=2,
        num_bkv=3,
        kv_q_gap=jnp.int32(32),
        actual_bq_sz=16,
        bkv_sz=16,
        sliding_window=sliding_window,
        next_seq_start_bkv_idx=jnp.int32(3),
    )
    assert tuple(int(x) for x in got) == expected


@pytest.mark.parametrize(
    ("span_dtype_name", "processed_q", "tile_starts", "effective_kv", "sliding_window"),
    [
        ("int32", 4, (0, 4), 6, None),
        ("int16", 1024, (894, 898), 902, 128),
    ],
)
def test_flash_attention_steps_cpu(
    span_dtype_name: str,
    processed_q: int,
    tile_starts: tuple[int, int],
    effective_kv: int,
    sliding_window: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: SimpleNamespace(num_lanes=128))
    rng = np.random.default_rng(0)
    nq_per_kv, q_tokens, bkv_csz, head_dim = 2, 2, 4, 128
    q_rows = q_tokens * nq_per_kv
    q = jnp.asarray(rng.standard_normal((q_rows, head_dim), dtype=np.float32))
    ks = [
        jnp.asarray(rng.standard_normal((bkv_csz, head_dim), dtype=np.float32))
        for _ in tile_starts
    ]
    vs = [
        jnp.asarray(rng.standard_normal((bkv_csz, head_dim), dtype=np.float32))
        for _ in tile_starts
    ]
    l_ref = jax.new_ref(jnp.zeros((q_rows, 128), dtype=jnp.float32))
    m_ref = jax.new_ref(jnp.full((q_rows, 128), -jnp.inf, dtype=jnp.float32))
    o_ref = jax.new_ref(jnp.zeros((q_rows, head_dim), dtype=jnp.float32))
    span_ty = getattr(jnp, span_dtype_name)
    mask_val = float(jnp.finfo(jnp.float32).min)
    cfg = flash_attention.FlashAttentionConfig(
        num_q_heads_per_kv_head=nq_per_kv,
        sm_scale=0.125,
        out_dtype=jnp.float32,
        mask_value=mask_val,
        span_int_ty=span_ty,
        sliding_window=sliding_window,
    )

    with jax.default_matmul_precision("float32"):
        for t_start, k_tile, v_tile in zip(tile_starts, ks, vs, strict=True):
            p, v_masked, exp_m_diff = flash_attention.flash_attention_step1_qk_softmax(
                q,
                k_tile,
                v_tile,
                l_ref,
                m_ref,
                processed_q_len=processed_q,
                processed_kv_len=t_start,
                effective_kv_len=effective_kv,
                cfg=cfg,
            )
            flash_attention.flash_attention_step2_pv(
                p, v_masked, exp_m_diff, o_ref, cfg=cfg
            )

    got = np.asarray(o_ref[...] / l_ref[:, :1])
    k_all = np.concatenate([np.asarray(x) for x in ks], axis=0)
    v_all = np.concatenate([np.asarray(x) for x in vs], axis=0)
    k_pos = np.concatenate(
        [np.arange(s, s + bkv_csz) for s in tile_starts], axis=0
    )  # [2 * bkv_csz]
    q_pos = processed_q + (np.arange(q_rows) // nq_per_kv)[:, None]
    scores = (np.asarray(q) @ k_all.T) * 0.125
    valid = (k_pos[None, :] < effective_kv) & (q_pos >= k_pos[None, :])
    if sliding_window is not None:
        valid &= q_pos < k_pos[None, :] + sliding_window
    scores = np.where(valid, scores, mask_val)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    ref = probs @ np.where((k_pos < effective_kv)[:, None], v_all, 0.0)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_scalar_prefetches_and_bundled_rpa() -> None:
    kl, pi = jnp.array([16, 32], jnp.int32), jnp.array([0, 2, 1, 3], jnp.int32)
    cql, dist = jnp.array([0, 1, 5], jnp.int32), jnp.array([1, 1, 2], jnp.int32)
    sp = launch.make_scalar_prefetches(kl, pi, cql, dist)
    assert [x.shape for x in sp[4:]] == [(3,), (4,), (6,)]
    assert int(sp[5][0]) == -1 and int(sp[6][0]) == -1

    bundle = jnp.arange(24, dtype=jnp.float32).reshape(4, 3, 2)
    seen: dict[str, object] = {}

    def fake_rpa(q_a, _k, _v, fc, _kl, fpi, _cql, _d, *, sm_scale: float = 1.0):
        seen["shape"], seen["pi"] = fc.shape, np.asarray(fpi)
        return q_a * sm_scale, fc.at[fpi[0]].set(-1.0)

    qa, ka = jnp.ones((5, 4, 128), jnp.float32), jnp.ones((5, 2, 128), jnp.float32)
    out, new_b = util.run_bundled_rpa(
        fake_rpa, qa, ka, ka, bundle, jnp.int32(2), kl, pi, cql, dist, sm_scale=0.5
    )
    assert seen["shape"] == (12, 2) and float(out[0, 0, 0]) == 0.5
    np.testing.assert_array_equal(seen["pi"], np.array([2, 8, 5, 11]))
    np.testing.assert_array_equal(new_b[0, 2], np.array([-1.0, -1.0]))
    np.testing.assert_array_equal(new_b[0, 0], bundle[0, 0])


def test_invoke_rpa_pallas_kernel_vmem_blockspec_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `pin_hbm=True` constrains HBM inputs and skips VMEM/None inputs."""
    constrained: list[tuple[int, object]] = []
    captured: dict[str, object] = {}

    def fake_constraint(arr, space):
        constrained.append((arr.shape[0], space))
        return arr

    def fake_pallas_call(fn, *, grid_spec, compiler_params, out_shape, **kw):
        del fn, compiler_params, kw
        captured["in_specs"] = grid_spec.in_specs
        captured["out_shape"] = out_shape

        def _runner(*args):
            return args[1], args[2]

        return _runner

    monkeypatch.setattr(pltpu, "with_memory_space_constraint", fake_constraint)
    monkeypatch.setattr(pl, "pallas_call", fake_pallas_call)
    q = jnp.ones((8, 128), jnp.float32)
    cache = jnp.ones((16, 128), jnp.float32)
    sink = jnp.ones((8, 128), jnp.float32)
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.VMEM),
        None,
    ]
    out_q, out_c = launch.invoke_rpa_pallas_kernel(
        kernel_fn=lambda *_a: None,
        scalar_prefetches=(jnp.zeros((1,), jnp.int32),),
        inputs=(q, cache, sink, None),
        scratch_shapes=[],
        out_arrays=(q, cache),
        input_output_aliases={},
        vmem_limit_bytes=1 << 20,
        scope_name="test_vmem_spec",
        tpu_version=7,
        in_specs=in_specs,
        disable_bounds_checks=False,
        disable_semaphore_checks=False,
        pin_hbm=True,
    )
    assert out_q.shape == q.shape and out_c.shape == cache.shape
    # Only the two HBM inputs (`q` of len 8 and `cache` of len 16) are constrained.
    assert constrained == [(8, pltpu.HBM), (16, pltpu.HBM)]


class _ParityCase(NamedTuple):
    name: str
    is_hd64: bool
    head_dim: int
    q_dtype: str
    kv_dtype: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    dist: tuple[int, int, int]
    kwargs: dict[str, object]
    atol: float
    use_sink: bool = False


_PARITY_CASES = (
    # Exercises sub-tile loops (`bq_csz < bq_sz`, `bkv_csz < bkv_sz`),
    # `DECODE` + `PREFILL` + `MIXED`, and `sliding_window` skipping leading KV blocks.
    _ParityCase(
        name="bf16_subtiles_sliding_window",
        is_hd64=False,
        head_dim=128,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        q_lens=(1, 16, 20),
        kv_lens=(112, 112, 116),
        dist=(1, 2, 3),
        kwargs=dict(
            sliding_window=32,
            chunk_prefill_size=16,
            d_block_sizes=(1, 32, 1, 16),
            p_block_sizes=(16, 32, 8, 16),
            m_block_sizes=(16, 32, 8, 16),
        ),
        atol=2e-2,
    ),
    # Exercises `head_dim=256` (`folds == 2`) in `strided_load` and `strided_store`.
    _ParityCase(
        name="bf16_hd256_multifold",
        is_hd64=False,
        head_dim=256,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        q_lens=(1, 20),
        kv_lens=(48, 64),
        dist=(1, 1, 2),
        kwargs=dict(
            d_block_sizes=(1, 32, 1, 16),
            m_block_sizes=(16, 32, 8, 16),
        ),
        atol=2e-2,
    ),
    # Exercises `kv_packing == 1` (`float32`) in `load_bkv` and sub-tile loops.
    _ParityCase(
        name="f32_unpacked_subtiles",
        is_hd64=False,
        head_dim=128,
        q_dtype="float32",
        kv_dtype="float32",
        q_lens=(1, 20),
        kv_lens=(32, 52),
        dist=(1, 1, 2),
        kwargs=dict(
            d_block_sizes=(1, 32, 1, 16),
            m_block_sizes=(16, 32, 8, 16),
        ),
        atol=1e-3,
    ),
    # Exercises `float8_e4m3fn` KV cache with explicit `k_scale` and `v_scale`.
    _ParityCase(
        name="fp8_scaled",
        is_hd64=False,
        head_dim=128,
        q_dtype="bfloat16",
        kv_dtype="float8_e4m3fn",
        q_lens=(1, 20),
        kv_lens=(32, 52),
        dist=(1, 1, 2),
        kwargs=dict(
            k_scale=0.5,
            v_scale=0.5,
            d_block_sizes=(1, 16, 1, 16),
            m_block_sizes=(16, 16, 16, 16),
        ),
        atol=3e-2,
    ),
    # Exercises `kernel_hd64` with `DECODE` + `PREFILL` + `MIXED`, `attention_sink`,
    # and `sliding_window` skipping leading KV blocks.
    _ParityCase(
        name="hd64_sliding_window_sink_prefill",
        is_hd64=True,
        head_dim=64,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        q_lens=(1, 16, 24),
        kv_lens=(112, 16, 128),
        dist=(1, 2, 3),
        kwargs=dict(
            sliding_window=32,
            chunk_prefill_size=16,
            num_kv_pages_per_block=1,
            num_queries_per_block=8,
        ),
        atol=2e-2,
        use_sink=True,
    ),
)


@pytest.mark.tpu_test
@pytest.mark.parametrize("case", _PARITY_CASES, ids=lambda c: c.name)
def test_rpa_v3_kernel_parity(case: _ParityCase) -> None:
    require_tpu()
    rng = np.random.default_rng(42)
    page_size = 16
    num_kv_heads, num_q_heads = 2, 8
    q_dtype = getattr(jnp, case.q_dtype)
    kv_dtype = getattr(jnp, case.kv_dtype)

    num_seqs = len(case.q_lens)
    pages_per_seq = max(case.kv_lens) // page_size + 1
    total_pages = num_seqs * pages_per_seq + 2
    cu_q_lens = jnp.asarray(np.cumsum([0, *case.q_lens]), dtype=jnp.int32)
    kv_lens = jnp.asarray(case.kv_lens, dtype=jnp.int32)
    dist = jnp.asarray(case.dist, dtype=jnp.int32)
    page_indices = jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32)
    total_q = int(cu_q_lens[-1])

    q = jnp.asarray(
        rng.standard_normal((total_q, num_q_heads, case.head_dim), dtype=np.float32)
        * 0.5,
        dtype=q_dtype,
    )
    k = jnp.asarray(
        rng.standard_normal((total_q, num_kv_heads, case.head_dim), dtype=np.float32)
        * 0.5,
        dtype=kv_dtype,
    )
    v = jnp.asarray(
        rng.standard_normal((total_q, num_kv_heads, case.head_dim), dtype=np.float32)
        * 0.5,
        dtype=kv_dtype,
    )

    shape_fn = (
        kernel_hd64.get_kv_cache_shape if case.is_hd64 else kernel.get_kv_cache_shape
    )
    ref_fn = (
        kernel_hd64.ref_ragged_paged_attention_hd64
        if case.is_hd64
        else kernel.ref_ragged_paged_attention
    )
    run_fn = (
        kernel_hd64.ragged_paged_attention_hd64
        if case.is_hd64
        else kernel.ragged_paged_attention
    )

    cache_shape = shape_fn(
        total_pages, page_size, num_kv_heads, case.head_dim, kv_dtype
    )
    init_cache = jnp.asarray(
        rng.standard_normal(cache_shape, dtype=np.float32) * 0.5, dtype=kv_dtype
    )

    extra_kw: dict[str, object] = {}
    if case.use_sink:
        extra_kw["attention_sink"] = jnp.asarray(
            rng.standard_normal((num_q_heads,), dtype=np.float32) * 0.5,
            dtype=jnp.float32,
        )

    ref_kw = {
        k_name: val
        for k_name, val in (case.kwargs | extra_kw).items()
        if k_name
        in ("sliding_window", "sm_scale", "k_scale", "v_scale", "attention_sink")
    }
    ref_out, ref_cache = ref_fn(
        q, k, v, init_cache, kv_lens, page_indices, cu_q_lens, dist, **ref_kw
    )
    got_out, got_cache = run_fn(
        q.copy(),
        k.copy(),
        v.copy(),
        init_cache.copy(),
        kv_lens,
        page_indices,
        cu_q_lens,
        dist,
        **(case.kwargs | extra_kw),
    )

    np.testing.assert_allclose(
        np.asarray(got_out, dtype=np.float32),
        np.asarray(ref_out, dtype=np.float32),
        rtol=5e-2,
        atol=case.atol,
    )
    np.testing.assert_array_equal(
        np.asarray(got_cache, dtype=np.float32),
        np.asarray(ref_cache, dtype=np.float32),
    )
