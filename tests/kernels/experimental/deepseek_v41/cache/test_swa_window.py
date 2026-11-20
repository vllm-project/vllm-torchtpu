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
"""Lean CPU interpret and TPU7x hardware tests for paged SWA window gather."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import jax
import jax.export
import jax.extend.core
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.deepseek_v41.cache import (
    ROW_WORDS,
    gather_swa_windows,
    write_sub_rows,
)

ROWS_PER_TOKEN = 2
WINDOW_TOKENS = 4
WINDOW_ROWS = WINDOW_TOKENS * ROWS_PER_TOKEN
ROWS_PER_PAGE = 4
PAGES_PER_WINDOW = 3
TPU_WINDOW_ROWS = 256
TPU_ROWS_PER_PAGE = 32
TPU_PAGES_PER_WINDOW = 9

_JIT_SWA_WINDOWS = jax.jit(
    gather_swa_windows,
    static_argnames=(
        "window_rows",
        "rows_per_page",
        "page_row_stride",
        "page_row_offset",
        "seqs_per_step",
        "interpret",
        "enable_named_scope",
    ),
)
_JIT_WRITE_SUB_ROWS = jax.jit(
    write_sub_rows,
    static_argnames=("page_tokens", "interpret", "enable_named_scope"),
)


@pytest.fixture(autouse=True)
def _tpu7x_abstract_mesh(request: pytest.FixtureRequest) -> Iterator[None]:
    if jax.default_backend() == "tpu":
        yield
        return
    if request.node.get_closest_marker("tpu_test") is not None:
        pytest.skip("Requires TPU hardware backend")
    device = jax.sharding.AbstractDevice("TPU7x", 1, "tpu")
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    with jax.sharding.use_abstract_mesh(mesh):
        yield


def _reference_gather_swa_windows(
    pool: np.ndarray,
    page_ids: np.ndarray,
    phase_rows: np.ndarray,
    *,
    window_rows: int,
    rows_per_page: int,
    page_row_stride: int,
    page_row_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_seqs, pages_per_window = page_ids.shape
    max_pages = (pool.shape[0] - page_row_offset - rows_per_page) // page_row_stride + 1
    out = np.zeros((num_seqs, window_rows, ROW_WORDS), dtype=np.int32)
    valid = np.zeros((num_seqs, window_rows), dtype=bool)
    for seq_idx in range(num_seqs):
        staged = np.zeros((pages_per_window * rows_per_page, ROW_WORDS), dtype=np.int32)
        staged_valid = np.zeros((pages_per_window * rows_per_page,), dtype=bool)
        for p_slot in range(pages_per_window):
            page = int(page_ids[seq_idx, p_slot])
            if 0 <= page < max_pages:
                src_start = page * page_row_stride + page_row_offset
                dst_start = p_slot * rows_per_page
                staged[dst_start : dst_start + rows_per_page] = pool[
                    src_start : src_start + rows_per_page
                ]
                staged_valid[dst_start : dst_start + rows_per_page] = True
        phase = int(phase_rows[seq_idx])
        out[seq_idx] = staged[phase : phase + window_rows]
        valid[seq_idx] = staged_valid[phase : phase + window_rows]
    return out, valid


def _make_inputs(
    num_seqs: int,
    num_pages: int,
    page_row_stride: int,
    *,
    rows_per_page: int = ROWS_PER_PAGE,
    pages_per_window: int = PAGES_PER_WINDOW,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pool = rng.integers(
        10, 50_000, size=(num_pages * page_row_stride, ROW_WORDS), dtype=np.int32
    )
    page_ids = rng.integers(
        0, num_pages, size=(num_seqs, pages_per_window), dtype=np.int32
    )
    phase_rows = rng.integers(0, rows_per_page, size=(num_seqs,), dtype=np.int32)
    return pool, page_ids, phase_rows


@pytest.mark.parametrize(
    (
        "num_seqs",
        "seqs_per_step",
        "page_row_stride",
        "page_row_offset",
        "enable_named_scope",
    ),
    [
        (2, 2, ROWS_PER_PAGE, 0, False),
        (3, 2, ROWS_PER_PAGE + 4, 4, True),
    ],
)
def test_should_match_reference_window_rows_when_pages_valid(
    num_seqs: int,
    seqs_per_step: int,
    page_row_stride: int,
    page_row_offset: int,
    enable_named_scope: bool,
) -> None:
    pool, page_ids, phase_rows = _make_inputs(
        num_seqs, num_pages=8, page_row_stride=page_row_stride, seed=23 + num_seqs
    )
    phase_rows[0] = 0
    phase_rows[-1] = ROWS_PER_PAGE - 1

    got = np.asarray(
        _JIT_SWA_WINDOWS(
            jnp.asarray(pool),
            jnp.asarray(page_ids),
            jnp.asarray(phase_rows),
            window_rows=WINDOW_ROWS,
            rows_per_page=ROWS_PER_PAGE,
            page_row_stride=page_row_stride,
            page_row_offset=page_row_offset,
            seqs_per_step=seqs_per_step,
            interpret=True,
            enable_named_scope=enable_named_scope,
        )
    )
    exp, valid = _reference_gather_swa_windows(
        pool,
        page_ids,
        phase_rows,
        window_rows=WINDOW_ROWS,
        rows_per_page=ROWS_PER_PAGE,
        page_row_stride=page_row_stride,
        page_row_offset=page_row_offset,
    )
    assert valid.all()
    np.testing.assert_array_equal(got, exp)


def test_should_return_empty_windows_when_num_seqs_is_zero() -> None:
    pool = jnp.zeros((ROWS_PER_PAGE * 2, ROW_WORDS), dtype=jnp.int32)
    page_ids = jnp.zeros((0, PAGES_PER_WINDOW), dtype=jnp.int32)
    phase_rows = jnp.zeros((0,), dtype=jnp.int32)
    out = gather_swa_windows(
        pool,
        page_ids,
        phase_rows,
        window_rows=WINDOW_ROWS,
        rows_per_page=ROWS_PER_PAGE,
        page_row_stride=ROWS_PER_PAGE,
        seqs_per_step=2,
        interpret=True,
    )
    assert out.shape == (0, WINDOW_ROWS, ROW_WORDS)
    assert out.dtype == jnp.int32


def test_should_roundtrip_write_sub_rows_into_gather_swa_windows_with_bf16_oracle() -> (
    None
):
    rng = np.random.default_rng(77)
    num_pages = 8
    written_rows = 3 * ROWS_PER_PAGE
    pool_bf16 = rng.normal(0.0, 2.0, size=(written_rows, 256)).astype(np.float32)
    bf16_np = pool_bf16.astype(jnp.bfloat16)
    u16 = bf16_np.view(np.uint16)
    updates_i32 = jnp.asarray(
        (u16[:, 0::2].astype(np.uint32) | (u16[:, 1::2].astype(np.uint32) << 16)).view(
            np.int32
        )
    )
    pool = jnp.asarray(np.zeros((num_pages * ROWS_PER_PAGE, ROW_WORDS), dtype=np.int32))
    slots = jnp.asarray(
        np.concatenate(
            [
                np.arange(1 * ROWS_PER_PAGE, 2 * ROWS_PER_PAGE, dtype=np.int32),
                np.arange(3 * ROWS_PER_PAGE, 4 * ROWS_PER_PAGE, dtype=np.int32),
                np.arange(5 * ROWS_PER_PAGE, 6 * ROWS_PER_PAGE, dtype=np.int32),
            ]
        )
    )
    pool = _JIT_WRITE_SUB_ROWS(pool, updates_i32, slots, interpret=True)

    page_ids = np.array([[1, 3, 5], [5, 1, 3]], dtype=np.int32)
    phase_rows = np.array([1, 3], dtype=np.int32)
    got = np.asarray(
        _JIT_SWA_WINDOWS(
            pool,
            jnp.asarray(page_ids),
            jnp.asarray(phase_rows),
            window_rows=WINDOW_ROWS,
            rows_per_page=ROWS_PER_PAGE,
            page_row_stride=ROWS_PER_PAGE,
            page_row_offset=0,
            seqs_per_step=2,
            interpret=True,
            enable_named_scope=False,
        )
    )
    exp, valid = _reference_gather_swa_windows(
        np.asarray(pool),
        page_ids,
        phase_rows,
        window_rows=WINDOW_ROWS,
        rows_per_page=ROWS_PER_PAGE,
        page_row_stride=ROWS_PER_PAGE,
        page_row_offset=0,
    )
    assert valid.all()
    np.testing.assert_array_equal(got, exp)

    # Positive control: shifting phase by +1 breaks oracle equality.
    shifted_exp, _ = _reference_gather_swa_windows(
        np.asarray(pool),
        page_ids,
        np.array([2, 3], dtype=np.int32),
        window_rows=WINDOW_ROWS,
        rows_per_page=ROWS_PER_PAGE,
        page_row_stride=ROWS_PER_PAGE,
        page_row_offset=0,
    )
    assert not np.array_equal(got, shifted_exp)


def test_should_guard_oob_pages_and_omit_external_gather_epilogue() -> None:
    # Arrange: non-multiple-of-stride pool (63 rows, stride=8, offset=4, rpp=4) gives
    # max_pages = (63 - 4 - 4) // 8 + 1 = 7, so page=6 ([52, 56)) fits while tight
    # OOB page=7 starts at row 60 (< 63) yet ends at 64 (> 63).
    stride, offset = ROWS_PER_PAGE + 4, 4
    rng = np.random.default_rng(91)
    pool = rng.integers(10, 50_000, size=(7 * stride + 7, ROW_WORDS), dtype=np.int32)
    max_pages = (pool.shape[0] - offset - ROWS_PER_PAGE) // stride + 1
    assert max_pages == 7
    int32_max = (1 << 31) - 1
    page_ids = np.array(
        [[-1, max_pages - 1, max_pages], [0, max_pages + 1, int32_max]],
        dtype=np.int32,
    )
    phase_rows = np.array([2, 1], dtype=np.int32)

    got = np.asarray(
        _JIT_SWA_WINDOWS(
            jnp.asarray(pool),
            jnp.asarray(page_ids),
            jnp.asarray(phase_rows),
            window_rows=WINDOW_ROWS,
            rows_per_page=ROWS_PER_PAGE,
            page_row_stride=stride,
            page_row_offset=offset,
            seqs_per_step=2,
            interpret=True,
            enable_named_scope=False,
        )
    )
    exp, valid = _reference_gather_swa_windows(
        pool,
        page_ids,
        phase_rows,
        window_rows=WINDOW_ROWS,
        rows_per_page=ROWS_PER_PAGE,
        page_row_stride=stride,
        page_row_offset=offset,
    )
    assert valid.any() and not valid.all()
    np.testing.assert_array_equal(got[valid], exp[valid])

    seqs_per_step = 2
    jaxpr = jax.make_jaxpr(
        lambda p, pids, ph: gather_swa_windows(
            p,
            pids,
            ph,
            window_rows=WINDOW_ROWS,
            rows_per_page=ROWS_PER_PAGE,
            page_row_stride=stride,
            page_row_offset=offset,
            seqs_per_step=seqs_per_step,
            interpret=False,
        )
    )(jnp.asarray(pool), jnp.asarray(page_ids), jnp.asarray(phase_rows))
    op_names = [eqn.primitive.name for eqn in jaxpr.jaxpr.eqns]
    assert "gather" not in op_names
    pallas_eqns = [
        eqn for eqn in jaxpr.jaxpr.eqns if eqn.primitive.name == "pallas_call"
    ]
    assert len(pallas_eqns) == 1
    inner_jaxpr = pallas_eqns[0].params["jaxpr"]
    ge_zero_eqns = [
        eqn
        for eqn in inner_jaxpr.eqns
        if eqn.primitive.name == "ge"
        and isinstance(eqn.invars[1], jax.extend.core.Literal)
        and eqn.invars[1].val == 0
    ]
    lt_max_pages_eqns = [
        eqn
        for eqn in inner_jaxpr.eqns
        if eqn.primitive.name == "lt"
        and isinstance(eqn.invars[1], jax.extend.core.Literal)
        and eqn.invars[1].val == max_pages
    ]
    cond_eqns = [eqn for eqn in inner_jaxpr.eqns if eqn.primitive.name == "cond"]
    expected_guards = 2 * seqs_per_step * PAGES_PER_WINDOW
    assert len(ge_zero_eqns) == expected_guards
    assert len(lt_max_pages_eqns) == expected_guards
    assert len(cond_eqns) == expected_guards


@pytest.mark.parametrize(
    (
        "window_rows",
        "rows_per_page",
        "page_row_stride",
        "page_row_offset",
        "pages_per_window",
        "seqs_per_step",
        "pool_rows",
    ),
    [
        (0, ROWS_PER_PAGE, ROWS_PER_PAGE, 0, PAGES_PER_WINDOW, 2, 16),
        (WINDOW_ROWS, 0, ROWS_PER_PAGE, 0, PAGES_PER_WINDOW, 2, 16),
        (WINDOW_ROWS, ROWS_PER_PAGE, ROWS_PER_PAGE, 0, PAGES_PER_WINDOW, 0, 16),
        (WINDOW_ROWS, ROWS_PER_PAGE, ROWS_PER_PAGE - 1, 0, PAGES_PER_WINDOW, 2, 16),
        (WINDOW_ROWS, ROWS_PER_PAGE, ROWS_PER_PAGE, -1, PAGES_PER_WINDOW, 2, 16),
        (WINDOW_ROWS, ROWS_PER_PAGE, ROWS_PER_PAGE, 1, PAGES_PER_WINDOW, 2, 16),
        (WINDOW_ROWS, ROWS_PER_PAGE, ROWS_PER_PAGE + 4, 4, PAGES_PER_WINDOW, 2, 6),
        (WINDOW_ROWS, ROWS_PER_PAGE, ROWS_PER_PAGE, 0, 2, 2, 16),
    ],
)
def test_should_reject_geometry_when_contract_violated(
    window_rows: int,
    rows_per_page: int,
    page_row_stride: int,
    page_row_offset: int,
    pages_per_window: int,
    seqs_per_step: int,
    pool_rows: int,
) -> None:
    pool = jnp.zeros((pool_rows, ROW_WORDS), dtype=jnp.int32)
    page_ids = jnp.zeros((2, pages_per_window), dtype=jnp.int32)
    phase_rows = jnp.zeros((2,), dtype=jnp.int32)
    with pytest.raises(ValueError):
        gather_swa_windows(
            pool,
            page_ids,
            phase_rows,
            window_rows=window_rows,
            rows_per_page=rows_per_page,
            page_row_stride=page_row_stride,
            page_row_offset=page_row_offset,
            seqs_per_step=seqs_per_step,
            interpret=True,
        )


@pytest.mark.parametrize(
    (
        "pool_shape",
        "pool_dtype",
        "pids_shape",
        "pids_dtype",
        "phase_shape",
        "phase_dtype",
    ),
    [
        ((ROW_WORDS,), jnp.int32, (2, PAGES_PER_WINDOW), jnp.int32, (2,), jnp.int32),
        ((16, 64), jnp.int32, (2, PAGES_PER_WINDOW), jnp.int32, (2,), jnp.int32),
        (
            (16, ROW_WORDS),
            jnp.float32,
            (2, PAGES_PER_WINDOW),
            jnp.int32,
            (2,),
            jnp.int32,
        ),
        ((16, ROW_WORDS), jnp.int32, (2,), jnp.int32, (2,), jnp.int32),
        (
            (16, ROW_WORDS),
            jnp.int32,
            (2, PAGES_PER_WINDOW),
            jnp.float32,
            (2,),
            jnp.int32,
        ),
        ((16, ROW_WORDS), jnp.int32, (2, PAGES_PER_WINDOW), jnp.int32, (3,), jnp.int32),
        (
            (16, ROW_WORDS),
            jnp.int32,
            (2, PAGES_PER_WINDOW),
            jnp.int32,
            (2,),
            jnp.float32,
        ),
    ],
)
def test_should_reject_operands_when_shape_or_dtype_wrong(
    pool_shape: tuple[int, ...],
    pool_dtype: jnp.dtype,
    pids_shape: tuple[int, ...],
    pids_dtype: jnp.dtype,
    phase_shape: tuple[int, ...],
    phase_dtype: jnp.dtype,
) -> None:
    pool = jnp.zeros(pool_shape, dtype=pool_dtype)
    page_ids = jnp.zeros(pids_shape, dtype=pids_dtype)
    phase_rows = jnp.zeros(phase_shape, dtype=phase_dtype)
    with pytest.raises(ValueError):
        gather_swa_windows(
            pool,
            page_ids,
            phase_rows,
            window_rows=WINDOW_ROWS,
            rows_per_page=ROWS_PER_PAGE,
            page_row_stride=ROWS_PER_PAGE,
            seqs_per_step=2,
            interpret=True,
        )


def test_should_accept_exact_vmem_capacity_and_reject_one_byte_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exact VMEM bytes for seqs_per_step=1:
    # 1 * (PAGES_PER_WINDOW * ROWS_PER_PAGE + 2 * WINDOW_ROWS) * 512.
    exact_bytes = 1 * (PAGES_PER_WINDOW * ROWS_PER_PAGE + 2 * WINDOW_ROWS) * 512
    pool = jnp.zeros((ROWS_PER_PAGE * 4, ROW_WORDS), dtype=jnp.int32)
    pids = jnp.zeros((1, PAGES_PER_WINDOW), dtype=jnp.int32)
    phase = jnp.zeros((1,), dtype=jnp.int32)

    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            generation=7, num_lanes=ROW_WORDS, vmem_capacity_bytes=exact_bytes
        ),
    )
    # Avoid JIT cache reuse bypassing Python-side _check_vmem_budget.
    closed = jax.make_jaxpr(
        lambda p, i, r: gather_swa_windows(
            p,
            i,
            r,
            window_rows=WINDOW_ROWS,
            rows_per_page=ROWS_PER_PAGE,
            page_row_stride=ROWS_PER_PAGE,
            seqs_per_step=1,
            interpret=True,
        )
    )(pool, pids, phase)
    assert tuple(v.aval.shape for v in closed.jaxpr.outvars) == (
        (1, WINDOW_ROWS, ROW_WORDS),
    )

    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            generation=7, num_lanes=ROW_WORDS, vmem_capacity_bytes=exact_bytes - 1
        ),
    )
    with pytest.raises(ValueError, match="VMEM"):
        gather_swa_windows(
            pool,
            pids,
            phase,
            window_rows=WINDOW_ROWS,
            rows_per_page=ROWS_PER_PAGE,
            page_row_stride=ROWS_PER_PAGE,
            seqs_per_step=1,
            interpret=True,
        )


def test_should_aot_export_gather_swa_windows_for_tpu7x() -> None:
    if jax.default_backend() == "tpu":
        pytest.skip("Offline AOT export check runs under CPU AbstractDevice mesh")
    pool = jax.ShapeDtypeStruct((4096, ROW_WORDS), jnp.int32)
    pids = jax.ShapeDtypeStruct((13, TPU_PAGES_PER_WINDOW), jnp.int32)
    ph = jax.ShapeDtypeStruct((13,), jnp.int32)
    exported = jax.export.export(
        jax.jit(
            lambda p, i, r: gather_swa_windows(
                p,
                i,
                r,
                window_rows=TPU_WINDOW_ROWS,
                rows_per_page=TPU_ROWS_PER_PAGE,
                page_row_stride=TPU_ROWS_PER_PAGE,
                interpret=False,
            )
        ),
        platforms=["tpu"],
    )(pool, pids, ph)
    assert "tpu_custom_call" in exported.mlir_module()


def test_should_fail_fast_when_target_is_not_tpu7x() -> None:
    if jax.default_backend() == "tpu":
        pytest.skip("Abstract non-TPU7x check runs under CPU backend")
    v6e = jax.sharding.AbstractDevice("TPU v6e", 1, "tpu")
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=v6e)
    with jax.sharding.use_abstract_mesh(mesh):
        pool = jnp.zeros((ROWS_PER_PAGE * 4, ROW_WORDS), dtype=jnp.int32)
        pids = jnp.zeros((1, PAGES_PER_WINDOW), dtype=jnp.int32)
        ph = jnp.zeros((1,), dtype=jnp.int32)
        with pytest.raises(RuntimeError, match="TPU7x"):
            gather_swa_windows(
                pool,
                pids,
                ph,
                window_rows=WINDOW_ROWS,
                rows_per_page=ROWS_PER_PAGE,
                page_row_stride=ROWS_PER_PAGE,
                seqs_per_step=1,
                interpret=True,
            )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("num_seqs", "page_row_stride", "page_row_offset"),
    [(8, 32, 0), (13, 64, 16), (48, 32, 0), (288, 32, 0)],
)
def test_tpu_should_gather_swa_windows_accurately_on_hardware(
    num_seqs: int, page_row_stride: int, page_row_offset: int
) -> None:
    pool, page_ids, phase_rows = _make_inputs(
        num_seqs,
        num_pages=120,
        page_row_stride=page_row_stride,
        rows_per_page=TPU_ROWS_PER_PAGE,
        pages_per_window=TPU_PAGES_PER_WINDOW,
        seed=400 + num_seqs,
    )
    max_pages = (
        pool.shape[0] - page_row_offset - TPU_ROWS_PER_PAGE
    ) // page_row_stride + 1
    page_ids[0, 0] = -1
    page_ids[0, 1] = max_pages
    page_ids[1, 0] = (1 << 31) - 1
    page_ids[1, 1] = max_pages - 1
    got = np.asarray(
        jax.jit(
            lambda p, i, r: gather_swa_windows(
                p,
                i,
                r,
                window_rows=TPU_WINDOW_ROWS,
                rows_per_page=TPU_ROWS_PER_PAGE,
                page_row_stride=page_row_stride,
                page_row_offset=page_row_offset,
                interpret=False,
            )
        )(jnp.asarray(pool), jnp.asarray(page_ids), jnp.asarray(phase_rows))
    )
    exp, valid = _reference_gather_swa_windows(
        pool,
        page_ids,
        phase_rows,
        window_rows=TPU_WINDOW_ROWS,
        rows_per_page=TPU_ROWS_PER_PAGE,
        page_row_stride=page_row_stride,
        page_row_offset=page_row_offset,
    )
    assert valid.any() and not valid.all()
    np.testing.assert_array_equal(got[valid], exp[valid])
