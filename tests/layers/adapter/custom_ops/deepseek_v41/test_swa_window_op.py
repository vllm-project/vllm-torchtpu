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
"""CPU stub, FakeTensor, and TPU7x tests for DSv4.1 SWA window custom op."""

import inspect
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh
from torch._dynamo.testing import CompileCounterWithBackend
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    cache_op,
    dsv41_gather_swa_windows,
    dsv41_pack_fp8_kv_rows,
    dsv41_write_sub_rows_in_place,
    swa_window_op,
)

ROW_WORDS = 128


def _torch_dtype_for(jax_dtype: jnp.dtype) -> torch.dtype:
    mapping = {
        jnp.dtype(jnp.int32): torch.int32,
        jnp.dtype(jnp.bfloat16): torch.bfloat16,
        jnp.dtype(jnp.uint8): torch.uint8,
        jnp.dtype(jnp.float32): torch.float32,
    }
    return mapping[jnp.dtype(jax_dtype)]


class _JaxBackedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem: jax.Array, env: object = None) -> "_JaxBackedTensor":
        del env
        shape = tuple(int(dim) for dim in elem.shape)
        base = torch.empty(shape, dtype=_torch_dtype_for(elem.dtype))
        return base.as_subclass(cls)

    def __init__(self, elem: jax.Array, env: object = None) -> None:
        self._elem = elem
        self._env = env

    @property
    def elem(self) -> jax.Array:
        return self._elem

    def copy_(
        self, src: torch.Tensor, non_blocking: bool = False
    ) -> "_JaxBackedTensor":
        del non_blocking
        assert isinstance(src, _JaxBackedTensor)
        self._elem = src.elem
        return self

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._elem)


def _clear_jax_op_caches() -> None:
    ops: tuple[Any, ...] = (
        swa_window_op._gather_swa_windows_op,  # noqa: SLF001 -- clear per-op cache
        cache_op._write_sub_rows_op,  # noqa: SLF001 -- clear per-op cache
        cache_op._write_sub_rows_after_op,  # noqa: SLF001 -- clear per-op cache
        cache_op._pack_fp8_kv_rows_op,  # noqa: SLF001 -- clear per-op cache
    )
    for op in ops:
        init_fn = op._init_fn  # noqa: SLF001 -- JaxCustomOpDef holds JaxCallable
        init_fn.jit_fn.clear_cache()
        if init_fn.output_shapes:
            init_fn.output_shapes.clear()
        if init_fn.kernel_key_to_mlir_fingerprint:
            init_fn.kernel_key_to_mlir_fingerprint.clear()


@pytest.fixture(autouse=True)
def _reset_jax_op_caches() -> Iterator[None]:
    yield
    _clear_jax_op_caches()


@pytest.fixture(autouse=True)
def _tpu7x_abstract_mesh(request: pytest.FixtureRequest) -> Iterator[None]:
    if jax.default_backend() == "tpu":
        yield
        return
    if request.node.get_closest_marker("tpu_test") is not None:
        pytest.skip("Requires TPU hardware backend")
    mesh = AbstractMesh((), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        yield


def _stub_write_sub_rows(
    cache_words: jax.Array,
    updates: jax.Array,
    slot_mapping: jax.Array,
    *,
    after: jax.Array | None = None,
    page_tokens: int = 16,
) -> jax.Array:
    del after, page_tokens
    if cache_words.ndim != 2 or cache_words.shape[1] != ROW_WORDS:
        raise ValueError("cache_words must be int32[num_rows, 128].")
    total_rows = cache_words.shape[0]

    def _apply(token_idx: jax.Array, acc: jax.Array) -> jax.Array:
        slot = slot_mapping[token_idx]
        valid = (slot >= 0) & (slot < total_rows)
        safe = jnp.where(valid, slot, 0)
        row = jnp.where(valid, updates[token_idx], acc[safe])
        return acc.at[safe].set(row)

    return jax.lax.fori_loop(0, updates.shape[0], _apply, cache_words)


def _stub_gather_swa_windows(
    pool: jax.Array,
    page_ids: jax.Array,
    phase_rows: jax.Array,
    *,
    window_rows: int,
    rows_per_page: int,
    page_row_stride: int,
    page_row_offset: int = 0,
) -> jax.Array:
    if pool.ndim != 2 or pool.shape[1] != ROW_WORDS or pool.dtype != jnp.int32:
        raise ValueError("pool must be int32[num_rows, 128].")
    num_seqs, pages_per_window = page_ids.shape
    if num_seqs == 0:
        return jnp.zeros((0, window_rows, ROW_WORDS), dtype=jnp.int32)
    rel_rows = jnp.arange(rows_per_page, dtype=jnp.int32)[None, None, :]
    valid_page = page_ids >= 0
    starts = jnp.where(valid_page, page_ids, 0) * page_row_stride + page_row_offset
    row_addrs = starts[:, :, None] + rel_rows
    staged = jnp.where(valid_page[:, :, None, None], pool[row_addrs], jnp.int32(0))
    staged = staged.reshape(num_seqs, pages_per_window * rows_per_page, ROW_WORDS)
    take = phase_rows[:, None] + jnp.arange(window_rows, dtype=jnp.int32)[None, :]
    seq_idx = jnp.arange(num_seqs, dtype=jnp.int32)[:, None]
    return staged[seq_idx, take]


def _reference_swa_windows(
    pool: np.ndarray,
    page_ids: np.ndarray,
    phase_rows: np.ndarray,
    *,
    window_rows: int,
    rows_per_page: int,
    page_row_stride: int,
    page_row_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    pool_2d = pool.reshape(-1, ROW_WORDS)
    num_seqs, pages_per_window = page_ids.shape
    staged = np.zeros(
        (num_seqs, pages_per_window * rows_per_page, ROW_WORDS), dtype=np.int32
    )
    valid = np.zeros(staged.shape[:2], dtype=bool)
    for seq in range(num_seqs):
        for slot, page in enumerate(page_ids[seq].tolist()):
            if page < 0:
                continue
            start = page * page_row_stride + page_row_offset
            dst = slice(slot * rows_per_page, (slot + 1) * rows_per_page)
            staged[seq, dst] = pool_2d[start : start + rows_per_page]
            valid[seq, dst] = True
    take = phase_rows[:, None] + np.arange(window_rows)[None, :]
    return (
        np.take_along_axis(staged, take[:, :, None], axis=1),
        np.take_along_axis(valid, take, axis=1),
    )


def test_should_export_dsv41_swa_window_op_with_expected_signature() -> None:
    assert swa_window_op.__all__ == ["dsv41_gather_swa_windows"]
    sig = inspect.signature(dsv41_gather_swa_windows)
    assert list(sig.parameters) == [
        "pool",
        "page_ids",
        "phase_rows",
        "window_rows",
        "rows_per_page",
        "page_row_stride",
        "page_row_offset",
    ]
    assert sig.parameters["page_row_offset"].default == 0


@pytest.mark.parametrize("pool_shape", [(256, 128), (8, 32, 128)])
@pytest.mark.parametrize("num_seqs", [0, 1, 7])
def test_should_trace_fake_swa_windows_across_2d_3d_pools_and_bucket_sizes(
    pool_shape: tuple[int, ...],
    num_seqs: int,
) -> None:
    torch.compiler.reset()
    counter = CompileCounterWithBackend("aot_eager")

    def _gather(
        pool: torch.Tensor,
        page_ids: torch.Tensor,
        phase_rows: torch.Tensor,
    ) -> torch.Tensor:
        return dsv41_gather_swa_windows(
            pool,
            page_ids,
            phase_rows,
            window_rows=64,
            rows_per_page=16,
            page_row_stride=32,
            page_row_offset=8,
        )

    compiled = torch.compile(_gather, backend=counter, fullgraph=True)
    with FakeTensorMode() as mode:
        pool = torch.empty(pool_shape, dtype=torch.int32)
        page_ids = torch.empty((num_seqs, 5), dtype=torch.int32)
        phase_rows = torch.empty((num_seqs,), dtype=torch.int32)

        out = compiled(pool, page_ids, phase_rows)

        assert counter.frame_count == 1
        assert isinstance(out, FakeTensor) and out.fake_mode is mode
        assert out.shape == (num_seqs, 64, ROW_WORDS)
        assert out.dtype == torch.int32


@pytest.mark.parametrize("use_3d_pool", [False, True])
@pytest.mark.parametrize(
    ("num_seqs", "window_rows", "rows_per_page", "page_row_stride", "page_row_offset"),
    [
        (4, 64, 16, 16, 0),
        (5, 96, 16, 48, 24),
        (0, 64, 16, 16, 0),
    ],
)
def test_should_gather_swa_windows_from_2d_and_3d_pools_via_bridge(
    monkeypatch: pytest.MonkeyPatch,
    use_3d_pool: bool,
    num_seqs: int,
    window_rows: int,
    rows_per_page: int,
    page_row_stride: int,
    page_row_offset: int,
) -> None:
    monkeypatch.setattr(swa_window_op, "gather_swa_windows", _stub_gather_swa_windows)
    rng = np.random.default_rng(300 + num_seqs + page_row_stride + int(use_3d_pool))
    pages_per_window = -(-(window_rows - 1) // rows_per_page) + 1
    num_pages = max(12, num_seqs * pages_per_window + 2)
    pool_shape = (
        (num_pages, page_row_stride, ROW_WORDS)
        if use_3d_pool
        else (num_pages * page_row_stride, ROW_WORDS)
    )
    # Non-zero sentinel values in page 0 ensure page_ids == -1 never reads page 0.
    pool_np = rng.integers(10_000, 999_999, size=pool_shape, dtype=np.int32)
    page_ids_np = rng.integers(
        1, num_pages, size=(num_seqs, pages_per_window), dtype=np.int32
    )
    if num_seqs > 0:
        page_ids_np[::2, 0] = -1
        page_ids_np[1::3, pages_per_window // 2] = -1
        page_ids_np[0, -1] = -1
    phase_rows_np = rng.integers(0, rows_per_page, size=(num_seqs,), dtype=np.int32)
    if num_seqs >= 1:
        phase_rows_np[0] = rows_per_page - 1
    if num_seqs >= 2:
        phase_rows_np[1] = 0

    expected, valid = _reference_swa_windows(
        pool_np,
        page_ids_np,
        phase_rows_np,
        window_rows=window_rows,
        rows_per_page=rows_per_page,
        page_row_stride=page_row_stride,
        page_row_offset=page_row_offset,
    )

    out_t = dsv41_gather_swa_windows(
        _JaxBackedTensor(jnp.asarray(pool_np)),
        _JaxBackedTensor(jnp.asarray(page_ids_np)),
        _JaxBackedTensor(jnp.asarray(phase_rows_np)),
        window_rows=window_rows,
        rows_per_page=rows_per_page,
        page_row_stride=page_row_stride,
        page_row_offset=page_row_offset,
    )

    assert isinstance(out_t, _JaxBackedTensor)
    assert out_t.shape == (num_seqs, window_rows, ROW_WORDS)
    actual = out_t.to_numpy()
    np.testing.assert_array_equal(actual[valid], expected[valid])


def test_should_compose_3d_write_sub_rows_into_gather_swa_windows_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_op, "write_sub_rows", _stub_write_sub_rows)
    monkeypatch.setattr(swa_window_op, "gather_swa_windows", _stub_gather_swa_windows)
    rng = np.random.default_rng(911)
    num_blocks = 8
    page_row_stride = 64
    rows_per_page = 32
    page_row_offset = 16
    window_rows = 64
    pool_np = np.zeros((num_blocks, page_row_stride, ROW_WORDS), dtype=np.int32)
    updates_np = rng.integers(1, 900_000, size=(24, ROW_WORDS), dtype=np.int32)
    slots_np = np.concatenate(
        [
            2 * page_row_stride + page_row_offset + np.arange(8, dtype=np.int32),
            5 * page_row_stride + page_row_offset + np.arange(8, dtype=np.int32),
            7 * page_row_stride + page_row_offset + np.arange(8, dtype=np.int32),
        ]
    )

    pool_t = _JaxBackedTensor(jnp.asarray(pool_np))
    res = dsv41_write_sub_rows_in_place(
        pool_t,
        _JaxBackedTensor(jnp.asarray(updates_np)),
        _JaxBackedTensor(jnp.asarray(slots_np)),
        page_tokens=16,
    )
    assert res is None
    assert pool_t.shape == (num_blocks, page_row_stride, ROW_WORDS)

    page_ids_np = np.array([[2, 5, 7], [-1, 7, 2]], dtype=np.int32)
    phase_rows_np = np.array([4, 0], dtype=np.int32)
    windows_t = dsv41_gather_swa_windows(
        pool_t,
        _JaxBackedTensor(jnp.asarray(page_ids_np)),
        _JaxBackedTensor(jnp.asarray(phase_rows_np)),
        window_rows=window_rows,
        rows_per_page=rows_per_page,
        page_row_stride=page_row_stride,
        page_row_offset=page_row_offset,
    )
    assert isinstance(windows_t, _JaxBackedTensor)
    expected_pool = pool_np.reshape(-1, ROW_WORDS).copy()
    for token_idx, slot in enumerate(slots_np.tolist()):
        if 0 <= slot < expected_pool.shape[0]:
            expected_pool[slot] = updates_np[token_idx]
    expected, valid = _reference_swa_windows(
        expected_pool,
        page_ids_np,
        phase_rows_np,
        window_rows=window_rows,
        rows_per_page=rows_per_page,
        page_row_stride=page_row_stride,
        page_row_offset=page_row_offset,
    )
    assert np.any(expected[valid] != 0)
    np.testing.assert_array_equal(windows_t.to_numpy()[valid], expected[valid])


def test_should_retrace_on_same_shape_after_clearing_swa_op_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_t = _JaxBackedTensor(jnp.zeros((4, 32, ROW_WORDS), dtype=jnp.int32))
    pages_t = _JaxBackedTensor(jnp.zeros((2, 3), dtype=jnp.int32))
    phases_t = _JaxBackedTensor(jnp.zeros((2,), dtype=jnp.int32))

    monkeypatch.setattr(
        swa_window_op,
        "gather_swa_windows",
        lambda pool, page_ids, phase_rows, **kw: jnp.full(
            (page_ids.shape[0], kw["window_rows"], ROW_WORDS), 11, dtype=jnp.int32
        ),
    )
    first = dsv41_gather_swa_windows(
        pool_t,
        pages_t,
        phases_t,
        window_rows=32,
        rows_per_page=16,
        page_row_stride=32,
    )
    assert isinstance(first, _JaxBackedTensor)
    np.testing.assert_array_equal(first.to_numpy(), 11)

    _clear_jax_op_caches()
    monkeypatch.setattr(
        swa_window_op,
        "gather_swa_windows",
        lambda pool, page_ids, phase_rows, **kw: jnp.full(
            (page_ids.shape[0], kw["window_rows"], ROW_WORDS), 22, dtype=jnp.int32
        ),
    )
    second = dsv41_gather_swa_windows(
        pool_t,
        pages_t,
        phases_t,
        window_rows=32,
        rows_per_page=16,
        page_row_stride=32,
    )
    assert isinstance(second, _JaxBackedTensor)
    np.testing.assert_array_equal(second.to_numpy(), 22)


def test_should_fail_fast_when_swa_window_inputs_or_device_are_invalid() -> None:
    valid_pool = torch.zeros((4, 32, ROW_WORDS), dtype=torch.int32)
    valid_pages = torch.zeros((2, 3), dtype=torch.int32)
    valid_phases = torch.zeros((2,), dtype=torch.int32)

    # Malformed 3D width (4, 32, 64) must NOT silently reshape into (64, 128).
    with pytest.raises(ValueError, match="128"):
        dsv41_gather_swa_windows(
            torch.zeros((4, 32, 64), dtype=torch.int32),
            valid_pages,
            valid_phases,
            window_rows=32,
            rows_per_page=16,
            page_row_stride=32,
        )

    with pytest.raises(ValueError, match="128"):
        dsv41_gather_swa_windows(
            torch.zeros((1, 4, 32, ROW_WORDS), dtype=torch.int32),
            valid_pages,
            valid_phases,
            window_rows=32,
            rows_per_page=16,
            page_row_stride=32,
        )

    with pytest.raises(ValueError, match="too small"):
        dsv41_gather_swa_windows(
            valid_pool,
            torch.zeros((2, 2), dtype=torch.int32),
            valid_phases,
            window_rows=32,
            rows_per_page=16,
            page_row_stride=32,
        )

    with pytest.raises(ValueError, match="page_row_stride"):
        dsv41_gather_swa_windows(
            valid_pool,
            valid_pages,
            valid_phases,
            window_rows=32,
            rows_per_page=16,
            page_row_stride=24,
            page_row_offset=12,
        )

    with pytest.raises(ValueError, match="phase_rows"):
        dsv41_gather_swa_windows(
            valid_pool,
            valid_pages,
            torch.zeros((3,), dtype=torch.int32),
            window_rows=32,
            rows_per_page=16,
            page_row_stride=32,
        )

    v6e_mesh = AbstractMesh((), (), abstract_device=AbstractDevice("TPU v6e", 1, "tpu"))
    with use_abstract_mesh(v6e_mesh), pytest.raises(RuntimeError, match="TPU7x"):
        dsv41_gather_swa_windows(
            valid_pool,
            valid_pages,
            valid_phases,
            window_rows=32,
            rows_per_page=16,
            page_row_stride=32,
        )


@pytest.mark.tpu_test
def test_tpu_should_gather_swa_windows_from_3d_pool_under_tpu_compile() -> None:
    device = torch.device("tpu")
    rng = np.random.default_rng(707)
    num_pages = 10
    page_row_stride = 64
    rows_per_page = 32
    page_row_offset = 16
    window_rows = 64

    pool_np = rng.integers(
        10_000, 900_000, size=(num_pages, page_row_stride, ROW_WORDS), dtype=np.int32
    )
    page_ids_np = np.array([[2, 5, 8], [-1, 3, 9], [1, -1, 4]], dtype=np.int32)
    phase_rows_np = np.array([12, 0, 31], dtype=np.int32)
    expected, valid = _reference_swa_windows(
        pool_np,
        page_ids_np,
        phase_rows_np,
        window_rows=window_rows,
        rows_per_page=rows_per_page,
        page_row_stride=page_row_stride,
        page_row_offset=page_row_offset,
    )

    pool_tpu = torch.from_numpy(pool_np).to(device)
    page_ids_tpu = torch.from_numpy(page_ids_np).to(device)
    phase_rows_tpu = torch.from_numpy(phase_rows_np).to(device)

    @torch.compile(backend="tpu", fullgraph=True)
    def _compiled_swa(
        pool: torch.Tensor, page_ids: torch.Tensor, phase_rows: torch.Tensor
    ) -> torch.Tensor:
        return dsv41_gather_swa_windows(
            pool,
            page_ids,
            phase_rows,
            window_rows=window_rows,
            rows_per_page=rows_per_page,
            page_row_stride=page_row_stride,
            page_row_offset=page_row_offset,
        )

    out_tpu = _compiled_swa(pool_tpu, page_ids_tpu, phase_rows_tpu)
    torch.accelerator.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        out_tpu = _compiled_swa(pool_tpu, page_ids_tpu, phase_rows_tpu)
        torch.accelerator.synchronize()
    assert not any(
        "tt_jit_execute_as_strided" in evt.key for evt in prof.key_averages()
    )

    out_cpu = out_tpu.cpu().numpy()
    assert out_cpu.shape == (3, window_rows, ROW_WORDS)
    np.testing.assert_array_equal(out_cpu[valid], expected[valid])

    out_2d_cpu = (
        dsv41_gather_swa_windows(
            pool_tpu.reshape(-1, ROW_WORDS),
            page_ids_tpu,
            phase_rows_tpu,
            window_rows=window_rows,
            rows_per_page=rows_per_page,
            page_row_stride=page_row_stride,
            page_row_offset=page_row_offset,
        )
        .cpu()
        .numpy()
    )
    np.testing.assert_array_equal(out_2d_cpu[valid], expected[valid])

    composed_pool = torch.zeros(
        (num_pages, page_row_stride, ROW_WORDS), dtype=torch.int32, device=device
    )
    latent_tpu = torch.randn((8, 512), dtype=torch.float32, device=device).to(
        torch.bfloat16
    )
    nope_w, _ = dsv41_pack_fp8_kv_rows(latent_tpu)
    write_slots = torch.arange(8, dtype=torch.int32, device=device) + (
        2 * page_row_stride + page_row_offset
    )
    dsv41_write_sub_rows_in_place(composed_pool, nope_w, write_slots, page_tokens=16)
    composed_win = (
        dsv41_gather_swa_windows(
            composed_pool,
            torch.tensor([[2, 5, 8]], dtype=torch.int32, device=device),
            torch.tensor([0], dtype=torch.int32, device=device),
            window_rows=window_rows,
            rows_per_page=rows_per_page,
            page_row_stride=page_row_stride,
            page_row_offset=page_row_offset,
        )
        .cpu()
        .numpy()
    )
    np.testing.assert_array_equal(composed_win[0, :8], nope_w.cpu().numpy())
