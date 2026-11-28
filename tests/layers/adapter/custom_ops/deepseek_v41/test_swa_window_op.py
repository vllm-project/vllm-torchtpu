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
"""FakeTensor, bridge forwarding, and TPU7x tests for DSv4.1 SWA window op."""

import inspect
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh
from torch._dynamo.testing import CompileCounterWithBackend
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    dsv41_gather_swa_windows,
    dsv41_pack_fp8_kv_rows,
    dsv41_write_sub_rows_in_place,
    swa_window_op,
)

ROW_WORDS = 128


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
    max_pages = (
        pool_2d.shape[0] - page_row_offset - rows_per_page
    ) // page_row_stride + 1
    num_seqs, pages_per_window = page_ids.shape
    staged = np.zeros(
        (num_seqs, pages_per_window * rows_per_page, ROW_WORDS), dtype=np.int32
    )
    valid = np.zeros(staged.shape[:2], dtype=bool)
    for seq in range(num_seqs):
        for slot, page in enumerate(page_ids[seq].tolist()):
            if page < 0 or page >= max_pages:
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


def test_should_not_donate_inputs_on_swa_window_op() -> None:
    op = swa_window_op._gather_swa_windows_op  # noqa: SLF001 -- check jax_op donate config
    assert list(op._init_fn.donate_argnums) == []  # noqa: SLF001 -- JaxCustomOpDef holds JaxCallable


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


@pytest.mark.parametrize(
    ("pool_shape", "expected_flat_shape"),
    [
        ((64, ROW_WORDS), (64, ROW_WORDS)),
        ((4, 16, ROW_WORDS), (64, ROW_WORDS)),
        ((4, 16, 64), (64, 64)),
    ],
)
def test_should_forward_swa_window_jax_wrapper_and_preserve_trailing_width(
    monkeypatch: pytest.MonkeyPatch,
    pool_shape: tuple[int, ...],
    expected_flat_shape: tuple[int, int],
) -> None:
    calls: list[dict[str, object]] = []

    def _spy_gather_swa(
        pool: jax.Array,
        page_ids: jax.Array,
        phase_rows: jax.Array,
        *,
        window_rows: int,
        rows_per_page: int,
        page_row_stride: int,
        page_row_offset: int,
    ) -> jax.Array:
        calls.append(
            {
                "pool_shape": pool.shape,
                "page_ids": page_ids,
                "phase_rows": phase_rows,
                "window_rows": window_rows,
                "rows_per_page": rows_per_page,
                "page_row_stride": page_row_stride,
                "page_row_offset": page_row_offset,
            }
        )
        return jnp.full((page_ids.shape[0], window_rows, ROW_WORDS), 9, dtype=jnp.int32)

    monkeypatch.setattr(swa_window_op, "gather_swa_windows", _spy_gather_swa)

    pool = jnp.zeros(pool_shape, dtype=jnp.int32)
    page_ids = jnp.zeros((2, 3), dtype=jnp.int32)
    phase_rows = jnp.array([4, 0], dtype=jnp.int32)
    out = swa_window_op._gather_swa_windows_jax(  # noqa: SLF001 -- test internal JAX bridge wrapper directly
        pool, page_ids, phase_rows, 32, 16, 48, 24
    )

    assert out.shape == (2, 32, ROW_WORDS)
    assert calls == [
        {
            "pool_shape": expected_flat_shape,
            "page_ids": page_ids,
            "phase_rows": phase_rows,
            "window_rows": 32,
            "rows_per_page": 16,
            "page_row_stride": 48,
            "page_row_offset": 24,
        }
    ]


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
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_tpu_should_gather_swa_windows_from_3d_pool_under_tpu_compile(
    compiled: bool,
) -> None:
    device = torch.device("tpu")
    rng = np.random.default_rng(707)
    num_pages = 10
    page_row_stride = 64
    rows_per_page = 32
    page_row_offset = 16
    window_rows = 64
    max_pages = (
        num_pages * page_row_stride - page_row_offset - rows_per_page
    ) // page_row_stride + 1

    pool_np = rng.integers(
        10_000, 900_000, size=(num_pages, page_row_stride, ROW_WORDS), dtype=np.int32
    )
    page_ids_np = np.array(
        [[2, 5, 8], [-1, 3, max_pages - 1], [1, max_pages, 4]], dtype=np.int32
    )
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

    def _gather(
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

    if compiled:
        torch.compiler.reset()
        step_fn = torch.compile(_gather, backend="tpu", fullgraph=True)
        out_tpu = step_fn(pool_tpu, page_ids_tpu, phase_rows_tpu)
        torch.accelerator.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            out_tpu = step_fn(pool_tpu, page_ids_tpu, phase_rows_tpu)
            torch.accelerator.synchronize()
        assert not any(
            "tt_jit_execute_as_strided" in evt.key for evt in prof.key_averages()
        )
    else:
        out_tpu = _gather(pool_tpu, page_ids_tpu, phase_rows_tpu)
        torch.accelerator.synchronize()

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
    write_slots = torch.arange(8, dtype=torch.int32, device=device) + (
        2 * page_row_stride + page_row_offset
    )
    gather_pages = torch.tensor([[2, 5, 8]], dtype=torch.int32, device=device)
    gather_phase = torch.tensor([0], dtype=torch.int32, device=device)

    def _compose(
        pool: torch.Tensor,
        latent: torch.Tensor,
        slots: torch.Tensor,
        pages: torch.Tensor,
        phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nope_w, _ = dsv41_pack_fp8_kv_rows(latent)
        dsv41_write_sub_rows_in_place(pool, nope_w, slots, page_tokens=16)
        win = dsv41_gather_swa_windows(
            pool,
            pages,
            phase,
            window_rows=window_rows,
            rows_per_page=rows_per_page,
            page_row_stride=page_row_stride,
            page_row_offset=page_row_offset,
        )
        return nope_w, win

    compose_fn = (
        torch.compile(_compose, backend="tpu", fullgraph=True) if compiled else _compose
    )
    nope_w, composed_win = compose_fn(
        composed_pool, latent_tpu, write_slots, gather_pages, gather_phase
    )
    torch.accelerator.synchronize()
    np.testing.assert_array_equal(
        composed_win.cpu().numpy()[0, :8], nope_w.cpu().numpy()
    )
