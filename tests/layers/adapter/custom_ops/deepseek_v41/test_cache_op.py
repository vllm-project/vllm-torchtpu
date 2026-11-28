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
"""FakeTensor, bridge forwarding, and TPU7x tests for DSv4.1 cache ops."""

import inspect
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh
from torch._dynamo import mark_dynamic
from torch._dynamo.testing import CompileCounterWithBackend
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from vllm_torchtpu.layers.adapter.custom_ops import deepseek_v41 as dsv41_ops
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    cache_op,
    dsv41_gather_cache_rows,
    dsv41_gather_swa_windows,
    dsv41_pack_fp8_kv_rows,
    dsv41_write_sub_rows_in_place,
)

ROW_WORDS = 128
ROPE_WORDS = 32
LATENT_DIM = 512
NOPE_DIM = 448


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


def _dequantize_packed_nope(nope_words: np.ndarray) -> np.ndarray:
    raw_bytes = nope_words.astype(np.int32, copy=False).view(np.uint8)
    raw_bytes = raw_bytes.reshape(nope_words.shape[0], ROW_WORDS, 4)
    fp8_bytes = np.concatenate(
        [
            raw_bytes[:, :, 0],
            raw_bytes[:, :, 1],
            raw_bytes[:, :, 2],
            raw_bytes[:, :64, 3],
        ],
        axis=1,
    )
    fp8_vals = jax.lax.bitcast_convert_type(
        jnp.asarray(fp8_bytes), jnp.float8_e4m3fn
    ).astype(jnp.float32)
    exponents = raw_bytes[:, 64:, 3].astype(np.int32) - 127
    scales = np.exp2(exponents, dtype=np.float32)
    return np.asarray(fp8_vals) * np.tile(scales, (1, 7))


def test_should_export_dsv41_cache_ops_with_single_bank_signatures() -> None:
    expected_exports = [
        "dsv41_gather_cache_rows",
        "dsv41_gather_swa_windows",
        "dsv41_pack_fp8_kv_rows",
        "dsv41_write_sub_rows_in_place",
    ]
    assert dsv41_ops.__all__ == expected_exports
    for name in expected_exports:
        assert callable(getattr(dsv41_ops, name))

    write_sig = inspect.signature(dsv41_write_sub_rows_in_place)
    assert list(write_sig.parameters) == [
        "cache_words",
        "updates",
        "slot_mapping",
        "after",
        "page_tokens",
    ]
    assert write_sig.parameters["after"].kind is inspect.Parameter.KEYWORD_ONLY
    assert write_sig.parameters["page_tokens"].kind is inspect.Parameter.KEYWORD_ONLY
    assert write_sig.parameters["page_tokens"].default == 16
    assert write_sig.return_annotation is None

    gather_sig = inspect.signature(dsv41_gather_cache_rows)
    assert list(gather_sig.parameters) == ["pool", "row_indices"]


@pytest.mark.parametrize("pool_shape", [(16, 128), (2, 8, 128)])
@pytest.mark.parametrize("slice_words", [32, 64, 128])
@pytest.mark.parametrize(
    "after_dtype", [None, torch.int32, torch.bfloat16, torch.uint8]
)
def test_should_trace_fake_write_sub_rows_on_2d_and_3d_pools(
    pool_shape: tuple[int, ...],
    slice_words: int,
    after_dtype: torch.dtype | None,
) -> None:
    with FakeTensorMode() as mode:
        cache = torch.empty(pool_shape, dtype=torch.int32)
        updates = torch.empty((7, slice_words), dtype=torch.int32)
        slots = torch.empty((7,), dtype=torch.int32)
        after = (
            None if after_dtype is None else torch.empty((3, 128), dtype=after_dtype)
        )

        result = dsv41_write_sub_rows_in_place(
            cache,
            updates,
            slots,
            after=after,
            page_tokens=8,
        )

        assert result is None
        assert isinstance(cache, FakeTensor) and cache.fake_mode is mode
        assert cache.shape == pool_shape
        assert cache.dtype == torch.int32


@pytest.mark.parametrize("num_tokens", [0, 1, 13])
@pytest.mark.parametrize("num_queries", [0, 1, 9])
def test_should_trace_fake_pack_and_gather_output_shapes_under_aot_eager(
    num_tokens: int,
    num_queries: int,
) -> None:
    torch.compiler.reset()
    counter = CompileCounterWithBackend("aot_eager")

    def _pipeline(
        pool: torch.Tensor,
        latent: torch.Tensor,
        slots: torch.Tensor,
        query_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nope_w, rope_w = dsv41_pack_fp8_kv_rows(latent)
        prior = dsv41_gather_cache_rows(pool, query_rows)
        dsv41_write_sub_rows_in_place(pool, nope_w, slots, after=prior, page_tokens=16)
        return rope_w, prior

    compiled = torch.compile(_pipeline, backend=counter, fullgraph=True)
    with FakeTensorMode():
        pool = torch.empty((2, 8, 128), dtype=torch.int32)
        latent = torch.empty((num_tokens, LATENT_DIM), dtype=torch.bfloat16)
        slots = torch.empty((num_tokens,), dtype=torch.int32)
        query_rows = torch.empty((num_queries,), dtype=torch.int32)

        rope_w, gathered = compiled(pool, latent, slots, query_rows)

        assert counter.frame_count == 1
        assert pool.shape == (2, 8, 128) and pool.dtype == torch.int32
        assert rope_w.shape == (num_tokens, ROPE_WORDS)
        assert rope_w.dtype == torch.int32
        assert gathered.shape == (num_queries, ROW_WORDS)
        assert gathered.dtype == torch.int32


def test_should_trace_all_four_dsv41_ops_with_dynamic_shapes_in_one_graph() -> None:
    torch.compiler.reset()
    counter = CompileCounterWithBackend("aot_eager")

    def _step(
        pool: torch.Tensor,
        latent: torch.Tensor,
        slots: torch.Tensor,
        query_rows: torch.Tensor,
        page_ids: torch.Tensor,
        phase_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nope_w, rope_w = dsv41_pack_fp8_kv_rows(latent)
        prior = dsv41_gather_cache_rows(pool, query_rows)
        dsv41_write_sub_rows_in_place(pool, nope_w, slots, after=prior, page_tokens=16)
        swa_win = dsv41_gather_swa_windows(
            pool,
            page_ids,
            phase_rows,
            window_rows=32,
            rows_per_page=16,
            page_row_stride=32,
            page_row_offset=0,
        )
        return rope_w, prior, swa_win

    compiled = torch.compile(_step, backend=counter, fullgraph=True, dynamic=True)
    with FakeTensorMode():
        pool = torch.empty((4, 32, ROW_WORDS), dtype=torch.int32)
        for num_tokens, num_queries, num_seqs in ((3, 5, 2), (7, 9, 4)):
            latent = torch.empty((num_tokens, LATENT_DIM), dtype=torch.bfloat16)
            slots = torch.empty((num_tokens,), dtype=torch.int32)
            query_rows = torch.empty((num_queries,), dtype=torch.int32)
            page_ids = torch.empty((num_seqs, 3), dtype=torch.int32)
            phase_rows = torch.empty((num_seqs,), dtype=torch.int32)
            for tensor in (latent, slots, query_rows, page_ids, phase_rows):
                mark_dynamic(tensor, 0)
            rope_w, prior, swa_win = compiled(
                pool, latent, slots, query_rows, page_ids, phase_rows
            )
            assert rope_w.shape == (num_tokens, ROPE_WORDS)
            assert prior.shape == (num_queries, ROW_WORDS)
            assert swa_win.shape == (num_seqs, 32, ROW_WORDS)
    assert counter.frame_count == 1


def test_should_wire_after_ordering_dependency_in_fx_graph() -> None:
    torch.compiler.reset()
    captured_no_after: list[tuple[str, int]] = []
    captured_with_after: list[tuple[str, int]] = []

    def _record_to(log: list[tuple[str, int]]):
        def _backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
            del example_inputs
            for node in gm.graph.nodes:
                if node.op in ("call_function", "call_method"):
                    log.append((str(node.target), len(node.args)))
            return gm.forward

        return _backend

    fn_no_after = torch.compile(
        lambda c, u, s: dsv41_write_sub_rows_in_place(c, u, s, page_tokens=8),
        backend=_record_to(captured_no_after),
        fullgraph=True,
    )
    fn_with_after = torch.compile(
        lambda c, u, s, a: dsv41_write_sub_rows_in_place(
            c, u, s, after=a, page_tokens=8
        ),
        backend=_record_to(captured_with_after),
        fullgraph=True,
    )
    with FakeTensorMode():
        c_fake = torch.empty((2, 4, 128), dtype=torch.int32)
        u_fake = torch.empty((4, 32), dtype=torch.int32)
        s_fake = torch.empty((4,), dtype=torch.int32)
        a_fake = torch.empty((2, 128), dtype=torch.int32)
        fn_no_after(c_fake, u_fake, s_fake)
        fn_with_after(c_fake, u_fake, s_fake, a_fake)

    assert ("pallas.dsv41_write_sub_rows.default", 4) in captured_no_after
    assert ("copy_", 2) in captured_no_after
    assert not any("write_sub_rows_after" in name for name, _ in captured_no_after)
    assert ("pallas.dsv41_write_sub_rows_after.default", 5) in captured_with_after
    assert ("copy_", 2) in captured_with_after


def test_should_configure_donate_argnums_only_on_write_ops() -> None:
    write_op = cache_op._write_sub_rows_op  # noqa: SLF001 -- check jax_op donate config
    after_op = cache_op._write_sub_rows_after_op  # noqa: SLF001 -- check jax_op donate config
    pack_op = cache_op._pack_fp8_kv_rows_op  # noqa: SLF001 -- check jax_op donate config
    gather_op = cache_op._gather_cache_rows_op  # noqa: SLF001 -- check jax_op donate config

    assert list(write_op._init_fn.donate_argnums) == [0]  # noqa: SLF001 -- JaxCustomOpDef holds JaxCallable
    assert list(after_op._init_fn.donate_argnums) == [0]  # noqa: SLF001 -- JaxCustomOpDef holds JaxCallable
    assert list(pack_op._init_fn.donate_argnums) == []  # noqa: SLF001 -- JaxCustomOpDef holds JaxCallable
    assert list(gather_op._init_fn.donate_argnums) == []  # noqa: SLF001 -- JaxCustomOpDef holds JaxCallable


@pytest.mark.parametrize(
    ("pool_shape", "expected_flat_shape"),
    [
        ((8, ROW_WORDS), (8, ROW_WORDS)),
        ((2, 4, ROW_WORDS), (8, ROW_WORDS)),
        ((2, 4, 64), (8, 64)),
    ],
)
def test_should_forward_cache_jax_wrappers_and_preserve_pool_shape(
    monkeypatch: pytest.MonkeyPatch,
    pool_shape: tuple[int, ...],
    expected_flat_shape: tuple[int, int],
) -> None:
    write_calls: list[dict[str, object]] = []

    def _spy_write(
        cache_words: jax.Array,
        updates: jax.Array,
        slot_mapping: jax.Array,
        *,
        after: jax.Array | None = None,
        page_tokens: int = 16,
    ) -> jax.Array:
        write_calls.append(
            {
                "shape": cache_words.shape,
                "updates": updates,
                "slots": slot_mapping,
                "after": after,
                "page_tokens": page_tokens,
            }
        )
        return cache_words + 1

    gather_calls: list[tuple[tuple[int, ...], jax.Array]] = []

    def _spy_gather(pool: jax.Array, row_indices: jax.Array) -> jax.Array:
        gather_calls.append((pool.shape, row_indices))
        return jnp.full((row_indices.shape[0], ROW_WORDS), 7, dtype=jnp.int32)

    pack_calls: list[jax.Array] = []

    def _spy_pack(latent: jax.Array) -> tuple[jax.Array, jax.Array]:
        pack_calls.append(latent)
        return (
            jnp.full((latent.shape[0], ROW_WORDS), 3, dtype=jnp.int32),
            jnp.full((latent.shape[0], ROPE_WORDS), 5, dtype=jnp.int32),
        )

    monkeypatch.setattr(cache_op, "write_sub_rows", _spy_write)
    monkeypatch.setattr(cache_op, "gather_cache_rows", _spy_gather)
    monkeypatch.setattr(cache_op, "pack_fp8_kv_rows", _spy_pack)

    pool = jnp.zeros(pool_shape, dtype=jnp.int32)
    updates = jnp.ones((3, 32), dtype=jnp.int32)
    slots = jnp.array([0, 1, 2], dtype=jnp.int32)
    after = jnp.array([42, 99], dtype=jnp.int32)
    row_indices = jnp.array([1, 0], dtype=jnp.int32)
    latent = jnp.zeros((2, LATENT_DIM), dtype=jnp.bfloat16)

    out_write = cache_op._write_sub_rows_jax(pool, updates, slots, 8)  # noqa: SLF001 -- test internal JAX bridge wrapper directly
    out_after = cache_op._write_sub_rows_after_jax(pool, updates, slots, 32, after)  # noqa: SLF001 -- test internal JAX bridge wrapper directly
    out_gather = cache_op._gather_cache_rows_jax(pool, row_indices)  # noqa: SLF001 -- test internal JAX bridge wrapper directly
    out_nope, out_rope = cache_op._pack_fp8_kv_rows_jax(latent)  # noqa: SLF001 -- test internal JAX bridge wrapper directly

    assert out_write.shape == pool_shape
    assert out_after.shape == pool_shape
    np.testing.assert_array_equal(np.asarray(out_write), 1)
    np.testing.assert_array_equal(np.asarray(out_after), 1)
    assert write_calls == [
        {
            "shape": expected_flat_shape,
            "updates": updates,
            "slots": slots,
            "after": None,
            "page_tokens": 8,
        },
        {
            "shape": expected_flat_shape,
            "updates": updates,
            "slots": slots,
            "after": after,
            "page_tokens": 32,
        },
    ]
    assert gather_calls == [(expected_flat_shape, row_indices)]
    assert out_gather.shape == (2, ROW_WORDS)
    assert pack_calls == [latent]
    assert out_nope.shape == (2, ROW_WORDS)
    assert out_rope.shape == (2, ROPE_WORDS)


def test_should_fail_fast_when_cache_inputs_or_target_device_are_invalid() -> None:
    valid_cache = torch.zeros((4, 128), dtype=torch.int32)
    valid_updates = torch.zeros((2, 32), dtype=torch.int32)
    valid_slots = torch.zeros((2,), dtype=torch.int32)

    # Malformed 3D width (2, 4, 64) must NOT silently reshape into (4, 128).
    bad_3d_cache = torch.zeros((2, 4, 64), dtype=torch.int32)
    with pytest.raises(ValueError, match="128"):
        dsv41_write_sub_rows_in_place(bad_3d_cache, valid_updates, valid_slots)

    with pytest.raises(ValueError, match="128"):
        dsv41_write_sub_rows_in_place(
            bad_3d_cache, valid_updates, valid_slots, after=valid_slots
        )

    with pytest.raises(ValueError, match="128"):
        dsv41_write_sub_rows_in_place(
            torch.zeros((1, 2, 4, 128), dtype=torch.int32),
            valid_updates,
            valid_slots,
        )

    with pytest.raises(ValueError, match="int32"):
        dsv41_write_sub_rows_in_place(
            torch.zeros((4, 128), dtype=torch.float32),
            valid_updates,
            valid_slots,
        )

    with pytest.raises(ValueError, match=r"32\|64\|128"):
        dsv41_write_sub_rows_in_place(
            valid_cache, torch.zeros((2, 48), dtype=torch.int32), valid_slots
        )

    with pytest.raises(ValueError, match="power of two"):
        dsv41_write_sub_rows_in_place(
            valid_cache, valid_updates, valid_slots, page_tokens=12
        )

    with pytest.raises(ValueError, match="power of two"):
        dsv41_write_sub_rows_in_place(
            valid_cache,
            valid_updates,
            valid_slots,
            after=valid_slots,
            page_tokens=12,
        )

    with pytest.raises(ValueError, match="512"):
        dsv41_pack_fp8_kv_rows(torch.zeros((2, 256), dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="128"):
        dsv41_gather_cache_rows(bad_3d_cache, valid_slots)

    v6e_mesh = AbstractMesh((), (), abstract_device=AbstractDevice("TPU v6e", 1, "tpu"))
    with use_abstract_mesh(v6e_mesh), pytest.raises(RuntimeError, match="TPU7x"):
        dsv41_write_sub_rows_in_place(valid_cache, valid_updates, valid_slots)


@pytest.mark.tpu_test
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_tpu_should_execute_2d_and_3d_cache_custom_ops_end_to_end(
    compiled: bool,
) -> None:
    device = torch.device("tpu")
    pool_3d = torch.zeros((3, 64, ROW_WORDS), dtype=torch.int32, device=device)
    cache_2d = torch.zeros((10, ROW_WORDS), dtype=torch.int32, device=device)

    gen = torch.Generator().manual_seed(42)
    latent_cpu = torch.randn((6, LATENT_DIM), dtype=torch.float32, generator=gen).to(
        torch.bfloat16
    )
    latent_tpu = latent_cpu.to(device)
    slots_tpu = torch.tensor(
        [64, 65, -1, 130, 64, 70], dtype=torch.int32, device=device
    )
    queries_tpu = torch.tensor(
        [64, 65, 130, -1, 70, 999], dtype=torch.int32, device=device
    )
    rope_slots_tpu = torch.tensor([0, 1, -1, 3, 0, 4], dtype=torch.int32, device=device)

    def _step(
        pool: torch.Tensor,
        rope_pool: torch.Tensor,
        latent: torch.Tensor,
        slots: torch.Tensor,
        queries: torch.Tensor,
        rope_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nope_w, rope_w = dsv41_pack_fp8_kv_rows(latent)
        dsv41_write_sub_rows_in_place(pool, nope_w, slots, page_tokens=16)
        gathered = dsv41_gather_cache_rows(pool, queries)
        dsv41_write_sub_rows_in_place(
            rope_pool, rope_w, rope_slots, after=gathered, page_tokens=8
        )
        return nope_w, rope_w, gathered

    if compiled:
        torch.compiler.reset()
        step_fn = torch.compile(_step, backend="tpu", fullgraph=True)
        nope_tpu, rope_tpu, gathered_tpu = step_fn(
            pool_3d, cache_2d, latent_tpu, slots_tpu, queries_tpu, rope_slots_tpu
        )
        torch.accelerator.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            nope_tpu, rope_tpu, gathered_tpu = step_fn(
                pool_3d, cache_2d, latent_tpu, slots_tpu, queries_tpu, rope_slots_tpu
            )
            torch.accelerator.synchronize()
        assert not any(
            "tt_jit_execute_as_strided" in evt.key for evt in prof.key_averages()
        )
    else:
        nope_tpu, rope_tpu, gathered_tpu = _step(
            pool_3d, cache_2d, latent_tpu, slots_tpu, queries_tpu, rope_slots_tpu
        )
        torch.accelerator.synchronize()

    nope_cpu = nope_tpu.cpu().numpy()
    rope_cpu = rope_tpu.cpu().numpy()
    gathered_cpu = gathered_tpu.cpu().numpy()
    pool_cpu = pool_3d.cpu().numpy()
    cache_2d_cpu = cache_2d.cpu().numpy()
    zeros_row = np.zeros(ROW_WORDS, dtype=np.int32)

    assert pool_cpu.shape == (3, 64, ROW_WORDS)
    assert rope_cpu.shape == (6, ROPE_WORDS)
    np.testing.assert_allclose(
        _dequantize_packed_nope(nope_cpu),
        latent_cpu[:, :NOPE_DIM].float().numpy(),
        rtol=0.15,
        atol=1e-2,
    )
    np.testing.assert_array_equal(pool_cpu[1, 0], nope_cpu[4])
    np.testing.assert_array_equal(pool_cpu[1, 1], nope_cpu[1])
    np.testing.assert_array_equal(pool_cpu[2, 2], nope_cpu[3])
    np.testing.assert_array_equal(pool_cpu[1, 6], nope_cpu[5])

    np.testing.assert_array_equal(gathered_cpu[0], nope_cpu[4])
    np.testing.assert_array_equal(gathered_cpu[1], nope_cpu[1])
    np.testing.assert_array_equal(gathered_cpu[2], nope_cpu[3])
    np.testing.assert_array_equal(gathered_cpu[3], zeros_row)
    np.testing.assert_array_equal(gathered_cpu[4], nope_cpu[5])
    np.testing.assert_array_equal(gathered_cpu[5], zeros_row)

    np.testing.assert_array_equal(cache_2d_cpu[0, 0:32], rope_cpu[4])
    np.testing.assert_array_equal(cache_2d_cpu[0, 32:64], rope_cpu[1])
    np.testing.assert_array_equal(cache_2d_cpu[0, 64:96], zeros_row[:32])
    np.testing.assert_array_equal(cache_2d_cpu[0, 96:128], rope_cpu[3])
    np.testing.assert_array_equal(cache_2d_cpu[1, 0:32], rope_cpu[5])
