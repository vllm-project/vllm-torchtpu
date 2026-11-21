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
"""CPU stub, FakeTensor, and TPU7x tests for DSv4.1 single-bank cache ops."""

import inspect
from collections.abc import Iterator
from typing import Any

import jax
import jax.extend.core
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
    swa_window_op,
)

ROW_WORDS = 128
ROPE_WORDS = 32
LATENT_DIM = 512
NOPE_DIM = 448


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


def _unwrap_jax(tensor: torch.Tensor) -> jax.Array:
    assert isinstance(tensor, _JaxBackedTensor)
    return tensor.elem


def _clear_jax_op_caches() -> None:
    ops: tuple[Any, ...] = (
        cache_op._write_sub_rows_op,  # noqa: SLF001 -- clear per-op cache
        cache_op._write_sub_rows_after_op,  # noqa: SLF001 -- clear per-op cache
        cache_op._pack_fp8_kv_rows_op,  # noqa: SLF001 -- clear per-op cache
        cache_op._gather_cache_rows_op,  # noqa: SLF001 -- clear per-op cache
        swa_window_op._gather_swa_windows_op,  # noqa: SLF001 -- clear per-op cache
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
    if cache_words.ndim != 2 or cache_words.shape[1] != ROW_WORDS:
        raise ValueError("cache_words must be int32[num_rows, 128].")
    if cache_words.dtype != jnp.int32:
        raise ValueError("cache_words must be int32[num_rows, 128].")
    if page_tokens <= 0 or (page_tokens & (page_tokens - 1)) != 0:
        raise ValueError("page_tokens must be a positive power of two.")
    slice_words = updates.shape[1]
    slices_per_row = ROW_WORDS // slice_words
    num_rows = cache_words.shape[0]
    total_slots = num_rows * slices_per_row
    out = cache_words.reshape(total_slots, slice_words)
    if after is not None:
        out = out + (jnp.sum(after).astype(jnp.int32) * 0)

    def _apply(token_idx: jax.Array, acc: jax.Array) -> jax.Array:
        slot = slot_mapping[token_idx]
        valid = (slot >= 0) & (slot < total_slots)
        safe_slot = jnp.where(valid, slot, 0)
        new_val = jnp.where(valid, updates[token_idx], acc[safe_slot])
        return acc.at[safe_slot].set(new_val)

    out = jax.lax.fori_loop(0, updates.shape[0], _apply, out)
    return out.reshape(num_rows, ROW_WORDS)


def _stub_pack_fp8_kv_rows(latent: jax.Array) -> tuple[jax.Array, jax.Array]:
    if (
        latent.ndim != 2
        or latent.shape[1] != LATENT_DIM
        or latent.dtype != jnp.bfloat16
    ):
        raise ValueError("latent must be bf16[num_tokens, 512].")
    num_tokens = latent.shape[0]
    rope_u16 = jax.lax.bitcast_convert_type(latent[:, NOPE_DIM:], jnp.uint16)
    rope_w = (
        rope_u16[:, :ROPE_WORDS].astype(jnp.uint32)
        | (rope_u16[:, ROPE_WORDS:].astype(jnp.uint32) << 16)
    ).astype(jnp.int32)
    nope_u16 = jax.lax.bitcast_convert_type(latent[:, :ROW_WORDS], jnp.uint16)
    nope_w = (
        nope_u16.astype(jnp.int32) + jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
    )
    return nope_w, rope_w


def _stub_gather_cache_rows(pool: jax.Array, row_indices: jax.Array) -> jax.Array:
    if pool.ndim != 2 or pool.shape[1] != ROW_WORDS or pool.dtype != jnp.int32:
        raise ValueError("pool must be int32[num_rows, 128].")
    valid = (row_indices >= 0) & (row_indices < pool.shape[0])
    safe_idx = jnp.where(valid, row_indices, 0)
    return jnp.where(valid[:, None], pool[safe_idx], jnp.int32(0))


def _reference_write_sub_rows(
    cache_words: np.ndarray,
    updates: np.ndarray,
    slot_mapping: np.ndarray,
) -> np.ndarray:
    orig_shape = cache_words.shape
    flat = cache_words.reshape(-1, ROW_WORDS).copy()
    slice_words = updates.shape[1]
    slices_per_row = ROW_WORDS // slice_words
    total_slots = flat.shape[0] * slices_per_row
    for token_idx, slot in enumerate(slot_mapping.tolist()):
        if 0 <= slot < total_slots:
            row = slot // slices_per_row
            col = (slot % slices_per_row) * slice_words
            flat[row, col : col + slice_words] = updates[token_idx]
    return flat.reshape(orig_shape)


def _reference_gather_cache_rows(
    pool: np.ndarray,
    row_indices: np.ndarray,
) -> np.ndarray:
    flat = pool.reshape(-1, ROW_WORDS)
    out = np.zeros((row_indices.shape[0], ROW_WORDS), dtype=np.int32)
    for out_idx, row_idx in enumerate(row_indices.tolist()):
        if 0 <= row_idx < flat.shape[0]:
            out[out_idx] = flat[row_idx]
    return out


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
        "dsv41_gather_sparse_fp8_kv",
        "dsv41_gather_swa_windows",
        "dsv41_pack_fp8_kv_rows",
        "dsv41_paged_indexer_topk",
        "dsv41_sparse_mla_swa_attention",
        "dsv41_write_sub_rows_in_place",
        "get_mhc_post_op",
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
def test_should_scale_fake_pack_and_gather_linearly_under_aot_eager(
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


def test_should_wire_after_ordering_dependency_in_fx_graph_and_jaxpr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_op, "write_sub_rows", _stub_write_sub_rows)
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

    def _trace_ordered(
        c: jax.Array, u: jax.Array, s: jax.Array, a: jax.Array
    ) -> jax.Array:
        cache_t = _JaxBackedTensor(c)
        dsv41_write_sub_rows_in_place(
            cache_t,
            _JaxBackedTensor(u),
            _JaxBackedTensor(s),
            after=_JaxBackedTensor(a),
            page_tokens=8,
        )
        return _unwrap_jax(cache_t)

    closed = jax.make_jaxpr(_trace_ordered)(
        jax.ShapeDtypeStruct((2, 4, 128), jnp.int32),
        jax.ShapeDtypeStruct((4, 32), jnp.int32),
        jax.ShapeDtypeStruct((4,), jnp.int32),
        jax.ShapeDtypeStruct((2, 128), jnp.int32),
    )
    jit_jaxpr = closed.jaxpr.eqns[0].params["jaxpr"].jaxpr
    after_invar = jit_jaxpr.invars[3]
    used_invars = {
        var
        for eqn in jit_jaxpr.eqns
        for var in eqn.invars
        if isinstance(var, jax.extend.core.Var)
    }
    assert after_invar in used_invars


@pytest.mark.parametrize("pool_shape", [(8, 128), (2, 4, 128)])
def test_should_donate_only_cache_words_in_write_sub_rows_jaxpr(
    monkeypatch: pytest.MonkeyPatch,
    pool_shape: tuple[int, ...],
) -> None:
    monkeypatch.setattr(cache_op, "write_sub_rows", _stub_write_sub_rows)
    cache_spec = jax.ShapeDtypeStruct(pool_shape, jnp.int32)
    updates_spec = jax.ShapeDtypeStruct((4, 32), jnp.int32)
    slots_spec = jax.ShapeDtypeStruct((4,), jnp.int32)
    after_spec = jax.ShapeDtypeStruct((2, 128), jnp.int32)

    def _trace_unordered(c: jax.Array, u: jax.Array, s: jax.Array) -> jax.Array:
        cache_t = _JaxBackedTensor(c)
        dsv41_write_sub_rows_in_place(
            cache_t, _JaxBackedTensor(u), _JaxBackedTensor(s), page_tokens=8
        )
        return _unwrap_jax(cache_t)

    closed_unordered = jax.make_jaxpr(_trace_unordered)(
        cache_spec, updates_spec, slots_spec
    )
    donated_unordered = closed_unordered.jaxpr.eqns[0].params["donated_invars"]
    assert donated_unordered[0] is True
    assert all(not flag for flag in donated_unordered[1:])

    def _trace_ordered(
        c: jax.Array, u: jax.Array, s: jax.Array, a: jax.Array
    ) -> jax.Array:
        cache_t = _JaxBackedTensor(c)
        dsv41_write_sub_rows_in_place(
            cache_t,
            _JaxBackedTensor(u),
            _JaxBackedTensor(s),
            after=_JaxBackedTensor(a),
            page_tokens=8,
        )
        return _unwrap_jax(cache_t)

    closed_ordered = jax.make_jaxpr(_trace_ordered)(
        cache_spec, updates_spec, slots_spec, after_spec
    )
    donated_ordered = closed_ordered.jaxpr.eqns[0].params["donated_invars"]
    assert donated_ordered[0] is True
    assert all(not flag for flag in donated_ordered[1:])


@pytest.mark.parametrize("pool_shape", [(8, 128), (2, 4, 128)])
@pytest.mark.parametrize("slice_words", [32, 64, 128])
def test_should_write_sub_rows_into_2d_and_3d_pools_via_bridge(
    monkeypatch: pytest.MonkeyPatch,
    pool_shape: tuple[int, ...],
    slice_words: int,
) -> None:
    seen_page_tokens: list[int] = []

    def _recording_write_stub(
        cache_words: jax.Array,
        updates: jax.Array,
        slot_mapping: jax.Array,
        *,
        after: jax.Array | None = None,
        page_tokens: int = 16,
    ) -> jax.Array:
        seen_page_tokens.append(page_tokens)
        return _stub_write_sub_rows(
            cache_words,
            updates,
            slot_mapping,
            after=after,
            page_tokens=page_tokens,
        )

    monkeypatch.setattr(cache_op, "write_sub_rows", _recording_write_stub)
    rng = np.random.default_rng(100 + slice_words + len(pool_shape))
    init_pool = rng.integers(10_000, 90_000, size=pool_shape, dtype=np.int32)
    updates_np = rng.integers(100_000, 900_000, size=(7, slice_words), dtype=np.int32)
    slices_per_row = ROW_WORDS // slice_words
    max_slot = 8 * slices_per_row
    slots_np = np.array(
        [
            5,
            -1,  # negative null slot -> ignored
            0,
            max_slot,  # exact upper OOB boundary -> ignored
            5,  # duplicate -> last writer (token 4) wins over token 0
            max_slot - 1,  # exact last valid in-bank slot
            3,
        ],
        dtype=np.int32,
    )
    expected = _reference_write_sub_rows(init_pool, updates_np, slots_np)

    cache_t = _JaxBackedTensor(jnp.asarray(init_pool))
    res = dsv41_write_sub_rows_in_place(
        cache_t,
        _JaxBackedTensor(jnp.asarray(updates_np)),
        _JaxBackedTensor(jnp.asarray(slots_np)),
    )
    assert res is None
    assert cache_t.shape == pool_shape
    np.testing.assert_array_equal(cache_t.to_numpy(), expected)

    cache_after_t = _JaxBackedTensor(jnp.asarray(init_pool))
    res_after = dsv41_write_sub_rows_in_place(
        cache_after_t,
        _JaxBackedTensor(jnp.asarray(updates_np)),
        _JaxBackedTensor(jnp.asarray(slots_np)),
        after=_JaxBackedTensor(jnp.array([42, 99], dtype=jnp.int32)),
        page_tokens=8,
    )
    assert res_after is None
    assert cache_after_t.shape == pool_shape
    np.testing.assert_array_equal(cache_after_t.to_numpy(), expected)
    assert 16 in seen_page_tokens
    assert 8 in seen_page_tokens


def test_should_retrace_on_same_shape_after_clearing_jax_op_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_t = _JaxBackedTensor(jnp.zeros((4, ROW_WORDS), dtype=jnp.int32))
    idx_t = _JaxBackedTensor(jnp.array([0, 1], dtype=jnp.int32))

    monkeypatch.setattr(
        cache_op,
        "gather_cache_rows",
        lambda pool, idx: jnp.full((idx.shape[0], ROW_WORDS), 11, dtype=jnp.int32),
    )
    first = dsv41_gather_cache_rows(pool_t, idx_t)
    assert isinstance(first, _JaxBackedTensor)
    np.testing.assert_array_equal(first.to_numpy(), 11)

    _clear_jax_op_caches()
    monkeypatch.setattr(
        cache_op,
        "gather_cache_rows",
        lambda pool, idx: jnp.full((idx.shape[0], ROW_WORDS), 22, dtype=jnp.int32),
    )
    second = dsv41_gather_cache_rows(pool_t, idx_t)
    assert isinstance(second, _JaxBackedTensor)
    np.testing.assert_array_equal(second.to_numpy(), 22)


@pytest.mark.parametrize("use_3d_pool", [False, True])
def test_should_pack_and_gather_single_bank_2d_3d_pools_via_bridge(
    monkeypatch: pytest.MonkeyPatch,
    use_3d_pool: bool,
) -> None:
    monkeypatch.setattr(cache_op, "pack_fp8_kv_rows", _stub_pack_fp8_kv_rows)
    monkeypatch.setattr(cache_op, "gather_cache_rows", _stub_gather_cache_rows)
    rng = np.random.default_rng(77 if use_3d_pool else 55)
    latent_bf16 = jnp.asarray(
        (rng.standard_normal((5, LATENT_DIM)) * 1.75).astype(np.float32),
        dtype=jnp.bfloat16,
    )

    nope_t, rope_t = dsv41_pack_fp8_kv_rows(_JaxBackedTensor(latent_bf16))
    assert isinstance(nope_t, _JaxBackedTensor)
    assert isinstance(rope_t, _JaxBackedTensor)
    assert nope_t.shape == (5, ROW_WORDS)
    assert rope_t.shape == (5, ROPE_WORDS)

    rope_u16 = np.asarray(
        jax.lax.bitcast_convert_type(latent_bf16[:, NOPE_DIM:], jnp.uint16)
    )
    expected_rope = (
        rope_u16[:, :ROPE_WORDS].astype(np.uint32)
        | (rope_u16[:, ROPE_WORDS:].astype(np.uint32) << 16)
    ).astype(np.int32)
    np.testing.assert_array_equal(rope_t.to_numpy(), expected_rope)

    pool_shape = (3, 4, ROW_WORDS) if use_3d_pool else (12, ROW_WORDS)
    pool_np = rng.integers(1, 500_000, size=pool_shape, dtype=np.int32)
    indices_np = np.array([11, 0, -1, 4, 12, 7, 9, -99], dtype=np.int32)

    gathered = dsv41_gather_cache_rows(
        _JaxBackedTensor(jnp.asarray(pool_np)),
        _JaxBackedTensor(jnp.asarray(indices_np)),
    )
    assert isinstance(gathered, _JaxBackedTensor)
    np.testing.assert_array_equal(
        gathered.to_numpy(),
        _reference_gather_cache_rows(pool_np, indices_np),
    )


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
def test_tpu_should_execute_2d_and_3d_cache_custom_ops_end_to_end() -> None:
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

    @torch.compile(backend="tpu", fullgraph=True)
    def _compiled_step(
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

    nope_tpu, rope_tpu, gathered_tpu = _compiled_step(
        pool_3d, cache_2d, latent_tpu, slots_tpu, queries_tpu, rope_slots_tpu
    )
    torch.accelerator.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        nope_tpu, rope_tpu, gathered_tpu = _compiled_step(
            pool_3d, cache_2d, latent_tpu, slots_tpu, queries_tpu, rope_slots_tpu
        )
        torch.accelerator.synchronize()
    assert not any(
        "tt_jit_execute_as_strided" in evt.key for evt in prof.key_averages()
    )

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
