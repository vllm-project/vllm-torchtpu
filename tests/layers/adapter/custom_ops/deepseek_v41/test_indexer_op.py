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
"""CPU stub, FakeTensor, and TPU7x tests for DSv4.1 indexer custom op."""

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

import vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 as dsv41_ops
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
    paged_indexer_topk,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    dsv41_paged_indexer_topk,
    dsv41_write_sub_rows_in_place,
    indexer_op,
)

INDEXER_DIM = 128
ROW_WORDS = 128
KEY_WORDS = 64
KEYS_PER_ROW = ROW_WORDS // KEY_WORDS


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

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._elem)


def _clear_jax_op_caches() -> None:
    ops: tuple[Any, ...] = (
        indexer_op._paged_indexer_topk_op,  # noqa: SLF001 -- clear per-op cache
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


def _pack_split_bf16_keys_to_words(split_keys_bf16: jax.Array) -> np.ndarray:
    u16 = np.asarray(jax.lax.bitcast_convert_type(split_keys_bf16, jnp.uint16))
    return (
        u16[..., :KEY_WORDS].astype(np.uint32)
        | (u16[..., KEY_WORDS:].astype(np.uint32) << 16)
    ).astype(np.int32)


def _u16_bf16_to_f32(bits_u16: np.ndarray) -> np.ndarray:
    return (bits_u16.astype(np.uint32) << 16).view(np.float32)


def _stub_paged_indexer_topk(
    q: jax.Array,
    weights: jax.Array,
    k_pool: jax.Array,
    block_tables: jax.Array,
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    *,
    topk: int,
    candidate_indices: jax.Array | None = None,
    compress_ratio: int = 1,
) -> jax.Array:
    num_tokens = q.shape[0]
    num_pages, page_rows, _ = k_pool.shape
    num_seqs, max_pages = block_tables.shape
    if (
        num_tokens == 0
        or num_seqs == 0
        or max_pages == 0
        or (candidate_indices is not None and candidate_indices.shape[1] == 0)
    ):
        return jnp.full((num_tokens, topk), -1, dtype=jnp.int32)

    page_ok = (block_tables >= 0) & (block_tables < num_pages)
    safe_bt = jnp.where(page_ok, block_tables, 0)
    gathered_words = jnp.where(
        page_ok[:, :, None, None], k_pool[safe_bt], jnp.int32(0)
    ).reshape(num_seqs, max_pages * page_rows, KEYS_PER_ROW, KEY_WORDS)
    words_u32 = gathered_words.astype(jnp.uint32)
    even_bf16 = jax.lax.bitcast_convert_type(
        (words_u32 & jnp.uint32(0xFFFF)).astype(jnp.uint16), jnp.bfloat16
    )
    odd_bf16 = jax.lax.bitcast_convert_type(
        (words_u32 >> jnp.uint32(16)).astype(jnp.uint16), jnp.bfloat16
    )
    keys_f32 = (
        jnp.concatenate([even_bf16, odd_bf16], axis=-1)
        .reshape(num_seqs, max_pages * page_rows * KEYS_PER_ROW, INDEXER_DIM)
        .astype(jnp.float32)
    )

    tok_iota = jnp.arange(num_tokens, dtype=jnp.int32)
    in_seq = (tok_iota[:, None] >= cu_q_lens[:-1][None, :]) & (
        tok_iota[:, None] < cu_q_lens[1:][None, :]
    )
    active = jnp.any(in_seq, axis=1)
    seq_id = jnp.argmax(in_seq, axis=1).astype(jnp.int32)
    tok_pos = seq_lens[seq_id] - (cu_q_lens[seq_id + 1] - tok_iota)
    eff_len = jnp.where(active, jnp.maximum(0, (tok_pos + 1) // compress_ratio), 0)

    max_keys = max_pages * page_rows * KEYS_PER_ROW
    key_page = jnp.arange(max_keys, dtype=jnp.int32) // (page_rows * KEYS_PER_ROW)
    key_ok = page_ok[seq_id][:, key_page] & (
        jnp.arange(max_keys, dtype=jnp.int32)[None, :] < eff_len[:, None]
    )
    raw_dots = jnp.einsum("thd,tkd->thk", q.astype(jnp.float32), keys_f32[seq_id])
    scores = jnp.sum(jnp.maximum(raw_dots, 0.0) * weights, axis=1)
    base_scores = jnp.where(key_ok, scores, -jnp.inf)

    if candidate_indices is None:
        eff_k = min(topk, max_keys)
        top_vals, top_idxs = jax.lax.top_k(base_scores, eff_k)
        out = jnp.where(jnp.isfinite(top_vals), top_idxs, -1)
        if eff_k < topk:
            out = jnp.pad(out, ((0, 0), (0, topk - eff_k)), constant_values=-1)
        return out.astype(jnp.int32)

    valid_cand = (candidate_indices >= 0) & (candidate_indices < max_keys)
    cand_sorted = jnp.sort(jnp.where(valid_cand, candidate_indices, max_keys), axis=1)
    dedup = (cand_sorted < max_keys) & jnp.concatenate(
        [
            jnp.ones((num_tokens, 1), dtype=jnp.bool_),
            cand_sorted[:, 1:] != cand_sorted[:, :-1],
        ],
        axis=1,
    )
    safe_cand = jnp.where(dedup, cand_sorted, 0)
    tok_col = tok_iota[:, None]
    cand_scores = jnp.where(dedup, base_scores[tok_col, safe_cand], -jnp.inf)
    cand_k = min(topk, candidate_indices.shape[1])
    top_vals, local_idxs = jax.lax.top_k(cand_scores, cand_k)
    mapped = jnp.take_along_axis(cand_sorted, local_idxs, axis=1)
    out = jnp.where(jnp.isfinite(top_vals), mapped, -1)
    if cand_k < topk:
        out = jnp.pad(out, ((0, 0), (0, topk - cand_k)), constant_values=-1)
    return out.astype(jnp.int32)


def _reference_paged_indexer_topk(
    q_bf16: jax.Array,
    weights_f32: np.ndarray,
    k_pool: np.ndarray,
    block_tables: np.ndarray,
    seq_lens: np.ndarray,
    cu_q_lens: np.ndarray,
    *,
    topk: int,
    candidate_indices: np.ndarray | None = None,
    compress_ratio: int = 1,
) -> np.ndarray:
    num_tokens = q_bf16.shape[0]
    num_pages, page_rows, _ = k_pool.shape
    num_seqs, max_pages = block_tables.shape
    if (
        num_tokens == 0
        or num_seqs == 0
        or max_pages == 0
        or (candidate_indices is not None and candidate_indices.shape[1] == 0)
    ):
        return np.full((num_tokens, topk), -1, dtype=np.int32)

    q_f32 = np.asarray(q_bf16, dtype=np.float32)
    out = np.full((num_tokens, topk), -1, dtype=np.int32)
    keys_per_page = page_rows * KEYS_PER_ROW
    max_keys = max_pages * keys_per_page

    for seq in range(num_seqs):
        start_tok = int(cu_q_lens[seq])
        end_tok = int(cu_q_lens[seq + 1])
        seq_len = int(seq_lens[seq])
        seq_keys = np.zeros((max_keys, INDEXER_DIM), dtype=np.float32)
        valid_page = np.zeros((max_keys,), dtype=bool)
        for lp, pp in enumerate(block_tables[seq].tolist()):
            if 0 <= pp < num_pages:
                words = k_pool[pp].reshape(keys_per_page, KEY_WORDS).astype(np.uint32)
                even = _u16_bf16_to_f32((words & 0xFFFF).astype(np.uint16))
                odd = _u16_bf16_to_f32((words >> 16).astype(np.uint16))
                dst = slice(lp * keys_per_page, (lp + 1) * keys_per_page)
                seq_keys[dst] = np.concatenate([even, odd], axis=1)
                valid_page[dst] = True

        for tok in range(max(0, start_tok), min(num_tokens, end_tok)):
            tok_pos = seq_len - (end_tok - tok)
            eff_len = min(max(0, (tok_pos + 1) // compress_ratio), max_keys)
            dots = q_f32[tok] @ seq_keys.T
            scores = (np.maximum(dots, 0.0) * weights_f32[tok]).sum(axis=0)
            valid_key = valid_page & (np.arange(max_keys) < eff_len)
            if candidate_indices is None:
                cands = np.nonzero(valid_key)[0]
            else:
                uniq = np.unique(candidate_indices[tok])
                cands = uniq[
                    (uniq >= 0)
                    & (uniq < max_keys)
                    & valid_key[np.clip(uniq, 0, max_keys - 1)]
                ]
            if cands.size == 0:
                continue
            order = np.argsort(-scores[cands], kind="stable")[:topk]
            out[tok, : order.size] = cands[order].astype(np.int32)
    return out


def test_should_export_single_bank_indexer_op_without_inert_partition_specs() -> None:
    assert indexer_op.__all__ == ["dsv41_paged_indexer_topk"]
    assert "dsv41_paged_indexer_topk" in dsv41_ops.__all__
    assert dsv41_ops.dsv41_paged_indexer_topk is dsv41_paged_indexer_topk

    sig = inspect.signature(dsv41_paged_indexer_topk)
    assert list(sig.parameters) == [
        "q",
        "weights",
        "k_pool",
        "block_tables",
        "seq_lens",
        "cu_q_lens",
        "topk",
        "candidate_indices",
        "compress_ratio",
    ]
    assert sig.parameters["candidate_indices"].default is None
    assert sig.parameters["compress_ratio"].default == 1
    assert not any("PARTITION_SPECS" in name for name in dir(indexer_op))


@pytest.mark.parametrize("with_candidates", [False, True])
@pytest.mark.parametrize("num_tokens", [0, 7])
@pytest.mark.parametrize("topk", [16, 2048])
def test_should_trace_fake_paged_indexer_topk_with_fullgraph_compile(
    with_candidates: bool,
    num_tokens: int,
    topk: int,
) -> None:
    torch.compiler.reset()
    counter = CompileCounterWithBackend("aot_eager")

    def _run(
        q: torch.Tensor,
        w: torch.Tensor,
        k_pool: torch.Tensor,
        bt: torch.Tensor,
        sl: torch.Tensor,
        cu: torch.Tensor,
        cand: torch.Tensor | None,
    ) -> torch.Tensor:
        return dsv41_paged_indexer_topk(
            q,
            w,
            k_pool,
            bt,
            sl,
            cu,
            topk=topk,
            candidate_indices=cand,
            compress_ratio=2,
        )

    compiled = torch.compile(_run, backend=counter, fullgraph=True)
    with FakeTensorMode() as mode:
        q = torch.empty((num_tokens, 8, INDEXER_DIM), dtype=torch.bfloat16)
        w = torch.empty((num_tokens, 8, 1), dtype=torch.float32)
        k_pool = torch.empty((4, 8, ROW_WORDS), dtype=torch.int32)
        bt = torch.empty((2, 3), dtype=torch.int32)
        sl = torch.empty((2,), dtype=torch.int32)
        cu = torch.empty((3,), dtype=torch.int32)
        cand = (
            torch.empty((num_tokens, 32), dtype=torch.int32)
            if with_candidates
            else None
        )

        out = compiled(q, w, k_pool, bt, sl, cu, cand)

        assert counter.frame_count == 1
        assert isinstance(out, FakeTensor) and out.fake_mode is mode
        assert out.shape == (num_tokens, topk)
        assert out.dtype == torch.int32


def test_should_bridge_single_bank_paged_indexer_topk_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_calls: list[dict[str, object]] = []

    def _recording_stub(
        q: jax.Array,
        weights: jax.Array,
        k_pool: jax.Array,
        block_tables: jax.Array,
        seq_lens: jax.Array,
        cu_q_lens: jax.Array,
        *,
        topk: int,
        candidate_indices: jax.Array | None = None,
        compress_ratio: int = 1,
    ) -> jax.Array:
        seen_calls.append(
            {
                "topk": topk,
                "has_candidates": candidate_indices is not None,
                "compress_ratio": compress_ratio,
            }
        )
        return _stub_paged_indexer_topk(
            q,
            weights,
            k_pool,
            block_tables,
            seq_lens,
            cu_q_lens,
            topk=topk,
            candidate_indices=candidate_indices,
            compress_ratio=compress_ratio,
        )

    monkeypatch.setattr(indexer_op, "paged_indexer_topk", _recording_stub)

    rng = np.random.default_rng(404)
    num_tokens, num_heads, num_pages, page_rows = 6, 4, 5, 8
    q_bf16 = jnp.asarray(
        rng.normal(size=(num_tokens, num_heads, INDEXER_DIM)), dtype=jnp.bfloat16
    )
    w_np = rng.uniform(0.25, 2.0, size=(num_tokens, num_heads, 1)).astype(np.float32)
    raw_keys = jnp.asarray(
        rng.normal(size=(num_pages, page_rows * KEYS_PER_ROW, INDEXER_DIM)),
        dtype=jnp.bfloat16,
    )
    # Non-zero page 0 sentinel detects invalid -1 / OOB page reads.
    raw_keys = raw_keys.at[0].set(jnp.bfloat16(10.0))
    k_pool_np = _pack_split_bf16_keys_to_words(raw_keys).reshape(
        num_pages, page_rows, ROW_WORDS
    )
    bt_np = np.array([[1, -1, 3], [2, 4, 99]], dtype=np.int32)
    seq_lens_np = np.array([70, 76], dtype=np.int32)
    cu_q_lens_np = np.array([0, 3, 5], dtype=np.int32)  # tok 5 is inactive padding
    cand_np = np.array(
        [
            [0, 5, 5, -1, 17, 32, 999, 33],
            [1, 3, 7, 16, 33, 2, -3, 33],
            [0, 4, 6, 8, 9, 10, 18, 34],
            [0, 1, 2, 10, 16, 20, 32, 35],
            [2, 3, 9, 11, 22, 31, 36, -1],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ],
        dtype=np.int32,
    )

    q_t = _JaxBackedTensor(q_bf16)
    w_t = _JaxBackedTensor(jnp.asarray(w_np))
    pool_t = _JaxBackedTensor(jnp.asarray(k_pool_np))
    bt_t = _JaxBackedTensor(jnp.asarray(bt_np))
    sl_t = _JaxBackedTensor(jnp.asarray(seq_lens_np))
    cu_t = _JaxBackedTensor(jnp.asarray(cu_q_lens_np))
    cand_t = _JaxBackedTensor(jnp.asarray(cand_np))

    full_out = dsv41_paged_indexer_topk(
        q_t, w_t, pool_t, bt_t, sl_t, cu_t, topk=6, compress_ratio=2
    )
    cand_out = dsv41_paged_indexer_topk(
        q_t,
        w_t,
        pool_t,
        bt_t,
        sl_t,
        cu_t,
        topk=6,
        candidate_indices=cand_t,
        compress_ratio=2,
    )

    assert isinstance(full_out, _JaxBackedTensor)
    assert isinstance(cand_out, _JaxBackedTensor)
    assert seen_calls == [
        {"topk": 6, "has_candidates": False, "compress_ratio": 2},
        {"topk": 6, "has_candidates": True, "compress_ratio": 2},
    ]

    ref_full = _reference_paged_indexer_topk(
        q_bf16,
        w_np,
        k_pool_np,
        bt_np,
        seq_lens_np,
        cu_q_lens_np,
        topk=6,
        compress_ratio=2,
    )
    ref_cand = _reference_paged_indexer_topk(
        q_bf16,
        w_np,
        k_pool_np,
        bt_np,
        seq_lens_np,
        cu_q_lens_np,
        topk=6,
        candidate_indices=cand_np,
        compress_ratio=2,
    )
    np.testing.assert_array_equal(
        np.sort(full_out.to_numpy(), axis=1), np.sort(ref_full, axis=1)
    )
    np.testing.assert_array_equal(
        np.sort(cand_out.to_numpy(), axis=1), np.sort(ref_cand, axis=1)
    )
    np.testing.assert_array_equal(full_out.to_numpy()[5], -1)
    np.testing.assert_array_equal(cand_out.to_numpy()[5], -1)


def test_should_fail_fast_when_paged_indexer_inputs_or_device_are_invalid() -> None:
    q = torch.zeros((4, 8, INDEXER_DIM), dtype=torch.bfloat16)
    w = torch.ones((4, 8, 1), dtype=torch.float32)
    pool = torch.zeros((2, 8, ROW_WORDS), dtype=torch.int32)
    bt = torch.zeros((1, 2), dtype=torch.int32)
    sl = torch.tensor([4], dtype=torch.int32)
    cu = torch.tensor([0, 4], dtype=torch.int32)

    with pytest.raises(ValueError, match="q"):
        dsv41_paged_indexer_topk(
            torch.zeros((4, 8, 64), dtype=torch.bfloat16), w, pool, bt, sl, cu, topk=4
        )
    with pytest.raises(ValueError, match="k_pool"):
        dsv41_paged_indexer_topk(
            q, w, torch.zeros((2, 8, 64), dtype=torch.int32), bt, sl, cu, topk=4
        )
    with pytest.raises(ValueError, match="k_pool"):
        dsv41_paged_indexer_topk(
            q, w, torch.zeros((2, 4, ROW_WORDS), dtype=torch.int32), bt, sl, cu, topk=4
        )
    with pytest.raises(ValueError, match="topk"):
        dsv41_paged_indexer_topk(q, w, pool, bt, sl, cu, topk=0)
    with pytest.raises(ValueError, match="compress_ratio"):
        dsv41_paged_indexer_topk(q, w, pool, bt, sl, cu, topk=4, compress_ratio=3)
    with pytest.raises(ValueError, match="candidate_indices"):
        dsv41_paged_indexer_topk(
            q,
            w,
            pool,
            bt,
            sl,
            cu,
            topk=4,
            candidate_indices=torch.zeros((3, 8), dtype=torch.int32),
        )

    v6e_mesh = AbstractMesh((), (), abstract_device=AbstractDevice("TPU v6e", 1, "tpu"))
    with use_abstract_mesh(v6e_mesh), pytest.raises(RuntimeError, match="TPU7x"):
        dsv41_paged_indexer_topk(q, w, pool, bt, sl, cu, topk=4)


@pytest.mark.tpu_test
def test_tpu_should_execute_single_bank_paged_indexer_topk_under_tpu_compile() -> None:
    device = torch.device("tpu")
    rng = np.random.default_rng(808)
    num_tokens, num_heads, num_pages, page_rows = 6, 8, 6, 16
    keys_per_page = page_rows * KEYS_PER_ROW

    q_cpu = torch.from_numpy(
        rng.normal(size=(num_tokens, num_heads, INDEXER_DIM)).astype(np.float32)
    ).to(torch.bfloat16)
    weights_cpu = torch.from_numpy(
        rng.uniform(0.1, 1.5, size=(num_tokens, num_heads, 1)).astype(np.float32)
    )
    keys_bf16 = jnp.asarray(
        rng.normal(size=(num_pages * keys_per_page, INDEXER_DIM)), dtype=jnp.bfloat16
    )
    packed_subrows = _pack_split_bf16_keys_to_words(keys_bf16)  # [P*page_rows*2, 64]

    pool_tpu = torch.zeros(
        (num_pages, page_rows, ROW_WORDS), dtype=torch.int32, device=device
    )
    updates_tpu = torch.from_numpy(packed_subrows).to(device)
    slots_tpu = torch.arange(
        num_pages * keys_per_page, dtype=torch.int32, device=device
    )
    dsv41_write_sub_rows_in_place(
        pool_tpu, updates_tpu, slots_tpu, page_tokens=keys_per_page
    )
    pool_cpu = pool_tpu.cpu()

    block_tables_cpu = torch.tensor([[1, -1, 4], [2, 5, 0]], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([60, 140], dtype=torch.int32)
    cu_q_lens_cpu = torch.tensor([0, 3, 5], dtype=torch.int32)
    candidates_cpu = torch.from_numpy(
        rng.integers(-2, 100, size=(num_tokens, 32), dtype=np.int32)
    )

    @torch.compile(backend="tpu", fullgraph=True)
    def _compiled_indexer(
        q: torch.Tensor,
        w: torch.Tensor,
        k_pool: torch.Tensor,
        bt: torch.Tensor,
        sl: torch.Tensor,
        cu: torch.Tensor,
        cand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_topk = dsv41_paged_indexer_topk(
            q, w, k_pool, bt, sl, cu, topk=16, compress_ratio=2
        )
        cand_topk = dsv41_paged_indexer_topk(
            q,
            w,
            k_pool,
            bt,
            sl,
            cu,
            topk=16,
            candidate_indices=cand,
            compress_ratio=2,
        )
        return full_topk, cand_topk

    q_tpu = q_cpu.to(device)
    w_tpu = weights_cpu.to(device)
    bt_tpu = block_tables_cpu.to(device)
    sl_tpu = seq_lens_cpu.to(device)
    cu_tpu = cu_q_lens_cpu.to(device)
    cand_tpu = candidates_cpu.to(device)

    full_tpu, cand_tpu_out = _compiled_indexer(
        q_tpu, w_tpu, pool_tpu, bt_tpu, sl_tpu, cu_tpu, cand_tpu
    )
    torch.accelerator.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        full_tpu, cand_tpu_out = _compiled_indexer(
            q_tpu, w_tpu, pool_tpu, bt_tpu, sl_tpu, cu_tpu, cand_tpu
        )
        torch.accelerator.synchronize()
    assert not any(
        "tt_jit_execute_as_strided" in evt.key for evt in prof.key_averages()
    )

    ref_full = np.asarray(
        paged_indexer_topk(
            jnp.asarray(q_cpu.float().numpy(), dtype=jnp.bfloat16),
            jnp.asarray(weights_cpu.numpy()),
            jnp.asarray(pool_cpu.numpy()),
            jnp.asarray(block_tables_cpu.numpy()),
            jnp.asarray(seq_lens_cpu.numpy()),
            jnp.asarray(cu_q_lens_cpu.numpy()),
            topk=16,
            compress_ratio=2,
            interpret=True,
        )
    )
    ref_cand = np.asarray(
        paged_indexer_topk(
            jnp.asarray(q_cpu.float().numpy(), dtype=jnp.bfloat16),
            jnp.asarray(weights_cpu.numpy()),
            jnp.asarray(pool_cpu.numpy()),
            jnp.asarray(block_tables_cpu.numpy()),
            jnp.asarray(seq_lens_cpu.numpy()),
            jnp.asarray(cu_q_lens_cpu.numpy()),
            topk=16,
            candidate_indices=jnp.asarray(candidates_cpu.numpy()),
            compress_ratio=2,
            interpret=True,
        )
    )
    np.testing.assert_array_equal(
        np.sort(full_tpu.cpu().numpy(), axis=1), np.sort(ref_full, axis=1)
    )
    np.testing.assert_array_equal(
        np.sort(cand_tpu_out.cpu().numpy(), axis=1), np.sort(ref_cand, axis=1)
    )

    scale_pool = torch.zeros((4, 32, ROW_WORDS), dtype=torch.int32, device=device)
    scale_bt = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=device)
    scale_sl = torch.tensor([256], dtype=torch.int32, device=device)
    for bucket_tokens in (8, 48, 288):
        scale_q = torch.zeros(
            (bucket_tokens, 8, INDEXER_DIM), dtype=torch.bfloat16, device=device
        )
        scale_w = torch.ones((bucket_tokens, 8, 1), dtype=torch.float32, device=device)
        scale_cu = torch.tensor([0, bucket_tokens], dtype=torch.int32, device=device)
        for topk in (512, 2048):
            scaled = dsv41_paged_indexer_topk(
                scale_q, scale_w, scale_pool, scale_bt, scale_sl, scale_cu, topk=topk
            )
            assert scaled.shape == (bucket_tokens, topk)
            assert scaled.dtype == torch.int32
