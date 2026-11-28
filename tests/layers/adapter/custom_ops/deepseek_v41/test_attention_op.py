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
"""CPU stub, FakeTensor, and TPU7x tests for DSv4.1 attention custom ops."""

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
from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
    gather_sparse_fp8_kv,
    sparse_mla_swa_attention,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    attention_op,
    dsv41_gather_sparse_fp8_kv,
    dsv41_pack_fp8_kv_rows,
    dsv41_sparse_mla_swa_attention,
    dsv41_write_sub_rows_in_place,
)

ROW_WORDS = 128
ROPE_WORDS = 32
ROPE_TOKENS_PER_ROW = ROW_WORDS // ROPE_WORDS
NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM


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
        attention_op._gather_sparse_fp8_kv_op,  # noqa: SLF001 -- clear per-op cache
        attention_op._sparse_mla_swa_attention_op,  # noqa: SLF001 -- clear per-op cache
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


def _stub_gather_sparse_fp8_kv(
    kv_pool: jax.Array,
    rope_pool: jax.Array,
    topk_indices: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    num_tokens, topk = topk_indices.shape
    num_pages, _, _ = kv_pool.shape
    max_pages = block_tables.shape[1]
    if num_tokens == 0 or topk == 0:
        z = jnp.zeros((num_tokens, topk, ROW_WORDS), dtype=jnp.int32)
        return z, z[:, : topk // ROPE_TOKENS_PER_ROW, :]

    tok_iota = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
    in_seq = (tok_iota >= cu_q_lens[:-1][None, :]) & (tok_iota < cu_q_lens[1:][None, :])
    valid_tok = jnp.any(in_seq, axis=1)
    seq_ids = jnp.argmax(in_seq, axis=1).astype(jnp.int32)
    logical_pages = topk_indices // page_size
    page_offsets = topk_indices % page_size
    valid_logical = (
        valid_tok[:, None] & (topk_indices >= 0) & (logical_pages < max_pages)
    )
    safe_lp = jnp.clip(logical_pages, 0, max_pages - 1)
    phys_pages = block_tables[seq_ids[:, None], safe_lp]
    valid = valid_logical & (phys_pages >= 0) & (phys_pages < num_pages)
    safe_row = jnp.where(valid, phys_pages * page_size + page_offsets, 0)

    kv_flat = kv_pool.reshape(-1, ROW_WORDS)
    rope_flat = rope_pool.reshape(-1, ROPE_WORDS)
    out_nope = jnp.where(valid[:, :, None], kv_flat[safe_row], jnp.int32(0))
    out_rope = jnp.where(valid[:, :, None], rope_flat[safe_row], jnp.int32(0)).reshape(
        num_tokens, topk // ROPE_TOKENS_PER_ROW, ROW_WORDS
    )
    return out_nope, out_rope


def _dequantize_nope_jax(nope_words: jax.Array) -> jax.Array:
    words_u32 = nope_words.astype(jnp.uint32)
    top_u32 = (words_u32 >> jnp.uint32(24)) & jnp.uint32(0xFF)
    scale_u8 = top_u32[..., 64:].astype(jnp.uint8)
    scales = jnp.broadcast_to(
        jax.lax.bitcast_convert_type(scale_u8, jnp.float8_e8m0fnu).astype(jnp.float32),
        (*nope_words.shape[:-1], 64),
    )
    slabs = []
    for shift in (0, 8, 16):
        b = ((words_u32 >> jnp.uint32(shift)) & jnp.uint32(0xFF)).astype(jnp.uint8)
        slabs.append(
            jax.lax.bitcast_convert_type(b, jnp.float8_e4m3fn).astype(jnp.float32)
        )
    tail_b = top_u32[..., :64].astype(jnp.uint8)
    slabs.append(
        jax.lax.bitcast_convert_type(tail_b, jnp.float8_e4m3fn).astype(jnp.float32)
    )
    fp8_vals = jnp.concatenate(slabs, axis=-1)
    return fp8_vals * jnp.tile(scales, (1, 1, 7))


def _unpack_rope_jax(rope_words: jax.Array) -> jax.Array:
    quarters = rope_words.reshape(*rope_words.shape[:-2], -1, ROPE_WORDS).astype(
        jnp.uint32
    )
    lo = jax.lax.bitcast_convert_type(
        (quarters & jnp.uint32(0xFFFF)).astype(jnp.uint16), jnp.bfloat16
    ).astype(jnp.float32)
    hi = jax.lax.bitcast_convert_type(
        (quarters >> jnp.uint32(16)).astype(jnp.uint16), jnp.bfloat16
    ).astype(jnp.float32)
    return jnp.concatenate([lo, hi], axis=-1)


def _stub_sparse_mla_swa_attention(
    q: jax.Array,
    gathered_nope: jax.Array,
    gathered_rope: jax.Array,
    topk_indices: jax.Array,
    attention_sinks: jax.Array,
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    *,
    sm_scale: float,
    swa_window: int,
    swa_kv: jax.Array | None = None,
    kv_compress_ratio: int = 1,
    swa_min_pos: jax.Array | None = None,
) -> jax.Array:
    num_tokens, num_heads, _ = q.shape
    if num_tokens == 0 or num_heads == 0:
        return jnp.zeros((num_tokens, num_heads, HEAD_DIM), dtype=jnp.bfloat16)

    tok_iota = jnp.arange(num_tokens, dtype=jnp.int32)
    in_seq = (tok_iota[:, None] >= cu_q_lens[:-1][None, :]) & (
        tok_iota[:, None] < cu_q_lens[1:][None, :]
    )
    valid_tok = jnp.any(in_seq, axis=1)
    seq_ids = jnp.argmax(in_seq, axis=1).astype(jnp.int32)
    tok_pos = seq_lens[seq_ids] - (cu_q_lens[seq_ids + 1] - tok_iota)

    sparse_kv = jnp.concatenate(
        [_dequantize_nope_jax(gathered_nope), _unpack_rope_jax(gathered_rope)], axis=-1
    )
    max_comp = jnp.maximum(0, (tok_pos + 1) // kv_compress_ratio)
    sparse_valid = (
        valid_tok[:, None] & (topk_indices >= 0) & (topk_indices < max_comp[:, None])
    )

    if swa_kv is not None and swa_window > 0:
        num_seqs = seq_lens.shape[0]
        if swa_kv.ndim == 2:
            width = (swa_kv.shape[0] - num_tokens) // num_seqs
            prefix = swa_kv[: num_seqs * width].reshape(num_seqs, width, HEAD_DIM)
            curr = swa_kv[num_seqs * width :]
            deltas = tok_iota - cu_q_lens[seq_ids]
            col_iota = jnp.arange(width, dtype=jnp.int32)[None, :]
            use_prefix = col_iota < (width - 1 - deltas[:, None])
            p_idx = jnp.clip(col_iota + deltas[:, None] + 1, 0, width - 1)
            c_idx = jnp.clip(
                tok_iota[:, None] - (width - 1 - col_iota), 0, num_tokens - 1
            )
            swa_3d = jnp.where(
                use_prefix[:, :, None],
                prefix[seq_ids[:, None], p_idx],
                curr[c_idx],
            )
        else:
            width = swa_kv.shape[1]
            swa_3d = swa_kv
        swa_pos = tok_pos[:, None] - (
            width - 1 - jnp.arange(width, dtype=jnp.int32)[None, :]
        )
        min_pos = 0 if swa_min_pos is None else swa_min_pos[seq_ids][:, None]
        swa_valid = (
            valid_tok[:, None]
            & (swa_pos >= jnp.maximum(min_pos, tok_pos[:, None] - swa_window + 1))
            & (swa_pos <= tok_pos[:, None])
        )
        all_kv = jnp.concatenate([swa_3d.astype(jnp.float32), sparse_kv], axis=1)
        all_valid = jnp.concatenate([swa_valid, sparse_valid], axis=1)
    else:
        all_kv = sparse_kv
        all_valid = sparse_valid

    kv_masked = jnp.where(all_valid[:, :, None], all_kv, 0.0)
    logits = jnp.einsum("thd,tkd->thk", q.astype(jnp.float32), kv_masked) * sm_scale
    logits = jnp.where(all_valid[:, None, :], logits, -jnp.inf)
    sink = jnp.broadcast_to(attention_sinks[None, :, None], (num_tokens, num_heads, 1))
    all_logits = jnp.concatenate([logits, sink], axis=-1)
    row_max = jnp.max(all_logits, axis=-1, keepdims=True)
    probs = jnp.exp(logits - row_max) * all_valid[:, None, :]
    denom = jnp.sum(probs, axis=-1, keepdims=True) + jnp.exp(sink - row_max)
    out = jnp.einsum("thk,tkd->thd", probs, kv_masked) / denom
    return jnp.where(valid_tok[:, None, None], out, 0.0).astype(jnp.bfloat16)


def test_should_export_single_bank_attention_ops_without_inert_partition_specs() -> (
    None
):
    assert attention_op.__all__ == [
        "dsv41_gather_sparse_fp8_kv",
        "dsv41_sparse_mla_swa_attention",
    ]
    assert "dsv41_gather_sparse_fp8_kv" in dsv41_ops.__all__
    assert "dsv41_sparse_mla_swa_attention" in dsv41_ops.__all__
    assert dsv41_ops.dsv41_gather_sparse_fp8_kv is dsv41_gather_sparse_fp8_kv
    assert dsv41_ops.dsv41_sparse_mla_swa_attention is dsv41_sparse_mla_swa_attention

    gather_sig = inspect.signature(dsv41_gather_sparse_fp8_kv)
    assert list(gather_sig.parameters) == [
        "kv_pool",
        "rope_pool",
        "topk_indices",
        "block_tables",
        "cu_q_lens",
        "page_size",
    ]
    attn_sig = inspect.signature(dsv41_sparse_mla_swa_attention)
    assert list(attn_sig.parameters) == [
        "q",
        "gathered_nope",
        "gathered_rope",
        "topk_indices",
        "attention_sinks",
        "seq_lens",
        "cu_q_lens",
        "sm_scale",
        "swa_window",
        "swa_kv",
        "kv_compress_ratio",
        "swa_min_pos",
    ]
    assert attn_sig.parameters["swa_kv"].default is None
    assert attn_sig.parameters["kv_compress_ratio"].default == 1
    assert attn_sig.parameters["swa_min_pos"].default is None
    assert not any("PARTITION_SPECS" in name for name in dir(attention_op))


@pytest.mark.parametrize("swa_mode", ["none", "3d", "2d"])
@pytest.mark.parametrize("with_min_pos", [False, True])
@pytest.mark.parametrize("num_tokens", [0, 5])
@pytest.mark.parametrize("topk", [16, 2048])
def test_should_trace_fake_gather_and_sparse_mla_swa_with_fullgraph_compile(
    swa_mode: str,
    with_min_pos: bool,
    num_tokens: int,
    topk: int,
) -> None:
    torch.compiler.reset()
    counter = CompileCounterWithBackend("aot_eager")
    num_seqs, num_heads, swa_width = 2, 8, 8

    def _run(
        kv_pool: torch.Tensor,
        rope_pool: torch.Tensor,
        topk_idx: torch.Tensor,
        bt: torch.Tensor,
        cu: torch.Tensor,
        q: torch.Tensor,
        sinks: torch.Tensor,
        sl: torch.Tensor,
        swa_kv: torch.Tensor | None,
        min_pos: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nope, rope = dsv41_gather_sparse_fp8_kv(
            kv_pool, rope_pool, topk_idx, bt, cu, page_size=16
        )
        out = dsv41_sparse_mla_swa_attention(
            q,
            nope,
            rope,
            topk_idx,
            sinks,
            sl,
            cu,
            sm_scale=0.0625,
            swa_window=swa_width,
            swa_kv=swa_kv,
            kv_compress_ratio=2,
            swa_min_pos=min_pos,
        )
        return nope, rope, out

    compiled = torch.compile(_run, backend=counter, fullgraph=True)
    with FakeTensorMode() as mode:
        kv_pool = torch.empty((4, 16, ROW_WORDS), dtype=torch.int32)
        rope_pool = torch.empty((4, 4, ROW_WORDS), dtype=torch.int32)
        topk_idx = torch.empty((num_tokens, topk), dtype=torch.int32)
        bt = torch.empty((num_seqs, 3), dtype=torch.int32)
        cu = torch.empty((num_seqs + 1,), dtype=torch.int32)
        q = torch.empty((num_tokens, num_heads, HEAD_DIM), dtype=torch.bfloat16)
        sinks = torch.empty((num_heads,), dtype=torch.float32)
        sl = torch.empty((num_seqs,), dtype=torch.int32)
        if swa_mode == "3d":
            swa_kv = torch.empty(
                (num_tokens, swa_width, HEAD_DIM), dtype=torch.bfloat16
            )
        elif swa_mode == "2d":
            swa_kv = torch.empty(
                (num_seqs * swa_width + num_tokens, HEAD_DIM), dtype=torch.bfloat16
            )
        else:
            swa_kv = None
        min_pos = torch.empty((num_seqs,), dtype=torch.int32) if with_min_pos else None

        nope, rope, out = compiled(
            kv_pool, rope_pool, topk_idx, bt, cu, q, sinks, sl, swa_kv, min_pos
        )

        assert counter.frame_count == 1
        assert isinstance(nope, FakeTensor) and nope.fake_mode is mode
        assert isinstance(rope, FakeTensor) and rope.fake_mode is mode
        assert isinstance(out, FakeTensor) and out.fake_mode is mode
        assert nope.shape == (num_tokens, topk, ROW_WORDS)
        assert nope.dtype == torch.int32
        assert rope.shape == (num_tokens, topk // ROPE_TOKENS_PER_ROW, ROW_WORDS)
        assert rope.dtype == torch.int32
        assert out.shape == (num_tokens, num_heads, HEAD_DIM)
        assert out.dtype == torch.bfloat16


def test_should_bridge_single_bank_gather_and_attention_options_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_calls: list[int] = []
    attn_calls: list[dict[str, object]] = []

    def _recording_gather(
        kv_pool: jax.Array,
        rope_pool: jax.Array,
        topk_indices: jax.Array,
        block_tables: jax.Array,
        cu_q_lens: jax.Array,
        *,
        page_size: int,
    ) -> tuple[jax.Array, jax.Array]:
        gather_calls.append(page_size)
        return _stub_gather_sparse_fp8_kv(
            kv_pool,
            rope_pool,
            topk_indices,
            block_tables,
            cu_q_lens,
            page_size=page_size,
        )

    def _recording_attn(
        q: jax.Array,
        gathered_nope: jax.Array,
        gathered_rope: jax.Array,
        topk_indices: jax.Array,
        attention_sinks: jax.Array,
        seq_lens: jax.Array,
        cu_q_lens: jax.Array,
        *,
        sm_scale: float,
        swa_window: int,
        swa_kv: jax.Array | None = None,
        kv_compress_ratio: int = 1,
        swa_min_pos: jax.Array | None = None,
    ) -> jax.Array:
        attn_calls.append(
            {
                "sm_scale": sm_scale,
                "swa_window": swa_window,
                "swa_ndim": None if swa_kv is None else swa_kv.ndim,
                "kv_compress_ratio": kv_compress_ratio,
                "has_min_pos": swa_min_pos is not None,
            }
        )
        return _stub_sparse_mla_swa_attention(
            q,
            gathered_nope,
            gathered_rope,
            topk_indices,
            attention_sinks,
            seq_lens,
            cu_q_lens,
            sm_scale=sm_scale,
            swa_window=swa_window,
            swa_kv=swa_kv,
            kv_compress_ratio=kv_compress_ratio,
            swa_min_pos=swa_min_pos,
        )

    monkeypatch.setattr(attention_op, "gather_sparse_fp8_kv", _recording_gather)
    monkeypatch.setattr(attention_op, "sparse_mla_swa_attention", _recording_attn)

    rng = np.random.default_rng(606)
    num_pages, page_size, num_tokens, topk, num_heads, swa_width = 5, 16, 5, 8, 4, 4
    num_seqs = 2

    # Valid UE8M0 exponent 127 (scale=1.0) in lanes [64..127] top byte avoids NaN.
    kv_pool_np = np.zeros((num_pages, page_size, ROW_WORDS), dtype=np.int32)
    kv_pool_np[:, :, :64] = rng.integers(
        1, 1000, size=(num_pages, page_size, 64), dtype=np.int32
    )
    kv_pool_np[:, :, 64:] = np.int32(127 << 24) | rng.integers(
        1, 0x00010000, size=(num_pages, page_size, 64), dtype=np.int32
    )
    # Non-zero page 0 sentinel detects unmapped/OOB fallback reads.
    rope_pool_np = rng.integers(
        10,
        50_000,
        size=(num_pages, page_size // ROPE_TOKENS_PER_ROW, ROW_WORDS),
        dtype=np.int32,
    )
    bt_np = np.array([[1, -1, 3], [2, 4, 99]], dtype=np.int32)
    cu_np = np.array([0, 2, 4], dtype=np.int32)  # tok 4 is inactive padding
    sl_np = np.array([20, 30], dtype=np.int32)
    topk_np = np.array(
        [
            [0, 3, 17, -1, 32, 35, 99, 7],
            [1, 4, 18, 2, 33, 36, -1, 8],
            [0, 1, 16, 17, 32, 5, 6, 7],
            [2, 3, 18, 19, 33, -1, 10, 11],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ],
        dtype=np.int32,
    )

    kv_t = _JaxBackedTensor(jnp.asarray(kv_pool_np))
    rope_t = _JaxBackedTensor(jnp.asarray(rope_pool_np))
    topk_t = _JaxBackedTensor(jnp.asarray(topk_np))
    bt_t = _JaxBackedTensor(jnp.asarray(bt_np))
    cu_t = _JaxBackedTensor(jnp.asarray(cu_np))

    nope_out, rope_out = dsv41_gather_sparse_fp8_kv(
        kv_t, rope_t, topk_t, bt_t, cu_t, page_size=page_size
    )
    assert isinstance(nope_out, _JaxBackedTensor)
    assert isinstance(rope_out, _JaxBackedTensor)
    assert gather_calls == [page_size]

    # Invalid slots (unmapped page -1 -> topk 17/18, OOB page 99 -> topk 32/33,
    # negative topk -1, OOB topk 99, inactive tok 4) MUST be exact zero.
    nope_np = nope_out.to_numpy()
    rope_np = rope_out.to_numpy().reshape(num_tokens, topk, ROPE_WORDS)
    np.testing.assert_array_equal(nope_np[0, 2], 0)  # page -1
    np.testing.assert_array_equal(nope_np[0, 3], 0)  # topk -1
    np.testing.assert_array_equal(nope_np[0, 6], 0)  # topk 99 OOB
    np.testing.assert_array_equal(nope_np[2, 4], 0)  # page 99 OOB
    np.testing.assert_array_equal(nope_np[4], 0)  # inactive tok 4
    np.testing.assert_array_equal(rope_np[4], 0)
    np.testing.assert_array_equal(nope_np[0, 0], kv_pool_np[1, 0])
    np.testing.assert_array_equal(nope_np[0, 4], kv_pool_np[3, 0])
    np.testing.assert_array_equal(
        rope_np[0, 0], rope_pool_np[1].reshape(page_size, ROPE_WORDS)[0]
    )

    q_bf16 = jnp.asarray(
        rng.normal(scale=0.2, size=(num_tokens, num_heads, HEAD_DIM)),
        dtype=jnp.bfloat16,
    )
    sinks_np = rng.normal(scale=0.5, size=(num_heads,)).astype(np.float32)
    swa_3d_bf16 = jnp.asarray(
        rng.normal(scale=0.3, size=(num_tokens, swa_width, HEAD_DIM)),
        dtype=jnp.bfloat16,
    )
    swa_2d_bf16 = jnp.asarray(
        rng.normal(scale=0.3, size=(num_seqs * swa_width + num_tokens, HEAD_DIM)),
        dtype=jnp.bfloat16,
    )
    min_pos_np = np.array([18, 28], dtype=np.int32)

    q_t = _JaxBackedTensor(q_bf16)
    sinks_t = _JaxBackedTensor(jnp.asarray(sinks_np))
    sl_t = _JaxBackedTensor(jnp.asarray(sl_np))
    swa_3d_t = _JaxBackedTensor(swa_3d_bf16)
    swa_2d_t = _JaxBackedTensor(swa_2d_bf16)
    min_pos_t = _JaxBackedTensor(jnp.asarray(min_pos_np))

    out_sparse = dsv41_sparse_mla_swa_attention(
        q_t,
        nope_out,
        rope_out,
        topk_t,
        sinks_t,
        sl_t,
        cu_t,
        sm_scale=0.125,
        swa_window=0,
    )
    out_3d = dsv41_sparse_mla_swa_attention(
        q_t,
        nope_out,
        rope_out,
        topk_t,
        sinks_t,
        sl_t,
        cu_t,
        sm_scale=0.125,
        swa_window=swa_width,
        swa_kv=swa_3d_t,
        kv_compress_ratio=2,
    )
    out_3d_min = dsv41_sparse_mla_swa_attention(
        q_t,
        nope_out,
        rope_out,
        topk_t,
        sinks_t,
        sl_t,
        cu_t,
        sm_scale=0.125,
        swa_window=swa_width,
        swa_kv=swa_3d_t,
        kv_compress_ratio=2,
        swa_min_pos=min_pos_t,
    )
    out_2d = dsv41_sparse_mla_swa_attention(
        q_t,
        nope_out,
        rope_out,
        topk_t,
        sinks_t,
        sl_t,
        cu_t,
        sm_scale=0.125,
        swa_window=swa_width,
        swa_kv=swa_2d_t,
        kv_compress_ratio=2,
        swa_min_pos=min_pos_t,
    )

    assert attn_calls == [
        {
            "sm_scale": 0.125,
            "swa_window": 0,
            "swa_ndim": None,
            "kv_compress_ratio": 1,
            "has_min_pos": False,
        },
        {
            "sm_scale": 0.125,
            "swa_window": swa_width,
            "swa_ndim": 3,
            "kv_compress_ratio": 2,
            "has_min_pos": False,
        },
        {
            "sm_scale": 0.125,
            "swa_window": swa_width,
            "swa_ndim": 3,
            "kv_compress_ratio": 2,
            "has_min_pos": True,
        },
        {
            "sm_scale": 0.125,
            "swa_window": swa_width,
            "swa_ndim": 2,
            "kv_compress_ratio": 2,
            "has_min_pos": True,
        },
    ]
    for tensor in (out_sparse, out_3d, out_3d_min, out_2d):
        assert isinstance(tensor, _JaxBackedTensor)
        assert tensor.shape == (num_tokens, num_heads, HEAD_DIM)
        assert tensor.dtype == torch.bfloat16
        np.testing.assert_array_equal(tensor.to_numpy()[4], 0.0)
    assert not np.allclose(out_3d.to_numpy()[:4], out_3d_min.to_numpy()[:4], atol=1e-4)
    assert not np.allclose(out_sparse.to_numpy()[:4], out_3d.to_numpy()[:4], atol=1e-4)


def test_should_fail_fast_when_attention_op_inputs_or_device_are_invalid() -> None:
    kv_pool = torch.zeros((2, 16, ROW_WORDS), dtype=torch.int32)
    rope_pool = torch.zeros((2, 4, ROW_WORDS), dtype=torch.int32)
    topk_idx = torch.zeros((4, 8), dtype=torch.int32)
    bt = torch.zeros((1, 2), dtype=torch.int32)
    cu = torch.tensor([0, 4], dtype=torch.int32)

    with pytest.raises(ValueError, match="page_size"):
        dsv41_gather_sparse_fp8_kv(kv_pool, rope_pool, topk_idx, bt, cu, page_size=6)
    with pytest.raises(ValueError, match="topk_indices"):
        dsv41_gather_sparse_fp8_kv(
            kv_pool,
            rope_pool,
            torch.zeros((4, 6), dtype=torch.int32),
            bt,
            cu,
            page_size=16,
        )

    q = torch.zeros((4, 8, HEAD_DIM), dtype=torch.bfloat16)
    nope = torch.zeros((4, 8, ROW_WORDS), dtype=torch.int32)
    rope = torch.zeros((4, 2, ROW_WORDS), dtype=torch.int32)
    sinks = torch.zeros((8,), dtype=torch.float32)
    sl = torch.tensor([4], dtype=torch.int32)

    with pytest.raises(ValueError, match="q"):
        dsv41_sparse_mla_swa_attention(
            torch.zeros((4, 8, 64), dtype=torch.bfloat16),
            nope,
            rope,
            topk_idx,
            sinks,
            sl,
            cu,
            sm_scale=0.1,
            swa_window=8,
        )
    with pytest.raises(ValueError, match="swa_window"):
        dsv41_sparse_mla_swa_attention(
            q, nope, rope, topk_idx, sinks, sl, cu, sm_scale=0.1, swa_window=-1
        )
    with pytest.raises(ValueError, match="ratio"):
        dsv41_sparse_mla_swa_attention(
            q,
            nope,
            rope,
            topk_idx,
            sinks,
            sl,
            cu,
            sm_scale=0.1,
            swa_window=8,
            kv_compress_ratio=3,
        )

    v6e_mesh = AbstractMesh((), (), abstract_device=AbstractDevice("TPU v6e", 1, "tpu"))
    with use_abstract_mesh(v6e_mesh), pytest.raises(RuntimeError, match="TPU7x"):
        dsv41_gather_sparse_fp8_kv(kv_pool, rope_pool, topk_idx, bt, cu, page_size=16)
    with use_abstract_mesh(v6e_mesh), pytest.raises(RuntimeError, match="TPU7x"):
        dsv41_sparse_mla_swa_attention(
            q, nope, rope, topk_idx, sinks, sl, cu, sm_scale=0.1, swa_window=8
        )


@pytest.mark.tpu_test
def test_tpu_should_run_single_bank_gather_and_mla_swa_under_tpu_compile() -> None:
    device = torch.device("tpu")
    rng = np.random.default_rng(909)
    num_pages, page_size, num_tokens, topk, num_heads, swa_width = 6, 16, 6, 16, 8, 8
    num_seqs = 2

    latents = torch.from_numpy(
        rng.normal(scale=0.5, size=(num_pages * page_size, HEAD_DIM)).astype(np.float32)
    ).to(dtype=torch.bfloat16, device=device)
    packed_nope, packed_rope = dsv41_pack_fp8_kv_rows(latents)

    kv_pool = torch.zeros(
        (num_pages, page_size, ROW_WORDS), dtype=torch.int32, device=device
    )
    rope_pool = torch.zeros(
        (num_pages, page_size // ROPE_TOKENS_PER_ROW, ROW_WORDS),
        dtype=torch.int32,
        device=device,
    )
    slots = torch.arange(num_pages * page_size, dtype=torch.int32, device=device)
    dsv41_write_sub_rows_in_place(kv_pool, packed_nope, slots, page_tokens=page_size)
    dsv41_write_sub_rows_in_place(rope_pool, packed_rope, slots, page_tokens=page_size)

    topk_cpu = torch.from_numpy(
        rng.integers(-2, 44, size=(num_tokens, topk), dtype=np.int32)
    )
    bt_cpu = torch.tensor([[1, 3, -1], [2, 4, 5]], dtype=torch.int32)
    cu_cpu = torch.tensor([0, 3, 5], dtype=torch.int32)
    sl_cpu = torch.tensor([60, 88], dtype=torch.int32)
    q_cpu = torch.from_numpy(
        rng.normal(scale=0.4, size=(num_tokens, num_heads, HEAD_DIM)).astype(np.float32)
    ).to(torch.bfloat16)
    sinks_cpu = torch.from_numpy(
        rng.normal(scale=0.5, size=(num_heads,)).astype(np.float32)
    )
    swa_cpu = torch.from_numpy(
        rng.normal(scale=0.4, size=(num_tokens, swa_width, HEAD_DIM)).astype(np.float32)
    ).to(torch.bfloat16)
    swa_2d_cpu = torch.from_numpy(
        rng.normal(
            scale=0.4, size=(num_seqs * swa_width + num_tokens, HEAD_DIM)
        ).astype(np.float32)
    ).to(torch.bfloat16)
    min_pos_cpu = torch.tensor([58, 86], dtype=torch.int32)

    @torch.compile(backend="tpu", fullgraph=True)
    def _compiled_attn(
        kv_p: torch.Tensor,
        rope_p: torch.Tensor,
        topk_i: torch.Tensor,
        bt: torch.Tensor,
        cu: torch.Tensor,
        sl: torch.Tensor,
        q: torch.Tensor,
        sinks: torch.Tensor,
        swa_3d: torch.Tensor,
        swa_2d: torch.Tensor,
        min_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nope, rope = dsv41_gather_sparse_fp8_kv(
            kv_p, rope_p, topk_i, bt, cu, page_size=page_size
        )
        out = dsv41_sparse_mla_swa_attention(
            q,
            nope,
            rope,
            topk_i,
            sinks,
            sl,
            cu,
            sm_scale=0.0625,
            swa_window=swa_width,
            swa_kv=swa_3d,
            kv_compress_ratio=2,
        )
        out_min = dsv41_sparse_mla_swa_attention(
            q,
            nope,
            rope,
            topk_i,
            sinks,
            sl,
            cu,
            sm_scale=0.0625,
            swa_window=swa_width,
            swa_kv=swa_3d,
            kv_compress_ratio=2,
            swa_min_pos=min_pos,
        )
        out_2d = dsv41_sparse_mla_swa_attention(
            q,
            nope,
            rope,
            topk_i,
            sinks,
            sl,
            cu,
            sm_scale=0.0625,
            swa_window=swa_width,
            swa_kv=swa_2d,
            kv_compress_ratio=2,
            swa_min_pos=min_pos,
        )
        return nope, rope, out, out_min, out_2d

    topk_tpu = topk_cpu.to(device)
    bt_tpu = bt_cpu.to(device)
    cu_tpu = cu_cpu.to(device)
    sl_tpu = sl_cpu.to(device)
    q_tpu = q_cpu.to(device)
    sinks_tpu = sinks_cpu.to(device)
    swa_3d_tpu = swa_cpu.to(device)
    swa_2d_tpu = swa_2d_cpu.to(device)
    min_pos_tpu = min_pos_cpu.to(device)

    nope, rope, out, out_min, out_2d = _compiled_attn(
        kv_pool,
        rope_pool,
        topk_tpu,
        bt_tpu,
        cu_tpu,
        sl_tpu,
        q_tpu,
        sinks_tpu,
        swa_3d_tpu,
        swa_2d_tpu,
        min_pos_tpu,
    )
    torch.accelerator.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        nope, rope, out, out_min, out_2d = _compiled_attn(
            kv_pool,
            rope_pool,
            topk_tpu,
            bt_tpu,
            cu_tpu,
            sl_tpu,
            q_tpu,
            sinks_tpu,
            swa_3d_tpu,
            swa_2d_tpu,
            min_pos_tpu,
        )
        torch.accelerator.synchronize()
    assert not any(
        "tt_jit_execute_as_strided" in evt.key for evt in prof.key_averages()
    )

    ref_gn, ref_gr = gather_sparse_fp8_kv(
        jnp.asarray(kv_pool.cpu().numpy()),
        jnp.asarray(rope_pool.cpu().numpy()),
        jnp.asarray(topk_cpu.numpy()),
        jnp.asarray(bt_cpu.numpy()),
        jnp.asarray(cu_cpu.numpy()),
        page_size=page_size,
        interpret=True,
    )
    q_j = jnp.asarray(q_cpu.float().numpy(), dtype=jnp.bfloat16)
    tk_j = jnp.asarray(topk_cpu.numpy())
    sinks_j = jnp.asarray(sinks_cpu.numpy())
    sl_j = jnp.asarray(sl_cpu.numpy())
    cu_j = jnp.asarray(cu_cpu.numpy())
    swa_j = jnp.asarray(swa_cpu.float().numpy(), dtype=jnp.bfloat16)
    swa_2d_j = jnp.asarray(swa_2d_cpu.float().numpy(), dtype=jnp.bfloat16)
    min_pos_j = jnp.asarray(min_pos_cpu.numpy())

    ref_out = sparse_mla_swa_attention(
        q_j,
        ref_gn,
        ref_gr,
        tk_j,
        sinks_j,
        sl_j,
        cu_j,
        sm_scale=0.0625,
        swa_window=swa_width,
        swa_kv=swa_j,
        kv_compress_ratio=2,
        interpret=True,
    )
    ref_min = sparse_mla_swa_attention(
        q_j,
        ref_gn,
        ref_gr,
        tk_j,
        sinks_j,
        sl_j,
        cu_j,
        sm_scale=0.0625,
        swa_window=swa_width,
        swa_kv=swa_j,
        kv_compress_ratio=2,
        swa_min_pos=min_pos_j,
        interpret=True,
    )
    ref_2d = sparse_mla_swa_attention(
        q_j,
        ref_gn,
        ref_gr,
        tk_j,
        sinks_j,
        sl_j,
        cu_j,
        sm_scale=0.0625,
        swa_window=swa_width,
        swa_kv=swa_2d_j,
        kv_compress_ratio=2,
        swa_min_pos=min_pos_j,
        interpret=True,
    )

    np.testing.assert_array_equal(nope.cpu().numpy(), np.asarray(ref_gn))
    np.testing.assert_array_equal(rope.cpu().numpy(), np.asarray(ref_gr))
    np.testing.assert_allclose(
        out.cpu().float().numpy(),
        np.asarray(ref_out, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        out_min.cpu().float().numpy(),
        np.asarray(ref_min, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        out_2d.cpu().float().numpy(),
        np.asarray(ref_2d, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    assert not np.allclose(
        out_min.cpu().float().numpy()[:5],
        out.cpu().float().numpy()[:5],
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_array_equal(out.cpu().float().numpy()[5:], 0.0)
    np.testing.assert_array_equal(out_2d.cpu().float().numpy()[5:], 0.0)

    scale_bt = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    scale_sl = torch.tensor([32], dtype=torch.int32, device=device)
    scale_sinks = torch.zeros((8,), dtype=torch.float32, device=device)
    for bucket_tokens in (8, 48, 288):
        scale_q = torch.zeros(
            (bucket_tokens, 8, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        scale_cu = torch.tensor([0, bucket_tokens], dtype=torch.int32, device=device)
        for topk_k in (512, 2048):
            scale_topk = torch.full(
                (bucket_tokens, topk_k), -1, dtype=torch.int32, device=device
            )
            gn, gr = dsv41_gather_sparse_fp8_kv(
                kv_pool, rope_pool, scale_topk, scale_bt, scale_cu, page_size=page_size
            )
            scaled_attn = dsv41_sparse_mla_swa_attention(
                scale_q,
                gn,
                gr,
                scale_topk,
                scale_sinks,
                scale_sl,
                scale_cu,
                sm_scale=0.0625,
                swa_window=128,
            )
            assert scaled_attn.shape == (bucket_tokens, 8, HEAD_DIM)
            assert scaled_attn.dtype == torch.bfloat16
