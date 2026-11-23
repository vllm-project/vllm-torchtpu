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
"""TurboQuant caches in the MTP KV-only writer (`MlaKvWriteOp`,
`mla_kv_write_caches`): every row it writes must be the bytes the full pass
(`sparse_mla_attention` with `tq_tables`) writes.

CPU wiring tests: the SparseCore latent writers are replaced by their
pure-XLA goldens and the attention kernel by a stub, so only the cache writes
run, at block sizes 32 and 256. They show the KV-only writer stores what the
full pass's write path stores; whether those bytes decode correctly is the
TurboQuant pack's own tests' concern, and the SparseCore writers need a TPU.
`k_pe` is stored as opaque fp8 bytes, so its column order (NeoX permute or
not) is the projection's, not the writer's.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.layers.core.attention_interface as attention_interface
from tests.layers.adapter import test_mla_kv_write_op as kvw
from tests.layers.adapter.mla_test_utils import (
    HIDDEN,
    IDX_HEAD_DIM,
    KV_LORA,
    ROPE,
    TPU_PAGE,
    TPU_PREFIX,
    build_tpu_sparse_mla,
    on_tpu,
    tpu_compiled,
    tpu_md,
)
from tests.layers.adapter.test_mla_kv_write_op import (
    _contexts,
    _inputs,
    _md,
    _op_wrapper,
    _record_jax_ops,
    _xla_latent_writers,
)
from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.layers.core import turboquant as tq

PAGES = 16
# The served block size (256) and a small one.
PAGE_SIZES = [32, 256]
HEADS = 8
TOPK = 4


def _tq_specs(page):
    from vllm_torchtpu.kernels.mla.kv_cache_utils import (
        KVCacheLayout,
        KVCacheType,
        SparseMLAKVCacheSpec,
    )

    layout = KVCacheLayout.SPARSECORE
    return (
        SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE, layout, PAGES, page, tq.MLA_TQ_NOPE_BYTES
        ),
        SparseMLAKVCacheSpec.create(KVCacheType.ROPE, layout, PAGES, page, ROPE + 2),
    )


def _step(page=32, seed=5):
    """One step's unquantized latents and fp8 RoPE keys for two requests at
    positions 33..35 and 6..9 (7 real rows + 1 padded)."""
    nope_spec, rope_spec = _tq_specs(page)
    rng = np.random.default_rng(seed)
    num_tokens = 8
    return dict(
        specs=(nope_spec, rope_spec),
        nope=jnp.zeros(nope_spec.shape, nope_spec.jax_dtype),
        rope=jnp.zeros(rope_spec.shape, rope_spec.jax_dtype),
        kv_c=jnp.asarray(rng.standard_normal((num_tokens, KV_LORA)), jnp.bfloat16),
        k_pe=jnp.asarray(rng.standard_normal((num_tokens, ROPE)), jnp.float8_e4m3fn),
        seq_lens=jnp.array([36, 10], jnp.int32),
        block_tables=jnp.array([5, 2, 7, 7, 9, 7, 7, 7], jnp.int32),
        qsl=jnp.array([0, 3, 7], jnp.int32),
    )


def _mesh():
    return jax.sharding.Mesh(np.asarray(jax.local_devices()[:1]), ("x",))


def _full_pass_caches(monkeypatch, s, tables):
    """The caches `sparse_mla_attention` returns, with the kernel stubbed."""

    def kernel(q, *args, **kwargs):
        del args, kwargs
        return jnp.zeros(q.shape, jnp.bfloat16)

    monkeypatch.setattr(
        attention_interface, "sparse_mla_ragged_paged_attention", kernel
    )
    n = s["kv_c"].shape[0]
    nope_spec, rope_spec = s["specs"]
    nope, rope, _ = attention_interface.sparse_mla_attention(
        jnp.zeros((n, HEADS, KV_LORA), jnp.bfloat16),
        jnp.zeros((n, HEADS, ROPE), jnp.bfloat16),
        s["kv_c"],
        s["k_pe"],
        s["nope"],
        s["rope"],
        jnp.zeros((n, TOPK), jnp.int32),
        s["seq_lens"],
        s["block_tables"],
        s["qsl"],
        jnp.array([0, 2, 2], jnp.int32),
        _mesh(),
        nope_spec,
        rope_spec,
        sm_scale=1.0,
        k_scale=1.0,
        tq_tables=tables,
        tq_prerotated=True,
    )
    return nope, rope


def _kv_only_caches(s, tables, dcp_size=1):
    idx_cache, k_packed, idx_seq_lens, idx_pages, idx_cu = _indexer_inputs()
    nope, rope, _ = mla_op.mla_kv_write_caches(
        s["nope"],
        s["rope"],
        idx_cache,
        s["kv_c"],
        s["k_pe"],
        k_packed,
        s["seq_lens"],
        s["block_tables"],
        s["qsl"],
        idx_seq_lens,
        idx_pages,
        idx_cu,
        mla_kv_spec=s["specs"],
        mesh=_mesh(),
        kv_cache_mode="xla",
        dcp_size=dcp_size,
        tq_tables=tables,
    )
    return nope, rope


def _indexer_inputs():
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        KVLayout,
        get_kv_cache_shape,
    )

    width = IDX_HEAD_DIM + 1
    rng = np.random.default_rng(3)
    shape = get_kv_cache_shape(PAGES, 16, width, KVLayout.HEAD_ALONG_SUBLANE)
    return (
        jnp.zeros(shape, jnp.uint8),
        jnp.asarray(rng.integers(1, 255, (8, width), dtype=np.uint8)),
        jnp.array([36, 10], jnp.int32),
        jnp.array([1, 3, 4, 0, 6, 0, 0, 0], jnp.int32),
        jnp.array([0, 3, 7], jnp.int32),
    )


def _same_bytes(got, want):
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint8), np.asarray(want).view(np.uint8)
    )


@pytest.mark.parametrize("page", PAGE_SIZES)
def test_kv_only_write_matches_the_full_pass(monkeypatch, page):
    """Same latents in, same NoPE codes and RoPE rows (fp8 `k_pe` + norm)
    out, as the full pass writes them."""
    _xla_latent_writers(monkeypatch)
    s = _step(page)
    tables = tq.make_mla_latent_tables()
    want_nope, want_rope = _full_pass_caches(monkeypatch, s, tables)
    got_nope, got_rope = _kv_only_caches(s, tables)
    _same_bytes(got_nope, want_nope)
    _same_bytes(got_rope, want_rope)
    # Something was written.
    assert np.asarray(want_nope).any() and np.asarray(want_rope).any()


@pytest.mark.parametrize("page", PAGE_SIZES)
def test_kv_only_write_comparison_sees_one_changed_row(monkeypatch, page):
    """Positive control: one negated latent row and one changed RoPE key
    change the bytes the KV-only writer stores."""
    _xla_latent_writers(monkeypatch)
    s = _step(page)
    tables = tq.make_mla_latent_tables()
    want_nope, want_rope = _full_pass_caches(monkeypatch, s, tables)
    s["kv_c"] = s["kv_c"].at[0].multiply(-1)
    s["k_pe"] = s["k_pe"].at[1].set(jnp.float8_e4m3fn(3.0))
    got_nope, got_rope = _kv_only_caches(s, tables)
    assert not np.array_equal(np.asarray(got_nope), np.asarray(want_nope))
    assert not np.array_equal(
        np.asarray(got_rope).view(np.uint8), np.asarray(want_rope).view(np.uint8)
    )


def test_write_caches_refuses_turboquant_under_dcp():
    with pytest.raises(NotImplementedError, match="TurboQuant"):
        _kv_only_caches(_step(), tq.make_mla_latent_tables(), dcp_size=2)


def _tq_wrapper(dcp_size=1):
    w = _op_wrapper(dcp_size=dcp_size)
    w.mla_attn.kv_cache_dtype = "turboquant_4bit_nc"
    return w


def test_write_op_hands_turboquant_tables_and_fp8_rope(monkeypatch):
    """On a TurboQuant layer the op passes the latent tables to the writer,
    the latent unquantized and `k_pe` in fp8 (the full pass's
    `_quantized_kv_latents` branch)."""
    from vllm_torchtpu.layers.core.quantization import quantize_kv

    built = _record_jax_ops(monkeypatch)
    op = mla_op.MlaKvWriteOp(_tq_wrapper(), mesh=None)
    assert op.tq_tables is tq.make_mla_latent_tables()
    (jax_op,) = built

    seen = {}

    def writer(*args, **kwargs):
        seen.update(kwargs)
        return args[0], args[1], args[2]

    monkeypatch.setattr(mla_op, "mla_kv_write_caches", writer)
    jax_op.fn(*range(12))
    assert seen["tq_tables"] is op.tq_tables

    nope, rope = torch.zeros(3), torch.zeros(3)
    hidden, positions = _inputs(16)
    with torch.no_grad(), _contexts((nope, rope), _md(10), _md(20)):
        op(positions, hidden)
        kv_c, k_pe = op.wrapper.kv_latents(positions, hidden)
    k_scale = op.mla_layer._k_scale_float
    k_pe, _ = quantize_kv(torch.float8_e4m3fn, k_pe, value=None, k_scale=k_scale)
    (args,) = jax_op.calls
    assert args[3].dtype == kv_c.dtype != torch.float8_e4m3fn
    assert torch.equal(args[3], kv_c.view(-1, KV_LORA))
    assert args[4].dtype == torch.float8_e4m3fn
    assert torch.equal(args[4].view(torch.uint8), k_pe.view(-1, ROPE).view(torch.uint8))


def test_write_op_without_turboquant_has_no_tables(monkeypatch):
    _record_jax_ops(monkeypatch)
    assert mla_op.MlaKvWriteOp(_op_wrapper(), mesh=None).tq_tables is None


def test_write_op_refuses_turboquant_under_dcp(monkeypatch):
    _record_jax_ops(monkeypatch)
    with pytest.raises(NotImplementedError, match="TurboQuant"):
        mla_op.MlaKvWriteOp(_tq_wrapper(dcp_size=2), mesh=None)


# --- On TPU: the real SparseCore writers and the compiled ops -------------

# Names no other build in a test process uses (ops register under them).
TQ_MLA = "model.layers.7.self_attn.attn"
TQ_IDX = "model.layers.7.self_attn.indexer.k_cache"


def _tq_md(seq_lens, query_start_loc, request_distribution):
    return tpu_md((TQ_MLA, TQ_IDX), seq_lens, query_start_loc, request_distribution)


@pytest.fixture(scope="module")
def tq_tpu_layer():
    """`test_mla_kv_write_op.tpu_layer` with TurboQuant caches: the full pass
    writes request 0's `TPU_PREFIX` tokens, then the step is a 64-token
    chunk (request 0's next 32 tokens, request 1's first 24, 8 padded)."""
    from vllm_torchtpu.layers.adapter import turboquant_attention
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    previous = turboquant_attention._active_preset
    turboquant_attention.set_active_turboquant_preset("turboquant_4bit_nc")
    torch._dynamo.reset()
    layer = build_tpu_sparse_mla(TQ_MLA, TQ_IDX, cache_dtype="turboquant_4bit_nc")
    # Both passes are in TurboQuant mode: the full pass's op was built for a
    # TurboQuant layer over TurboQuant-shaped caches, and the write op packs.
    mla = layer.mla_layer
    assert PallasMLAttentionBackendImpl._is_turboquant(mla)
    nope_spec, rope_spec = mla.mla_kv_spec
    assert tuple(nope_spec.shape[1:]) == (TPU_PAGE * tq.MLA_TQ_NOPE_BYTES // 512, 128)
    assert tuple(mla.kv_cache[0].shape) == tuple(nope_spec.shape)
    with set_vllm_model_wrapper_context(mesh=layer.mesh):
        write_op = mla_op.MlaKvWriteOp(layer.wrapper, layer.mesh)
    assert write_op.tq_tables is not None
    layer.full_c = tpu_compiled(kvw._FullPass(layer.wrapper))
    layer.kv_only_c = tpu_compiled(kvw._KvOnly(layer.wrapper, write_op))
    gen = torch.Generator().manual_seed(9)
    dev = torch.device("tpu")

    def tokens(positions):
        n = len(positions)
        return (
            torch.tensor(positions, dtype=torch.int32, device=dev),
            torch.randn(n, HIDDEN, generator=gen).bfloat16().to(dev),
        )

    end = TPU_PREFIX + 32
    prefix = tokens(list(range(TPU_PREFIX)))
    prefix_md = _tq_md([TPU_PREFIX, 0], [0, TPU_PREFIX, TPU_PREFIX], [0, 0, 1])
    on_tpu(layer, layer.full_c, prefix_md, *prefix)
    layer.prefix_caches = [c.cpu() for c in layer.caches]
    layer.step = tokens(list(range(TPU_PREFIX, end)) + list(range(24)) + [0] * 8)
    layer.step_md = lambda: _tq_md([end, 24], [0, 32, 56], [0, 0, 2])
    layer.last_rows = torch.tensor([31, 55] + [0] * 6, dtype=torch.int64, device=dev)
    layer.rows_md = lambda: _tq_md([end, 24], [0, 1, 2], [2, 2, 2])
    yield layer
    turboquant_attention._active_preset = previous


@pytest.mark.tpu_test
def test_write_op_writes_the_full_pass_turboquant_cache_entries_on_tpu(
    tq_tpu_layer,
):
    """Compiled for TPU, with TurboQuant caches and the SparseCore writers:
    the KV-only write leaves the same MLA and indexer-K bytes as the full
    forward over the same rows."""
    _, want, _ = kvw._single_pass(tq_tpu_layer)
    assert any(
        not torch.equal(w, p) for w, p in zip(want, tq_tpu_layer.prefix_caches)
    ), "the step wrote nothing"
    kvw._reset(tq_tpu_layer)
    on_tpu(
        tq_tpu_layer, tq_tpu_layer.kv_only_c, tq_tpu_layer.step_md(), *tq_tpu_layer.step
    )
    _assert_same_cache_bytes([c.cpu() for c in tq_tpu_layer.caches], want)


@pytest.mark.tpu_test
def test_turboquant_rows_read_back_after_the_write_op_match_the_full_pass_on_tpu(
    tq_tpu_layer,
):
    """The KV-only write over every row, then the full forward on one row
    per request as decode rows: the caches equal one full pass's, byte for
    byte (the gate), and each decode row's output points the same way as the
    full pass's row (cosine >= 0.999; decode and prefill rows compute
    differently, so not bitwise). The decode rows read the TurboQuant rows
    the KV-only write stored."""
    want_out, want_caches, _ = kvw._single_pass(tq_tpu_layer)
    out, caches, _ = kvw._two_pass(tq_tpu_layer)
    _assert_same_cache_bytes(caches, want_caches)
    got, ref = out[:2].float(), want_out[[31, 55]].float()
    cos = torch.nn.functional.cosine_similarity(
        got.reshape(2, -1), ref.reshape(2, -1), dim=-1
    )
    assert bool((cos >= 0.999).all()), f"decode vs prefill row cosine {cos.tolist()}"


def _assert_same_cache_bytes(got, want):
    """Byte equality per cache, naming where the bytes differ (cache, count,
    the first flat byte offsets and their (page, row) in a `[pages, rows,
    ...]` cache)."""
    problems = []
    for i, (g, w) in enumerate(zip(got, want)):
        gb, wb = g.contiguous().view(torch.uint8), w.contiguous().view(torch.uint8)
        assert gb.shape == wb.shape, (i, gb.shape, wb.shape)
        diff = (gb != wb).reshape(-1).nonzero().reshape(-1)
        if diff.numel():
            page_bytes = gb[0].numel()
            row_bytes = gb[0, 0].numel() if gb.dim() > 2 else page_bytes
            first = diff[:8].tolist()
            where = [(o // page_bytes, (o % page_bytes) // row_bytes) for o in first]
            problems.append(
                f"cache {i}: {diff.numel()} bytes differ; first {first} at {where}"
            )
    assert not problems, "; ".join(problems)
