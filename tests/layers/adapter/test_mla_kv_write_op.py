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
"""`MlaKvWriteOp` and its parts: every row it writes must be bit-identical to
what the full forward writes.

* `VllmTPUMultiHeadLatentAttentionWrapper.kv_latents` vs `forward`'s
  `(kv_c_normed, k_pe)` (bf16 and fp8 projections, both rope classes);
* `VllmTPUIndexer.k_packed` vs `VllmTPUIndexer.forward` + the packing in
  `VllmTPUSparseAttnIndexer.forward_oot`;
* the non-DCP XLA indexer-K append, read back slot by slot;
* `mla_kv_write_caches`, unsharded and on two DCP ranks, against the writers'
  CPU goldens;
* `MlaKvWriteOp`: its operands, its registration and its refusals;
* on TPU: the compiled op against the full pass (single chip), the two-pass
  step that reads those rows back, and under DCP on several chips.

The CPU tests use stand-in projections in the processed runtime layouts.
"""

import contextlib
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.layers.core.attention_interface as attention_interface
from tests.layers.adapter.mla_test_utils import (
    HIDDEN,
    IDX_HEAD_DIM,
    IDX_N_HEAD,
    KV_LORA,
    MAX_POS,
    Q_LORA,
    QK,
    ROPE,
    TPU_HEADS,
    TPU_PAGE,
    TPU_PAGES,
    TPU_PREFIX,
    TPU_TOPK,
    assert_same_bytes,
    build_tpu_sparse_mla,
    dsa_indexer,
    mla_impl,
    mla_rope,
    mla_wrapper,
    on_tpu,
    reset_caches,
    tpu_compiled,
    tpu_md,
)
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata


def _wrapper_latents(w, hidden, positions):
    """What `forward` hands the MLA attention as (kv_c_normed, k_pe)."""
    mla_op.VllmTPUMultiHeadLatentAttentionWrapper.forward(
        w, positions=positions, hidden_states=hidden
    )
    _, kv_c, k_pe = w.mla_attn.calls[-1]
    return kv_c, k_pe


def _inputs(num_tokens, seed=1):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(num_tokens, HIDDEN, generator=gen).bfloat16()
    positions = torch.randperm(MAX_POS, generator=gen)[:num_tokens]
    return hidden, positions


@pytest.mark.parametrize("layout", ["flipped", "fp8"])
@pytest.mark.parametrize("scaling", [False, True], ids=["rope", "deepseek_scaling"])
@pytest.mark.parametrize("num_tokens", [16, 96])
def test_kv_latents_match_the_wrapper(layout, scaling, num_tokens):
    w = mla_wrapper(layout=layout, rope=mla_rope(scaling=scaling))
    hidden, positions = _inputs(num_tokens)
    with torch.no_grad():
        want_kv, want_pe = _wrapper_latents(w, hidden, positions)
        got_kv, got_pe = w.kv_latents(positions, hidden)
    assert torch.equal(got_kv, want_kv)
    assert got_pe.shape == want_pe.shape
    assert torch.equal(got_pe, want_pe)


@pytest.mark.parametrize(
    "kwargs", [dict(q_lora=False), dict(gate_is_fused=True)], ids=["no_q_lora", "gate"]
)
def test_kv_latents_match_the_wrapper_projection_variants(kwargs):
    w = mla_wrapper(**kwargs)
    hidden, positions = _inputs(16)
    with torch.no_grad():
        want_kv, want_pe = _wrapper_latents(w, hidden, positions)
        got_kv, got_pe = w.kv_latents(positions, hidden)
    assert torch.equal(got_kv, want_kv)
    assert torch.equal(got_pe, want_pe)


def test_kv_latents_match_the_wrapper_with_an_fp32_rope():
    """The DeepSeek scaling RoPE with an fp32 cache promotes the rotated
    k_pe to fp32; both paths cast it back to the projection dtype."""
    w = mla_wrapper(rope=mla_rope(scaling=True, dtype=torch.float32))
    hidden, positions = _inputs(16)
    with torch.no_grad():
        want_kv, want_pe = _wrapper_latents(w, hidden, positions)
        got_kv, got_pe = w.kv_latents(positions, hidden)
    assert got_pe.dtype == want_pe.dtype == torch.bfloat16
    assert torch.equal(got_kv, want_kv)
    assert torch.equal(got_pe, want_pe)


def test_kv_latents_positive_control():
    """Different positions must change k_pe: the comparison above can fail."""
    w = mla_wrapper()
    hidden, positions = _inputs(16)
    with torch.no_grad():
        _, want_pe = _wrapper_latents(w, hidden, positions)
        _, got_pe = w.kv_latents(positions + 1, hidden)
    assert not torch.equal(got_pe, want_pe)


# ---------------------------------------------------------------------------
# Indexer K
# ---------------------------------------------------------------------------


class _Captured(Exception):
    def __init__(self, args):
        super().__init__()
        self.args_ = args


def _packed_by_the_indexer_op(hidden, q_fp8, k, weights):
    """The packed rows `forward_oot` hands the top-k op (and its writer)."""

    def capture(*args):
        raise _Captured(args)

    op = SimpleNamespace(
        k_cache=SimpleNamespace(prefix="idx", kv_cache=torch.zeros(1)),
        topk_op_dcp=None,
        topk_op=capture,
    )
    md = AttentionMetadata(
        input_positions=torch.zeros(1),
        block_tables=torch.zeros(1, dtype=torch.int32),
        seq_lens=torch.zeros(1, dtype=torch.int32),
        query_start_loc=torch.zeros(2, dtype=torch.int32),
        request_distribution=torch.zeros(3, dtype=torch.int32),
    )
    with (
        patch.object(
            mla_op,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"idx": md}),
        ),
        pytest.raises(_Captured) as exc,
    ):
        mla_op.VllmTPUSparseAttnIndexer.forward_oot(op, hidden, q_fp8, k, weights)
    return exc.value.args_[3]


@pytest.mark.parametrize("num_tokens", [16, 96])
@pytest.mark.parametrize("scaling", [False, True], ids=["rope", "deepseek_scaling"])
def test_k_packed_matches_the_indexer(num_tokens, scaling):
    ind = dsa_indexer()
    rope = mla_rope(scaling=scaling)
    hidden, positions = _inputs(num_tokens)
    gen = torch.Generator().manual_seed(5)
    qr = torch.randn(num_tokens, Q_LORA, generator=gen).bfloat16()
    with torch.no_grad():
        h, q_fp8, k, weights = ind(hidden, qr, positions, rope)
        want = _packed_by_the_indexer_op(h, q_fp8, k, weights)
        got = ind.k_packed(hidden, positions, rope)
    assert got.dtype == want.dtype == torch.uint8
    assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# The cache writes
# ---------------------------------------------------------------------------

PAGE = 32
PAGES = 16

# Both cache layouts: SPARSECORE packs the fp8 RoPE cache to 64 B/token.
MLA_LAYOUTS = ["tensorcore", "sparsecore"]


def _mla_specs(layout="tensorcore"):
    from vllm_torchtpu.kernels.mla.kv_cache_utils import (
        KVCacheLayout,
        KVCacheType,
        SparseMLAKVCacheSpec,
    )

    cache_layout = KVCacheLayout(layout)
    return (
        SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE, cache_layout, PAGES, PAGE, KV_LORA, 4
        ),
        SparseMLAKVCacheSpec.create(
            KVCacheType.ROPE, cache_layout, PAGES, PAGE, ROPE, 4
        ),
    )


def _write_inputs(layout="tensorcore"):
    """One step's rows plus *different* MLA and indexer metadata, so a swap
    of the two metadata sets cannot go unnoticed."""
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        KVLayout,
        get_kv_cache_shape,
    )

    nope_spec, rope_spec = _mla_specs(layout)
    rng = np.random.default_rng(11)
    num_tokens = 8  # 7 real rows + 1 padded
    kv_c = jnp.asarray(rng.standard_normal((num_tokens, KV_LORA)), jnp.float8_e4m3fn)
    k_pe = jnp.asarray(rng.standard_normal((num_tokens, ROPE)), jnp.float8_e4m3fn)
    width = IDX_HEAD_DIM + 1
    k_packed = jnp.asarray(rng.integers(1, 255, (num_tokens, width), dtype=np.uint8))
    idx_shape = get_kv_cache_shape(PAGES, 16, width, KVLayout.HEAD_ALONG_SUBLANE)
    return dict(
        specs=(nope_spec, rope_spec),
        nope=jnp.zeros(nope_spec.shape, nope_spec.jax_dtype),
        rope=jnp.zeros(rope_spec.shape, rope_spec.jax_dtype),
        idx_cache=jnp.zeros(idx_shape, jnp.uint8),
        kv_c=kv_c,
        k_pe=k_pe,
        k_packed=k_packed,
        # MLA cache group: page 32, requests at positions 33..35 and 6..9
        # (with DCP 2 x interleave 8, rows 6, 7 and 33..35 sit on rank 0 and
        # rows 8, 9 on rank 1).
        seq_lens=jnp.array([36, 10], jnp.int32),
        block_tables=jnp.array([5, 2, 7, 7, 9, 7, 7, 7], jnp.int32),
        qsl=jnp.array([0, 3, 7], jnp.int32),
        # Indexer cache group: page 16, its own page table.
        idx_seq_lens=jnp.array([36, 10], jnp.int32),
        idx_pages=jnp.array([1, 3, 4, 0, 6, 0, 0, 0], jnp.int32),
        idx_cu=jnp.array([0, 3, 7], jnp.int32),
    )


def _golden(inp):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import KVLayout
    from vllm_torchtpu.kernels.mla import kv_cache_utils

    nope_spec, rope_spec = inp["specs"]
    nope, rope = kv_cache_utils.update_sparse_mla_kv_cache_jax(
        inp["nope"],
        inp["rope"],
        inp["kv_c"],
        inp["k_pe"],
        inp["seq_lens"],
        inp["block_tables"],
        inp["qsl"],
        nope_spec=nope_spec,
        rope_spec=rope_spec,
    )
    idx = mla_op.indexer_k_append(
        inp["idx_cache"],
        inp["k_packed"],
        inp["idx_seq_lens"],
        inp["idx_pages"],
        inp["idx_cu"],
        writer="xla",
        kv_layout=KVLayout.HEAD_ALONG_SUBLANE,
    )
    return nope, rope, idx


def _ordered(inp):
    return [
        inp[k]
        for k in (
            "nope",
            "rope",
            "idx_cache",
            "kv_c",
            "k_pe",
            "k_packed",
            "seq_lens",
            "block_tables",
            "qsl",
            "idx_seq_lens",
            "idx_pages",
            "idx_cu",
        )
    ]


def _assert_caches_equal(got, want):
    for g, w in zip(got, want):
        np.testing.assert_array_equal(
            np.asarray(g).view(np.uint8), np.asarray(w).view(np.uint8)
        )
    # Something was written at all.
    assert any(np.asarray(w).view(np.uint8).any() for w in want)


def _xla_latent_writers(monkeypatch):
    """The SparseCore latent writers need a TPU; use their pure-XLA goldens,
    which take the same arguments."""
    from vllm_torchtpu.kernels.mla import kv_cache_utils

    monkeypatch.setattr(
        attention_interface,
        "update_sparse_mla_kv_cache",
        kv_cache_utils.update_sparse_mla_kv_cache_jax,
    )
    monkeypatch.setattr(
        attention_interface,
        "update_sparse_mla_kv_cache_dcp",
        kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax,
    )


@pytest.mark.parametrize("layout", MLA_LAYOUTS)
def test_write_caches_matches_the_writers(monkeypatch, layout):
    import jax

    _xla_latent_writers(monkeypatch)
    inp = _write_inputs(layout)
    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()[:1]), ("x",))
    got = mla_op.mla_kv_write_caches(
        *_ordered(inp), mla_kv_spec=inp["specs"], mesh=mesh, kv_cache_mode="xla"
    )
    _assert_caches_equal(got, _golden(inp))


# The DCP write needs two devices; on CPU they exist only when XLA is told so
# before jax initializes, so the check runs in a subprocess.
_TWO_RANK_WORKER_ENV = "MLA_KV_WRITE_TWO_RANK_WORKER"


@pytest.mark.parametrize("layout", MLA_LAYOUTS)
def test_write_caches_dcp_two_ranks_keep_their_own_rows(request, layout):
    """Each DCP rank writes exactly the rows it owns into its own shard: the
    shard_map specs and the rank (`cp_rank_as_data`) are what the per-rank
    golden says. The indexer half is `_cp_scatter_indexer_k` itself."""
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count=2",
        **{_TWO_RANK_WORKER_ENV: "1"},
    )
    node = f"{__file__}::test_two_rank_worker[{layout}]"
    # The plugins this run loaded explicitly, so the child collects alike.
    plugins = [arg for p in request.config.option.plugins or () for arg in ("-p", p)]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *plugins,
            node,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "1 passed" in proc.stdout, proc.stdout[-3000:]


if os.environ.get(_TWO_RANK_WORKER_ENV) == "1":

    @pytest.mark.parametrize("layout", MLA_LAYOUTS)
    def test_two_rank_worker(monkeypatch, layout):
        _two_ranks_keep_their_own_rows(monkeypatch, layout)


def _two_ranks_keep_their_own_rows(monkeypatch, layout):
    import jax

    from vllm_torchtpu.distributed.dcp import DCP_LANE_AXIS_NAME
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import DCP_AXIS_NAME
    from vllm_torchtpu.kernels.mla import kv_cache_utils

    assert len(jax.devices()) >= 2, jax.devices()
    _xla_latent_writers(monkeypatch)
    dcp, interleave = 2, 8
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:dcp]).reshape(1, dcp),
        (DCP_LANE_AXIS_NAME, DCP_AXIS_NAME),
    )
    inp = _write_inputs(layout)
    nope_spec, rope_spec = inp["specs"]
    sharded = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(DCP_AXIS_NAME)
    )
    # Global caches: dcp shards of the per-rank spec stacked on dim 0.
    for key, spec in (("nope", nope_spec), ("rope", rope_spec)):
        inp[key] = jax.device_put(
            jnp.zeros((dcp * spec.shape[0], *spec.shape[1:]), spec.jax_dtype), sharded
        )
    idx_shape = inp["idx_cache"].shape
    inp["idx_cache"] = jax.device_put(
        jnp.zeros((dcp * idx_shape[0], *idx_shape[1:]), jnp.uint8), sharded
    )
    got = mla_op.mla_kv_write_caches(
        *_ordered(inp),
        mla_kv_spec=(nope_spec, rope_spec),
        mesh=mesh,
        kv_cache_mode="xla",
        dcp_size=dcp,
        interleave_size=interleave,
    )
    for r in range(dcp):
        want_nope, want_rope = kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax(
            jnp.zeros(nope_spec.shape, nope_spec.jax_dtype),
            jnp.zeros(rope_spec.shape, rope_spec.jax_dtype),
            inp["kv_c"],
            inp["k_pe"],
            inp["seq_lens"],
            inp["block_tables"],
            inp["qsl"],
            jnp.int32(r),
            nope_spec=nope_spec,
            rope_spec=rope_spec,
            dcp_size=dcp,
            interleave_size=interleave,
        )
        n0, r0 = nope_spec.shape[0], rope_spec.shape[0]
        np.testing.assert_array_equal(
            np.asarray(got[0])[r * n0 : (r + 1) * n0].view(np.uint8),
            np.asarray(want_nope).view(np.uint8),
        )
        np.testing.assert_array_equal(
            np.asarray(got[1])[r * r0 : (r + 1) * r0].view(np.uint8),
            np.asarray(want_rope).view(np.uint8),
        )
    want_idx = mla_op._cp_scatter_indexer_k(
        jax.device_put(jnp.zeros_like(inp["idx_cache"]), sharded),
        inp["k_packed"],
        inp["idx_seq_lens"],
        inp["idx_pages"],
        inp["idx_cu"],
        mesh=mesh,
        cp_size=dcp,
        interleave_size=interleave,
        axis_name=DCP_AXIS_NAME,
        writer="xla",
    )
    np.testing.assert_array_equal(np.asarray(got[2]), np.asarray(want_idx))
    # Both ranks got rows (the interleave spreads this step over both).
    for cache in got[:2]:
        halves = np.split(np.asarray(cache).view(np.uint8), dcp)
        assert all(h.any() for h in halves)


# ---------------------------------------------------------------------------
# MlaKvWriteOp
# ---------------------------------------------------------------------------

ATTN_NAME = "model.layers.0.self_attn.attn"
IDX_NAME = "model.layers.0.self_attn.indexer.k_cache"


class _RecordedJaxOp:
    """What `pallas.jax_op` / `pcp_streaming_jax_op` were handed; called, it
    returns the three caches plus 1, 2 and 3."""

    def __init__(self, name, fn, **kwargs):
        self.name, self.fn, self.kwargs = name, fn, kwargs
        self.fake = None
        self.calls = []

    def register_fake(self, fake):
        self.fake = fake

    def __call__(self, *args):
        self.calls.append(args)
        return args[0] + 1, args[1] + 2, args[2] + 3


def _record_jax_ops(monkeypatch):
    built = []

    def jax_op(name, fn, **kwargs):
        built.append(_RecordedJaxOp(name, fn, **kwargs))
        return built[-1]

    monkeypatch.setattr(mla_op.pallas, "jax_op", jax_op)
    monkeypatch.setattr(mla_op, "pcp_streaming_jax_op", jax_op)
    return built


def _indexer_op(dcp_size=1, kv_cache_mode="xla"):
    op = mla_op.VllmTPUSparseAttnIndexer.__new__(mla_op.VllmTPUSparseAttnIndexer)
    torch.nn.Module.__init__(op)
    op.dcp_size = dcp_size
    op.kv_cache_mode = kv_cache_mode
    op.k_cache = SimpleNamespace(
        prefix=IDX_NAME, kv_cache=torch.zeros(4, dtype=torch.uint8)
    )
    return op


def _op_wrapper(
    *,
    dcp_size=1,
    indexer_dcp_size=None,
    layout="tensorcore",
    quantized=False,
    kv_cache_mode="xla",
):
    """A wrapper whose MLA layer, backend impl and indexer op carry what
    `MlaKvWriteOp` reads."""
    layer = SimpleNamespace(
        layer_name=ATTN_NAME,
        mla_kv_spec=_mla_specs(layout),
        kv_cache_quantized_dtype=torch.float8_e4m3fn if quantized else None,
        _k_scale_float=0.5,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        impl=mla_impl(dcp_size),
    )
    w = mla_wrapper(indexer=dsa_indexer(), mla_attn=layer)
    w.indexer.indexer_op = _indexer_op(
        dcp_size if indexer_dcp_size is None else indexer_dcp_size, kv_cache_mode
    )
    return w


def _md(n):
    return AttentionMetadata(
        input_positions=torch.zeros(1),
        block_tables=torch.full((4,), n, dtype=torch.int32),
        seq_lens=torch.full((2,), n + 1, dtype=torch.int32),
        query_start_loc=torch.full((3,), n + 2, dtype=torch.int32),
        request_distribution=torch.zeros(3, dtype=torch.int32),
    )


@contextlib.contextmanager
def _contexts(kv_cache, md_mla, md_idx):
    with (
        patch.object(
            mla_op,
            "get_attention_context",
            lambda name: (md_mla if name == ATTN_NAME else None, None, kv_cache, None),
        ),
        patch.object(
            mla_op,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={IDX_NAME: md_idx}),
        ),
    ):
        yield


@pytest.mark.parametrize("quantized", [True, False])
def test_write_op_operands(monkeypatch, quantized):
    """The op hands its writer the latents `kv_latents` / `k_packed` produce,
    quantized and flattened by the impl's `_quantized_kv_latents` (the
    method the full pass uses), each cache group's own metadata, and writes
    the result back in place."""
    from vllm_torchtpu.layers.core.quantization import quantize_kv

    built = _record_jax_ops(monkeypatch)
    op = mla_op.MlaKvWriteOp(_op_wrapper(quantized=quantized), mesh=None)
    (jax_op,) = built
    nope, rope = torch.zeros(3), torch.zeros(3)
    idx_cache = op.indexer_op.k_cache.kv_cache
    md_mla, md_idx = _md(10), _md(20)
    hidden, positions = _inputs(16)
    with torch.no_grad(), _contexts((nope, rope), md_mla, md_idx):
        out = op(positions, hidden)
        kv_c, k_pe = op.wrapper.kv_latents(positions, hidden)
        k_packed = op.indexer.k_packed(hidden, positions, op.wrapper.indexer_rope_emb)
    assert torch.equal(out, k_pe)
    if quantized:
        kv_c, _ = quantize_kv(torch.float8_e4m3fn, kv_c, value=None, k_scale=0.5)
        k_pe, _ = quantize_kv(torch.float8_e4m3fn, k_pe, value=None, k_scale=0.5)
    (args,) = jax_op.calls
    assert args[0] is nope and args[1] is rope and args[2] is idx_cache
    assert args[3].dtype == kv_c.dtype and args[4].dtype == k_pe.dtype
    assert torch.equal(
        args[3].view(torch.uint8), kv_c.view(-1, KV_LORA).view(torch.uint8)
    )
    assert torch.equal(args[4].view(torch.uint8), k_pe.view(-1, ROPE).view(torch.uint8))
    assert torch.equal(args[5], k_packed)
    # MLA metadata feeds the MLA caches, the indexer's feeds the indexer cache.
    assert args[6] is md_mla.seq_lens
    assert args[7] is md_mla.block_tables
    assert args[8] is md_mla.query_start_loc
    assert args[9] is md_idx.seq_lens
    assert args[10] is md_idx.block_tables
    assert args[11] is md_idx.query_start_loc
    assert len(args) == 12
    # Written back in place.
    assert torch.equal(nope, torch.ones(3))
    assert torch.equal(rope, torch.full((3,), 2.0))
    assert torch.equal(idx_cache, torch.full((4,), 3, dtype=torch.uint8))


@pytest.mark.parametrize("unbound", ["mla", "indexer"])
def test_write_op_refuses_unbound_caches(monkeypatch, unbound):
    """The memory probe does not run this op, so a trace without bound
    caches would compile a program that never writes: refuse it."""
    built = _record_jax_ops(monkeypatch)
    op = mla_op.MlaKvWriteOp(_op_wrapper(), mesh=None)
    if unbound == "indexer":
        op.indexer_op.k_cache.kv_cache = torch.zeros(0, dtype=torch.uint8)
        kv_cache = (torch.zeros(3), torch.zeros(3))
    else:
        kv_cache = torch.zeros(0)
    hidden, positions = _inputs(16)
    with (
        torch.no_grad(),
        _contexts(kv_cache, _md(1), _md(2)),
        pytest.raises(RuntimeError, match="not bound"),
    ):
        op(positions, hidden)
    assert built[0].calls == []


@pytest.mark.parametrize("missing", ["mla", "indexer"])
def test_write_op_refuses_bound_caches_without_metadata(monkeypatch, missing):
    built = _record_jax_ops(monkeypatch)
    op = mla_op.MlaKvWriteOp(_op_wrapper(), mesh=None)
    hidden, positions = _inputs(16)
    md_mla = None if missing == "mla" else _md(1)
    md_idx = None if missing == "indexer" else _md(2)
    with (
        torch.no_grad(),
        _contexts((torch.zeros(3), torch.zeros(3)), md_mla, md_idx),
        pytest.raises(RuntimeError, match="no attention metadata"),
    ):
        op(positions, hidden)
    assert built[0].calls == []


@pytest.mark.parametrize("layout", MLA_LAYOUTS)
def test_write_op_registers_one_donating_op_around_the_writers(monkeypatch, layout):
    """`__init__` registers one op that donates the three caches; the function
    it registers is the unsharded writer pair (checked against the goldens)."""
    import jax

    built = _record_jax_ops(monkeypatch)
    _xla_latent_writers(monkeypatch)
    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()[:1]), ("x",))
    op = mla_op.MlaKvWriteOp(_op_wrapper(layout=layout), mesh)
    (jax_op,) = built
    assert op.jax_op is jax_op
    assert jax_op.name == "pallas::mla_kv_write_" + ATTN_NAME.replace(".", "_")
    assert jax_op.kwargs == {"donate_argnums": (0, 1, 2)}
    inp = _write_inputs(layout)
    _assert_caches_equal(jax_op.fn(*_ordered(inp)), _golden(inp))
    fake = jax_op.fake(torch.zeros(3), torch.zeros(4), torch.zeros(5), torch.zeros(1))
    assert [t.shape[0] for t in fake] == [3, 4, 5]


def test_write_op_writes_with_the_indexer_ops_mode(monkeypatch):
    """The indexer writer uses the mode the indexer op resolved, not a fresh
    read of VLLM_TPU_INDEXER_KV_CACHE_MODE."""
    import jax

    from vllm_torchtpu import envs

    built = _record_jax_ops(monkeypatch)
    _xla_latent_writers(monkeypatch)
    monkeypatch.setattr(envs, "is_indexer_kv_cache_mode_explicit", lambda: True)
    monkeypatch.setattr(envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE", "xla")
    mesh = jax.sharding.Mesh(np.asarray(jax.local_devices()[:1]), ("x",))
    mla_op.MlaKvWriteOp(_op_wrapper(kv_cache_mode="bogus"), mesh)
    (jax_op,) = built
    with pytest.raises(ValueError, match="'bogus'"):
        jax_op.fn(*_ordered(_write_inputs()))


def test_write_op_under_dcp_registers_a_dcp_sharded_op(monkeypatch):
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import DCP_AXIS_NAME

    built = _record_jax_ops(monkeypatch)
    mesh = object()
    monkeypatch.setattr(mla_op, "get_or_create_dcp_mesh", lambda axis_name: mesh)
    op = mla_op.MlaKvWriteOp(_op_wrapper(dcp_size=2), mesh=None)
    (jax_op,) = built
    assert op.dcp_size == 2
    assert jax_op.name == "pallas::mla_kv_write_dcp_" + ATTN_NAME.replace(".", "_")
    ax = P(DCP_AXIS_NAME)
    assert jax_op.kwargs == {
        "donate_argnums": (0, 1, 2),
        "mesh": mesh,
        "input_partition_specs": (ax, ax, ax) + (P(),) * 9,
        "output_partition_specs": (ax, ax, ax),
    }
    assert jax_op.fake is not None


def test_write_op_refuses_disagreeing_dcp_sizes(monkeypatch):
    built = _record_jax_ops(monkeypatch)
    with pytest.raises(ValueError, match="disagree on the DCP size: 2 vs 1"):
        mla_op.MlaKvWriteOp(_op_wrapper(dcp_size=2, indexer_dcp_size=1), mesh=None)
    assert built == []


def test_write_op_under_dcp_writes_with_its_layout(monkeypatch):
    """The registered function hands the write helper the DCP size, the
    interleave, the mesh, the cache specs and the mode it was built with."""
    built = _record_jax_ops(monkeypatch)
    mesh = object()
    monkeypatch.setattr(mla_op, "get_or_create_dcp_mesh", lambda axis_name: mesh)
    calls = []
    monkeypatch.setattr(
        mla_op, "mla_kv_write_caches", lambda *a, **kw: calls.append(kw)
    )
    w = _op_wrapper(dcp_size=2)
    mla_op.MlaKvWriteOp(w, mesh=None)
    (jax_op,) = built
    jax_op.fn(*range(12))
    assert calls == [
        dict(
            mla_kv_spec=w.mla_attn.mla_kv_spec,
            mesh=mesh,
            kv_cache_mode="xla",
            dcp_size=2,
            interleave_size=8,
        )
    ]


def test_write_op_refuses_the_kimi_preprocess(monkeypatch):
    built = _record_jax_ops(monkeypatch)
    w = _op_wrapper()
    w.kimi_preprocess = object()
    with pytest.raises(NotImplementedError, match="Kimi"):
        mla_op.MlaKvWriteOp(w, mesh=None)
    assert built == []


@pytest.mark.parametrize("part", ["impl", "indexer_op"])
def test_write_op_refuses_other_impls_and_indexer_ops(monkeypatch, part):
    built = _record_jax_ops(monkeypatch)
    w = _op_wrapper()
    if part == "impl":
        w.mla_attn.impl = SimpleNamespace(dcp_size=1)
    else:
        w.indexer.indexer_op = torch.nn.Module()
    with pytest.raises(TypeError, match="expected a"):
        mla_op.MlaKvWriteOp(w, mesh=None)
    assert built == []


# ---------------------------------------------------------------------------
# On TPU: the compiled op, the default writers and an fp8 cache
# ---------------------------------------------------------------------------

TPU_MLA = "model.layers.0.self_attn.mla_attn.mla_attn"
TPU_IDX = "model.layers.0.self_attn.mla_attn.indexer.k_cache"


class _FullPass(torch.nn.Module):
    """The wrapper's forward: projections, indexer (top-k + K write), sparse
    MLA (latent write + attention)."""

    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, positions, hidden_states):
        return self.wrapper(positions, hidden_states)


class _KvOnly(torch.nn.Module):
    """The same rows through `MlaKvWriteOp` only."""

    def __init__(self, wrapper, op):
        super().__init__()
        self.wrapper = wrapper
        self.op = op

    def forward(self, positions, hidden_states):
        return self.op(positions, hidden_states)


def _tpu_layer():
    """`build_tpu_sparse_mla` plus the write op around its wrapper."""
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    built = build_tpu_sparse_mla(TPU_MLA, TPU_IDX)
    w = built.wrapper
    with set_vllm_model_wrapper_context(mesh=built.mesh):
        write_op = mla_op.MlaKvWriteOp(w, built.mesh)
    built.full = _FullPass(w)
    built.kv_only = _KvOnly(w, write_op)
    return built


def _tpu_md(seq_lens, query_start_loc, request_distribution):
    return tpu_md((TPU_MLA, TPU_IDX), seq_lens, query_start_loc, request_distribution)


@pytest.fixture(scope="module")
def tpu_layer():
    """One build per module: the ops register under fixed names. Request 0
    has `TPU_PREFIX` tokens cached before the step (written here by the full
    pass); the step is a 64-token chunk: request 0's next 32 tokens, request
    1's first 24, 8 padded rows."""
    torch._dynamo.reset()
    layer = _tpu_layer()
    layer.full_c = tpu_compiled(layer.full)
    layer.kv_only_c = tpu_compiled(layer.kv_only)
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
    prefix_md = _tpu_md([TPU_PREFIX, 0], [0, TPU_PREFIX, TPU_PREFIX], [0, 0, 1])
    on_tpu(layer, layer.full_c, prefix_md, *prefix)
    # Host copies: a deferred on-device clone can lose its source to the
    # next op that donates the cache.
    layer.prefix_caches = [c.cpu() for c in layer.caches]
    layer.step = tokens(list(range(TPU_PREFIX, end)) + list(range(24)) + [0] * 8)
    layer.step_md = lambda: _tpu_md([end, 24], [0, 32, 56], [0, 0, 2])
    # Each request's last row, padded to 8 rows with row 0, then read back as
    # decode rows (one query per request) over the cache the step filled.
    layer.last_rows = torch.tensor([31, 55] + [0] * 6, dtype=torch.int64, device=dev)
    layer.rows_md = lambda: _tpu_md([end, 24], [0, 1, 2], [2, 2, 2])
    return layer


def _reset(layer):
    reset_caches(layer, layer.prefix_caches)


def _single_pass(layer):
    _reset(layer)
    out = on_tpu(layer, layer.full_c, layer.step_md(), *layer.step)
    topk = layer.topk_rows[[31, 55]].cpu()
    return out.cpu(), [c.cpu() for c in layer.caches], topk


def _two_pass(layer):
    """`MlaKvWriteOp` over every row, then the full forward on each
    request's last row with decode metadata."""
    _reset(layer)
    on_tpu(layer, layer.kv_only_c, layer.step_md(), *layer.step)
    positions, hidden = layer.step
    rows = layer.last_rows
    out = on_tpu(
        layer,
        layer.full_c,
        layer.rows_md(),
        positions.index_select(0, rows),
        hidden.index_select(0, rows),
    )
    topk = layer.topk_rows[:2].cpu()
    return out.cpu(), [c.cpu() for c in layer.caches], topk


@pytest.mark.tpu_test
def test_write_op_writes_the_full_pass_cache_entries_on_tpu(tpu_layer):
    """`MlaKvWriteOp`, compiled for TPU with the default `sal` indexer writer
    and an fp8 latent cache, leaves the same MLA and indexer-K caches as the
    full forward over the same rows."""
    _, want, _ = _single_pass(tpu_layer)
    assert any(not torch.equal(w, p) for w, p in zip(want, tpu_layer.prefix_caches)), (
        "the step wrote nothing"
    )
    _reset(tpu_layer)
    on_tpu(tpu_layer, tpu_layer.kv_only_c, tpu_layer.step_md(), *tpu_layer.step)
    assert_same_bytes(tpu_layer.caches, want, "write op vs full pass")


@pytest.mark.tpu_test
def test_rows_read_back_after_the_write_op_match_the_full_pass_on_tpu(tpu_layer):
    """The write op over every row, then the full forward on one row per
    request as decode rows: same caches as the single pass, and each
    request's last row gets the same attention output."""
    want_out, want_caches, _ = _single_pass(tpu_layer)
    out, caches, _ = _two_pass(tpu_layer)
    assert_same_bytes(caches, want_caches, "write op + decode rows vs full pass")
    torch.testing.assert_close(out[:2], want_out[[31, 55]], rtol=1e-2, atol=1e-2)


@pytest.mark.tpu_test
def test_decode_rows_select_the_full_pass_top_k_on_tpu(tpu_layer):
    """Request 0's last row has a context (352) longer than the top-k (256),
    so its indexer really selects. Read back as a decode row after the write
    op, it selects the same positions as the full pass's prefill row; request
    1 (24 tokens) keeps every position.

    The decode and prefill paths of the top-k kernel may order the selected
    positions differently, so the rows are compared as sets. They may also
    break exact score ties differently; the scores here come from random
    projections of random hidden states, so a tie at the cut-off has
    negligible probability, and a mismatch is reported with both sets.
    """
    _, _, want = _single_pass(tpu_layer)
    _, _, got = _two_pass(tpu_layer)
    for r, context in ((0, TPU_PREFIX + 32), (1, 24)):
        want_set = {int(i) for i in want[r] if i >= 0}
        got_set = {int(i) for i in got[r] if i >= 0}
        assert len(want_set) == min(context, TPU_TOPK), (r, len(want_set))
        assert want_set <= set(range(context))
        assert got_set == want_set, (
            f"request {r}: decode row selects {sorted(got_set - want_set)} "
            f"instead of {sorted(want_set - got_set)}"
        )


# ---------------------------------------------------------------------------
# On several TPU chips: under DCP, the write op vs the full pass's writers
# ---------------------------------------------------------------------------


@pytest.mark.tpu_test
@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_write_op_matches_the_full_pass_caches_under_dcp_on_tpu(monkeypatch, dcp_size):
    """Under DCP, the function `MlaKvWriteOp` registers, run with the
    partition specs it registers on `dcp_size` chips, leaves the same latent
    and indexer-K caches (every rank's shard) as the full pass's two DCP ops:
    the sparse MLA attention (latent write) and the indexer's top-k op (K
    write), each run with its own registered partition specs. Real TPU
    writers, default modes, fp8 latent cache."""
    import jax

    from vllm_torchtpu.distributed.dcp import DCP_LANE_AXIS_NAME
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import DCP_AXIS_NAME
    from vllm_torchtpu.layers.adapter import attention as attention_adapter
    from vllm_torchtpu.layers.adapter.attention import (
        PallasMLAttentionBackend,
        VllmTPUDeepseekV32IndexerBackend,
    )

    devices = jax.devices()
    if len(devices) < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {len(devices)}")
    mesh = jax.sharding.Mesh(
        np.asarray(devices[:dcp_size]).reshape(1, dcp_size),
        (DCP_LANE_AXIS_NAME, DCP_AXIS_NAME),
    )
    built = _record_jax_ops(monkeypatch)
    monkeypatch.setattr(
        attention_adapter, "pcp_streaming_jax_op", mla_op.pcp_streaming_jax_op
    )
    monkeypatch.setattr(mla_op, "get_or_create_dcp_mesh", lambda axis_name=None: mesh)
    interleave = 8

    # The full pass's two DCP ops.
    specs = PallasMLAttentionBackend.get_sparse_kv_cache_specs(
        TPU_PAGES, TPU_PAGE, KV_LORA + ROPE, "fp8"
    )
    impl = mla_impl(dcp_size)
    impl.dcp_interleave_size = interleave
    impl._dcp_mesh = mesh
    layer = SimpleNamespace(
        layer_name=TPU_MLA,
        mla_kv_spec=specs,
        kv_cache_quantized_dtype=torch.float8_e4m3fn,
        _k_scale_float=1.0,
        scale=QK**-0.5,
        num_heads=TPU_HEADS,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        impl=impl,
    )
    impl._build_sparse_mla_dcp_op(layer, dcp_size, interleave)
    indexer_op = _indexer_op(dcp_size, mla_op._explicit_indexer_kv_cache_mode())
    indexer_op.topk_tokens = TPU_TOPK
    indexer_op.k_cache.prefix = TPU_IDX
    indexer_op._build_streamidx_dcp_op(dcp_size, interleave)
    # The write op, around the same layer and indexer op.
    w = mla_wrapper(indexer=dsa_indexer(), mla_attn=layer)
    w.indexer.indexer_op = indexer_op
    mla_op.MlaKvWriteOp(w, mesh=None)
    attn_op, topk_op, write_op = built

    def run(recorded, *args):
        def sharding(specs):
            return tuple(jax.sharding.NamedSharding(mesh, s) for s in specs)

        kwargs = recorded.kwargs
        return jax.jit(
            recorded.fn,
            in_shardings=sharding(kwargs["input_partition_specs"]),
            out_shardings=sharding(kwargs["output_partition_specs"]),
        )(*args)

    rng = np.random.default_rng(dcp_size)
    # Request 0: 64 new tokens at positions 236..299, nine interleave chunks,
    # so every rank owns some; request 1: its first 24; 8 padded rows. Both
    # cache groups share the page table, as their KV cache groups do; the
    # entries are virtual page ordinals.
    n = 96
    seq_lens = jnp.array([300, 24], jnp.int32)
    block_tables = jnp.array([0, 1, 2, 3], jnp.int32)
    qsl = jnp.array([0, 64, 88], jnp.int32)
    distribution = jnp.array([0, 0, 2], jnp.int32)
    kv_c = jnp.asarray(rng.standard_normal((n, KV_LORA)), jnp.float8_e4m3fn)
    k_pe = jnp.asarray(rng.standard_normal((n, ROPE)), jnp.float8_e4m3fn)
    k_fp8 = jnp.asarray(rng.standard_normal((n, IDX_HEAD_DIM)), jnp.float8_e4m3fn)
    scale = jnp.full((n, 1), 127, jnp.uint8)  # e8m0 2**0
    k_packed = jnp.concatenate(
        [jax.lax.bitcast_convert_type(k_fp8, jnp.uint8), scale], axis=-1
    )
    q_bytes = jax.lax.bitcast_convert_type(
        jnp.asarray(
            rng.standard_normal((n, IDX_N_HEAD, IDX_HEAD_DIM)), jnp.float8_e4m3fn
        ),
        jnp.uint8,
    )
    weights = jnp.asarray(rng.standard_normal((n, IDX_N_HEAD)), jnp.bfloat16)
    ql_nope = jnp.asarray(rng.standard_normal((n, TPU_HEADS, KV_LORA)), jnp.bfloat16)
    q_pe = jnp.asarray(rng.standard_normal((n, TPU_HEADS, ROPE)), jnp.bfloat16)
    local_topk = jnp.zeros((dcp_size * n, TPU_TOPK), jnp.int32)

    def empty_caches():
        nope_spec, rope_spec = specs
        idx_shape = VllmTPUDeepseekV32IndexerBackend.get_kv_cache_shape(
            TPU_PAGES, TPU_PAGE, 1, IDX_HEAD_DIM
        )
        return tuple(
            jnp.zeros((dcp_size * shape[0], *shape[1:]), dtype)
            for shape, dtype in (
                (nope_spec.shape, nope_spec.jax_dtype),
                (rope_spec.shape, rope_spec.jax_dtype),
                (idx_shape, jnp.uint8),
            )
        )

    nope, rope, idx = empty_caches()
    want_nope, want_rope, _, _ = run(
        attn_op,
        nope,
        rope,
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        local_topk,
        seq_lens,
        block_tables,
        qsl,
        distribution,
    )
    want_idx, _ = run(
        topk_op,
        q_bytes,
        weights,
        idx,
        k_packed,
        seq_lens,
        block_tables,
        qsl,
        distribution,
    )
    nope, rope, idx = empty_caches()
    got = run(
        write_op,
        nope,
        rope,
        idx,
        kv_c,
        k_pe,
        k_packed,
        seq_lens,
        block_tables,
        qsl,
        seq_lens,
        block_tables,
        qsl,
    )
    for name, g, w_ in zip(
        ("nope", "rope", "indexer K"), got, (want_nope, want_rope, want_idx)
    ):
        g8 = np.asarray(g).view(np.uint8)
        w8 = np.asarray(w_).view(np.uint8)
        assert (g8 != w8).sum() == 0, f"dcp={dcp_size}: {name} cache differs"
        # Every rank's shard got rows.
        assert all(s.any() for s in np.split(w8, dcp_size)), name
