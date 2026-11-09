# SPDX-License-Identifier: Apache-2.0
"""Rank-rows draft context K/V write vs the v29 sharded write, byte for byte.

`store_context_kv_rank_rows` splits the DSpark context K/V work over the DCP
group by token rows (fc, a K/V-only projection of every group KV head, the
packed cache rows), all-gathers the packed rows and copies each rank's own
tokens into its pages (`write_owned_rows`). The v29 path (frozen below) runs
fc and qkv_proj on every row on every rank, all-gathers bf16 K|V and
scatters with `write_kv`. Every rank's whole cache must come out identical.

The DCP group is simulated with one thread per rank and a barrier
all-gather; JAX calls are serialized.
"""

import threading
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch import nn

from vllm_torchtpu.layers.adapter import cp_draft_attention as cda
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.models.minimax_m3 import dspark

PAGE = 128
HEAD_DIM = 128
HIDDEN = 256
NUM_AUX = 5
LAYERS = 2
KV_HEADS = 4
TP = 8
REPLICAS = TP // KV_HEADS
Q_HEADS_PER_RANK = 2
ROT = 64
EPS = 1e-6

_JAX_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Frozen v29 write (cp_draft_attention.write_kv at 80f0fed879).
# ---------------------------------------------------------------------------


def _v29_token_layout(seq_lens, cu_q_lens, num_tokens):
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.searchsorted(cu_q_lens[1:], tok, side="right", method="compare_all")
    seq = jnp.minimum(seq.astype(jnp.int32), seq_lens.shape[0] - 1)
    q_len = cu_q_lens[seq + 1] - cu_q_lens[seq]
    pos = seq_lens[seq] - q_len + (tok - cu_q_lens[seq])
    valid = tok < cu_q_lens[-1]
    return seq, jnp.where(valid, pos, 0), valid


def _v29_merge_rows(key, value, page_shape):
    num_tokens, kv_heads, head_dim = key.shape
    words, packing, width = page_shape
    kv = jnp.concatenate([key, value], axis=-1).reshape(num_tokens, 2 * kv_heads, head_dim)
    pad_heads = words * packing - 2 * kv_heads
    kv = jnp.pad(kv, ((0, 0), (0, pad_heads), (0, width - head_dim)))
    return kv.reshape(num_tokens, words, packing, width)


def _v29_quantize(x, dtype, scale):
    if scale is not None:
        x = x.astype(jnp.float32) / scale
    if jnp.issubdtype(dtype, jnp.floating) and jnp.finfo(dtype).bits == 8:
        limit = float(jnp.finfo(dtype).max)
        x = jnp.clip(x, -limit, limit)
    return x.astype(dtype)


def _v29_write_kv(kv_cache, key, value, seq_lens, block_tables, cu_q_lens, *, dcp, rank,
                  k_scale=None, v_scale=None):
    num_tokens = key.shape[0]
    num_pages, page_size = kv_cache.shape[:2]
    max_seqs = seq_lens.shape[0]
    tables = block_tables.reshape(max_seqs, -1)
    seq, pos, valid = _v29_token_layout(seq_lens, cu_q_lens, num_tokens)
    block = pos // page_size
    mine = valid & (block % dcp == rank)
    page = jnp.where(mine, tables[seq, block // dcp], num_pages)
    rows = _v29_merge_rows(
        _v29_quantize(key, kv_cache.dtype, k_scale),
        _v29_quantize(value, kv_cache.dtype, v_scale),
        kv_cache.shape[2:],
    )
    return kv_cache.at[page, pos % page_size].set(rows, mode="drop")


# ---------------------------------------------------------------------------
# torch <-> jax, simulated group, tiny drafter
# ---------------------------------------------------------------------------

_VIEW = {
    torch.float8_e4m3fn: (torch.uint8, jnp.float8_e4m3fn),
    torch.bfloat16: (torch.int16, jnp.bfloat16),
}


def _t2j(t: torch.Tensor) -> jax.Array:
    if t.dtype in _VIEW:
        raw, jdt = _VIEW[t.dtype]
        return jnp.asarray(t.contiguous().view(raw).numpy()).view(jdt)
    return jnp.asarray(t.numpy())


def _j2t(a: jax.Array, dtype: torch.dtype) -> torch.Tensor:
    if dtype in _VIEW:
        raw, _ = _VIEW[dtype]
        np_raw = {torch.uint8: np.uint8, torch.int16: np.int16}[raw]
        return torch.from_numpy(np.asarray(a).view(np_raw).copy()).view(dtype)
    return torch.from_numpy(np.asarray(a).copy())


def _jdtype(dtype: torch.dtype):
    return _VIEW[dtype][1]


class _Group:
    def __init__(self, size: int):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots: list = [None] * size
        self.local = threading.local()

    def all_gather(self, t: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self.slots[self.local.rank] = t.clone()
        self.barrier.wait()
        out = torch.cat(self.slots, dim=dim)
        self.barrier.wait()
        return out


def _run_ranks(group: _Group, fn):
    errors = []

    def run(rank):
        group.local.rank = rank
        try:
            fn(rank)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            group.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(group.size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


class _GemmaNorm(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + EPS) * (1.0 + self.weight.float())).to(x.dtype)


class _Rope:
    """Partial NeoX RoPE on every head of ``[T, heads * D]``, as vLLM's."""

    def __call__(self, positions, query, key=None):
        x = query.view(query.shape[0], -1, HEAD_DIM)
        inv = 1.0 / (10000.0 ** (torch.arange(0, ROT, 2, dtype=torch.float32) / ROT))
        ang = positions.float()[:, None] * inv[None, :]
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
        x1, x2 = x[..., : ROT // 2].float(), x[..., ROT // 2 : ROT].float()
        rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)
        return torch.cat([rot, x[..., ROT:]], dim=-1).reshape(query.shape), key


class _Linear(nn.Module):
    def __init__(self, weight, returns_tuple):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.quant_method = VllmUnquantizedLinearMethod()
        setattr(self, WEIGHT_FLIPPED_ATTR, True)
        self.returns_tuple = returns_tuple

    def forward(self, x):
        out = torch.matmul(x, self.weight)
        return (out, None) if self.returns_tuple else out


def _weights(seed):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    return dict(
        fc=rnd(NUM_AUX * HIDDEN, HIDDEN, scale=0.05),
        hidden_norm=rnd(HIDDEN, scale=0.1),
        layers=[
            dict(
                q=rnd(TP, HIDDEN, Q_HEADS_PER_RANK * HEAD_DIM, scale=0.06),
                k=rnd(KV_HEADS, HIDDEN, HEAD_DIM, scale=0.06),
                v=rnd(KV_HEADS, HIDDEN, HEAD_DIM, scale=0.06),
                k_norm=rnd(HEAD_DIM, scale=0.1),
            )
            for _ in range(LAYERS)
        ],
    )


def _rank_model(w, tp_rank, dcp_rank, dcp, cache_dtype, scale, init_cache):
    """The drafter as TP rank `tp_rank` holds it: own Q heads, KV head
    ``tp_rank // REPLICAS``."""

    def write(kv_cache, key, value, seq_lens, block_tables, qsl, rank):
        with _JAX_LOCK:
            out = _v29_write_kv(
                _t2j(kv_cache), _t2j(key), _t2j(value), _t2j(seq_lens),
                _t2j(block_tables), _t2j(qsl), dcp=dcp, rank=jnp.int32(int(rank[0])),
                k_scale=scale, v_scale=scale,
            )
            return _j2t(out, kv_cache.dtype)

    def pack(kv_cache, key, value):
        with _JAX_LOCK:
            out = cda.pack_rows(
                _t2j(key), _t2j(value), tuple(kv_cache.shape[2:]), _jdtype(kv_cache.dtype),
                k_scale=scale, v_scale=scale,
            )
            return _j2t(out, kv_cache.dtype)

    def write_rows(kv_cache, rows, seq_lens, block_tables, qsl, rank):
        with _JAX_LOCK:
            out = cda.write_owned_rows(
                _t2j(kv_cache), _t2j(rows), _t2j(seq_lens), _t2j(block_tables),
                _t2j(qsl), dcp=dcp, rank=jnp.int32(int(rank[0])), interpret=True,
            )
            return _j2t(out, kv_cache.dtype)

    impl = SimpleNamespace(
        sharded_draft_dcp=dict(ops=dict(write=write, pack=pack, write_rows=write_rows))
    )
    head = tp_rank // REPLICAS
    layers = []
    for lw in w["layers"]:
        attn = nn.Module()
        qkv = torch.cat([lw["q"][tp_rank], lw["k"][head], lw["v"][head]], dim=1)
        attn.qkv_proj = _Linear(qkv, returns_tuple=True)
        attn.q_size = Q_HEADS_PER_RANK * HEAD_DIM
        attn.kv_size = HEAD_DIM
        attn.head_dim = HEAD_DIM
        attn.num_kv_heads = 1
        attn.k_norm = _GemmaNorm(lw["k_norm"])
        attn.rotary_emb = _Rope()
        attn.attn = SimpleNamespace(
            impl=impl,
            kv_cache=init_cache.clone(),
            draft_dcp_rank=torch.tensor([dcp_rank], dtype=torch.int32),
        )
        layers.append(SimpleNamespace(self_attn=attn))
    return SimpleNamespace(
        fc=_Linear(w["fc"], returns_tuple=False),
        hidden_norm=_GemmaNorm(w["hidden_norm"]),
        layers=layers,
        rank_rows_context_kv=None,
    )


def _v29_store(model, aux, positions, metadata, group, dcp):
    """v29 `store_context_kv` (sharded branch) + `write_with_dcp`."""
    context = model.hidden_norm(model.fc(torch.cat(aux, dim=-1)))
    for layer, md in zip(model.layers, metadata, strict=True):
        attn = layer.self_attn
        qkv, _ = attn.qkv_proj(context)
        k, v = qkv[:, attn.q_size :].split([attn.kv_size, attn.kv_size], dim=-1)
        k = dspark._per_head_norm(k, attn.k_norm, attn.head_dim)
        k, _ = attn.rotary_emb(positions, k, None)
        num_tokens = k.shape[0]
        key = k.view(num_tokens, attn.num_kv_heads, HEAD_DIM)
        value = v.view(num_tokens, attn.num_kv_heads, HEAD_DIM)
        if dcp > REPLICAS:
            gathered = group.all_gather(torch.cat([key, value], dim=1).contiguous(), dim=1)
            per_rank = gathered.view(num_tokens, dcp, 2, HEAD_DIM)[:, ::REPLICAS]
            key, value = per_rank[:, :, 0].contiguous(), per_rank[:, :, 1].contiguous()
        cache = attn.attn.kv_cache
        new = attn.attn.impl.sharded_draft_dcp["ops"]["write"](
            cache, key, value, md.seq_lens, md.block_tables, md.query_start_loc,
            attn.attn.draft_dcp_rank,
        )
        cache.copy_(new)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _metadata(case, dcp, rng):
    """(seq_lens, block_tables, query_start_loc, num_tokens) for a case.

    ``seq_lens`` include the step's tokens; each request's tokens start at
    ``ctx``. Pages are distinct and below the last page (the target of a
    wrapped -1)."""
    if case == "prefill_window":  # one chunk crossing the 4096 window, 4 padding rows
        reqs, pad = [(3900, 1100), (0, 0), (0, 0), (0, 0)], 4
    elif case == "mixed":  # chunk + verify blocks (some straddling a page) + empty
        reqs = [(4000, 700), (125, 8), (250, 8), (1021, 8), (5, 8), (383, 8), (64, 8), (0, 0)]
        pad = 5
    elif case == "decode":
        ctx = rng.integers(0, 2500, 12)
        ctx[:4] = [120, 127, 128, 255]
        reqs, pad = [(int(c), 8) for c in ctx], 0
    elif case == "bad_pages":
        ctx = rng.integers(0, 3000, 8)
        reqs, pad = [(int(c), 7) for c in ctx], 3
    else:
        raise AssertionError(case)
    max_seqs = len(reqs)
    q_lens = np.array([q for _, q in reqs], np.int32)
    seq_lens = np.array([c + q for c, q in reqs], np.int32)
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    num_tokens = int(cu[-1]) + pad
    cols = -(-int(seq_lens.max()) // (PAGE * dcp)) + 1
    num_pages = max_seqs * cols + 2
    tables = rng.permutation(num_pages - 1)[: num_pages - 2]
    tables = tables.reshape(max_seqs, cols).astype(np.int32)
    if case == "bad_pages":
        # Wrapped (-1 -> the last page), dropped (too negative, past the end).
        tables[0, 0] = -1
        tables[1, :] = -num_pages - 2
        tables[2, :] = num_pages + 3
    return (
        torch.from_numpy(seq_lens),
        torch.from_numpy(tables.reshape(-1).copy()),
        torch.from_numpy(cu),
        num_tokens,
        num_pages,
    )


def _caches(case, dcp, cache_dtype, scale, *, fault=None, monkeypatch=None, seed=0):
    rng = np.random.default_rng(seed + dcp)
    w = _weights(seed)
    seq_lens, tables, qsl, num_tokens, num_pages = _metadata(case, dcp, rng)
    g = torch.Generator().manual_seed(seed + 7)
    aux = [
        torch.randn(num_tokens, HIDDEN, generator=g).to(torch.bfloat16)
        for _ in range(NUM_AUX)
    ]
    positions = torch.zeros(num_tokens, dtype=torch.int64)
    for s in range(seq_lens.shape[0]):
        lo, hi = int(qsl[s]), int(qsl[s + 1])
        positions[lo:hi] = torch.arange(int(seq_lens[s]) - (hi - lo), int(seq_lens[s]))
    md = SimpleNamespace(seq_lens=seq_lens, block_tables=tables, query_start_loc=qsl)
    metadata = (md,) * LAYERS
    G = max(1, dcp // REPLICAS)
    packing = 4 // torch.tensor([], dtype=cache_dtype).element_size()
    words = -(-2 * G // packing)
    raw = torch.from_numpy(
        rng.integers(0, 256, (num_pages, PAGE, words, packing, HEAD_DIM * cache_dtype.itemsize),
                     dtype=np.uint8)
    )
    init = raw.view(cache_dtype).view(num_pages, PAGE, words, packing, HEAD_DIM)

    group = _Group(dcp)
    old = [_rank_model(w, r, r, dcp, cache_dtype, scale, init) for r in range(dcp)]
    new = [_rank_model(w, r, r, dcp, cache_dtype, scale, init) for r in range(dcp)]
    _run_ranks(group, lambda r: _v29_store(old[r], aux, positions, metadata, group, dcp))

    if monkeypatch is not None:
        monkeypatch.setattr("vllm_torchtpu.distributed.dcp.get_dcp_group", lambda: group)
    if fault == "max_runs":
        monkeypatch.setattr(cda, "max_owned_runs", lambda *a: 1)

    def run_new(r):
        assert dspark.prepare_rank_rows_context_kv(new[r], group, dcp, REPLICAS)
        if fault == "head_order" and G > 1:
            for layer in new[r].layers:
                wkv = layer.self_attn.context_kv_weight
                k = wkv[:, : G * HEAD_DIM].view(HIDDEN, G, HEAD_DIM).flip(1)
                wkv[:, : G * HEAD_DIM] = k.reshape(HIDDEN, -1).clone()
        if fault == "wrong_rank":
            for layer in new[r].layers:
                layer.self_attn.attn.draft_dcp_rank = torch.tensor(
                    [(r + 1) % dcp], dtype=torch.int32
                )
        dspark.MiniMaxM3DSparkModel.store_context_kv(new[r], aux, positions, metadata)

    _run_ranks(group, run_new)
    pairs = []
    for r in range(dcp):
        for lo, ln in zip(old[r].layers, new[r].layers, strict=True):
            a = lo.self_attn.attn.kv_cache
            b = ln.self_attn.attn.kv_cache
            pairs.append((a, b))
    return pairs, init


def _raw(t):
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)


CASES = ["prefill_window", "mixed", "decode", "bad_pages"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dcp", [2, 4, 8])
@pytest.mark.parametrize(
    "cache_dtype,scale", [(torch.float8_e4m3fn, 0.5), (torch.bfloat16, None)], ids=["fp8", "bf16"]
)
def test_rank_rows_cache_bytes_match_v29(case, dcp, cache_dtype, scale, monkeypatch):
    if cache_dtype == torch.bfloat16 and dcp == 4:
        pytest.skip("bf16 covered at dcp 2 and 8")
    pairs, init = _caches(case, dcp, cache_dtype, scale, monkeypatch=monkeypatch)
    wrote = False
    for a, b in pairs:
        assert torch.equal(_raw(a), _raw(b))
        wrote |= not torch.equal(_raw(a), _raw(init))
    assert wrote  # the write did something


@pytest.mark.parametrize(
    "fault,case,dcp",
    [
        ("head_order", "mixed", 8),
        ("max_runs", "mixed", 2),
        ("wrong_rank", "decode", 4),
    ],
)
def test_parity_catches_faults(fault, case, dcp, monkeypatch):
    pairs, _ = _caches(case, dcp, torch.float8_e4m3fn, 0.5, fault=fault, monkeypatch=monkeypatch)
    assert any(not torch.equal(_raw(a), _raw(b)) for a, b in pairs)


@pytest.mark.parametrize("dcp", [1, 2, 8])
def test_owned_runs_within_bound(dcp):
    """`max_owned_runs` bounds the unclipped run count for random batches."""
    rng = np.random.default_rng(dcp)
    for _ in range(30):
        max_seqs = int(rng.integers(1, 40))
        q = rng.integers(0, 600, max_seqs) * (rng.random(max_seqs) < 0.8)
        ctx = rng.integers(0, 9000, max_seqs)
        cu = np.concatenate([[0], np.cumsum(q)])
        num_tokens = int(cu[-1]) + int(rng.integers(0, 9))
        if num_tokens == 0:
            continue
        cols = -(-int((ctx + q).max()) // (PAGE * dcp)) + 1
        tables = np.arange(max_seqs * cols, dtype=np.int32)
        for rank in range(dcp):
            n, *_ = cda.owned_runs(
                jnp.asarray(ctx + q, jnp.int32), jnp.asarray(tables), jnp.asarray(cu, jnp.int32),
                num_tokens=num_tokens, num_pages=max_seqs * cols, page_size=PAGE, dcp=dcp,
                rank=jnp.int32(rank), max_runs=num_tokens,
            )
            assert int(n[0]) <= cda.max_owned_runs(num_tokens, max_seqs, PAGE, dcp)


@pytest.mark.parametrize(
    "cache_shape,dtype,num_tokens,max_seqs",
    [
        ((16384, PAGE, 2, 4, HEAD_DIM), jnp.float8_e4m3fn, 8192, 64),  # G = 4 fp8, 8K chunk
        ((16384, PAGE, 2, 4, HEAD_DIM), jnp.float8_e4m3fn, 512, 64),  # c64 verify
        ((16384, PAGE, 1, 4, HEAD_DIM), jnp.float8_e4m3fn, 1003, 256),  # G = 1 (DCP 2)
        ((16384, PAGE, 4, 2, HEAD_DIM), jnp.bfloat16, 8192, 64),  # unquantized
    ],
)
def test_write_owned_rows_compiles_for_tpu7x_without_a_device(
    cache_shape, dtype, num_tokens, max_seqs
):
    """Mosaic compiles the DMA write (deviceless tpu7x AOT through libtpu)."""
    from jax.experimental import topologies
    from jax.sharding import AbstractDevice, AbstractMesh, SingleDeviceSharding, use_abstract_mesh

    try:
        topology = topologies.get_topology_desc("tpu7x:2x2x1", "tpu")
    except Exception as e:  # no libtpu
        pytest.skip(f"no deviceless tpu7x topology: {e}")
    sharding = SingleDeviceSharding(topology.devices[0])

    def spec(shape, dt):
        return jax.ShapeDtypeStruct(shape, dt, sharding=sharding)

    pages_per_seq = 64
    args = [
        spec(cache_shape, dtype),
        spec((num_tokens, *cache_shape[2:]), dtype),
        spec((max_seqs,), jnp.int32),
        spec((max_seqs * pages_per_seq,), jnp.int32),
        spec((max_seqs + 1,), jnp.int32),
        spec((1,), jnp.int32),
    ]

    def fn(cache, rows, seq_lens, tables, cu, rank):
        return cda.write_owned_rows(cache, rows, seq_lens, tables, cu, dcp=8, rank=rank[0])

    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(fn, donate_argnums=0).lower(*args)
    assert "dcp_draft_write_owned_rows" in lowered.as_text()
    lowered.compile()
