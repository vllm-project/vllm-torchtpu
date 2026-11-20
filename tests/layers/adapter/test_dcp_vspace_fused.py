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
"""TPU_DCP_MERGE_VSPACE_FUSED: the chunked merge with its operand
built in V space equals the latent merge followed by W_UV."""

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

D, T, LOCAL, C, V = 4, 16, 2, 32, 16
H = D * LOCAL


def _operand(o, lse_r, lses, w_uv):
    """Rank r's heads-major V-space operand [H, T, V] in f64."""
    merged = torch.logsumexp(lses.double(), dim=0)
    w = torch.exp(lse_r.double() - merged)  # [T, H]
    v = torch.einsum("thc,hcv->htv", o.double(), w_uv.double())
    wt = w.transpose(0, 1).unsqueeze(-1)
    return torch.where(wt > 0, v * wt, torch.zeros_like(v))


class _Group:
    """all_gather by row match; reduce_scatter checks this rank's operand and
    returns its head block of the sum over every rank's operand."""

    def __init__(self, outputs, lses, w_uv, rank):
        self.world_size, self.rank_in_group = D, rank
        self.outputs, self.lses, self.w_uv = outputs, lses, w_uv
        self.done = 0  # rows already reduce-scattered (chunks run in order)

    def all_gather(self, tensor, dim=0):
        n = tensor.shape[1]
        own = self.lses[self.rank_in_group]
        for off in range(0, T - n + 1):
            if torch.equal(own[off : off + n], tensor[0]):
                return self.lses[:, off : off + n].clone()
        raise AssertionError("lse rows not found")

    def reduce_scatter(self, tensor, dim=0):
        assert dim == 0
        n = tensor.shape[1]
        rows = slice(self.done, self.done + n)
        self.done += n
        lses = self.lses[:, rows]
        ops = [
            _operand(self.outputs[r][rows], self.lses[r][rows], lses, self.w_uv)
            for r in range(D)
        ]
        torch.testing.assert_close(
            tensor.double(), ops[self.rank_in_group], rtol=1e-5, atol=1e-5
        )
        total = sum(ops).to(tensor.dtype)
        return total[self.rank_in_group * LOCAL : (self.rank_in_group + 1) * LOCAL]


@pytest.mark.parametrize("lse_ag", [False, True])
def test_vspace_merge_is_the_latent_merge_then_w_uv(monkeypatch, lse_ag):
    rng = np.random.default_rng(7)
    outputs = torch.from_numpy(rng.standard_normal((D, T, H, C)).astype(np.float32))
    lses = torch.from_numpy((rng.standard_normal((D, T, H)) * 3).astype(np.float32))
    lses[1, 3] = float("-inf")
    w_uv = torch.from_numpy(rng.standard_normal((H, C, V)).astype(np.float32))
    for k, val in {
        "TPU_DCP_MERGE_VSPACE_FUSED": True,
        "TPU_DCP_LSE_MERGE_A2A": False,
        "TPU_DCP_MERGE_CHUNKED": 2,
        "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS": 0,
        "TPU_DCP_MERGE_CHUNKED_SMALL": 0,
        "TPU_DCP_MERGE_CHUNKED_LSE_AG": lse_ag,
        "TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS": 0,
        "TPU_DCP_MLA_MERGE_PIECES": 0,
        "TPU_DCP_MERGE_RS_BF16": 0,
        "TPU_DCP_RS_OUT_PRECISION": False,
    }.items():
        monkeypatch.setattr(cp_mla.envs, k, val, raising=False)
    merged = torch.logsumexp(lses.double(), dim=0)
    w = torch.exp(lses.double() - merged)  # [D, T, H]
    latent = torch.einsum("dth,dthc->thc", w.nan_to_num(0), outputs.double())
    want = torch.einsum("thc,hcv->thv", latent, w_uv.double())
    for rank in range(D):
        group = _Group(outputs, lses, w_uv, rank)
        monkeypatch.setattr(cp_mla, "get_dcp_group", lambda g=group: g)
        assert cp_mla.vspace_fused_applies(T, True)
        assert not cp_mla.vspace_fused_applies(T, False)
        got = cp_mla.merge_lse_partials_scatter_heads_vspace(
            outputs[rank], lses[rank], w_uv, None
        )
        assert got.shape == (T, LOCAL, V)
        torch.testing.assert_close(
            got.double(),
            want[:, rank * LOCAL : (rank + 1) * LOCAL],
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("scaled", [False, True])
def test_vspace_operand_kernel_matches_reference(scaled):
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.layers.adapter import dcp_vspace

    if jax.default_backend() != "tpu":
        pytest.skip("requires a TPU")
    t, h, c, v = 512, 32, 512, 256
    k = jax.random.split(jax.random.key(0), 4)
    o = jax.random.normal(k[0], (t, h, c)).astype(jnp.bfloat16)
    w = jax.nn.softmax(jax.random.normal(k[1], (t, h)) * 3, axis=-1)
    w = w.at[5].set(0.0)
    o = o.at[5].set(jnp.nan)  # zero-weight partial may be NaN
    w_uv = (jax.random.normal(k[2], (h, c, v)) * 0.05).astype(jnp.bfloat16)
    scale = (jax.random.uniform(k[3], (h, 1, v)) + 0.5) if scaled else None
    got = dcp_vspace.vspace_weighted_heads_major_jax(
        o, w, w_uv, scale, out_dtype=jnp.float32
    )
    of = jnp.nan_to_num(o.astype(jnp.float32))
    ref = jnp.einsum("thc,hcv->htv", of, w_uv.astype(jnp.float32))
    if scaled:
        ref = ref * scale
    ref = ref * jnp.transpose(w, (1, 0))[:, :, None]
    got, ref = np.asarray(got), np.asarray(ref)
    assert np.isfinite(got).all()
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-2, rel
    bf = np.asarray(
        dcp_vspace.vspace_weighted_heads_major_jax(
            o, w, w_uv, scale, out_dtype=jnp.bfloat16
        )
    ).astype(np.float32)
    np.testing.assert_allclose(bf, got, rtol=1e-2, atol=1e-3)


class _ReplayGroup:
    """Records each rank's all-gather shards on a first pass over the ranks,
    returns every rank's recorded shards on a second; reduce_scatter returns
    this rank's operand unreduced."""

    def __init__(self, rank, shards, replay):
        self.world_size, self.rank = D, rank
        self.shards, self.replay, self.calls = shards, replay, 0

    def all_gather(self, tensor, dim=0):
        call, self.calls = self.calls, self.calls + 1
        if not self.replay:
            self.shards.setdefault(call, {})[self.rank] = tensor.clone()
            return torch.cat([tensor] * D, dim=dim)
        return torch.cat([self.shards[call][r] for r in range(D)], dim=dim)

    def reduce_scatter(self, tensor, dim=0):
        return tensor


def _operands(monkeypatch, row_pairs, merge, rows, roll=False):
    """Every rank's V-space reduce-scatter operand and the gathered shards."""
    monkeypatch.setattr(
        cp_mla.envs, "TPU_DCP_MERGE_LSE_ROW_PAIRS", row_pairs, raising=False
    )
    g = torch.Generator().manual_seed(rows)
    lses = torch.randn(D, rows, H, generator=g) * 3 + 4
    lses[torch.rand(D, rows, H, generator=g) < 0.08] = float("-inf")
    outputs = torch.randn(D, rows, H, C, generator=g).to(torch.bfloat16)
    vspace = (torch.randn(H, C, V, generator=g).to(torch.bfloat16), None)
    shards: dict = {}
    for rank in range(D):
        merge(
            outputs[rank],
            lses[rank],
            _ReplayGroup(rank, shards, False),
            2,
            None,
            vspace=vspace,
        )
    if roll:
        # Positive control: every shard's rows moved by one.
        shards = {
            k: {r: torch.roll(x, 1, dims=1) for r, x in v.items()}
            for k, v in shards.items()
        }
    got = [
        merge(
            outputs[r], lses[r], _ReplayGroup(r, shards, True), 2, None, vspace=vspace
        )
        for r in range(D)
    ]
    return got, sorted({tuple(v[0].shape) for v in shards.values()})


@pytest.mark.parametrize(
    "path, pieces",
    [("per_chunk", 0), ("gathered_once", 0), ("gathered_once", 2)],
)
def test_row_paired_lse_gather_gives_the_row_major_operand_bit_for_bit(
    monkeypatch, path, pieces
):
    merge = {
        "per_chunk": cp_mla._merge_chunked,
        "gathered_once": cp_mla._merge_chunked_lse_gathered,
    }[path]
    for k, val in {
        "TPU_DCP_MERGE_CHUNKED_LSE_AG": False,
        "TPU_DCP_MLA_MERGE_PIECES": pieces,
        "TPU_DCP_MERGE_RS_BF16": 1,
    }.items():
        monkeypatch.setattr(cp_mla.envs, k, val, raising=False)
    rows = 32
    want, want_shards = _operands(monkeypatch, False, merge, rows)
    got, shards = _operands(monkeypatch, True, merge, rows)
    assert all(s[-1] == H for s in want_shards)
    assert all(s[-1] == 2 * H for s in shards)  # [1, rows / 2, 2H]: no padded lane
    for a, b in zip(want, got):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    rolled, _ = _operands(monkeypatch, True, merge, rows, roll=True)
    assert not torch.equal(want[0].float(), rolled[0].float())


def test_row_paired_lse_gather_needs_an_even_row_count():
    group = _ReplayGroup(0, {}, False)
    with pytest.raises(NotImplementedError, match="even row count"):
        cp_mla._gather_lse_row_pairs(torch.zeros(7, H), group)


class _TwoRankEchoGroup:
    def all_gather(self, x, dim=0):
        return torch.cat([x, x + 1], dim=dim)


def test_row_paired_lse_gather_compiles_with_a_symbolic_row_count():
    def gather(lse):
        return cp_mla._gather_lse_row_pairs(lse, _TwoRankEchoGroup())

    compiled = torch.compile(gather, fullgraph=True, dynamic=True, backend="eager")
    for rows in (8, 12):
        lse = torch.randn(rows, H)
        torch.testing.assert_close(compiled(lse), gather(lse))


def _built_lse_reduce_scatter(monkeypatch, op, cap):
    monkeypatch.setattr(cp_mla, "_LSE_RS_OPS", {False: op, True: op})
    monkeypatch.setattr(cp_mla, "_LSE_RS_WORLD", 16)
    monkeypatch.setattr(cp_mla, "_DCP_RANK", torch.zeros(1, dtype=torch.int32))
    monkeypatch.setattr(
        cp_mla.envs, "TPU_DCP_MERGE_LSE_RS_MAX_TOKENS", cap, raising=False
    )
    monkeypatch.setattr(cp_mla.envs, "TPU_DCP_MERGE_RS_BF16", 1, raising=False)


@pytest.mark.parametrize(
    "tokens, chunks, cap, applies",
    [
        (1024, 2, 2048, True),
        (2048, 2, 2048, True),
        (4096, 8, 2048, False),  # over the cap
        (4096, 8, 0, False),  # more attention chunks than the op takes
        (4096, 4, 0, True),
    ],
)
def test_lse_reduce_scatter_takes_buckets_up_to_the_cap(
    monkeypatch, tokens, chunks, cap, applies
):
    _built_lse_reduce_scatter(monkeypatch, None, cap)
    assert cp_mla.lse_rs_applies(tokens, chunks, 64, 512, 256, 2) is applies


@pytest.mark.parametrize("rs_bf16", [0, 2])
def test_lse_reduce_scatter_takes_only_the_bf16_reduce_scatter_merge(
    monkeypatch, rs_bf16
):
    """The kernel rounds every partial sum to bf16, as the merge of
    TPU_DCP_MERGE_RS_BF16=1 does; the f32 merge and the all-to-all merge keep
    their own arithmetic."""
    _built_lse_reduce_scatter(monkeypatch, None, 2048)
    monkeypatch.setattr(cp_mla.envs, "TPU_DCP_MERGE_RS_BF16", rs_bf16)
    assert not cp_mla.lse_rs_applies(2048, 2, 64, 512, 256, 2)


def test_lse_reduce_scatter_takes_only_bf16_partials(monkeypatch):
    """A float32 attention output keeps the chunked merge: the kernel reads 2
    bf16 heads from each 32-bit word."""
    _built_lse_reduce_scatter(monkeypatch, None, 2048)
    monkeypatch.setattr(cp_mla, "lse_rs_applies", lambda *a: True)
    monkeypatch.setattr(cp_mla, "get_dcp_group", lambda: None)
    monkeypatch.setattr(cp_mla, "_merge_chunks", lambda tokens: 2)
    monkeypatch.setattr(cp_mla, "_retain_rank", lambda x: x)
    taken = []
    monkeypatch.setattr(
        cp_mla, "_merge_lse_rs", lambda *a: taken.append("lse_rs") or "lse_rs"
    )
    monkeypatch.setattr(
        cp_mla, "_merge_chunked", lambda *a, **k: taken.append("chunked") or "chunked"
    )
    w_uv = torch.zeros(H, C, V, dtype=torch.bfloat16)
    lse = torch.zeros(T, H)
    for dtype in (torch.float32, torch.bfloat16):
        output = torch.zeros(T, H, C, dtype=dtype)
        cp_mla.merge_lse_partials_scatter_heads_vspace(output, lse, w_uv, None)
    assert taken == ["chunked", "lse_rs"]


def test_lse_reduce_scatter_needs_its_ops_built(monkeypatch):
    _built_lse_reduce_scatter(monkeypatch, None, 2048)
    monkeypatch.setattr(cp_mla, "_LSE_RS_OPS", {})
    assert not cp_mla.lse_rs_applies(2048, 2, 64, 512, 256, 2)


@pytest.mark.parametrize("scaled", [False, True])
def test_lse_reduce_scatter_gets_each_attention_chunk_as_an_operand(
    monkeypatch, scaled
):
    """The op takes the 2 attention chunks, the last repeated to fill its 4
    operands, W_UV (and its scale per head and V column), the f32
    log-sum-exps and the rank; its heads-major result comes back token-major."""
    calls = []
    heads_out = H // D

    def op(*args):
        calls.append(args)
        return (
            torch.arange(heads_out * T * V, dtype=torch.float32)
            .reshape(heads_out, T, V)
            .to(torch.bfloat16)
        )

    _built_lse_reduce_scatter(monkeypatch, op, 2048)
    rng = np.random.default_rng(3)
    output = torch.from_numpy(rng.standard_normal((T, H, C)).astype(np.float32))
    lse = torch.from_numpy(rng.standard_normal((T, H)).astype(np.float32))
    w_uv = torch.zeros(H, C, V)
    w_scale = torch.arange(H, dtype=torch.float32).reshape(H, 1) if scaled else None
    got = cp_mla._merge_lse_rs(output, lse, w_uv, w_scale, 2)

    (args,) = calls
    half = T // 2
    for part, rows in zip(args[:4], (0, half, half, half)):
        assert torch.equal(part, output[rows : rows + half])
    assert args[4] is w_uv
    if scaled:
        assert torch.equal(args[5], w_scale.expand(H, V))
    assert args[-2].dtype == torch.float32 and torch.equal(args[-2], lse)
    assert got.dtype == output.dtype
    assert torch.equal(got, op().transpose(0, 1).to(output.dtype))
