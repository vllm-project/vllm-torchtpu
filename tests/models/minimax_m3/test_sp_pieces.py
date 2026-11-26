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
"""MiniMax-M3 SP row pieces (TPU_MINIMAX_M3_SP_FFN_PIECES and friends), on CPU.

Runs the threaded TP harness of test_sp_mode.py on a step large enough to
split each rank's shard into whole 32-row pieces. Pieces only regroup
row-local work (norm, e4m3 row quantization, gathers, projections, routing)
and split the reduce-scatters by rows, so every variant must give the
one-piece result bit for bit.
"""

import contextlib
import threading
import types

import pytest
import torch

pytest.importorskip("torch_tpu")

from tests.models.minimax_m3 import test_sp_mode as base  # noqa: E402
from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.distributed import pallas_all_reduce  # noqa: E402
from vllm_torchtpu.layers.adapter import token_padding  # noqa: E402
from vllm_torchtpu.models.minimax_m3 import attention as attn_mod  # noqa: E402
from vllm_torchtpu.models.minimax_m3 import model as mm  # noqa: E402

BIG_T = 256  # TP 4: 64-row shards, two 32-row pieces

_PIECES_ENV = {
    "TPU_MINIMAX_M3_SP_FFN_PIECES": 1,
    "TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS": 4096,
    "TPU_MINIMAX_M3_SP_FFN_GATHER_PIECES": True,
    "TPU_MINIMAX_M3_SP_DENSE_FFN_PER_PIECE": False,
    "TPU_MINIMAX_M3_SP_E4M3_SHARED": False,
    "TPU_MINIMAX_M3_SP_ATTN_PIECES": False,
}


@pytest.fixture(name="tp")
def tp_fixture(monkeypatch):
    return base._install_tp(monkeypatch)


@pytest.fixture(autouse=True)
def big_step(monkeypatch):
    monkeypatch.setattr(base, "T", BIG_T)
    # The rank threads share torch's intra-op pool; a CPU matmul split over a
    # varying thread count can round differently from run to run, which a
    # bit-exact comparison would read as a pieces difference.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _env(monkeypatch, **kw):
    base._set_env(monkeypatch, **kw)
    for key, default in _PIECES_ENV.items():
        monkeypatch.setattr(envs, key, kw.get(key, default), raising=False)


def _run(tp, monkeypatch, **kw):
    kw = {"TPU_MINIMAX_M3_SP_MIN_TOKENS": BIG_T, **kw}
    _env(monkeypatch, **kw)
    tp.calls.clear()
    outs, models = base._forward(
        tp, monkeypatch, kw.get("TPU_MINIMAX_M3_SP_MODE", "full"), aux=(0, 2, 4)
    )
    return outs, list(tp.calls), models


def _pieces(n=2, **kw):
    return dict(
        TPU_MINIMAX_M3_SP_FFN_PIECES=n,
        TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS=BIG_T,
        **kw,
    )


def _assert_same(outs, want):
    for got, ref in zip(outs, want):
        torch.testing.assert_close(got[0], ref[0], rtol=0, atol=0)
        for a, b in zip(got[1], ref[1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


_CONFIGS = {
    "full": dict(TPU_MINIMAX_M3_SP_MODE="full"),
    "full-fp8": dict(
        TPU_MINIMAX_M3_SP_MODE="full",
        TPU_MINIMAX_M3_SP_FP8_GATHER=True,
        TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER=True,
    ),
    "target": base._TARGET,
}


# ------------------------------------------------------------ FFN pieces
@pytest.mark.parametrize("dense_per_piece", [False, True])
@pytest.mark.parametrize("name", list(_CONFIGS))
def test_ffn_pieces_bit_exact(tp, monkeypatch, name, dense_per_piece):
    """The FFN input gathered in two pieces and its output reduce-scattered
    per piece: same bits as one piece; one more reduce-scatter per layer."""
    cfg = dict(_CONFIGS[name])
    want, calls1, _ = _run(tp, monkeypatch, **cfg)
    outs, calls2, models = _run(
        tp,
        monkeypatch,
        **cfg,
        **_pieces(TPU_MINIMAX_M3_SP_DENSE_FFN_PER_PIECE=dense_per_piece),
    )
    _assert_same(outs, want)
    base._flags_restored(models)
    assert calls2.count("rs") == calls1.count("rs") + 4
    assert "ar" not in calls2


def test_ffn_pieces_hand_the_ffn_a_list(tp, monkeypatch):
    """The MoE and MLP get the gathered pieces as a list of e4m3 rows."""
    seen = []
    forward = base.MiniMaxM3MoE.forward

    def spy(self, x):
        seen.append(x)
        return forward(self, x)

    monkeypatch.setattr(base.MiniMaxM3MoE, "forward", spy)
    _run(tp, monkeypatch, **base._TARGET, **_pieces())
    assert seen and all(isinstance(x, list) and len(x) == 2 for x in seen)
    assert all(isinstance(p, tuple) for x in seen for p in x)


def test_ffn_pieces_off_below_min_tokens(tp, monkeypatch):
    """Below FFN_PIECES_MIN_TOKENS (or with shards not in whole 32-row
    pieces) the step makes exactly the one-piece calls."""
    _, calls1, _ = _run(tp, monkeypatch, **base._TARGET)
    _, calls2, _ = _run(
        tp,
        monkeypatch,
        **base._TARGET,
        TPU_MINIMAX_M3_SP_FFN_PIECES=2,
        TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS=BIG_T + 1,
    )
    assert calls2 == calls1
    _, calls3, _ = _run(tp, monkeypatch, **base._TARGET, **_pieces(4))  # 16-row pieces
    assert calls3 == calls1


def test_ffn_gather_pieces_off_keeps_one_ffn_gather(tp, monkeypatch):
    _, calls1, _ = _run(tp, monkeypatch, **base._TARGET)
    _, calls2, _ = _run(
        tp,
        monkeypatch,
        **base._TARGET,
        **_pieces(TPU_MINIMAX_M3_SP_FFN_GATHER_PIECES=False),
    )
    assert calls2 == calls1


def test_ffn_pieces_gate():
    f = mm._ffn_pieces
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(envs, "TPU_MINIMAX_M3_SP_FFN_PIECES", 2, raising=False)
        mp.setattr(envs, "TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS", 4096, raising=False)
        assert f(512, 8) == 2 and f(2048, 8) == 2
        assert f(256, 8) == 1  # 2048 tokens: below the minimum
        assert f(4096 // 8 + 32, 8) == 1  # 288-row shards: not whole 32-row pieces
        mp.setattr(envs, "TPU_MINIMAX_M3_SP_FFN_PIECES", 1, raising=False)
        assert f(2048, 8) == 1


def test_pieces_env_defaults(monkeypatch):
    for key, default in _PIECES_ENV.items():
        monkeypatch.delenv(key, raising=False)
        assert envs.environment_variables[key]() == default, key


@pytest.mark.parametrize("op", ["qrs", "hier"])
def test_ffn_pieces_reduce_scatter_through_sp_rows(tp, monkeypatch, op):
    """Every per-piece reduce-scatter goes through `_reduce_scatter_rows`
    and so through the armed QRS / HIER_RS op."""
    key = "TPU_MINIMAX_M3_SP_QRS" if op == "qrs" else "TPU_MINIMAX_M3_SP_HIER_RS"
    emu = base._qrs_emulation(tp) if op == "qrs" else base._exact_rs(tp)
    monkeypatch.setattr(mm, "get_tp_group", tp.group)
    monkeypatch.setattr(pallas_all_reduce, "reduce_scatter", emu)
    _, calls, _ = _run(tp, monkeypatch, **base._TARGET, **{key: True}, **_pieces())
    assert "rs" not in calls
    assert calls.count(op) == 8  # 4 layers x 2 FFN pieces


def test_piece_major_rows():
    x = torch.arange(32)
    got = mm._piece_major_rows(16, 2, 2, x)
    # rank 0 shard: rows 0..7 (pieces 0..3, 4..7); rank 1: rows 8..15.
    want = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
    torch.testing.assert_close(got, torch.tensor(want))


class _MaskedExperts(base._Experts):
    """Experts that zero the rows the token-padding mask marks (as the
    routed weights of padded tokens are zeroed)."""

    def __call__(self, hidden_states, router_logits):
        out = super().__call__(hidden_states, router_logits)
        if token_padding._padding_state is None:
            return out
        mask = token_padding._padding_state.get_local_padding_mask(out.shape[0])
        return torch.where(mask[:, None], torch.zeros_like(out), out)


def test_ffn_pieces_padding_mask_follows_rows(tp, monkeypatch):
    """The MoE sees its rows piece-major; the padding mask it reads must be
    in the same order (token_padding.padding_of_rows), or a valid row would
    be zeroed: same bits as one piece with a padded tail."""
    monkeypatch.setattr(base, "_Experts", _MaskedExperts)
    # The rank threads share the module's mask state: one rank at a time
    # inside padding_of_rows (the FFN makes no collective under it).
    real, lock = token_padding.padding_of_rows, threading.Lock()

    @contextlib.contextmanager
    def one_rank_at_a_time(select):
        with lock, real(select):
            yield

    monkeypatch.setattr(token_padding, "padding_of_rows", one_rank_at_a_time)
    state = token_padding.TokenPaddingState.create(BIG_T, "cpu")
    state.update(BIG_T - 40, BIG_T)
    monkeypatch.setattr(token_padding, "_padding_state", state)
    want, _, _ = _run(tp, monkeypatch, **base._TARGET)
    # (The rank threads share the module's state; reset it between runs.)
    monkeypatch.setattr(token_padding, "_padding_state", state)
    outs, _, _ = _run(tp, monkeypatch, **base._TARGET, **_pieces())
    _assert_same(outs, want)
    monkeypatch.setattr(token_padding, "_padding_state", None)
    unmasked, _, _ = _run(tp, monkeypatch, **base._TARGET)
    assert not torch.equal(unmasked[0][0], want[0][0])


def test_padding_of_rows_restores_state(monkeypatch):
    state = token_padding.TokenPaddingState.create(8, "cpu")
    state.update(5, 8)
    monkeypatch.setattr(token_padding, "_padding_state", state)
    with token_padding.padding_of_rows(lambda m: m[:8].flip(0)):
        mask = token_padding._padding_state.get_local_padding_mask(8)
        assert mask.tolist() == [True] * 3 + [False] * 5
    assert token_padding._padding_state is state
    monkeypatch.setattr(token_padding, "_padding_state", None)
    with token_padding.padding_of_rows(lambda m: m):
        assert token_padding._padding_state is None


def _jax_e4m3_matmul(monkeypatch):
    quant = base.quant
    monkeypatch.setattr(
        quant, "_e4m3_matmul", lambda *a: base._jax_call(quant._e4m3_rows_matmul, *a)
    )


def test_moe_list_input_matches_block(monkeypatch):
    """MiniMaxM3MoE given gathered pieces equals it given their rows in one
    block, with the e4m3 shared expert and with the shared expert in the
    combine."""
    _jax_e4m3_matmul(monkeypatch)
    x = base._input()
    q8, scale = mm._quantize_rows(x)
    pieces = [(q8[:128], scale[:128]), (q8[128:], scale[128:])]
    for e4m3_shared in (False, True):
        monkeypatch.setattr(
            envs, "TPU_MINIMAX_M3_SP_E4M3_SHARED", e4m3_shared, raising=False
        )
        moe, mlp = base._shared_moe()
        want = moe((q8, scale))
        got = moe(list(pieces))
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert "_e4m3_rows" not in mlp.__dict__
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_SP_E4M3_SHARED", True, raising=False)
    moe, runner = base._moe_with_shared(monkeypatch, True)
    want = moe((q8, scale))
    got = moe(list(pieces))
    assert moe.experts.routed_experts.quant_method.calls == 2
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_mlp_list_input_matches_block(monkeypatch):
    _jax_e4m3_matmul(monkeypatch)
    t = base._TP(1)
    t.local.rank = 0
    mlp = base._mlp(("l",), 0, t)
    mlp.down_proj.reduce_results = False
    x = base._input()
    q8, scale = mm._quantize_rows(x)
    pieces = [(q8[:64], scale[:64]), (q8[64:], scale[64:])]
    torch.testing.assert_close(mlp(pieces), mlp((q8, scale)), rtol=0, atol=0)
    torch.testing.assert_close(mlp([x[:64], x[64:]]), mlp(x), rtol=0, atol=0)


# ------------------------------------------------------ attention pieces
class _PieceAttn(base._Attn):
    """The stand-in attention, also given gathered row pieces: their rows
    back in step order (attention.py `_step_rows`), as qkv_proj's output is."""

    def __call__(self, positions, x, index_reuse=None, project=True):
        if not isinstance(x, list):
            return super().__call__(positions, x, index_reuse, project)
        deq = [(p[0].float() * p[1]).to(torch.bfloat16) for p in x]
        out = super().__call__(
            positions, attn_mod._step_rows(deq), index_reuse, project
        )
        self.calls[-1] = ("list", index_reuse, project)
        return out


@pytest.fixture
def piece_attn(monkeypatch):
    monkeypatch.setattr(base, "_Attn", _PieceAttn)
    monkeypatch.setattr(
        attn_mod, "get_tensor_model_parallel_world_size", lambda: base.TP
    )


_ATTN_CONFIGS = {
    # The r15-merge job's SP flags: e4m3 gathers, fused norm, e4m3 qkv,
    # e4m3 all-to-all o_proj, FFN and attention pieces.
    "target": base._TARGET,
    "rs-o_proj": dict(base._TARGET, TPU_MINIMAX_M3_SP_A2A_OPROJ="off"),
}


@pytest.mark.parametrize("name", list(_ATTN_CONFIGS))
def test_attn_pieces_bit_exact(tp, monkeypatch, piece_attn, name):
    """The attention input gathered in pieces (qkv per piece) and o_proj's
    e4m3 all-to-all + projection per piece: same bits as one piece."""
    cfg = _ATTN_CONFIGS[name]
    want, calls1, _ = _run(tp, monkeypatch, **cfg)
    outs, calls2, models = _run(
        tp, monkeypatch, **cfg, **_pieces(TPU_MINIMAX_M3_SP_ATTN_PIECES=True)
    )
    _assert_same(outs, want)
    base._flags_restored(models)
    # Layers 1-3 get their attention input as pieces (layer 0 attends to
    # the embedding, which is not gathered).
    kinds = [
        c[0] for m in models[:1] for layer in m.layers for c in layer.self_attn.calls
    ]
    assert kinds == ["Tensor", "list", "list", "list"]
    a2a = [c for c in calls2 if c.startswith("a2a")]
    if cfg["TPU_MINIMAX_M3_SP_A2A_OPROJ"] != "off":
        # q and scales per piece on layers 1-3, once on layer 0.
        assert len(a2a) == 2 + 3 * 2 * 2
    else:
        assert not a2a
    # Gathers: one per piece on both sides of layers 1-3.
    assert calls2.count("ag:torch.float8_e4m3fn") > calls1.count(
        "ag:torch.float8_e4m3fn"
    )


def test_attn_pieces_without_ffn_gather_pieces(tp, monkeypatch, piece_attn):
    want, _, _ = _run(tp, monkeypatch, **base._TARGET)
    outs, _, _ = _run(
        tp,
        monkeypatch,
        **base._TARGET,
        **_pieces(
            TPU_MINIMAX_M3_SP_ATTN_PIECES=True,
            TPU_MINIMAX_M3_SP_FFN_GATHER_PIECES=False,
        ),
    )
    _assert_same(outs, want)


def test_attention_projects_each_piece(monkeypatch):
    """The real attention forward given gathered pieces runs qkv_proj on each
    piece and hands the step-ordered rows on."""
    seen = []

    def stop(qkv, *a):
        seen.append(qkv)
        raise base._Stop

    monkeypatch.setattr(attn_mod, "norm_rope_heads", stop)
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 2)
    _jax_e4m3_matmul(monkeypatch)
    attn = object.__new__(attn_mod.MiniMaxM3SparseAttention)
    torch.nn.Module.__init__(attn)
    attn.qkv_proj = base._Linear(base.H, 96, ("qkv",))
    for name in ("num_heads", "num_kv_heads", "head_dim", "rotary_emb", "norm_rope_op"):
        object.__setattr__(attn, name, None)
    for name in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
        object.__setattr__(attn, name, None)
    x = base._input()[:64]
    q8, scale = mm._quantize_rows(x)
    # TP 2, two pieces of 16 rows: piece i = (rank 0 block i, rank 1 block i).
    blocks = [(q8[i : i + 16], scale[i : i + 16]) for i in range(0, 64, 16)]
    pieces = [
        tuple(torch.cat([blocks[r * 2 + i][j] for r in range(2)]) for j in range(2))
        for i in range(2)
    ]
    for hidden in ((q8, scale), pieces):
        with pytest.raises(base._Stop):
            attn.forward(torch.arange(64), hidden)
    torch.testing.assert_close(seen[1], seen[0], rtol=0, atol=0)


# ------------------------------------------------------ trace signatures
class _IdentityTP(base._TP):
    """TP 1 without threads, so Dynamo can trace the collectives."""

    def __init__(self):
        super().__init__(1)
        self.local.rank = 0

    def _exchange(self, x):
        return [x]

    def group(self):
        return _GROUP


def _quantize_pow2(x):
    amax = x.float().abs().amax(-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(amax.clamp(min=1e-30) / 448.0)))
    return (x.float() / scale).to(base.E4M3), scale


def _o_proj_e4m3(q, r, w, s):
    top = r.amax(0)
    q = (q.float() * (r / top)).to(base.E4M3)
    acc = torch.einsum("eti,ein->tn", q.float(), w.float())
    return (acc * top * s.float()).to(torch.bfloat16)


def _e4m3_matmul(q, rs, w, ws):
    return ((q.float() @ w.float()) * rs * ws.float()).to(torch.bfloat16)


def _add_norm_quant(shard, update, weight):
    x = shard + update
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + base.EPS) * weight
    q, scale = mm._quantize_rows(normed.to(x.dtype))
    return x, q, scale


class _MoeConfig:
    skip_final_all_reduce = False


_RANK = torch.zeros((1, 1), dtype=torch.int32)
_GROUP = types.SimpleNamespace(world_size=1, rank_in_group=0, ranks=[0])


def _signatures(monkeypatch, sizes, **kw):
    """graph_signature of the whole model's Dynamo trace (no split graph on
    TPU: one subgraph whose placeholders are the inputs and every parameter /
    buffer, in first-use order) at each token count, traced dynamic as the
    warmup traces each bucket group."""
    from vllm_torchtpu.compilation import shape_variants

    _env(monkeypatch, **kw)
    g = _IdentityTP()
    for name in (
        "tensor_model_parallel_reduce_scatter",
        "tensor_model_parallel_all_gather",
        "tensor_model_parallel_all_reduce",
    ):
        monkeypatch.setattr(mm, name, getattr(g, name.split("model_parallel_")[1]))
    monkeypatch.setattr(
        base.sp_o_proj, "tensor_model_parallel_all_gather", g.all_gather
    )
    monkeypatch.setattr(base.sp_o_proj, "all_to_all_rows", g.all_to_all_rows)
    monkeypatch.setattr(base.sp_o_proj, "get_tp_group", g.group)
    monkeypatch.setattr(mm, "get_tp_group", g.group)
    monkeypatch.setattr(attn_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pallas_all_reduce, "armed_rank", lambda ranks: _RANK)
    monkeypatch.setattr(base.sp_o_proj, "_quantize_op", _quantize_pow2)
    monkeypatch.setattr(base.sp_o_proj, "_o_proj_e4m3_op", _o_proj_e4m3)
    monkeypatch.setattr(base.quant, "_e4m3_matmul", _e4m3_matmul)
    monkeypatch.setattr(base, "_fused_norm_op", _add_norm_quant)
    monkeypatch.setattr(base, "_Experts", _MaskedExperts)
    monkeypatch.setattr(base, "_Attn", _PieceAttn)
    monkeypatch.setattr(base, "TP", 1)
    min_tokens = envs.TPU_MINIMAX_M3_SP_MIN_TOKENS
    monkeypatch.setattr(
        mm, "_sp_step", lambda n, size: n >= min_tokens and n % size == 0
    )
    state = token_padding.TokenPaddingState.create(max(sizes), "cpu")
    monkeypatch.setattr(token_padding, "_padding_state", state)
    model = base._model(0, g, "full" if kw.get("TPU_MINIMAX_M3_SP_MODE") else "off")
    model.tp_size = 1
    for layer in model.layers:
        layer.tp_size = 1
        if layer.is_moe_layer:
            layer.block_sparse_moe.experts.moe_config = _MoeConfig()
    out = {}
    for t in sizes:
        graphs = []

        def backend(gm, example_inputs, graphs=graphs):
            graphs.append((shape_variants.graph_signature(gm), len(gm.graph.nodes)))
            raise base._Stop  # traced; running needs the TPU ops

        torch._dynamo.reset()
        x = torch.randn(t, base.H).to(torch.bfloat16)
        pos = torch.arange(t)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(pos, 0)
        fn = torch.compile(
            lambda p, e: mm.MiniMaxM3Model.forward(model, None, p, inputs_embeds=e),
            backend=backend,
            fullgraph=True,
        )
        with pytest.raises(Exception) as err:
            fn(pos, x)
        assert graphs, err.value
        assert len(graphs) == 1
        out[t] = graphs[0][0]
        NODES[t] = graphs[0][1]
    torch._dynamo.reset()
    return out


# Node count of the last _signatures trace per token count.
NODES: dict[int, int] = {}

_SMALL = [16, 32, 64, 128, 256, 512]
_SP_ONLY = [1024, 2048]
_LARGE = [8192, 16384]

_JOB = dict(
    base._TARGET,
    TPU_MINIMAX_M3_SP_MIN_TOKENS=1024,
    TPU_MINIMAX_M3_SP_FFN_PIECES=2,
    TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS=4096,
    TPU_MINIMAX_M3_SP_ATTN_PIECES=True,
)


def test_piece_traces_keep_the_signature(monkeypatch):
    """The r15-merge job's flags (SP from 1024 tokens, FFN + attention pieces
    from 4096). shape_variants hands a bucket between traces only when they
    take the same inputs in the same order. The SP traces without pieces
    (1024, 2048) and with pieces (8192, 16384) must agree, and every trace
    must be exactly the one with the pieces flags off: the small plain
    buckets (16..512) trace unchanged. (Plain vs SP is not compared here:
    the stand-ins read the armed rank only on the SP path, unlike serving.)"""
    sizes = _SMALL + _SP_ONLY + _LARGE
    on = _signatures(monkeypatch, sizes, **_JOB)
    nodes_on = dict(NODES)
    off = _signatures(
        monkeypatch,
        sizes,
        **dict(
            _JOB, TPU_MINIMAX_M3_SP_FFN_PIECES=1, TPU_MINIMAX_M3_SP_ATTN_PIECES=False
        ),
    )
    # The pieces engage only from 4096 tokens: smaller traces are identical.
    assert all(nodes_on[t] == NODES[t] for t in _SMALL + _SP_ONLY)
    assert all(nodes_on[t] > NODES[t] for t in _LARGE)
    assert on == off
    assert len({on[t] for t in _SMALL}) == 1
    assert len({on[t] for t in _SP_ONLY + _LARGE}) == 1
