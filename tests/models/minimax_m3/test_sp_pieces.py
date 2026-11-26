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

import pytest
import torch

pytest.importorskip("torch_tpu")

from tests.models.minimax_m3 import test_sp_mode as base  # noqa: E402
from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.distributed import pallas_all_reduce  # noqa: E402
from vllm_torchtpu.layers.adapter import token_padding  # noqa: E402
from vllm_torchtpu.models.minimax_m3 import model as mm  # noqa: E402

BIG_T = 256  # TP 4: 64-row shards, two 32-row pieces

_PIECES_ENV = {
    "TPU_MINIMAX_M3_SP_FFN_PIECES": 1,
    "TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS": 4096,
    "TPU_MINIMAX_M3_SP_FFN_GATHER_PIECES": True,
    "TPU_MINIMAX_M3_SP_DENSE_FFN_PER_PIECE": False,
    "TPU_MINIMAX_M3_SP_E4M3_SHARED": False,
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
