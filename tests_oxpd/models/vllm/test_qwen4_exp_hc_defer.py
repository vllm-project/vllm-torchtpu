# SPDX-License-Identifier: Apache-2.0
"""TPU_QWEN4_HC_PREFILL_DEFER_COMBINE (P-DENSE): the stage loop of
Qwen4ExpBackbone._forward_impl hands each layer's mlp combine to the next layer
of the stage, never into the PLE layer and never across the stage boundary.
CPU, with recording stand-in layers whose combine/mix are exact arithmetic, so
the deferred and direct paths must agree exactly."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpBackbone


class _HC:
    """combine(moe, residual) = residual + moe; mix(h) = (2h, h); combine_mix as its docstring."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def mix(self, hidden):
        return hidden * 2, hidden

    def combine(self, moe, residual):
        return residual + moe

    def combine_mix(self, moe, residual, next_hc):
        self.log.append(("combine_mix", self.name, next_hc.name))
        return next_hc.mix(self.combine(moe, residual))


class _Layer:
    def __init__(self, log, i, ple=False):
        self.log, self.i = log, i
        self.ple = object() if ple else None
        self.attn_hyper_connection = _HC(log, f"attn{i}")
        self.mlp_hyper_connection = _HC(log, f"mlp{i}")

    def __call__(self, hidden, *args, sequence_parallel=False, sp_pad=0, pending=None, defer=False, **_):
        self.log.append(("layer", self.i, pending is not None, defer))
        if pending is not None:
            assert hidden is None and self.ple is None
            moe_prev, residual_prev, hc_prev = pending
            block, residual = hc_prev.combine_mix(moe_prev, residual_prev, self.attn_hyper_connection)
        else:
            if self.ple is not None:
                hidden = hidden + 0.5   # the PLE addend needs the combined streams
            block, residual = self.attn_hyper_connection.mix(hidden)
        moe = block * (self.i + 1) + 1.0   # the layer's block work
        if defer:
            return moe, residual, self.mlp_hyper_connection
        return self.mlp_hyper_connection.combine(moe, residual)


def _run(monkeypatch, defer_flag, stage, ple_at, sequence_parallel=False):
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_DEFER_COMBINE", defer_flag)
    log = []
    layers = {i: _Layer(log, i, ple=(i == ple_at)) for i in stage}
    fake = SimpleNamespace(
        host_ple_index=None, stage=range(stage[0], stage[-1] + 1), first_stage=False, last_stage=False,
        layers=layers, hc_count=4, sp_pad_rows=0, embed_tokens=None, hyper_connection_mixer=None,
        emit_mtp_hidden=False,
    )
    hidden = torch.arange(8, dtype=torch.float64).reshape(2, 4)
    out = Qwen4ExpBackbone._forward_impl(
        fake, torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64), None,
        sequence_parallel=sequence_parallel, intermediate_tensors={"hidden_states": hidden},
        query_start_loc=None, ngram_context=None, state_slots=None, context_lens=None,
        valid_tokens=None, main_block_table=None, compressed_block_table=None,
    )
    return out["hidden_states"], log


@pytest.mark.parametrize("stage,ple_at", [([0, 1, 2, 3, 4, 5], 1), ([6, 7, 8, 9, 10, 11], None)])
def test_defer_matches_direct_and_skips_ple_and_stage_end(monkeypatch, stage, ple_at):
    direct, log0 = _run(monkeypatch, False, stage, ple_at)
    deferred, log1 = _run(monkeypatch, True, stage, ple_at)
    assert torch.equal(direct, deferred)
    assert not any(e[0] == "combine_mix" for e in log0)
    calls = [e for e in log1 if e[0] == "layer"]
    for k, (_, i, took_pending, deferred_out) in enumerate(calls):
        nxt = stage[k + 1] if k + 1 < len(stage) else None
        assert deferred_out == (nxt is not None and nxt != ple_at), (i, deferred_out)
        assert took_pending == (k > 0 and calls[k - 1][3]), (i, took_pending)
    # Every deferred hand-off ran as the previous layer's mlp HC into this layer's attn HC.
    hands = [e for e in log1 if e[0] == "combine_mix"]
    expected = [("combine_mix", f"mlp{stage[k]}", f"attn{stage[k + 1]}")
                for k in range(len(stage) - 1) if stage[k + 1] != ple_at]
    assert hands == expected


class _TiledHC(_HC):
    """_HC with a prefill tile, recording combine_mix's reread flag."""

    prefill_full_tile = 4

    def combine_mix(self, moe, residual, next_hc, reread=False):
        self.log.append(("combine_mix", self.name, next_hc.name, reread))
        return next_hc.mix(self.combine(moe, residual))


class _TiledLayer(_Layer):
    def __init__(self, log, i, ple=False):
        super().__init__(log, i, ple)
        self.attn_hyper_connection = _TiledHC(log, f"attn{i}")
        self.mlp_hyper_connection = _TiledHC(log, f"mlp{i}")

    def __call__(self, hidden, *args, sequence_parallel=False, sp_pad=0, pending=None, defer=False, **_):
        self.log.append(("layer", self.i, pending is not None, defer))
        if pending is not None:
            moe_prev, residual_prev, hc_prev = pending
            block, residual = hc_prev.combine_mix(moe_prev, residual_prev, self.attn_hyper_connection,
                                                  reread=bool(envs.TPU_QWEN4_HC_PREFILL_DEFER_WHOLE))
        else:
            block, residual = self.attn_hyper_connection.mix(hidden)
        moe = block * (self.i + 1) + 1.0
        if defer:
            return moe, residual, self.mlp_hyper_connection
        return self.mlp_hyper_connection.combine(moe, residual)


def _run_whole(monkeypatch, rows, whole=True, combine=False):
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_DEFER_COMBINE", combine)
    monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_DEFER_WHOLE", whole)
    log = []
    stage = [6, 7, 8, 9]
    layers = {i: _TiledLayer(log, i) for i in stage}
    fake = SimpleNamespace(
        host_ple_index=None, stage=range(stage[0], stage[-1] + 1), first_stage=False, last_stage=False,
        layers=layers, hc_count=4, sp_pad_rows=0, embed_tokens=None, hyper_connection_mixer=None,
        emit_mtp_hidden=False,
    )
    hidden = torch.arange(rows * 4, dtype=torch.float64).reshape(rows, 4)
    out = Qwen4ExpBackbone._forward_impl(
        fake, torch.zeros(rows, dtype=torch.int64), torch.zeros(rows, dtype=torch.int64), None,
        sequence_parallel=False, intermediate_tensors={"hidden_states": hidden},
        query_start_loc=None, ngram_context=None, state_slots=None, context_lens=None,
        valid_tokens=None, main_block_table=None, compressed_block_table=None,
    )
    return out["hidden_states"], log


@pytest.mark.parametrize("rows,defers", [(8, True), (16, True), (12, True), (10, False), (6, False), (4, False)])
def test_defer_whole_only_on_whole_tile_steps_with_reread(monkeypatch, rows, defers):
    """TPU_QWEN4_HC_PREFILL_DEFER_WHOLE: hand-offs only when rows are >= 2 whole tiles (tile 4 here); each
    hand-off asks for the reread prologue; other steps run the direct path; the result equals direct."""
    direct, _ = _run_whole(monkeypatch, rows, whole=False)
    out, log = _run_whole(monkeypatch, rows, whole=True)
    assert torch.equal(direct, out)
    hands = [e for e in log if e[0] == "combine_mix"]
    if defers:
        assert hands == [("combine_mix", f"mlp{i}", f"attn{i + 1}", True) for i in (6, 7, 8)]
    else:
        assert hands == [], hands


def test_defer_combine_flag_alone_keeps_every_step_and_no_reread(monkeypatch):
    """The older TPU_QWEN4_HC_PREFILL_DEFER_COMBINE still defers on any step, without reread."""
    _, log = _run_whole(monkeypatch, 10, whole=False, combine=True)
    assert [e for e in log if e[0] == "combine_mix"] == [("combine_mix", f"mlp{i}", f"attn{i + 1}", False)
                                                          for i in (6, 7, 8)]


def test_combine_mix_picks_the_reread_op(monkeypatch):
    """GatedResidual.combine_mix(reread=True) runs next_hc's reread prologue op on the fused prefill path."""
    from vllm_torchtpu.models.vllm.qwen4_exp import hyperconnection as H

    calls = []
    next_hc = SimpleNamespace(
        _decode_combine_mix_op=None,
        _prefill_combine_mix_op=lambda *a: calls.append("plain") or (a[1], a[2], a[0]),
        _prefill_combine_mix_rr_op=lambda *a: calls.append("rr") or (a[1], a[2], a[0]),
        _prefill_fused_mix=lambda x: True, _decode_weights=lambda: (),
        _hcx_xla=lambda x: False,  # q38 L3-HCX: HCX (VLLM_TPU_HC_XLA_MAX_ROWS) off
    )
    me = SimpleNamespace(hc_count=4)
    residual = torch.zeros(8, 16, dtype=torch.bfloat16)
    gate = torch.zeros(8, 4, dtype=torch.float32)
    block = torch.zeros(8, 4, dtype=torch.bfloat16)
    H.GatedResidual.combine_mix(me, block, (residual, gate), next_hc)
    H.GatedResidual.combine_mix(me, block, (residual, gate), next_hc, reread=True)
    next_hc._prefill_combine_mix_rr_op = None
    H.GatedResidual.combine_mix(me, block, (residual, gate), next_hc, reread=True)
    assert calls == ["plain", "rr", "plain"]


class _Stop(Exception):
    pass


@pytest.mark.parametrize("flag", [False, True])
def test_decoder_layer_passes_reread_on_the_deferred_boundary(monkeypatch, flag):
    """Qwen4ExpDecoderLayer.forward's pending branch asks combine_mix for the reread prologue exactly when
    TPU_QWEN4_HC_PREFILL_DEFER_WHOLE is set."""
    from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpDecoderLayer

    monkeypatch.setattr(envs, "TPU_QWEN4_HC_PREFILL_DEFER_WHOLE", flag)
    seen = []

    class _Rec:
        def combine_mix(self, moe, residual, next_hc, reread=False):
            seen.append(reread)
            raise _Stop

    me = SimpleNamespace(ple=None, attn_hyper_connection=object())
    with pytest.raises(_Stop):
        Qwen4ExpDecoderLayer.forward(me, None, None, torch.zeros(4, dtype=torch.int64), None, None, None, None,
                                     None, None, None, pending=(None, None, _Rec()))
    assert seen == [flag]
