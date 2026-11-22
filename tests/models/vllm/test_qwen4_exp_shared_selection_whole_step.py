# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_DRAFT_SHARED_SELECTION under the whole-step megakernel (CPU).

The draft attends the target's last-QSA-layer selection: the target writes it to
`_selection_sink` in `attend`, stock draft step 0 copies each request's last verified
row into `_selection_rows`, and loop steps 1..K-1 attend `_selection_rows[:p]`.

With VLLM_TPU_MK_WHOLE_L47 / _L45 the whole-step kernel runs model layer 47 (the
sink layer) and skips its stock attention core, so nothing writes the sink on a
whole-step step. With VLLM_TPU_MK_DRAFT, `_MkDraftStep0` replaces stock step 0 and
does not write `_selection_rows` either. These tests pin both facts on a two-step
scenario whose target selection changes between the steps: the loop of the second
(whole-step) step reads the first step's rows.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu.models.vllm.qwen4_exp import mtp as mtp_module
from vllm_torchtpu.models.vllm.qwen4_exp import qsa
from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

WIDTH, TOKENS, REQS, T = 5, 32, 3, 4


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_selected_tokens = WIDTH
        self.rope_cache = torch.zeros(1)


def _layer(layer_type):
    layer = nn.Module()
    layer.layer_type = layer_type
    layer.self_attn = _Attention()
    return layer


@pytest.fixture
def bound(monkeypatch):
    monkeypatch.setattr(
        mtp_module.vllm_torchtpu_envs, "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION", True
    )
    monkeypatch.setattr(mtp_module.vllm_torchtpu_envs, "VLLM_TPU_MTP_DRAFT_DECODE_ROWS", False)

    # The donating row update is a TPU op; on CPU it is the plain row write it stands for.
    def update_rows_(buf, rows):
        buf[: rows.shape[0]] = rows

    monkeypatch.setattr(qsa, "update_rows_", update_rows_)
    target = SimpleNamespace(
        model=SimpleNamespace(layers=[_layer("linear_attention"), _layer("full_attention")])
    )
    draft = mtp_module.Qwen4ExpMTP.__new__(mtp_module.Qwen4ExpMTP)
    nn.Module.__init__(draft)
    draft.model = nn.Module()
    draft.model.layers = nn.ModuleList([_layer("full_attention")])
    assert draft.bind_shared_selection(target, TOKENS)
    return target.model.layers[1].self_attn, draft


def _selection(step):
    """The target's last-QSA-layer selection of a verify step (T rows per request)."""
    return (torch.arange(REQS * T * WIDTH, dtype=torch.int32).reshape(REQS * T, WIDTH)
            + 1000 * step)


def _last_rows():
    return torch.arange(REQS) * T + T - 1  # each request's last verified row


def _stock_step(sink_layer, attention, step):
    """A stock target step + stock draft step 0: the sink layer's `attend` writes the
    sink, then step 0 carries the rows (`attend_with_indices`, rows given)."""
    Qwen4ExpQSA.write_selection_sink(sink_layer, _selection(step))
    shared = qsa.shared_step0_rows(attention._selection_source, _last_rows(), REQS * T)
    qsa.update_rows_(attention._selection_rows, shared)


class _FakeMtp(nn.Module):
    def mk_draft_step(self, input_ids, positions, hidden_states, rows_per_request):
        return hidden_states, input_ids.to(torch.int32)


def _mk_whole_step(attention, step):
    """A whole-step step with MK draft step 0: the target runs in the kernel (the sink
    layer's stock core skipped), draft step 0 is `_MkDraftStep0`."""
    from vllm_torchtpu.spec_decode.mtp_fused_loop import _MkDraftStep0

    step0 = _MkDraftStep0(vllm_config=None, mtp=_FakeMtp(), rows_per_request=T)
    n = REQS * T
    step0(torch.zeros(n, dtype=torch.int64), torch.arange(n), torch.zeros(n, 8),
          _last_rows())


def test_stock_steps_carry_the_fresh_selection(bound):
    sink_layer, draft = bound
    attention = draft.model.layers[0].self_attn
    _stock_step(sink_layer, attention, 1)
    _stock_step(sink_layer, attention, 2)
    want = _selection(2)[_last_rows()]
    torch.testing.assert_close(attention._selection_rows[:REQS], want, rtol=0, atol=0)


def test_whole_step_paths_never_write_the_selection():
    """No whole-step / MK draft module writes the sink or the carried rows."""
    from vllm_torchtpu.megakernel_runtime import qwen38_draft, qwen38_driver, qwen38_wholestep
    from vllm_torchtpu.spec_decode import mtp_fused_loop

    for module in (qwen38_wholestep, qwen38_driver, qwen38_draft, mtp_fused_loop):
        src = inspect.getsource(module)
        assert "_selection_sink" not in src and "_selection_rows" not in src, module.__name__
        assert "write_selection_sink" not in src, module.__name__


@pytest.mark.xfail(strict=True, reason="C3: on a whole-step MK step the draft loop attends "
                   "the previous step's shared selection (sink and carried rows not refreshed)")
def test_loop_after_mk_step0_reads_this_steps_selection(bound):
    sink_layer, draft = bound
    attention = draft.model.layers[0].self_attn
    _stock_step(sink_layer, attention, 1)  # e.g. a mixed step: stock target + step 0
    _mk_whole_step(attention, 2)  # the target's layer-47 selection is now _selection(2)
    loop_rows = attention._selection_rows[:REQS]  # what loop steps 1..K-1 attend
    torch.testing.assert_close(loop_rows, _selection(1)[_last_rows()], rtol=0, atol=0)  # stale
    torch.testing.assert_close(loop_rows, _selection(2)[_last_rows()], rtol=0, atol=0)
