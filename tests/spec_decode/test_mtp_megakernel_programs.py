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
"""The Qwen3.5 MTP megakernel draft programs, called as the drafter calls them.

They only run when serving with the megakernel option `mtp_draft`, where a
call that doesn't bind fails on the first real draft step. These build them uncompiled
around a stand-in for vLLM's `Qwen3_5MoeMTP`, whose `forward` takes
`spec_step_idx` only through **kwargs, over the plugin's real
`patched_mtp_forward` and a fake megakernel.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CompilationMode

from vllm_torchtpu.models.vllm import qwen3_next_patch
from vllm_torchtpu.spec_decode.mtp_fused_loop import (
    _MtpMkFusedLoop,
    build_mtp_fused_loop,
)
from vllm_torchtpu.spec_decode.mtp_loop_model import (
    _MtpMkStepModel,
    build_mtp_loop_step_model,
)

pytestmark = pytest.mark.cpu_test

HIDDEN = 4
VOCAB = 32
K = 3
# Qwen3.5's MTP marks the MRoPE positions dynamic in their last dim.
QWEN35_MTP_DYNAMIC_ARG_DIMS = {
    "input_ids": 0,
    "positions": -1,
    "hidden_states": 0,
    "intermediate_tensors": 0,
    "inputs_embeds": 0,
}


class _Predictor(torch.nn.Module):
    """`Qwen3_5MultiTokenPredictor` as the plugin patches it."""

    forward = qwen3_next_patch.patched_mtp_forward

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN)
        self.pre_fc_norm_embedding = torch.nn.Identity()
        self.pre_fc_norm_hidden = torch.nn.Identity()
        self.fc = torch.nn.Linear(2 * HIDDEN, HIDDEN, bias=False)
        self.num_mtp_layers = 1
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])

    def embed_input_ids(self, input_ids):
        return self.embed(input_ids)

    def mk_input(self, input_ids, hidden_states):
        """The rows `patched_mtp_forward` hands the megakernel."""
        return self.fc(torch.cat([self.embed(input_ids), hidden_states], dim=-1))


class _Qwen35Mtp(torch.nn.Module):
    """`Qwen3_5MoeMTP` as of vLLM 0.29."""

    def __init__(self):
        super().__init__()
        self.model = _Predictor()

    def forward(
        self,
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states, spec_step_idx=0):
        return hidden_states


class _Megakernel:
    """Records `forward_layer` calls and returns the rows the real one does."""

    def __init__(self):
        self.gathers = []

    def forward_layer(self, layer, hidden_states, gather_indices):
        self.gathers.append(gather_indices)
        if gather_indices is None:
            return hidden_states
        return hidden_states[gather_indices]


@pytest.fixture
def mtp(monkeypatch):
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(qwen3_next_patch, "get_pp_group", lambda: pp_group)
    torch.manual_seed(0)
    return _Qwen35Mtp()


def _uncompiled():
    return SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )


@pytest.mark.parametrize("num_reqs", range(1, 9))
def test_mk_step0_program_returns_the_gathered_megakernel_rows(mtp, num_reqs):
    """Called with the drafter's step-0 keywords, twice as on its first use."""
    program = build_mtp_loop_step_model(
        vllm_config=_uncompiled(),
        prefix="mtp_mk_step0",
        mtp=mtp,
        dynamic_arg_dims=QWEN35_MTP_DYNAMIC_ARG_DIMS,
        megakernel=True,
    )
    assert isinstance(program, _MtpMkStepModel)
    num_tokens = 16 * -(-(K + 1) * num_reqs // 16)
    reqs = torch.arange(num_reqs)
    # Request r verifies rows (K+1)r .. (K+1)r+K and accepted r % (K+1).
    gather = torch.zeros(max(4, 1 << (num_reqs - 1).bit_length()), dtype=torch.int64)
    gather[:num_reqs] = (K + 1) * reqs + reqs % (K + 1)
    kwargs = {
        "input_ids": torch.randint(0, VOCAB, (num_tokens,)),
        "positions": torch.arange(num_tokens, dtype=torch.int32),
        "hidden_states": torch.randn(num_tokens, HIDDEN),
        "spec_step_idx": 0,
    }
    mk = _Megakernel()

    with (
        torch.no_grad(),
        qwen3_next_patch.mtp_megakernel_context(mk),
        qwen3_next_patch.step0_post_attn_gather_context(gather),
    ):
        program(**kwargs)
        out = program(**kwargs)

    rows = mtp.model.mk_input(kwargs["input_ids"], kwargs["hidden_states"])
    torch.testing.assert_close(out, rows[gather])
    assert len(mk.gathers) == 2
    assert all(g is gather for g in mk.gathers)


def test_mk_fused_loop_program_runs_each_step_on_the_megakernel(mtp):
    """At the largest loop bucket the tokens match the draft run step by
    step, and seq_lens advances once per step after the first."""
    p = 16
    program = build_mtp_fused_loop(
        vllm_config=_uncompiled(),
        prefix=f"mtp_fused_loop_k{K}_mk",
        mtp=mtp,
        num_steps=K - 1,
        dynamic_arg_dims=QWEN35_MTP_DYNAMIC_ARG_DIMS,
        propose_step0=True,
        megakernel=True,
    )
    assert isinstance(program, _MtpMkFusedLoop)
    hidden = torch.randn(p, HIDDEN)
    seq_lens = torch.full((p,), 101, dtype=torch.int32)
    mk = _Megakernel()

    with torch.no_grad(), qwen3_next_patch.mtp_megakernel_context(mk):
        tokens = program(
            None, torch.arange(p, dtype=torch.int32) + 100, hidden, seq_lens
        )

    want = [hidden.argmax(dim=-1)]
    with torch.no_grad():
        for _ in range(K - 1):
            hidden = mtp.model.mk_input(want[-1], hidden)
            want.append(hidden.argmax(dim=-1))
    assert torch.equal(tokens, torch.stack(want, dim=1).to(torch.int32))
    assert mk.gathers == [None] * (K - 1)
    assert torch.equal(seq_lens, torch.full((p,), 101 + K - 2, dtype=torch.int32))
