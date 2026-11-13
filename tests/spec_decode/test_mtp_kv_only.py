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
"""The two-pass MTP draft step 0 (spec_decode/mtp_kv_only.py) on CPU.

* `mtp_layer_input` vs the upstream `DeepSeekMultiTokenPredictor` ->
  `DeepSeekMultiTokenPredictorLayer` -> `DeepseekV2DecoderLayer` prefix;
* which drafts `kv_only_unsupported_reason` accepts;
* `MtpKvOnlyModel` hands the write op the MTP layer's attention input.

The upstream modules are real instances built without their `__init__`
(which needs a full vLLM config and a TP group), with stand-in submodules.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config.compilation import CompilationMode
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMTP,
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MLAAttention,
)

from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.spec_decode import mtp_kv_only

MTP_HIDDEN = 32
MTP_VOCAB = 64

# ---------------------------------------------------------------------------
# The MTP layer prefix
# ---------------------------------------------------------------------------


def _module(cls, **attrs):
    """An instance of `cls` with only `attrs` set."""
    obj = cls.__new__(cls)
    nn.Module.__init__(obj)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


class _Stop(Exception):
    pass


class _Capture(nn.Module):
    """The decoder layer's `self_attn`: records its input, then stops."""

    def forward(self, positions, hidden_states, *_args):
        self.seen = hidden_states
        raise _Stop


def _rms_norm(gen):
    norm = nn.RMSNorm(MTP_HIDDEN, eps=1e-6).bfloat16()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(MTP_HIDDEN, generator=gen))
    return norm


def _embedding(gen):
    # Seeded like every other weight here: the default init would draw from
    # the global RNG, whose state depends on the tests that ran before.
    embed = nn.Embedding(MTP_VOCAB, MTP_HIDDEN).bfloat16()
    with torch.no_grad():
        embed.weight.copy_(torch.randn(MTP_VOCAB, MTP_HIDDEN, generator=gen))
    return embed


def _upstream_mtp(self_attn: nn.Module, seed=0) -> DeepSeekMTP:
    gen = torch.Generator().manual_seed(seed)
    block = _module(
        DeepseekV2DecoderLayer,
        use_sequence_parallel_moe=False,
        use_mha=False,
        input_layernorm=_rms_norm(gen),
        self_attn=self_attn,
    )
    eh_proj = nn.Linear(2 * MTP_HIDDEN, MTP_HIDDEN, bias=False).bfloat16()
    nn.init.normal_(eh_proj.weight, generator=gen)
    layer = _module(
        DeepSeekMultiTokenPredictorLayer,
        enorm=_rms_norm(gen),
        hnorm=_rms_norm(gen),
        eh_proj=eh_proj,
        mtp_block=block,
    )
    predictor = _module(
        DeepSeekMultiTokenPredictor,
        layers=nn.ModuleDict({"78": layer}),
        mtp_start_layer_idx=78,
        num_mtp_layers=1,
        embed_tokens=_embedding(gen),
    )
    return _module(DeepSeekMTP, model=predictor)


def _mtp_inputs():
    gen = torch.Generator().manual_seed(3)
    input_ids = torch.randint(0, MTP_VOCAB, (12,), generator=gen)
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 7, 8, 9, 10, 0, 5])
    hidden = torch.randn(12, MTP_HIDDEN, generator=gen).bfloat16()
    return input_ids, positions, hidden


def test_mtp_layer_input_matches_the_upstream_prefix():
    """The guard on a vLLM pin bump: `mtp_layer_input` is a copy of the
    upstream forwards up to the attention call."""
    capture = _Capture()
    mtp = _upstream_mtp(capture)
    input_ids, positions, hidden = _mtp_inputs()
    with torch.no_grad(), pytest.raises(_Stop):
        mtp.model(input_ids, positions, hidden, None, 0)
    with torch.no_grad():
        layer, got = mtp_kv_only.mtp_layer_input(mtp, input_ids, positions, hidden)
    assert layer is mtp.model.layers["78"]
    assert torch.equal(got, capture.seen)


# ---------------------------------------------------------------------------
# Support check
# ---------------------------------------------------------------------------


def _impl():
    return PallasMLAttentionBackendImpl.__new__(PallasMLAttentionBackendImpl)


def _dsa_wrapper(mla_attn=None):
    indexer_op = _module(mla_op.VllmTPUSparseAttnIndexer)
    return _module(
        mla_op.VllmTPUMultiHeadLatentAttentionWrapper,
        is_sparse=True,
        kimi_preprocess=None,
        indexer=_module(mla_op.VllmTPUIndexer, indexer_op=indexer_op),
        mla_attn=mla_attn or SimpleNamespace(impl=_impl()),
    )


def _supported_mtp(wrapper=None) -> DeepSeekMTP:
    self_attn = _module(DeepseekV2MLAAttention, mla_attn=wrapper or _dsa_wrapper())
    return _upstream_mtp(self_attn)


def _parts(mtp):
    layer = mtp.model.layers["78"]
    wrapper = layer.mtp_block.self_attn.mla_attn
    return layer, wrapper


def test_kv_only_supported_for_a_dsa_mtp():
    assert mtp_kv_only.kv_only_unsupported_reason(_supported_mtp()) is None


def test_kv_only_supported_for_a_subclass_that_keeps_the_forwards():
    """Pass 1 shares the wrapper's projections, so a subclass that keeps
    `forward` computes the same entries."""
    mtp = _supported_mtp()
    _, w = _parts(mtp)
    w.__class__ = type("_Sub", (type(w),), {})
    assert mtp_kv_only.kv_only_unsupported_reason(mtp) is None


BREAKAGES = {
    "not_deepseek_mtp": lambda mtp: setattr(mtp, "__class__", nn.Module),
    "two_layers": lambda mtp: mtp.model.layers.update({"79": mtp.model.layers["78"]}),
    "not_mla": lambda mtp: setattr(
        mtp.model.layers["78"].mtp_block, "self_attn", nn.Identity()
    ),
    "plain_wrapper": lambda mtp: setattr(
        mtp.model.layers["78"].mtp_block.self_attn, "mla_attn", nn.Identity()
    ),
    "dense": lambda mtp: setattr(_parts(mtp)[1], "is_sparse", False),
    "kimi": lambda mtp: setattr(_parts(mtp)[1], "kimi_preprocess", object()),
    "shared_indexer": lambda mtp: setattr(_parts(mtp)[1], "indexer", nn.Identity()),
    "indexer_op_stub": lambda mtp: setattr(
        _parts(mtp)[1].indexer, "indexer_op", nn.Identity()
    ),
    "impl_stub": lambda mtp: setattr(
        _parts(mtp)[1], "mla_attn", SimpleNamespace(impl=object())
    ),
    "wrapper_forward_overridden": lambda mtp: setattr(
        _parts(mtp)[1],
        "__class__",
        type("_Sub", (type(_parts(mtp)[1]),), {"forward": lambda *a: a}),
    ),
    # E.g. a draft that runs step 0 on selected rows inside its own forward.
    "draft_forward_overridden": lambda mtp: setattr(
        mtp, "__class__", type("_Sub", (type(mtp),), {"forward": lambda *a: a})
    ),
    "layer_forward_overridden": lambda mtp: setattr(
        mtp.model.layers["78"],
        "__class__",
        type("_Sub", (DeepSeekMultiTokenPredictorLayer,), {"forward": lambda *a: a}),
    ),
}


@pytest.mark.parametrize("breakage", sorted(BREAKAGES))
def test_kv_only_unsupported_reasons(breakage):
    mtp = _supported_mtp()
    BREAKAGES[breakage](mtp)
    assert mtp_kv_only.kv_only_unsupported_reason(mtp) is not None


@pytest.mark.parametrize(
    "where",
    [
        "draft",
        "predictor",
        "layer",
        "block",
        "wrapper",
        "indexer",
        "indexer_op",
        "impl",
    ],
)
def test_kv_only_refuses_a_forward_patched_on_the_class(monkeypatch, where):
    """A patch (`patch_registry`) replaces the method on the class itself,
    so the subclass check passes; the method's home module and name give it
    away."""
    mtp = _supported_mtp()
    layer, wrapper = _parts(mtp)
    obj, name = {
        "draft": (mtp, "forward"),
        "predictor": (mtp.model, "forward"),
        "layer": (layer, "forward"),
        "block": (layer.mtp_block, "forward"),
        "wrapper": (wrapper, "forward"),
        "indexer": (wrapper.indexer, "forward"),
        "indexer_op": (wrapper.indexer.indexer_op, "forward_oot"),
        "impl": (wrapper.mla_attn.impl, "forward"),
    }[where]
    cls = type(obj)

    def patched_forward(*args):
        return args

    monkeypatch.setattr(cls, name, patched_forward)
    reason = mtp_kv_only.kv_only_unsupported_reason(mtp)
    assert reason is not None and "patched_forward" in reason


# ---------------------------------------------------------------------------
# MtpKvOnlyModel
# ---------------------------------------------------------------------------


class _RecordedWriteOp:
    """`MlaKvWriteOp` (tested with its writers in
    tests/layers/adapter/test_mla_kv_write_op.py), recording its calls."""

    def __init__(self, wrapper, mesh):
        self.wrapper = wrapper
        self.mesh = mesh
        self.calls = []

    def __call__(self, positions, hidden_states):
        self.calls.append((positions, hidden_states))
        return hidden_states


def _uncompiled_config():
    return SimpleNamespace(
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE)
    )


def test_mtp_kv_only_model_writes_the_mtp_layers_attention_input(monkeypatch):
    monkeypatch.setattr(mla_op, "MlaKvWriteOp", _RecordedWriteOp)
    mtp = _supported_mtp()
    mesh = object()
    model = mtp_kv_only.MtpKvOnlyModel(
        vllm_config=_uncompiled_config(),
        prefix="mtp_prefill_kv_only",
        mtp=mtp,
        mesh=mesh,
    )
    assert model.prefix == "mtp_prefill_kv_only"
    write = model._write
    assert isinstance(write, _RecordedWriteOp)
    assert write.wrapper is _parts(mtp)[1]
    assert write.mesh is mesh
    input_ids, positions, hidden = _mtp_inputs()
    with torch.no_grad():
        model(input_ids, positions, hidden)
        _, attn_input = mtp_kv_only.mtp_layer_input(mtp, input_ids, positions, hidden)
    ((got_positions, got_hidden),) = write.calls
    assert got_positions is positions
    assert torch.equal(got_hidden, attn_input)


def test_mtp_kv_only_model_refuses_an_unsupported_draft(monkeypatch):
    monkeypatch.setattr(mla_op, "MlaKvWriteOp", _RecordedWriteOp)
    mtp = _supported_mtp()
    _parts(mtp)[1].is_sparse = False
    with pytest.raises(NotImplementedError, match="not sparse"):
        mtp_kv_only.MtpKvOnlyModel(vllm_config=_uncompiled_config(), mtp=mtp)
