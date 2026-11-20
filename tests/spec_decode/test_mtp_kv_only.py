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

* pass 1's decoder-layer stand-in sees what the upstream
  `DeepSeekMultiTokenPredictor` -> `DeepSeekMultiTokenPredictorLayer` ->
  `DeepseekV2DecoderLayer` forwards hand its attention;
* which drafts `kv_only_unsupported_reason` accepts;
* `MtpKvOnlyModel` hands the write op the MTP layer's attention input;
* the proposer side (`TPU_MTP_KV_ONLY_STEP0`): `MtpKvOnlyStep0`'s build,
  refusals, gate and compile sizes, the narrowed pass-2 metadata, and that a
  propose with the switch on leaves the same draft tokens, KV / indexer-K
  caches and IndexShare rows as the single full pass. That draft is a toy
  that keeps the property the argument rests on: rows couple only through a
  causal read of caches every row writes first.

The upstream modules are real instances built without their `__init__`
(which needs a full vLLM config and a TP group), with stand-in submodules.
"""

import contextlib
import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from torch import nn
from vllm.config.compilation import CompilationMode
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMTP,
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MLAAttention,
    DeepseekV32IndexerCache,
)

import vllm_torchtpu.spec_decode.eagle3 as e3
from tests.layers.adapter import mla_test_utils as mla
from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.spec_decode import mtp_kv_only
from vllm_torchtpu.spec_decode.eagle3 import DraftChunkInputs
from vllm_torchtpu.spec_decode.mtp_kv_only import (
    MtpKvOnlyStep0,
    mtp_kv_only_step0_config_reason,
)
from vllm_torchtpu.spec_decode.utils import schedules_prefill

from .proposer_test_utils import make_proposer

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


def _rms_norm(gen, hidden):
    norm = nn.RMSNorm(hidden, eps=1e-6).bfloat16()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden, generator=gen))
    return norm


def _embedding(gen, hidden):
    # Seeded like every other weight here: the default init would draw from
    # the global RNG, whose state depends on the tests that ran before.
    embed = nn.Embedding(MTP_VOCAB, hidden).bfloat16()
    with torch.no_grad():
        embed.weight.copy_(torch.randn(MTP_VOCAB, hidden, generator=gen))
    return embed


def _upstream_mtp(
    self_attn: nn.Module, seed=0, hidden=MTP_HIDDEN, sequence_parallel_moe=False
) -> DeepSeekMTP:
    gen = torch.Generator().manual_seed(seed)
    block = _module(
        DeepseekV2DecoderLayer,
        use_sequence_parallel_moe=sequence_parallel_moe,
        use_mha=False,
        input_layernorm=_rms_norm(gen, hidden),
        self_attn=self_attn,
    )
    eh_proj = nn.Linear(2 * hidden, hidden, bias=False).bfloat16()
    nn.init.normal_(eh_proj.weight, generator=gen)
    layer = _module(
        DeepSeekMultiTokenPredictorLayer,
        enorm=_rms_norm(gen, hidden),
        hnorm=_rms_norm(gen, hidden),
        eh_proj=eh_proj,
        mtp_block=block,
        shared_head=nn.Identity(),
    )
    predictor = _module(
        DeepSeekMultiTokenPredictor,
        layers=nn.ModuleDict({"78": layer}),
        mtp_start_layer_idx=78,
        num_mtp_layers=1,
        embed_tokens=_embedding(gen, hidden),
    )
    return _module(DeepSeekMTP, model=predictor)


def _mtp_inputs():
    gen = torch.Generator().manual_seed(3)
    input_ids = torch.randint(0, MTP_VOCAB, (12,), generator=gen)
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 7, 8, 9, 10, 0, 5])
    hidden = torch.randn(12, MTP_HIDDEN, generator=gen).bfloat16()
    return input_ids, positions, hidden


def test_kv_only_stand_in_sees_the_attention_input():
    """The guard on a vLLM pin bump: the upstream predictor and MTP layer
    forwards hand pass 1's stand-in the input norm of what they hand the
    decoder layer, which is what its `self_attn` receives."""
    capture = _Capture()
    mtp = _upstream_mtp(capture)
    input_ids, positions, hidden = _mtp_inputs()
    with torch.no_grad(), pytest.raises(_Stop):
        mtp.model(input_ids, positions, hidden, None, 0)
    seen = []
    predictor = mtp_kv_only.kv_only_predictor(mtp, lambda pos, h: seen.append((pos, h)))
    with torch.no_grad():
        predictor(input_ids, positions, hidden)
    ((got_positions, got),) = seen
    assert got_positions is positions
    assert torch.equal(got, capture.seen)
    # The draft itself keeps its decoder layer; the copy shares its weights.
    layer = mtp.model.layers["78"]
    assert layer.mtp_block.self_attn is capture
    kv_layer = predictor.layers["78"]
    assert kv_layer is not layer and kv_layer.eh_proj is layer.eh_proj
    assert predictor.embed_tokens is mtp.model.embed_tokens


def test_kv_only_stand_in_skips_the_sequence_parallel_gather(monkeypatch):
    """With a sequence-parallel MoE the MTP layer all-gathers its decoder
    layer's token shards; the stand-in returns every row unsharded, so pass 1
    must not run that gather."""
    from vllm.model_executor.models import deepseek_mtp

    def no_gather(*_args, **_kwargs):
        raise AssertionError("pass 1 all-gathered the decoder layer's output")

    monkeypatch.setattr(deepseek_mtp, "tensor_model_parallel_all_gather", no_gather)
    mtp = _upstream_mtp(_Capture(), sequence_parallel_moe=True)
    input_ids, positions, hidden = _mtp_inputs()
    seen = []
    predictor = mtp_kv_only.kv_only_predictor(mtp, lambda pos, h: seen.append(h))
    with torch.no_grad():
        out, _ = predictor(input_ids, positions, hidden)
    assert len(seen) == 1
    assert out.shape == hidden.shape
    # The draft's own decoder layer keeps its sequence-parallel MoE.
    assert mtp.model.layers["78"].mtp_block.use_sequence_parallel_moe


# ---------------------------------------------------------------------------
# Support check
# ---------------------------------------------------------------------------


class TestMtpKvOnlyStep0ConfigReason:
    """Which configurations keep the MTP draft's single step-0 pass, decided
    before any weights load (TPU_MTP_KV_ONLY_STEP0 is on by default)."""

    @staticmethod
    def _config(method="mtp"):
        vllm_config = MagicMock()
        vllm_config.speculative_config = (
            None if method is None else SimpleNamespace(method=method)
        )
        return vllm_config

    def test_an_mtp_draft_without_pcp_may_take_the_two_passes(self, monkeypatch):
        monkeypatch.delenv("TPU_MTP_KV_ONLY_STEP0", raising=False)
        assert mtp_kv_only_step0_config_reason(self._config(), False) is None

    def test_the_kill_switch_keeps_the_single_pass(self, monkeypatch):
        monkeypatch.setenv("TPU_MTP_KV_ONLY_STEP0", "0")
        reason = mtp_kv_only_step0_config_reason(self._config(), False)
        assert reason == "TPU_MTP_KV_ONLY_STEP0 is 0"

    @pytest.mark.parametrize(
        "method, message", [("eagle3", "'eagle3'"), (None, "None")]
    )
    def test_a_draft_other_than_mtp_keeps_the_single_pass(
        self, monkeypatch, method, message
    ):
        monkeypatch.delenv("TPU_MTP_KV_ONLY_STEP0", raising=False)
        reason = mtp_kv_only_step0_config_reason(self._config(method), False)
        assert reason == f"the draft method is {message}"

    def test_prefill_context_parallelism_keeps_the_single_pass(self, monkeypatch):
        monkeypatch.delenv("TPU_MTP_KV_ONLY_STEP0", raising=False)
        reason = mtp_kv_only_step0_config_reason(self._config(), True)
        assert reason == "prefill context parallelism is on"


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
    capture = _Capture()
    _parts(mtp)[0].mtp_block.self_attn = capture
    with torch.no_grad(), pytest.raises(_Stop):
        mtp.model(input_ids, positions, hidden, None, 0)
    with torch.no_grad():
        out = model(input_ids, positions, hidden)
    ((got_positions, got_hidden),) = write.calls
    assert got_positions is positions
    assert torch.equal(got_hidden, capture.seen)
    # One row of output: the writes are the point.
    assert out.shape == (1, hidden.shape[1])


def test_mtp_kv_only_model_refuses_an_unsupported_draft(monkeypatch):
    monkeypatch.setattr(mla_op, "MlaKvWriteOp", _RecordedWriteOp)
    mtp = _supported_mtp()
    _parts(mtp)[1].is_sparse = False
    with pytest.raises(NotImplementedError, match="not sparse"):
        mtp_kv_only.MtpKvOnlyModel(vllm_config=_uncompiled_config(), mtp=mtp)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

DRAFT_LAYER = "model.layers.78.mtp_block.self_attn.attn"
HIDDEN = 4
VOCAB = 32
TOPK = 2


@pytest.mark.parametrize(
    "q_lens, expected",
    [
        ([4, 4, 2], False),  # verify only: at most 1 + K tokens each
        ([4, 5], True),  # one more than a verify step gives
        ([1, 1], False),  # decode, or one-token prefill spans
        ([], False),
    ],
)
def test_schedules_prefill(q_lens, expected):
    assert schedules_prefill(q_lens, num_speculative_tokens=3) is expected


@pytest.fixture(autouse=True)
def _measured_chip(monkeypatch):
    """The dispatch table is measured on TPU7x: build there, without asking
    the host which chip it is."""
    monkeypatch.setattr(
        mtp_kv_only, "get_chip_version", lambda: pltpu.ChipVersion.TPU_7X
    )


@pytest.fixture
def fixed_gate(monkeypatch):
    """Dispatch on `MtpKvOnlyStep0.min_tokens` alone, as if every loop
    bucket's measured threshold were below it."""
    monkeypatch.setattr(mtp_kv_only, "two_pass_min_tokens", lambda loop_bucket: 0)


def test_two_pass_min_tokens_reads_the_table(monkeypatch):
    monkeypatch.setattr(mtp_kv_only, "_TWO_PASS_MIN_TOKENS", {16: 256, 64: 512})
    log = MagicMock()
    monkeypatch.setattr(mtp_kv_only.logger, "info_once", log)
    assert mtp_kv_only.two_pass_min_tokens(16) == 256
    assert mtp_kv_only.two_pass_min_tokens(64) == 512
    assert log.call_count == 0
    # Between entries, and below the smallest: the next larger entry's
    # threshold (a smaller p only makes pass 2 cheaper), so it never falls
    # as p grows.
    assert mtp_kv_only.two_pass_min_tokens(24) == 512
    assert mtp_kv_only.two_pass_min_tokens(8) == 256
    # Above the largest: its T/p, rounded up.
    assert mtp_kv_only.two_pass_min_tokens(96) == 768
    # Each miss names its loop bucket, the threshold and the entry it used.
    assert [c.args[1:] for c in log.call_args_list] == [
        (24, 512, 64),
        (8, 256, 16),
        (96, 768, 64),
    ]


def test_takes_two_pass_needs_the_loop_buckets_threshold(monkeypatch):
    monkeypatch.setattr(mtp_kv_only, "_TWO_PASS_MIN_TOKENS", {16: 64, 32: 256})
    step0 = MtpKvOnlyStep0(object(), 64, object())
    assert step0.takes_two_pass(64, 16, True)
    assert not step0.takes_two_pass(32, 16, True)  # below the p=16 threshold
    assert not step0.takes_two_pass(128, 32, True)  # below the p=32 threshold
    assert step0.takes_two_pass(256, 32, True)
    assert not step0.takes_two_pass(4096, 16, False)  # verify only


def _gate_proposer(min_tokens=8):
    proposer = make_proposer(method="mtp")
    proposer.speculative_config.num_speculative_tokens = 3
    proposer.runner.num_tokens_paddings = [8, 16, 32]
    proposer._mtp_kv_only = MtpKvOnlyStep0(object(), min_tokens, object())
    return proposer


def _qsl(q_lens):
    return SimpleNamespace(
        num_reqs=len(q_lens),
        query_start_loc_np=np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32),
    )


@pytest.mark.parametrize(
    "q_lens, num_tokens, expected",
    [
        ([4, 6, 5], 16, True),
        ([5, 3], 8, True),  # exactly the gate
        ([4, 3], 7, False),  # below the gate
        ([4, 4, 4, 4], 16, False),  # verify only
    ],
)
def test_takes_two_pass_step0_gate(fixed_gate, q_lens, num_tokens, expected):
    proposer = _gate_proposer()
    assert proposer._takes_two_pass_step0(_qsl(q_lens), num_tokens) is expected


def test_takes_two_pass_step0_reads_the_chunks_loop_bucket(monkeypatch):
    # The real table lookup: p is the token bucket the chunk's requests pad to.
    monkeypatch.setattr(mtp_kv_only, "_TWO_PASS_MIN_TOKENS", {8: 16, 16: 32})
    proposer = _gate_proposer()
    three = _qsl([4, 6, 5])  # p = 8
    nine = [1] * 8  # with one prefill span below, p = 16
    assert proposer._takes_two_pass_step0(three, 16) is True
    assert proposer._takes_two_pass_step0(_qsl([*nine, 8]), 16) is False
    assert proposer._takes_two_pass_step0(_qsl([*nine, 24]), 32) is True
    assert proposer.takes_two_pass_prefill(16, 3) is True
    assert proposer.takes_two_pass_prefill(16, 9) is False
    assert proposer.takes_two_pass_prefill(32, 9) is True
    with proposer.single_pass_step0():
        assert proposer._takes_two_pass_step0(three, 16) is False
    assert proposer._takes_two_pass_step0(three, 16) is True


def test_takes_two_pass_step0_needs_the_switch():
    proposer = _gate_proposer()
    proposer._mtp_kv_only = None
    assert proposer._takes_two_pass_step0(_qsl([4096]), 4096) is False


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

BUCKETS = [16, 32, 64, 128, 1024, 4096]
MLA_NAME = "mtp.attn"
INDEXER_NAME = "mtp.indexer.k_cache"


def _attn_layers(num_mla=1, num_indexer=1):
    """The draft's attention-layer registry entries, as load_model sees
    them."""
    layers = {
        f"{MLA_NAME}{i or ''}": MLAAttention.__new__(MLAAttention)
        for i in range(num_mla)
    }
    for i in range(num_indexer):
        layers[f"{INDEXER_NAME}{i or ''}"] = DeepseekV32IndexerCache.__new__(
            DeepseekV32IndexerCache
        )
    return layers


def _build_proposer(method="mtp"):
    proposer = make_proposer(method=method)
    proposer.draft_model = object()
    proposer.draft_vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(compile_sizes=list(BUCKETS))
    )
    proposer.runner = SimpleNamespace(
        _pcp_enabled=False,
        mesh=object(),
        _dp_lockstep_enabled=lambda: False,
        # Loop buckets up to 32 (max_num_reqs 32): the gate is 33.
        num_tokens_paddings=list(BUCKETS),
        max_num_reqs=32,
    )
    proposer._draft_attn_layer_names = set(_attn_layers())
    return proposer


def _patch_builder(monkeypatch, reason=None, raises=False):
    built = []

    class _FakeModel:
        def __init__(self, **kwargs):
            if raises:
                raise RuntimeError("boom")
            built.append(kwargs)

    monkeypatch.setattr(mtp_kv_only, "kv_only_unsupported_reason", lambda mtp: reason)
    monkeypatch.setattr(mtp_kv_only, "MtpKvOnlyModel", _FakeModel)
    for name in (
        "set_model_tag",
        "set_vllm_model_wrapper_context",
        "set_current_vllm_config",
    ):
        monkeypatch.setattr(mtp_kv_only, name, lambda *a, **k: contextlib.nullcontext())
    return built


def test_build_compiles_only_the_buckets_it_dispatches(monkeypatch):
    proposer = _build_proposer()
    built = _patch_builder(monkeypatch)
    monkeypatch.setattr(mtp_kv_only, "_TWO_PASS_MIN_TOKENS", {16: 64, 32: 128})
    logs = []
    monkeypatch.setattr(
        mtp_kv_only.logger, "info", lambda msg, *args: logs.append(msg % args)
    )
    step0 = MtpKvOnlyStep0.build(proposer, _attn_layers())
    # The loop buckets reach 32 (max_num_reqs 32); the smallest threshold
    # among them is p=16's.
    assert step0.min_tokens == 64
    assert step0.takes_two_pass(64, 16, True) and not step0.takes_two_pass(64, 32, True)
    assert step0.takes_two_pass(128, 32, True)
    (kwargs,) = built
    assert step0.model is not None
    config = kwargs["vllm_config"]
    assert config is step0.vllm_config
    assert config.compilation_config.compile_sizes == [64, 128, 1024, 4096]
    # The draft's own programs still compile every bucket.
    assert proposer.draft_vllm_config.compilation_config.compile_sizes == BUCKETS
    assert kwargs["mtp"] is proposer.draft_model
    assert kwargs["mesh"] is proposer.runner.mesh
    assert kwargs["prefix"] == "mtp_prefill_kv_only"
    (log,) = logs
    assert "from 64 padded tokens" in log


def test_set_draft_compile_sizes_keeps_the_kv_only_copy_at_its_gate(monkeypatch):
    """Pipeline chunks add token buckets after load: the draft compiles all
    of them, the KV-only pass only those at or above its gate."""
    proposer = _build_proposer()
    _patch_builder(monkeypatch)
    proposer._mtp_kv_only = MtpKvOnlyStep0.build(proposer, _attn_layers())
    proposer.set_draft_compile_sizes([16, 32, 512, 1024, 4096])
    assert proposer.draft_vllm_config.compilation_config.compile_sizes == [
        16,
        32,
        512,
        1024,
        4096,
    ]
    kv_only_config = proposer._mtp_kv_only.vllm_config
    assert kv_only_config.compilation_config.compile_sizes == [512, 1024, 4096]
    # Every target bucket at or above the gate has a KV-only graph.
    assert proposer.kv_only_missing_buckets([16, 32, 512, 1024, 4096]) == []


def test_kv_only_missing_buckets_names_a_bucket_its_copy_lacks(monkeypatch):
    """A target bucket at or above the gate that only the draft's own copy
    of the sizes got is reported; buckets below the gate never take the two
    passes and are not."""
    proposer = _build_proposer()
    _patch_builder(monkeypatch)
    proposer._mtp_kv_only = MtpKvOnlyStep0.build(proposer, _attn_layers())
    proposer.set_draft_compile_sizes([16, 512, 4096])
    proposer.draft_vllm_config.compilation_config.compile_sizes = [16, 512, 2048, 4096]
    assert proposer.kv_only_missing_buckets([16, 512, 2048, 4096]) == [2048]


def test_kv_only_missing_buckets_is_empty_while_the_switch_is_off():
    proposer = _build_proposer()
    proposer._mtp_kv_only = None
    assert proposer.kv_only_missing_buckets([16, 512, 4096]) == []


@pytest.mark.parametrize(
    "setup, message",
    [
        ("kv_shared", "reads the target's KV cache"),
        ("unsupported", "not DSA"),
        ("two_mla_layers", "not one of each"),
        ("no_indexer_cache", "not one of each"),
        ("no_bucket_reaches_it", "no token bucket reaches 8192 tokens"),
        ("unmeasured_chip", "dispatch table is measured on TPU7x, not v6e"),
    ],
)
def test_build_keeps_the_single_pass(monkeypatch, setup, message):
    """A draft or layout that cannot take the two-pass step 0 (or no token
    bucket where it would pay off) keeps the single pass and says why. (The
    draft method and PCP are decided before load, by the platform.)"""
    proposer = _build_proposer()
    built = _patch_builder(
        monkeypatch, reason="not DSA" if setup == "unsupported" else None
    )
    layers = _attn_layers(
        num_mla=2 if setup == "two_mla_layers" else 1,
        num_indexer=0 if setup == "no_indexer_cache" else 1,
    )
    proposer._draft_attn_layer_names = set(layers)
    if setup == "kv_shared":
        proposer._draft_kv_shared = True
    if setup == "no_bucket_reaches_it":
        monkeypatch.setattr(mtp_kv_only, "_TWO_PASS_MIN_TOKENS", {16: 8192})
    if setup == "unmeasured_chip":
        monkeypatch.setattr(
            mtp_kv_only, "get_chip_version", lambda: pltpu.ChipVersion.TPU_V6E
        )
    logs = []
    monkeypatch.setattr(
        mtp_kv_only.logger, "info", lambda msg, *args: logs.append(msg % args)
    )
    assert MtpKvOnlyStep0.build(proposer, layers) is None
    (log,) = logs
    assert "keeps the single pass" in log and message in log
    # Decided before the op is built (and registered).
    assert built == []


def test_build_under_ep_dp_lockstep(monkeypatch):
    """The idle ranks replay the two passes (see the lockstep tests below)."""
    proposer = _build_proposer()
    built = _patch_builder(monkeypatch)
    proposer._dp_lockstep_sharded = lambda: True
    step0 = MtpKvOnlyStep0.build(proposer, _attn_layers())
    assert step0 is not None
    assert len(built) == 1


def test_build_error_propagates(monkeypatch):
    """A failed build is not caught: falling back on one rank would leave it
    on the single pass while its TP peers run the two passes."""
    proposer = _build_proposer()
    _patch_builder(monkeypatch, raises=True)
    with pytest.raises(RuntimeError, match="boom"):
        MtpKvOnlyStep0.build(proposer, _attn_layers())


def test_precompile_gather_rows_warms_every_shape_the_two_passes_take(
    monkeypatch,
):
    """Every (T, p) the table selects: T a token bucket, p a loop bucket."""
    monkeypatch.setattr(mtp_kv_only, "_TWO_PASS_MIN_TOKENS", {16: 1024, 32: 4096})
    proposer = make_proposer(method="mtp")
    proposer.runner = SimpleNamespace(
        num_tokens_paddings=[16, 32, 64, 128, 1024, 4096],
        max_num_reqs=32,
        device=torch.device("cpu"),
        uses_mrope=False,
        _hidden_states_dtype=torch.bfloat16,
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )
    proposer._draft_hidden_size = lambda: 8
    seen = []
    proposer._draft_gather_rows = lambda x, pos, y, idx: seen.append(
        (x.shape, pos.shape, y.shape, idx.shape)
    )
    monkeypatch.setattr(mtp_kv_only, "synchronize_tensors", lambda out: None)

    MtpKvOnlyStep0(object(), 1024, object()).precompile_gather_rows(proposer)
    assert [(x[0], idx[0]) for x, _, _, idx in seen] == [
        (1024, 16),
        (4096, 16),
        (4096, 32),
    ]
    assert all(pos == x and y == (x[0], 8) for x, pos, y, _ in seen)


# ---------------------------------------------------------------------------
# Pass-2 metadata
# ---------------------------------------------------------------------------


def _ctx(seq_lens, qsl, num_reqs, knr):
    seq = np.zeros(knr, dtype=np.int32)
    seq[: len(seq_lens)] = seq_lens
    qsl_full = np.full(knr + 1, qsl[-1], dtype=np.int32)
    qsl_full[: len(qsl)] = qsl
    return SimpleNamespace(
        use_max_model_len=True,
        seq_lens=torch.from_numpy(seq),
        query_start_loc=torch.from_numpy(qsl_full),
        request_distribution=torch.tensor([0, num_reqs, num_reqs], dtype=torch.int32),
        position_ids_override=None,
    )


@pytest.mark.parametrize("async_rejected", [False, True])
def test_pass2_attn_metadata_is_one_query_per_request(async_rejected):
    proposer = make_proposer(method="mtp")
    knr = 6
    proposer.runner = SimpleNamespace(
        num_reqs_max_model_len=knr, num_reqs_most_model_len=None, device="cpu"
    )
    ctx = _ctx([10, 6, 8], [0, 4, 10, 15], 3, knr)
    md = AttentionMetadata(
        input_positions=torch.zeros(16, dtype=torch.int32),
        block_tables=torch.arange(12, dtype=torch.int32),
        seq_lens=ctx.seq_lens,
        query_start_loc=ctx.query_start_loc,
        request_distribution=ctx.request_distribution,
    )
    other = AttentionMetadata(input_positions=torch.zeros(16, dtype=torch.int32))
    chunk = SimpleNamespace(
        num_reqs=3, attn_ctx=ctx, attn_metadata={"a": md, "b": md, "c": other}
    )
    proposer._draft_attn_layer_names = {"a", "b", "c"}
    rejected = np.array([2, 0, 0], dtype=np.int32)
    if async_rejected:
        rejected = torch.from_numpy(rejected)
    rejected_np = None if async_rejected else rejected

    loop_inputs = proposer._chunk_loop_inputs(chunk, rejected)
    out = proposer._pass2_attn_metadata(chunk, rejected_np, loop_inputs, 4)

    assert out["a"] is out["b"]  # shared metadata stays shared
    new = out["a"]
    assert new.block_tables is md.block_tables
    assert new.seq_lens.tolist() == [8, 6, 8, 0, 0, 0]
    assert new.query_start_loc.tolist() == [0, 1, 2, 3, 3, 3, 3]
    assert new.request_distribution.tolist() == [3, 3, 3]
    # The loop steps reuse the same query starts and distribution.
    assert new.query_start_loc is loop_inputs.query_start_loc
    assert new.request_distribution is loop_inputs.request_distribution
    # The target's own metadata is untouched.
    assert md.seq_lens is ctx.seq_lens
    assert md.query_start_loc is ctx.query_start_loc


def test_two_pass_step0_takes_the_gather_indices_after_pass_1(monkeypatch):
    """The single pass takes its gather indices after its forward. A compiled
    call materializes the pending eager ops, so indices taken before pass 1
    would fuse with the index padding into a program warmup never compiles:
    pass 1, then the indices, then pass 2's gather and step 0."""
    events = []
    real = e3._step0_gather_indices

    def indices(last_token_indices):
        events.append("indices")
        return real(last_token_indices)

    monkeypatch.setattr(e3, "_step0_gather_indices", indices)
    proposer = make_proposer(method="mtp")
    proposer._build_draft_attn_metadata = lambda **_k: {}
    proposer._run_kv_only_pass = lambda *_a: events.append("pass 1")

    def gather(x, positions, y, idx):
        events.append("gather")
        return x[idx], positions[idx], y[idx]

    def pass2(**kwargs):
        events.append("pass 2")
        return kwargs["target_hidden_states"], kwargs["target_hidden_states"]

    proposer._draft_gather_rows = gather
    proposer._chunk_loop_inputs = lambda chunk, rejected: "loop inputs"
    proposer._pass2_attn_metadata = lambda *_a: {}
    proposer._run_draft_model = pass2
    rows = torch.tensor([3, 7, -1, 7])

    *_, loop_inputs, owner_mask = proposer._two_pass_step0(
        chunk=object(),
        input_ids=torch.arange(8),
        positions=torch.arange(8),
        target_hidden_states=torch.arange(8.0),
        last_token_indices=rows,
        num_rejected=np.zeros(3, dtype=np.int32),
    )
    assert events == ["pass 1", "indices", "gather", "pass 2"]
    assert loop_inputs == "loop inputs"
    assert owner_mask.tolist() == [True, True, False, True]


def test_save_topk_snapshots_the_front_rows():
    proposer = make_proposer(method="mtp")
    buf = torch.arange(24, dtype=torch.int32).reshape(6, 4)
    proposer._mtp_topk_buffers = [buf]
    (saved,) = proposer._mtp_save_topk(2)
    assert torch.equal(saved, buf[:2])
    buf.zero_()
    assert saved.sum() > 0  # a copy, not a view


def test_chunk_loop_inputs_trims_a_padded_async_rejected_count():
    """Async: the rejected count arrives at the sampler's request-padding
    width, wider than kernel_num_reqs. The loop steps' copy is cut to knr, as
    `propose` cuts its own, so pass 2 and the loop steps see one shape."""
    proposer = make_proposer(method="mtp")
    knr = 4
    proposer.runner = SimpleNamespace(
        num_reqs_max_model_len=knr, num_reqs_most_model_len=None, device="cpu"
    )
    chunk = SimpleNamespace(
        num_reqs=3, attn_ctx=SimpleNamespace(use_max_model_len=True)
    )
    rejected = torch.tensor([2, 0, 1, 0, 0, 0], dtype=torch.int32)
    rejected_dev, qsl, reqdist = proposer._chunk_loop_inputs(chunk, rejected)
    assert rejected_dev is not None
    assert rejected_dev.tolist() == [2, 0, 1, 0]
    assert qsl.tolist() == [0, 1, 2, 3, 3]
    assert reqdist.tolist() == [3, 3, 3]
    # Narrower than knr: zero-padded up to it.
    rejected_dev, _, _ = proposer._chunk_loop_inputs(chunk, rejected[:3])
    assert rejected_dev is not None
    assert rejected_dev.tolist() == [2, 0, 1, 0]


# ---------------------------------------------------------------------------
# Switch on == switch off, on a toy draft
# ---------------------------------------------------------------------------


class _ToyWorld:
    """Per-request caches indexed by absolute position, plus the shared top-k
    table. `write` and `attend` are the only row couplings, as in the MTP
    layer (KV + indexer-K writes, causal sparse attention over them)."""

    def __init__(self, num_reqs: int, max_len: int, prefix_lens):
        g = torch.Generator().manual_seed(0)
        self.kv = torch.zeros(num_reqs, max_len, HIDDEN)
        self.idx = torch.zeros(num_reqs, max_len)
        for r, n in enumerate(prefix_lens):
            self.kv[r, :n] = torch.randn(n, HIDDEN, generator=g)
            self.idx[r, :n] = torch.randn(n, generator=g)
        self.topk_buf = torch.full((32, TOPK), -1, dtype=torch.int32)
        self.md = None
        self.skip_topk = False
        self.step0_rows_seen: list[int] = []
        self.kv_only_rows_seen: list[int] = []

    def rows(self, num_tokens: int, positions: torch.Tensor):
        """(row, request, position) of every row the metadata places."""
        (md,) = {id(m): m for m in self.md.values()}.values()
        qsl = md.query_start_loc.tolist()
        seq_lens = md.seq_lens.tolist()
        num_reqs = int(md.request_distribution[2])
        # The toy's "block table": entry r * PAGES_PER_REQ is request r's
        # batch-wide id, i.e. which cache rows it owns.
        owners = md.block_tables.tolist()
        out = []
        for t in range(num_tokens):
            r = next((r for r in range(num_reqs) if qsl[r] <= t < qsl[r + 1]), None)
            if r is None:
                continue
            pos = seq_lens[r] - (qsl[r + 1] - qsl[r]) + (t - qsl[r])
            # The metadata and the row's own position agree.
            assert pos == int(positions[t]), (t, r, pos, int(positions[t]))
            out.append((t, owners[r * PAGES_PER_REQ], pos))
        return out

    def write(self, input_ids, positions, hidden):
        placed = self.rows(input_ids.shape[0], positions)
        for t, r, pos in placed:
            self.kv[r, pos] = hidden[t] * 0.5 + input_ids[t] * 0.01 + pos * 0.001
            self.idx[r, pos] = hidden[t].sum() + input_ids[t] * 0.1
        return placed

    def forward(self, input_ids, positions, hidden_states, spec_step_idx=0):
        placed = self.write(input_ids, positions, hidden_states)
        out = hidden_states.clone()
        for t, r, pos in placed:
            if self.skip_topk:
                sel = self.topk_buf[t]
            else:
                scores = self.idx[r, : pos + 1]
                k = min(TOPK, pos + 1)
                sel = torch.full((TOPK,), -1, dtype=torch.int32)
                sel[:k] = torch.topk(scores, k).indices.to(torch.int32)
                self.topk_buf[t] = sel
            valid = sel[sel >= 0].long()
            out[t] = (
                hidden_states[t] * 0.9
                + self.kv[r, : pos + 1].mean(dim=0) * 0.3
                + self.kv[r, valid].sum(dim=0) * 0.2
            )
        if spec_step_idx == 0 and not self.skip_topk:
            self.step0_rows_seen.append(input_ids.shape[0])
        return out, out * 2.0


class _ToyMtpModel:
    def __init__(self, world):
        self.world = world

    def set_skip_topk(self, skip):
        self.world.skip_topk = skip

    def compact_topk_indices(self, slot_ids):
        buf = self.world.topk_buf
        buf[: slot_ids.numel()] = buf[slot_ids]


class _ToyDraft:
    def __init__(self, world):
        self.world = world
        self.model = _ToyMtpModel(world)
        g = torch.Generator().manual_seed(1)
        self.head = torch.randn(HIDDEN, VOCAB, generator=g)

    def __call__(self, *, input_ids, positions, hidden_states, spec_step_idx):
        return self.world.forward(input_ids, positions, hidden_states, spec_step_idx)

    def compute_logits(self, hidden):
        return hidden @ self.head


class _ToyLoopProgram:
    """IndexShare's loop-step program: the same forward, run with the
    indexer off (it reads the saved top-k rows)."""

    def __init__(self, world):
        self.world = world

    def __call__(self, *, input_ids, positions, hidden_states, spec_step_idx):
        assert self.world.skip_topk
        return self.world.forward(input_ids, positions, hidden_states, spec_step_idx)


class _ToyFusedLoop:
    """The fused loop program (VLLM_TPU_FUSED_DRAFT_LOOP, IndexShare off):
    draft steps 1..K-1 of one chunk in one call, each with its own indexer,
    bumping the loop metadata's seq_lens in place between steps."""

    def __init__(self, world, proposer, num_steps):
        self.world = world
        self.proposer = proposer
        self.num_steps = num_steps
        self.calls = 0

    def __call__(self, input_ids, positions, hidden_states, seq_lens):
        assert not self.world.skip_topk
        self.calls += 1
        tokens = []
        for step in range(1, self.num_steps + 1):
            last_hidden, hidden_states = self.world.forward(
                input_ids, positions, hidden_states, spec_step_idx=step
            )
            input_ids = self.proposer._draft_propose_token(last_hidden)
            tokens.append(input_ids)
            positions = positions + 1
            seq_lens.add_(1)
        return tuple(tokens)


class _ToyKvOnly:
    def __init__(self, world):
        self.world = world
        self.calls = 0

    def __call__(self, input_ids, positions, hidden_states):
        self.calls += 1
        placed = self.world.write(input_ids, positions, hidden_states)
        self.world.kv_only_rows_seen.append(len(placed))
        return hidden_states


class _Req:
    def __init__(self, tokens, num_computed):
        self.tokens = tokens
        self.num_computed_tokens = num_computed

    def get_token_id(self, i):
        return self.tokens[i]


PAGES_PER_REQ = 4
KNR = 4
GATE = 9


@dataclasses.dataclass
class _ReqSpec:
    rid: str
    context: int
    q_len: int
    rejected: int = 0
    # Sync: the sampled tokens; async: the rejection-sampler row.
    sampled: tuple = ()
    async_row: tuple = (0, -1, -1, -1)
    num_drafts: int = 0
    discard: bool = False


# Chunk 0 (16 padded tokens, at the gate or above -> two-pass), K = 3:
#   r0 verify: context 6, 1 + K = 4 tokens (positions 6..9), 2 drafts rejected;
#   r1 prefill whose prompt completes: 6 tokens (positions 0..5);
#   r2 intermediate prefill chunk: 5 tokens (positions 3..7), discarded.
_CHUNK0 = [
    _ReqSpec("r0", 6, 4, 2, (11, 12), (11, 12, -1, -1), 3),
    _ReqSpec("r1", 0, 6, 0, (13,), (13, -1, -1, -1)),
    _ReqSpec("r2", 3, 5, discard=True),
]
# A prefill-only chunk (16 padded tokens), as a prefill-only producer drafts
# it with K = 1: two prompts complete (6 tokens at positions 0..5, 5 tokens at
# 4..8) and an intermediate chunk (4 tokens at 3..6) is discarded.
_PREFILL_CHUNK = [
    _ReqSpec("p0", 0, 6, 0, (13,)),
    _ReqSpec("p1", 4, 5, 0, (14,)),
    _ReqSpec("p2", 3, 4, discard=True),
]
# Chunk 1 (8 tokens, below the gate -> single pass): two verify requests.
_CHUNK1 = [
    _ReqSpec("r3", 4, 4, 1, (21, 22, 23), (21, 22, 23, -1), 3),
    _ReqSpec("r4", 2, 4, 0, (31, 32, 33, 34), (31, 32, 33, 34), 3),
]


def _chunk(specs, start, padded, request_width=None):
    num_reqs = len(specs)
    q_lens = [sp.q_len for sp in specs]
    qsl = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    seq_lens = [sp.context + sp.q_len for sp in specs]
    positions = np.zeros(padded, dtype=np.int32)
    for r, sp in enumerate(specs):
        positions[qsl[r] : qsl[r + 1]] = np.arange(sp.context, seq_lens[r])
    real = int(qsl[-1])
    input_ids = np.zeros(padded, dtype=np.int32)
    input_ids[:real] = (np.arange(real) + 5 * start) % 7 + 3
    g = torch.Generator().manual_seed(2 + start)
    hidden = torch.randn(padded, HIDDEN, generator=g)
    owners = np.zeros(KNR * PAGES_PER_REQ, dtype=np.int32)
    owners[: num_reqs * PAGES_PER_REQ : PAGES_PER_REQ] = np.arange(
        start, start + num_reqs
    )
    md = AttentionMetadata(
        input_positions=torch.from_numpy(positions),
        block_tables=torch.from_numpy(owners),
        seq_lens=torch.tensor(seq_lens + [0] * (KNR - num_reqs), dtype=torch.int32),
        query_start_loc=torch.tensor(
            list(qsl) + [qsl[-1]] * (KNR - num_reqs), dtype=torch.int32
        ),
        request_distribution=torch.tensor([0, num_reqs, num_reqs], dtype=torch.int32),
    )
    attn_ctx = SimpleNamespace(
        use_max_model_len=True,
        seq_lens=md.seq_lens,
        query_start_loc=md.query_start_loc,
        request_distribution=md.request_distribution,
        position_ids_override=None,
    )
    return DraftChunkInputs(
        input_ids=torch.from_numpy(input_ids),
        position_ids=torch.from_numpy(positions),
        query_start_loc_np=qsl,
        attn_ctx=attn_ctx,
        start_index=start,
        num_reqs=num_reqs,
        aux_hidden_states=[],
        hidden_states=hidden,
        attn_metadata={DRAFT_LAYER: md, "target.attn": md},
        draft_lengths=torch.tensor(
            [sp.num_drafts for sp in specs]
            + [0] * ((request_width or num_reqs) - num_reqs),
            dtype=torch.int32,
        ),
    )


# "async_padded": the rejection-sampler rows at the sampler's request-padding
# width, wider than the chunk's kernel_num_reqs (KNR), as `main` hands them to
# `_prepare_draft_inputs`; the rows past num_reqs are padding.
REQUEST_WIDTH = KNR + 2


def _scenario(mode: str, num_chunks: int, chunk_specs=None):
    chunk_specs = chunk_specs or [_CHUNK0, _CHUNK1][:num_chunks]
    specs = [sp for cs in chunk_specs for sp in cs]
    width = REQUEST_WIDTH if mode == "async_padded" else None
    chunks, start = [], 0
    for cs, padded in zip(chunk_specs, (16, 8)):
        chunks.append(_chunk(cs, start, padded, width))
        start += len(cs)
    requests = {
        sp.rid: _Req(list(range(100, 120)), sp.context) for sp in specs if sp.discard
    }
    kwargs = dict(
        discard_sampled_tokens_req_indices=[
            i for i, sp in enumerate(specs) if sp.discard
        ],
        scheduler_output=SimpleNamespace(
            num_scheduled_tokens={sp.rid: sp.q_len for sp in specs},
            scheduled_spec_decode_tokens={},
        ),
    )
    if mode == "sync":
        kwargs.update(
            sampled_token_ids=[list(sp.sampled) for sp in specs],
            num_rejected_tokens_np=np.array(
                [sp.rejected for sp in specs], dtype=np.int32
            ),
        )
    else:
        # The rejection-sampler rows that give the same seeds and rejections.
        kwargs.update(
            sampled_token_ids=None,
            num_rejected_tokens_np=None,
            next_tokens_per_chunk=[
                torch.tensor(
                    [sp.async_row for sp in cs]
                    + [(0, -1, -1, -1)] * ((width or len(cs)) - len(cs)),
                    dtype=torch.int32,
                )
                for cs in chunk_specs
            ],
            return_device=True,
        )
    return chunks, requests, kwargs, [sp.context for sp in specs], specs


# Eagle3Proposer's @torch.compile(backend="tpu") helpers that propose() calls.
_COMPILED_DRAFT_HELPERS = (
    "_draft_seed_input_ids",
    "_draft_combine_hidden_states",
    "_draft_gather_rows",
    "_draft_propose_token",
)


def _eager_body(method):
    """The Python body under `_static_shape_args(torch.compile(...)(fn))`
    (idempotent: `_run` runs twice per test)."""
    compiled = getattr(method, "__wrapped__", method)
    return getattr(compiled, "_torchdynamo_orig_callable", compiled)


def _run(
    monkeypatch,
    *,
    two_pass: bool,
    mode: str,
    num_chunks: int = 1,
    loop: str = "indexshare",
    num_spec: int = 3,
    chunk_specs=None,
):
    import torch._dynamo

    torch._dynamo.reset()
    # propose() runs its small helpers in @torch.compile(backend="tpu")
    # regions, which need a TPU device; on the CPU lane run their bodies
    # eagerly (the TPU equality run covers the compiled path).
    for name in _COMPILED_DRAFT_HELPERS:
        monkeypatch.setattr(
            e3.Eagle3Proposer, name, _eager_body(getattr(e3.Eagle3Proposer, name))
        )
    chunks, requests, kwargs, prefix_lens, specs = _scenario(
        mode, num_chunks, chunk_specs
    )
    world = _ToyWorld(len(specs), 24, prefix_lens)
    K = num_spec

    proposer = make_proposer(method="mtp")
    proposer.speculative_config.num_speculative_tokens = K
    proposer.draft_vllm_config = object()
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=len(specs), req_ids=[sp.rid for sp in specs]
        ),
        requests=requests,
        device=torch.device("cpu"),
        num_tokens_paddings=[4, 8, 16, 32],
        max_num_reqs=4,
        num_reqs_max_model_len=KNR,
        num_reqs_most_model_len=None,
        uses_mrope=False,
        mesh=object(),
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda n: n,
        _has_mamba_state=False,
        _unified_kv_layout=False,
    )
    proposer._draft_attn_layer_names = {DRAFT_LAYER}
    proposer.draft_model = _ToyDraft(world)
    proposer.draft_chunks = chunks
    fused_loop = _ToyFusedLoop(world, proposer, K - 1)
    if loop == "indexshare":
        # IndexShare, on the toy's shared top-k table: the separate per-step
        # loop, each step restoring its chunk's saved step-0 rows.
        proposer._share_mtp_indices = True
        proposer._mtp_topk_buffers = [world.topk_buf]
        proposer._mtp_loop_model = _ToyLoopProgram(world)
        proposer._mtp_loop_model_warm = True
    elif loop == "fused":
        # VLLM_TPU_FUSED_DRAFT_LOOP: steps 1..K-1 in one program (IndexShare
        # off, as `_build_mtp_fused_loop` requires).
        proposer._mtp_fused_loop = fused_loop
        proposer._mtp_fused_loop_warm = False
    else:
        assert loop == "separate"

    kv_only = _ToyKvOnly(world)
    proposer._mtp_kv_only = (
        MtpKvOnlyStep0(kv_only, GATE, object()) if two_pass else None
    )

    @contextlib.contextmanager
    def fake_forward_context(attn_metadata, *_a, **_k):
        world.md = attn_metadata
        yield

    monkeypatch.setattr(e3, "set_forward_context", fake_forward_context)
    for name in ("set_model_tag", "set_vllm_model_wrapper_context"):
        monkeypatch.setattr(e3, name, lambda *a, **k: contextlib.nullcontext())

    drafts = proposer.propose(**kwargs)
    if isinstance(drafts, torch.Tensor):
        drafts = drafts.tolist()
    return SimpleNamespace(
        drafts=drafts,
        world=world,
        kv_only_calls=kv_only.calls,
        fused_calls=fused_loop.calls,
    )


@pytest.mark.parametrize("loop", ["indexshare", "separate", "fused"])
@pytest.mark.parametrize("num_chunks", [1, 2])
@pytest.mark.parametrize("mode", ["sync", "async", "async_padded"])
def test_two_pass_step0_matches_the_single_pass(
    fixed_gate, monkeypatch, mode, num_chunks, loop
):
    opts = dict(mode=mode, num_chunks=num_chunks, loop=loop)
    ref = _run(monkeypatch, two_pass=False, **opts)
    new = _run(monkeypatch, two_pass=True, **opts)
    if loop == "fused":
        # Steps 1..K-1 ran in the program for every chunk (the first call
        # runs it twice: trace, then compiled).
        assert ref.fused_calls == new.fused_calls == num_chunks + 1
        # And match the separate per-step loop.
        sep = _run(
            monkeypatch,
            two_pass=False,
            mode=mode,
            num_chunks=num_chunks,
            loop="separate",
        )
        assert new.drafts == sep.drafts
        assert torch.equal(new.world.kv, sep.world.kv)

    # The paths actually differ: one full step 0 on all 16 rows vs a KV-only
    # pass over the 15 real rows and a full pass on p = 4 rows. A second
    # chunk (8 tokens) is below the gate and keeps the single pass.
    assert ref.kv_only_calls == 0
    assert ref.world.step0_rows_seen == [16, 8][:num_chunks]
    assert new.kv_only_calls == 1
    assert new.world.kv_only_rows_seen == [15]
    assert new.world.step0_rows_seen == [4, 8][:num_chunks]

    assert new.drafts == ref.drafts
    # Same cache entries everywhere: prefix, this step's rows (rejected tail
    # included), and the K - 1 loop steps' draft entries.
    assert torch.equal(new.world.kv, ref.world.kv)
    assert torch.equal(new.world.idx, ref.world.idx)
    # The last chunk's per-request top-k rows (IndexShare) are the same.
    num_last = len(_CHUNK0 if num_chunks == 1 else _CHUNK1)
    assert torch.equal(new.world.topk_buf[:num_last], ref.world.topk_buf[:num_last])


def test_two_pass_step0_matches_the_single_pass_with_one_draft_token(
    fixed_gate, monkeypatch
):
    """K = 1, as a prefill-only producer drafts: step 0 and no loop step."""
    opts = dict(mode="sync", loop="separate", num_spec=1, chunk_specs=[_PREFILL_CHUNK])
    ref = _run(monkeypatch, two_pass=False, **opts)
    new = _run(monkeypatch, two_pass=True, **opts)
    assert ref.kv_only_calls == 0
    assert ref.world.step0_rows_seen == [16]
    assert new.kv_only_calls == 1
    assert new.world.kv_only_rows_seen == [15]
    assert new.world.step0_rows_seen == [4]
    assert new.drafts == ref.drafts
    assert torch.equal(new.world.kv, ref.world.kv)
    assert torch.equal(new.world.idx, ref.world.idx)


# ---------------------------------------------------------------------------
# EP-DP lockstep: idle ranks replay the busy ranks' programs
# ---------------------------------------------------------------------------

LOCKSTEP_GATE = 9
LOCKSTEP_K = 3


class _RecordingDraft:
    def __init__(self, trace):
        self.trace = trace

    def __call__(self, *, input_ids, positions, hidden_states, spec_step_idx):
        self.trace.append(("draft", spec_step_idx, input_ids.shape[0]))
        return hidden_states, hidden_states


class _RecordingKvOnly:
    def __init__(self, trace):
        self.trace = trace

    def __call__(self, input_ids, positions, hidden_states):
        self.trace.append(("kv_only", input_ids.shape[0]))
        return hidden_states


def _lockstep_proposer(monkeypatch, *, bucket, any_prefill, num_chunks):
    """A proposer under EP-DP lockstep at one step's DP-wide values, whose
    programs (pass 1, draft forward, lm head) and metadata builds append
    `(name, ..., num_tokens)` to the returned trace instead of running. The
    programs are where the collectives are; everything between them is
    local."""
    monkeypatch.setattr(
        e3.Eagle3Proposer,
        "_draft_gather_rows",
        _eager_body(e3.Eagle3Proposer._draft_gather_rows),
    )
    for name in ("set_model_tag", "set_vllm_model_wrapper_context"):
        monkeypatch.setattr(e3, name, lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(
        e3, "set_forward_context", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(e3, "synchronize_tensors", lambda *a, **k: None)
    trace: list[tuple] = []
    proposer = make_proposer(method="mtp")
    proposer.speculative_config.num_speculative_tokens = LOCKSTEP_K
    # A sharded draft: its collectives pair across the DP group.
    proposer._draft_replicated = False
    proposer.draft_vllm_config = object()
    proposer.runner = SimpleNamespace(
        device=torch.device("cpu"),
        uses_mrope=False,
        _hidden_states_dtype=torch.float32,
        mesh=object(),
        num_tokens_paddings=[4, 8, 16, 32],
        max_num_reqs=4,
        num_reqs_max_model_len=KNR,
        num_reqs_most_model_len=None,
        input_batch=SimpleNamespace(num_reqs=3),
        _dp_lockstep_enabled=lambda: True,
        _dp_num_tokens_across_dp=lambda n: n,
        # The step's all-reduced values (TPUModelRunner._dp_coordinated_step).
        _dp_target_bucket=bucket,
        _dp_step_max_reqs=3,
        _dp_step_num_chunks=num_chunks,
        _dp_step_any_prefill=any_prefill,
    )
    proposer._draft_attn_layer_names = {DRAFT_LAYER}
    proposer._draft_hidden_size = lambda: HIDDEN
    proposer.draft_model = _RecordingDraft(trace)
    proposer._mtp_kv_only = MtpKvOnlyStep0(
        _RecordingKvOnly(trace), LOCKSTEP_GATE, object()
    )

    def build_md(chunk, step_idx, seq_lens_delta, num_rejected_np, **kwargs):
        trace.append(("md", step_idx, seq_lens_delta, kwargs["num_tokens_padded"]))
        return {DRAFT_LAYER: None}

    def lm_head(hidden):
        trace.append(("lm_head", hidden.shape[0]))
        return torch.zeros(hidden.shape[0], dtype=torch.int32)

    proposer._build_draft_attn_metadata = build_md
    proposer._draft_propose_token = lm_head
    return proposer, trace


def _busy_chunk(q_lens, bucket):
    num_reqs = len(q_lens)
    qsl = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    return DraftChunkInputs(
        input_ids=torch.zeros(bucket, dtype=torch.int32),
        position_ids=torch.arange(bucket, dtype=torch.int32),
        query_start_loc_np=qsl,
        attn_ctx=SimpleNamespace(use_max_model_len=True),
        start_index=0,
        num_reqs=num_reqs,
        aux_hidden_states=[],
        hidden_states=torch.zeros(bucket, HIDDEN),
    )


def _busy_rank_trace(monkeypatch, *, q_lens, bucket, any_prefill, num_chunks):
    """One busy rank: one real chunk, padded with dummy chunks up to the
    step's chunk count."""
    proposer, trace = _lockstep_proposer(
        monkeypatch, bucket=bucket, any_prefill=any_prefill, num_chunks=num_chunks
    )
    chunk = _busy_chunk(q_lens, bucket)
    proposer.draft_chunks = [chunk]
    p = proposer._loop_bucket(chunk.num_reqs)

    def prepare(chunk, *args, **kwargs):
        last_token_indices = torch.zeros(p, dtype=torch.int64)
        last_token_indices[: chunk.num_reqs] = torch.from_numpy(
            chunk.query_start_loc_np[1:] - 1
        )
        rejected = np.zeros(chunk.num_reqs, dtype=np.int32)
        return chunk.input_ids, chunk.position_ids, last_token_indices, rejected

    proposer._prepare_draft_inputs = prepare
    proposer.propose(None, [], None, SimpleNamespace(), return_device=True)
    return trace


def _idle_rank_trace(monkeypatch, *, bucket, any_prefill, num_chunks):
    proposer, trace = _lockstep_proposer(
        monkeypatch, bucket=bucket, any_prefill=any_prefill, num_chunks=num_chunks
    )
    proposer.run_dp_dummy_draft(num_chunks)
    return trace


def _expected_trace(bucket, num_chunks, two_pass, p=4):
    if two_pass:
        step0 = [
            ("md", 0, 0, bucket),
            ("kv_only", bucket),
            ("md", 1, 0, p),
            ("draft", 0, p),
            ("lm_head", p),
        ]
    else:
        step0 = [("md", 0, 0, bucket), ("draft", 0, bucket), ("lm_head", p)]
    trace = step0 * num_chunks
    for step in range(1, LOCKSTEP_K):
        trace += [("md", step, step, p), ("draft", step, p), ("lm_head", p)] * (
            num_chunks
        )
    return trace


@pytest.mark.parametrize(
    "q_lens, bucket, any_prefill, two_pass",
    [
        # This rank's chunk carries a prefill span: the two passes.
        ([4, 6, 5], 16, True, True),
        # Verify only here, but a peer prefills: the step's decision wins.
        ([4, 4, 4], 16, True, True),
        # No rank prefills: the single pass, at any size.
        ([4, 4, 4], 16, False, False),
        # Below the gate: the single pass.
        ([2, 3, 2], 8, True, False),
    ],
)
def test_lockstep_idle_ranks_replay_the_busy_ranks_programs(
    fixed_gate, monkeypatch, q_lens, bucket, any_prefill, two_pass
):
    """Collectives pair across EP-DP ranks in emission order, so a busy
    rank (one real chunk and one padding chunk) and an idle rank (two
    padding chunks) must run the same programs at the same shapes, in the
    same order: pass 1 at the bucket and pass 2 at p, or the single pass at
    the bucket, then the lm head at p."""
    num_chunks = 2
    busy = _busy_rank_trace(
        monkeypatch,
        q_lens=q_lens,
        bucket=bucket,
        any_prefill=any_prefill,
        num_chunks=num_chunks,
    )
    idle = _idle_rank_trace(
        monkeypatch, bucket=bucket, any_prefill=any_prefill, num_chunks=num_chunks
    )
    assert busy == _expected_trace(bucket, num_chunks, two_pass)
    assert idle == busy


def test_lockstep_decision_ignores_the_chunks_own_prefill(fixed_gate, monkeypatch):
    """A rank whose own chunk prefills follows the DP-wide bit too: it is
    set whenever this rank prefills, so a False here is only reachable in
    this test, and the chunk keeps the single pass like its peers."""
    proposer, _ = _lockstep_proposer(
        monkeypatch, bucket=16, any_prefill=False, num_chunks=1
    )
    assert proposer._takes_two_pass_step0(_busy_chunk([4, 6, 5], 16), 16) is False
    proposer.runner._dp_step_any_prefill = True
    assert proposer._takes_two_pass_step0(_busy_chunk([4, 4, 4], 16), 16) is True


def test_lockstep_decision_reads_the_dp_wide_loop_bucket(monkeypatch):
    """Under lockstep p is the step's DP-wide loop bucket, not the chunk's:
    a peer with more requests raises the threshold on every rank."""
    monkeypatch.setattr(mtp_kv_only, "_TWO_PASS_MIN_TOKENS", {4: 16, 16: 32})
    proposer, _ = _lockstep_proposer(
        monkeypatch, bucket=16, any_prefill=True, num_chunks=1
    )
    chunk = _busy_chunk([4, 6, 5], 16)
    assert proposer._takes_two_pass_step0(chunk, 16) is True  # p = 4
    proposer.runner._dp_step_max_reqs = 9  # a peer's chunk: p = 16
    assert proposer._takes_two_pass_step0(chunk, 16) is False
    assert proposer._dp_two_pass_step0() is False
    proposer.runner._dp_step_max_reqs = 3
    with proposer.single_pass_step0():
        assert proposer._dp_two_pass_step0() is False


# ---------------------------------------------------------------------------
# On TPU: the KV-only pass and the two-pass step 0 vs the full MTP step 0
# ---------------------------------------------------------------------------

# Own op names: test_mla_kv_write_op.py builds the same layer under its own.
TPU_NAMES = ("kv_only_test.mtp.mla_attn", "kv_only_test.mtp.indexer.k_cache")


class _AttentionOutput(nn.Module):
    """The MTP decoder layer up to its attention output: the input norm, then
    the wrapper's forward (indexer, sparse MLA). Its zero residual and the
    identity shared head leave that output as the MTP layer's output."""

    def __init__(self, block):
        super().__init__()
        self.input_layernorm = block.input_layernorm
        self.mla_attn = block.self_attn.mla_attn
        self.use_sequence_parallel_moe = False

    def forward(self, positions, hidden_states, residual):
        out = self.mla_attn(positions, self.input_layernorm(hidden_states))
        return out, torch.zeros_like(out)


class _Step0(nn.Module):
    """Draft step 0 as the full pass runs it, up to the attention output, on
    the upstream predictor and MTP layer forwards."""

    def __init__(self, mtp):
        super().__init__()
        layer = mtp.model.layers["78"]
        step0_layer = mtp_kv_only._with_child(
            layer, "mtp_block", _AttentionOutput(layer.mtp_block)
        )
        self.predictor = mtp_kv_only._with_child(
            mtp.model, "layers", nn.ModuleDict({"78": step0_layer})
        )

    def forward(self, input_ids, positions, hidden_states):
        out, _ = self.predictor(input_ids, positions, hidden_states)
        return out


@pytest.fixture(scope="module")
def tpu_mtp():
    """A synthetic single-layer DSA MTP draft on one TPU chip (the shared
    sparse MLA layer under the upstream MTP prefix), built once per module:
    the ops register under fixed names. Request 0 has `TPU_PREFIX` tokens
    cached before the step (written here by the full step 0); the step is a
    64-token chunk: request 0's next 32 tokens, request 1's first 24, 8
    padded rows."""
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    torch._dynamo.reset()
    dev = torch.device("tpu")
    built = mla.build_tpu_sparse_mla(*TPU_NAMES)
    self_attn = _module(DeepseekV2MLAAttention, mla_attn=built.wrapper)
    mtp = _upstream_mtp(self_attn, hidden=mla.HIDDEN)
    mtp.model.to(dev)
    assert mtp_kv_only.kv_only_unsupported_reason(mtp) is None
    with set_vllm_model_wrapper_context(mesh=built.mesh):
        kv_only = mtp_kv_only.MtpKvOnlyModel(
            vllm_config=_uncompiled_config(),
            prefix="mtp_prefill_kv_only",
            mtp=mtp,
            mesh=built.mesh,
        )
    built.step0_c = mla.tpu_compiled(_Step0(mtp))
    built.kv_only_c = mla.tpu_compiled(kv_only)
    gen = torch.Generator().manual_seed(9)

    def tokens(positions):
        n = len(positions)
        return (
            torch.randint(0, MTP_VOCAB, (n,), generator=gen, dtype=torch.int32).to(dev),
            torch.tensor(positions, dtype=torch.int32, device=dev),
            torch.randn(n, mla.HIDDEN, generator=gen).bfloat16().to(dev),
        )

    prefix_len = mla.TPU_PREFIX
    end = prefix_len + 32
    prefix_md = mla.tpu_md(
        TPU_NAMES, [prefix_len, 0], [0, prefix_len, prefix_len], [0, 0, 1]
    )
    mla.on_tpu(built, built.step0_c, prefix_md, *tokens(list(range(prefix_len))))
    # Host copies: a deferred on-device clone can lose its source to the
    # next op that donates the cache.
    built.prefix_caches = [c.cpu() for c in built.caches]
    built.end = end
    built.step = tokens(list(range(prefix_len, end)) + list(range(24)) + [0] * 8)
    built.step_qsl = [0, 32, 56]
    built.step_md = lambda: mla.tpu_md(TPU_NAMES, [end, 24], built.step_qsl, [0, 0, 2])
    # Pass 2's rows: each request's last row, padded to p = 8 with row 0.
    built.last_rows = torch.tensor([31, 55] + [0] * 6, dtype=torch.int64, device=dev)
    built.rows_md = lambda: mla.tpu_md(TPU_NAMES, [end, 24], [0, 1, 2], [2, 2, 2])
    return built


def _tpu_single_pass(built):
    mla.reset_caches(built, built.prefix_caches)
    out = mla.on_tpu(built, built.step0_c, built.step_md(), *built.step)
    return out.cpu(), [c.cpu() for c in built.caches]


def _selected(rows):
    return [{int(i) for i in row if i >= 0} for row in rows]


@pytest.mark.tpu_test
def test_kv_only_pass_writes_the_full_step0_cache_entries_on_tpu(tpu_mtp):
    """`MtpKvOnlyModel` (MTP prefix + `MlaKvWriteOp`), compiled for TPU with
    the default writers and an fp8 latent cache, leaves the same MLA and
    indexer-K caches as the full step 0 over the same rows."""
    _, want = _tpu_single_pass(tpu_mtp)
    assert any(not torch.equal(w, p) for w, p in zip(want, tpu_mtp.prefix_caches)), (
        "the step wrote nothing"
    )
    mla.reset_caches(tpu_mtp, tpu_mtp.prefix_caches)
    mla.on_tpu(tpu_mtp, tpu_mtp.kv_only_c, tpu_mtp.step_md(), *tpu_mtp.step)
    mla.assert_same_bytes(tpu_mtp.caches, want, "KV-only pass vs full step 0")


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "rejected, index_share",
    [
        pytest.param(0, False, id="all-accepted"),
        # Request 0's last 2 tokens are rejected drafts: pass 1 still writes
        # their rows, as the single pass does, and pass 2 narrows request 0's
        # window back to its accepted prefix.
        pytest.param(2, True, id="rejected-indexshare"),
    ],
)
def test_proposer_two_pass_step0_matches_the_single_pass_on_tpu(
    monkeypatch, tpu_mtp, rejected, index_share
):
    """`Eagle3Proposer._two_pass_step0` on TPU: pass 1 over every row, the
    compiled `_draft_gather_rows`, then step 0 on one row per request with
    the pass-2 metadata it narrows from the chunk's own. Same caches as the
    single pass, and each request's last accepted row gets the same attention
    output (within a tolerance: pass 2 runs the decode path of the kernels)
    and the same top-k, which is what IndexShare saves for the loop steps."""
    from vllm.forward_context import ForwardContext, override_forward_context

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    built = tpu_mtp
    want_out, want_caches = _tpu_single_pass(built)
    # Each request's last accepted row: request 0's context (352) is longer
    # than the top-k (256), so its indexer really selects.
    last = [31 - rejected, 55]
    want_topk = _selected(built.topk_rows[last].cpu())
    mla.reset_caches(built, built.prefix_caches)

    @contextlib.contextmanager
    def forward_context(attn_metadata, *_args, **_kwargs):
        ctx = ForwardContext(
            no_compile_layers={built.mla_layer.layer_name: built.mla_layer},
            attn_metadata=attn_metadata,
            slot_mapping={},
        )
        with override_forward_context(ctx):
            yield

    monkeypatch.setattr(e3, "set_forward_context", forward_context)
    monkeypatch.setattr(
        e3,
        "set_vllm_model_wrapper_context",
        lambda **_kwargs: set_vllm_model_wrapper_context(mesh=built.mesh),
    )
    proposer = make_proposer(method="mtp")
    proposer.runner = SimpleNamespace(
        device=torch.device("tpu"),
        mesh=built.mesh,
        num_reqs_max_model_len=2,
        num_reqs_most_model_len=None,
        num_tokens_paddings=[8, 64, 512],
        _dp_num_tokens_across_dp=lambda n: None,
        _dp_lockstep_enabled=lambda: False,
    )
    proposer.draft_vllm_config = object()
    proposer._draft_attn_layer_names = set(TPU_NAMES)
    skips = []

    class _Draft:
        """The full step 0, with the IndexShare switch a DSA MTP draft has."""

        model = SimpleNamespace(set_skip_topk=skips.append)

        def __call__(self, *, input_ids, positions, hidden_states, spec_step_idx):
            return built.step0_c(input_ids, positions, hidden_states)

    class _LoopStepModel:
        """IndexShare's loop-step program (indexer off): step 0 never runs it."""

        def __call__(self, **_kwargs):
            raise AssertionError("draft step 0 ran the loop-step program")

    proposer.draft_model = _Draft()
    proposer._share_mtp_indices = index_share
    proposer._mtp_loop_model = _LoopStepModel() if index_share else None
    proposer._mtp_topk_buffers = [built.topk_rows] if index_share else []
    proposer._mtp_kv_only = MtpKvOnlyStep0(built.kv_only_c, 9, object())
    ids, positions, hidden = built.step
    md = built.step_md()
    chunk = DraftChunkInputs(
        input_ids=ids,
        position_ids=positions,
        query_start_loc_np=np.array(built.step_qsl, dtype=np.int32),
        attn_ctx=SimpleNamespace(
            use_max_model_len=True,
            seq_lens=md[TPU_NAMES[0]].seq_lens,
            position_ids_override=None,
        ),
        start_index=0,
        num_reqs=2,
        aux_hidden_states=[],
        hidden_states=hidden,
        attn_metadata=md,
    )
    last_rows = torch.tensor(
        last + [0] * 6, dtype=torch.int64, device=built.last_rows.device
    )
    with torch.no_grad():
        last_hidden, _, rows_positions, _, owner_mask = proposer._two_pass_step0(
            chunk=chunk,
            input_ids=ids,
            positions=positions,
            target_hidden_states=hidden,
            last_token_indices=last_rows,
            num_rejected=np.array([rejected, 0], dtype=np.int32),
        )
    mla.assert_same_bytes(built.caches, want_caches, "two-pass vs single pass")
    assert rows_positions.cpu()[:2].tolist() == [built.end - 1 - rejected, 23]
    assert owner_mask.cpu().all()
    # Pass 2 runs the decode path of the top-k, which may order the selected
    # positions differently: compare them as sets. With IndexShare, these rows
    # are what `_mtp_save_topk` keeps for the loop steps.
    assert _selected(built.topk_rows[:2].cpu()) == want_topk
    if index_share:
        # Pass 2 ran the draft with its indexer on.
        assert skips == [False]
    torch.testing.assert_close(
        last_hidden.cpu()[:2], want_out[last], rtol=1e-2, atol=1e-2
    )
