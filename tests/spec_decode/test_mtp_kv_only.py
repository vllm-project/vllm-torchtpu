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

import numpy as np
import pytest
import torch
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
from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackendImpl
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.spec_decode import mtp_kv_only
from vllm_torchtpu.spec_decode.eagle3 import DraftChunkInputs
from vllm_torchtpu.spec_decode.mtp_kv_only import MtpKvOnlyStep0
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


def _gate_proposer(min_tokens=8):
    proposer = make_proposer(method="mtp")
    proposer.speculative_config.num_speculative_tokens = 3
    proposer._mtp_kv_only = MtpKvOnlyStep0(object(), min_tokens, object())
    return proposer


def _qsl(q_lens):
    return SimpleNamespace(
        query_start_loc_np=np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
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
def test_takes_two_pass_step0_gate(q_lens, num_tokens, expected):
    proposer = _gate_proposer()
    assert proposer._takes_two_pass_step0(_qsl(q_lens), num_tokens) is expected


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
    logs = []
    monkeypatch.setattr(
        mtp_kv_only.logger, "info", lambda msg, *args: logs.append(msg % args)
    )
    step0 = MtpKvOnlyStep0.build(proposer, _attn_layers())
    # One token above the largest loop bucket (32).
    assert step0.min_tokens == 33
    assert step0.takes_two_pass(33, True) and not step0.takes_two_pass(32, True)
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
    assert ">= 33 padded tokens" in log


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


@pytest.mark.parametrize(
    "setup, message",
    [
        ("eagle3", "draft method is 'eagle3'"),
        ("kv_shared", "reads the target's KV cache"),
        ("pcp", "prefill context parallelism"),
        ("unsupported", "not DSA"),
        ("two_mla_layers", "not one of each"),
        ("no_indexer_cache", "not one of each"),
        ("loop_bucket_is_the_largest", "the largest is 4096"),
    ],
)
def test_build_refusals_raise(monkeypatch, setup, message):
    """The switch is explicit: a draft or layout that cannot take the
    two-pass step 0 raises at load instead of keeping the single pass."""
    proposer = _build_proposer(method="eagle3" if setup == "eagle3" else "mtp")
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
    if setup == "pcp":
        proposer.runner._pcp_enabled = True
    if setup == "loop_bucket_is_the_largest":
        # max_num_reqs pads to the largest bucket, so no chunk could ever
        # reach the gate one past it.
        proposer.runner.max_num_reqs = 4096
    # A draft or layout it does not support yet is NotImplementedError; a
    # bucket set that cannot reach the gate is a bad value.
    error = ValueError if setup == "loop_bucket_is_the_largest" else NotImplementedError
    with pytest.raises(error, match="TPU_MTP_KV_ONLY_STEP0 is set") as e:
        MtpKvOnlyStep0.build(proposer, layers)
    assert message in str(e.value)
    # Every refusal is decided before the op is built (and registered).
    assert built == []


def test_build_under_ep_dp_lockstep(monkeypatch):
    """The idle ranks replay the two passes (see the lockstep tests below)."""
    proposer = _build_proposer()
    built = _patch_builder(monkeypatch)
    proposer._dp_lockstep_sharded = lambda: True
    assert MtpKvOnlyStep0.build(proposer, _attn_layers()).min_tokens == 33
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
    """T: every token bucket at or above the gate; p: every loop bucket."""
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
        (1024, 32),
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


def _scenario(mode: str, num_chunks: int):
    chunk_specs = [_CHUNK0, _CHUNK1][:num_chunks]
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
    chunks, requests, kwargs, prefix_lens, specs = _scenario(mode, num_chunks)
    world = _ToyWorld(len(specs), 24, prefix_lens)
    K = 3

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
def test_two_pass_step0_matches_the_single_pass(monkeypatch, mode, num_chunks, loop):
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
    monkeypatch, q_lens, bucket, any_prefill, two_pass
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


def test_lockstep_decision_ignores_the_chunks_own_prefill(monkeypatch):
    """A rank whose own chunk prefills follows the DP-wide bit too: it is
    set whenever this rank prefills, so a False here is only reachable in
    this test, and the chunk keeps the single pass like its peers."""
    proposer, _ = _lockstep_proposer(
        monkeypatch, bucket=16, any_prefill=False, num_chunks=1
    )
    assert proposer._takes_two_pass_step0(_busy_chunk([4, 6, 5], 16), 16) is False
    proposer.runner._dp_step_any_prefill = True
    assert proposer._takes_two_pass_step0(_busy_chunk([4, 4, 4], 16), 16) is True
