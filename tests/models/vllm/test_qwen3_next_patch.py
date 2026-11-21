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
"""Pin upstream Qwen3.5 MoE forwards and verify the trace-stable SP forward
patch and the FP8 fusions."""

import contextlib
import hashlib
import importlib
import inspect
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm
import vllm.model_executor.custom_op as custom_op_mod
from vllm.config import CompilationConfig, CompilationMode
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models import qwen3_5, qwen3_5_mtp, qwen3_next, qwen3_next_mtp

import vllm_torchtpu.envs as envs
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.models.qwen3_5 import hooks as qwen3_5_hooks
from vllm_torchtpu.models.vllm import qwen3_next_patch as patch_mod
from vllm_torchtpu.models.vllm.qwen3_5_mtp import UPSTREAM_PREDICTOR_FORWARD
from vllm_torchtpu.models.vllm.qwen3_next_patch import (
    UPSTREAM_DECODER_LAYER_FORWARD,
    UPSTREAM_MODEL_FORWARD,
    UPSTREAM_MTP_FORWARD,
    _check_tokens_divide_by_tp,
    compute_fused_gate_and_shared_expert,
    entry_chunk_decoder_layer_forward,
    entry_chunk_model_forward,
    maybe_flip_gdn_in_proj_qkvz_weight,
    maybe_patch_qwen3_next,
    maybe_setup_fused_gate_se,
    patched_decoder_layer_forward,
    patched_model_forward,
    patched_mtp_forward,
    patched_sparse_moe_forward,
    step0_post_attn_gather_context,
)

# The Qwen3.5 drafter forward maybe_patch_qwen3_next replaces, captured before
# any test installs the patch and after the qwen3_5_mtp.py import above, which
# installs its own copy over upstream's.
UNPATCHED_QWEN3_5_MTP_FORWARD = qwen3_5_mtp.Qwen3_5MultiTokenPredictor.forward

# The patched copies track vLLM v0.33.0 (pyproject.toml pin).
PINNED_VLLM_VERSION = "0.33.0"
PINNED_DECODER_LAYER_HASH = (
    "359c047f120f585e0c519a3bc438f037f996f4d48c7b64495dadddc59a709b4a"
)
PINNED_MODEL_FORWARD_HASH = (
    "a9c1265f14e88e54138394b9038db621813bb6357a9077ad50d1691db0ac4398"
)
# patched_mtp_forward copies this Qwen3Next drafter forward, and tpu_platform
# admits MTP under SP because its layer slices the rows after attention and it
# all-gathers after the final norm.
PINNED_MTP_FORWARD_HASH = (
    "54034d76e48fccf2cd98bcf05b831207580ad9e8886e6e387193e1bd70e9e472"
)
# qwen3_5_mtp.py replaces this upstream drafter forward with a copy that
# keeps its SP chunk before the MTP layer and all-gather after the final
# norm, which is what lets tpu_platform admit MTP under SP.
# maybe_patch_qwen3_next replaces it with patched_mtp_forward on a Qwen3.5
# target.
PINNED_QWEN3_5_MTP_FORWARD_HASH = (
    "af988ccd1ba49bf2953bb9858d7a2082c405d260d2ebb64413c6707c34a80d22"
)
requires_pinned_vllm = pytest.mark.skipif(
    vllm.__version__ != PINNED_VLLM_VERSION,
    reason=f"patched forwards are copies of vLLM {PINNED_VLLM_VERSION}",
)


def _source_hash(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()


@pytest.fixture
def restore_forwards(monkeypatch):
    """Start from the unpatched forwards and undo whatever the patch installs."""
    monkeypatch.setattr(
        qwen3_next.Qwen3NextDecoderLayer, "forward", UPSTREAM_DECODER_LAYER_FORWARD
    )
    monkeypatch.setattr(qwen3_next.Qwen3NextModel, "forward", UPSTREAM_MODEL_FORWARD)
    monkeypatch.setattr(
        qwen3_next_mtp.Qwen3NextMultiTokenPredictor, "forward", UPSTREAM_MTP_FORWARD
    )
    monkeypatch.setattr(
        qwen3_5_mtp.Qwen3_5MultiTokenPredictor, "forward", UNPATCHED_QWEN3_5_MTP_FORWARD
    )
    for layer_cls in (qwen3_next.Qwen3NextDecoderLayer, qwen3_5.Qwen3_5DecoderLayer):
        if "__init__" in vars(layer_cls):
            monkeypatch.setattr(layer_cls, "__init__", layer_cls.__init__)
    monkeypatch.setattr(
        patch_mod.sequence_parallel,
        "_FOLLOWS_STEP_FORM",
        patch_mod.sequence_parallel._FOLLOWS_STEP_FORM,
    )
    monkeypatch.setattr(patch_mod.sequence_parallel, "_HAS_DRAFTER", False)
    return monkeypatch


# The model configs the Qwen3.5 rule tells apart. QWEN3_5 is the checkpoint
# the patch was measured on (Qwen3.5-397B-A17B's text config, on TPU7x through
# the autouse fixture below); QWEN3_5_35B is main's CI model.
_SHAPE_397B = {"num_experts": 512, "hidden_size": 4096, "num_hidden_layers": 60}
_SHAPE_35B = {"num_experts": 256, "hidden_size": 2048, "num_hidden_layers": 40}
QWEN3_5 = SimpleNamespace(
    architectures=["Qwen3_5MoeForConditionalGeneration"],
    hf_text_config=SimpleNamespace(**_SHAPE_397B),
)
QWEN3_5_35B = SimpleNamespace(
    architectures=["Qwen3_5MoeForConditionalGeneration"],
    hf_text_config=SimpleNamespace(**_SHAPE_35B),
)
QWEN3_NEXT = SimpleNamespace(architectures=["Qwen3NextForCausalLM"])


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    """The validation entries' record runs were on TPU7x; a test that needs
    another device overrides it."""
    monkeypatch.setattr(qwen3_5_hooks, "_chip_family", lambda: "7x")


@pytest.fixture
def v6_install(restore_forwards):
    """Records whether the Qwen3.5 FP8 fusions are installed, without patching
    the shared classes for the rest of the session."""
    install = MagicMock()
    restore_forwards.setattr(patch_mod, "install_qwen3_next_v6_patches", install)
    return install


def _sequence_parallel(monkeypatch, *, min_rows: int = 0, fp8_gather: bool = False):
    monkeypatch.setattr(patch_mod.sequence_parallel, "min_rows", lambda: min_rows)
    monkeypatch.setattr(patch_mod.sequence_parallel, "fp8_gather", lambda: fp8_gather)


@requires_pinned_vllm
def test_upstream_qwen3_next_forward_unchanged():
    """Fail when upstream vLLM modifies either method, requiring a re-sync."""
    assert _source_hash(UPSTREAM_DECODER_LAYER_FORWARD) == PINNED_DECODER_LAYER_HASH, (
        "Upstream Qwen3NextDecoderLayer.forward has drifted. "
        "Re-sync src/vllm_torchtpu/models/vllm/qwen3_next_patch.py."
    )
    assert _source_hash(UPSTREAM_MODEL_FORWARD) == PINNED_MODEL_FORWARD_HASH, (
        "Upstream Qwen3NextModel.forward has drifted. "
        "Re-sync src/vllm_torchtpu/models/vllm/qwen3_next_patch.py."
    )


@requires_pinned_vllm
def test_upstream_mtp_forwards_unchanged():
    """Pin the Qwen3Next drafter forward patched_mtp_forward copies and replaces."""
    assert _source_hash(UPSTREAM_MTP_FORWARD) == PINNED_MTP_FORWARD_HASH, (
        "Upstream Qwen3NextMultiTokenPredictor.forward has drifted. "
        "Re-sync patched_mtp_forward in qwen3_next_patch.py."
    )


@requires_pinned_vllm
def test_upstream_qwen3_5_mtp_forward_unchanged():
    """Pin the upstream Qwen3.5 drafter forward that qwen3_5_mtp.py copies and
    patched_mtp_forward replaces."""
    source = inspect.getsource(UPSTREAM_PREDICTOR_FORWARD)
    actual_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert actual_hash == PINNED_QWEN3_5_MTP_FORWARD_HASH, (
        "Upstream Qwen3_5MultiTokenPredictor.forward has drifted. Re-sync "
        "src/vllm_torchtpu/models/vllm/qwen3_5_mtp.py and patched_mtp_forward "
        "in qwen3_next_patch.py, and re-check that they still chunk or slice "
        "and all-gather under SP before admitting MTP in tpu_platform.py."
    )


def test_qwen3_5_inherits_forwards():
    """Verify Qwen3.5 classes inherit the patched Qwen3Next forwards."""

    assert (
        qwen3_5.Qwen3_5DecoderLayer.forward is qwen3_next.Qwen3NextDecoderLayer.forward
    )
    assert qwen3_5.Qwen3_5Model.forward is qwen3_next.Qwen3NextModel.forward


def test_moe_block_accepts_already_sequence_parallel():
    """Verify Qwen3NextSparseMoeBlock.forward still accepts the SP keyword."""
    params = inspect.signature(qwen3_next.Qwen3NextSparseMoeBlock.forward).parameters
    assert "already_sequence_parallel" in params
    params = inspect.signature(patched_sparse_moe_forward).parameters
    assert "already_sequence_parallel" in params


@pytest.mark.parametrize(
    "forward", [patched_decoder_layer_forward, entry_chunk_decoder_layer_forward]
)
def test_patched_decoder_layer_forward_parity(forward):
    """Verify patched forward matches upstream output on the non-SP branch."""

    class DummyLayer:
        def __init__(self):
            self.use_attn_reduce_scatter_for_moe = False
            self.layer_type = "linear_attention"
            self.layer_scale = False

        def input_layernorm(self, x, residual=None):
            if residual is None:
                return x * 1.05
            return x * 1.05, residual + x * 0.05

        def linear_attn(self, hidden_states):
            return hidden_states + 1.0

        def post_attention_layernorm(self, x, residual):
            return x * 1.1, residual + x * 0.1

        def mlp(self, x, already_sequence_parallel=False):
            return x * 2.0

    layer = DummyLayer()
    hidden = torch.randn(8, 64)
    residual = torch.randn(8, 64)
    positions = torch.arange(8)

    patched_h, patched_r = forward(layer, hidden, residual, positions)
    upstream_h, upstream_r = UPSTREAM_DECODER_LAYER_FORWARD(
        layer, hidden, residual, positions
    )
    torch.testing.assert_close(patched_h, upstream_h)
    torch.testing.assert_close(patched_r, upstream_r)


def _sp_dummy_layer(attn_calls, mlp_kwargs):
    class DummyLayer:
        use_attn_reduce_scatter_for_moe = True
        layer_type = "linear_attention"
        layer_scale = False

        def input_layernorm(self, x, residual=None):
            return x * 1.05 if residual is None else (x * 1.05, residual + x * 0.05)

        def linear_attn(self, hidden_states, **kwargs):
            attn_calls.append(kwargs)
            return hidden_states.float() + 1.0

        def post_attention_layernorm(self, x, residual):
            return x * 1.1, residual + x * 0.1

        def mlp(self, x, **kwargs):
            mlp_kwargs.append(kwargs)
            return x * 2.0

    return DummyLayer()


def _fake_tp_collectives(monkeypatch, tp, all_reduced=None):
    monkeypatch.setattr(
        patch_mod,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x] * tp, dim=dim),
    )
    monkeypatch.setattr(
        patch_mod,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim=0: x.narrow(dim, 0, x.shape[dim] // tp) * tp,
    )

    def all_reduce(x):
        if all_reduced is not None:
            all_reduced.append(x)
        return x * tp

    monkeypatch.setattr(patch_mod, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(patch_mod, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(patch_mod, "get_tensor_model_parallel_rank", lambda: 0)


def test_patched_decoder_layer_forward_sequence_parallel(monkeypatch):
    """Verify the SP branch runs all-gather, reduce-scatter, and SP MLP."""

    tp = 2
    _fake_tp_collectives(monkeypatch, tp)
    _sequence_parallel(monkeypatch, min_rows=0)

    attn_calls: list[dict[str, object]] = []
    mlp_kwargs: list[dict[str, object]] = []
    layer = _sp_dummy_layer(attn_calls, mlp_kwargs)
    num_tokens = 8
    hidden = torch.randn(num_tokens // tp, 64)
    residual = torch.randn(num_tokens // tp, 64)
    positions = torch.arange(num_tokens)

    patched_h, patched_r = patched_decoder_layer_forward(
        layer, hidden, residual, positions
    )

    normed = hidden * 1.05
    expected_r = residual + hidden * 0.05
    attn_shard = (normed + 1.0) * tp
    expected_h = (attn_shard * 1.1) * 2.0
    expected_r = expected_r + attn_shard * 0.1

    assert patched_h.shape == (num_tokens // tp, 64)
    assert patched_r.shape == (num_tokens // tp, 64)
    torch.testing.assert_close(patched_h, expected_h)
    torch.testing.assert_close(patched_r, expected_r)
    assert attn_calls == [{}]
    assert mlp_kwargs == [{"already_sequence_parallel": True}]


def test_patched_decoder_layer_forward_replicated_below_min_rows(monkeypatch):
    """A step below min_rows keeps its rows and all-reduces the attention out."""

    tp = 2
    all_reduced: list[torch.Tensor] = []
    _fake_tp_collectives(monkeypatch, tp, all_reduced)
    _sequence_parallel(monkeypatch, min_rows=16)

    attn_calls: list[dict[str, object]] = []
    mlp_kwargs: list[dict[str, object]] = []
    layer = _sp_dummy_layer(attn_calls, mlp_kwargs)
    num_tokens = 8
    hidden = torch.randn(num_tokens, 64)
    residual = torch.randn(num_tokens, 64)
    positions = torch.arange(num_tokens)

    patched_h, patched_r = patched_decoder_layer_forward(
        layer, hidden, residual, positions
    )

    attn_out = (hidden * 1.05 + 1.0) * tp
    torch.testing.assert_close(patched_h, attn_out * 1.1 * 2.0)
    torch.testing.assert_close(patched_r, residual + hidden * 0.05 + attn_out * 0.1)
    assert len(all_reduced) == 1
    assert mlp_kwargs == [{"already_sequence_parallel": True}]


def test_patched_decoder_layer_forward_fp8_gather(monkeypatch):
    """With fp8 gather on, attention receives the gathered rows pre-quantized."""

    tp = 2
    _fake_tp_collectives(monkeypatch, tp)
    _sequence_parallel(monkeypatch, min_rows=0, fp8_gather=True)
    # A bf16 stand-in for the fp8 rows keeps the fake collectives dtype-agnostic.
    monkeypatch.setattr(
        patch_mod.linear_common,
        "quantize_rows",
        lambda x: (x.to(torch.bfloat16), torch.ones(x.shape[0], 1)),
    )

    attn_calls: list[dict[str, object]] = []
    layer = _sp_dummy_layer(attn_calls, [])
    # Its attention's main projection takes pre-quantized rows.
    layer._pre_norm_quant_eligible = True
    num_tokens = 8
    hidden = torch.randn(num_tokens // tp, 64)
    residual = torch.randn(num_tokens // tp, 64)

    patched_decoder_layer_forward(layer, hidden, residual, torch.arange(num_tokens))

    (kwargs,) = attn_calls
    x_q, x_s = kwargs["prequant"]
    assert x_q.dtype == torch.bfloat16
    assert x_q.shape == (num_tokens, 64)
    assert x_s.shape == (num_tokens, 1)


@pytest.mark.parametrize("marked", [False, None], ids=["bf16_projection", "unmarked"])
def test_a_sliced_layer_without_fp8_projections_gathers_bf16_rows(monkeypatch, marked):
    """A layer whose attention's main projection does not take pre-quantized
    rows (a bf16 checkpoint, say) gathers its bf16 rows on a sliced step, as
    on main, and attention gets no prequant."""
    tp = 2
    _fake_tp_collectives(monkeypatch, tp)
    _sequence_parallel(monkeypatch, min_rows=0, fp8_gather=True)
    quantized = []
    monkeypatch.setattr(
        patch_mod.linear_common,
        "quantize_rows",
        lambda x: quantized.append(x) or (x, torch.ones(x.shape[0], 1)),
    )
    gathered = _record_all_gathers(monkeypatch)

    attn_calls: list[dict[str, object]] = []
    received = []
    layer = _sp_dummy_layer(attn_calls, [])
    if marked is not None:
        layer._pre_norm_quant_eligible = marked
    linear_attn = layer.linear_attn
    layer.linear_attn = lambda hidden_states, **kwargs: (
        received.append(hidden_states),
        linear_attn(hidden_states, **kwargs),
    )[1]
    num_tokens = 8
    hidden = torch.randn(num_tokens // tp, 64, dtype=torch.bfloat16)
    residual = torch.randn(num_tokens // tp, 64, dtype=torch.bfloat16)

    patched_decoder_layer_forward(layer, hidden, residual, torch.arange(num_tokens))

    assert attn_calls == [{}]
    assert quantized == []
    assert gathered == [num_tokens // tp]
    (rows,) = received
    assert rows.dtype == torch.bfloat16 and rows.shape == (num_tokens, 64)
    torch.testing.assert_close(rows, torch.cat([hidden * 1.05] * tp))


def _record_all_gathers(monkeypatch) -> list[int]:
    """Record the rows each (fake) all-gather receives."""
    rows: list[int] = []
    gather = patch_mod.tensor_model_parallel_all_gather

    def all_gather(x, dim=0):
        rows.append(x.shape[dim])
        return gather(x, dim)

    monkeypatch.setattr(patch_mod, "tensor_model_parallel_all_gather", all_gather)
    return rows


def test_patched_decoder_layer_forward_first_layer_replicated(monkeypatch):
    """The first layer (residual None) runs attention on the replicated rows
    without an all-gather, then the reduce-scatter and the residual both leave
    this rank's slice for the MoE."""

    tp = 2
    _fake_tp_collectives(monkeypatch, tp)
    gathered = _record_all_gathers(monkeypatch)
    _sequence_parallel(monkeypatch, min_rows=0, fp8_gather=True)

    attn_calls: list[dict[str, object]] = []
    mlp_kwargs: list[dict[str, object]] = []
    layer = _sp_dummy_layer(attn_calls, mlp_kwargs)
    layer._tpu_sp_rank = torch.tensor([0], dtype=torch.int32)
    num_tokens = 8
    chunk = num_tokens // tp
    hidden = torch.randn(num_tokens, 64)

    patched_h, patched_r = patched_decoder_layer_forward(
        layer, hidden, None, torch.arange(num_tokens)
    )

    attn_shard = (hidden[:chunk] * 1.05 + 1.0) * tp
    torch.testing.assert_close(patched_h, attn_shard * 1.1 * 2.0)
    torch.testing.assert_close(patched_r, hidden[:chunk] + attn_shard * 0.1)
    assert gathered == []
    # Replicated rows reach attention unquantized, even with fp8 gather on.
    assert attn_calls == [{}]
    assert mlp_kwargs == [{"already_sequence_parallel": True}]


def test_patched_model_forward_sequence_parallel(monkeypatch):
    """The model does not chunk at entry: the first layer gets every row and
    slices them, later layers get the slice, and the final norm's output (with
    the aux hidden states) is all-gathered back to every row."""

    tp = 2
    _fake_tp_collectives(monkeypatch, tp)
    gathered = _record_all_gathers(monkeypatch)
    _sequence_parallel(monkeypatch, min_rows=0)
    monkeypatch.setattr(
        patch_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    seen: list[tuple[int, bool]] = []

    class Layer:
        def __init__(self):
            self.inner = _sp_dummy_layer([], [])
            self._tpu_sp_rank = torch.tensor([0], dtype=torch.int32)
            self.inner._tpu_sp_rank = self._tpu_sp_rank

        def __call__(self, positions, hidden_states, residual):
            seen.append((hidden_states.shape[0], residual is None))
            return patched_decoder_layer_forward(
                self.inner, hidden_states, residual, positions
            )

    def add_aux(aux, layer_idx, hidden_states, residual):
        if layer_idx in (0, 2):
            aux.append(hidden_states if residual is None else hidden_states + residual)
        return aux

    num_tokens = 8
    chunk = num_tokens // tp
    embeds = torch.randn(num_tokens, 64)
    model = SimpleNamespace(
        use_sequence_parallel=True,
        start_layer=0,
        end_layer=2,
        layers=[Layer(), Layer()],
        embed_input_ids=lambda ids: embeds,
        norm=lambda h, r: (h + r, r),
        _maybe_add_hidden_state=add_aux,
    )

    hidden, aux = patched_model_forward(
        model, torch.zeros(num_tokens, dtype=torch.long), torch.arange(num_tokens)
    )

    assert seen == [(num_tokens, True), (chunk, False)]
    # Layer 1's attention input, then the final hidden + aux gather.
    assert gathered == [chunk, chunk]
    assert hidden.shape == (num_tokens, 64)
    assert [a.shape for a in aux] == [(num_tokens, 64), (num_tokens, 64)]
    torch.testing.assert_close(aux[0][:chunk], embeds[:chunk])


def _dummy_mtp(layer_rows: list[int], out_rows_divisor: int):
    """An MTP predictor whose single layer records the rows it receives and
    returns 1/`out_rows_divisor` of them, like a layer slicing after attention."""

    class Layer:
        use_attn_reduce_scatter_for_moe = True

        def __call__(self, positions, hidden_states, residual):
            layer_rows.append(hidden_states.shape[0])
            out = hidden_states.narrow(0, 0, hidden_states.shape[0] // out_rows_divisor)
            return out, out

    return SimpleNamespace(
        num_mtp_layers=1,
        layers=[Layer()],
        embed_input_ids=lambda ids: torch.ones(ids.shape[0], 4),
        pre_fc_norm_embedding=lambda x: x,
        pre_fc_norm_hidden=lambda x: x,
        fc=lambda x: x[:, :4],
        norm=lambda h, r: (h + r, r),
    )


@pytest.mark.parametrize("with_step0_gather", [False, True])
def test_patched_mtp_forward_sequence_parallel(monkeypatch, with_step0_gather):
    """The drafter hands its layer every row and gathers the layer's slice
    after the norm, unless the step-0 post-attention gather keeps its rows
    replicated."""

    tp = 2
    _fake_tp_collectives(monkeypatch, tp)
    _sequence_parallel(monkeypatch, min_rows=0)
    monkeypatch.setattr(
        patch_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    layer_rows: list[int] = []
    model = _dummy_mtp(layer_rows, 1 if with_step0_gather else tp)
    num_tokens = 8
    indices = torch.arange(2) if with_step0_gather else None
    with step0_post_attn_gather_context(indices):
        out = patched_mtp_forward(
            model,
            input_ids=torch.zeros(num_tokens, dtype=torch.long),
            positions=torch.arange(num_tokens),
            hidden_states=torch.ones(num_tokens, 4),
        )
    assert layer_rows == [num_tokens]
    assert out.shape == (num_tokens, 4)


def test_entry_chunk_decoder_layer_forward_sequence_parallel(monkeypatch):
    """Verify the SP branch runs all-gather, narrow, reduce-scatter, and SP MLP."""

    tp = 2
    monkeypatch.setattr(
        patch_mod,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x] * tp, dim=dim),
    )
    monkeypatch.setattr(
        patch_mod,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim=0: x.narrow(dim, 0, x.shape[dim] // tp) * tp,
    )
    monkeypatch.setattr(
        patch_mod,
        "get_tensor_model_parallel_world_size",
        lambda: tp,
    )

    mlp_kwargs: list[dict[str, object]] = []

    class DummyLayer:
        use_attn_reduce_scatter_for_moe = True
        layer_type = "linear_attention"
        layer_scale = False

        def input_layernorm(self, x, residual=None):
            return x * 1.05 if residual is None else (x * 1.05, residual + x * 0.05)

        def linear_attn(self, hidden_states):
            return hidden_states + 1.0

        def post_attention_layernorm(self, x, residual):
            return x * 1.1, residual + x * 0.1

        def mlp(self, x, **kwargs):
            mlp_kwargs.append(kwargs)
            return x * 2.0

    layer = DummyLayer()
    num_tokens = 8
    hidden = torch.randn(num_tokens // tp, 64)
    residual = torch.randn(num_tokens // tp, 64)
    positions = torch.arange(num_tokens)

    patched_h, patched_r = entry_chunk_decoder_layer_forward(
        layer, hidden, residual, positions
    )

    normed = hidden * 1.05
    expected_r = residual + hidden * 0.05
    attn_shard = (normed + 1.0) * tp
    expected_h = (attn_shard * 1.1) * 2.0
    expected_r = expected_r + attn_shard * 0.1

    assert patched_h.shape == (num_tokens // tp, 64)
    assert patched_r.shape == (num_tokens // tp, 64)
    torch.testing.assert_close(patched_h, expected_h)
    torch.testing.assert_close(patched_r, expected_r)
    assert mlp_kwargs == [{"already_sequence_parallel": True}]


def test_check_tokens_divide_by_tp():
    """Divisibility check passes when divisible, raises RuntimeError when not."""
    patch_target = (
        "vllm_torchtpu.models.vllm.qwen3_next_patch."
        "get_tensor_model_parallel_world_size"
    )
    with patch(patch_target, return_value=8):
        _check_tokens_divide_by_tp(16)
        _check_tokens_divide_by_tp(8)
        with pytest.raises(RuntimeError):
            _check_tokens_divide_by_tp(7)


def test_entry_chunk_model_forward_sequence_parallel(monkeypatch):
    """The model-entry forward chunks the rows before the first layer, so every
    layer gets this rank's slice, and gathers the final norm's output back to
    the step's rows."""

    tp = 2
    _fake_tp_collectives(monkeypatch, tp)
    gathered = _record_all_gathers(monkeypatch)
    monkeypatch.setattr(
        patch_mod.qwen3_next,
        "sequence_parallel_chunk",
        lambda x: x.narrow(0, 0, x.shape[0] // tp),
    )
    monkeypatch.setattr(
        patch_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    seen: list[tuple[int, bool]] = []

    def layer(positions, hidden_states, residual):
        seen.append((hidden_states.shape[0], residual is None))
        return hidden_states + 1.0, hidden_states if residual is None else residual

    num_tokens = 8
    chunk = num_tokens // tp
    embeds = torch.randn(num_tokens, 64)
    model = SimpleNamespace(
        use_sequence_parallel=True,
        start_layer=0,
        end_layer=2,
        layers=[layer, layer],
        embed_input_ids=lambda ids: embeds,
        norm=lambda h, r: (h + r, r),
        _maybe_add_hidden_state=lambda aux, *args: aux,
    )

    hidden = entry_chunk_model_forward(
        model, torch.zeros(num_tokens, dtype=torch.long), torch.arange(num_tokens)
    )

    assert seen == [(chunk, True), (chunk, False)]
    assert gathered == [chunk]
    assert hidden.shape == (num_tokens, 64)
    torch.testing.assert_close(hidden[:chunk], embeds[:chunk] * 2 + 2.0)


def test_forward_patch_skipped_with_flag_off(monkeypatch):
    """When TPU_MOE_SEQUENCE_PARALLEL=0, methods remain the upstream objects."""
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False)
    result = maybe_patch_qwen3_next(QWEN3_NEXT)
    assert result is None
    assert qwen3_next.Qwen3NextDecoderLayer.forward is UPSTREAM_DECODER_LAYER_FORWARD
    assert qwen3_next.Qwen3NextModel.forward is UPSTREAM_MODEL_FORWARD


def test_forward_patch_applied(monkeypatch):
    """With the flag on, the inherited Qwen3Next forwards are replaced."""
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    monkeypatch.setattr(
        qwen3_next.Qwen3NextDecoderLayer,
        "forward",
        UPSTREAM_DECODER_LAYER_FORWARD,
    )
    monkeypatch.setattr(qwen3_next.Qwen3NextModel, "forward", UPSTREAM_MODEL_FORWARD)

    result = maybe_patch_qwen3_next(QWEN3_NEXT)
    assert result is None
    assert (
        qwen3_next.Qwen3NextDecoderLayer.forward is not UPSTREAM_DECODER_LAYER_FORWARD
    )
    assert qwen3_next.Qwen3NextModel.forward is not UPSTREAM_MODEL_FORWARD


def test_forward_patch_skipped_for_other_models_without_sp(
    restore_forwards, v6_install
):
    """Another model on these classes (Qwen3-Next) without
    TPU_MOE_SEQUENCE_PARALLEL=1 keeps the upstream forwards and gets no FP8
    fusion, as on main."""
    restore_forwards.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False)
    result = maybe_patch_qwen3_next(QWEN3_NEXT)
    assert result is None
    v6_install.assert_not_called()
    assert qwen3_next.Qwen3NextDecoderLayer.forward is UPSTREAM_DECODER_LAYER_FORWARD
    assert qwen3_next.Qwen3NextModel.forward is UPSTREAM_MODEL_FORWARD
    assert qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward is UPSTREAM_MTP_FORWARD
    assert (
        qwen3_5_mtp.Qwen3_5MultiTokenPredictor.forward is UNPATCHED_QWEN3_5_MTP_FORWARD
    )


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_a_config_without_architectures_gets_mains_behaviour(
    restore_forwards, v6_install, sequence_parallel
):
    """A config that does not name its architectures (a stand-in such as the
    patch registry's tests pass) is not a Qwen3.5 target: the patch installs
    what main installs and nothing of the Qwen3.5 rule."""
    restore_forwards.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", sequence_parallel)
    restore_forwards.setattr(patch_mod.sequence_parallel, "_FOLLOWS_STEP_FORM", False)
    stand_in = SimpleNamespace(hf_config=SimpleNamespace(model_type="llama"))
    assert maybe_patch_qwen3_next(stand_in) is None
    v6_install.assert_not_called()
    assert not patch_mod.sequence_parallel.follows_step_form()
    expected = (
        (entry_chunk_decoder_layer_forward, entry_chunk_model_forward)
        if sequence_parallel
        else (UPSTREAM_DECODER_LAYER_FORWARD, UPSTREAM_MODEL_FORWARD)
    )
    assert (
        qwen3_next.Qwen3NextDecoderLayer.forward,
        qwen3_next.Qwen3NextModel.forward,
    ) == expected


def test_forward_patch_waits_for_a_model_config(restore_forwards, v6_install):
    """Without a model config the rule cannot tell Qwen3.5 from other models,
    so it installs nothing and asks the registry to call it again."""
    restore_forwards.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    assert maybe_patch_qwen3_next(None) is False
    v6_install.assert_not_called()
    assert qwen3_next.Qwen3NextDecoderLayer.forward is UPSTREAM_DECODER_LAYER_FORWARD
    assert not patch_mod.sequence_parallel.follows_step_form()


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3NextForCausalLM",
        # Qwen3.5 classes outside models/qwen3_5/settings.ARCHITECTURES.
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
    ],
)
def test_other_models_keep_the_entry_chunk_forwards(
    restore_forwards, v6_install, architecture
):
    """Under TPU_MOE_SEQUENCE_PARALLEL=1 another model takes the model-entry
    forwards, not the Qwen3.5 ones. It gets no FP8 fusion, no row gating
    (every sequence-parallel step stays sliced), no rank tensor on its decoder
    layers and no FP8 gather ops, and both drafter forwards stay as they are."""
    restore_forwards.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    restore_forwards.setattr(patch_mod.sequence_parallel, "_FOLLOWS_STEP_FORM", False)
    prebuild = MagicMock()
    restore_forwards.setattr(patch_mod.linear_common, "prebuild_prequant_ops", prebuild)
    restore_forwards.setattr(patch_mod, "_prebuild_norm_ops", prebuild)

    model_config = SimpleNamespace(architectures=[architecture])
    assert maybe_patch_qwen3_next(model_config) is None
    v6_install.assert_not_called()
    prebuild.assert_not_called()
    assert not patch_mod.sequence_parallel.follows_step_form()
    assert qwen3_next.Qwen3NextDecoderLayer.forward is entry_chunk_decoder_layer_forward
    assert qwen3_next.Qwen3NextModel.forward is entry_chunk_model_forward
    assert qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward is UPSTREAM_MTP_FORWARD
    assert (
        qwen3_5_mtp.Qwen3_5MultiTokenPredictor.forward is UNPATCHED_QWEN3_5_MTP_FORWARD
    )
    for layer_cls in (qwen3_next.Qwen3NextDecoderLayer, qwen3_5.Qwen3_5DecoderLayer):
        assert not getattr(layer_cls.__init__, "_tpu_sp_rank_init", False)


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize(
    "case, chip",
    [("35b", "7x"), ("397b", "v6e"), ("397b", None), ("397b-60-to-40-layers", "7x")],
)
def test_qwen3_5_off_the_measured_model_gets_mains_behaviour(
    restore_forwards, v6_install, monkeypatch, sequence_parallel, case, chip
):
    """A Qwen3.5 checkpoint or device the patch was not measured on (main's CI
    model, Qwen3.5-35B-A3B; the 397B on another or an unknown device; another
    checkpoint shape) is not a target: no FP8 fusion, no row gating, no FP8
    gather ops, both drafter forwards unchanged, and main's forwards (the
    model-entry SP forwards only under TPU_MOE_SEQUENCE_PARALLEL=1)."""
    monkeypatch.setattr(qwen3_5_hooks, "_chip_family", lambda: chip)
    restore_forwards.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", sequence_parallel)
    restore_forwards.setattr(patch_mod.sequence_parallel, "_FOLLOWS_STEP_FORM", False)
    prebuild = MagicMock()
    restore_forwards.setattr(patch_mod.linear_common, "prebuild_prequant_ops", prebuild)
    restore_forwards.setattr(patch_mod, "_prebuild_norm_ops", prebuild)
    model_config = {
        "35b": QWEN3_5_35B,
        "397b": QWEN3_5,
        "397b-60-to-40-layers": SimpleNamespace(
            architectures=QWEN3_5.architectures,
            hf_text_config=SimpleNamespace(**{**_SHAPE_397B, "num_hidden_layers": 40}),
        ),
    }[case]
    assert maybe_patch_qwen3_next(model_config) is None
    v6_install.assert_not_called()
    prebuild.assert_not_called()
    assert not patch_mod.sequence_parallel.follows_step_form()
    expected = (
        (entry_chunk_decoder_layer_forward, entry_chunk_model_forward)
        if sequence_parallel
        else (UPSTREAM_DECODER_LAYER_FORWARD, UPSTREAM_MODEL_FORWARD)
    )
    assert (
        qwen3_next.Qwen3NextDecoderLayer.forward,
        qwen3_next.Qwen3NextModel.forward,
    ) == expected
    assert qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward is UPSTREAM_MTP_FORWARD
    assert (
        qwen3_5_mtp.Qwen3_5MultiTokenPredictor.forward is UNPATCHED_QWEN3_5_MTP_FORWARD
    )
    for layer_cls in (qwen3_next.Qwen3NextDecoderLayer, qwen3_5.Qwen3_5DecoderLayer):
        assert not getattr(layer_cls.__init__, "_tpu_sp_rank_init", False)


def test_main_ci_model_never_reads_the_device(
    restore_forwards, v6_install, monkeypatch
):
    """The checkpoint shape is checked first, so a 35B engine never reads the
    accelerator type."""
    chip = MagicMock(return_value="7x")
    monkeypatch.setattr(qwen3_5_hooks, "_chip_family", chip)
    restore_forwards.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False)
    assert maybe_patch_qwen3_next(QWEN3_5_35B) is None
    chip.assert_not_called()
    v6_install.assert_not_called()


@pytest.mark.parametrize("sequence_parallel", [True, False])
def test_qwen3_5_forward_patch_applied(restore_forwards, v6_install, sequence_parallel):
    """On a Qwen3.5 target the FP8 fusions are installed and the inherited
    Qwen3.5 forwards and the drafter forwards are replaced, with or without
    sequence parallelism (the MTP step-0 gather needs them); row gating
    follows sequence parallelism."""
    restore_forwards.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", sequence_parallel)
    restore_forwards.setattr(patch_mod.sequence_parallel, "fp8_gather", lambda: False)
    restore_forwards.setattr(patch_mod.sequence_parallel, "_FOLLOWS_STEP_FORM", False)

    result = maybe_patch_qwen3_next(QWEN3_5)
    assert result is None
    v6_install.assert_called_once_with()
    assert patch_mod.sequence_parallel.follows_step_form() is sequence_parallel
    assert qwen3_next.Qwen3NextDecoderLayer.forward is patched_decoder_layer_forward
    assert qwen3_next.Qwen3NextModel.forward is patched_model_forward
    assert qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward is patched_mtp_forward
    assert qwen3_5_mtp.Qwen3_5MultiTokenPredictor.forward is patched_mtp_forward
    assert (
        getattr(qwen3_next.Qwen3NextDecoderLayer.__init__, "_tpu_sp_rank_init", False)
        is sequence_parallel
    )


def test_mtp_draft_module_import_keeps_patched_forward(restore_forwards, v6_install):
    """Loading a Qwen3.5 MTP draft imports vllm_torchtpu.models.vllm.qwen3_5_mtp,
    whose module-level forward assignment must not replace patched_mtp_forward."""
    import vllm_torchtpu.models.vllm as tpu_models

    restore_forwards.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False)
    restore_forwards.setattr(patch_mod.sequence_parallel, "fp8_gather", lambda: False)
    restore_forwards.setattr(tpu_models, "qwen3_5_mtp", tpu_models.qwen3_5_mtp)
    restore_forwards.delitem(sys.modules, "vllm_torchtpu.models.vllm.qwen3_5_mtp")

    maybe_patch_qwen3_next(QWEN3_5)
    importlib.import_module("vllm_torchtpu.models.vllm.qwen3_5_mtp")
    assert qwen3_5_mtp.Qwen3_5MultiTokenPredictor.forward is patched_mtp_forward


def test_maybe_flip_gdn_in_proj_qkvz_weight():
    """Verify GDN in_proj_qkvz weight is flipped to (k, n) only when PCP is off."""
    k_in, n_out = 256, 128
    w_nk = torch.ones((n_out, k_in), dtype=torch.float8_e4m3fn)
    w_scale = torch.full((n_out,), 0.01, dtype=torch.float32)

    def _make_gdn(pcp: bool):
        proj = nn.Module()
        proj.register_parameter(
            "weight", nn.Parameter(w_nk.clone(), requires_grad=False)
        )
        proj.register_parameter(
            "weight_scale", nn.Parameter(w_scale.clone(), requires_grad=False)
        )
        proj.bias = None
        setattr(proj, KEEP_VLLM_LAYOUT_ATTR, True)
        return SimpleNamespace(
            _pcp_streaming_configured=pcp,
            in_proj_qkvz=proj,
        )

    gdn_pcp = _make_gdn(pcp=True)
    assert maybe_flip_gdn_in_proj_qkvz_weight(gdn_pcp) is False
    assert gdn_pcp.in_proj_qkvz.weight.shape == (n_out, k_in)
    assert getattr(gdn_pcp.in_proj_qkvz, KEEP_VLLM_LAYOUT_ATTR, False) is True

    gdn_non_pcp = _make_gdn(pcp=False)
    assert maybe_flip_gdn_in_proj_qkvz_weight(gdn_non_pcp) is True
    assert gdn_non_pcp.in_proj_qkvz.weight.shape == (k_in, n_out)
    assert getattr(gdn_non_pcp.in_proj_qkvz, KEEP_VLLM_LAYOUT_ATTR, False) is False
    assert getattr(gdn_non_pcp.in_proj_qkvz, WEIGHT_FLIPPED_ATTR, False) is True


def test_cat768_fp8_fusion_and_weight_deletion():
    """Verify Cat_FP8_256 fuses [gate; gate_up_proj] into 256-column-aligned FP8
    ([4096, 768] at TP=8 I=128 and [4096, 2560] at TP=1 I=1024), folds sigmoid(se_gate)
    into down_proj per-token scale, deletes original weights, and achieves cos > 0.9988.
    """
    torch.manual_seed(0)
    hidden_dim = 4096
    n_gate = 512
    tokens = 16

    for n_inter, tp_size, rep_se in ((128, 8, False), (1024, 1, True)):
        n_se_gu = 2 * n_inter
        w_gate = torch.randn(n_gate, hidden_dim, dtype=torch.bfloat16) * 0.02
        w_se_gu_bf16 = torch.randn(hidden_dim, n_se_gu, dtype=torch.bfloat16) * 0.02
        s_se_gu = w_se_gu_bf16.float().abs().amax(dim=0) / 448.0
        w_se_gu_fp8 = (
            (w_se_gu_bf16.float() / s_se_gu.unsqueeze(0))
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        w_se_down_bf16 = torch.randn(n_inter, hidden_dim, dtype=torch.bfloat16) * 0.02
        s_se_down = w_se_down_bf16.float().abs().amax(dim=0) / 448.0
        w_se_down_fp8 = (
            (w_se_down_bf16.float() / s_se_down.unsqueeze(0))
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        w_se_gate = torch.randn(1, hidden_dim, dtype=torch.bfloat16) * 0.02

        x = torch.randn(tokens, hidden_dim, dtype=torch.bfloat16)

        ref_router_logits = F.linear(x, w_gate)
        ref_gu = torch.matmul(x.float(), w_se_gu_fp8.float() * s_se_gu.unsqueeze(0)).to(
            torch.bfloat16
        )
        ref_inter = F.silu(ref_gu[:, :n_inter]) * ref_gu[:, n_inter:]
        ref_down = torch.matmul(
            ref_inter.float(), w_se_down_fp8.float() * s_se_down.unsqueeze(0)
        ).to(torch.bfloat16)
        ref_se_gate = F.linear(x, w_se_gate)
        ref_shared_out = ref_down * torch.sigmoid(ref_se_gate)

        gate_mod = nn.Module()
        gate_mod.register_parameter(
            "weight", nn.Parameter(w_gate.clone(), requires_grad=False)
        )
        gate_mod.bias = None

        gu_mod = nn.Module()
        gu_mod.register_parameter(
            "weight", nn.Parameter(w_se_gu_fp8.clone(), requires_grad=False)
        )
        gu_mod.register_parameter(
            "weight_scale", nn.Parameter(s_se_gu.float().clone(), requires_grad=False)
        )
        gu_mod.bias = None
        setattr(gu_mod, WEIGHT_FLIPPED_ATTR, True)
        gu_mod.quant_method = _fp8_method()

        down_mod = nn.Module()
        down_mod.register_parameter(
            "weight", nn.Parameter(w_se_down_fp8.clone(), requires_grad=False)
        )
        down_mod.register_parameter(
            "weight_scale", nn.Parameter(s_se_down.float().clone(), requires_grad=False)
        )
        down_mod.bias = None
        setattr(down_mod, WEIGHT_FLIPPED_ATTR, True)
        down_mod.quant_method = _fp8_method()

        se_gate_mod = nn.Module()
        se_gate_mod.register_parameter(
            "weight", nn.Parameter(w_se_gate.clone(), requires_grad=False)
        )
        se_gate_mod.bias = None

        moe = nn.Module()
        moe.replicate_shared_expert = rep_se
        moe.use_sequence_parallel = False
        moe.tp_size = tp_size
        moe.gate = gate_mod
        moe.shared_expert = SimpleNamespace(
            gate_up_proj=gu_mod,
            down_proj=down_mod,
            act_fn=lambda t, n=n_inter: F.silu(t[:, :n]) * t[:, n:],
        )
        moe.shared_expert_gate = se_gate_mod
        moe.experts = type(
            "_E",
            (),
            {
                "gate": gate_mod,
                "shared_expert_gate": se_gate_mod,
                "_fse_fuse_gate": True,
                "_shared_experts": moe.shared_expert,
                "moe_config": SimpleNamespace(
                    skip_final_all_reduce=False, is_sequence_parallel=False
                ),
                "maybe_all_reduce_tensor_model_parallel": lambda self, t: t,
                "__call__": lambda self, hidden_states, router_logits: hidden_states
                * 0.5,
            },
        )()

        assert maybe_setup_fused_gate_se(moe) is True
        assert getattr(moe, "_fused_gate_se_is_fp8", False) is True
        assert moe._fused_gate_gu_fp8_weight.shape == (hidden_dim, n_gate + n_se_gu)
        assert (n_gate + n_se_gu) % 256 == 0
        assert moe._fused_gate_gu_fp8_weight.dtype == torch.float8_e4m3fn
        assert moe._fused_se_down_fp8_weight.shape == (n_inter, hidden_dim)
        assert not hasattr(moe.gate, "weight")
        assert not hasattr(moe.shared_expert.gate_up_proj, "weight")
        assert not hasattr(moe.shared_expert.gate_up_proj, "weight_scale")
        assert not hasattr(moe.shared_expert.down_proj, "weight")
        assert not hasattr(moe.shared_expert.down_proj, "weight_scale")
        assert not hasattr(moe.shared_expert_gate, "weight")

        got_logits, got_shared = compute_fused_gate_and_shared_expert(moe, x)
        cos_logits = F.cosine_similarity(
            got_logits.float().flatten(), ref_router_logits.float().flatten(), dim=0
        ).item()
        cos_shared = F.cosine_similarity(
            got_shared.float().flatten(), ref_shared_out.float().flatten(), dim=0
        ).item()
        assert cos_logits > 0.999, f"cos_logits={cos_logits}"
        assert cos_shared > 0.9988, f"cos_shared={cos_shared}"

        got_moe_out = patched_sparse_moe_forward(moe, x)
        cos_moe = F.cosine_similarity(
            got_moe_out.float().flatten(),
            (x * 0.5 + ref_shared_out).float().flatten(),
            dim=0,
        ).item()
        assert cos_moe > 0.999, f"cos_moe={cos_moe}"


def test_an_unflipped_fp8_shared_expert_is_checked_again_after_the_flip():
    """The fused gate setup waits for an FP8 gate_up_proj to be flipped to
    (k, n), matching the FP8 method by class, subclasses included, and then
    fuses."""

    class _Fp8Method(VllmFp8LinearMethodTPU):
        pass

    hidden_dim, n_gate, n_inter = 512, 128, 64

    def linear(weight, scale=None):
        mod = nn.Module()
        mod.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
        if scale is not None:
            mod.register_parameter(
                "weight_scale", nn.Parameter(scale, requires_grad=False)
            )
        mod.bias = None
        return mod

    def fp8(shape):
        return (torch.randn(*shape) * 0.02).to(torch.float8_e4m3fn)

    gate_up = linear(fp8((hidden_dim, 2 * n_inter)), torch.ones(2 * n_inter))
    gate_up.quant_method = object.__new__(_Fp8Method)
    gate_up.quant_method.w8a16_max_rows = 0
    down = linear(fp8((n_inter, hidden_dim)), torch.ones(hidden_dim))
    setattr(down, WEIGHT_FLIPPED_ATTR, True)
    down.quant_method = _fp8_method()
    moe = nn.Module()
    moe.gate = linear(torch.randn(n_gate, hidden_dim, dtype=torch.bfloat16))
    moe.shared_expert = SimpleNamespace(
        gate_up_proj=gate_up, down_proj=down, act_fn=F.silu
    )
    moe.shared_expert_gate = linear(
        torch.randn(1, hidden_dim, dtype=torch.bfloat16) * 0.02
    )

    assert maybe_setup_fused_gate_se(moe) is False
    assert not getattr(moe, "_fused_gate_se_checked", False)
    setattr(gate_up, WEIGHT_FLIPPED_ATTR, True)
    assert maybe_setup_fused_gate_se(moe) is True


def test_the_shared_expert_gate_weight_takes_a_host_round_trip(monkeypatch):
    """The fused shared-expert gate weight is copied through host memory: on
    TPU an on-device copy of the [1, hidden] gate can alias the buffer behind
    the [hidden] parameter, and a host round trip cannot. On the host the
    round trip shows as the .cpu() of the [hidden] gate row; the fused weight
    keeps the gate's values."""
    torch.manual_seed(0)
    hidden_dim, n_gate, n_inter = 512, 128, 64

    def linear(name, weight, scale=None):
        mod = nn.Module()
        mod.register_parameter(name, nn.Parameter(weight, requires_grad=False))
        if scale is not None:
            mod.register_parameter(
                "weight_scale", nn.Parameter(scale, requires_grad=False)
            )
            setattr(mod, WEIGHT_FLIPPED_ATTR, True)
            mod.quant_method = _fp8_method()
        mod.bias = None
        return mod

    def fp8(shape):
        return (torch.randn(*shape) * 0.02).to(torch.float8_e4m3fn)

    w_se_gate = torch.randn(1, hidden_dim, dtype=torch.bfloat16) * 0.02
    moe = nn.Module()
    moe.gate = linear("weight", torch.randn(n_gate, hidden_dim, dtype=torch.bfloat16))
    moe.shared_expert = SimpleNamespace(
        gate_up_proj=linear(
            "weight", fp8((hidden_dim, 2 * n_inter)), torch.ones(2 * n_inter)
        ),
        down_proj=linear("weight", fp8((n_inter, hidden_dim)), torch.ones(hidden_dim)),
        act_fn=F.silu,
    )
    moe.shared_expert_gate = linear("weight", w_se_gate.clone())

    round_trips: list[tuple[int, ...]] = []
    tensor_cpu = torch.Tensor.cpu

    def cpu(self, *args, **kwargs):
        if self.ndim == 1 and self.dtype == torch.bfloat16:
            round_trips.append(tuple(self.shape))
        return tensor_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    assert maybe_setup_fused_gate_se(moe) is True
    assert round_trips == [(hidden_dim,)]
    torch.testing.assert_close(
        moe._fused_se_gate_weight.data, w_se_gate.reshape(hidden_dim), rtol=0, atol=0
    )


@pytest.mark.parametrize("w8a16_max_rows", [0, 64])
def test_the_fused_fp8_ops_take_a_projection_whose_method_runs_w8a8(w8a16_max_rows):
    """The sigmoid-gate o_proj op and the fused router/shared-expert op
    quantize their rows and run a W8A8 matmul, so they take an FP8 per-channel
    projection only where its method serves pre-quantized rows: not where it
    runs small steps W8A16 (TPU_FP8_LINEAR_W8A16_MAX_ROWS)."""
    hidden_dim, n_gate, n_inter = 512, 128, 64
    w8a8 = w8a16_max_rows == 0

    def linear(weight, fp8_scale=None):
        mod = nn.Module()
        mod.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
        if fp8_scale is not None:
            mod.register_parameter(
                "weight_scale", nn.Parameter(fp8_scale, requires_grad=False)
            )
            setattr(mod, WEIGHT_FLIPPED_ATTR, True)
            mod.quant_method = _fp8_method(w8a16_max_rows)
        mod.bias = None
        return mod

    def fp8(shape):
        return (torch.randn(*shape) * 0.02).to(torch.float8_e4m3fn)

    o_proj = linear(fp8((hidden_dim, hidden_dim)), torch.ones(hidden_dim))
    attention = SimpleNamespace(o_proj=o_proj)
    rows = torch.zeros(4, hidden_dim, dtype=torch.bfloat16)
    assert patch_mod._can_use_pal_gate_q(attention, rows, rows) is w8a8

    moe = nn.Module()
    moe.gate = linear(torch.randn(n_gate, hidden_dim, dtype=torch.bfloat16))
    moe.shared_expert = SimpleNamespace(
        gate_up_proj=linear(fp8((hidden_dim, 2 * n_inter)), torch.ones(2 * n_inter)),
        down_proj=linear(fp8((n_inter, hidden_dim)), torch.ones(hidden_dim)),
        act_fn=F.silu,
    )
    moe.shared_expert_gate = linear(
        torch.randn(1, hidden_dim, dtype=torch.bfloat16) * 0.02
    )
    assert maybe_setup_fused_gate_se(moe) is w8a8


def _bf16_jnp(t: torch.Tensor):
    import jax.numpy as jnp

    return jnp.asarray(t.float().numpy(), dtype=jnp.bfloat16)


def _gemma_norm(weight: torch.Tensor, eps: float = 1e-6) -> SimpleNamespace:
    return SimpleNamespace(variance_epsilon=eps, weight=weight)


@pytest.mark.parametrize("with_residual", [True, False])
def test_fused_norm_quantize_rows_matches_norm_then_quantize(with_residual):
    """The fused input-norm op returns the
    residual sum and `quantize_rows` of the GemmaRMSNorm output."""
    import jax.numpy as jnp

    from vllm_torchtpu.layers.adapter import linear_common

    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn(16, 256, dtype=torch.bfloat16)
    residual = torch.randn(16, 256, dtype=torch.bfloat16)
    weight = torch.randn(256) * 0.1
    norm = _gemma_norm(weight, eps)
    w_plus_one = jnp.asarray((1.0 + weight.float()).numpy())
    if with_residual:
        h_ref, res_ref = patch_mod._patched_gemma_rmsnorm_forward_native(
            norm, x, residual
        )
        x_q, x_s, res_out = patch_mod._norm_quantize_rows_with_res_jax(
            _bf16_jnp(x), _bf16_jnp(residual), w_plus_one, eps=eps
        )
        np.testing.assert_array_equal(
            np.asarray(res_out, np.float32), res_ref.float().numpy()
        )
    else:
        h_ref = patch_mod._patched_gemma_rmsnorm_forward_native(norm, x)
        x_q, x_s = patch_mod._norm_quantize_rows_no_res_jax(
            _bf16_jnp(x), w_plus_one, eps=eps
        )
    assert x_q.dtype == jnp.float8_e4m3fn and x_q.shape == (16, 256)
    assert x_s.dtype == jnp.float32 and x_s.shape == (16, 1)
    ref_q, ref_s = linear_common._quantize_rows_jax(_bf16_jnp(h_ref))
    np.testing.assert_allclose(np.asarray(x_s), np.asarray(ref_s), rtol=1e-2)
    deq = np.asarray(x_q, np.float32) * np.asarray(x_s)
    np.testing.assert_allclose(deq, h_ref.float().numpy(), rtol=0.07, atol=1e-3)
    assert np.mean(np.asarray(x_q, np.float32) == np.asarray(ref_q, np.float32)) > 0.99


def test_fused_norm_gate_and_se_matches_norm_then_gate_and_se():
    """The fused post-attention op is the
    GemmaRMSNorm of the residual sum followed by `_fp8_gate_and_se_jax`."""
    import jax.numpy as jnp

    torch.manual_seed(0)
    eps = 1e-6
    hidden, n_gate, n_inter, rows = 256, 64, 32, 8
    x = torch.randn(rows, hidden, dtype=torch.bfloat16)
    residual = torch.randn(rows, hidden, dtype=torch.bfloat16)
    weight = torch.randn(hidden) * 0.1
    w_gate_gu = jnp.asarray(
        (torch.randn(hidden, n_gate + 2 * n_inter) * 8).numpy(), jnp.float8_e4m3fn
    )
    s_gate_gu = jnp.full((n_gate + 2 * n_inter,), 0.01, jnp.float32)
    w_se_gate = _bf16_jnp(torch.randn(hidden) * 0.02)
    w_down = jnp.asarray((torch.randn(n_inter, hidden) * 8).numpy(), jnp.float8_e4m3fn)
    s_down = jnp.full((hidden,), 0.01, jnp.float32)
    w_plus_one = jnp.asarray((1.0 + weight.float()).numpy())

    logits, se_out, h_bf16, res_out = patch_mod._norm_fp8_gate_and_se_jax(
        _bf16_jnp(x),
        _bf16_jnp(residual),
        w_plus_one,
        w_gate_gu,
        s_gate_gu,
        w_se_gate,
        w_down,
        s_down,
        eps=eps,
    )
    h_ref, res_ref = patch_mod._patched_gemma_rmsnorm_forward_native(
        _gemma_norm(weight, eps), x, residual
    )
    np.testing.assert_array_equal(np.asarray(res_out, np.float32), res_ref.float())
    # The norm runs in JAX here and in torch there: one bf16 ulp at most.
    np.testing.assert_allclose(
        np.asarray(h_bf16, np.float32), h_ref.float().numpy(), rtol=2**-7, atol=1e-6
    )
    ref_logits, ref_se = patch_mod._fp8_gate_and_se_jax(
        h_bf16, w_gate_gu, s_gate_gu, w_se_gate, w_down, s_down
    )
    np.testing.assert_array_equal(
        np.asarray(logits, np.float32), np.asarray(ref_logits, np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(se_out, np.float32), np.asarray(ref_se, np.float32)
    )


def _tpu_rows(dtype=torch.bfloat16) -> SimpleNamespace:
    """What the fusion predicates read of a TPU activation."""
    return SimpleNamespace(device=SimpleNamespace(type="tpu"), dtype=dtype)


def test_pre_attn_norm_quant_fusion_predicate(monkeypatch):
    """Sliced rows take the fused input norm whenever the layer is eligible;
    replicated rows only where the pre-quantized rows are exact."""
    _sequence_parallel(monkeypatch, fp8_gather=True)
    fuses = patch_mod._fuses_pre_attn_norm_quant
    layer = SimpleNamespace(_pre_norm_quant_eligible=True, _pre_norm_quant_exact=False)

    assert fuses(layer, _tpu_rows(), True)
    assert not fuses(layer, _tpu_rows(), False)
    layer._pre_norm_quant_exact = True
    assert fuses(layer, _tpu_rows(), False)

    assert not fuses(layer, torch.zeros(2, 2, dtype=torch.bfloat16), True)
    assert not fuses(layer, _tpu_rows(torch.float32), True)
    assert not fuses(SimpleNamespace(), _tpu_rows(), True)
    _sequence_parallel(monkeypatch, fp8_gather=False)
    assert not fuses(layer, _tpu_rows(), True)


def test_post_attn_norm_moe_fusion_predicate(monkeypatch):
    fuses = patch_mod._fuses_post_attn_norm_moe
    norm = GemmaRMSNorm.__new__(GemmaRMSNorm)
    layer = SimpleNamespace(
        mlp=SimpleNamespace(_fused_gate_gu_fp8_weight=torch.zeros(1)),
        post_attention_layernorm=norm,
    )
    assert fuses(layer, _tpu_rows())
    assert not fuses(layer, torch.zeros(2, 2, dtype=torch.bfloat16))
    assert not fuses(layer, _tpu_rows(torch.float32))
    layer.mlp = SimpleNamespace()
    assert not fuses(layer, _tpu_rows())
    layer.mlp = SimpleNamespace(_fused_gate_gu_fp8_weight=torch.zeros(1))
    layer.post_attention_layernorm = _gemma_norm(torch.zeros(4))
    assert not fuses(layer, _tpu_rows())


def _proj(
    dtype=torch.float8_e4m3fn, scale_shape=(4,), quant_method=None
) -> SimpleNamespace:
    return SimpleNamespace(
        weight=torch.zeros(4, 4, dtype=dtype),
        weight_scale=torch.ones(scale_shape),
        quant_method=_fp8_method() if quant_method is None else quant_method,
    )


def test_takes_prequant_rows_dispatches_on_the_method_class():
    """A method takes pre-quantized rows by being a PrequantLinearMethod, not by
    having an `apply_prequant` attribute."""
    assert patch_mod._takes_prequant_rows(_proj())
    stand_in = SimpleNamespace(apply_prequant=lambda *args: None)
    assert not patch_mod._takes_prequant_rows(_proj(quant_method=stand_in))
    assert not patch_mod._takes_prequant_rows(None)


def _fp8_method(w8a16_max_rows=0):
    method = VllmFp8LinearMethodTPU.__new__(VllmFp8LinearMethodTPU)
    method.w8a16_max_rows = w8a16_max_rows
    return method


def test_takes_prequant_rows_follows_what_the_fp8_method_serves():
    """The TPU FP8 method takes pre-quantized rows only where its pre-quantized
    matmul computes what `apply` would: one scale per output column, and no
    W8A16 small steps (the pre-quantized matmul is W8A8)."""
    assert patch_mod._takes_prequant_rows(_proj(quant_method=_fp8_method()))
    assert not patch_mod._takes_prequant_rows(
        _proj(scale_shape=(2,), quant_method=_fp8_method())
    )
    assert not patch_mod._takes_prequant_rows(
        _proj(quant_method=_fp8_method(w8a16_max_rows=8))
    )


def test_mark_pre_attn_norm_quant():
    """Eligible: a GemmaRMSNorm and an FP8 per-channel main projection. Exact:
    every projection of the attention input is FP8 per-channel."""
    norm = GemmaRMSNorm.__new__(GemmaRMSNorm)

    def mark(layer):
        patch_mod._mark_pre_attn_norm_quant(layer)
        return layer._pre_norm_quant_eligible, layer._pre_norm_quant_exact

    full = SimpleNamespace(
        layer_type="full_attention",
        input_layernorm=norm,
        self_attn=SimpleNamespace(qkv_proj=_proj()),
    )
    assert mark(full) == (True, True)

    def gdn(ba_dtype=torch.bfloat16, pcp=False):
        return SimpleNamespace(
            layer_type="linear_attention",
            input_layernorm=norm,
            linear_attn=SimpleNamespace(
                in_proj_qkvz=_proj(),
                in_proj_ba=_proj(ba_dtype),
                _pcp_streaming_configured=pcp,
            ),
        )

    # A bf16 in_proj_ba would see FP8-rounded replicated rows.
    assert mark(gdn()) == (True, False)
    assert mark(gdn(torch.float8_e4m3fn)) == (True, True)
    # The pooled PCP forward refuses prequant.
    assert mark(gdn(torch.float8_e4m3fn, pcp=True)) == (False, False)

    full.self_attn.qkv_proj = _proj(scale_shape=(1, 4))
    assert mark(full) == (False, False)
    full.self_attn.qkv_proj = _proj()
    full.input_layernorm = _gemma_norm(torch.zeros(4))
    assert mark(full) == (False, False)


class _CallableNorm:
    """A norm stand-in the fused-path fakes below reproduce exactly."""

    variance_epsilon = 1e-6

    def __init__(self, scale: float):
        self.weight = torch.zeros(64)
        self.scale = scale

    def __call__(self, x, residual=None):
        if residual is None:
            return x * self.scale
        x = x + residual
        return x * self.scale, x


class _FusedMoeBlock:
    """A MoE block with the fused gate/shared-expert weights; doubles rows."""

    _fused_gate_se_checked = True
    _fused_gate_gu_fp8_weight = torch.zeros(1)
    _fused_gate_gu_fp8_scale = torch.zeros(1)
    _fused_se_gate_weight = torch.zeros(1)
    _fused_se_down_fp8_weight = torch.zeros(1)
    _fused_se_down_fp8_scale = torch.zeros(1)

    def __init__(self, calls):
        self.calls = calls

    def __call__(self, x, **kwargs):
        self.calls.append(("unfused", kwargs))
        return x * 2.0


def _fused_norm_layer(attn_calls, mlp_calls):
    class Layer:
        use_attn_reduce_scatter_for_moe = True
        layer_type = "linear_attention"
        layer_scale = False
        input_layernorm = _CallableNorm(1.05)
        post_attention_layernorm = _CallableNorm(1.1)

        def linear_attn(self, hidden_states, **kwargs):
            attn_calls.append(kwargs)
            return hidden_states.float() + 1.0

    layer = Layer()
    layer.mlp = _FusedMoeBlock(mlp_calls)
    layer._tpu_sp_rank = torch.tensor([0], dtype=torch.int32)
    return layer


def _fake_fused_norm_ops(monkeypatch, moe_calls):
    """CPU stand-ins for the fused norm ops matching `_CallableNorm`, with the
    quantized rows kept in bf16 (as `quantize_rows` is faked)."""

    def with_res(eps):
        def op(x, residual, w_plus_one):
            s = x + residual
            return (s * 1.05).to(torch.bfloat16), torch.ones(x.shape[0], 1), s

        return op

    def no_res(eps):
        return lambda x, w_plus_one: (
            (x * 1.05).to(torch.bfloat16),
            torch.ones(x.shape[0], 1),
        )

    def gate_se(eps):
        def op(x, residual, w_plus_one, *weights):
            assert len(weights) == 5
            s = x + residual
            h = s * 1.1
            return h[:, :4], torch.zeros_like(h), h, s

        return op

    def moe_forward(mlp, h, already_sequence_parallel=False, *, prenorm_outputs):
        moe_calls.append((already_sequence_parallel, prenorm_outputs))
        return h * 2.0

    monkeypatch.setattr(patch_mod, "_get_norm_quantize_rows_with_res_op", with_res)
    monkeypatch.setattr(patch_mod, "_get_norm_quantize_rows_no_res_op", no_res)
    monkeypatch.setattr(patch_mod, "_get_norm_fp8_gate_and_se_op", gate_se)
    monkeypatch.setattr(patch_mod, "patched_sparse_moe_forward", moe_forward)


@pytest.mark.parametrize("first_layer", [False, True])
def test_patched_decoder_layer_forward_fused_norms_match_unfused(
    monkeypatch, first_layer
):
    """With both fusions on, the SP layer keeps its row handling (gather the
    slice into attention, or slice replicated rows after it) and its result;
    attention gets the fused op's rows pre-quantized, and the MoE gets the
    fused op's normed rows, router logits and shared expert."""
    tp = 2
    _fake_tp_collectives(monkeypatch, tp)
    _sequence_parallel(monkeypatch, min_rows=0, fp8_gather=True)
    monkeypatch.setattr(
        patch_mod.linear_common,
        "quantize_rows",
        lambda x: (x.to(torch.bfloat16), torch.ones(x.shape[0], 1)),
    )
    num_tokens = 8
    rows = num_tokens if first_layer else num_tokens // tp
    # bf16 rows: the fakes stand in bf16 for fp8, so both paths stay exact.
    hidden = torch.randn(rows, 64, dtype=torch.bfloat16)
    residual = None if first_layer else torch.randn(rows, 64, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens)

    attn_calls: list[dict[str, object]] = []
    mlp_calls: list = []
    unfused_h, unfused_r = patched_decoder_layer_forward(
        _fused_norm_layer(attn_calls, mlp_calls), hidden, residual, positions
    )
    assert mlp_calls == [("unfused", {"already_sequence_parallel": True})]

    moe_calls: list = []
    fused_attn_calls: list[dict[str, object]] = []
    _fake_fused_norm_ops(monkeypatch, moe_calls)
    monkeypatch.setattr(patch_mod, "_fuses_pre_attn_norm_quant", lambda *a: True)
    monkeypatch.setattr(patch_mod, "_fuses_post_attn_norm_moe", lambda *a: True)
    fused_h, fused_r = patched_decoder_layer_forward(
        _fused_norm_layer(fused_attn_calls, mlp_calls), hidden, residual, positions
    )

    torch.testing.assert_close(fused_h, unfused_h)
    torch.testing.assert_close(fused_r, unfused_r)
    assert fused_h.shape == (num_tokens // tp, 64)
    (kwargs,) = fused_attn_calls
    x_q, x_s = kwargs["prequant"]
    assert x_q.shape == (num_tokens, 64) and x_s.shape == (num_tokens, 1)
    ((already_sp, (logits, se_out)),) = moe_calls
    assert already_sp is True
    assert logits.shape == (num_tokens // tp, 4)
    assert se_out.shape == (num_tokens // tp, 64)


def test_patched_sparse_moe_forward_takes_prenorm_outputs(monkeypatch):
    """`prenorm_outputs` replaces the fused gate/shared-expert op."""

    def no_recompute(*args, **kwargs):
        raise AssertionError("prenorm_outputs already holds the gate outputs")

    monkeypatch.setattr(patch_mod, "compute_fused_gate_and_shared_expert", no_recompute)
    seen = {}

    def experts(hidden_states, router_logits):
        seen["router_logits"] = router_logits
        return hidden_states * 0.5

    moe = SimpleNamespace(
        _fused_gate_se_checked=True,
        _fused_gate_se_weight=torch.zeros(1),
        _fused_se_is_replicated=True,
        tp_size=1,
        experts=experts,
    )
    x = torch.randn(4, 8)
    logits = torch.randn(4, 3)
    se_out = torch.randn(4, 8)
    out = patched_sparse_moe_forward(moe, x, prenorm_outputs=(logits, se_out))
    torch.testing.assert_close(out, x * 0.5 + se_out)
    assert seen["router_logits"] is logits


def test_patched_sparse_moe_forward_v2_owns_combine_no_double_all_reduce(monkeypatch):
    """When fused_moe_ep v2 is armed (_fused_moe_ep_op is not None) and
    se_is_replicated=False (TP=8, SP=0), patched_sparse_moe_forward must only
    all-reduce shared_output and NOT all-reduce final_hidden_states a second time.
    """
    tp = 8
    all_reduced: list[torch.Tensor] = []
    _fake_tp_collectives(monkeypatch, tp, all_reduced=all_reduced)
    monkeypatch.setattr(patch_mod.sequence_parallel, "is_active", lambda: False)
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)

    hidden_dim = 64
    tokens = 4
    x = torch.ones(tokens, hidden_dim, dtype=torch.bfloat16)
    routed_out = torch.full((tokens, hidden_dim), 2.0, dtype=torch.bfloat16)
    shared_partial = torch.full((tokens, hidden_dim), 0.25, dtype=torch.bfloat16)

    monkeypatch.setattr(
        patch_mod,
        "compute_fused_gate_and_shared_expert",
        lambda _moe, _h: (torch.zeros(tokens, 8), shared_partial),
    )

    qm = SimpleNamespace(_fused_moe_ep_op=object())
    routed_experts = SimpleNamespace(quant_method=qm)
    moe = SimpleNamespace(
        _fused_gate_se_checked=True,
        _fused_gate_se_weight=torch.ones(1),
        _fused_se_is_replicated=False,
        replicate_shared_expert=False,
        is_sequence_parallel=False,
        tp_size=tp,
        experts=type(
            "_E",
            (),
            {
                "routed_experts": routed_experts,
                "__call__": lambda self, hidden_states, router_logits: routed_out,
            },
        )(),
    )

    out = patched_sparse_moe_forward(moe, x)
    assert len(all_reduced) == 1
    torch.testing.assert_close(all_reduced[0], shared_partial)
    # Expected: routed_out (2.0) + tp * shared_partial (8 * 0.25 = 2.0) = 4.0
    torch.testing.assert_close(out, routed_out + shared_partial * tp)


def test_sp_row_decision_inside_and_outside_shape_variant_warmup(monkeypatch):
    """A trace outside shape_variants.warmup() takes the caller's
    sequence-parallel decision; inside it, the trace compares its own token
    count with the SP row minimum."""
    from vllm_torchtpu.compilation import shape_variants

    assert not shape_variants.in_warmup()
    with shape_variants.warmup():
        assert shape_variants.in_warmup()
    assert not shape_variants.in_warmup()

    monkeypatch.setattr(patch_mod.sequence_parallel, "is_active", lambda: True)
    monkeypatch.setattr(patch_mod.sequence_parallel, "min_rows", lambda: 512)
    assert not patch_mod._rows_are_sequence_parallel(4)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert patch_mod._rows_are_sequence_parallel(4)
    with shape_variants.warmup():
        assert not patch_mod._rows_are_sequence_parallel(4)
        assert patch_mod._rows_are_sequence_parallel(512)


def test_a_draft_trace_reads_a_sliced_form_until_the_draft_form_is_recorded(
    monkeypatch,
):
    """The MTP drafter is compiled once outside the shape-variant warmup, so
    its sequence-parallel decision is the last recorded step form. Without a
    proposer that records each draft forward's form, a target step below the
    layout's threshold would leave a replicated form behind; with a drafter
    the threshold therefore stays 0, so every recorded form is sliced and the
    draft slices its rows, as main's drafter does. With the record, the
    target's small step is replicated and the draft's own record decides."""
    sp = patch_mod.sequence_parallel
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    monkeypatch.setattr(sp, "_FOLLOWS_STEP_FORM", True)
    monkeypatch.setattr(sp, "_HAS_DRAFTER", True)
    monkeypatch.setattr(sp, "_ACTIVE", sp._ACTIVE)
    monkeypatch.setattr(sp, "parallel_layout", lambda parallel_config=None: (1, 8))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(patch_mod, "_step0_gather_indices", None)

    monkeypatch.setattr(sp, "DRAFT_RECORDS_STEP_FORM", False)
    sp.set_active(sp.step_is_sequence_parallel(4))  # the runner's 4-row target step
    assert patch_mod._rows_are_sequence_parallel(4)

    monkeypatch.setattr(sp, "DRAFT_RECORDS_STEP_FORM", True)
    sp.set_active(sp.step_is_sequence_parallel(4))
    assert not patch_mod._rows_are_sequence_parallel(4)
    sp.set_active(True)  # the proposer records a sliced draft forward
    assert patch_mod._rows_are_sequence_parallel(4)


class _QkvProj:
    """`qkv_proj` stand-in recording which entry the attention forward used."""

    def __init__(self, skip_bias_add):
        self.skip_bias_add = skip_bias_add
        self.bias = torch.full((24,), 0.5)
        self.calls: list[tuple] = []
        self.quant_method = SimpleNamespace(apply_prequant=self._apply_prequant)

    def _apply_prequant(self, layer, x_q, x_s, bias):
        assert layer is self
        self.calls.append(("prequant", x_q, x_s, bias))
        return torch.zeros(x_q.shape[0], 24)

    def __call__(self, x):
        self.calls.append(("plain", x))
        return torch.zeros(x.shape[0], 24), None


@pytest.mark.parametrize("skip_bias_add", [False, True])
def test_attn_forward_takes_the_rows_or_the_prequantized_rows(
    monkeypatch, skip_bias_add
):
    """Without prequant the attention runs qkv_proj on its rows; with it, the
    projection's apply_prequant on the FP8 rows and scales, with the bias the
    projection adds itself."""
    seen = []
    monkeypatch.setattr(
        patch_mod,
        "_forward_qwen3_next_attn_from_qkv",
        lambda layer, qkv, positions: seen.append((qkv.shape, positions)) or qkv,
    )
    proj = _QkvProj(skip_bias_add)
    attn = SimpleNamespace(qkv_proj=proj)
    rows, positions = torch.randn(4, 16), torch.arange(4)
    patch_mod.patched_qwen3_next_attn_forward(attn, positions, rows)
    x_q, x_s = torch.randn(4, 16), torch.ones(4, 1)
    patch_mod.patched_qwen3_next_attn_forward(attn, positions, x_q, (x_q, x_s))

    (plain, prequant) = proj.calls
    assert plain[0] == "plain" and plain[1] is rows
    assert prequant[0] == "prequant" and prequant[1] is x_q and prequant[2] is x_s
    assert prequant[3] is (None if skip_bias_add else proj.bias)
    assert [s for s, _ in seen] == [(4, 24), (4, 24)]


def _attn_from_qkv_layer(pal_gate_q_eligible=False):
    o_proj_calls = []

    def o_proj(x):
        o_proj_calls.append(x)
        return x * 3.0, None

    layer = SimpleNamespace(
        _project_qkv_gate=lambda qkv, positions: (
            qkv[:, :8],
            qkv[:, 8:16],
            qkv[:, 16:24],
            qkv[:, 24:32],
        ),
        attn=lambda q, k, v: q * 2.0,
        o_proj=o_proj,
        _pal_gate_q_eligible=pal_gate_q_eligible,
    )
    return layer, o_proj_calls


@pytest.mark.parametrize("gather", [None, [1, 3]], ids=["all_rows", "step0_gather"])
def test_attention_from_qkv_gates_the_attention_rows(gather):
    """The attention output is gated by sigmoid(gate) before o_proj; on MTP
    step 0 the post-attention gather keeps only the requested rows of both."""
    layer, o_proj_calls = _attn_from_qkv_layer()
    qkv = torch.randn(4, 32)
    indices = None if gather is None else torch.tensor(gather)
    with patch_mod.step0_post_attn_gather_context(indices):
        out = patch_mod._forward_qwen3_next_attn_from_qkv(layer, qkv, None)
    rows = slice(None) if gather is None else gather
    want = qkv[rows, :8] * 2.0 * torch.sigmoid(qkv[rows, 24:32])
    (o_proj_in,) = o_proj_calls
    torch.testing.assert_close(o_proj_in, want)
    torch.testing.assert_close(out, want * 3.0)
    assert patch_mod._step0_gather_indices is None


def test_pal_gate_q_o_proj_runs_the_fused_op_and_all_reduces(monkeypatch):
    """The fused sigmoid gate + FP8 o_proj takes 2D rows, all-reduces a TP
    o_proj's output and gives it the attention output's leading shape."""
    calls, reduced = [], []

    def op(attn_2d, gate_2d, weight, weight_scale):
        calls.append((attn_2d.shape, gate_2d.shape, weight, weight_scale))
        return attn_2d[:, :16] + gate_2d[:, :16]

    monkeypatch.setattr(patch_mod, "_get_sigmoid_gate_o_proj_op", lambda: op)
    monkeypatch.setattr(
        patch_mod,
        "tensor_model_parallel_all_reduce",
        lambda x: reduced.append(x) or x * 2.0,
    )
    o_proj = SimpleNamespace(
        weight=object(), weight_scale=object(), reduce_results=True, tp_size=2
    )
    attn_out, gate = torch.randn(2, 3, 128), torch.randn(2, 3, 128)
    out = patch_mod._apply_pal_gate_q_o_proj(o_proj, attn_out, gate)
    ((a_shape, g_shape, w, s),) = calls
    assert a_shape == g_shape == (6, 128)
    assert w is o_proj.weight and s is o_proj.weight_scale
    assert len(reduced) == 1
    want = (attn_out.reshape(6, 128)[:, :16] + gate.reshape(6, 128)[:, :16]) * 2.0
    torch.testing.assert_close(out, want.reshape(2, 3, 16))


def test_decoder_layer_step0_gather_keeps_the_requested_residual_rows(monkeypatch):
    """On MTP step 0 the attention keeps only the gathered rows, and the
    decoder layer gathers the residual to match before the MoE."""
    _sequence_parallel(monkeypatch, min_rows=0)

    class Layer:
        use_attn_reduce_scatter_for_moe = False
        layer_type = "full_attention"
        layer_scale = False

        def input_layernorm(self, x, residual=None):
            return x if residual is None else (x, residual)

        def self_attn(self, hidden_states, positions):
            return torch.index_select(hidden_states, 0, patch_mod._step0_gather_indices)

        def post_attention_layernorm(self, x, residual):
            return x, residual + x

        def mlp(self, x):
            return x

    hidden, residual = torch.randn(4, 8), torch.randn(4, 8)
    with patch_mod.step0_post_attn_gather_context(torch.tensor([1, 3])):
        out, new_residual = patched_decoder_layer_forward(
            Layer(), hidden, residual, torch.arange(4)
        )
    torch.testing.assert_close(out, hidden[[1, 3]])
    torch.testing.assert_close(new_residual, residual[[1, 3]] + hidden[[1, 3]])


@pytest.mark.parametrize(
    ("num_tokens", "in_warmup", "rows"),
    [(4, False, 16), (4, True, 4), (5, True, 8), (8, True, 8), (12, True, 16)],
)
def test_step0_gather_slices_to_4_or_8_rows_only_inside_a_warmup(
    num_tokens, in_warmup, rows
):
    """Inside shape_variants.warmup() a step of at most 4 or 8 tokens gathers
    the first 4 or 8 rows of the step-0 gather, which hold every request a
    step that small can have; outside it, or at more tokens, the whole
    gather."""
    gather = torch.arange(16)
    window = shape_variants.warmup() if in_warmup else contextlib.nullcontext()
    with step0_post_attn_gather_context(gather), window:
        effective = patch_mod._step0_effective_gather_indices(
            torch.zeros((3, num_tokens))
        )
    assert torch.equal(effective, gather[:rows])


class _Step0GatherMtpLayer:
    """An MTP layer on the patched decoder-layer forward whose attention runs
    the real post-attention gather (`_forward_qwen3_next_attn_from_qkv`). The
    MoE stand-in records the rows it receives and the padding mask it reads
    over them."""

    use_attn_reduce_scatter_for_moe = False
    layer_type = "full_attention"
    layer_scale = False

    def __init__(self):
        self.attn, _ = _attn_from_qkv_layer()
        self.moe_rows: list[int] = []
        self.moe_masks: list[list[bool]] = []

    def __call__(self, positions, hidden_states, residual):
        return patched_decoder_layer_forward(self, hidden_states, residual, positions)

    def input_layernorm(self, x, residual=None):
        return x if residual is None else (x, residual)

    def self_attn(self, hidden_states, positions):
        qkv = hidden_states.repeat(1, 4)
        return patch_mod._forward_qwen3_next_attn_from_qkv(self.attn, qkv, positions)

    def post_attention_layernorm(self, x, residual):
        return x, residual + x

    def mlp(self, x):
        self.moe_rows.append(x.shape[0])
        mask = token_padding.padding_mask_for(x.shape[0], is_local_tensor=True)
        self.moe_masks.append(mask.tolist())
        return x


def _step0_gather_mtp_forward(monkeypatch, num_tokens, in_warmup):
    """patched_mtp_forward over `num_tokens` rows of 8 under a 16-row step-0
    gather of 2 requests (last tokens at rows 1 and 3, padding gathers row 0),
    with the padding mask written as Eagle3Proposer writes it for that gather
    (2 valid rows of 16)."""
    monkeypatch.setattr(
        patch_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    layer = _Step0GatherMtpLayer()
    model = SimpleNamespace(
        num_mtp_layers=1,
        layers=[layer],
        embed_input_ids=lambda ids: torch.zeros(ids.shape[0], 8),
        pre_fc_norm_embedding=lambda x: x,
        pre_fc_norm_hidden=lambda x: x,
        fc=lambda x: x[:, 8:],
        norm=lambda h, r: (h + r, r),
    )
    hidden = torch.arange(num_tokens * 8, dtype=torch.float32).reshape(-1, 8) / 64
    gather = torch.tensor([1, 3] + [0] * 14)
    padding = token_padding.TokenPaddingState.create(16, "cpu")
    padding.update(2, 16)
    window = shape_variants.warmup() if in_warmup else contextlib.nullcontext()
    with (
        token_padding.use_padding_state(padding),
        step0_post_attn_gather_context(gather),
        window,
    ):
        out = patched_mtp_forward(
            model,
            input_ids=torch.zeros(num_tokens, dtype=torch.long),
            positions=torch.arange(num_tokens),
            hidden_states=hidden,
        )
    return out, layer, hidden, gather


@pytest.mark.parametrize(
    ("num_tokens", "moe_rows"), [(4, 4), (8, 8), (16, 16)], ids=["T4", "T8", "T16"]
)
def test_mtp_forward_runs_a_sliced_step0_gather_and_pads_back(
    monkeypatch, num_tokens, moe_rows
):
    """Traced inside shape_variants.warmup(), a step of at most 4 or 8 tokens
    runs the post-attention layers on the first 4 or 8 gathered rows only, and
    patched_mtp_forward pads the result back to the gather's 16 rows; those
    rows equal the unsliced forward's, and the padding rows are zero."""
    out, layer, hidden, gather = _step0_gather_mtp_forward(
        monkeypatch, num_tokens, in_warmup=True
    )
    full, full_layer, _, _ = _step0_gather_mtp_forward(
        monkeypatch, num_tokens, in_warmup=False
    )

    assert layer.moe_rows == [moe_rows]
    assert full_layer.moe_rows == [16]
    assert out.shape == full.shape == (16, 8)
    # Attention gives a = 6 h sigmoid(h) on the gathered rows (2 h gated by
    # sigmoid(h), o_proj * 3); the post-attention norm makes the residual
    # h + a, and the final norm returns a + (h + a).
    rows = hidden[gather[:moe_rows]]
    torch.testing.assert_close(out[:moe_rows], 12 * rows * torch.sigmoid(rows) + rows)
    torch.testing.assert_close(out[:moe_rows], full[:moe_rows])
    assert not out[moe_rows:].any()


def test_a_sliced_step0_gather_reads_the_padding_mask_over_its_rows(monkeypatch):
    """Eagle3Proposer writes the step-0 gather's mask for all 16 rows (2
    valid). The MoE on the 4-row slice reads its first 4 entries, which mark
    the same rows padding as the full gather does: a step of at most 4 tokens
    has at most 4 requests, all at the front of the gather."""
    _, layer, _, _ = _step0_gather_mtp_forward(monkeypatch, 4, in_warmup=True)
    _, full_layer, _, _ = _step0_gather_mtp_forward(monkeypatch, 4, in_warmup=False)

    assert layer.moe_masks == [[False, False, True, True]]
    assert full_layer.moe_masks == [[False, False] + [True] * 14]


@pytest.fixture
def unpatched_v6(monkeypatch):
    """The classes and loader install_qwen3_next_v6_patches rewrites, restored
    at teardown, with the install flag cleared."""
    from vllm.model_executor.model_loader import base_loader

    monkeypatch.setattr(
        qwen3_next.Qwen3NextAttention,
        "__init__",
        patch_mod.UPSTREAM_ATTN_INIT,
        raising=False,
    )
    monkeypatch.setattr(
        qwen3_next.Qwen3NextAttention, "forward", patch_mod.UPSTREAM_ATTN_FORWARD
    )
    monkeypatch.setattr(
        qwen3_next.Qwen3NextSparseMoeBlock,
        "forward",
        patch_mod.UPSTREAM_SPARSE_MOE_FORWARD,
    )
    monkeypatch.setattr(
        GemmaRMSNorm, "forward_native", patch_mod.UPSTREAM_GEMMA_RMSNORM_FORWARD_NATIVE
    )
    monkeypatch.setattr(
        base_loader,
        "process_weights_after_loading",
        lambda model, model_config, target_device: "processed",
    )
    monkeypatch.setattr(
        base_loader, "_qwen3_next_v6_process_patched", False, raising=False
    )
    monkeypatch.setattr(patch_mod, "_V6_PATCHES_INSTALLED", False)
    return base_loader


def test_install_qwen3_next_v6_patches_wraps_the_loader_once(monkeypatch, unpatched_v6):
    """The install routes Qwen3NextAttention and the sparse MoE block through
    the patched forwards, prebuilds the fused gate's ops when an attention
    layer is built, and has the loader run the post-load setup on every model
    it processes; a second install changes nothing."""
    setups = []
    prebuilds = []
    monkeypatch.setattr(patch_mod, "setup_qwen3_next_v6_after_loading", setups.append)
    monkeypatch.setattr(
        patch_mod, "_prebuild_sigmoid_gate_ops", lambda: prebuilds.append(True)
    )
    attn_inits = []
    monkeypatch.setattr(
        qwen3_next.Qwen3NextAttention,
        "__init__",
        lambda self, *args, **kwargs: attn_inits.append(args),
    )
    patch_mod.install_qwen3_next_v6_patches()
    assert (
        qwen3_next.Qwen3NextAttention.forward
        is patch_mod.patched_qwen3_next_attn_forward
    )
    assert (
        qwen3_next.Qwen3NextSparseMoeBlock.forward
        is patch_mod.patched_sparse_moe_forward
    )
    assert (
        GemmaRMSNorm.forward_native is patch_mod._patched_gemma_rmsnorm_forward_native
    )
    wrapped = unpatched_v6.process_weights_after_loading
    model = torch.nn.Linear(2, 2)
    assert wrapped(model, None, "cpu") == "processed"
    assert setups == [model]

    # Constructing an attention layer runs vLLM's __init__, then prebuilds the
    # fused gate's ops.
    qwen3_next.Qwen3NextAttention.__init__(
        object.__new__(qwen3_next.Qwen3NextAttention), "config"
    )
    assert attn_inits == [("config",)]
    assert prebuilds == [True]

    patch_mod.install_qwen3_next_v6_patches()
    assert unpatched_v6.process_weights_after_loading is wrapped


def test_step0_post_attn_gather_is_installed_with_the_patched_forwards(
    monkeypatch, unpatched_v6
):
    """A draft model runs the step-0 post-attention gather when the attention,
    decoder-layer and MTP forwards are the patched ones."""
    monkeypatch.setattr(patch_mod, "_prebuild_sigmoid_gate_ops", lambda: None)

    class Mtp:
        forward = patch_mod.patched_mtp_forward

    draft = SimpleNamespace(model=Mtp())
    assert not patch_mod.is_step0_post_attn_gather_installed(None)
    assert not patch_mod.is_step0_post_attn_gather_installed(draft)
    patch_mod.install_qwen3_next_v6_patches()
    monkeypatch.setattr(
        qwen3_next.Qwen3NextDecoderLayer, "forward", patched_decoder_layer_forward
    )
    assert patch_mod.is_step0_post_attn_gather_installed(draft)
    assert not patch_mod.is_step0_post_attn_gather_installed(
        SimpleNamespace(model=SimpleNamespace())
    )


def _decoder_layer(cls, qkv_weight_dtype):
    """A real decoder-layer instance carrying only what the post-load setup
    reads: its type, its input norm and its attention's qkv_proj."""
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    layer.layer_type = "full_attention"
    layer.layer_scale = False
    compilation_config = CompilationConfig(
        mode=CompilationMode.NONE, custom_ops=["none"]
    )
    with patch.object(
        custom_op_mod, "get_cached_compilation_config", lambda: compilation_config
    ):
        layer.input_layernorm = GemmaRMSNorm(64, eps=1e-6)
    qkv_proj = nn.Module()
    weight = torch.randn(64, 96).to(qkv_weight_dtype)
    qkv_proj.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
    qkv_proj.register_parameter(
        "weight_scale", nn.Parameter(torch.ones(96), requires_grad=False)
    )
    qkv_proj.quant_method = (
        _fp8_method() if qkv_weight_dtype == torch.float8_e4m3fn else SimpleNamespace()
    )
    layer.self_attn = nn.Module()
    layer.self_attn.qkv_proj = qkv_proj
    layer.mlp = nn.Module()
    return layer


@pytest.mark.parametrize(
    "cls", [qwen3_next.Qwen3NextDecoderLayer, qwen3_5.Qwen3_5DecoderLayer]
)
def test_setup_after_loading_marks_where_the_fused_input_norm_quant_applies(
    monkeypatch, cls
):
    """The post-load setup hoists each GemmaRMSNorm's (1 + w) and marks a
    decoder layer, Qwen3.5's subclass included, eligible for the fused input
    norm + FP8 quantization only when its qkv_proj takes pre-quantized rows."""
    monkeypatch.setattr(patch_mod, "_prebuild_norm_ops", lambda eps: None)
    fp8 = _decoder_layer(cls, torch.float8_e4m3fn)
    bf16 = _decoder_layer(cls, torch.bfloat16)
    model = nn.ModuleList([fp8, bf16])
    patch_mod.setup_qwen3_next_v6_after_loading(model)
    assert fp8._pre_norm_quant_eligible and fp8._pre_norm_quant_exact
    assert not bf16._pre_norm_quant_eligible and not bf16._pre_norm_quant_exact
    for layer in (fp8, bf16):
        norm = layer.input_layernorm
        torch.testing.assert_close(
            norm._tpu_weight_plus_one, 1.0 + norm.weight.data.float()
        )
