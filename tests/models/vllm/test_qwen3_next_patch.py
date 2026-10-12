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
"""Pin upstream Qwen3.5 MoE forwards and verify trace-stable SP forward patch."""

import hashlib
import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import vllm
from vllm.model_executor.models import qwen3_5, qwen3_5_mtp, qwen3_next

import vllm_torchtpu.envs as envs
from vllm_torchtpu.models.vllm import qwen3_next_patch as patch_mod
from vllm_torchtpu.models.vllm.qwen3_next_patch import (
    UPSTREAM_ATTENTION_FORWARD,
    UPSTREAM_DECODER_LAYER_FORWARD,
    UPSTREAM_MODEL_FORWARD,
    _check_tokens_divide_by_tp,
    maybe_patch_qwen3_next,
    patched_attention_forward,
    patched_decoder_layer_forward,
)

# The patched copies track vLLM v0.29.0 (pyproject.toml pin).
PINNED_VLLM_VERSION = "0.29.0"
PINNED_DECODER_LAYER_HASH = (
    "359c047f120f585e0c519a3bc438f037f996f4d48c7b64495dadddc59a709b4a"
)
PINNED_MODEL_FORWARD_HASH = (
    "a9c1265f14e88e54138394b9038db621813bb6357a9077ad50d1691db0ac4398"
)
PINNED_ATTENTION_FORWARD_HASH = (
    "ec7ceb7de698d7a05fab4d77b9a5579afeb66edc4960c05a65a3144cfabf9578"
)
# tpu_platform admits MTP under SP because this upstream drafter forward
# chunks before the MTP layer and all-gathers after the final norm.
PINNED_QWEN3_5_MTP_FORWARD_HASH = (
    "54034d76e48fccf2cd98bcf05b831207580ad9e8886e6e387193e1bd70e9e472"
)
requires_pinned_vllm = pytest.mark.skipif(
    vllm.__version__ != PINNED_VLLM_VERSION,
    reason=f"patched forwards are copies of vLLM {PINNED_VLLM_VERSION}",
)


@requires_pinned_vllm
def test_upstream_qwen3_next_forward_unchanged():
    """Fail when upstream vLLM modifies either method, requiring a re-sync."""
    actual_decoder_hash = hashlib.sha256(
        inspect.getsource(UPSTREAM_DECODER_LAYER_FORWARD).encode("utf-8")
    ).hexdigest()
    assert actual_decoder_hash == PINNED_DECODER_LAYER_HASH, (
        "Upstream Qwen3NextDecoderLayer.forward has drifted. "
        "Re-sync src/vllm_torchtpu/models/vllm/qwen3_next_patch.py."
    )

    actual_model_hash = hashlib.sha256(
        inspect.getsource(UPSTREAM_MODEL_FORWARD).encode("utf-8")
    ).hexdigest()
    assert actual_model_hash == PINNED_MODEL_FORWARD_HASH, (
        "Upstream Qwen3NextModel.forward has drifted. "
        "Re-sync src/vllm_torchtpu/models/vllm/qwen3_next_patch.py."
    )


@requires_pinned_vllm
def test_upstream_qwen3_next_attention_forward_unchanged():
    """patched_attention_forward copies this body for the fp8 gather."""
    source = inspect.getsource(UPSTREAM_ATTENTION_FORWARD)
    actual_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert actual_hash == PINNED_ATTENTION_FORWARD_HASH, (
        "Upstream Qwen3NextAttention.forward has drifted. "
        "Re-sync patched_attention_forward in qwen3_next_patch.py."
    )


@requires_pinned_vllm
def test_upstream_qwen3_5_mtp_forward_unchanged():
    """Pin the Qwen3.5 drafter forward that the SP admission gate relies on."""
    source = inspect.getsource(qwen3_5_mtp.Qwen3_5MultiTokenPredictor.forward)
    actual_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert actual_hash == PINNED_QWEN3_5_MTP_FORWARD_HASH, (
        "Upstream Qwen3_5MultiTokenPredictor.forward has drifted. Re-check that "
        "it still chunks and all-gathers under SP before admitting MTP in "
        "tpu_platform.py."
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


def test_patched_decoder_layer_forward_parity():
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

    patched_h, patched_r = patched_decoder_layer_forward(
        layer, hidden, residual, positions
    )
    upstream_h, upstream_r = UPSTREAM_DECODER_LAYER_FORWARD(
        layer, hidden, residual, positions
    )
    torch.testing.assert_close(patched_h, upstream_h)
    torch.testing.assert_close(patched_r, upstream_r)


def test_patched_decoder_layer_forward_sequence_parallel(monkeypatch):
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


def test_forward_patch_skipped_with_flag_off(monkeypatch):
    """When TPU_MOE_SEQUENCE_PARALLEL=0, methods remain the upstream objects."""
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False)
    result = maybe_patch_qwen3_next()
    assert result is None
    assert qwen3_next.Qwen3NextDecoderLayer.forward is UPSTREAM_DECODER_LAYER_FORWARD
    assert qwen3_next.Qwen3NextModel.forward is UPSTREAM_MODEL_FORWARD


def test_forward_patch_applied(monkeypatch):
    """With the flag on, the inherited Qwen3.5 forwards are replaced."""
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    monkeypatch.setattr(
        qwen3_next.Qwen3NextDecoderLayer,
        "forward",
        UPSTREAM_DECODER_LAYER_FORWARD,
    )
    monkeypatch.setattr(qwen3_next.Qwen3NextModel, "forward", UPSTREAM_MODEL_FORWARD)

    result = maybe_patch_qwen3_next()
    assert result is None
    assert (
        qwen3_next.Qwen3NextDecoderLayer.forward is not UPSTREAM_DECODER_LAYER_FORWARD
    )
    assert qwen3_next.Qwen3NextModel.forward is not UPSTREAM_MODEL_FORWARD


def _sp_layer(seen):
    class DummyLayer:
        use_attn_reduce_scatter_for_moe = True
        layer_type = "linear_attention"
        layer_scale = False

        def input_layernorm(self, x, residual=None):
            return x * 1.05 if residual is None else (x * 1.05, residual + x * 0.05)

        def linear_attn(self, hidden_states, **kwargs):
            seen.append((hidden_states, kwargs))
            prequant = kwargs.get("prequant")
            rows = hidden_states if prequant is None else prequant[0] * prequant[1]
            return rows + 1.0

        def post_attention_layernorm(self, x, residual):
            return x * 1.1, residual + x * 0.1

        def mlp(self, x, **kwargs):
            return x * 2.0

    return DummyLayer()


@pytest.mark.parametrize("fp8_gather", [False, True], ids=["bf16", "fp8"])
def test_patched_decoder_layer_forward_fp8_gather(monkeypatch, fp8_gather):
    """The fp8 gather moves the rows and scales each rank quantized; the
    projections see the step's rows either way."""
    tp = 2
    gathers = []

    def all_gather(x, dim=0):
        gathers.append(x.dtype)
        return torch.cat([x] * tp, dim=dim)

    monkeypatch.setattr(patch_mod, "tensor_model_parallel_all_gather", all_gather)
    monkeypatch.setattr(
        patch_mod,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim=0: x.narrow(dim, 0, x.shape[dim] // tp) * tp,
    )
    monkeypatch.setattr(patch_mod, "get_tensor_model_parallel_world_size", lambda: tp)
    # A stand-in quantizer that is exact, so both forms give the same output.
    scale = 0.5
    monkeypatch.setattr(
        patch_mod.linear_common,
        "quantize_rows",
        lambda x: (x / scale, x.new_full((x.shape[0], 1), scale)),
    )
    monkeypatch.setattr(patch_mod, "_fp8_gather", fp8_gather)

    seen = []
    num_tokens = 8
    hidden = torch.randn(num_tokens // tp, 64)
    residual = torch.randn(num_tokens // tp, 64)
    positions = torch.arange(num_tokens)
    out_h, out_r = patched_decoder_layer_forward(
        _sp_layer(seen), hidden, residual, positions
    )

    attn_shard = (hidden * 1.05 + 1.0) * tp
    torch.testing.assert_close(out_h, attn_shard * 1.1 * 2.0)
    torch.testing.assert_close(out_r, residual + hidden * 0.05 + attn_shard * 0.1)
    ((rows, kwargs),) = seen
    assert rows.shape[0] == num_tokens
    if fp8_gather:
        x_q, x_s = kwargs["prequant"]
        assert rows is x_q
        assert x_q.shape == (num_tokens, 64) and x_s.shape == (num_tokens, 1)
        assert len(gathers) == 2
    else:
        assert kwargs == {}
        assert len(gathers) == 1


def test_patched_attention_forward_projects_the_gathered_rows(monkeypatch):
    original = Mock(return_value="upstream")
    monkeypatch.setattr(patch_mod, "UPSTREAM_ATTENTION_FORWARD", original)
    monkeypatch.setattr(
        patch_mod.linear_common,
        "project_rows",
        lambda layer, x, prequant: prequant[0] + 1,
    )
    attn = SimpleNamespace(
        qkv_proj=object(),
        _project_qkv_gate=lambda qkv, positions: (qkv, qkv, qkv, None),
        attn=lambda q, k, v: q + k + v,
        o_proj=lambda x: (x * 2, None),
    )
    positions, x = torch.arange(4), torch.ones(4, 8)
    assert patched_attention_forward(attn, positions, x) == "upstream"
    original.assert_called_once_with(attn, positions, x)
    torch.testing.assert_close(
        patched_attention_forward(attn, positions, x, prequant=(x, None)),
        torch.full((4, 8), 12.0),
    )


@pytest.mark.parametrize("fp8_gather", [False, True])
def test_attention_patch_applied_only_with_fp8_gather(monkeypatch, fp8_gather):
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", True)
    monkeypatch.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL_FP8_GATHER", fp8_gather)
    monkeypatch.setattr(
        qwen3_next.Qwen3NextDecoderLayer, "forward", UPSTREAM_DECODER_LAYER_FORWARD
    )
    monkeypatch.setattr(qwen3_next.Qwen3NextModel, "forward", UPSTREAM_MODEL_FORWARD)
    monkeypatch.setattr(
        qwen3_next.Qwen3NextAttention, "forward", UPSTREAM_ATTENTION_FORWARD
    )
    monkeypatch.setattr(patch_mod, "_fp8_gather", False)

    maybe_patch_qwen3_next()
    assert patch_mod._fp8_gather is fp8_gather
    patched = qwen3_next.Qwen3NextAttention.forward is patched_attention_forward
    assert patched is fp8_gather
