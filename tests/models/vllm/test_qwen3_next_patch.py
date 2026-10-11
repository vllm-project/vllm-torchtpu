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
from unittest.mock import patch

import pytest
import torch
import vllm
from vllm.model_executor.models import qwen3_5, qwen3_5_mtp, qwen3_next

import vllm_torchtpu.envs as envs
from vllm_torchtpu.models.vllm import qwen3_next_patch as patch_mod
from vllm_torchtpu.models.vllm.qwen3_next_patch import (
    UPSTREAM_DECODER_LAYER_FORWARD,
    UPSTREAM_MODEL_FORWARD,
    _check_tokens_divide_by_tp,
    maybe_patch_qwen3_next,
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


def test_patched_decoder_layer_forward_fp8_gather(monkeypatch):
    """When VLLM_TPU_SEQUENCE_PARALLEL_FP8_GATHER=1, quantize before all-gather."""
    tp = 2
    monkeypatch.setattr(envs, "VLLM_TPU_SEQUENCE_PARALLEL_FP8_GATHER", True)
    monkeypatch.setattr(envs, "VLLM_TPU_SP_FP8_GATHER_MIN_TOKENS", 0)
    monkeypatch.setattr(patch_mod, "_can_use_sp_fp8_gather", lambda self, n=None: True)
    monkeypatch.setattr(
        patch_mod,
        "quantize_rows",
        lambda x: (
            x.to(torch.float8_e4m3fn),
            torch.ones((x.shape[0], 1), dtype=torch.float32),
        ),
    )
    gathered_dtypes: list[torch.dtype] = []

    def fake_all_gather(x, dim=0):
        gathered_dtypes.append(x.dtype)
        return torch.cat([x] * tp, dim=dim)

    monkeypatch.setattr(patch_mod, "tensor_model_parallel_all_gather", fake_all_gather)
    monkeypatch.setattr(
        patch_mod,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim=0: x.narrow(dim, 0, x.shape[dim] // tp),
    )
    monkeypatch.setattr(patch_mod, "get_tensor_model_parallel_world_size", lambda: tp)

    class DummySelfAttn:
        def __init__(self):
            self.qkv_proj = torch.nn.Linear(8, 8, bias=False)
            self.seen_prequant = None

        def __call__(self, hidden_states, positions):
            self.seen_prequant = getattr(self.qkv_proj, "_tpu_sp_prequant", None)
            return torch.ones((positions.shape[-1], 8), dtype=torch.bfloat16)

    class DummyLayer:
        use_attn_reduce_scatter_for_moe = True
        layer_type = "full_attention"
        layer_scale = False

        def __init__(self):
            self.self_attn = DummySelfAttn()

        def input_layernorm(self, x, residual=None):
            return x if residual is None else (x, residual)

        def post_attention_layernorm(self, x, residual):
            return x, residual

        def mlp(self, x, **kwargs):
            return x

    layer = DummyLayer()
    hidden = torch.ones((4, 8), dtype=torch.bfloat16)
    positions = torch.arange(8)
    out_h, _ = patched_decoder_layer_forward(layer, hidden, None, positions)
    assert out_h.shape == (4, 8)
    assert gathered_dtypes == [torch.float8_e4m3fn, torch.float32]
    assert layer.self_attn.seen_prequant is not None
    assert getattr(layer.self_attn.qkv_proj, "_tpu_sp_prequant", None) is None


def test_patched_decoder_layer_forward_sp_min_tokens(monkeypatch):
    """Buckets below VLLM_TPU_SP_MIN_TOKENS use all-reduce instead of SP."""
    monkeypatch.setattr(envs, "VLLM_TPU_SP_MIN_TOKENS", 64)
    all_reduce_called = []
    monkeypatch.setattr(
        patch_mod,
        "tensor_model_parallel_all_reduce",
        lambda x: (all_reduce_called.append(True), x)[1],
    )

    class DummyLayer:
        use_attn_reduce_scatter_for_moe = True
        layer_type = "linear_attention"
        layer_scale = False

        def input_layernorm(self, x, residual=None):
            return x if residual is None else (x, residual)

        def linear_attn(self, hidden_states):
            return hidden_states

        def post_attention_layernorm(self, x, residual):
            return x, residual

        def mlp(self, x, **kwargs):
            assert not kwargs.get("already_sequence_parallel", False)
            return x

    layer = DummyLayer()
    hidden = torch.ones((16, 8), dtype=torch.bfloat16)
    positions = torch.arange(16)
    out_h, _ = patched_decoder_layer_forward(layer, hidden, None, positions)
    assert out_h.shape == (16, 8)
    assert all_reduce_called == [True]
