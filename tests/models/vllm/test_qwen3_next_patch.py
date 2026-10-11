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

import hashlib
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm
from vllm.model_executor.models import qwen3_5, qwen3_5_mtp, qwen3_next, qwen3_next_mtp

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.models.vllm import qwen3_next_patch as patch_mod
from vllm_torchtpu.models.vllm.qwen3_next_patch import (
    UPSTREAM_DECODER_LAYER_FORWARD,
    UPSTREAM_MODEL_FORWARD,
    UPSTREAM_MTP_FORWARD,
    _check_tokens_divide_by_tp,
    compute_fused_gate_and_shared_expert,
    maybe_flip_gdn_in_proj_qkvz_weight,
    maybe_patch_qwen3_next,
    maybe_setup_fused_gate_se,
    pallas_sigmoid_gate_quant,
    patched_decoder_layer_forward,
    patched_mtp_forward,
    patched_sparse_moe_forward,
    sigmoid_gate_quant_reference,
    step0_post_attn_gather_context,
)

# Captured before any test installs the patch.
UPSTREAM_QWEN3_5_MTP_FORWARD = qwen3_5_mtp.Qwen3_5MultiTokenPredictor.forward

# The patched copies track vLLM v0.29.0 (pyproject.toml pin).
PINNED_VLLM_VERSION = "0.29.0"
PINNED_DECODER_LAYER_HASH = (
    "359c047f120f585e0c519a3bc438f037f996f4d48c7b64495dadddc59a709b4a"
)
PINNED_MODEL_FORWARD_HASH = (
    "a9c1265f14e88e54138394b9038db621813bb6357a9077ad50d1691db0ac4398"
)
# patched_mtp_forward copies this drafter forward (Qwen3Next and Qwen3.5 share
# its source), and tpu_platform admits MTP under SP because it chunks before
# the MTP layer and all-gathers after the final norm.
PINNED_MTP_FORWARD_HASH = (
    "54034d76e48fccf2cd98bcf05b831207580ad9e8886e6e387193e1bd70e9e472"
)
requires_pinned_vllm = pytest.mark.skipif(
    vllm.__version__ != PINNED_VLLM_VERSION,
    reason=f"patched forwards are copies of vLLM {PINNED_VLLM_VERSION}",
)


def _source_hash(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()


@pytest.fixture
def restore_forwards(monkeypatch):
    """Start from the upstream forwards and undo whatever the patch installs."""
    monkeypatch.setattr(
        qwen3_next.Qwen3NextDecoderLayer, "forward", UPSTREAM_DECODER_LAYER_FORWARD
    )
    monkeypatch.setattr(qwen3_next.Qwen3NextModel, "forward", UPSTREAM_MODEL_FORWARD)
    monkeypatch.setattr(
        qwen3_next_mtp.Qwen3NextMultiTokenPredictor, "forward", UPSTREAM_MTP_FORWARD
    )
    monkeypatch.setattr(
        qwen3_5_mtp.Qwen3_5MultiTokenPredictor, "forward", UPSTREAM_QWEN3_5_MTP_FORWARD
    )
    for layer_cls in (qwen3_next.Qwen3NextDecoderLayer, qwen3_5.Qwen3_5DecoderLayer):
        if "__init__" in vars(layer_cls):
            monkeypatch.setattr(layer_cls, "__init__", layer_cls.__init__)
    return monkeypatch


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
    """Pin the drafter forwards patched_mtp_forward copies and replaces."""
    assert _source_hash(UPSTREAM_MTP_FORWARD) == PINNED_MTP_FORWARD_HASH, (
        "Upstream Qwen3NextMultiTokenPredictor.forward has drifted. "
        "Re-sync patched_mtp_forward in qwen3_next_patch.py."
    )
    assert _source_hash(UPSTREAM_QWEN3_5_MTP_FORWARD) == PINNED_MTP_FORWARD_HASH, (
        "Upstream Qwen3_5MultiTokenPredictor.forward has drifted. Re-sync "
        "patched_mtp_forward and re-check that it still chunks and all-gathers "
        "under SP before admitting MTP in tpu_platform.py."
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
    """Verify the SP branch runs all-gather, narrow, reduce-scatter, and SP MLP."""

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
    num_tokens = 8
    hidden = torch.randn(num_tokens // tp, 64)
    residual = torch.randn(num_tokens // tp, 64)

    patched_decoder_layer_forward(layer, hidden, residual, torch.arange(num_tokens))

    (kwargs,) = attn_calls
    x_q, x_s = kwargs["prequant"]
    assert x_q.dtype == torch.bfloat16
    assert x_q.shape == (num_tokens, 64)
    assert x_s.shape == (num_tokens, 1)


def test_patched_decoder_layer_forward_mtp_entry_prechunked(monkeypatch):
    """Verify MTP layer entry: residual=None with pre-chunked hidden_states."""
    full_num_tokens = 16
    tp_world_size = 8
    chunk = full_num_tokens // tp_world_size
    hidden_size = 64

    _fake_tp_collectives(monkeypatch, tp_world_size)
    _sequence_parallel(monkeypatch, min_rows=0)

    layer = MagicMock()
    layer.use_attn_reduce_scatter_for_moe = True
    layer.layer_type = "linear_attention"
    layer.layer_scale = False
    layer.input_layernorm.side_effect = lambda h, r=None: (h, r) if r is not None else h
    layer.linear_attn.side_effect = lambda hidden_states: hidden_states
    layer.post_attention_layernorm.side_effect = lambda h, r: (h, r)
    layer.mlp.side_effect = lambda h, already_sequence_parallel=False: h

    prechunked_hidden = torch.randn(chunk, hidden_size)
    positions = torch.arange(full_num_tokens).unsqueeze(0)

    out_h, out_r = patched_decoder_layer_forward(
        layer,
        hidden_states=prechunked_hidden,
        residual=None,
        positions=positions,
    )

    assert out_h.shape == (chunk, hidden_size)
    assert out_r.shape == (chunk, hidden_size)


def _dummy_mtp(layer_rows: list[int]):
    """An MTP predictor whose single layer records the rows it receives."""

    class Layer:
        use_attn_reduce_scatter_for_moe = True

        def __call__(self, positions, hidden_states, residual):
            layer_rows.append(hidden_states.shape[0])
            return hidden_states, hidden_states

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
    """The drafter chunks before its layer and gathers after the norm, unless
    the step-0 post-attention gather keeps its rows replicated."""

    tp = 2
    _fake_tp_collectives(monkeypatch, tp)
    _sequence_parallel(monkeypatch, min_rows=0)
    monkeypatch.setattr(
        patch_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        patch_mod.qwen3_next,
        "sequence_parallel_chunk",
        lambda x: x.narrow(0, 0, x.shape[0] // tp),
    )
    layer_rows: list[int] = []
    model = _dummy_mtp(layer_rows)
    num_tokens = 8
    indices = torch.arange(2) if with_step0_gather else None
    with step0_post_attn_gather_context(indices):
        out = patched_mtp_forward(
            model,
            input_ids=torch.zeros(num_tokens, dtype=torch.long),
            positions=torch.arange(num_tokens),
            hidden_states=torch.ones(num_tokens, 4),
        )
    assert layer_rows == [num_tokens if with_step0_gather else num_tokens // tp]
    assert out.shape == (num_tokens, 4)


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


def test_forward_patch_skipped_with_flags_off(restore_forwards):
    """With TPU_MOE_SEQUENCE_PARALLEL=0 and TPU_MTP_STEP0_POST_ATTN_GATHER=0,
    the forwards remain the upstream objects."""
    restore_forwards.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False)
    restore_forwards.setattr(envs, "TPU_MTP_STEP0_POST_ATTN_GATHER", False)
    result = maybe_patch_qwen3_next()
    assert result is None
    assert qwen3_next.Qwen3NextDecoderLayer.forward is UPSTREAM_DECODER_LAYER_FORWARD
    assert qwen3_next.Qwen3NextModel.forward is UPSTREAM_MODEL_FORWARD
    assert qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward is UPSTREAM_MTP_FORWARD
    assert (
        qwen3_5_mtp.Qwen3_5MultiTokenPredictor.forward is UPSTREAM_QWEN3_5_MTP_FORWARD
    )


@pytest.mark.parametrize("sequence_parallel", [True, False])
def test_forward_patch_applied(restore_forwards, sequence_parallel):
    """With either flag on, the inherited Qwen3.5 forwards and the drafter
    forwards are replaced."""
    restore_forwards.setattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", sequence_parallel)
    restore_forwards.setattr(envs, "TPU_MTP_STEP0_POST_ATTN_GATHER", True)
    restore_forwards.setattr(patch_mod.sequence_parallel, "fp8_gather", lambda: False)

    result = maybe_patch_qwen3_next()
    assert result is None
    assert qwen3_next.Qwen3NextDecoderLayer.forward is patched_decoder_layer_forward
    assert qwen3_next.Qwen3NextModel.forward is not UPSTREAM_MODEL_FORWARD
    assert qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward is patched_mtp_forward
    assert qwen3_5_mtp.Qwen3_5MultiTokenPredictor.forward is patched_mtp_forward
    assert (
        getattr(qwen3_next.Qwen3NextDecoderLayer.__init__, "_tpu_sp_rank_init", False)
        is sequence_parallel
    )


def test_pal_gate_q_interpreter_parity():
    """Verify pallas_sigmoid_gate_quant in Pallas CPU interpreter."""
    import jax.numpy as jnp

    rng = np.random.default_rng(42)
    for m in (4, 8, 16, 64):
        a = jnp.asarray(rng.standard_normal((m, 256)), dtype=jnp.bfloat16)
        g = jnp.asarray(rng.standard_normal((m, 256)), dtype=jnp.bfloat16)
        q_pal, s_pal = pallas_sigmoid_gate_quant(a, g, interpret=True)
        q_ref, s_ref = sigmoid_gate_quant_reference(a, g)
        assert q_pal.shape == (m, 256) and s_pal.shape == (m, 1)
        np.testing.assert_allclose(
            np.asarray(s_pal, np.float32),
            np.asarray(s_ref, np.float32),
            rtol=1e-2,
            atol=1e-3,
        )
        deq_pal = np.asarray(q_pal, np.float32) * np.asarray(s_pal, np.float32)
        deq_ref = np.asarray(q_ref, np.float32) * np.asarray(s_ref, np.float32)
        np.testing.assert_allclose(deq_pal, deq_ref, rtol=0.15, atol=0.15)
        assert np.mean(np.abs(deq_pal - deq_ref)) < 5e-3


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

        down_mod = nn.Module()
        down_mod.register_parameter(
            "weight", nn.Parameter(w_se_down_fp8.clone(), requires_grad=False)
        )
        down_mod.register_parameter(
            "weight_scale", nn.Parameter(s_se_down.float().clone(), requires_grad=False)
        )
        down_mod.bias = None
        setattr(down_mod, WEIGHT_FLIPPED_ATTR, True)

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
                "__call__": lambda self, hidden_states, router_logits: (
                    hidden_states * 0.5
                ),
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


@pytest.mark.parametrize(
    ("rows", "cols", "expected"),
    [
        (16, 1024, 16),  # < 32: full shape
        (32, 1024, 32),  # 32-aligned decode bucket (TP=8)
        (64, 1024, 32),  # 32-aligned sub-block for 2D float8_e4m3fn (32, 128) tiling
        (512, 1024, 64),  # TP=8 prefill bucket
        (4096, 4096, 32),  # TP=2 prefill bucket (must stay 32, not 4096)
    ],
)
def test_sigmoid_gate_quant_tile_selection(rows, cols, expected):
    bm = patch_mod._legal_tile(rows, patch_mod._select_gate_q_tile(rows, cols))
    assert bm == expected
    assert bm == rows or (rows % bm == 0 and bm % 32 == 0)


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
