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

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.models import qwen3_5, qwen3_5_mtp, qwen3_next, qwen3_next_mtp

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.models.vllm import qwen3_next_patch as patch_mod
from vllm_torchtpu.models.vllm.qwen3_5_mtp import (
    UPSTREAM_PREDICTOR_FORWARD as UPSTREAM_QWEN3_5_MTP_FORWARD,
)
from vllm_torchtpu.models.vllm.qwen3_next_patch import (
    UPSTREAM_DECODER_LAYER_FORWARD,
    UPSTREAM_MODEL_FORWARD,
    UPSTREAM_MTP_FORWARD,
    compute_fused_gate_and_shared_expert,
    maybe_flip_gdn_in_proj_qkvz_weight,
    maybe_patch_qwen3_next,
    maybe_setup_fused_gate_se,
    pallas_sigmoid_gate_quant,
    patched_decoder_layer_forward,
    patched_model_forward,
    patched_mtp_forward,
    patched_sparse_moe_forward,
    sigmoid_gate_quant_reference,
    step0_post_attn_gather_context,
)

# The patched copies track vLLM v0.32.0 (pyproject.toml pin).
PINNED_VLLM_VERSION = "0.32.0"
PINNED_DECODER_LAYER_HASH = (
    "359c047f120f585e0c519a3bc438f037f996f4d48c7b64495dadddc59a709b4a"
)
PINNED_MODEL_FORWARD_HASH = (
    "a9c1265f14e88e54138394b9038db621813bb6357a9077ad50d1691db0ac4398"
)
# qwen3_5_mtp.py replaces this upstream drafter forward with a copy that
# keeps its SP chunk before the MTP layer and all-gather after the final
# norm, which is what lets tpu_platform admit MTP under SP.
PINNED_QWEN3_5_MTP_FORWARD_HASH = (
    "af988ccd1ba49bf2953bb9858d7a2082c405d260d2ebb64413c6707c34a80d22"
)
# patched_mtp_forward copies this drafter forward (Qwen3Next and Qwen3.5 share
# its source), and tpu_platform admits MTP under SP because its layer slices
# the rows after attention and it all-gathers after the final norm.
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
    assert (
        _source_hash(UPSTREAM_QWEN3_5_MTP_FORWARD) == PINNED_QWEN3_5_MTP_FORWARD_HASH
    ), (
        "Upstream Qwen3_5MultiTokenPredictor.forward has drifted. Re-sync "
        "patched_mtp_forward and re-check that its layer still slices and it "
        "all-gathers under SP before admitting MTP in tpu_platform.py."
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
    num_tokens = 8
    hidden = torch.randn(num_tokens // tp, 64)
    residual = torch.randn(num_tokens // tp, 64)

    patched_decoder_layer_forward(layer, hidden, residual, torch.arange(num_tokens))

    (kwargs,) = attn_calls
    x_q, x_s = kwargs["prequant"]
    assert x_q.dtype == torch.bfloat16
    assert x_q.shape == (num_tokens, 64)
    assert x_s.shape == (num_tokens, 1)


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


def _bf16_jnp(t: torch.Tensor):
    import jax.numpy as jnp

    return jnp.asarray(t.float().numpy(), dtype=jnp.bfloat16)


def _gemma_norm(weight: torch.Tensor, eps: float = 1e-6) -> SimpleNamespace:
    return SimpleNamespace(variance_epsilon=eps, weight=weight)


@pytest.mark.parametrize("with_residual", [True, False])
def test_fused_norm_quantize_rows_matches_norm_then_quantize(with_residual):
    """The fused input-norm op (TPU_FUSE_PRE_ATTN_NORM_QUANT) returns the
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
    """The fused post-attention op (TPU_FUSE_POST_ATTN_NORM_MOE) is the
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
    monkeypatch.delenv("TPU_FUSE_PRE_ATTN_NORM_QUANT", raising=False)
    fuses = patch_mod._fuses_pre_attn_norm_quant
    layer = SimpleNamespace(_pre_norm_quant_eligible=True, _pre_norm_quant_exact=False)

    assert fuses(layer, _tpu_rows(), True)
    assert not fuses(layer, _tpu_rows(), False)
    layer._pre_norm_quant_exact = True
    assert fuses(layer, _tpu_rows(), False)

    assert not fuses(layer, torch.zeros(2, 2, dtype=torch.bfloat16), True)
    assert not fuses(layer, _tpu_rows(torch.float32), True)
    assert not fuses(SimpleNamespace(), _tpu_rows(), True)
    monkeypatch.setenv("TPU_FUSE_PRE_ATTN_NORM_QUANT", "0")
    assert not fuses(layer, _tpu_rows(), True)
    monkeypatch.delenv("TPU_FUSE_PRE_ATTN_NORM_QUANT")
    _sequence_parallel(monkeypatch, fp8_gather=False)
    assert not fuses(layer, _tpu_rows(), True)


def test_post_attn_norm_moe_fusion_predicate(monkeypatch):
    monkeypatch.delenv("TPU_FUSE_POST_ATTN_NORM_MOE", raising=False)
    fuses = patch_mod._fuses_post_attn_norm_moe
    norm = GemmaRMSNorm.__new__(GemmaRMSNorm)
    layer = SimpleNamespace(
        mlp=SimpleNamespace(_fused_gate_gu_fp8_weight=torch.zeros(1)),
        post_attention_layernorm=norm,
    )
    assert fuses(layer, _tpu_rows())
    assert not fuses(layer, torch.zeros(2, 2, dtype=torch.bfloat16))
    assert not fuses(layer, _tpu_rows(torch.float32))
    monkeypatch.setenv("TPU_FUSE_POST_ATTN_NORM_MOE", "off")
    assert not fuses(layer, _tpu_rows())
    monkeypatch.delenv("TPU_FUSE_POST_ATTN_NORM_MOE")
    layer.mlp = SimpleNamespace()
    assert not fuses(layer, _tpu_rows())
    layer.mlp = SimpleNamespace(_fused_gate_gu_fp8_weight=torch.zeros(1))
    layer.post_attention_layernorm = _gemma_norm(torch.zeros(4))
    assert not fuses(layer, _tpu_rows())


def _proj(dtype=torch.float8_e4m3fn, scale_shape=(4,)) -> SimpleNamespace:
    return SimpleNamespace(
        weight=torch.zeros(4, 4, dtype=dtype),
        weight_scale=torch.ones(scale_shape),
        quant_method=SimpleNamespace(apply_prequant=lambda *args: None),
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
    # The pooled PCP forward ignores prequant.
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
