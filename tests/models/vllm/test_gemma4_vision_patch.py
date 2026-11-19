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

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from tests.layers.adapter.test_vision_attention import (
    _reference_varlen_sdpa,
    make_attention,
)
from vllm_torchtpu.layers.adapter.attention_backend import (
    ensure_flywheel_enum_registered,
)
from vllm_torchtpu.layers.adapter.vision_attention import TpuMMEncoderAttention
from vllm_torchtpu.models.vllm.gemma4_vision_patch import (
    build_padded_row_cu_seqlens,
    install_gemma4_vision_patches,
    maybe_patch_gemma4_vision,
)


def test_maybe_patch_gemma4_vision_skips_non_gemma4_configs():
    assert maybe_patch_gemma4_vision(None) is False
    non_vision = SimpleNamespace(hf_config=SimpleNamespace(vision_config=None))
    assert maybe_patch_gemma4_vision(non_vision) is False
    other_vision = SimpleNamespace(
        hf_config=SimpleNamespace(
            vision_config=SimpleNamespace(model_type="qwen2_5_vl")
        )
    )
    assert maybe_patch_gemma4_vision(other_vision) is False


def test_build_padded_row_cu_seqlens_2d_and_4d_masks():
    device = torch.device("cpu")
    mask_2d = torch.tensor(
        [[True, True, False, False], [True, True, True, True]], dtype=torch.bool
    )
    cu_2d, valid_2d = build_padded_row_cu_seqlens(2, 4, mask_2d, device)
    assert cu_2d.tolist() == [0, 2, 4, 8, 8]
    assert valid_2d is not None and torch.equal(valid_2d, mask_2d)

    mask_4d = torch.full((2, 1, 4, 4), float("-inf"))
    mask_4d[0, 0, :, :2] = 0.0
    mask_4d[1, 0, :, :3] = 0.0
    cu_4d, valid_4d = build_padded_row_cu_seqlens(2, 4, mask_4d, device)
    assert cu_4d.tolist() == [0, 2, 4, 7, 8]
    assert valid_4d is not None
    assert valid_4d.tolist() == [
        [True, True, False, False],
        [True, True, True, False],
    ]

    with pytest.raises(ValueError, match="Expected 2D or 4D attention_mask"):
        build_padded_row_cu_seqlens(2, 4, torch.ones(2, 4, 4), device)


def test_gemma4_vision_patch_matches_reference_masked_attention(monkeypatch):
    modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    cfg = Gemma4VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    ref_encoder = modeling_gemma4.Gemma4VisionEncoder(cfg).eval()

    bsz, seq_len = 3, 8
    inputs_embeds = torch.randn(bsz, seq_len, 64)
    # Row 0: 5 valid + 3 pad; Row 1: 8 valid + 0 pad; Row 2: 2 valid + 6 pad.
    valid_lens = [5, 8, 2]
    attention_mask = torch.zeros(bsz, seq_len, dtype=torch.bool)
    pixel_pos = torch.full((bsz, seq_len, 2), -1, dtype=torch.long)
    for b, vlen in enumerate(valid_lens):
        attention_mask[b, :vlen] = True
        pixel_pos[b, :vlen, 0] = torch.arange(vlen)
        pixel_pos[b, :vlen, 1] = 0

    with torch.no_grad():
        ref_out = ref_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_position_ids=pixel_pos,
        ).last_hidden_state

    class FakeMM(torch.nn.Module):
        def __init__(self, num_heads, head_size, scale=None, num_kv_heads=None, **kw):
            super().__init__()
            self.num_heads = num_heads
            self.head_size = head_size
            self.scale = 1.0 if scale is None else scale
            self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
            self.attn_backend = ensure_flywheel_enum_registered()

        def forward(self, q, k, v, cu_seqlens=None, **kw):
            return _reference_varlen_sdpa(q, k, v, cu_seqlens, self.scale)

    import vllm_torchtpu.models.vllm.gemma4_vision_patch as g4_patch

    monkeypatch.setattr(g4_patch, "MMEncoderAttention", FakeMM)
    monkeypatch.setattr(
        modeling_gemma4.Gemma4VisionAttention,
        "__init__",
        modeling_gemma4.Gemma4VisionAttention.__init__,
    )
    monkeypatch.setattr(
        modeling_gemma4.Gemma4VisionAttention,
        "forward",
        modeling_gemma4.Gemma4VisionAttention.forward,
    )
    monkeypatch.setattr(
        modeling_gemma4.Gemma4VisionAttention,
        "_tpu_mm_encoder_attn_patched",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        modeling_gemma4.Gemma4VisionEncoder,
        "forward",
        modeling_gemma4.Gemma4VisionEncoder.forward,
    )
    install_gemma4_vision_patches(modeling_gemma4)

    patched_encoder = modeling_gemma4.Gemma4VisionEncoder(cfg).eval()
    patched_encoder.load_state_dict(ref_encoder.state_dict(), strict=True)

    with torch.no_grad():
        patched_out = patched_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_position_ids=pixel_pos,
        ).last_hidden_state

    for b, vlen in enumerate(valid_lens):
        torch.testing.assert_close(
            patched_out[b, :vlen], ref_out[b, :vlen], atol=1e-5, rtol=1e-5
        )


@pytest.mark.tpu_test
def test_tpu_gemma4_vision_encoder_fullgraph_parity(monkeypatch):
    modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    flywheel = ensure_flywheel_enum_registered()
    if TpuMMEncoderAttention._flywheel_head_major_op is None:
        from vllm_torchtpu.layers.adapter.vision_attention import _build_flywheel_ops

        tm_op, hm_op = _build_flywheel_ops()
        TpuMMEncoderAttention._flywheel_token_major_op = tm_op
        TpuMMEncoderAttention._flywheel_head_major_op = hm_op

    cfg = Gemma4VisionConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(11)
    ref_encoder = modeling_gemma4.Gemma4VisionEncoder(cfg).eval()
    with torch.no_grad():
        for layer in ref_encoder.layers:
            layer.self_attn.q_norm.weight.fill_(64**-0.25)
            layer.self_attn.k_norm.weight.fill_(64**-0.25)

    bsz, seq_len = 2, 128
    valid_lens = [96, 128]
    embeds_cpu = torch.randn(bsz, seq_len, 256, dtype=torch.bfloat16).float()
    mask_cpu = torch.zeros(bsz, seq_len, dtype=torch.bool)
    pos_cpu = torch.full((bsz, seq_len, 2), -1, dtype=torch.long)
    for b, vlen in enumerate(valid_lens):
        mask_cpu[b, :vlen] = True
        pos_cpu[b, :vlen, 0] = torch.arange(vlen) % 16
        pos_cpu[b, :vlen, 1] = torch.arange(vlen) // 16

    with torch.no_grad():
        ref_out = ref_encoder(
            inputs_embeds=embeds_cpu,
            attention_mask=mask_cpu,
            pixel_position_ids=pos_cpu,
        ).last_hidden_state

    import vllm_torchtpu.models.vllm.gemma4_vision_patch as g4_patch

    if TpuMMEncoderAttention._vision_op is None:
        from torch_tpu._internal import pallas

        from vllm_torchtpu.layers.adapter.vision_attention import _vision_flash_core

        op = pallas.jax_op("pallas::vision_flash_attention", _vision_flash_core)
        op.register_fake(lambda q, k, v, q_seg, kv_seg: torch.empty_like(q))
        TpuMMEncoderAttention._vision_op = op

    active_backend = flywheel

    def make_tpu_mm(num_heads, head_size, scale=None, num_kv_heads=None, **kw):
        return make_attention(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0 if scale is None else scale,
            num_kv_heads=num_kv_heads,
            attn_backend=active_backend,
        )

    monkeypatch.setattr(g4_patch, "MMEncoderAttention", make_tpu_mm)
    monkeypatch.setattr(
        modeling_gemma4.Gemma4VisionAttention,
        "__init__",
        modeling_gemma4.Gemma4VisionAttention.__init__,
    )
    monkeypatch.setattr(
        modeling_gemma4.Gemma4VisionAttention,
        "forward",
        modeling_gemma4.Gemma4VisionAttention.forward,
    )
    monkeypatch.setattr(
        modeling_gemma4.Gemma4VisionAttention,
        "_tpu_mm_encoder_attn_patched",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        modeling_gemma4.Gemma4VisionEncoder,
        "forward",
        modeling_gemma4.Gemma4VisionEncoder.forward,
    )
    install_gemma4_vision_patches(modeling_gemma4)

    device = torch.device("tpu:0")
    tpu_encoder = modeling_gemma4.Gemma4VisionEncoder(cfg).eval()
    tpu_encoder.load_state_dict(ref_encoder.state_dict(), strict=True)
    tpu_encoder = tpu_encoder.to(device=device, dtype=torch.bfloat16)

    active_backend = AttentionBackendEnum.FLASH_ATTN
    flash_encoder = modeling_gemma4.Gemma4VisionEncoder(cfg).eval()
    flash_encoder.load_state_dict(ref_encoder.state_dict(), strict=True)
    flash_encoder = flash_encoder.to(device=device, dtype=torch.bfloat16)

    def run_fw_encoder(embeds, mask, pos):
        return tpu_encoder(
            inputs_embeds=embeds,
            attention_mask=mask,
            pixel_position_ids=pos,
        ).last_hidden_state

    def run_flash_encoder(embeds, mask, pos):
        return flash_encoder(
            inputs_embeds=embeds,
            attention_mask=mask,
            pixel_position_ids=pos,
        ).last_hidden_state

    compiled_fw = torch.compile(
        run_fw_encoder, backend="tpu", fullgraph=True, dynamic=False
    )
    compiled_flash = torch.compile(
        run_flash_encoder, backend="tpu", fullgraph=True, dynamic=False
    )
    embeds_tpu = embeds_cpu.to(device=device, dtype=torch.bfloat16)
    mask_tpu = mask_cpu.to(device=device)
    pos_tpu = pos_cpu.to(device=device)
    with torch.no_grad():
        out_fw = compiled_fw(embeds_tpu, mask_tpu, pos_tpu).cpu()
        out_flash = compiled_flash(embeds_tpu, mask_tpu, pos_tpu).cpu()

    for b, vlen in enumerate(valid_lens):
        cos_sim = F.cosine_similarity(
            out_fw[b, :vlen].float(), ref_out[b, :vlen], dim=-1
        )
        assert float(cos_sim.min()) > 0.998
        torch.testing.assert_close(
            out_fw[b, :vlen].float(),
            ref_out[b, :vlen],
            atol=1e-1,
            rtol=1e-1,
        )
        fw_flash_cos = F.cosine_similarity(
            out_fw[b, :vlen].float(), out_flash[b, :vlen].float(), dim=-1
        )
        assert float(fw_flash_cos.min()) > 0.999
        torch.testing.assert_close(
            out_fw[b, :vlen],
            out_flash[b, :vlen],
            atol=1e-1,
            rtol=1e-1,
        )
