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
import pytest
import torch
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

MAX_REL_L2 = 2e-2
MAX_ABS = 5e-2


def _reference_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    freqs: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU fp32 RMSNorm + interleaved complex RoPE reference."""
    q_fp32 = q.detach().cpu().float()
    k_fp32 = k.detach().cpu().float()
    w_q = norm_q_weight.detach().cpu().float()
    w_k = norm_k_weight.detach().cpu().float()
    q_norm = q_fp32 * torch.rsqrt(q_fp32.pow(2).mean(dim=-1, keepdim=True) + eps) * w_q
    k_norm = k_fp32 * torch.rsqrt(k_fp32.pow(2).mean(dim=-1, keepdim=True) + eps) * w_k
    freqs_cpu = freqs.detach().cpu().unsqueeze(0).unsqueeze(2)
    q_rot = torch.view_as_real(
        torch.view_as_complex(q_norm.reshape(*q_norm.shape[:-1], -1, 2)) * freqs_cpu
    )
    k_rot = torch.view_as_real(
        torch.view_as_complex(k_norm.reshape(*k_norm.shape[:-1], -1, 2)) * freqs_cpu
    )
    return q_rot.flatten(-2), k_rot.flatten(-2)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "_fused_cuda_supported requires current_platform.is_cuda(), "
        "so _qwen_image_qk_norm_rope stays on eager path on TPU"
    ),
)
def test_fused_qk_norm_rope_token_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fused QK-Norm RoPE Token Gate

    - TPU path: set min_tokens=0 and run _qwen_image_qk_norm_rope on TPU bf16 Q/K.
    - Setup check: assert min_tokens == 0 and spy records 1 fused call (0 at 1024).
    - Outcome check: compare TPU bf16 Q/K against CPU fp32 via assert_parity.
    """
    fused_calls: list[int] = []
    orig_fused = qwen_image_transformer.fused_qk_norm_rope

    def _spy_fused(
        *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_calls.append(1)
        return orig_fused(*args, **kwargs)

    monkeypatch.setattr(qwen_image_transformer, "fused_qk_norm_rope", _spy_fused)

    torch.manual_seed(0)
    batch, seq_len, heads, head_dim = 2, 32, 4, 128
    eps = 1e-6
    q_cpu = torch.randn((batch, seq_len, heads, head_dim), dtype=torch.bfloat16)
    k_cpu = torch.randn((batch, seq_len, heads, head_dim), dtype=torch.bfloat16)
    angles = torch.randn((seq_len, head_dim // 2), dtype=torch.float32)
    freqs_cpu = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)

    norm_q_cpu = RMSNorm(head_dim, eps=eps)
    norm_k_cpu = RMSNorm(head_dim, eps=eps)
    ref_q, ref_k = _reference_qk_norm_rope(
        q_cpu, k_cpu, norm_q_cpu.weight, norm_k_cpu.weight, freqs_cpu, eps
    )

    tpu = torch.device("tpu")
    norm_q = norm_q_cpu.to(device=tpu, dtype=torch.bfloat16)
    norm_k = norm_k_cpu.to(device=tpu, dtype=torch.bfloat16)
    rope = RotaryEmbedding(is_neox_style=False).to(tpu)
    q_tpu = q_cpu.to(tpu)
    k_tpu = k_cpu.to(tpu)
    freqs_tpu = freqs_cpu.to(tpu)

    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "1024")
    qwen_image_transformer._qwen_image_qk_norm_rope(
        q_tpu, k_tpu, norm_q, norm_k, freqs_tpu, rope, eps
    )
    assert len(fused_calls) == 0, (
        f"expected 0 fused calls when min_tokens=1024, got {len(fused_calls)}"
    )

    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
    min_tokens = qwen_image_transformer.fused_qk_norm_rope_min_tokens(2048)
    assert min_tokens == 0, (
        f"expected fused_qk_norm_rope_min_tokens(2048) == 0, got {min_tokens}"
    )

    out_q, out_k = qwen_image_transformer._qwen_image_qk_norm_rope(
        q_tpu, k_tpu, norm_q, norm_k, freqs_tpu, rope, eps
    )
    assert len(fused_calls) == 1, (
        f"expected 1 fused call when min_tokens=0, got {len(fused_calls)}"
    )
    assert_parity(
        out_q,
        ref_q,
        name="fused_qk_norm_rope_token_gate_q",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
    assert_parity(
        out_k,
        ref_k,
        name="fused_qk_norm_rope_token_gate_k",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
