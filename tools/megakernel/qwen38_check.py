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
"""One Qwen3.8 decoder layer on the host (torch, FP32), from the checkpoint's
tensors, for ``qwen38 probe --check``: a few fresh tokens through the
layer's mixer (:func:`mixer`, no cached state) or its MoE (:func:`sparse`);
:func:`logits` for the LM head. NumPy rows in and out.

Independent of the stage's loader and layouts: every tensor is read by its
checkpoint name; the FP8 projections run as static W8A8 (``reference.py``'s
rounding), the routed experts from their NVFP4 bytes dequantized at the
checkpoint's own block-16 scales (the model's native weights).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


class Checkpoint:
    def __init__(self, path: Path) -> None:
        from safetensors import safe_open

        self.path = Path(path)
        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.files = index["weight_map"]
        self._open = {}
        self._safe_open = safe_open

    def raw(self, name: str) -> torch.Tensor:
        file = self.files[name]
        if file not in self._open:
            self._open[file] = self._safe_open(self.path / file, framework="pt")
        return self._open[file].get_tensor(name)

    def get(self, name: str) -> torch.Tensor:
        """FP32 (FP8 and BF16 widened; integer tensors as stored)."""
        tensor = self.raw(name)
        return tensor if tensor.dtype == torch.uint8 else tensor.float()

    def scalar(self, name: str) -> float:
        return float(self.get(name).reshape(-1)[0])


def rms(x, w, eps, plus_one=True):
    value = x * torch.rsqrt(torch.mean(x * x, -1, keepdim=True) + eps)
    return bf16(value * ((1 + w) if plus_one else w))


def fp8_linear(ckpt, name, x):
    s_in, s_w = ckpt.scalar(name + ".input_scale"), ckpt.scalar(name + ".weight_scale")
    q = torch.clamp(x / s_in, -448, 448).to(torch.float8_e4m3fn).float()
    return bf16((q @ ckpt.get(name + ".weight").T) * (s_in * s_w))


def nvfp4(ckpt, name):
    """``[out, in]`` FP32 from the NVFP4 bytes and block-16 scales."""
    packed = ckpt.get(name + ".weight")  # [out, in / 2] uint8, low nibble first
    codes = torch.stack((packed & 0xF, packed >> 4), -1).reshape(packed.shape[0], -1)
    magnitude = E2M1[(codes & 7).long()]
    value = torch.where((codes & 8) > 0, -magnitude, magnitude)
    scale = ckpt.get(name + ".weight_scale") * ckpt.scalar(name + ".weight_scale_2")
    return value * scale.repeat_interleave(16, dim=1)


def rope(x, positions, rotary, theta):
    inv = theta ** (-2 * torch.arange(rotary // 2, dtype=torch.float64) / rotary)
    angles = positions[:, None].double() * inv
    cos = torch.cat((angles.cos(), angles.cos()), -1)[:, None].float()
    sin = torch.cat((angles.sin(), angles.sin()), -1)[:, None].float()
    rot, rest = x[..., :rotary], x[..., rotary:]
    a, b = rot.chunk(2, -1)
    return bf16(torch.cat((rot * cos + torch.cat((-b, a), -1) * sin, rest), -1))


def gdn(ckpt, p, x, d):
    t = x.shape[0]
    qkv = fp8_linear(ckpt, p + "linear_attn.in_proj_qkv", x)
    z = fp8_linear(ckpt, p + "linear_attn.in_proj_z", x)
    b = fp8_linear(ckpt, p + "linear_attn.in_proj_b", x)
    a = fp8_linear(ckpt, p + "linear_attn.in_proj_a", x)
    conv = ckpt.get(p + "linear_attn.conv1d.weight")[:, 0, :]  # [C, K]
    history = torch.cat((torch.zeros(d.conv - 1, qkv.shape[1]), qkv))
    convolved = bf16(sum(history[i : i + t] * conv[:, i] for i in range(d.conv)))
    convolved = torch.nn.functional.silu(convolved)
    q, k, v = convolved.split((d.key_dim, d.key_dim, d.value_dim), -1)
    q = q.reshape(t, d.k_heads, d.head_k)
    k = k.reshape(t, d.k_heads, d.head_k)
    q = q * torch.rsqrt(torch.sum(q * q, -1, keepdim=True) + 1e-6) * d.head_k**-0.5
    k = k * torch.rsqrt(torch.sum(k * k, -1, keepdim=True) + 1e-6)
    group = d.v_heads // d.k_heads
    q, k = q.repeat_interleave(group, 1), k.repeat_interleave(group, 1)
    v = v.reshape(t, d.v_heads, d.head_v)
    beta = torch.sigmoid(b)
    a_log = ckpt.get(p + "linear_attn.A_log")
    dt_bias = ckpt.get(p + "linear_attn.dt_bias")
    decay = torch.exp(-torch.exp(a_log) * torch.nn.functional.softplus(a + dt_bias))
    state = torch.zeros(d.v_heads, d.head_v, d.head_k)
    out = torch.zeros(t, d.v_heads, d.head_v)
    for i in range(t):
        state = state * decay[i][:, None, None]
        prediction = torch.einsum("hk,hvk->hv", k[i], state)
        update = (beta[i][:, None] * (v[i] - prediction))[..., None] * k[i][:, None, :]
        state = state + update
        out[i] = torch.einsum("hk,hvk->hv", q[i], state)
    norm = ckpt.get(p + "linear_attn.norm.weight")
    gated = rms(bf16(out), norm, d.eps, plus_one=False)
    gate = z.reshape(t, d.v_heads, d.head_v)
    gated = bf16(gated * torch.nn.functional.silu(gate))
    return fp8_linear(ckpt, p + "linear_attn.out_proj", gated.reshape(t, -1))


def attention(ckpt, p, x, positions, d):
    t = x.shape[0]
    qg = fp8_linear(ckpt, p + "self_attn.q_proj", x)
    q, gate = qg.reshape(t, d.q_heads, 2 * d.head_dim).chunk(2, -1)
    k = fp8_linear(ckpt, p + "self_attn.k_proj", x).reshape(t, d.kv_heads, d.head_dim)
    v = fp8_linear(ckpt, p + "self_attn.v_proj", x).reshape(t, d.kv_heads, d.head_dim)
    q_norm = ckpt.get(p + "self_attn.q_norm.weight")
    k_norm = ckpt.get(p + "self_attn.k_norm.weight")
    q = rope(rms(q, q_norm, d.eps), positions, d.rotary_dim, d.rope_theta)
    k = rope(rms(k, k_norm, d.eps), positions, d.rotary_dim, d.rope_theta)
    k = k.to(torch.float8_e4m3fn).float()  # the FP8 KV cache, scale 1
    v = v.to(torch.float8_e4m3fn).float()
    group = d.q_heads // d.kv_heads
    k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
    scores = torch.einsum("thd,shd->hts", q, k) * d.head_dim**-0.5
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool))
    probs = torch.softmax(scores.masked_fill(~causal, float("-inf")), -1)
    out = bf16(torch.einsum("hts,shd->thd", probs, v))
    out = bf16(out * torch.sigmoid(gate))
    return fp8_linear(ckpt, p + "self_attn.o_proj", out.reshape(t, -1))


def moe(ckpt, p, x, d):
    logits = bf16(x @ ckpt.get(p + "mlp.gate.weight").T)
    probs = torch.softmax(logits, -1)
    weights, top = torch.topk(probs, d.top_k, -1)
    weights = weights / weights.sum(-1, keepdim=True)
    routed = torch.zeros_like(x)
    for e in torch.unique(top).tolist():
        rows = torch.nonzero((top == e).any(-1)).reshape(-1)
        base = f"{p}mlp.experts.{e}."
        gate = x[rows] @ nvfp4(ckpt, base + "gate_proj").T
        up = x[rows] @ nvfp4(ckpt, base + "up_proj").T
        hidden = bf16(torch.nn.functional.silu(gate) * up)
        out = bf16(hidden @ nvfp4(ckpt, base + "down_proj").T)
        w = torch.where(top[rows] == e, weights[rows], 0).sum(-1, keepdim=True)
        routed[rows] += out * w
    shared_in = torch.cat(
        (
            ckpt.get(p + "mlp.shared_expert.gate_proj.weight"),
            ckpt.get(p + "mlp.shared_expert.up_proj.weight"),
        )
    )
    g, u = bf16(x @ shared_in.T).chunk(2, -1)
    hidden = bf16(torch.nn.functional.silu(g) * u)
    shared = bf16(hidden @ ckpt.get(p + "mlp.shared_expert.down_proj.weight").T)
    scale = torch.sigmoid(bf16(x @ ckpt.get(p + "mlp.shared_expert_gate.weight").T))
    return routed + shared * scale


def _rows(h) -> torch.Tensor:
    return torch.as_tensor(np.asarray(h, np.float32))


def mixer(ckpt, index, h, positions, d, full) -> np.ndarray:
    """The residual rows after layer ``index``'s mixer."""
    p, h = f"model.layers.{index}.", _rows(h)
    x = rms(h, ckpt.get(p + "input_layernorm.weight"), d.eps)
    positions = torch.as_tensor(np.asarray(positions))
    mixed = attention(ckpt, p, x, positions, d) if full else gdn(ckpt, p, x, d)
    return bf16(h + mixed).numpy()


def sparse(ckpt, index, h, d) -> np.ndarray:
    """The residual rows after layer ``index``'s MoE, from rows after its mixer."""
    p, h = f"model.layers.{index}.", _rows(h)
    x = rms(h, ckpt.get(p + "post_attention_layernorm.weight"), d.eps)
    return bf16(h + moe(ckpt, p, x, d)).numpy()


def logits(ckpt, h, d) -> np.ndarray:
    """The LM head's logits for final residual rows ``h``."""
    x = rms(_rows(h), ckpt.get("model.norm.weight"), d.eps)
    return (x @ ckpt.get("lm_head.weight").T).numpy()
