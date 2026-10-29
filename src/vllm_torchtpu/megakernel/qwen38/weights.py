# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""A stage core's weights as device tensors, in the layers' layouts.

Per layer (``layers`` docstrings): the FP8 projections ``[out, in]`` (the
checkpoint's layout) with their scalar scales, replicated on every core of
the stage, the GDN input projections sent to the heads' cores fused into one
(:func:`fuse_gdn_in`); this core's head shard of the GDN conv weight,
``a_log``, ``dt_bias`` and norm; the routed experts it holds
(``moe.held_experts``: its 64 and copies of the previous core's) as
``experts.expert_products`` takes them (``w13`` ``[groups, H, 2 I]`` and
``w2`` ``[groups, I, H]`` E2M1 with per-channel scales ``[groups, 1, N]`` and
per-block ratios ``[groups, K / 64, N]``); the BF16 router, shared expert and
norms (``[out, in]``). Stage 0 adds the embedding, the last stage the final
norm and this core's vocabulary shard of the LM head.

Every tensor is created once in its final dtype on the device, FP8 as FP8:
no step converts a weight.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from typing import Any

import torch

from vllm_torchtpu.megakernel.qwen38.experts import nvfp4_expert_weights
from vllm_torchtpu.megakernel.qwen38.moe import COPIES, held_experts
from vllm_torchtpu.megakernel.qwen38.program import is_full_attention
from vllm_torchtpu.megakernel.qwen38.reference import Dims


@dataclasses.dataclass
class StageWeights:
    layers: dict[int, dict[str, torch.Tensor]]
    embedding: torch.Tensor | None  # [vocab, H] BF16 (stage 0)
    final_norm: torch.Tensor | None  # [H] (last stage)
    lm_head: torch.Tensor | None  # [vocab / cores, H] BF16 (last stage)


GDN_FP8 = ("qkv", "z", "b", "a", "out")
GDN_IN = ("qkv", "z", "b", "a")  # one FP8 quantization of the input serves all four
ATTENTION_FP8 = ("q", "k", "v", "o")


def gdn_in_rows(d: Dims, cores: int) -> int:
    """A core's ``[q | k | v | b | a]`` rows of the GDN input projections."""
    lk, lv = d.k_heads // cores, d.v_heads // cores
    return 2 * lk * d.head_k + lv * d.head_v + 2 * lv


def gdn_in_width(d: Dims, cores: int) -> int:
    """A core's block of ``in_proj``: its rows padded to a lane multiple (the
    block is a column slice of the projection's output, sent as one DMA)."""
    return -(-gdn_in_rows(d, cores) // 128) * 128


def fuse_gdn_in(
    d: Dims, cores: int, projections: dict[str, tuple[torch.Tensor, ...]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``in_proj`` ``[cores * width, H]``, its input scale and its per-row
    weight scales from the GDN input projections (``GDN_IN`` -> ``(weight,
    input_scale, weight_scale)``, the checkpoint's): core ``c``'s block holds
    its heads' rows ``[q_c | k_c | v_c | b_c | a_c]`` then zero rows to
    :func:`gdn_in_width`, so the projection's output is the head exchange's
    operand (``exchange.project_to_heads``). The gate ``z`` is not part of
    it (the output norm runs on the rows' owner), but it shares the input's
    one FP8 quantization, so all four input scales must agree."""
    input_scales = {float(projections[name][1]) for name in GDN_IN}
    if len(input_scales) != 1:
        raise ValueError(
            f"GDN input projections with input scales {input_scales}: one "
            "quantization of the input needs one"
        )
    lk, lv = d.k_heads // cores, d.v_heads // cores
    qkv = projections["qkv"][0]
    pad = gdn_in_width(d, cores) - gdn_in_rows(d, cores)
    rows, scales = [], []
    for c in range(cores):
        k_rows = slice(c * lk * d.head_k, (c + 1) * lk * d.head_k)
        v_rows = slice(c * lv * d.head_v, (c + 1) * lv * d.head_v)
        heads = slice(c * lv, (c + 1) * lv)
        parts = (
            ("qkv", qkv[k_rows]),
            ("qkv", qkv[d.key_dim :][k_rows]),
            ("qkv", qkv[2 * d.key_dim :][v_rows]),
            ("b", projections["b"][0][heads]),
            ("a", projections["a"][0][heads]),
        )
        for name, part in parts:
            rows.append(part)
            scales.append(projections[name][2].reshape(1).expand(part.shape[0]))
        rows.append(torch.zeros(pad, d.hidden, dtype=qkv.dtype))
        scales.append(torch.zeros(pad))
    return (
        torch.cat(rows),
        projections["qkv"][1],
        torch.cat(scales).to(torch.float32),
    )


def fp8_shapes(d: Dims, full_attention: bool) -> dict[str, tuple[int, int]]:
    """Each FP8 projection's ``[out, in]`` shape (the checkpoint's)."""
    if full_attention:
        return {
            "q": (2 * d.q_heads * d.head_dim, d.hidden),
            "k": (d.kv_heads * d.head_dim, d.hidden),
            "v": (d.kv_heads * d.head_dim, d.hidden),
            "o": (d.hidden, d.q_heads * d.head_dim),
        }
    return {
        "qkv": (d.conv_dim, d.hidden),
        "z": (d.value_dim, d.hidden),
        "b": (d.v_heads, d.hidden),
        "a": (d.v_heads, d.hidden),
        "out": (d.hidden, d.value_dim),
    }


def synthetic(
    d: Dims,
    layers: range,
    cores: int,
    vocab: int,
    *,
    first_stage: bool,
    last_stage: bool,
    seed: int,
    device: str = "tpu",
) -> StageWeights:
    """Random weights of the stage's shapes and dtypes (timing and
    bring-up), drawn on the device from ``seed`` and the layer, so every
    core of the stage holds the same replicated weights (and the same
    experts); scaled so activations stay in range through the layers."""

    def normal(*shape, scale=1.0, dtype=torch.bfloat16):
        value = torch.randn(*shape, device=device) * scale
        return value.to(dtype)

    def scalar(value):
        return torch.tensor(value, dtype=torch.float32).to(device)

    def experts(out_features: int, in_features: int):
        """Random NVFP4 bytes and block-16 scales of every local expert and of
        the copies of the previous core's (every core draws the same
        experts), in the kernel's block-64 form by the loader's own conversion
        (so the result has the layout the kernel reads, not a byte view's)."""
        local = d.experts // cores
        packed = torch.randint(
            0,
            256,
            (local, out_features, in_features // 2),
            dtype=torch.uint8,
            device=device,
        )
        scale = torch.full(
            (local, out_features, in_features // 16), 0.01, device=device
        )
        copies = slice(None, None, local // COPIES)
        packed = torch.cat([packed, packed[copies]])
        scale = torch.cat([scale, scale[copies]])
        return tuple(_expert_weights()(packed, scale))

    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        torch.manual_seed(seed * 1000 + layer)
        full = is_full_attention(layer)
        w: dict[str, torch.Tensor] = {
            "input_norm": normal(d.hidden, scale=0.0),
            "post_norm": normal(d.hidden, scale=0.0),
        }
        for name, (n, k) in fp8_shapes(d, full).items():
            if name in GDN_IN:
                continue
            w[name] = normal(n, k, dtype=torch.float8_e4m3fn)
            w[f"{name}_input_scale"] = scalar(1.0 / 16)
            w[f"{name}_weight_scale"] = scalar(k**-0.5)
        if not full:
            w["z"] = normal(d.value_dim, d.hidden, dtype=torch.float8_e4m3fn)
            w["z_weight_scale"] = scalar(d.hidden**-0.5)
            # fuse_gdn_in's layout, drawn in place.
            used = gdn_in_rows(d, cores)
            width = gdn_in_width(d, cores)
            in_proj = normal(cores, width, d.hidden)
            in_proj[:, used:] = 0
            w["in_proj"] = in_proj.reshape(cores * width, d.hidden).to(
                torch.float8_e4m3fn
            )
            w["in_proj_input_scale"] = scalar(1.0 / 16)
            scale = torch.zeros(cores, width)
            scale[:, :used] = d.hidden**-0.5
            w["in_proj_weight_scale"] = scale.reshape(-1).to(device)
        if full:
            w["q_norm"] = normal(d.head_dim, scale=0.0)
            w["k_norm"] = normal(d.head_dim, scale=0.0)
        else:
            lk, lv = d.k_heads // cores, d.v_heads // cores
            conv_dim = 2 * lk * d.head_k + lv * d.head_v
            w["conv"] = normal(conv_dim, 1, d.conv, scale=0.5)
            w["a_log"] = normal(lv, scale=0.5, dtype=torch.float32)
            w["dt_bias"] = normal(lv, scale=0.5)
            w["norm"] = normal(d.head_v, scale=0.1) + 1
        w["router"] = normal(d.experts, d.hidden, scale=d.hidden**-0.5)
        w["shared_gate_up"] = normal(2 * d.shared_inter, d.hidden, scale=d.hidden**-0.5)
        w["shared_down"] = normal(d.hidden, d.shared_inter, scale=d.shared_inter**-0.5)
        w["shared_gate"] = normal(1, d.hidden, scale=d.hidden**-0.5)
        w["w13"], w["w13_channel"], w["w13_ratio"] = experts(
            2 * d.expert_inter, d.hidden
        )
        w["w2"], w["w2_channel"], w["w2_ratio"] = experts(d.hidden, d.expert_inter)
        out[layer] = w
    torch.manual_seed(seed)
    return StageWeights(
        layers=out,
        embedding=normal(vocab, d.hidden) if first_stage else None,
        final_norm=normal(d.hidden, scale=0.0) if last_stage else None,
        lm_head=normal(vocab // cores, d.hidden, scale=d.hidden**-0.5)
        if last_stage
        else None,
    )


_FP8_TENSORS = {
    "qkv": "linear_attn.in_proj_qkv",
    "z": "linear_attn.in_proj_z",
    "b": "linear_attn.in_proj_b",
    "a": "linear_attn.in_proj_a",
    "out": "linear_attn.out_proj",
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
}


READ_THREADS = 8  # concurrent tensor reads per core (the checkpoint is remote)


class _Checkpoint:
    """Tensors of a safetensors checkpoint by name. :meth:`prefetch` starts
    reading a batch on ``READ_THREADS`` threads; :meth:`get` returns a
    prefetched tensor, or reads it then."""

    def __init__(self, path: Path) -> None:
        import json
        import threading
        from concurrent.futures import ThreadPoolExecutor

        self.path = Path(path)
        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.files: dict[str, str] = index["weight_map"]
        self._open: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(READ_THREADS)
        self._pending: dict[str, Any] = {}

    def _handle(self, file: str):
        from safetensors import safe_open

        with self._lock:
            if file not in self._open:
                self._open[file] = safe_open(self.path / file, framework="pt")
            return self._open[file]

    def _read(self, name: str) -> torch.Tensor:
        # A copy: the read happens here, on the worker thread, not on first use.
        return self._handle(self.files[name]).get_tensor(name).clone()

    def prefetch(self, names) -> None:
        for name in names:
            if name not in self._pending:
                self._pending[name] = self._pool.submit(self._read, name)

    def get(self, name: str) -> torch.Tensor:
        future = self._pending.pop(name, None)
        return self._read(name) if future is None else future.result()

    def close(self) -> None:
        self._pool.shutdown(cancel_futures=True)


def load(
    checkpoint: Path,
    d: Dims,
    layers: range,
    cores: int,
    core: int,
    vocab: int,
    *,
    first_stage: bool,
    last_stage: bool,
    device: str = "tpu",
) -> StageWeights:
    """The stage's weights from the Hugging Face NVFP4 checkpoint: this
    core's reads only (its head shards and the experts it holds; the replicated
    families are the same tensors on every core of the host, so the page
    cache serves the other seven), the next layer's read while this one's
    are converted. Experts are converted on the device to
    ``experts.expert_products``' form."""
    ckpt = _Checkpoint(checkpoint)

    def put(tensor: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        return (tensor if dtype is None else tensor.to(dtype)).contiguous().to(device)

    lk, lv = d.k_heads // cores, d.v_heads // cores
    q_rows = slice(core * lk * d.head_k, (core + 1) * lk * d.head_k)
    v_rows = slice(core * lv * d.head_v, (core + 1) * lv * d.head_v)
    heads = slice(core * lv, (core + 1) * lv)
    experts = held_experts(d, cores, core)

    def names(layer: int) -> list[str]:
        """The checkpoint tensors layer ``layer`` reads on this core."""
        prefix = f"model.layers.{layer}."
        parts = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert.down_proj.weight",
            "mlp.shared_expert_gate.weight",
        ]
        for name in fp8_shapes(d, is_full_attention(layer)):
            parts += [
                f"{_FP8_TENSORS[name]}.{x}"
                for x in ("weight", "input_scale", "weight_scale")
            ]
        if is_full_attention(layer):
            parts += ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
        else:
            parts += [
                f"linear_attn.{x}"
                for x in ("conv1d.weight", "A_log", "dt_bias", "norm.weight")
            ]
        for e in experts:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                parts += [
                    f"mlp.experts.{e}.{proj}.{x}"
                    for x in ("weight", "weight_scale", "weight_scale_2")
                ]
        return [prefix + part for part in parts]

    out: dict[int, dict[str, torch.Tensor]] = {}
    ckpt.prefetch(names(layers[0]) if len(layers) else [])
    for i, layer in enumerate(layers):
        # The next layer's reads overlap this layer's device work.
        if i + 1 < len(layers):
            ckpt.prefetch(names(layers[i + 1]))
        prefix = f"model.layers.{layer}."
        full = is_full_attention(layer)
        w = {
            "input_norm": put(ckpt.get(prefix + "input_layernorm.weight")),
            "post_norm": put(ckpt.get(prefix + "post_attention_layernorm.weight")),
        }
        gdn_in = {}
        for name in fp8_shapes(d, full):
            tensor = prefix + _FP8_TENSORS[name]
            weight = ckpt.get(tensor + ".weight")
            input_scale = ckpt.get(tensor + ".input_scale").to(torch.float32)
            weight_scale = ckpt.get(tensor + ".weight_scale").to(torch.float32)
            if name in GDN_IN:
                gdn_in[name] = (weight, input_scale, weight_scale)
                continue
            w[name] = put(weight)
            w[f"{name}_input_scale"] = put(input_scale)
            w[f"{name}_weight_scale"] = put(weight_scale)
        if gdn_in:
            w["in_proj"], w["in_proj_input_scale"], w["in_proj_weight_scale"] = (
                put(t) for t in fuse_gdn_in(d, cores, gdn_in)
            )
            w["z"], w["z_weight_scale"] = put(gdn_in["z"][0]), put(gdn_in["z"][2])
        if full:
            w["q_norm"] = put(ckpt.get(prefix + "self_attn.q_norm.weight"))
            w["k_norm"] = put(ckpt.get(prefix + "self_attn.k_norm.weight"))
        else:
            conv = ckpt.get(prefix + "linear_attn.conv1d.weight")  # [conv_dim, 1, K]
            w["conv"] = put(
                torch.cat(
                    (
                        conv[q_rows],
                        conv[d.key_dim :][q_rows],
                        conv[2 * d.key_dim :][v_rows],
                    )
                )
            )
            w["a_log"] = put(
                ckpt.get(prefix + "linear_attn.A_log")[heads], torch.float32
            )
            w["dt_bias"] = put(ckpt.get(prefix + "linear_attn.dt_bias")[heads])
            w["norm"] = put(ckpt.get(prefix + "linear_attn.norm.weight"))
        w["router"] = put(ckpt.get(prefix + "mlp.gate.weight"))
        w["shared_gate_up"] = put(
            torch.cat(
                (
                    ckpt.get(prefix + "mlp.shared_expert.gate_proj.weight"),
                    ckpt.get(prefix + "mlp.shared_expert.up_proj.weight"),
                )
            )
        )
        w["shared_down"] = put(ckpt.get(prefix + "mlp.shared_expert.down_proj.weight"))
        w["shared_gate"] = put(ckpt.get(prefix + "mlp.shared_expert_gate.weight"))
        w13, s13, w2, s2 = [], [], [], []
        for e in experts:
            base = f"{prefix}mlp.experts.{e}."
            halves, scales = [], []
            for proj in ("gate_proj", "up_proj"):
                halves.append(ckpt.get(base + proj + ".weight"))
                scales.append(
                    ckpt.get(base + proj + ".weight_scale").to(torch.float32)
                    * ckpt.get(base + proj + ".weight_scale_2").to(torch.float32)
                )
            w13.append(torch.cat(halves))
            s13.append(torch.cat(scales))
            w2.append(ckpt.get(base + "down_proj.weight"))
            s2.append(
                ckpt.get(base + "down_proj.weight_scale").to(torch.float32)
                * ckpt.get(base + "down_proj.weight_scale_2").to(torch.float32)
            )
        for name, weight, scale in (("w13", w13, s13), ("w2", w2, s2)):
            w[name], w[f"{name}_channel"], w[f"{name}_ratio"] = _expert_weights()(
                put(torch.stack(weight)), put(torch.stack(scale))
            )
        out[layer] = w
    shard = slice(core * vocab // cores, (core + 1) * vocab // cores)
    weights = StageWeights(
        layers=out,
        embedding=put(ckpt.get("model.embed_tokens.weight")) if first_stage else None,
        final_norm=put(ckpt.get("model.norm.weight")) if last_stage else None,
        lm_head=put(ckpt.get("lm_head.weight")[shard]) if last_stage else None,
    )
    ckpt.close()
    return weights


@functools.cache
def _expert_weights():
    """``experts.nvfp4_expert_weights`` as a device op on torch tensors."""
    from torch_tpu._internal import pallas

    return pallas.jax_op("pallas::qwen38_nvfp4_expert_weights", nvfp4_expert_weights)
