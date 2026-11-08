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
"""ERNIE 5.5 building blocks on TPU: VHA head mix, HCA attention, SiTU MLP, latent
multi-router MoE, MultiMax.

Math follows github.com/baidu/vLLM-ERNIE @ 34cceda1 (Apache-2.0)
``models/ernie5_5/nvidia/{attention,moe,model}.py``; the HCA attention reuses the
DeepSeek-V4 TPU attention (SWA + compressed-KV Pallas kernels, fused inverse-RoPE
``wo_a``) and only adds ERNIE's low-rank output gate and VHA around it.
"""

import dataclasses
from functools import partial

import torch
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import SituAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    config as compressor_config,
)
from vllm_torchtpu.models.vllm.deepseek_v4.attention import VllmDeepseekV4MLAAttention

VHA_RANK = 16  # vLLM-ERNIE nvidia/attention.py:85


class ErnieVHA(nn.Module):
    """Virtual-head post-mix ``o[h] <- o[h] + sum_r V[h, r] sum_g U[g, r] o[g]``
    (m = I + V U^T).

    Under attention TP each rank holds its heads' rows of U / V; the rank-16 projection
    ``U^T o`` is summed over every head, so it is all-reduced
    (fp32 ``[T, 16, head_dim]``) before ``V`` maps it back to the local heads.
    """

    def __init__(self, num_heads: int, rank: int = VHA_RANK) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        n_local = num_heads // self.tp_size
        self.U = nn.Parameter(torch.zeros(n_local, rank), requires_grad=False)
        self.V = nn.Parameter(torch.zeros(n_local, rank), requires_grad=False)
        for p in (self.U, self.V):
            set_weight_attrs(p, {"weight_loader": self._load_local_heads})

    def _load_local_heads(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        rows = param.shape[0]
        param.data.copy_(loaded[self.tp_rank * rows : (self.tp_rank + 1) * rows])

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        # o: [num_tokens, num_local_heads, head_dim]
        z = torch.einsum("hr,thd->trd", self.U.float(), o.float())
        if self.tp_size > 1:
            z = tensor_model_parallel_all_reduce(z)
        return (o.float() + torch.einsum("hr,trd->thd", self.V.float(), z)).to(o.dtype)


class ErnieHCAAttention(VllmDeepseekV4MLAAttention):
    """DeepSeek-V4 HCA attention (compress ratio 128 + 128-token window, sink) with
    ERNIE's output gate and VHA.

    ``wo_b(wo_a(inv_rope(VHA(o))) * sigmoid(o_gate(qr)))``: VHA mixes heads and the
    inverse RoPE rotates head dims, so mixing before the fused inverse-RoPE ``wo_a``
    kernel is exact (vLLM-ERNIE attention.py:227-243).
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None:
        super().__init__(vllm_config, prefix=prefix, **kwargs)
        # Column-split by group, like wo_a, so each rank gates its own groups' wo_a
        # output.
        self.o_gate = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=vllm_config.quant_config,
            return_bias=False,
            prefix=f"{prefix}.o_gate",
        )
        self.vha = ErnieVHA(self.n_heads)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None:
            return spec
        # ERNIE has no CSA layer to host this layer's compressor state, so the DSv4 KV
        # allocator gives every HCA state its own CSA-NoPE-shaped array
        # (kv_cache_dsv4.py). Charging that page here makes the engine size num_blocks
        # for what the TPU allocates (the SWA caches share those arrays across groups at
        # no extra cost).
        mode = compressor_config.Mode
        host_rows = compressor_config.state_host_page_size(
            mode.HCA, spec.block_size, self.compress_ratio
        )
        host_bytes = host_rows * compressor_config.row_size_bytes(mode.CSA)
        return dataclasses.replace(
            spec, page_size_padded=spec.page_size_bytes + host_bytes
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if positions.ndim > 1:
            positions = positions.flatten()
        qr_kv, indexer_weights = self.attn_gemm(hidden_states)
        qr, kv = qr_kv.split(
            [self.mla_attn.q_lora_rank, self.mla_attn.head_dim], dim=-1
        )
        qr = self.q_norm(qr)
        kv = self.kv_norm(kv)
        o = self.attention_impl(hidden_states, qr, kv, indexer_weights, positions, None)
        o = self.vha(o)
        z = self.o_proj_op(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a.weight,
            self.wo_a.weight_scale,
        )
        out = self.wo_b(z * torch.sigmoid(self.o_gate(qr)))
        return out[0] if isinstance(out, tuple) else out


class ErnieMLP(nn.Module):
    """SiTU-GLU MLP (dense layer 0 and the shared expert)."""

    def __init__(
        self,
        config,
        intermediate_size: int,
        quant_config,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SituAndMul(
            beta=config.activation_situ_beta,
            linear_beta=config.activation_situ_linear_beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gate_up_proj(x)
        out, _ = self.down_proj(self.act_fn(h))
        return out


def ernie_routing(hidden_states, gating_output, topk, renormalize, *, bias, scale):
    """vLLM-ERNIE moe.py:66-87: top-k of the summed split-feature sigmoid scores +
    correction bias; weights = renormalised selected scores x the learnable
    per-expert scale (indexed by logical expert id)."""
    ids = torch.topk(gating_output + bias, topk, dim=-1, sorted=False).indices
    w = gating_output.gather(1, ids)
    if renormalize:
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
    return (w * scale[ids]).float(), ids.int()


class ErnieRoutedOutputTransform(nn.Module):
    """Latent RMSNorm + fc2 after the expert combine (non-linear: the runner all-reduces
    the latent sum first)."""

    def __init__(self, norm: RMSNorm, up_proj: ReplicatedLinear) -> None:
        super().__init__()
        self.norm, self.up_proj = norm, up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.up_proj(self.norm(x))
        return out


class ErnieMoE(nn.Module):
    """Multi-router latent MoE: two sigmoid routers summed (one fp32 [2E, H] GEMM),
    experts SiTU-GLU in the 2048-d latent between fc1 and latent RMSNorm + fc2,
    shared expert on the hidden state (vLLM-ERNIE moe.py:99-223)."""

    def __init__(self, config, quant_config, prefix: str = ""):
        super().__init__()
        if get_current_vllm_config().parallel_config.enable_eplb:
            raise NotImplementedError(
                "ERNIE 5.5 MoE: the per-expert scale is indexed by logical id; no EPLB."
            )
        H, L, E = config.hidden_size, config.moe_latent_size, config.n_routed_experts
        self.n_routed_experts = E
        self.ep_size = get_ep_group().world_size
        self.gate = GateLinear(
            H,
            2 * E,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.routed_scaling_factor_param = nn.Parameter(
            torch.ones(E, dtype=torch.float32), requires_grad=False
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(E, dtype=torch.float32), requires_grad=False
        )
        self.shared_experts = ErnieMLP(
            config,
            config.moe_intermediate_size * config.n_shared_experts,
            quant_config,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )
        self.fc1_latent_proj = ReplicatedLinear(
            H,
            L,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1_latent_proj",
        )
        self.latent_norm = RMSNorm(L, eps=config.rms_norm_eps)
        self.fc2_latent_proj = ReplicatedLinear(
            L,
            H,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2_latent_proj",
        )
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=E,
            top_k=config.num_experts_per_tok,
            hidden_size=L,
            intermediate_size=config.moe_intermediate_size,
            activation=config.hidden_act,
            activation_situ_beta=config.activation_situ_beta,
            activation_situ_linear_beta=config.activation_situ_linear_beta,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=partial(
                ernie_routing,
                bias=self.e_score_correction_bias,
                scale=self.gate.routed_scaling_factor_param,
            ),
            router_logits_dtype=torch.float32,
            routed_output_transform=ErnieRoutedOutputTransform(
                self.latent_norm, self.fc2_latent_proj
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        E = self.n_routed_experts
        logits, _ = self.gate(hidden_states)
        scores = logits[:, :E].sigmoid() + logits[:, E:].sigmoid()
        latent, _ = self.fc1_latent_proj(hidden_states)
        return self.experts(
            hidden_states=latent,
            router_logits=scores,
            shared_experts_input=hidden_states,
        )


def multimax(
    logits: torch.Tensor, ranges: torch.Tensor, ts: torch.Tensor
) -> torch.Tensor:
    """vLLM-ERNIE model.py:92-100 learned piecewise logit transform, in fp32."""
    x, r, t = logits.float(), ranges.float(), ts.float()
    return (
        x
        + t[0] * (r[0] - x).relu()
        + t[1] * (x - r[1]).relu()
        + t[2] * (r[2] - x).relu().square()
        + t[3] * (x - r[3]).relu().square()
    )
