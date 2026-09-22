# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp W8A16 experts, FP32 routing/combine and BF16 shared expert.

Router/shared work stays local. EP dispatches physical DP slots to complete
owned experts. Without EP, every rank evaluates a TP slice of all experts.
Routed contributions are summed in FP32; the shared expert is added once.
"""
import functools
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import torch
from torch import nn
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2
from vllm_torchtpu.layers.core.fused_moe_gmm import prepare_routed_gmm_inputs
from vllm_torchtpu.models.vllm.qwen4_exp.fp8 import (ExpertGeometry,
                                                     pack_expert_projection)


def _bf16(x):
    return jax.lax.optimization_barrier(x.astype(jnp.bfloat16))


def _linear(x, weight):
    return _bf16(
        jnp.dot(x,
                weight.T,
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32))


def route_tokens(x, router, valid_tokens):
    """BF16 logits, FP32 softmax/top10/renorm; equal scores use smaller id."""
    logits = _linear(x, router).astype(jnp.float32)
    scores = jax.nn.softmax(logits, axis=-1)
    # lax.top_k defines lower-index-first ties. Selected probabilities stay
    # FP32 through dispatch, expert weighting and the inter-rank reduction.
    probs, ids = jax.lax.top_k(scores, 10)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    probs = jnp.where(valid_tokens[:, None], probs, 0)
    return probs, ids, logits


def shared_expert(x, gate, up, down, shared_gate):
    gate_out, up_out = _linear(x, gate), _linear(x, up)
    act = _bf16(jax.nn.silu(gate_out.astype(jnp.float32)))
    hidden = _bf16(act * up_out)
    output = _linear(hidden, down)
    probability = _bf16(
        jax.nn.sigmoid(_linear(x, shared_gate).astype(jnp.float32)))
    return _bf16(output * probability)


def routed_experts(x, w13, s13, w2, s2, probs, ids, experts_start):
    """Local contribution, FP32 [T,H]; no communication or activation quant."""
    local_ids = ids - experts_start
    local_ids = jnp.where((local_ids >= 0) & (local_ids < w13.shape[0]),
                          local_ids, -1)
    dispatched, groups, inverse, flat_probs, valid, _ = prepare_routed_gmm_inputs(
        x,
        local_ids,
        probs,
        local_num_experts=w13.shape[0],
        topk=10,
        use_ep=True,
        use_sparse_core=False,
        skip_padded_tokens=True)
    hidden = gmm_v2(dispatched,
                    w13,
                    groups,
                    rhs_scale=s13,
                    maybe_quantize_lhs=False,
                    acc_dtype=jnp.float32,
                    preferred_element_type=jnp.bfloat16,
                    fuse_act="silu")
    expert_output = gmm_v2(hidden,
                           w2,
                           groups,
                           rhs_scale=s2,
                           maybe_quantize_lhs=False,
                           acc_dtype=jnp.float32,
                           preferred_element_type=jnp.bfloat16)
    # Uncomputed routes may contain NaNs. Select before weighting: 0*NaN
    # would otherwise contaminate padded tokens and idle expert owners.
    selected = jnp.where(valid[:, None], expert_output[inverse],
                         0).astype(jnp.float32)
    return (selected * flat_probs[:, None]).reshape(x.shape[0], 10,
                                                    -1).sum(axis=1)


@functools.cache
def build_moe_op():

    def fn(x: jax.Array, router: jax.Array, w13: jax.Array, s13: jax.Array,
           w2: jax.Array, s2: jax.Array, gate: jax.Array, up: jax.Array,
           down: jax.Array, shared_gate: jax.Array, experts_start: jax.Array,
           valid_tokens: jax.Array) -> tuple[jax.Array, jax.Array]:
        probs, ids, _ = route_tokens(x, router, valid_tokens)
        routed = routed_experts(x, w13, s13, w2, s2, probs, ids, experts_start)
        shared = shared_expert(x, gate, up, down, shared_gate)
        return routed, jnp.where(valid_tokens[:, None], shared, 0)

    op = pallas.jax_op("qwen4_exp::moe_w8a16_fp32_routes", fn)

    @op.register_fake
    def fake(x, *args):
        return torch.empty_like(x, dtype=torch.float32), torch.empty_like(x)

    return op


class _SharedExpert(nn.Module):

    def __init__(self, device):
        super().__init__()
        for name, n, k in (("gate_proj", 640, 2560), ("up_proj", 640, 2560),
                           ("down_proj", 2560, 640)):
            setattr(
                self, name,
                nn.Linear(k,
                          n,
                          bias=False,
                          dtype=torch.bfloat16,
                          device=device))


@functools.cache
def build_local_routing_op():

    def fn(
        x: jax.Array, router: jax.Array, gate: jax.Array, up: jax.Array,
        down: jax.Array, shared_gate: jax.Array, valid_tokens: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # GDN need not initialize padding output. Zero it before routing and
        # remote expert computation, without changing any valid row's math.
        x = jnp.where(valid_tokens[:, None], x, 0)
        probs, ids, _ = route_tokens(x, router, valid_tokens)
        shared = shared_expert(x, gate, up, down, shared_gate)
        return x, probs, ids, jnp.where(valid_tokens[:, None], shared, 0)

    op = pallas.jax_op("qwen4_exp::moe_local_routes_shared", fn)

    @op.register_fake
    def fake(x, *args):
        return (torch.empty_like(x),
                x.new_empty((x.shape[0], 10), dtype=torch.float32),
                x.new_empty((x.shape[0], 10),
                            dtype=torch.int32), torch.empty_like(x))

    return op


@functools.cache
def build_routed_experts_op():

    def fn(x: jax.Array, w13: jax.Array, s13: jax.Array, w2: jax.Array,
           s2: jax.Array, probs: jax.Array, ids: jax.Array,
           experts_start: jax.Array) -> jax.Array:
        return routed_experts(x, w13, s13, w2, s2, probs, ids, experts_start)

    op = pallas.jax_op("qwen4_exp::moe_owned_w8a16_experts", fn)

    @op.register_fake
    def fake(x, *args):
        return torch.empty_like(x, dtype=torch.float32)

    return op


class Qwen4ExpSparseMoeBlock(nn.Module):
    """EP owns complete experts; EP off owns a TP slice of every expert.

    ``load_weights`` takes names relative to ``mlp.`` in arbitrary order.
    ``finalize_weights`` verifies every owned tensor was supplied and builds
    the custom op outside Dynamo. No default FP8 post-load hook is invoked.
    """

    def __init__(self,
                 *,
                 ep_rank: int,
                 ep_size: int = 8,
                 tp_rank: int = 0,
                 tp_size: int = 1,
                 enable_ep: bool = True,
                 device=None):
        super().__init__()
        if enable_ep and (ep_size != 8 or not 0 <= ep_rank < ep_size):
            raise ValueError(
                "Qwen4Exp S1 MoE requires EP8, contiguous experts")
        self.ep_rank = ep_rank
        self.enable_ep = enable_ep
        self.geometry = (ExpertGeometry() if enable_ep else ExpertGeometry(
            tp_size, tp_rank))
        self.expert_ids = (range(ep_rank * 64, (ep_rank + 1) *
                                 64) if enable_ep else range(512))
        count, width = len(self.expert_ids), self.geometry.physical_size
        self.gate = nn.Linear(2560,
                              512,
                              bias=False,
                              dtype=torch.bfloat16,
                              device=device)
        self.shared_expert = _SharedExpert(device)
        self.shared_expert_gate = nn.Linear(2560,
                                            1,
                                            bias=False,
                                            dtype=torch.bfloat16,
                                            device=device)
        self.register_buffer("experts_start",
                             torch.tensor(self.expert_ids.start,
                                          dtype=torch.int32,
                                          device=device),
                             persistent=False)
        # Stream into one CPU bank per local rank. Device view copy_ can
        # rematerialize an entire TorchTPU parameter for every expert.
        self._packed = {}
        for name, shape, dtype in (("w13", (count, 2560, 2 * width),
                                    torch.float8_e4m3fn),
                                   ("s13", (count, 20, 1, 2 * width),
                                    torch.float32), ("w2", (count, width,
                                                            2560),
                                                     torch.float8_e4m3fn),
                                   ("s2", (count, width // 128, 1, 2560),
                                    torch.float32)):
            self._packed[name] = torch.empty(shape, dtype=dtype, device="cpu")
            self.register_parameter(
                name,
                nn.Parameter(torch.empty(0, dtype=dtype, device=device),
                             requires_grad=False))
        self._pending = {}
        self._loaded = set()
        self._local_op = None
        self._experts_op = None

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loaded = set()
        dense = dict(self.named_parameters())
        for name, tensor in weights:
            if not name.startswith("experts."):
                if name not in dense or name in ("w13", "s13", "w2", "s2"):
                    raise ValueError(
                        f"Unknown Qwen4Exp MoE checkpoint tensor: {name}")
                if tensor.shape != dense[
                        name].shape or tensor.dtype != torch.bfloat16:
                    raise ValueError(f"Expected BF16 checkpoint tensor {name}")
                with torch.no_grad():
                    dense[name].copy_(tensor)
                self._loaded.add(name)
                loaded.add(name)
                continue
            _, expert, projection, field = name.split(".")
            expert = int(expert)
            if not 0 <= expert < 512:
                raise ValueError(f"Invalid expert id: {expert}")
            if expert not in self.expert_ids:
                continue
            if projection not in ("gate_proj", "up_proj",
                                  "down_proj") or field not in (
                                      "weight", "weight_scale_inv"):
                raise ValueError(f"Unknown expert checkpoint tensor: {name}")
            key = (expert, projection)
            pair = self._pending.setdefault(key, {})
            pair[field] = tensor.detach().cpu()
            loaded.add(name)
            if len(pair) == 2:
                packed, scale = pack_expert_projection(
                    pair["weight"].unsqueeze(0),
                    pair["weight_scale_inv"].unsqueeze(0),
                    projection.removesuffix("_proj"), self.geometry)
                local = expert - self.expert_ids.start
                with torch.no_grad():
                    if projection == "down_proj":
                        self._packed["w2"][local].copy_(packed[0])
                        self._packed["s2"][local].copy_(scale[0])
                    else:
                        width = self.geometry.physical_size
                        start = 0 if projection == "gate_proj" else width
                        self._packed["w13"][local, :, start:start +
                                            width].copy_(packed[0])
                        self._packed["s13"][local, :, :, start:start +
                                            width].copy_(scale[0])
                self._loaded.add(key)
                del self._pending[key]
        return loaded

    def finalize_weights(self):
        expected = {(e, p)
                    for e in self.expert_ids
                    for p in ("gate_proj", "up_proj", "down_proj")}
        expected.update(
            ("gate.weight", "shared_expert_gate.weight",
             "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
             "shared_expert.down_proj.weight"))
        missing = expected - self._loaded
        if missing or self._pending:
            raise ValueError(
                f"Incomplete Qwen4Exp MoE checkpoint: {len(missing)} tensors")
        for name, value in self._packed.items():
            setattr(
                self, name,
                nn.Parameter(value.to(self.gate.weight.device),
                             requires_grad=False))
        self._packed.clear()
        self._local_op = build_local_routing_op()
        self._experts_op = build_routed_experts_op()

    def forward(self, hidden_states, valid_tokens):
        from vllm.distributed import (get_ep_group,
                                      tensor_model_parallel_all_reduce)

        x, probs, ids, shared = self._local_op(
            hidden_states, self.gate.weight,
            self.shared_expert.gate_proj.weight,
            self.shared_expert.up_proj.weight,
            self.shared_expert.down_proj.weight,
            self.shared_expert_gate.weight, valid_tokens)
        if self.enable_ep:
            ep = get_ep_group()
            x, probs, ids = ep.dispatch(x, probs, ids)
        routed = self._experts_op(x, self.w13, self.s13, self.w2, self.s2,
                                  probs, ids, self.experts_start)
        if self.enable_ep:
            routed = ep.combine(routed)
        routed = tensor_model_parallel_all_reduce(routed).to(torch.bfloat16)
        return routed + shared
