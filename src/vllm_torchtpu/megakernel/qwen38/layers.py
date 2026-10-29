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
"""The stage's layers as global-view JAX functions over the stage's cores.

The residual stream is sequence parallel: core ``c`` holds the step's flat
rows ``rows.RowPlan.core_rows(c)`` (``[rows, H]`` per core, ``P(PCP)``).
Each layer moves only activations between the cores:

* GDN, one kernel per core (``gdn``): the core projects its own rows (FP8,
  replicated weights, one fused input projection whose output columns are
  grouped by destination core) and pushes each block to the cores of its
  value-head shards (16 value and 2 key heads) as it is computed, so every
  core gets all rows of its shard in flat order (``exchange``); the conv +
  delta rule runs over each request's tokens there as they arrive
  (``gdn_pcp_fused``'s tiles, states from vLLM's unified pool); each finished
  stage's output goes back to the rows' owners, which apply the gate
  (projected from their own rows), the output norm and the output
  projection.
* Full attention: the core projects its rows. In a step without cached
  prefixes every core gathers the step's keys and values (FP8) and attends
  its queries over them (``attention``); otherwise the PCP streaming
  attention kernel passes every KV page around the stage. Either way each
  new KV row is written on the core owning its page.
* MoE: the router runs on the owner's rows, the shared expert while the rows
  are all-gathered (``prologue``); every core runs its 64 experts, and
  copies of the previous core's where that evens the cores' routes, over the
  routes of all rows (``moe``), and the weighted sums go back to the rows'
  owners.

The kernels are vllm-torchtpu's own (``kernels.experimental.gdn_pcp_fused``,
``kernels.experimental.pcp_streaming_rpa``), ``gdn``, ``exchange``,
``attention``, ``prologue``, ``experts``, ``moe``, ``ring`` and
``projection`` (the output projections with the residual add); the other
projections, norms and RoPE are XLA. No vLLM dependencies.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.qwen38 import attention, gdn
from vllm_torchtpu.megakernel.qwen38.moe import routed_experts
from vllm_torchtpu.megakernel.qwen38.projection import fp8_projection
from vllm_torchtpu.megakernel.qwen38.reference import (
    BF16,
    F32,
    FP8,
    FP8_MAX,
    Dims,
    rms,
    rope,
)

PCP = "pcp"  # the stage's core axis (the stock PCP kernels' axis name)

Weights = Mapping[str, jax.Array]


@dataclasses.dataclass(frozen=True)
class Geometry:
    """A compiled step's static shape."""

    dims: Dims
    cores: int
    ring: tuple[int, ...]  # the cores in ring order (ring.ring_order)
    chunk: int  # flat chunk = KV interleave
    rows: int  # rows per core
    max_requests: int
    pool_block_tokens: int  # a GDN state block's manager tokens (x PCP)
    q_block: int  # the attention kernel's query tile
    seq_along_lane: bool  # the pool's pages are [heads, dim, tokens] (vLLM's HND)
    k_scale: float = 1.0  # the FP8 KV cache's scales (compiled in)
    v_scale: float = 1.0

    @property
    def flat_rows(self) -> int:
        return self.cores * self.rows

    @property
    def local_k_heads(self) -> int:
        return self.dims.k_heads // self.cores

    @property
    def local_v_heads(self) -> int:
        return self.dims.v_heads // self.cores


def _fp8_dot(x, weight, input_scale, weight_scale):
    """Static W8A8 on the MXU: FP8 x FP8 -> FP32, scaled, BF16 out (weights
    ``[out, in]``, as the checkpoint stores them)."""
    return _quantized_dot(_quantize(x, input_scale), weight, input_scale, weight_scale)


def _quantized_dot(x, weight, input_scale, weight_scale):
    """:func:`_fp8_dot` of an input already quantized by ``input_scale``."""
    value = jax.lax.dot_general(
        x, weight, (((1,), (1,)), ((), ())), preferred_element_type=F32
    )
    return (value * (input_scale * weight_scale)).astype(BF16)


def _quantize(x, input_scale):
    return jnp.clip(x.astype(F32) / input_scale, -FP8_MAX, FP8_MAX).astype(FP8)


def _bf16_dot(x, weight):
    return jax.lax.dot_general(
        x.astype(BF16), weight, (((1,), (1,)), ((), ())), preferred_element_type=F32
    )


def _gdn_dims(g: Geometry) -> dict:
    d = g.dims
    return dict(
        n_kq=g.local_k_heads,
        n_v=g.local_v_heads,
        d_k=d.head_k,
        d_v=d.head_v,
        kernel_size=d.conv,
    )


def _gdn_tiles(g: Geometry, meta: Mapping[str, jax.Array], state_indices):
    return gdn.tile_metadata(
        g.flat_rows,
        meta["seq_lens"],
        meta["query_start_loc"],
        state_indices,
        meta["distribution"],
        **_gdn_dims(g),
    )


def gdn_plan(g: Geometry, meta: Mapping[str, jax.Array], state):
    """The step's GDN tiles (``gdn.tile_metadata``) as flat replicated arrays:
    every GDN layer whose states ``state`` names shares them."""
    return tuple(jax.tree.leaves(_gdn_tiles(g, meta, state)))


def gdn_mixer(
    g: Geometry,
    mesh: Mesh,
    h,
    w: Weights,
    pool,
    meta: Mapping[str, jax.Array],
    plan=(),
):
    """``h + GDN(rms(h))`` and the pool with the requests' new states.

    ``w``: replicated FP8 projections (``in_proj`` ``[cores * width, H]``,
    ``weights.fuse_gdn_in``'s layout, ``z`` and ``out``) with their scales
    and norms; this core's shard of the conv weight ``conv`` ``[local conv_dim,
    1, kernel]`` (rows ``[q_c | k_c | v_c]``), ``a_log``, ``dt_bias``,
    ``norm``. ``plan``: the step's :func:`gdn_plan`, or empty to compute it
    here.
    """
    d = g.dims
    lk, lv = g.local_k_heads, g.local_v_heads
    tiles = jax.tree.structure(
        jax.eval_shape(
            lambda meta, state: _gdn_tiles(g, meta, state),
            dict(meta),
            meta["state_indices"],
        )
    )

    def local(
        h,
        pool,
        conv,
        a_log,
        dt_bias,
        norm,
        qsl,
        state_indices,
        seq_lens,
        distribution,
        *leaves,
    ):
        x = rms(h, w["input_norm"], d.eps)
        input_scale = w["in_proj_input_scale"]
        normed, pool = gdn.mixer(
            _quantize(x, input_scale),
            w["in_proj"],
            (input_scale * w["in_proj_weight_scale"])[None],
            w["z"],
            jnp.broadcast_to(input_scale * w["z_weight_scale"], (1, d.value_dim)),
            pool,
            conv,
            a_log,
            dt_bias,
            norm,
            w["out_input_scale"],
            qsl,
            state_indices,
            seq_lens,
            distribution,
            axis=PCP,
            cores=g.cores,
            chunk=g.chunk,
            n_kq=lk,
            n_v=lv,
            d_k=d.head_k,
            d_v=d.head_v,
            kernel_size=d.conv,
            pool_block_tokens=g.pool_block_tokens,
            eps=d.eps,
            meta=jax.tree.unflatten(tiles, leaves) if leaves else None,
        )
        scale = w["out_input_scale"] * w["out_weight_scale"]
        return fp8_projection(normed, w["out"], scale, h), pool

    return jax.shard_map(
        local,
        mesh=mesh,
        # The norm weight is every head's; the tiles' metadata every core's.
        in_specs=(P(PCP),) * 5 + (P(),) * 5 + (P(),) * len(plan),
        out_specs=(P(PCP), P(PCP)),
        check_vma=False,
    )(
        h,
        pool,
        w["conv"],
        w["a_log"],
        w["dt_bias"],
        w["norm"],
        meta["query_start_loc"],
        meta["state_indices"],
        meta["seq_lens"],
        meta["distribution"],
        *plan,
    )


def _kv_layout(g: Geometry):
    from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout

    return KVLayout.SEQ_ALONG_LANE if g.seq_along_lane else KVLayout.HEAD_ALONG_SUBLANE


def attention_plan(g: Geometry, mesh: Mesh, cache_shape, meta: Mapping[str, jax.Array]):
    """The step's :func:`attention.attention_plan` (every attention layer's;
    ``cache_shape`` a core's pool shape)."""
    return attention.attention_plan(
        mesh,
        PCP,
        meta["seq_lens"],
        meta["block_tables"],
        meta["query_start_loc"],
        meta["distribution"],
        meta["positions"],
        rows=g.rows,
        interleave=g.chunk,
        kv_layout=_kv_layout(g),
        cache_shape=cache_shape,
        rotary_dim=g.dims.rotary_dim,
        theta=g.dims.rope_theta,
    )


def attention_mixer(
    g: Geometry, mesh: Mesh, h, w: Weights, pool, meta: Mapping[str, jax.Array], plan
):
    """``h + Attention(rms(h))`` and the pool with the new tokens' KV
    (``plan``: the step's :func:`attention_plan`)."""
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import (
        sharded_pcp_ragged_paged_attention,
    )

    d = g.dims

    def project(h, positions):
        """Each head's query and gate (``[rows, Hq, 2 D]``, the query not yet
        normalized or rotated), and the keys and values."""
        x = rms(h, w["input_norm"], d.eps)
        query_gate = _fp8_dot(x, w["q"], w["q_input_scale"], w["q_weight_scale"])
        query_gate = query_gate.reshape(g.rows, d.q_heads, 2 * d.head_dim)
        key = _fp8_dot(x, w["k"], w["k_input_scale"], w["k_weight_scale"])
        value = _fp8_dot(x, w["v"], w["v_input_scale"], w["v_weight_scale"])
        key = key.reshape(g.rows, d.kv_heads, d.head_dim)
        value = value.reshape(g.rows, d.kv_heads, d.head_dim)
        key = rope(rms(key, w["k_norm"], d.eps), positions, d.rotary_dim, d.rope_theta)
        return query_gate, key, value

    query_gate, key, value = jax.shard_map(
        project,
        mesh=mesh,
        in_specs=(P(PCP), P(PCP)),
        out_specs=(P(PCP), P(PCP), P(PCP)),
        check_vma=False,
    )(h, meta["positions"])
    kv_layout = _kv_layout(g)
    metadata = (
        meta["seq_lens"],
        meta["block_tables"],
        meta["query_start_loc"],
        meta["distribution"],
    )

    def streaming():
        def normed(query_gate, positions):
            query = query_gate[..., : d.head_dim]
            query = rms(query, w["q_norm"], d.eps)
            return rope(query, positions, d.rotary_dim, d.rope_theta)

        query = jax.shard_map(
            normed,
            mesh=mesh,
            in_specs=(P(PCP), P(PCP)),
            out_specs=P(PCP),
            check_vma=False,
        )(query_gate, meta["positions"])
        return sharded_pcp_ragged_paged_attention(
            mesh=mesh,
            q=query,
            k=key,
            v=value,
            kv_cache=pool,
            kv_lens=metadata[0],
            page_indices=metadata[1],
            cu_q_lens=metadata[2],
            distribution=metadata[3],
            attention_sink=None,
            sm_scale=d.head_dim**-0.5,
            k_scale=g.k_scale,
            v_scale=g.v_scale,
            cp_kv_cache_interleave_size=g.chunk,
            q_block_size=g.q_block,
            kv_layout=kv_layout,
        )

    def gathered():
        return attention.stage_attention(
            mesh,
            g.ring,
            PCP,
            query_gate,
            key,
            value,
            pool,
            plan,
            sm_scale=d.head_dim**-0.5,
            kv_layout=kv_layout,
            q_norm=w["q_norm"],
            rotary_dim=d.rotary_dim,
            eps=d.eps,
        )

    # Without a cached prefix every core gathers the step's keys and values
    # once instead of streaming them around the ring for every query tile.
    if g.k_scale == g.v_scale == 1.0:
        attended, pool = jax.lax.cond(plan[-1][0] > 0, streaming, gathered)
    else:
        attended, pool = streaming()

    def finish(h, attended, query_gate):
        gate = query_gate[..., d.head_dim :]
        out = attended.astype(BF16) * jax.nn.sigmoid(gate.astype(F32)).astype(BF16)
        input_scale = w["o_input_scale"]
        return fp8_projection(
            _quantize(out.reshape(g.rows, d.q_heads * d.head_dim), input_scale),
            w["o"],
            input_scale * w["o_weight_scale"],
            h,
        )

    h = jax.shard_map(
        finish,
        mesh=mesh,
        in_specs=(P(PCP), P(PCP), P(PCP)),
        out_specs=P(PCP),
        check_vma=False,
    )(h, attended, query_gate)
    return h, pool


def moe_block(g: Geometry, mesh: Mesh, h, w: Weights):
    """``h + MoE(rms(h))``: routed experts (``moe.routed_experts``;
    ``moe.held_experts`` per core, as ``experts.expert_products`` takes them)
    and the gated shared expert."""
    d = g.dims
    names = ("w13", "w13_channel", "w13_ratio", "w2", "w2_channel", "w2_ratio")

    def local(h, *expert_weights):
        x = rms(h, w["post_norm"], d.eps)
        logits = _bf16_dot(x, w["router"])
        scale = jax.nn.sigmoid(_bf16_dot(x, w["shared_gate"]))
        shared_weights = (w["shared_gate_up"], w["shared_down"])
        routed, shared = routed_experts(
            d, g.ring, PCP, x, logits, expert_weights, shared_weights
        )
        partial = h.astype(F32) + shared.astype(F32) * scale
        return (partial + routed.astype(F32)).astype(BF16)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(PCP),) * (1 + len(names)),
        out_specs=P(PCP),
        check_vma=False,
    )(h, *(w[name] for name in names))
