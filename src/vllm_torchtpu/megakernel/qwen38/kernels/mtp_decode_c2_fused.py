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
"""Experimental fused multi-request native BF16 draft, with donated KV leaves."""

from dataclasses import replace

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import PartitionSpec as P

from .attention_append import exchange_scratch
from .attention_batched_append import attend_paged_batched_append
from .collectives import hierarchical_sum, scattered_sum
from .decode_megakernel import KV_BUFFERS
from .mtp_c2_layout import cache_contract
from .mtp_config import weight_shapes
from .mtp_dense import native_experts, native_project

BF16 = jnp.bfloat16
F32 = jnp.float32


def _round_bf16_f32(value):
    """Round finite FP32 to BF16 ties-to-even while retaining an FP32 result.

    A BF16-result reduction can narrow each product before summation on TPU.
    Inspecting the FP32 sum's bits keeps that reduction in FP32 and makes the
    rounding boundary observable without Pallas's unsupported reduce_precision.
    The low retained bit resolves exact ties. Preserve signed zero, overflow to
    infinity, and the original NaN/Inf bits (including low-payload NaNs).
    """
    if value.dtype != F32:
        raise TypeError("Native MTP projection rounding requires an FP32 sum")
    bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
    retained_odd = (bits >> jnp.uint32(16)) & jnp.uint32(1)
    rounded = (bits + jnp.uint32(0x7FFF) + retained_odd) & jnp.uint32(0xFFFF0000)
    special = (bits & jnp.uint32(0x7F800000)) == jnp.uint32(0x7F800000)
    return jax.lax.bitcast_convert_type(jnp.where(special, bits, rounded), F32)


def _fused_stack(
    config, layout, context, target, weights, state, tokens, positions, hidden, interpret,
):
    requests = len(state)
    c, tp, rows = config.target, layout.tp, max(8, requests)
    # Wide lane reductions run per eight-row tile: Mosaic associates a 16-row one
    # differently.
    tiles = tuple(slice(start, start + 8) for start in range(0, rows, 8))
    h, hd, qh = c.dim, c.head_dim, layout.local_q_heads
    fc_width, router_width = h // tp, c.experts // layout.router_tp
    fc_packet = max(128, (fc_width + 127) // 128 * 128)
    width = max(h, 2 * qh * hd, layout.expert_width, c.experts, layout.vocab_shard_padded, 128)
    # Only the attention helper reads this layout; draft experts are native BF16.
    native_layout = replace(
        layout,
        precision="native",
        expert_output_tile=0,
        expert_reduction="exact",
        expert_overlap=0,
    )
    names = tuple(weight_shapes(config, layout))
    compact = h % 1024 == 0
    packet = requests * h // rows if compact else h
    local_size, hosts = min(tp, 8), max(1, tp // 8)
    # Multi-request packets reduce-scatter over the host-local group, keeping
    # hierarchical_sum's FP32 sum order.
    scattered = compact and local_size == 8 and packet % (local_size * 128) == 0
    channels = 2 * (local_size - 1) + hosts - 1 if scattered else local_size + hosts - 2
    if scattered:
        reduction = {
            "stage": tpu.VMEM((rows, packet), F32),
            "scatter": tpu.VMEM((2, local_size, rows, packet // local_size), F32),
            "host": tpu.VMEM((2, hosts, rows, packet // local_size), F32),
            "gathered": tpu.VMEM((2, rows, packet), BF16),
        }
    else:
        reduction = {
            "local": tpu.VMEM((2, local_size, rows, packet), F32),
            "remote": tpu.VMEM((2, hosts, rows, packet), F32),
        }
    inbox = exchange_scratch(native_layout, requests)
    scratch = {
        "residual": tpu.VMEM((rows, h), BF16),
        "normed": tpu.VMEM((rows, h), BF16),
        "joined": tpu.VMEM((rows, 2 * h), BF16),
        "hidden": tpu.VMEM((requests, h), BF16),
        "row_tile": tpu.VMEM((8, h), BF16),
        "norm": tpu.VMEM((h,), BF16),
        "shared_gate_weight": tpu.VMEM((1, h), BF16),
        "qn": tpu.VMEM((hd,), BF16),
        "kn": tpu.VMEM((hd,), BF16),
        "proj": tpu.VMEM((rows, width), F32),
        "qgate": tpu.VMEM((rows, 2 * qh * hd), BF16),
        "key": tpu.VMEM((rows, hd), BF16),
        "value": tpu.VMEM((rows, hd), BF16),
        "attended": tpu.VMEM((rows, qh * hd), BF16),
        "gate": tpu.VMEM((rows, max(128, layout.expert_width, layout.shared_width)), BF16),
        "middle": tpu.VMEM((rows, max(128, layout.expert_width, layout.shared_width)), BF16),
        "moe": tpu.VMEM((rows, h), F32),
        "fc_parts": tpu.VMEM((tp, rows, fc_packet), BF16),
        "router_parts": tpu.VMEM((layout.router_tp, rows, router_width), BF16),
        "router": tpu.VMEM((rows, c.experts), F32),
        "ids": tpu.VMEM((rows, c.top_k), jnp.int32),
        "probabilities": tpu.VMEM((rows, c.top_k), F32),
        "candidates": tpu.VMEM((tp, rows, 128), F32),
        **reduction,
        "sends": tpu.SemaphoreType.DMA((max(1, channels),)),
        "recvs": tpu.SemaphoreType.DMA((max(1, channels),)),
        "small_sends": tpu.SemaphoreType.DMA((tp - 1,)),
        "small_recvs": tpu.SemaphoreType.DMA((tp - 1,)),
        "copy_sem": tpu.SemaphoreType.DMA,
        **inbox,
    }

    def kernel(position_ref, token_ref, embedding_ref, lm_ref, *args):
        w = dict(zip(names, args[: len(names)], strict=True))
        # Input state refs are aliased to outputs below; no historical KV copy.
        leaves = 2 * requests
        hidden_ref = args[len(names) + leaves]
        logits_out, token_out = args[len(names) + leaves + 1 : len(names) + leaves + 3]
        caches = args[len(names) + leaves + 3 : len(names) + 2 * leaves + 3]
        hidden_out = args[len(names) + 2 * leaves + 3]
        r = dict(zip(scratch, args[len(names) + 2 * leaves + 4 :], strict=True))
        rank = jax.lax.axis_index("tp")
        barrier = tpu.get_barrier_semaphore()
        for peer in range(1, tp):
            pl.semaphore_signal(
                barrier, 1, device_id=(rank ^ peer,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, tp - 1)

        def copy(source, destination):
            dma = tpu.make_async_copy(source, destination, r["copy_sem"])
            dma.start()
            dma.wait()

        def exchange(buffer, count):
            for peer in range(1, count):
                tpu.make_async_remote_copy(
                    buffer.at[0],
                    buffer.at[peer],
                    r["small_sends"].at[peer - 1],
                    r["small_recvs"].at[peer - 1],
                    device_id=(rank ^ peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
            for peer in range(1, count):
                tpu.make_async_remote_copy(
                    buffer.at[0],
                    buffer.at[peer],
                    r["small_sends"].at[peer - 1],
                    r["small_recvs"].at[peer - 1],
                    device_id=(rank ^ peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait()

        def reduce(iteration):
            if scattered:
                parts = ("stage", "scatter", "host", "gathered", "sends", "recvs")
                return scattered_sum(r["proj"].at[:, :h], iteration, rank,
                                     {part: r[part] for part in parts},
                                     id_type=pl.DeviceIdType.MESH)
            return hierarchical_sum(
                r["proj"].at[:, :h],
                iteration,
                rank,
                r["local"],
                r["remote"],
                r["sends"],
                r["recvs"],
                compact=compact,
                id_type=pl.DeviceIdType.MESH,
            )

        def norm(value, weight):
            fp32 = value.astype(F32)
            fp32 *= jax.lax.rsqrt(jnp.mean(fp32 * fp32, axis=-1, keepdims=True) + c.eps)
            return (fp32 * (1 + weight.astype(F32))).astype(BF16)

        def normalize(family):
            copy(w[family], r["norm"])
            for tile in tiles:
                r["normed"][tile] = norm(r["residual"][tile], r["norm"][...])

        def rope(value, position):
            half = c.rotary_dim // 2
            frequency = c.rope_theta ** (
                -jnp.arange(half, dtype=jnp.int32).astype(F32) * 2 / c.rotary_dim
            )
            angles = position.astype(F32) * frequency
            cosine, sine = jnp.cos(angles).astype(BF16), jnp.sin(angles).astype(BF16)
            first, second = value[..., :half], value[..., half : c.rotary_dim]
            return jnp.concatenate(
                (
                    first * cosine - second * sine,
                    second * cosine + first * sine,
                    value[..., c.rotary_dim :],
                ),
                axis=-1,
            ).astype(BF16)

        start = rank * layout.vocab_shard
        r["proj"][...] = jnp.zeros(r["proj"].shape, F32)
        for request in range(requests):
            token = token_ref[request]
            owned = ((position_ref[request] >= 0) & (token >= start)
                     & (token < jnp.minimum(start + layout.vocab_shard, c.vocab)))
            @pl.when(owned)
            def embedding(request=request, token=token):
                local = token - start
                base = pl.multiple_of(local // 8 * 8, 8)
                copy(embedding_ref.at[pl.ds(base, 8), :], r["row_tile"])
                row = jnp.sum(jnp.where(jnp.arange(8)[:, None] == local % 8,
                                       r["row_tile"][...].astype(F32), 0), axis=0)
                r["proj"][request, :h] = row
        r["residual"][...] = reduce(0).astype(BF16)
        r["joined"][...] = jnp.zeros(r["joined"].shape, BF16)
        copy(w["embedding_norm"], r["norm"])
        live_tiles = tuple(
            slice(start, min(start + 8, requests)) for start in range(0, requests, 8)
        )
        for live in live_tiles:
            r["joined"][live, :h] = norm(r["residual"][live], r["norm"][...])
        copy(hidden_ref, r["hidden"])
        copy(w["hidden_norm"], r["norm"])
        for live in live_tiles:
            r["joined"][live, h:] = norm(r["hidden"][live], r["norm"][...])
        native_project(r["joined"], w["fc"], r["proj"])
        r["fc_parts"][0, ...] = r["proj"][:, :fc_packet].astype(BF16)
        exchange(r["fc_parts"], tp)
        for absolute in range(tp):
            r["residual"][:, pl.ds(absolute * fc_width, fc_width)] = r["fc_parts"][
                rank ^ absolute, :, :fc_width
            ]
        normalize("norm_in")
        native_project(r["normed"], w["qw"], r["qgate"])
        native_project(r["normed"], w["kw"], r["key"])
        native_project(r["normed"], w["vw"], r["value"])
        copy(w["qn"], r["qn"])
        copy(w["kn"], r["kn"])
        r["attended"][...] = jnp.zeros(r["attended"].shape, BF16)
        # One attention scope for every request precedes rank-dependent expert
        # scopes. Each request owns separate aliased cache leaves and its own
        # position; an inactive request's -1 neither attends nor appends.
        queries, keys, gates = [], [], []
        for request in range(requests):
            packed = r["qgate"][request, ...].reshape(qh, 2 * hd)
            queries.append(rope(norm(packed[:, :hd], r["qn"][...]), position_ref[request]))
            keys.append(rope(norm(r["key"][request, :], r["kn"][...]), position_ref[request]))
            gates.append(jax.nn.sigmoid(packed[:, hd:].astype(F32)).astype(BF16))
        attended = attend_paged_batched_append(
            jnp.stack(queries), jnp.stack(keys), r["value"][:requests, :],
            tuple(caches[2 * request].at[0, 0] for request in range(requests)),
            tuple(caches[2 * request + 1].at[0, 0] for request in range(requests)),
            position_ref, rank, native_layout, (1.0, 1.0), pl.DeviceIdType.MESH,
            {name: r[name] for name in inbox}, buffers=KV_BUFFERS)
        for request in range(requests):
            @pl.when(position_ref[request] >= 0)
            def gated(request=request):
                r["attended"][request, :] = (
                    attended[request] * gates[request]
                ).reshape(-1).astype(BF16)
        native_project(r["attended"], w["ow"], r["proj"], rounded=False)
        # Native MTP reference rounds the reduction before its BF16 residual add.
        partial = reduce(1).astype(BF16)
        r["residual"][...] = (r["residual"][...] + partial).astype(BF16)
        normalize("norm_post")
        native_project(r["normed"], w["router"], r["proj"])
        r["router_parts"][0, ...] = r["proj"][:, :router_width].astype(BF16)
        exchange(r["router_parts"], layout.router_tp)
        for peer in range(layout.router_tp):
            column = ((rank % layout.router_tp) ^ peer) * router_width
            r["router"][:, pl.ds(column, router_width)] = r["router_parts"][peer, ...].astype(F32)
        for tile in tiles:
            r["router"][tile] = jax.nn.softmax(r["router"][tile], axis=-1)
        indices = jnp.arange(c.experts)[None, :]
        for slot in range(c.top_k):
            best = jnp.max(r["router"][...], axis=-1, keepdims=True)
            expert = jnp.min(jnp.where(r["router"][...] == best, indices, c.experts), axis=-1)
            r["ids"][:, slot] = expert
            r["probabilities"][:, slot] = best[:, 0]
            r["router"][...] = jnp.where(indices == expert[:, None], -jnp.inf, r["router"][...])
        probabilities = r["probabilities"][...]
        r["probabilities"][...] = (
            (probabilities / jnp.maximum(jnp.sum(probabilities, axis=1, keepdims=True), 1e-20))
            .astype(BF16)
            .astype(F32)
        )
        # Keep the checkpoint's [1,H] shape during HBM DMA: its row is a
        # tiled dimension and cannot be squeezed when H spans multiple tiles.
        copy(w["shared_gate"], r["shared_gate_weight"])
        multiplier = _round_bf16_f32(
            jnp.concatenate([
                jnp.sum(
                    r["normed"][tile].astype(F32) * r["shared_gate_weight"][...].astype(F32),
                    axis=1,
                )
                for tile in tiles
            ])
        )
        multiplier = jax.nn.sigmoid(multiplier).astype(BF16).astype(F32)
        native_project(r["normed"], w["sg"], r["gate"])
        native_project(r["normed"], w["su"], r["proj"])
        sw = layout.shared_width
        r["middle"][...] = jnp.zeros(r["middle"].shape, BF16)
        r["middle"][:, :sw] = (
            jax.nn.silu(r["gate"][:, :sw].astype(F32)).astype(BF16) * r["proj"][:, :sw].astype(BF16)
        ).astype(BF16)
        native_project(r["middle"], w["sd"], r["moe"], rounded=False)
        r["moe"][...] *= multiplier[:, None]
        first = rank // layout.expert_tp * layout.local_experts
        valid = (
            (jnp.arange(rows)[:, None] < requests)
            & (r["ids"][...] >= first)
            & (r["ids"][...] < first + layout.local_experts)
        )
        r["ids"][...] = jnp.where(valid, r["ids"][...], c.experts)
        for request in range(requests):
            r["ids"][request, :] = jnp.where(position_ref[request] >= 0,
                                             r["ids"][request, :], c.experts)

        native_experts(
            r["normed"], r["ids"], r["probabilities"], (w["eg"], w["eu"], w["ed"]), first,
            c.experts, r["gate"], r["proj"], r["middle"], r["proj"], r["moe"],
        )
        r["proj"][:, :h] = r["moe"][...]
        partial = reduce(2).astype(BF16)
        r["residual"][...] = (r["residual"][...] + partial).astype(BF16)
        normalize("norm")
        for request in range(requests):
            r["normed"][request, :] = jnp.where(position_ref[request] >= 0,
                                               r["normed"][request, :], r["hidden"][request, :])
        copy(r["normed"], hidden_out)
        native_project(r["normed"], lm_ref, r["proj"], rounded=False)
        vocab_ids = start + jnp.arange(layout.vocab_shard_padded)
        valid = (jnp.arange(layout.vocab_shard_padded) < layout.vocab_shard) & (vocab_ids < c.vocab)
        logits = jnp.where(valid[None], r["proj"][:, : layout.vocab_shard_padded], -jnp.inf)
        logits_out[...] = jnp.stack(tuple(jnp.where(
            position_ref[request] >= 0, logits[request], jnp.where(valid, 0., -jnp.inf))
            for request in range(requests)))
        maximum = jnp.max(logits, axis=-1)
        best_id = jnp.min(jnp.where(logits == maximum[:, None], vocab_ids[None], c.vocab), axis=-1)
        r["candidates"][0, ...] = jnp.zeros((rows, 128), F32)
        r["candidates"][0, :, 0] = maximum
        r["candidates"][0, :, 1] = best_id.astype(F32)
        exchange(r["candidates"], tp)
        largest = jnp.max(r["candidates"][:, :, 0], axis=0)
        result = jnp.min(
            jnp.where(r["candidates"][:, :, 0] == largest[None], r["candidates"][:, :, 1], c.vocab),
            axis=0,
        )
        token_out[...] = jnp.stack(tuple(jnp.where(
            position_ref[request] >= 0, result[request].astype(jnp.int32), token_ref[request])
            for request in range(requests)))

    arguments = (
        positions, tokens, target["emb"], target["lmw"],
        *(weights[name] for name in names),
        *(request[name] for request in state for name in ("kcache", "vcache")),
        hidden,
    )
    state_offset = 4 + len(names)

    leaves = 2 * requests

    def owner_kernel(*refs):
        refs = list(refs)
        for index in (*range(state_offset, state_offset + leaves),
                      *range(len(arguments) + 2, len(arguments) + 2 + leaves)):
            refs[index] = refs[index].at[0].bitcast(BF16)
        kernel(*refs)

    result = pl.pallas_call(
        owner_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((requests, layout.vocab_shard_padded), F32),
            jax.ShapeDtypeStruct((requests,), jnp.int32),
            *(jax.ShapeDtypeStruct(request[name].shape, request[name].dtype)
              for request in state for name in ("kcache", "vcache")),
            jax.ShapeDtypeStruct((rows, h), BF16),
        ),
        input_output_aliases={state_offset + i: 2 + i for i in range(leaves)},
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * (len(arguments) - 2),
            out_specs=[pl.BlockSpec(), pl.BlockSpec(),
                       *[pl.BlockSpec(memory_space=tpu.HBM)] * (leaves + 1)],
            scratch_shapes=tuple(scratch.values()),
        ),
        compiler_params=tpu.CompilerParams(
            collective_id=89, vmem_limit_bytes=64 << 20, shape_invariant_numerics=True,
            allow_input_fusion=(False,) * len(arguments)),
        interpret=interpret, name="qwen_native_bf16_mtp_draft_c2_fused",
    )(*arguments)
    caches = tuple(dict(zip(("kcache", "vcache"), result[2+2*i:4+2*i], strict=True))
                   for i in range(requests))
    return result[1], result[0], caches, result[2 + leaves][:requests]


def make_fused_draft(mesh, config, layout, context, *, interpret=False):
    """Three calls per K3 round, each computing every independent request."""
    def local(target, weights, state, tokens, positions, hidden):
        target, weights = (jax.tree.map(lambda x: x[0], tree) for tree in (target, weights))
        following, logits, updated, carry = _fused_stack(
            config, layout, context, target, weights, cache_contract(state),
            tokens, positions, hidden, interpret)
        return following, logits[None], cache_contract(updated), carry

    def step(target, weights, state, tokens, positions, hidden, active):
        expected = (layout.tp, 1, 1, layout.local_kv_pages(context), 64, config.target.head_dim)
        requests = len(state) if isinstance(state, tuple) else 0
        if (requests not in (2, 4, 8, 16) or
                any(set(request) != {"kcache", "vcache"} or
                    any(x.shape != expected or x.dtype != jnp.uint32 for x in request.values())
                    for request in state)):
            raise ValueError("Two, four, eight or sixteen independent packed BF16 cache owners "
                             "required")
        if (tokens.shape != (requests,) or positions.shape != (requests,)
                or active.shape != (requests,)
                or hidden.shape != (requests, config.target.dim) or hidden.dtype != BF16
                or tokens.dtype != jnp.int32 or positions.dtype != jnp.int32
                or active.dtype != jnp.bool_):
            raise ValueError(
                "Explicit per-request token, position, hidden and activity metadata required")
        return jax.shard_map(
            local, mesh=mesh,
            in_specs=(P("tp"), P("tp"), P("tp"), P(), P(), P()),
            out_specs=(P(), P("tp"), P("tp"), P()), check_vma=False,
        )(target, weights, state, tokens, jnp.where(active, positions, -1), hidden)
    return jax.jit(step)
