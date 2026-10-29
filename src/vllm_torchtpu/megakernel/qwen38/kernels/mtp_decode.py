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
"""One native-BF16 MTP draft layer in one Pallas call; unqualified development path."""

from dataclasses import replace
from functools import wraps

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import PartitionSpec as P

from .attention_append import attend_paged_append
from .collectives import hierarchical_sum
from .decode_megakernel import KV_BUFFERS
from .mtp_config import weight_shapes
from .mtp_dense import native_experts, native_project
from .paged_kv import PoolPages

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


def _native_stack(
    config, layout, context, target, weights, state, tokens, positions, hidden, interpret,
    *, _state_owner_axis=False, _requests=1, _request_index=0, paged=None,
):
    """One draft call. ``paged`` (``{"pool", "tables", "pages_per_block"}``)
    reads the draft KV from vLLM's MTP layer pool (``paged_kv``) through the
    ``[1, width]`` rank-local block table instead of ``state``'s caches, the
    pool a read-only operand; the call's new key/value column (FP8 ``[1, 1, 2,
    D]``) is returned as the state's ``"columns"`` for the caller to commit."""
    if paged is not None and (_state_owner_axis or _requests != 1):
        raise ValueError("Pool pages serve one request's draft")
    c, tp, rows = config.target, layout.tp, 8
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
    packet = h // 8 if compact else h
    local_size, hosts = min(tp, 8), max(1, tp // 8)
    channels = local_size + hosts - 2
    scratch = {
        "residual": tpu.VMEM((rows, h), BF16),
        "normed": tpu.VMEM((rows, h), BF16),
        "joined": tpu.VMEM((rows, 2 * h), BF16),
        "hidden": tpu.VMEM((1, h), BF16),
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
        "local": tpu.VMEM((2, local_size, rows, packet), F32),
        "remote": tpu.VMEM((2, hosts, rows, packet), F32),
        "sends": tpu.SemaphoreType.DMA((max(1, channels),)),
        "recvs": tpu.SemaphoreType.DMA((max(1, channels),)),
        "small_sends": tpu.SemaphoreType.DMA((tp - 1,)),
        "small_recvs": tpu.SemaphoreType.DMA((tp - 1,)),
        "copy_sem": tpu.SemaphoreType.DMA,
        "attention_barriers": tpu.SemaphoreType.REGULAR((2,)),
    }
    if paged is not None:
        scratch["new_columns"] = tpu.VMEM((1, 2, hd), jnp.float8_e4m3fn)

    def kernel(position_ref, token_ref, *args):
        tables_ref = None
        if paged is not None:
            tables_ref, args = args[0], args[1:]
        embedding_ref, lm_ref, args = args[0], args[1], args[2:]
        w = dict(zip(names, args[: len(names)], strict=True))
        if paged is not None:
            # The pool is read only; the call's column leaves for the commit.
            pool_ref, hidden_ref = args[len(names)], args[len(names) + 1]
            logits_out, token_out, columns_out, hidden_out = args[len(names) + 2 : len(names) + 6]
            r = dict(zip(scratch, args[len(names) + 6 :], strict=True))
        else:
            # Input state refs are aliased to outputs below; no historical KV copy.
            hidden_ref = args[len(names) + 2]
            logits_out, token_out, key_out, value_out, hidden_out = args[
                len(names) + 3 : len(names) + 8
            ]
            r = dict(zip(scratch, args[len(names) + 8 :], strict=True))
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
            r["normed"][...] = norm(r["residual"][...], r["norm"][...])

        def rope(value):
            half = c.rotary_dim // 2
            frequency = c.rope_theta ** (
                -jnp.arange(half, dtype=jnp.int32).astype(F32) * 2 / c.rotary_dim
            )
            angles = position_ref[0].astype(F32) * frequency
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

        token = token_ref[0]
        start = rank * layout.vocab_shard
        owned = (token >= start) & (token < jnp.minimum(start + layout.vocab_shard, c.vocab))
        r["proj"][...] = jnp.zeros(r["proj"].shape, F32)

        @pl.when(owned)
        def embedding():
            local = token - start
            base = pl.multiple_of(local // 8 * 8, 8)
            copy(embedding_ref.at[pl.ds(base, 8), :], r["row_tile"])
            row = jnp.sum(
                jnp.where(jnp.arange(8)[:, None] == local % 8, r["row_tile"][...].astype(F32), 0),
                axis=0,
            )
            r["proj"][0, :h] = row

        r["residual"][...] = reduce(0).astype(BF16)
        r["joined"][...] = jnp.zeros(r["joined"].shape, BF16)
        copy(w["embedding_norm"], r["norm"])
        r["joined"][0, :h] = norm(r["residual"][0, :], r["norm"][...])
        copy(hidden_ref, r["hidden"])
        copy(w["hidden_norm"], r["norm"])
        r["joined"][0, h:] = norm(r["hidden"][0, :], r["norm"][...])
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
        packed = r["qgate"][0, ...].reshape(qh, 2 * hd)
        query, gate = packed[:, :hd], packed[:, hd:]
        copy(w["qn"], r["qn"])
        copy(w["kn"], r["kn"])
        query = rope(norm(query, r["qn"][...]))
        key = rope(norm(r["key"][0, :], r["kn"][...]))
        if paged is not None:
            for kind, row in enumerate((key, r["value"][0, :])):
                encoded = jnp.clip(row.astype(F32), -448, 448).astype(jnp.float8_e4m3fn)
                r["new_columns"][0, kind] = encoded
            copy(r["new_columns"], columns_out.at[0])
            caches = (PoolPages((pool_ref,), jnp.int32(0), tables_ref, 0, paged["pages_per_block"]), None)
        else:
            caches = (key_out.at[0, 0], value_out.at[0, 0])
        attended = attend_paged_append(
            query,
            key,
            r["value"][0, :],
            *caches,
            position_ref[0],
            rank,
            native_layout,
            (1.0, 1.0),
            pl.DeviceIdType.MESH,
            barriers=r["attention_barriers"],
            buffers=KV_BUFFERS,
        )
        r["attended"][...] = jnp.zeros(r["attended"].shape, BF16)
        r["attended"][0, :] = (
            (attended * jax.nn.sigmoid(gate.astype(F32)).astype(BF16)).reshape(-1).astype(BF16)
        )
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
        r["router"][...] = jax.nn.softmax(r["router"][...], axis=-1)
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
            jnp.sum(
                r["normed"][...].astype(F32) * r["shared_gate_weight"][...].astype(F32),
                axis=1,
            )
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
            (jnp.arange(rows)[:, None] == 0)
            & (r["ids"][...] >= first)
            & (r["ids"][...] < first + layout.local_experts)
        )
        r["ids"][...] = jnp.where(valid, r["ids"][...], c.experts)

        native_experts(
            r["normed"], r["ids"], r["probabilities"], (w["eg"], w["eu"], w["ed"]), first,
            c.experts, r["gate"], r["proj"], r["middle"], r["proj"], r["moe"],
        )
        r["proj"][:, :h] = r["moe"][...]
        partial = reduce(2).astype(BF16)
        r["residual"][...] = (r["residual"][...] + partial).astype(BF16)
        normalize("norm")
        copy(r["normed"], hidden_out)
        native_project(r["normed"], lm_ref, r["proj"], rounded=False)
        vocab_ids = start + jnp.arange(layout.vocab_shard_padded)
        valid = (jnp.arange(layout.vocab_shard_padded) < layout.vocab_shard) & (vocab_ids < c.vocab)
        logits = jnp.where(valid[None], r["proj"][:, : layout.vocab_shard_padded], -jnp.inf)
        logits_out[...] = logits[:1]
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
        token_out[...] = result[:1].astype(jnp.int32)

    if paged is not None:
        arguments = (
            positions,
            tokens,
            paged["tables"].astype(jnp.int32),
            target["emb"],
            target["lmw"],
            *(weights[name] for name in names),
            paged["pool"],
            hidden,
        )
        result = pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((1, layout.vocab_shard_padded), F32),
                jax.ShapeDtypeStruct((1,), jnp.int32),
                jax.ShapeDtypeStruct((1, 1, 2, hd), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8, h), BF16),
            ),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * (len(arguments) - 3),
                out_specs=[pl.BlockSpec(), pl.BlockSpec(), *[pl.BlockSpec(memory_space=tpu.HBM)] * 2],
                scratch_shapes=tuple(scratch.values()),
            ),
            compiler_params=tpu.CompilerParams(
                collective_id=83, vmem_limit_bytes=64 << 20, shape_invariant_numerics=True
            ),
            interpret=interpret,
            name="qwen_native_bf16_mtp_draft_paged",
        )(*arguments)
        return result[1], result[0], {"columns": result[2]}, result[3][:1]
    arguments = (
        positions,
        tokens,
        target["emb"],
        target["lmw"],
        *(weights[name] for name in names),
        state["kcache"],
        state["vcache"],
        hidden,
    )
    state_offset = 4 + len(names)
    if _state_owner_axis:
        if any(value.ndim != 6 or value.shape[0] != 1 for value in state.values()):
            raise ValueError("Native MTP cache must retain one local owner axis")

        def owner_kernel(*refs):
            # State stays full-shape/aliased at the HBM call boundary. Ref views
            # expose the original kernel cache shape without a JAX array squeeze.
            refs = list(refs)
            for index in (state_offset, state_offset + 1, len(arguments) + 2, len(arguments) + 3):
                refs[index] = refs[index].at[0]
            kernel(*refs)

    if _requests == 2:
        def requests_kernel(*refs):
            # One independent request per Pallas call. All full HBM cache
            # operands retain their aliases; only Ref views select a request.
            # A second native body in this call would allocate remote attention
            # inboxes after rank-dependent expert run_scoped allocations.
            output_offset = len(arguments)
            scratch_offset = output_offset + 5
            hidden_tile = refs[scratch_offset + tuple(scratch).index("hidden")]
            padded_hidden = refs[scratch_offset + tuple(scratch).index("normed")]
            copy_sem = refs[scratch_offset + tuple(scratch).index("copy_sem")]

            def copy_ref(source, destination):
                dma = tpu.make_async_copy(source, destination, copy_sem)
                dma.start()
                dma.wait()

            selected = list(refs)
            for index in (0, 1, state_offset, state_offset + 1,
                          state_offset + 2, output_offset + 2, output_offset + 3):
                selected[index] = selected[index].at[pl.ds(_request_index, 1)]

            @pl.when(refs[0][_request_index] >= 0)
            def active_request():
                kernel(*selected)

            @pl.when(refs[0][_request_index] < 0)
            def inactive_request():
                # Do not touch a cache or embedding for inactive requests.
                # The small outputs retain token/carry; ignored valid logits
                # are zero and padded vocabulary columns remain -inf.
                selected[output_offset + 1][...] = jnp.full((1,), selected[1][0], jnp.int32)
                start = jax.lax.axis_index("tp") * layout.vocab_shard
                ids = start + jnp.arange(layout.vocab_shard_padded)
                valid = (jnp.arange(layout.vocab_shard_padded) < layout.vocab_shard) & (ids < c.vocab)
                selected[output_offset][...] = jnp.where(valid[None], jnp.float32(0), -jnp.inf)
                copy_ref(selected[state_offset + 2], hidden_tile)
                padded_hidden[...] = jnp.zeros(padded_hidden.shape, BF16)
                padded_hidden[0, :] = hidden_tile[0, :]
                copy_ref(padded_hidden, selected[output_offset + 4])

        def owner_requests_kernel(*refs):
            refs = list(refs)
            for index in (state_offset, state_offset + 1, len(arguments) + 2, len(arguments) + 3):
                refs[index] = refs[index].at[0]
            requests_kernel(*refs)

    result = pl.pallas_call(
        (owner_kernel if _state_owner_axis else kernel) if _requests == 1
        else (owner_requests_kernel if _state_owner_axis else requests_kernel),
        out_shape=(
            jax.ShapeDtypeStruct((1, layout.vocab_shard_padded), F32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct(state["kcache"].shape, BF16),
            jax.ShapeDtypeStruct(state["vcache"].shape, BF16),
            jax.ShapeDtypeStruct((8, h), BF16),
        ),
        input_output_aliases={state_offset: 2, state_offset + 1: 3},
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * (len(arguments) - 2),
            out_specs=[pl.BlockSpec(), pl.BlockSpec(), *[pl.BlockSpec(memory_space=tpu.HBM)] * 3],
            scratch_shapes=tuple(scratch.values()),
        ),
        compiler_params=tpu.CompilerParams(
            collective_id=83, vmem_limit_bytes=64 << 20, shape_invariant_numerics=True
        ),
        interpret=interpret,
        name="qwen_native_bf16_mtp_draft",
    )(*arguments)
    return result[1], result[0], {"kcache": result[2], "vcache": result[3]}, result[4][:1]


def make_mtp_decode(
    mesh, config, layout, context, *, donate=False, interpret=False, _state_owner_axis=False,
    _requests=1,
):
    """Native one-layer C1 draft; same callable/result contract as MTP reference.

    Unqualified until independent arithmetic and TPU gates pass. Positions are
    predecessor target positions, admitted once by the device generation loop.
    Target embedding/head are shared; all MTP weights/cache remain native BF16.
    """
    if config.target != layout.config or layout.tp != mesh.size or layout.context_parallel != 8:
        raise ValueError("Initial native MTP draft requires matching TP/CP8 identity")
    if layout.tp not in (8, 32) or layout.local_kv_heads != 1:
        raise ValueError("Native MTP draft requires TP8/32 and one local KV head")
    if context < 128 or context % 128 or config.target.dim % layout.tp:
        raise ValueError("MTP draft context/FC output partition is invalid")
    if config.target.experts // layout.router_tp % 128:
        raise ValueError("MTP router shards must occupy complete128-column tiles")

    if type(_requests) is not int or _requests not in (1, 2):
        raise ValueError("Private native MTP requests must be exactly1 or2")
    if _requests == 2:
        return _make_mtp_decode_c2(
            mesh, config, layout, context, donate=donate, interpret=interpret,
            _state_owner_axis=_state_owner_axis,
        )

    def local(target, weights, states, tokens, positions, hidden):
        target, weights = (jax.tree.map(lambda x: x[0], tree) for tree in (target, weights))
        states = states if _state_owner_axis else jax.tree.map(lambda x: x[0], states)
        following, logits, updated, output = _native_stack(
            config, layout, context, target, weights, states, tokens, positions, hidden, interpret,
            _state_owner_axis=_state_owner_axis,
        )
        updated = updated if _state_owner_axis else jax.tree.map(lambda x: x[None], updated)
        return following, logits[None], updated, output

    def step(target, weights, state, tokens, positions, hidden):
        if (
            tokens.shape != (1,)
            or positions.shape != (1,)
            or not jnp.issubdtype(positions.dtype, jnp.signedinteger)
        ):
            raise ValueError(
                "Native MTP draft requires one request and a signed predecessor position"
            )
        if hidden.shape != (1, config.target.dim) or hidden.dtype != BF16:
            raise ValueError("Native MTP draft requires target post-final-norm BF16 [1,H]")
        expected_weights = weight_shapes(config, layout)
        expected_cache = (
            layout.tp,
            1,
            1,
            layout.local_kv_pages(context),
            128,
            config.target.head_dim,
        )
        if set(weights) != set(expected_weights) or any(
            weights[name].shape != (layout.tp, *shape) for name, shape in expected_weights.items()
        ):
            raise ValueError("Native MTP weight keys/shapes differ from explicit ownership")
        if set(state) != {"kcache", "vcache"} or any(
            value.shape != expected_cache for value in state.values()
        ):
            raise ValueError("Native MTP draft owns exactly one request and its own CP8 cache")
        if any(value.dtype != BF16 for value in weights.values()) or any(
            value.dtype != BF16 for value in state.values()
        ):
            raise ValueError("MTP-owned weights and cache must remain native BF16")
        if target["emb"].dtype != BF16 or target["lmw"].dtype != BF16:
            raise ValueError("MTP shared target embedding/head must remain BF16")
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("tp"), P("tp"), P("tp"), P(), P(), P()),
            out_specs=(P(), P("tp"), P("tp"), P()),
            check_vma=False,
        )(target, weights, state, tokens, positions, hidden)

    return jax.jit(step, donate_argnums=(2,) if donate else ())



def _admit_native_c2(compiled, *, context, vocab):
    """Admit only six replicated metadata words before an outer donation.

    Device-loop callers must validate their concrete entry metadata once, then
    use __wrapped__ with positions/tokens proven valid whenever active. Admission
    cannot run inside a surrounding trace. Inactive metadata is deliberately
    opaque and is never passed to embedding or cache indexing.
    """
    def validate_entry(*args, **kwargs):
        names = ("tokens", "positions", "active")
        indexes = (3, 4, 6)
        values = [args[index] if len(args) > index else kwargs[name]
                  for name, index in zip(names, indexes, strict=True)]
        if any(isinstance(value, jax.core.Tracer) for value in values):
            raise TypeError("C2 entry admission must precede surrounding JIT tracing")
        tokens, positions, active = values
        if (tokens.shape != (2,) or positions.shape != (2,) or active.shape != (2,)
                or tokens.dtype != jnp.int32 or positions.dtype != jnp.int32
                or active.dtype != jnp.bool_):
            raise ValueError("C2 metadata requires int32 tokens/positions and boolean active [2]")
        host = [value.tolist() for value in jax.device_get(tuple(values))]
        for token, position, enabled in zip(*host, strict=True):
            if enabled and not (0 <= token < vocab and 0 <= position < context):
                raise ValueError("Active C2 token or predecessor position is out of bounds")

    @wraps(compiled)
    def admitted(*args, **kwargs):
        validate_entry(*args, **kwargs)
        return compiled(*args, **kwargs)

    def lower(*args, **kwargs):
        validate_entry(*args, **kwargs)
        return compiled.lower(*args, **kwargs)

    admitted.lower = lower
    admitted.validate_entry = validate_entry
    return admitted


def _make_mtp_decode_c2(mesh, config, layout, context, *, donate, interpret, _state_owner_axis):
    """Private two-independent-request component; public runtime remains C1.

    Two sequential aliased Pallas calls each execute the exact C1 kernel on one
    Ref view of the full C2 cache. A replicated mask skips inactive native work.
    This isolates request-local scoped scratch while chaining full cache aliases;
    only small outputs concatenate. It does not claim batched dense speedup.
    """
    def local(target, weights, states, tokens, positions, hidden):
        target, weights = (jax.tree.map(lambda x: x[0], tree) for tree in (target, weights))
        states = states if _state_owner_axis else jax.tree.map(lambda x: x[0], states)
        small_outputs = []
        for request in range(2):
            following, logits, states, output = _native_stack(
                config, layout, context, target, weights, states, tokens, positions, hidden,
                interpret, _state_owner_axis=_state_owner_axis, _requests=2,
                _request_index=request,
            )
            small_outputs.append((following, logits, output))
        following, logits, output = (
            jnp.concatenate(values, axis=0) for values in zip(*small_outputs, strict=True)
        )
        updated = states if _state_owner_axis else jax.tree.map(lambda x: x[None], states)
        return following, logits[None], updated, output

    def step(target, weights, state, tokens, positions, hidden, active):
        if (tokens.shape != (2,) or positions.shape != (2,) or active.shape != (2,)
                or tokens.dtype != jnp.int32 or positions.dtype != jnp.int32
                or active.dtype != jnp.bool_):
            raise ValueError("C2 metadata requires int32 tokens/positions and boolean active [2]")
        if hidden.shape != (2, config.target.dim) or hidden.dtype != BF16:
            raise ValueError("C2 native MTP requires normalized target BF16 hidden [2,H]")
        expected_weights = weight_shapes(config, layout)
        if set(weights) != set(expected_weights) or any(
            weights[name].shape != (layout.tp, *shape) or weights[name].dtype != BF16
            for name, shape in expected_weights.items()
        ):
            raise ValueError("C2 native MTP requires exact native BF16 weight ownership")
        expected_cache = (layout.tp, 2, 1, layout.local_kv_pages(context), 128,
                          config.target.head_dim)
        if set(state) != {"kcache", "vcache"} or any(
            value.shape != expected_cache or value.dtype != BF16 for value in state.values()
        ):
            raise ValueError("C2 native MTP requires two independently owned BF16 CP8 caches")
        shapes = {"emb": (layout.tp, layout.vocab_shard_padded, config.target.dim),
                  "lmw": (layout.tp, config.target.dim, layout.vocab_shard_padded)}
        if any(target[name].shape != shape or target[name].dtype != BF16
               for name, shape in shapes.items()):
            raise ValueError("C2 native MTP must share the original BF16 embedding and head")
        # Only a two-word scalar-prefetch vector changes. All full caches pass
        # directly into Pallas; a negative sentinel is private to the C2 wrapper.
        admitted_positions = jnp.where(active, positions, jnp.int32(-1))
        return jax.shard_map(
            local, mesh=mesh,
            in_specs=(P("tp"), P("tp"), P("tp"), P(), P(), P()),
            out_specs=(P(), P("tp"), P("tp"), P()), check_vma=False,
        )(target, weights, state, tokens, admitted_positions, hidden)

    compiled = jax.jit(step, donate_argnums=(2,) if donate else ())
    return _admit_native_c2(compiled, context=context, vocab=config.target.vocab)
