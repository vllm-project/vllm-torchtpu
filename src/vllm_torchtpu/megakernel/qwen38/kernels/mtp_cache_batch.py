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
"""Private C1 native BF16 repair of <=3 causal rows in one KV-only invocation.

No runtime selects this candidate. The request axis remains one; three causal
rows share the qualified eight-row projection tile. No attention or FFN runs.
All active rows require verified target hidden, never provisional draft hidden.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import PartitionSpec as P

from .collectives import hierarchical_sum
from .mtp_config import weight_shapes
from .mtp_dense import native_project

BF16 = jnp.bfloat16
F32 = jnp.float32
FAMILIES = ("embedding_norm", "hidden_norm", "fc", "norm_in", "kw", "vw", "kn")


def _cache_batch_stack(config, layout, context, embedding, weights, state, tokens, positions,
                       hidden, active, interpret, *, paged=False):
    """KV rows of up to three causal draft positions. With ``paged`` no cache
    is read or written: the rows' key/value columns (FP8 ``[1, 3, 2, D]``, as
    vLLM's pools store them) are returned as ``{"columns"}`` for the caller's
    commit (``state`` unused)."""
    c, tp, rows = config.target, layout.tp, 8
    h, hd = c.dim, c.head_dim
    fc_width = h // tp
    fc_packet = max(128, (fc_width + 127) // 128 * 128)
    compact = h % 1024 == 0
    # All three causal embedding rows must cross the compact collective.
    packet = 3 * h // 8 if compact else h
    local_size, hosts = min(tp, 8), max(1, tp // 8)
    channels = local_size + hosts - 2
    scratch = {
        "residual": tpu.VMEM((rows, h), BF16),
        "normed": tpu.VMEM((rows, h), BF16),
        "joined": tpu.VMEM((rows, 2 * h), BF16),
        "hidden": tpu.VMEM((1, h), BF16),
        "row_tile": tpu.VMEM((8, h), BF16),
        "norm": tpu.VMEM((h,), BF16),
        "kn": tpu.VMEM((hd,), BF16),
        "proj": tpu.VMEM((rows, max(h, hd, 128)), F32),
        "key": tpu.VMEM((rows, hd), BF16),
        "value": tpu.VMEM((rows, hd), BF16),
        "fc_parts": tpu.VMEM((tp, rows, fc_packet), BF16),
        "local": tpu.VMEM((2, local_size, rows, packet), F32),
        "remote": tpu.VMEM((2, hosts, rows, packet), F32),
        "sends": tpu.SemaphoreType.DMA((max(1, channels),)),
        "recvs": tpu.SemaphoreType.DMA((max(1, channels),)),
        "small_sends": tpu.SemaphoreType.DMA((tp - 1,)),
        "small_recvs": tpu.SemaphoreType.DMA((tp - 1,)),
        "copy_sem": tpu.SemaphoreType.DMA,
        "write_page": tpu.VMEM((128, hd), BF16),
    }
    if paged:
        scratch["new_columns"] = tpu.VMEM((3, 2, hd), jnp.float8_e4m3fn)

    def kernel(position_ref, token_ref, active_ref, embedding_ref, *args):
        w = dict(zip(FAMILIES, args[:len(FAMILIES)], strict=True))
        if paged:
            hidden_refs = args[len(FAMILIES):len(FAMILIES) + 3]
            (columns_out,) = args[len(FAMILIES) + 3:len(FAMILIES) + 4]
            r = dict(zip(scratch, args[len(FAMILIES) + 4:], strict=True))
        else:
            hidden_refs = args[len(FAMILIES) + 2:len(FAMILIES) + 5]
            key_out, value_out = args[len(FAMILIES) + 5:len(FAMILIES) + 7]
            r = dict(zip(scratch, args[len(FAMILIES) + 7:], strict=True))
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

        # Three causal rows of one request occupy the first three of the
        # existing eight padded MXU rows. The cache still owns one request.
        start = rank * layout.vocab_shard
        r["proj"][...] = jnp.zeros(r["proj"].shape, F32)
        for slot in range(3):
            token = token_ref[slot]
            owned = (active_ref[slot] != 0) & (token >= start) & (
                token < jnp.minimum(start + layout.vocab_shard, c.vocab)
            )

            @pl.when(owned)
            def embedding(slot=slot, token=token):
                local = token - start
                base = pl.multiple_of(local // 8 * 8, 8)
                copy(embedding_ref.at[pl.ds(base, 8), :], r["row_tile"])
                row = jnp.sum(
                    jnp.where(jnp.arange(8)[:, None] == local % 8, r["row_tile"][...].astype(F32), 0),
                    axis=0,
                )
                r["proj"][slot, :h] = row

        r["residual"][...] = reduce(0).astype(BF16)
        r["joined"][...] = jnp.zeros(r["joined"].shape, BF16)
        copy(w["embedding_norm"], r["norm"])
        for slot in range(3):
            @pl.when(active_ref[slot] != 0)
            def normalize_embedding(slot=slot):
                r["joined"][slot, :h] = norm(r["residual"][slot, :], r["norm"][...])
        copy(w["hidden_norm"], r["norm"])
        for slot in range(3):
            @pl.when(active_ref[slot] != 0)
            def normalize_hidden(slot=slot):
                # Inactive hidden may be poison and must never be loaded.
                copy(hidden_refs[slot], r["hidden"])
                r["joined"][slot, h:] = norm(r["hidden"][0, :], r["norm"][...])
        native_project(r["joined"], w["fc"], r["proj"])
        r["fc_parts"][0, ...] = r["proj"][:, :fc_packet].astype(BF16)
        exchange(r["fc_parts"], tp)
        for absolute in range(tp):
            r["residual"][:, pl.ds(absolute * fc_width, fc_width)] = r["fc_parts"][
                rank ^ absolute, :, :fc_width
            ]
        normalize("norm_in")
        native_project(r["normed"], w["kw"], r["key"])
        native_project(r["normed"], w["vw"], r["value"])
        copy(w["kn"], r["kn"])
        if paged:
            # Every rank's rows of its KV head; the commit writes them on the owners.
            for slot in range(3):
                key = rope(norm(r["key"][slot, :], r["kn"][...]), position_ref[slot])
                for kind, row in enumerate((key, r["value"][slot, :])):
                    r["new_columns"][slot, kind] = jnp.clip(row.astype(F32), -448, 448).astype(
                        jnp.float8_e4m3fn
                    )
            copy(r["new_columns"], columns_out.at[0])
            return
        for slot in range(3):
            position = position_ref[slot]
            owner, page = position // 128 % 8, position // 1024

            @pl.when((active_ref[slot] != 0) & (owner == rank % 8))
            def append(slot=slot, position=position, page=page):
                key = rope(norm(r["key"][slot, :], r["kn"][...]), position)
                # Sequential completed page RMW preserves prior rows when two
                # or three repairs share a page; other owners never touch it.
                mask = jnp.arange(128)[:, None] == position % 128
                for cache, row in ((key_out, key), (value_out, r["value"][slot, :])):
                    destination = cache.at[0, 0, page, :, :]
                    copy(destination, r["write_page"])
                    r["write_page"][...] = jnp.where(mask, row[None, :], r["write_page"][...])
                    copy(r["write_page"], destination)

    # Split only the tiny hidden operand before entering Pallas. A whole [1,H]
    # DMA is legal even when a combined [3,H] HBM buffer has a four-row tile.
    # Inactive rows remain untouched by the original per-slot active branch.
    if paged:
        arguments = (positions, tokens, active.astype(jnp.int32), embedding,
                     *(weights[name] for name in FAMILIES),
                     *(hidden[slot:slot + 1, :] for slot in range(3)))
        (columns,) = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((1, 3, 2, hd), jnp.float8_e4m3fn),),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=3,
                in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * (len(arguments) - 3),
                out_specs=[pl.BlockSpec(memory_space=tpu.HBM)],
                scratch_shapes=tuple(scratch.values()),
            ),
            compiler_params=tpu.CompilerParams(
                collective_id=87, vmem_limit_bytes=64 << 20, shape_invariant_numerics=True
            ),
            interpret=interpret,
            name="qwen_native_bf16_mtp_cache_batch_columns",
        )(*arguments)
        return {"columns": columns}
    arguments = (positions, tokens, active.astype(jnp.int32), embedding, *(weights[name] for name in FAMILIES),
                 state["kcache"], state["vcache"],
                 *(hidden[slot:slot + 1, :] for slot in range(3)))
    state_offset = 4 + len(FAMILIES)
    result = pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(state[name].shape, BF16)
                        for name in ("kcache", "vcache")),
        input_output_aliases={state_offset: 0, state_offset + 1: 1},
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * (len(arguments) - 3),
            out_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * 2,
            scratch_shapes=tuple(scratch.values()),
        ),
        compiler_params=tpu.CompilerParams(
            collective_id=87, vmem_limit_bytes=64 << 20, shape_invariant_numerics=True
        ),
        interpret=interpret,
        name="qwen_native_bf16_mtp_cache_batch",
    )(*arguments)
    return dict(zip(("kcache", "vcache"), result, strict=True))


def make_mtp_cache_batch_update(mesh, config, layout, context, *, donate=False, interpret=False):
    """Return an optional C1 updater with explicit [request=1,causal=3] axes.

    tokens/positions/active are [1,3]; hidden is normalized BF16 [1,3,H]. Active
    rows must be a consecutive prefix and use verified target hidden. State is
    the existing BF16 [TP,request=1,KVhead=1,CPpages,128,HD] draft cache. We never
    allocate three request states or copy a prefix. Only active CP owners write.

    ``validate_entry`` and ``lower`` admit concrete metadata before donation.
    Device-loop callers may use ``update`` after maintaining that contract,
    as the existing single-row updater does. Explicit compiled invocations take
    ``executable_arguments(*full_args)``. Outer generation alone owns donation.
    This factory is not selected by any existing runtime or controller.
    """
    if (config.target != layout.config or layout.tp != mesh.size
            or layout.context_parallel != 8 or layout.tp not in (8, 32)
            or layout.local_kv_heads != 1):
        raise ValueError("Batched MTP repair requires matching C1 TP8/32 CP8 layout")
    if context < 128 or context % 128 or config.target.dim % layout.tp:
        raise ValueError("Invalid batched MTP cache capacity/FC ownership")
    specs = weight_shapes(config, layout)
    expected_cache = (layout.tp, 1, 1, layout.local_kv_pages(context), 128, config.target.head_dim)

    def shapes(embedding, weights, state, tokens, positions, hidden, active):
        if (tokens.shape != (1, 3) or positions.shape != (1, 3)
                or tokens.dtype != jnp.int32 or positions.dtype != jnp.int32
                or active.shape != (1, 3) or active.dtype != jnp.bool_):
            raise ValueError("Repair needs separate [request=1,causal=3] int32 positions/tokens and bool activity")
        if hidden.shape != (1, 3, config.target.dim) or hidden.dtype != BF16:
            raise ValueError("Repair needs verified normalized BF16 target hidden [1,3,H]")
        if embedding.shape != (layout.tp, layout.vocab_shard_padded, config.target.dim) or embedding.dtype != BF16:
            raise ValueError("Repair must share the original BF16 target embedding")
        if set(weights) != set(FAMILIES) or any(
            weights[name].shape != (layout.tp, *specs[name]) or weights[name].dtype != BF16
            for name in FAMILIES
        ):
            raise ValueError("Repair needs exactly seven native BF16 weight families")
        if set(state) != {"kcache", "vcache"} or any(
            value.shape != expected_cache or value.dtype != BF16 for value in state.values()
        ):
            raise ValueError("Three causal rows must retain one independently owned BF16 request cache")

    def local(embedding, weights, state, tokens, positions, hidden, active):
        embedding = embedding[0]
        weights, state = (jax.tree.map(lambda value: value[0], tree) for tree in (weights, state))
        updated = _cache_batch_stack(
            config, layout, context, embedding, weights, state,
            tokens[0], positions[0], hidden[0], active[0], interpret,
        )
        return jax.tree.map(lambda value: value[None], updated)

    def step(embedding, weights, state, tokens, positions, hidden, active):
        shapes(embedding, weights, state, tokens, positions, hidden, active)
        return jax.shard_map(
            local, mesh=mesh,
            in_specs=(P("tp"), P("tp"), P("tp"), P(), P(), P(), P()),
            out_specs=P("tp"), check_vma=False,
        )(embedding, weights, state, tokens, positions, hidden, active)

    compiled = jax.jit(step, donate_argnums=(2,) if donate else ())

    def select(target, weights, state, tokens, positions, hidden, active):
        return target["emb"], {name: weights[name] for name in FAMILIES}, state, tokens, positions, hidden, active

    def validate_entry(*args):
        values = select(*args)
        shapes(*values)
        tokens, positions, active = values[3], values[4], values[6]
        if any(isinstance(value, jax.core.Tracer) for value in (tokens, positions, active)):
            raise TypeError("Concrete batched repair admission must precede tracing/donation")
        token_rows, position_rows, masks = [value.tolist()[0] for value in jax.device_get((tokens, positions, active))]
        count = sum(masks)
        if masks != [slot < count for slot in range(3)]:
            raise ValueError("Consumed repairs must form a causal prefix")
        for slot in range(count):
            if not (0 <= token_rows[slot] < config.target.vocab and 0 <= position_rows[slot] < context):
                raise ValueError("Active repair token/position outside model bounds")
            if slot and position_rows[slot] != position_rows[0] + slot:
                raise ValueError("Repair rows must occupy consecutive shifted positions")

    def update(*args):
        return compiled(*select(*args))

    def lower(*args):
        validate_entry(*args)
        return compiled.lower(*select(*args))

    update.executable_arguments = select
    update.validate_entry = validate_entry
    update.lower = lower
    update.mtp_cache_batch_shape = (1, 3)
    return update
