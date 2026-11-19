# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 BF16 prefill rings over an ascending-device-id production mesh.

Physical ring position, mesh index, and Torch token owner are distinct.
Tables explicitly translate all three; never permute the JAX mesh to match
physical coordinates in the multi-process runtime.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

CYCLES = (
    (0, 4, 12, 13, 14, 15, 7, 3, 11, 10, 2, 6, 5, 1, 9, 8),
    (0, 1, 9, 13, 14, 10, 2, 3, 11, 15, 7, 6, 5, 4, 12, 8),
    (0, 1, 2, 3, 7, 6, 14, 15, 11, 10, 9, 8, 12, 13, 5, 4),
)


def mla_chip_layout(devices, owners):
    """Physical pairs and same-core groups, ordered by Torch token ownership."""
    chips = {}
    for index, device in enumerate(devices):
        chips.setdefault(tuple(device.coords), []).append(index)
    pairs = tuple(
        tuple(sorted(pair, key=lambda i: owners[i]))
        for _, pair in sorted(chips.items())
    )
    if len(pairs) != 16 or any(len(pair) != 2 for pair in pairs):
        raise ValueError("K3 MLA preprocessing requires sixteen two-core chips")
    parity = tuple(tuple(pair[i] for pair in pairs) for i in range(2))
    token_order = tuple(sorted(range(32), key=lambda i: owners[pairs[i // 2][i % 2]]))
    return pairs, parity, token_order


def pack_mla_gate(weight, mesh, pairs):
    """Replicate complementary TP2 gate shards from the original TP32 weights."""
    parity = tuple(tuple(pair[i] for pair in pairs) for i in range(2))

    def local(weight):
        return jax.lax.all_gather(
            weight[:, 2112:], "k3_sp", axis=1, tiled=True, axis_index_groups=parity
        )

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=P("k3_sp", None),
        out_specs=P("k3_sp", None),
        check_vma=False,
    )(weight)


def mla_preprocess(
    x, weight, packed_gate, qnorm, kvnorm, mesh, owners, *, sequence_parallel, eps
):
    """Source MLA layout with a common parameter interface for all buckets."""
    pairs, parity, token_order = mla_chip_layout(list(mesh.devices.flat), owners)
    owner_order = tuple(sorted(range(32), key=lambda i: owners[i]))

    def local(x, weight, packed_gate, qnorm, kvnorm):
        def norm(value, scale):
            value = value.astype(jnp.float32)
            return (
                value
                * jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + eps)
                * scale.astype(jnp.float32)
            ).astype(x.dtype)

        if sequence_parallel:
            projected = x @ weight[:, :2112]
            gate_input = jax.lax.all_gather(
                x, "k3_sp", axis=0, tiled=True, axis_index_groups=pairs
            )
            gate = gate_input @ packed_gate
            gate = jax.lax.all_to_all(
                gate,
                "k3_sp",
                split_axis=1,
                concat_axis=0,
                tiled=True,
                axis_index_groups=parity,
            )
            gate = gate.reshape(32, x.shape[0], 384)[jnp.asarray(token_order)]
            gate = gate.reshape(32 * x.shape[0], 384)
        else:
            projected = x @ weight
            projected, gate = projected[:, :2112], projected[:, 2112:]
        q, kv, rope = jnp.split(projected, [1536, 2048], axis=-1)
        q, kv = norm(q, qnorm), norm(kv, kvnorm)
        if sequence_parallel:
            latent = jnp.pad(jnp.concatenate((q, kv, rope), axis=-1), ((0, 0), (0, 64)))
            latent = jax.lax.all_gather(latent, "k3_sp", axis=0, tiled=True)
            latent = latent.reshape(32, x.shape[0], 2176)[jnp.asarray(owner_order)]
            latent = latent.reshape(32 * x.shape[0], 2176)
            q, kv, rope = latent[:, :1536], latent[:, 1536:2048], latent[:, 2048:2112]
        return q, kv, rope, gate

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P("k3_sp", None),) * 3 + (P(), P()),
        out_specs=(P("k3_sp", None),) * 4,
        check_vma=False,
    )(x, weight, packed_gate, qnorm, kvnorm)


def gather_moe_inputs(x, weights, ids, mesh, tables, *, collective_id):
    """Gather latent tokens, BF16 weights and signed 16-bit IDs together.

    Byte-valued BF16 lanes preserve IDs without NaN/subnormal bit encodings.
    The caller ensures IDs fit int16; K3 has 896 global experts.
    """

    def pack(x, weights, ids):
        low = (ids & 255).astype(jnp.bfloat16)
        high = ((ids >> 8) & 255).astype(jnp.bfloat16)
        payload = jnp.concatenate((x, weights, low, high), axis=1)
        return jnp.pad(payload, ((0, 0), (0, (-payload.shape[1]) % 128)))

    packed = jax.shard_map(
        pack,
        mesh=mesh,
        in_specs=(P("k3_sp", None),) * 3,
        out_specs=P("k3_sp", None),
        check_vma=False,
    )(x, weights, ids)
    result = ring_collective(
        packed, mesh, tables, gather=True, collective_id=collective_id
    )
    width, topk = x.shape[1], weights.shape[1]
    gathered_weights = result[:, width : width + topk]
    low = result[:, width + topk : width + 2 * topk].astype(jnp.int32)
    high = result[:, width + 2 * topk : width + 3 * topk].astype(jnp.int32)
    gathered_ids = ((high * 256 + low) << 16) >> 16
    return result[:, :width], gathered_weights, gathered_ids


def ring_tables(devices, token_ranks):
    """Build [family,position/peer/token-owner,mesh-index] lookup tables."""
    if len(devices) != 32 or sorted(token_ranks) != list(range(32)):
        raise ValueError("K3 rings require 32 devices and a token-owner permutation")
    if [d.id for d in devices] != sorted(d.id for d in devices):
        raise ValueError("Production collective mesh must use ascending device IDs")
    coords = [sorted({d.coords[a] for d in devices}) for a in range(3)]
    if tuple(map(len, coords)) != (2, 2, 4):
        raise ValueError("K3 rings require a physical 2x2x4 slice")
    physical_to_mesh = {}
    for i, device in enumerate(devices):
        x, y, z = [coords[a].index(device.coords[a]) for a in range(3)]
        if device.core_on_chip not in (0, 1):
            raise ValueError("Expected two cores per chip")
        physical = ((x * 2 + y) * 4 + z) * 2 + device.core_on_chip
        physical_to_mesh[physical] = i
    if len(physical_to_mesh) != 32:
        raise ValueError("Duplicate physical cores")
    result = []
    for cycle in CYCLES:
        peers = tuple(
            physical_to_mesh[chip * 2 + core] for chip in cycle for core in range(2)
        )
        positions = tuple(peers.index(i) for i in range(32))
        result.append((positions, peers, tuple(token_ranks)))
    return np.asarray(result, np.int32)


def _regions(rows, width):
    if rows % 64 or width % 128 or width < 384:
        raise ValueError("K3 rings need 64-row ownership and 128-column alignment")
    blocks = width // 128
    widths = [128 * (blocks // 3 + (i < blocks % 3)) for i in range(3)]
    cols = [0, widths[0], widths[0] + widths[1]]
    return tuple(
        ((2 * group + direction) * (rows // 4), rows // 4, cols[family], widths[family])
        for group in range(2)
        for family in range(3)
        for direction in range(2)
    )


def _reduce_scatter_kernel(
    tables_ref,
    x_ref,
    o_ref,
    local_sems,
    send_sems,
    recv_sems,
    credits,
    store_sem,
    local_buf,
    payload,
    received,
    *,
    streams,
    buffers,
):
    tile = pl.program_id(0)
    step = pl.program_id(1)
    rank = lax.axis_index("k3_sp")
    descriptors = []
    reads = []
    positions = [tables_ref[f, 0, rank] for f in range(3)]

    def ring_coords(pos, family):
        peer = tables_ref[family, 1, pos]
        return (peer,)

    @pl.when(step == 0)
    def enter():
        peers = [
            ring_coords((positions[f] + direction) % 32, jnp.int32(f))
            for f in range(3)
            for direction in (-1, 1)
        ]
        sem = pltpu.get_barrier_semaphore()
        for peer in peers:
            pl.semaphore_signal(
                sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(sem, len(peers))

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready):
            for peer in peers:
                pl.semaphore_signal(
                    ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(ready, len(peers))

    for stream, region in enumerate(streams):
        begin, size = region[:2]
        col_begin, width = region[2:] if len(region) == 4 else (0, o_ref.shape[1])
        cols = pl.ds(col_begin, width)
        family = (stream // 2) % 3
        direction = -1 if stream % 2 == 0 else 1
        position = positions[family]
        target = ring_coords((position + direction) % 32, jnp.int32(family))
        source = ring_coords((position - direction) % 32, jnp.int32(family))
        owner_mesh = ring_coords(
            (position - direction * (step + 1)) % 32, jnp.int32(family)
        )[0]
        owner = tables_ref[family, 2, owner_mesh]
        rows = pl.ds(begin, size)
        read = pltpu.make_async_copy(
            x_ref.at[
                pl.ds(owner * o_ref.shape[0] + tile * payload.shape[0] + begin, size),
                cols,
            ],
            local_buf.at[rows, cols],
            local_sems.at[stream],
        )
        read.start()
        reads.append(read)
        descriptors.append((target, source, rows, cols))

    for stream, (target, source, rows, cols) in enumerate(descriptors):

        def remote(index):
            return pltpu.make_async_remote_copy(
                payload.at[rows, cols],
                received.at[index % buffers, rows, cols],
                send_sems.at[stream],
                recv_sems.at[index % buffers, stream],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step > 0)
        def wait():
            remote(step - 1).wait()

        reads[stream].wait()

        @pl.when(step == 0)
        def seed():
            payload.at[rows, cols][...] = local_buf.at[rows, cols][...].astype(
                payload.dtype
            )

        @pl.when(step > 0)
        def add():
            payload.at[rows, cols][...] = (
                local_buf.at[rows, cols][...].astype(payload.dtype)
                + received.at[(step - 1) % buffers, rows, cols][...]
            )
            pl.semaphore_signal(
                credits.at[stream],
                1,
                device_id=source,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step < 31)
        def forward():
            @pl.when(step >= buffers)
            def acquire():
                pl.semaphore_wait(credits.at[stream], 1)

            remote(step).start()

        @pl.when(step == 31)
        def finish():
            pl.semaphore_wait(credits.at[stream], buffers)
            local_buf.at[rows, cols][...] = payload.at[rows, cols][...].astype(
                local_buf.dtype
            )
            output = pltpu.make_async_copy(
                local_buf.at[rows, cols],
                o_ref.at[pl.ds(tile * payload.shape[0] + rows.start, rows.size), cols],
                store_sem,
            )
            output.start()
            output.wait()


def _all_gather_kernel(
    tables,
    x,
    out,
    load_sems,
    send_sems,
    recv_sems,
    credits,
    store_sems,
    payload,
    received,
    *,
    regions,
    buffers,
):
    tile = pl.program_id(0)
    step = pl.program_id(1)
    rank = lax.axis_index("k3_sp")
    positions = [tables[f, 0, rank] for f in range(3)]

    def owner(pos, family):
        return tables[family, 1, pos % 32]

    def peer(pos, family):
        r = owner(pos, family)
        return (r,)

    @pl.when(step == 0)
    def enter():
        peers = [
            peer(positions[f] + d, jnp.int32(f)) for f in range(3) for d in (-1, 1)
        ]
        barrier = pltpu.get_barrier_semaphore()
        for p in peers:
            pl.semaphore_signal(
                barrier, 1, device_id=p, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, len(peers))

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready):
            for p in peers:
                pl.semaphore_signal(
                    ready, 1, device_id=p, device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(ready, len(peers))

        for s, (begin, size, col, width) in enumerate(regions):
            rows, cols = pl.ds(begin, size), pl.ds(col, width)
            pltpu.make_async_copy(
                x.at[pl.ds(tile * payload.shape[0] + begin, size), cols],
                payload.at[rows, cols],
                load_sems.at[s],
            ).start()

    for s, (begin, size, col, width) in enumerate(regions):
        family = (s // 2) % 3
        direction = -1 if s % 2 == 0 else 1
        pos = positions[family]
        target = peer(pos + direction, jnp.int32(family))
        source = peer(pos - direction, jnp.int32(family))
        rows, cols = pl.ds(begin, size), pl.ds(col, width)

        def remote(index):
            return pltpu.make_async_remote_copy(
                payload.at[rows, cols],
                received.at[index % buffers, rows, cols],
                send_sems.at[s],
                recv_sems.at[index % buffers, s],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            )

        def store(index):
            owner_mesh = owner(pos - direction * index, jnp.int32(family))
            dest = tables[family, 2, owner_mesh]
            return pltpu.make_async_copy(
                payload.at[rows, cols],
                out.at[
                    pl.ds(dest * x.shape[0] + tile * payload.shape[0] + begin, size),
                    cols,
                ],
                store_sems.at[s],
            )

        @pl.when(step == 0)
        def seed():
            pltpu.make_async_copy(
                x.at[pl.ds(tile * payload.shape[0] + begin, size), cols],
                payload.at[rows, cols],
                load_sems.at[s],
            ).wait()

        @pl.when(step > 0)
        def receive():
            remote(step - 1).wait()
            store(step - 1).wait()
            payload.at[rows, cols][...] = received.at[(step - 1) % buffers, rows, cols][
                ...
            ]
            pl.semaphore_signal(
                credits.at[s], 1, device_id=source, device_id_type=pl.DeviceIdType.MESH
            )

        store(step).start()

        @pl.when(step < 31)
        def forward():
            @pl.when(step >= buffers)
            def acquire():
                pl.semaphore_wait(credits.at[s], 1)

            remote(step).start()

        @pl.when(step == 31)
        def finish():
            store(step).wait()
            pl.semaphore_wait(credits.at[s], buffers)


def ring_collective(x, mesh, tables, *, gather, collective_id):
    """AG takes global [T,D]; RS takes global [32,T,D] partials."""
    if tuple(mesh.axis_names) != ("k3_sp",) or mesh.size != 32:
        raise ValueError("Expected a flat 32-device k3_sp mesh")
    if x.dtype != jnp.bfloat16:
        raise ValueError("K3 rings currently transfer and reduce BF16")
    tokens = x.shape[0] if gather else x.shape[1]
    width = x.shape[-1]
    rows = tokens // 32
    if tokens % 32 or (not gather and x.shape[0] != 32):
        raise ValueError("Invalid token or partial-shard count")
    if rows % 64 or rows > 256:
        # The ring's DMA tiles require 64-row alignment and bounded VMEM.
        # XLA handles other shard sizes, preserving Torch token ownership.
        owners = tuple(int(i) for i in tables[0, 2])
        owner_order = tuple(sorted(range(32), key=lambda i: owners[i]))

        def general(part):
            if gather:
                return lax.all_gather(
                    part, "k3_sp", axis=0, tiled=True, axis_index_groups=(owner_order,)
                )
            ordered = part[0].reshape(32, rows, width)[jnp.asarray(owners)]
            return lax.psum_scatter(
                ordered.reshape(tokens, width), "k3_sp", scatter_dimension=0, tiled=True
            )

        return jax.shard_map(
            general,
            mesh=mesh,
            in_specs=P("k3_sp", None) if gather else P("k3_sp", None, None),
            out_specs=P() if gather else P("k3_sp", None),
            check_vma=False,
        )(x)
    regions = _regions(rows, width)
    streams, buffers = len(regions), 2

    def local(part):
        scratch = (
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((buffers, streams)),
            pltpu.SemaphoreType.REGULAR((streams,)),
        )
        if gather:
            scratch += (
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.VMEM((rows, width), part.dtype),
                pltpu.VMEM((buffers, rows, width), part.dtype),
            )
            kernel = functools.partial(
                _all_gather_kernel, regions=regions, buffers=buffers
            )
        else:
            scratch += (
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM((rows, width), part.dtype),
                pltpu.VMEM((rows, width), part.dtype),
                pltpu.VMEM((buffers, rows, width), part.dtype),
            )
            kernel = functools.partial(
                _reduce_scatter_kernel, streams=regions, buffers=buffers
            )
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (tokens if gather else rows, width), part.dtype
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(1, 32),
                in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                scratch_shapes=scratch,
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=(3 if gather else 4) * rows * width * 2 + 8 * 1024**2,
            ),
            name="kimi_ring_ag" if gather else "kimi_ring_rs",
        )(jnp.asarray(tables, jnp.int32), part if gather else part[0])

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=P("k3_sp", None) if gather else P("k3_sp", None, None),
        out_specs=P() if gather else P("k3_sp", None),
        check_vma=False,
    )(x)


def _project_ring_kernel(
    tables_ref,
    x_ref,
    w_ref,
    o_ref,
    local_sems,
    send_sems,
    recv_sems,
    credits,
    store_sem,
    local_buf,
    payload,
    received,
    weights,
    inputs,
    weight_sem,
    *,
    streams,
    buffers,
):
    tile = pl.program_id(0)
    step = pl.program_id(1)

    descriptors = []
    reads = []
    rank = lax.axis_index("k3_sp")
    positions = [tables_ref[f, 0, rank] for f in range(3)]

    def ring_coords(pos, family):
        peer = tables_ref[family, 1, pos]
        return (peer,)

    @pl.when(step == 0)
    def enter():
        peers = [
            ring_coords((positions[f] + direction) % 32, jnp.int32(f))
            for f in range(3)
            for direction in (-1, 1)
        ]
        sem = pltpu.get_barrier_semaphore()
        for peer in peers:
            pl.semaphore_signal(
                sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(sem, len(peers))

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready):
            for peer in peers:
                pl.semaphore_signal(
                    ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(ready, len(peers))

    @pl.when(step == 0)
    def load_weight():
        pltpu.async_copy(w_ref, weights, weight_sem).wait()

    for stream, region in enumerate(streams):
        begin, size = region[:2]
        col_begin, width = region[2:] if len(region) == 4 else (0, o_ref.shape[1])
        cols = pl.ds(col_begin, width)
        family = (stream // 2) % 3
        direction = -1 if stream % 2 == 0 else 1
        position = positions[family]
        target = ring_coords((position + direction) % 32, jnp.int32(family))
        source = ring_coords((position - direction) % 32, jnp.int32(family))
        peer = tables_ref[family, 1, (position - direction * (step + 1)) % 32]
        owner = tables_ref[family, 2, peer]
        rows = pl.ds(begin, size)
        read = pltpu.make_async_copy(
            x_ref.at[
                pl.ds(owner * o_ref.shape[0] + tile * payload.shape[0] + begin, size), :
            ],
            inputs.at[stream],
            local_sems.at[stream],
        )
        read.start()
        reads.append(read)
        descriptors.append((target, source, rows, cols))

    for stream, (target, source, rows, cols) in enumerate(descriptors):

        def remote(index):
            return pltpu.make_async_remote_copy(
                payload.at[rows, cols],
                received.at[index % buffers, rows, cols],
                send_sems.at[stream],
                recv_sems.at[index % buffers, stream],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step > 0)
        def wait_before():
            remote(step - 1).wait()

        reads[stream].wait()
        local_buf.at[rows, cols][...] = jnp.dot(
            inputs[stream], weights[:, cols], preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)

        @pl.when(step == 0)
        def seed():
            payload.at[rows, cols][...] = local_buf.at[rows, cols][...].astype(
                payload.dtype
            )

        @pl.when(step > 0)
        def add():
            payload.at[rows, cols][...] = (
                local_buf.at[rows, cols][...].astype(payload.dtype)
                + received.at[(step - 1) % buffers, rows, cols][...]
            )
            pl.semaphore_signal(
                credits.at[stream],
                1,
                device_id=source,
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step < 31)
        def forward():
            @pl.when(step >= buffers)
            def acquire():
                pl.semaphore_wait(credits.at[stream], 1)

            remote(step).start()

        @pl.when(step == 31)
        def finish():
            pl.semaphore_wait(credits.at[stream], buffers)
            local_buf.at[rows, cols][...] = payload.at[rows, cols][...].astype(
                local_buf.dtype
            )
            output = pltpu.make_async_copy(
                local_buf.at[rows, cols],
                o_ref.at[pl.ds(tile * payload.shape[0] + rows.start, rows.size), cols],
                store_sem,
            )
            output.start()
            output.wait()


def project_reduce_scatter(x, w, mesh, tables, *, collective_id):
    """Fuse a three-head BF16 attention projection with SP32 RS."""
    if (
        x.shape[0] != 32
        or x.shape[1] % 32
        or x.shape[2] != 384
        or w.shape != (32, 384, 7168)
    ):
        raise ValueError("K3 projection/RS requires SP32 and 3 local heads")
    if x.dtype != jnp.bfloat16 or w.dtype != jnp.bfloat16:
        raise ValueError("K3 projection/RS requires BF16 operands")
    tokens = x.shape[1]
    rows = tokens // 32
    if rows % 64 or rows > 256:
        projected = jax.shard_map(
            lambda a, b: a @ b,
            mesh=mesh,
            in_specs=(P("k3_sp", None, None),) * 2,
            out_specs=P("k3_sp", None, None),
            check_vma=False,
        )(x, w)
        return ring_collective(
            projected, mesh, tables, gather=False, collective_id=collective_id
        )
    regions = _regions(rows, 7168)
    streams = len(regions)
    chunk = rows // 4

    def local(x, w):
        x = x.reshape(tokens, 384)
        w = w.reshape(384, 7168)
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        scratch = (
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((streams,)),
            pltpu.SemaphoreType.DMA((2, streams)),
            pltpu.SemaphoreType.REGULAR((streams,)),
            pltpu.SemaphoreType.DMA,
            pltpu.VMEM((rows, 7168), jnp.bfloat16),
            pltpu.VMEM((rows, 7168), jnp.bfloat16),
            pltpu.VMEM((2, rows, 7168), jnp.bfloat16),
            pltpu.VMEM((384, 7168), jnp.bfloat16),
            pltpu.VMEM((streams, chunk, 384), jnp.bfloat16),
            pltpu.SemaphoreType.DMA,
        )
        return pl.pallas_call(
            functools.partial(_project_ring_kernel, streams=regions, buffers=2),
            out_shape=jax.ShapeDtypeStruct((rows, 7168), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=(hbm, hbm),
                out_specs=hbm,
                scratch_shapes=scratch,
                grid=(1, 32),
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id, vmem_limit_bytes=40 * 1024**2
            ),
            name="kimi_attention_project_ring_rs",
        )(jnp.asarray(tables, jnp.int32), x, w)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P("k3_sp", None, None), P("k3_sp", None, None)),
        out_specs=P("k3_sp", None),
        check_vma=False,
    )(x, w)


def _gather_project_kernel(
    tables,
    x,
    w,
    out,
    load_sems,
    send_sems,
    recv_sems,
    credits,
    payload,
    received,
    weights,
    acc,
    output_buffers,
    wsem,
    output_sems,
    *,
    regions,
):
    step = pl.program_id(1)
    tile = pl.program_id(0)
    rank = lax.axis_index("k3_sp")
    positions = [tables[f, 0, rank] for f in range(3)]

    def owner(pos, f):
        return tables[f, 2, tables[f, 1, pos % 32]]

    def peer(pos, f):
        return (tables[f, 1, pos % 32],)

    @pl.when(step == 0)
    def enter():
        peers = [
            peer(positions[f] + d, jnp.int32(f)) for f in range(3) for d in (-1, 1)
        ]
        barrier = pltpu.get_barrier_semaphore()
        for p in peers:
            pl.semaphore_signal(
                barrier, 1, device_id=p, device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, len(peers))

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready):
            for p in peers:
                pl.semaphore_signal(
                    ready, 1, device_id=p, device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(ready, len(peers))

        @pl.when(tile == 0)
        def load_weight():
            pltpu.make_async_copy(w, weights, wsem).start()

        acc[...] = jnp.zeros(acc.shape, jnp.float32)
        for s, (begin, size, col, width) in enumerate(regions):
            pltpu.make_async_copy(
                x.at[pl.ds(tile * 128 + begin, size), pl.ds(col, width)],
                payload.at[pl.ds(begin, size), pl.ds(col, width)],
                load_sems.at[s],
            ).start()

    def remote(s, index):
        begin, size, col, width = regions[s]
        f = (s // 2) % 3
        direction = -1 if s % 2 == 0 else 1
        return pltpu.make_async_remote_copy(
            payload.at[pl.ds(begin, size), pl.ds(col, width)],
            received.at[index % 2, pl.ds(begin, size), pl.ds(col, width)],
            send_sems.at[s],
            recv_sems.at[index % 2, s],
            device_id=peer(positions[f] + direction, jnp.int32(f)),
            device_id_type=pl.DeviceIdType.MESH,
        )

    for s, (begin, size, col, width) in enumerate(regions):
        rows, cols = pl.ds(begin, size), pl.ds(col, width)

        @pl.when(step == 0)
        def seed():
            pltpu.make_async_copy(
                x.at[pl.ds(tile * 128 + begin, size), cols],
                payload.at[rows, cols],
                load_sems.at[s],
            ).wait()

        @pl.when(step > 0)
        def receive():
            remote(s, step - 1).wait()
            payload.at[rows, cols][...] = received.at[(step - 1) % 2, rows, cols][...]
            f = (s // 2) % 3
            direction = -1 if s % 2 == 0 else 1
            pl.semaphore_signal(
                credits.at[s],
                1,
                device_id=peer(positions[f] - direction, jnp.int32(f)),
                device_id_type=pl.DeviceIdType.MESH,
            )

        @pl.when(step < 31)
        def forward():
            @pl.when(step >= 2)
            def acquire():
                pl.semaphore_wait(credits.at[s], 1)

            remote(s, step).start()

    @pl.when((step == 0) & (tile == 0))
    def weights_ready():
        pltpu.make_async_copy(w, weights, wsem).wait()

    for pair in range(len(regions) // 2):
        s = pair * 2
        begin, size, col, width = regions[s]
        next_begin = regions[s + 1][0]
        lhs = jnp.concatenate(
            (
                payload[pl.ds(begin, size), pl.ds(col, width)],
                payload[pl.ds(next_begin, size), pl.ds(col, width)],
            ),
            axis=0,
        )
        value = jnp.dot(
            lhs, weights[pl.ds(col, width), :], preferred_element_type=jnp.float32
        )
        family = pair % 3
        for offset in range(2):
            direction = -1 if offset == 0 else 1
            dest = owner(positions[family] - direction * step, jnp.int32(family))
            row_begin = regions[s + offset][0]
            rows = pl.ds(dest * 128 + row_begin, size)
            acc[rows, :] = acc[rows, :] + value[offset * size : (offset + 1) * size, :]

    @pl.when(step == 31)
    def finish():
        for s in range(len(regions)):
            pl.semaphore_wait(credits.at[s], 2)

        def copy(owner_id):
            return pltpu.make_async_copy(
                output_buffers.at[owner_id % 2],
                out.at[pl.ds(owner_id * x.shape[0] + tile * 128, 128), :],
                output_sems.at[owner_id % 2],
            )

        for owner_id in range(32):
            if owner_id >= 2:
                copy(owner_id - 2).wait()
            output_buffers[owner_id % 2, ...] = acc[
                pl.ds(owner_id * 128, 128), :
            ].astype(jnp.bfloat16)
            copy(owner_id).start()
        copy(30).wait()
        copy(31).wait()


def _gather_project(x, w, tables, collective_id):
    assert x.shape[0] % 128 == 0 and x.shape[1] == 7168 and w.shape == (7168, 1792)
    streams = 6
    regions = tuple(
        (direction * 64, 64, col, width)
        for col, width in [(0, 2432), (2432, 2432), (4864, 2304)]
        for direction in (0, 1)
    )
    scratch = (
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((2, streams)),
        pltpu.SemaphoreType.REGULAR((streams,)),
        pltpu.VMEM((128, 7168), jnp.bfloat16),
        pltpu.VMEM((2, 128, 7168), jnp.bfloat16),
        pltpu.VMEM((7168, 1792), jnp.bfloat16),
        pltpu.VMEM((4096, 1792), jnp.float32),
        pltpu.VMEM((2, 128, 1792), jnp.bfloat16),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((2,)),
    )
    hb = pl.BlockSpec(memory_space=pltpu.HBM)
    partials = pl.pallas_call(
        functools.partial(_gather_project_kernel, regions=regions),
        out_shape=jax.ShapeDtypeStruct((32 * x.shape[0], 1792), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(hb, hb),
            out_specs=hb,
            scratch_shapes=scratch,
            grid=(x.shape[0] // 128, 32),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=63 * 1024**2,
            dimension_semantics=("arbitrary", "arbitrary"),
        ),
        name="kda_gather_project_vmem_sum",
    )(jnp.asarray(tables, jnp.int32), x, w)
    return partials


def gather_project(x, w, mesh, tables, *, collective_id):
    """Gather SP32 hidden states while projecting into packed KDA inputs."""

    def local(x, w):
        if x.shape[0] % 128:
            owners = tuple(int(i) for i in tables[0, 2])
            order = tuple(sorted(range(32), key=lambda i: owners[i]))
            gathered = lax.all_gather(
                x, "k3_sp", axis=0, tiled=True, axis_index_groups=(order,)
            )
            return (gathered @ w[0])[None]
        return _gather_project(x, w[0], tables, collective_id)[None]

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P("k3_sp", None), P("k3_sp", None, None)),
        out_specs=P("k3_sp", None, None),
        check_vma=False,
    )(x, w)
