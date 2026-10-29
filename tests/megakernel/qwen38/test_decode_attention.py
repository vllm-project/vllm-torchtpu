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
"""The decode megakernel's attention over vLLM's HND pool pages
(``megakernel.qwen38.kernels.paged_kv``) against dense attention, and the
column commit.

The attention runs compiled on one TPU host's eight cores (one CP8 group): it
reads pool pages through packed-word ref views, which the TPU interpreter does
not model. The commit runs interpreted on an eight-device CPU mesh
(XLA_FLAGS=--xla_force_host_platform_device_count=8).
"""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel.qwen38.kernels.attention_append import (  # noqa: E402
    attend_paged_append,
    exchange_scratch,
)
from vllm_torchtpu.megakernel.qwen38.kernels.attention_batched_append import (  # noqa: E402
    attend_paged_batched_append,
)
from vllm_torchtpu.megakernel.qwen38.kernels.paged_kv import (  # noqa: E402
    PoolPages,
    commit_columns,
)

CP, PAGE, DIM, LOCAL_HEADS = 8, 256, 256, 2
PAGES_PER_BLOCK, WIDTH = 2, 3  # six local pages per request and rank
FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
LAYOUT = SimpleNamespace(
    context_parallel=CP,
    kv_page_size=PAGE,
    config=SimpleNamespace(head_dim=DIM),
    fast=True,
    local_q_heads=LOCAL_HEADS,
    local_kv_heads=1,
)


def _mesh(platform):
    devices = (
        [d for d in jax.devices() if d.platform == platform]
        if platform == "tpu"
        else []
    )
    if platform == "cpu":
        devices = jax.devices("cpu")
    if len(devices) < CP:
        pytest.skip(f"requires eight {platform} devices")
    return Mesh(np.asarray(devices[:CP]), ("tp",))


def _pools(rng, batch, positions=None):
    """FP8 pools ``[rank, pages, 2, D / 4, 4, page]`` and each request's distinct
    blocks (the same ids on every rank, as DCP block tables are). With
    ``positions``, request ``b``'s tokens before ``positions[b]`` are random and
    every other slot holds an FP8 NaN pattern; otherwise every slot is random."""
    tables = 1 + np.arange(batch * WIDTH, dtype=np.int32).reshape(batch, WIDTH)
    pages = (1 + batch * WIDTH) * PAGES_PER_BLOCK
    shape = (CP, pages, 2, DIM // 4, 4, PAGE)
    if positions is None:
        return np.asarray(jnp.asarray(rng.standard_normal(shape), FP8)), tables
    pools = np.full(shape, 0x7F, np.uint8).view(np.asarray(jnp.zeros((), FP8)).dtype)
    for b, position in enumerate(positions):
        for p in range(position):
            rank, target, column = _slot(tables, b, p)
            view = pools[rank, target].reshape(2, DIM, PAGE)
            view[:, :, column] = np.asarray(
                jnp.asarray(rng.standard_normal((2, DIM)), FP8)
            )
            pools[rank, target] = view.reshape(shape[2:])
    return pools, tables


def _slot(tables, b, position):
    """(rank, kernel page, column) of request ``b``'s token ``position``."""
    page = position // PAGE
    rank, local = page % CP, page // CP
    target = (
        tables[b, local // PAGES_PER_BLOCK] * PAGES_PER_BLOCK + local % PAGES_PER_BLOCK
    )
    return rank, target, position % PAGE


def _token(pools, tables, b, position):
    """Request ``b``'s key and value at ``position`` (float32 ``[2, D]``)."""
    rank, target, column = _slot(tables, b, position)
    return np.asarray(pools[rank, target], np.float32).reshape(2, DIM, PAGE)[
        :, :, column
    ]


def _attend(mesh, query, key, value, pools, tables, positions):
    batch = positions.shape[0]
    inbox_names = tuple(exchange_scratch(LAYOUT, batch))

    def local(query, key, value, pool, tables, positions):
        def kernel(
            position_ref,
            tables_ref,
            query_ref,
            key_ref,
            value_ref,
            pool_ref,
            out_ref,
            *inbox_refs,
        ):
            rank = jax.lax.axis_index("tp")
            barrier = tpu.get_barrier_semaphore()
            for peer in range(1, CP):
                pl.semaphore_signal(
                    barrier,
                    1,
                    device_id=(rank ^ peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier, CP - 1)
            inbox = dict(zip(inbox_names, inbox_refs, strict=True))
            pages = PoolPages((pool_ref,), jnp.int32(0), tables_ref, 0, PAGES_PER_BLOCK)
            if batch > 1:
                out_ref[...] = attend_paged_batched_append(
                    query_ref[...],
                    key_ref[...],
                    value_ref[...],
                    pages,
                    None,
                    tuple(position_ref[b] for b in range(batch)),
                    rank,
                    LAYOUT,
                    (1.0, 1.0),
                    pl.DeviceIdType.MESH,
                    inbox,
                    buffers=3,
                )
            else:
                out_ref[0] = attend_paged_append(
                    query_ref[0],
                    key_ref[0],
                    value_ref[0],
                    pages,
                    None,
                    position_ref[0],
                    rank,
                    LAYOUT,
                    (1.0, 1.0),
                    pl.DeviceIdType.MESH,
                    inbox=inbox,
                    buffers=3,
                )

        vmem = pl.BlockSpec(memory_space=tpu.VMEM)
        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(query.shape[1:], BF16),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                in_specs=[vmem, vmem, vmem, pl.BlockSpec(memory_space=tpu.HBM)],
                out_specs=vmem,
                scratch_shapes=tuple(exchange_scratch(LAYOUT, batch).values()),
            ),
            compiler_params=tpu.CompilerParams(collective_id=0),
        )(positions, tables, query[0], key, value, pool[0])
        return out[None]

    sharded, replicated = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    out = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("tp"), P(), P(), P("tp"), P(), P()),
            out_specs=P("tp"),
            check_vma=False,
        )
    )(
        jax.device_put(query, sharded),
        jax.device_put(key, replicated),
        jax.device_put(value, replicated),
        jax.device_put(pools, sharded),
        jax.device_put(tables, replicated),
        jax.device_put(positions, replicated),
    )
    return np.asarray(out, np.float32)


# Ranks own zero to three blocks of a request; the current token lands in the
# first, a middle or the last page of its block.
@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "positions", [[5], [2200], [12287], [5, 12287], [6000, 260, 0, 4095]]
)
def test_paged_decode_attention_matches_dense_attention(positions):
    mesh = _mesh("tpu")
    positions = np.asarray(positions, np.int32)
    batch = positions.shape[0]
    rng = np.random.default_rng(int(positions.sum()))
    pools, tables = _pools(rng, batch, positions)
    query = rng.standard_normal((CP, batch, LOCAL_HEADS, DIM)).astype(BF16)
    key = rng.standard_normal((batch, DIM)).astype(BF16)
    value = rng.standard_normal((batch, DIM)).astype(BF16)
    out = _attend(mesh, query, key, value, pools, tables, positions)

    for b in range(batch):
        position = positions[b]
        tokens = np.stack(
            [_token(pools, tables, b, p) for p in range(position)]
            or [np.zeros((2, DIM))]
        )
        current = np.stack(
            [
                np.asarray(jnp.asarray(np.asarray(x[b], np.float32), FP8), np.float32)
                for x in (key, value)
            ]
        )
        visible = np.concatenate([tokens[:position], current[None]])
        keys, values = visible[:, 0], visible[:, 1]
        for rank in range(CP):
            heads = np.asarray(query[rank, b], np.float32)
            scores = heads @ keys.T / np.sqrt(DIM)
            weights = np.exp(scores - scores.max(axis=1, keepdims=True))
            expected = (weights / weights.sum(axis=1, keepdims=True)) @ values
            np.testing.assert_allclose(out[rank, b], expected, rtol=2e-2, atol=2e-2)


def test_commit_columns_writes_each_request_on_its_owner():
    mesh = _mesh("cpu")
    rng = np.random.default_rng(0)
    positions = np.asarray([5, 1100, -1, 6143], np.int32)
    batch, layers = positions.shape[0], 2
    pools, tables = _pools(rng, batch)
    pools = np.stack([pools] * layers)  # [layer, rank, ...]
    columns = np.asarray(jnp.asarray(rng.standard_normal((layers, batch, 2, DIM)), FP8))

    def local(columns, positions, tables, *pools):
        updated = commit_columns(
            tuple(p[0] for p in pools),
            columns,
            positions,
            tables,
            pages_per_block=PAGES_PER_BLOCK,
            interpret=True,
        )
        return tuple(p[None] for p in updated)

    sharded, replicated = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    updated = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(), P(), P()) + (P("tp"),) * layers,
            out_specs=(P("tp"),) * layers,
            check_vma=False,
        )
    )(
        jax.device_put(columns, replicated),
        jax.device_put(positions, replicated),
        jax.device_put(tables, replicated),
        *(jax.device_put(p, sharded) for p in pools),
    )
    for o in range(layers):
        expected = np.asarray(pools[o], np.float32).copy()
        for b, position in enumerate(positions):
            if position < 0:
                continue
            rank, target, column = _slot(tables, b, position)
            view = expected[rank, target].reshape(2, DIM, PAGE)
            view[:, :, column] = np.asarray(columns[o, b], np.float32)
            expected[rank, target] = view.reshape(expected.shape[2:])
        np.testing.assert_array_equal(np.asarray(updated[o], np.float32), expected)
