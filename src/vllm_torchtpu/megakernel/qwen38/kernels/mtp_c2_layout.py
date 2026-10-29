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
"""Shared HBM cache layout and bounded hidden selection for optional C2/C4 MTP."""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import PartitionSpec as P

CACHE_TILE = 8


def cache_contract(state):
    return jax.tree.map(lambda value: tpu.with_memory_space_constraint(value, tpu.HBM), state)


def pack_cache(state):
    def pack(value):
        if value.dtype != jnp.bfloat16 or value.shape[-2] != 128:
            raise ValueError('Native BF16 cache pages required')
        paired = value.reshape(*value.shape[:-2], 64, 2, value.shape[-1]).swapaxes(-1, -2)
        return jax.lax.bitcast_convert_type(paired, jnp.uint32)
    return jax.tree.map(pack, state)


def unpack_cache(state):
    def unpack(value):
        bits = jax.lax.bitcast_convert_type(value, jnp.bfloat16)
        return bits.swapaxes(-1, -2).reshape(*value.shape[:-2], 128, value.shape[-1])
    return jax.tree.map(unpack, state)


def make_hidden_selector(mesh, *, interpret=False):
    def local(hidden, previous, count):
        hidden = tpu.with_memory_space_constraint(hidden, tpu.HBM)
        previous = tpu.with_memory_space_constraint(previous, tpu.HBM)
        requests, h = previous.shape

        def kernel(count_ref, hidden_ref, previous_ref, output_ref, tile, carry, sem):
            def copy(source, destination):
                dma = tpu.make_async_copy(source, destination, sem)
                dma.start()
                dma.wait()
            copy(hidden_ref, tile)
            copy(previous_ref, carry)
            for request in range(requests):
                for slot in range(4):
                    @pl.when(count_ref[request] == slot + 1)
                    def select(request=request, slot=slot):
                        carry[request, :] = tile[request, slot, :]
            copy(carry, output_ref)

        return pl.pallas_call(
            kernel, out_shape=tpu.HBM((requests, h), jnp.bfloat16),
            input_output_aliases={2: 0},
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * 2,
                out_specs=pl.BlockSpec(memory_space=tpu.HBM),
                scratch_shapes=(tpu.VMEM((requests, 4, h), jnp.bfloat16),
                                tpu.VMEM((requests, h), jnp.bfloat16), tpu.SemaphoreType.DMA)),
            compiler_params=tpu.CompilerParams(vmem_limit_bytes=64 << 20),
            interpret=interpret, name='qwen_mtp_c2_hidden_select',
        )(count, hidden, previous)

    def select(hidden, previous, count):
        return jax.shard_map(local, mesh=mesh, in_specs=(P(), P(), P()),
                             out_specs=P(), check_vma=False)(hidden, previous, count)
    return jax.jit(select)
