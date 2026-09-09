# SPDX-License-Identifier: Apache-2.0
"""Kimi Stage-3 declarations must reconstruct the state the KDA kernel reads.

The rank-blocked pool adapters write real TPU source pools. Raiden stages their
physical bytes, the declared spans reshard them, and Raiden uploads into a
destination pool. The adapters must read the original dense states.
This isolates the layout contract from the controller and model weights.
"""

import ctypes
from functools import partial

import numpy as np
import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import \
    lower_kda_state_shard_spans
from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout

from .raiden_test_utils import kimi_pool_manifest
from .test_pool_byte_spans import _copy_registration
from .test_raiden_pool_liveness_tpu import _extent_bytes, _live_extents
from .tpu_test_utils import run_in_isolated_process


def _run_state_transfer(parallelism, dst_shards):
    import jax
    import jax.numpy as jnp
    import torch
    from torch_tpu._internal import pallas, sync
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

    from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import \
        measured_kimi_k3_layout_fingerprint
    from vllm_torchtpu.kernels import pool_adapters

    src_heads = 96 // parallelism

    def ops(heads):
        layout = derive_pooled_gdn_state_layout(ssm_bytes=heads * 128 * 128 *
                                                4,
                                                conv_bytes=9 * heads * 128 * 2,
                                                token_bytes=2560)
        fragments = heads // src_heads
        fragment_elems = layout.conv_tokens * 1280 // fragments

        def seed(pool: jax.Array, conv: jax.Array,
                 ssm: jax.Array) -> jax.Array:
            ids = jnp.array([1], jnp.int32)
            parts = []
            for start in range(0, heads, src_heads):
                part = conv[:, :, start:start + src_heads].reshape(-1)
                parts.append(jnp.pad(part, (0, fragment_elems - part.size)))
            conv = jnp.concatenate(parts)
            pool = pool_adapters.scatter_region(pool,
                                                conv.reshape(1, -1, 640),
                                                ids,
                                                tok0=layout.ssm_tokens,
                                                ntok=layout.conv_tokens)
            ssm = jnp.pad(ssm.reshape(1, -1, 128),
                          ((0, 0), (0, layout.ssm_tokens * 5 - heads * 128),
                           (0, 0)))
            return pool_adapters.scatter_region(pool,
                                                ssm,
                                                ids,
                                                tok0=0,
                                                ntok=layout.ssm_tokens)

        def read(pool: jax.Array) -> tuple[jax.Array, jax.Array]:
            ids = jnp.array([1], jnp.int32)
            conv = pool_adapters.gather_region(pool,
                                               ids,
                                               tok0=layout.ssm_tokens,
                                               ntok=layout.conv_tokens,
                                               out_dtype=jnp.bfloat16).reshape(
                                                   fragments, fragment_elems)
            conv = jnp.concatenate([
                conv[i, :9 * src_heads * 128].reshape(3, 3, src_heads, 128)
                for i in range(fragments)
            ],
                                   axis=2)
            ssm = pool_adapters.gather_region(pool,
                                              ids,
                                              tok0=0,
                                              ntok=layout.ssm_tokens,
                                              out_dtype=jnp.float32,
                                              out_lanes=128)[:, :heads * 128]
            return conv.reshape(3, 3, heads, 128), ssm.reshape(heads, 128, 128)

        return (pallas.jax_op(f"pallas::stage3_kda_seed_{heads}",
                              seed,
                              donate_argnums=(0, )),
                pallas.jax_op(f"pallas::stage3_kda_read_{heads}",
                              read), layout)

    fan_in = parallelism // dst_shards
    dst_heads = src_heads * fan_in
    src_seed, src_read, src_layout = ops(src_heads)
    if fan_in == 1:
        dst_read = src_read
        dst_layout = src_layout
    else:
        _, dst_read, dst_layout = ops(dst_heads)
    rng = np.random.default_rng(72)
    conv = torch.from_numpy(
        rng.normal(size=(3, 3, dst_heads,
                         128)).astype(np.float32)).to(torch.bfloat16)
    ssm = torch.from_numpy(
        rng.normal(size=(dst_heads, 128, 128)).astype(np.float32))

    def manager(pool, heads):
        sync.synchronize([pool], wait=True)
        manifest = kimi_pool_manifest(heads, pool)
        measured_kimi_k3_layout_fingerprint(manifest,
                                            page_tokens=pool.shape[1] * 2,
                                            state_fragment_heads=src_heads,
                                            state_fragments=heads // src_heads)
        engine = KVCacheManager(kv_caches=[pool],
                                node_id=0,
                                local_control_port=-1,
                                max_blocks=1,
                                num_slots=1,
                                unsafe_skip_buffer_lock=True)
        engine.register_pools(manifest.pool_dicts())
        return engine, manifest

    destination = torch.zeros((4, dst_layout.required_tokens, 2, 640),
                              dtype=torch.bfloat16,
                              device="tpu")
    dst_engine, dst_manifest = manager(destination, dst_heads)
    compact_destinations = [
        bytearray(p.live_bytes_per_block) for p in dst_manifest.pools[:2]
    ]
    sources = []  # Keep admitted source buffers/managers alive through upload.
    for rank in range(fan_in):
        pool = torch.zeros((4, src_layout.required_tokens, 2, 640),
                           dtype=torch.bfloat16,
                           device="tpu")
        head_slice = slice(rank * src_heads, (rank + 1) * src_heads)
        pool = src_seed(pool, conv[:, :, head_slice].contiguous().to("tpu"),
                        ssm[head_slice].contiguous().to("tpu"))
        engine, manifest = manager(pool, src_heads)
        sources.append((pool, engine))
        for index, entry in enumerate(manifest.pools[:2]):
            engine.d2h_pool_blocks(index, [1]).wait()
            ref = engine.get_block_ref(index, 1)
            extent = max(r.extent_end_bytes for r in entry.regions)
            raw = ctypes.string_at(ref["ptr"], extent)
            registration = lower_kda_state_shard_spans(tag=entry.tag,
                                                       block_id=1,
                                                       transfer_rank=rank,
                                                       parallelism=parallelism,
                                                       dst_shards=dst_shards,
                                                       regions=entry.regions)
            assert len(registration.spans) == 1
            assert registration.spans[0].count == 1
            _copy_registration(compact_destinations[index],
                               _extent_bytes(raw, _live_extents(entry)),
                               registration)
    for index, entry in enumerate(dst_manifest.pools[:2]):
        ref = dst_engine.get_block_ref(index, 1)
        cursor = 0
        for offset, size in _live_extents(entry):
            ctypes.memmove(
                ref["ptr"] + offset,
                bytes(compact_destinations[index][cursor:cursor + size]), size)
            cursor += size
        dst_engine.h2d_pool_blocks(index, [1]).wait()

    actual_conv, actual_ssm = dst_read(destination)
    np.testing.assert_array_equal(actual_conv.cpu().view(torch.uint16).numpy(),
                                  conv.view(torch.uint16).numpy())
    np.testing.assert_array_equal(actual_ssm.cpu().numpy(), ssm.numpy())


@pytest.mark.parametrize(
    "parallelism,dst_shards", [(8, 8), (8, 4), (32, 32), (32, 8)],
    ids=["tp8-to-tp8", "tp8-to-tp4", "tp32-to-tp32", "tp32-to-tp8"])
def test_kimi_stage3_state_transfer_tpu(parallelism, dst_shards):
    run_in_isolated_process(
        partial(_run_state_transfer, parallelism, dst_shards))
