# SPDX-License-Identifier: Apache-2.0
"""Eight-device GDN state seam check. Each device holds one worker's unified
pool, written/read by the host-side pool layout helpers; span-directed byte
movement between them is emulated on the host. This complements, rather than
replaces, the native Raiden DMA test.
"""

import os

import pytest

from .tpu_test_utils import run_in_isolated_process

pytestmark = pytest.mark.tpu_test


def _run_state_seam():
    os.environ["VLLM_KV_CACHE_LAYOUT"] = "HND"
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from kernels.gdn.v3.gdn_pool_state_utils import read_states, write_states

    from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
    from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol

    from .test_pcp_tp_dcp_connector import _manifest
    from .test_raiden_fa_layout_calibration_tpu import _apply_xla_tiled_layout

    mesh = Mesh(np.array(jax.devices()), ("rank",))
    if mesh.size != 8:
        pytest.skip("requires eight local TPU devices")
    sharding = NamedSharding(mesh, P("rank"))
    minor = (4, 3, 2, 1, 0)
    tiles = ((4, 128), (4, 1))
    ssm = np.arange(8 * 512 * 128).reshape(8, 512, 128) % 61 + 1
    conv = np.arange(8 * 18 * 128).reshape(8, 18, 128) % 61 + 1
    # The pooled conv state is fp32, so its regions are twice the BF16
    # `_manifest` ones; the BF16 SSM regions and both pool bases still hold.
    conv_regions = (
        rpm.RegionSpec("gdn_conv_qk", 0, 3072, 1024, 3),
        rpm.RegionSpec("gdn_conv_v", 1024, 3072, 512, 3, 4),
    )
    idx = jnp.array([0])
    for src_heads in (1, 2):
        src_pools = [
            np.asarray(
                write_states(
                    jnp.zeros((3, 2 * src_heads, 64, 4, 128), jnp.float8_e4m3fn),
                    jnp.asarray(ssm[g].reshape(1, 4, 128, 128), jnp.bfloat16),
                    conv[g].reshape(1, 3, 768),
                    idx,
                    split=3,
                    recurrent_dtype=jnp.bfloat16,
                )
            )
            for g in range(8)
        ]
        # Worker g's pool lives on device g; the host gathers the raw pools.
        pools = np.asarray(jax.device_put(np.stack(src_pools), sharding))
        shape = (3, 2, 64, 4, 128)
        # Source worker p*TP+t stores GDN head rank t*PCP+p.
        raw_dst = np.zeros((8, np.prod(shape)), dtype=np.uint8)
        for rank in range(8):
            p, t = divmod(rank, 2)
            g = t * 4 + p
            raw_src = _apply_xla_tiled_layout(pools[g].view(np.uint8), minor, tiles)
            for pool in _manifest(src_heads).pools[1:]:
                reg = sol.lower_gdn_state_shard_spans_sol(
                    tag=pool.tag,
                    block_id=0,
                    transfer_rank=p,
                    parallelism=4,
                    producer_tp_rank=t,
                    producer_tp_size=2,
                    dst_shards=8,
                    regions=conv_regions if pool.tag == "gdn.conv.g0" else pool.regions,
                    physical_granule_bytes=512,
                )
                dst_base = next(
                    e.base_offset_bytes for e in _manifest(1).pools if e.tag == pool.tag
                )
                for span in reg.spans:
                    # span offsets are relative to their pool, not raw storage.
                    for i in range(span.count):
                        src = (
                            pool.base_offset_bytes
                            + span.src_offset_bytes
                            + i * span.src_stride_bytes
                        )
                        dst = (
                            dst_base + span.dst_offset_bytes + i * span.dst_stride_bytes
                        )
                        raw_dst[span.dst_unit_ordinal, dst : dst + span.size_bytes] = (
                            raw_src[src : src + span.size_bytes]
                        )
        index = _apply_xla_tiled_layout(
            np.arange(np.prod(shape)).reshape(shape), minor, tiles
        )
        logical = np.empty_like(raw_dst)
        logical[:, index] = raw_dst
        logical = logical.reshape((8,) + shape).view(jnp.float8_e4m3fn)
        # Decode shard d's pool lives on device d.
        logical = np.asarray(jax.device_put(logical, sharding))
        for d in range(8):
            actual_s, actual_c = read_states(
                logical[d],
                idx,
                split=3,
                recurrent_shape=(4, 128, 128),
                conv_shape=(3, 768),
                recurrent_dtype=jnp.bfloat16,
            )
            np.testing.assert_array_equal(
                np.asarray(actual_s, np.float32).reshape(512, 128), ssm[d]
            )
            np.testing.assert_array_equal(
                np.asarray(actual_c).reshape(18, 128), conv[d]
            )
        print("PASS GDN SPMD", src_heads, "-> 1", flush=True)


def test_pcp_tp_dcp_state_seam_spmd():
    run_in_isolated_process(_run_state_seam, timeout=120)
