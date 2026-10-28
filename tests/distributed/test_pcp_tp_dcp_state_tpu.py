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


def _gdn_heads(keys, values, degree, rank):
    """Key and value heads of a GDN head-axis rank: value heads shard
    contiguously, key heads shard or replicate across adjacent ranks."""
    width = values // degree
    value_heads = list(range(rank * width, (rank + 1) * width))
    if keys >= degree:
        key_width = keys // degree
        return list(range(rank * key_width, (rank + 1) * key_width)), value_heads
    return [rank // (degree // keys)], value_heads


def _run_fanout_state_seam():
    os.environ["VLLM_KV_CACHE_LAYOUT"] = "HND"
    import jax.numpy as jnp
    import numpy as np
    from kernels.gdn.v3.gdn_pool_state_utils import read_states, write_states

    from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
    from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol

    from .test_raiden_fa_layout_calibration_tpu import _apply_xla_tiled_layout

    minor = (4, 3, 2, 1, 0)
    tiles = ((4, 128), (4, 1))
    d = 128
    taps = 3
    idx = jnp.array([0])

    def side(keys, values, degree, fa_heads):
        key_heads, value_heads = _gdn_heads(keys, values, degree, 0)
        lk, lv = len(key_heads), len(value_heads)
        conv_dim = (2 * lk + lv) * d
        # One state spans the pool's three blocks (`split=3`).
        shape = (3, 2 * fa_heads, 64, 4, 128)
        gdn = rpm.GdnHeadGeometry(
            local_key_heads=lk, local_value_heads=lv, key_head_dim=d, value_head_dim=d
        )
        # The unified pool keeps the conv state in fp32, segment-major, right
        # after the BF16 SSM state.
        regions = {
            "gdn.conv.g0": rpm._gdn_conv_regions(
                conv_shape=(1, taps, conv_dim), itemsize=4, geometry=gdn
            ),
            "gdn.ssm.g0": rpm._gdn_ssm_regions(ssm_shape=(1, lv, d, d), itemsize=2),
        }
        bases = {"gdn.conv.g0": lv * d * d * 2, "gdn.ssm.g0": 0}
        return shape, regions, bases, lk, lv, conv_dim

    # (key heads, value heads, producer degree, decode degree, producer FA
    # heads, decode FA heads, producer rank): a PCP8 stage rank with 16 key /
    # 128 value heads (K2/V16) feeding four TP32 ranks (K1/V4), and one with
    # 8 key / 64 value heads (K1/V8) feeding two TP16 ranks (K1/V4); both add
    # Q/K replicas.
    for keys, values, src_degree, dst_degree, src_fa, dst_fa, rank in (
        (16, 128, 8, 32, 4, 1, 5),
        (8, 64, 8, 16, 2, 1, 3),
    ):
        fan_out = dst_degree // src_degree
        src_shape, regions, src_bases, lk, lv, src_conv_dim = side(
            keys, values, src_degree, src_fa
        )
        dst_shape, _, dst_bases, dst_lk, dst_lv, dst_conv_dim = side(
            keys, values, dst_degree, dst_fa
        )
        src_keys, src_values = _gdn_heads(keys, values, src_degree, rank)

        # Finite random BF16 SSM and fp32 conv bit patterns, compared
        # bit-exactly.
        rng = np.random.default_rng(rank)
        ssm = rng.integers(0, 0x7F00, (lv, d, d), dtype=np.uint16).view(jnp.bfloat16)
        # Normal fp32 values only, so no denormal is flushed on the way.
        conv = rng.integers(
            0x00800000, 0x7F000000, (taps, src_conv_dim), dtype=np.uint32
        ).view(np.float32)
        pool = np.asarray(
            write_states(
                jnp.zeros(src_shape, jnp.float8_e4m3fn),
                jnp.asarray(ssm)[None],
                conv[None],
                idx,
                split=3,
                recurrent_dtype=jnp.bfloat16,
            )
        )
        raw_src = _apply_xla_tiled_layout(pool.view(np.uint8), minor, tiles)
        raw_dst = np.zeros((fan_out, np.prod(dst_shape)), dtype=np.uint8)
        for tag, tag_regions in regions.items():
            reg = sol.lower_gdn_state_shard_spans_sol(
                tag=tag,
                block_id=0,
                transfer_rank=rank,
                parallelism=src_degree,
                regions=tag_regions,
                dst_shards=dst_degree,
                physical_granule_bytes=512,
                num_key_heads=keys,
                num_value_heads=values,
            )
            for span in reg.spans:
                local = span.dst_unit_ordinal - rank * fan_out
                assert 0 <= local < fan_out
                for i in range(span.count):
                    src = src_bases[tag] + span.src_offset_bytes
                    src += i * span.src_stride_bytes
                    dst = dst_bases[tag] + span.dst_offset_bytes
                    dst += i * span.dst_stride_bytes
                    raw_dst[local, dst : dst + span.size_bytes] = raw_src[
                        src : src + span.size_bytes
                    ]
        index = _apply_xla_tiled_layout(
            np.arange(np.prod(dst_shape)).reshape(dst_shape), minor, tiles
        )
        logical = np.empty_like(raw_dst)
        logical[:, index] = raw_dst
        logical = logical.reshape((fan_out,) + dst_shape).view(jnp.float8_e4m3fn)

        # Logical source rows per tap: [Q heads, K heads, V heads], one
        # 128-lane row per head.
        conv_rows = conv.reshape(taps, 2 * lk + lv, d)
        for j in range(fan_out):
            actual_s, actual_c = read_states(
                jnp.asarray(logical[j]),
                idx,
                split=3,
                recurrent_shape=(dst_lv, d, d),
                conv_shape=(taps, dst_conv_dim),
                recurrent_dtype=jnp.bfloat16,
            )
            dst_keys, dst_values = _gdn_heads(
                keys, values, dst_degree, rank * fan_out + j
            )
            key_rows = [src_keys.index(h) for h in dst_keys]
            value_rows = [src_values.index(h) for h in dst_values]
            assert len(key_rows) == dst_lk and len(value_rows) == dst_lv
            expected_s = ssm[value_rows]
            rows = (
                key_rows + [lk + k for k in key_rows] + [2 * lk + v for v in value_rows]
            )
            expected_c = conv_rows[:, rows].reshape(taps, dst_conv_dim)
            np.testing.assert_array_equal(
                np.asarray(actual_s).view(np.uint16).reshape(expected_s.shape),
                expected_s.view(np.uint16),
            )
            np.testing.assert_array_equal(
                np.asarray(actual_c).view(np.uint32).reshape(expected_c.shape),
                expected_c.view(np.uint32),
            )
        print("PASS GDN fan-out", src_degree, "->", dst_degree, flush=True)


def test_pp_pcp_fanout_bf16_state_seam():
    run_in_isolated_process(_run_fanout_state_seam, timeout=300)
