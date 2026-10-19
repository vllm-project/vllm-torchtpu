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
"""Eight-device GDN state seam check. Pallas writes/reads real TPU pools;
span-directed byte movement between them is emulated on the host.
This complements, rather than replaces, the native Raiden DMA test.
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
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol
    from vllm_torchtpu.kernels.gdn.v3.vmem_ldst import (
        load_state_region,
        store_state_region,
    )
    from vllm_torchtpu.layers.core.gdn_attention import _build_v3_pool_state_plan

    from .test_pcp_tp_dcp_connector import _manifest
    from .test_raiden_fa_layout_calibration_tpu import _apply_xla_tiled_layout

    mesh = Mesh(np.array(jax.devices()), ("rank",))
    if mesh.size != 8:
        pytest.skip("requires eight local TPU devices")
    minor = (4, 3, 2, 1, 0)
    tiles = ((4, 128), (4, 1))
    ssm = np.arange(8 * 512 * 128).reshape(8, 512, 128) % 61 + 1
    conv = np.arange(8 * 18 * 128).reshape(8, 18, 128) % 61 + 1
    for src_heads in (1, 2):
        pools = {}
        plans = {}
        for heads in set((src_heads, 1)):
            shape = (3, 2 * heads, 64, 4, 128)
            plan = _build_v3_pool_state_plan(
                jax.ShapeDtypeStruct(shape, jnp.float8_e4m3fn),
                conv_dim=768,
                n_v=4,
                d_k=128,
                d_v=128,
                kernel_size=4,
                pool_block_tokens=384,
                qk_pair_layout=True,
                recurrent_state_dtype=jnp.bfloat16,
            )
            plans[heads] = plan

            def write(s, c, o, *, shape=shape, plan=plan):
                o[...] = jnp.zeros(shape, jnp.float8_e4m3fn)
                for region, values in ((plan.recurrent, s[...]), (plan.conv, c[...])):
                    store_state_region(
                        o.at[pl.ds(region.kb0, region.nblocks)], region, values
                    )

            call = pl.pallas_call(
                write,
                out_shape=jax.ShapeDtypeStruct(shape, jnp.float8_e4m3fn),
                compiler_params=pltpu.CompilerParams(),
            )

            def local_write(s, c, *, call=call):
                return call(s[0], c[0])[None]

            fn = jax.jit(
                jax.shard_map(
                    local_write,
                    mesh=mesh,
                    in_specs=(P("rank"), P("rank")),
                    out_specs=P("rank"),
                    check_vma=False,
                )
            )
            pools[heads] = np.asarray(
                fn(
                    jax.device_put(
                        jnp.asarray(ssm, dtype=jnp.bfloat16),
                        NamedSharding(mesh, P("rank")),
                    ),
                    jax.device_put(
                        jnp.asarray(conv, dtype=jnp.bfloat16),
                        NamedSharding(mesh, P("rank")),
                    ),
                )
            )
        shape = (3, 2, 64, 4, 128)
        # Source worker p*TP+t stores GDN head rank t*PCP+p.
        raw_dst = np.zeros((8, np.prod(shape)), dtype=np.uint8)
        for rank in range(8):
            p, t = divmod(rank, 2)
            g = t * 4 + p
            raw_src = _apply_xla_tiled_layout(
                pools[src_heads][g].view(np.uint8), minor, tiles
            )
            for pool in _manifest(src_heads).pools[1:]:
                reg = sol.lower_gdn_state_shard_spans_sol(
                    tag=pool.tag,
                    block_id=0,
                    transfer_rank=p,
                    parallelism=4,
                    producer_tp_rank=t,
                    producer_tp_size=2,
                    dst_shards=8,
                    regions=pool.regions,
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
        plan = plans[1]

        def read(i, s, c, *, plan=plan):
            s[...] = load_state_region(
                i.at[pl.ds(plan.recurrent.kb0, plan.recurrent.nblocks)],
                plan.recurrent,
                (512, 128),
            )
            c[...] = load_state_region(
                i.at[pl.ds(plan.conv.kb0, plan.conv.nblocks)], plan.conv, (18, 128)
            )

        call = pl.pallas_call(
            read,
            out_shape=(
                jax.ShapeDtypeStruct((512, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((18, 128), jnp.bfloat16),
            ),
        )

        def local_read(x, *, call=call):
            s, c = call(x[0])
            return s[None], c[None]

        fn = jax.jit(
            jax.shard_map(
                local_read,
                mesh=mesh,
                in_specs=P("rank"),
                out_specs=(P("rank"), P("rank")),
                check_vma=False,
            )
        )
        actual_s, actual_c = fn(jax.device_put(logical, NamedSharding(mesh, P("rank"))))
        np.testing.assert_array_equal(np.asarray(actual_s), ssm)
        np.testing.assert_array_equal(np.asarray(actual_c), conv)
        print("PASS GDN BF16 SPMD", src_heads, "-> 1", flush=True)


def test_pcp_tp_dcp_bf16_state_seam_spmd():
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
    os.environ["TPU_GDN_CONV_QK_PAIR_LAYOUT"] = "1"
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
    from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol
    from vllm_torchtpu.kernels.gdn.v3.vmem_ldst import (
        load_state_region,
        store_state_region,
    )
    from vllm_torchtpu.layers.core.gdn_attention import _build_v3_pool_state_plan

    from .test_raiden_fa_layout_calibration_tpu import _apply_xla_tiled_layout

    minor = (4, 3, 2, 1, 0)
    tiles = ((4, 128), (4, 1))
    d = 128
    lane_bytes = sol.sol_gdn_conv_lane_bytes(sol.SolFaGeometry(2, 64, 128))

    def side(keys, values, degree, fa_heads):
        key_heads, value_heads = _gdn_heads(keys, values, degree, 0)
        lk, lv = len(key_heads), len(value_heads)
        conv_dim = (2 * lk + lv) * d
        shape = (3, 2 * fa_heads, 64, 4, 128)
        plan = _build_v3_pool_state_plan(
            jax.ShapeDtypeStruct(shape, jnp.float8_e4m3fn),
            conv_dim=conv_dim,
            n_v=lv,
            d_k=d,
            d_v=d,
            kernel_size=4,
            pool_block_tokens=384,
            qk_pair_layout=True,
            recurrent_state_dtype=jnp.bfloat16,
        )
        gdn = rpm.GdnHeadGeometry(
            local_key_heads=lk, local_value_heads=lv, key_head_dim=d, value_head_dim=d
        )
        regions = {
            "gdn.conv.g0": rpm._gdn_conv_regions(
                conv_shape=(1, 3, conv_dim), itemsize=2, geometry=gdn
            ),
            "gdn.ssm.g0": rpm._gdn_ssm_regions(ssm_shape=(1, lv, d, d), itemsize=2),
        }
        bases = {"gdn.conv.g0": lv * d * d * 2, "gdn.ssm.g0": 0}
        rows = (lv * d, 3 * conv_dim // 128)
        return shape, plan, regions, bases, rows

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
        src_shape, src_plan, regions, src_bases, src_rows = side(
            keys, values, src_degree, src_fa
        )
        dst_shape, dst_plan, _, dst_bases, dst_rows = side(
            keys, values, dst_degree, dst_fa
        )
        src_keys, src_values = _gdn_heads(keys, values, src_degree, rank)
        lk, lv = len(src_keys), len(src_values)
        assert (src_plan.conv.rows_perm is None) == (lk == 1)
        assert dst_plan.conv.rows_perm is None

        # Finite random BF16 bit patterns, compared bit-exactly.
        rng = np.random.default_rng(rank)
        ssm = rng.integers(0, 0x7F00, (src_rows[0], 128), dtype=np.uint16).view(
            jnp.bfloat16
        )
        conv = rng.integers(0, 0x7F00, (src_rows[1], 128), dtype=np.uint16).view(
            jnp.bfloat16
        )

        def write(s, c, o, *, shape=src_shape, plan=src_plan):
            o[...] = jnp.zeros(shape, jnp.float8_e4m3fn)
            # The GDN kernels store their float32 compute states.
            for region, state in ((plan.recurrent, s[...]), (plan.conv, c[...])):
                store_state_region(
                    o.at[pl.ds(region.kb0, region.nblocks)],
                    region,
                    state.astype(jnp.float32),
                )

        pool = np.asarray(
            jax.jit(
                pl.pallas_call(
                    write,
                    out_shape=jax.ShapeDtypeStruct(src_shape, jnp.float8_e4m3fn),
                    compiler_params=pltpu.CompilerParams(),
                )
            )(jnp.asarray(ssm), jnp.asarray(conv))
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
                conv_lane_bytes=lane_bytes,
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

        def read(i, s, c, *, plan=dst_plan, rows=dst_rows):
            s[...] = load_state_region(
                i.at[pl.ds(plan.recurrent.kb0, plan.recurrent.nblocks)],
                plan.recurrent,
                (rows[0], 128),
            )
            c[...] = load_state_region(
                i.at[pl.ds(plan.conv.kb0, plan.conv.nblocks)], plan.conv, (rows[1], 128)
            )

        reader = jax.jit(
            pl.pallas_call(
                read,
                out_shape=(
                    jax.ShapeDtypeStruct((dst_rows[0], 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((dst_rows[1], 128), jnp.bfloat16),
                ),
            )
        )
        # Logical source rows per tap: [Q heads, K heads, V heads], one
        # 128-lane row per head.
        tap_rows = src_rows[1] // 3
        assert tap_rows == 2 * lk + lv
        for j in range(fan_out):
            actual_s, actual_c = reader(jnp.asarray(logical[j]))
            dst_keys, dst_values = _gdn_heads(
                keys, values, dst_degree, rank * fan_out + j
            )
            key_rows = [src_keys.index(h) for h in dst_keys]
            value_rows = [src_values.index(h) for h in dst_values]
            expected_s = np.concatenate([ssm[v * d : (v + 1) * d] for v in value_rows])
            rows = (
                key_rows + [lk + k for k in key_rows] + [2 * lk + v for v in value_rows]
            )
            expected_c = np.concatenate(
                [conv[tap * tap_rows + np.array(rows)] for tap in range(3)]
            )
            np.testing.assert_array_equal(
                np.asarray(actual_s).view(np.uint16), expected_s.view(np.uint16)
            )
            np.testing.assert_array_equal(
                np.asarray(actual_c).view(np.uint16), expected_c.view(np.uint16)
            )
        print("PASS GDN BF16 fan-out", src_degree, "->", dst_degree, flush=True)


def test_pp_pcp_fanout_bf16_state_seam():
    run_in_isolated_process(_run_fanout_state_seam, timeout=300)
