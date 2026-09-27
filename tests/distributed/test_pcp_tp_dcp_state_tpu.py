# SPDX-License-Identifier: Apache-2.0
"""Eight-device GDN state seam check. Pallas writes/reads real TPU pools;
span-directed byte movement between them is emulated on the host.
This complements, rather than replaces, the native Raiden DMA test.
"""

import os

import pytest

from .tpu_test_utils import run_in_isolated_process


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
