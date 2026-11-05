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
"""1-host TPU v7x (8-core) microbenchmark comparing baseline DCP LSE combine
paths (``cp_lse_ag_out_rs`` in bf16/f32 and packed ``all_to_all``) against
``dcp_online_lse_combine`` (``mode='pairs'`` and ``mode='flat'``), both
standalone and VMEM-chained into ``fp8_uv_matmul_all_reduce`` across
``T = 16, 32, 64, 80, 96``.

Checks numerical accuracy (``rel_l2``, ``cos``, ``max_abs``, dead-row zeroing)
against the IEEE float64 reference on every path and prints ``RESULT {...}``
JSON lines.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOKEN_BUCKETS = (16, 32, 64, 80, 96)
LAYERS = 32
REPS = 20
HIDDEN = 6144
TOLERANCE = {"rel_l2": 0.015, "cos": 0.9998, "max_abs": 0.125}


def _baseline_rs_local(o, lse, *, axis_name: str, wire_bf16: bool):
    """Exact JAX ``shard_map`` body of ``cp_mla_attention.cp_lse_ag_out_rs``."""
    import jax
    import jax.numpy as jnp

    lse_f32 = lse.astype(jnp.float32)
    lses = jax.lax.all_gather(lse_f32, axis_name, axis=0)
    merged_lse = jax.nn.logsumexp(lses, axis=0)
    weight = jnp.exp(lse_f32 - merged_lse)[..., None]
    weighted = o.astype(jnp.float32) * weight
    weighted = jnp.where(weight > 0, weighted, jnp.zeros_like(weighted))
    if wire_bf16:
        weighted = weighted.astype(jnp.bfloat16)
    out = jax.lax.psum_scatter(weighted, axis_name, scatter_dimension=1, tiled=True)
    return out.astype(jnp.bfloat16)


def _baseline_a2a_packed_local(o, lse, *, world: int, axis_name: str):
    """Exact JAX ``shard_map`` body of packed ``merge_lse_partials_all_to_all``."""
    import jax
    import jax.numpy as jnp

    tokens, heads, vdim = o.shape
    local = heads // world
    rows = tokens * local
    words = rows * 2
    extra = -(-words // vdim)
    o_send = o.reshape(tokens, world, local, vdim).transpose(1, 0, 2, 3)
    l_send = lse.astype(jnp.float32).reshape(tokens, world, local).transpose(1, 0, 2)
    o_words = jax.lax.bitcast_convert_type(o_send, jnp.int16).reshape(world, rows, vdim)
    bits = jax.lax.bitcast_convert_type(l_send, jnp.int32).reshape(world, rows)
    hi = (bits >> 16).astype(jnp.int16)
    lo = (((bits & 0xFFFF) ^ 0x8000) - 0x8000).astype(jnp.int16)
    l_words = jnp.pad(
        jnp.concatenate([hi, lo], axis=1), ((0, 0), (0, extra * vdim - words))
    )
    send = jnp.concatenate([o_words, l_words.reshape(world, extra, vdim)], axis=1)
    recv = jax.lax.all_to_all(send, axis_name, 0, 0, tiled=True)
    o_recv = jax.lax.bitcast_convert_type(recv[:, :rows], jnp.bfloat16).reshape(
        world, tokens, local, vdim
    )
    lw = recv[:, rows:].reshape(world, -1)
    r_hi = lw[:, :rows].astype(jnp.int32)
    r_lo = lw[:, rows : 2 * rows].astype(jnp.int32)
    l_recv = jax.lax.bitcast_convert_type(
        (r_hi << 16) | (r_lo & 0xFFFF), jnp.float32
    ).reshape(world, tokens, local)

    finite = jnp.isfinite(l_recv)
    shift = jnp.where(finite, l_recv, -jnp.inf).max(axis=0)
    shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
    weights = jnp.where(finite, jnp.exp(l_recv - shift[None]), 0.0)
    denom = weights.sum(axis=0)
    weights = weights / jnp.where(denom > 0, denom, 1.0)
    of_raw = o_recv.astype(jnp.float32)
    of = jnp.where(jnp.isnan(of_raw), 0.0, of_raw)
    acc = of[0] * weights[0][..., None]
    for r in range(1, world):
        acc = acc + of[r] * weights[r][..., None]
    return acc.astype(jnp.bfloat16)


def _accuracy_metrics(got_bf16, ref_f32, live: int) -> dict:
    import numpy as np

    got_live = got_bf16[:, :live].astype(np.float64)
    ref_live = ref_f32[:, :live].astype(np.float64)
    diff = got_live - ref_live
    denom = float(np.linalg.norm(ref_live))
    rel_l2 = float(np.linalg.norm(diff) / denom) if denom > 0 else 0.0
    cos = (
        float(np.vdot(got_live, ref_live) / (np.linalg.norm(got_live) * denom))
        if denom > 0
        else 1.0
    )
    max_abs = float(np.max(np.abs(diff))) if got_live.size else 0.0
    dead = got_bf16[:, live:].astype(np.float32)
    dead_zero = bool(dead.size == 0 or np.all((dead == 0.0) & ~np.signbit(dead)))
    return {
        "rel_l2": round(rel_l2, 6),
        "cos": round(cos, 6),
        "max_abs": round(max_abs, 6),
        "nan": int(np.isnan(got_bf16.astype(np.float32)).sum()),
        "dead_positive_zero": dead_zero,
    }


def _project_dcp16_2host_us(
    dcp8_us: float,
    *,
    tokens: int,
    local_dcp16: int,
    vdim: int,
    mode: str,
) -> float:
    """Analytical `est.` projection from measured 1-host DCP8 latency to 2-host
    DCP16 (`W=16`, `C=8` chips across 2 hosts, inter-host ICI hop ~1.6 us est.,
    per-link ICI unidirectional bandwidth ~180 GB/s est.)."""
    block_bytes = (16 + tokens * local_dcp16) * vdim * 2
    if mode == "pairs":
        # DCP8 pairs sends 3 remote blocks; DCP16 pairs sends 7 remote blocks
        # (including 4 inter-host blocks over ICI) + 4 extra VPU slot terms.
        extra_bw_us = (4 * block_bytes) / (180e9 / 1e6)
        inter_host_hop_us = 1.6
        extra_vpu_us = 0.35
        return round(dcp8_us + inter_host_hop_us + extra_bw_us + extra_vpu_us, 2)
    extra_bw_us = (8 * block_bytes) / (180e9 / 1e6)
    inter_host_hop_us = 2.2
    extra_vpu_us = 0.8
    return round(dcp8_us + inter_host_hop_us + extra_bw_us + extra_vpu_us, 2)


def _run_bench() -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    here = Path(__file__).resolve()
    src_dir = str(here.parents[4] / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    if str(here.parent) not in sys.path:
        sys.path.insert(0, str(here.parent))
    import oracle

    import vllm_torchtpu
    from vllm_torchtpu.kernels.experimental.small_collectives import (
        dcp_lse_combine as comb,
    )
    from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce as oproj_ar

    devices = jax.devices()
    world = len(devices)
    mesh = Mesh(np.asarray(devices), ("dcp",))
    chip_pairs = comb.mesh_has_chip_pairs(mesh, "dcp")
    spec = P("dcp")
    sharding = NamedSharding(mesh, spec)
    rank_ids = jax.device_put(jnp.arange(world, dtype=jnp.int32), sharding)

    result = {
        "devices": world,
        "device_kind": devices[0].device_kind,
        "chip_pairs": chip_pairs,
        "jax": jax.__version__,
        "vllm_torchtpu": vllm_torchtpu.__file__,
        "vmem_capacity_bytes": int(pltpu.get_tpu_info().vmem_capacity_bytes),
        "layers": LAYERS,
        "reps": REPS,
        "standalone": {},
        "oproj_chained": {},
    }

    def make_data(tokens: int, local: int, vdim: int, live: int, seed: int):
        rng = np.random.default_rng(seed)
        heads = world * local
        part = rng.standard_normal((world, tokens, heads, vdim), dtype=np.float32)
        lse = (
            rng.standard_normal((world, tokens, heads), dtype=np.float32) * 2.5
        ).astype(np.float32)
        empty = rng.random((world, tokens, heads)) < 0.15
        lse[empty] = -np.inf
        part[empty] = np.nan
        part_full_bf16 = part.astype(jnp.bfloat16)
        lse_full = lse.copy()
        part_live_bf16 = part_full_bf16.copy()
        lse_live = lse_full.copy()
        if live < tokens:
            part_live_bf16[:, live:, :, :4] = np.array(np.nan, dtype=jnp.bfloat16)
            lse_live[:, live:, :] = 50.0
        ref_full = comb.reference_float64(part_full_bf16, lse_full, live_tokens=None)
        ref_live = comb.reference_float64(part_live_bf16, lse_live, live_tokens=live)
        p_full_dev = jax.device_put(
            jnp.asarray(part_full_bf16).reshape(world * tokens, heads, vdim), sharding
        )
        l_full_dev = jax.device_put(
            jnp.asarray(lse_full).reshape(world * tokens, heads), sharding
        )
        p_live_dev = jax.device_put(
            jnp.asarray(part_live_bf16).reshape(world * tokens, heads, vdim), sharding
        )
        l_live_dev = jax.device_put(
            jnp.asarray(lse_live).reshape(world * tokens, heads), sharding
        )
        # Clean carry without NaNs for the 32-layer scan timing loop.
        part_clean = np.where(np.isnan(part), 0.0, part).astype(jnp.bfloat16)
        lse_clean = np.where(np.isfinite(lse), lse, -10.0).astype(np.float32)
        p_clean_dev = jax.device_put(
            jnp.asarray(part_clean).reshape(world * tokens, heads, vdim), sharding
        )
        l_clean_dev = jax.device_put(
            jnp.asarray(lse_clean).reshape(world * tokens, heads), sharding
        )
        return (
            ref_full,
            ref_live,
            p_full_dev,
            l_full_dev,
            p_live_dev,
            l_live_dev,
            p_clean_dev,
            l_clean_dev,
        )

    def time_scan(step_body, p_init, l_init, lv_val):
        def scanned(p0, l0, rk, lv):
            def step(carry, _):
                y = step_body(carry, l0, rk, lv)
                # Thread `[T, local, D]` into next step's `[T, H, D]` carry;
                # local VPU overhead (~0.1 us) is measured and subtracted.
                carry_next = carry + jnp.tile(y, (1, world, 1)) * jnp.bfloat16(1e-4)
                return carry_next, None

            out, _ = lax.scan(step, p0, None, length=LAYERS)
            return out

        fn = jax.jit(
            jax.shard_map(
                scanned,
                mesh=mesh,
                in_specs=(spec, spec, spec, P()),
                out_specs=spec,
                check_vma=False,
            )
        )
        t0 = time.monotonic()
        out = fn(p_init, l_init, rank_ids, lv_val)
        out.block_until_ready()
        compile_s = time.monotonic() - t0
        fn(p_init, l_init, rank_ids, lv_val).block_until_ready()
        t1 = time.perf_counter()
        for _ in range(REPS):
            out = fn(p_init, l_init, rank_ids, lv_val)
        out.block_until_ready()
        raw_us = (time.perf_counter() - t1) / REPS / LAYERS * 1e6
        return raw_us, round(compile_s, 2)

    # Standalone geometries:
    # - (local=8, D=512): H=64 latent combine (matches GLM-5.2 input [T, 64, 512])
    # - (local=8, D=256): H=64 V-space combine (uv_first)
    # - (local=4, D=512): local=4 per-rank slice (matches GLM-5.2 DCP16 block bytes)
    # - (local=4, D=256): local=4 V-space slice (matches GLM-5.2 DCP16 uv_first bytes)
    # - (local=1, D=512): H=8 minimal per-rank head slice
    geometries = [
        (8, 512),
        (8, 256),
        (4, 512),
        (4, 256),
        (1, 512),
    ]

    for local, vdim in geometries:
        heads = world * local
        for tokens in TOKEN_BUCKETS:
            live = max(1, tokens - (tokens // 4))
            lv_full = jnp.asarray([tokens], dtype=jnp.int32)
            lv_part = jnp.asarray([live], dtype=jnp.int32)
            seed_val = 1000 + 31 * tokens + 7 * local + vdim
            (
                ref_full,
                ref_live,
                p_full_dev,
                l_full_dev,
                p_live_dev,
                l_live_dev,
                p_clean_dev,
                l_clean_dev,
            ) = make_data(tokens, local, vdim, live=live, seed=seed_val)

            overhead_us, _ = time_scan(
                lambda c, l0, rk, lv, _loc=local: c[:, :_loc, :],
                p_clean_dev,
                l_clean_dev,
                lv_full,
            )

            variants = {
                "rs_bf16": (
                    lambda c, l0, rk, lv: _baseline_rs_local(
                        c, l0, axis_name="dcp", wire_bf16=True
                    ),
                    lv_full,
                    tokens,
                    ref_full,
                ),
                "rs_f32": (
                    lambda c, l0, rk, lv: _baseline_rs_local(
                        c, l0, axis_name="dcp", wire_bf16=False
                    ),
                    lv_full,
                    tokens,
                    ref_full,
                ),
                "a2a_pack": (
                    lambda c, l0, rk, lv: _baseline_a2a_packed_local(
                        c, l0, world=world, axis_name="dcp"
                    ),
                    lv_full,
                    tokens,
                    ref_full,
                ),
                "online_flat": (
                    lambda c, l0, rk, lv: comb.dcp_online_lse_combine(
                        c,
                        l0,
                        num_cores=world,
                        axis="dcp",
                        mode="flat",
                        rank=rk,
                        out_vmem=False,
                    ),
                    lv_full,
                    tokens,
                    ref_full,
                ),
                "online_flat_live75": (
                    lambda c, l0, rk, lv: comb.dcp_online_lse_combine(
                        c,
                        l0,
                        num_cores=world,
                        axis="dcp",
                        mode="flat",
                        rank=rk,
                        live_tokens=lv,
                        out_vmem=False,
                    ),
                    lv_part,
                    live,
                    ref_live,
                ),
            }
            if chip_pairs:
                variants["online_pairs"] = (
                    lambda c, l0, rk, lv: comb.dcp_online_lse_combine(
                        c,
                        l0,
                        num_cores=world,
                        axis="dcp",
                        mode="pairs",
                        rank=rk,
                        out_vmem=False,
                    ),
                    lv_full,
                    tokens,
                    ref_full,
                )
                variants["online_pairs_live75"] = (
                    lambda c, l0, rk, lv: comb.dcp_online_lse_combine(
                        c,
                        l0,
                        num_cores=world,
                        axis="dcp",
                        mode="pairs",
                        rank=rk,
                        live_tokens=lv,
                        out_vmem=False,
                    ),
                    lv_part,
                    live,
                    ref_live,
                )

            bucket_key = f"T{tokens}_H{heads}_L{local}_D{vdim}"
            bucket_res = {"chain_overhead_us": round(overhead_us, 2), "variants": {}}
            for vname, (vbody, lv_arg, eff_live, ref_arr) in variants.items():
                fn_once = jax.jit(
                    jax.shard_map(
                        vbody,
                        mesh=mesh,
                        in_specs=(spec, spec, spec, P()),
                        out_specs=spec,
                        check_vma=False,
                    )
                )
                p_in, l_in = (
                    (p_live_dev, l_live_dev)
                    if eff_live < tokens
                    else (p_full_dev, l_full_dev)
                )
                got = np.asarray(fn_once(p_in, l_in, rank_ids, lv_arg)).reshape(
                    world, tokens, local, vdim
                )
                acc = _accuracy_metrics(got, ref_arr, eff_live)
                raw_us, comp_s = time_scan(vbody, p_clean_dev, l_clean_dev, lv_arg)
                net_us = round(max(0.0, raw_us - overhead_us), 2)
                entry = {
                    **acc,
                    "raw_us": round(raw_us, 2),
                    "us": net_us,
                    "compile_s": comp_s,
                }
                if local == 4 and vname in ("online_pairs", "online_flat"):
                    entry["est_dcp16_2host_us"] = _project_dcp16_2host_us(
                        net_us,
                        tokens=tokens,
                        local_dcp16=4,
                        vdim=vdim,
                        mode="pairs" if "pairs" in vname else "flat",
                    )
                bucket_res["variants"][vname] = entry

            rs_us = bucket_res["variants"]["rs_bf16"]["us"]
            if "online_pairs" in bucket_res["variants"] and rs_us > 0:
                p_us = bucket_res["variants"]["online_pairs"]["us"]
                bucket_res["speedup_pairs_vs_rs_bf16"] = (
                    round(rs_us / p_us, 3) if p_us > 0 else None
                )
            result["standalone"][bucket_key] = bucket_res
            print(f"CASE {bucket_key} {json.dumps(bucket_res)}", flush=True)

    # End-to-end VMEM-chained Combine + FP8 O-proj All-Reduce across T=16..96
    # (exercises LIMIT_M = 96 at T = 80, 96).
    if chip_pairs:
        rng = np.random.default_rng(77)
        local_c = 4
        w_uv_np = (
            rng.standard_normal((world * local_c, 512, 256), dtype=np.float32) * 0.05
        ).astype(jnp.float8_e4m3fn)
        w_q_np = (
            rng.standard_normal((world * local_c * 256, HIDDEN), dtype=np.float32)
            * 0.02
        ).astype(jnp.float8_e4m3fn)
        w_s_np = np.ones((HIDDEN,), dtype=np.float32) * 0.05
        w_uv_dev = jax.device_put(jnp.asarray(w_uv_np), sharding)
        w_q_dev = jax.device_put(jnp.asarray(w_q_np), sharding)
        w_s_dev = jax.device_put(jnp.asarray(w_s_np), NamedSharding(mesh, P()))

        for tokens in TOKEN_BUCKETS:
            *_, p512_clean, l512_clean = make_data(
                tokens, local_c, 512, live=tokens, seed=2000 + tokens
            )
            lv_full = jnp.asarray([tokens], dtype=jnp.int32)

            def make_chained_fn(use_online: bool, out_vmem: bool = True, _t=tokens):
                def step_fn(p0, l0, rk, lv, uv, q, s):
                    def step(carry, _):
                        if use_online:
                            lat = comb.dcp_online_lse_combine(
                                carry,
                                l0,
                                num_cores=world,
                                axis="dcp",
                                mode="pairs",
                                rank=rk,
                                out_vmem=out_vmem,
                            )
                        else:
                            lat = _baseline_rs_local(
                                carry, l0, axis_name="dcp", wire_bf16=True
                            )
                        y = oproj_ar.fp8_uv_matmul_all_reduce(
                            lat.reshape(_t, local_c * 512),
                            uv,
                            q,
                            s,
                            rk,
                            num_cores=world,
                            axis="dcp",
                            out_dtype=jnp.bfloat16,
                            max_m=96,
                        )
                        carry_next = carry + y[:, None, :512] * jnp.bfloat16(1e-4)
                        return carry_next, None

                    out, _ = lax.scan(step, p0, None, length=LAYERS)
                    return out

                return jax.jit(
                    jax.shard_map(
                        step_fn,
                        mesh=mesh,
                        in_specs=(spec, spec, spec, P(), spec, spec, P()),
                        out_specs=spec,
                        check_vma=False,
                    )
                )

            c_res = {}
            for cname, (use_onl, ov) in (
                ("rs_bf16_then_uv_oproj_ar", (False, False)),
                ("online_pairs_hbm_then_uv_oproj_ar", (True, False)),
                ("online_pairs_vmem_then_uv_oproj_ar", (True, True)),
            ):
                fn_c = make_chained_fn(use_onl, ov)
                c_args = (
                    p512_clean,
                    l512_clean,
                    rank_ids,
                    lv_full,
                    w_uv_dev,
                    w_q_dev,
                    w_s_dev,
                )
                compiled = fn_c.lower(*c_args).compile()
                hlo_txt = compiled.as_text()
                vmem_ops = oracle.custom_call_operands_in_vmem(
                    hlo_txt, "fp8_uv_matmul_all_reduce"
                )
                out = compiled(*c_args)
                out.block_until_ready()
                compiled(*c_args).block_until_ready()
                t1 = time.perf_counter()
                for _ in range(REPS):
                    out = compiled(*c_args)
                out.block_until_ready()
                us = round((time.perf_counter() - t1) / REPS / LAYERS * 1e6, 2)
                c_res[cname] = {
                    "us": us,
                    "oproj_vmem_operands": {k: len(v) for k, v in vmem_ops.items()},
                    "finite": bool(np.isfinite(np.asarray(out, np.float32)).all()),
                }
            ck = f"T{tokens}_L{local_c}_D512_uv_oproj_ar"
            result["oproj_chained"][ck] = c_res
            print(f"CASE {ck} {json.dumps(c_res)}", flush=True)

    print("RESULT " + json.dumps(result), flush=True)
    return result


def test_bench_baselines_and_projection_cpu():
    """Verify benchmark baseline shard_map bodies and projection math on CPU."""
    for tokens in TOKEN_BUCKETS:
        p_us = _project_dcp16_2host_us(
            12.0, tokens=tokens, local_dcp16=4, vdim=512, mode="pairs"
        )
        f_us = _project_dcp16_2host_us(
            12.0, tokens=tokens, local_dcp16=4, vdim=512, mode="flat"
        )
        assert 12.0 < p_us < f_us

    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count=8",
    )
    proc = subprocess.run(
        [sys.executable, __file__, "cpu_smoke"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _cpu_smoke():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[4] / "src"))
    from vllm_torchtpu.kernels.experimental.small_collectives import (
        dcp_lse_combine as comb,
    )

    world, tokens, local, vdim = 8, 16, 4, 256
    heads = world * local
    mesh = Mesh(np.asarray(jax.devices()[:world]), ("dcp",))
    sharding = NamedSharding(mesh, P("dcp"))
    rng = np.random.default_rng(12)
    part = rng.standard_normal((world, tokens, heads, vdim), dtype=np.float32).astype(
        jnp.bfloat16
    )
    lse = rng.standard_normal((world, tokens, heads), dtype=np.float32)
    ref = comb.reference_float64(part, lse)
    p_dev = jax.device_put(
        jnp.asarray(part).reshape(world * tokens, heads, vdim), sharding
    )
    l_dev = jax.device_put(jnp.asarray(lse).reshape(world * tokens, heads), sharding)
    for fn_body in (
        lambda o, ls: _baseline_rs_local(o, ls, axis_name="dcp", wire_bf16=True),
        lambda o, ls: _baseline_rs_local(o, ls, axis_name="dcp", wire_bf16=False),
        lambda o, ls: _baseline_a2a_packed_local(o, ls, world=world, axis_name="dcp"),
    ):
        fn = jax.jit(
            jax.shard_map(
                fn_body,
                mesh=mesh,
                in_specs=(P("dcp"), P("dcp")),
                out_specs=P("dcp"),
                check_vma=False,
            )
        )
        got = np.asarray(fn(p_dev, l_dev)).reshape(world, tokens, local, vdim)
        acc = _accuracy_metrics(got, ref, tokens)
        assert acc["nan"] == 0 and acc["rel_l2"] < 0.01 and acc["cos"] > 0.9999


@pytest.mark.multichip
def test_dcp_lse_combine_bench_tpu():
    import jax

    if jax.default_backend() != "tpu" or len(jax.devices()) not in (8, 16):
        pytest.skip("needs a TPU host (8 or 16 cores)")
    result = _run_bench()
    for bkey, bval in sorted(result["standalone"].items()):
        for vname, entry in sorted(bval["variants"].items()):
            assert entry["nan"] == 0, (bkey, vname, entry)
            assert entry["dead_positive_zero"], (bkey, vname, entry)
            assert entry["rel_l2"] < TOLERANCE["rel_l2"], (bkey, vname, entry)
            assert entry["cos"] > TOLERANCE["cos"], (bkey, vname, entry)
            assert entry["max_abs"] <= TOLERANCE["max_abs"], (bkey, vname, entry)
    for ckey, cval in sorted(result["oproj_chained"].items()):
        for cname, entry in sorted(cval.items()):
            assert entry["finite"], (ckey, cname, entry)


if __name__ == "__main__":
    if sys.argv[1:] == ["cpu_smoke"]:
        _cpu_smoke()
    elif sys.argv[1:] == ["bench"]:
        import jax

        if jax.default_backend() != "tpu" or len(jax.devices()) not in (8, 16):
            print("needs a TPU host (8 or 16 cores)", flush=True)
            sys.exit(3)
        _run_bench()
