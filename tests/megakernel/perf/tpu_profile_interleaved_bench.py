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
"""On-TPU (v7x-8) interleaved A/B benchmark for Qwen3.5 execution profiles.

Evaluates `qwen35-candidate30`, `qwen35-candidate30-fused`, and tile / pool-I/O
tuning variants across decode (`T=1`) and speculative verify (`T=4`) buckets on
a single `v7x-8` allocation using 6 interleaved rounds per bucket:
  - `cand1_*_gpr_on`: `pool_gather=True` on `T=4, B <= 16` (`approx`).
  - `cand1b_c30f_fused_mid`: `fused_pool_io=True` on `24 < B < 56`
    (`B in {32, 40, 48}`), closing the middle-bucket split-pool gap (`approx`).
  - `cand2_*`: `nbe=16, nkv=4` for `8 < B <= 48` (`approx` when built on
    `qwen35-candidate30*`).
  - `cand12_*` / `cand1b2_*`: combined variants where both apply.

Numerics (`exact` vs `approx` relative to `qwen35-baseline`) are recorded and
reported separately in every result row and table.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

import vllm_torchtpu.megakernel as mk
from vllm_torchtpu.megakernel import cache, kernel
from vllm_torchtpu.megakernel.cache import should_fuse_pool_io
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.profile import get_profile

BF16 = jnp.bfloat16
FP8 = jnp.float8_e4m3fn
F32 = jnp.float32
I32 = jnp.int32


def _has_tpu_v7x8() -> bool:
    try:
        devs = jax.devices("tpu")
    except RuntimeError:
        return False
    return len(devs) >= 8 and "7x" in devs[0].device_kind.lower()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a.ravel(), b.ravel())
        / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12)
    )


def build_bucket_variants(
    spec: Any,
    base: ExecSpec,
    *,
    T: int,
    B: int,
) -> list[tuple[str, ExecSpec, bool, str]]:
    """Return `(name, ex, fused_pool_io, numerics)` variants for bucket `(T, B)`."""
    g = spec.config
    p_c30 = get_profile("qwen35-candidate30")

    ex_c30 = p_c30.exec_for(base, B=B, T=T, g=g)
    ex_c30_base_fused = mk.with_cache(ex_c30, pool_gather=False)
    fused_auto = should_fuse_pool_io(
        spec, ex_c30_base_fused, fused_pool_io="auto", max_b_low=24, min_b_hi=56
    )

    c30_num = p_c30.numerics

    variants: list[tuple[str, ExecSpec, bool, str]] = [
        ("c30", ex_c30, False, c30_num),
    ]
    if fused_auto:
        variants.append(("c30_fused", ex_c30_base_fused, True, c30_num))

    # Variant 1a: enable pool_gather=True on T=4, B <= 16 (approx numerics).
    if T == 4 and B <= 16:
        ex_c1 = mk.with_cache(ex_c30, pool_gather=True)
        variants.append(("cand1_c30_gpr_on", ex_c1, False, "approx"))
        if fused_auto:
            ex_c1f = mk.with_cache(ex_c30_base_fused, pool_gather=True)
            variants.append(("cand1_c30f_gpr_on", ex_c1f, True, "approx"))

    # Variant 1b: close the 24 < B < 56 gap with fused_pool_io=True.
    if 24 < B < 56 and should_fuse_pool_io(spec, ex_c30_base_fused, fused_pool_io=True):
        variants.append(("cand1b_c30f_fused_mid", ex_c30_base_fused, True, c30_num))

    # Variant 2: nbe=16, nkv=4 for 8 < B <= 48.
    if 8 < B <= 48:
        t_c2 = dataclasses.replace(ex_c30.resolved_tiles, nbe=16).with_modules(
            {"gqa": {"nkv": 4}}
        )
        ex_c2 = dataclasses.replace(ex_c30, tiles=t_c2)
        variants.append(("cand2_c30", ex_c2, False, c30_num))
        if fused_auto:
            ex_c2f = dataclasses.replace(ex_c30_base_fused, tiles=t_c2)
            variants.append(("cand2_c30_fused", ex_c2f, True, c30_num))
        elif 24 < B < 56:
            ex_c2f_mid = dataclasses.replace(ex_c30_base_fused, tiles=t_c2)
            if should_fuse_pool_io(spec, ex_c2f_mid, fused_pool_io=True):
                variants.append(
                    ("cand1b2_c30f_fused_mid_nbe16", ex_c2f_mid, True, c30_num)
                )
        if T == 4 and B <= 16 and fused_auto:
            ex_c12f = mk.with_cache(
                dataclasses.replace(ex_c30_base_fused, tiles=t_c2), pool_gather=True
            )
            variants.append(("cand12_c30f_gpr_on_nbe16", ex_c12f, True, "approx"))

    return variants


def summarize_interleaved_pairs(
    samples_by_name: dict[str, list[float]],
    ref_name: str,
) -> dict[str, dict[str, float | bool]]:
    """Compute paired A/B statistics from interleaved per-round latency samples."""
    ref_arr = np.asarray(samples_by_name[ref_name], dtype=np.float64)
    n_pairs = len(ref_arr)
    assert n_pairs >= 3, f"Expected >= 3 interleaved pairs, got {n_pairs}"
    out: dict[str, dict[str, float | bool]] = {}
    for name, vals in samples_by_name.items():
        arr = np.asarray(vals, dtype=np.float64)
        assert len(arr) == n_pairs
        rel_deltas = (arr - ref_arr) / ref_arr * 100.0
        med_delta = float(np.median(rel_deltas))
        mean_delta = float(np.mean(rel_deltas))
        spread = float(np.max(np.abs(rel_deltas - med_delta)))
        std_delta = float(np.std(rel_deltas, ddof=1)) if n_pairs > 1 else 0.0
        se_delta = std_delta / float(np.sqrt(n_pairs)) if n_pairs > 1 else 0.0
        # Significant if all pairs agree in sign AND |median_delta| > max(spread, 2*SE).
        same_sign = bool(np.all(rel_deltas < 0.0) or np.all(rel_deltas > 0.0))
        sig = bool(
            name != ref_name
            and same_sign
            and abs(med_delta) > max(spread, 2.0 * se_delta, 0.5)
        )
        out[name] = {
            "median_us": round(float(np.median(arr)), 1),
            "mean_us": round(float(np.mean(arr)), 1),
            "p10_us": round(float(np.percentile(arr, 10)), 1),
            "p90_us": round(float(np.percentile(arr, 90)), 1),
            "paired_median_delta_pct": round(med_delta, 2),
            "paired_mean_delta_pct": round(mean_delta, 2),
            "paired_spread_pct": round(spread, 2),
            "paired_se_pct": round(se_delta, 2),
            "all_pairs_same_sign": same_sign,
            "significant": sig,
        }
    return out


def test_profile_interleaved_helper_unit():
    """Fast CPU unit test for variant builder and paired interleaved stats."""
    g = dataclasses.replace(qwen35.FULL_EP, L=60, att_every=4)
    spec = qwen35.qwen35_spec(g)
    base = ExecSpec(
        R=8, ctx=4096, stack_e=True, cache=mk.CacheExec(paged_kv=True, num_pages=4096)
    )
    v_t4_b16 = build_bucket_variants(spec, base, T=4, B=16)
    names_16 = [v[0] for v in v_t4_b16]
    assert "c30" in names_16
    assert "c30_fused" in names_16
    assert "cand1_c30f_gpr_on" in names_16
    assert "cand2_c30_fused" in names_16
    assert "cand12_c30f_gpr_on_nbe16" in names_16

    v_t4_b32 = build_bucket_variants(spec, base, T=4, B=32)
    names_32 = [v[0] for v in v_t4_b32]
    assert "c30" in names_32
    assert "cand1b_c30f_fused_mid" in names_32
    assert "cand2_c30" in names_32
    assert "cand1b2_c30f_fused_mid_nbe16" in names_32

    fake_samples = {
        "c30": [100.0, 101.0, 99.5, 100.5, 100.0, 100.2],
        "c30_fused": [92.0, 92.8, 91.5, 92.4, 92.1, 92.2],
    }
    summary = summarize_interleaved_pairs(fake_samples, "c30")
    assert summary["c30_fused"]["significant"] is True
    assert summary["c30_fused"]["paired_median_delta_pct"] < -7.0


def test_tpu_v7x8_profile_interleaved_candidates():
    """v7x-8 interleaved A/B benchmark across T=4 and T=1 buckets."""
    if not _has_tpu_v7x8():
        pytest.skip("Requires 8 TPU TensorCores (v7x-8)")

    from tests.megakernel.test_tpu_v7x8_bench import (
        _check_pooled_hlo_or_fail,
        _init_pooled_bucket_on_tpu,
        _init_shared_weights_on_tpu,
    )

    tpu_devs = jax.devices("tpu")
    mesh = Mesh(tpu_devs[:8], ("tp",))

    g = dataclasses.replace(qwen35.FULL_EP, L=60, att_every=4)
    spec = qwen35.qwen35_spec(g)
    ex_init = ExecSpec(B=4, T=1, R=8, ctx=4096, stack_e=True)
    w_lib, w_f = _init_shared_weights_on_tpu(mesh, g, ex_init)
    del w_f
    gc.collect()

    p_def = get_profile("qwen35-baseline")
    base = ExecSpec(
        R=8, ctx=4096, stack_e=True, cache=mk.CacheExec(paged_kv=True, num_pages=4096)
    )
    t4_buckets = [(4, b) for b in (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)]
    t1_buckets = [(1, b) for b in (4, 8, 16, 32, 64)]
    all_buckets = t4_buckets + t1_buckets

    n_rounds = 6
    steps_per_round = 10
    warmup_steps = 10
    all_rows: list[dict[str, Any]] = []

    for T, B in all_buckets:
        tag = f"T={T},B={B}"
        ex_def = p_def.exec_for(base, B=B, T=T, g=g)
        pos, hid, raw_init, pt, ckpt, roff = _init_pooled_bucket_on_tpu(mesh, g, ex_def)

        # 1. Exact baseline reference output (`qwen35-baseline`).
        step_def, info_def = cache.make_pooled_step(
            mesh, spec, ex_def, donate_pools=True, fused_pool_io=False
        )
        raw_def = tuple(jnp.copy(p) for p in raw_init)
        out_def, raw_def = step_def(pos, hid, w_lib, raw_def, pt, ckpt, roff)
        jax.block_until_ready(out_def)
        hid_def_raw = np.asarray(jax.device_get(kernel.unfold(out_def, 8)))[:, :B, :]
        hid_def_u8 = hid_def_raw.view(np.uint8).copy()
        hid_def_f32 = hid_def_raw.astype(np.float32)
        del out_def, raw_def, step_def, info_def
        gc.collect()

        # 2. Compile and verify HLO + numerics for all variants in this bucket.
        variants = build_bucket_variants(spec, base, T=T, B=B)
        compiled_steps: dict[str, Any] = {}
        meta_by_name: dict[str, dict[str, Any]] = {}
        c30_u8: np.ndarray | None = None

        for v_name, ex_v, fused_v, num_kind in variants:
            step_v, info_v = cache.make_pooled_step(
                mesh, spec, ex_v, donate_pools=True, fused_pool_io=fused_v
            )
            comp = step_v.lower(pos, hid, w_lib, raw_init, pt, ckpt, roff).compile()
            _check_pooled_hlo_or_fail(
                comp.as_text(), spec, ex_v, fused=fused_v, tag=f"{v_name}:{tag}"
            )
            del comp

            # Run 1 verification step on cloned pools to check numerics cleanly.
            raw_clone = tuple(jnp.copy(p) for p in raw_init)
            out_v, raw_v_out = step_v(pos, hid, w_lib, raw_clone, pt, ckpt, roff)
            jax.block_until_ready(out_v)
            hid_v_raw = np.asarray(jax.device_get(kernel.unfold(out_v, 8)))[:, :B, :]
            hid_v_u8 = hid_v_raw.view(np.uint8).copy()
            hid_v_f32 = hid_v_raw.astype(np.float32)
            del out_v, raw_v_out, raw_clone
            gc.collect()
            assert np.all(np.isfinite(hid_v_f32)), (
                f"Non-finite output in {v_name}:{tag}"
            )
            if v_name == "c30":
                c30_u8 = hid_v_u8.copy()

            bit_exact_def = bool(np.array_equal(hid_v_u8, hid_def_u8))
            bit_exact_c30 = (
                bool(np.array_equal(hid_v_u8, c30_u8)) if c30_u8 is not None else True
            )
            cos_def = round(_cos(hid_def_f32, hid_v_f32), 6)
            max_diff_def = float(np.max(np.abs(hid_v_f32 - hid_def_f32)))

            compiled_steps[v_name] = step_v
            meta_by_name[v_name] = {
                "numerics": num_kind,
                "fused_pool_io": fused_v,
                "pool_gather": bool(info_v["plan"].module_params["gdn"].GPR),
                "nbe": int(ex_v.resolved_tiles.nbe),
                "nkv": int(ex_v.resolved_tiles.module("gqa").nkv),
                "vmem_mib": round(info_v["vmem_scratch_bytes"] / (1 << 20), 2),
                "hid_bit_exact_vs_default": bit_exact_def,
                "hid_bit_exact_vs_c30": bit_exact_c30,
                "cos_vs_default": cos_def,
                "max_abs_diff_vs_default": round(max_diff_def, 5),
            }

        # Warm up all compiled variants on the shared timing pools.
        v_names = [v[0] for v in variants]
        pools_timing = raw_init
        del raw_init
        for v_name in v_names:
            step_v = compiled_steps[v_name]
            for _ in range(warmup_steps):
                out_v, pools_timing = step_v(
                    pos, hid, w_lib, pools_timing, pt, ckpt, roff
                )
            jax.block_until_ready(out_v)
            del out_v

        # 3. Interleaved timing rounds (forward order on even rounds, reverse on odd).
        samples: dict[str, list[float]] = {n: [] for n in v_names}
        for r_idx in range(n_rounds):
            order = v_names if (r_idx % 2 == 0) else list(reversed(v_names))
            for v_name in order:
                step_v = compiled_steps[v_name]
                t0 = time.perf_counter()
                for _ in range(steps_per_round):
                    out_v, pools_timing = step_v(
                        pos, hid, w_lib, pools_timing, pt, ckpt, roff
                    )
                jax.block_until_ready(out_v)
                dt_us = (time.perf_counter() - t0) * 1e6 / steps_per_round
                samples[v_name].append(dt_us)
                del out_v

        del pools_timing

        stats_vs_c30 = summarize_interleaved_pairs(samples, "c30")
        ref_fused = "c30_fused" if "c30_fused" in samples else "c30"
        stats_vs_c30f = summarize_interleaved_pairs(samples, ref_fused)

        bucket_entry: dict[str, Any] = {
            "T": T,
            "B": B,
            "NS": B // T,
            "n_rounds": n_rounds,
            "steps_per_round": steps_per_round,
            "variants": {},
        }
        for v_name in v_names:
            v_entry = {
                **meta_by_name[v_name],
                **stats_vs_c30[v_name],
                "vs_c30_fused_median_delta_pct": stats_vs_c30f[v_name][
                    "paired_median_delta_pct"
                ],
                "vs_c30_fused_spread_pct": stats_vs_c30f[v_name]["paired_spread_pct"],
                "vs_c30_fused_significant": stats_vs_c30f[v_name]["significant"],
                "round_samples_us": [round(x, 1) for x in samples[v_name]],
            }
            bucket_entry["variants"][v_name] = v_entry

        all_rows.append(bucket_entry)
        summary_str = " | ".join(
            f"{n}({bucket_entry['variants'][n]['numerics']}): "
            f"{bucket_entry['variants'][n]['median_us']:.1f}us "
            f"(vs_c30={bucket_entry['variants'][n]['paired_median_delta_pct']:+.2f}%, "
            f"vs_c30f={bucket_entry['variants'][n]['vs_c30_fused_median_delta_pct']:+.2f}%)"
            for n in v_names
        )
        print(f"[Profile Interleaved {tag}] {summary_str}", flush=True)

        del compiled_steps, pos, hid, pt, ckpt, roff
        gc.collect()

    print("\n=== PROFILE_INTERLEAVED_TPU_V7X8_JSON_START ===")
    print(json.dumps({"profile_interleaved_bench": all_rows}, indent=2))
    print("=== PROFILE_INTERLEAVED_TPU_V7X8_JSON_END ===", flush=True)
