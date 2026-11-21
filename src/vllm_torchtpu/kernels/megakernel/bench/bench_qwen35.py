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
"""M0 benchmark & verification harness for frozen Qwen3.5 megakernel (`_frozen.qwen35`).

Supports three execution modes:
  1. `interpret`: CPU `interpret=True` on `TINY` / `TINY_EP` across virtual host devices.
     Verifies `make_step` (unpooled) and `make_pooled_step(fused_pool_io=False)` (pooled)
     for T in {1, 4} and B in {4, 8}.
  2. `aot`: TPU7x (`tpu7x:2x2x1`) ahead-of-time XLA/Mosaic compilation of `FULL_EP` / `FULL`
     using zero-memory `jax.ShapeDtypeStruct` descriptors across B in {4, 8, 16, 32} and
     T in {1, 4}, recording compile time, VMEM scratch bytes, tile counts, and HLO metrics.
  3. `tpu`: Hardware execution on TPU v7x-8, measuring step latency (median / p90 / min).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.megakernel._frozen.qwen35 import megakernel as K
from vllm_torchtpu.kernels.megakernel._frozen.qwen35 import pool_io


def ensure_cpu_devices(min_devices: int = 8) -> list[jax.Device]:
    """Return available JAX devices; requires XLA_FLAGS=--xla_force_host_platform_device_count=N on CPU."""
    devs = jax.devices()
    if len(devs) < min_devices:
        raise RuntimeError(
            f"Need at least {min_devices} JAX devices, got {len(devs)}. "
            f"Set XLA_FLAGS='--xla_force_host_platform_device_count={min_devices}' before importing jax."
        )
    return devs[:min_devices]


def make_tpu7x_aot_mesh(num_ranks: int = 8, axis_name: str = "tp") -> Mesh:
    """Create a compile-only TPU7x mesh (`tpu7x:2x2x1` for 8 ranks, `tpu7x:2x2x2` for 16 ranks)."""
    from jax.experimental import topologies

    topo_str = "tpu7x:2x2x1" if num_ranks == 8 else "tpu7x:2x2x2"
    topo = topologies.get_topology_desc(topology_name=topo_str, platform="tpu")
    return Mesh(np.array(topo.devices[:num_ranks]), (axis_name,))


def make_random_qwen35_inputs(
    c: K.MKConfig,
    tiles: K.Tiles | None = None,
    *,
    seed: int = 0,
    init_pos: int = 32,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, jax.Array]]:
    """Build deterministic random `(pos, hidden, weights, states)` for `K.make_step`."""
    tiles = K.default_tiles(c) if tiles is None else tiles
    rng = np.random.default_rng(seed)

    # 1. pos [B] int32: sequence s has tokens at init_pos + 16*s + t
    pos_np = np.array(
        [init_pos + 16 * s + t for s in range(c.NS) for t in range(c.T)],
        dtype=np.int32,
    )
    pos = jnp.asarray(pos_np)

    # 2. hidden [R * M, H] bf16 (rank-replicated rows, zero-padded past B)
    hid_1 = rng.standard_normal((c.M, c.H), dtype=np.float32) * 0.1
    hid_1[c.B :] = 0.0
    hid_rep = np.tile(hid_1[None, :, :], (c.R, 1, 1)).reshape(c.R * c.M, c.H)
    hidden = jnp.asarray(hid_rep).astype(K.BF16)

    # 3. weights dict with leading rank axis [R, ...]
    wshapes = K.weight_shapes(c, tiles)
    lay, S = K.smalls_layout(c)
    weights: dict[str, Any] = {}

    def _rand_tensor(shape: tuple[int, ...], dtype: Any, scale: float = 0.05) -> jax.Array:
        dt = jnp.dtype(dtype)
        if dt == jnp.dtype(K.FP8):
            arr = np.clip(rng.standard_normal(shape, dtype=np.float32) * 2.0, -16.0, 16.0)
            return jnp.asarray(arr).astype(K.FP8)
        if dt == jnp.dtype(K.BF16):
            arr = rng.standard_normal(shape, dtype=np.float32) * scale
            return jnp.asarray(arr).astype(K.BF16)
        arr = rng.standard_normal(shape, dtype=np.float32) * scale
        return jnp.asarray(arr).astype(K.F32)

    for k, spec in wshapes.items():
        if isinstance(spec, tuple) and isinstance(spec[0], K.Spec):
            # Per-layer tuple of expert views: each leaf [R, EW, n_tiles, tk, N]
            weights[k] = tuple(
                _rand_tensor((c.R, *leaf.shape), leaf.dtype) for leaf in spec
            )
            continue
        sh, dt = spec
        if k == "smalls":
            sm = np.zeros((c.R, c.L, 1, S), dtype=np.float32)
            for name, (off, width) in lay.items():
                if name in ("in_norm", "post_norm", "gnorm", "qn", "kn"):
                    sm[:, :, 0, off : off + width] = 0.1
                elif name in ("in_s", "out_s", "sh13_s", "sh2_s"):
                    sm[:, :, 0, off : off + width] = 0.01
                elif name == "alog":
                    sm[:, :, 0, off : off + width] = -2.0
                elif name == "dtb":
                    sm[:, :, 0, off : off + width] = -1.0
                elif name == "ecopy":
                    sm[:, :, 0, off : off + width] = 0.0
                else:
                    sm[:, :, 0, off : off + width] = (
                        rng.standard_normal((c.R, c.L, width), dtype=np.float32) * 0.02
                    )
            weights[k] = jnp.asarray(sm)
        elif k == "es":
            es = np.full((c.R, *sh), 0.01, dtype=np.float32)
            weights[k] = jnp.asarray(es)
        elif k == "final_norm":
            fn = np.full((c.R, *sh), 0.1, dtype=np.float32)
            weights[k] = jnp.asarray(fn)
        else:
            weights[k] = _rand_tensor((c.R, *sh), dt)

    # 4. states dict (rank-folded [R * N0, ...])
    sshapes = K.state_shapes(dataclasses.replace(c, paged_kv=False, fused_pool_io=False))
    states_unfolded: dict[str, jax.Array] = {}
    for k, (sh, dt) in sshapes.items():
        states_unfolded[k] = _rand_tensor((c.R, *sh), dt, scale=0.02)
    states = {k: K.fold(v) for k, v in states_unfolded.items()}
    return pos, hidden, weights, states


def make_abstract_qwen35_inputs(
    mesh: Mesh,
    c: K.MKConfig,
    tiles: K.Tiles | None = None,
    *,
    pooled: bool = False,
    fused_pool_io: bool = False,
) -> tuple[Any, ...]:
    """Create zero-memory `jax.ShapeDtypeStruct` descriptors for AOT compilation."""
    tiles = K.default_tiles(c) if tiles is None else tiles
    axis = mesh.axis_names[0]
    sh_rep = NamedSharding(mesh, P())
    sh_tp = NamedSharding(mesh, P(axis))

    def _sd(shape: tuple[int, ...], dtype: Any, sharding: NamedSharding) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(shape, jnp.dtype(dtype), sharding=sharding)

    c_eff = dataclasses.replace(c, paged_kv=pooled, fused_pool_io=(pooled and fused_pool_io))
    wshapes = K.weight_shapes(c_eff, tiles)

    if pooled:
        # Flat positional args for pool_io.make_flat_pooled_step:
        # every input is sharded along axis 0 over the R ranks (global shape = (R * d0, ...))
        def _tp_global(local_shape: tuple[int, ...], dtype: Any) -> jax.ShapeDtypeStruct:
            return _sd((c.R * local_shape[0], *local_shape[1:]), dtype, sh_tp)

        pos_spec = _tp_global((c.B,), K.I32)
        hid_spec = _tp_global((c.M, c.H), K.BF16)
        w_flat_specs = []
        for k in K.W_NAMES:
            spec = wshapes[k]
            if isinstance(spec, tuple) and isinstance(spec[0], K.Spec):
                for leaf in spec:
                    w_flat_specs.append(_tp_global(leaf.shape, leaf.dtype))
            else:
                sh, dt = spec
                w_flat_specs.append(_tp_global(sh, dt))
        raw_specs = [
            _tp_global((c_eff.num_pages, 2, 64, 4, c_eff.page_size), K.FP8)
            for _ in range(c_eff.NATT)
        ]
        G_PER_ATT = max(1, c_eff.att_every - 1)
        pt_spec = _tp_global((c_eff.NS, c_eff.max_pages_per_seq), K.I32)
        ckpt_spec = _tp_global((G_PER_ATT, c_eff.NS, c_eff.T), K.I32)
        roff_spec = _tp_global((G_PER_ATT, c_eff.NS), K.I32)
        return (pos_spec, hid_spec, *w_flat_specs, *raw_specs, pt_spec, ckpt_spec, roff_spec)

    # Unpooled `K.make_step(mesh, c, tiles)` signature: `(pos, hidden, weights, states)`
    pos_spec = _sd((c.B,), K.I32, sh_rep)
    hid_spec = _sd((c.R * c.M, c.H), K.BF16, sh_tp)
    weights_spec: dict[str, Any] = {}
    for k, spec in wshapes.items():
        if isinstance(spec, tuple) and isinstance(spec[0], K.Spec):
            weights_spec[k] = tuple(_sd((c.R, *leaf.shape), leaf.dtype, sh_tp) for leaf in spec)
        else:
            sh, dt = spec
            weights_spec[k] = _sd((c.R, *sh), dt, sh_tp)
    sshapes = K.state_shapes(c_eff)
    states_spec = {
        k: _sd((c.R * sh[0], *sh[1:]), dt, sh_tp) for k, (sh, dt) in sshapes.items()
    }
    return (pos_spec, hid_spec, weights_spec, states_spec)


def analyze_hlo_text(hlo_text: str) -> dict[str, Any]:
    """Extract HLO structural metrics from compiled HLO text."""
    lines = hlo_text.splitlines()
    custom_calls = [ln for ln in lines if "custom-call(" in ln]
    copy_starts = [ln for ln in lines if "copy-start(" in ln]
    async_starts = [ln for ln in lines if "async-start(" in ln]
    return {
        "hlo_lines": len(lines),
        "custom_calls": len(custom_calls),
        "copy_starts": len(copy_starts),
        "async_starts": len(async_starts),
    }


def run_qwen35_interpret(
    configs: list[tuple[int, int]] = [(1, 2), (2, 4)],
    *,
    moe_tp: bool = False,
    full_tiny: bool = False,
) -> list[dict[str, Any]]:
    """Run CPU `interpret=True` verification on `TINY` / `CPU_TINY` for unpooled and pooled steps."""
    n_ranks = 8 if full_tiny else 1
    devs = ensure_cpu_devices(n_ranks)
    mesh = Mesh(np.array(devs[:n_ranks]), ("tp",))
    if full_tiny:
        base_cfg = K.TINY if moe_tp else K.TINY_EP
    else:
        base_cfg = K.MKConfig(
            R=1,
            L=4,
            att_every=4,
            H=512,
            NH=1,
            NKV=1,
            KH=1,
            VH=2,
            I=128,
            IS=128,
            E=128,
            TOPK=2,
            ctx=512,
            moe_tp=moe_tp,
        )
    results: list[dict[str, Any]] = []

    for T, B in configs:
        c = dataclasses.replace(base_cfg, T=T, B=B)
        tiles = dataclasses.replace(K.TINY_TILES, hier=0, pack=0, mb8=0, kv_blk=256, nkv=2)
        pos, hidden, weights, states = make_random_qwen35_inputs(c, tiles, seed=42)

        # 1. Unpooled `make_step`
        t0 = time.perf_counter()
        step_fn, info = K.make_step(mesh, c, tiles, interpret=True)
        hid_out, st_out = step_fn(pos, hidden, weights, states)
        jax.block_until_ready(hid_out)
        dt_unpooled_s = time.perf_counter() - t0

        hid_np = np.asarray(hid_out[: c.B], dtype=np.float32)
        assert np.all(np.isfinite(hid_np)), f"Non-finite output in unpooled TINY (T={T}, B={B})"
        norm_unpooled = float(np.linalg.norm(hid_np))

        results.append(
            {
                "mode": "interpret",
                "step_kind": "make_step",
                "config": f"TINY_{'TP' if moe_tp else 'EP'}",
                "T": T,
                "B": B,
                "vmem_scratch_mib": round(info["vmem_scratch_bytes"] / (1024 * 1024), 2),
                "T_TOTAL": info["T_TOTAL"],
                "wall_s": round(dt_unpooled_s, 3),
                "out_l2_norm": round(norm_unpooled, 4),
            }
        )

        # 2. Pooled `make_pooled_step(fused_pool_io=False)`
        pt, ckpt, roff, min_pages = pool_io.default_pool_layout(c)
        c_pool = dataclasses.replace(
            c,
            paged_kv=True,
            fused_pool_io=False,
            num_pages=max(128, min_pages + 16),
            max_pages_per_seq=max(16, int(pt.shape[1])),
        )
        st_unfolded = {k: K.unfold(v, c.R) for k, v in states.items()}
        raw_pools = pool_io.states_to_pool(c_pool, st_unfolded, pt, ckpt, roff)
        pooled_states = {
            "raw": raw_pools,
            "page_table": pt,
            "ckpt_indices": ckpt,
            "read_offsets": roff,
        }

        t1 = time.perf_counter()
        pooled_step, p_info = pool_io.make_pooled_step(
            mesh, c_pool, tiles, interpret=True, fused_pool_io=False
        )
        hid_pool, _ = pooled_step(pos, hidden, weights, pooled_states)
        jax.block_until_ready(hid_pool)
        dt_pooled_s = time.perf_counter() - t1

        hid_pool_np = np.asarray(hid_pool[: c.B], dtype=np.float32)
        assert np.all(np.isfinite(hid_pool_np)), f"Non-finite output in pooled TINY (T={T}, B={B})"
        norm_pooled = float(np.linalg.norm(hid_pool_np))
        cos_sim = float(
            np.sum(hid_np * hid_pool_np)
            / (np.linalg.norm(hid_np) * np.linalg.norm(hid_pool_np) + 1e-12)
        )

        results.append(
            {
                "mode": "interpret",
                "step_kind": "make_pooled_step(fused=False)",
                "config": f"TINY_{'TP' if moe_tp else 'EP'}",
                "T": T,
                "B": B,
                "vmem_scratch_mib": round(p_info["vmem_scratch_bytes"] / (1024 * 1024), 2),
                "T_TOTAL": p_info["T_TOTAL"],
                "wall_s": round(dt_pooled_s, 3),
                "out_l2_norm": round(norm_pooled, 4),
                "cosine_vs_unpooled": round(cos_sim, 6),
            }
        )
    return results


def compile_qwen35_aot(
    configs: list[tuple[int, int]] = [
        (1, 4),
        (1, 8),
        (1, 16),
        (1, 32),
        (4, 4),
        (4, 8),
        (4, 16),
        (4, 32),
    ],
    *,
    moe_tp: bool = False,
    include_pooled: bool = True,
) -> list[dict[str, Any]]:
    """Run TPU7x (`tpu7x:2x2x1`) AOT compilation on `FULL` / `FULL_EP` across (T, B) buckets."""
    mesh = make_tpu7x_aot_mesh(num_ranks=8, axis_name="tp")
    base_cfg = K.FULL if moe_tp else K.FULL_EP
    results: list[dict[str, Any]] = []

    for T, B in configs:
        c = dataclasses.replace(
            base_cfg,
            T=T,
            B=B,
            stack_e=False,
            e_slots=(0 if moe_tp else 72),
        )
        tiles = K.default_tiles(c)

        # 1. Unpooled `make_step`
        t0 = time.perf_counter()
        step_fn, info = K.make_step(mesh, c, tiles, interpret=False)
        abs_args = make_abstract_qwen35_inputs(mesh, c, tiles, pooled=False)
        lowered = jax.jit(step_fn).lower(*abs_args)
        compiled = lowered.compile()
        compile_s = time.perf_counter() - t0
        hlo_stats = analyze_hlo_text(compiled.as_text())
        results.append(
            {
                "mode": "aot_tpu7x",
                "step_kind": "make_step",
                "config": f"FULL_{'TP' if moe_tp else 'EP72'}",
                "T": T,
                "B": B,
                "nbd": tiles.nbd,
                "nbe": tiles.nbe,
                "nkv": tiles.nkv,
                "hier": tiles.hier,
                "rsag": tiles.rsag,
                "vmem_scratch_mib": round(info["vmem_scratch_bytes"] / (1024 * 1024), 2),
                "T_TOTAL": info["T_TOTAL"],
                "compile_s": round(compile_s, 2),
                **hlo_stats,
            }
        )

        # 2. Pooled `make_flat_pooled_step(fused_pool_io=False)`
        if include_pooled:
            c_paged = dataclasses.replace(c, paged_kv=True, fused_pool_io=False)
            t1 = time.perf_counter()
            smap_fn, _, _, donate_argnums = pool_io.make_flat_pooled_step(
                mesh, c_paged, tiles, interpret=False, fused_pool_io=False
            )
            _, p_info = K.build(c_paged, tiles, interpret=False)
            abs_pool_args = make_abstract_qwen35_inputs(
                mesh, c_paged, tiles, pooled=True, fused_pool_io=False
            )
            lowered_p = jax.jit(smap_fn, donate_argnums=donate_argnums).lower(*abs_pool_args)
            compiled_p = lowered_p.compile()
            compile_p_s = time.perf_counter() - t1
            hlo_p_stats = analyze_hlo_text(compiled_p.as_text())
            results.append(
                {
                    "mode": "aot_tpu7x",
                    "step_kind": "make_pooled_step(fused=False)",
                    "config": f"FULL_{'TP' if moe_tp else 'EP72'}",
                    "T": T,
                    "B": B,
                    "nbd": tiles.nbd,
                    "nbe": tiles.nbe,
                    "nkv": tiles.nkv,
                    "hier": tiles.hier,
                    "rsag": tiles.rsag,
                    "vmem_scratch_mib": round(p_info["vmem_scratch_bytes"] / (1024 * 1024), 2),
                    "T_TOTAL": p_info["T_TOTAL"],
                    "compile_s": round(compile_p_s, 2),
                    **hlo_p_stats,
                }
            )
    return results


def bench_qwen35_tpu(
    configs: list[tuple[int, int]] = [(1, 4), (1, 8), (1, 16), (1, 32), (4, 8), (4, 16), (4, 32)],
    *,
    moe_tp: bool = False,
    warmup: int = 5,
    iters: int = 20,
) -> list[dict[str, Any]]:
    """Run hardware latency benchmarks on an attached TPU v7x-8 node."""
    devs = jax.devices("tpu")
    if len(devs) < 8:
        raise RuntimeError(f"Expected 8 TPU devices, found {len(devs)}")
    mesh = Mesh(np.array(devs[:8]), ("tp",))
    base_cfg = K.FULL if moe_tp else K.FULL_EP
    results: list[dict[str, Any]] = []

    for T, B in configs:
        c = dataclasses.replace(base_cfg, T=T, B=B, stack_e=False, e_slots=(0 if moe_tp else 72))
        tiles = K.default_tiles(c)
        pos, hidden, weights, states = make_random_qwen35_inputs(c, tiles, seed=0, init_pos=1024)
        step_fn, info = K.make_step(mesh, c, tiles, interpret=False)
        jitted = jax.jit(step_fn)

        t_c0 = time.perf_counter()
        out_h, _ = jitted(pos, hidden, weights, states)
        jax.block_until_ready(out_h)
        compile_s = time.perf_counter() - t_c0

        for _ in range(warmup):
            out_h, _ = jitted(pos, hidden, weights, states)
            jax.block_until_ready(out_h)

        times_ms = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out_h, _ = jitted(pos, hidden, weights, states)
            jax.block_until_ready(out_h)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        results.append(
            {
                "mode": "tpu_v7x",
                "step_kind": "make_step",
                "config": f"FULL_{'TP' if moe_tp else 'EP72'}",
                "T": T,
                "B": B,
                "vmem_scratch_mib": round(info["vmem_scratch_bytes"] / (1024 * 1024), 2),
                "compile_s": round(compile_s, 2),
                "min_ms": round(float(np.min(times_ms)), 3),
                "median_ms": round(float(np.median(times_ms)), 3),
                "p90_ms": round(float(np.percentile(times_ms, 90)), 3),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frozen Qwen3.5 megakernel.")
    parser.add_argument(
        "--mode",
        choices=("interpret", "aot", "tpu"),
        default="interpret",
        help="Execution mode: interpret (CPU TINY), aot (TPU7x FULL compile), or tpu (v7x hardware).",
    )
    parser.add_argument("--moe-tp", action="store_true", help="Use TP-sliced MoE instead of EP.")
    parser.add_argument("--no-pooled", action="store_true", help="Skip pooled step in AOT mode.")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to write JSON results.")
    args = parser.parse_args()

    if args.mode == "interpret":
        res = run_qwen35_interpret(moe_tp=args.moe_tp)
    elif args.mode == "aot":
        res = compile_qwen35_aot(moe_tp=args.moe_tp, include_pooled=not args.no_pooled)
    else:
        res = bench_qwen35_tpu(moe_tp=args.moe_tp)

    print(json.dumps(res, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

