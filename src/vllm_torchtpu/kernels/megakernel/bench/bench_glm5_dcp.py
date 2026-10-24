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
"""M0 benchmark & verification harness for frozen GLM-5.2 DCP megakernel (`_frozen.glm5_mk_dcp`).

Supports three execution modes:
  1. `interpret`: CPU `interpret=True` on `TINY` (R=4, L=5, first_dense=1) across 4 virtual CPU devices.
  2. `aot`: TPU7x (`tpu7x:2x2x2`, 16 ranks) ahead-of-time XLA/Mosaic compilation of `FULL` (R=16, L=78).
  3. `tpu`: Hardware execution on TPU v7x-16.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.megakernel._frozen.glm5_mk_dcp import config as C
from vllm_torchtpu.kernels.megakernel._frozen.glm5_mk_dcp import kernel as K
from vllm_torchtpu.kernels.megakernel._frozen.glm5_mk_dcp import layout as L
from vllm_torchtpu.kernels.megakernel._frozen.glm5_mk_dcp import wrapper as W


def default_stack_w(cfg: C.MKDConfig) -> frozenset[str]:
    """Default stacked weight kinds (all static weights except per-layer routed experts ew13/ew2)."""
    wk = K.weight_kinds(cfg)
    return frozenset(set(wk) - {"ew13", "ew2"})


def make_random_glm5_dcp_inputs(
    cfg: C.MKDConfig,
    *,
    nb: int = 16,
    max_pages: int = 8,
    stack_w: frozenset[str] | None = None,
    stack_c: Any = None,
    seed: int = 0,
) -> tuple[list[jax.Array], dict[str, Any]]:
    """Build deterministic random operands `[R, ...]` for `wrapper.make(cfg, ...)`."""
    rng = np.random.default_rng(seed)
    stack_w = default_stack_w(cfg) if stack_w is None else stack_w
    _, info = K.build(
        cfg,
        nb=nb,
        max_pages=max_pages,
        bmax=cfg.TMAX,
        stack_w=stack_w,
        stack_c=stack_c,
        interpret=True,
    )
    R, TMAX, M, DR = cfg.R, cfg.TMAX, info["M"], cfg.DR
    BMAX = TMAX

    # Simulate 2 active requests of 2 tokens each (total T=4 <= TMAX)
    n_reqs = min(2, BMAX)
    q_per_req = min(2, cfg.QMAX, TMAX // max(1, n_reqs))
    T_live = n_reqs * q_per_req

    scal_1 = np.zeros((8,), dtype=np.int32)
    scal_1[0] = T_live
    scal_1[1] = n_reqs

    rmeta_1 = np.zeros((TMAX, 8), dtype=np.int32)
    qmeta_1 = np.zeros((BMAX, 4), dtype=np.int32)
    bt_1 = np.zeros((BMAX, max_pages), dtype=np.int32)
    for r_idx in range(n_reqs):
        seq_len = 64 + 16 * r_idx
        t0 = r_idx * q_per_req
        qmeta_1[r_idx] = [t0, q_per_req, seq_len, 1]
        for p_idx in range(max_pages):
            bt_1[r_idx, p_idx] = (r_idx * max_pages + p_idx + 1) % nb
        for q_i in range(q_per_req):
            row = t0 + q_i
            pos = seq_len - q_per_req + q_i
            rmeta_1[row] = [1, pos, r_idx, seq_len, 0, 0, 0, 0]

    rvec_1 = np.zeros((M, 128), dtype=np.int32)
    rvec_1[:TMAX, 0] = rmeta_1[:, 0]
    rvec_1[:TMAX, 1] = rmeta_1[:, 1]

    h0_1 = (rng.standard_normal((TMAX, cfg.H), dtype=np.float32) * 0.1).astype(np.float32)
    h0_1[T_live:] = 0.0

    # Identity RoPE table [4, M, 128]: cos=1, sin=0
    tab_1 = np.zeros((4, M, 128), dtype=np.float32)
    tab_1[0] = 1.0
    tab_1[2] = 1.0

    # Smalls [R, L + 1, 1, S]
    lay, S = L.smalls_layout(cfg)
    smalls_r = np.zeros((R, cfg.L + 1, 1, S), dtype=np.float32)
    for name, (off, width) in lay.items():
        if "ln" in name or name == "idx_kn_w":
            smalls_r[:, :, 0, off : off + width] = 0.1
        elif name.endswith("_s"):
            smalls_r[:, :, 0, off : off + width] = 0.01

    def _rep(arr: np.ndarray) -> jax.Array:
        return jnp.asarray(np.tile(arr[None, ...], (R,) + (1,) * arr.ndim))

    fixed = [
        _rep(scal_1),
        _rep(rmeta_1),
        _rep(qmeta_1),
        _rep(bt_1),
        _rep(rvec_1),
        _rep(h0_1),
        _rep(tab_1),
        jnp.asarray(smalls_r),
    ]

    def _rand_w(shape: tuple[int, ...], dtype: Any) -> jax.Array:
        dt = jnp.dtype(dtype)
        if dt == jnp.dtype(K.FP8):
            arr = np.clip(rng.standard_normal(shape, dtype=np.float32) * 2.0, -16.0, 16.0)
            return jnp.asarray(arr).astype(K.FP8)
        if dt == jnp.dtype(K.BF16):
            arr = rng.standard_normal(shape, dtype=np.float32) * 0.05
            return jnp.asarray(arr).astype(K.BF16)
        return jnp.asarray(np.full(shape, 0.01, dtype=np.float32))

    wk = info["wk"]
    weights = []
    for k, tag in info["w_names"]:
        layers, sh, dt = wk[k]
        full_sh = (R, len(layers), *sh) if tag == "stk" else (R, *sh)
        weights.append(_rand_w(full_sh, dt))

    ck = info["ck"]
    scg = K.cache_groups(stack_c, ck)
    caches = []
    for k, tag in info["c_names"]:
        layers, sh, dt = ck[k]
        if isinstance(tag, str):
            gs = len(layers) // scg[k]
            c_sh = (R, gs, *sh)
        else:
            c_sh = (R, *sh)
        caches.append(jnp.zeros(c_sh, dtype=dt))

    return fixed + weights + caches, info


def make_abstract_glm5_dcp_inputs(
    mesh: Mesh,
    cfg: C.MKDConfig,
    *,
    nb: int = 64,
    max_pages: int = 16,
    stack_w: frozenset[str] | None = None,
    stack_c: Any = None,
) -> tuple[list[jax.ShapeDtypeStruct], dict[str, Any]]:
    """Build zero-memory `jax.ShapeDtypeStruct` descriptors `[R, ...]` for `wrapper.make` AOT compile."""
    stack_w = default_stack_w(cfg) if stack_w is None else stack_w
    _, info = K.build(
        cfg,
        nb=nb,
        max_pages=max_pages,
        bmax=cfg.TMAX,
        stack_w=stack_w,
        stack_c=stack_c,
        interpret=False,
    )
    R, TMAX, M = cfg.R, cfg.TMAX, info["M"]
    BMAX = TMAX
    S = info["S"]
    sh_d = NamedSharding(mesh, P(mesh.axis_names[0]))

    def _sd(shape_no_r: tuple[int, ...], dtype: Any) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((R, *shape_no_r), jnp.dtype(dtype), sharding=sh_d)

    fixed = [
        _sd((8,), K.I32),
        _sd((TMAX, 8), K.I32),
        _sd((BMAX, 4), K.I32),
        _sd((BMAX, max_pages), K.I32),
        _sd((M, 128), K.I32),
        _sd((TMAX, cfg.H), K.F32),
        _sd((4, M, 128), K.F32),
        _sd((cfg.L + 1, 1, S), K.F32),
    ]

    wk = info["wk"]
    weights = []
    for k, tag in info["w_names"]:
        layers, sh, dt = wk[k]
        full_sh = (len(layers), *sh) if tag == "stk" else sh
        weights.append(_sd(full_sh, dt))

    ck = info["ck"]
    scg = K.cache_groups(stack_c, ck)
    caches = []
    for k, tag in info["c_names"]:
        layers, sh, dt = ck[k]
        if isinstance(tag, str):
            gs = len(layers) // scg[k]
            c_sh = (gs, *sh)
        else:
            c_sh = sh
        caches.append(_sd(c_sh, dt))

    return fixed + weights + caches, info


def run_glm5_dcp_interpret(*, full_tiny: bool = False) -> list[dict[str, Any]]:
    """Run CPU `interpret=True` verification on `CPU_TINY` (R=2, L=3) or `TINY` (R=4, L=5)."""
    cfg = (
        C.TINY
        if full_tiny
        else dataclasses.replace(
            C.TINY,
            R=2,
            L=3,
            first_dense=1,
            NH=4,
            E=4,
            I=128,
            IS=256,
            ID=256,
            IH=2,
            idx_freq=2,
            idx_skip=1,
        )
    )
    devs = jax.devices()
    if len(devs) < cfg.R:
        raise RuntimeError(
            f"Need at least {cfg.R} JAX devices, got {len(devs)}. "
            f"Set XLA_FLAGS='--xla_force_host_platform_device_count={cfg.R}'."
        )
    mesh = Mesh(np.array(devs[: cfg.R]), ("d",))
    nb, max_pages = 8, 4
    stack_w = default_stack_w(cfg)
    stack_c = {"nope": 1, "rope": 1, "idx": 1}

    ops, _ = make_random_glm5_dcp_inputs(
        cfg, nb=nb, max_pages=max_pages, stack_w=stack_w, stack_c=stack_c, seed=0
    )
    t0 = time.perf_counter()
    fn, info = W.make(
        cfg,
        mesh,
        axis="d",
        donate=False,
        nb=nb,
        max_pages=max_pages,
        bmax=cfg.TMAX,
        stack_w=stack_w,
        stack_c=stack_c,
        interpret=True,
    )
    outs = fn(*ops)
    out_h = outs[0]
    jax.block_until_ready(out_h)
    wall_s = time.perf_counter() - t0

    out_np = np.asarray(out_h[:, : cfg.TMAX], dtype=np.float32)
    assert np.all(np.isfinite(out_np)), "Non-finite output in GLM-5.2 DCP TINY interpret"
    # Verify rank 0 and rank 1 produce identical all-gathered output rows
    max_rank_diff = float(np.max(np.abs(out_np[0] - out_np[1])))
    assert max_rank_diff == 0.0, f"Rank output mismatch: {max_rank_diff}"

    return [
        {
            "mode": "interpret",
            "config": "TINY",
            "R": cfg.R,
            "L": cfg.L,
            "TMAX": cfg.TMAX,
            "vmem_scratch_mib": round(info["vmem_scratch_bytes"] / (1024 * 1024), 2),
            "wall_s": round(wall_s, 3),
            "out_l2_norm": round(float(np.linalg.norm(out_np[0])), 4),
        }
    ]


def compile_glm5_dcp_aot() -> list[dict[str, Any]]:
    """Run TPU7x (`tpu7x:2x2x2`, 16 ranks) AOT compilation on `FULL` (R=16, L=78)."""
    from jax.experimental import topologies

    topo = topologies.get_topology_desc(topology_name="tpu7x:2x2x2", platform="tpu")
    cfg = C.FULL
    mesh = Mesh(np.array(topo.devices[: cfg.R]), ("d",))
    nb, max_pages = 64, 16
    stack_w = default_stack_w(cfg)
    stack_c = {"nope": 2, "rope": 2, "idx": 1}

    abs_ops, _ = make_abstract_glm5_dcp_inputs(
        mesh, cfg, nb=nb, max_pages=max_pages, stack_w=stack_w, stack_c=stack_c
    )
    t0 = time.perf_counter()
    fn, info = W.make(
        cfg,
        mesh,
        axis="d",
        donate=True,
        nb=nb,
        max_pages=max_pages,
        bmax=cfg.TMAX,
        stack_w=stack_w,
        stack_c=stack_c,
        interpret=False,
    )
    compiled = fn.lower(*abs_ops).compile()
    compile_s = time.perf_counter() - t0
    hlo_lines = len(compiled.as_text().splitlines())

    return [
        {
            "mode": "aot_tpu7x",
            "config": "FULL",
            "R": cfg.R,
            "L": cfg.L,
            "TMAX": cfg.TMAX,
            "vmem_scratch_mib": round(info["vmem_scratch_bytes"] / (1024 * 1024), 2),
            "compile_s": round(compile_s, 2),
            "hlo_lines": hlo_lines,
            "n_weights": len(info["w_names"]),
            "n_caches": len(info["c_names"]),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frozen GLM-5.2 DCP megakernel.")
    parser.add_argument(
        "--mode",
        choices=("interpret", "aot"),
        default="interpret",
        help="Execution mode: interpret (CPU TINY) or aot (TPU7x FULL 16-rank compile).",
    )
    parser.add_argument("--json-out", type=str, default="", help="Optional path to write JSON results.")
    args = parser.parse_args()

    if args.mode == "interpret":
        res = run_glm5_dcp_interpret()
    else:
        res = compile_glm5_dcp_aot()

    print(json.dumps(res, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

