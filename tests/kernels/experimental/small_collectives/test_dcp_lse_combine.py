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
"""Unit, Pallas TPU interpreter + race-detector, and TPU v7x Mosaic lowering
tests for ``dcp_online_lse_combine`` and its MLA adapter wiring."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

_REPO_SRC = str(Path(__file__).resolve().parents[4] / "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from vllm_torchtpu.kernels.experimental.small_collectives import (  # noqa: E402
    dcp_lse_combine as comb,
)

if __package__:
    from .oracle import AXIS, N, bf16_ulp
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from oracle import AXIS, N, bf16_ulp

TOKEN_BUCKETS = (16, 32, 64, 80, 96)


class _FakeDevice:
    def __init__(self, coords: tuple[int, ...], core_on_chip: int):
        self.coords = coords
        self.core_on_chip = core_on_chip


def _make_inputs(
    world: int,
    tokens: int,
    local: int,
    vdim: int,
    *,
    live: int | None = None,
    seed: int = 0,
):
    """Synthetic per-rank `[world, T, H, D]` bf16 partials and `[world, T, H]`
    f32 LSE with empty-shard `-inf`/NaN entries, an all-empty row, and NaN
    garbage in dead rows `>= live`."""
    rng = np.random.default_rng(seed)
    heads = world * local
    part = rng.standard_normal((world, tokens, heads, vdim), dtype=np.float32)
    lse = (rng.standard_normal((world, tokens, heads), dtype=np.float32) * 2.5).astype(
        np.float32
    )
    empty_rank = rng.random((world, tokens, heads)) < 0.25
    lse[empty_rank] = -np.inf
    part[empty_rank] = np.nan
    if tokens >= 2:
        lse[:, 1, : min(heads, 3)] = -np.inf
        part[:, 1, : min(heads, 3)] = np.nan
    part_bf16 = part.astype(jnp.bfloat16)
    if live is not None and live < tokens:
        dead = np.array(np.nan, dtype=jnp.bfloat16)
        part_bf16[:, live:, :, :4] = dead
        lse[:, live:, :] = 80.0
    return part_bf16, lse


def _weighted_magnitude_f64(
    partials: np.ndarray, lses: np.ndarray, live_tokens: int | None = None
) -> np.ndarray:
    """Per-element `sum_r w_r |O_r|` in float64 `[world, T, local, D]`, the
    pre-cancellation magnitude bounding intermediate Stage-0 bf16 and f32 sum
    rounding (matching `oracle.sum_tolerance`)."""
    p_f64 = np.asarray(jax.device_get(partials).astype(jnp.float32), dtype=np.float64)
    l_f64 = np.asarray(lses, dtype=np.float64)
    world, tokens, heads, vdim = p_f64.shape
    local = heads // world
    finite = np.isfinite(l_f64)
    m = np.max(np.where(finite, l_f64, -np.inf), axis=0, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(invalid="ignore", over="ignore"):
        exps = np.where(finite, np.exp(l_f64 - m_safe), 0.0)
    denom = np.sum(exps, axis=0, keepdims=True)
    weights = np.where(denom > 0.0, exps / np.where(denom > 0.0, denom, 1.0), 0.0)
    p_abs = np.where(np.isnan(p_f64) | ~finite[..., None], 0.0, np.abs(p_f64))
    mag = np.sum(p_abs * weights[..., None], axis=0)
    if live_tokens is not None:
        mag[max(0, min(live_tokens, tokens)) :] = 0.0
    return mag.reshape(tokens, world, local, vdim).transpose(1, 0, 2, 3)


def _metrics_vs_f64(got_bf16: np.ndarray, ref_bf16: np.ndarray, live: int) -> dict:
    got_live = got_bf16[:, :live].astype(np.float64)
    ref_live = ref_bf16[:, :live].astype(np.float64)
    if live == 0 or np.linalg.norm(ref_live) == 0.0:
        rel_l2 = 0.0
        cos = 1.0
        max_abs = float(np.max(np.abs(got_live - ref_live))) if live > 0 else 0.0
    else:
        diff = got_live - ref_live
        rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(ref_live))
        cos = float(
            np.vdot(got_live, ref_live)
            / (np.linalg.norm(got_live) * np.linalg.norm(ref_live))
        )
        max_abs = float(np.max(np.abs(diff)))
    dead = got_bf16[:, live:].astype(np.float32)
    dead_positive_zero = bool(
        dead.size == 0 or np.all((dead == 0.0) & ~np.signbit(dead))
    )
    return {
        "rel_l2": rel_l2,
        "cos": cos,
        "max_abs": max_abs,
        "nan": int(np.isnan(got_bf16.astype(np.float32)).sum()),
        "dead_positive_zero": dead_positive_zero,
    }


def test_supported_geometries_and_validation():
    for tokens in TOKEN_BUCKETS:
        for world, local in ((8, 8), (8, 4), (8, 1), (16, 4), (16, 1)):
            heads = world * local
            for vdim in (256, 512):
                assert comb.online_lse_combine_supported(
                    tokens, heads, world, vdim, mode="pairs"
                )
                assert comb.online_lse_combine_supported(
                    tokens, heads, world, vdim, mode="flat"
                )

    assert not comb.online_lse_combine_supported(8, 64, 16, 512, mode="pairs")
    assert not comb.online_lse_combine_supported(24, 64, 16, 512, mode="flat")
    assert not comb.online_lse_combine_supported(16, 64, 16, 64, mode="pairs")
    assert not comb.online_lse_combine_supported(16, 12, 3, 256, mode="pairs")
    assert comb.online_lse_combine_supported(16, 12, 3, 256, mode="flat")
    assert not comb.online_lse_combine_supported(16, 4, 2, 256, mode="pairs")
    assert comb.online_lse_combine_supported(16, 4, 2, 256, mode="flat")
    assert not comb.online_lse_combine_supported(16, 64, 16, 256, mode="unknown")
    # Header capacity: tokens * local <= 8 * vdim -> 512 * 8 = 4096 > 8 * 256.
    assert not comb.online_lse_combine_supported(512, 64, 8, 256, mode="pairs")

    part = jnp.zeros((16, 64, 256), jnp.bfloat16)
    lse = jnp.zeros((16, 64), jnp.float32)
    with pytest.raises(ValueError, match="bf16 partial"):
        comb.dcp_online_lse_combine(
            part.astype(jnp.float32), lse, num_cores=8, axis="dcp"
        )
    with pytest.raises(ValueError, match="f32 lse"):
        comb.dcp_online_lse_combine(
            part, lse.astype(jnp.bfloat16), num_cores=8, axis="dcp"
        )
    with pytest.raises(ValueError, match="does not match"):
        comb.dcp_online_lse_combine(
            part, jnp.zeros((16, 32), jnp.float32), num_cores=8, axis="dcp"
        )


def test_mesh_has_chip_pairs_1d_and_2d():
    class _FakeMesh:
        def __init__(self, devices, axis_names):
            self.devices = np.asarray(devices, dtype=object)
            self.axis_names = axis_names
            self.shape = dict(zip(axis_names, self.devices.shape, strict=True))

    paired_8 = [_FakeDevice((i // 2, 0, 0), i % 2) for i in range(8)]
    assert comb.mesh_has_chip_pairs(_FakeMesh(paired_8, ("dcp",)), "dcp")
    assert not comb.mesh_has_chip_pairs(_FakeMesh(paired_8, ("dcp",)), "other")

    scrambled_8 = [
        paired_8[0],
        paired_8[2],
        paired_8[1],
        paired_8[3],
        *paired_8[4:],
    ]
    assert not comb.mesh_has_chip_pairs(_FakeMesh(scrambled_8, ("dcp",)), "dcp")

    paired_2x8 = np.array(
        [
            [_FakeDevice((0, i // 2, 0), i % 2) for i in range(8)],
            [_FakeDevice((1, i // 2, 0), i % 2) for i in range(8)],
        ],
        dtype=object,
    )
    assert comb.mesh_has_chip_pairs(_FakeMesh(paired_2x8, ("dcp_lane", "dcp")), "dcp")
    bad_lane = paired_2x8.copy()
    bad_lane[1, 0], bad_lane[1, 2] = bad_lane[1, 2], bad_lane[1, 0]
    assert not comb.mesh_has_chip_pairs(_FakeMesh(bad_lane, ("dcp_lane", "dcp")), "dcp")


def test_envs_defaults_and_oproj_limit_m(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce

    assert fp8_oproj_all_reduce.LIMIT_M == 96
    monkeypatch.delenv("TPU_DCP_ONLINE_LSE_COMBINE", raising=False)
    monkeypatch.delenv("TPU_DCP_ONLINE_LSE_COMBINE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("TPU_DCP_ONLINE_LSE_COMBINE_MIN_TOKENS", raising=False)
    monkeypatch.delenv("TPU_DCP_ONLINE_LSE_COMBINE_LIVE_TOKENS", raising=False)
    monkeypatch.delenv("TPU_DCP_ONLINE_LSE_COMBINE_OUT_VMEM", raising=False)
    assert envs.TPU_DCP_ONLINE_LSE_COMBINE == ""
    assert envs.TPU_DCP_ONLINE_LSE_COMBINE_MAX_TOKENS == 96
    assert envs.TPU_DCP_ONLINE_LSE_COMBINE_MIN_TOKENS == 16
    assert envs.TPU_DCP_ONLINE_LSE_COMBINE_LIVE_TOKENS is True
    assert envs.TPU_DCP_ONLINE_LSE_COMBINE_OUT_VMEM is True
    assert envs.bucket_gate_edge(envs.TPU_DCP_ONLINE_LSE_COMBINE_MIN_TOKENS) == 8


def test_adapter_env_gate_and_uv_compat(monkeypatch):
    pytest.importorskip("torch")
    import torch

    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp

    fake_group = types.SimpleNamespace(
        world_size=16, rank_in_group=0, ranks=list(range(16))
    )
    monkeypatch.setattr(cp, "get_dcp_group", lambda: fake_group)

    # Default-off: unset env leaves all predicates False / unchanged.
    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", "")
    assert cp.online_lse_combine_mode_requested() == ""
    for t in (4, 8, 16, 32, 64, 80, 96, 128):
        assert not cp.online_lse_combine_applies(t, True)
        assert not cp.online_lse_combine_applies(t, False)

    for raw, want in (
        ("0", ""),
        ("off", ""),
        ("false", ""),
        ("1", "auto"),
        ("auto", "auto"),
        ("PAIRS", "pairs"),
        ("flat", "flat"),
    ):
        monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", raw)
        assert cp.online_lse_combine_mode_requested() == want

    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", "invalid_mode")
    with pytest.raises(ValueError, match="TPU_DCP_ONLINE_LSE_COMBINE"):
        cp.online_lse_combine_mode_requested()

    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", "pairs")
    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE_MIN_TOKENS", "16")
    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE_MAX_TOKENS", "96")
    for t in (1, 4, 8, 128, 256):
        assert not cp.online_lse_combine_applies(t, True)
    for t in TOKEN_BUCKETS:
        assert cp.online_lse_combine_applies(t, True)
        assert not cp.online_lse_combine_applies(t, False)
        assert not cp.send_layout_applies(t, 64, 512, torch.bfloat16, True)
        assert not cp.vspace_fused_applies(t, True)


def _run_16cpu_worker() -> dict:
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count=16"
        " --xla_allow_excess_precision=false",
    )
    proc = subprocess.run(
        [sys.executable, __file__, "worker16"],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-4000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def worker16_results():
    return _run_16cpu_worker()


def test_interpret_dcp8_and_dcp16_race_free_and_accurate(worker16_results):
    assert not worker16_results["wrong_rank"]["matches_ref"]
    for key, case in sorted(worker16_results["interpret"].items()):
        assert not case["races_found"], (key, case)
        assert case["nan"] == 0, (key, case)
        assert case["dead_positive_zero"], (key, case)
        assert case["rel_l2"] < 8e-3, (key, case)
        assert case["cos"] > 0.99995, (key, case)
        assert case["within_f64_bound"], (key, case)
        if "pairs_exact_frac" in case:
            assert case["pairs_within_2ulp"], (key, case)
            assert case["pairs_exact_frac"] >= 0.999, (key, case)


def test_tpu7x_mosaic_lowering_and_fp8_oproj_chain(worker16_results):
    lowered = worker16_results["lowered"]
    assert len(lowered) >= len(TOKEN_BUCKETS) * 4
    for key, info in sorted(lowered.items()):
        assert info["ok"], (key, info)
        assert not info["has_partition_id"], (key, info)
    for key, info in sorted(worker16_results["chained"].items()):
        assert info["ok"], (key, info)
        assert not info["has_partition_id"], (key, info)


@pytest.mark.multichip
@pytest.mark.parametrize("mode", ["pairs", "flat"])
@pytest.mark.parametrize("tokens", TOKEN_BUCKETS)
def test_multichip_tpu_dcp_online_lse_combine(mesh, mode, tokens):
    """Compiled execution on the 8 TPU cores of one host across T=16..96."""
    local = 8
    vdim = 512
    live = max(0, tokens - 5)
    seed = 100 + tokens + (0 if mode == "pairs" else 1)
    part, lse = _make_inputs(N, tokens, local, vdim, live=live, seed=seed)
    ref_f64 = np.asarray(comb.reference_float64(part, lse, live_tokens=live))
    sharding = NamedSharding(mesh, P(AXIS))
    p_dev = jax.device_put(
        jnp.asarray(part).reshape(N * tokens, N * local, vdim), sharding
    )
    l_dev = jax.device_put(jnp.asarray(lse).reshape(N * tokens, N * local), sharding)
    r_dev = jax.device_put(jnp.arange(N, dtype=jnp.int32), sharding)
    lv_dev = jnp.asarray([live], dtype=jnp.int32)

    fn = jax.jit(
        jax.shard_map(
            lambda p, ls, r, lv: comb.dcp_online_lse_combine(
                p,
                ls,
                num_cores=N,
                axis=AXIS,
                mode=mode,
                rank=r,
                live_tokens=lv,
                out_vmem=False,
            ),
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS), P(AXIS), P()),
            out_specs=P(AXIS),
            check_vma=False,
        )
    )
    got = np.asarray(fn(p_dev, l_dev, r_dev, lv_dev)).reshape(N, tokens, local, vdim)
    metrics = _metrics_vs_f64(got, ref_f64, live)
    assert metrics["nan"] == 0 and metrics["dead_positive_zero"], metrics
    assert metrics["rel_l2"] < 8e-3 and metrics["cos"] > 0.99995, metrics


def _worker16_main():
    from jax._src.pallas.mosaic import tpu_info
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.interpreters import mlir

    from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce as oproj_ar

    devices = jax.devices()
    assert len(devices) == 16, len(devices)
    mesh16 = Mesh(np.asarray(devices), ("dcp",))
    mesh8 = Mesh(np.asarray(devices[:8]), ("dcp",))
    mesh2x8 = Mesh(np.asarray(devices).reshape(2, 8), ("dcp_lane", "dcp"))

    def run_interpret_case(
        mesh_obj,
        world: int,
        tokens: int,
        local: int,
        vdim: int,
        *,
        mode: str,
        live: int | None,
        dma_mode: str = "on_wait",
        use_rank_data: bool = True,
        multi_lane: bool = False,
        seed: int = 0,
    ) -> dict:
        lanes = 2 if multi_lane else 1
        heads = world * local
        p_all = []
        l_all = []
        refs_f64 = []
        mags_f64 = []
        refs_pairs = []
        eff_live = tokens if live is None else live
        for ln in range(lanes):
            part, lse = _make_inputs(
                world, tokens, local, vdim, live=live, seed=seed + 97 * ln
            )
            p_all.append(part)
            l_all.append(lse)
            refs_f64.append(
                np.asarray(comb.reference_float64(part, lse, live_tokens=live))
            )
            mags_f64.append(_weighted_magnitude_f64(part, lse, live_tokens=live))
            if mode == "pairs":
                refs_pairs.append(
                    np.asarray(comb.reference_pairs_f32(part, lse, live_tokens=live))
                )

        if multi_lane:
            spec = P("dcp_lane", "dcp")
            p_in = jax.device_put(
                jnp.asarray(np.stack(p_all, axis=0)), NamedSharding(mesh_obj, spec)
            )
            l_in = jax.device_put(
                jnp.asarray(np.stack(l_all, axis=0)), NamedSharding(mesh_obj, spec)
            )
            r_in = jax.device_put(
                jnp.broadcast_to(
                    jnp.arange(world, dtype=jnp.int32)[None, :], (2, world)
                ),
                NamedSharding(mesh_obj, spec),
            )
            ln_in = jax.device_put(
                jnp.broadcast_to(jnp.arange(2, dtype=jnp.int32)[:, None], (2, world)),
                NamedSharding(mesh_obj, spec),
            )
            lv_in = jnp.asarray([eff_live], dtype=jnp.int32)

            def body(p, ls, r, ln_id, lv):
                return comb.dcp_online_lse_combine(
                    p[0, 0],
                    ls[0, 0],
                    num_cores=world,
                    axis="dcp",
                    mode=mode,
                    rank=jnp.reshape(r, (1,)),
                    lane_axis="dcp_lane",
                    lane=jnp.reshape(ln_id, (1,)),
                    live_tokens=lv if live is not None else None,
                    out_vmem=True,
                )[None, None]

            fn = jax.jit(
                jax.shard_map(
                    body,
                    mesh=mesh_obj,
                    in_specs=(spec, spec, spec, spec, P()),
                    out_specs=spec,
                    check_vma=False,
                )
            )
            call_args = (p_in, l_in, r_in, ln_in, lv_in)
        else:
            spec = P("dcp")
            p_in = jax.device_put(
                jnp.asarray(p_all[0]).reshape(world * tokens, heads, vdim),
                NamedSharding(mesh_obj, spec),
            )
            l_in = jax.device_put(
                jnp.asarray(l_all[0]).reshape(world * tokens, heads),
                NamedSharding(mesh_obj, spec),
            )
            r_in = jax.device_put(
                jnp.arange(world, dtype=jnp.int32), NamedSharding(mesh_obj, spec)
            )
            lv_in = jnp.asarray([eff_live], dtype=jnp.int32)

            def body(p, ls, r, lv):
                return comb.dcp_online_lse_combine(
                    p,
                    ls,
                    num_cores=world,
                    axis="dcp",
                    mode=mode,
                    rank=r if use_rank_data else None,
                    live_tokens=lv if live is not None else None,
                    out_vmem=True,
                )

            fn = jax.jit(
                jax.shard_map(
                    body,
                    mesh=mesh_obj,
                    in_specs=(spec, spec, spec, P()),
                    out_specs=spec,
                    check_vma=False,
                )
            )
            call_args = (p_in, l_in, r_in, lv_in)

        params = pltpu.InterpretParams(detect_races=True, dma_execution_mode=dma_mode)
        pltpu.reset_tpu_interpret_mode_state()
        with (
            pltpu.force_tpu_interpret_mode(params),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            raw = np.asarray(jax.block_until_ready(fn(*call_args)))
        races = bool(interpret_pallas_call.races.races_found)

        got = raw.reshape(lanes, world, tokens, local, vdim)
        ref_f64 = np.stack(refs_f64, axis=0)
        mag_f64 = np.stack(mags_f64, axis=0)
        m = _metrics_vs_f64(
            got.reshape(lanes * world, tokens, local, vdim),
            ref_f64.reshape(lanes * world, tokens, local, vdim),
            eff_live,
        )
        ref_live = ref_f64[:, :, :eff_live].astype(np.float64)
        mag_live = mag_f64[:, :, :eff_live]
        # In `pairs`, Stage 0 rounds each chip partial (bounded by `mag_live`)
        # to bf16 before Stage 1 rounds at `|ref_live|`; `flat` rounds once.
        stage0_err = (
            bf16_ulp(mag_live) if mode == "pairs" else world * 2.0**-22 * mag_live
        )
        tol = bf16_ulp(np.abs(ref_live) + stage0_err) + stage0_err + 1e-6
        m["within_f64_bound"] = bool(
            eff_live == 0
            or np.all(np.abs(got[:, :, :eff_live].astype(np.float64) - ref_live) <= tol)
        )
        m["races_found"] = races
        if mode == "pairs":
            ref_p = np.stack(refs_pairs, axis=0)
            g_live = got[:, :, :eff_live].astype(np.float64)
            rp_live = ref_p[:, :, :eff_live].astype(np.float64)
            ptol = bf16_ulp(np.maximum(np.abs(rp_live), mag_live)) + bf16_ulp(
                np.maximum(np.abs(rp_live), 1e-6)
            )
            m["pairs_within_2ulp"] = bool(
                eff_live == 0 or np.all(np.abs(g_live - rp_live) <= ptol)
            )
            m["pairs_exact_frac"] = (
                1.0
                if eff_live == 0
                else float(
                    np.mean(
                        got[:, :, :eff_live].view(np.int16)
                        == ref_p[:, :, :eff_live].astype(jnp.bfloat16).view(np.int16)
                    )
                )
            )
        return m

    interpret_cases = {}
    # Sweep DCP16 across T = 16, 32, 64, 80, 96 (pairs + flat, sized + full).
    for idx, tokens in enumerate(TOKEN_BUCKETS):
        for mode in ("pairs", "flat"):
            live = max(0, tokens - 7)
            dma_m = "eager" if (idx % 2 == 1 and mode == "pairs") else "on_wait"
            interpret_cases[f"w16_t{tokens}_l4_d256_{mode}_live{live}"] = (
                run_interpret_case(
                    mesh16,
                    16,
                    tokens,
                    4,
                    256,
                    mode=mode,
                    live=live,
                    dma_mode=dma_m,
                    seed=10 + idx,
                )
            )
    # Full D=512 latent width at T=16, 64, 96 on DCP16, plus live=0 edge case.
    for tokens, live in ((16, None), (16, 0), (64, 49), (96, 83)):
        interpret_cases[f"w16_t{tokens}_l4_d512_pairs_live{live}"] = run_interpret_case(
            mesh16, 16, tokens, 4, 512, mode="pairs", live=live, seed=200 + tokens
        )
    # DCP8 1-lane (local=8, D=512) across T=16..96 + 2-lane TP16xDCP8 + axis_index.
    for tokens in TOKEN_BUCKETS:
        interpret_cases[f"w8_t{tokens}_l8_d512_pairs"] = run_interpret_case(
            mesh8,
            8,
            tokens,
            8,
            512,
            mode="pairs",
            live=tokens - 3,
            seed=300 + tokens,
        )
    interpret_cases["w2x8_t32_l4_d256_pairs"] = run_interpret_case(
        mesh2x8,
        8,
        32,
        4,
        256,
        mode="pairs",
        live=21,
        multi_lane=True,
        seed=401,
    )
    interpret_cases["w8_t16_l4_d256_axis_index"] = run_interpret_case(
        mesh8,
        8,
        16,
        4,
        256,
        mode="pairs",
        live=None,
        use_rank_data=False,
        seed=402,
    )

    # Positive control: swapping rank 0 and rank 1 must fail the oracle on rank 0.
    part_bad, lse_bad = _make_inputs(8, 16, 4, 256, live=None, seed=505)
    ref_bad = np.asarray(comb.reference_float64(part_bad, lse_bad))[0].astype(
        np.float64
    )
    sh8 = NamedSharding(mesh8, P("dcp"))
    swapped_ranks = jax.device_put(
        jnp.asarray([1, 0, 2, 3, 4, 5, 6, 7], dtype=jnp.int32), sh8
    )
    fn_bad = jax.jit(
        jax.shard_map(
            lambda p, ls, r: comb.dcp_online_lse_combine(
                p, ls, num_cores=8, axis="dcp", mode="pairs", rank=r
            ),
            mesh=mesh8,
            in_specs=(P("dcp"), P("dcp"), P("dcp")),
            out_specs=P("dcp"),
            check_vma=False,
        )
    )
    pltpu.reset_tpu_interpret_mode_state()
    with (
        pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=False)),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        got_bad = (
            np.asarray(
                fn_bad(
                    jax.device_put(jnp.asarray(part_bad).reshape(8 * 16, 32, 256), sh8),
                    jax.device_put(jnp.asarray(lse_bad).reshape(8 * 16, 32), sh8),
                    swapped_ranks,
                )
            )
            .reshape(8, 16, 4, 256)[0]
            .astype(np.float64)
        )
    wrong_matches = bool(np.max(np.abs(got_bad - ref_bad)) < 1e-2)

    # Register TPU_7X info and verify jax.export lowering across T=16..96.
    kind = devices[0].device_kind
    tpu_info.registry[kind] = lambda: tpu_info.get_tpu_info_for_chip(
        tpu_info.ChipVersion.TPU_7X, 1
    )
    tpu_info.get_tpu_info.cache_clear()

    def export_asm(fn, *sds):
        exported = jax.export.export(jax.jit(fn), platforms=["tpu"])(*sds)
        with mlir.make_ir_context():
            module = mlir.ir.Module.parse(exported.mlir_module())
            return str(module.operation.get_asm(enable_debug_info=False))

    lowered = {}
    S = jax.ShapeDtypeStruct
    for world, mesh_obj, local in ((16, mesh16, 4), (8, mesh8, 8)):
        heads = world * local
        sh = NamedSharding(mesh_obj, P("dcp"))
        sh_rep = NamedSharding(mesh_obj, P())
        for tokens in TOKEN_BUCKETS:
            for vdim in (256, 512):
                for mode in ("pairs", "flat"):
                    key = f"w{world}_t{tokens}_h{heads}_d{vdim}_{mode}"

                    def fn_exp(
                        p,
                        ls,
                        lv,
                        r,
                        *,
                        _m=mesh_obj,
                        _w=world,
                        _md=mode,
                        _ov=(tokens <= 64),
                    ):
                        return jax.shard_map(
                            lambda a, b, c, rk: comb.dcp_online_lse_combine(
                                a,
                                b,
                                num_cores=_w,
                                axis="dcp",
                                mode=_md,
                                rank=rk,
                                live_tokens=c,
                                out_vmem=_ov,
                            ),
                            mesh=_m,
                            in_specs=(P("dcp"), P("dcp"), P(), P("dcp")),
                            out_specs=P("dcp"),
                            check_vma=False,
                        )(p, ls, lv, r)

                    asm = export_asm(
                        fn_exp,
                        S((world * tokens, heads, vdim), jnp.bfloat16, sharding=sh),
                        S((world * tokens, heads), jnp.float32, sharding=sh),
                        S((1,), jnp.int32, sharding=sh_rep),
                        S((world,), jnp.int32, sharding=sh),
                    )
                    lowered[key] = {
                        "ok": "tpu.custom_call" in asm or "custom_call" in asm,
                        "has_partition_id": (
                            "partition_id" in asm or "partition-id" in asm
                        ),
                    }

    # Verify VMEM chaining into fp8_uv_matmul_all_reduce and fp8_matmul_all_reduce
    # across T = 16, 64, 80, 96 (exercising LIMIT_M = 96).
    chained = {}
    sh16 = NamedSharding(mesh16, P("dcp"))
    sh16_rep = NamedSharding(mesh16, P())
    hidden = 6144
    local16 = 4
    heads16 = 16 * local16
    for tokens in (16, 64, 80, 96):
        # 1) Latent combine (D=512, out_vmem=True) -> fp8_uv_matmul_all_reduce
        def fn_uv_chain(p, ls, lv, r, w_uv, w_q, w_s, *, _t=tokens):
            def body(a, b, c, rk, uv, q, s):
                lat = comb.dcp_online_lse_combine(
                    a,
                    b,
                    num_cores=16,
                    axis="dcp",
                    mode="pairs",
                    rank=rk,
                    live_tokens=c,
                    out_vmem=True,
                ).reshape(_t, local16 * 512)
                return oproj_ar.fp8_uv_matmul_all_reduce(
                    lat,
                    uv,
                    q,
                    s,
                    rk,
                    num_cores=16,
                    axis="dcp",
                    out_dtype=jnp.bfloat16,
                    max_m=96,
                )

            return jax.shard_map(
                body,
                mesh=mesh16,
                in_specs=(
                    P("dcp"),
                    P("dcp"),
                    P(),
                    P("dcp"),
                    P("dcp"),
                    P("dcp"),
                    P(),
                ),
                out_specs=P("dcp"),
                check_vma=False,
            )(p, ls, lv, r, w_uv, w_q, w_s)

        asm_uv = export_asm(
            fn_uv_chain,
            S((16 * tokens, heads16, 512), jnp.bfloat16, sharding=sh16),
            S((16 * tokens, heads16), jnp.float32, sharding=sh16),
            S((1,), jnp.int32, sharding=sh16_rep),
            S((16,), jnp.int32, sharding=sh16),
            S((16 * local16, 512, 256), jnp.float8_e4m3fn, sharding=sh16),
            S((16 * local16 * 256, hidden), jnp.float8_e4m3fn, sharding=sh16),
            S((hidden,), jnp.float32, sharding=sh16_rep),
        )
        chained[f"uv_chain_t{tokens}"] = {
            "ok": "custom_call" in asm_uv,
            "has_partition_id": "partition_id" in asm_uv or "partition-id" in asm_uv,
        }

        # 2) V-space combine (D=256, out_vmem=True) -> fp8_matmul_all_reduce
        def fn_v_chain(p, ls, lv, r, w_q, w_s, *, _t=tokens):
            def body(a, b, c, rk, q, s):
                v_out = comb.dcp_online_lse_combine(
                    a,
                    b,
                    num_cores=16,
                    axis="dcp",
                    mode="pairs",
                    rank=rk,
                    live_tokens=c,
                    out_vmem=True,
                ).reshape(_t, local16 * 256)
                return oproj_ar.fp8_matmul_all_reduce(
                    v_out,
                    q,
                    s,
                    rk,
                    num_cores=16,
                    axis="dcp",
                    max_m=96,
                )

            return jax.shard_map(
                body,
                mesh=mesh16,
                in_specs=(P("dcp"), P("dcp"), P(), P("dcp"), P("dcp"), P()),
                out_specs=P("dcp"),
                check_vma=False,
            )(p, ls, lv, r, w_q, w_s)

        asm_v = export_asm(
            fn_v_chain,
            S((16 * tokens, heads16, 256), jnp.bfloat16, sharding=sh16),
            S((16 * tokens, heads16), jnp.float32, sharding=sh16),
            S((1,), jnp.int32, sharding=sh16_rep),
            S((16,), jnp.int32, sharding=sh16),
            S((16 * local16 * 256, hidden), jnp.float8_e4m3fn, sharding=sh16),
            S((hidden,), jnp.float32, sharding=sh16_rep),
        )
        chained[f"v_chain_t{tokens}"] = {
            "ok": "custom_call" in asm_v,
            "has_partition_id": "partition_id" in asm_v or "partition-id" in asm_v,
        }

    print(
        "RESULT "
        + json.dumps(
            {
                "interpret": interpret_cases,
                "wrong_rank": {"matches_ref": wrong_matches},
                "lowered": lowered,
                "chained": chained,
            }
        ),
        flush=True,
    )


if __name__ == "__main__" and sys.argv[1:] == ["worker16"]:
    _worker16_main()
