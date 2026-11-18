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
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

if __name__ == "__main__":
    # The CPU worker subprocesses run this file as a script; make `src`
    # importable there without changing `sys.path` for the pytest session.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from vllm_torchtpu.kernels.experimental.small_collectives import (  # noqa: E402
    dcp_lse_combine as comb,
)
from vllm_torchtpu.kernels.quantized_matmul import (  # noqa: E402
    fp8_oproj_all_reduce as oproj_ar,
)

if __package__:
    from .oracle import AXIS, N, bf16_ulp
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from oracle import AXIS, N, bf16_ulp

TOKEN_BUCKETS = (16, 32, 64, 80, 96)


class _FakeDevice:
    def __init__(
        self, coords: tuple[int, ...], core_on_chip: int, platform: str = "tpu"
    ):
        self.coords = coords
        self.core_on_chip = core_on_chip
        self.platform = platform


class _FakeMesh:
    def __init__(self, devices, axis_names):
        self.devices = np.asarray(devices, dtype=object)
        self.axis_names = axis_names
        self.shape = dict(zip(axis_names, self.devices.shape, strict=True))


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
    # Header capacity: tokens * local <= 8 * vdim -> 96 * 16 = 1536 > 8 * 128.
    assert not comb.online_lse_combine_supported(96, 256, 16, 128, mode="pairs")
    # Single pass only: buckets above MAX_COMBINE_TOKENS keep the XLA merge.
    for tokens in (112, 128, 192, 256, 512):
        for mode in ("pairs", "flat"):
            assert not comb.online_lse_combine_supported(tokens, 64, 16, 512, mode=mode)
    # VMEM budget: 128 heads over 16 cores fit at 32 tokens but not at 96
    # (about 39 MB of buffers against 80% of the 32 MiB scoped limit) unless
    # the kernel is compiled with a larger limit.
    assert comb.online_lse_combine_supported(32, 128, 16, 512, mode="flat")
    assert not comb.online_lse_combine_supported(96, 128, 16, 512, mode="flat")
    assert comb.online_lse_combine_supported(
        96, 128, 16, 512, mode="flat", vmem_limit_bytes=64 * 1024 * 1024
    )

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


def test_geometry_supported_bounds_every_admitted_bucket():
    for mode in ("pairs", "flat"):
        assert comb.online_lse_combine_geometry_supported(64, 16, 512, mode=mode)
        assert comb.online_lse_combine_geometry_supported(64, 8, 512, mode=mode)
        # 128 heads over 16 cores fit at 32 tokens but not at the largest
        # admitted bucket, so trace-time dispatch keeps the XLA merge.
        assert comb.online_lse_combine_supported(32, 128, 16, 512, mode=mode)
        assert not comb.online_lse_combine_geometry_supported(128, 16, 512, mode=mode)
        # Header capacity at the largest bucket: 96 * 16 > 8 * 128.
        assert not comb.online_lse_combine_geometry_supported(256, 16, 128, mode=mode)
    assert not comb.online_lse_combine_geometry_supported(12, 3, 256, mode="pairs")
    assert comb.online_lse_combine_geometry_supported(12, 3, 256, mode="flat")
    assert not comb.online_lse_combine_geometry_supported(64, 16, 512, mode="unknown")


def test_trace_time_gate_keeps_the_token_count_dynamic():
    """MLA dispatch evaluates the gate inside vLLM's compile, which marks the
    token dimension dynamic. Token gates must not pin it (for example a lower
    bound ``T >= 16`` next to another gate's ``T <= 16``), and the gate must not
    use builtins Dynamo cannot model on symbolic ints."""
    torch = pytest.importorskip("torch")

    def route(x):
        t = x.shape[0]
        y = x * 2 if t <= 16 else x  # another decode gate at the 16-token edge
        if (
            t > 8
            and t <= comb.MAX_COMBINE_TOKENS
            and t % comb.TILE_ROWS == 0
            and comb.online_lse_combine_geometry_supported(64, 16, 512, mode="pairs")
        ):
            return y + 1
        return y - 1

    for tokens, want in ((16, 3.0), (32, 2.0), (96, 2.0), (192, 0.0)):
        torch._dynamo.reset()
        x = torch.ones(tokens)
        torch._dynamo.mark_dynamic(x, 0)
        compiled = torch.compile(route, fullgraph=True, backend="eager")
        assert float(compiled(x)[0]) == want


def test_mesh_has_chip_pairs_1d_and_2d():
    paired_8 = [_FakeDevice((i // 2, 0, 0), i % 2) for i in range(8)]
    assert comb.mesh_has_chip_pairs(_FakeMesh(paired_8, ("dcp",)), "dcp")
    assert not comb.mesh_has_chip_pairs(_FakeMesh(paired_8, ("dcp",)), "other")
    # The same placement on non-TPU devices (CPU virtual devices) is not paired.
    cpu_8 = [_FakeDevice((i // 2, 0, 0), i % 2, platform="cpu") for i in range(8)]
    assert not comb.mesh_has_chip_pairs(_FakeMesh(cpu_8, ("dcp",)), "dcp")

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


def test_clean_part_masks_inf_and_nan_padding():
    raw_bf16 = jnp.zeros((16, 128), dtype=jnp.bfloat16)
    raw_bf16 = raw_bf16.at[0, 0].set(jnp.bfloat16(2.5))
    raw_bf16 = raw_bf16.at[0, 1].set(jnp.bfloat16(jnp.inf))
    raw_bf16 = raw_bf16.at[0, 2].set(jnp.bfloat16(-jnp.inf))
    raw_bf16 = raw_bf16.at[0, 3].set(jnp.bfloat16(jnp.nan))
    raw_bf16 = raw_bf16.at[8:, :].set(jnp.bfloat16(-jnp.inf))
    raw_i16 = jax.lax.bitcast_convert_type(raw_bf16, jnp.int16)
    live_mask = (jnp.arange(16)[:, None] < 4).astype(jnp.int32)

    # pltpu.bitcast only lowers inside a kernel, so run the helper in one
    # (TPU interpret mode keeps this on the CPU lane).
    def kernel(part_ref, mask_ref, out_ref):
        out_ref[...] = comb._clean_part(part_ref[...], mask_ref[...] != 0)

    cleaned = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(raw_i16.shape, jnp.float32),
        interpret=pltpu.InterpretParams(),
    )(raw_i16, live_mask)
    assert bool(jnp.all(jnp.isfinite(cleaned)))
    assert float(cleaned[0, 0]) == 2.5
    assert float(cleaned[0, 1]) == 0.0
    assert float(cleaned[0, 2]) == 0.0
    assert float(cleaned[0, 3]) == 0.0
    assert float(jnp.max(jnp.abs(cleaned[4:, :]))) == 0.0


def test_env_default_and_choices(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv("TPU_DCP_ONLINE_LSE_COMBINE", raising=False)
    assert envs.TPU_DCP_ONLINE_LSE_COMBINE == ""
    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", "invalid_mode")
    with pytest.raises(ValueError, match="TPU_DCP_ONLINE_LSE_COMBINE"):
        _ = envs.TPU_DCP_ONLINE_LSE_COMBINE


def test_adapter_env_gate_and_uv_compat(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    import torch

    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp

    fake_group = types.SimpleNamespace(
        world_size=16, rank_in_group=0, ranks=list(range(16)), cpu_group=None
    )
    monkeypatch.setattr(cp, "get_dcp_group", lambda: fake_group)

    # Default-off: unset env leaves all predicates False / unchanged.
    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", "")
    assert cp.online_lse_combine_mode_requested() == ""
    for t in (4, 8, 16, 24, 32, 64, 80, 96, 128):
        assert not cp.online_lse_combine_applies(t, True, num_heads=64, vdim=512)
        assert not cp.online_lse_combine_applies(t, False, num_heads=64, vdim=512)

    for raw, want in (
        ("0", ""),
        ("1", "auto"),
        ("auto", "auto"),
        ("PAIRS", "pairs"),
        ("flat", "flat"),
    ):
        monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", raw)
        assert cp.online_lse_combine_mode_requested() == want

    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", "pairs")
    # Non-multiple-of-16 buckets (24, 40, 56, 72, 88), tiny buckets (<=8),
    # oversized buckets (>96), and float32 partials must return False so
    # send_layout / oproj_all_reduce / vspace_fused are not disabled when the
    # kernel cannot run.
    for t in (1, 4, 8, 24, 40, 56, 72, 88, 128, 256):
        assert not cp.online_lse_combine_applies(t, True, num_heads=64, vdim=512)
    assert cp.online_lse_combine_applies(48, True, num_heads=64, vdim=512)
    for t in TOKEN_BUCKETS:
        assert cp.online_lse_combine_applies(t, True, num_heads=64, vdim=512)
        assert not cp.online_lse_combine_applies(
            t, True, num_heads=64, vdim=128, combine_dtype=torch.float32
        )
        assert not cp.online_lse_combine_applies(t, False, num_heads=64, vdim=512)
        assert not cp.send_layout_applies(t, 64, 512, torch.bfloat16, True)
        assert not cp.vspace_fused_applies(
            t, True, num_heads=64, vdim=128, combine_dtype=torch.bfloat16
        )


def test_adapter_gate_traces_under_torch_compile(monkeypatch):
    """``online_lse_combine_applies`` runs inside vLLM's compile of the MLA
    forward, which marks the token dimension dynamic. The gate must trace
    without pinning the token count next to another gate at the 16-token edge,
    and keep the eager routing."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    import torch

    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp

    fake_group = types.SimpleNamespace(
        world_size=16, rank_in_group=0, ranks=list(range(16)), cpu_group=None
    )
    monkeypatch.setattr(cp, "get_dcp_group", lambda: fake_group)
    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", "pairs")

    def route(x):
        t = x.shape[0]
        y = x * 2 if t <= 16 else x  # another decode gate at the 16-token edge
        if cp.online_lse_combine_applies(t, True, num_heads=64, vdim=512):
            return y + 1
        return y - 1

    for tokens, want in ((16, 3.0), (32, 2.0), (96, 2.0), (192, 0.0)):
        torch._dynamo.reset()
        x = torch.ones(tokens)
        torch._dynamo.mark_dynamic(x, 0)
        compiled = torch.compile(route, fullgraph=True, backend="eager")
        assert float(compiled(x)[0]) == want


@pytest.mark.parametrize(("mode", "lifted"), [("", False), ("pairs", True)])
def test_a2a_merge_lifts_rank_and_lane_only_with_the_combine_on(
    monkeypatch, mode, lifted
):
    """TPU_DCP_LSE_MERGE_A2A's branch returns the all-to-all merge as is with
    the combine off. With it on, an 8-token bucket (which the combine does not
    take) also lifts the rank and lane buffers the combine's buckets read."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    import torch

    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp

    fake_group = types.SimpleNamespace(
        world_size=16, rank_in_group=0, ranks=list(range(16)), cpu_group=None
    )
    monkeypatch.setattr(cp, "get_dcp_group", lambda: fake_group)
    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", mode)
    monkeypatch.setenv("TPU_DCP_LSE_MERGE_A2A", "1")
    merged = torch.zeros(8, 4, 512, dtype=torch.bfloat16)
    monkeypatch.setattr(
        cp, "merge_lse_partials_all_to_all", lambda *args, **kwargs: merged
    )
    retained = []
    monkeypatch.setattr(cp, "_retain_rank", lambda out: retained.append(out) or out + 1)
    out = cp.merge_lse_partials_scatter_heads(
        torch.zeros(8, 64, 512, dtype=torch.bfloat16), torch.zeros(8, 64)
    )
    assert retained == ([merged] if lifted else [])
    assert (out is merged) == (not lifted)


@pytest.mark.parametrize("exchange", ["small", "group_rows", "unpack", "ragged"])
@pytest.mark.parametrize(("mode", "lifted"), [("", False), ("pairs", True)])
def test_send_layout_merge_lifts_rank_and_lane_only_with_the_combine_on(
    monkeypatch, exchange, mode, lifted
):
    """As for the all-to-all branch: each exchange of the send-layout merge
    (an 8-token bucket, which the combine does not take) returns its merge as
    is with the combine off, and with it on also lifts the rank and lane
    buffers the combine's buckets read."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    import torch

    from vllm_torchtpu.distributed import dcp
    from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS
    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp

    world, tokens, local, vdim = 16, 8, 4, 512
    rows = tokens * local
    fake_group = types.SimpleNamespace(
        world_size=world, rank_in_group=0, ranks=list(range(world)), cpu_group=None
    )
    monkeypatch.setattr(cp, "get_dcp_group", lambda: fake_group)
    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", mode)
    monkeypatch.setattr(cp, "send_small_a2a_applies", lambda t: exchange == "small")
    monkeypatch.setenv(
        "TPU_DCP_MLA_SEND_GROUP_ROWS", "8" if exchange == "group_rows" else "0"
    )
    monkeypatch.setenv(
        "TPU_DCP_MLA_SEND_LAYOUT_UNPACK", "1" if exchange == "unpack" else "0"
    )
    recv = (
        torch.zeros(world, rows, vdim, dtype=torch.bfloat16),
        torch.zeros(world, rows),
    )
    monkeypatch.setattr(dcp, "dcp_lane_operand", lambda device=None: None)
    for prebuild in (
        "prebuild_send_small",
        "prebuild_send_group",
        "prebuild_send_unpack",
    ):
        monkeypatch.setattr(cp, prebuild, lambda *args: None)
    monkeypatch.setitem(cp._SEND_SMALL_OPS, world, lambda *args: recv)
    monkeypatch.setitem(cp._SEND_GROUP_OPS, (world, 8), lambda *args: recv)
    monkeypatch.setitem(cp._SEND_UNPACK_OPS, world, lambda *args: recv)
    monkeypatch.setattr(cp, "_ragged_all_to_all", lambda send, *args, **kw: send)
    merged = torch.zeros(tokens, local, vdim, dtype=torch.bfloat16)
    monkeypatch.setattr(cp, "_merge_received", lambda o_recv, l_recv: merged)
    retained = []
    monkeypatch.setattr(cp, "_retain_rank", lambda out: retained.append(out) or out + 1)
    send = torch.zeros(world, SEND_PAD_ROWS + rows, vdim, dtype=torch.int16)
    out = cp.merge_send_layout(send, torch.tensor([0, 5], dtype=torch.int32), tokens)
    assert retained == ([merged] if lifted else [])
    assert (out is merged) == (not lifted)


@pytest.mark.parametrize(
    ("requested", "placed", "want"),
    [
        ("auto", "paired", "pairs"),
        ("auto", "cpu", "flat"),
        ("pairs", "paired", "pairs"),
        ("pairs", "cpu", ValueError),
        ("pairs", "scrambled", ValueError),
        ("flat", "paired", "flat"),
    ],
)
def test_resolve_online_combine_mode(monkeypatch, requested, placed, want):
    """'auto' takes 'pairs' only on a chip-paired DCP mesh, 'pairs' raises on
    any other mesh, and 'flat' never runs the pairing check (which may gather
    across ranks)."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp

    platform = "cpu" if placed == "cpu" else "tpu"
    devices = [_FakeDevice((i // 2, 0, 0), i % 2, platform) for i in range(8)]
    if placed == "scrambled":
        devices[1], devices[2] = devices[2], devices[1]
    mesh = _FakeMesh(
        np.asarray(devices, dtype=object).reshape(1, 8), ("dcp_lane", "dcp")
    )
    monkeypatch.setattr(cp, "_DCP_CHIP_PAIRS_CACHE", {})
    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", requested)
    checked = []
    pairs_ok = cp._dcp_chip_pairs_ok
    monkeypatch.setattr(
        cp, "_dcp_chip_pairs_ok", lambda m, g=None: checked.append(m) or pairs_ok(m, g)
    )
    if want is ValueError:
        with pytest.raises(ValueError, match="requires even DCP world >= 4"):
            cp._resolve_online_combine_mode(mesh)
    else:
        assert cp._resolve_online_combine_mode(mesh) == want
    assert checked == ([] if requested == "flat" else [mesh])


@pytest.mark.parametrize(
    ("paired", "other_lane", "want"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_dcp_chip_pairs_ok_gathers_placement_the_mesh_lacks(
    monkeypatch, paired, other_lane, want
):
    """When the mesh's devices carry no TPU placement, every rank of the DCP
    group contributes its own device's chip and core, pairing is checked in
    rank order, and on a multi-lane world every lane must agree."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    import torch

    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp

    world = 4
    places = [((i // 2, 0, 0), i % 2) for i in range(world)]
    if not paired:
        places[1], places[2] = places[2], places[1]
    cpu_devices = [_FakeDevice((0, 0, 0), 0, "cpu") for _ in range(2 * world)]
    mesh = _FakeMesh(
        np.asarray(cpu_devices, dtype=object).reshape(2, world), ("dcp_lane", "dcp")
    )
    group = types.SimpleNamespace(
        world_size=world, rank_in_group=0, ranks=list(range(world)), cpu_group=object()
    )
    monkeypatch.setattr(cp, "_DCP_CHIP_PAIRS_CACHE", {})
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: 2 * world
    )
    monkeypatch.setattr(cp.jax, "local_devices", lambda: [_FakeDevice(*places[0])])
    gathers = []

    def all_gather_object(out, obj, group=None):
        gathers.append(group)
        if group is None:  # the world: this lane's answer, then the other's
            out[:] = [obj] * world + [other_lane] * world
        else:  # the DCP group: every rank's (rank, coords, core), out of order
            out[:] = [(r, *places[r]) for r in reversed(range(world))]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    assert cp._dcp_chip_pairs_ok(mesh, group) is want
    assert gathers == [group.cpu_group, None]


@pytest.mark.parametrize("multi_lane", [False, True])
@pytest.mark.parametrize("with_qsl", [False, True])
@pytest.mark.parametrize("partial_dtype", ["bfloat16", "float32"])
def test_flag_on_merge_hands_the_combine_op_its_operands(
    monkeypatch, multi_lane, with_qsl, partial_dtype
):
    """With TPU_DCP_ONLINE_LSE_COMBINE set, `merge_lse_partials_scatter_heads`
    on a 32-token backbone bucket calls the combine op (a stand-in recording
    its operands) with the partial in bf16, the f32 lse, the live count from
    `query_start_loc`, then the rank and (multi-lane) lane buffers; lifts them
    during warmup; and returns the op's result in the partial's dtype. The
    kernel itself runs in the interpret and multichip tests."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    import torch

    from vllm_torchtpu.compilation import shape_variants
    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp

    world, rank, tokens, local, vdim, live = 8, 3, 32, 8, 512, 27
    group = types.SimpleNamespace(
        world_size=world, rank_in_group=rank, ranks=list(range(world)), cpu_group=None
    )
    monkeypatch.setattr(cp, "get_dcp_group", lambda: group)
    monkeypatch.setenv("TPU_DCP_ONLINE_LSE_COMBINE", "flat")
    monkeypatch.setenv("TPU_DCP_RS_OUT_PRECISION", "1")
    rank_buf = torch.tensor([rank], dtype=torch.int32)
    lane_buf = torch.tensor([1], dtype=torch.int32) if multi_lane else None
    monkeypatch.setattr(cp, "_DCP_RANK", rank_buf)
    monkeypatch.setattr(cp, "_DCP_LANE", lane_buf)
    merged = torch.randn(tokens, local, vdim).to(torch.bfloat16)
    calls, built, retained = [], [], []

    def op(*args):
        calls.append(args)
        return merged

    def ensure(world_arg, *, use_live, out_vmem=True, group=None):
        built.append((world_arg, use_live, group))
        return op, multi_lane

    monkeypatch.setattr(cp, "_ensure_online_lse_combine_op", ensure)
    retain = cp._retain_rank
    monkeypatch.setattr(
        cp, "_retain_rank", lambda out: retained.append(out) or retain(out)
    )

    dtype = getattr(torch, partial_dtype)
    output = torch.randn(tokens, world * local, vdim).to(dtype)
    lse = torch.randn(tokens, world * local)
    qsl = torch.tensor([0, 11, live], dtype=torch.int32) if with_qsl else None
    with shape_variants.warmup():
        got = cp.merge_lse_partials_scatter_heads(
            output, lse, wire_dtype=torch.bfloat16, query_start_loc=qsl
        )

    assert built == [(world, with_qsl, group)]
    (args,) = calls
    part, lse_arg, *rest = args
    assert part.dtype == torch.bfloat16
    assert torch.equal(part, output.to(torch.bfloat16))
    assert lse_arg.dtype == torch.float32 and torch.equal(lse_arg, lse)
    if with_qsl:
        live_arg, *rest = rest
        assert live_arg.dtype == torch.int32 and live_arg.tolist() == [live]
    assert len(rest) == (2 if multi_lane else 1)
    assert rest[0] is rank_buf
    assert not multi_lane or rest[1] is lane_buf
    assert len(retained) == 1 and retained[0] is merged
    assert got.dtype == dtype
    assert torch.equal(got, merged.to(dtype))


# The V-space o_proj chain the worker exports: DCP16, 4 heads per rank of
# v_head_dim 256 into fp8_matmul_all_reduce's [1024, 6144] o_proj.
CHAIN_CORES = 16
CHAIN_LOCAL = 4
CHAIN_VDIM = 256
CHAIN_HIDDEN = 6144


def _oproj_chain_tokens() -> list[int]:
    """The combine buckets whose output `fp8_matmul_all_reduce` takes."""
    k = CHAIN_LOCAL * CHAIN_VDIM
    return [
        t for t in TOKEN_BUCKETS if oproj_ar.supports(t, k, CHAIN_HIDDEN, CHAIN_CORES)
    ]


# Under the CPU lane's 600 s pytest timeout, so a slow or hung worker is killed
# (and its output shown) by the test rather than by pytest. A passing worker16
# run takes about 150-165 s on a 64-core host, adapter8 about 30 s.
WORKER_TIMEOUT_SECONDS = 540


def _run_cpu_worker(name: str, num_devices: int) -> dict:
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={num_devices}"
        " --xla_allow_excess_precision=false",
    )
    try:
        proc = subprocess.run(
            [sys.executable, __file__, name],
            env=env,
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        # Output captured so far, as bytes even with text=True.
        tail = (exc.stdout or b"")[-4000:] + (exc.stderr or b"")[-4000:]
        pytest.fail(
            f"{name} timed out after {WORKER_TIMEOUT_SECONDS} s\n"
            + tail.decode(errors="replace")
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-4000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT ") :])


def test_adapter_combine_op_function_in_interpret_mode():
    """The JAX function the MLA adapter registers as the combine op runs the
    kernel (not its XLA fallback), race free, within bf16 accuracy on every
    rank of both lanes (`_adapter8_main`)."""
    pytest.importorskip("torch_tpu")
    cases = _run_cpu_worker("adapter8", 8)
    assert len(cases) == 2 * 2 * 2 * 8, sorted(cases)
    for key, case in sorted(cases.items()):
        assert case["kernel"], (key, case)
        assert not case["races_found"], (key, case)
        assert case["nan"] == 0 and case["dead_positive_zero"], (key, case)
        assert case["rel_l2"] < 8e-3 and case["cos"] > 0.99995, (key, case)


@pytest.fixture(scope="module")
def worker16_results():
    return _run_cpu_worker("worker16", 16)


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
    chain_tokens = _oproj_chain_tokens()
    assert chain_tokens
    chained = worker16_results["chained"]
    assert sorted(chained) == sorted(f"v_chain_t{t}" for t in chain_tokens), chained
    for key, info in sorted(chained.items()):
        assert info["ok"], (key, info)
        assert not info["has_partition_id"], (key, info)


@pytest.mark.multichip
@pytest.mark.parametrize("out_vmem", [True, False])
@pytest.mark.parametrize("mode", ["pairs", "flat"])
@pytest.mark.parametrize("tokens", TOKEN_BUCKETS)
def test_multichip_tpu_dcp_online_lse_combine(mesh, mode, tokens, out_vmem):
    """Compiled execution on the 8 TPU cores of one host across T=16..96, with
    the output in VMEM (as the MLA adapter builds the op) and in HBM."""
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
                out_vmem=out_vmem,
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
                        "ok": f"dcp_online_lse_combine_{mode}_n{world}_t{tokens}_"
                        in asm,
                        "has_partition_id": (
                            "partition_id" in asm or "partition-id" in asm
                        ),
                    }

    # The combine's VMEM output (V space, D=256, out_vmem=True) chains into
    # fp8_matmul_all_reduce for every bucket that kernel admits.
    chained = {}
    sh16 = NamedSharding(mesh16, P("dcp"))
    sh16_rep = NamedSharding(mesh16, P())
    heads16 = CHAIN_CORES * CHAIN_LOCAL
    k16 = CHAIN_LOCAL * CHAIN_VDIM
    for tokens in _oproj_chain_tokens():

        def fn_v_chain(p, ls, lv, r, w_q, w_s, *, _t=tokens):
            def body(a, b, c, rk, q, s):
                v_out = comb.dcp_online_lse_combine(
                    a,
                    b,
                    num_cores=CHAIN_CORES,
                    axis="dcp",
                    mode="pairs",
                    rank=rk,
                    live_tokens=c,
                    out_vmem=True,
                ).reshape(_t, k16)
                return oproj_ar.fp8_matmul_all_reduce(
                    v_out, q, s, rk, num_cores=CHAIN_CORES, axis="dcp"
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
            S((CHAIN_CORES * tokens, heads16, CHAIN_VDIM), jnp.bfloat16, sharding=sh16),
            S((CHAIN_CORES * tokens, heads16), jnp.float32, sharding=sh16),
            S((1,), jnp.int32, sharding=sh16_rep),
            S((CHAIN_CORES,), jnp.int32, sharding=sh16),
            S((CHAIN_CORES * k16, CHAIN_HIDDEN), jnp.float8_e4m3fn, sharding=sh16),
            S((CHAIN_HIDDEN,), jnp.float32, sharding=sh16_rep),
        )
        chained[f"v_chain_t{tokens}"] = {
            "ok": f"dcp_online_lse_combine_pairs_n{CHAIN_CORES}_t{tokens}_" in asm_v
            and f"fp8_oproj_all_reduce_m{tokens}_" in asm_v,
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


def _adapter8_main():
    """The JAX function `_ensure_online_lse_combine_op` hands `sharded_jax_op`
    (its shard_map, operand order, rank and lane operands, live count and mode
    resolution), run with the kernel in TPU interpret mode on 8 CPU devices: as
    one 8-rank DCP group and as two lanes of 4, `flat` and `pairs`, with and
    without a live count. Both lanes get the same inputs, as replicated
    operands; the lane operand differs per lane, as each rank's own buffer."""
    import torch
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    from vllm_torchtpu.distributed import dcp
    from vllm_torchtpu.distributed import sharded_jax_op as sjo
    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp

    devices = jax.devices()
    assert len(devices) == 8, len(devices)
    built = {}

    def capture(tag, fn, **kwargs):
        built["fn"] = fn
        return tag

    sjo.sharded_jax_op = capture
    # CPU devices carry no chip placement; the pairing check has its own tests.
    cp._dcp_chip_pairs_ok = lambda mesh, group=None: True
    cp._DCP_RANK = torch.zeros(1, dtype=torch.int32)
    cp._DCP_LANE = torch.zeros(1, dtype=torch.int32)
    tokens, heads, vdim = 32, 64, 256
    cases = {}
    for lanes, world in ((1, 8), (2, 4)):
        mesh = Mesh(np.asarray(devices).reshape(lanes, world), ("dcp_lane", "dcp"))
        dcp.get_or_create_dcp_mesh = lambda *args, _m=mesh, **kwargs: _m
        group = types.SimpleNamespace(
            world_size=world, rank_in_group=0, ranks=list(range(world)), cpu_group=None
        )
        spec = NamedSharding(mesh, P("dcp"))
        lane_of = {dev: ln for (ln, _), dev in np.ndenumerate(mesh.devices)}
        rank_of = {dev: d for (_, d), dev in np.ndenumerate(mesh.devices)}
        rank = jax.device_put(jnp.arange(world, dtype=jnp.int32), spec)
        lane = jax.make_array_from_single_device_arrays(
            (world,),
            spec,
            [
                jax.device_put(np.asarray([lane_of[dev]], np.int32), dev)
                for dev in spec.addressable_devices_indices_map((world,))
            ],
        )
        for mode in ("flat", "pairs"):
            os.environ["TPU_DCP_ONLINE_LSE_COMBINE"] = mode
            for live in (None, tokens - 5):
                cp._ONLINE_COMBINE_OPS.clear()
                _, multi_lane = cp._ensure_online_lse_combine_op(
                    world, use_live=live is not None, group=group
                )
                assert multi_lane == (lanes > 1), (lanes, multi_lane)
                part, lse = _make_inputs(world, tokens, heads // world, vdim, live=live)
                ref = np.asarray(comb.reference_float64(part, lse, live_tokens=live))
                args = [
                    jax.device_put(
                        jnp.asarray(part).reshape(world * tokens, heads, vdim), spec
                    ),
                    jax.device_put(
                        jnp.asarray(lse).reshape(world * tokens, heads), spec
                    ),
                ]
                if live is not None:
                    args.append(jnp.asarray([live], dtype=jnp.int32))
                args.append(rank)
                if multi_lane:
                    args.append(lane)
                fn = jax.jit(built["fn"])
                kernel = "dcp_online_lse_combine_" in str(jax.make_jaxpr(fn)(*args))
                params = pltpu.InterpretParams(detect_races=True)
                pltpu.reset_tpu_interpret_mode_state()
                with (
                    pltpu.force_tpu_interpret_mode(params),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    out = jax.block_until_ready(fn(*args))
                races = bool(interpret_pallas_call.races.races_found)
                eff_live = tokens if live is None else live
                for shard in out.addressable_shards:
                    d, ln = rank_of[shard.device], lane_of[shard.device]
                    got = np.asarray(shard.data)[None]
                    m = _metrics_vs_f64(got, ref[d][None], eff_live)
                    m.update(kernel=kernel, races_found=races)
                    cases[f"{lanes}x{world}_{mode}_live{live}_lane{ln}_rank{d}"] = m
    print("RESULT " + json.dumps(cases), flush=True)


if __name__ == "__main__" and sys.argv[1:] == ["worker16"]:
    _worker16_main()
if __name__ == "__main__" and sys.argv[1:] == ["adapter8"]:
    _adapter8_main()
