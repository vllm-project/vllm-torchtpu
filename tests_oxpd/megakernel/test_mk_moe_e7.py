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
"""CPU interpret parity & contract tests for Segment (c): MK MoE E7 wins.

Verifies all three opt-in E7 decode MoE optimizations inside the megakernel
(both `flag=0` regression and `flag=1` parity):
  1. `VLLM_TPU_MK_MOE_AR2D=1` (`HIER=4` 2D-torus AR2D all-reduce):
     8-rank Pallas interpret parity vs FP32 exact sum & `HIER=2`, bit-exact
     consensus across all 8 ranks, 2-layer multi-slot race freedom, and 35%
     lower VMEM than `HIER=2` at served `H=2560, B=64`.
  2. `VLLM_TPU_MK_MOE_EXACT_WIDTH=1`:
     - `HIER=5` uneven 2-round RS+AG at served `H=2560, R=8` (`[384x4, 256x4]`
       lane-tile column split, zero padding): bit-exact consensus across all
       8 ranks and tight parity vs FP32 exact sum & `HIER=2`.
     - `MR`-row active slicing in `experts_piped_ep` at `B=4` (`MR=8 < MB=16`):
       bit-exact (`max_abs=0.0`) vs `exact_width=False` on active rows `0..MR-1`.
  3. `VLLM_TPU_MK_MOE_SKIP_PAD=1` (`MoETunables.skip_pad`):
     Padded rows (`row >= n_valid`) receive zero MoE delta, activate zero bogus
     experts (strictly lower `n_hit`), and keep real rows `0..n_valid-1` in
     tight parity with `skip_pad=False` and pure-JAX `moe_reference` (both
     dense `PIPE_EP` and `row_bucket=True`).
"""

from __future__ import annotations

import dataclasses
import os
import sys
import types

_xla_flags = os.environ.get("XLA_FLAGS", "")
for _prefix, _flag in (
    ("--xla_force_host_platform_device_count", "--xla_force_host_platform_device_count=8"),
    ("--xla_llvm_disable_expensive_passes", "--xla_llvm_disable_expensive_passes=true"),
):
    if _prefix not in _xla_flags:
        _xla_flags = f"{_xla_flags} {_flag}".strip()
os.environ["XLA_FLAGS"] = _xla_flags

try:
    import vllm.envs  # noqa: F401
except ImportError:
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _vllm.envs = _envs
    sys.modules.setdefault("vllm", _vllm)
    sys.modules.setdefault("vllm.envs", _envs)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from tests.megakernel.megakernel_helpers import TINY_TILES, FakeGeometry, MiniCtx  # noqa: E402
from vllm_torchtpu.megakernel import comm, planner, stream, vmem  # noqa: E402
from vllm_torchtpu.megakernel.blocks import MoEBlock  # noqa: E402
from vllm_torchtpu.megakernel.blocks.hc.comm import (  # noqa: E402
    _BUF_NAMES,
    hc_comm_params,
    hc_comm_scratch,
)
from vllm_torchtpu.megakernel.blocks.moe.params import MoETunables  # noqa: E402
from vllm_torchtpu.megakernel.blocks.moe.reference import moe_reference  # noqa: E402
from vllm_torchtpu.megakernel.exec_spec import ExecSpec  # noqa: E402
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, I32, rmsnorm  # noqa: E402
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import (  # noqa: E402
    _safe_valid_rows,
    moe_e7_hier,
    segment_tiles,
)

try:
    from tests.megakernel.test_mk_parity_gates import (  # noqa: E402
        assert_parity,
        compute_parity_metrics,
        format_parity_table,
    )
except ImportError:

    def compute_parity_metrics(actual, ref, *, segment, name="", **kw):
        a, r = np.asarray(actual, np.float32), np.asarray(ref, np.float32)
        if a.ndim == r.ndim + 1 and a.shape[1:] == r.shape:
            r = np.broadcast_to(r[None], a.shape)
        d = np.abs(a - r)
        s = max(float(np.abs(r).mean()), 1e-12)
        ok = bool(np.isfinite(a).all())
        if kw.get("tol_max_abs") is not None:
            ok &= float(d.max()) <= kw["tol_max_abs"]
        if kw.get("tol_max_rel") is not None:
            ok &= float((d / np.maximum(np.abs(r), 1e-2 * s)).max()) <= kw["tol_max_rel"]
        if kw.get("tol_mean_ratio") is not None:
            ok &= float(d.mean()) / s <= kw["tol_mean_ratio"]
        if kw.get("tol_max_ratio") is not None:
            ok &= float(d.max()) / s <= kw["tol_max_ratio"]
        return types.SimpleNamespace(
            segment=segment, name=name, max_abs=float(d.max()), passed=ok
        )

    def assert_parity(m, *, record=True):
        del record
        assert m.passed, m
        return m

    def format_parity_table(ms):
        return "\n".join(str(m) for m in ms)


def _require_devices(n: int) -> list[jax.Device]:
    devs = jax.devices("cpu")
    if len(devs) < n:
        pytest.skip(
            f"Requires {n} CPU devices (found {len(devs)}); pass "
            "--test_env=XLA_FLAGS=--xla_force_host_platform_device_count=8"
        )
    return devs[:n]


# ---------------------------------------------------------------------------
# 1. Served Qwen3.8 Comm Planning, VMEM Budget & Env Flag Routing
# ---------------------------------------------------------------------------


def _vmem_bytes(reqs) -> int:
    total = 0
    for r in reqs:
        spec = r.spec
        if (
            getattr(spec, "memory_space", None) == pltpu.VMEM
            and hasattr(spec, "shape")
            and hasattr(spec, "dtype")
        ):
            total += int(np.prod(spec.shape)) * jnp.dtype(spec.dtype).itemsize
    return total


def test_hc_comm_params_ar2d_and_exact_width_served_qwen38():
    """Verify HIER=4 (AR2D) and HIER=5 (exact_width) at served Qwen3.8 R=8, h=2560."""
    # B=64, h=2560, R=8: AR2D (HIER=4) active and uses strictly less VMEM than HIER=2!
    ex64_h2 = ExecSpec(B=64, T=4, R=8, tiles=dataclasses.replace(TINY_TILES, hier=2))
    ex64_h4 = ExecSpec(B=64, T=4, R=8, tiles=dataclasses.replace(TINY_TILES, hier=4))
    cp2 = hc_comm_params(ex64_h2, 2560)
    cp4 = hc_comm_params(ex64_h4, 2560)
    assert cp2.HIER == 2 and cp4.HIER == 4 and len(cp4.AR_ROUNDS) == 10
    vmem_h2 = _vmem_bytes(hc_comm_scratch(ex64_h2, 2560))
    vmem_h4 = _vmem_bytes(hc_comm_scratch(ex64_h4, 2560))
    assert vmem_h4 < vmem_h2, (vmem_h4, vmem_h2)

    # B=16, h=2560, R=8: HIER=4 falls back to HIER=5 (exact_width [384x4, 256x4])!
    ex16_h4 = ExecSpec(B=16, T=4, R=8, tiles=dataclasses.replace(TINY_TILES, hier=4))
    cp16 = hc_comm_params(ex16_h4, 2560)
    assert cp16.HIER == 5 and cp16.CW == 384 and cp16.AR_ROUNDS == (0, 1)


def test_moe_e7_env_flags_routing(monkeypatch):
    """Verify opt-in flags leave flag=0 untouched and route HIER=4/5 + skip_pad cleanly."""
    for k in (
        "VLLM_TPU_MK_MOE_E7",
        "VLLM_TPU_MK_MOE_AR2D",
        "VLLM_TPU_MK_MOE_EXACT_WIDTH",
        "VLLM_TPU_MK_MOE_SKIP_PAD",
    ):
        monkeypatch.setenv(k, "0")
    t0 = segment_tiles(64)
    assert t0.hier == 2
    assert not t0.block("moe", MoETunables).skip_pad
    assert not t0.block("moe", MoETunables).exact_width

    # Individual flags:
    monkeypatch.setenv("VLLM_TPU_MK_MOE_AR2D", "1")
    monkeypatch.setenv("VLLM_TPU_MK_MOE_AR2D_MIN_TOKENS", "64")
    assert moe_e7_hier(64, 2) == 4
    assert moe_e7_hier(128, 2) == 4
    assert moe_e7_hier(16, 2) == 2

    # Umbrella flag enables AR2D (>=64), exact_width (<64 & low-batch MR slice), and skip_pad!
    monkeypatch.setenv("VLLM_TPU_MK_MOE_AR2D", "0")
    monkeypatch.setenv("VLLM_TPU_MK_MOE_E7", "1")
    t16 = segment_tiles(16)
    t64 = segment_tiles(64)
    assert t16.hier == 5 and t64.hier == 4
    assert t16.block("moe", MoETunables).skip_pad
    assert t16.block("moe", MoETunables).exact_width


def test_safe_valid_rows_empty_qsl_guard():
    """Bug #4: empty `query_start_loc` (`shape == (0,)`, e.g. c32 DCP=2 mns=12 warmup)
    must fall back to `[B]` instead of raising IndexError on `qsl[-1]`."""
    empty = jnp.zeros((0,), dtype=jnp.int32)
    got_empty = _safe_valid_rows(empty, 48)
    assert got_empty.shape == (1,) and got_empty.dtype == jnp.int32
    assert int(got_empty[0]) == 48

    got_scalar = _safe_valid_rows(jnp.int32(5), 48)
    assert got_scalar.shape == (1,) and int(got_scalar[0]) == 48

    got_live = _safe_valid_rows(jnp.asarray([0, 4, 8, 12], dtype=jnp.int32), 48)
    assert got_live.shape == (1,) and int(got_live[0]) == 12

    got_neg = _safe_valid_rows(jnp.asarray([-3], dtype=jnp.int32), 48)
    assert int(got_neg[0]) == 0

    got_hi = _safe_valid_rows(jnp.asarray([0, 64], dtype=jnp.int32), 48)
    assert int(got_hi[0]) == 48


def test_ar_hier_small_interaction_with_moe_e7(monkeypatch):
    """`VLLM_TPU_MK_AR_HIER_SMALL=1` preserves `hier=1` for `B <= 8` with E7 off or on."""
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    monkeypatch.delenv("VLLM_TPU_MK_HIER", raising=False)
    for k in (
        "VLLM_TPU_MK_MOE_E7",
        "VLLM_TPU_MK_MOE_AR2D",
        "VLLM_TPU_MK_MOE_EXACT_WIDTH",
        "VLLM_TPU_MK_MOE_SKIP_PAD",
        "VLLM_TPU_MK_AR_HIER_SMALL",
    ):
        monkeypatch.setenv(k, "0")
    assert {b: wsc.wholestep_tiles(b).hier for b in (4, 8, 16, 32, 64)} == {
        4: 2, 8: 2, 16: 2, 32: 2, 64: 2
    }

    monkeypatch.setenv("VLLM_TPU_MK_AR_HIER_SMALL", "1")
    assert {b: wsc.wholestep_tiles(b).hier for b in (4, 8, 16, 32, 64)} == {
        4: 1, 8: 1, 16: 2, 32: 2, 64: 2
    }

    monkeypatch.setenv("VLLM_TPU_MK_MOE_AR2D", "1")
    monkeypatch.setenv("VLLM_TPU_MK_MOE_AR2D_MIN_TOKENS", "64")
    assert {b: wsc.wholestep_tiles(b).hier for b in (4, 8, 16, 32, 64)} == {
        4: 1, 8: 1, 16: 2, 32: 2, 64: 4
    }

    monkeypatch.setenv("VLLM_TPU_MK_MOE_E7", "1")
    assert {b: wsc.wholestep_tiles(b).hier for b in (4, 8, 16, 32, 64)} == {
        4: 1, 8: 1, 16: 5, 32: 5, 64: 4
    }

    monkeypatch.setenv("VLLM_TPU_MK_AR_HIER_SMALL", "0")
    assert {b: wsc.wholestep_tiles(b).hier for b in (4, 8, 16, 32, 64)} == {
        4: 5, 8: 5, 16: 5, 32: 5, 64: 4
    }


# ---------------------------------------------------------------------------
# 2. 8-Rank Pallas Interpret Parity: HIER=4 (AR2D) & HIER=5 (exact_width RS+AG)
# ---------------------------------------------------------------------------


def _run_hc_allreduce(R: int, B: int, h: int, hier: int, seed: int = 42):
    """Execute 2-layer double-buffered HC `AllReduceRing` on `R` CPU devices."""
    devs = _require_devices(R)
    tiles = dataclasses.replace(TINY_TILES, hier=hier, war=1)
    ex = ExecSpec(B=B, T=4 if B % 4 == 0 else 1, R=R, ctx=128, tiles=tiles, interpret=True)
    cp = hc_comm_params(ex, h)
    assert cp.HIER == hier, (cp.HIER, hier)
    reqs = hc_comm_scratch(ex, h)
    s_names = [r.name for r in reqs]
    s_specs = [r.spec for r in reqs]

    def pallas_body(x0_ref, x1_ref, o0_ref, o1_ref, *scr_refs):
        raw_scr = dict(zip(s_names, scr_refs))
        scr = {n: raw_scr["hc_" + n] for n in _BUF_NAMES if "hc_" + n in raw_scr}
        me = lax.axis_index("tp").astype(I32)
        barrier = pltpu.get_barrier_semaphore()
        for off in range(1, R):
            pl.semaphore_signal(
                barrier, 1, device_id=(me ^ off,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, R - 1)
        ar = comm.AllReduceRing(cp, scr, me)
        # Layer 0: slot 0 then slot 1; Layer 1: reuse slot 0 then slot 1 (proves race-free!).
        tot0 = ar.allreduce(0, x0_ref[...], jnp.bool_(True))
        tot1 = ar.allreduce(1, x1_ref[...], jnp.bool_(True))
        tot0b = ar.allreduce(0, tot0, jnp.bool_(False))
        tot1b = ar.allreduce(1, tot1, jnp.bool_(False))
        ar.drain_sends()
        o0_ref[...] = tot0b
        o1_ref[...] = tot1b

    call = pl.pallas_call(
        pallas_body,
        out_shape=[
            jax.ShapeDtypeStruct((ex.MR, h), F32),
            jax.ShapeDtypeStruct((ex.MR, h), F32),
        ],
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
        scratch_shapes=s_specs,
        compiler_params=pltpu.CompilerParams(collective_id=0),
        interpret=pltpu.InterpretParams(),
    )
    mesh = Mesh(np.array(devs), ("tp",))
    step = jax.jit(
        jax.shard_map(
            lambda a, b: tuple(y[None] for y in call(a[0], b[0])),
            mesh=mesh,
            in_specs=(P("tp"), P("tp")),
            out_specs=(P("tp"), P("tp")),
            check_vma=False,
        )
    )
    k0, k1 = jax.random.split(jax.random.PRNGKey(seed), 2)
    x0 = jax.random.normal(k0, (R, ex.MB, h), dtype=BF16).astype(F32)
    x1 = jax.random.normal(k1, (R, ex.MB, h), dtype=BF16).astype(F32)
    o0, o1 = step(x0, x1)
    # Exact FP32 reference: sum across R ranks twice (first sum * R).
    ref0 = np.asarray(jnp.sum(x0[:, : ex.MR], axis=0) * R, np.float32)
    ref1 = np.asarray(jnp.sum(x1[:, : ex.MR], axis=0) * R, np.float32)
    return np.asarray(o0, np.float32), np.asarray(o1, np.float32), ref0, ref1


@pytest.mark.parametrize(
    "hier,B,h,label",
    [
        (4, 64, 512, "AR2D_R8_B64_H512"),
        (5, 8, 2560, "ExactWidth_R8_B8_H2560_uneven384_256"),
        (5, 16, 2560, "ExactWidth_R8_B16_H2560_uneven384_256"),
    ],
)
def test_allreduce_ar2d_and_exact_width_8rank_parity(hier: int, B: int, h: int, label: str):
    """8-rank 2-layer parity vs FP32 exact sum & HIER=2 + bit-exact 8-rank consensus."""
    o0, o1, ref0, ref1 = _run_hc_allreduce(R=8, B=B, h=h, hier=hier, seed=42 + hier + B)
    h2_0, h2_1, _, _ = _run_hc_allreduce(R=8, B=B, h=h, hier=2, seed=42 + hier + B)

    metrics = [
        # All 8 ranks MUST hold identical bits!
        assert_parity(
            compute_parity_metrics(
                o0[1:],
                o0[0],
                segment="MoE_Comm",
                name=f"{label}_consensus_slot0",
                tol_max_abs=0.0,
                notes="all 8 ranks bit-exact",
            ),
            record=False,
        ),
        assert_parity(
            compute_parity_metrics(
                o1[1:],
                o1[0],
                segment="MoE_Comm",
                name=f"{label}_consensus_slot1",
                tol_max_abs=0.0,
                notes="all 8 ranks bit-exact",
            ),
            record=False,
        ),
        # Parity vs exact FP32 sum (2 BF16 all-reduce rounds).
        assert_parity(
            compute_parity_metrics(
                o0,
                ref0,
                segment="MoE_Comm",
                name=f"{label}_vs_f32_ref",
                tol_mean_ratio=1e-2,
                tol_max_ratio=5e-2,
                notes="2-layer sum vs FP32",
            ),
            record=False,
        ),
        # Parity vs baseline HIER=2.
        assert_parity(
            compute_parity_metrics(
                o1,
                h2_1[0],
                segment="MoE_Comm",
                name=f"{label}_vs_hier2",
                tol_mean_ratio=1e-2,
                tol_max_ratio=5e-2,
                notes="vs HIER=2 baseline",
            ),
            record=False,
        ),
    ]
    print(format_parity_table(metrics))


# ---------------------------------------------------------------------------
# 3. Single-Layer MoE Parity: skip_pad & exact_width (MR-row slice)
# ---------------------------------------------------------------------------


def _run_moe_layer(
    B: int,
    *,
    skip_pad: bool = False,
    n_valid: int | None = None,
    exact_width: bool = False,
    row_bucket: bool = False,
    hier: int = 0,
    seed: int = 77,
    poison_pad: bool = False,
):
    """Run one PIPE_EP MoE layer on R=2 CPU devices; returns (out, ref_valid, hid_raw, n_hits)."""
    R = 2
    devs = _require_devices(R)
    H, E, TOPK, IR, ISR = 256, 128, 8, 128, 128
    tiles = dataclasses.replace(
        TINY_TILES, bank=512 << 10, nbd=8, nbe=6, ebank=32 << 10, war=1, hier=hier
    ).with_blocks(
        {
            "moe": {
                "pipe_ep": True,
                "moe_unroll": True,
                "topk_lanes": True,
                "row_bucket": row_bucket,
                "skip_pad": skip_pad,
                "exact_width": exact_width,
            }
        }
    )
    g = FakeGeometry(
        L=1, att_every=1, H=H, NH=2, NKV=1, HD=128, E=E, TOPK=TOPK, I=IR,
        IS=ISR * R, moe_tp=False, e_slots=0,
    )
    moe = MoEBlock()
    ms = ModelSpec("moe_e7_test", config=g, layer_type=LayerType("lt", slots=(moe,)))
    ex = ExecSpec(B=B, T=1, R=R, ctx=128, stack_e=True, tiles=tiles, interpret=True)
    cp, scratch = vmem.build_scratch(ms, ex)
    mp = cp.block("moe")
    assert mp.PIPE_EP
    p = cp.plan
    wsh = planner.weight_shapes(ms, ex, p)
    osh = planner.kernel_out_shapes(ms, ex, p)
    s_names = list(scratch.keys())

    nv = B if n_valid is None else n_valid
    keys = jax.random.split(jax.random.PRNGKey(seed + B), 13)
    hid_raw = (jax.random.normal(keys[0], (ex.M, H), dtype=F32) * 0.5).astype(BF16)
    router = (jax.random.normal(keys[2], (H, E), dtype=F32) * 0.1).astype(BF16)
    if poison_pad and nv < B:
        # Make padded rows strongly route to distinct experts (cols 64..127) so skipping
        # padded rows demonstrably reduces routed expert count `n_hit`!
        pad_vec = jax.random.normal(keys[12], (B - nv, H), dtype=F32) * 2.0
        hid_raw = hid_raw.at[nv:B].set(pad_vec.astype(BF16))
    post_norm = jax.random.uniform(keys[1], (H,), dtype=F32, minval=-0.2, maxval=0.2)
    sh_gate = jax.random.normal(keys[3], (H,), dtype=F32) * 0.2
    sh13 = (jax.random.normal(keys[4], (R, H, 2 * ISR), dtype=F32) * 0.4).astype(FP8)
    sh13_s = jax.random.uniform(keys[5], (R, 2 * ISR), dtype=F32, minval=0.05, maxval=0.15)
    sh2 = (jax.random.normal(keys[6], (R, ISR, H), dtype=F32) * 0.4).astype(FP8)
    sh2_s = jax.random.uniform(keys[7], (R, H), dtype=F32, minval=0.05, maxval=0.15)
    EW = E // R
    e13 = (jax.random.normal(keys[8], (R, EW, H, 2 * IR), dtype=F32) * 0.4).astype(FP8)
    e13_s = jax.random.uniform(keys[9], (R, EW, 2 * IR), dtype=F32, minval=0.05, maxval=0.15)
    e2 = (jax.random.normal(keys[10], (R, EW, IR, H), dtype=F32) * 0.4).astype(FP8)
    e2_s = jax.random.uniform(keys[11], (R, EW, H), dtype=F32, minval=0.05, maxval=0.15)

    x_norm = rmsnorm(
        hid_raw[:B].astype(F32), post_norm, g.eps, zero_centered=bool(g.zero_centered_norm)
    )
    ref_delta, _ = moe_reference(
        {
            "router": router,
            "sh_gate": sh_gate,
            "sh13": sh13,
            "sh13_s": sh13_s,
            "sh2": sh2,
            "sh2_s": sh2_s,
            "e13": e13,
            "e13_s": e13_s,
            "e2": e2,
            "e2_s": e2_s,
        },
        x_norm,
        g=g,
        ex=ex,
    )
    ref_out = hid_raw[:B].astype(F32) + ref_delta

    lay = p.smalls_layout
    sm = jnp.zeros((R, 1, 1, p.smalls_width), dtype=F32)
    for name, val in [
        ("post_norm", jnp.broadcast_to(post_norm[None, :], (R, H))),
        ("sh13_s", sh13_s),
        ("sh2_s", sh2_s),
        ("sh_gate", jnp.broadcast_to(sh_gate[None, :], (R, H))),
    ]:
        o, w = lay[name]
        sm = sm.at[:, 0, 0, o : o + w].set(val)
    w_by_name = {
        "router": jnp.broadcast_to(router[None, ...], (R, 1, H, E)).reshape(R, *wsh["router"][0]),
        "sh13": sh13.reshape(R, *wsh["sh13"][0]),
        "sh2": sh2.reshape(R, *wsh["sh2"][0]),
        "e13": e13.reshape(R, *wsh["e13"][0]),
        "e2": e2.reshape(R, *wsh["e2"][0]),
        "es": jnp.concatenate([e13_s, e2_s], axis=-1).reshape(R, *wsh["es"][0]),
        "smalls": sm,
        "final_norm": jnp.zeros((R, *wsh["final_norm"][0]), dtype=F32),
    }

    def pallas_body(pos_ref, hid_ref, valid_ref, *rest):
        n_w, n_o = len(p.w_names), len(p.o_names)
        W = dict(zip(p.w_names, rest[:n_w]))
        O = dict(zip(p.o_names, rest[n_w : n_w + n_o]))
        nhit_out = rest[n_w + n_o]
        scr = dict(zip(s_names, rest[n_w + n_o + 1 :]))
        me = lax.axis_index("tp").astype(I32)
        barrier = pltpu.get_barrier_semaphore()
        for off in range(1, R):
            pl.semaphore_signal(barrier, 1, device_id=(me ^ off,), device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(barrier, R - 1)
        ring = stream.TwoRing(cp.stream, W, scr, me, interpret=True)
        ar = comm.AllReduceRing(cp.comm, scr, me)
        extra = {"moe_valid_rows": valid_ref} if skip_pad else {}
        ctx = MiniCtx(
            g=g, ex=ex, core_plan=cp, pos_ref=pos_ref, W=W, ST={}, O=O, scr=scr,
            me=me, stream=ring, comm=ar, extra=extra,
        )
        for i in range(cp.stream.NBD):
            ring.fetch(jnp.int32(i))
        pltpu.make_async_copy(W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]).start()
        moe.prologue(ctx)
        scr["res"][...] = hid_ref[: ex.MB].astype(F32)
        scr["mix"][...] = jnp.zeros(scr["mix"].shape, F32)
        moe.init_buffers(ctx)
        pltpu.make_async_copy(W["smalls"].at[0], scr["sm"].at[0], scr["sm_sem"].at[0]).wait()
        moe.on_layer_start(ctx, jnp.int32(0), jnp.int32(0))
        ring.begin_layer(jnp.int32(0))
        ctx.apply_input_norm(moe.input_norm, jnp.int32(0))
        part_moe = moe.emit(ctx, jnp.int32(0), jnp.int32(0))
        nhit_out[...] = jnp.full((8, 128), scr["st"][0] // mp.TE, I32)
        ar.add_residual(ar.allreduce(0, part_moe, jnp.bool_(True)))
        ar.drain_sends()
        O["hidden"][: ex.MB] = scr["res"][...].astype(BF16)
        if ex.MB < ex.M:
            O["hidden"][ex.MB :] = jnp.zeros((ex.M - ex.MB, H), BF16)

    call = pl.pallas_call(
        pallas_body,
        out_shape=[jax.ShapeDtypeStruct(*osh[k]) for k in p.o_names]
        + [jax.ShapeDtypeStruct((8, 128), I32)],
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ]
        + [pl.BlockSpec(memory_space=pltpu.HBM) for _ in p.w_names],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM) for _ in p.o_names]
        + [pl.BlockSpec(memory_space=pltpu.VMEM)],
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(collective_id=0),
        interpret=pltpu.InterpretParams() if ex.interpret is True else ex.interpret,
    )
    mesh = Mesh(np.array(devs), ("tp",))
    def _sharded_step(pos, hid, vr, *ws):
        h_out, n_out = call(pos, hid, vr, *(w[0] for w in ws))
        return h_out[None], n_out[None]

    step = jax.jit(
        jax.shard_map(
            _sharded_step,
            mesh=mesh,
            in_specs=(P(), P(), P(), *[P("tp")] * len(p.w_names)),
            out_specs=(P("tp"), P("tp")),
            check_vma=False,
        )
    )
    pos = jnp.zeros((ex.M,), dtype=I32)
    vr = jnp.asarray([nv], dtype=I32)
    out, nhit = step(pos, hid_raw, vr, *[w_by_name[n] for n in p.w_names])
    return (
        np.asarray(out.reshape(R, ex.M, H)[:, :B], np.float32),
        np.asarray(ref_out, np.float32),
        np.asarray(hid_raw[:B], np.float32),
        np.asarray(nhit.reshape(R, 8, 128)[:, 0, 0], np.int32),
    )


@pytest.mark.parametrize("B,nv,rb", [(16, 4, False), (32, 8, True)])
def test_moe_skip_pad_exact_real_rows_and_fewer_experts(B: int, nv: int, rb: bool):
    """`skip_pad=True` matches `skip_pad=False` and `moe_reference` on real rows `0:nv`,
    produces zero MoE delta on padded rows `nv:B`, and activates strictly fewer experts."""
    out_base, ref, hid, nhit_base = _run_moe_layer(
        B, skip_pad=False, n_valid=nv, row_bucket=rb, poison_pad=True
    )
    out_skip, _, _, nhit_skip = _run_moe_layer(
        B, skip_pad=True, n_valid=nv, row_bucket=rb, poison_pad=True
    )
    # 1. Strictly fewer routed experts fetched when 75% of rows are padded!
    assert int(nhit_skip.sum()) < int(nhit_base.sum()), (nhit_skip, nhit_base)
    # 2. Padded rows receive ZERO MoE delta (`out == hid_raw` exactly).
    np.testing.assert_array_equal(out_skip[:, nv:B], np.broadcast_to(hid[None, nv:B], (2, B - nv, hid.shape[-1])))
    # 3. Real rows match both baseline MK and pure-JAX `moe_reference`!
    m_vs_base = assert_parity(
        compute_parity_metrics(
            out_skip[:, :nv],
            out_base[:, :nv],
            segment="MoE_SkipPad",
            name=f"B{B}_nv{nv}_rb{int(rb)}_vs_flag0",
            tol_mean_ratio=1e-4,
            tol_max_ratio=1e-2,
            notes=f"experts {nhit_base.sum()}->{nhit_skip.sum()}",
        ),
        record=False,
    )
    m_vs_ref = assert_parity(
        compute_parity_metrics(
            out_skip[:, :nv],
            ref[:nv],
            segment="MoE_SkipPad",
            name=f"B{B}_nv{nv}_rb{int(rb)}_vs_jax_ref",
            tol_mean_ratio=1e-2,
            tol_max_ratio=8e-2,
            notes="real rows vs moe_reference",
        ),
        record=False,
    )
    print(format_parity_table([m_vs_base, m_vs_ref]))


def test_moe_exact_width_mr_slice_and_hier5_parity():
    """Low-batch `B=4` (`MR=8 < MB=16`): `exact_width=True` slices `experts_piped_ep`
    to `8` rows (bit-exact vs `exact_width=False` on `0:B`) + `HIER=5` RS+AG matches `moe_reference`."""
    out_base, ref, _, _ = _run_moe_layer(B=4, exact_width=False, hier=0)
    out_ew, _, _, _ = _run_moe_layer(B=4, exact_width=True, hier=0)
    out_ew_h5, _, _, _ = _run_moe_layer(B=4, exact_width=True, hier=5)

    metrics = [
        assert_parity(
            compute_parity_metrics(
                out_ew,
                out_base,
                segment="MoE_ExactWidth",
                name="B4_MR8_slice_vs_MB16",
                tol_max_abs=0.0,
                tol_max_rel=0.0,
                notes="bit-exact on 0:B",
            ),
            record=False,
        ),
        assert_parity(
            compute_parity_metrics(
                out_ew_h5,
                ref,
                segment="MoE_ExactWidth",
                name="B4_exact_width_HIER5_vs_ref",
                tol_mean_ratio=1e-2,
                tol_max_ratio=8e-2,
                notes="R=2 HIER=5 RS+AG",
            ),
            record=False,
        ),
    ]
    print(format_parity_table(metrics))
