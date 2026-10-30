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
"""Behaviour-preservation gate: golden hashes of kernel and reference_step outputs.

Two families of cases are checked:
- Kernel cases (`run_case`): run one tiny paged pooled step (CPU interpret) and
  hash the hidden output, every returned pool (KV + GDN state), and the
  resolved `use_gpr` flag.
- Reference cases (`ref_tiny_*` via `run_ref_case`): run pure-JAX
  `qwen35.reference_step` across TP, EP+EPLB, paged pool I/O, float32 RoPE
  intermediates, and attention-only configs, and hash `out` and `states`.

Two execution modes are supported:
1. Committed-golden mode (default, `MK_GOLDEN_JSON` unset): compares against
   `golden/b1_bitexact.json` when the current host fingerprint matches the
   recorded `"_meta"` entry (per-case `"_meta"` for `ref_tiny_*` entries, which
   never fall back to the file-level `"_meta"`, or file-level `"_meta"` for
   kernel entries). Because CPU XLA / Pallas lowering depends on the host CPU
   microarchitecture and library versions, a host fingerprint mismatch skips
   with instructions to run base-vs-head mode.
2. Base-vs-head mode (`MK_GOLDEN_JSON=<path>` or
   `tests/megakernel/golden/compare_base_head.sh <base-ref> [-n N]`): records
   fresh reference hashes at `<base-ref>` on the current host with
   `MK_REGEN_GOLDEN=1 MK_GOLDEN_JSON=$TMP/base.json pytest <this file>` (falling
   back to running HEAD's `ref_tiny_*` cases against `<base-ref>`'s `src/` when
   `<base-ref>` predates those cases), then verifies the working tree against
   `$TMP/base.json` with `MK_GOLDEN_JSON=$TMP/base.json pytest <this file>`.

Regenerate the committed golden file only for an intentional numerics change,
and say so in the PR: `MK_REGEN_GOLDEN=1 pytest <this file>`.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import pathlib
import platform
import sys
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel.modules.gqa import GQATunables
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35

GOLDEN = pathlib.Path(__file__).parent / "golden" / "b1_bitexact.json"
_META_KEY = "_meta"
R = 2


def _cpu_model_name() -> str:
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                _, _, val = line.partition(":")
                if val.strip():
                    return val.strip()
    return platform.processor() or "unknown"


def _host_fingerprint() -> dict[str, Any]:
    return {
        "cpu": f"{_cpu_model_name()} ({platform.machine()})",
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def _fingerprint_matches(meta: Any, cur: dict[str, Any]) -> bool:
    if not isinstance(meta, dict):
        return False
    return {k: v for k, v in meta.items() if k != "gpr_resolved"} == cur


def _format_fingerprint(fp: dict[str, Any] | None) -> str:
    if not isinstance(fp, dict):
        return "<missing _meta>"
    return (
        f"{fp.get('cpu', '?')} "
        f"(jax={fp.get('jax', '?')}, jaxlib={fp.get('jaxlib', '?')}, "
        f"numpy={fp.get('numpy', '?')}, python={fp.get('python', '?')})"
    )


def _golden_path() -> tuple[pathlib.Path, bool]:
    override = os.environ.get("MK_GOLDEN_JSON", "").strip()
    if override:
        return pathlib.Path(override), True
    return GOLDEN, False


def _write_golden_case(
    golden_path: pathlib.Path, name: str, got: dict[str, Any]
) -> None:
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = golden_path.with_name(golden_path.name + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        gold = json.loads(golden_path.read_text()) if golden_path.exists() else {}
        fp = _host_fingerprint()
        if "gpr" in got:
            gold[_META_KEY] = {**fp, "gpr_resolved": True}
            gold[name] = got
        else:
            gold.setdefault(_META_KEY, fp)
            gold[name] = {**got, _META_KEY: fp}
        golden_path.write_text(json.dumps(gold, indent=1, sort_keys=True) + "\n")


CASES: dict[str, dict[str, Any]] = {
    # Qwen3.5 default profile (qwen35-baseline), production-size and tiny tiles
    "def_t1_b4_prod": dict(profile="qwen35-baseline", T=1, B=4, tiles_mode="prod"),
    "def_t1_b8": dict(profile="qwen35-baseline", T=1, B=8),
    "def_t1_b20_wide": dict(profile="qwen35-baseline", T=1, B=20),
    "def_t2_b4": dict(profile="qwen35-baseline", T=2, B=4),
    "def_t4_b16": dict(profile="qwen35-baseline", T=4, B=16),
    "def_t4_b24_gch": dict(profile="qwen35-baseline", T=4, B=24),
    "def_t4_b48": dict(profile="qwen35-baseline", T=4, B=48),
    "def_t1_b8_moetp": dict(profile="qwen35-baseline", T=1, B=8, moe_tp=True),
    "def_t1_b8_eplb": dict(profile="qwen35-baseline", T=1, B=8, e_slots=72),
    "def_t2_b4_stacke": dict(profile="qwen35-baseline", T=2, B=4, stack_e=True),
    "def_t4_b24_gpr": dict(profile="qwen35-baseline", T=4, B=24, pool_gather=True),
    # qwen35-candidate30 (approx), split / both sides of moe_group_rows
    "c30_t1_b4": dict(profile="qwen35-candidate30", T=1, B=4),
    "c30_t1_b4_prod": dict(profile="qwen35-candidate30", T=1, B=4, tiles_mode="prod"),
    "c30_t4_b8": dict(profile="qwen35-candidate30", T=4, B=8),
    "c30_t4_b16": dict(profile="qwen35-candidate30", T=4, B=16),
    "c30_t4_b24": dict(profile="qwen35-candidate30", T=4, B=24),
    "c30_t4_b40": dict(profile="qwen35-candidate30", T=4, B=40),
    "c30_t4_b48": dict(profile="qwen35-candidate30", T=4, B=48),
    "c30_t1_b8_moetp": dict(profile="qwen35-candidate30", T=1, B=8, moe_tp=True),
    "c30_t1_b8_eplb": dict(profile="qwen35-candidate30", T=1, B=8, e_slots=72),
    # qwen35-candidate30 tiles + fused pool I/O (= qwen35-candidate30-fused)
    "c30_fused_t1_b4": dict(profile="qwen35-candidate30", T=1, B=4, fused=True),
    "c30_fused_t4_b8": dict(profile="qwen35-candidate30", T=4, B=8, fused=True),
    "c30_fused_t4_b24": dict(profile="qwen35-candidate30", T=4, B=24, fused=True),
    # Pure-JAX reference_step output and state hashes across TP, EP+EPLB,
    # paged pool I/O, float32 RoPE intermediates, and attention-only configs.
    "ref_tiny_t1_tp": dict(kind="ref", seed=0, T=1, B=2, ctx=64),
    "ref_tiny_t2_tp": dict(kind="ref", seed=1, T=2, B=4, ctx=64),
    "ref_tiny_t1_ep_eplb72": dict(
        kind="ref", seed=2, T=1, B=2, ctx=64, moe_tp=False, e_slots=72
    ),
    "ref_tiny_t2_paged_pool": dict(
        kind="ref", seed=3, T=2, B=4, ctx=128, paged_pool=True
    ),
    "ref_tiny_t1_rope_fp32": dict(kind="ref", seed=4, T=1, B=2, ctx=64, rope_fp32=True),
    "ref_tiny_t1_att_only": dict(
        kind="ref", seed=5, T=1, B=2, ctx=64, L=2, att_every=1
    ),
}


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _geom(moe_tp: bool = False, e_slots: int = 0) -> qwen35.Qwen35Geometry:
    return qwen35.Qwen35Geometry(
        L=4, H=256, I=256, IS=256, E=128, TOPK=2, NH=4, NKV=2, HD=256, ROT=64,
        KH=2, VH=4, KD=128, VD=128, att_every=4, moe_tp=moe_tp, e_slots=e_slots,
    )  # fmt: skip


def _tiny(t: mk.Tiles, ctx: int) -> mk.Tiles:
    """Shrink ring / bank sizes to the CPU geometry, keep every schedule flag."""
    return dataclasses.replace(
        t,
        bank=512 << 10,
        nbd=10,
        nbe=max(6, min(t.nbe, 12)),
        ebank=256 << 10,
        vmem_mib=32,
    ).with_modules(
        {
            "gqa": {
                "kv_blk": min(256, ctx),
                "kv_sub": 128 if t.module("gqa").kv_sub else 0,
            }
        }
    )


def _h(a: Any) -> str:
    return hashlib.sha256(np.asarray(a).view(np.uint8).tobytes()).hexdigest()[:24]


def _arr_hash(arr: jax.Array) -> str:
    np_arr = np.asarray(arr)
    raw = np_arr.view(np.uint8).tobytes()
    return f"{np_arr.dtype}:{list(np_arr.shape)}:{hashlib.sha256(raw).hexdigest()}"


def _pinned_b1_golden_inputs(
    key: jax.Array, g: qwen35.Qwen35Geometry, ex: mk.ExecSpec
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Pinned input tensor stream matching `b1_bitexact.json` / `aaafa5d66`."""
    from vllm_torchtpu.megakernel.modules.gdn import GDNModule
    from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, I32

    R_, B, T, NS, ctx = ex.R, ex.B, ex.T, ex.NS, ex.ctx
    L, H, E, HD = int(g.L), int(g.H), int(g.E), int(g.HD)
    nh = int(g.nh(R_))
    lay_gdn = GDNModule.lin_in_layout(g, ex)
    vh, KD, VD = lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
    qkv, z, nli = lay_gdn.QKV, lay_gdn.Z, lay_gdn.NLI
    isr, ir = int(g.ISR(R_)), int(g.IR(R_))
    er, ew = int(g.ER(R_)), int(g.EW(R_))
    natt, ngdn = int(g.NATT), int(g.NGDN)
    it = iter(jax.random.split(key, 36))

    base_pos = jnp.arange(1, NS + 1, dtype=I32) * min(16, ctx // 4)
    pos = (
        base_pos
        if T == 1
        else (base_pos[:, None] + jnp.arange(T, dtype=I32)[None, :]).reshape(B)
    )
    hidden = jax.random.normal(next(it), (B, H), dtype=BF16) * 0.1

    def rfp8(shape: tuple[int, ...]) -> jax.Array:
        return (jax.random.normal(next(it), shape, dtype=F32) * 0.25).astype(FP8)

    def rscale(shape: tuple[int, ...]) -> jax.Array:
        return jnp.abs(jax.random.normal(next(it), shape, dtype=F32)) * 0.02 + 0.01

    G, nkv, koff = int(g.G(R_)), int(g.NKV), nh * 2 * HD
    att_q = rfp8((R_, natt, H, koff))
    att_kv = jnp.repeat(rfp8((nkv, natt, H, 2 * HD)), G, axis=0)
    att_in = jnp.concatenate([att_q, att_kv], axis=-1)
    att_q_s = rscale((R_, natt, koff))
    att_kv_s = jnp.repeat(rscale((nkv, natt, 2 * HD)), G, axis=0)
    att_in_s = jnp.concatenate([att_q_s, att_kv_s], axis=-1)
    kn = jnp.repeat(
        jax.random.normal(next(it), (nkv, natt, HD), dtype=F32) * 0.05, G, axis=0
    )
    sh_gate = jnp.broadcast_to(
        (jax.random.normal(next(it), (L, H), dtype=F32) * 0.1)[None, :, :], (R_, L, H)
    )
    e13, e13_s = rfp8((R_, L, ew, H, 2 * ir)), rscale((R_, L, ew, 2 * ir))
    e2, e2_s = rfp8((R_, L, ew, ir, H)), rscale((R_, L, ew, H))
    in_n = jax.random.normal(next(it), (L, H), dtype=F32) * 0.05
    post_n_w = jax.random.normal(next(it), (L, H), dtype=F32) * 0.05
    final_n_w = jax.random.normal(next(it), (H,), dtype=F32) * 0.05
    att_out, att_out_s = rfp8((R_, natt, nh * HD, H)), rscale((R_, natt, H))
    qn = jax.random.normal(next(it), (R_, natt, HD), dtype=F32) * 0.05
    router = jax.random.normal(next(it), (L, H, E), dtype=BF16) * 0.1
    sh13, sh13_s = rfp8((R_, L, H, 2 * isr)), rscale((R_, L, 2 * isr))
    sh2, sh2_s = rfp8((R_, L, isr, H)), rscale((R_, L, H))
    logical_weights: dict[str, Any] = dict(
        in_norm=in_n, post_norm=post_n_w, final_norm=final_n_w, qn=qn, kn=kn
    )
    logical_weights.update(
        att_in=att_in, att_in_s=att_in_s, att_out=att_out, att_out_s=att_out_s
    )
    logical_weights.update(
        router=router, sh_gate=sh_gate, sh13=sh13, sh13_s=sh13_s, sh2=sh2, sh2_s=sh2_s
    )
    logical_weights.update(e13=e13, e13_s=e13_s, e2=e2, e2_s=e2_s)
    if not g.moe_tp and ew > er and R_ > 1:
        e_cores = mk.default_eplb_cores(g, R_)
        for lyr in range(L):
            for r1 in range(R_):
                for k_idx, g_id in enumerate(e_cores[lyr][r1][er:ew]):
                    r0, s0 = divmod(int(g_id), er)
                    slot = er + k_idx
                    e13 = e13.at[r1, lyr, slot].set(e13[r0, lyr, s0])
                    e13_s = e13_s.at[r1, lyr, slot].set(e13_s[r0, lyr, s0])
                    e2 = e2.at[r1, lyr, slot].set(e2[r0, lyr, s0])
                    e2_s = e2_s.at[r1, lyr, slot].set(e2_s[r0, lyr, s0])
        ecopy = mk.encode_eplb_ecopy(g, R_, e_cores)
        logical_weights.update(
            e13=e13, e13_s=e13_s, e2=e2, e2_s=e2_s, e_cores=e_cores, ecopy=ecopy
        )
    if ngdn > 0:
        logical_weights.update(
            {
                "lin_in": rfp8((R_, ngdn, H, nli)),
                "lin_in_s": rscale((R_, ngdn, nli)),
                "lin_out": rfp8((R_, ngdn, z, H)),
                "lin_out_s": rscale((R_, ngdn, H)),
                "conv": jax.random.normal(next(it), (R_, ngdn, 4 * qkv), dtype=F32)
                * 0.05,
                "alog": jax.random.normal(next(it), (R_, ngdn, vh), dtype=F32) * 0.1
                - 1.0,
                "dtb": jax.random.normal(next(it), (R_, ngdn, vh), dtype=F32) * 0.1,
                "gnorm": jnp.ones((R_, ngdn, VD), dtype=F32)
                + jax.random.normal(next(it), (R_, ngdn, VD), dtype=F32) * 0.05,
            }
        )
    kc_nkv = jax.random.normal(next(it), (nkv, natt, NS, ctx, HD), dtype=BF16) * 0.1
    vc_nkv = jax.random.normal(next(it), (nkv, natt, NS, ctx, HD), dtype=BF16) * 0.1
    logical_states: dict[str, Any] = {
        "kc": jnp.repeat(kc_nkv, G, axis=0),
        "vc": jnp.repeat(vc_nkv, G, axis=0),
    }
    if ngdn > 0:
        logical_states["conv"] = (
            jax.random.normal(next(it), (R_, ngdn, NS, 3, qkv), dtype=F32) * 0.05
        )
        logical_states["rec"] = (
            jax.random.normal(next(it), (R_, ngdn, NS, vh, KD, VD), dtype=BF16) * 0.05
        )
    return pos, hidden, logical_weights, logical_states


def run_ref_case(
    *,
    seed: int,
    T: int,
    B: int,
    ctx: int,
    moe_tp: bool = True,
    e_slots: int = 0,
    paged_pool: bool = False,
    rope_fp32: bool = False,
    L: int | None = None,
    att_every: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    g = qwen35.TINY if moe_tp else qwen35.TINY_EP
    if e_slots:
        g = dataclasses.replace(g, e_slots=e_slots)
    if L is not None:
        g = dataclasses.replace(g, L=L)
    if att_every is not None:
        g = dataclasses.replace(g, att_every=att_every)
    cache = mk.CacheExec(paged_kv=True, num_pages=64) if paged_pool else mk.CacheExec()
    tiles: mk.Tiles | None = None
    if rope_fp32:
        if "rope_fp32" not in GQATunables.__dataclass_fields__:
            pytest.skip("GQATunables.rope_fp32 not supported at this ref")
        tiles = mk.Tiles(modules={"gqa": GQATunables(rope_fp32=True)})
    ex = mk.ExecSpec(R=R, B=B, T=T, ctx=ctx, cache=cache, tiles=tiles)
    pos, hid, w_l, st_l = _pinned_b1_golden_inputs(jax.random.PRNGKey(seed), g, ex)
    out, st = qwen35.reference_step(g, ex, pos, hid, w_l, st_l, pool_io=paged_pool)
    return {
        "out": _arr_hash(out),
        "states": {k: _arr_hash(st[k]) for k in sorted(st)},
    }


def _dispatch_case(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg.get("kind") == "ref":
        return run_ref_case(**cfg)
    return run_case(**cfg)


def run_case(
    *,
    profile: str,
    T: int,
    B: int,
    fused: bool = False,
    pool_gather: Any = None,
    tiles_mode: str = "tiny",
    moe_tp: bool = False,
    e_slots: int = 0,
    stack_e: bool = False,
) -> dict[str, Any]:
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"Need at least {R} CPU devices")
    mesh = Mesh(np.array(devs[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    g = _geom(moe_tp=moe_tp, e_slots=e_slots)
    ctx = 256
    base = mk.ExecSpec(
        R=R, B=B, T=T, ctx=ctx, stack_e=stack_e, interpret=True,
        cache=mk.CacheExec(paged_kv=True, num_pages=512 if B <= 8 else 2048),
    )  # fmt: skip
    ex = mk.get_profile(profile).exec_for(base, B=B, T=T, g=g)
    t = ex.resolved_tiles
    if tiles_mode == "tiny":
        t = _tiny(t, ctx)
    else:  # production ring / bank sizes; only the KV block must fit `ctx`
        gqa = t.module("gqa")
        t = t.with_modules(
            {
                "gqa": {
                    "kv_blk": min(gqa.kv_blk, ctx),
                    "kv_sub": min(gqa.kv_sub, ctx // 2) if gqa.kv_sub else 0,
                }
            }
        )
    ex = dataclasses.replace(ex, tiles=t)
    if pool_gather is not None:
        ex = mk.with_cache(ex, pool_gather=pool_gather)
    spec = qwen35.qwen35_spec(g)
    pos, hid_l, w_l, st_l = _pinned_b1_golden_inputs(
        jax.random.key(900 + 10 * T + B), g, ex
    )
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, _ = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    if T == 1:
        n_read = int(ckpt.max()) - int(ckpt.min()) + 1
        ckpt = jnp.concatenate([ckpt, ckpt + n_read + 4], axis=-1)
        roff = roff.at[:, 1].set(-1)
    pools = tuple(
        jax.device_put(np.asarray(r).copy(), tp)
        for r in mk.states_to_pool(g, ex, st_l, pt, ckpt[..., :T], jnp.maximum(roff, 0))
    )
    step, info = mk.PallasWriter(fused_pool_io=fused).make_step(
        mesh, spec, ex, donate=fused
    )
    out, raw = step(pos, hid_mk, w_mk, pools, pt, ckpt, roff)
    return {
        "hid": _h(out),
        "pools": [_h(r) for r in raw],
        "gpr": bool(info["use_gpr"]),
    }


@pytest.mark.golden
@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_bitexact(name: str):
    """Hidden + every pool or reference_step output reproduces the golden hash."""
    golden_path, is_custom_golden = _golden_path()
    if os.environ.get("MK_REGEN_GOLDEN"):
        got = _dispatch_case(CASES[name])
        _write_golden_case(golden_path, name, got)
        pytest.skip("regenerated golden")
    all_gold = json.loads(golden_path.read_text())
    if not is_custom_golden:
        case_entry = all_gold.get(name, {})
        is_ref_case = CASES[name].get("kind") == "ref"
        meta = (
            case_entry.get(_META_KEY)
            if is_ref_case or (isinstance(case_entry, dict) and _META_KEY in case_entry)
            else all_gold.get(_META_KEY)
        )
        cur = _host_fingerprint()
        if not _fingerprint_matches(meta, cur):
            pytest.skip(
                f"golden recorded on {_format_fingerprint(meta)}, "
                f"this host is {_format_fingerprint(cur)}; "
                "use base-vs-head mode: "
                "tests/megakernel/golden/compare_base_head.sh <base-ref>"
            )
    elif name not in all_gold:
        pytest.skip(f"case {name!r} only exists in HEAD (not in {golden_path})")
    got = _dispatch_case(CASES[name])
    gold = all_gold[name]
    if "out" in gold:
        assert got["out"] == gold["out"], (name, "out", got["out"], gold["out"])
        assert got["states"] == gold["states"], (
            name,
            "states",
            got["states"],
            gold["states"],
        )
    else:
        assert got["hid"] == gold["hid"], (name, "hidden", got["hid"], gold["hid"])
        assert got["pools"] == gold["pools"], (
            name,
            "pools",
            got["pools"],
            gold["pools"],
        )
        if all_gold.get(_META_KEY, {}).get("gpr_resolved"):
            assert got["gpr"] == gold["gpr"], (name, "gpr", got["gpr"], gold["gpr"])


def test_golden_covers_every_case():
    golden_path, is_custom_golden = _golden_path()
    if os.environ.get("MK_REGEN_GOLDEN") and os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.skip("regenerating golden across xdist workers")
    gold_cases = set(json.loads(golden_path.read_text())) - {_META_KEY}
    head_cases = set(CASES)
    if is_custom_golden:
        base_only = sorted(gold_cases - head_cases)
        head_only = sorted(head_cases - gold_cases)
        if base_only or head_only:
            pytest.skip(
                f"case set differs from {golden_path}: "
                f"base_only={base_only}, head_only={head_only}"
            )
    assert gold_cases == head_cases


def test_golden_regen_writes_per_case_meta(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """`MK_REGEN_GOLDEN=1` writes per-case `_meta` and never uses global `_meta`."""
    tmp_json = tmp_path / "golden.json"
    other_fp = {
        "cpu": "Other CPU (x86_64)",
        "jax": "0.0.0",
        "jaxlib": "0.0.0",
        "numpy": "0.0.0",
        "python": "3.12.0",
    }
    tmp_json.write_text(json.dumps({_META_KEY: other_fp}) + "\n")
    this_mod = sys.modules[__name__]
    fake_ref = {"out": "bf16:[2,128]:deadbeef", "states": {"kv": "fp8:[2]:cafe"}}
    monkeypatch.setattr(this_mod, "_dispatch_case", lambda cfg: dict(fake_ref))
    monkeypatch.setenv("MK_REGEN_GOLDEN", "1")
    monkeypatch.setenv("MK_GOLDEN_JSON", str(tmp_json))
    with pytest.raises(pytest.skip.Exception, match="regenerated golden"):
        test_golden_bitexact("ref_tiny_t1_tp")

    data = json.loads(tmp_json.read_text())
    assert data[_META_KEY] == other_fp
    assert data["ref_tiny_t1_tp"][_META_KEY] == _host_fingerprint()
    assert data["ref_tiny_t1_tp"]["out"] == fake_ref["out"]

    # In committed-golden mode, per-case `_meta` matches this host even though
    # global `_meta` is `other_fp`; removing per-case `_meta` must skip even if
    # global `_meta` matches `_host_fingerprint()`.
    monkeypatch.delenv("MK_REGEN_GOLDEN")
    monkeypatch.delenv("MK_GOLDEN_JSON")
    monkeypatch.setattr(this_mod, "GOLDEN", tmp_json)
    test_golden_bitexact("ref_tiny_t1_tp")

    data[_META_KEY] = _host_fingerprint()
    del data["ref_tiny_t1_tp"][_META_KEY]
    tmp_json.write_text(json.dumps(data) + "\n")
    with pytest.raises(
        pytest.skip.Exception, match="golden recorded on <missing _meta>"
    ):
        test_golden_bitexact("ref_tiny_t1_tp")
