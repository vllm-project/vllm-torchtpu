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
"""Cloudtop AOT TPU7x compiler stats, full normalized LLO/Mosaic diffing, and roofline
math.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

# Default path where cloudtop isolated libtpu package is installed if not in active
# venv.
DEFAULT_CLOUDTOP_LIBTPU_SITE = pathlib.Path.home() / "mk" / "tpu_aot_pkgs"
DEFAULT_AOT_TOPOLOGY = "tpu7x:2x2x1"

# Analytic roofline constants from megakernel_roofline.md / roofline.py for
# Qwen3.5-397B-A17B-FP8 on v7x-8.
V7X_HBM_BW_BYTES_PER_US = 3.69e6  # 3.69 TB/s per TensorCore = 3.69e6 B/us
V7X_BF16_FLOPS_PER_US = 1153.5e6  # 1,153.5 TFLOP/s per TensorCore = 1153.5e6 FLOP/us
QWEN35_FIXED_WEIGHT_BYTES_PER_RANK = 1_252_833_280  # 1.253 GB across 60 layers
QWEN35_EXPERT_WEIGHT_BYTES = (
    12_597_248  # 12.60 MB per routed expert per layer (FP8 + scales)
)
QWEN35_NUM_LAYERS = 60
QWEN35_NUM_GDN_LAYERS = 45
QWEN35_NUM_GQA_LAYERS = 15
QWEN35_NUM_EXPERTS = 512
QWEN35_TOP_K = 10

# Multinomial order-statistic max-rank / mean-rank EP load factor L_EP(B) for E=512,
# K=10, R=8.
_EP_IMBALANCE_TABLE: dict[int, float] = {
    1: 1.98,
    2: 1.78,
    4: 1.61,
    8: 1.45,
    12: 1.35,
    16: 1.29,
    20: 1.24,
    24: 1.20,
    32: 1.19,
    40: 1.15,
    48: 1.13,
    56: 1.11,
    64: 1.10,
}


class ToolchainMismatchError(RuntimeError):
    """Raised when comparing compiler dumps produced by mismatched toolchains
    or flags.
    """


def ensure_libtpu_on_path() -> None:
    """Add `~/mk/tpu_aot_pkgs` to `sys.path` if `libtpu` is not already importable."""
    if not pathlib.Path("/dev/vfio").is_dir():
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("TPU_WORKER_HOSTNAMES", "localhost")
    try:
        import libtpu  # noqa: F401
    except ImportError:
        if DEFAULT_CLOUDTOP_LIBTPU_SITE.is_dir():
            site_str = str(DEFAULT_CLOUDTOP_LIBTPU_SITE)
            if site_str not in sys.path:
                sys.path.insert(0, site_str)


def _strip_dump_flags(flag_str: str) -> str:
    """Strip per-invocation dump directory flags so toolchain comparisons check semantic
    flags.
    """
    if not flag_str:
        return ""
    tokens = []
    for tok in flag_str.split():
        if tok.startswith(("--xla_mosaic_dump_to=", "--xla_jf_dump_to=")):
            continue
        tokens.append(tok)
    return " ".join(sorted(tokens))


def get_toolchain_info(topology_name: str = DEFAULT_AOT_TOPOLOGY) -> dict[str, str]:
    """Return canonical toolchain version and flag metadata for gate validation."""
    ensure_libtpu_on_path()
    dist_versions: dict[str, str] = {}
    for pkg in ("jax", "jaxlib", "libtpu"):
        try:
            dist_versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            dist_versions[pkg] = "not-installed"

    libtpu_so_sha = "none"
    try:
        import libtpu

        so_path = pathlib.Path(libtpu.get_library_path())
        if so_path.is_file():
            st = so_path.stat()
            libtpu_so_sha = f"size={st.st_size}"
    except Exception:
        libtpu_so_sha = "unknown"

    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    return {
        "python": py_ver,
        "jax": dist_versions["jax"],
        "jaxlib": dist_versions["jaxlib"],
        "libtpu": dist_versions["libtpu"],
        "libtpu_so": libtpu_so_sha,
        "topology_name": topology_name,
        "libtpu_init_args": _strip_dump_flags(os.environ.get("LIBTPU_INIT_ARGS", "")),
        "xla_flags": _strip_dump_flags(os.environ.get("XLA_FLAGS", "")),
    }


def check_toolchain_match(
    tc_a: dict[str, Any], tc_b: dict[str, Any]
) -> tuple[bool, str]:
    """Verify two toolchain metadata dicts match on compiler versions and flags."""
    keys = (
        "python",
        "jax",
        "jaxlib",
        "libtpu",
        "libtpu_so",
        "topology_name",
        "libtpu_init_args",
        "xla_flags",
    )
    diffs = []
    for k in keys:
        va, vb = tc_a.get(k), tc_b.get(k)
        if va != vb:
            diffs.append(f"{k}: base={va!r} vs head={vb!r}")
    if diffs:
        return False, "INVALID_TOOLCHAIN (" + "; ".join(diffs) + ")"
    return True, "OK"


def ep_imbalance_factor(B: int) -> float:
    """Return the EP max-rank/mean-rank load imbalance factor `L_EP(B)`."""
    if B in _EP_IMBALANCE_TABLE:
        return _EP_IMBALANCE_TABLE[B]
    keys = sorted(_EP_IMBALANCE_TABLE.keys())
    if keys[0] >= B:
        return _EP_IMBALANCE_TABLE[keys[0]]
    if keys[-1] <= B:
        return _EP_IMBALANCE_TABLE[keys[-1]]
    for i in range(len(keys) - 1):
        k0, k1 = keys[i], keys[i + 1]
        if k0 <= B <= k1:
            w = (B - k0) / (k1 - k0)
            return _EP_IMBALANCE_TABLE[k0] * (1.0 - w) + _EP_IMBALANCE_TABLE[k1] * w
    return 1.10


def roofline_stats(
    B: int,
    T: int = 1,
    *,
    ctx: int = 4096,
    pos: int | None = None,
    moe_tp: bool = False,
    R: int = 8,
    median_us: float | None = None,
) -> dict[str, float]:
    """Compute analytic HBM roofline floor and headroom for
    Qwen3.5-397B-A17B-FP8 on v7x-8.
    """
    ns = max(1, B // max(1, T))
    mb = ((B + 7) // 8) * 8
    eff_pos = (65536 if ctx > 32768 else min(2048, ctx // 2)) if pos is None else pos

    # Expected unique routed experts U(B) across E=512 with top-10 uniform routing:
    # A single row picks K=10 distinct experts out of E=512; P(miss 1 expert) = (E - K)
    # / E.
    p_miss_one_row = (QWEN35_NUM_EXPERTS - QWEN35_TOP_K) / QWEN35_NUM_EXPERTS
    u_b = QWEN35_NUM_EXPERTS * (1.0 - (p_miss_one_row**B))

    if moe_tp:
        # Under MoE TP, every rank loads 1/R of every hit expert's weights (no EP
        # imbalance).
        expert_bytes_per_layer = u_b * (QWEN35_EXPERT_WEIGHT_BYTES / R)
        l_ep = 1.0
    else:
        l_ep = ep_imbalance_factor(B)
        mean_experts_per_rank = u_b / R
        max_experts_on_crit_rank = min(
            QWEN35_NUM_EXPERTS // R, mean_experts_per_rank * l_ep
        )
        expert_bytes_per_layer = max_experts_on_crit_rank * QWEN35_EXPERT_WEIGHT_BYTES

    total_expert_bytes = QWEN35_NUM_LAYERS * expert_bytes_per_layer
    # GDN reads NS recurrent states (256 KiB each) and writes B state checkpoints (256
    # KiB each)
    gdn_state_bytes = QWEN35_NUM_GDN_LAYERS * (ns + B) * 262_144
    kv_cache_bytes = QWEN35_NUM_GQA_LAYERS * ns * eff_pos * 512  # 2 * 256 B per token

    crit_bytes = (
        QWEN35_FIXED_WEIGHT_BYTES_PER_RANK
        + total_expert_bytes
        + gdn_state_bytes
        + kv_cache_bytes
    )
    floor_us = crit_bytes / V7X_HBM_BW_BYTES_PER_US
    bf16_flops = 2.0 * mb * (QWEN35_FIXED_WEIGHT_BYTES_PER_RANK + total_expert_bytes)
    mxu_bf16_us = bf16_flops / V7X_BF16_FLOPS_PER_US

    out: dict[str, float] = {
        "u_experts": round(u_b, 1),
        "ep_imbalance": round(l_ep, 2),
        "crit_bytes_gb": round(crit_bytes / 1e9, 3),
        "expert_share_pct": round(total_expert_bytes / crit_bytes * 100.0, 1),
        "roofline_floor_us": round(floor_us, 1),
        "mxu_bf16_us": round(mxu_bf16_us, 1),
    }
    if median_us is not None and median_us > 0:
        out["hbm_bw_util_pct"] = round((floor_us / median_us) * 100.0, 1)
        out["headroom_us"] = round(median_us - floor_us, 1)
        out["headroom_x"] = round(median_us / floor_us, 2)
    return out


def plan_bucket_stats(
    profile_or_name: Any,
    T: int,
    B: int,
    *,
    ctx: int = 4096,
    num_pages: int = 4096,
    e_slots: int = 72,
    moe_tp: bool = False,
    R: int = 8,
    stack_e: bool = False,
    fused_pool_io: bool | str | None = None,
    fused_pool_io_max_b: int | None = None,
    fused_pool_io_min_b_hi: int | None = None,
) -> dict[str, Any]:
    """Compute fast CPU static plan, VMEM/SMEM usage, and tunable breakdown for
    `(T, B)`.
    """
    from vllm_torchtpu.megakernel import vmem
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.models import qwen35
    from vllm_torchtpu.megakernel.modules.gdn import GDNModule, GDNTunables
    from vllm_torchtpu.megakernel.modules.gqa import GQAModule, GQATunables
    from vllm_torchtpu.megakernel.modules.moe import MoEModule, MoETunables
    from vllm_torchtpu.megakernel.profile import Profile, get_profile
    from vllm_torchtpu.megakernel.runtime.policy import (
        align_cache_exec,
        bucket_exec_spec,
    )

    prof = (
        profile_or_name
        if isinstance(profile_or_name, Profile)
        else get_profile(profile_or_name)
    )
    base_g = qwen35.FULL if moe_tp else qwen35.FULL_EP
    g = dataclasses.replace(base_g, e_slots=(base_g.E // R if moe_tp else int(e_slots)))
    spec = qwen35.qwen35_spec(g)

    max_p = max(32, (ctx + 127) // 128)
    eff_pages = max(num_pages, (B // T) * max_p + 512)
    base_ex = align_cache_exec(
        ExecSpec(B=B, T=T, R=R, ctx=ctx, stack_e=stack_e),
        prof,
        g,
        num_pages=eff_pages,
        page_size=128,
        max_pages_per_seq=max_p,
    )
    ex, use_fused, use_gpr = bucket_exec_spec(
        spec,
        base_ex,
        prof,
        bucket_t=B,
        tokens_per_seq=T,
        fused_pool_io=fused_pool_io,
        fused_pool_io_max_b=fused_pool_io_max_b,
        fused_pool_io_min_b_hi=fused_pool_io_min_b_hi,
    )
    tiles = ex.resolved_tiles
    gqa_t = tiles.module(GQAModule.name, GQATunables)
    gdn_t = tiles.module(GDNModule.name, GDNTunables)
    moe_t = tiles.module(MoEModule.name, MoETunables)

    fits_vmem = True
    error_msg = ""
    vmem_bytes = 0
    smem_bytes = 0
    subkernel_vmem: dict[str, int] = {}

    try:
        import vllm_torchtpu.megakernel.experimental.fused_pool_io  # noqa: F401
        from vllm_torchtpu.megakernel.pool.config import cache_exec, with_cache

        cx = cache_exec(ex)
        default_ckpt_k = cx.ckpt_k if cx.ckpt_k > 0 else ex.T
        ex_k = with_cache(
            ex,
            act_mode=cx.act_mode,
            ckpt_2d=cx.ckpt_2d,
            ckpt_k=default_ckpt_k,
            roff_1d=cx.roff_1d,
        )
        _, info = qwen35.build(g, ex_k)
        vmem_bytes = int(info["vmem_scratch_bytes"])
        smem_bytes = int(info["smem_scratch_bytes"])
        subkernel_vmem[
            "fused_verify"
            if (T > 1 and use_fused)
            else ("split_verify_mk" if T > 1 else "decode")
        ] = vmem_bytes
    except Exception as exc:
        fits_vmem = False
        error_msg = f"{type(exc).__name__}: {exc}"
        try:
            cp, _ = vmem.build_scratch(spec, ex)
            vmem_bytes = int(cp.vmem_bytes)
            smem_bytes = int(cp.smem_bytes)
        except Exception as exc2:
            if not error_msg:
                error_msg = f"{type(exc2).__name__}: {exc2}"

    limit_bytes = int(tiles.vmem_mib) * (1 << 20)
    if vmem_bytes > limit_bytes:
        fits_vmem = False

    rl = roofline_stats(B=B, T=T, ctx=ctx, moe_tp=moe_tp, R=R)
    return {
        "profile": prof.name,
        "numerics": prof.numerics,
        "T": T,
        "B": B,
        "NS": ex.NS,
        "ctx": ctx,
        "moe_tp": moe_tp,
        "e_slots": g.e_slots,
        "fits_vmem": fits_vmem,
        "error": error_msg,
        "vmem_bytes": vmem_bytes,
        "vmem_mib": round(vmem_bytes / (1024 * 1024), 2),
        "vmem_limit_mib": tiles.vmem_mib,
        "smem_bytes": smem_bytes,
        "smem_kib": round(smem_bytes / 1024, 1),
        "subkernel_vmem": subkernel_vmem,
        "FUSED": bool(use_fused),
        "GPR": bool(use_gpr),
        "GCH": bool(gdn_t.conv_stage and T > 1 and use_fused),
        "tiles": {
            "nbd": tiles.nbd,
            "nbe": tiles.nbe,
            "hier": tiles.hier,
            "rsag": tiles.rsag,
            "mb8": tiles.mb8,
            "dense_acc_pair": tiles.dense_acc_pair,
            "dmm_unroll": tiles.dmm_unroll,
            "rel_fetch": tiles.rel_fetch,
            "nkv": gqa_t.nkv,
            "kv_blk": gqa_t.kv_blk,
            "kv_sub": gqa_t.kv_sub,
            "kv_ering": gqa_t.kv_ering,
            "pair_kv": gqa_t.pair_kv,
            "gq_pack": gqa_t.gq_pack,
            "gqa_qv": gqa_t.gqa_qv,
            "pack": gqa_t.pack,
            "gdn_rows": gdn_t.gdn_rows,
            "gdn_head_major": gdn_t.gdn_head_major,
            "conv_stage": gdn_t.conv_stage,
            "ecopy_opt": moe_t.ecopy_opt,
            "pipe_ep": moe_t.pipe_ep,
            "r2_direct_store": moe_t.r2_direct_store,
            "route_vec": moe_t.route_vec,
            "topk_lanes": moe_t.topk_lanes,
            "moe_unroll": moe_t.moe_unroll,
        },
        **rl,
    }


_SUBKERNEL_CANON_RE = re.compile(r"\.\d+$")
_STACK_FRAME_RE = re.compile(r"stack_frame_id=\d+")
_CUSTOM_BODY_RE = re.compile(r'"body":"[^"]*"')
_REPO_PATH_RE = re.compile(r'"[^"\s]*?/(src/vllm_torchtpu/|tests/megakernel/)')
_MLIR_LOC_LINE_RE = re.compile(r'(\.(?:py|cc|h)"):\d+:\d+(?:\s+to\s+:\d+:\d+)?')
_LINE_LOC_RE = re.compile(r"\bline=\d+\b")
_TS_PREFIX_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{6}\s+", re.MULTILINE)
_INLINE_SUBKERNEL_ID_RE = re.compile(
    r"((?:qwen3_5_decode_megakernel|decode_megakernel|qwen3_5_gather_gdn_states|gather_gdn_states|qwen3_5_scatter_pool_states|scatter_pool_states|stage[123]))\.\d+\b"
)


def _canonical_subkernel_name(raw_name: str) -> str:
    """Normalize `<kernel_name>.<id>` to `<kernel_name>` while preserving distinct copy
    ops.
    """
    if raw_name.startswith(("copy.", "<", "TLP")):
        return raw_name
    name = _SUBKERNEL_CANON_RE.sub("", raw_name)
    if name.startswith("qwen3_5_") and (
        "gather_gdn_states" in name or "scatter_pool_states" in name
    ):
        name = name[len("qwen3_5_") :]
    return name


def normalize_llo_text(text: str, *, is_schedule_analysis: bool = True) -> str:
    """Normalize volatile compiler metadata while keeping all instructions, bundles, and
    regs.
    """
    if "stack_frame_id=" in text:
        text = _STACK_FRAME_RE.sub("stack_frame_id=N", text)
    if (
        "megakernel." in text
        or "gather_gdn_states." in text
        or "scatter_pool_states." in text
        or "stage" in text
    ):
        text = _INLINE_SUBKERNEL_ID_RE.sub(r"\1", text)
    if is_schedule_analysis:
        if '"body":"' in text:
            # The serialized MLIR bytecode in HLO/schedule-analysis custom_call_config
            # embeds source file paths and line numbers, while its actual IR is compared
            # in text form by the Mosaic MLIR dump and LLO passes.
            text = _CUSTOM_BODY_RE.sub('"body":"<MLIR_BYTECODE>"', text)
        if "src/vllm_torchtpu/" in text or "tests/megakernel/" in text:
            text = _REPO_PATH_RE.sub(r'"<REPO>/\1', text)
        if '.py":' in text or '.cc":' in text or '.h":' in text:
            text = _MLIR_LOC_LINE_RE.sub(r"\1:0:0", text)
        if "line=" in text:
            text = _LINE_LOC_RE.sub("line=N", text)
        if "timestamp|heap_size_bytes" in text:
            text = _TS_PREFIX_RE.sub("", text)
            lines = text.splitlines()
            if len(lines) > 2:
                text = "\n".join(lines[:2] + sorted(lines[2:])) + "\n"
    if not text.endswith("\n"):
        text += "\n"
    return text


def parse_llo_dump_dir(dump_dir: str | pathlib.Path) -> dict[str, Any]:
    """Parse and hash the FULL LLO + Mosaic dump directory per subkernel."""
    dpath = pathlib.Path(dump_dir)
    files = sorted(f.name for f in dpath.iterdir() if f.is_file())

    mosaic_files: list[str] = []
    kernel_pass_files: dict[str, list[tuple[str, str]]] = {}

    for fname in files:
        if "mosaic-dump" in fname:
            mosaic_files.append(fname)
            continue
        m = re.match(r"^\d+-(.+)-(\d+-.+|hlo)\.txt$", fname)
        if m:
            raw_kname, pass_stage = m.group(1), m.group(2)
            kname = _canonical_subkernel_name(raw_kname)
            kernel_pass_files.setdefault(kname, []).append((pass_stage, fname))

    # Group Mosaic dumps by pallas_call prefix so multi-kernel modules sort
    # deterministically regardless of location-hashed prefix IDs.
    mosaic_groups: dict[str, list[tuple[str, str]]] = {}
    for mfname in mosaic_files:
        prefix, rest = mfname.split("-", 1) if "-" in mfname else ("0", mfname)
        raw_txt = (dpath / mfname).read_text(encoding="utf-8", errors="replace")
        norm_txt = normalize_llo_text(raw_txt, is_schedule_analysis=False)
        mosaic_groups.setdefault(prefix, []).append((rest, norm_txt))

    sorted_groups: list[list[tuple[str, str]]] = []
    for passes in mosaic_groups.values():
        passes.sort(key=lambda x: x[0])
        sorted_groups.append(passes)
    sorted_groups.sort(key=lambda g: (len(g), g[0][1] if g else ""))

    mosaic_hasher = hashlib.sha256()
    mosaic_entries: list[dict[str, Any]] = []
    idx = 0
    for grp in sorted_groups:
        for pass_rest, norm_txt in grp:
            m_sha = hashlib.sha256(norm_txt.encode("utf-8")).hexdigest()
            mosaic_hasher.update(f"=== mosaic[{idx}:{pass_rest}] ===\n".encode())
            mosaic_hasher.update(norm_txt.encode("utf-8"))
            mosaic_entries.append(
                {
                    "index": idx,
                    "pass": pass_rest,
                    "sha256": m_sha,
                    "bytes": len(norm_txt),
                }
            )
            idx += 1

    # Parse TLP bufferinfo for scoped VMEM allocations per subkernel
    scoped_vmem_by_kernel: dict[str, int] = {}
    for pass_stage, fname in kernel_pass_files.get("TLP", []):
        if "memory-space-assignment-bufferinfo" in pass_stage:
            txt = (dpath / fname).read_text(encoding="utf-8", errors="replace")
            for line in txt.splitlines():
                m_sc = re.search(
                    r'"scoped allocation for ([^"]+)",[^,]+,(\d+),', line
                ) or re.search(
                    r"scoped allocation for ([^\s\":]+)\s*:\s*size\s*(\d+)", line
                )
                if m_sc:
                    sk = _canonical_subkernel_name(m_sc.group(1))
                    scoped_vmem_by_kernel[sk] = max(
                        scoped_vmem_by_kernel.get(sk, 0), int(m_sc.group(2))
                    )

    subkernels: dict[str, dict[str, Any]] = {}
    full_dir_hasher = hashlib.sha256()
    full_dir_hasher.update(f"mosaic_all:{mosaic_hasher.hexdigest()}\n".encode())

    for kname in sorted(kernel_pass_files.keys()):
        passes = sorted(kernel_pass_files[kname], key=lambda x: x[0])
        k_full_hasher = hashlib.sha256()
        pass_hashes: dict[str, str] = {}
        total_bundles = 0
        non_empty_bundles = 0
        empty_bundles = 0
        final_bundles_sha256 = ""
        fingerprints_sha256 = ""
        last_fingerprint_line = ""
        spill_ops = 0
        vmem_spill_ops = 0
        smem_spill_ops = 0

        for pass_stage, fname in passes:
            raw_txt = (dpath / fname).read_text(encoding="utf-8", errors="replace")
            is_sched = (
                "schedule-analysis" in pass_stage
                or "static-bundle-profile" in pass_stage
                or "heap_sizes" in pass_stage
                or pass_stage == "hlo"
            )
            norm_txt = normalize_llo_text(raw_txt, is_schedule_analysis=is_sched)
            p_sha = hashlib.sha256(norm_txt.encode("utf-8")).hexdigest()
            pass_hashes[pass_stage] = p_sha
            k_full_hasher.update(f"=== {kname}:{pass_stage} ===\n".encode())
            k_full_hasher.update(norm_txt.encode("utf-8"))

            if pass_stage.endswith("-fingerprints"):
                fingerprints_sha256 = p_sha
                lines = [ln.strip() for ln in norm_txt.splitlines() if ln.strip()]
                if lines:
                    last_fingerprint_line = lines[-1]
            elif pass_stage.endswith("schedule-analysis_final_bundles"):
                m_tot = re.search(r"total scheduled bundles:\s+(\d+)", norm_txt)
                m_emp = re.search(r"empty scheduled bundles:\s+(\d+)", norm_txt)
                m_non = re.search(r"non empty scheduled bundles:\s+(\d+)", norm_txt)
                if m_tot:
                    total_bundles = int(m_tot.group(1))
                if m_emp:
                    empty_bundles = int(m_emp.group(1))
                if m_non:
                    non_empty_bundles = int(m_non.group(1))
            elif pass_stage.endswith("final_bundles"):
                final_bundles_sha256 = p_sha
                spill_ops = len(re.findall(r"_spill\b", norm_txt))
                vmem_spill_ops = len(
                    re.findall(r"vmem:\[#allocation\d+_spill", norm_txt)
                )
                smem_spill_ops = len(
                    re.findall(r"smem:\[#allocation\d+_spill", norm_txt)
                )
                if total_bundles == 0:
                    total_bundles = sum(
                        1
                        for ln in norm_txt.splitlines()
                        if re.match(r"^\s*(?:0x[0-9a-fA-F]+|\d+)\s*:\s*\{", ln)
                    )
                    non_empty_bundles = total_bundles

        k_full_sha = k_full_hasher.hexdigest()
        full_dir_hasher.update(f"kernel:{kname}:{k_full_sha}\n".encode())
        sc_bytes = scoped_vmem_by_kernel.get(kname, 0)
        subkernels[kname] = {
            "num_pass_files": len(passes),
            "full_llo_sha256": k_full_sha,
            "normalized_pass_sha256": k_full_sha,
            "final_bundles_sha256": final_bundles_sha256,
            "fingerprints_sha256": fingerprints_sha256,
            "last_fingerprint": last_fingerprint_line,
            "total_bundles": total_bundles,
            "non_empty_bundles": non_empty_bundles,
            "empty_bundles": empty_bundles,
            "spill_ops": spill_ops,
            "spill_count": spill_ops,
            "vmem_spill_ops": vmem_spill_ops,
            "smem_spill_ops": smem_spill_ops,
            "scoped_vmem_bytes": sc_bytes,
            "scoped_vmem_mib": round(sc_bytes / (1024 * 1024), 2),
            "pass_hashes": pass_hashes,
        }

    return {
        "full_llo_mosaic_sha256": full_dir_hasher.hexdigest(),
        "mosaic_sha256": mosaic_hasher.hexdigest(),
        "mosaic_files": mosaic_entries,
        "num_dump_files": len(files),
        "subkernels": subkernels,
    }


def compile_bucket_in_process(
    profile_or_name: Any,
    T: int,
    B: int,
    *,
    ctx: int = 4096,
    num_pages: int = 4096,
    e_slots: int = 72,
    moe_tp: bool = False,
    R: int = 8,
    stack_e: bool = True,
    fused_pool_io: bool | str | None = None,
    fused_pool_io_max_b: int | None = None,
    fused_pool_io_min_b_hi: int | None = None,
    topology_name: str = DEFAULT_AOT_TOPOLOGY,
    dump_dir: str | pathlib.Path | None = None,
    skip_parse: bool = False,
) -> dict[str, Any]:
    """Compile a single `(profile, T, B)` bucket via AOT `tpu7x` topology in the current
    process.
    """
    ensure_libtpu_on_path()
    own_tmp = False
    if dump_dir is None and "jax" not in sys.modules:
        dump_dir = tempfile.mkdtemp(prefix=f"mk_llo_inproc_t{T}_b{B}_")
        own_tmp = True
        base_init = _strip_dump_flags(os.environ.get("LIBTPU_INIT_ARGS", ""))
        dflags = f"--xla_mosaic_dump_to={dump_dir} --xla_jf_dump_to={dump_dir}"
        os.environ["LIBTPU_INIT_ARGS"] = (
            f"{base_init} {dflags}".strip() if base_init else dflags
        )
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("TPU_WORKER_HOSTNAMES", "localhost")

    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    import vllm_torchtpu.megakernel as mk
    from vllm_torchtpu.megakernel import cache
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.models import qwen35
    from vllm_torchtpu.megakernel.profile import Profile, get_profile
    from vllm_torchtpu.megakernel.runtime.policy import (
        align_cache_exec,
        bucket_exec_spec,
    )

    static_stats = plan_bucket_stats(
        profile_or_name,
        T,
        B,
        ctx=ctx,
        num_pages=num_pages,
        e_slots=e_slots,
        moe_tp=moe_tp,
        R=R,
        stack_e=stack_e,
        fused_pool_io=fused_pool_io,
        fused_pool_io_max_b=fused_pool_io_max_b,
        fused_pool_io_min_b_hi=fused_pool_io_min_b_hi,
    )
    tc_info = get_toolchain_info(topology_name=topology_name)

    topo = topologies.get_topology_desc(platform="tpu", topology_name=topology_name)
    mesh = Mesh(topo.devices[:R], ("tp",))
    base_g = qwen35.FULL if moe_tp else qwen35.FULL_EP
    g = dataclasses.replace(base_g, e_slots=(base_g.E // R if moe_tp else int(e_slots)))
    spec = qwen35.qwen35_spec(g)
    prof = (
        profile_or_name
        if isinstance(profile_or_name, Profile)
        else get_profile(profile_or_name)
    )

    max_p = max(32, (ctx + 127) // 128)
    eff_pages = max(num_pages, (B // T) * max_p + 512)
    base_ex = align_cache_exec(
        ExecSpec(B=B, T=T, R=R, ctx=ctx, stack_e=stack_e),
        prof,
        g,
        num_pages=eff_pages,
        page_size=128,
        max_pages_per_seq=max_p,
    )
    ex, _, _ = bucket_exec_spec(
        spec,
        base_ex,
        prof,
        bucket_t=B,
        tokens_per_seq=T,
        fused_pool_io=fused_pool_io,
        fused_pool_io_max_b=fused_pool_io_max_b,
        fused_pool_io_min_b_hi=fused_pool_io_min_b_hi,
    )

    t0 = time.perf_counter()
    step, info = cache.PallasWriter(
        fused_pool_io=fused_pool_io
        if fused_pool_io is not None
        else prof.fused_pool_io,
        max_b_low=int(
            fused_pool_io_max_b
            if fused_pool_io_max_b is not None
            else (prof.fused_pool_io_max_b or 24)
        ),
        min_b_hi=int(
            fused_pool_io_min_b_hi
            if fused_pool_io_min_b_hi is not None
            else (prof.fused_pool_io_min_b_hi or 56)
        ),
    ).make_step(mesh, spec, ex, donate=True)
    rep = NamedSharding(mesh, P())
    tp = NamedSharding(mesh, P("tp"))

    ce = mk.cache_exec(ex)
    pt_cols = mk.pool.resolved_max_pages_per_seq(ex)
    g_per_att = g.NGDN // g.NATT
    pos = jax.ShapeDtypeStruct((ex.B,), jnp.int32, sharding=rep)
    hid = jax.ShapeDtypeStruct((ex.R * ex.M, g.H), jnp.bfloat16, sharding=tp)
    pt = jax.ShapeDtypeStruct((ex.NS, pt_cols), jnp.int32, sharding=rep)
    ckpt = jax.ShapeDtypeStruct((g_per_att, ex.NS, ex.T + 1), jnp.int32, sharding=rep)
    roff = jax.ShapeDtypeStruct((g_per_att, ex.NS), jnp.int32, sharding=rep)
    raw_pools = tuple(
        jax.ShapeDtypeStruct(
            (ex.R * ce.num_pages, 2, 64, 4, ce.page_size),
            jnp.float8_e4m3fn,
            sharding=tp,
        )
        for _ in range(g.NATT)
    )
    wshape = info["wshape"]
    weights = {
        k: tuple(
            jax.ShapeDtypeStruct((ex.R, *s.shape), s.dtype, sharding=tp) for s in v
        )
        if isinstance(v, tuple) and hasattr(v[0], "shape")
        else jax.ShapeDtypeStruct((ex.R, *v[0]), v[1], sharding=tp)
        for k, v in wshape.items()
    }

    lowered = step.lower(pos, hid, weights, raw_pools, pt, ckpt, roff)
    hlo_text = normalize_llo_text(lowered.as_text())
    hlo_sha256 = hashlib.sha256(hlo_text.encode("utf-8")).hexdigest()
    compiled = lowered.compile()
    compile_time_s = time.perf_counter() - t0
    del compiled

    llo_summary: dict[str, Any] = {}
    if not skip_parse and dump_dir is not None and pathlib.Path(dump_dir).is_dir():
        llo_summary = parse_llo_dump_dir(dump_dir)
        if own_tmp:
            for f in pathlib.Path(dump_dir).iterdir():
                f.unlink(missing_ok=True)
            pathlib.Path(dump_dir).rmdir()

    # Extract primary megakernel bundle count for quick table display
    primary_bundles = 0
    primary_non_empty = 0
    primary_spills = 0
    for kname, kinfo in llo_summary.get("subkernels", {}).items():
        if "megakernel" in kname:
            primary_bundles += int(kinfo.get("total_bundles", 0))
            primary_non_empty += int(kinfo.get("non_empty_bundles", 0))
            primary_spills += int(kinfo.get("spill_ops", 0))

    return {
        **static_stats,
        "vmem_bytes": int(info.get("vmem_scratch_bytes", static_stats["vmem_bytes"])),
        "vmem_mib": round(
            int(info.get("vmem_scratch_bytes", static_stats["vmem_bytes"]))
            / (1024 * 1024),
            2,
        ),
        "smem_bytes": int(info.get("smem_scratch_bytes", static_stats["smem_bytes"])),
        "smem_kib": round(
            int(info.get("smem_scratch_bytes", static_stats["smem_bytes"])) / 1024,
            1,
        ),
        "compile_ok": True,
        "compile_s": round(compile_time_s, 2),
        "compile_time_s": round(compile_time_s, 2),
        "hlo_sha256": hlo_sha256,
        "primary_total_bundles": primary_bundles,
        "primary_non_empty_bundles": primary_non_empty,
        "primary_spill_ops": primary_spills,
        "llo": llo_summary,
        "toolchain": tc_info,
    }


def compile_bucket_subprocess(
    profile_name: str,
    T: int,
    B: int,
    *,
    worktree: str | pathlib.Path | None = None,
    ctx: int = 4096,
    e_slots: int = 72,
    moe_tp: bool = False,
    stack_e: bool = True,
    fused_pool_io: bool | str | None = None,
    fused_pool_io_max_b: int | None = None,
    fused_pool_io_min_b_hi: int | None = None,
    dump_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Run AOT compilation + full LLO/Mosaic dump for one bucket in an isolated
    subprocess.
    """
    perf_repo_root = pathlib.Path(__file__).resolve().parents[3]
    wt_path = perf_repo_root if worktree is None else pathlib.Path(worktree).resolve()
    own_tmp = dump_dir is None
    eff_dump_dir = (
        pathlib.Path(tempfile.mkdtemp(prefix=f"mk_llo_{profile_name}_t{T}_b{B}_"))
        if own_tmp
        else pathlib.Path(dump_dir).resolve()
    )
    eff_dump_dir.mkdir(parents=True, exist_ok=True)
    out_json = eff_dump_dir / "bucket_compile_stats.json"

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["TPU_WORKER_HOSTNAMES"] = "localhost"
    py_paths = [str(wt_path / "src"), str(wt_path), str(perf_repo_root)]
    if DEFAULT_CLOUDTOP_LIBTPU_SITE.is_dir():
        py_paths.insert(0, str(DEFAULT_CLOUDTOP_LIBTPU_SITE))
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_paths)

    base_init_args = _strip_dump_flags(env.get("LIBTPU_INIT_ARGS", ""))
    dump_flags = f"--xla_mosaic_dump_to={eff_dump_dir} --xla_jf_dump_to={eff_dump_dir}"
    env["LIBTPU_INIT_ARGS"] = (
        f"{base_init_args} {dump_flags}".strip() if base_init_args else dump_flags
    )

    cmd = [
        sys.executable,
        "-m",
        "tests.megakernel.perf.compiler_stats",
        "--single-bucket",
        "--profile",
        profile_name,
        "--T",
        str(T),
        "--B",
        str(B),
        "--ctx",
        str(ctx),
        "--e-slots",
        str(e_slots),
        "--dump-dir",
        str(eff_dump_dir),
        "--out-json",
        str(out_json),
        "--skip-parse",
    ]
    if moe_tp:
        cmd.append("--moe-tp")
    if stack_e:
        cmd.append("--stack-e")
    if fused_pool_io is not None:
        cmd.extend(["--fused-pool-io", str(fused_pool_io)])
    if fused_pool_io_max_b is not None:
        cmd.extend(["--fused-pool-io-max-b", str(fused_pool_io_max_b)])
    if fused_pool_io_min_b_hi is not None:
        cmd.extend(["--fused-pool-io-min-b-hi", str(fused_pool_io_min_b_hi)])

    import fcntl

    lock_path = pathlib.Path(tempfile.gettempdir()) / "mk_aot_libtpu_compile.lock"
    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(wt_path),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    if proc.returncode != 0 or not out_json.is_file():
        raise RuntimeError(
            f"AOT compile failed for profile={profile_name} T={T} B={B} in {wt_path}:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    res = json.loads(out_json.read_text(encoding="utf-8"))
    llo_summary = parse_llo_dump_dir(eff_dump_dir)
    primary_bundles = 0
    primary_non_empty = 0
    primary_spills = 0
    for kname, kinfo in llo_summary.get("subkernels", {}).items():
        if "megakernel" in kname:
            primary_bundles += int(kinfo.get("total_bundles", 0))
            primary_non_empty += int(kinfo.get("non_empty_bundles", 0))
            primary_spills += int(kinfo.get("spill_ops", 0))
    res["llo"] = llo_summary
    res["primary_total_bundles"] = primary_bundles
    res["primary_non_empty_bundles"] = primary_non_empty
    res["primary_spill_ops"] = primary_spills
    out_json.write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
    if own_tmp:
        # Clean up raw multi-MB dump files if caller did not request keeping dump_dir
        for f in eff_dump_dir.iterdir():
            f.unlink(missing_ok=True)
        eff_dump_dir.rmdir()
    else:
        res["dump_dir"] = str(eff_dump_dir)
    return res


def diff_compiled_buckets(
    base_res: dict[str, Any],
    head_res: dict[str, Any],
    *,
    diff_out_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Compare two compiled bucket results (toolchain check + full LLO/Mosaic diff per
    subkernel).
    """
    tc_ok, tc_reason = check_toolchain_match(
        base_res.get("toolchain", {}), head_res.get("toolchain", {})
    )
    if not tc_ok:
        return {
            "status": "INVALID_TOOLCHAIN",
            "reason": tc_reason,
            "exact_match": False,
        }

    base_llo = base_res.get("llo", {})
    head_llo = head_res.get("llo", {})
    full_match = (
        base_llo.get("full_llo_mosaic_sha256") == head_llo.get("full_llo_mosaic_sha256")
        and base_res.get("vmem_bytes") == head_res.get("vmem_bytes")
        and base_res.get("smem_bytes") == head_res.get("smem_bytes")
    )

    base_sk = base_llo.get("subkernels", {})
    head_sk = head_llo.get("subkernels", {})
    all_k = sorted(set(base_sk.keys()) | set(head_sk.keys()))

    subkernel_diffs: dict[str, Any] = {}
    for k in all_k:
        bk = base_sk.get(k, {})
        hk = head_sk.get(k, {})
        k_match = bk.get("full_llo_sha256") == hk.get("full_llo_sha256")
        b_bund = int(bk.get("total_bundles", 0))
        h_bund = int(hk.get("total_bundles", 0))
        b_ne = int(bk.get("non_empty_bundles", 0))
        h_ne = int(hk.get("non_empty_bundles", 0))

        changed_passes = []
        b_passes = bk.get("pass_hashes", {})
        h_passes = hk.get("pass_hashes", {})
        for p_stage in sorted(set(b_passes.keys()) | set(h_passes.keys())):
            if b_passes.get(p_stage) != h_passes.get(p_stage):
                changed_passes.append(p_stage)

        subkernel_diffs[k] = {
            "full_llo_match": k_match,
            "final_bundles_match": bk.get("final_bundles_sha256")
            == hk.get("final_bundles_sha256"),
            "base_total_bundles": b_bund,
            "head_total_bundles": h_bund,
            "delta_total_bundles": h_bund - b_bund,
            "base_non_empty_bundles": b_ne,
            "head_non_empty_bundles": h_ne,
            "delta_non_empty_bundles": h_ne - b_ne,
            "base_spill_ops": int(bk.get("spill_ops", 0)),
            "head_spill_ops": int(hk.get("spill_ops", 0)),
            "changed_passes_count": len(changed_passes),
            "first_changed_pass": changed_passes[0] if changed_passes else None,
        }

    # If dump directories exist and there is a diff, write unified diff of final_bundles
    diff_files: list[str] = []
    if (
        not full_match
        and diff_out_dir is not None
        and base_res.get("dump_dir")
        and head_res.get("dump_dir")
    ):
        out_d = pathlib.Path(diff_out_dir)
        out_d.mkdir(parents=True, exist_ok=True)
        b_dir = pathlib.Path(base_res["dump_dir"])
        h_dir = pathlib.Path(head_res["dump_dir"])
        for k, sdiff in subkernel_diffs.items():
            if sdiff["full_llo_match"] or k == "TLP":
                continue
            b_fb = sorted(b_dir.glob(f"*{k}*-final_bundles.txt"))
            h_fb = sorted(h_dir.glob(f"*{k}*-final_bundles.txt"))
            if b_fb and h_fb:
                b_lines = normalize_llo_text(
                    b_fb[0].read_text(encoding="utf-8", errors="replace")
                ).splitlines(keepends=True)
                h_lines = normalize_llo_text(
                    h_fb[0].read_text(encoding="utf-8", errors="replace")
                ).splitlines(keepends=True)
                udiff = list(
                    difflib.unified_diff(
                        b_lines,
                        h_lines,
                        fromfile=f"base/{k}/final_bundles.txt",
                        tofile=f"head/{k}/final_bundles.txt",
                        n=3,
                    )
                )
                if udiff:
                    safe_k = re.sub(r"[^a-zA-Z0-9_.-]+", "_", k)
                    df_name = (
                        f"diff_{base_res['profile']}_t{base_res['T']}"
                        f"_b{base_res['B']}_{safe_k}.diff"
                    )
                    df_path = out_d / df_name
                    df_path.write_text("".join(udiff[:5000]), encoding="utf-8")
                    diff_files.append(str(df_path))

    return {
        "status": "EQUAL" if full_match else "DIFF",
        "exact_match": full_match,
        "mosaic_match": base_llo.get("mosaic_sha256") == head_llo.get("mosaic_sha256"),
        "hlo_match": base_res.get("hlo_sha256") == head_res.get("hlo_sha256"),
        "base_vmem_bytes": base_res.get("vmem_bytes"),
        "head_vmem_bytes": head_res.get("vmem_bytes"),
        "delta_vmem_bytes": int(head_res.get("vmem_bytes", 0))
        - int(base_res.get("vmem_bytes", 0)),
        "base_smem_bytes": base_res.get("smem_bytes"),
        "head_smem_bytes": head_res.get("smem_bytes"),
        "delta_smem_bytes": int(head_res.get("smem_bytes", 0))
        - int(base_res.get("smem_bytes", 0)),
        "base_primary_bundles": base_res.get("primary_total_bundles", 0),
        "head_primary_bundles": head_res.get("primary_total_bundles", 0),
        "delta_primary_bundles": int(head_res.get("primary_total_bundles", 0))
        - int(base_res.get("primary_total_bundles", 0)),
        "subkernels": subkernel_diffs,
        "diff_files": diff_files,
    }


def parse_bucket_spec_list(buckets_str: str) -> list[tuple[int, int]]:
    """Parse `'1:8,1:16,4:24,4:32'` into `[(1, 8), (1, 16), (4, 24), (4, 32)]`."""
    out: list[tuple[int, int]] = []
    for item in buckets_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            t_s, b_s = item.split(":", 1)
            out.append((int(t_s.strip()), int(b_s.strip())))
        else:
            out.append((1, int(item)))
    return out


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AOT TPU7x compiler & LLO stats helper."
    )
    parser.add_argument("--single-bucket", action="store_true")
    parser.add_argument("--profile", default="qwen35-candidate30-fused")
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--e-slots", type=int, default=72)
    parser.add_argument("--moe-tp", action="store_true")
    parser.add_argument("--stack-e", action="store_true")
    parser.add_argument("--fused-pool-io", default=None)
    parser.add_argument("--fused-pool-io-max-b", type=int, default=None)
    parser.add_argument("--fused-pool-io-min-b-hi", type=int, default=None)
    parser.add_argument("--dump-dir", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--skip-parse", action="store_true")
    args = parser.parse_args()

    if args.single_bucket:
        res = compile_bucket_in_process(
            args.profile,
            args.T,
            args.B,
            ctx=args.ctx,
            e_slots=args.e_slots,
            moe_tp=args.moe_tp,
            stack_e=args.stack_e,
            fused_pool_io=args.fused_pool_io,
            fused_pool_io_max_b=args.fused_pool_io_max_b,
            fused_pool_io_min_b_hi=args.fused_pool_io_min_b_hi,
            dump_dir=args.dump_dir,
            skip_parse=args.skip_parse,
        )
        if args.out_json:
            pathlib.Path(args.out_json).write_text(
                json.dumps(res, indent=2, sort_keys=True), encoding="utf-8"
            )
        else:
            print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    _cli_main()
