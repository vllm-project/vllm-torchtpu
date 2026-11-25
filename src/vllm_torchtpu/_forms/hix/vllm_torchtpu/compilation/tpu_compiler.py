# SPDX-License-Identifier: Apache-2.0
"""TPU compiler adaptor for vLLM's VLLM_COMPILE mode.

This module implements the CompilerInterface for TPU, enabling
PiecewiseBackend to compile FX graphs via TorchTPU's FX->MLIR->PjRt
pipeline. Each shape bucket gets its own compiled executable.
"""

import contextlib
import copy
import dataclasses
import glob
import hashlib
import importlib.util
import json
import os
import pickle
import re
import shutil
import threading
import time
from collections.abc import Callable
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch
import torch._guards
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch_tpu._internal.compile._backend import TpuBackend
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig
from vllm.config.utils import Range

from vllm_torchtpu import envs, utils
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class TpuCompilationHandle:
    """Metadata handle saved to vLLM's compilation cache directory.

    `entry` is torch's bundled AOTAutograd cache entry: the aot_autograd
    wrapper metadata and the compiled PjRt executable as a single serializable
    object. Keeping them together is the point -- a wrapper rebuilt from a
    fresh trace and an executable fetched separately can disagree, and the
    disagreement is only observable once the program runs.

    `signature` pins the graph the entry was built from, so a replay against a
    structurally different graph is refused instead of silently executing.

    `fingerprint` pins the whole graph, body included (see
    `_graph_fingerprint`). Another rank's artifact is only replayed if it
    matches, since vLLM's key names a (bucket, subgraph) and nothing else.

    Defaults are deliberately "unusable": an older pickle restored into this
    class keeps its own __dict__ and picks up these class defaults for the
    fields it lacks, so it fails the version check rather than being replayed.
    """

    key: str
    entry: Any | None = None
    signature: tuple | None = None
    was_wrapped: bool = False
    fingerprint: str | None = None


_tpu_backend = TpuBackend()

# How long a rank waits for the rank of its host that compiles a bucket, and
# how often it looks. The wait is bounded only for a peer that never saves the
# artifact (its serialize() failed); the rank then compiles the bucket itself.
PEER_WAIT_S = 3600.0
_PEER_POLL_S = 5.0
# get_attr tensors up to this size are hashed by value; bigger ones, which are
# weights rather than constants, by dtype and shape.
_FINGERPRINT_MAX_NUMEL = 1 << 16

_peer_wait = threading.local()

# Op-name fragments whose lowering depends on the calling rank although the
# traced graph does not (e.g. a JAX callable over the TP mesh whose program is
# specialised to the rank's mesh position). A graph calling one is
# fingerprinted with the rank, so no other rank replays its artifact.
# vLLM's sequence_parallel_chunk slices this rank's rows (tp_rank * chunk)
# inside the custom op, so prefill sequence parallelism is one from the start.
_RANK_DEPENDENT_OPS: set[str] = {"sequence_parallel_chunk"}


def mark_rank_dependent(op_fragment: str) -> None:
    """Never replay a graph whose code mentions `op_fragment` across ranks."""
    _RANK_DEPENDENT_OPS.add(op_fragment)


def _rank_salt(code: str) -> str:
    if not any(fragment in code for fragment in _RANK_DEPENDENT_OPS):
        return ""
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    return f"\nrank-dependent:{rank}"


def graph_is_rank_dependent(graph: fx.GraphModule) -> bool:
    """Whether `graph` (or a submodule) calls a rank-dependent op: its artifacts are salted
    with the rank, so no peer ever replays them and waiting for a peer's is pointless."""
    try:
        return any(
            _rank_salt(module.code)
            for _, module in graph.named_modules()
            if isinstance(module, fx.GraphModule)
        )
    except Exception:
        return False


def agree_all_ranks(ok: bool) -> bool:
    """True only if every rank passes `ok` (a MIN all-reduce on the world CPU group).

    Lets the ranks fail a rank-dependent compile together: a rank that failed alone would leave
    its peers running an SPMD program it never joins, or waiting on an artifact it never saves
    (w04 j-05391f11: one rank's VMEM OOM left the other seven idle ~62 min).
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return ok
    group = None
    try:
        from vllm.distributed.parallel_state import get_world_group

        group = get_world_group().cpu_group
    except Exception:
        pass
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


@contextlib.contextmanager
def wait_for_peer_artifact(timeout_s: float):
    """Within this block, `compile` waits up to `timeout_s` for another rank
    of this host to save the bucket, and replays it, before compiling it.

    Outside it, `compile` still replays an artifact a peer has already saved,
    but does not wait for one.
    """
    _peer_wait.timeout_s = timeout_s
    try:
        yield
    finally:
        _peer_wait.timeout_s = 0.0


def _graph_fingerprint(graph: fx.GraphModule) -> str | None:
    """Hash of the code of `graph` and its submodules and of their constants.

    Two ranks with the same fingerprint traced the same program, so either can
    replay the other's artifact. A rank-dependent value baked into the graph,
    e.g. a Python int from `rank_in_group`, changes the code and so the hash.
    None if the graph cannot be hashed; its artifact is then only replayed by
    the rank that saved it.
    """
    try:
        h = hashlib.sha256()
        for name, module in graph.named_modules():
            if not isinstance(module, fx.GraphModule):
                continue
            h.update(f"{name}\n{module.code}{_rank_salt(module.code)}".encode())
            for node in module.graph.nodes:
                if node.op != "get_attr":
                    continue
                value = module
                for atom in str(node.target).split("."):
                    value = getattr(value, atom)
                h.update(str(node.target).encode())
                if not isinstance(value, torch.Tensor):
                    h.update(type(value).__qualname__.encode())
                    continue
                h.update(f"{value.dtype}{tuple(value.shape)}".encode())
                if value.numel() <= _FINGERPRINT_MAX_NUMEL:
                    flat = value.detach().to("cpu").contiguous().reshape(-1)
                    h.update(flat.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    except Exception as e:
        logger.warning("[TpuCompilerAdaptor] Could not fingerprint graph: %s", e)
        return None


def _deserialize_entry(entry: Any) -> Callable[..., Any]:
    """Rebuild a runnable from a bundled AOTAutograd entry.

    Restores the aot_autograd wrapper *and* the executable from the one entry,
    and returns a varargs callable over the graph's placeholders -- the same
    calling convention TpuBackend hands back, so `compile` and `load` produce
    interchangeable runnables.

    Module-level, and paired with `_tpu_backend`: a test that swaps the backend
    for a fake has to swap this too, since the fake's "entry" never came from
    torch and cannot go back through it.
    """
    from torch._functorch._aot_autograd.aot_autograd_result import (
        deserialize_bundled_cache_entry,
    )

    return deserialize_bundled_cache_entry(entry)


_TPU_COMPILE_ENV_IGNORED = {
    # Warm-up sweep cap: which spec-verify programs are precompiled, not
    # their contents.
    "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS",
    # Warm-up served-only: which spec-verify programs are precompiled, not
    # their contents.
    "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY",
    # Imported megakernel tree (MM 37b960ea9), as MM ignores them there: a host-side operand dump switch,
    # the step-mix log interval and the dump directory; none changes a compiled program.
    "VLLM_TPU_MK_OPERAND_DUMP",
    "VLLM_TPU_MK_STEP_MIX_INTERVAL_S",
    "CDK_OUTPUT_DIR",
    # q38 L5 CTXCENSUS: host-side context census lines of the Qwen3.8 driver (log-only).
    "VLLM_TPU_Q38_CTX_CENSUS",
    # VPS diagnostics (runner/vps_wiring.py): host-side choices of whether to run
    # the existing stock verify as a reference and how often to log; the served
    # programs are unchanged.
    "VLLM_TPU_VPS_CHECK",
    "VLLM_TPU_VPS_CHECK_FAULT",
    "VLLM_TPU_VPS_STATS_EVERY",
    # q38 L4 SEEDFIX: admission of seeded requests and an eager overlay of their
    # rows on the shared uniform draws (runner/seedfix.py); no traced code reads it.
    "VLLM_TPU_SEEDED_SAMPLING",
    # q38 L4 D83's read-offset fold check: whether the first N folded chunks also
    # run the separate read-offset update on a copy (eager); the served programs
    # are unchanged.
    "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK",
    "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT",
    # q38 L4 VFY-ONE: its CHECK count and its window switch pick between programs
    # that are precompiled whenever VLLM_TPU_VPS_VFY_ONE (hashed) is on.
    "VLLM_TPU_VPS_VFY_ONE_CHECK",
    "VLLM_TPU_VPS_VFY_ONE_AB",
    # q38 L4 KEYLITE: how VFY-ONE's warm-key test is computed (host-only; same verdicts).
    "VLLM_TPU_VPS_VFY_ONE_KEYLITE",
    # q38 STALL-HUNT: VFY-ONE's warm-up passes the seed serving's argument aliasing (warm-up
    # only; serving traces that same aliased graph with or without it).
    "VLLM_TPU_VFY_WARM_ALIAS",
    # DRAFT_TOP_TOKENS' inline check: whether to also run the full-logits draft
    # argmax (its own program) on the first N proposals; served programs unchanged.
    "VLLM_TPU_DRAFT_TOP_TOKENS_CHECK",
    # FAST_METADATA_BUILD: the host-side metadata walk builds the same
    # AttentionMetadata objects inline instead of one builder call per group
    # (bitwise-equal fields, tests/runner/test_fast_metadata_build_port.py);
    # nothing traced reads it, so the compiled programs do not change.
    "VLLM_TPU_FAST_METADATA_BUILD",
    # Startup, diagnostics, and orchestration only; these do not affect the
    # compiled model or custom-kernel lowering.
    # VLLM_TPU_CPU_PROBE_S: the CPUPROBE daemon thread (worker/cpu_pinning.py) logs each worker's
    # thread placement, migrations and context switches every N seconds; it pins nothing with
    # VLLM_TPU_CPU_PIN off and no traced code reads it (q38 L1, STACK-D v3).
    "VLLM_TPU_CPU_PROBE_S",
    "DP_SCHED_ENABLED",
    "VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS",
    # P-DENSE trace-time records (debug; pdense_engagement), host-only.
    "VLLM_TPU_PDENSE_TRACE_RECORDS",
    "PYTHON_TRACER_LEVEL",
    "RAY_USAGE_STATS_ENABLED",
    "TPU_NAME",
    "USE_PHASED_PROFILER",
    # Raiden KV-transfer orchestration/identity only; consumed at serving time
    # by the KV connector and never affect the compiled graph.
    # TPU_RAIDEN_ADVERTISE_HOST is the pod's own IP (recipes set it from
    # status.podIP; read only by tpu_connector's reshard-store address map), so
    # hashing it gave every pod, and every restart of a pod, its own vLLM
    # torch.compile cache directory: a restored compile cache was never reused.
    "TPU_RAIDEN_ADVERTISE_HOST",
    "TPU_RAIDEN_CONTROLLER_ADDRESS",
    "TPU_RAIDEN_ENGINE_ID",
    "TPU_RAIDEN_JOB_NAME",
    "TPU_RAIDEN_TRANSFER_PARALLELISM",
    "TPU_WORKER_ID",
    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE",
    "VLLM_XLA_CHECK_RECOMPILATION",
    "TPU_PARALLEL_PRECOMPILE",
    "TPU_PRECOMPILE_BIGGEST_FIRST",
    "TPU_PRECOMPILE_PEER_REPLAY",
    # KV-transfer wiring: ports, socket paths, staging pool, pool sizes,
    # timeouts. Same reasoning as the Raiden group above. Every registry
    # entry not named here joins the cache key, so omitting these would
    # recompile the world whenever an operator retuned a port.
    "TPU_IPC_SOCKET_DIR",
    "TPU_KV_CHANNEL_EXECUTOR_MAX_WORKERS",
    "TPU_KV_COORD_EXECUTOR_MAX_WORKERS",
    "TPU_KV_LATENCY_LOG_INTERVAL",
    "TPU_KV_PIN_SHM",
    "TPU_KV_SHM_POOL_GB",
    "TPU_KV_STAGE_WAITER_POOL_SIZE",
    "TPU_KV_STAGE_WAIT_TIMEOUT_SECS",
    "TPU_KV_TRANSFER_CHANNEL_NUMBER",
    "TPU_KV_TRANSFER_NAMESPACE",
    "TPU_KV_TRANSFER_PORT",
    # Compiles the same artifacts, just eagerly at startup.
    "TPU_KV_WARMUP_ENABLED",
    "TPU_NODE_ID",
    "TPU_P2P_WAIT_PULL_TIMEOUT",
    "TPU_RAIDEN_INLINE_LOAD",
    "TPU_RAIDEN_POOL_STAGING_LEASES",
    # Host staging arena size on a Qwen PD producer (host memory only).
    "QWEN4_EXP_PD_STAGING_LEASES",
    # When a staged Qwen PD decode runs its (unchanged) repack program.
    "QWEN4_EXP_PD_SYNC_REPACK",
    "QWEN4_EXP_PD_SYNC_REPACK_AB_S",
    # When a staged decode's released staging pages may be reused (host-only).
    "QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS",
    # A staged decode's per-load landing log lines (host clocks only).
    "QWEN4_EXP_PD_LANDING_LOG",
    # How a staged decode chunks its repack: adds shape-keyed repack programs
    # (compiled at startup by warm_repack_v2), changes none of the model's.
    "QWEN4_EXP_PD_REPACK_V2",
    # q38 L2 D74: one descriptor-driven landing program (its own HLO; warmed at startup).
    "QWEN4_EXP_PD_REPACK_L2",
    # When a staged decode plans its repack (host-only numpy, same programs).
    "QWEN4_EXP_PD_REPACK_PLAN_AT_LAND",
    # Whether a TP1 pipeline producer ships its MTP draft's KV and the staged
    # decode lands it (q38 F2): admission and host-side staging indices only;
    # the producer's draft itself is keyed by its speculative config.
    "QWEN4_EXP_PD_DRAFT_KV",
    # How much of each load's draft KV a staged decode zero-fills (q38 F2-3's
    # forced-gap arm): host-side repack indices only.
    "QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP",
    # Whether an F2 producer holds MTP draft blocks only above each request's
    # prefix hit (q38 F2-HOLD): scheduler block tables and send spans only.
    "QWEN4_EXP_PD_DRAFT_HOLD",
    # Where a staged decode's compressed keys land (once per host, then a
    # landing all-gather): adds shape-keyed landing programs, changes none.
    "QWEN4_EXP_PD_SHARED_COMPRESSED",
    # Host-side Stage-3 admission collectives (same decisions; q38 D-MK).
    "QWEN4_EXP_PD_BATCHED_ADMISSION",
    "QWEN4_EXP_PD_BATCHED_ADMISSION_AB_PASSES",
    "QWEN4_EXP_PD_BATCHED_ADMISSION_CHECK",
    # Per-step host phase logging (runner/host_phases.py).
    "VLLM_TPU_HOST_PHASES",
    # q38 L4 D98 / X-H14-D98R direct cache-entry dispatch, guard memo, GLOBAL_STATE guard
    # filter and engagement log: host-side dispatch of the same compiled programs
    # (compilation/fast_compiled_call.py). Unlike AOTSKIP it acts on every dispatch of a
    # cached helper, warm or cold, so sharing the compile key does not hide it.
    "VLLM_TPU_FAST_COMPILED_CALL",
    "VLLM_TPU_FAST_CALL_GUARD_MEMO",
    "VLLM_TPU_FAST_CALL_DROP_GLOBAL_STATE",
    "VLLM_TPU_FAST_CALL_LOG_EVERY",
    # q38 L4 EAGER-FIX: full-length InputBatch sampling copies, the eager-shape log and
    # the VPS draw warm-up are host-side; the compiled programs are unchanged
    # (VLLM_TPU_SPEC_TAIL_REJ_PAD changes the spec tail program and is hashed).
    "VLLM_TPU_SAMPLING_COPY_FULL",
    "VLLM_TPU_EAGER_SHAPE_LOG",
    "VLLM_TPU_VPS_WARM_DRAWS",
    # q38 L4 VPSHOST host levers (l6_ab classes): host-side order / copy changes only; the
    # compiled programs and the inputs they receive are unchanged.
    "VLLM_TPU_L4_SAMP_FB_ORDER",
    "VLLM_TPU_L4_SAMP_SNAP",
    "VLLM_TPU_L4_SAMP_SNAP_CHECK",
    "VLLM_TPU_L4_VPS_STATS_ASYNC",
    # q38 L4 DRAFT-ASM: the CHECK step count and the l6_ab window switch pick between
    # precompiled programs (VLLM_TPU_DRAFT_ASM_FOLD adds the programs and is hashed).
    "VLLM_TPU_DRAFT_ASM_CHECK_STEPS",
    "VLLM_TPU_DRAFT_ASM_AB",
    # q38 L4 DRAFT1-B: the stage window switches pick between precompiled programs
    # (VLLM_TPU_DRAFT_ASM_CARRY_LOOP / _ONE add the programs and are hashed).
    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB",
    "VLLM_TPU_DRAFT_ASM_ONE_AB",
    # q38 L4 T1-K1FUSE: the window switch picks between precompiled programs
    # (VLLM_TPU_K1FUSE adds the programs and is hashed).
    "VLLM_TPU_K1FUSE_AB",
    "VLLM_TPU_K1FUSE_CHECK",
    # E-d: which served VPS verify programs are precompiled, not their contents.
    "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_FULL_DRAFT_MAX_SEQS",
    # P-DENSE engagement counters (host-side, log-only; models/vllm/qwen4_exp/pdense_engagement.py).
    "VLLM_TPU_PDENSE_COUNT",
    # Engine-core phase logging (core/engine_phases.py).
    "VLLM_TPU_ENGINE_PHASES",
    # q38 L1-C48 DECODE-TRIM: which decode requests the engine core schedules per step; the
    # compiled programs are the same (core/decode_trim.py).
    "VLLM_TPU_DECODE_TRIM",
    "VLLM_TPU_DECODE_TRIM_LOG",
    # q38 L1-DSTASH: re-runs the already-warmed _substitute_placeholder_token program at a
    # warmed shape with a held-back request's stashed source (runner/draft_stash.py); no new program.
    "VLLM_TPU_DECODE_TRIM_DRAFT_STASH",
    # q38 S3A-ENG T4: the per-rank DSTASH state line (log-only, read once per runner).
    "VLLM_TPU_DSTASH_STATE_LOG",
    # Container memory telemetry, host-side and log-only (core/memstat.py).
    "VLLM_TPU_MEMSTAT_INTERVAL_S",
    # Decode host tier: scheduler/connector-side load split and host DMA; the
    # compiled programs are the same (distributed/kv_transfer/tpu_multi_connector.py).
    "VLLM_TPU_DECODE_HOST_TIER",
    "VLLM_TPU_DECODE_HOST_TIER_VERIFY",
    # What the engine core ships to the workers (core/pd_prompt_tail.py); the
    # workers' programs are the same.
    "VLLM_TPU_PD_PROMPT_TAIL",
    "VLLM_TPU_HOST_PHASES_SLOW_MS",
    # q38 L4 host sub-mark instrument: HOSTSUBS log lines (runner/host_phases.py).
    "VLLM_TPU_HOST_PHASES_SUBS",
    # q38 L6-INSTR: GCCENSUS log lines (runner/host_phases.py).
    "VLLM_TPU_HOST_PHASES_GC",
    # Engine-core scheduler bookkeeping only (core/kv_twin_groups.py).
    "VLLM_TPU_KV_TWIN_GROUPS",
    # Interpreter knobs only (core/host_knobs.py): thread switching and cyclic GC.
    "VLLM_TPU_HOST_SWITCH_US",
    "VLLM_TPU_HOST_GC",
    "VLLM_TPU_HOST_GC_AFTER",
    "VLLM_TPU_HOST_GC_GEN0",
    "VLLM_TPU_HOST_AB_PERIOD_MS",
    # q38 D1-norm L-GDNCOPY packed seed-copy uploads and its check (the index
    # values and every program are unchanged), and the D1-norm A/B windows.
    "VLLM_TPU_GDN_SEED_COPY_PACKED",
    "VLLM_TPU_GDN_SEED_COPY_PACKED_CHECK",
    "VLLM_TPU_D1N_AB_PERIOD_MS",
    # q38 L6-HOST step-time host levers (core/l6_ab.py, core/l6_envcache.py): read per
    # worker step, never at trace time; every program is the same.
    "VLLM_TPU_L6_AB_PERIOD_MS",
    "VLLM_TPU_L6_AB_CLASSES",
    "VLLM_TPU_L6_AB_STEPS",
    "VLLM_TPU_L6_AB_CONFIRM",
    "VLLM_TPU_L6_AB_DIR",
    "VLLM_TPU_L6_CHECK_STEPS",
    "VLLM_TPU_L6_ENVS_CACHE",
    "VLLM_TPU_L6_ENVS_CACHE_CHECK",
    "VLLM_TPU_L6_ENVS_CENSUS",
    "VLLM_TPU_L6_FAST_FWD_CTX",
    "VLLM_TPU_L6_GC_FREEZE",
    "VLLM_TPU_L6_SWITCH",
    "VLLM_TPU_L6_SWITCH_US",
    "VLLM_TPU_L6_MAIN_PIN",
    "VLLM_TPU_L6_MAIN_PIN_CPU",
    # q38 L6 MAINPIN-2 / NUMA bind / SYNC-GC (core/l6_jitter.py, worker/numa_bind.py): thread placement
    # and cyclic-GC timing on the host only.
    "VLLM_TPU_L6_MAIN_PIN2",
    "VLLM_TPU_L6_NUMA_BIND",
    "VLLM_TPU_L6_SYNC_GC",
    "VLLM_TPU_L6_SYNC_GC_EVERY",
    "VLLM_TPU_L6_SYNC_GC_VALVE",
    "VLLM_TPU_L6_DEFER_PREV",
    "VLLM_TPU_L6_STEADY_STEP",
    "VLLM_TPU_L6_STEADY_STEP_CHECK",
    # Lever carry (runner/decode_carry.py): host-only switch; the carry program it adds is
    # its own small compiled function (compiled at warmup when the flag is set), and no
    # existing program changes.
    "VLLM_TPU_L6_DECODE_CARRY",
    "VLLM_TPU_L6_DECODE_CARRY_CHECK",
    # q38 L6-GATE batch-size gates (core/l6_ab.py): which steps run CARRY / STEADY; host-only.
    "VLLM_TPU_L6_CARRY_MIN_REQS",
    "VLLM_TPU_L6_STEADY_MIN_REQS",
    # Lever adcarry (runner/adapt_carry.py): host-only switch between precompiled programs; the two merged
    # programs it adds are their own small compiled functions (compiled at warm-up when the flag is set), and
    # no existing program changes.
    "VLLM_TPU_L6_ADAPT_CARRY",
    "VLLM_TPU_L6_ADAPT_CARRY_CHECK",
    # q38 L6 lever engalloc (core/l6_engine_alloc.py): engine-core KV slot bookkeeping only.
    "VLLM_TPU_L6_ENGINE_ALLOC_FAST",
    "VLLM_TPU_L6_ENGINE_ALLOC_FAST_CHECK",
    # q38 L6 lever setupfree (runner/l6_setupfree.py): STEADY's metadata objects reused in place;
    # the same values, programs and uploads.
    "VLLM_TPU_L6_SETUP_FREE",
    "VLLM_TPU_L6_SETUP_FREE_CHECK",
    # q38 L6-H2 levers: host-only (no program, input or call ABI changes). aotskip writes the
    # existing AOT artifacts' _guard_check_enabled attribute per window; nothing recompiles.
    "VLLM_TPU_L6_FENCE_SKIP",
    "VLLM_TPU_L6_TOKEN_LISTS",
    "VLLM_TPU_L6_TOKEN_LISTS_CHECK",
    "VLLM_TPU_L6_AOT_SKIP",
    "VLLM_TPU_L6_AOT_SKIP_CHECK",
    # q38 L6 lever gcadmit (core/l6_gcadmit.py): gc.freeze() after admissions, host-only.
    "VLLM_TPU_L6_GC_ADMIT_FREEZE",
    "VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP",
    "VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK",
    # q38 L6 gen-2 heap type census (core/l6_gccensus.py): log lines only.
    "VLLM_TPU_L6_GC_CENSUS_TYPES",
    "VLLM_TPU_L6_GC_CENSUS_MS",
    "VLLM_TPU_L6_GC_CENSUS_MAX",
    # q38 L6 end-token canary (core/l6_eos_canary.py): host-side counts in the L6AB line only.
    "VLLM_TPU_L6_EOS_CANARY",
    # q38 L6 lever ptrims (runner/l6_ptrims.py): host-side input-prep trims; same host arrays,
    # same programs.
    "VLLM_TPU_L6_PREP_TRIMS",
    "VLLM_TPU_L6_PREP_TRIMS_CHECK",
    # q38 L6 lever admitinc (layers/core/attention_metadata.py): host-side block-table pack and
    # row writes; the staged bytes, and so every program, are the same.
    "VLLM_TPU_L6_ADMIT_INC",
    "VLLM_TPU_L6_ADMIT_INC_CHECK",
    # q38 L6-MK levers (runner/l6_mk.py, core/l6_fuse_rpc.py): host-side call paths only; every
    # compiled program and its arguments are the same with them on or off.
    "VLLM_TPU_L6_MK_CTX",
    "VLLM_TPU_L6_MK_CTX_CHECK",
    "VLLM_TPU_L6_MK_THIN_TT",
    "VLLM_TPU_L6_MK_THIN_TT_CHECK",
    "VLLM_TPU_L6_MK_DIRECT",
    "VLLM_TPU_L6_MK_DIRECT_CHECK",
    "VLLM_TPU_L6_FUSE_RPC",
    "VLLM_TPU_L6_OUT_KVONLY",
    "VLLM_TPU_L6_OUT_KVONLY_CHECK",
    # q38 L6 levers clean (runner/l6_carry_lean.py) and pshort (runner/l6_prep_short.py): host-side
    # carry check / record and input-prep short path; same programs.
    "VLLM_TPU_L6_CARRY_LEAN",
    "VLLM_TPU_L6_CARRY_LEAN_CHECK",
    "VLLM_TPU_L6_PREP_SHORT",
    "VLLM_TPU_L6_PREP_SHORT_CHECK",
    # q38 L6 P-METER (runner/l6_pmeter.py): log lines and a host busy spin; no program changes.
    "VLLM_TPU_L6_PMETER",
    "VLLM_TPU_L6_PMETER_WINDOW",
    "VLLM_TPU_L6_INJECT",
    "VLLM_TPU_L6_INJECT_US",
    "VLLM_TPU_L6_INJECT_RANK",
    "VLLM_TPU_L6_INJECT_ALL",
    "VLLM_TPU_L6_INJECT_ALL_US",
    # q38 L6 NODEFER census (runner/l6_nodefer.py): an is_materializing probe and log counts.
    "VLLM_TPU_L6_NODEFER_CENSUS",
    "VLLM_TPU_L6_NODEFER_EVERY",
    # q38 L6-S3ASYNC (distributed/kv_transfer/stage3_async.py): where the Stage-3 load coordination
    # runs (model-runner thread or a coordinator thread) and its log lines / host sub-marks; the
    # transfer, the KV bytes and every compiled program are the same.
    "VLLM_TPU_S3_ASYNC_SUBMIT",
    "VLLM_TPU_S3_ASYNC_LAG",
    "VLLM_TPU_S3_ASYNC_JOIN_TIMEOUT_S",
    "VLLM_TPU_S3_ASYNC_STATS_EVERY",
    "VLLM_TPU_S3_SUBMARKS",
    # q38 L6-S3ASYNC S3-DEFER-AR: when the Stage-3 submit pass runs (and its admission collective is
    # issued); the transfer and every compiled program are the same.
    "VLLM_TPU_S3_DEFER_AR",
    # q38 L1-S3DGATE: which L6 windows run lever s3defer (the S3D load gate); host-only like the lever.
    "VLLM_TPU_S3_DEFER_GATE_MIN_REQS",
    # Incremental block-table staging (layers/core/attention_metadata.py): the
    # staged bytes, and so every program, are the same as the full pack.
    "VLLM_TPU_BLOCK_TABLE_STAGE_INC",
    "VLLM_TPU_BLOCK_TABLE_STAGE_INC_CHECK",
    # DEC-HC engagement counter: host-side log lines only.
    "VLLM_TPU_HC_DECODE_COUNT",
    # q38 L2 STACK-MARK: host-side per-bucket route counter (log-only).
    "VLLM_TPU_Q38_STACK_ROUTE_COUNT",
    # Per-step token-bucket counts (runner/step_buckets.py).
    "VLLM_TPU_LOG_STEP_BUCKETS",
    # ... and their host step periods (q38 L5 STEPBUCKETS2; log-only).
    "VLLM_TPU_LOG_STEP_BUCKETS_PERIOD",
    # q38 L5 SAB: routes whole-eligible steps between existing programs (host-side).
    "VLLM_TPU_SERVED_AB",
    # Runner stage timing off (runner/stage_timing.py): host-side events only.
    "VLLM_TPU_STAGE_TIMING_OFF",
    "TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT",
    "TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S",
    "TPU_RAIDEN_STAGE3_STATUS_PROBE_S",
    "TPU_RAIDEN_TEST_REGISTRATION_DELAY_S",
    "TPU_RAIDEN_TRANSFER_NUM_SLOTS",
    "TPU_SIDE_CHANNEL_PORT",
    # The HBM reserve that feeds block-count sizing comes from
    # kv_transfer_config, not from this variable.
    "TPU_USE_RAIDEN_CONNECTOR",
    # Pipeline chunk sizing: scheduling and the extra token buckets, which
    # are keyed by shape inside the same cache.
    "TPU_PP_DYNAMIC_CHUNKS",
    "TPU_PP_CHUNK_SLACK",
    # Engine-side timing of the pipeline pushes; the workers' programs are
    # the same either way.
    "TPU_PP_LAZY_PUSH",
    # The hand-off between pipeline stages and its depth: it runs outside
    # the compiled forwards, which take the same inputs either way.
    "TPU_PP_HANDOFF",
    "TPU_PP_RING_DEPTH",
    # Engine-side queue depth and the ring's host-side byte check: the
    # compiled forwards are the same either way.
    "TPU_PP_EXTRA_STEPS_IN_FLIGHT",
    "TPU_PP_RING_VERIFY",
    "TPU_PP_RING_VERIFY_MUTANT",
    # Startup-only host barrier before the hand-off warmup: the compiled
    # forwards are the same either way.
    "TPU_PP_WARMUP_BARRIER",
    # The same barrier before each prompt length of the MTP prefill warmup.
    "TPU_PP_MTP_WARMUP_BARRIER",
    # The chunk scheduler's spread (engine side): it changes which buckets
    # steps use, not the compiled programs.
    "TPU_PP_SPREAD_MAX_PREFIX",
    # The chunk scheduler's waiting order (engine side; q38 HIGHC-WARMFIRST,
    # port of benliangw d523c2ce0): it changes which requests a step carries,
    # not the compiled programs.
    "TPU_PP_WARM_FIRST",
    "TPU_PP_WARM_FIRST_AGING",
    "TPU_PP_WARM_FIRST_TIER",
    # Startup and host-side knobs: ports, an IPC secret, two raiden startup
    # checks, the raiden offload storage tier, offload retry and drain
    # limits, a weight-load sync cadence, a debug log switch, and the
    # local-rank offsets, which move a worker to a different chip without
    # changing the executable it runs.
    "DEBUG_TPU_LOCAL_RANK_OFFSET",
    "EXPERIMENTAL_RAIDEN_STORAGE_SKIP_PARALLELISM_CHECK",
    "RAIDEN_DISABLE_SINGLETON_WORKER",
    "RAIDEN_SHM_KEY",
    "RAIDEN_STORAGE_CAPACITY_GB",
    "RAIDEN_STORAGE_PATH",
    "RAIDEN_STORAGE_PROPERTIES",
    "RAIDEN_STORAGE_TYPE",
    # Compiles the same spec-decode artifacts, just eagerly at startup.
    "SPEC_WARMUP",
    "TORCH_TPU_BASE_PORT",
    "TORCH_TPU_MP_RENDEZVOUS_PORT",
    "TPU_LOCAL_RANK_OFFSET",
    "TPU_SHARDED_LOAD_SYNC_EVERY",
    "VLLM_TORCHTPU_IPC_KEY",
    "VLLM_TPU_DEBUG_PCP_LAYOUT",
    "VLLM_TPU_OFFLOAD_SAVE_RETRIES",
    "VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S",
    # Scheduler prefix-cache hit length only.
    "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP",
    # Scheduler prefix-cache block lifetime only.
    "VLLM_TPU_MAMBA_CHECKPOINT_RETENTION",
    "VLLM_TPU_MAMBA_DECODE_CHECKPOINT",
    # Scheduler prefix-cache eviction order only.
    "VLLM_TPU_PREFIX_DEMOTE_STALE_CHECKPOINT",
    "VLLM_TPU_PREFIX_TAIL_EVICT_FIRST",
    "VLLM_TPU_PREFIX_UNIT_ALLOW_F2_PRODUCER",
    "VLLM_TPU_PREFIX_REFRESH_WAITING",
    "VLLM_TPU_PREFIX_REFRESH_INTERVAL_S",
    "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
    # Only changes LIBTPU_INIT_ARGS, which is hashed on its own.
    "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES",
    # Pooling: the static pooler is a separate backend="tpu" helper outside the
    # backbone graph, and the KV cap reaches the hash through num_gpu_blocks.
    "VLLM_TPU_STATIC_POOLER",
    "VLLM_TPU_POOLING_KV_CACHE_TOKENS",
    # API-server DP routing only (core/dp_affinity.py): which engine serves a
    # request, never what an engine compiles.
    "VLLM_TPU_DP_AFFINITY",
    "VLLM_TPU_DP_AFFINITY_CHUNK",
    "VLLM_TPU_DP_AFFINITY_MAX_ENTRIES",
    "VLLM_TPU_DP_AFFINITY_SLACK",
    "VLLM_TPU_DP_AFFINITY_SLOT_GUARD",
    "VLLM_TPU_DP_AFFINITY_SHARED",
    "VLLM_TPU_DP_AFFINITY_SHARED_SLOTS",
    # q38 L4 VRNG-VPS path switch: routes between programs that are all warmed (host only).
    "VLLM_TPU_VPS_INGRAPH_RNG_AB",
    # q38 PORT-BL GDN (begin): VLLM_TPU_GDN_LEAN_GLUE3's engagement-line period (host-side, log-only).
    "VLLM_TPU_GDN_LEAN_GLUE3_COUNT",
    # q38 PORT-BL GDN (end)
}

_NATIVE_TPU_COMPILE_ENV_VARS = (
    "LIBTPU_INIT_ARGS",
    "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
    "TORCH_TPU_INTERNAL_XLA_OPTIONS",
    "XLA_FLAGS",
)

_RUNTIME_CACHE_KEY_PATHS = (
    # Pallas implementations execute behind torch custom-op boundaries, so
    # Dynamo's traced-file validation cannot see their source changes.
    "vllm_torchtpu/kernels",
    "vllm_torchtpu/layers/core",
    "vllm_torchtpu/layers/adapter/attention.py",
    "vllm_torchtpu/layers/adapter/cp_attention.py",
    "vllm_torchtpu/layers/adapter/custom_ops",
    "vllm_torchtpu/layers/adapter/fused_moe.py",
    # The fused EP MoE op and the helper that registers it. Both sit behind a
    # torch custom-op boundary Dynamo cannot see through, and the kernel they
    # export is chosen from their source, so an edit here has to invalidate
    # the compiled executables like any other kernel change.
    "vllm_torchtpu/layers/adapter/fused_moe_ep.py",
    # Same for the TP decode MoE op, whose body this module builds.
    "vllm_torchtpu/layers/adapter/moe_tp_decode.py",
    # Still listed: on this base `pp_shift.py` is a second caller, so the
    # module stays even though the fused EP op no longer needs it.
    "vllm_torchtpu/distributed/sharded_jax_op.py",
    # And the mesh they export with: `build_ep_mesh` decides the axis, the
    # shape and the device order (the DeviceAssignment) the kernel runs on.
    "vllm_torchtpu/distributed/ep_mesh.py",
    "vllm_torchtpu/distributed/pcp.py",
    # Same reason: dcp.py owns the DCP kernels' mesh shape and device order.
    "vllm_torchtpu/distributed/dcp.py",
    "vllm_torchtpu/distributed/pallas_shapes.py",
    "vllm_torchtpu/layers/adapter/linear_common.py",
    "vllm_torchtpu/layers/adapter/router_topk.py",
    "vllm_torchtpu/layers/adapter/quantization",
    "vllm_torchtpu/layers/adapter/vision_attention.py",
    "vllm_torchtpu/models/vllm/deepseek_v4/attention.py",
    # `get_pallas_op_mesh` decides whether the DSv4 Pallas ops are built with
    # the wrapper mesh or with none, like `build_ep_mesh` above.
    "vllm_torchtpu/models/vllm/vllm_model_wrapper_context.py",
    # K3 custom-op registration, sharding, and Attention Residual bridges.
    "vllm_torchtpu/models/vllm/kimi_k3/collective_ops.py",
    "vllm_torchtpu/models/vllm/kimi_k3/layers.py",
    # Model-owned pooled cache geometry changes graph argument extents.
    "vllm_torchtpu/models/vllm/qwen4_exp/pool_layout.py",
    # The staged-decode repack and landing-scatter ops (qwen4_pd_staged_repack,
    # qwen4_pd_staged_repack_v2, qwen4_pd_shared_scatter) are built here with
    # the Pallas jax-op wrapper, behind a custom-op boundary Dynamo cannot see
    # through. (Keep that wrapper's dotted name out of this file: the coverage
    # test treats any module that mentions it as a custom-op boundary.)
    "vllm_torchtpu/models/vllm/qwen4_exp/pd_staging.py",
    # q38 L1-G1 GDN-COMMIT1 (VLLM_TPU_MK_GDN_COMMIT1): the commit and full-flush ops (q38_g1_gdn_commit1 / commit2 / full)
    # are built here with the same Pallas jax-op wrapper, compiled with torch.compile(backend="tpu") behind a custom-op
    # boundary Dynamo cannot see through, so their source must move the runtime cache key (wrapper name kept out of
    # this file, as above).
    "vllm_torchtpu/megakernel_runtime/qwen38_gdn_commit.py",
)

# vLLM-Omni diffusion code, dropped from the hashed set above. Diffusion
# workers never compile through TpuCompilerAdaptor (OmniTpuPlatform does not
# support torch inductor and vLLM-Omni skips torch.compile), and nothing
# outside these paths imports their kernels, so they cannot change a cached
# executable; hashing them would only invalidate LLM compile caches.
_RUNTIME_CACHE_KEY_EXCLUDED_PATHS = (
    "vllm_torchtpu/kernels/diffusion_attention",
    "vllm_torchtpu/omni/attention.py",
    "vllm_torchtpu/omni/rope.py",
)


def _iter_runtime_cache_key_files(repo_root: Path) -> list[Path]:
    """Return runtime source files that affect TPU custom-op lowering."""
    files_by_relpath: dict[str, Path] = {}

    for rel_path in _RUNTIME_CACHE_KEY_PATHS:
        path = repo_root / rel_path
        if path.is_file():
            files_by_relpath[rel_path] = path
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*.py")):
                child_relpath = child.relative_to(repo_root).as_posix()
                files_by_relpath[child_relpath] = child
            continue
        logger.warning("[TpuCompilerAdaptor] Cache key source path missing: %s", path)

    excluded_files = set(_RUNTIME_CACHE_KEY_EXCLUDED_PATHS)
    excluded_dirs = tuple(
        f"{rel_path}/" for rel_path in _RUNTIME_CACHE_KEY_EXCLUDED_PATHS
    )
    for relpath in [
        key
        for key in files_by_relpath
        if key in excluded_files or key.startswith(excluded_dirs)
    ]:
        del files_by_relpath[relpath]

    if envs.TPU_KERNEL_ITER_MODE:
        for relpath in _reloadable_kernel_relpaths(repo_root):
            files_by_relpath.pop(relpath, None)

    return [files_by_relpath[key] for key in sorted(files_by_relpath)]


def _reloadable_kernel_relpaths(repo_root: Path) -> list[str]:
    """Repo-relative paths of the hot-reloadable kernel modules.

    In kernel-iteration mode these execute eagerly outside the compiled
    pieces (splitting_ops in tpu_platform.py) and are re-lowered from source
    per process / per reload, so their bytes no longer affect the cached
    executables and are dropped from the cache key. Everything else in
    _RUNTIME_CACHE_KEY_PATHS stays hashed.
    """
    from vllm_torchtpu.compilation import kernel_reload

    relpaths = []
    for mod_name in kernel_reload.default_reload_modules():
        spec = importlib.util.find_spec(mod_name)
        origin = spec.origin if spec else None
        if not origin:
            logger.warning(
                "[TpuCompilerAdaptor] Cannot resolve reload module %s; its "
                "file stays in the cache key.",
                mod_name,
            )
            continue
        try:
            relpaths.append(Path(origin).resolve().relative_to(repo_root).as_posix())
        except ValueError:
            # Outside the repo root: never part of the hashed set anyway.
            continue
    return relpaths


def _tpu_compile_env_factors() -> dict[str, Any]:
    """Return TorchTPU plugin env values that can affect compilation."""
    factors = {}
    for name, getter in envs.environment_variables.items():
        if name in _TPU_COMPILE_ENV_IGNORED:
            continue
        try:
            factors[name] = getter()
        except Exception as exc:
            logger.warning(
                "Skipping TPU compile environment variable %s: %s", name, exc
            )
    # VPS flags (runner/vps_wiring.py) are registered in envs.py: the hashed ones
    # key the cache through the loop above, the diagnostics are ignored.
    # q38mm DSTOCK-B lean KV writer (fused_kv_write.LEAN_ENV) changes the traced QSA graph, so it keys the
    # cache when set, and only then: not registered in envs.py, so unset leaves every key unchanged.
    # q38 (DS-O1S) one-window KV block (pooled_qsa.ONE_WINDOW_ENV) changes the traced decode indexer: keyed the same way.
    # q38 L3-HCX (port of q38mm 20e6fcf23): the HCX row limit
    # (hc_decode.DECODE_XLA_MAX_ROWS_ENV) changes which HC mixes trace the
    # prefill-tile ops: keyed the same way.
    for switch in (
        "VLLM_TPU_QSA_FUSED_KV_WRITE_LEAN",
        "VLLM_TPU_QSA_DECODE_KV_BLOCK_ONE_WINDOW",
        "VLLM_TPU_HC_XLA_MAX_ROWS",
    ):
        value = os.environ.get(switch, "").strip()
        if value:
            factors[switch] = value
    # q38 L3-HCGATE: the fused decode HC row limit (hc_decode.DECODE_FUSED_MAX_ROWS_ENV)
    # changes which HC mixes trace the fused decode kernel: keyed the same way, only when set.
    hcgate = os.environ.get("VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS", "").strip()
    if hcgate:
        factors["VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS"] = hcgate
    # q38 AR-FIX decode TP all-reduce kernel (layers/adapter/tp_ar_decode.py): keyed the same way, only when set.
    # q38 INT-4 (H8 A2 85c05d64): import only when the flag is set; compile_factors() is {} whenever it is
    # empty, so the factors are unchanged in every env state and unset imports nothing (as 8c87f6f6b).
    if os.environ.get("VLLM_TPU_DECODE_TP_AR", "").strip():
        from vllm_torchtpu.kernels.collectives import tp8_ar_decode_fn

        factors.update(tp8_ar_decode_fn.compile_factors())
    # q38 INT-4 end
    return factors


def compute_tpu_compilation_hash(vllm_config: VllmConfig) -> str:
    """Hash TPU-specific factors used by both AOT and piecewise caches."""
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    spec = vllm_config.speculative_config
    factors = {
        "torch": torch.__version__,
        "torch_tpu": importlib_metadata.version("torch-tpu"),
        "jax": importlib_metadata.version("jax"),
        "jaxlib": importlib_metadata.version("jaxlib"),
        "libtpu": importlib_metadata.version("libtpu"),
        "env": _tpu_compile_env_factors(),
        "native_env": {name: os.getenv(name) for name in _NATIVE_TPU_COMPILE_ENV_VARS},
        "parallel": {
            "effective_data_parallel_size": utils.get_dp_size(
                vllm_config.parallel_config
            ),
        },
        # TPU AOT graphs embed the KV tensor extent as a literal. Hash the
        # resolved block count, which every rank shares via kv_cache_config.
        # The raw per-worker num_gpu_blocks_override is a pre-alignment HBM
        # measurement that diverges by a few blocks across ranks (the executor
        # aligns the value actually used to min()); hashing it would split
        # ranks into separate cache dirs and force asymmetric recompiles.
        "kv_cache": {
            "gpu_memory_utilization": cache_config.gpu_memory_utilization,
            "num_gpu_blocks": cache_config.num_gpu_blocks,
            "enable_prefix_caching": cache_config.enable_prefix_caching,
            "mamba_cache_mode": cache_config.mamba_cache_mode,
        },
        "scheduler": {
            "max_model_len": vllm_config.model_config.max_model_len,
            "max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
            "max_num_seqs": scheduler_config.max_num_seqs,
        },
        "speculative": {
            "model": spec.model if spec is not None else None,
            "draft_tensor_parallel_size": spec.draft_tensor_parallel_size
            if spec is not None
            else None,
            "num_speculative_tokens": spec.num_speculative_tokens
            if spec is not None
            else None,
        },
    }

    repo_root = Path(__file__).resolve().parents[2]
    hash_obj = hashlib.sha256(
        json.dumps(factors, sort_keys=True, separators=(",", ":")).encode()
    )
    for path in _iter_runtime_cache_key_files(repo_root):
        relpath = path.relative_to(repo_root).as_posix()
        hash_obj.update(relpath.encode())
        try:
            hash_obj.update(path.read_bytes())
        except OSError as exc:
            logger.warning(
                "[TpuCompilerAdaptor] Failed to read cache key source %s: %s",
                path,
                exc,
            )

    return hash_obj.hexdigest()[:10]


# TODO(geyuhao): Switch this cache-key hashing to upstream vllm.ir.util.hash_source
# once our local vLLM checkout exposes that helper.
def _ensure_tuple_output(graph: fx.GraphModule) -> tuple[fx.GraphModule, bool]:
    """Wrap a graph module to return a tuple if it returns a single tensor.

    aot_autograd requires graph outputs to be tuples. PiecewiseBackend
    sub-graphs may return a single tensor.

    Returns (wrapped_graph, was_wrapped).
    """
    output_node = None
    for node in graph.graph.nodes:
        if node.op == "output":
            output_node = node
            break

    if output_node is None:
        return graph, False

    output_args = output_node.args[0]
    if isinstance(output_args, (tuple, list)):
        return graph, False

    # Single return value — wrap in a tuple
    with graph.graph.inserting_before(output_node):
        output_node.args = ((output_args,),)
    graph.graph.lint()
    graph.recompile()
    return graph, True


def _single_size(compile_range: Range) -> int | None:
    """The token bucket a compile range pins, or None for a size range."""
    return compile_range.start if compile_range.is_single_size() else None


class TpuCompilerAdaptor(CompilerInterface):
    """CompilerInterface implementation for TPU.

    Delegates to TpuBackend which handles aot_autograd decomposition
    and FX->MLIR->PjRt compilation. Supports per-shape disk caching
    via pickle serialization of compiled executables.
    """

    name = "tpu"

    def __init__(self):
        self.cache_dir: str | None = None
        self._disable_cache = False

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ) -> None:
        self.cache_dir = cache_dir
        self._disable_cache = disable_cache

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        """Hash TPU runtime sources that sit behind custom-op boundaries."""
        return compute_tpu_compilation_hash(vllm_config)

    @staticmethod
    def _unwrap_compiled_fn(compiled_fn):
        inner_fn = compiled_fn

        def unwrap_fn(*args, **kwargs):
            result = inner_fn(*args, **kwargs)
            if isinstance(result, (tuple, list)) and len(result) == 1:
                return result[0]
            return result

        return unwrap_fn

    def _run_backend(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
    ) -> tuple[Callable[..., Any], Any | None]:
        """Compile `graph` with TpuBackend, and grab its serializable entry.

        Shared by `compile` and by `load`'s fallback, so a cache miss and a
        cold start reach the backend by exactly the same route.

        Returns (runnable, entry). The runnable still returns a tuple -- the
        caller applies the `_ensure_tuple_output` unwrap. `entry` is None when
        the backend attached no `serialize`, which it skips for dynamic-symint
        graphs and in debug mode (see TpuBackend.__call__).
        """
        # `_tpu_backend` calls detect_fake_mode(), which asserts a single
        # FakeTensorMode. The Dynamo tracing context's mode differs from the
        # example inputs' mode, so run under a context built from the example
        # inputs' own fake mode (consistent for detect_fake_mode). We also ensure
        # that mode has a ShapeEnv: with enable_serialization the bundled
        # AOTAutogradCache key computation (FxGraphCache._check_can_cache)
        # bypasses with "No shape env" when the tracing context has none, which
        # is fatal for these static graphs. An (empty) ShapeEnv yields empty
        # guards -- correct for static shapes -- and lets the artifact cache work.
        from torch._subclasses.fake_tensor import FakeTensor

        fake_mode = next(
            (t.fake_mode for t in example_inputs if isinstance(t, FakeTensor)), None
        )
        if fake_mode is not None and fake_mode.shape_env is None:
            fake_mode.shape_env = ShapeEnv()
        tracing_ctx = (
            torch._guards.TracingContext(fake_mode) if fake_mode is not None else None
        )
        # Each call builds one program. Kernels number their scheduling groups
        # from 1 within it, so the HLO, and with it torch_tpu's cache key, does
        # not depend on what this process compiled before: every rank then
        # builds the same key for a bucket and can hit an executable another
        # rank already put in the Tier-2 cache.
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import config

        config.reset_scheduling_group_ids()
        with torch._guards.tracing(tracing_ctx):
            compiled_fn = _tpu_backend(graph, example_inputs)

        # Read `serialize` off the object TpuBackend returned, before any
        # wrapping: the unwrap below replaces it with a plain closure that does
        # not carry attributes, and the entry would be unreachable after that.
        entry = None
        serialize = getattr(compiled_fn, "serialize", None)
        if serialize is not None:
            try:
                entry = serialize()
            except Exception as e:
                logger.warning(
                    "[TpuCompilerAdaptor] serialize() failed, this graph will "
                    "not be cached: %s",
                    e,
                )
        return compiled_fn, entry

    def _save_handle(
        self,
        key: str,
        path: str,
        entry: Any,
        graph: fx.GraphModule,
        was_wrapped: bool,
        fingerprint: str | None,
    ) -> bool:
        """Write the bundled entry for `graph` to `path`. True if it landed.

        `fingerprint` is `_graph_fingerprint(graph)` taken before the backend
        ran, as a peer replaying this artifact computes it.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = TpuCompilationHandle(
                key=key,
                entry=entry,
                signature=shape_variants.graph_signature(graph),
                was_wrapped=was_wrapped,
                fingerprint=fingerprint,
            )
            # Write-then-rename: a crash or a full disk partway through
            # pickling a 400 MB entry would otherwise leave a truncated file
            # that the next run reads back as a corrupt artifact.
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f)
            os.replace(tmp_path, path)
            logger.info("[TpuCompilerAdaptor] Saved compiled artifact to %s", path)
            return True
        except Exception as e:
            # Non-fatal by design: a failed save costs a recompile next run,
            # which is strictly better than persisting something unloadable.
            logger.warning("[TpuCompilerAdaptor] Failed to save artifact: %s", e)
            return False

    def _replay(
        self,
        path: str,
        graph: fx.GraphModule,
        was_wrapped: bool,
        fingerprint: str | None = None,
    ) -> Callable[..., Any] | None:
        """Rebuild the runnable saved at `path`, or None to recompile.

        Every rejection here costs one recompile and nothing else, so this is
        deliberately strict: an artifact is replayed only if it is the current
        format and was built from a graph with this one's input signature.
        With `fingerprint` (another rank's artifact), it must also have been
        built from exactly this graph.
        """
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning("[TpuCompilerAdaptor] Unreadable artifact %s: %s", path, e)
            return None

        reason = None
        if not isinstance(payload, TpuCompilationHandle):
            reason = f"unexpected payload type {type(payload).__name__}"
        elif payload.entry is None:
            reason = "no bundled entry"
        elif payload.signature != shape_variants.graph_signature(graph):
            # Not what makes the bundled entry safe -- that comes from wrapper
            # and executable being one object, with no lookup left to drift.
            # This is the same check BucketExecutables.get already applies to
            # the in-memory hand-over, extended across processes, and it earns
            # its place because nothing else validates the entry against the
            # graph: torch_tpu injects a synthetic cache_info and stubs out
            # generate_guards_expression, and replaying a pickle we wrote
            # ourselves never goes through AOTAutogradCache.try_load.
            # Placeholders only, so a changed graph *body* still passes -- that
            # is compute_tpu_compilation_hash's job, not this one.
            reason = (
                f"graph signature changed ({payload.signature} vs "
                f"{shape_variants.graph_signature(graph)})"
            )
        elif payload.was_wrapped != was_wrapped:
            reason = "output-tuple wrapping differs from the saved graph"
        elif fingerprint is not None and payload.fingerprint != fingerprint:
            reason = (
                f"built from a different graph ({payload.fingerprint} vs {fingerprint})"
            )
        if reason is not None:
            logger.warning(
                "[TpuCompilerAdaptor] Ignoring artifact %s: %s", path, reason
            )
            return None

        try:
            return _deserialize_entry(payload.entry)
        except Exception as e:
            logger.warning(
                "[TpuCompilerAdaptor] Could not replay artifact %s, recompiling: %s",
                path,
                e,
            )
            return None

    def _peer_paths(self, key: str) -> list[str]:
        """Where the other ranks have saved their artifact for `key`.

        vLLM gives each rank its own `<hash>/rank_<rank>_<dp_rank>/<prefix>`
        cache dir, so another rank's artifact for the same bucket sits at the
        same relative path under its `rank_*` dir. Any DP rank counts: the
        owner of a bucket can be in another DP group of the host, and the
        fingerprint check decides whether its artifact fits. Only dirs on this
        host's disk are seen unless the cache root is shared.
        """
        assert self.cache_dir is not None
        own = os.path.normpath(self.cache_dir)
        rank_dir, prefix = os.path.split(own)
        if re.fullmatch(r"rank_\d+_\d+", os.path.basename(rank_dir)) is None:
            return []
        pattern = os.path.join(
            glob.escape(os.path.dirname(rank_dir)),
            "rank_*_*",
            glob.escape(prefix),
            glob.escape(key),
        )
        mine = os.path.join(own, key)
        return sorted(p for p in glob.glob(pattern) if p != mine)

    def _replay_from_peer(
        self,
        key: str,
        graph: fx.GraphModule,
        was_wrapped: bool,
        fingerprint: str,
    ) -> tuple[Callable[..., Any], str] | None:
        """Replay the artifact another rank saved for this bucket.

        Waits for one to appear inside `wait_for_peer_artifact`. Returns the
        runnable and the peer's path, or None to compile: nothing appeared in
        time, or what appeared was built from a different graph -- then the
        ranks do not trace the same program and waiting on would not help.
        """
        deadline = time.monotonic() + getattr(_peer_wait, "timeout_s", 0.0)
        paths = self._peer_paths(key)
        while not paths and time.monotonic() < deadline:
            time.sleep(_PEER_POLL_S)
            paths = self._peer_paths(key)
        if not paths:
            return None
        compiled_fn = self._replay(paths[0], graph, was_wrapped, fingerprint)
        return None if compiled_fn is None else (compiled_fn, paths[0])

    @staticmethod
    def _link_artifact(src: str, dst: str) -> bool:
        """Make `dst` the same file as `src`: a hard link, a copy if `src`
        is on another filesystem. True if `dst` now exists."""
        tmp = f"{dst}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.link(src, tmp)
            except OSError:
                shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            return True
        except Exception as e:
            logger.warning(
                "[TpuCompilerAdaptor] Could not link %s to %s: %s", src, dst, e
            )
            with contextlib.suppress(OSError):
                os.remove(tmp)
            return False

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]:
        """Compile a Dynamo-level FX graph to a TPU PjRt executable.

        Delegates to TpuBackend which handles aot_autograd + fx_to_mlir
        + compile_mlir internally.

        Called once per (subgraph, token bucket). vLLM keeps the returned
        callable for this process and writes the returned handle into its
        compilation cache directory, where a later process reads it back and
        passes it to `load`. The handle is therefore an on-disk format.

        Returning `(fn, None)` means "usable now, but nothing was persisted" --
        vLLM will call `compile` again next run.
        """
        # A bucket this graph cannot serve, or one an earlier trace of the same
        # model already built, is not compiled here (see shape_variants).
        size = _single_size(compile_range)
        if key is not None and size is not None:
            elsewhere = shape_variants.runnable_for(self.cache_dir, key, graph, size)
            if elsewhere is not None:
                return elsewhere, None

        logger.info(
            "[TpuCompilerAdaptor] Compiling FX graph for range %s",
            compile_range,
        )

        # aot_autograd (inside TpuBackend) requires tuple outputs. Deep-copied
        # first because _ensure_tuple_output rewrites the output node in place
        # and vLLM keeps its own reference to `graph`. Only the output node
        # moves, so the placeholder signature is unaffected.
        graph = copy.deepcopy(graph)
        graph, was_wrapped = _ensure_tuple_output(graph)

        caching = (
            key is not None and self.cache_dir is not None and not self._disable_cache
        )

        # Another rank of this host compiled, or is compiling, this bucket:
        # replay its artifact instead of tracing, lowering and compiling it
        # again. The artifact is the one vLLM's compile cache already holds for
        # that rank; this rank's cache entry becomes a hard link to it.
        # With peer replay off, nothing replays another rank's artifact, so
        # none records a fingerprint either: `load` replays a rank's own
        # artifact without one, and a peer with replay on refuses it.
        peer = None
        fingerprint = (
            _graph_fingerprint(graph)
            if caching and envs.TPU_PRECOMPILE_PEER_REPLAY
            else None
        )
        if fingerprint is not None:
            assert key is not None
            peer = self._replay_from_peer(key, graph, was_wrapped, fingerprint)
        if peer is not None:
            assert key is not None and self.cache_dir is not None
            compiled_fn, peer_path = peer
            logger.info(
                "[TpuCompilerAdaptor] Replayed range %s from %s",
                compile_range,
                peer_path,
            )
            save_path = os.path.join(self.cache_dir, key)
            handle = (
                (key, save_path, was_wrapped)
                if self._link_artifact(peer_path, save_path)
                else None
            )
            if was_wrapped:
                compiled_fn = self._unwrap_compiled_fn(compiled_fn)
            if size is not None:
                shape_variants.remember(self.cache_dir, key, graph, compiled_fn)
            return compiled_fn, handle

        compiled_fn, entry = self._run_backend(graph, example_inputs)

        if was_wrapped:
            # Undo _ensure_tuple_output on the result side, so callers see the
            # single tensor the original graph returned.
            compiled_fn = self._unwrap_compiled_fn(compiled_fn)

        # Persist torch's bundled AOTAutograd entry -- the aot_autograd wrapper
        # metadata and the compiled executable as one object. It has to be one
        # object: `load` cannot rebuild a wrapper from a fresh trace and pair
        # it with a separately-fetched executable, because nothing at that
        # point can check the two describe the same program, and a mismatch is
        # only observable once the program runs -- as e0102
        # RuntimeProgramInputMismatch if the padded input buffers differ in
        # size, and as a silently wrong result if they happen not to.
        #
        # When the Tier-3 C++ persistent cache is active torch_tpu also keeps
        # the PJRT binary itself (torch_tpu_tier3/*.bin); that covers the
        # recompile path in `load`, not this one.
        handle = None
        if caching and entry is not None:
            assert self.cache_dir is not None
            save_path = os.path.join(self.cache_dir, key)
            if self._save_handle(
                key, save_path, entry, graph, was_wrapped, fingerprint
            ):
                handle = (key, save_path, was_wrapped)
        elif caching:
            logger.info(
                "[TpuCompilerAdaptor] No serializable artifact for range %s; "
                "it will be recompiled on the next start",
                compile_range,
            )

        # Offer this bucket to later traces of the same model, so a graph that
        # refuses it reuses this runnable instead of building a second
        # executable for it (see shape_variants.BucketExecutables).
        if key is not None and size is not None:
            shape_variants.remember(self.cache_dir, key, graph, compiled_fn)
        return compiled_fn, handle

    def load(
        self,
        handle: Any,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        compile_range: Range,
    ) -> Callable[..., Any]:
        """Rebuild a compiled runnable from the artifact `compile` saved.

        This is the warm-start path, and the reason a warm start is minutes
        rather than tens of minutes: FX->MLIR lowering and XLA compilation are
        both skipped. Replaying torch's bundled AOTAutograd entry restores the
        wrapper and the executable together, so there is no point at which a
        freshly traced wrapper could be paired with the wrong program.

        Falls back to a real compile whenever the artifact is missing, stale,
        or built from a different graph. That is never wrong, only slower --
        and with the Tier-3 C++ cache active it is not even much slower, since
        torch_tpu serves the PJRT binary natively.
        """
        assert isinstance(handle, tuple) and len(handle) == 3
        key, path, _saved_was_wrapped = handle
        size = _single_size(compile_range)
        # Screened on the load path too, and for a second reason beyond
        # "unsupported bucket": vLLM's key is (subgraph, bucket) with no graph
        # identity, so without this a re-traced variant would adopt an artifact
        # another trace compiled (see shape_variants.runnable_for).
        elsewhere = shape_variants.runnable_for(self.cache_dir, key, graph, size)
        if elsewhere is not None:
            return elsewhere

        logger.info("[TpuCompilerAdaptor] Loading compiled artifact from %s", path)

        # Same deepcopy + output-tuple rewrite as `compile`, so the graph the
        # artifact is checked against is shaped the way it was when saved.
        graph = copy.deepcopy(graph)
        graph, was_wrapped = _ensure_tuple_output(graph)

        compiled_fn = self._replay(path, graph, was_wrapped)
        if compiled_fn is None:
            fingerprint = (
                _graph_fingerprint(graph) if envs.TPU_PRECOMPILE_PEER_REPLAY else None
            )
            compiled_fn, entry = self._run_backend(graph, example_inputs)
            # Self-heal: overwrite the artifact vLLM will hand us again next
            # run. Without this a single stale entry recompiles forever.
            if (
                entry is not None
                and self.cache_dir is not None
                and not self._disable_cache
            ):
                self._save_handle(key, path, entry, graph, was_wrapped, fingerprint)

        if was_wrapped:
            compiled_fn = self._unwrap_compiled_fn(compiled_fn)

        # A loaded bucket is offered to later traces exactly like a compiled
        # one, so a graph that refuses this bucket reuses this runnable instead
        # of replaying the artifact into a second HBM program region.
        if size is not None:
            shape_variants.remember(self.cache_dir, key, graph, compiled_fn)
        return compiled_fn
