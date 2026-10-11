# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

"""Model-aware runtime environment configuration for Qwen3.5 serving on TPU.

Applies default kernel, scheduler, and compilation flags when serving
Qwen3.5-397B-A17B unless explicitly overridden in the process environment.
"""

import json
import os
import re

# envs.py imports nothing from this package, so it is safe to load here,
# while the package's own __init__ is still running.
from vllm_torchtpu import envs

# Configure TorchTPU compilation cache environment variables directly
if envs.TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT:
    # Enable Tier-2 cache in memory (required by Tier-3)
    os.environ.setdefault("TORCH_TPU_TIER2_COMPILATION_CACHE", "tpu_tier2_cache")
    os.environ.setdefault(
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
        envs.TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT,
    )
    os.environ.setdefault(
        "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE", "tpu_tier2_cache"
    )

# Default VLLM_CACHE_ROOT to ~/.cache/vllm if unset, ensuring a writable default path
os.environ.setdefault("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))

# Disable CUDA-specific shared experts stream for TPU
# This prevents errors when trying to create CUDA streams on TPU hardware
# The issue was introduced by vllm-project/vllm#26440
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

# Disable forced graph breaks for functional collectives by default.
# torch_tpu commit 477f611f made the collective-ops graph-break guardrail
# active by default on module import in non-absl environments, causing a
# Dynamo Unsupported crash during vLLM startup (fullgraph=True). Setting
# this to false allows functional collectives to compile under SPMD, which
# is safe and required for Tensor Parallelism.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")

# vLLM 0.27.0 auto-enables breakable cudagraph for Kimi K3, disabling its
# torch.compile pipeline. TPU does not support CUDA graphs and requires that
# pipeline, so force the flag off before VllmConfig initialization.
os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"

# vLLM's "mega" AOT artifact is inductor-only by construction: it saves whatever
# a runnable's serialize() returns but reloads it through
# torch._inductor.standalone_compile.AOTCompiledArtifact.deserialize
# (vllm/compilation/caching.py load_all), which an out-of-tree CompilerInterface
# cannot produce without forging that format. So the artifact is written -- 689 MB
# per rank on the 12-layer Qwen3.5-35B -- and always fails to load:
#   "Compiling model again due to a load failure ..., reason: a bytes-like object
#    is required, not 'BundledAOTAutogradResult'"
# every start silently re-traces, and the 0-entry artifact the failed reload then
# saves makes the *next* start raise TypeError. It defaults on for torch >= 2.12
# (vllm/envs.py use_mega_aot_artifact). Our per-range executable cache already
# persists the same bytes, keyed one entry per (compile range, subgraph), so
# turning this off loses nothing and is what lets a warm start reuse them.
os.environ.setdefault("VLLM_USE_MEGA_AOT_ARTIFACT", "0")

# Size libtpu's premapped (DMA-mapped) host buffer pool when the Raiden KV
# transfer stack is enabled. Raiden's XlaHostMemoryAllocator deliberately
# skips per-allocation DMA mapping on multi-process v7x workers
# (host_memory_allocator.cc), and torch_tpu pins TPU_PREMAPPED_BUFFER_SIZE
# to "0" when unset (discovery.cc, memoized at torch_tpu import). The
# combination routes every KV-cache D2H/H2D through libtpu's synchronous
# staged-copy slow path (~0.4 GB/s with per-transfer IOMMU map/unmap).
# A sized pool restores the pinned fast path; 16 GiB per worker covers the
# connector's host mirrors with headroom. Must run before torch_tpu is
# imported; overridable by presetting the variable.
if (
    os.getenv("TPU_KV_RESHARD_TRANSPORT") == "raiden"
    or os.getenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER") == "1"
):
    os.environ.setdefault("TPU_PREMAPPED_BUFFER_SIZE", str(16 << 30))

# TPU_GDN_CONV_QK_PAIR_LAYOUT resolves in two layers:
#   1. An explicitly set value always wins (enable or disable), for both
#      raiden and non-raiden runs — hence setdefault below.
#   2. Otherwise, when the Raiden reshard stack is enabled, the layout
#      follows the reshard's source fan-in: enabled iff
#      TPU_RAIDEN_TRANSFER_PARALLELISM >= 8.  At such degrees a rank's
#      conv Q and K segments are sub-token (512 B halves of the 1024-byte
#      physical pool token under TP8), so only the pair-blocked layout
#      gives whole-token — placement-exact — raw transfer spans; at lower
#      source degrees the segment-major layout is already token-aligned.
# The fan-in is parsed exactly like envs.TPU_RAIDEN_TRANSFER_PARALLELISM
# (unset/empty means the default degree of 8 — keep the two in sync), so
# the layout decision and the transfer stack always agree; today that
# makes the pair layout the raiden default.  The launcher exports the
# variable identically on both roles, so producer and consumer resolve the
# same layout; a mismatched pair still fails closed through the layout
# fingerprint.  An unparsable value raises here, at startup, rather than
# letting the two readers diverge.
if (
    os.getenv("TPU_KV_RESHARD_TRANSPORT") == "raiden"
    or os.getenv("TPU_USE_RAIDEN_KV_CACHE_MANAGER") == "1"
):
    _raiden_transfer_degree = int(os.getenv("TPU_RAIDEN_TRANSFER_PARALLELISM") or "8")
    if _raiden_transfer_degree >= 8:
        os.environ.setdefault("TPU_GDN_CONV_QK_PAIR_LAYOUT", "1")

# Per-tensor-core VMEM capacity by TPU family (used to set the SC-offload
# threshold below). Values mirror JAX's pallas chip table at
# jax/_src/pallas/mosaic/tpu_info.py. We resolve from the chip family string
# because we run before libtpu is loadable and cannot query pltpu directly.
# Unknown families fall through to libtpu's default (no flag set); add new
# entries here when a new chip family ships.
_TPU_VMEM_BYTES_PER_TENSOR_CORE = {
    # TPU v5p: 64 MiB per tensor core.
    "v5p": 67108864,
    # TPU v5e (accelerator-type prefix "v5litepod"): 128 MiB per tensor core.
    "v5litepod": 134217728,
    # TPU v6e (Trillium): 128 MiB per tensor core.
    "v6e": 134217728,
    # TPU v7 (accelerator-type prefix "tpu7x" / device-kind "v7x"):
    # 64 MiB per tensor core.
    "tpu7x": 67108864,
    "v7x": 67108864,
}


def _fetch_tpu_type_from_metadata() -> str | None:
    # Inline the GCE metadata probe instead of delegating to
    # vllm_torchtpu.tpu_info, because that helper transitively imports
    # vllm.logger -> vllm -> torch -> torch_tpu autoload -> libtpu.so, which
    # would violate the "set LIBTPU_INIT_ARGS before libtpu loads" invariant
    # this file exists to enforce.
    try:
        import requests

        resp = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/"
            "instance/attributes/accelerator-type",
            headers={"Metadata-Flavor": "Google"},
        )
        if resp.status_code == 200 and resp.text:
            return resp.text
    except Exception:  # noqa: S110
        pass
    return None


def _resolve_sc_offload_threshold_bytes() -> int | None:
    # An explicit value wins, with 0 meaning no flag. None (the variable
    # was unset or "auto") falls through to the chip-family table below.
    override = envs.SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES
    if override is not None:
        return override if override > 0 else None

    tpu_type = os.getenv("TPU_ACCELERATOR_TYPE")
    if not tpu_type:
        tpu_type = _fetch_tpu_type_from_metadata()
    if not tpu_type:
        return None
    tpu_type = tpu_type.strip().lower()
    for prefix, size in _TPU_VMEM_BYTES_PER_TENSOR_CORE.items():
        if tpu_type.startswith(prefix):
            return size
    return None


# Every rewrite below (recipe env, LIBTPU_INIT_ARGS, serve argv) is recorded
# here as [kind, key, old, new] and exported once as VLLM_TORCHTPU_CONFIG_CHANGES
# (first process wins: spawned children inherit the rewritten env and record
# nothing), so TpuPlatform.check_and_update_config logs it in its
# [EFFECTIVE_CONFIG] line.
CHANGES: list[list] = []


def _set_env(key: str, val: str) -> None:
    old = os.environ.get(key)
    if old != val:
        os.environ[key] = val
        CHANGES.append(["env", key, old, val])


# Recipe tokens rewritten in place, so the rest of the string (and with it the
# native compile key) stays byte-identical. The CDK recipe pairs fused EP with
# SparseCore all-gather offload at a 256 KiB threshold.
_LIBTPU_REWRITES = {
    "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144": (
        "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=8388608"
    ),
    "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true": (
        "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false"
    ),
}
_SC_HAR_FLAG = "--xla_tpu_enable_sparse_core_hierarchical_all_reduce"
_SC_AR_MIN_FLAG = "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes"
# Either flag as a whole token, with its value and the whitespace before it.
_SC_HAR_TOKENS = re.compile(
    rf"\s*(?<!\S)(?:{re.escape(_SC_HAR_FLAG)}|{re.escape(_SC_AR_MIN_FLAG)})"
    r"(?:=\S*)?(?!\S)"
)


def _without_sc_hierarchical_all_reduce(args: str) -> str:
    """SparseCore hierarchical all-reduce: recipes inject it directly
    (HAR_MIN_TOKENS, together with an all-reduce offload threshold of
    HAR_MIN_TOKENS x 8 KiB). It halts the SparseCore sequencer at DCP=4 and is
    the suspect for corrupted >= 8 MiB all-reduces at DCP=1 (the embedding
    all-reduce of a >= 1024-row step, every draft all-reduce at >= 1024 rows).
    Unless every spelling of it already disables it, strip it together with
    the recipe's all-reduce threshold and append both: disabled, and 64 MiB,
    the threshold every validated run used (8 MiB without HAR never ran)."""
    har = [t for t in args.split() if t.split("=", 1)[0] == _SC_HAR_FLAG]
    if all(t.partition("=")[2].lower() in ("false", "0") for t in har):
        return args
    dropped = " ".join(t.strip() for t in _SC_HAR_TOKENS.findall(args))
    added = f"{_SC_HAR_FLAG}=false {_SC_AR_MIN_FLAG}={64 << 20}"
    CHANGES.append(["libtpu_rewrite", dropped, added])
    return f"{_SC_HAR_TOKENS.sub('', args)} {added}"


def _recipe_libtpu_args(args: str) -> str:
    for old, new in _LIBTPU_REWRITES.items():
        if old in args:
            args = args.replace(old, new)
            CHANGES.append(["libtpu_rewrite", old, new])
    return _without_sc_hierarchical_all_reduce(args)


_existing_libtpu_args = _recipe_libtpu_args(os.environ.get("LIBTPU_INIT_ARGS", ""))

# The CDK recipe (infx-qwen35-agentx) exports stale fused-EP settings in the
# pod spec: USE_MOE_FUSED_EP_KERNEL=0 and TPU_MOE_FUSED_EP_STEP_MIN_TOKENS=33.
# When TPU_DISABLE_PR989_KERNELS=1 is set (for leave-one-out PR #989 ablation),
# disable the PR #989 kernel/placement stack while keeping the main GMMv2
# single-sort MoE path and all non-PR#989 runtime settings intact.
if os.environ.get("TPU_DISABLE_PR989_KERNELS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
):
    _set_env("USE_MOE_FUSED_EP_KERNEL", "0")
    _set_env("TPU_MOE_SEQUENCE_PARALLEL", "0")
    _set_env("TPU_MOE_LOWTOK", "0")
    _set_env("TPU_SMALL_ALL_REDUCE", "0")
    _set_env("TPU_GDN_FUSED_NORM_QUANT", "0")
    _set_env("TPU_FUSED_QK_NORM_ROPE", "0")
    _set_env("TPU_EXPERT_PLACEMENT", "")
    _set_env("TPU_GREEDY_TOP_TOKENS", "0")
    _set_env("TPU_SAMPLER_TOPK_CANDIDATES", "0")
    _set_env("VLLM_TPU_FUSED_DRAFT_LOOP", "0")
    _set_env("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", "0")
elif os.environ.get("TPU_RAIDEN_JOB_NAME", "").strip().lower() == "decode":
    if "USE_MOE_FUSED_EP_KERNEL" not in os.environ:
        _set_env("USE_MOE_FUSED_EP_KERNEL", "0")
else:
    _set_env("USE_MOE_FUSED_EP_KERNEL", "1")
    if os.environ.get("TPU_MOE_FUSED_EP_STEP_MIN_TOKENS") in (None, "", "33"):
        _set_env("TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", "64")


def _user_has_flag(flag_name: str) -> bool:
    # Tokenized startswith("<flag>=") check so a hypothetical longer flag
    # that embeds ours as a prefix doesn't false-positive.
    return any(
        token.startswith(flag_name + "=") or token == flag_name
        for token in _existing_libtpu_args.split()
    )


_libtpu_extra_args: list[str] = []

for _default_flag, _default_val in (
    ("--xla_tpu_use_dynamic_smem_negotiation", "true"),
    ("--xla_tpu_dvfs_p_state", "7"),
    ("--xla_tpu_use_minor_sharding_for_major_trivial_input", "true"),
    ("--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter", "false"),
    ("--xla_tpu_enable_sparse_core_collective_offload_all_gather", "false"),
    ("--xla_tpu_ars_combiner_threshold_in_bytes", "0"),
    ("--xla_tpu_enable_async_collective_merger", "false"),
):
    if not _user_has_flag(_default_flag):
        _libtpu_extra_args.append(f"{_default_flag}={_default_val}")

_sc_offload_bytes = _resolve_sc_offload_threshold_bytes()
if _sc_offload_bytes is not None:
    # No SparseCore hierarchical all-reduce: with 8MiB AR/AG offload it halts
    # the SparseCore sequencer (E0200) during DCP=4 prefill bursts on tpu7x.
    for _flag_name in (
        "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes",
        "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes",
    ):
        if not _user_has_flag(_flag_name):
            _libtpu_extra_args.append(f"{_flag_name}={_sc_offload_bytes}")

CHANGES.extend(["libtpu_add", tok] for tok in _libtpu_extra_args)
if _libtpu_extra_args or _existing_libtpu_args:
    _assembled = " ".join(
        x for x in [" ".join(_libtpu_extra_args), _existing_libtpu_args] if x
    )
    os.environ["LIBTPU_INIT_ARGS"] = _assembled


def _has_cli_opt(argv: list[str], opt: str) -> bool:
    return any(a == opt or a.startswith(opt + "=") for a in argv)


def _get_cli_opt(argv: list[str], opt: str) -> str | None:
    for i, a in enumerate(argv):
        if a == opt and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(opt + "="):
            return a.split("=", 1)[1]
    return None


def _apply_qwen35_serve_cli_defaults() -> None:
    """`vllm serve` of the Qwen3.5-397B checkpoint: MTP=3 speculative decoding
    and 4 tokenizer renderer workers unless the command line sets them."""
    import sys

    argv = sys.argv
    if (
        "serve" not in argv[:3]
        or "bench" in argv[:3]
        or "qwen3.5-397b" not in " ".join(argv).lower()
    ):
        return
    for opt, val in (
        ("--speculative-config", '{"method":"mtp","num_speculative_tokens":3}'),
        ("--renderer-num-workers", "4"),
    ):
        if not _has_cli_opt(argv, opt):
            argv.extend([opt, val])
            CHANGES.append(["argv_add", opt, val])
    # <=64-row decode/MTP-draft buckets use half VMEM (32 MiB on v7x) so XLA
    # prefetches next layers' weights across batched RPA, while 128-row c=32
    # MTP=3 target verification keeps full VMEM.
    if "TPU_RPA_DECODE_BATCHED" not in os.environ:
        _set_env("TPU_RPA_DECODE_BATCHED", "1")
    # Pack MoE permutation into a single int32 radix sort (60 sorts/step vs
    # 180 across 60 MoE layers) whenever steps route via fused_moe_gmm.
    if "TPU_MOE_SINGLE_SORT" not in os.environ:
        _set_env("TPU_MOE_SINGLE_SORT", "1")
    dcp = _get_cli_opt(argv, "--decode-context-parallel-size") or ""
    if dcp.isdigit() and int(dcp) > 1:
        if "TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS" not in os.environ:
            # torch_tpu's reassociate_norm_weights pass is the DCP>1 crash suspect.
            _set_env("TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS", "0")
        if "MOE_FUSED_EP_KERNEL_IMPL" not in os.environ:
            # Fused EP v3 triggers HBM OOM (E0101 -> E0200) under DCP>1 when
            # combined with dual-pass DCP RPA graphs; use v2 for DCP>1.
            _set_env("MOE_FUSED_EP_KERNEL_IMPL", "v2")


_apply_qwen35_serve_cli_defaults()
os.environ.setdefault(
    "VLLM_TORCHTPU_CONFIG_CHANGES", json.dumps(CHANGES, separators=(",", ":"))
)


def _patch_jax_pallas_fori_lowering() -> None:
    """Elide the dead remainder ``scf.for`` from Pallas ``fori_loop`` lowering.

    JAX Pallas' Mosaic lowering unconditionally emits a remainder loop for
    dynamic-bound ``jax.lax.fori_loop`` even when ``unroll=1`` (the default),
    where the main loop already covers the full range and the remainder is a
    no-op. libtpu still lowers the empty remainder to device code, which
    materially grows the emitted Mosaic body and slows decode-heavy MoE
    workloads that rely on ``fori_loop`` in Pallas scheduler kernels.

    Rewrites ``_lower_jaxpr_to_for_loop`` in place to gate the remainder
    emission on ``unroll != 1``. Applied at import so it takes effect before
    any Pallas kernel is traced. No-op if the source no longer contains the
    exact needle (upstream fix landed, or the file was refactored), so it
    remains safe across JAX bumps.
    """
    try:
        from jax._src.pallas.mosaic import lowering as _lowering
    except Exception:
        return
    if getattr(_lowering, "_torchtpu_fori_patch_applied", False):
        return
    orig = getattr(_lowering, "_lower_jaxpr_to_for_loop", None)
    if orig is None:
        return

    import inspect
    import textwrap

    try:
        src = textwrap.dedent(inspect.getsource(orig))
    except Exception:
        return
    needle = "elif has_dynamic_remainder:"
    fix = "elif has_dynamic_remainder and unroll != 1:"
    if needle not in src or fix in src:
        return
    patched_src = src.replace(needle, fix, 1)
    ns: dict = {"__name__": _lowering.__name__}
    try:
        exec(compile(patched_src, _lowering.__file__, "exec"), _lowering.__dict__, ns)
    except Exception:
        return
    new_fn = ns.get("_lower_jaxpr_to_for_loop")
    if new_fn is None:
        return
    _lowering._lower_jaxpr_to_for_loop = new_fn
    _lowering._torchtpu_fori_patch_applied = True


from vllm_torchtpu import patch_registry  # noqa: E402

patch_registry.apply("import")
