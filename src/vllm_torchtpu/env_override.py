# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import os

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
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                      "false")

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
    except Exception:
        pass
    return None


def _resolve_sc_offload_threshold_bytes() -> int | None:
    # SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES:
    #   "auto" (default): look up per-family constant
    #   "<integer>":      use that byte value
    #   "0":              disable the flag (fall through to libtpu default)
    override = os.getenv("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", "auto")
    if override != "auto":
        try:
            value = int(override)
        except ValueError:
            # Log-and-fall-through on bad values so a typo doesn't silently
            # no-op. (No vllm logger here; env_override runs before it's
            # importable.)
            import sys
            print(
                "vllm_torchtpu.env_override: invalid "
                f"SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES={override!r}; "
                "ignoring (SC-offload flag not set).",
                file=sys.stderr,
            )
            return None
        return value if value > 0 else None

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


_existing_libtpu_args = os.environ.get("LIBTPU_INIT_ARGS", "")


def _user_has_flag(flag_name: str) -> bool:
    # Tokenized startswith("<flag>=") check so a hypothetical longer flag
    # that embeds ours as a prefix doesn't false-positive.
    return any(
        token.startswith(flag_name + "=") or token == flag_name
        for token in _existing_libtpu_args.split())


_libtpu_extra_args: list[str] = []

# --xla_tpu_use_dynamic_smem_negotiation: let libtpu size SMEM to fit the
# batched RPA attention kernel's scheduling metadata (default budget is too
# small on --attention-backend CUSTOM).
if not _user_has_flag("--xla_tpu_use_dynamic_smem_negotiation"):
    _libtpu_extra_args.append("--xla_tpu_use_dynamic_smem_negotiation=true")

# --xla_tpu_sparse_core_{all_reduce,all_gather}_offload_min_size_in_bytes:
# raise the SC-offload threshold to the per-tensor-core VMEM capacity so
# typical per-layer AR/AG collectives run on TensorCore instead of being
# offloaded to SparseCore. The chip family is detected from
# TPU_ACCELERATOR_TYPE / GCE metadata and mapped to a VMEM constant above;
# each flag is added only if the user hasn't supplied it explicitly.
_sc_offload_bytes = _resolve_sc_offload_threshold_bytes()
if _sc_offload_bytes is not None:
    for _flag_name in (
            "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes",
            "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes",
    ):
        if not _user_has_flag(_flag_name):
            _libtpu_extra_args.append(f"{_flag_name}={_sc_offload_bytes}")

if _libtpu_extra_args:
    _assembled = " ".join(_libtpu_extra_args)
    if _existing_libtpu_args:
        _assembled = _assembled + " " + _existing_libtpu_args
    os.environ["LIBTPU_INIT_ARGS"] = _assembled


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
        exec(compile(patched_src, _lowering.__file__, "exec"),
             _lowering.__dict__, ns)
    except Exception:
        return
    new_fn = ns.get("_lower_jaxpr_to_for_loop")
    if new_fn is None:
        return
    _lowering._lower_jaxpr_to_for_loop = new_fn
    _lowering._torchtpu_fori_patch_applied = True


_patch_jax_pallas_fori_lowering()

# Map VLLM_XLA_CACHE_PATH to TorchTPU compilation cache environment variables
vllm_xla_cache_path = os.getenv("VLLM_XLA_CACHE_PATH")
if vllm_xla_cache_path:
    # Route Tier-3 native cache to a subdirectory under VLLM_XLA_CACHE_PATH
    os.environ.setdefault("TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
                          os.path.join(vllm_xla_cache_path, "torch_tpu_tier3"))
    # Enable Tier-2 cache in memory (required by Tier-3)
    os.environ.setdefault("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE",
                          "tpu_tier2_cache")
