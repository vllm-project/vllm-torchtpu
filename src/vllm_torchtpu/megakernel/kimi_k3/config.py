# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel settings for the Kimi-K3 megakernels.

Kernel modules read their settings from :func:`active` when they are
imported (most settings are module constants), so a non-default config must
be installed with :func:`install` before the first kernel import. Serving and
the standalone tools install overrides before building the program.
"""

import dataclasses
import hashlib
import json
from typing import Any

from vllm_torchtpu.megakernel.config import set_active_kernel_config


@dataclasses.dataclass(frozen=True)
class KernelConfig:
    # Decode context parallel attention.
    dcp_group: int = 32
    # Attention data parallelism of the decoder (TP32 / dp heads per DP group
    # of 32 / dp ranks, each with its own rows); needs dcp_group = 32 / dp.
    attention_dp: int = 1
    # Block tables: "local" (vLLM's rank-local DCP tables) or "global" (global
    # page ids).
    dcp_tables: str = "local"
    # MLA decode key tile (tokens per page fold). 2048 is opt-in and applies
    # only to FP8 plain decode at attention_dp=2, dcp_group=16 with 32 rows per
    # DP group (B64); everything else uses 1024.
    mla_decode_ktile: int = 1024
    # Routed experts and weight layout (the pre-shard format).
    expert_gate_up: str = "mxfp4"  # "mxfp4" or "fp8" (reblocked)
    mxfp4_block: int = 2048
    mxfp4_pair_tile: int = 512
    mxfp4_static_tile: int = 0
    # None: 1024 at B96, 512 at power-of-two B1-B128, else unsplit (0).
    mxfp4_dot_tile: int | None = None
    pooled_weights: bool = True
    shared_shards: bool | None = None  # None: at B128
    compact_only: bool = False
    singleton_block: bool | None = None  # None: at B16-B128
    stage_blocks: bool | None = None  # None: at B16-B128
    b16_compact_experts: bool = True
    shared_expert_finishers: bool = True
    compact_overflow: bool = True
    b4_attention_gather: bool = True
    fused_cache_commit: bool = True
    decode_no_hlo_calls: bool = True  # no deduplicated HLO calls at B8..B128, DP B2/B4
    shard_routing: bool = False
    kda_staged: bool = False
    weight_streams: tuple[tuple[str, tuple], ...] = ()  # (weight, (axis, streams))
    # VMEM and compiler settings.
    vmem_limit_mb: int = 64
    vmem_spill_reserve_mb: float = 5.0
    internal_scratch_kb: int = 0
    bounds_checks: bool = False
    # Prefill attention.
    # Prefill MLA flash tiles (queries x keys per step).
    mla_flash_qtile: int = 1024
    mla_flash_ktile: int = 2048
    # Prefix keys in mla_flash_ktile tiles and a whole-window query tile under
    # mla_flash_qtile rows (False: the window's key tile, as before).
    mla_flash_prefix_tiles: bool = True
    # Prefill MoE expert parallelism: 8 (EP8/TP4, the pre-shard layout), 16
    # (EP16/TP2: 56 experts, TP2 within the chip) or 32 (28 whole experts);
    # 16 and 32 regroup the EP8 group's shards at load.
    prefill_expert_parallel: int = 16
    # Packed dispatch/shared overlap and fused down/combine/reduce-scatter.
    # EP16: FP32 at all serving windows, BF16 at 4K/8K only.
    # False: the unfused SparseCore gather/combine.
    prefill_fused_moe: bool = True
    # Both prefill MoE rings: BF16 halves transfer bytes, rounding every hop.
    moe_reduce_dtype: str = "float32"
    # Weight loading.
    read_streams: int = 4
    load_inflight_gb: float = 16.0
    # Diagnostics (off in serving): dcp_skip and debug_skip skip named work, so
    # results are invalid (timing or hang bisection only); the rest add trace
    # scopes, VMEM reports or VMEM padding.
    dcp_skip: tuple[str, ...] = ()
    debug_skip: tuple[str, ...] = ()
    phase_scopes: bool = False
    pool_report: bool = False
    vmem_report: bool = False
    vmem_pad_kb: int = 0

    def __post_init__(self) -> None:
        if self.prefill_expert_parallel not in (8, 16, 32):
            raise ValueError("prefill_expert_parallel is 8, 16 or 32")
        if self.moe_reduce_dtype not in ("float32", "bfloat16"):
            raise ValueError("moe_reduce_dtype is float32 or bfloat16")
        if self.mla_decode_ktile not in (1024, 2048):
            raise ValueError("mla_decode_ktile supports 1024 or experimental 2048")
        if self.mla_decode_ktile == 2048 and (
            self.attention_dp != 2 or self.dcp_group != 16
        ):
            raise ValueError(
                "mla_decode_ktile=2048 requires attention_dp=2 and dcp_group=16"
            )

    def key(self) -> str:
        """A stable digest of every field, for caches of compiled kernels."""
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def replace(self, **fields: Any) -> "KernelConfig":
        return dataclasses.replace(
            self, **{name: _coerce(self, name, value) for name, value in fields.items()}
        )


def _coerce(config: KernelConfig, name: str, value: Any) -> Any:
    """Parse a command-line string into the field's type."""
    current = getattr(config, name)
    if not isinstance(value, str):
        return value
    if current is None:  # an optional override: parse by the declared type
        if value.lower() in ("", "none"):
            return None
        kind = str(KernelConfig.__dataclass_fields__[name].type)
        if kind.startswith("bool"):
            return value.lower() in ("1", "true", "yes", "on")
        if kind.startswith("int"):
            return int(value)
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if name == "weight_streams":
        return tuple(sorted((k, tuple(v)) for k, v in json.loads(value).items()))
    if isinstance(current, tuple):
        return tuple(filter(None, value.split(",")))
    return value


_ACTIVE: KernelConfig | None = None


def install(config: KernelConfig) -> KernelConfig:
    """Fix the process's kernel config. Kernel modules read it at import, so
    installing a different config after the first import is an error."""
    global _ACTIVE
    if _ACTIVE is not None and config != _ACTIVE:
        raise RuntimeError(
            f"K3 kernel config is already fixed ({_ACTIVE}); install {config} before "
            "the first kernel import"
        )
    _ACTIVE = config
    set_active_kernel_config(config)
    return config


def active() -> KernelConfig:
    """The installed config, or the defaults (installed on first use)."""
    return _ACTIVE if _ACTIVE is not None else install(KernelConfig())
