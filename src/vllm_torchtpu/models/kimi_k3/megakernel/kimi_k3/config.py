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

"""Kernel settings for the Kimi-K3 megakernels.

Kernel modules read their settings from :func:`active` when they are
imported (most settings are module constants), so a non-default config must
be installed with :func:`install` before the first kernel import. Serving and
the standalone tools install overrides before building the program.
"""

import dataclasses
import hashlib
import json
from typing import Any, Final

from vllm_torchtpu.models.kimi_k3.megakernel.config import set_active_kernel_config

_DISABLED_PLAN_VALUES: Final[frozenset[str]] = frozenset(
    {"", "0", "none", "off", "false"}
)
EP8_DEFAULT_EXPERT_PLAN: Final[str] = "plan_ep8_real_hotfirst.npz"
EP16_DEFAULT_EXPERT_PLAN: Final[str] = "plan_ep16_real_hotfirst.npz"
EP8_DEFAULT_REPLICA_PLAN: Final[str] = "replica_plan_ep8_real_plain4.npz"
EP16_DEFAULT_REPLICA_PLAN: Final[str] = "replica_plan_ep16_real_plain4.npz"

# Production defaults for Kimi-K3 TPU7x megakernel serving and prefill scheduling.
DEFAULT_AGGREGATE_EP16: bool = False
DEFAULT_DCP_COMBINE_CHUNKS: bool = True
DEFAULT_DECODE_EXPERT_BANKS: bool = False
DEFAULT_FP8_WIRES: str = ""
DEFAULT_HOT_PLAN: str = ""
DEFAULT_MAX_PREFILL_STEP_TOKENS: int = 16384
DEFAULT_PREFILL_A2A_CHIP_SHARD: bool = True
DEFAULT_PREFILL_EXPERT_PLAN: str = EP16_DEFAULT_EXPERT_PLAN
DEFAULT_PREFILL_KDA_A2A: bool = True
DEFAULT_PREFILL_KDA_DUAL_GATHER: bool = False
DEFAULT_PREFILL_KDA_RING_SET: bool = True
DEFAULT_PREFILL_KDA_A2A_FP8: bool = False
DEFAULT_PREFILL_KDA_FP8_GATHER: bool = False
DEFAULT_PREFILL_MLA_A2A: bool = True
DEFAULT_PREFILL_MLA_A2A_FP8: bool = False
DEFAULT_PREFILL_MOE_16K_MONOLITHIC: bool = True
DEFAULT_PREFILL_MOE_DENSE_FP8: bool = False
DEFAULT_PREFILL_MOE_W4A8: bool = True
DEFAULT_PREFILL_PREFIX_FP8: bool = True
DEFAULT_PREFILL_REPLICA_PLAN: str = EP16_DEFAULT_REPLICA_PLAN
DEFAULT_PREFILL_REPLICA_PLAN_EXPLICIT: bool = False
DEFAULT_PREFILL_ROUTE_COUNTS: bool = False
DEFAULT_PREFILL_ROUTE_COUNTS_DIR: str = ""
DEFAULT_REPLICA_PLAN: str = EP16_DEFAULT_REPLICA_PLAN
DEFAULT_SERIAL_PREFILL: bool = True
DEFAULT_UNIFIED_EP16: bool = True

PRODUCTION_DEFAULTS: Final[dict[str, Any]] = {
    "DEFAULT_AGGREGATE_EP16": DEFAULT_AGGREGATE_EP16,
    "DEFAULT_DCP_COMBINE_CHUNKS": DEFAULT_DCP_COMBINE_CHUNKS,
    "DEFAULT_DECODE_EXPERT_BANKS": DEFAULT_DECODE_EXPERT_BANKS,
    "DEFAULT_FP8_WIRES": DEFAULT_FP8_WIRES,
    "DEFAULT_HOT_PLAN": DEFAULT_HOT_PLAN,
    "DEFAULT_MAX_PREFILL_STEP_TOKENS": DEFAULT_MAX_PREFILL_STEP_TOKENS,
    "DEFAULT_PREFILL_A2A_CHIP_SHARD": DEFAULT_PREFILL_A2A_CHIP_SHARD,
    "DEFAULT_PREFILL_EXPERT_PLAN": DEFAULT_PREFILL_EXPERT_PLAN,
    "DEFAULT_PREFILL_KDA_A2A": DEFAULT_PREFILL_KDA_A2A,
    "DEFAULT_PREFILL_KDA_DUAL_GATHER": DEFAULT_PREFILL_KDA_DUAL_GATHER,
    "DEFAULT_PREFILL_KDA_RING_SET": DEFAULT_PREFILL_KDA_RING_SET,
    "DEFAULT_PREFILL_KDA_A2A_FP8": DEFAULT_PREFILL_KDA_A2A_FP8,
    "DEFAULT_PREFILL_KDA_FP8_GATHER": DEFAULT_PREFILL_KDA_FP8_GATHER,
    "DEFAULT_PREFILL_MLA_A2A": DEFAULT_PREFILL_MLA_A2A,
    "DEFAULT_PREFILL_MLA_A2A_FP8": DEFAULT_PREFILL_MLA_A2A_FP8,
    "DEFAULT_PREFILL_MOE_16K_MONOLITHIC": DEFAULT_PREFILL_MOE_16K_MONOLITHIC,
    "DEFAULT_PREFILL_MOE_DENSE_FP8": DEFAULT_PREFILL_MOE_DENSE_FP8,
    "DEFAULT_PREFILL_MOE_W4A8": DEFAULT_PREFILL_MOE_W4A8,
    "DEFAULT_PREFILL_PREFIX_FP8": DEFAULT_PREFILL_PREFIX_FP8,
    "DEFAULT_PREFILL_REPLICA_PLAN": DEFAULT_PREFILL_REPLICA_PLAN,
    "DEFAULT_PREFILL_REPLICA_PLAN_EXPLICIT": DEFAULT_PREFILL_REPLICA_PLAN_EXPLICIT,
    "DEFAULT_PREFILL_ROUTE_COUNTS": DEFAULT_PREFILL_ROUTE_COUNTS,
    "DEFAULT_PREFILL_ROUTE_COUNTS_DIR": DEFAULT_PREFILL_ROUTE_COUNTS_DIR,
    "DEFAULT_REPLICA_PLAN": DEFAULT_REPLICA_PLAN,
    "DEFAULT_SERIAL_PREFILL": DEFAULT_SERIAL_PREFILL,
    "DEFAULT_UNIFIED_EP16": DEFAULT_UNIFIED_EP16,
}


def resolve_expert_parallel_settings(
    *,
    aggregate: bool,
    unified_ep16: bool,
    aggregate_ep16: bool,
    raw_expert_plan: str,
    raw_replica_plan: str,
    raw_hot_plan: str,
) -> tuple[int, int, str, str]:
    expert_disabled = raw_expert_plan.strip().lower() in _DISABLED_PLAN_VALUES
    replica_disabled = raw_replica_plan.strip().lower() in _DISABLED_PLAN_VALUES
    if not aggregate:
        # Disaggregated 1P1D: prefill runs EP=16, decode runs EP=8. Never pass
        # an EP=16 replica plan or a disabled token ("none"/"0"/"off"/"false")
        # to the EP=8 decoder, and suppress replica plans when hot_plan is set.
        expert_plan = (
            ""
            if expert_disabled or raw_expert_plan == EP8_DEFAULT_EXPERT_PLAN
            else raw_expert_plan
        )
        if (
            replica_disabled
            or raw_hot_plan
            or raw_replica_plan == EP16_DEFAULT_REPLICA_PLAN
        ):
            replica_plan = ""
        else:
            replica_plan = raw_replica_plan
        return 16, 8, expert_plan, replica_plan
    if unified_ep16:
        if raw_expert_plan == "":
            expert_plan = EP16_DEFAULT_EXPERT_PLAN
        elif expert_disabled:
            expert_plan = ""
        elif raw_expert_plan == EP8_DEFAULT_EXPERT_PLAN:
            expert_plan = EP16_DEFAULT_EXPERT_PLAN
        else:
            expert_plan = raw_expert_plan
        if raw_replica_plan == "":
            replica_plan = "" if raw_hot_plan else EP16_DEFAULT_REPLICA_PLAN
        elif replica_disabled or raw_hot_plan:
            replica_plan = ""
        elif raw_replica_plan == EP8_DEFAULT_REPLICA_PLAN:
            replica_plan = EP16_DEFAULT_REPLICA_PLAN
        else:
            replica_plan = raw_replica_plan
        return 16, 16, expert_plan, replica_plan
    prefill_ep = 16 if aggregate_ep16 else 8
    if prefill_ep == 8:
        if raw_expert_plan == "" or raw_expert_plan == EP16_DEFAULT_EXPERT_PLAN:
            expert_plan = EP8_DEFAULT_EXPERT_PLAN
        elif expert_disabled:
            expert_plan = ""
        else:
            expert_plan = raw_expert_plan
        if raw_replica_plan == "" or raw_replica_plan == EP16_DEFAULT_REPLICA_PLAN:
            replica_plan = "" if raw_hot_plan else EP8_DEFAULT_REPLICA_PLAN
        elif replica_disabled or raw_hot_plan:
            replica_plan = ""
        else:
            replica_plan = raw_replica_plan
    else:
        expert_plan = (
            ""
            if expert_disabled or raw_expert_plan == EP8_DEFAULT_EXPERT_PLAN
            else raw_expert_plan
        )
        replica_plan = (
            ""
            if (
                replica_disabled
                or raw_hot_plan
                or raw_replica_plan == EP16_DEFAULT_REPLICA_PLAN
            )
            else raw_replica_plan
        )
    return prefill_ep, 8, expert_plan, replica_plan


def resolve_decode_expert_banks(
    *, prefill_ep: int, decode_ep: int, enabled: bool
) -> bool:
    return enabled and prefill_ep == decode_ep == 16


def _normalize_plan_value(value: str) -> str:
    stripped = value.strip()
    return "" if stripped.lower() in _DISABLED_PLAN_VALUES else stripped


def resolve_role_kernel_settings(
    *,
    role: str,
    unified_ep16: bool,
    aggregate_ep16: bool,
    raw_expert_plan: str,
    raw_replica_plan: str,
    raw_prefill_replica_plan: str,
    raw_hot_plan: str,
    prefill_replica_plan_explicit: bool = False,
) -> tuple[int, int, str, str, str, str]:
    """Resolve ``(prefill_ep, decode_ep, prefill_expert_plan, kd_replica_plan,
    prefill_replica_plan, kd_hot_plan)`` for ``role``."""
    hot_plan = _normalize_plan_value(raw_hot_plan)
    prefill_rep = _normalize_plan_value(raw_prefill_replica_plan)
    if prefill_rep == EP8_DEFAULT_REPLICA_PLAN:
        prefill_rep = ""

    if role == "decode":
        _, decode_ep, _, kd_rep = resolve_expert_parallel_settings(
            aggregate=False,
            unified_ep16=unified_ep16,
            aggregate_ep16=aggregate_ep16,
            raw_expert_plan="",
            raw_replica_plan=raw_replica_plan,
            raw_hot_plan=hot_plan,
        )
        return 16, decode_ep, "", kd_rep, "", hot_plan

    if role == "prefill":
        prefill_ep, decode_ep, expert_plan, _ = resolve_expert_parallel_settings(
            aggregate=False,
            unified_ep16=unified_ep16,
            aggregate_ep16=aggregate_ep16,
            raw_expert_plan=raw_expert_plan,
            raw_replica_plan="",
            raw_hot_plan="",
        )
        return prefill_ep, decode_ep, expert_plan, "", prefill_rep, ""

    if role != "aggregate":
        raise ValueError(f"unknown megakernel role: {role!r}")

    prefill_ep, decode_ep, expert_plan, kd_rep = resolve_expert_parallel_settings(
        aggregate=True,
        unified_ep16=unified_ep16,
        aggregate_ep16=aggregate_ep16,
        raw_expert_plan=raw_expert_plan,
        raw_replica_plan=raw_replica_plan,
        raw_hot_plan=hot_plan,
    )
    if prefill_ep != 16 or hot_plan:
        return prefill_ep, decode_ep, expert_plan, kd_rep, "", hot_plan
    if decode_ep == 16:
        if (
            not prefill_replica_plan_explicit
            or kd_rep
            and prefill_rep == EP16_DEFAULT_REPLICA_PLAN
        ):
            resolved_prefill_rep = kd_rep
        else:
            resolved_prefill_rep = prefill_rep
        return (
            prefill_ep,
            decode_ep,
            expert_plan,
            kd_rep,
            resolved_prefill_rep,
            hot_plan,
        )
    # Aggregate with prefill_ep=16, decode_ep=8: avoid colliding with an EP8
    # decode replica plan on KernelConfig.__post_init__ equality check.
    resolved_prefill_rep = "" if kd_rep else prefill_rep
    return prefill_ep, decode_ep, expert_plan, kd_rep, resolved_prefill_rep, hot_plan


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
    dcp_combine_chunks: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_DCP_COMBINE_CHUNKS
    )
    # MLA decode key tile (tokens per page fold). 2048 is opt-in and applies
    # only to FP8 plain decode at attention_dp=2, dcp_group=16 with 32 rows per
    # DP group (B64); everything else uses 1024.
    mla_decode_ktile: int = 1024
    # Routed experts and weight layout (the pre-shard format).
    expert_gate_up: str = "mxfp4"  # "mxfp4" or "fp8" (reblocked)
    decode_expert_parallel: int = 8
    decode_expert_banks: bool = False
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
    # DP2 plain decode above B16: the kernel writes KDA post-states into their slots.
    kda_inplace_writes: bool = True
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
    # mla_flash_qtile rows. False: prefix keys use the window's key tile.
    mla_flash_prefix_tiles: bool = True
    # Cached-prefix attention in FP8 (E4M3FN keys/values from the expansion, FP8
    # queries, probabilities x16 in E4M3FN, FP32 statistics); the window's own
    prefill_prefix_fp8: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_PREFIX_FP8
    )
    # Prefill MoE route counts (diagnostic): each fused MoE call also returns its
    # chip's 56 local expert counts; the prefiller saves cumulative snapshots.
    prefill_route_counts: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_ROUTE_COUNTS
    )
    prefill_route_counts_every: int = 200
    # Prefill EP16 expert placement (empty: the pre-shard's): an npz whose
    # ``order`` [92, 896] is the logical expert at each physical slot (chip c:
    # slots 56c..56c+55); the router columns and expert weights are loaded in
    # that order. A bare name loads from kimi_k3/data.
    prefill_expert_plan: str = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_EXPERT_PLAN
    )
    prefill_replica_plan: str = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_REPLICA_PLAN
    )
    # Prefill MoE expert parallelism: 8 (EP8/TP4, the pre-shard layout), 16
    # (EP16/TP2: 56 experts, TP2 within the chip) or 32 (28 whole experts);
    # 16 and 32 regroup the EP8 group's shards at load.
    prefill_expert_parallel: int = 16
    # Packed dispatch/shared overlap and fused down/combine/reduce-scatter.
    # EP16: all serving windows (FP32 reduce-scatter).
    # False: the unfused SparseCore gather/combine.
    prefill_fused_moe: bool = True
    prefill_moe_16k_monolithic: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_MOE_16K_MONOLITHIC
    )
    prefill_moe_dense_fp8: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_MOE_DENSE_FP8
    )
    prefill_moe_w4a8: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_MOE_W4A8
    )
    prefill_kda_fp8_gather: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_KDA_FP8_GATHER
    )
    prefill_kda_a2a: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_KDA_A2A
    )
    prefill_kda_dual_gather: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_KDA_DUAL_GATHER
    )
    prefill_kda_ring_set: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_KDA_RING_SET
    )
    prefill_kda_a2a_fp8: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_KDA_A2A_FP8
    )
    prefill_mla_a2a: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_MLA_A2A
    )
    prefill_mla_a2a_fp8: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_MLA_A2A_FP8
    )
    prefill_a2a_chip_shard: bool = dataclasses.field(
        default_factory=lambda: DEFAULT_PREFILL_A2A_CHIP_SHARD
    )
    # Weight loading.
    read_streams: int = 4
    load_inflight_gb: float = 16.0
    # Diagnostics (off in serving): dcp_skip and debug_skip skip named work, so
    # results are invalid (timing or hang bisection only); the rest add trace
    # scopes, VMEM reports or VMEM padding.
    dcp_skip: tuple[str, ...] = ()
    debug_skip: tuple[str, ...] = ()
    phase_scopes: bool = False
    # Hot-expert fp8 copies (off when empty): an npz whose ``experts``
    # [92, 8, S] holds the global expert id in copy slot j of EP group g
    # per MoE layer (-1: empty). A bare name loads from kimi_k3/data.
    kd_hot_plan: str = dataclasses.field(default_factory=lambda: DEFAULT_HOT_PLAN)
    # A hot expert's routes use its fp8 copy from this many rows.
    hot_min_rows: int = 9
    # Hot-expert MXFP4 replicas on other EP groups (off when empty; exclusive
    # with kd_hot_plan): an npz with ``experts`` [92, 8, S] (global expert id
    # replicated in copy slot j of EP group g, -1: empty) and ``order``
    # [92, 8 S] (the dispatch order of the slots). Per step, each replicated
    # expert's routes stay home, split by row parity, or move to the replica,
    # whichever gives the lowest modeled slowest-group expert time. A bare
    kd_replica_plan: str = dataclasses.field(
        default_factory=lambda: DEFAULT_REPLICA_PLAN
    )
    # FP8 wires for attention DP's hidden-row gathers (off when empty): "t0"
    # the attention input, "t10" the MoE input. Rows travel as e4m3 with one
    # FP32 scale each; every rank uses the dequantized rows.
    fp8_wires: tuple[str, ...] = dataclasses.field(
        default_factory=lambda: tuple(filter(None, DEFAULT_FP8_WIRES.split(",")))
    )
    pool_report: bool = False
    vmem_report: bool = False
    vmem_pad_kb: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kd_replica_plan", _normalize_plan_value(self.kd_replica_plan)
        )
        object.__setattr__(
            self,
            "prefill_replica_plan",
            _normalize_plan_value(self.prefill_replica_plan),
        )
        object.__setattr__(
            self,
            "prefill_expert_plan",
            _normalize_plan_value(self.prefill_expert_plan),
        )
        object.__setattr__(self, "kd_hot_plan", _normalize_plan_value(self.kd_hot_plan))
        if (
            self.decode_expert_parallel == 8
            and self.kd_replica_plan == EP16_DEFAULT_REPLICA_PLAN
        ):
            object.__setattr__(self, "kd_replica_plan", "")
        if self.kd_hot_plan and self.kd_replica_plan:
            raise ValueError("kd_hot_plan and kd_replica_plan are exclusive")
        if self.kd_hot_plan and self.prefill_replica_plan:
            raise ValueError("kd_hot_plan and prefill_replica_plan are exclusive")
        if self.prefill_replica_plan and self.prefill_expert_parallel != 16:
            raise ValueError("prefill_replica_plan requires prefill_expert_parallel=16")
        if self.prefill_replica_plan and not self.prefill_fused_moe:
            raise ValueError("prefill_replica_plan requires prefill_fused_moe=True")
        if (
            self.kd_replica_plan
            and self.prefill_replica_plan
            and self.kd_replica_plan != self.prefill_replica_plan
        ):
            raise ValueError(
                "kd_replica_plan and prefill_replica_plan must match when both are set"
            )
        if not set(self.fp8_wires) <= {"t0", "t10"}:
            raise ValueError("fp8_wires holds t0 and/or t10")
        if self.prefill_expert_parallel not in (8, 16, 32):
            raise ValueError("prefill_expert_parallel is 8, 16 or 32")
        if self.decode_expert_parallel not in (8, 16):
            raise ValueError("decode_expert_parallel is 8 or 16")
        if self.decode_expert_parallel == 16 and self.kd_hot_plan:
            raise ValueError("kd_hot_plan needs decode_expert_parallel 8")
        if self.decode_expert_banks and self.decode_expert_parallel != 16:
            raise ValueError("decode_expert_banks needs decode_expert_parallel 16")
        if self.prefill_kda_a2a_fp8 and not self.prefill_kda_a2a:
            raise ValueError("prefill_kda_a2a_fp8 requires prefill_kda_a2a=True")
        if self.prefill_mla_a2a_fp8 and not self.prefill_mla_a2a:
            raise ValueError("prefill_mla_a2a_fp8 requires prefill_mla_a2a=True")
        if self.prefill_a2a_chip_shard and not (
            self.prefill_kda_a2a or self.prefill_mla_a2a
        ):
            raise ValueError(
                "prefill_a2a_chip_shard requires prefill_kda_a2a=True "
                "or prefill_mla_a2a=True"
            )
        if self.prefill_a2a_chip_shard and (
            self.prefill_kda_a2a_fp8 or self.prefill_mla_a2a_fp8
        ):
            raise ValueError(
                "prefill_a2a_chip_shard is mutually exclusive with "
                "prefill_kda_a2a_fp8 and prefill_mla_a2a_fp8"
            )
        if self.prefill_moe_dense_fp8 and not self.prefill_fused_moe:
            raise ValueError("prefill_moe_dense_fp8 requires prefill_fused_moe=True")
        if self.prefill_moe_dense_fp8 and self.prefill_expert_parallel not in (8, 16):
            raise ValueError(
                "prefill_moe_dense_fp8 requires prefill_expert_parallel in (8, 16)"
            )
        if self.prefill_moe_w4a8 and not self.prefill_fused_moe:
            raise ValueError("prefill_moe_w4a8 requires prefill_fused_moe=True")
        if self.prefill_moe_w4a8 and not self.prefill_moe_16k_monolithic:
            raise ValueError(
                "prefill_moe_w4a8 requires prefill_moe_16k_monolithic=True"
            )
        if self.prefill_moe_w4a8 and self.prefill_expert_parallel != 16:
            raise ValueError("prefill_moe_w4a8 requires prefill_expert_parallel=16")
        if self.prefill_moe_w4a8 and self.prefill_moe_dense_fp8:
            raise ValueError(
                "prefill_moe_w4a8 is mutually exclusive with prefill_moe_dense_fp8"
            )
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
