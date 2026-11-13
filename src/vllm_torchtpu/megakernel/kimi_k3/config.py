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


def _host_side(default: Any) -> Any:
    """A field the host-side code reads and no compiled kernel depends on."""
    return dataclasses.field(default=default, metadata={"host_side": True})


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
    # Decode routed experts: 8 (EP8/TP4, the pre-shard layout) or 16 (EP16/TP2,
    # the prefill's families; each expert runs as two TP4-width halves).
    decode_expert_parallel: int = 8
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
    # keys stay BF16.
    prefill_prefix_fp8: bool = True
    # Prefill MoE route counts (diagnostic): each fused MoE call also returns its
    # chip's 56 local expert counts; the prefiller saves cumulative snapshots
    # every prefill_route_counts_every windows under prefill_route_counts_dir
    # (default: $CDK_OUTPUT_DIR/prefill_route_counts).
    prefill_route_counts: bool = False
    prefill_route_counts_every: int = _host_side(200)
    prefill_route_counts_dir: str = _host_side("")
    # EP16 expert placement, shared by the prefill and the aggregate's decode
    # (empty: the pre-shard's): an npz whose ``order`` [92, 896] is the logical
    # expert at each physical slot (chip c: slots 56c..56c+55, hottest first);
    # the router columns and expert weights are loaded in that order
    # (rank_loader.expert_plan). A bare name loads from kimi_k3/data; the
    # default is built by tools/megakernel/expert_plan.py from the route counts
    # of a served random-text benchmark (every context depth) and of prefix-free
    # random-token, id-sweep and real-text prefills.
    expert_plan: str = "plan_ep16_real_hotfirst.npz"
    # Weight loading.
    read_streams: int = _host_side(4)
    load_inflight_gb: float = _host_side(16.0)
    # Named XProf trace scopes around the decode's phases (profiling only).
    phase_scopes: bool = False

    def __post_init__(self) -> None:
        if self.decode_expert_parallel not in (8, 16):
            raise ValueError("decode_expert_parallel is 8 or 16")
        if self.mla_decode_ktile not in (1024, 2048):
            raise ValueError("mla_decode_ktile supports 1024 or experimental 2048")
        if self.mla_decode_ktile == 2048 and (
            self.attention_dp != 2 or self.dcp_group != 16
        ):
            raise ValueError(
                "mla_decode_ktile=2048 requires attention_dp=2 and dcp_group=16"
            )

    def key(self) -> str:
        """A stable digest of the fields that shape the compiled kernels, for
        their caches (host-side fields, which may differ per host, are left
        out)."""
        shaping = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if not field.metadata.get("host_side")
        }
        blob = json.dumps(shaping, sort_keys=True, default=list)
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
