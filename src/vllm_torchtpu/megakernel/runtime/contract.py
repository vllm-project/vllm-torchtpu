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
"""The megakernel model contract (framework-free, stdlib only).

A model folder exports `SPEC = TpuMonoSpec(...)`. Its declarative fields
(chips, TP sizes, parallelism, features, required server settings) are
enforced by the runtime before `create`; `create(view, options)` returns a
`MonoProgram` (one per TP rank) or a named `Decline`. Importing this module
imports no jax / torch / vLLM.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from ..errors import MegakernelConfigError, MegakernelLayoutError

API_VERSION = 1

# --- Features: framework features a spec declares it can serve under. A
# feature that is on but undeclared declines the megakernel at startup.
FEATURE_LORA = "lora"
FEATURE_MULTIMODAL = "multimodal"
FEATURE_AUX_HIDDEN_STATES = "aux_hidden_states"
FEATURE_SPEC_DECODE = "spec_decode"
FEATURE_MROPE = "mrope"
FEATURE_PREFIX_CACHING = "prefix_caching"
FEATURE_KV_OFFLOAD = "kv_offload"
FEATURE_KV_TRANSFER = "kv_transfer"
FEATURE_EPLB_DYNAMIC = "eplb_dynamic"
FEATURE_SLEEP_MODE = "sleep_mode"
FEATURES: frozenset[str] = frozenset(
    {
        FEATURE_LORA,
        FEATURE_MULTIMODAL,
        FEATURE_AUX_HIDDEN_STATES,
        FEATURE_SPEC_DECODE,
        FEATURE_MROPE,
        FEATURE_PREFIX_CACHING,
        FEATURE_KV_OFFLOAD,
        FEATURE_KV_TRANSFER,
        FEATURE_EPLB_DYNAMIC,
        FEATURE_SLEEP_MODE,
    }
)

# --- Parallelism kinds. A spec declares the kinds it serves under ("tp" is
# required); the runtime detects the others from the config.
PAR_TP = "tp"
PAR_EP_IN_TP = "ep_in_tp"
PAR_DP = "dp"
PAR_DP_LOCKSTEP = "dp_lockstep"
PAR_PP = "pp"
PAR_DCP = "dcp"
PAR_PCP = "pcp"
PAR_EP_WIDER_THAN_TP = "ep_wider_than_tp"
PARALLELISM: frozenset[str] = frozenset(
    {
        PAR_TP,
        PAR_EP_IN_TP,
        PAR_DP,
        PAR_DP_LOCKSTEP,
        PAR_PP,
        PAR_DCP,
        PAR_PCP,
        PAR_EP_WIDER_THAN_TP,
    }
)

# --- `requires`: server settings a spec can require, by key. The runtime
# maps each key to the vLLM config value it compares (as a lowercase string).
REQUIRES_KEYS: Mapping[str, str] = {
    "mamba_ssm_cache_dtype": "cache_config.mamba_ssm_cache_dtype",
    "kv_cache_dtype": "cache_config.cache_dtype",
    "block_size": "cache_config.block_size",
    "mamba_cache_mode": "cache_config.mamba_cache_mode",
    "enable_prefix_caching": "cache_config.enable_prefix_caching",
}

NUMERICS: frozenset[str] = frozenset({"exact", "approx"})

# --- Collective domains: the group of ranks a spec's kernel communicates
# across, which is the group the runtime's startup / bind agreements must
# span. Only "tp" (the TP group) is served today; the others decline at
# startup with `parallelism_unsupported:<domain>`.
DOMAIN_TP = "tp"
DOMAIN_TP_X_DCP = "tp_x_dcp"
DOMAIN_KERNEL_MESH = "kernel_mesh"
COLLECTIVE_DOMAINS: frozenset[str] = frozenset(
    {DOMAIN_TP, DOMAIN_TP_X_DCP, DOMAIN_KERNEL_MESH}
)

# --- Decline codes the runtime itself can produce.
UNSUPPORTED_TPU_CHIP = "unsupported_tpu_chip"
UNSUPPORTED_TP_SIZE = "unsupported_tp_size"
PARALLELISM_UNSUPPORTED = "parallelism_unsupported"  # + ":<kind>"
REQUIRES_UNMET = "requires_unmet"  # + ":<key>"
PLUGIN_MISMATCH = "plugin_mismatch"
CREATE_ERROR = "create_error"
BIND_ERROR = "bind_error"
DRAFT_DECLINED = "draft_declined"
BUCKET_SET_MISMATCH = "bucket_set_mismatch"
PEER_DECLINED = "peer_declined"  # + ":<code>"
CACHE_LAYOUT_INCOMPATIBLE = "cache_layout_incompatible"
POOL_LAYOUT_UNKNOWN = "pool_layout_unknown"
POOL_LAYOUT_MISMATCH = "pool_layout_mismatch"
CONV_STATE_DTYPE_UNSUPPORTED = "conv_state_dtype_unsupported"
MESH_SIZE_MISMATCH = "mesh_size_mismatch"
PALLAS_MESH_UNAVAILABLE = "pallas_mesh_unavailable"
PALLAS_MESH_MISMATCH = "pallas_mesh_mismatch"


@dataclasses.dataclass(frozen=True)
class EligibilityResult:
    """Structured eligibility result returned by startup and config gates."""

    eligible: bool
    reason: str = "ok"
    message: str = ""
    level: str = "warning"

    @property
    def code(self) -> str:
        return self.reason


@dataclasses.dataclass(frozen=True)
class StepDecision:
    """Per-step routing decision (a program's `decide`)."""

    eligible: bool
    reason: str
    num_reqs: int = 0
    tokens_per_seq: int = 0
    bucket_t: int = 0


@dataclasses.dataclass(frozen=True)
class Decline:
    """A named, expected reason not to serve (`create`'s other result)."""

    code: str
    message: str = ""
    level: str = "warning"

    @property
    def result(self) -> EligibilityResult:
        """This decline as an `EligibilityResult`."""
        return EligibilityResult(False, self.code, self.message, self.level)


class LayoutDecline(MegakernelLayoutError):
    """The runner's cache layout isn't the one the megakernel was built for
    (raised by a program's `bind`). Carries a named `EligibilityResult`; the
    runtime makes it rank-uniform."""

    def __init__(self, result: EligibilityResult) -> None:
        super().__init__(f"Megakernel: {result.message}")
        self.result = result


@dataclasses.dataclass(frozen=True)
class Bucket:
    """One compiled step shape: `T` tokens per sequence, `B` rows.

    `out_rows=None` lets the runtime pick the runner's padded token count.
    """

    T: int
    B: int
    out_rows: int | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (int(self.T), int(self.B))


@dataclasses.dataclass(frozen=True)
class RunnerShape:
    """One runner input shape a bucket is warmed up with."""

    tok_pad: int
    req_pad: int
    blk_pad: int
    ckpt_cols: int


@dataclasses.dataclass(frozen=True)
class WarmupCase:
    """Positional args for a bucket's `StepFn`, and an optional post-call check."""

    args: tuple[Any, ...]
    check: Callable[[Any], None] | None = None


StepFn = Callable[..., Any]


class MonoProgram(Protocol):
    """One megakernel instance on one TP rank (returned by `create`).

    `view` is the runtime's read-only `RunnerView`; `fwd` its per-step
    `ForwardView`. Optional methods (looked up with `getattr`):
    `warmup_session()` (a context manager around warmup),
    `check_compiled_ir(debug, bucket)`, `attach_draft(drafter)` /
    `detach_draft()`, `on_warmup_done()` (called once per bind from the
    runtime's H2 `on_warmup_done(runner)`, after the worker's whole warmup).
    `decide` may return `None` for a step it doesn't see (no metadata): stock,
    not counted in the step mix.
    """

    display_name: str

    def buckets(self, token_paddings: tuple[int, ...]) -> Sequence[Bucket]: ...

    def load(self, mesh: Any) -> None: ...

    def bind(self, view: Any) -> Any: ...

    def build(self, bucket: Bucket) -> StepFn: ...

    def warmup_inputs(
        self, bucket: Bucket, shape: RunnerShape
    ) -> WarmupCase | None: ...

    def decide(self, ctx: Any, fwd: Any) -> StepDecision | None: ...

    def forward(self, decision: StepDecision | None, ctx: Any, fwd: Any) -> Any: ...


def _frozen_strs(values: Any) -> frozenset[str]:
    return frozenset(str(v) for v in values)


@dataclasses.dataclass(frozen=True)
class TpuMonoSpec:
    """What a megakernel model folder exports as `SPEC`.

    `create` is a callable or a `"module:attr"` string (`".adapter:create"` is
    relative to the folder's package), imported lazily, so importing the
    folder imports nothing heavy. `create(view, options)` returns a program or
    a `Decline`; it runs no collective and loads no weights.

    The runtime enforces, before `create` (config and local device facts
    only, no collective): `chips` (device_kind prefixes; `()` = any),
    `tp_sizes` (`()` = any), `parallelism` (kinds served; "tp" is required),
    `features` (framework features served) and `requires` (server settings,
    keys from `REQUIRES_KEYS`). `numerics` ("exact" | "approx", of the
    default profile) is logged and part of the fingerprint, not a gate.
    `collective_domain` names the ranks the kernel's collectives span (see
    `COLLECTIVE_DOMAINS`); the runtime's agreements run over that group.
    """

    name: str
    version: str
    create: str | Callable[..., Any]
    chips: tuple[str, ...] = ()
    tp_sizes: tuple[int, ...] = ()
    parallelism: frozenset[str] = frozenset({PAR_TP})
    features: frozenset[str] = frozenset()
    requires: Mapping[str, str] = dataclasses.field(default_factory=dict)
    numerics: str = "exact"
    collective_domain: str = DOMAIN_TP
    layout_request: str | Callable[..., Any] | None = None
    decline_codes: tuple[str, ...] = ()
    source_root: str | None = None
    api_version: int = API_VERSION

    def __post_init__(self) -> None:
        def fix(name: str, value: Any) -> None:
            object.__setattr__(self, name, value)

        if not isinstance(self.name, str) or not self.name:
            raise MegakernelConfigError("TpuMonoSpec.name must be non-empty")
        fix("chips", tuple(str(c) for c in self.chips))
        fix("tp_sizes", tuple(int(t) for t in self.tp_sizes))
        fix("parallelism", _frozen_strs(self.parallelism))
        fix("features", _frozen_strs(self.features))
        fix("requires", {str(k): str(v) for k, v in dict(self.requires).items()})
        fix("decline_codes", tuple(self.decline_codes))
        where = f"TpuMonoSpec {self.name!r}"
        if unknown := sorted(self.features - FEATURES):
            raise MegakernelConfigError(
                f"{where}: unknown features {unknown}; known: {sorted(FEATURES)}"
            )
        if unknown := sorted(self.parallelism - PARALLELISM):
            raise MegakernelConfigError(
                f"{where}: unknown parallelism {unknown}; known: {sorted(PARALLELISM)}"
            )
        if PAR_TP not in self.parallelism:
            raise MegakernelConfigError(f"{where}: parallelism must include 'tp'")
        if unknown := sorted(set(self.requires) - set(REQUIRES_KEYS)):
            raise MegakernelConfigError(
                f"{where}: unknown requires key {unknown[0]!r}; "
                f"known: {sorted(REQUIRES_KEYS)}"
            )
        if self.numerics not in NUMERICS:
            raise MegakernelConfigError(
                f"{where}: numerics={self.numerics!r}; expected one of "
                f"{sorted(NUMERICS)}"
            )
        if self.collective_domain not in COLLECTIVE_DOMAINS:
            raise MegakernelConfigError(
                f"{where}: collective_domain={self.collective_domain!r}; "
                f"expected one of {sorted(COLLECTIVE_DOMAINS)}"
            )


__all__ = [
    "API_VERSION",
    "COLLECTIVE_DOMAINS",
    "FEATURES",
    "NUMERICS",
    "PARALLELISM",
    "REQUIRES_KEYS",
    "Bucket",
    "Decline",
    "EligibilityResult",
    "LayoutDecline",
    "MonoProgram",
    "RunnerShape",
    "StepDecision",
    "StepFn",
    "TpuMonoSpec",
    "WarmupCase",
]
