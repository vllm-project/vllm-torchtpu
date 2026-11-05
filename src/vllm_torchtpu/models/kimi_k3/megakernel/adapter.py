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
"""Kimi K3 MonoProgram adapter and pre-quantization hooks.

Limitations:
- On TPU, `create()` validates topology and weights and returns
  `Decline("runner_lifecycle_unwired", ...)` until runner step-execution,
  output-sampling, and single-copy weight-ownership hooks land.
- `StepContext` and `ForwardView` carry no request IDs, so per-step `SlotMap`
  slot admission is not driven from `decide()`; callers use `admit_requests()`
  directly until the runner step-execution hook passes `StepInput`.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Iterable, Mapping, Sequence

import jax
import numpy as np
import torch

from vllm_torchtpu.megakernel.runtime.contract import EligibilityResult
from vllm_torchtpu.megakernel.runtime.eligibility import check_step_eligibility
from vllm_torchtpu.megakernel.runtime.policy import BucketPolicy
from vllm_torchtpu.megakernel_runtime.plugin import (
    Bucket,
    Decline,
    ForwardView,
    LayoutDecline,
    RunnerShape,
    RunnerView,
    StepDecision,
    StepFn,
    WarmupCase,
)

from .kernels.load import pack_expert_bytes
from .layout import (
    FP8_LAYOUT,
    LAYOUT,
    NUM_LAYERS,
)
from .pools import (
    KDA_LAYERS,
    MLA_LAYERS,
    SlotMap,
    is_mla_layer,
    layer_index,
)
from .program import (
    DRAFT_TOKENS,
    HIDDEN_SIZE,
    TP_SIZE,
)

KIMI_K3_NUM_EXPERTS = 896
KIMI_K3_EP_GROUPS = 8
KIMI_K3_ETP_PARTS = 4
SUPPORTED_ARCHS = frozenset({"KimiK3ForConditionalGeneration"})
SUPPORTED_CPU_TP_SIZES = (1, 2, TP_SIZE)
SUPPORTED_ATTENTION_DP_SIZES = (1, 2, 4)
DEFAULT_DECODE_BUCKETS = (1, 2, 4, 8, 16)
VERIFY_TOKEN_COUNT = DRAFT_TOKENS + 1
DEFAULT_VERIFY_BUCKETS = (8, 16)
PACKED_MATRIX_NDIM = 2
STACKED_EXPERT_NDIM = 3
U32_PACK_FACTOR = 4
MIN_OUT_ROW_ALIGNMENT = 16
DEFAULT_MAX_SLOTS = 64


@dataclasses.dataclass
class ObservedMxfp4Entry:
    """Captured pre-quantization MXFP4 packed tensor or E8M0 scale tensor."""

    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    packed_u32: np.ndarray | None = None
    raw_u8: np.ndarray | None = None


class ObservedWeightStore:
    """Thread-safe store populated by `load_tensor` during weight streaming."""

    def __init__(self) -> None:
        """Initialize an empty thread-safe weight observation store."""
        self._lock = threading.Lock()
        self._entries: dict[str, ObservedMxfp4Entry] = {}

    def record(self, name: str, entry: ObservedMxfp4Entry) -> None:
        """Store an observed weight or scale entry by parameter name."""
        with self._lock:
            self._entries[name] = entry

    def snapshot(self) -> dict[str, ObservedMxfp4Entry]:
        """Return a shallow copy of all recorded weight entries."""
        with self._lock:
            return dict(self._entries)

    def clear(self) -> None:
        """Remove all recorded entries from the store."""
        with self._lock:
            self._entries.clear()


WEIGHT_STORE = ObservedWeightStore()


def observed_weights_snapshot() -> dict[str, ObservedMxfp4Entry]:
    """Return a shallow snapshot of MXFP4 / E8M0 tensors observed at load time."""
    return WEIGHT_STORE.snapshot()


def clear_observed_weights() -> None:
    """Clear the module-local MXFP4 weight observation store."""
    WEIGHT_STORE.clear()


def _is_fp8_cache_dtype(dtype_str: str) -> bool:
    """Return True when the configured KV cache dtype uses FP8 storage."""
    low = (dtype_str or "auto").lower()
    return "fp8" in low or "float8" in low


def _lookup_mapping_or_attr(container: object, field: str) -> object:
    """Read a field from either a mapping or an attribute container."""
    if isinstance(container, Mapping):
        return container.get(field)
    return getattr(container, field, None)


def _extract_cfg_field(hf_cfg: object, field: str, default: object = None) -> object:
    """Extract a configuration field from `hf_cfg` or nested `text_config`."""
    if hf_cfg is None:
        return default
    direct = _lookup_mapping_or_attr(hf_cfg, field)
    if direct is not None:
        return direct
    text_cfg = _lookup_mapping_or_attr(hf_cfg, "text_config")
    if text_cfg is not None:
        nested = _lookup_mapping_or_attr(text_cfg, field)
        if nested is not None:
            return nested
    return default


def _as_int(val: object, default: int = 0) -> int:
    """Convert a dynamic configuration value to `int`."""
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, (float, str)):
        return int(val)
    return default


def layout_request(
    vllm_config: object, options: Mapping[str, object] | None = None
) -> dict[str, int]:
    """Return unified MLA + KDA pool `block_size` and `dcp_interleave_size`."""
    del options
    cache_cfg = getattr(vllm_config, "cache_config", None)
    raw_dtype = getattr(cache_cfg, "cache_dtype", "auto")
    cache_dtype = raw_dtype if isinstance(raw_dtype, str) and raw_dtype else "auto"
    block = (
        FP8_LAYOUT.tokens_per_block
        if _is_fp8_cache_dtype(cache_dtype)
        else LAYOUT.tokens_per_block
    )
    return {
        "block_size": block,
        "dcp_interleave_size": block,
    }


def _default_kernel_position() -> int:
    """Resolve local worker position in ascending `device.id` order (matching `pools.tp_rank_permutation`)."""
    devices = sorted(jax.devices(), key=lambda d: int(d.id))
    if len(devices) != TP_SIZE:
        msg = (
            f"Kimi K3 weight loading requires {TP_SIZE} global JAX devices "
            f"or explicit kernel_rank in options, got {len(devices)} devices"
        )
        raise ValueError(msg)
    local_devs = jax.local_devices()
    if not local_devs:
        msg = "Kimi K3 weight loading found no local JAX devices"
        raise ValueError(msg)
    ids = [int(d.id) for d in devices]
    dev_id = int(local_devs[0].id)
    return ids.index(dev_id)


def _resolve_kernel_position(opts: Mapping[str, object]) -> int:
    """Return ascending-`device.id` rank position `p` in `0..31`."""
    raw_pos = opts.get("kernel_rank", opts.get("device_position"))
    if raw_pos is None:
        pos = _default_kernel_position()
    elif isinstance(raw_pos, bool) or not isinstance(raw_pos, (int, str)):
        msg = f"kernel_rank must be an integer, got {raw_pos!r}"
        raise ValueError(msg)
    else:
        pos = int(raw_pos)
    if not 0 <= pos < TP_SIZE:
        msg = f"kernel_rank position must be in 0..{TP_SIZE - 1}, got {pos}"
        raise ValueError(msg)
    return pos


def _extract_expert_index(name: str) -> int | None:
    """Extract per-expert integer index from parameter `name` when present."""
    parts = name.split(".")
    for idx, seg in enumerate(parts[:-1]):
        if seg == "experts" and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    return None


def _slice_2d_etp(
    mat: np.ndarray, *, name: str, part: int, is_down: bool
) -> np.ndarray:
    """Slice 1/4 ETP shard along columns (`down`) or rows (`gate`/`up`)."""
    axis = 1 if is_down else 0
    dim = mat.shape[axis]
    if dim % KIMI_K3_ETP_PARTS != 0:
        msg = (
            f"expert tensor {name!r} with shape {mat.shape} is not divisible "
            f"by ETP={KIMI_K3_ETP_PARTS} along axis {axis}"
        )
        raise ValueError(msg)
    width = dim // KIMI_K3_ETP_PARTS
    if is_down:
        return np.ascontiguousarray(mat[:, part * width : (part + 1) * width])
    return np.ascontiguousarray(mat[part * width : (part + 1) * width, :])


def _slice_expert_shard(
    arr: np.ndarray,
    *,
    name: str,
    position: int,
    num_experts: int = KIMI_K3_NUM_EXPERTS,
) -> np.ndarray | None:
    """Slice `1/32` (`EP8 x ETP4`) shard for ascending-`device.id` position."""
    ep_group = position // KIMI_K3_ETP_PARTS
    part = position % KIMI_K3_ETP_PARTS
    low = name.lower()
    is_down = "w2" in low or "down" in low
    expert_idx = _extract_expert_index(name)
    if expert_idx is not None:
        if num_experts % KIMI_K3_EP_GROUPS != 0:
            msg = (
                f"expert tensor {name!r} num_experts={num_experts} is not "
                f"divisible by EP={KIMI_K3_EP_GROUPS}"
            )
            raise ValueError(msg)
        ep_count = num_experts // KIMI_K3_EP_GROUPS
        first_expert = ep_count * ep_group
        if not first_expert <= expert_idx < first_expert + ep_count:
            return None
        return _slice_2d_etp(arr, name=name, part=part, is_down=is_down)
    if arr.ndim == STACKED_EXPERT_NDIM:
        if arr.shape[0] % KIMI_K3_EP_GROUPS != 0:
            msg = (
                f"expert tensor {name!r} with shape {arr.shape} is not divisible "
                f"by EP={KIMI_K3_EP_GROUPS} along axis 0"
            )
            raise ValueError(msg)
        e_count = arr.shape[0] // KIMI_K3_EP_GROUPS
        e_first = e_count * ep_group
        sub = arr[e_first : e_first + e_count]
        etp_axis = 2 if is_down else 1
        if sub.shape[etp_axis] % KIMI_K3_ETP_PARTS != 0:
            msg = (
                f"expert tensor {name!r} with shape {arr.shape} is not divisible "
                f"by ETP={KIMI_K3_ETP_PARTS} along axis {etp_axis}"
            )
            raise ValueError(msg)
        width = sub.shape[etp_axis] // KIMI_K3_ETP_PARTS
        if is_down:
            sub = sub[:, :, part * width : (part + 1) * width]
        else:
            sub = sub[:, part * width : (part + 1) * width, :]
        return np.ascontiguousarray(
            sub.reshape(sub.shape[0] * sub.shape[1], sub.shape[2])
        )
    if arr.ndim == PACKED_MATRIX_NDIM:
        if arr.shape[0] % KIMI_K3_EP_GROUPS != 0:
            msg = (
                f"expert tensor {name!r} with shape {arr.shape} is not divisible "
                f"by EP={KIMI_K3_EP_GROUPS} along axis 0"
            )
            raise ValueError(msg)
        ep_rows = arr.shape[0] // KIMI_K3_EP_GROUPS
        r_first = ep_rows * ep_group
        ep_slice = arr[r_first : r_first + ep_rows, :]
        return _slice_2d_etp(ep_slice, name=name, part=part, is_down=is_down)
    msg = f"expert tensor {name!r} with shape {arr.shape} must be 2D or 3D"
    raise ValueError(msg)


def load_tensor(
    name: str, tensor: torch.Tensor, options: Mapping[str, object] | None = None
) -> None:
    """Capture `1/32` rank-sliced routed-expert MXFP4 bytes and E8M0 scales before delattr."""
    opts = dict(options) if isinstance(options, Mapping) else {}
    if not opts.get("load_tensors", False):
        return
    low = name.lower()
    if "shared_experts" in low:
        return
    is_scale = any(
        tok in low for tok in ("weight_scale", "scale_inv", "e8m0", ".scales")
    )
    is_packed = any(
        tok in low
        for tok in ("weight_packed", "blocks", "mxfp4", ".experts.", "w13", "w2")
    )
    if (
        not (is_scale or is_packed)
        or tensor is None
        or not isinstance(tensor, torch.Tensor)
    ):
        return

    position = _resolve_kernel_position(opts)
    num_experts = _as_int(
        opts.get("num_experts", KIMI_K3_NUM_EXPERTS),
        default=KIMI_K3_NUM_EXPERTS,
    )
    t_cpu = tensor.detach().cpu()
    dtype_name = str(t_cpu.dtype).replace("torch.", "")
    raw_shape = tuple(t_cpu.shape)
    if t_cpu.dtype == torch.uint8 and t_cpu.ndim in (
        PACKED_MATRIX_NDIM,
        STACKED_EXPERT_NDIM,
    ):
        shard = _slice_expert_shard(
            t_cpu.numpy(),
            name=name,
            position=position,
            num_experts=num_experts,
        )
        del t_cpu
        if shard is None:
            return
        if is_scale:
            scale_t = np.ascontiguousarray(shard.T).copy()
            del shard
            WEIGHT_STORE.record(
                name,
                ObservedMxfp4Entry(
                    name=name,
                    kind="e8m0_scale",
                    shape=tuple(scale_t.shape),
                    dtype=dtype_name,
                    raw_u8=scale_t,
                ),
            )
            return
        if shard.shape[1] >= U32_PACK_FACTOR and shard.shape[1] % U32_PACK_FACTOR == 0:
            packed = pack_expert_bytes(shard)
            del shard
            WEIGHT_STORE.record(
                name,
                ObservedMxfp4Entry(
                    name=name,
                    kind="mxfp4_packed",
                    shape=tuple(packed.shape),
                    dtype=dtype_name,
                    packed_u32=packed,
                ),
            )
            return
        shard_shape = tuple(shard.shape)
        del shard
        WEIGHT_STORE.record(
            name,
            ObservedMxfp4Entry(
                name=name,
                kind="weight_meta",
                shape=shard_shape,
                dtype=dtype_name,
            ),
        )
        return

    del t_cpu
    expert_idx = _extract_expert_index(name)
    if expert_idx is not None:
        ep_count = max(1, num_experts // KIMI_K3_EP_GROUPS)
        first_expert = ep_count * (position // KIMI_K3_ETP_PARTS)
        if not first_expert <= expert_idx < first_expert + ep_count:
            return
    WEIGHT_STORE.record(
        name,
        ObservedMxfp4Entry(
            name=name,
            kind="scale_meta" if is_scale else "weight_meta",
            shape=raw_shape,
            dtype=dtype_name,
        ),
    )


def _parse_strict_int(item: object, field_name: str) -> int:
    """Parse an integer from `int` or decimal `str`, raising `ValueError` otherwise."""
    if isinstance(item, bool) or not isinstance(item, (int, str)):
        msg = f"invalid {field_name} entry {item!r}: expected integer"
        raise ValueError(msg)
    try:
        return int(item)
    except ValueError as exc:
        msg = f"invalid {field_name} entry {item!r}: expected integer"
        raise ValueError(msg) from exc


def _parse_decode_buckets(raw_decode: object) -> tuple[int, ...]:
    """Parse and validate positive decode bucket sizes."""
    if raw_decode is None:
        return DEFAULT_DECODE_BUCKETS
    if raw_decode in ((), [], "none", False):
        return ()
    if isinstance(raw_decode, str):
        items: list[object] = [p.strip() for p in raw_decode.split(",") if p.strip()]
    elif isinstance(raw_decode, Iterable) and not isinstance(raw_decode, bytes):
        items = list(raw_decode)
    else:
        msg = (
            "buckets must be a comma-separated string or iterable of positive "
            f"integers, got {raw_decode!r}"
        )
        raise ValueError(msg)
    parsed: list[int] = []
    for item in items:
        val = _parse_strict_int(item, "buckets")
        if val <= 0:
            msg = f"invalid buckets entry {val}: must be > 0"
            raise ValueError(msg)
        parsed.append(val)
    return tuple(sorted(set(parsed)))


def _parse_verify_buckets(raw_verify: object) -> tuple[int, ...]:
    """Parse and validate Kimi K3 verify bucket sizes (multiples of `VERIFY_TOKEN_COUNT`)."""
    if raw_verify is None:
        return DEFAULT_VERIFY_BUCKETS
    if raw_verify in ((), [], "none", False):
        return ()
    if isinstance(raw_verify, str):
        items: list[object] = [p.strip() for p in raw_verify.split(",") if p.strip()]
    elif isinstance(raw_verify, Iterable) and not isinstance(raw_verify, bytes):
        items = list(raw_verify)
    else:
        msg = (
            "verify_buckets must be a comma-separated string or iterable of "
            f"integers, got {raw_verify!r}"
        )
        raise ValueError(msg)
    parsed: list[int] = []
    for item in items:
        val = _parse_strict_int(item, "verify_buckets")
        if val < VERIFY_TOKEN_COUNT or val % VERIFY_TOKEN_COUNT != 0:
            msg = (
                f"invalid verify_buckets entry {val}: Kimi K3 verify buckets "
                f"must be positive multiples of VERIFY_TOKEN_COUNT={VERIFY_TOKEN_COUNT}"
            )
            raise ValueError(msg)
        parsed.append(val)
    return tuple(sorted(set(parsed)))


def _validate_hf_config(
    hf_cfg: object,
) -> tuple[int | None, int | None, Decline | None]:
    """Validate Kimi K3 architecture and dimensions from HuggingFace config."""
    if hf_cfg is None:
        return None, None, None
    archs = _extract_cfg_field(hf_cfg, "architectures", None)
    model_type = _extract_cfg_field(hf_cfg, "model_type", None)
    arch_seq = (
        archs if isinstance(archs, Iterable) and not isinstance(archs, str) else ()
    )
    arch_ok = any(a in SUPPORTED_ARCHS for a in arch_seq)
    type_ok = model_type == "kimi_k3"
    if (archs is not None or model_type is not None) and not (arch_ok or type_ok):
        return (
            None,
            None,
            Decline(
                "unsupported_arch",
                f"Kimi K3 adapter does not support architectures={archs!r} "
                f"model_type={model_type!r}",
            ),
        )
    raw_layers = _extract_cfg_field(hf_cfg, "num_hidden_layers", None)
    raw_hidden = _extract_cfg_field(hf_cfg, "hidden_size", None)
    n_layers = _as_int(raw_layers) if raw_layers is not None else None
    hidden = _as_int(raw_hidden) if raw_hidden is not None else None
    if n_layers is not None and n_layers != NUM_LAYERS:
        return (
            n_layers,
            hidden,
            Decline(
                "invalid_geometry",
                f"expected {NUM_LAYERS} layers, got {n_layers}",
            ),
        )
    if hidden is not None and hidden != HIDDEN_SIZE:
        return (
            n_layers,
            hidden,
            Decline(
                "invalid_geometry",
                f"expected hidden_size={HIDDEN_SIZE}, got {hidden}",
            ),
        )
    return n_layers, hidden, None


def validate_tpu_topology(
    view: RunnerView,
    opts: Mapping[str, object],
    tp_size: int,
    attention_dp: int,
    geometry: tuple[int | None, int | None],
    observed_weights: Mapping[str, ObservedMxfp4Entry],
) -> Decline | None:
    """Validate TPU-specific Kimi K3 parallelism, weights, and decoder state."""
    n_layers, hidden = geometry
    par_cfg = getattr(view.vllm_config, "parallel_config", None)
    dcp_raw = getattr(par_cfg, "decode_context_parallel_size", None)
    dcp_size = _as_int(dcp_raw) if dcp_raw is not None else None
    expected_dcp = TP_SIZE // max(1, attention_dp)
    checks: tuple[tuple[bool, str, str], ...] = (
        (
            n_layers is None or hidden is None,
            "invalid_geometry",
            "missing num_hidden_layers or hidden_size on TPU",
        ),
        (
            tp_size != TP_SIZE,
            "unsupported_tp_size",
            f"Kimi K3 TPU megakernel requires tp_size == {TP_SIZE}, got {tp_size}",
        ),
        (
            attention_dp not in SUPPORTED_ATTENTION_DP_SIZES,
            "unsupported_attention_dp",
            "Kimi K3 attention_dp on TPU must be in "
            f"{SUPPORTED_ATTENTION_DP_SIZES}, got {attention_dp}",
        ),
        (
            dcp_size is not None and dcp_size > 1 and dcp_size != expected_dcp,
            "unsupported_dcp_size",
            f"expected decode_context_parallel_size == {expected_dcp}, got {dcp_size}",
        ),
        (
            view.model is None and not opts.get("presharded"),
            "backbone_not_found",
            "Kimi K3 TPU megakernel requires a loaded model",
        ),
        (
            bool(opts.get("require_observed_mxfp4", False)) and not observed_weights,
            "missing_mxfp4_weights",
            "no MXFP4 weights captured by load_tensor prior to create()",
        ),
    )
    for failed, code, msg in checks:
        if failed:
            return Decline(code, msg)
    return Decline(
        "runner_lifecycle_unwired",
        "Kimi K3 TPU megakernel requires runner step-execution, output-sampling, "
        "and single-copy weight-ownership hooks that are not wired yet",
    )


def _resolve_buckets(
    opts: Mapping[str, object],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve sorted decode and verify bucket tuples from plugin options."""
    decode_buckets = _parse_decode_buckets(opts.get("buckets"))
    verify_buckets = _parse_verify_buckets(
        opts.get("verify_buckets", DEFAULT_VERIFY_BUCKETS)
    )
    return decode_buckets, verify_buckets


@dataclasses.dataclass(frozen=True, slots=True)
class KimiK3ProgramConfig:
    """Resolved configuration bundle for `KimiK3MonoProgram`."""

    tp_rank: int
    tp_size: int
    attention_dp: int
    hidden_size: int
    decode_buckets: tuple[int, ...]
    verify_buckets: tuple[int, ...]
    max_slots: int
    is_tpu: bool
    options: Mapping[str, object]
    view: RunnerView | None = None
    observed_weights: Mapping[str, ObservedMxfp4Entry] = dataclasses.field(
        default_factory=dict
    )


def create(
    view: RunnerView, options: Mapping[str, object]
) -> KimiK3MonoProgram | Decline:
    """Validate Kimi K3 topology, snapshot observed weights, and clear store."""
    observed = observed_weights_snapshot()
    try:
        opts = dict(options or {})
        hf_cfg = view.hf_config
        n_layers, hidden, hf_decline = _validate_hf_config(hf_cfg)
        if hf_decline is not None:
            return hf_decline

        raw_dev_type = getattr(view.device, "type", "cpu")
        dev_type = (raw_dev_type if isinstance(raw_dev_type, str) else "cpu").lower()
        is_tpu = dev_type == "tpu"
        tp_size = view.tp_size
        par_cfg = getattr(view.vllm_config, "parallel_config", None)
        default_dp = _as_int(getattr(par_cfg, "data_parallel_size", 1), default=1)
        attention_dp = _as_int(opts.get("attention_dp", default_dp), default=default_dp)

        if is_tpu:
            tpu_decline = validate_tpu_topology(
                view,
                opts,
                tp_size,
                attention_dp,
                (n_layers, hidden),
                observed,
            )
            if tpu_decline is not None:
                return tpu_decline
        elif tp_size not in SUPPORTED_CPU_TP_SIZES:
            return Decline(
                "unsupported_tp_size",
                f"CPU test runner tp_size={tp_size} outside {SUPPORTED_CPU_TP_SIZES}",
            )

        decode_buckets, verify_buckets = _resolve_buckets(opts)
        if not decode_buckets and not verify_buckets:
            return Decline("empty_buckets", "no decode or verify buckets configured")

        default_hidden = (
            _as_int(
                _extract_cfg_field(hf_cfg, "hidden_size", HIDDEN_SIZE),
                default=HIDDEN_SIZE,
            )
            if hf_cfg is not None
            else HIDDEN_SIZE
        )
        hidden_size = _as_int(
            opts.get("synthetic_hidden_size", default_hidden), default=default_hidden
        )
        max_slots = _as_int(
            opts.get("max_slots", DEFAULT_MAX_SLOTS), default=DEFAULT_MAX_SLOTS
        )

        return KimiK3MonoProgram(
            KimiK3ProgramConfig(
                tp_rank=view.tp_rank,
                tp_size=tp_size,
                attention_dp=max(1, attention_dp),
                hidden_size=hidden_size,
                decode_buckets=decode_buckets,
                verify_buckets=verify_buckets,
                max_slots=max_slots,
                is_tpu=is_tpu,
                options=opts,
                view=view,
                observed_weights=observed,
            )
        )
    finally:
        clear_observed_weights()


class KimiK3MonoProgram:
    """MonoProgram wrapping Kimi K3 decoder stack, unified pools, and SlotMap."""

    def __init__(self, config: KimiK3ProgramConfig) -> None:
        """Initialize the Kimi K3 MonoProgram from a validated configuration."""
        self.tp_rank = config.tp_rank
        self.tp_size = config.tp_size
        self.attention_dp = config.attention_dp
        self.hidden_size = config.hidden_size
        self.decode_buckets = config.decode_buckets
        self.verify_buckets = config.verify_buckets
        self.is_tpu = config.is_tpu
        self.options = dict(config.options)
        self.slot_map = SlotMap(num_slots=config.max_slots, groups=config.attention_dp)
        self._group: dict[str, tuple[int, int]] = {}
        self.warmed: list[tuple[int, int]] = []
        self.mesh: object = None
        self.observed_weights: dict[str, ObservedMxfp4Entry] = dict(
            config.observed_weights
        )
        self.decoder: object = None
        self.bound_pools: object = None
        self.bucket_out_rows: dict[tuple[int, int], int] = {}
        prof_ns = dataclasses.make_dataclass(
            "KimiBucketProfile",
            [("buckets", tuple[int, ...]), ("verify_buckets", tuple[int, ...])],
        )(buckets=config.decode_buckets, verify_buckets=config.verify_buckets)
        spec_toks = _as_int(getattr(config.view, "speculative_tokens", 0), default=0)
        verify_toks = (
            (VERIFY_TOKEN_COUNT,)
            if spec_toks == DRAFT_TOKENS
            or bool(self.options.get("enable_verify_bucket", False))
            else ()
        )
        raw_paddings = (
            (getattr(config.view, "token_paddings", ()) or ())
            if config.view is not None
            else ()
        )
        token_paddings = tuple(
            sorted(_as_int(p) for p in raw_paddings if _as_int(p) > 0)
        )
        self.policy = BucketPolicy.resolve(
            prof_ns, verify_tokens=verify_toks, token_paddings=token_paddings
        )

    def _group_of(self, req_id: str, context_len: int = 0) -> int:
        """Select or look up the attention-DP group (`0 .. attention_dp - 1`) for `req_id`."""
        if self.attention_dp == 1:
            return 0
        if req_id in self._group:
            g, _ = self._group[req_id]
            self._group[req_id] = (g, context_len)
            return g
        count = [0] * self.attention_dp
        context = [0] * self.attention_dp
        for g, c in self._group.values():
            count[g] += 1
            context[g] += c
        g = min(range(self.attention_dp), key=lambda k: (count[k], context[k]))
        self._group[req_id] = (g, context_len)
        return g

    def admit_requests(
        self,
        active_req_ids: Sequence[str],
        finished_req_ids: Iterable[str] = (),
        *,
        group: int | None = None,
        context_lens: Mapping[str, int] | None = None,
    ) -> list[int]:
        """Release finished requests and allocate attention-DP-grouped SlotMap slots."""
        fin_list = [str(req_id) for req_id in finished_req_ids]
        self.slot_map.release(fin_list)
        for req_id in fin_list:
            self._group.pop(req_id, None)
        slots: list[int] = []
        for req_id in active_req_ids:
            ctx_len = (
                int(context_lens.get(req_id, 0)) if context_lens is not None else 0
            )
            target_group = (
                group
                if group is not None
                else self._group_of(req_id, context_len=ctx_len)
            )
            slots.append(self.slot_map.allocate(req_id, group=target_group))
        return slots

    def buckets(self, token_paddings: Sequence[int]) -> Sequence[Bucket]:
        """Return decode and verify compilation buckets aligned to token paddings."""
        pads = sorted(p for p in token_paddings if p > 0)

        def _out_rows(b: int) -> int:
            """Compute padded output row count for bucket size `b`."""
            for p in pads:
                if p >= b:
                    return p
            return max(
                MIN_OUT_ROW_ALIGNMENT,
                -(-b // MIN_OUT_ROW_ALIGNMENT) * MIN_OUT_ROW_ALIGNMENT,
            )

        result: list[Bucket] = []
        self.bucket_out_rows.clear()
        for b in self.decode_buckets:
            out_r = _out_rows(b)
            bkt = Bucket(1, b, out_r)
            result.append(bkt)
            self.bucket_out_rows[bkt.key] = out_r
        for vb in self.verify_buckets:
            out_r = _out_rows(vb)
            bkt = Bucket(VERIFY_TOKEN_COUNT, vb, out_r)
            result.append(bkt)
            self.bucket_out_rows[bkt.key] = out_r
        return result

    def load(self, mesh: object) -> None:
        """Bind the runtime mesh and flush any remaining observed weights."""
        self.mesh = mesh
        latest = observed_weights_snapshot()
        if latest:
            self.observed_weights.update(latest)
        clear_observed_weights()

    def bind(self, view: RunnerView) -> dict[str, int]:
        """Validate unified MLA and KDA cache layout and return pool geometry."""
        vllm_cfg = view.vllm_config
        cache_cfg = getattr(vllm_cfg, "cache_config", None)
        block_size = view.cache_block_size
        raw_dtype = getattr(cache_cfg, "cache_dtype", "auto")
        cache_dtype = raw_dtype if isinstance(raw_dtype, str) and raw_dtype else "auto"
        expected_layout = FP8_LAYOUT if _is_fp8_cache_dtype(cache_dtype) else LAYOUT
        if block_size != expected_layout.tokens_per_block:
            req_blk = expected_layout.tokens_per_block
            raise LayoutDecline(
                EligibilityResult(
                    eligible=False,
                    reason="unaligned_cache_layout",
                    message=(
                        "unaligned_cache_layout: Kimi K3 unified KV+KDA pool "
                        f"requires block_size={req_blk}, got {block_size}"
                    ),
                )
            )

        par_cfg = getattr(vllm_cfg, "parallel_config", None)
        interleave = (
            getattr(par_cfg, "cp_kv_cache_interleave_size", None)
            if par_cfg is not None
            else None
        )

        if interleave is not None and _as_int(interleave) != block_size:
            raise LayoutDecline(
                EligibilityResult(
                    eligible=False,
                    reason="unaligned_cache_layout",
                    message=(
                        "unaligned_cache_layout: "
                        f"cp_kv_cache_interleave_size={interleave} "
                        f"must equal block_size={block_size}"
                    ),
                )
            )

        return {
            "block_size": block_size,
            "entry_width": LAYOUT.block_shape[2],
            "tokens_per_row": expected_layout.tokens_per_row,
            "mla_pools": MLA_LAYERS,
            "kda_layers": KDA_LAYERS,
        }

    def build(self, bucket: Bucket) -> StepFn:
        """Build a CPU-only contract-test step callable for `bucket`."""
        if self.is_tpu:
            msg = (
                "KimiK3MonoProgram contract step callable is CPU-only and "
                "cannot be built for TPU"
            )
            raise RuntimeError(msg)
        out_rows = bucket.out_rows if bucket.out_rows is not None else bucket.B
        hidden = self.hidden_size

        def step_fn(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            """Execute one bucketed CPU contract-test forward step."""
            dev = input_ids.device
            if dev.type != "cpu":
                msg = (
                    "KimiK3MonoProgram contract step callable is CPU-only; "
                    f"got device={dev}"
                )
                raise RuntimeError(msg)
            tok_f = input_ids.to(dtype=torch.float32).unsqueeze(-1)
            pos_f = positions.to(dtype=torch.float32).unsqueeze(-1)
            row_val = (tok_f * 0.015625 + pos_f * 0.00390625).to(dtype=torch.bfloat16)
            out = torch.zeros((out_rows, hidden), dtype=torch.bfloat16, device=dev)
            n = min(int(row_val.shape[0]), out_rows)
            out[:n, :] = row_val[:n].expand(n, hidden)
            return out

        return step_fn

    def warmup_inputs(self, bucket: Bucket, shape: RunnerShape) -> WarmupCase:
        """Construct synthetic warmup inputs and output validator for `bucket`."""
        del shape
        b_len = bucket.B
        out_rows = bucket.out_rows if bucket.out_rows is not None else b_len
        dev = torch.device("tpu") if self.is_tpu else torch.device("cpu")
        ids = torch.ones((b_len,), dtype=torch.int32, device=dev)
        pos = torch.arange(b_len, dtype=torch.int32, device=dev)
        key = bucket.key

        def _check(out: torch.Tensor) -> None:
            """Verify warmup output tensor dimensions and record warmed key."""
            out_shape = tuple(out.shape)
            expected_shape = (out_rows, self.hidden_size)
            if out_shape != expected_shape:
                msg = f"warmup output shape {out_shape} != {expected_shape}"
                raise AssertionError(msg)
            self.warmed.append(key)

        return WarmupCase(args=(ids, pos), check=_check)

    def _extract_attn_pair(self, md: object) -> tuple[object, object]:
        """Extract `(mla_md, kda_md)` from per-layer attention metadata via `pools.is_mla_layer`."""
        if not isinstance(md, Mapping):
            return None, None
        mla_md: object = None
        kda_md: object = None
        for k, val in md.items():
            try:
                idx = layer_index(str(k))
            except ValueError:
                continue
            if is_mla_layer(idx):
                if mla_md is None:
                    mla_md = val
            elif kda_md is None:
                kda_md = val
        return mla_md, kda_md

    def decide(self, ctx: object, fwd: ForwardView) -> StepDecision | None:
        """Evaluate step eligibility and select the target bucket."""
        attn_md = fwd.attn_metadata or {}
        input_ids = getattr(ctx, "input_ids", None)
        positions = getattr(ctx, "positions", None)
        if (
            input_ids is None
            or positions is None
            or not isinstance(input_ids, torch.Tensor)
        ):
            return None
        mla_md, kda_md = self._extract_attn_pair(attn_md)
        b_shape = fwd.batch_shape(mla_md, kda_md)
        return check_step_eligibility(
            b_shape,
            self.policy,
            compiled_buckets=set(fwd.steps.keys()),
            bucket_out_rows=self.bucket_out_rows,
            input_rows=int(input_ids.shape[0]),
        )

    def forward(
        self, decision: StepDecision | None, ctx: object, fwd: ForwardView
    ) -> torch.Tensor:
        """Run the compiled bucket step callable and slice to active tokens."""
        if decision is None:
            decision = self.decide(ctx, fwd)
        if decision is None or not decision.eligible:
            msg = "KimiK3MonoProgram.forward invoked on ineligible step"
            raise RuntimeError(msg)
        key = (decision.tokens_per_seq, decision.bucket_t)
        step_fn = fwd.steps[key]
        raw_ids = getattr(ctx, "input_ids", None)
        raw_pos = getattr(ctx, "positions", None)
        if (
            raw_ids is None
            or raw_pos is None
            or not isinstance(raw_ids, torch.Tensor)
            or not isinstance(raw_pos, torch.Tensor)
        ):
            msg = "KimiK3MonoProgram.forward requires tensor input_ids and positions"
            raise TypeError(msg)
        ids = raw_ids
        pos = raw_pos
        target_b = decision.bucket_t
        cur_n = int(ids.shape[0])
        if cur_n < target_b:
            pad_n = target_b - cur_n
            ids = torch.cat(
                [ids, torch.zeros((pad_n,), dtype=ids.dtype, device=ids.device)], dim=0
            )
            pos = torch.cat(
                [pos, torch.zeros((pad_n,), dtype=pos.dtype, device=pos.device)], dim=0
            )
        elif cur_n > target_b:
            ids = ids[:target_b]
            pos = pos[:target_b]
        out = step_fn(ids, pos)
        n_out = fwd.num_tokens if fwd.num_tokens is not None else cur_n
        return out[:n_out]


__all__ = [
    "KimiK3MonoProgram",
    "KimiK3ProgramConfig",
    "ObservedMxfp4Entry",
    "clear_observed_weights",
    "create",
    "layout_request",
    "load_tensor",
    "observed_weights_snapshot",
]
