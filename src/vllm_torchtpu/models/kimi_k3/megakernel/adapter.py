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
- `StepContext` and `ForwardView` carry no request IDs, so per-step `SlotMap`
  slot admission is not driven from `decide()`; callers use `admit_requests()`
  directly until the runner step-execution hook passes `StepInput`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import importlib
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.runtime.contract import (
    Bucket,
    Decline,
    EligibilityResult,
    LayoutDecline,
    RunnerShape,
    StepDecision,
    StepFn,
    WarmupCase,
)
from vllm_torchtpu.megakernel.runtime.eligibility import check_step_eligibility
from vllm_torchtpu.megakernel.runtime.policy import BucketPolicy
from vllm_torchtpu.megakernel_runtime.hooks import StepContext

from .kernels import paged
from .kernels.dspark_mla import MlaDraftConfig
from .kernels.load import BF16, pack_expert_bytes
from .kernels.paged import commit_kda_states
from .layout import (
    CONV_HEAD_ROWS,
    CONV_TAPS,
    FP8_LAYOUT,
    HEAD_DIM,
    KDA_HEADS,
    LAYOUT,
    NUM_LAYERS,
)
from .pools import (
    CONV_CHANNEL_GROUPS,
    KDA_LAYERS,
    MLA_LAYERS,
    KdaImporter,
    SlotMap,
    decode_step_options,
    dp_batch_sizes,
    dp_group_rows,
    draft_raw_pools,
    group_slots,
    is_mla_layer,
    kda_state_index,
    layer_index,
    pack_kda_state,
    read_kda_states,
    select_verify_kda_rows,
    unpack_kda_state,
    write_kda_states,
)

if TYPE_CHECKING:
    from vllm_torchtpu.megakernel_runtime.plugin import ForwardView, RunnerView

    from .kernels.paged import _DecodePagedOptions
from .program import (
    DRAFT_TOKENS,
    HIDDEN_SIZE,
    TP_SIZE,
    recurrent_state_dtype,
)
from .weight_stream import RankWeightStreamer, StreamStats

_bridge: object = None


def _resolve_tpu_bridge() -> object:
    """Return overridden `_bridge` or import the `runtime_shim.bridge` module."""
    if _bridge is not None:
        return _bridge
    return importlib.import_module(".runtime_shim.bridge", __package__)


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
BATCHED_HIDDEN_NDIM = 3
U32_PACK_FACTOR = 4
MIN_OUT_ROW_ALIGNMENT = 16
DEFAULT_MAX_SLOTS = 64
VERIFY_CONV_SEED_SCALE = 0.0078125
VERIFY_REC_SEED_SCALE = 0.001953125


def commit_verify_kda_states(
    conv_slots: jax.Array,
    recurrent_slots: jax.Array,
    new_conv: jax.Array,
    new_recurrent: jax.Array,
    verify_meta: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Commit only accepted verifier KDA rows into persistent slot banks."""
    slots, accepted_counts = verify_meta
    requests = int(accepted_counts.shape[0])
    sources, targets = select_verify_kda_rows(
        slots,
        accepted_counts,
        requests=requests,
        attention_dp=1,
        bank_slots=int(conv_slots.shape[0]),
    )
    return commit_kda_states(
        conv_slots,
        recurrent_slots,
        new_conv,
        new_recurrent,
        targets,
        source_rows=sources,
    )


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


class StreamingWeightStore:
    """Thread-safe per-rank single-copy streaming repacker store."""

    def __init__(self) -> None:
        """Initialize an empty per-rank streaming store."""
        self._lock = threading.Lock()
        self._streamers: dict[int, RankWeightStreamer] = {}

    def get_streamer(
        self,
        rank: int,
        *,
        num_layers: int = NUM_LAYERS,
        device: str = "cpu",
    ) -> RankWeightStreamer:
        """Return or initialize the `RankWeightStreamer` for `rank`."""
        with self._lock:
            streamer = self._streamers.get(rank)
            if (
                streamer is None
                or streamer.num_layers != num_layers
                or streamer.device != device
            ):
                streamer = RankWeightStreamer(
                    rank, num_layers=num_layers, device=device
                )
                self._streamers[rank] = streamer
            return streamer

    def consume(
        self,
        rank: int,
        name: str,
        tensor: torch.Tensor,
        *,
        num_layers: int = NUM_LAYERS,
        device: str = "cpu",
    ) -> None:
        """Route `tensor` to the rank streamer."""
        streamer = self.get_streamer(rank, num_layers=num_layers, device=device)
        streamer.consume(name, tensor)

    def snapshot(self, rank: int = 0) -> dict[str, torch.Tensor]:
        """Return a shallow copy of uploaded device tensors for `rank`."""
        with self._lock:
            streamer = self._streamers.get(rank)
            return dict(streamer.device_tensors) if streamer is not None else {}

    def stats(self, rank: int = 0) -> StreamStats | None:
        """Return staging telemetry for `rank`, or `None` if uninitialized."""
        with self._lock:
            streamer = self._streamers.get(rank)
            return streamer.stats() if streamer is not None else None

    def verify_complete(self, rank: int = 0) -> None:
        """Assert stream completeness and clean staging buffers for `rank`."""
        with self._lock:
            streamer = self._streamers.get(rank)
            if streamer is None:
                msg = f"no weights were streamed for rank {rank}"
                raise MegakernelConfigError(msg)
            streamer.verify_complete()

    def clear(self, rank: int | None = None) -> None:
        """Evict streamer state for `rank` or all ranks when `rank` is `None`."""
        with self._lock:
            if rank is None:
                self._streamers.clear()
            else:
                self._streamers.pop(rank, None)


STREAM_STORE = StreamingWeightStore()


def streamed_weights_snapshot(rank: int = 0) -> dict[str, torch.Tensor]:
    """Return uploaded device tensors captured by `load_tensor` on `rank`."""
    return STREAM_STORE.snapshot(rank)


def streamed_staging_stats(rank: int = 0) -> StreamStats | None:
    """Return staging telemetry for `rank` or None when uninitialized."""
    return STREAM_STORE.stats(rank)


def clear_streamed_weights(rank: int | None = None) -> None:
    """Evict per-rank or global weight streamer caches from `STREAM_STORE`."""
    STREAM_STORE.clear(rank)


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
    vllm_config: object, _options: Mapping[str, object] | None = None
) -> dict[str, int]:
    """Return unified MLA + KDA pool `block_size` and `dcp_interleave_size`."""
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
    """Resolve local worker position in ascending `device.id` order."""
    devices = sorted(jax.devices(), key=lambda d: int(d.id))
    if len(devices) != TP_SIZE:
        msg = (
            f"Kimi K3 weight loading requires {TP_SIZE} global JAX devices "
            f"or explicit kernel_rank in options, got {len(devices)} devices"
        )
        raise ValueError(msg)
    ids = [int(d.id) for d in devices]
    with contextlib.suppress(Exception):
        tpu_dist = importlib.import_module(
            "torch_tpu._internal.distributed.tpu_distributed"
        )
        dev_id = int(tpu_dist.global_device_id())
        if dev_id in ids:
            return ids.index(dev_id)
    local_devs = jax.local_devices()
    if not local_devs:
        msg = "Kimi K3 weight loading found no local JAX devices"
        raise ValueError(msg)
    dev_id = int(local_devs[0].id)
    return ids.index(dev_id)


def _coerce_kernel_rank_int(raw_pos: object) -> int:
    """Convert `raw_pos` to `int` or raise `ValueError` for non-integer inputs."""
    if not isinstance(raw_pos, bool) and isinstance(raw_pos, (int, str, np.integer)):
        try:
            return (
                int(raw_pos) if isinstance(raw_pos, (int, str)) else int(str(raw_pos))
            )
        except (ValueError, TypeError) as err:
            msg = f"kernel_rank must be an integer, got {raw_pos!r}"
            raise ValueError(msg) from err
    msg = f"kernel_rank must be an integer, got {raw_pos!r}"
    raise ValueError(msg)


def resolve_kernel_position(
    opts: Mapping[str, object], default_pos: int | None = None
) -> int:
    """Return ascending-`device.id` rank position `p` in `0..31`."""
    raw_pos = opts.get("kernel_rank", opts.get("device_position"))
    if raw_pos is not None:
        pos = _coerce_kernel_rank_int(raw_pos)
    elif jax.default_backend() == "tpu" and len(jax.devices()) == TP_SIZE:
        try:
            pos = _default_kernel_position()
        except ValueError:
            if default_pos is None:
                raise
            pos = int(default_pos)
    elif default_pos is not None:
        pos = int(default_pos)
    else:
        pos = _default_kernel_position()
    if not 0 <= pos < TP_SIZE:
        msg = f"kernel_rank position must be in 0..{TP_SIZE - 1}, got {pos}"
        raise ValueError(msg)
    return pos


_resolve_kernel_position = resolve_kernel_position


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


def _record_observed_mxfp4(
    name: str,
    tensor: torch.Tensor,
    opts: Mapping[str, object],
    *,
    is_scale: bool,
) -> None:
    """Slice and record an observed MXFP4 weight or E8M0 scale entry."""
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
        if shard is None:
            return
        if is_scale:
            scale_t = np.ascontiguousarray(shard.T).copy()
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
        WEIGHT_STORE.record(
            name,
            ObservedMxfp4Entry(
                name=name,
                kind="weight_meta",
                shape=tuple(shard.shape),
                dtype=dtype_name,
            ),
        )
        return

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


_ROUTED_EXPERT_ATTR_HINTS: tuple[str, ...] = (
    "w13_weight_packed",
    "w2_weight_packed",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight",
    "w2_weight",
)
_KDA_BLOCK_TABLE_NDIM = 2


def _is_decode_only_worker(
    view: object,
    options: Mapping[str, object],
) -> bool:
    """Return True when `view` or `options` identifies a decode-only worker."""
    if "reclaim_routed_experts" in options:
        return bool(options["reclaim_routed_experts"])
    for holder in (
        view,
        getattr(view, "vllm_config", None),
        getattr(view, "_runner", None),
    ):
        if holder is None:
            continue
        kv_cfg = getattr(holder, "kv_transfer_config", None)
        if kv_cfg is not None:
            is_consumer = bool(getattr(kv_cfg, "is_kv_consumer", False))
            is_producer = bool(getattr(kv_cfg, "is_kv_producer", False))
            if is_consumer and not is_producer:
                return True
    return False


def _named_model_tensors(
    model: object,
) -> list[Iterable[tuple[str, torch.Tensor]]]:
    """Collect parameter and buffer `(name, tensor)` iterables from `model`."""
    named_iters: list[Iterable[tuple[str, torch.Tensor]]] = []
    named_params = getattr(model, "named_parameters", None)
    if callable(named_params):
        named_iters.append(named_params())
    named_buffers = getattr(model, "named_buffers", None)
    if callable(named_buffers):
        named_iters.append(named_buffers())
    if not named_iters and isinstance(model, Mapping):
        named_iters.append(
            (str(k), v) for k, v in model.items() if isinstance(v, torch.Tensor)
        )
    return named_iters


def _free_routed_tensor_storage(
    name: str,
    tensor: torch.Tensor,
    seen_ptrs: set[int],
) -> int:
    """Release `tensor` storage if `name` is a routed-expert weight."""
    if not isinstance(tensor, torch.Tensor):
        return 0
    is_routed = any(hint in name for hint in _ROUTED_EXPERT_ATTR_HINTS) or (
        ".experts." in name and "shared_experts" not in name
    )
    if not is_routed:
        return 0
    storage = tensor.untyped_storage()
    nbytes = int(storage.nbytes())
    if nbytes <= 0:
        return 0
    ptr = int(storage.data_ptr())
    if ptr in seen_ptrs:
        return 0
    seen_ptrs.add(ptr)
    storage.resize_(0)
    return nbytes


def reclaim_routed_expert_hbm(model: object) -> int:
    """Release routed-expert parameter/buffer storages on `model`."""
    if model is None:
        return 0
    freed_bytes = 0
    seen_ptrs: set[int] = set()
    for pairs in _named_model_tensors(model):
        for name, tensor in pairs:
            freed_bytes += _free_routed_tensor_storage(name, tensor, seen_ptrs)
    return freed_bytes


def _resolve_load_tensor_flags(
    opts: Mapping[str, object],
) -> tuple[bool, bool]:
    """Return `(load_enabled, consume_mode)` with host-staged streaming defaults."""
    observe_only = bool(opts.get("observe_weights", False))
    if "load_tensors" in opts:
        load_enabled = bool(opts["load_tensors"])
    else:
        load_enabled = not observe_only
    if "consume_weights" in opts:
        consume_mode = bool(opts["consume_weights"])
    else:
        consume_mode = not observe_only and "load_tensors" not in opts
    return load_enabled, consume_mode


def load_tensor(
    name: str, tensor: torch.Tensor, options: Mapping[str, object] | None = None
) -> None:
    """Stream single-copy TP32 weights in host RAM or capture `1/32` MXFP4 shards."""
    opts = dict(options) if isinstance(options, Mapping) else {}
    load_enabled, consume_mode = _resolve_load_tensor_flags(opts)
    if not load_enabled and not bool(opts.get("observe_weights", False)):
        if bool(opts.get("consume_weights", False)):
            msg = "consume_weights=True requires load_tensors=True"
            raise ValueError(msg)
        return
    low = name.lower()
    if consume_mode:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return
        position = resolve_kernel_position(opts)
        num_layers = _as_int(opts.get("num_layers", NUM_LAYERS), default=NUM_LAYERS)
        raw_dev = opts.get("device", "cpu")
        dev = raw_dev if isinstance(raw_dev, str) and raw_dev else "cpu"
        streamer = STREAM_STORE.get_streamer(
            position, num_layers=num_layers, device=dev
        )
        t_in = tensor.detach().to(device="cpu") if dev == "cpu" else tensor
        streamer.consume(name, t_in)
        return
    if "shared_expert" not in low and isinstance(tensor, torch.Tensor):
        is_scale = any(
            tok in low for tok in ("weight_scale", "scale_inv", "e8m0", ".scales")
        )
        is_packed = any(
            tok in low
            for tok in (
                "weight_packed",
                "blocks",
                "mxfp4",
                ".experts.",
                "w13",
                "w2",
            )
        )
        if is_scale or is_packed:
            _record_observed_mxfp4(name, tensor, opts, is_scale=is_scale)


def _parse_strict_int(item: object, field_name: str) -> int:
    """Parse an integer from `int` or decimal `str`, raising `ValueError` otherwise."""
    if not isinstance(item, bool) and isinstance(item, (int, str)):
        try:
            return int(item)
        except ValueError as exc:
            msg = f"invalid {field_name} entry {item!r}: expected integer"
            raise ValueError(msg) from exc
    msg = f"invalid {field_name} entry {item!r}: expected integer"
    raise ValueError(msg)


def _split_bucket_items(raw_buckets: object, error_msg: str) -> list[object]:
    """Extract raw bucket entries from a comma-separated string or iterable."""
    if isinstance(raw_buckets, str):
        return [p.strip() for p in raw_buckets.split(",") if p.strip()]
    if isinstance(raw_buckets, Iterable) and not isinstance(raw_buckets, bytes):
        return list(raw_buckets)
    raise ValueError(error_msg)


def _parse_decode_buckets(raw_decode: object) -> tuple[int, ...]:
    """Parse and validate positive decode bucket sizes."""
    if raw_decode is None:
        return DEFAULT_DECODE_BUCKETS
    if raw_decode in ((), [], "none", False):
        return ()
    items = _split_bucket_items(
        raw_decode,
        "buckets must be a comma-separated string or iterable of positive "
        f"integers, got {raw_decode!r}",
    )
    parsed: list[int] = []
    for item in items:
        val = _parse_strict_int(item, "buckets")
        if val <= 0:
            msg = f"invalid buckets entry {val}: must be > 0"
            raise ValueError(msg)
        parsed.append(val)
    return tuple(sorted(set(parsed)))


def _parse_verify_buckets(raw_verify: object) -> tuple[int, ...]:
    """Parse and validate Kimi K3 verify bucket sizes."""
    if raw_verify is None:
        return DEFAULT_VERIFY_BUCKETS
    if raw_verify in ((), [], "none", False):
        return ()
    items = _split_bucket_items(
        raw_verify,
        "verify_buckets must be a comma-separated string or iterable of "
        f"integers, got {raw_verify!r}",
    )
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
    valid_layers = n_layers is None or (
        n_layers == NUM_LAYERS
        or (1 <= n_layers <= NUM_LAYERS and n_layers % KIMI_K3_ETP_PARTS == 0)
    )
    if not valid_layers:
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
    attention_dp: int,
    geometry: tuple[int | None, int | None],
    observed_weights: Mapping[str, ObservedMxfp4Entry],
) -> Decline | None:
    """Validate TPU-specific Kimi K3 parallelism, weights, and decoder state."""
    n_layers, hidden = geometry
    tp_size = view.tp_size
    par_cfg = getattr(view.vllm_config, "parallel_config", None)
    dcp_raw = getattr(par_cfg, "decode_context_parallel_size", None)
    dcp_size = _as_int(dcp_raw) if dcp_raw is not None else None
    expected_dcp = TP_SIZE // max(1, attention_dp)
    stream_rank = _resolve_kernel_position(opts, default_pos=view.tp_rank)
    has_mxfp4 = bool(observed_weights) or (
        bool(opts.get("consume_weights", False))
        and (
            bool(streamed_weights_snapshot(stream_rank))
            or STREAM_STORE.stats(stream_rank) is not None
        )
    )
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
            bool(opts.get("require_observed_mxfp4", False)) and not has_mxfp4,
            "missing_mxfp4_weights",
            "no MXFP4 weights captured by load_tensor prior to create()",
        ),
    )
    for failed, code, msg in checks:
        if failed:
            return Decline(code, msg)
    return None


def _resolve_buckets(
    opts: Mapping[str, object],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve sorted decode and verify bucket tuples from plugin options."""
    decode_buckets = _parse_decode_buckets(opts.get("buckets"))
    verify_buckets = _parse_verify_buckets(
        opts.get("verify_buckets", DEFAULT_VERIFY_BUCKETS)
    )
    return decode_buckets, verify_buckets


@overload
def _decline_or_raise_consumed(
    decline: Decline, *, consume_mode: Literal[False]
) -> Decline: ...


@overload
def _decline_or_raise_consumed(
    decline: Decline, *, consume_mode: Literal[True]
) -> NoReturn: ...


@overload
def _decline_or_raise_consumed(decline: Decline, *, consume_mode: bool) -> Decline: ...


def _decline_or_raise_consumed(decline: Decline, *, consume_mode: bool) -> Decline:
    """Raise `MegakernelConfigError` in consume mode, else return `decline`."""
    if consume_mode:
        msg = (
            "Kimi-K3 checkpoint weights were consumed; cannot decline "
            f"megakernel: {decline.message}"
        )
        raise MegakernelConfigError(msg)
    return decline


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
    num_layers: int = NUM_LAYERS
    view: RunnerView | None = None
    observed_weights: Mapping[str, ObservedMxfp4Entry] = dataclasses.field(
        default_factory=dict
    )
    streamed_weights: Mapping[str, torch.Tensor] = dataclasses.field(
        default_factory=dict
    )


def _check_topology_decline(
    view: RunnerView,
    opts: Mapping[str, object],
    observed: Mapping[str, ObservedMxfp4Entry],
    *,
    is_tpu: bool,
    attention_dp: int,
) -> Decline | None:
    """Check HF geometry and TPU/CPU topology, returning `Decline` on mismatch."""
    n_layers, hidden, hf_decline = _validate_hf_config(view.hf_config)
    if hf_decline is not None:
        return hf_decline
    if is_tpu:
        return validate_tpu_topology(
            view, opts, attention_dp, (n_layers, hidden), observed
        )
    if view.tp_size not in SUPPORTED_CPU_TP_SIZES:
        return Decline(
            "unsupported_tp_size",
            f"CPU test runner tp_size={view.tp_size} outside {SUPPORTED_CPU_TP_SIZES}",
        )
    return None


def create(
    view: RunnerView, options: Mapping[str, object] | None = None
) -> KimiK3MonoProgram | Decline:
    """Validate Kimi K3 topology, snapshot observed weights, and clear store."""
    observed = observed_weights_snapshot()
    opts = dict(options or {})
    stream_rank = _resolve_kernel_position(opts, default_pos=view.tp_rank)
    has_streamed = STREAM_STORE.stats(stream_rank) is not None
    consume_mode = (
        bool(opts["consume_weights"]) if "consume_weights" in opts else has_streamed
    )
    try:
        raw_dev_type = getattr(view.device, "type", "cpu")
        dev_type = (raw_dev_type if isinstance(raw_dev_type, str) else "cpu").lower()
        is_tpu = dev_type == "tpu"
        par_cfg = getattr(view.vllm_config, "parallel_config", None)
        default_dp = _as_int(getattr(par_cfg, "data_parallel_size", 1), default=1)
        attention_dp = _as_int(opts.get("attention_dp", default_dp), default=default_dp)

        top_decline = _check_topology_decline(
            view,
            opts,
            observed,
            is_tpu=is_tpu,
            attention_dp=attention_dp,
        )
        if top_decline is not None:
            return _decline_or_raise_consumed(top_decline, consume_mode=consume_mode)

        decode_buckets, verify_buckets = _resolve_buckets(opts)
        if is_tpu and attention_dp > 1 and decode_buckets:
            decode_buckets = dp_batch_sizes(decode_buckets, attention_dp)
        if not decode_buckets and not verify_buckets:
            return _decline_or_raise_consumed(
                Decline("empty_buckets", "no decode or verify buckets configured"),
                consume_mode=consume_mode,
            )

        hf_cfg = view.hf_config
        default_hidden = (
            _as_int(
                _extract_cfg_field(hf_cfg, "hidden_size", HIDDEN_SIZE),
                default=HIDDEN_SIZE,
            )
            if hf_cfg is not None
            else HIDDEN_SIZE
        )
        default_layers = (
            _as_int(
                _extract_cfg_field(hf_cfg, "num_hidden_layers", NUM_LAYERS),
                default=NUM_LAYERS,
            )
            if hf_cfg is not None
            else NUM_LAYERS
        )
        num_layers = _as_int(
            opts.get("num_layers", default_layers),
            default=default_layers,
        )
        hidden_size = _as_int(
            opts.get("synthetic_hidden_size", default_hidden),
            default=default_hidden,
        )
        max_slots = _as_int(
            opts.get("max_slots", DEFAULT_MAX_SLOTS), default=DEFAULT_MAX_SLOTS
        )

        if consume_mode:
            try:
                STREAM_STORE.verify_complete(stream_rank)
            except (ValueError, MegakernelConfigError) as err:
                msg = f"Kimi-K3 streaming weight verification failed: {err}"
                raise MegakernelConfigError(msg) from err
            st_stats = STREAM_STORE.stats(stream_rank)
            if st_stats is not None and st_stats.num_layers != num_layers:
                msg = (
                    f"Kimi-K3 streaming weight verification failed: "
                    f"streamed {st_stats.num_layers} layers != "
                    f"hf_config.num_hidden_layers {num_layers}"
                )
                raise MegakernelConfigError(msg)

        return KimiK3MonoProgram(
            KimiK3ProgramConfig(
                tp_rank=view.tp_rank,
                tp_size=view.tp_size,
                attention_dp=max(1, attention_dp),
                hidden_size=hidden_size,
                decode_buckets=decode_buckets,
                verify_buckets=verify_buckets,
                max_slots=max_slots,
                is_tpu=is_tpu,
                options=opts,
                num_layers=num_layers,
                view=view,
                observed_weights=observed,
                streamed_weights=(
                    streamed_weights_snapshot(stream_rank) if consume_mode else {}
                ),
            )
        )
    finally:
        clear_observed_weights()
        clear_streamed_weights(stream_rank)


def _extract_static_context_caches(
    static_ctx: Mapping[object, object],
) -> dict[str, object]:
    """Collect layer-name -> cache tensor entries from `static_forward_context`."""
    extracted: dict[str, object] = {}
    for name, mod in static_ctx.items():
        kv_attr = getattr(mod, "kv_cache", None)
        if kv_attr is not None:
            extracted[str(name)] = kv_attr
        elif isinstance(mod, (torch.Tensor, list, tuple)):
            extracted[str(name)] = mod
    return extracted


def _extract_model_layer_caches(layers: Iterable[object]) -> dict[str, object]:
    """Collect `kv_cache` attributes attached to model decoder layers."""
    by_layer: dict[str, object] = {}
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
        kv_attr = getattr(attn, "kv_cache", None)
        if kv_attr is not None:
            by_layer[f"model.layers.{idx}.self_attn"] = kv_attr
    return by_layer


def _discover_runner_caches(
    view: RunnerView, opts: Mapping[str, object]
) -> dict[str, object] | None:
    """Extract layer-name -> unified KV cache mapping from options or runner."""
    opt_caches = opts.get("caches")
    if isinstance(opt_caches, Mapping) and opt_caches:
        return {str(k): v for k, v in opt_caches.items()}
    comp_cfg = getattr(view.vllm_config, "compilation_config", None)
    static_ctx = getattr(comp_cfg, "static_forward_context", None)
    if isinstance(static_ctx, Mapping) and static_ctx:
        extracted = _extract_static_context_caches(static_ctx)
        if extracted:
            return extracted
    backbone = getattr(view.model, "model", view.model)
    layers = getattr(backbone, "layers", None)
    if isinstance(layers, Iterable):
        by_layer = _extract_model_layer_caches(layers)
        if by_layer:
            return by_layer
    return None


def _resolve_speculative_tokens(view: RunnerView | None) -> int:
    """Resolve speculative draft token count from `RunnerView` or `vllm_config`."""
    if view is None:
        return 0
    direct = _as_int(getattr(view, "speculative_tokens", 0), default=0)
    if direct > 0:
        return direct
    vllm_cfg = getattr(view, "vllm_config", None)
    spec_cfg = getattr(vllm_cfg, "speculative_config", None)
    return _as_int(getattr(spec_cfg, "num_speculative_tokens", 0), default=0)


def _resolve_bucket_policy(
    config: KimiK3ProgramConfig, options: Mapping[str, object]
) -> tuple[BucketPolicy, bool]:
    """Build `BucketPolicy` and report whether `verify8` speculation is active."""
    prof_ns = dataclasses.make_dataclass(
        "KimiBucketProfile",
        [("buckets", tuple[int, ...]), ("verify_buckets", tuple[int, ...])],
    )(buckets=config.decode_buckets, verify_buckets=config.verify_buckets)
    spec_toks = _resolve_speculative_tokens(config.view)
    enable_verify = (
        bool(options["enable_verify_bucket"])
        if "enable_verify_bucket" in options
        else (spec_toks == DRAFT_TOKENS)
    )
    verify_toks = (VERIFY_TOKEN_COUNT,) if enable_verify else ()
    raw_paddings = (
        (getattr(config.view, "token_paddings", ()) or ())
        if config.view is not None
        else ()
    )
    token_paddings = tuple(sorted(_as_int(p) for p in raw_paddings if _as_int(p) > 0))
    policy = BucketPolicy.resolve(
        prof_ns, verify_tokens=verify_toks, token_paddings=token_paddings
    )
    return policy, len(verify_toks) > 0


class KimiK3MonoProgram:
    """MonoProgram wrapping Kimi K3 decoder stack, unified pools, and SlotMap."""

    def __init__(self, config: KimiK3ProgramConfig) -> None:
        """Initialize the Kimi K3 MonoProgram from a validated configuration."""
        self.tp_rank = config.tp_rank
        self.tp_size = config.tp_size
        self.attention_dp = config.attention_dp
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.decode_buckets = config.decode_buckets
        self.verify_buckets = config.verify_buckets
        self.max_slots = config.max_slots
        self.is_tpu = config.is_tpu
        self.options = dict(config.options)
        self.slot_map = SlotMap(num_slots=config.max_slots, groups=config.attention_dp)
        self.dp_groups: dict[str, tuple[int, int] | int] = {}
        self._group = self.dp_groups
        self.warmed: list[tuple[int, int]] = []
        self.mesh: object = None
        self._driver_mesh: object = None
        self.kernel_pos: tuple[int, ...] = tuple(range(TP_SIZE))
        self.state_perm: tuple[tuple[int, int], ...] | None = None
        self.dcp_owner_table: tuple[int, ...] | None = None
        self.observed_weights: dict[str, ObservedMxfp4Entry] = dict(
            config.observed_weights
        )
        self.streamed_weights: dict[str, torch.Tensor] = dict(config.streamed_weights)
        self.decoder: object = None
        self.decoder_weights: dict[str, torch.Tensor] = dict(config.streamed_weights)
        self.bound_pools: object = None
        self.kda_importer: KdaImporter | None = None
        self.conv_slots: torch.Tensor | None = None
        self.recurrent_slots: torch.Tensor | None = None
        self.last_verify_conv: torch.Tensor | None = None
        self.last_verify_recurrent: torch.Tensor | None = None
        self.raw_pools: list[torch.Tensor] = []
        self.draft_pools: list[torch.Tensor] = []
        self.draft_aux_layers: tuple[int, ...] = ()
        self.mla_raw: tuple[int, ...] = ()
        self.kda_raw: tuple[int, ...] = ()
        self.max_pages_per_seq: int = 1
        self._view: RunnerView | None = config.view
        self.reclaimed_expert_bytes: int = 0
        self._slot_synced: set[str] = set()
        self._req_kda_blocks: dict[str, torch.Tensor] = {}
        self._tpu_ops: dict[tuple[int, int, int], object] = {}
        self._tpu_finish_ops: dict[tuple[int, int, int], object] = {}
        self.active_slots: list[int] = []
        self._active_block_tables: torch.Tensor | None = None
        self.bucket_out_rows: dict[tuple[int, int], int] = {}
        self.policy, self.speculative_enabled = _resolve_bucket_policy(
            config, self.options
        )

    def _group_of(self, req_id: str, context_len: int = 0) -> int:
        """Select or look up the attention-DP group for `req_id`."""
        if self.attention_dp == 1:
            return 0
        if req_id in self.dp_groups:
            existing = self.dp_groups[req_id]
            group_idx = existing[0] if isinstance(existing, tuple) else int(existing)
            self.dp_groups[req_id] = (group_idx, context_len)
            return group_idx
        count = [0] * self.attention_dp
        context = [0] * self.attention_dp
        for val in self.dp_groups.values():
            if isinstance(val, tuple):
                g_val, c_val = val
            else:
                g_val, c_val = int(val), 0
            count[g_val] += 1
            context[g_val] += c_val
        group_idx = min(range(self.attention_dp), key=lambda k: (count[k], context[k]))
        self.dp_groups[req_id] = (group_idx, context_len)
        return group_idx

    def _forget_requests(self, finished_req_ids: Iterable[str]) -> None:
        """Release `finished_req_ids` from `slot_map`, `dp_groups`, and KDA state."""
        fin_list = [str(req_id) for req_id in finished_req_ids]
        if not fin_list:
            return
        self.slot_map.release(fin_list)
        for req_id in fin_list:
            self.dp_groups.pop(req_id, None)
            self._slot_synced.discard(req_id)
            self._req_kda_blocks.pop(req_id, None)

    def admit_requests(
        self,
        active_req_ids: Sequence[str],
        finished_req_ids: Iterable[str] = (),
        *,
        group: int | None = None,
        context_lens: Mapping[str, int] | None = None,
    ) -> list[int]:
        """Release finished requests and allocate attention-DP-grouped SlotMap slots."""
        self._forget_requests(finished_req_ids)
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
        self.active_slots = list(slots)
        return slots

    def set_active_batch(
        self,
        slots: Sequence[int],
        block_tables: torch.Tensor | None = None,
    ) -> None:
        """Bind active KDA slot indices and DCP block tables for the next step."""
        self.active_slots = [int(s) for s in slots]
        self._active_block_tables = block_tables

    def clear_tpu_op_cache(self) -> None:
        """Clear cached TPU custom-op callables when rebinding devices or bridges."""
        self._tpu_ops.clear()
        self._tpu_finish_ops.clear()

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
        if self.speculative_enabled or "verify_buckets" in self.options:
            for vb in self.verify_buckets:
                out_r = _out_rows(vb)
                bkt = Bucket(VERIFY_TOKEN_COUNT, vb, out_r)
                result.append(bkt)
                self.bucket_out_rows[bkt.key] = out_r
        return result

    def _mark_runner_reclaimed(self, view: object) -> None:
        """Mark bound runner instances as having reclaimed routed-expert HBM."""
        for holder in (view, self._view):
            if holder is not None:
                runner_obj = getattr(holder, "_runner", None)
                if runner_obj is not None and hasattr(runner_obj, "__dict__"):
                    vars(runner_obj)["_megakernel_reclaimed_routed_experts"] = True

    def _upload_tpu_decoder_weights(self) -> None:
        """Upload host-staged decoder weights to TPU HBM via `bridge.upload`."""
        bridge_mod = _resolve_tpu_bridge()
        bridge_upload = getattr(bridge_mod, "upload", None)
        materialize_fn = getattr(bridge_mod, "materialize", None)
        if not callable(bridge_upload):
            msg = "KimiK3MonoProgram TPU weight upload requires bridge.upload"
            raise TypeError(msg)
        last_uploaded: torch.Tensor | None = None
        for w_name, w_tensor in list(self.decoder_weights.items()):
            if isinstance(w_tensor, torch.Tensor) and w_tensor.device.type == "cpu":
                w_squeezed = (
                    w_tensor.squeeze(0)
                    if (w_tensor.ndim > 1 and int(w_tensor.shape[0]) == 1)
                    else w_tensor
                )
                w_contig = w_squeezed.detach().contiguous()
                if w_contig.dtype == torch.bfloat16:
                    host_arr = np.ascontiguousarray(
                        w_contig.view(torch.uint16).numpy().view(BF16)
                    )
                else:
                    host_arr = np.ascontiguousarray(w_contig.numpy())
                uploaded = bridge_upload(host_arr)
                if isinstance(uploaded, torch.Tensor):
                    self.decoder_weights[w_name] = uploaded
                    last_uploaded = uploaded
        if callable(materialize_fn) and last_uploaded is not None:
            materialize_fn(last_uploaded)

    def _maybe_reclaim_and_upload_weights(self, view: object) -> int:
        """Reclaim `model` routed-expert HBM on decode workers before TPU upload."""
        should_reclaim = _is_decode_only_worker(view, self.options)
        if should_reclaim and self.reclaimed_expert_bytes == 0:
            model = (
                getattr(view, "model", None)
                or getattr(self._view, "model", None)
                or self.options.get("model")
            )
            self.reclaimed_expert_bytes += reclaim_routed_expert_hbm(model)
            if self.reclaimed_expert_bytes > 0:
                gc.collect()
                self._mark_runner_reclaimed(view)
        upload_fn = self.options.get("upload_fn")
        if callable(upload_fn) and self.decoder_weights:
            for w_name, w_tensor in list(self.decoder_weights.items()):
                if isinstance(w_tensor, torch.Tensor) and w_tensor.device.type == "cpu":
                    uploaded = upload_fn(w_tensor)
                    if isinstance(uploaded, torch.Tensor):
                        self.decoder_weights[w_name] = uploaded
        elif (
            self.is_tpu
            and self.decoder_weights
            and (
                (_bridge is not None and hasattr(_bridge, "upload"))
                or sys.modules.get("torch_tpu") is not None
            )
        ):
            self._upload_tpu_decoder_weights()
        self.streamed_weights.clear()
        return self.reclaimed_expert_bytes

    def _try_bridge_kernel_mesh(self) -> Mesh | None:
        """Return the TPU runtime bridge `kernel_mesh` when available."""
        if _bridge is not None or self.is_tpu:
            with contextlib.suppress(Exception):
                bridge_obj = _resolve_tpu_bridge()
                km_fn = getattr(bridge_obj, "kernel_mesh", None)
                if callable(km_fn):
                    cand_mesh, _ = km_fn("tp", TP_SIZE)
                    if (
                        isinstance(cand_mesh, Mesh)
                        and int(cand_mesh.devices.size) == TP_SIZE
                    ):
                        return cand_mesh
        return None

    def _resolve_kernel_mesh_and_alignment(self, driver_mesh: object) -> None:
        """Build ascending-`device.id` kernel mesh and alignment tables."""
        self._driver_mesh = driver_mesh
        if isinstance(driver_mesh, Mesh) and int(driver_mesh.devices.size) == TP_SIZE:
            driver_devs = list(driver_mesh.devices.flat)
            resolved_mesh = self._try_bridge_kernel_mesh()
            if resolved_mesh is not None:
                self.mesh = resolved_mesh
                kernel_devs = list(resolved_mesh.devices.flat)
            else:
                sorted_devs = sorted(driver_devs, key=lambda d: int(d.id))
                if [int(d.id) for d in driver_devs] == [int(d.id) for d in sorted_devs]:
                    self.mesh = driver_mesh
                    kernel_devs = driver_devs
                else:
                    kernel_devs = sorted_devs
                    self.mesh = Mesh(np.asarray(kernel_devs, dtype=object), ("tp",))
            id_to_p = {int(d.id): p for p, d in enumerate(kernel_devs)}
            driver_to_p = tuple(
                id_to_p.get(int(d.id), i) for i, d in enumerate(driver_devs)
            )
            if len(set(driver_to_p)) == TP_SIZE:
                self.kernel_pos = driver_to_p
                if driver_to_p == tuple(range(TP_SIZE)):
                    self.state_perm = None
                    self.dcp_owner_table = None
                else:
                    self.state_perm = tuple((driver_to_p[q], q) for q in range(TP_SIZE))
                    self.dcp_owner_table = tuple(
                        driver_to_p.index(p) for p in range(TP_SIZE)
                    )
            else:
                self.kernel_pos = tuple(range(TP_SIZE))
                self.state_perm = None
                self.dcp_owner_table = None
        else:
            self.mesh = driver_mesh
            self.kernel_pos = tuple(range(TP_SIZE))
            self.state_perm = None
            self.dcp_owner_table = None

    def load(self, mesh: object) -> None:
        """Bind the runtime mesh and flush observed or streamed weights."""
        self._resolve_kernel_mesh_and_alignment(mesh)
        stream_rank = _resolve_kernel_position(self.options, default_pos=self.tp_rank)
        try:
            opt_weights = self.options.get("decoder_weights")
            if isinstance(opt_weights, Mapping):
                self.decoder_weights = {
                    str(k): v
                    for k, v in opt_weights.items()
                    if isinstance(v, torch.Tensor)
                }
            if "consume_weights" in self.options:
                should_consume = bool(self.options["consume_weights"])
            else:
                should_consume = self.is_tpu or bool(
                    streamed_weights_snapshot(stream_rank)
                )
            if should_consume and STREAM_STORE.stats(stream_rank) is not None:
                STREAM_STORE.verify_complete(stream_rank)
                st_stats = STREAM_STORE.stats(stream_rank)
                if st_stats is not None and st_stats.num_layers != self.num_layers:
                    msg = (
                        f"streamed {st_stats.num_layers} layers != "
                        f"hf_config.num_hidden_layers {self.num_layers}"
                    )
                    raise MegakernelConfigError(msg)
                snap = streamed_weights_snapshot(stream_rank)
                self.streamed_weights.update(snap)
                self.decoder_weights.update(snap)
            latest = observed_weights_snapshot()
            if latest:
                self.observed_weights.update(latest)
        finally:
            clear_streamed_weights()
            clear_observed_weights()

    def import_kda_state(self, req_id: str, kda_blocks: torch.Tensor) -> int:
        """Allocate or look up `req_id`'s KDA slot and import `kda_blocks`."""
        if (
            self.kda_importer is None
            or self.conv_slots is None
            or self.recurrent_slots is None
        ):
            msg = "KimiK3MonoProgram.import_kda_state requires bound unified pools"
            raise RuntimeError(msg)
        r_key = str(req_id)
        slot = self.slot_map.allocate(r_key, group=self._group_of(r_key))
        self._req_kda_blocks[r_key] = kda_blocks
        self.conv_slots, self.recurrent_slots = self.kda_importer.import_state(
            self.conv_slots,
            self.recurrent_slots,
            kda_blocks,
            slot,
        )
        self._slot_synced.add(r_key)
        return slot

    def export_kda_state(
        self,
        req_id: str,
        kda_blocks: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Export `req_id`'s private slot back into its unified-pool KDA blocks."""
        if (
            self.kda_importer is None
            or self.conv_slots is None
            or self.recurrent_slots is None
        ):
            msg = "KimiK3MonoProgram.export_kda_state requires bound unified pools"
            raise RuntimeError(msg)
        r_key = str(req_id)
        blocks = (
            kda_blocks if kda_blocks is not None else self._req_kda_blocks.get(r_key)
        )
        if blocks is None:
            msg = f"no KDA blocks recorded for {r_key!r}"
            raise KeyError(msg)
        slot = self.slot_map.allocate(r_key, group=self._group_of(r_key))
        updated_pools = self.kda_importer.export_state(
            self.conv_slots,
            self.recurrent_slots,
            blocks,
            slot,
        )
        self._adopt_pools(updated_pools)
        self._slot_synced.discard(r_key)
        return self.raw_pools

    def _bind_draft_pools(self, caches: Mapping[str, object]) -> None:
        """Validate and bind the 5 MLA DSpark draft cache pools when present."""
        d_pools = draft_raw_pools(
            cast("Mapping[str, Sequence[torch.Tensor] | torch.Tensor]", caches)
        )
        if not d_pools:
            self.draft_pools = []
            self.draft_aux_layers = ()
            return
        draft_cfg = MlaDraftConfig()
        if len(d_pools) != draft_cfg.layers:
            msg = f"expected {draft_cfg.layers} MLA draft pools, got {len(d_pools)}"
            raise ValueError(msg)
        if self.raw_pools:
            tmpl = self.raw_pools[0]
            if any(p.shape != tmpl.shape or p.dtype != tmpl.dtype for p in d_pools):
                msg = "MLA draft pool shape or dtype does not match target MLA pool"
                raise ValueError(msg)
        self.draft_pools = list(d_pools)
        self.draft_aux_layers = draft_cfg.aux_layers

    def bind(self, view: RunnerView) -> dict[str, int]:
        """Validate unified MLA and KDA cache layout and bind KdaImporter."""
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

        self.max_pages_per_seq = max(
            1, _as_int(getattr(view, "max_pages_per_seq", 1), default=1)
        )
        num_blocks = 0
        caches = _discover_runner_caches(view, self.options)
        if caches:
            mesh_obj = (
                self.mesh
                if isinstance(self.mesh, Mesh)
                else Mesh(jax.devices()[:1], ("tp",))
            )
            raw_mamba = getattr(cache_cfg, "mamba_ssm_cache_dtype", "auto")
            mamba_dtype = recurrent_state_dtype(
                raw_mamba if isinstance(raw_mamba, str) and raw_mamba else "auto"
            )
            rec_dtype = torch.float32 if mamba_dtype == "float32" else torch.bfloat16
            importer = KdaImporter(
                mesh=mesh_obj,
                attention_dp=self.attention_dp,
                num_slots=self.max_slots,
                state_perm=self.state_perm,
            )
            num_blocks = importer.bind(
                cast("Mapping[str, Sequence[torch.Tensor] | torch.Tensor]", caches),
                cache_dtype,
            )
            self.kda_importer = importer
            self.raw_pools = list(importer.raw_pools)
            self.mla_raw = tuple(importer.mla_raw)
            self.kda_raw = tuple(importer.kda_raw)
            self.bound_pools = tuple(self.raw_pools)
            pool_dev = self.raw_pools[0].device
            state_slots = (
                group_slots(self.max_slots, self.attention_dp) * self.attention_dp
            )
            num_kda = len(self.kda_raw) or (self.num_layers * 3 // KIMI_K3_ETP_PARTS)
            self.conv_slots = torch.zeros(
                (
                    1,
                    state_slots,
                    num_kda,
                    CONV_TAPS,
                    CONV_CHANNEL_GROUPS,
                    CONV_HEAD_ROWS,
                    HEAD_DIM,
                ),
                dtype=torch.bfloat16,
                device=pool_dev,
            )
            self.recurrent_slots = torch.zeros(
                (1, state_slots, num_kda, KDA_HEADS, HEAD_DIM, HEAD_DIM),
                dtype=rec_dtype,
                device=pool_dev,
            )
            self._bind_draft_pools(caches)
        else:
            self.draft_pools = []
            self.draft_aux_layers = ()

        self._maybe_reclaim_and_upload_weights(view)
        if (
            caches
            and self.kda_importer is not None
            and self.conv_slots is not None
            and self.recurrent_slots is not None
            and self.conv_slots.device.type == "tpu"
            and isinstance(self.mesh, Mesh)
            and int(self.mesh.devices.size) == TP_SIZE
            and sys.modules.get("torch_tpu") is not None
        ):
            num_kda_warm = len(self.kda_raw) or (
                self.num_layers * 3 // KIMI_K3_ETP_PARTS
            )
            zero_kda = torch.zeros(
                (num_kda_warm,),
                dtype=torch.int32,
                device=self.conv_slots.device,
            )
            self.conv_slots, self.recurrent_slots = self.kda_importer.import_state(
                self.conv_slots,
                self.recurrent_slots,
                zero_kda,
                0,
            )
            self._adopt_pools(
                self.kda_importer.export_state(
                    self.conv_slots,
                    self.recurrent_slots,
                    zero_kda,
                    0,
                )
            )
        return {
            "block_size": block_size,
            "entry_width": LAYOUT.block_shape[2],
            "tokens_per_row": expected_layout.tokens_per_row,
            "mla_pools": len(self.mla_raw) or MLA_LAYERS,
            "kda_layers": len(self.kda_raw) or KDA_LAYERS,
            "num_blocks": num_blocks,
            "draft_pools": len(self.draft_pools),
        }

    def make_tpu_decode_body(
        self,
        bucket: Bucket,
        *,
        layers: int = NUM_LAYERS,
    ) -> tuple[
        Callable[..., Any],
        tuple[P, ...],
        tuple[P, ...],
        tuple[int, ...],
        str,
    ]:
        """Build the per-shard `decode_paged` body and sharding specs for `bucket`."""
        batch_size = int(bucket.B)
        tokens_per_seq = int(bucket.T)
        out_rows = int(bucket.out_rows if bucket.out_rows is not None else batch_size)
        cache_key = (tokens_per_seq, batch_size, out_rows)
        weight_names = tuple(sorted(self.decoder_weights.keys()))
        num_weights = len(weight_names)
        num_pools = len(self.raw_pools)
        raw_map = tuple(self.mla_raw)
        pad_pool = raw_map[-1] if raw_map else 0
        pool_map = raw_map + (pad_pool,) * max(0, MLA_LAYERS - len(raw_map))
        dp = self.attention_dp
        defer_commit = tokens_per_seq == VERIFY_TOKEN_COUNT
        step_opts = cast(
            "_DecodePagedOptions",
            {
                **decode_step_options(batch_size, dp),
                "batch_size": batch_size,
                "layers": layers,
                "dcp_pool_map": pool_map,
                "sequence_rows": tokens_per_seq,
                "defer_kda_commit": defer_commit,
                "return_hidden": True,
            },
        )
        if self.dcp_owner_table is not None:
            step_opts["dcp_owner_table"] = self.dcp_owner_table
        placed: np.ndarray | None = None
        order: np.ndarray | None = None
        if dp > 1:
            placed = dp_group_rows(batch_size, dp)
            order = np.argsort(placed)

        def body(*args: jax.Array) -> tuple[jax.Array, ...]:
            """Run `paged.decode_paged` with `return_hidden=True` per shard."""
            w_vals = args[:num_weights]
            conv_in = args[num_weights][0]
            rec_in = args[num_weights + 1][0]
            pools_in = args[num_weights + 2 : num_weights + 2 + num_pools]
            tokens, pos, tables, slots = args[num_weights + 2 + num_pools :]
            if order is not None:
                tokens, pos, tables, slots = (
                    tokens[order],
                    pos[order],
                    tables[order],
                    slots[order],
                )
            weights = {k: v[0] for k, v in zip(weight_names, w_vals, strict=True)}
            paged_out = paged.decode_paged(
                weights,
                conv_in,
                rec_in,
                pools_in,
                tokens,
                pos,
                tables,
                slots,
                **step_opts,
            )
            hidden = paged_out[0]
            if placed is not None:
                hidden = hidden[placed]
            conv_out = paged_out[-3]
            rec_out = paged_out[-2]
            pools_out = paged_out[-1]
            if out_rows > batch_size:
                hidden = jnp.pad(hidden, ((0, out_rows - batch_size), (0, 0)))
            return (
                hidden,
                conv_out[None],
                rec_out[None],
                *pools_out,
            )

        in_specs = (P("tp"),) * (num_weights + 2 + num_pools) + (P(),) * 4
        out_specs = (P(),) + (P("tp"),) * (2 + num_pools)
        donate_start = num_weights + 2 if defer_commit else num_weights
        donate = tuple(range(donate_start, num_weights + 2 + num_pools))
        export_key = repr(
            (
                "k3_mono_decode",
                cache_key,
                pool_map,
                dp,
                defer_commit,
                layers,
                self.dcp_owner_table,
            )
        )
        return body, in_specs, out_specs, donate, export_key

    def _require_tpu_register_op(self, caller: str) -> Callable[..., object]:
        """Validate `self.mesh` and resolve the `torch_tpu` `register_op` hook."""
        if not isinstance(self.mesh, Mesh):
            msg = f"KimiK3MonoProgram {caller} requires a JAX Mesh"
            raise TypeError(msg)
        bridge_obj = _resolve_tpu_bridge()
        register_fn = getattr(bridge_obj, "register_op", None)
        if callable(register_fn):
            return register_fn
        msg = f"KimiK3MonoProgram {caller} requires torch_tpu"
        raise RuntimeError(msg)

    def _register_bridge_op(
        self,
        caller: str,
        name: str,
        lowered: tuple[
            Callable[..., Any],
            tuple[P, ...],
            tuple[P, ...],
            tuple[int, ...],
            str,
        ],
    ) -> object:
        """Register a sharded JAX custom op using `OpRegistrationSpec`."""
        body, in_specs, out_specs, donate, export_key = lowered
        register_fn = self._require_tpu_register_op(caller)
        bridge_obj = _resolve_tpu_bridge()
        spec_cls = getattr(bridge_obj, "OpRegistrationSpec", None)
        if callable(spec_cls):
            spec = spec_cls(
                in_specs=in_specs,
                out_specs=out_specs,
                donate_argnums=donate,
                export_key=export_key,
            )
            return register_fn(name, self.mesh, body, spec)
        return register_fn(
            name,
            self.mesh,
            body,
            in_specs,
            out_specs,
            donate_argnums=donate,
            export_key=export_key,
        )

    def _get_or_register_tpu_op(self, bucket: Bucket) -> object:
        """Compile and cache the `sharded_jax_op` for `bucket` over `decode_paged`."""
        batch_size = int(bucket.B)
        tokens_per_seq = int(bucket.T)
        out_rows = int(bucket.out_rows if bucket.out_rows is not None else batch_size)
        cache_key = (tokens_per_seq, batch_size, out_rows)
        cached = self._tpu_ops.get(cache_key)
        if cached is not None:
            return cached
        lowered = self.make_tpu_decode_body(
            bucket,
            layers=self.num_layers,
        )
        op = self._register_bridge_op(
            "TPU op registration",
            f"pallas::k3_mono_decode_t{tokens_per_seq}_b{batch_size}_r{out_rows}",
            lowered,
        )
        self._tpu_ops[cache_key] = op
        return op

    def _adopt_pools(self, pools_out: Sequence[torch.Tensor]) -> None:
        """Update bound pool storages in place to preserve vLLM tensor identity."""
        for old, fresh in zip(self.raw_pools, pools_out, strict=True):
            if fresh is not old:
                old.set_(
                    fresh.untyped_storage(),
                    int(fresh.storage_offset()),
                    tuple(fresh.shape),
                    tuple(fresh.stride()),
                )

    def _sync_tpu_tensors(self, tensors: Sequence[torch.Tensor]) -> None:
        """Synchronize lazy TPU tensors through the runtime bridge when on TPU."""
        tpu_tensors = [
            t for t in tensors if isinstance(t, torch.Tensor) and t.device.type == "tpu"
        ]
        if not tpu_tensors:
            return
        if _bridge is None and sys.modules.get("torch_tpu") is None:
            return
        bridge_obj = _resolve_tpu_bridge()
        sync_fn = getattr(bridge_obj, "synchronize", None)
        if callable(sync_fn):
            sync_fn(tpu_tensors, wait=False)

    def _adopt_tpu_step_outputs(
        self,
        outputs: Sequence[torch.Tensor],
        tokens_per_seq: int,
    ) -> torch.Tensor:
        """Adopt donated KDA slots and unified KV pools from a TPU decode step."""
        hidden_out = outputs[0]
        if tokens_per_seq == VERIFY_TOKEN_COUNT:
            self.last_verify_conv = outputs[1]
            self.last_verify_recurrent = outputs[2]
        else:
            self.conv_slots = outputs[1]
            self.recurrent_slots = outputs[2]
        self._adopt_pools(outputs[3:])
        return hidden_out[0] if hidden_out.ndim == BATCHED_HIDDEN_NDIM else hidden_out

    def prepare_tpu_step_inputs(
        self, batch_size: int, tokens_per_seq: int, dev: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad or slice block tables and KDA slot indices for a TPU step."""
        slots_list = list(self.active_slots)
        if tokens_per_seq > 1 and slots_list:
            expanded: list[int] = []
            for s_val in slots_list:
                expanded.extend([s_val] * tokens_per_seq)
            slots_list = expanded
        if len(slots_list) < batch_size:
            slots_list.extend([0] * (batch_size - len(slots_list)))
        slots_tensor = torch.tensor(
            slots_list[:batch_size], dtype=torch.int32, device=dev
        )
        tables = self._active_block_tables
        if tables is None:
            tables_tensor = torch.zeros(
                (batch_size, self.max_pages_per_seq),
                dtype=torch.int32,
                device=dev,
            )
            return tables_tensor, slots_tensor
        tables_tensor = tables.to(dtype=torch.int32, device=dev)
        if tokens_per_seq > 1 and (
            (self.active_slots and tables_tensor.shape[0] == len(self.active_slots))
            or tables_tensor.shape[0] <= batch_size // tokens_per_seq
        ):
            tables_tensor = tables_tensor.repeat_interleave(tokens_per_seq, dim=0)
        if tables_tensor.shape[0] < batch_size:
            pad_r = batch_size - int(tables_tensor.shape[0])
            tables_tensor = torch.cat(
                [
                    tables_tensor,
                    torch.zeros(
                        (pad_r, tables_tensor.shape[1]),
                        dtype=torch.int32,
                        device=dev,
                    ),
                ],
                dim=0,
            )
            return tables_tensor, slots_tensor
        return tables_tensor[:batch_size], slots_tensor

    _prepare_tpu_step_inputs = prepare_tpu_step_inputs

    def _execute_uncompiled_tpu_step(
        self,
        op_and_weights: tuple[Callable[..., Any], tuple[torch.Tensor, ...]],
        shape: tuple[int, int],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Execute one uncompiled 2-argument TPU decode step for unit tests."""
        op, w_tensors = op_and_weights
        batch_size, tokens_per_seq = shape
        if (
            self.conv_slots is None
            or self.recurrent_slots is None
            or not self.raw_pools
        ):
            msg = (
                "KimiK3MonoProgram TPU step requires loaded decoder_weights "
                "and bound unified pools"
            )
            raise RuntimeError(msg)
        if self.last_verify_recurrent is not None:
            msg = (
                "uncommitted deferred verify KDA state; call finish_verify "
                "before running the next TPU step"
            )
            raise RuntimeError(msg)
        ids_i32 = input_ids.to(dtype=torch.int32)
        pos_i32 = positions.to(dtype=torch.int32)
        tables_tensor, slots_tensor = self.prepare_tpu_step_inputs(
            batch_size, tokens_per_seq, input_ids.device
        )
        self._sync_tpu_tensors(
            [
                ids_i32,
                pos_i32,
                tables_tensor,
                slots_tensor,
                self.conv_slots,
                self.recurrent_slots,
                *self.raw_pools,
            ]
        )
        outputs = op(
            *w_tensors,
            self.conv_slots,
            self.recurrent_slots,
            *self.raw_pools,
            ids_i32,
            pos_i32,
            tables_tensor,
            slots_tensor,
        )
        self._sync_tpu_tensors(outputs)
        return self._adopt_tpu_step_outputs(outputs, tokens_per_seq)

    def _build_tpu_step(self, bucket: Bucket) -> StepFn:
        """Build a TPU `StepFn` running `paged.decode_paged`."""
        batch_size = int(bucket.B)
        tokens_per_seq = int(bucket.T)
        if (
            not self.decoder_weights
            or self.conv_slots is None
            or self.recurrent_slots is None
            or not self.raw_pools
        ):

            def _unready_tpu_step_fn(
                _input_ids: torch.Tensor, _positions: torch.Tensor
            ) -> torch.Tensor:
                """Reject execution when weights or pools are not yet bound."""
                msg = (
                    "KimiK3MonoProgram TPU step requires loaded decoder_weights "
                    "and bound unified pools"
                )
                raise RuntimeError(msg)

            return _unready_tpu_step_fn

        op = self._get_or_register_tpu_op(bucket)
        if not callable(op):
            msg = "registered TPU decode op is not callable"
            raise TypeError(msg)
        weight_names = tuple(sorted(self.decoder_weights.keys()))
        w_tensors = tuple(self.decoder_weights[k] for k in weight_names)

        def tpu_step_fn(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            *state_and_pools: torch.Tensor,
        ) -> torch.Tensor | tuple[torch.Tensor, ...]:
            """Execute one bucketed TPU decode step and return hidden states."""
            if not state_and_pools:
                return self._execute_uncompiled_tpu_step(
                    (op, w_tensors),
                    (batch_size, tokens_per_seq),
                    input_ids,
                    positions,
                )
            tables_tensor = state_and_pools[0]
            slots_tensor = state_and_pools[1]
            conv_slots = state_and_pools[2]
            recurrent_slots = state_and_pools[3]
            raw_pools = state_and_pools[4:]
            outputs = op(
                *w_tensors,
                conv_slots,
                recurrent_slots,
                *raw_pools,
                input_ids,
                positions,
                tables_tensor,
                slots_tensor,
            )
            hidden_out = outputs[0]
            hidden_2d = (
                hidden_out[0] if hidden_out.ndim == BATCHED_HIDDEN_NDIM else hidden_out
            )
            return (hidden_2d, *outputs[1:])

        return cast("StepFn", tpu_step_fn)

    def _record_cpu_verify_states(
        self, input_ids: torch.Tensor, positions: torch.Tensor, batch_rows: int
    ) -> None:
        """Populate per-row deferred KDA states for a CPU `verify8` step."""
        if self.conv_slots is None or self.recurrent_slots is None:
            return
        dev = self.conv_slots.device
        n_tok = min(int(input_ids.shape[0]), batch_rows)
        ids_pad = torch.zeros((batch_rows,), dtype=torch.float32, device=dev)
        pos_pad = torch.zeros((batch_rows,), dtype=torch.float32, device=dev)
        ids_pad[:n_tok] = input_ids[:n_tok].to(dtype=torch.float32, device=dev)
        pos_pad[:n_tok] = positions[:n_tok].to(dtype=torch.float32, device=dev)
        seed = (
            ids_pad * VERIFY_CONV_SEED_SCALE
            + pos_pad * VERIFY_REC_SEED_SCALE
            + torch.arange(1, batch_rows + 1, dtype=torch.float32, device=dev)
        )
        num_kda = int(self.conv_slots.shape[2])
        conv_shape = (
            1,
            batch_rows,
            num_kda,
            CONV_TAPS,
            CONV_CHANNEL_GROUPS,
            CONV_HEAD_ROWS,
            HEAD_DIM,
        )
        rec_shape = (1, batch_rows, num_kda, KDA_HEADS, HEAD_DIM, HEAD_DIM)
        conv_view = seed.view(1, batch_rows, 1, 1, 1, 1, 1).to(dtype=torch.bfloat16)
        rec_view = seed.view(1, batch_rows, 1, 1, 1, 1).to(
            dtype=self.recurrent_slots.dtype
        )
        self.last_verify_conv = conv_view.expand(conv_shape).clone()
        self.last_verify_recurrent = rec_view.expand(rec_shape).clone()

    def make_tpu_finish_body(
        self, requests: int, bank_slots: int
    ) -> tuple[
        Callable[..., Any],
        tuple[P, ...],
        tuple[P, ...],
        tuple[int, ...],
        str,
    ]:
        """Build the per-shard `commit_kda_states` body and sharding specs."""
        dp = self.attention_dp
        key = (requests, dp, bank_slots)

        def body(*args: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Commit accepted verifier KDA rows per shard."""
            conv_in, rec_in, new_conv, new_rec, slots_arr, acc_arr = args
            srcs, tgts = select_verify_kda_rows(
                slots_arr,
                acc_arr,
                requests=requests,
                attention_dp=dp,
                bank_slots=bank_slots,
            )
            c_out, r_out = commit_kda_states(
                conv_in[0],
                rec_in[0],
                new_conv[0],
                new_rec[0],
                tgts,
                source_rows=srcs,
            )
            return c_out[None], r_out[None]

        in_specs = (P("tp"), P("tp"), P("tp"), P("tp"), P(), P())
        out_specs = (P("tp"), P("tp"))
        export_key = repr(("k3_mono_finish_verify", key))
        return body, in_specs, out_specs, (0, 1), export_key

    def _get_or_register_tpu_finish_op(self, requests: int) -> object:
        """Compile and cache the TPU `kda_selected_state_commit` op for `requests`."""
        bank_slots = (
            int(self.conv_slots.shape[1])
            if self.conv_slots is not None
            else DEFAULT_MAX_SLOTS
        )
        key = (requests, self.attention_dp, bank_slots)
        cached = self._tpu_finish_ops.get(key)
        if cached is not None:
            return cached
        dp = self.attention_dp
        lowered = self.make_tpu_finish_body(requests, bank_slots)
        op = self._register_bridge_op(
            "TPU finish_verify",
            f"pallas::k3_mono_finish_verify_r{requests}_dp{dp}_s{bank_slots}",
            lowered,
        )
        self._tpu_finish_ops[key] = op
        return op

    def _finish_verify_cpu(
        self, accepted_counts: torch.Tensor, req_slots: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Commit accepted verifier KDA rows into CPU slot banks."""
        if (
            self.conv_slots is None
            or self.recurrent_slots is None
            or self.last_verify_conv is None
            or self.last_verify_recurrent is None
        ):
            msg = "finish_verify requires bound slots and deferred verify states"
            raise RuntimeError(msg)
        requests = int(accepted_counts.numel())
        rows_per = int(self.last_verify_recurrent.shape[1]) // max(1, requests)
        conv_out = self.conv_slots.clone()
        rec_out = self.recurrent_slots.clone()
        acc_flat = accepted_counts.reshape(-1)
        for req_idx in range(requests):
            target_slot = int(req_slots[req_idx])
            if target_slot <= 0:
                continue
            acc_tok = min(max(int(acc_flat[req_idx].item()), 0), rows_per - 1)
            src_row = req_idx * rows_per + acc_tok
            local_slot = target_slot
            if self.attention_dp > 1:
                bank = group_slots(self.max_slots, self.attention_dp)
                share = bank - 1
                owner = (target_slot - 1) // share
                local_slot = (target_slot - owner * share) + owner * bank
            conv_out[0, local_slot] = self.last_verify_conv[0, src_row]
            rec_out[0, local_slot] = self.last_verify_recurrent[0, src_row]
        self.conv_slots = conv_out
        self.recurrent_slots = rec_out
        self.last_verify_conv = None
        self.last_verify_recurrent = None
        return conv_out, rec_out

    def finish_verify(
        self,
        accepted_counts: torch.Tensor,
        slots: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Commit only the accepted verifier KDA rows into persistent slot banks."""
        if (
            self.conv_slots is None
            or self.recurrent_slots is None
            or self.last_verify_conv is None
            or self.last_verify_recurrent is None
        ):
            msg = "finish_verify requires bound slots and deferred verify states"
            raise RuntimeError(msg)
        requests = int(accepted_counts.numel())
        req_slots = list(slots) if slots is not None else list(self.active_slots)
        if len(req_slots) < requests:
            req_slots.extend([0] * (requests - len(req_slots)))
        else:
            req_slots = req_slots[:requests]
        if self.conv_slots.device.type == "cpu":
            return self._finish_verify_cpu(accepted_counts, req_slots)
        op = self._get_or_register_tpu_finish_op(requests)
        if not callable(op):
            msg = "registered TPU finish_verify op is not callable"
            raise TypeError(msg)
        expanded_slots = [s for s in req_slots for _ in range(VERIFY_TOKEN_COUNT)]
        dev = self.conv_slots.device
        slots_t = torch.tensor(expanded_slots, dtype=torch.int32, device=dev)
        acc_t = accepted_counts.reshape(requests).to(dtype=torch.int32, device=dev)
        self._sync_tpu_tensors(
            [
                self.conv_slots,
                self.recurrent_slots,
                self.last_verify_conv,
                self.last_verify_recurrent,
                slots_t,
                acc_t,
            ]
        )
        conv_out, rec_out = op(
            self.conv_slots,
            self.recurrent_slots,
            self.last_verify_conv,
            self.last_verify_recurrent,
            slots_t,
            acc_t,
        )
        self._sync_tpu_tensors([conv_out, rec_out])
        self.conv_slots = conv_out
        self.recurrent_slots = rec_out
        self.last_verify_conv = None
        self.last_verify_recurrent = None
        return conv_out, rec_out

    def build(self, bucket: Bucket) -> StepFn:
        """Build a TPU `decode_paged` step or CPU contract-test step for `bucket`."""
        if self.is_tpu:
            return self._build_tpu_step(bucket)
        out_rows = bucket.out_rows if bucket.out_rows is not None else bucket.B
        batch_rows = int(bucket.B)
        is_verify = int(bucket.T) == VERIFY_TOKEN_COUNT
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
            if is_verify:
                self._record_cpu_verify_states(input_ids, positions, batch_rows)
            tok_f = input_ids.to(dtype=torch.float32).unsqueeze(-1)
            pos_f = positions.to(dtype=torch.float32).unsqueeze(-1)
            row_val = (tok_f * 0.015625 + pos_f * 0.00390625).to(dtype=torch.bfloat16)
            out = torch.zeros((out_rows, hidden), dtype=torch.bfloat16, device=dev)
            n = min(int(row_val.shape[0]), out_rows)
            out[:n, :] = row_val[:n].expand(n, hidden)
            return out

        return step_fn

    def warmup_inputs(self, bucket: Bucket, _shape: RunnerShape) -> WarmupCase:
        """Construct synthetic warmup inputs and output validator for `bucket`."""
        b_len = int(bucket.B)
        t_len = int(bucket.T)
        out_rows = int(bucket.out_rows if bucket.out_rows is not None else b_len)
        if self.conv_slots is not None:
            dev = self.conv_slots.device
        elif self.is_tpu and sys.modules.get("torch_tpu") is not None:
            dev = torch.device("tpu")
        else:
            dev = torch.device("cpu")
        ids = torch.ones((b_len,), dtype=torch.int32, device=dev)
        pos = torch.arange(b_len, dtype=torch.int32, device=dev)
        key = bucket.key
        conv_s = self.conv_slots
        rec_s = self.recurrent_slots
        if self.is_tpu and conv_s is not None and rec_s is not None and self.raw_pools:
            tables_tensor, slots_tensor = self.prepare_tpu_step_inputs(
                b_len, t_len, dev
            )
            warmup_args: tuple[torch.Tensor, ...] = (
                ids,
                pos,
                tables_tensor,
                slots_tensor,
                conv_s,
                rec_s,
                *self.raw_pools,
            )
            self._sync_tpu_tensors(warmup_args)
        else:
            warmup_args = (ids, pos)

        def _check(out: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
            """Verify warmup output tensor dimensions and record warmed key."""
            if isinstance(out, tuple):
                hidden_t = self._adopt_tpu_step_outputs(out, t_len)
            else:
                hidden_t = out
            out_shape = tuple(hidden_t.shape)
            expected_shape = (out_rows, self.hidden_size)
            if out_shape != expected_shape:
                msg = f"warmup output shape {out_shape} != {expected_shape}"
                raise AssertionError(msg)
            if t_len == VERIFY_TOKEN_COUNT:
                self.last_verify_conv = None
                self.last_verify_recurrent = None
            self.warmed.append(key)

        return WarmupCase(args=warmup_args, check=_check)

    def _extract_attn_pair(self, md: object) -> tuple[object, object]:
        """Extract `(mla_md, kda_md)` from per-layer attention metadata."""
        if not isinstance(md, Mapping):
            return md, None
        mla_md: object = md.get("mla")
        kda_md: object = md.get("kda_0")
        for layer_key, val in md.items():
            try:
                idx = layer_index(str(layer_key))
            except ValueError:
                continue
            if is_mla_layer(idx):
                if mla_md is None:
                    mla_md = val
            elif kda_md is None:
                kda_md = val
        return mla_md, kda_md

    def _release_finished_from_metadata(
        self,
        md: object,
        mla_md: object = None,
        ctx: object = None,
    ) -> None:
        """Release finished request slots from `ctx` or step metadata."""
        for src in (ctx, md, mla_md):
            if src is None:
                continue
            fin_ids = _lookup_mapping_or_attr(src, "finished_req_ids")
            if isinstance(fin_ids, Iterable) and not isinstance(fin_ids, (str, bytes)):
                self._forget_requests(str(req_id) for req_id in fin_ids)

    def _resolve_step_req_ids(
        self,
        md: object,
        mla_md: object = None,
        ctx: object = None,
    ) -> list[str] | None:
        """Extract active request IDs from `ctx` or step attention metadata."""
        for src in (ctx, md, mla_md):
            if src is None:
                continue
            req_ids = _lookup_mapping_or_attr(src, "req_ids")
            if isinstance(req_ids, Sequence) and not isinstance(req_ids, (str, bytes)):
                return [str(req_id) for req_id in req_ids]
        return None

    @staticmethod
    def _effective_mamba_state_blocks(
        val: object,
        n_reqs: int,
    ) -> torch.Tensor | None:
        """Resolve the active KDA checkpoint block per request from layer metadata."""
        m_raw = _lookup_mapping_or_attr(val, "mamba_state_indices")
        if not isinstance(m_raw, torch.Tensor):
            return None
        m_idx = cast("torch.Tensor", m_raw)
        if m_idx.ndim != 1 or int(m_idx.shape[0]) < n_reqs:
            return None
        base = m_idx[:n_reqs]
        ckpt_raw = _lookup_mapping_or_attr(val, "mamba_ckpt_indices")
        off_raw = _lookup_mapping_or_attr(val, "mamba_slot_read_offsets")
        if isinstance(ckpt_raw, torch.Tensor) and isinstance(off_raw, torch.Tensor):
            ckpt = cast("torch.Tensor", ckpt_raw)
            offsets = cast("torch.Tensor", off_raw)
            if (
                ckpt.ndim == _KDA_BLOCK_TABLE_NDIM
                and int(ckpt.shape[0]) >= n_reqs
                and int(ckpt.shape[1]) > 0
                and offsets.ndim == 1
                and int(offsets.shape[0]) > 0
            ):
                safe_base = base.long().clamp(min=0, max=int(offsets.shape[0]) - 1)
                row_offsets = (
                    offsets[safe_base].clamp(min=0, max=int(ckpt.shape[1]) - 1).long()
                )
                return ckpt[:n_reqs].gather(1, row_offsets.unsqueeze(1)).squeeze(1)
        return base

    def _apply_kda_table_entry(
        self,
        tbl: object,
        req_ids: Sequence[str] | None,
    ) -> bool:
        """Store KDA block IDs from mapping or tensor `tbl` into `_req_kda_blocks`."""
        expected_kda = len(self.kda_raw) or (self.num_layers * 3 // KIMI_K3_ETP_PARTS)
        applied = False
        if isinstance(tbl, Mapping):
            for r_id, blocks in tbl.items():
                if (
                    isinstance(blocks, torch.Tensor)
                    and blocks.ndim == 1
                    and int(blocks.shape[0]) == expected_kda
                ):
                    self._req_kda_blocks[str(r_id)] = blocks
                    applied = True
            return applied
        if not isinstance(tbl, torch.Tensor) or req_ids is None:
            return False
        tbl_t = cast("torch.Tensor", tbl)
        if (
            tbl_t.ndim == _KDA_BLOCK_TABLE_NDIM
            and int(tbl_t.shape[0]) >= len(req_ids)
            and int(tbl_t.shape[1]) == expected_kda
        ):
            for idx, r_id in enumerate(req_ids):
                self._req_kda_blocks[str(r_id)] = tbl_t[idx]
            return True
        if (
            tbl_t.ndim == 1
            and len(req_ids) == 1
            and int(tbl_t.shape[0]) == expected_kda
        ):
            self._req_kda_blocks[str(req_ids[0])] = tbl_t
            return True
        return False

    def _extract_kda_table_from_attn_md(
        self,
        attn_md: object,
        req_ids: Sequence[str] | None,
    ) -> bool:
        """Build and store a per-request KDA block table from per-layer `attn_md`."""
        if not req_ids or not isinstance(attn_md, Mapping):
            return False
        by_kda: list[tuple[int, torch.Tensor]] = []
        n_reqs = len(req_ids)
        for layer_key, val in attn_md.items():
            try:
                idx = layer_index(str(layer_key))
            except ValueError:
                continue
            if not is_mla_layer(idx):
                eff = self._effective_mamba_state_blocks(val, n_reqs)
                if eff is not None:
                    by_kda.append((kda_state_index(idx), eff))
        if not by_kda:
            return False
        by_kda.sort(key=lambda item: item[0])
        stacked = torch.stack([t for _, t in by_kda], dim=1)
        return self._apply_kda_table_entry(stacked, req_ids)

    def _record_kda_block_tables(
        self,
        ctx: object,
        mla_md: object,
        kda_md: object,
        attn_md: object,
        req_ids: Sequence[str] | None,
    ) -> None:
        """Record per-request KDA block tables from layer metadata or `ctx`."""
        if self._extract_kda_table_from_attn_md(attn_md, req_ids):
            return
        for src in (ctx, kda_md, mla_md, attn_md):
            if src is None:
                continue
            tbl = _lookup_mapping_or_attr(src, "kda_block_tables")
            if tbl is not None and self._apply_kda_table_entry(tbl, req_ids):
                return

    def _sync_kda_for_eligible_step(self, req_ids: Sequence[str] | None) -> None:
        """Import KDA state from unified pools for any unsynced decode request."""
        if (
            not req_ids
            or self.kda_importer is None
            or self.conv_slots is None
            or self.recurrent_slots is None
        ):
            return
        for r_id in req_ids:
            r_key = str(r_id)
            if r_key not in self._slot_synced and r_key in self._req_kda_blocks:
                self.import_kda_state(r_key, self._req_kda_blocks[r_key])

    def _export_kda_for_ineligible_step(self) -> None:
        """Export synced private KDA slots back to unified pools before fallback."""
        if (
            not self._slot_synced
            or self.kda_importer is None
            or self.conv_slots is None
            or self.recurrent_slots is None
        ):
            return
        for r_key in list(self._slot_synced):
            if r_key in self.slot_map and r_key in self._req_kda_blocks:
                self.export_kda_state(r_key, self._req_kda_blocks[r_key])
        self._slot_synced.clear()

    def _allocate_active_from_metadata(
        self,
        md: object,
        mla_md: object = None,
        ctx: object = None,
    ) -> list[str] | None:
        """Allocate slots for active requests on eligible steps."""
        req_ids = self._resolve_step_req_ids(md, mla_md, ctx)
        if req_ids is not None:
            self.active_slots = [
                self.slot_map.allocate(req_id, group=self._group_of(req_id))
                for req_id in req_ids
            ]
        return req_ids

    def _capture_step_block_tables(self, mla_md: object, md: object = None) -> None:
        """Record active block tables from MLA attention metadata on eligible steps."""
        raw_tables = _lookup_mapping_or_attr(mla_md, "block_table")
        if raw_tables is None:
            raw_tables = _lookup_mapping_or_attr(mla_md, "block_tables")
        if raw_tables is None and md is not None:
            raw_tables = _lookup_mapping_or_attr(md, "block_table")
            if raw_tables is None:
                raw_tables = _lookup_mapping_or_attr(md, "block_tables")
        if isinstance(raw_tables, torch.Tensor):
            self._active_block_tables = raw_tables

    def _maybe_commit_accepted_verify(self, ctx: object) -> None:
        """Commit pending verify KDA states when `ctx` carries `accepted_counts`."""
        if self.last_verify_recurrent is None:
            return
        accepted = getattr(ctx, "accepted_counts", None)
        if isinstance(accepted, torch.Tensor):
            self.finish_verify(accepted)

    def on_verify_accepted(
        self,
        req_ids: Sequence[str] | None,
        num_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Commit speculative verify8 KDA states from `num_valid` accepted counts."""
        if self.last_verify_conv is None or self.last_verify_recurrent is None:
            return None
        slots: list[int] | None = None
        if req_ids is not None:
            slots = [
                self.slot_map.allocate(str(r), group=self._group_of(str(r)))
                for r in req_ids
            ]
        valid_i32 = num_valid.to(dtype=torch.int32)
        accepted_offsets = (
            valid_i32
            if bool(self.options.get("on_verify_zero_indexed", False))
            else (valid_i32 - 1).clamp(min=0)
        )
        return self.finish_verify(accepted_offsets, slots=slots)

    def decide(self, ctx: object, fwd: ForwardView) -> StepDecision | None:
        """Evaluate step eligibility and select the target bucket."""
        self._maybe_commit_accepted_verify(ctx)
        attn_md = fwd.attn_metadata or {}
        mla_md, kda_md = self._extract_attn_pair(attn_md)
        self._release_finished_from_metadata(attn_md, mla_md, ctx)
        req_ids = self._resolve_step_req_ids(attn_md, mla_md, ctx)
        self._record_kda_block_tables(ctx, mla_md, kda_md, attn_md, req_ids)
        input_ids = getattr(ctx, "input_ids", None)
        positions = getattr(ctx, "positions", None)
        if (
            input_ids is None
            or positions is None
            or not isinstance(input_ids, torch.Tensor)
        ):
            self._export_kda_for_ineligible_step()
            return None
        b_shape = fwd.batch_shape(mla_md, kda_md)
        decision = check_step_eligibility(
            b_shape,
            self.policy,
            compiled_buckets=set(fwd.steps.keys()),
            bucket_out_rows=self.bucket_out_rows,
            input_rows=int(input_ids.shape[0]),
        )
        if decision.eligible and decision.tokens_per_seq == VERIFY_TOKEN_COUNT:
            runner_obj = getattr(self._view, "_runner", None)
            explicit_hook_opt = self.options.get("verify_commit_hook_wired")
            has_commit_hook = (
                callable(getattr(ctx, "commit_verify", None))
                or getattr(ctx, "accepted_counts", None) is not None
                or (
                    bool(explicit_hook_opt)
                    if explicit_hook_opt is not None
                    else (
                        isinstance(ctx, StepContext)
                        or callable(
                            getattr(runner_obj, "_update_mamba_slot_read_offsets", None)
                        )
                    )
                )
            )
            if not has_commit_hook:
                self._export_kda_for_ineligible_step()
                decline = Decline(
                    "verify_commit_hook_unwired",
                    "verify8 step requires post-verify accepted_counts commit "
                    "hook on step context",
                )
                if self.reclaimed_expert_bytes > 0:
                    msg = (
                        "cannot fall back to standard model after reclaiming "
                        f"routed-expert HBM: {decline.code}"
                    )
                    raise RuntimeError(msg)
                return StepDecision(
                    eligible=False,
                    reason=decline.code,
                    num_reqs=decision.num_reqs,
                    tokens_per_seq=decision.tokens_per_seq,
                    bucket_t=0,
                )
        if decision.eligible:
            active_reqs = self._allocate_active_from_metadata(attn_md, mla_md, ctx)
            self._sync_kda_for_eligible_step(active_reqs)
            self._capture_step_block_tables(mla_md, attn_md)
        else:
            self._export_kda_for_ineligible_step()
            if self.reclaimed_expert_bytes > 0:
                msg = (
                    "cannot fall back to standard model after reclaiming "
                    f"routed-expert HBM: {decision.reason}"
                )
                raise RuntimeError(msg)
        return decision

    def forward(
        self, decision: StepDecision | None, ctx: object, fwd: ForwardView
    ) -> torch.Tensor:
        """Run the compiled bucket step callable and slice to active tokens."""
        self._maybe_commit_accepted_verify(ctx)
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
        if (
            self.is_tpu
            and self.conv_slots is not None
            and self.recurrent_slots is not None
            and self.raw_pools
        ):
            if self.last_verify_recurrent is not None:
                msg = (
                    "uncommitted deferred verify KDA state; call finish_verify "
                    "before running the next TPU step"
                )
                raise RuntimeError(msg)
            ids_i32 = ids.to(dtype=torch.int32)
            pos_i32 = pos.to(dtype=torch.int32)
            tables_tensor, slots_tensor = self.prepare_tpu_step_inputs(
                target_b, decision.tokens_per_seq, ids.device
            )
            step_inputs: tuple[torch.Tensor, ...] = (
                ids_i32,
                pos_i32,
                tables_tensor,
                slots_tensor,
                self.conv_slots,
                self.recurrent_slots,
                *self.raw_pools,
            )
            self._sync_tpu_tensors(step_inputs)
            raw_out = cast("Callable[..., Any]", step_fn)(*step_inputs)
            if isinstance(raw_out, tuple):
                self._sync_tpu_tensors(raw_out)
                out = self._adopt_tpu_step_outputs(raw_out, decision.tokens_per_seq)
            else:
                out = raw_out
        else:
            out = step_fn(ids, pos)
        n_out = fwd.num_tokens if fwd.num_tokens is not None else cur_n
        return out[:n_out]


__all__ = [
    "STREAM_STORE",
    "KimiK3MonoProgram",
    "KimiK3ProgramConfig",
    "ObservedMxfp4Entry",
    "RankWeightStreamer",
    "StreamStats",
    "StreamingWeightStore",
    "clear_observed_weights",
    "clear_streamed_weights",
    "commit_verify_kda_states",
    "create",
    "layout_request",
    "load_tensor",
    "observed_weights_snapshot",
    "pack_kda_state",
    "read_kda_states",
    "reclaim_routed_expert_hbm",
    "resolve_kernel_position",
    "select_verify_kda_rows",
    "streamed_staging_stats",
    "streamed_weights_snapshot",
    "unpack_kda_state",
    "write_kda_states",
]
