# SPDX-License-Identifier: Apache-2.0
"""TPU patches for the vLLM-Omni MiniMax-H3 pipeline.

The patches fall into two groups, in file order:

- Correctness: the pipeline's hand-rolled broadcasts and object collectives,
  the video VAE device context and the Triton-only indexed-modulation
  kernels, none of which run on torch_tpu as written upstream.
- Denoise performance: keep every DiT step on one set of shapes and keep
  host syncs out of the denoise loop (AdaLN cache, timestep table, RoPE
  table, finite checks, row gathers and scatters).

Each patch carries its rationale in a comment block at its entry point.
Patches only touch modules that are already loaded, do nothing without
vLLM-Omni or the MiniMax-H3 pipeline, and applying them twice changes
nothing.
"""

import contextlib
import functools
import hashlib
import inspect
import pickle
import sys
import types
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

PIPELINE_MODULE = "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3"
INDEXED_MODULATION_MODULE = "vllm_omni.diffusion.layers.indexed_modulation"
# Modules that bind the indexed-modulation functions by name at import time.
INDEXED_MODULATION_IMPORTERS = (
    "vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer",
)

# Header lengths of the upstream shape tensors (ndim + up to N-1 dims).
DIT_META_LEN = 5  # pipeline_minimax_h3._broadcast_tensor: torch.zeros(5)
ENCODER_META_LEN = 8  # MiniMaxH3Pipeline._encoder_group_broadcast_tensor: zeros(8)

_INT32_MAX = 2**31 - 1
_CPU = torch.device("cpu")


# -----------------------------------------------------------------------------
# Whole-tensor metadata / payload broadcast helpers.
# -----------------------------------------------------------------------------


def broadcast_int_list(
    values: Sequence[int] | None,
    length: int,
    *,
    src: int,
    device_group: Any,
    cpu_group: Any,
    device: torch.device,
) -> list[int]:
    """Broadcast a fixed-length list of ints from global rank ``src``.

    ``values`` is ``None`` on non-source ranks. The list is built on the host
    and sent as one whole tensor: over the gloo ``cpu_group`` if there is
    one, otherwise as one int32 device tensor over ``device_group`` between
    device syncs (see ``broadcast_tensor``), read back with one ``tolist()``.
    """
    if values is not None and len(values) != length:
        raise ValueError(f"expected {length} values to broadcast, got {len(values)}")
    if cpu_group is not None:
        if values is None:
            data = torch.zeros(length, dtype=torch.int64)
        else:
            data = torch.tensor([int(v) for v in values], dtype=torch.int64)
        torch.distributed.broadcast(data, src=src, group=cpu_group)
        return [int(v) for v in data.tolist()]
    if values is None:
        data = torch.zeros(length, dtype=torch.int32, device=device)
    else:
        host_values = [int(v) for v in values]
        if any(abs(v) > _INT32_MAX for v in host_values):
            raise ValueError(f"metadata value out of int32 range: {host_values}")
        data = torch.tensor(host_values, dtype=torch.int32, device=device)
    _synchronize(device)
    torch.distributed.broadcast(data, src=src, group=device_group)
    _synchronize(device)
    return [int(v) for v in data.cpu().tolist()]


def broadcast_shape(
    shape: Sequence[int] | None,
    meta_len: int,
    *,
    src: int,
    device_group: Any,
    cpu_group: Any,
    device: torch.device,
) -> tuple[int, ...]:
    """Broadcast ``[ndim, *shape, 0...]`` (``meta_len`` ints) from ``src``."""
    values = None
    if shape is not None:
        ndim = len(shape)
        if ndim + 1 > meta_len:
            # Upstream also fails here, on the slice assignment, before the
            # collective.
            raise RuntimeError(
                f"cannot broadcast a {ndim}-D tensor: the shape header holds "
                f"at most {meta_len - 1} dimensions"
            )
        values = [ndim, *(int(d) for d in shape)] + [0] * (meta_len - 1 - ndim)
    received = broadcast_int_list(
        values,
        meta_len,
        src=src,
        device_group=device_group,
        cpu_group=cpu_group,
        device=device,
    )
    ndim = received[0]
    return tuple(received[1 : ndim + 1])


def broadcast_tensor(
    tensor: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
    src: int,
    is_src: bool,
    device_group: Any,
    cpu_group: Any,
    meta_len: int,
) -> torch.Tensor:
    """Broadcast ``tensor`` from ``src``: shape header, then payload.

    Returns a ``dtype`` tensor on ``device`` on every rank, as the upstream
    helpers do.
    """
    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # Upstream: _broadcast_tensor and the encoder-group broadcast send the
    # payload with dist.broadcast over the device group
    # (pipeline_minimax_h3.py:612 and :1417-1490 at vllm-omni e1d77a3).
    #
    # Problem on TPU: a broadcast of a device tensor over the tpu_dist group
    # does not complete on all 8 v7x ranks. Some ranks finish and wait in the
    # next gloo all_gather_object while the others stay blocked in
    # torch_tpu::TpuMemcpyDtoHDirect reading the result. This hung all 8
    # diffusion workers in the Qwen3-VL text encoder, at the first host read
    # after the input-id broadcast (_get_rope_index's .tolist()). An
    # all_reduce over the same group completes; the broadcast also hangs when
    # every rank holds the same tensor. It completes when every rank fences
    # it with torch.accelerator.synchronize() before and after.
    #
    # Fix: with a gloo cpu_group, the source stages the payload on the host,
    # every rank receives into a host tensor and then copies it to the
    # device, so all device queues see the same ops. Without a cpu_group, the
    # device broadcast is fenced by device syncs on every rank.
    #
    # REMOVAL / FIX CONDITION:
    # Remove once torch_tpu's tpu_dist backend completes
    # torch.distributed.broadcast on every rank; then send the payload over
    # device_group again.
    # -------------------------------------------------------------------------
    shape = broadcast_shape(
        tuple(tensor.shape) if is_src else None,
        meta_len,
        src=src,
        device_group=device_group,
        cpu_group=cpu_group,
        device=device,
    )
    if cpu_group is not None:
        if is_src:
            payload = tensor.detach().to(device=_CPU, dtype=dtype).contiguous()
        else:
            payload = torch.empty(shape, dtype=dtype)
        torch.distributed.broadcast(payload, src=src, group=cpu_group)
        return payload.to(device)
    if is_src:
        output = tensor.to(device=device, dtype=dtype).contiguous()
    else:
        output = torch.empty(shape, device=device, dtype=dtype)
    _synchronize(device)
    torch.distributed.broadcast(output, src=src, group=device_group)
    _synchronize(device)
    return output


def _synchronize(device: torch.device) -> None:
    if device.type != "cpu":
        torch.accelerator.synchronize()


# -----------------------------------------------------------------------------
# Object collectives: fallback for groups without a gloo cpu_group. Device
# collectives are fenced with _synchronize, as in broadcast_tensor.
# -----------------------------------------------------------------------------


def _pickle_to_words(obj: Any, device: torch.device) -> tuple[torch.Tensor, int]:
    data = pickle.dumps(obj)
    nbytes = len(data)
    padded = bytearray(data)
    padded.extend(b"\0" * (-nbytes % 4))
    words = torch.frombuffer(padded, dtype=torch.int32).clone()
    return words.to(device), nbytes


def _words_to_obj(words: torch.Tensor, nbytes: int) -> Any:
    return pickle.loads(words.cpu().numpy().tobytes()[:nbytes])


def fallback_broadcast_object_list(
    object_list: list[Any],
    *,
    src: int,
    group: Any,
    device: torch.device,
) -> None:
    """``broadcast_object_list`` with a host-built length header + int32 payload.

    Same contract as ``torch.distributed.broadcast_object_list``: ``src`` is a
    global rank and non-source ranks get ``object_list`` filled in place.
    """
    is_src = torch.distributed.get_rank() == src
    words = None
    header = None
    if is_src:
        words, nbytes = _pickle_to_words(list(object_list), device)
        header = [nbytes, words.numel()]
    nbytes, nwords = broadcast_int_list(
        header, 2, src=src, device_group=group, cpu_group=None, device=device
    )
    if not is_src:
        words = torch.empty(nwords, dtype=torch.int32, device=device)
    _synchronize(device)
    torch.distributed.broadcast(words, src=src, group=group)
    _synchronize(device)
    if not is_src:
        for index, value in enumerate(_words_to_obj(words, nbytes)):
            object_list[index] = value


def fallback_all_gather_object(
    object_list: list[Any],
    obj: Any,
    *,
    group: Any,
    device: torch.device,
) -> None:
    """``all_gather_object`` with host-built length headers + int32 payloads."""
    world_size = torch.distributed.get_world_size(group)
    words, nbytes = _pickle_to_words(obj, device)
    local_header = torch.tensor(
        [nbytes, words.numel()], dtype=torch.int32, device=device
    )
    headers = [
        torch.zeros(2, dtype=torch.int32, device=device) for _ in range(world_size)
    ]
    _synchronize(device)
    torch.distributed.all_gather(headers, local_header, group=group)
    _synchronize(device)
    sizes = [[int(v) for v in header.cpu().tolist()] for header in headers]
    max_words = max(nwords for _, nwords in sizes)
    padding = torch.zeros(max_words - words.numel(), dtype=torch.int32, device=device)
    padded = torch.cat([words, padding])
    gathered = [
        torch.empty(max_words, dtype=torch.int32, device=device)
        for _ in range(world_size)
    ]
    _synchronize(device)
    torch.distributed.all_gather(gathered, padded, group=group)
    _synchronize(device)
    for index, (payload, (rank_nbytes, _)) in enumerate(zip(gathered, sizes)):
        object_list[index] = _words_to_obj(payload, rank_nbytes)


def _is_gloo_group(group: Any) -> bool:
    try:
        return str(torch.distributed.get_backend(group)) == "gloo"
    except Exception:
        return False


def _accelerator_device() -> torch.device:
    accelerator = torch.accelerator.current_accelerator()
    return torch.device(accelerator.type) if accelerator is not None else _CPU


class PipelineDist:
    """Module-local stand-in for ``torch.distributed`` in the H3 pipeline.

    Every attribute forwards to ``torch.distributed`` except the two object
    collectives, which route a device group to its gloo ``cpu_group`` (as
    vLLM and vLLM-Omni's own ``_all_gather_rank_values`` do). The process-wide
    ``torch.distributed`` functions are never replaced.
    """

    def __init__(self, cpu_group_for: Callable[[Any], Any]) -> None:
        self._cpu_group_for = cpu_group_for

    def __getattr__(self, name: str) -> Any:
        return getattr(torch.distributed, name)

    def cpu_group_for(self, group: Any) -> Any:
        return self._cpu_group_for(group)

    def broadcast_object_list(
        self,
        object_list: list[Any],
        src: int | None = None,
        group: Any = None,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        cpu_group = self._cpu_group_for(group)
        if cpu_group is not None:
            return torch.distributed.broadcast_object_list(
                object_list, src=src, group=cpu_group, **kwargs
            )
        if kwargs or _is_gloo_group(group):
            return torch.distributed.broadcast_object_list(
                object_list, src=src, group=group, device=device, **kwargs
            )
        return fallback_broadcast_object_list(
            object_list, src=src, group=group, device=_accelerator_device()
        )

    def all_gather_object(
        self, object_list: list[Any], obj: Any, group: Any = None
    ) -> None:
        cpu_group = self._cpu_group_for(group)
        if cpu_group is not None:
            return torch.distributed.all_gather_object(
                object_list, obj, group=cpu_group
            )
        if _is_gloo_group(group):
            return torch.distributed.all_gather_object(object_list, obj, group=group)
        return fallback_all_gather_object(
            object_list, obj, group=group, device=_accelerator_device()
        )


def world_cpu_group(pipeline: Any, group: Any) -> Any:
    """Gloo group of the DiT world coordinator if ``group`` belongs to it."""
    try:
        coordinator = pipeline.get_world_group()
    except Exception:
        return None
    cpu_group = getattr(coordinator, "cpu_group", None)
    if group is getattr(coordinator, "device_group", None) or group is cpu_group:
        return cpu_group
    return None


# -----------------------------------------------------------------------------
# Patches.
# -----------------------------------------------------------------------------


def patch_pipeline_tensor_broadcasts(pipeline: Any) -> bool:
    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # Upstream: _broadcast_tensor, MiniMaxH3Pipeline._encoder_group_broadcast_tensor
    # and _distribute_encode_inputs (pipeline_minimax_h3.py:584-613,
    # :1417-1490 at vllm-omni e1d77a3) build their shape and mask headers
    # with in-place slice writes into an int64 device tensor (shape[0] =
    # ndim; shape[1:ndim+1] = torch.tensor(...); mask[index] = ...) and read
    # them back with 0-d .item() on as_strided views.
    #
    # Problem on TPU: on 8x v7x the non-source encoder ranks received
    # ndim=0 ("input_ids must be 1-D, got []") and rank 0 halted the
    # SparseCore sequencer in
    # tt_jit_local_scalar_dense_to_copy_copy__as_strided.
    #
    # Fix: headers are built on the host and broadcast as one whole tensor
    # over the coordinator's gloo cpu_group, or as one int32 device tensor
    # if there is none. Payloads go through broadcast_tensor.
    #
    # REMOVAL / FIX CONDITION:
    # Remove once torch_tpu supports slice assignment and .item() on
    # as_strided int64 views feeding collectives, or vLLM-Omni sends these
    # headers over the coordinator's cpu_group (or as host-built whole
    # tensors) itself.
    # -------------------------------------------------------------------------
    cls = pipeline.MiniMaxH3Pipeline
    if getattr(pipeline, "_tpu_orig_broadcast_tensor", None) is not None:
        return False

    def _broadcast_tensor(
        tensor: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        group, rank, world_size = pipeline._dit_rank_world()
        if world_size == 1:
            if tensor is None:
                raise ValueError("source tensor is required for single-rank execution")
            return tensor.to(device=device, dtype=dtype)
        if rank == 0 and tensor is None:
            raise ValueError("rank 0 must provide a tensor to broadcast")
        return broadcast_tensor(
            tensor,
            dtype=dtype,
            device=device,
            src=0,
            is_src=rank == 0,
            device_group=group,
            cpu_group=world_cpu_group(pipeline, group),
            meta_len=DIT_META_LEN,
        )

    def _encoder_group_broadcast_tensor(
        self: Any,
        tensor: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Broadcast a tensor from encoder rank 0 over the encoder TP group."""
        group = self.text_encoder_group
        if group.world_size == 1:
            if tensor is None:
                raise ValueError("source tensor is required for single-rank execution")
            return tensor.to(device=device, dtype=dtype)
        if group.rank_in_group == 0 and tensor is None:
            raise ValueError("encoder rank 0 must provide a tensor to broadcast")
        return broadcast_tensor(
            tensor,
            dtype=dtype,
            device=device,
            src=group.ranks[0],
            is_src=group.rank_in_group == 0,
            device_group=group.device_group,
            cpu_group=getattr(group, "cpu_group", None),
            meta_len=ENCODER_META_LEN,
        )

    def _distribute_encode_inputs(
        self: Any,
        ids: torch.Tensor | None,
        vision_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fan out encode inputs from encoder rank 0 to the encoder TP ranks."""
        keys = (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        )
        key_dtypes = {
            "pixel_values": torch.bfloat16,
            "pixel_values_videos": torch.bfloat16,
            "image_grid_thw": torch.long,
            "video_grid_thw": torch.long,
        }
        group = self.text_encoder_group
        device = self.device
        if group.world_size == 1:
            if ids is None:
                raise ValueError("encoder rank 0 must produce input ids")
            return ids.to(device=device, dtype=torch.long)

        is_src = group.rank_in_group == 0
        mask = broadcast_int_list(
            [1 if key in vision_kwargs else 0 for key in keys] if is_src else None,
            len(keys),
            src=group.ranks[0],
            device_group=group.device_group,
            cpu_group=getattr(group, "cpu_group", None),
            device=device,
        )
        ids = self._encoder_group_broadcast_tensor(
            ids if is_src else None, dtype=torch.long, device=device
        )
        for index, key in enumerate(keys):
            if mask[index] == 0:
                continue
            source = vision_kwargs.get(key) if is_src else None
            vision_kwargs[key] = self._encoder_group_broadcast_tensor(
                source,
                dtype=key_dtypes[key],
                device=device,
            )
        return ids

    pipeline._tpu_orig_broadcast_tensor = pipeline._broadcast_tensor
    pipeline._broadcast_tensor = _broadcast_tensor
    cls._tpu_orig_encoder_group_broadcast_tensor = cls._encoder_group_broadcast_tensor
    cls._encoder_group_broadcast_tensor = _encoder_group_broadcast_tensor
    cls._tpu_orig_distribute_encode_inputs = cls._distribute_encode_inputs
    cls._distribute_encode_inputs = _distribute_encode_inputs
    logger.info(
        "Applied TPU patch: MiniMax-H3 host-built broadcast headers and "
        "host-staged payloads."
    )
    return True


def patch_pipeline_object_collectives(pipeline: Any) -> bool:
    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # Upstream: pipeline_minimax_h3 calls dist.broadcast_object_list (:562,
    # :2583, :2626, :2786) and dist.all_gather_object (:2659 at vllm-omni
    # e1d77a3) on the DiT world device group.
    #
    # Problem on TPU: for a backend that is neither NCCL nor gloo, torch
    # serializes the objects into accelerator tensors and slices them per
    # object with in-place copies and 0-d .item() size reads, the same
    # as_strided / .item() pattern that halts torch_tpu (see
    # patch_pipeline_tensor_broadcasts).
    #
    # Fix: the module's ``dist`` global becomes a PipelineDist shim that
    # sends these two collectives over the world coordinator's gloo
    # cpu_group, as vLLM-Omni already does in
    # diffusion_worker._all_gather_rank_values, with a host-built length
    # header protocol as fallback. torch.distributed itself is not modified.
    #
    # REMOVAL / FIX CONDITION:
    # Remove once vLLM-Omni passes get_world_group().cpu_group to these object
    # collectives, or torch_tpu supports torch's object-collective
    # serialization path on TPU process groups.
    # -------------------------------------------------------------------------
    if isinstance(pipeline.dist, PipelineDist):
        return False
    pipeline._tpu_orig_dist = pipeline.dist
    pipeline.dist = PipelineDist(lambda group: world_cpu_group(pipeline, group))
    logger.info("Applied TPU patch: MiniMax-H3 object collectives over gloo cpu_group.")
    return True


# -----------------------------------------------------------------------------
# Video VAE device context.
# -----------------------------------------------------------------------------


def _accelerator_is_tpu() -> bool:
    accelerator = torch.accelerator.current_accelerator()
    return accelerator is not None and accelerator.type == "tpu"


def _device_index(device: Any) -> int | None:
    """Index of ``device`` (int, str or torch.device); ``None`` stays ``None``."""
    if device is None or isinstance(device, int):
        return device
    return torch.device(device).index


class _TpuDeviceModuleView:
    """``torch.get_device_module()`` with ``device()`` via ``torch.accelerator``.

    ``device(d)`` returns ``torch.accelerator.device_index(index of d)``;
    every other attribute (``manual_seed``, ...) is the wrapped module's.
    """

    __slots__ = ("_target",)

    def __init__(self, target: Any) -> None:
        self._target = target

    def device(self, device: Any) -> Any:
        return torch.accelerator.device_index(_device_index(device))

    def __getattr__(self, name: str) -> Any:
        if name == "_target":
            raise AttributeError(name)
        return getattr(self._target, name)


class _VideoVAEDeviceModule:
    """Data descriptor for ``MiniMaxH3VideoVAE.device_module``.

    The module stays in the instance ``__dict__`` (``nn.Module.__setattr__``
    stores plain values with ``object.__setattr__``, which calls
    ``__set__``). On TPU, reads wrap it in ``_TpuDeviceModuleView``. Being a
    class attribute, it also covers VAEs built before the patch.
    """

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        if instance is None:
            return self
        try:
            module = instance.__dict__["device_module"]
        except KeyError:
            raise AttributeError("device_module") from None
        if _accelerator_is_tpu():
            return _TpuDeviceModuleView(module)
        return module

    def __set__(self, instance: Any, value: Any) -> None:
        instance.__dict__["device_module"] = value

    def __delete__(self, instance: Any) -> None:
        try:
            del instance.__dict__["device_module"]
        except KeyError:
            raise AttributeError("device_module") from None


def patch_video_vae_device_module(pipeline: Any) -> bool:
    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # Upstream: MiniMaxH3VideoVAE.__init__ sets self.device_module =
    # torch.get_device_module() (vae.py:377 at vllm-omni e1d77a3), which is
    # torch_tpu's TpuDeviceModule on TPU. encode_image and encode_video
    # reseed the keyframe encode with `with self.device_module.device(device):
    # self.device_module.manual_seed(...)` (vae.py:752, :921).
    #
    # Problem on TPU: TpuDeviceModule.device is _DeviceContext (torch_tpu
    # _internal/device/_device_module.py:222, :389), marked
    # internal_or_experimental(strict=True) for torch>=2.11, so the call
    # raised "RuntimeError: _DeviceContext is internal to TorchTPU and can
    # only be called by the PyTorch framework (torch and torch_tpu). Use
    # torch.accelerator.device_index instead." on the first-last-frame
    # (fl2va) task.
    #
    # Fix: on TPU, device_module reads as a view whose device() is
    # torch.accelerator.device_index(index) and whose other attributes are
    # TpuDeviceModule's (manual_seed is only experimental, not strict).
    # Other accelerators get the stored module unchanged.
    #
    # REMOVAL / FIX CONDITION:
    # Remove once vLLM-Omni uses torch.accelerator.device_index for the
    # keyframe-encode reseed, or torch_tpu allows external callers of
    # TpuDeviceModule.device.
    # -------------------------------------------------------------------------
    cls = getattr(pipeline, "MiniMaxH3VideoVAE", None)
    if cls is None:
        return False
    if isinstance(cls.__dict__.get("device_module"), _VideoVAEDeviceModule):
        return False
    cls._tpu_orig_device_module = cls.__dict__.get("device_module")
    cls.device_module = _VideoVAEDeviceModule()
    logger.info(
        "Applied TPU patch: MiniMax-H3 video VAE device context via "
        "torch.accelerator.device_index."
    )
    return True


# -----------------------------------------------------------------------------
# Indexed modulation (DiT AdaLN) PyTorch reference.
# -----------------------------------------------------------------------------


def indexed_scale_shift_reference(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Upstream CPU path of ``indexed_scale_shift_`` (updates ``x`` in place)."""
    x.copy_(
        (
            x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
        ).to(x.dtype)
    )
    return x


def indexed_gate_reference(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Upstream CPU path of ``indexed_gate``."""
    return (x + gate.index_select(0, indices) * other).to(x.dtype)


def rms_norm_indexed_scale_shift_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Upstream CPU path of ``rms_norm_indexed_scale_shift``."""
    input_dtype = x.dtype
    normalized = x.float()
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    normalized = (weight.float() * normalized).to(input_dtype)
    return (
        normalized * (1.0 + scale.index_select(0, indices))
        + shift.index_select(0, indices)
    ).to(input_dtype)


def indexed_gate_rms_norm_scale_shift_reference(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream CPU path of ``indexed_gate_rms_norm_scale_shift``."""
    input_dtype = residual.dtype
    residual_out = (residual + gate.index_select(0, indices) * branch).to(input_dtype)
    normalized = residual_out.float()
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    normalized = (weight.float() * normalized).to(input_dtype)
    modulated_out = (
        normalized * (1.0 + scale.index_select(0, indices))
        + shift.index_select(0, indices)
    ).to(input_dtype)
    return residual_out, modulated_out


# (public name, PyTorch reference, name of the tensor upstream dispatches on)
INDEXED_MODULATION_FUNCTIONS = (
    ("indexed_scale_shift_", indexed_scale_shift_reference, "x"),
    ("indexed_gate", indexed_gate_reference, "x"),
    ("rms_norm_indexed_scale_shift", rms_norm_indexed_scale_shift_reference, "x"),
    (
        "indexed_gate_rms_norm_scale_shift",
        indexed_gate_rms_norm_scale_shift_reference,
        "residual",
    ),
)


def _triton_available() -> bool:
    try:
        from vllm.triton_utils import HAS_TRITON
    except ImportError:
        return False
    return bool(HAS_TRITON)


def upstream_handles_device(tensor: torch.Tensor) -> bool:
    """Whether upstream indexed modulation can run on ``tensor``'s device.

    True on CPU (upstream's own PyTorch path) and on non-TPU devices when
    vLLM found Triton.
    """
    if tensor.is_cpu:
        return True
    return tensor.device.type != "tpu" and _triton_available()


def _indexed_modulation_wrapper(
    original: Callable[..., Any],
    reference: Callable[..., Any],
    dispatch_arg: str,
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tensor = args[0] if args else kwargs[dispatch_arg]
        if upstream_handles_device(tensor):
            return original(*args, **kwargs)
        return reference(*args, **kwargs)

    wrapper._tpu_reference = reference
    return wrapper


def patch_indexed_modulation() -> bool:
    # -------------------------------------------------------------------------
    # Upstream vLLM-Omni Root Cause Tracking:
    # Upstream: vllm_omni/diffusion/layers/indexed_modulation.py (vllm-omni
    # e1d77a3) does `from vllm.triton_utils import tl, triton`, which is
    # vLLM's TritonPlaceholder when Triton is unavailable. Its four public
    # functions (indexed_scale_shift_, indexed_gate,
    # rms_norm_indexed_scale_shift, indexed_gate_rms_norm_scale_shift) take
    # their PyTorch reference only under `if x.is_cpu:` / `if
    # residual.is_cpu:` and otherwise launch Triton kernels.
    #
    # Problem on TPU: the first DiT block failed at
    # minimax_h3_transformer.py:1059 -> indexed_modulation.py:278 with
    # "AttributeError: module 'triton' has no attribute 'next_power_of_2'".
    #
    # Fix: each function is wrapped. CPU tensors and Triton-capable devices
    # call the original; other devices (TPU) run a copy of upstream's CPU
    # path, expression by expression. minimax_h3_transformer binds the four
    # names at import (:49-54), so they are rebound there and in every
    # loaded vllm_omni module that holds an original. teacache/extractors.py
    # :1382 imports indexed_scale_shift_ at call time and gets the wrapper.
    #
    # REMOVAL / FIX CONDITION:
    # Remove once upstream indexed_modulation selects its PyTorch reference
    # path whenever Triton is unavailable (e.g. on HAS_TRITON / device
    # support) instead of checking `is_cpu`.
    # -------------------------------------------------------------------------
    module = sys.modules.get(INDEXED_MODULATION_MODULE)
    if module is None:
        logger.debug(
            "Skipping indexed modulation TPU patch: %s is not loaded.",
            INDEXED_MODULATION_MODULE,
        )
        return False
    importers = [
        loaded
        for loaded in (sys.modules.get(name) for name in INDEXED_MODULATION_IMPORTERS)
        if loaded is not None
    ]
    importers.extend(
        loaded
        for loaded_name, loaded in list(sys.modules.items())
        if loaded_name.startswith("vllm_omni.")
        and loaded is not None
        and loaded is not module
        and all(loaded is not importer for importer in importers)
    )
    changed = False
    for name, reference, dispatch_arg in INDEXED_MODULATION_FUNCTIONS:
        marker = f"_tpu_orig_{name}"
        original = getattr(module, marker, None)
        if original is None:
            original = getattr(module, name)
            setattr(module, marker, original)
            setattr(
                module,
                name,
                _indexed_modulation_wrapper(original, reference, dispatch_arg),
            )
            changed = True
        wrapper = getattr(module, name)
        for importer in importers:
            if getattr(importer, "__dict__", {}).get(name) is original:
                setattr(importer, name, wrapper)
                changed = True
    if changed:
        logger.info(
            "Applied TPU patch: MiniMax-H3 indexed modulation PyTorch reference "
            "on devices without Triton."
        )
    return changed


# -----------------------------------------------------------------------------
# DiT forward: exact AdaLN projection cache.
# -----------------------------------------------------------------------------

ADALN_CACHE_MODULE = "vllm_omni.diffusion.models.minimax_h3.adaln_cache"


def _is_tpu(value: Any) -> bool:
    """Whether a tensor, device or device string is on torch_tpu.

    Values with no device (e.g. ``None``) count as not TPU.
    """
    device = getattr(value, "device", value)
    if isinstance(device, str):
        return device.split(":", 1)[0] == "tpu"
    return getattr(device, "type", None) == "tpu"


def _loaded_module(name: str) -> types.ModuleType | None:
    """``sys.modules[name]`` if it is a loaded module (never imports it)."""
    module = sys.modules.get(name)
    return module if isinstance(module, types.ModuleType) else None


def patch_adaln_projection_cache() -> bool:
    # -------------------------------------------------------------------------
    # Upstream vLLM-Omni Root Cause Tracking:
    # Upstream: MiniMaxH3DiTModel.forward calls adaln_cache.prepare(t_emb)
    # after the embedders and the token refiner (minimax_h3_transformer.py
    # :1812-1813 at vllm-omni e1d77a3). ExactProjectionCache.prepare hashes
    # t_emb with tensor_digest, which copies it to the host
    # (exact_projection_cache.py:56-62, :175-180). Every AdaLN projection (50
    # blocks + the final layer, minimax_h3_transformer.py:885-903) then goes
    # through ExactProjectionCache.project, which at TP > 1 votes on the hit
    # with get_tp_group().all_reduce(vote).item() (exact_projection_cache.py
    # :136-138) before it runs or skips the GEMM. The cache is on by default
    # (256 MiB, minimax_h3_transformer.py:1294-1333).
    #
    # Problem on TPU: under torch_tpu eager DEFER_AND_FUSE every host read is
    # a device sync, so each forward pays one digest sync plus 51 all-reduce
    # + .item() round trips, and the host cannot queue block i+1 before block
    # i has finished. Hits only happen for a repeated schedule. They skip
    # ~2 ms of [M, 2688] x [2688, 96768 / TP] GEMMs per forward but run a
    # different (GEMM-free) program, which torch_tpu compiles again: on v7x-8
    # the second request spent 2.25 s / 2.85 s / 4.47 s on each of steps 1-2
    # against 0.27 / 0.5 / 1.26 s warm steps (256x256x4s / 480x480x4s /
    # 768x768x5s).
    #
    # Fix: on TPU, prepare() leaves the cache unarmed (_input = _key = None),
    # the state upstream itself uses for max_bytes=0 and grad mode. project()
    # then returns compute() before the vote, so every forward runs the same
    # projections without a host round trip. Outputs are bit-identical (a hit
    # returns a clone of the same projection). Other devices keep the cache.
    #
    # REMOVAL / FIX CONDITION:
    # Remove once upstream ExactProjectionCache decides hits without a host
    # digest and a per-projection TP vote (or skips arming itself on devices
    # where that sync costs more than the projection), or torch_tpu overlaps
    # host reads with already queued device work.
    # -------------------------------------------------------------------------
    module = _loaded_module(ADALN_CACHE_MODULE)
    cls = getattr(module, "MiniMaxH3RuntimeAdalnCache", None)
    if not isinstance(cls, type):
        msg = "Skipping MiniMax-H3 AdaLN cache TPU patch: %s is not loaded."
        if _loaded_module(PIPELINE_MODULE):
            # Skipping on only some TP ranks would hang the TP all-reduce vote.
            logger.warning(msg, ADALN_CACHE_MODULE)
        else:
            logger.debug(msg, ADALN_CACHE_MODULE)
        return False
    if "_tpu_orig_prepare" in cls.__dict__:
        return False
    original = cls.prepare

    @torch.compiler.disable
    @functools.wraps(original)
    def prepare(self: Any, embedding: torch.Tensor) -> None:
        if not _is_tpu(embedding):
            return original(self, embedding)
        self._input = None
        self._key = None
        # _sidecar_plan = None loses nothing on TPU because math_identity(device)
        # never matches a CUDA-built manifest.
        self._sidecar_plan = None
        return None

    cls._tpu_orig_prepare = original
    cls.prepare = prepare
    logger.info(
        "Applied TPU patch: MiniMax-H3 AdaLN projections without the host "
        "digest and TP hit vote."
    )
    return True


# -----------------------------------------------------------------------------
# Denoise loop: fixed-width pad for the timestep table.
# -----------------------------------------------------------------------------

DENOISE_LOOP_MODULE = "vllm_omni.diffusion.models.minimax_h3.denoise_loop"

# Distinct row timesteps one request without latent edits can carry
# (adaln_cache.MODES): the video target (also the text and padding rows), the
# audio target, the image/video condition and the audio reference.
H3_UNIQUE_TIMESTEP_ROWS = 4


def pad_unique_timesteps(unique_timesteps: torch.Tensor, rows: int) -> torch.Tensor:
    """Pad a sorted ``[M]`` timestep table to ``[rows]`` with its last value.

    ``inverse_indices`` only address rows ``< M``, so the padded rows are
    computed but never read, and the table stays sorted. Tables that are not
    1-D, are empty, or already have ``rows`` or more entries (latent-edit
    schedules) are returned unchanged. The pad runs on the host so that the
    unpadded width never reaches a traced device program.
    """
    if unique_timesteps.dim() != 1:
        return unique_timesteps
    count = unique_timesteps.shape[0]
    if count == 0 or count >= rows:
        return unique_timesteps

    if unique_timesteps.is_cpu:
        return torch.cat((unique_timesteps, unique_timesteps[-1:].expand(rows - count)))

    host = unique_timesteps.detach().cpu()
    padded = torch.cat((host, host[-1:].expand(rows - count)))
    return padded.to(unique_timesteps.device)


# -----------------------------------------------------------------------------
# Denoise loop: shared sync-free helpers (source pins, stream flush, deferred
# checks), the host timestep table and its forward_kwargs patch.
# -----------------------------------------------------------------------------

SCHEDULER_MODULE = (
    "vllm_omni.diffusion.models.minimax_h3.scheduling_minimax_h3_euler_ancestral"
)

# sha256 of inspect.getsource() of the vLLM-Omni e1d77a3 functions that the
# TPU paths below re-implement. A function with any other source keeps its
# upstream implementation, host syncs included, and logs a warning.
_SHA_LOOP = "dd782edbd769ab7c729a323d50bffa4f1e36f17ef461ad6c9b2b8d499ef45e7f"
_SHA_FWD_KW = "2bbefee104576679a2434913c7a6db95dc2f9418e20799dab44284f349a2222b"
_SHA_FILL_TS = "fdc969aba9c658a435ea40012a41f818fde4785f398a61893b015119d319d02e"
_SHA_FINITE = "759c5d1fc2af7ec655847a4ae7e74833cd093d92026438a63b3a609efa030e1e"
_SHA_UNIT_TS = "0d18f1abd26700e7f0cefc2f4010d481b88349a59d0bbd9532b7ee0ef76f800c"
_UPSTREAM_SOURCE_SHA256 = {
    "denoise_loop.minimax_h3_denoise_loop": _SHA_LOOP,
    "denoise_loop.MiniMaxH3DenoiseBranch.forward_kwargs": _SHA_FWD_KW,
    "denoise_loop.MiniMaxH3DenoiseBranch.fill_timesteps": _SHA_FILL_TS,
    "scheduling_minimax_h3_euler_ancestral._require_finite_tensor": _SHA_FINITE,
    "scheduling_minimax_h3_euler_ancestral._validate_unit_timestep": _SHA_UNIT_TS,
}
# Process globals: the diffusion worker runs the denoise loop single-threaded.
_SOURCE_MATCHES: dict[tuple[Any, str], bool] = {}


def _source_sha256(func: Any) -> str | None:
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode()).hexdigest()


def upstream_source_matches(func: Any, key: str) -> bool:
    """Whether ``func`` has the vLLM-Omni e1d77a3 source pinned under ``key``.

    Checked once per function object; a mismatch logs one warning and the
    caller keeps the upstream implementation.
    """
    expected = _UPSTREAM_SOURCE_SHA256[key]
    cache_key = (func, expected)
    matches = _SOURCE_MATCHES.get(cache_key)
    if matches is None:
        matches = _source_sha256(func) == expected
        _SOURCE_MATCHES[cache_key] = matches
        if not matches:
            logger.warning(
                "MiniMax-H3 TPU patch: %s differs from the vLLM-Omni e1d77a3 "
                "source its TPU copy mirrors; using the upstream implementation "
                "(with its per-step host syncs).",
                key,
            )
    return matches


@functools.cache
def _synchronize_tensors() -> Callable[..., None]:
    from vllm_torchtpu.utils import synchronize_tensors

    return synchronize_tensors


def _flush_stream(device: Any) -> None:
    """Start materializing every deferred torch_tpu op, without waiting.

    Called where upstream pays a blocking host sync (a boolean-mask index, a
    finite-check ``.item()``). A torch_tpu sync materializes the whole
    stream, so flushing at the same point cuts the same traced programs,
    hits the same compiled executables and gives the same numerics, but the
    host keeps queuing work instead of waiting. No-op off torch_tpu. It does
    not use ``_is_tpu``, so tests that force the TPU dispatch stay on CPU.
    """
    device = getattr(device, "device", device)
    if isinstance(device, str):
        kind = device.split(":", 1)[0]
    else:
        kind = getattr(device, "type", None)
    if kind == "tpu":
        _synchronize_tensors()(wait=False)


class _DeferredChecks:
    """Device-side results of upstream's finite / range checks for one run."""

    def __init__(self) -> None:
        self._checks: list[tuple[str, torch.Tensor, bool]] = []

    def record(self, message: str, flag: torch.Tensor, *, fails_if: bool) -> None:
        self._checks.append((message, flag, fails_if))

    def raise_first_failure(self) -> None:
        """One host read for the whole run; raise upstream's first error."""
        if not self._checks:
            return
        flags = torch.stack([flag for _, flag, _ in self._checks]).cpu().tolist()
        for (message, _, fails_if), flag in zip(self._checks, flags, strict=True):
            if bool(flag) == fails_if:
                raise ValueError(message)


_DEFERRED_CHECK_SCOPES: list[_DeferredChecks] = []


def _active_deferred_checks() -> _DeferredChecks | None:
    return _DEFERRED_CHECK_SCOPES[-1] if _DEFERRED_CHECK_SCOPES else None


def _branch_timestep_plan(
    branch: Any,
) -> tuple[torch.Tensor, tuple[int, ...], dict[tuple[int, ...], torch.Tensor]]:
    """Row groups of the branch's packed timestep table, built once on CPU.

    Group ids follow fill_timesteps' write order (denoise_loop.py:235-265 at
    vllm-omni e1d77a3), later writes winning as there: 0 = fill (text and
    padding rows, t_video), 1 = video target, 2 = video condition, 3 = audio
    target, 4 = audio reference. Returns the ``[seq_len]`` int64 group ids,
    the groups present, and the branch's cache of uploaded inverse indices.
    """
    plan = branch.__dict__.get("_tpu_timestep_plan")
    if plan is None:
        groups = torch.zeros(branch.seq_len, dtype=torch.long)
        img_pos = branch.img_pos.cpu()
        audio_pos = branch.audio_pos.cpu()
        update = branch.update_mask.cpu()
        audio_update = branch.audio_update_mask.cpu()
        for group, positions in (
            (1, img_pos[update]),
            (2, img_pos[~update]),
            (3, audio_pos[audio_update]),
            (4, audio_pos[~audio_update]),
        ):
            groups[positions] = group
        plan = (groups, tuple(torch.unique(groups).tolist()), {})
        branch._tpu_timestep_plan = plan
    return plan


def host_timestep_table(
    branch: Any,
    *,
    t_video: float,
    t_audio: float,
    imgvid_cond_timestep: float,
    audio_ref_cond_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream's ``torch.unique(timesteps, sorted=True, return_inverse=True)``.

    Computed on the host from the step's Python floats and the branch's row
    groups. The float32 group values are the ones fill_timesteps writes, so
    the sorted unique values and int64 inverse indices equal torch.unique's.
    Returns the CPU table padded to H3_UNIQUE_TIMESTEP_ROWS and the inverse
    indices on the branch device. The inverse indices are uploaded once per
    distinct group order and reused by later steps.
    """
    groups, present, inverse_by_order = _branch_timestep_plan(branch)
    locked = branch.locked_audio_rows is not None
    values = (
        float(t_video),
        float(t_video),
        float(imgvid_cond_timestep),
        1.0 if locked else float(t_audio),
        float(audio_ref_cond_timestep),
    )
    unique, order = torch.unique(
        torch.tensor([values[group] for group in present], dtype=torch.float32),
        sorted=True,
        return_inverse=True,
    )
    key = tuple(order.tolist())
    inverse = inverse_by_order.get(key)
    if inverse is None:
        # A host-to-device copy is a fresh device buffer, not a traced
        # constant, so every step's DiT program takes the same inputs.
        lut = torch.zeros(5, dtype=torch.int64)
        lut[list(present)] = order
        inverse = lut[groups].to(branch.device)
        inverse_by_order[key] = inverse
    return pad_unique_timesteps(unique, H3_UNIQUE_TIMESTEP_ROWS), inverse


_HOST_TABLE_KWARGS = frozenset(
    (
        "video_rows",
        "audio_rows",
        "t_video",
        "t_audio",
        "imgvid_cond_timestep",
        "audio_ref_cond_timestep",
    )
)
_EDIT_TIMESTEP_KWARGS = frozenset(("video_target_timesteps", "audio_target_timesteps"))

# Packed rows (branch.seq_len) from which forward_kwargs keeps upstream's
# table instead of the host one; see patch_denoise_branch_timestep_rows.
_HOST_TABLE_MAX_SEQ_LEN = 36_000


def _without_latent_edits(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Whether a forward_kwargs call is a plain step without latent edits."""
    if args or not kwargs.keys() >= _HOST_TABLE_KWARGS:
        return False
    if kwargs.keys() - _HOST_TABLE_KWARGS - _EDIT_TIMESTEP_KWARGS:
        return False
    return all(kwargs.get(name) is None for name in _EDIT_TIMESTEP_KWARGS)


def _forward_kwargs_host_table(
    branch: Any,
    *,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    t_video: float,
    t_audio: float,
    imgvid_cond_timestep: float,
    audio_ref_cond_timestep: float,
) -> dict[str, Any]:
    """MiniMaxH3DenoiseBranch.forward_kwargs (denoise_loop.py:182-215 at
    vllm-omni e1d77a3).

    Same x / audio_x ops as upstream; the timestep table comes from
    host_timestep_table instead of fill_timesteps + torch.unique.
    """
    x = branch.x_base.clone()
    x[0].index_copy_(0, branch.img_pos_dev, video_rows)
    audio_x = branch.audio_x_base.clone()
    audio_x[0].index_copy_(0, branch.audio_pos_dev, audio_rows)
    # Upstream's first fill_timesteps boolean index syncs here (:255).
    _flush_stream(branch.device)
    unique_timesteps, inverse_indices = host_timestep_table(
        branch,
        t_video=t_video,
        t_audio=t_audio,
        imgvid_cond_timestep=imgvid_cond_timestep,
        audio_ref_cond_timestep=audio_ref_cond_timestep,
    )
    return {
        **branch.static_kwargs,
        "x": x,
        "audio_x": audio_x,
        "unique_timesteps": unique_timesteps.to(branch.device),
        "inverse_indices": inverse_indices,
    }


def patch_denoise_branch_timestep_rows() -> bool:
    # -------------------------------------------------------------------------
    # Upstream vLLM-Omni Root Cause Tracking:
    # Upstream: MiniMaxH3DenoiseBranch.forward_kwargs builds the DiT's
    # timestep table with fill_timesteps and torch.unique(timesteps,
    # sorted=True, return_inverse=True) (denoise_loop.py:208 at vllm-omni
    # e1d77a3). Its length M is the number of distinct row timesteps and
    # changes with the step: at step 0 (sigma = 1) the video, audio, text and
    # padding rows all sit at t = 0 (M = 1 for t2va); later steps add the
    # audio timestep when the audio shift differs (M = 2), and fl2va / ref2va
    # add their condition timesteps. t_emb [M, 2688], every block's AdaLN
    # output [3M, 5376] (minimax_h3_transformer.py:898-906, :1709, :1815) and
    # the final layer's [M, 2 x 5376] depend on M.
    #
    # Problem on TPU, shapes: torch_tpu compiles every DiT program once per
    # M. On v7x-8 the first request compiled the whole DiT again at step 2:
    # 8.4 s / 10 s / 12 s for step 2 against 0.27 / 0.5 / 1.3 s warm steps
    # (256x256x4s / 480x480x4s / 768x768x5s).
    #
    # Problem on TPU, syncs: torch.unique (a dynamic output shape), the four
    # boolean-mask indexes in fill_timesteps (positions[update_mask] /
    # positions[~update_mask], denoise_loop.py:255, :265) and the pad's
    # .cpu() are six blocking host syncs per step, each draining the device
    # before the host can queue the step's DiT forward.
    #
    # Fix, shapes: on TPU the table is padded to H3_UNIQUE_TIMESTEP_ROWS rows
    # with its last value (the internal JAX port's fixed_num_unique), so
    # every step of every task without latent edits runs one set of shapes.
    # The padded rows only go through row-independent ops (time embedder,
    # AdaLN linears) and are never gathered: inverse_indices and the
    # combined_indices derived from them are unchanged. Wider tables (edit
    # schedules) and other devices keep the upstream table.
    #
    # Fix, syncs: without latent edits (video_target_timesteps and
    # audio_target_timesteps None) and below _HOST_TABLE_MAX_SEQ_LEN packed
    # rows, host_timestep_table computes the table on the host. The branch's
    # row groups are built once from its CPU masks; each step maps them to
    # the float32 values fill_timesteps writes, so the sorted unique values
    # and int64 inverse indices equal torch.unique's
    # (tests/omni/test_minimax_h3_sync.py checks t2va, fl2va, ref2va and
    # locked audio rows at every step of 8-step schedules). Both reach the
    # device as host-to-device copies, i.e. fresh buffers rather than traced
    # constants, so every step runs the same DiT program. One non-blocking
    # stream flush after x / audio_x keeps the program boundary of the first
    # sync it replaces. The x / audio_x ops are copied from upstream, so the
    # copy is pinned to the e1d77a3 sources of forward_kwargs and
    # fill_timesteps.
    #
    # Fix, sequence length: the host table only pays off on short sequences.
    # Warm s/step with / without it (v7x-8, TP8, 8 steps, frames bit-identical
    # either way): 256x256, 2,432 rows: 0.093 / 0.102; 768x768x5s, 21,760
    # rows: 0.750 / 0.751; 768x768x7s, 30,592 rows: 1.151 / 1.155;
    # 768x768x10s, 42,304 rows: 1.936 / 1.787, i.e. 148 ms/step (8%) slower
    # with it. Skipping the flush (1.943) or uploading fresh inverse indices
    # every step (1.929) does not recover it; the cause is not known. Longer
    # packed sequences therefore keep upstream's fill_timesteps +
    # torch.unique (and its syncs) and only get the pad.
    # _HOST_TABLE_MAX_SEQ_LEN sits between the 7 s and 10 s shapes;
    # branch.seq_len is a Python int, so the check costs no sync.
    #
    # REMOVAL / FIX CONDITION:
    # Remove the pad once upstream gives the timestep table a fixed width or
    # torch_tpu reuses one compiled program across M. Remove the host table
    # once upstream builds the table without device syncs; revisit the
    # length cut-off if the long-sequence slowdown is root-caused.
    # -------------------------------------------------------------------------
    module = _loaded_module(DENOISE_LOOP_MODULE)
    cls = getattr(module, "MiniMaxH3DenoiseBranch", None)
    if not isinstance(cls, type):
        msg = "Skipping MiniMax-H3 timestep table TPU patch: %s is not loaded."
        if _loaded_module(PIPELINE_MODULE):
            # Skipping on only some TP ranks would give them other DiT shapes.
            logger.warning(msg, DENOISE_LOOP_MODULE)
        else:
            logger.debug(msg, DENOISE_LOOP_MODULE)
        return False
    if "_tpu_orig_forward_kwargs" in cls.__dict__:
        return False
    original = cls.forward_kwargs

    @functools.wraps(original)
    def forward_kwargs(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if (
            _without_latent_edits(args, kwargs)
            and _is_tpu(getattr(self, "device", None))
            and upstream_source_matches(
                original, "denoise_loop.MiniMaxH3DenoiseBranch.forward_kwargs"
            )
            and upstream_source_matches(
                getattr(cls, "fill_timesteps", None),
                "denoise_loop.MiniMaxH3DenoiseBranch.fill_timesteps",
            )
            and self.seq_len < _HOST_TABLE_MAX_SEQ_LEN
        ):
            return _forward_kwargs_host_table(
                self, **{name: kwargs[name] for name in _HOST_TABLE_KWARGS}
            )
        result = original(self, *args, **kwargs)
        unique_timesteps = result.get("unique_timesteps")
        if isinstance(unique_timesteps, torch.Tensor) and _is_tpu(unique_timesteps):
            # forward_kwargs returns a fresh dict per call.
            result["unique_timesteps"] = pad_unique_timesteps(
                unique_timesteps, H3_UNIQUE_TIMESTEP_ROWS
            )
        return result

    cls._tpu_orig_forward_kwargs = original
    cls.forward_kwargs = forward_kwargs
    logger.info(
        "Applied TPU patch: MiniMax-H3 timestep table padded to %d rows and "
        "built on the host below %d packed rows.",
        H3_UNIQUE_TIMESTEP_ROWS,
        _HOST_TABLE_MAX_SEQ_LEN,
    )
    return True


# -----------------------------------------------------------------------------
# Denoise loop: per-request RoPE table.
# -----------------------------------------------------------------------------


def _rebind_vllm_omni_importers(name: str, original: Any, replacement: Any) -> bool:
    """Rebind ``name`` in every loaded ``vllm_omni.*`` module holding ``original``."""
    changed = False
    for loaded_name, loaded in list(sys.modules.items()):
        if not loaded_name.startswith("vllm_omni.") or loaded is None:
            continue
        if getattr(loaded, "__dict__", {}).get(name) is original:
            setattr(loaded, name, replacement)
            changed = True
    return changed


def patch_denoise_loop_rope_table() -> bool:
    # -------------------------------------------------------------------------
    # Upstream vLLM-Omni Root Cause Tracking:
    # Upstream: MiniMaxH3DiTModel.forward rebuilds the [S, 96] bf16 RoPE
    # table (rope frequencies of the packed position ids, then cos/sin) on
    # every call unless the caller passes static_kwargs["rope_table"]
    # (minimax_h3_transformer.py:1777-1789 at vllm-omni e1d77a3).
    # minimax_h3_denoise_loop builds that table once per denoise run with
    # MiniMaxH3DenoiseBranch.prepare_rope_table, but only on NPU
    # (denoise_loop.py:349-353; CUDA and CPU stay on the reference path of
    # the H100 accuracy suite).
    #
    # Problem on TPU: every step program repeats the position-frequency and
    # cos/sin work over S rows.
    #
    # Fix: on TPU the table is built once per request before the loop, as on
    # NPU. prepare_rope_table runs the same ops on the same inputs
    # (img_position_ids.narrow -> self.rope -> _build_rope_table,
    # minimax_h3_transformer.py:1469-1484), and the forward validates the
    # prepared table's shape, device and dtype (:1790-1794). The same
    # wrapper also sends TPU calls without latent edits to
    # minimax_h3_denoise_loop_tpu (see the block there).
    #
    # REMOVAL / FIX CONDITION:
    # Remove once upstream prepares the branch RoPE table on every platform
    # (or on TPU) in minimax_h3_denoise_loop.
    # -------------------------------------------------------------------------
    module = _loaded_module(DENOISE_LOOP_MODULE)
    if module is None:
        logger.debug(
            "Skipping MiniMax-H3 RoPE table TPU patch: %s is not loaded.",
            DENOISE_LOOP_MODULE,
        )
        return False
    marker = "_tpu_orig_minimax_h3_denoise_loop"
    original = getattr(module, marker, None)
    changed = False
    if original is None:
        original = getattr(module, "minimax_h3_denoise_loop", None)
        if not inspect.isfunction(original):
            return False

        @functools.wraps(original)
        def minimax_h3_denoise_loop(*args: Any, **kwargs: Any) -> Any:
            # Upstream's signature is keyword-only.
            positive = kwargs.get("positive")
            model = kwargs.get("model")
            on_tpu = _is_tpu(kwargs.get("device"))
            if positive is not None and model is not None and on_tpu:
                positive.prepare_rope_table(model)
            if on_tpu and not args:
                arguments = _tpu_denoise_loop_arguments(original, kwargs)
                if arguments is not None:
                    return minimax_h3_denoise_loop_tpu(module, **arguments)
            return original(*args, **kwargs)

        setattr(module, marker, original)
        module.minimax_h3_denoise_loop = minimax_h3_denoise_loop
        changed = True
    wrapper = module.minimax_h3_denoise_loop
    # pipeline_minimax_h3 binds the loop by name at import (:119-121).
    changed = (
        _rebind_vllm_omni_importers("minimax_h3_denoise_loop", original, wrapper)
        or changed
    )
    if changed:
        logger.info(
            "Applied TPU patch: MiniMax-H3 RoPE table built once per denoise run, "
            "denoise steps without host syncs."
        )
    return changed


# -----------------------------------------------------------------------------
# Denoise loop: deferred finite checks and the sync-free TPU loop.
# -----------------------------------------------------------------------------


def _deferred_require_finite_tensor(
    original: Callable[[torch.Tensor, str], None],
) -> Callable[[torch.Tensor, str], None]:
    @functools.wraps(original)
    def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
        checks = _active_deferred_checks()
        if checks is None or not _is_tpu(tensor):
            return original(tensor, name)
        # scheduling_minimax_h3_euler_ancestral.py:10-12.
        checks.record(
            f"{name} must be finite", torch.isfinite(tensor).all(), fails_if=False
        )
        _flush_stream(tensor.device)
        return None

    return _require_finite_tensor


def _deferred_validate_unit_timestep(
    original: Callable[[torch.Tensor, str], None],
) -> Callable[[torch.Tensor, str], None]:
    @functools.wraps(original)
    def _validate_unit_timestep(timestep: torch.Tensor, name: str) -> None:
        checks = _active_deferred_checks()
        if (
            checks is None
            or not isinstance(timestep, torch.Tensor)
            or not _is_tpu(timestep)
        ):
            return original(timestep, name)
        # scheduling_minimax_h3_euler_ancestral.py:15-23.
        if not torch.is_floating_point(timestep):
            raise ValueError(f"{name} must be a floating point tensor")
        checks.record(
            f"{name} must be finite", torch.isfinite(timestep).all(), fails_if=False
        )
        _flush_stream(timestep.device)
        out_of_range = (timestep < 0) | (timestep > 1)
        checks.record(f"{name} must be in [0, 1]", out_of_range.any(), fails_if=True)
        _flush_stream(timestep.device)
        return None

    return _validate_unit_timestep


_DEFERRED_VALIDATORS = (
    (
        "_require_finite_tensor",
        _deferred_require_finite_tensor,
        "scheduling_minimax_h3_euler_ancestral._require_finite_tensor",
    ),
    (
        "_validate_unit_timestep",
        _deferred_validate_unit_timestep,
        "scheduling_minimax_h3_euler_ancestral._validate_unit_timestep",
    ),
)


def patch_scheduler_finite_checks() -> bool:
    # -------------------------------------------------------------------------
    # Upstream vLLM-Omni Root Cause Tracking:
    # Upstream: minimax_h3_rf_v_to_x0 and minimax_h3_euler_eta0_step validate
    # every tensor they touch with _require_finite_tensor /
    # _validate_unit_timestep, i.e. bool(torch.isfinite(t).all().item()) and
    # bool(out_of_range.any().item())
    # (scheduling_minimax_h3_euler_ancestral.py:10-23, called from :60-68 and
    # :85-101 at vllm-omni e1d77a3). minimax_h3_denoise_loop runs both per
    # stream per step (denoise_loop.py:420-449): 8 .item() reads per stream,
    # 16 per step.
    #
    # Problem on TPU: under torch_tpu eager DEFER_AND_FUSE each .item() is a
    # blocking sync, so the host waits for the device 16 times per step
    # before it can queue the next step.
    #
    # Fix: inside minimax_h3_denoise_loop_tpu, and only there, a check on a
    # TPU tensor records its device-side result and starts a non-blocking
    # stream flush in place of the sync (same program boundaries). The loop
    # reads all results once after its last step and raises upstream's
    # ValueError for the first failed check, in upstream's order and with
    # upstream's message. The error now surfaces after the loop instead of
    # in the failing step. Calls outside that loop, and on other devices,
    # run upstream's checks. The checks are re-implemented, so they are
    # pinned to their e1d77a3 sources; vllm_omni modules that imported them
    # by name are rebound.
    #
    # REMOVAL / FIX CONDITION:
    # Remove once upstream validates the denoise state without per-tensor host
    # reads (e.g. one fused device flag per step or run), or torch_tpu can read
    # a scalar without draining the stream.
    # -------------------------------------------------------------------------
    module = _loaded_module(SCHEDULER_MODULE)
    if module is None:
        logger.debug(
            "Skipping MiniMax-H3 deferred finite check TPU patch: %s is not loaded.",
            SCHEDULER_MODULE,
        )
        return False
    changed = False
    for name, deferred, key in _DEFERRED_VALIDATORS:
        marker = f"_tpu_orig_{name}"
        original = getattr(module, marker, None)
        if original is None:
            original = getattr(module, name, None)
            if not inspect.isfunction(original):
                continue
            if not upstream_source_matches(original, key):
                continue
            setattr(module, marker, original)
            setattr(module, name, deferred(original))
            changed = True
        changed = (
            _rebind_vllm_omni_importers(name, original, getattr(module, name))
            or changed
        )
    if changed:
        logger.info(
            "Applied TPU patch: MiniMax-H3 denoise finite checks read once per run."
        )
    return changed


def _tpu_denoise_loop_arguments(
    original: Callable[..., Any], kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    """Bound arguments if minimax_h3_denoise_loop_tpu covers this call."""
    if not upstream_source_matches(original, "denoise_loop.minimax_h3_denoise_loop"):
        return None
    try:
        bound = inspect.signature(original).bind(**kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    if (
        arguments.get("video_edit") is not None
        or arguments.get("audio_edit") is not None
        or arguments.get("sampler") not in (None, "euler", "res_multistep")
    ):
        return None
    return arguments


def _branch_row_indices(
    positive: Any, device: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Indices of the update / condition rows of one branch, uploaded once.

    ``(video target, video condition, audio target, audio reference)`` rows,
    from the branch's CPU masks, so a step indexes rows without a nonzero.
    """
    cached = positive.__dict__.get("_tpu_row_indices")
    if cached is None or cached[0] != str(device):
        update = positive.update_mask.cpu()
        audio_update = positive.audio_update_mask.cpu()
        indices = tuple(
            torch.nonzero(mask).flatten().to(device)
            for mask in (update, ~update, audio_update, ~audio_update)
        )
        cached = (str(device), indices)
        positive._tpu_row_indices = cached
    return cached[1]


def minimax_h3_denoise_loop_tpu(
    module: types.ModuleType,
    *,
    model: Any,
    positive: Any,
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    keyframe_cond_rows: torch.Tensor | None,
    audio_ref_rows: torch.Tensor | None,
    video_edit: Any,
    audio_edit: Any,
    sigmas_video: list[float],
    sigmas_audio: list[float],
    device: torch.device,
    imgvid_cond_noise_aug_for_inference: float,
    audio_cond_noise_aug_for_inference: float,
    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None,
    step_profiler: Callable[[int], contextlib.AbstractContextManager] | None,
    sampler: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # -------------------------------------------------------------------------
    # Upstream vLLM-Omni Root Cause Tracking:
    # Upstream: after the DiT forward, minimax_h3_denoise_loop gathers and
    # scatters the target and condition rows with boolean masks:
    # v.float()[update], rows[update] (x2 per stream), rows[update] = new and
    # rows[~update] = anchor (denoise_loop.py:416-453 at vllm-omni e1d77a3).
    #
    # Problem on TPU: each boolean index runs a nonzero with a data-dependent
    # output shape, i.e. a blocking device sync: 8 per step for t2va, 9-10
    # with condition anchors. Together with the 16 finite-check reads
    # (patch_scheduler_finite_checks) and the 6 timestep-table syncs
    # (patch_denoise_branch_timestep_rows), the host waited for the device
    # about 30 times per step before it could queue more work; a warm step
    # took 0.136 s on v7x-8 (256x256x4s, TP8).
    #
    # Fix: a copy of upstream's loop body for TPU without latent edits
    # (euler and res_multistep). The masks become index tensors once per
    # request (_branch_row_indices), and the gathers / scatters use
    # index_select / index_put_, which read and write exactly the same rows
    # (index_put_ is what a boolean setitem dispatches to after its nonzero).
    # Every removed sync is replaced by _flush_stream at the same point, so
    # torch_tpu cuts and compiles the same programs and the output is
    # bit-identical, but the host no longer waits. The finite checks are
    # read once after the loop. Helpers are looked up on the upstream module
    # at call time. res_multistep keeps the inline isfinite() syncs of
    # H3SampleSolver.step (sampling.py:71-78). The copy is pinned to the
    # e1d77a3 source of minimax_h3_denoise_loop; any other source, latent
    # edits, other samplers and other devices run upstream's loop.
    #
    # REMOVAL / FIX CONDITION:
    # Remove once upstream's denoise loop indexes rows with precomputed
    # indices and validates without per-step host reads, or torch_tpu runs
    # boolean-mask indexing with a host-known mask without a device sync.
    # -------------------------------------------------------------------------
    if len(sigmas_video) != len(sigmas_audio):
        raise ValueError("video/audio sigma schedules must have equal length")
    if len(sigmas_video) < 2:
        raise ValueError("sigma schedules need at least 2 entries")
    if module.current_omni_platform.is_npu():
        positive.prepare_rope_table(model)
    video_rows, audio_rows, cond_anchor, audio_anchor = (
        module.minimax_h3_prepare_denoise_rows(
            positive=positive,
            initial_video_rows=initial_video_rows,
            initial_audio_rows=initial_audio_rows,
            keyframe_cond_rows=keyframe_cond_rows,
            audio_ref_rows=audio_ref_rows,
            device=device,
        )
    )
    update = positive.update_mask_dev
    audio_update = positive.audio_update_mask_dev
    update_idx, cond_idx, audio_update_idx, audio_ref_idx = _branch_row_indices(
        positive, device
    )

    video_solver = module.create_h3_sample_solver(sampler, sigmas_video)
    audio_solver = module.create_h3_sample_solver(sampler, sigmas_audio)
    num_steps = len(sigmas_video) - 1
    checks = _DeferredChecks()
    _DEFERRED_CHECK_SCOPES.append(checks)
    try:
        for step in range(num_steps):
            module.check_request_cancellation()
            step_cm = (
                step_profiler(step)
                if step_profiler is not None
                else contextlib.nullcontext()
            )
            with step_cm:
                s_v = sigmas_video[step]
                s_a = sigmas_audio[step]
                module.minimax_h3_publish_denoise_progress(step, s_v, num_steps)
                t_v, t_a = 1.0 - s_v, 1.0 - s_a
                imgvid_cond_t = max(t_v, float(imgvid_cond_noise_aug_for_inference))
                audio_ref_cond_t = max(t_a, float(audio_cond_noise_aug_for_inference))

                # Without an edit these return (rows, None).
                model_video_rows, video_target_timesteps = (
                    module.minimax_h3_prepare_edit_rows(
                        video_rows, update, None, t_v, imgvid_cond_t, sigma=s_v
                    )
                )
                model_audio_rows, audio_target_timesteps = (
                    module.minimax_h3_prepare_edit_rows(
                        audio_rows, audio_update, None, t_a, audio_ref_cond_t, sigma=s_a
                    )
                )

                fk = positive.forward_kwargs(
                    video_rows=model_video_rows,
                    audio_rows=model_audio_rows,
                    t_video=t_v,
                    t_audio=t_a,
                    imgvid_cond_timestep=imgvid_cond_t,
                    audio_ref_cond_timestep=audio_ref_cond_t,
                    video_target_timesteps=video_target_timesteps,
                    audio_target_timesteps=audio_target_timesteps,
                )
                with torch.inference_mode():
                    v_video, v_audio = model(**fk)
                # Each _flush_stream below stands where upstream syncs. A flush
                # materializes every live pending tensor, so the locals keep
                # upstream's lifetimes: v_video / v_audio stay referenced, the
                # float copies only until they are gathered.
                v_video_float = v_video.float()
                _flush_stream(device)  # :416 v.float()[update]
                mv_video_t = v_video_float.index_select(0, update_idx)
                del v_video_float
                v_audio_float = v_audio.float()
                _flush_stream(device)  # :417 v.float()[audio_update]
                mv_audio_t = v_audio_float.index_select(0, audio_update_idx)
                del v_audio_float

                _flush_stream(device)  # :421 video_rows[update]
                x0_video = module.minimax_h3_rf_v_to_x0(
                    video_rows.index_select(0, update_idx),
                    mv_video_t,
                    torch.tensor(t_v, dtype=torch.float32, device=device),
                )
                _flush_stream(device)  # :431 video_rows[update]
                new_target = video_solver.step(
                    video_rows.index_select(0, update_idx), x0_video, step
                )
                video_rows = video_rows.clone()
                _flush_stream(device)  # :433 video_rows[update] = new_target
                video_rows.index_put_((update_idx,), new_target)
                if cond_anchor is not None:
                    _flush_stream(device)  # :435 video_rows[~update] = anchor
                    video_rows.index_put_((cond_idx,), cond_anchor)

                _flush_stream(device)  # :439 audio_rows[audio_update]
                x0_audio = module.minimax_h3_rf_v_to_x0(
                    audio_rows.index_select(0, audio_update_idx),
                    mv_audio_t,
                    torch.tensor(t_a, dtype=torch.float32, device=device),
                )
                _flush_stream(device)  # :449 audio_rows[audio_update]
                new_audio = audio_solver.step(
                    audio_rows.index_select(0, audio_update_idx), x0_audio, step
                )
                audio_rows = audio_rows.clone()
                _flush_stream(device)  # :451 audio_rows[audio_update] = ...
                audio_rows.index_put_(
                    (audio_update_idx,),
                    new_audio
                    if positive.locked_audio_rows is None
                    else positive.locked_audio_rows,
                )
                if audio_anchor is not None:
                    _flush_stream(device)  # :453 audio_rows[~audio_update] = ...
                    audio_rows.index_put_((audio_ref_idx,), audio_anchor)
                if on_step is not None:
                    on_step(step, video_rows, audio_rows)
                module.check_request_cancellation(synchronize=True)
    except Exception:
        # An earlier failed deferred check wins over whatever was raised
        # later (an inline H3SampleSolver.step check, a request abort, a
        # device error); otherwise re-raise the original exception.
        try:
            checks.raise_first_failure()
        except ValueError:
            raise
        except Exception:  # noqa: S110
            pass
        raise
    finally:
        _DEFERRED_CHECK_SCOPES.remove(checks)
    checks.raise_first_failure()

    module.minimax_h3_publish_denoise_progress(None, None, None)
    return video_rows, audio_rows


def apply_minimax_h3_patches() -> bool:
    """Patch the MiniMax-H3 pipeline module if it is already loaded.

    ``apply_omni_model_specific_patches`` runs for every Omni model, so the
    pipeline is looked up in ``sys.modules`` rather than imported: unrelated
    workers never import MiniMax-H3. In an H3 worker the diffusion registry
    imports ``pipeline_minimax_h3`` (and with it ``minimax_h3_transformer``
    and ``indexed_modulation``) before ``MiniMaxH3Pipeline.__init__`` builds
    the attention layers that trigger these patches.
    """
    pipeline = sys.modules.get(PIPELINE_MODULE)
    if pipeline is None:
        logger.debug(
            "Skipping MiniMax-H3 TPU patches: %s is not loaded.", PIPELINE_MODULE
        )
        return False
    patched = patch_pipeline_tensor_broadcasts(pipeline)
    patched = patch_pipeline_object_collectives(pipeline) or patched
    patched = patch_video_vae_device_module(pipeline) or patched
    patched = patch_indexed_modulation() or patched
    patched = patch_adaln_projection_cache() or patched
    patched = patch_denoise_branch_timestep_rows() or patched
    patched = patch_scheduler_finite_checks() or patched
    return patch_denoise_loop_rope_table() or patched
