# SPDX-License-Identifier: Apache-2.0
"""TPU patches for the vLLM-Omni MiniMax-H3 pipeline.

The patches fall into two groups, in file order:

- Correctness: the pipeline's hand-rolled broadcasts and object collectives,
  the video VAE device context and the Triton-only indexed-modulation
  kernels, none of which run on torch_tpu as written upstream.
- Denoise performance: keep host syncs out of the denoise loop (AdaLN
  cache).

Each patch carries its rationale in a comment block at its entry point.
Patches only touch modules that are already loaded, do nothing without
vLLM-Omni or the MiniMax-H3 pipeline, and applying them twice changes
nothing.
"""

import functools
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
    return patch_adaln_projection_cache() or patched
