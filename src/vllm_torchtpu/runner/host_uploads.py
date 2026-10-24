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
"""Host-to-device uploads that keep their host tensor alive.

torch_tpu reads the host memory of a non-blocking upload when the transfer
runs, behind the device's queued work, so the host tensor must stay alive and
unchanged until a later event has completed. The TPU model runner does that
for its own uploads (`TPUModelRunner._upload`) and registers itself here;
`upload` hands it host tensors built elsewhere (multimodal batches, pooling),
and `hold` keeps the sources of uploads vLLM issues itself (encoder metadata,
penalty and data-parallel temporaries). With no runner registered, `upload`
copies synchronously and `hold` does nothing.
"""

import functools
import inspect
import sys
import weakref
from collections.abc import Callable
from typing import Any

import torch


def _unset() -> None:
    return None


_uploader: Callable[[], Callable[[torch.Tensor], Any] | None] = _unset
_holder: Callable[[], Callable[[torch.Tensor], Any] | None] = _unset
_device_type: str | None = None


def _ref(fn: Callable[[torch.Tensor], Any] | None):
    # A runner's bound methods are held weakly, so the registration does not
    # keep a torn-down runner (and its held tensors) alive.
    if fn is None:
        return _unset
    if inspect.ismethod(fn):
        return weakref.WeakMethod(fn)
    return lambda: fn  # noqa: E731


def set_uploader(
    uploader: Callable[[torch.Tensor], torch.Tensor] | None,
    device: torch.device | None,
    holder: Callable[[torch.Tensor], None] | None = None,
) -> None:
    """Route uploads to `device` through `uploader`, and `hold` through
    `holder`; None makes `upload` synchronous and `hold` a no-op."""
    global _uploader, _holder, _device_type
    _uploader = _ref(uploader)
    _holder = _ref(holder)
    _device_type = None if device is None else torch.device(device).type


def upload(host: torch.Tensor, device: torch.types.Device) -> torch.Tensor:
    """`host.to(device)`, held by the registered runner when it uploads to
    that device type, synchronous otherwise."""
    target = torch.device(device)
    if host.device.type != "cpu" or target.type == "cpu":
        return host.to(target)
    uploader = _uploader()
    if uploader is not None and target.type == _device_type:
        return uploader(host)
    return host.to(target)


def hold(host: torch.Tensor) -> None:
    """Keep `host` alive until the registered runner's next fence covering
    it has been waited on (for uploads issued by code that cannot route
    through `upload`)."""
    holder = _holder()
    if holder is not None:
        holder(host)


def _targets_device(args: tuple, kwargs: dict) -> bool:
    for arg in (*args, *kwargs.values()):
        if isinstance(arg, torch.Tensor) and arg.device.type != "cpu":
            return True
        if isinstance(arg, (torch.device, str)):
            try:
                if torch.device(arg).type != "cpu":
                    return True
            except (RuntimeError, TypeError):
                continue
    return False


class _HoldHostSources(torch.overrides.TorchFunctionMode):
    """Holds the host source of every host-to-device `Tensor.to` and
    `Tensor.copy_` issued inside, making a strided source contiguous first
    so that the held tensor is the one the copy reads."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (
            func is torch.Tensor.to
            and args
            and isinstance(args[0], torch.Tensor)
            and args[0].device.type == "cpu"
            and _targets_device(args[1:], kwargs)
        ):
            host = args[0].contiguous()
            out = func(host, *args[1:], **kwargs)
            hold(host)
            return out
        if (
            func is torch.Tensor.copy_
            and len(args) >= 2
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[1], torch.Tensor)
            and args[0].device.type != "cpu"
            and args[1].device.type == "cpu"
        ):
            host = args[1].contiguous()
            out = func(args[0], host, *args[2:], **kwargs)
            hold(host)
            return out
        return func(*args, **kwargs)


def hold_encoder_metadata_uploads(model: torch.nn.Module) -> None:
    """Hold the host tensors that vision towers upload while building their
    encoder metadata (rotary tables, window and position indices).

    `prepare_encoder_metadata` builds them on the host and moves them with
    non-blocking copies before dropping them. The interception covers only
    that call and is skipped while torch.compile traces.
    """
    for module in model.modules():
        prepare = getattr(module, "prepare_encoder_metadata", None)
        # A wrapped module holds a plain function here, so it is skipped.
        if not inspect.ismethod(prepare):
            continue

        def held(*args: Any, _prepare=prepare, **kwargs: Any) -> Any:
            if torch.compiler.is_compiling():
                return _prepare(*args, **kwargs)
            with _HoldHostSources():
                return _prepare(*args, **kwargs)

        module.prepare_encoder_metadata = functools.wraps(prepare)(held)


def patch_host_uploads() -> None:
    """Hold the host tensors vLLM uploads non-blocking and then drops.

    `async_tensor_h2d` and the multimodal batch transfer route through `upload`;
    the penalty prompt-token matrix and the data-parallel token-count all-reduce
    hold their temporary until the runner's fence. Modules that imported
    `async_tensor_h2d` by name are rebound as well.
    """
    from vllm.multimodal import inputs as mm_inputs
    from vllm.utils import torch_utils
    from vllm.utils.jsontree import json_map_leaves
    from vllm.v1.worker import dp_utils
    from vllm.v1.worker.gpu_input_batch import InputBatch

    original = torch_utils.async_tensor_h2d
    if getattr(original, "_tpu_held", False):
        return

    def async_tensor_h2d(
        data: Any,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dtype is None and out is not None:
            dtype = out.dtype
        host = torch.as_tensor(data, dtype=dtype)
        if out is not None:
            return out.copy_(host)
        assert device is not None, "must provide destination tensor or device"
        return upload(host, device)

    async_tensor_h2d._tpu_held = True  # type: ignore[attr-defined]

    def _nested_tensors_h2d(
        tensors: Any, device: torch.types.Device, *, pin_memory: bool = False
    ) -> Any:
        del pin_memory
        if device is None:
            return tensors
        return json_map_leaves(
            lambda x: upload(x, device) if isinstance(x, torch.Tensor) else x,
            tensors,
        )

    for module in list(sys.modules.values()):
        if getattr(module, "async_tensor_h2d", None) is original:
            module.__dict__["async_tensor_h2d"] = async_tensor_h2d
    mm_inputs._nested_tensors_h2d = _nested_tensors_h2d

    make_prompt_token_ids = InputBatch._make_prompt_token_ids_cpu_tensor

    def _make_prompt_token_ids_cpu_tensor(self: InputBatch) -> torch.Tensor:
        # The caller uploads it non-blocking during the state update, before
        # the step's fence is recorded.
        tensor = make_prompt_token_ids(self)
        hold(tensor)
        return tensor

    InputBatch._make_prompt_token_ids_cpu_tensor = _make_prompt_token_ids_cpu_tensor  # type: ignore[method-assign]

    run_ar = dp_utils._run_ar

    @functools.wraps(run_ar)
    def _run_ar(*args: Any, **kwargs: Any) -> torch.Tensor:
        # Uploads the ranks' token counts from a temporary for the all-reduce.
        with _HoldHostSources():
            return run_ar(*args, **kwargs)

    dp_utils._run_ar = _run_ar
