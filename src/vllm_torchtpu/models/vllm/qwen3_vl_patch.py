# SPDX-License-Identifier: Apache-2.0
"""Model-specific patches and PyTorch XLA op overrides for Qwen3-VL on TPU."""

import contextlib
import functools
import inspect
import sys
import threading
from typing import TYPE_CHECKING, Optional

import torch

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
else:
    ModelConfig = None

logger = init_logger(__name__)

_qwen3_vl_torch_ops_depth = 0
_orig_torch_ops: dict = {}
_torch_ops_lock = threading.RLock()


def _is_qwen3_vl_model(model_config: Optional["ModelConfig"]) -> bool:
    """Check if the provided ModelConfig corresponds to a Qwen3-VL model."""
    if model_config is None:
        return False
    hf_config = model_config.hf_config
    if hf_config is not None:
        model_type = getattr(hf_config, "model_type", "")
        if model_type and "qwen3_vl" in str(model_type).lower():
            return True
        architectures = getattr(hf_config, "architectures", [])
        if any("qwen3vl" in str(arch).lower() for arch in architectures):
            return True
    model_name = getattr(model_config, "model", "")
    if "qwen3-vl" in str(model_name).lower() or "qwen3vl" in str(
            model_name).lower():
        return True
    return False


def _patched_masked_scatter_(self: torch.Tensor, mask: torch.Tensor,
                             source: torch.Tensor) -> torch.Tensor:
    if self.device.type == "tpu":
        mask_bool = mask.bool().expand_as(self)
        flat_self = self.reshape(-1)
        flat_mask = mask_bool.reshape(-1)
        flat_source = source.reshape(-1)
        indices = torch.nonzero(flat_mask, as_tuple=True)[0]
        num_indices = indices.numel()
        if flat_source.numel() < num_indices:
            raise RuntimeError(
                f"Number of source elements ({flat_source.numel()}) is less than "
                f"number of selected elements in mask ({num_indices})")
        res = flat_self.clone()
        res[indices] = flat_source[:num_indices].to(dtype=self.dtype,
                                                    device=self.device)
        self.copy_(res.reshape(self.shape))
        return self
    orig_fn = _orig_torch_ops.get("tensor_masked_scatter_",
                                  torch.Tensor.masked_scatter_)
    return orig_fn(self, mask, source)


def _patched_masked_scatter(input: torch.Tensor, mask: torch.Tensor,
                            source: torch.Tensor) -> torch.Tensor:
    if isinstance(input, torch.Tensor) and input.device.type == "tpu":
        broadcast_input, broadcast_mask = torch.broadcast_tensors(input, mask)
        mask_bool = broadcast_mask.bool()
        flat_input = broadcast_input.reshape(-1)
        flat_mask = mask_bool.reshape(-1)
        flat_source = source.reshape(-1)
        indices = torch.nonzero(flat_mask, as_tuple=True)[0]
        num_indices = indices.numel()
        if flat_source.numel() < num_indices:
            raise RuntimeError(
                f"Number of source elements ({flat_source.numel()}) is less than "
                f"number of selected elements in mask ({num_indices})")
        res = flat_input.clone()
        res[indices] = flat_source[:num_indices].to(dtype=input.dtype,
                                                    device=input.device)
        return res.reshape(broadcast_input.shape)
    orig_fn = _orig_torch_ops.get("masked_scatter", torch.masked_scatter)
    return orig_fn(input, mask, source)


def _patched_repeat_interleave(input: torch.Tensor,
                               repeats: torch.Tensor | int,
                               dim: int | None = None,
                               output_size: int | None = None) -> torch.Tensor:
    if (isinstance(input, torch.Tensor) and input.device.type == "tpu"
            and input.dtype
            in (torch.int64, torch.int32, torch.int16, torch.bool)):
        input_cpu = input.cpu()
        repeats_cpu = repeats.cpu() if isinstance(repeats,
                                                  torch.Tensor) else repeats
        orig_fn = _orig_torch_ops.get("repeat_interleave",
                                      torch.repeat_interleave)
        res_cpu = orig_fn(input_cpu,
                          repeats_cpu,
                          dim=dim,
                          output_size=output_size)
        return res_cpu.to(input.device)
    orig_fn = _orig_torch_ops.get("repeat_interleave", torch.repeat_interleave)
    return orig_fn(input, repeats, dim=dim, output_size=output_size)


def _patched_cumsum(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    orig_fn = _orig_torch_ops.get("cumsum", torch.cumsum)
    if (isinstance(input, torch.Tensor) and input.device.type == "tpu"
            and input.dtype
            in (torch.int64, torch.int32, torch.int16, torch.bool)):
        return orig_fn(input.cpu(), *args, **kwargs).to(input.device)
    return orig_fn(input, *args, **kwargs)


def _patched_tensor_cumsum(self: torch.Tensor, *args,
                           **kwargs) -> torch.Tensor:
    orig_fn = _orig_torch_ops.get("tensor_cumsum", torch.Tensor.cumsum)
    if (self.device.type == "tpu" and self.dtype
            in (torch.int64, torch.int32, torch.int16, torch.bool)):
        return orig_fn(self.cpu(), *args, **kwargs).to(self.device)
    return orig_fn(self, *args, **kwargs)


@contextlib.contextmanager
def _scoped_qwen3_vl_torch_ops():
    """Temporarily override PyTorch ops with TPU workarounds during vision execution."""
    global _qwen3_vl_torch_ops_depth, _orig_torch_ops
    with _torch_ops_lock:
        if _qwen3_vl_torch_ops_depth == 0:
            _orig_torch_ops["masked_scatter"] = torch.masked_scatter
            _orig_torch_ops[
                "tensor_masked_scatter"] = torch.Tensor.masked_scatter
            _orig_torch_ops[
                "tensor_masked_scatter_"] = torch.Tensor.masked_scatter_
            _orig_torch_ops["repeat_interleave"] = torch.repeat_interleave
            _orig_torch_ops["cumsum"] = torch.cumsum
            _orig_torch_ops["tensor_cumsum"] = torch.Tensor.cumsum
            has_tb = hasattr(torch, "_C") and hasattr(torch._C, "_TensorBase")
            _orig_torch_ops["has_tb"] = has_tb
            if has_tb:
                _orig_torch_ops["tb_masked_scatter"] = getattr(
                    torch._C._TensorBase, "masked_scatter", None)
                _orig_torch_ops["tb_masked_scatter_"] = getattr(
                    torch._C._TensorBase, "masked_scatter_", None)

            torch.masked_scatter = _patched_masked_scatter
            torch.Tensor.masked_scatter = _patched_masked_scatter
            torch.Tensor.masked_scatter_ = _patched_masked_scatter_
            torch.repeat_interleave = _patched_repeat_interleave
            torch.cumsum = _patched_cumsum
            torch.Tensor.cumsum = _patched_tensor_cumsum
            if has_tb:
                try:
                    if _orig_torch_ops["tb_masked_scatter"] is not None:
                        torch._C._TensorBase.masked_scatter = _patched_masked_scatter
                    if _orig_torch_ops["tb_masked_scatter_"] is not None:
                        torch._C._TensorBase.masked_scatter_ = _patched_masked_scatter_
                except (TypeError, AttributeError):
                    pass

        _qwen3_vl_torch_ops_depth += 1
    try:
        yield
    finally:
        with _torch_ops_lock:
            _qwen3_vl_torch_ops_depth -= 1
            if _qwen3_vl_torch_ops_depth == 0:
                torch.masked_scatter = _orig_torch_ops["masked_scatter"]
                torch.Tensor.masked_scatter = _orig_torch_ops[
                    "tensor_masked_scatter"]
                torch.Tensor.masked_scatter_ = _orig_torch_ops[
                    "tensor_masked_scatter_"]
                torch.repeat_interleave = _orig_torch_ops["repeat_interleave"]
                torch.cumsum = _orig_torch_ops["cumsum"]
                torch.Tensor.cumsum = _orig_torch_ops["tensor_cumsum"]
                if _orig_torch_ops.get("has_tb"):
                    try:
                        if _orig_torch_ops["tb_masked_scatter"] is not None:
                            torch._C._TensorBase.masked_scatter = _orig_torch_ops[
                                "tb_masked_scatter"]
                        if _orig_torch_ops["tb_masked_scatter_"] is not None:
                            torch._C._TensorBase.masked_scatter_ = _orig_torch_ops[
                                "tb_masked_scatter_"]
                    except (TypeError, AttributeError):
                        pass
                _orig_torch_ops.clear()


def _patch_qwen3_vl_get_rope_index(modeling) -> None:
    """Patch Qwen3VLModel.get_rope_index with signature binding and CPU RoPE evaluation."""
    target_cls = getattr(modeling, "Qwen3VLModel", None)
    if target_cls is None:
        return

    _orig_get_rope_index = getattr(target_cls, "get_rope_index", None)
    if _orig_get_rope_index is None:
        return
    if getattr(_orig_get_rope_index, "_tpu_qwen3_vl_rope_index_patch", False):
        return

    def fix_grid(g):
        if g is not None and isinstance(g, torch.Tensor):
            if g.numel() == 0:
                return None
            if g.dim() == 1:
                return g.unsqueeze(0)
        return g

    def to_cpu(t, target_device_box):
        if isinstance(t, torch.Tensor):
            if target_device_box[0] is None and t.device.type != "cpu":
                target_device_box[0] = t.device
            return t.cpu()
        if isinstance(t, (tuple, list)):
            return type(t)(to_cpu(x, target_device_box) for x in t)
        if isinstance(t, dict):
            return {k: to_cpu(v, target_device_box) for k, v in t.items()}
        return t

    def to_device(r, target_device):
        if target_device is None:
            return r
        if isinstance(r, torch.Tensor):
            return r.to(target_device)
        if isinstance(r, (tuple, list)):
            return type(r)(to_device(x, target_device) for x in r)
        if isinstance(r, dict):
            return {k: to_device(v, target_device) for k, v in r.items()}
        return r

    sig = inspect.signature(_orig_get_rope_index)

    @functools.wraps(_orig_get_rope_index)
    def patched_get_rope_index(self, *args, **kwargs):
        try:
            bound = sig.bind(self, *args, **kwargs)
        except TypeError:
            bound = sig.bind_partial(self, *args, **kwargs)
        bound.apply_defaults()

        for param_name in ("image_grid_thw", "video_grid_thw"):
            if param_name in bound.arguments and bound.arguments[
                    param_name] is not None:
                bound.arguments[param_name] = fix_grid(
                    bound.arguments[param_name])

        target_device_box = [None]
        for k, v in list(bound.arguments.items()):
            if k == "self":
                continue
            bound.arguments[k] = to_cpu(v, target_device_box)

        call_args = bound.args[1:]
        call_kwargs = {k: v for k, v in bound.kwargs.items() if k != "self"}

        with _scoped_qwen3_vl_torch_ops():
            res = _orig_get_rope_index(self, *call_args, **call_kwargs)

        if target_device_box[0] is not None:
            res = to_device(res, target_device_box[0])
        return res

    patched_get_rope_index._tpu_qwen3_vl_rope_index_patch = True
    target_cls.get_rope_index = patched_get_rope_index
    logger.info("Applied TPU patch: Qwen3VLModel.get_rope_index.")


def _patch_qwen3_vl_vision_attention(modeling) -> None:
    """Patch Qwen3VLVisionAttention.forward to ensure cu_seqlens is on CPU."""
    target_cls = getattr(modeling, "Qwen3VLVisionAttention", None)
    if target_cls is None:
        return

    _orig_vision_attn_forward = getattr(target_cls, "forward", None)
    if _orig_vision_attn_forward is None:
        return
    if getattr(_orig_vision_attn_forward, "_tpu_qwen3_vl_vision_attn_patch",
               False):
        return

    sig = inspect.signature(_orig_vision_attn_forward)

    @functools.wraps(_orig_vision_attn_forward)
    def patched_vision_attn_forward(self, *args, **kwargs):
        try:
            bound = sig.bind(self, *args, **kwargs)
        except TypeError:
            bound = sig.bind_partial(self, *args, **kwargs)
        bound.apply_defaults()

        if "cu_seqlens" in bound.arguments:
            cu_seqlens = bound.arguments["cu_seqlens"]
            if isinstance(cu_seqlens,
                          torch.Tensor) and cu_seqlens.device.type != "cpu":
                bound.arguments["cu_seqlens"] = cu_seqlens.cpu()

        call_args = bound.args[1:]
        call_kwargs = {k: v for k, v in bound.kwargs.items() if k != "self"}

        with _scoped_qwen3_vl_torch_ops():
            return _orig_vision_attn_forward(self, *call_args, **call_kwargs)

    patched_vision_attn_forward._tpu_qwen3_vl_vision_attn_patch = True
    target_cls.forward = patched_vision_attn_forward
    logger.info("Applied TPU patch: Qwen3VLVisionAttention.forward.")


_VISION_MODEL_CANDIDATE_NAMES = (
    "Qwen3VLVisionModel",  # Standard Hugging Face transformers
    "Qwen3VisionTransformerPretrainedModel",  # HF PretrainedModel alternate convention
    "Qwen3_VisionTransformerPretrainedModel",  # Alternate HF naming convention
    "Qwen3_VisionTransformer",  # vLLM internal model naming
    "Qwen3VisionTransformer",  # Custom / alias modeling
)


def _patch_qwen3_vl_vision_transformer(modeling) -> None:
    """Patch vision encoder forward to scope PyTorch ops during vision encoding."""
    target_classes = []
    for name in _VISION_MODEL_CANDIDATE_NAMES:
        cls = getattr(modeling, name, None)
        if cls is not None and cls not in target_classes:
            target_classes.append(cls)

    def _make_patched_vision_forward(orig_forward):

        @functools.wraps(orig_forward)
        def patched_vision_forward(self, *args, **kwargs):
            with _scoped_qwen3_vl_torch_ops():
                return orig_forward(self, *args, **kwargs)

        patched_vision_forward._tpu_qwen3_vl_vision_transformer_patch = True
        return patched_vision_forward

    for target_cls in target_classes:
        _orig_vision_forward = getattr(target_cls, "forward", None)
        if _orig_vision_forward is None:
            continue
        if getattr(_orig_vision_forward,
                   "_tpu_qwen3_vl_vision_transformer_patch", False):
            continue

        target_cls.forward = _make_patched_vision_forward(_orig_vision_forward)
        logger.info("Applied TPU patch: %s.forward.", target_cls.__name__)


def maybe_patch_qwen3_vl(
        model_config: Optional["ModelConfig"] = None) -> bool | None:
    """Apply model-specific and PyTorch XLA op-level patches for Qwen3-VL on TPU.

    This function applies the following scoped patches:
    1. Qwen3VLModel.get_rope_index:
       - Grid Metadata Fix (`fix_grid`): Unsqueezes 1D `image_grid_thw`/`video_grid_thw`
         tensors (shape [3] -> [1, 3]) using signature binding.
       - CPU RoPE Evaluation: Evaluates 3D RoPE (Time, Height, Width) position ID
         calculation on host CPU and returns results back to TPU, preventing XLA
         lowering and device sync errors during 3D RoPE index construction.
    2. Qwen3VLVisionAttention.forward:
       - Moves `cu_seqlens` (cumulative visual patch sequence lengths) to CPU to avoid
         device mismatch in vision attention kernels without fabricating fallback sequences.
    3. Vision-encoder scoped PyTorch op overrides:
       - torch.masked_scatter / masked_scatter_: Replaces native TPU masked_scatter
         with strict 1D sequential indexing decomposition, validating source tensor size
         and eliminating unlowered XLA scan HLO nodes.
       - torch.repeat_interleave & torch.cumsum: Evaluates integer/bool TPU tensors on CPU
         scoped strictly to vision-encoder execution instead of process-wide mutation.
    """
    if not _is_qwen3_vl_model(model_config):
        return False

    modeling = None
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
    except ImportError as e:
        for mod_name, mod in sys.modules.items():
            if "qwen3_vl" in mod_name and hasattr(mod, "Qwen3VLModel"):
                modeling = mod
                break
        if modeling is None:
            logger.warning(
                "Qwen3-VL model detected but transformers.models.qwen3_vl "
                "could not be imported or located: %s", e)
            return None

    _patch_qwen3_vl_get_rope_index(modeling)
    _patch_qwen3_vl_vision_attention(modeling)
    _patch_qwen3_vl_vision_transformer(modeling)
    return None
