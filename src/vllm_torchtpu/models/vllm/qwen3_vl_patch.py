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
"""Model-specific patches and PyTorch XLA op overrides for Qwen3-VL on TPU."""

import contextlib
import functools
import inspect
import sys
import threading
from typing import TYPE_CHECKING, Optional

import numpy as np
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
    model_name = str(getattr(model_config, "model", "")).lower()
    return "qwen3-vl" in model_name or "qwen3vl" in model_name


def _patched_masked_scatter_(
    self: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
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
                f"number of selected elements in mask ({num_indices})"
            )
        res = flat_self.clone()
        res[indices] = flat_source[:num_indices].to(
            dtype=self.dtype, device=self.device
        )
        self.copy_(res.reshape(self.shape))
        return self
    orig_fn = _orig_torch_ops.get(
        "tensor_masked_scatter_", torch.Tensor.masked_scatter_
    )
    return orig_fn(self, mask, source)


def _patched_masked_scatter(
    input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
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
                f"number of selected elements in mask ({num_indices})"
            )
        res = flat_input.clone()
        res[indices] = flat_source[:num_indices].to(
            dtype=input.dtype, device=input.device
        )
        return res.reshape(broadcast_input.shape)
    orig_fn = _orig_torch_ops.get("masked_scatter", torch.masked_scatter)
    return orig_fn(input, mask, source)


def _patched_repeat_interleave(
    input: torch.Tensor,
    repeats: torch.Tensor | int,
    dim: int | None = None,
    output_size: int | None = None,
) -> torch.Tensor:
    if (
        isinstance(input, torch.Tensor)
        and input.device.type == "tpu"
        and input.dtype in (torch.int64, torch.int32, torch.int16, torch.bool)
    ):
        input_cpu = input.cpu()
        repeats_cpu = repeats.cpu() if isinstance(repeats, torch.Tensor) else repeats
        orig_fn = _orig_torch_ops.get("repeat_interleave", torch.repeat_interleave)
        res_cpu = orig_fn(input_cpu, repeats_cpu, dim=dim, output_size=output_size)
        return res_cpu.to(input.device)
    orig_fn = _orig_torch_ops.get("repeat_interleave", torch.repeat_interleave)
    return orig_fn(input, repeats, dim=dim, output_size=output_size)


def _patched_cumsum(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    orig_fn = _orig_torch_ops.get("cumsum", torch.cumsum)
    if (
        isinstance(input, torch.Tensor)
        and input.device.type == "tpu"
        and input.dtype in (torch.int64, torch.int32, torch.int16, torch.bool)
    ):
        return orig_fn(input.cpu(), *args, **kwargs).to(input.device)
    return orig_fn(input, *args, **kwargs)


def _patched_tensor_cumsum(self: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    orig_fn = _orig_torch_ops.get("tensor_cumsum", torch.Tensor.cumsum)
    if self.device.type == "tpu" and self.dtype in (
        torch.int64,
        torch.int32,
        torch.int16,
        torch.bool,
    ):
        return orig_fn(self.cpu(), *args, **kwargs).to(self.device)
    return orig_fn(self, *args, **kwargs)


@contextlib.contextmanager
def _scoped_qwen3_vl_torch_ops():
    """Temporarily override PyTorch ops with TPU workarounds during vision execution."""
    global _qwen3_vl_torch_ops_depth, _orig_torch_ops
    with _torch_ops_lock:
        if _qwen3_vl_torch_ops_depth == 0:
            _orig_torch_ops["masked_scatter"] = torch.masked_scatter
            _orig_torch_ops["tensor_masked_scatter"] = torch.Tensor.masked_scatter
            _orig_torch_ops["tensor_masked_scatter_"] = torch.Tensor.masked_scatter_
            _orig_torch_ops["repeat_interleave"] = torch.repeat_interleave
            _orig_torch_ops["cumsum"] = torch.cumsum
            _orig_torch_ops["tensor_cumsum"] = torch.Tensor.cumsum
            has_tb = hasattr(torch, "_C") and hasattr(torch._C, "_TensorBase")
            _orig_torch_ops["has_tb"] = has_tb
            if has_tb:
                _orig_torch_ops["tb_masked_scatter"] = getattr(
                    torch._C._TensorBase, "masked_scatter", None
                )
                _orig_torch_ops["tb_masked_scatter_"] = getattr(
                    torch._C._TensorBase, "masked_scatter_", None
                )

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
                torch.Tensor.masked_scatter = _orig_torch_ops["tensor_masked_scatter"]
                torch.Tensor.masked_scatter_ = _orig_torch_ops["tensor_masked_scatter_"]
                torch.repeat_interleave = _orig_torch_ops["repeat_interleave"]
                torch.cumsum = _orig_torch_ops["cumsum"]
                torch.Tensor.cumsum = _orig_torch_ops["tensor_cumsum"]
                if _orig_torch_ops.get("has_tb"):
                    try:
                        if _orig_torch_ops["tb_masked_scatter"] is not None:
                            torch._C._TensorBase.masked_scatter = _orig_torch_ops[
                                "tb_masked_scatter"
                            ]
                        if _orig_torch_ops["tb_masked_scatter_"] is not None:
                            torch._C._TensorBase.masked_scatter_ = _orig_torch_ops[
                                "tb_masked_scatter_"
                            ]
                    except (TypeError, AttributeError):
                        pass
                _orig_torch_ops.clear()


def _patch_qwen3_vl_get_rope_index(modeling) -> None:
    """Patch Qwen3VLModel.get_rope_index with signature binding and
    CPU RoPE evaluation."""
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
            if (
                param_name in bound.arguments
                and bound.arguments[param_name] is not None
            ):
                bound.arguments[param_name] = fix_grid(bound.arguments[param_name])

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
    if getattr(_orig_vision_attn_forward, "_tpu_qwen3_vl_vision_attn_patch", False):
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
            if isinstance(cu_seqlens, torch.Tensor) and cu_seqlens.device.type != "cpu":
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
        if getattr(
            _orig_vision_forward, "_tpu_qwen3_vl_vision_transformer_patch", False
        ):
            continue

        target_cls.forward = _make_patched_vision_forward(_orig_vision_forward)
        logger.info("Applied TPU patch: %s.forward.", target_cls.__name__)


def maybe_patch_qwen3_vl(model_config: Optional["ModelConfig"] = None) -> bool | None:
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
         device mismatch in vision attention kernels without fabricating
         fallback sequences.
    3. Vision-encoder scoped PyTorch op overrides:
       - torch.masked_scatter / masked_scatter_: Replaces native TPU masked_scatter
         with strict 1D sequential indexing decomposition, validating source tensor size
         and eliminating unlowered XLA scan HLO nodes.
       - torch.repeat_interleave & torch.cumsum: Evaluates integer/bool TPU tensors
         on CPU scoped strictly to vision-encoder execution instead of
         process-wide mutation.
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
                "could not be imported or located: %s",
                e,
            )
            return None

    _patch_qwen3_vl_get_rope_index(modeling)
    _patch_qwen3_vl_vision_attention(modeling)
    _patch_qwen3_vl_vision_transformer(modeling)
    return None


def _has_qwen3_vl_vision(model_config: "ModelConfig") -> bool:
    """Whether the model has a Qwen3-VL vision tower (Qwen3-VL, its MoE, Cosmos3)."""
    vision_config = getattr(model_config.hf_config, "vision_config", None)
    return "qwen3_vl" in str(getattr(vision_config, "model_type", ""))


def maybe_patch_qwen3_vl_video_prompt(
    model_config: Optional["ModelConfig"] = None,
) -> bool | None:
    """Expand Qwen3-VL videos in the prompt format the models were trained on.

    The chat template writes a video as ``<|vision_start|><|video_pad|><|vision_end|>``.
    The processor Qwen3-VL and Cosmos3 were trained with (transformers < 5.10)
    replaces that whole triple with ``<t seconds><|vision_start|>...<|vision_end|>``
    per frame. vLLM follows transformers >= 5.10, which replaces only the bare
    video token and so leaves an outer ``<|vision_start|>``/``<|vision_end|>``
    pair around every video, two tokens the model never saw (Cosmos3-Super: 474
    prompt tokens vs the official 472, and a different answer).
    """
    if model_config is None or not _has_qwen3_vl_vision(model_config):
        return False
    from vllm.model_executor.models import qwen3_vl

    processor_cls = qwen3_vl.Qwen3VLMultiModalProcessor
    if not getattr(processor_cls, "_tpu_trained_video_prompt", False):
        processor_cls._expands_only_video_token = staticmethod(
            lambda hf_processor: False
        )
        processor_cls._tpu_trained_video_prompt = True
        logger.info(
            "Applied TPU patch: Qwen3-VL videos expand in the trained prompt format."
        )
    return None


@functools.lru_cache(maxsize=1024)
def _bilinear_tables(
    h: int, w: int, num_grid_per_side: int, merge: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """`pos_embed_interpolate_native`'s corner indices and weights for an h x w
    grid, on the host, in spatial-merge order: `[4, h * w]` each."""
    h_idxs = torch.linspace(0, num_grid_per_side - 1, h, dtype=torch.float32)
    w_idxs = torch.linspace(0, num_grid_per_side - 1, w, dtype=torch.float32)
    h_floor, w_floor = h_idxs.to(torch.long), w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)
    dh_grid, dw_grid = torch.meshgrid(h_idxs - h_floor, w_idxs - w_floor, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")
    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01
    h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    indices = h_grid * num_grid_per_side + w_grid
    weights = torch.stack([w00, w01, w10, w11]).to(dtype)

    def merge_order(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(4, h // merge, merge, w // merge, merge)
        return x.permute(0, 1, 3, 2, 4).reshape(4, -1)

    return merge_order(indices), merge_order(weights)


def _prepare_encoder_budget_inputs(
    self,
    mm_kwargs: dict | None,
    template: dict[str, torch.Tensor | None],
    max_batch_size: int,
    max_frames_per_batch: int,
) -> dict[str, torch.Tensor | None]:
    """The vision tower's inputs for one budget graph, at its template's shapes.

    Upstream builds the position embeddings and rotary tables with eager device
    ops shaped by each item's grid, so every new image layout compiles them,
    and the fused graph can differ from one request to the next. Here the
    layout-only parts (corner indices, bilinear weights, rotary position ids)
    are built on the host and padded to the budget's rows; the device only
    gathers and sums at the budget's shape. Padding rows come out zero, like
    the template's. Device pixels pass through for the manager to pad; host
    pixels (a list of item slices, see `_select_encoder_items`) are copied
    into one host buffer of the budget's rows and uploaded once.
    `mm_kwargs=None` builds a stand-in that fills the budget, with host-path
    pixels (warm-up).
    """
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )

    visual = self.visual
    merge = visual.spatial_merge_size
    rows = template["pos_embeds"].shape[0]
    if mm_kwargs is None:
        side = int(rows**0.5) // merge * merge
        grid, is_video = [(1, side, side)], False
        pixels = _budget_pixels([], template["pixel_values"], visual.device)
    else:
        grid = [
            tuple(int(v) for v in g) for g in self._get_grid_thw_by_modality(mm_kwargs)
        ]
        is_video = self.get_input_modality(mm_kwargs) == "video"
        pixels = self._get_pixel_values_by_modality(mm_kwargs)
        if isinstance(pixels, list):
            pixels = _budget_pixels(pixels, template["pixel_values"], visual.device)

    cos, sin = visual.rotary_pos_emb.get_cos_sin(
        visual.rotary_pos_emb.cos_sin_cache.shape[0]
    )
    corners = torch.zeros(4, rows, dtype=torch.long)
    weights = torch.zeros(4, rows, dtype=visual.dtype)
    # Padding rows read the zero row appended to the rotary tables below.
    pos_ids = torch.full((rows, 2), cos.shape[0], dtype=torch.long)
    start = 0
    for t, h, w in grid:
        item_corners, item_weights = _bilinear_tables(
            h, w, visual.num_grid_per_side, merge, visual.dtype
        )
        item_pos_ids = visual.rot_pos_ids(h, w, merge)
        for _ in range(t):
            end = start + h * w
            corners[:, start:end] = item_corners
            weights[:, start:end] = item_weights
            pos_ids[start:end] = item_pos_ids
            start = end

    device = visual.device
    embeds = visual.pos_embed.weight[corners.to(device, non_blocking=True)]
    embeds *= weights.to(device, non_blocking=True).unsqueeze(-1)
    pos_ids = pos_ids.to(device, non_blocking=True)
    values: dict[str, torch.Tensor | None] = {
        "pixel_values": pixels,
        "pos_embeds": embeds.sum(dim=0).to(template["pos_embeds"].dtype),
        "rotary_pos_emb_cos": torch.nn.functional.pad(cos, (0, 0, 0, 1))[
            pos_ids
        ].flatten(1),
        "rotary_pos_emb_sin": torch.nn.functional.pad(sin, (0, 0, 0, 1))[
            pos_ids
        ].flatten(1),
    }

    # The attention metadata, as `Qwen3_VisionTransformer.prepare_encoder_metadata`
    # builds it for the replay buffers.
    grid_np = np.array(grid, dtype=np.int32)
    cu_seqlens = np.repeat(grid_np[:, 1] * grid_np[:, 2], grid_np[:, 0]).cumsum(
        dtype=np.int32
    )
    cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
    pad_to = max_frames_per_batch if is_video else max_batch_size
    if len(cu_seqlens) - 1 < pad_to:
        cu_seqlens = np.concatenate(
            [
                cu_seqlens,
                np.full(pad_to - len(cu_seqlens) + 1, cu_seqlens[-1], np.int32),
            ]
        )
    backend = visual.attn_backend
    values["sequence_lengths"] = MMEncoderAttention.maybe_compute_seq_lens(
        backend, cu_seqlens, device
    )
    values["max_seqlen"] = torch.tensor(
        MMEncoderAttention.compute_max_seqlen(backend, cu_seqlens), dtype=torch.int32
    )
    values["cu_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
        backend,
        cu_seqlens,
        visual.hidden_size,
        visual.tp_size,
        device,
        fp8_padded_hidden_size=visual.fp8_padded_hidden_size,
    )
    return values


def _budget_pixels(
    items: list[torch.Tensor], template: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Host item pixels in one buffer of the template's shape, zero-padded, on
    the device."""
    from vllm_torchtpu.mm_host_copy_patch import copy_bytes_

    buffer = torch.empty(template.shape, dtype=template.dtype)
    flat = buffer.view(-1).view(torch.uint8)
    pairs, offset = [], 0
    for item in items:
        src = item.to(template.dtype).contiguous().view(-1).view(torch.uint8)
        pairs.append((flat[offset : offset + src.numel()], src))
        offset += src.numel()
    copy_bytes_(pairs)
    flat[offset:] = 0
    return buffer.to(device, non_blocking=True)


def _select_encoder_items(upstream):
    """Qwen3-VL's item selection, leaving host pixels as per-item slices.

    Upstream concatenates the selected items' pixels; on host batches that is
    a one-thread copy the budget-inputs hook would repeat, so the slices go to
    the hook, which copies them once into the budget's buffer.
    """

    @functools.wraps(upstream)
    def select(self, mm_kwargs, indices):
        pixel_values = self._get_pixel_values_by_modality(mm_kwargs)
        if not indices or not pixel_values.is_cpu:
            return upstream(self, mm_kwargs, indices)
        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
        offsets = np.cumsum([0] + [t * h * w for t, h, w in grid_thw])
        modality = self.get_input_modality(mm_kwargs)
        key = "pixel_values" if modality == "image" else "pixel_values_videos"
        return {
            key: [pixel_values[offsets[i] : offsets[i + 1]] for i in indices],
            f"{modality}_grid_thw": [grid_thw[i] for i in indices],
        }

    return select


def maybe_patch_qwen3_vl_budget_inputs(
    model_config: Optional["ModelConfig"] = None,
) -> bool | None:
    """Give the Qwen3-VL vision tower the encoder manager's budget-inputs hook
    and host-batch item selection."""
    if model_config is None or not _has_qwen3_vl_vision(model_config):
        return False
    from vllm.model_executor.models import qwen3_vl

    from vllm_torchtpu.runner.mm_encoder_manager import BUDGET_INPUTS_HOOK

    model_cls = qwen3_vl.Qwen3VLForConditionalGeneration
    if (
        getattr(model_cls, BUDGET_INPUTS_HOOK, None)
        is not _prepare_encoder_budget_inputs
    ):
        setattr(model_cls, BUDGET_INPUTS_HOOK, _prepare_encoder_budget_inputs)
        model_cls.select_encoder_cudagraph_items = _select_encoder_items(
            model_cls.select_encoder_cudagraph_items
        )
    return None
