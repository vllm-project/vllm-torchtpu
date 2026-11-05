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
"""Model-specific patches for Gemma4 multimodal models on TPU."""

from typing import TYPE_CHECKING

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_GEMMA4_MM_ARCHITECTURES = {
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
}

_TEXT_ONLY_ARCHITECTURES = {
    "Gemma4ForCausalLM",
}

_orig_gemma4_layer_config = None


def _lane_swap_rotate_half(x):
    """In-register lane-swap rotate_half (l ^ n) for TPU acceleration (b/581011672).

    Adapted from the Lane-swap technique (go/fused_qkv_proj, cl/989795783).
    Replaces tensor slicing (torch.chunk) and concatenation (torch.cat)
    with an in-register bitwise XOR lane permutation (l ^ 32), eliminating
    crossbar lane masking and HBM round-trips on TPU hardware.
    """
    head_dim = x.shape[-1]
    # In Gemma-4 vision encoder, head_dim is 72 with 64 rotated lanes (n = 32 pairs).
    # For standard rotary heads, rotary_dim == head_dim (n = head_dim // 2).
    n = 32 if head_dim == 72 else head_dim // 2
    if (n & (n - 1)) == 0 and n >= 8:
        import torch

        partner = torch.arange(head_dim, device=x.device)
        partner[: 2 * n] = partner[: 2 * n] ^ n
        signs = torch.ones(head_dim, device=x.device, dtype=x.dtype)
        signs[:n] = -1.0
        return x.gather(-1, partner.expand_as(x)) * signs

    import torch

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    return torch.cat((-x2, x1), dim=-1)


setattr(_lane_swap_rotate_half, "_is_lane_swap_patched", True)  # noqa: B010


def patch_vision_mrope_lane_swap() -> None:
    """Patch SigLip vision encoder rotary embedding with in-register XOR lane-swap.

    Eliminates the 0.932 ms/layer slicing overhead in Gemma-4 vision encoder
    by replacing standard rotate_half with in-register lane-swap (b/581011672).
    """
    import sys

    modules_to_patch = [
        "transformers.models.siglip.modeling_siglip",
        "transformers.models.siglip2.modeling_siglip2",
        "vllm.model_executor.layers.rotary_embedding.common",
        "vllm.model_executor.layers.rotary_embedding",
    ]

    for mod_name in modules_to_patch:
        try:
            mod = sys.modules.get(mod_name)
            if mod is None:
                __import__(mod_name)
                mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, "rotate_half"):
                orig_fn = mod.rotate_half
                if not getattr(orig_fn, "_is_lane_swap_patched", False):
                    mod.rotate_half = _lane_swap_rotate_half
                    logger.info(
                        "Applied TPU patch: enabled in-register 2D mRoPE "
                        "lane-swap in %s.",
                        mod_name,
                    )
        except (ImportError, AttributeError):
            pass


_GEMMA4_HF_MODELING = "transformers.models.gemma4.modeling_gemma4"
_orig_apply_multidimensional_rope = None


def _rotate_half_matrix(num_channels: int, chunk: int, device, dtype):
    """Signed permutation P with ``x @ P == cat(rotate_half(x_k) for x_k)``.

    ``x`` is split into contiguous chunks of ``chunk`` channels; each chunk is
    rotated with HF's ``rotate_half`` (``cat(-x2, x1)``). Every output channel
    reads exactly one input channel scaled by +1 or -1, so the product is exact.
    Built from ``arange`` so a compiled graph folds it to a constant.
    """
    import torch

    idx = torch.arange(num_channels, device=device)
    row = idx.unsqueeze(1)  # input channel
    col = idx.unsqueeze(0)  # output channel
    half = chunk // 2
    same_chunk = (row // chunk) == (col // chunk)
    row_pos = row % chunk
    col_pos = col % chunk
    # out[j] = -x[j + half] for the first half of each chunk.
    neg = same_chunk & (col_pos < half) & (row_pos == col_pos + half)
    # out[j] = x[j - half] for the second half of each chunk.
    pos = same_chunk & (col_pos >= half) & (row_pos == col_pos - half)
    return pos.to(dtype) - neg.to(dtype)


def tpu_apply_multidimensional_rope(
    x,
    cos,
    sin,
    position_ids,
    unsqueeze_dim: int = 2,
):
    """Single-pass replacement for HF ``apply_multidimensional_rope`` (b/581011672).

    HF splits the head into ``ndim`` contiguous chunks, applies 1D RoPE to each
    chunk via ``rotate_half`` (slice + negate + concat) and concatenates the
    results. For the Gemma-4 vision tower (head_dim 72, ndim 2) this slices
    18-channel pieces, which pads badly on 128-lane TPU vregs.

    This computes the same result on the full head in one pass:
        y = x * cos + (x @ P) * sin
    where ``P`` is the block-diagonal signed permutation that applies
    ``rotate_half`` within each chunk. Falls back to the HF implementation for
    layouts that do not split evenly.
    """
    import torch

    ndim = position_ids.shape[-1]
    num_channels = x.shape[-1]
    chunk = 2 * (num_channels // (2 * ndim))
    if (
        chunk <= 0
        or chunk * ndim != num_channels
        or cos.shape[-1] != num_channels
        or sin.shape[-1] != num_channels
    ):
        assert _orig_apply_multidimensional_rope is not None
        return _orig_apply_multidimensional_rope(
            x, cos, sin, position_ids, unsqueeze_dim=unsqueeze_dim
        )

    rot = _rotate_half_matrix(num_channels, chunk, x.device, x.dtype)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (torch.matmul(x, rot) * sin)


setattr(tpu_apply_multidimensional_rope, "_tpu_rope_patched", True)  # noqa: B010


def patch_gemma4_vision_rope() -> None:
    """Route Gemma-4 vision attention RoPE through the single-pass version.

    ``Gemma4VisionAttention.forward`` resolves ``apply_multidimensional_rope``
    from the module globals at call time, so replacing the module attribute
    before the vision graph is traced is sufficient. Only the Gemma-4 HF
    modeling module is touched.
    """
    global _orig_apply_multidimensional_rope
    import sys

    try:
        mod = sys.modules.get(_GEMMA4_HF_MODELING)
        if mod is None:
            __import__(_GEMMA4_HF_MODELING)
            mod = sys.modules.get(_GEMMA4_HF_MODELING)
    except ImportError:
        return
    if mod is None:
        return

    current = getattr(mod, "apply_multidimensional_rope", None)
    if current is None or getattr(current, "_tpu_rope_patched", False):
        return
    _orig_apply_multidimensional_rope = current
    mod.apply_multidimensional_rope = tpu_apply_multidimensional_rope
    logger.info(
        "Applied TPU patch: single-pass multidimensional RoPE for Gemma-4 vision."
    )


def _make_eager_dp_embed_multimodal(orig_embed_multimodal):
    """Wrap embed_multimodal with eager-mode data-parallel sharding."""

    def patched_embed_multimodal(self, *args, **kwargs):
        # When EncoderCudaGraphManager is active, the runner handles rank sharding
        # and output gathering across DP ranks. Avoid redundant local sharding.
        if getattr(self, "is_encoder_cudagraph_active", False):
            return orig_embed_multimodal(self, *args, **kwargs)

        use_dp = getattr(self, "use_data_parallel", None)
        if use_dp is None:
            mm_config = getattr(
                getattr(self, "config", None), "multimodal_config", None
            )
            if mm_config is not None:
                use_dp = getattr(mm_config, "mm_encoder_tp_mode", None) == "data"
            else:
                use_dp = False

        if not use_dp:
            return orig_embed_multimodal(self, *args, **kwargs)

        all_gather_fn = None
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
                tensor_model_parallel_all_gather,
            )

            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            all_gather_fn = tensor_model_parallel_all_gather
        except (ImportError, AttributeError):
            tp_size = 1
            tp_rank = 0

        if tp_size <= 1 or all_gather_fn is None:
            return orig_embed_multimodal(self, *args, **kwargs)

        pixel_values = kwargs.get("pixel_values")
        if (
            pixel_values is not None
            and hasattr(pixel_values, "shape")
            and len(pixel_values.shape) > 0
        ):
            num_items = pixel_values.shape[0]
            items_per_rank = (num_items + tp_size - 1) // tp_size
            start_idx = min(tp_rank * items_per_rank, num_items)
            end_idx = min(start_idx + items_per_rank, num_items)

            sharded_kwargs = dict(kwargs)
            sharded_kwargs["pixel_values"] = pixel_values[start_idx:end_idx]

            local_output = orig_embed_multimodal(self, *args, **sharded_kwargs)
            if local_output is not None and hasattr(local_output, "dim"):
                return all_gather_fn(local_output, dim=0)
            return local_output

        return orig_embed_multimodal(self, *args, **kwargs)

    setattr(patched_embed_multimodal, "_tpu_dp_patched", True)  # noqa: B010
    return patched_embed_multimodal


def patch_gemma4_mm_model_classes() -> None:
    """Set supports_encoder_tp_data and eager DP ViT on Gemma4 multimodal classes."""
    for mod_path, cls_name in (
        (
            "vllm.model_executor.models.gemma4_mm",
            "Gemma4ForConditionalGeneration",
        ),
        (
            "vllm.model_executor.models.gemma4_unified",
            "Gemma4UnifiedForConditionalGeneration",
        ),
    ):
        try:
            import sys

            mod = sys.modules.get(mod_path)
            if mod is None:
                __import__(mod_path)
                mod = sys.modules.get(mod_path)
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue

            if not getattr(cls, "supports_encoder_tp_data", False):
                cls.supports_encoder_tp_data = True
                logger.info(
                    "Applied TPU patch: enabled supports_encoder_tp_data on %s.",
                    cls_name,
                )

            orig_embed = getattr(cls, "embed_multimodal", None)
            if orig_embed is not None and not getattr(
                orig_embed, "_tpu_dp_patched", False
            ):
                cls.embed_multimodal = _make_eager_dp_embed_multimodal(orig_embed)
                logger.info(
                    "Applied TPU patch: enabled eager DP ViT on %s.",
                    cls_name,
                )
        except (ImportError, AttributeError):
            pass


def patch_gemma4_attention_init() -> None:
    """Ensure Gemma4Attention instances for full-attention layers use
    global_head_dim.
    """
    try:
        import inspect
        import sys

        gemma4_mod = sys.modules.get("vllm.model_executor.models.gemma4")
        if gemma4_mod is None:
            from vllm.model_executor.models import gemma4 as gemma4_mod

        orig_init = gemma4_mod.Gemma4Attention.__init__
        if getattr(orig_init, "_tpu_gemma4_attn_patched", False):
            return

        sig = inspect.signature(orig_init)

        def patched_init(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            config = bound.arguments.get("config")
            prefix = bound.arguments.get("prefix", "")
            layer_idx = gemma4_mod.extract_layer_index(prefix)
            layer_types = getattr(config, "layer_types", None)
            is_full = (
                layer_types is not None
                and layer_idx < len(layer_types)
                and layer_types[layer_idx] == "full_attention"
            )
            if is_full:
                global_head_dim = getattr(config, "global_head_dim", None) or getattr(
                    config, "full_attention_head_dim", None
                )
                if global_head_dim is not None:
                    bound.arguments["head_dim"] = global_head_dim
                if getattr(config, "attention_k_eq_v", False):
                    global_kv = getattr(
                        config, "num_global_key_value_heads", None
                    ) or getattr(config, "full_attention_num_key_value_heads", None)
                    if global_kv is not None:
                        bound.arguments["num_kv_heads"] = global_kv
            return orig_init(*bound.args, **bound.kwargs)

        setattr(patched_init, "_tpu_gemma4_attn_patched", True)  # noqa: B010
        gemma4_mod.Gemma4Attention.__init__ = patched_init
    except (ImportError, AttributeError):
        pass


def patch_gemma4_layer_config() -> None:
    """Ensure full attention layers receive global_head_dim and global KV heads."""
    global _orig_gemma4_layer_config
    try:
        import sys
        from copy import copy

        import vllm.transformers_utils.configs.gemma4 as gemma4_cfg_mod

        if _orig_gemma4_layer_config is None:
            _orig_gemma4_layer_config = gemma4_cfg_mod.gemma4_layer_config
        orig_fn = _orig_gemma4_layer_config

        def patched_gemma4_layer_config(text_config, layer_idx: int):
            layer_types = getattr(text_config, "layer_types", None)
            is_full = (
                layer_types is not None
                and layer_idx < len(layer_types)
                and layer_types[layer_idx] == "full_attention"
            )

            # Resolve base layer config
            if getattr(text_config, "is_heterogeneous", False):
                per_layer = getattr(text_config, "per_layer_config", None)
                if per_layer and layer_idx < len(per_layer):
                    layer = copy(per_layer[layer_idx])
                else:
                    layer = copy(text_config)
            else:
                layer = orig_fn(text_config, layer_idx)

            if is_full:
                global_head_dim = getattr(
                    text_config, "global_head_dim", None
                ) or getattr(text_config, "full_attention_head_dim", None)
                if global_head_dim is None:
                    per_layer = getattr(text_config, "per_layer_config", None)
                    if per_layer and layer_idx < len(per_layer):
                        plc = per_layer[layer_idx]
                        global_head_dim = (
                            plc.get("head_dim")
                            if isinstance(plc, dict)
                            else getattr(plc, "head_dim", None)
                        )
                if global_head_dim is not None:
                    layer.head_dim = global_head_dim

                global_kv_heads = getattr(
                    text_config, "num_global_key_value_heads", None
                ) or getattr(text_config, "full_attention_num_key_value_heads", None)
                if global_kv_heads is None:
                    per_layer = getattr(text_config, "per_layer_config", None)
                    if per_layer and layer_idx < len(per_layer):
                        plc = per_layer[layer_idx]
                        global_kv_heads = (
                            plc.get("num_key_value_heads")
                            if isinstance(plc, dict)
                            else getattr(plc, "num_key_value_heads", None)
                        )
                if global_kv_heads is not None and getattr(
                    text_config, "attention_k_eq_v", False
                ):
                    layer.num_key_value_heads = global_kv_heads
            else:
                if getattr(layer, "head_dim", None) is None:
                    layer.head_dim = getattr(text_config, "head_dim", None)
                if getattr(layer, "num_key_value_heads", None) is None:
                    layer.num_key_value_heads = getattr(
                        text_config, "num_key_value_heads", None
                    )

            return layer

        gemma4_cfg_mod.gemma4_layer_config = patched_gemma4_layer_config
        setattr(gemma4_cfg_mod, "_tpu_layer_config_patched", True)  # noqa: B010

        for mod_name in [
            "vllm.model_executor.models.gemma4",
            "vllm.model_executor.models.gemma4_mtp",
            "vllm.model_executor.models.gemma4_dspark",
            "vllm.transformers_utils.model_arch_config_convertor",
        ]:
            if mod_name in sys.modules:
                mod = sys.modules[mod_name]
                if hasattr(mod, "gemma4_layer_config"):
                    setattr(  # noqa: B010
                        mod, "gemma4_layer_config", patched_gemma4_layer_config
                    )

        patch_gemma4_attention_init()

        logger.info(
            "Applied TPU patch: patched gemma4_layer_config and Gemma4Attention "
            "for heterogeneous layers."
        )
    except (ImportError, AttributeError):
        pass


def refresh_gemma4_mm() -> None:
    """Refresh Gemma4 multimodal and layer patches across lifecycle boundaries."""
    patch_gemma4_mm_model_classes()
    patch_gemma4_layer_config()
    patch_vision_mrope_lane_swap()


def _is_gemma4_mm_model(model_config: "ModelConfig | None") -> bool:
    """Check if the provided ModelConfig corresponds to a Gemma4 multimodal model."""
    if model_config is None:
        return False

    arch = getattr(model_config, "architecture", None)
    if arch in _GEMMA4_MM_ARCHITECTURES:
        return True
    if arch in _TEXT_ONLY_ARCHITECTURES:
        return False

    architectures = getattr(model_config, "architectures", []) or []
    if any(a in _GEMMA4_MM_ARCHITECTURES for a in architectures):
        return True
    if any(a in _TEXT_ONLY_ARCHITECTURES for a in architectures):
        return False

    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        hf_archs = getattr(hf_config, "architectures", []) or []
        if any(a in _GEMMA4_MM_ARCHITECTURES for a in hf_archs):
            return True
        if any(a in _TEXT_ONLY_ARCHITECTURES for a in hf_archs):
            return False
        model_type = getattr(hf_config, "model_type", "")
        if model_type == "gemma4":
            return hasattr(hf_config, "vision_config")

    model_name = str(getattr(model_config, "model", "")).lower()
    is_gemma4 = "gemma-4" in model_name or "gemma4" in model_name
    has_mm = getattr(model_config, "multimodal_config", None) is not None
    return is_gemma4 and has_mm


def maybe_patch_gemma4_mm(model_config: "ModelConfig | None" = None) -> bool | None:
    """Apply TPU patches for Gemma4 multimodal models.

    Enables data-parallel multimodal vision encoding when --mm-encoder-tp-mode
    data is configured by setting supports_encoder_tp_data = True on the model
    classes and updating model_config._model_info.
    """
    if not _is_gemma4_mm_model(model_config):
        return False

    patch_gemma4_mm_model_classes()
    patch_gemma4_layer_config()
    patch_vision_mrope_lane_swap()

    if model_config is not None:
        model_info = getattr(model_config, "_model_info", None)
        if model_info is not None:
            object.__setattr__(
                model_info,
                "supports_multimodal_encoder_tp_data",
                True,
            )

    return None
