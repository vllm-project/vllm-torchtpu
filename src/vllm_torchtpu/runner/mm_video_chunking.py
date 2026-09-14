# SPDX-License-Identifier: Apache-2.0
"""Temporal chunking of the vision encoder over video frames.

Encoding a long video in one shot makes the vision flash-attention kernel
process all `t*h*w` patches in a single call, which can exhaust VMEM and OOM
the step. `MM_ENCODER_FRAME_CHUNK_PATCH_SIZE` bounds each vision forward to
at most that many patches by splitting the video on the temporal axis and
concatenating the per-chunk outputs on the token axis.

The split is lossless for video encoders where vision attention is
segmented per frame (`cu_seqlens` has one entry per frame) and the position
embeddings are spatial-only, so a frame never attends across frame boundaries
and is encoded identically regardless of chunking. Deepstack features ride on
the hidden axis (`dim=-1`) and are preserved by the token-axis concat. The
one cross-frame exception is global video-token pruning
(`video_pruning_rate`), which selects tokens across the whole video; the
chunking auto-disables when it is on.

The chunking is applied by temporarily swapping the model's
`embed_multimodal` for a wrapper (see `chunked_video_encoding`) so the rest
of the upstream `GPUModelRunner._execute_mm_encoder` path — prompt_embeds,
multimodal LoRA, encoder cudagraphs, encoder-cache bookkeeping — keeps running
unmodified.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.model_executor.models.interfaces import MultiModalEmbeddings

logger = init_logger(__name__)

# Video kwargs field names, in preference order. Only an explicit video pixel
# field enables chunking: the wrapper sees the batched kwargs without the
# modality tag, and images must never be split (their attention spans the whole
# item, so chunking would not be lossless).
_VIDEO_PIXEL_KEYS = ("pixel_values_videos",)
_VIDEO_GRID_KEYS = ("video_grid_thw", "grid_thw")

# Sentinel for "the model had no instance-level embed_multimodal attribute".
_UNSET = object()


def get_video_chunk_patch_size(model_config: "ModelConfig | None") -> int:
    """Return the per-forward patch budget, or 0 when chunking is disabled.

    Args:
        model_config: Model configuration, optionally containing multimodal
            settings.

    Returns:
        The maximum number of patches allowed per forward, or 0 when
        chunking is disabled.
    """
    max_patches = envs.MM_ENCODER_FRAME_CHUNK_PATCH_SIZE
    if max_patches <= 0:
        return 0

    # EVS pruning selects tokens across the whole video; chunking would prune
    # each chunk independently, so skip it to stay lossless.
    mm_config = getattr(model_config, "multimodal_config", None)
    if getattr(mm_config, "video_pruning_rate", None):
        logger.warning_once(
            "[mm_encoder chunk] MM_ENCODER_FRAME_CHUNK_PATCH_SIZE=%d ignored: "
            "video token pruning (EVS) selects tokens across the whole video, "
            "so chunking it would change the output.",
            max_patches,
        )
        return 0
    return max_patches


@contextmanager
def chunked_video_encoding(
    model: torch.nn.Module,
    model_config: "ModelConfig | None",
) -> Iterator[None]:
    """Wrap the model's `embed_multimodal` to temporally chunk oversized videos.

    A no-op when chunking is disabled or the model has no `embed_multimodal`
    attribute (e.g. text-only models).

    Args:
        model: The model module whose `embed_multimodal` will be wrapped.
        model_config: Model configuration used to determine chunking parameters.

    Yields:
        None.
    """
    max_patches = get_video_chunk_patch_size(model_config)
    embed_fn = getattr(model, "embed_multimodal", None)
    if max_patches <= 0 or embed_fn is None:
        yield
        return

    def chunked_embed_multimodal(**mm_kwargs: object) -> "MultiModalEmbeddings":
        return embed_multimodal_chunked(embed_fn, mm_kwargs, max_patches)

    previous = model.__dict__.get("embed_multimodal", _UNSET)
    model.embed_multimodal = chunked_embed_multimodal
    try:
        yield
    finally:
        if previous is _UNSET:
            # Drop the instance attribute so the bound method is visible again.
            model.__dict__.pop("embed_multimodal", None)
        else:
            model.embed_multimodal = previous


def embed_multimodal_chunked(
    embed_fn: Callable[..., "MultiModalEmbeddings"],
    mm_kwargs: dict[str, object],
    max_patches: int,
) -> "MultiModalEmbeddings":
    """Run `embed_fn` on multimodal kwargs, splitting large videos into chunks.

    When several video items are batched into one encoder call, the batch is
    likewise segmented per item, so we split it back into per-item forwards and
    chunk each item that is too large. `pixel_values_videos` is concatenated
    on the patch axis; the grid is stacked one row per item; other stacked
    per-item fields are indexed by item. This is bit-exact for the same reason
    the temporal split is.

    Falls back to a single un-chunked call unless the group is a video group
    whose total patch count exceeds the budget.

    Args:
        embed_fn: The underlying multimodal embedding function to call.
        mm_kwargs: Batched group of multimodal keyword arguments.
        max_patches: Maximum number of patches allowed per vision forward pass.

    Returns:
        Encoded multimodal embeddings for the batch.
    """
    grid_key = next((k for k in _VIDEO_GRID_KEYS if mm_kwargs.get(k) is not None), None)
    pixel_key = next(
        (k for k in _VIDEO_PIXEL_KEYS if mm_kwargs.get(k) is not None), None
    )
    if grid_key is None or pixel_key is None:
        return embed_fn(**mm_kwargs)

    grid = mm_kwargs[grid_key]
    grid_list = grid.tolist() if hasattr(grid, "tolist") else list(grid)
    # One grid row [t, h, w] per item; bail on any unexpected layout.
    if not grid_list or not all(
        isinstance(row, (list, tuple)) and len(row) == 3 for row in grid_list
    ):
        logger.warning_once(
            "[mm_encoder chunk] %s has an unrecognized layout; encoding the "
            "whole group in one forward (chunking disabled for it).",
            grid_key,
        )
        return embed_fn(**mm_kwargs)

    num_items = len(grid_list)
    per_item_patches = [int(t) * int(h) * int(w) for t, h, w in grid_list]
    total_patches = sum(per_item_patches)
    # Whole batch already fits the budget -> keep the single batched call.
    if total_patches <= max_patches:
        return embed_fn(**mm_kwargs)

    # We slice the pixels by those per-item counts, so they have to account for
    # every row. A mismatch means the grid does not describe the pixel layout
    # the way we assume, so encode unsplit rather than feed the encoder
    # garbage.
    pixels = mm_kwargs[pixel_key]
    pixel_shape = getattr(pixels, "shape", None)
    pixel_rows = pixel_shape[0] if pixel_shape and len(pixel_shape) >= 1 else None
    if pixel_rows != total_patches:
        logger.warning_once(
            "[mm_encoder chunk] %s has %s rows but %s implies %d patches; "
            "encoding the whole group in one forward (chunking disabled for "
            "it).",
            pixel_key,
            pixel_rows,
            grid_key,
            total_patches,
        )
        return embed_fn(**mm_kwargs)

    # Single item: chunk it in place (no batch to split).
    if num_items == 1:
        return [
            _encode_video_item(embed_fn, pixel_key, grid_key, mm_kwargs, max_patches)
        ]

    # Multiple items batched together: split per item, then chunk each.
    timestamps = mm_kwargs.get("timestamps")
    logger.debug(
        "[mm_encoder chunk] %d video items, %d patches > budget %d; encoding per item",
        num_items,
        total_patches,
        max_patches,
    )

    outputs = []
    row = 0
    for i, num_patches in enumerate(per_item_patches):
        item_kwargs = dict(mm_kwargs)
        item_kwargs[pixel_key] = pixels[row : row + num_patches]
        item_kwargs[grid_key] = grid[i : i + 1]
        if timestamps is not None:
            item_kwargs["timestamps"] = _select_item_timestamps(timestamps, i)
        # Index any other stacked per-item field (shape[0] == num_items).
        for key, val in mm_kwargs.items():
            if key in (pixel_key, grid_key, "timestamps"):
                continue
            shape = getattr(val, "shape", None)
            if shape and len(shape) >= 1 and shape[0] == num_items:
                item_kwargs[key] = val[i : i + 1]
        outputs.append(
            _encode_video_item(embed_fn, pixel_key, grid_key, item_kwargs, max_patches)
        )
        row += num_patches
    return outputs


def _encode_video_item(
    embed_fn: Callable[..., "MultiModalEmbeddings"],
    pixel_key: str,
    grid_key: str,
    item_kwargs: dict[str, object],
    max_patches: int,
) -> torch.Tensor:
    """Encode one video item, splitting into chunks when patch count exceeds budget.

    Args:
        embed_fn: The multimodal embedding function to call for each chunk.
        pixel_key: Keyword argument name for the video pixel values tensor.
        grid_key: Keyword argument name for the video grid shape tensor.
        item_kwargs: Multimodal keyword arguments for a single video item
            (with grid shaped `[1, 3]`).
        max_patches: Maximum number of patches allowed per vision forward pass.

    Returns:
        Encoded multimodal embedding tensor concatenated across chunks along the
        token axis.
    """
    grid = item_kwargs[grid_key]
    grid_list = grid.tolist() if hasattr(grid, "tolist") else list(grid)
    t, h, w = (int(x) for x in grid_list[0])

    # Bound each vision forward to at most `max_patches` patches (t*h*w). The
    # flash-attention window scales with the per-chunk patch count, so budget on
    # patches (not frames) to stay under VMEM regardless of frame resolution.
    # Each temporal-patch contributes h*w patches, so pack as many
    # temporal-patches per chunk as fit within the budget.
    rows_per_t = h * w
    chunk_t = max(1, max_patches // rows_per_t)
    if t <= chunk_t:
        return embed_fn(**item_kwargs)[0]

    pixels = item_kwargs[pixel_key]
    timestamps = item_kwargs.get("timestamps")

    num_chunks = (t + chunk_t - 1) // chunk_t
    logger.info(
        "[mm_encoder chunk] video grid=(t=%d,h=%d,w=%d) %d patches -> "
        "%d chunks of <=%d temporal-patches (<=%d patches)",
        t,
        h,
        w,
        t * rows_per_t,
        num_chunks,
        chunk_t,
        chunk_t * rows_per_t,
    )

    chunk_outs = []
    for t_start in range(0, t, chunk_t):
        chunk_len = min(chunk_t, t - t_start)
        row_start = t_start * rows_per_t
        row_end = (t_start + chunk_len) * rows_per_t

        chunk_grid = grid[:1].clone()
        chunk_grid[0, 0] = chunk_len

        chunk_kwargs = dict(item_kwargs)
        chunk_kwargs[pixel_key] = pixels[row_start:row_end]
        chunk_kwargs[grid_key] = chunk_grid
        if timestamps is not None:
            chunk_kwargs["timestamps"] = _slice_timestamps(
                timestamps, t_start, t_start + chunk_len
            )

        out = embed_fn(**chunk_kwargs)
        # Enqueue the chunk to PJRT immediately so device activation
        # buffers can be freed between chunks without blocking the host CPU.
        synchronize_tensors(out[0], wait=False)
        chunk_outs.append(out[0])

    return torch.cat(chunk_outs, dim=0)


def _select_item_timestamps(timestamps: object, i: int) -> object:
    """Select item `i`'s timestamps from a batched group.

    Batching stacks a fixed-length timestamps tensor to `[num_items, T]` and
    keeps variable-length ones (the common video case) as a list with one entry
    per item. Returns item `i` in the same shape a single-item forward expects;
    passes anything else through unchanged.

    Args:
        timestamps: Batched timestamps, either as a stacked tensor or a list
            with one element per item.
        i: Index of the item to select.

    Returns:
        The timestamps for item `i`.
    """
    if hasattr(timestamps, "ndim") and timestamps.ndim >= 2:
        return timestamps[i : i + 1]
    if isinstance(timestamps, (list, tuple)):
        return timestamps[i]
    return timestamps


def _slice_timestamps(timestamps: object, start: int, end: int) -> object:
    """Slice a per-temporal-patch timestamps field to `[start, end)`.

    The vision encoder itself ignores timestamps, but the input schema may
    carry/validate them, so keep the length consistent with each chunk's
    temporal extent. Handles a 1D tensor, a batched `[1, T]` tensor, and
    nested Python lists; passes anything else through unchanged.

    Args:
        timestamps: Timestamps field for the single item being chunked.
        start: Starting index along the temporal patch dimension.
        end: Ending index along the temporal patch dimension.

    Returns:
        Sliced timestamps covering the range `[start, end)`.
    """
    if hasattr(timestamps, "ndim"):  # torch/np tensor
        if timestamps.ndim == 1:
            return timestamps[start:end]
        return timestamps[..., start:end]
    if isinstance(timestamps, (list, tuple)):
        if len(timestamps) == 1 and isinstance(timestamps[0], (list, tuple)):
            return type(timestamps)([timestamps[0][start:end]])
        return timestamps[start:end]
    return timestamps
