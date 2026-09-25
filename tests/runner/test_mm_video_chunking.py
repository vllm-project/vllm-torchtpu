# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the vision-encoder video frame chunking
(`MM_ENCODER_FRAME_CHUNK_PATCH_SIZE`).

The chunking is exercised directly through
`vllm_torchtpu.runner.mm_video_chunking` with a fake encoder callable, so no
TPU or real multimodal model is needed.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from vllm_torchtpu.runner import mm_video_chunking as mvc


def _make_video_kwargs(t: int, h: int, w: int, feat: int = 5):
    """Build an mm_kwargs group for a single video item.

    Mirrors the kwargs that `_execute_mm_encoder` passes for a single video item:
    flat pixels `[t*h*w, feat]`, grid `[1, 3]`, and per-temporal-patch
    timestamps `[t]`.

    Args:
        t: Number of temporal patches/frames.
        h: Grid height in patches.
        w: Grid width in patches.
        feat: Feature dimension for each pixel row.

    Returns:
        Tuple of (mm_kwargs, pixels, timestamps).
    """
    rows = t * h * w
    pixels = torch.arange(rows * feat, dtype=torch.float32).reshape(rows, feat)
    grid = torch.tensor([[t, h, w]], dtype=torch.int64)
    timestamps = torch.arange(t, dtype=torch.float32)
    return (
        {
            "pixel_values_videos": pixels,
            "video_grid_thw": grid,
            "timestamps": timestamps,
        },
        pixels,
        timestamps,
    )


def _make_multi_video_kwargs(specs, feat=5):
    """Build an mm_kwargs group for several batched video items.

    Mirrors `group_and_batch_mm_kwargs`: pixels concatenated on the patch
    axis, grid stacked one row per item, and timestamps formatted as a per-item
    list.

    Args:
        specs: Iterable of (t, h, w) dimensions for each video item.
        feat: Feature dimension for each pixel row.

    Returns:
        Tuple of (mm_kwargs, pixel_slabs, timestamps_list).
    """
    pixel_slabs, grid_rows, ts_list = [], [], []
    base = 0
    for t, h, w in specs:
        rows = t * h * w
        pixel_slabs.append(
            torch.arange(base, base + rows * feat, dtype=torch.float32).reshape(
                rows, feat
            )
        )
        base += rows * feat
        grid_rows.append([t, h, w])
        ts_list.append(torch.arange(t, dtype=torch.float32))
    kwargs = {
        "pixel_values_videos": torch.cat(pixel_slabs, dim=0),
        "video_grid_thw": torch.tensor(grid_rows, dtype=torch.int64),
        "timestamps": ts_list,
    }
    return kwargs, pixel_slabs, ts_list


class TestChunkPatchSize:
    """``get_video_chunk_patch_size`` gates the whole feature."""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MM_ENCODER_FRAME_CHUNK_PATCH_SIZE", None)
            assert (
                mvc.get_video_chunk_patch_size(SimpleNamespace(multimodal_config=None))
                == 0
            )

    def test_enabled_by_env(self):
        with patch.dict(os.environ, {"MM_ENCODER_FRAME_CHUNK_PATCH_SIZE": "4096"}):
            assert (
                mvc.get_video_chunk_patch_size(SimpleNamespace(multimodal_config=None))
                == 4096
            )

    def test_disabled_when_video_pruning_enabled(self):
        """EVS video-token pruning selects tokens across the whole video, so
        chunking is skipped to stay lossless."""
        model_config = SimpleNamespace(
            multimodal_config=SimpleNamespace(video_pruning_rate=0.5)
        )
        with patch.dict(os.environ, {"MM_ENCODER_FRAME_CHUNK_PATCH_SIZE": "4096"}):
            assert mvc.get_video_chunk_patch_size(model_config) == 0


class TestChunkedVideoEncodingContext:
    """``chunked_video_encoding`` swaps ``embed_multimodal`` only while the
    body runs, and only when chunking is enabled."""

    def test_wraps_and_restores(self):
        model = torch.nn.Linear(2, 2)
        calls = []

        def embed_multimodal(**mm_kwargs):
            calls.append(mm_kwargs)
            return (torch.ones((4, 3)),)

        model.embed_multimodal = embed_multimodal
        model_config = SimpleNamespace(multimodal_config=None)

        kwargs, _, _ = _make_video_kwargs(8, 2, 2)  # 32 patches
        with (
            patch.dict(os.environ, {"MM_ENCODER_FRAME_CHUNK_PATCH_SIZE": "8"}),
            mvc.chunked_video_encoding(model, model_config),
        ):
            assert model.embed_multimodal is not embed_multimodal
            model.embed_multimodal(**kwargs)

        # Chunked into 4 forwards of 2 temporal patches each ...
        assert len(calls) == 4
        # ... and the original callable is restored on exit.
        assert model.embed_multimodal is embed_multimodal

    def test_noop_when_disabled(self):
        model = torch.nn.Linear(2, 2)
        embed_multimodal = MagicMock()
        model.embed_multimodal = embed_multimodal

        with (
            patch.dict(os.environ, {"MM_ENCODER_FRAME_CHUNK_PATCH_SIZE": "0"}),
            mvc.chunked_video_encoding(model, SimpleNamespace(multimodal_config=None)),
        ):
            assert model.embed_multimodal is embed_multimodal

    def test_noop_for_text_only_model(self):
        """A model without ``embed_multimodal`` is left alone."""
        model = torch.nn.Linear(2, 2)
        with (
            patch.dict(os.environ, {"MM_ENCODER_FRAME_CHUNK_PATCH_SIZE": "8"}),
            mvc.chunked_video_encoding(model, SimpleNamespace(multimodal_config=None)),
        ):
            assert not hasattr(model, "embed_multimodal")


class TestVideoEncoderChunking:
    """Covers ``embed_multimodal_chunked``."""

    def test_chunking_splits_and_concats(self):
        """A large video is encoded in temporal chunks and the per-chunk
        outputs are concatenated on the token axis in order."""
        t, h, w = 8, 2, 2  # rows_per_temporal_patch = h*w = 4
        kwargs, pixels, timestamps = _make_video_kwargs(t, h, w)

        hidden = 6  # visual_dim + deepstack levels, packed on the hidden axis
        calls = []

        def fake_encode(**mm_kwargs):
            calls.append(mm_kwargs)
            idx = len(calls) - 1
            chunk_t = int(mm_kwargs["video_grid_thw"].tolist()[0][0])
            # one item -> length-1 tuple; mark rows with the chunk index.
            return (torch.full((chunk_t, hidden), float(idx), dtype=torch.float32),)

        # max_patches=8, rows_per_t=h*w=4 -> chunk_t=2 temporal patches.
        out = mvc.embed_multimodal_chunked(fake_encode, kwargs, max_patches=8)

        assert len(calls) == 4
        rows_per_t = h * w
        for i, call_kwargs in enumerate(calls):
            grid_row = call_kwargs["video_grid_thw"].tolist()[0]
            # temporal count reduced to the chunk size
            assert grid_row == [2, h, w]
            # pixels sliced to this chunk's rows, matching the original slab.
            expected_pixels = pixels[i * 2 * rows_per_t : (i + 1) * 2 * rows_per_t]
            assert torch.equal(call_kwargs["pixel_values_videos"], expected_pixels)
            # timestamps sliced to the chunk's temporal extent.
            assert torch.equal(
                call_kwargs["timestamps"], timestamps[i * 2 : (i + 1) * 2]
            )

        # Output: single item, concatenation of the 4 chunk outputs.
        assert isinstance(out, list) and len(out) == 1
        expected = np.concatenate(
            [np.full((2, hidden), float(i)) for i in range(4)], axis=0
        )
        np.testing.assert_array_equal(out[0].numpy(), expected)

    def test_no_chunk_when_fits_one_chunk(self):
        """A video smaller than one chunk is encoded in a single call."""
        kwargs, _, _ = _make_video_kwargs(4, 2, 2)  # 16 patches

        encode_fn = MagicMock(return_value=(torch.ones((16, 6), dtype=torch.float32),))
        out = mvc.embed_multimodal_chunked(encode_fn, kwargs, max_patches=400)

        encode_fn.assert_called_once()
        assert out is encode_fn.return_value  # returned as-is, not re-wrapped

    def test_no_chunk_for_images(self):
        """Chunking only applies to video; image kwargs are never split."""
        kwargs = {
            "pixel_values": torch.arange(32 * 5, dtype=torch.float32).reshape(32, 5),
            "image_grid_thw": torch.tensor([[8, 2, 2]], dtype=torch.int64),
        }

        encode_fn = MagicMock(return_value=(torch.ones((32, 6), dtype=torch.float32),))
        mvc.embed_multimodal_chunked(encode_fn, kwargs, max_patches=8)

        encode_fn.assert_called_once()

    def test_no_chunk_on_unexpected_grid_layout(self):
        """A malformed grid falls back to a single un-chunked call rather than
        mis-splitting the batch."""
        kwargs, _, _ = _make_video_kwargs(8, 2, 2)
        kwargs["video_grid_thw"] = torch.tensor([[8, 2]], dtype=torch.int64)

        encode_fn = MagicMock(return_value=(torch.ones((32, 6), dtype=torch.float32),))
        mvc.embed_multimodal_chunked(encode_fn, kwargs, max_patches=8)

        encode_fn.assert_called_once()

    def test_no_chunk_when_pixels_disagree_with_grid(self):
        """The pixels are sliced using the grid's patch counts, so a grid that
        does not account for every pixel row is not safe to split."""
        kwargs, _, _ = _make_video_kwargs(8, 2, 2)  # 32 pixel rows
        # 16 patches, i.e. half the pixel rows -> layout we don't understand.
        kwargs["video_grid_thw"] = torch.tensor([[4, 2, 2]], dtype=torch.int64)

        encode_fn = MagicMock(return_value=(torch.ones((32, 6), dtype=torch.float32),))
        mvc.embed_multimodal_chunked(encode_fn, kwargs, max_patches=8)

        encode_fn.assert_called_once()

    def test_no_chunk_when_pixels_none(self):
        """When pixel_values_videos is None, chunking falls back safely."""
        kwargs, _, _ = _make_video_kwargs(8, 2, 2)
        kwargs["pixel_values_videos"] = None

        encode_fn = MagicMock(return_value="done")
        out = mvc.embed_multimodal_chunked(encode_fn, kwargs, max_patches=8)

        encode_fn.assert_called_once_with(**kwargs)
        assert out == "done"

    def test_no_chunk_when_grid_none(self):
        """When video_grid_thw is None, chunking falls back safely."""
        kwargs, _, _ = _make_video_kwargs(8, 2, 2)
        kwargs["video_grid_thw"] = None

        encode_fn = MagicMock(return_value="done")
        out = mvc.embed_multimodal_chunked(encode_fn, kwargs, max_patches=8)

        encode_fn.assert_called_once_with(**kwargs)
        assert out == "done"

    def test_no_chunk_when_pixels_invalid_shape(self):
        """A scalar or non-tensor pixel field falls back safely without error."""
        kwargs, _, _ = _make_video_kwargs(8, 2, 2)
        kwargs["pixel_values_videos"] = torch.tensor(1.0)  # 0D tensor

        encode_fn = MagicMock(return_value="done")
        out = mvc.embed_multimodal_chunked(encode_fn, kwargs, max_patches=8)

        encode_fn.assert_called_once_with(**kwargs)
        assert out == "done"

    def test_caller_grid_not_mutated(self):
        """Each chunk narrows the temporal extent on a clone, never in place."""
        t, h, w = 8, 2, 2
        kwargs, _, _ = _make_video_kwargs(t, h, w)
        original = kwargs["video_grid_thw"].clone()

        def fake_encode(**mm_kwargs):
            chunk_t = int(mm_kwargs["video_grid_thw"].tolist()[0][0])
            return (torch.zeros((chunk_t * h * w, 6), dtype=torch.float32),)

        mvc.embed_multimodal_chunked(fake_encode, kwargs, max_patches=8)

        assert torch.equal(kwargs["video_grid_thw"], original)

    def test_multi_item_chunks_each(self):
        """Two batched videos that together blow the budget are encoded as
        separate per-item forwards, each temporally chunked, with per-item
        outputs returned in order. Guards the OOM from batching two long videos
        into one full-length vision attention call."""
        specs = [(8, 2, 2), (6, 2, 2)]  # rows_per_t=4; budget 8 -> chunk_t=2
        kwargs, pixel_slabs, ts_list = _make_multi_video_kwargs(specs)
        hidden = 6
        calls = []

        def fake_encode(**mm_kwargs):
            calls.append(mm_kwargs)
            chunk_t = int(mm_kwargs["video_grid_thw"].tolist()[0][0])
            return (
                torch.full(
                    (chunk_t * 4, hidden), float(len(calls)), dtype=torch.float32
                ),
            )

        out = mvc.embed_multimodal_chunked(fake_encode, kwargs, max_patches=8)

        # item0 t=8 -> chunks [2,2,2,2]; item1 t=6 -> chunks [2,2,2].
        chunk_ts = [int(kw["video_grid_thw"].tolist()[0][0]) for kw in calls]
        assert chunk_ts == [2, 2, 2, 2, 2, 2, 2]

        # Each call's pixels come from that item's own slab, never mixing the
        # two videos' patches.
        rows_per_t = 4
        item0_calls, item1_calls = calls[:4], calls[4:]
        for i, call_kwargs in enumerate(item0_calls):
            expected = pixel_slabs[0][i * 2 * rows_per_t : (i + 1) * 2 * rows_per_t]
            assert torch.equal(call_kwargs["pixel_values_videos"], expected)
            assert torch.equal(
                call_kwargs["timestamps"], ts_list[0][i * 2 : (i + 1) * 2]
            )
        for i, call_kwargs in enumerate(item1_calls):
            expected = pixel_slabs[1][i * 2 * rows_per_t : (i + 1) * 2 * rows_per_t]
            assert torch.equal(call_kwargs["pixel_values_videos"], expected)
            assert torch.equal(
                call_kwargs["timestamps"], ts_list[1][i * 2 : (i + 1) * 2]
            )

        # Two per-item outputs, each the concat of its own chunks.
        assert isinstance(out, list) and len(out) == 2
        assert out[0].shape == (8 * rows_per_t, hidden)
        assert out[1].shape == (6 * rows_per_t, hidden)

    def test_multi_item_fits_single_call(self):
        """Several small videos whose combined patches fit the budget are left
        as one batched call for efficiency."""
        kwargs, _, _ = _make_multi_video_kwargs([(2, 2, 2), (2, 2, 2)])

        encode_fn = MagicMock(
            return_value=(
                torch.ones((8, 6), dtype=torch.float32),
                torch.ones((8, 6), dtype=torch.float32),
            )
        )
        out = mvc.embed_multimodal_chunked(encode_fn, kwargs, max_patches=400)

        encode_fn.assert_called_once()
        assert len(out) == 2

    def test_multi_item_only_large_ones_chunked(self):
        """A batch mixing a small and a large video splits per item: the small
        one is a single call, the large one is temporally chunked."""
        specs = [(2, 2, 2), (8, 2, 2)]  # rows_per_t=4; budget 8 -> chunk_t=2
        kwargs, _, _ = _make_multi_video_kwargs(specs)
        chunk_ts = []

        def fake_encode(**mm_kwargs):
            chunk_t = int(mm_kwargs["video_grid_thw"].tolist()[0][0])
            chunk_ts.append(chunk_t)
            return (
                torch.full((chunk_t * 4, 6), float(len(chunk_ts)), dtype=torch.float32),
            )

        out = mvc.embed_multimodal_chunked(fake_encode, kwargs, max_patches=8)

        # item0 (t=2 <= chunk_t=2) -> one call; item1 (t=8) -> 4 chunks.
        assert chunk_ts == [2, 2, 2, 2, 2]
        assert len(out) == 2

    def test_ragged_last_chunk(self):
        """When t is not a multiple of chunk_t the last chunk is smaller, and
        the concatenated length still matches the full video."""
        t, h, w = 7, 1, 3  # budget 6 -> chunk_t = 6 // 3 = 2 -> [2, 2, 2, 1]
        kwargs, pixels, timestamps = _make_video_kwargs(t, h, w)
        hidden = 4
        calls = []

        def fake_encode(**mm_kwargs):
            calls.append(mm_kwargs)
            chunk_t = int(mm_kwargs["video_grid_thw"].tolist()[0][0])
            return (
                torch.full((chunk_t, hidden), float(len(calls)), dtype=torch.float32),
            )

        out = mvc.embed_multimodal_chunked(fake_encode, kwargs, max_patches=6)

        chunk_ts = [int(kw["video_grid_thw"].tolist()[0][0]) for kw in calls]
        assert chunk_ts == [2, 2, 2, 1]
        # The last chunk's pixels/timestamps cover the ragged tail.
        rows_per_t = h * w
        assert torch.equal(
            calls[-1]["pixel_values_videos"], pixels[6 * rows_per_t : 7 * rows_per_t]
        )
        assert torch.equal(calls[-1]["timestamps"], timestamps[6:7])
        assert out[0].shape == (t, hidden)

    def test_chunk_smaller_than_one_frame(self):
        """A budget below a single frame still encodes one temporal patch per
        forward instead of dividing by zero."""
        kwargs, _, _ = _make_video_kwargs(3, 4, 4)  # 16 patches per frame
        calls = []

        def fake_encode(**mm_kwargs):
            calls.append(mm_kwargs)
            chunk_t = int(mm_kwargs["video_grid_thw"].tolist()[0][0])
            return (torch.zeros((chunk_t * 16, 6), dtype=torch.float32),)

        mvc.embed_multimodal_chunked(fake_encode, kwargs, max_patches=4)

        assert [int(kw["video_grid_thw"].tolist()[0][0]) for kw in calls] == [1, 1, 1]


class TestSliceTimestamps:
    def test_slice_timestamps_variants(self):
        """``_slice_timestamps`` handles 1D/2D tensors, lists, nested lists and
        passes through non-sliceable values."""
        slice_fn = mvc._slice_timestamps

        # 1D tensor
        assert torch.equal(slice_fn(torch.arange(10), 2, 5), torch.tensor([2, 3, 4]))
        # 2D [1, T] tensor -> slice the last axis
        sliced = slice_fn(torch.arange(10).reshape(1, 10), 2, 5)
        assert sliced.shape == (1, 3)
        assert torch.equal(sliced, torch.tensor([[2, 3, 4]]))
        # flat list
        assert slice_fn(list(range(10)), 2, 5) == [2, 3, 4]
        # nested single-item list
        assert slice_fn([list(range(10))], 2, 5) == [[2, 3, 4]]
        # non-sliceable -> passthrough
        assert slice_fn(None, 2, 5) is None

    def test_select_item_timestamps_variants(self):
        """``_select_item_timestamps`` picks item ``i`` out of a batched
        group."""
        select_fn = mvc._select_item_timestamps

        batched = torch.arange(6).reshape(2, 3)
        assert torch.equal(select_fn(batched, 1), batched[1:2])
        per_item = [torch.arange(3), torch.arange(4)]
        assert torch.equal(select_fn(per_item, 1), per_item[1])
        assert select_fn(None, 1) is None
