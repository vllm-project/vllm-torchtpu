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
"""Encoder outputs kept as row handles into their encoder batch's output.

vLLM keeps each multimodal item's encoder output as its own tensor, sliced
out of the encoder batch, re-sliced per scheduled token range, concatenated
per step and gathered into the token embeddings. On TPU every one of those
ops takes its shape from the items' token counts, so each new image layout
compiles them again.

Here an item is the batch output it came from plus its row indices there
(`EncoderRows`); slicing a token range slices the row indices on the host.
A step's embeddings are gathers by host-built indices (`EncoderEmbeds.gather`),
whose programs take their shapes from encoder budgets and token buckets only,
compiled at warm-up. Nothing is copied when an output is cached, and the batch
output lives exactly as long as an item of it is in vLLM's encoder cache,
where a shared table would copy every output into it once more.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from vllm_torchtpu.utils import synchronize_tensors

# Uploads a host tensor without blocking, keeping it alive until the device
# has read it (the runner's `stage_h2d_upload`).
Upload = Callable[[torch.Tensor], torch.Tensor]


class EncoderRows:
    """Rows of one encoder batch output: an item, or a token range of one."""

    # vLLM tags each item it gathers with its `modality`.
    __slots__ = ("modality", "rows", "source")
    ndim = 2

    def __init__(self, source: torch.Tensor, rows: np.ndarray) -> None:
        self.source = source
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: slice) -> EncoderRows:
        if not isinstance(key, slice):
            raise TypeError("EncoderRows supports slices only")
        return EncoderRows(self.source, self.rows[key])

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), self.source.shape[1])

    def tensor(self, upload: Upload) -> torch.Tensor:
        """These rows as a tensor of their own (for layout-shaped consumers)."""
        return self.source[upload(torch.from_numpy(self.rows))]


def split_batch_output(
    output: torch.Tensor, lengths: Sequence[int]
) -> list[EncoderRows]:
    """The consecutive items of an encoder batch output, as row handles."""
    items, start = [], 0
    for n in lengths:
        items.append(EncoderRows(output, np.arange(start, start + n, dtype=np.int64)))
        start += n
    return items


class EncoderEmbeds:
    """A step's multimodal embeddings in token order, as row handles.

    Slicing gives a token range; `gather` materializes it at a token bucket.
    """

    def __init__(self, parts: Sequence[EncoderRows]) -> None:
        self.parts = list(parts)
        self.bounds = np.cumsum([0] + [len(p) for p in self.parts])

    def __len__(self) -> int:
        return int(self.bounds[-1])

    def __getitem__(self, key: slice) -> EncoderEmbeds:
        start, stop, step = key.indices(len(self))
        assert step == 1
        parts = []
        bounds = zip(self.parts, self.bounds[:-1], self.bounds[1:], strict=True)
        for part, lo, hi in bounds:
            a, b = max(start, lo), min(stop, hi)
            if a < b:
                parts.append(part[a - lo : b - lo])
        return EncoderEmbeds(parts)

    def gather(self, num_rows: int, max_rows: int, upload: Upload) -> torch.Tensor:
        """The embeddings in `num_rows` rows: these first, then rows the merge
        does not read (it reads only the first `len(self)`).

        Rows of one batch output are one gather, which fuses into the merge: a
        program per (encoder budget, token bucket). Rows of several are
        gathered output by output into `max_rows` rows (the largest token
        bucket), each its own program per encoder budget, then sliced into the
        merge: a program per token bucket. No program depends on how many
        outputs a step reads.
        """
        spans: dict[int, tuple[torch.Tensor, list[tuple[int, np.ndarray]]]] = {}
        offset = 0
        for part in self.parts:
            spans.setdefault(id(part.source), (part.source, []))[1].append(
                (offset, part.rows)
            )
            offset += len(part)

        def index(
            at: list[tuple[int, np.ndarray]], size: int, fill: int
        ) -> torch.Tensor:
            table = np.full(size, fill, dtype=np.int64)
            for start, rows in at:
                table[start : start + len(rows)] = rows
            return upload(torch.from_numpy(table))

        if len(spans) == 1:
            ((source, at),) = spans.values()
            return source[index(at, num_rows, 0)]
        out: torch.Tensor | None = None
        for source, at in spans.values():
            picks = index(at, max_rows, -1)
            rows = source[picks.clamp(min=0)]
            out = (
                rows
                if out is None
                else torch.where((picks >= 0).unsqueeze(-1), rows, out)
            )
            synchronize_tensors([out], wait=False)
        assert out is not None
        return out[:num_rows]
