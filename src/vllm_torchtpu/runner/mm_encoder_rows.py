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
compiles them again (about 0.7 s per first-seen size on Cosmos3-Super).

Here an item is the batch output it came from plus its row indices there
(`EncoderRows`); slicing a token range slices the row indices on the host.
A step's embeddings are one gather per distinct batch output by host-built
indices padded to the step's token bucket, combined by a mask: programs per
(encoder budget, token bucket), compiled at warm-up. Nothing is copied when
an output is cached, and the batch output lives exactly as long as an item
of it is in vLLM's encoder cache (a scatter into a shared table cost 8.6 ms
per 8192-row output).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


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

    def tensor(self) -> torch.Tensor:
        """These rows as a tensor of their own (for layout-shaped consumers)."""
        return self.source[self._index(self.rows)]

    def _index(self, rows: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(rows).to(self.source.device, non_blocking=True)


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

    def gather(self, num_rows: int) -> torch.Tensor:
        """The embeddings in `num_rows` rows: these first, then the first row
        of the first source (the merge reads only the first `len(self)`).

        One host-built table of row indices (-1 where a row is another
        source's) uploads once for all sources.
        """
        sources: dict[int, int] = {}
        for part in self.parts:
            sources.setdefault(id(part.source), len(sources))
        table = np.full((len(sources), num_rows), -1, dtype=np.int64)
        table[0] = 0
        offset, by_index = 0, {}
        for part in self.parts:
            s = sources[id(part.source)]
            by_index[s] = part.source
            table[s, offset : offset + len(part)] = part.rows
            offset += len(part)
        device = by_index[0].device
        table = torch.from_numpy(table).to(device, non_blocking=True)
        out = by_index[0][table[0].clamp(min=0)]
        for s in range(1, len(sources)):
            rows = by_index[s][table[s].clamp(min=0)]
            out = torch.where((table[s] >= 0).unsqueeze(-1), rows, out)
        return out
