# SPDX-License-Identifier: Apache-2.0
"""Token-to-request lookups into short per-request tables.

XLA:TPU lowers a gather from a table of a few dozen entries, and a
``bucketize`` against one, to a chain of selects that takes every table entry
as a separate scalar operand; at 48 requests each such fusion costs ~30 us of
mostly scalar-operand staging. Comparing against the whole table at once is a
single vector fusion. Both forms are exact for integer tables.
"""

import torch

from vllm_torchtpu import envs


def take(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``table[index]`` for a short 1-D integer ``table``.

    Negative indices wrap once and the result is clamped into range, as the
    lowered gather does.
    """
    if not envs.TPU_ONEHOT_TABLE_LOOKUP:
        return table[index]
    size = table.shape[0]
    index = torch.where(index < 0, index + size, index).clamp(0, size - 1)
    entries = torch.arange(size, device=index.device, dtype=index.dtype)
    hit = index.unsqueeze(-1) == entries
    return torch.where(hit, table, torch.zeros_like(table)).sum(-1, dtype=table.dtype)


def bucketize_right(
    values: torch.Tensor, boundaries: torch.Tensor, out_int32: bool = False
) -> torch.Tensor:
    """``torch.bucketize(values, boundaries, right=True, out_int32=out_int32)``.

    ``boundaries`` sorted.
    """
    if not envs.TPU_ONEHOT_TABLE_LOOKUP:
        return torch.bucketize(values, boundaries, right=True, out_int32=out_int32)
    dtype = torch.int32 if out_int32 else torch.int64
    return (boundaries <= values.unsqueeze(-1)).sum(-1, dtype=dtype)
