# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-exact coverage for paired host table gathering."""

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.common.host_gather import (
    gather_weight_scale,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize("rows", [0, 1, 3, 17, 1152])
@pytest.mark.parametrize("dim,blocks", [(32, 1), (256, 8), (96, 3)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layer_slice", [False, True])
def test_gather_weight_scale(
    rows: int, dim: int, blocks: int, index_dtype: torch.dtype, layer_slice: bool
) -> None:
    generator = torch.Generator().manual_seed(123)
    weight = torch.randint(256, (67, dim), dtype=torch.uint8, generator=generator)
    scales = torch.randint(256, (67, blocks), dtype=torch.uint8, generator=generator)
    indices = torch.randint(
        67,
        (rows, 2, 3) if layer_slice else (rows,),
        dtype=index_dtype,
        generator=generator,
    )
    if layer_slice:
        indices = indices[:, 1, :]
    if rows:
        indices[0] = 66
    count = indices.numel()
    packed = torch.full((count * (dim + blocks) + 9,), 43, dtype=torch.uint8)
    raw_out = packed[: count * dim].view(count, dim)
    scale_out = packed[count * dim : count * (dim + blocks)].view(count, blocks)

    gather_weight_scale(weight, scales, indices, raw_out, scale_out)

    assert torch.equal(raw_out, torch.index_select(weight, 0, indices.flatten()))
    assert torch.equal(scale_out, torch.index_select(scales, 0, indices.flatten()))
    assert torch.all(packed[-9:] == 43)


@pytest.mark.cpu_test
@pytest.mark.parametrize("index", [-(2**31), -1, 67, 2**31 - 1])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_gather_weight_scale_rejects_invalid_row(
    index: int, index_dtype: torch.dtype
) -> None:
    weight = torch.zeros((67, 32), dtype=torch.uint8)
    scales = torch.zeros((67, 1), dtype=torch.uint8)
    with pytest.raises(IndexError, match="outside the host table"):
        gather_weight_scale(
            weight,
            scales,
            torch.tensor([index], dtype=index_dtype),
            torch.empty((1, 32), dtype=torch.uint8),
            torch.empty((1, 1), dtype=torch.uint8),
        )
