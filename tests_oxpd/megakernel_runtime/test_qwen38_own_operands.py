# SPDX-License-Identifier: Apache-2.0
"""Megakernel operands registered as non-view tensors, with no value change."""

import torch

from vllm_torchtpu.megakernel_runtime.qwen38_weights import own_operands


def test_own_operands_drops_views_without_changing_values():
    base = torch.arange(2 * 3 * 1 * 4, dtype=torch.float32).reshape(2, 3, 1, 4)
    merged = base.reshape(2, 3, 4)  # contiguous view at offset 0 (the "es" operand)
    strided = torch.arange(24.0).reshape(4, 6).t()  # non-contiguous view
    sliced = torch.arange(24.0)[6:]  # offset view
    plain = torch.ones(5)
    out, fixed = own_operands([merged, strided, sliced, plain])
    assert fixed == 3
    assert all(t._base is None for t in out)
    assert out[3] is plain
    assert out[0].data_ptr() == base.data_ptr()  # no copy
    for got, want in zip(out, [merged, strided, sliced, plain]):
        assert torch.equal(got, want)
        assert got.shape == want.shape and got.dtype == want.dtype
