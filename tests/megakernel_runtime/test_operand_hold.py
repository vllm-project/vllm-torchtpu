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
"""host_round_trip_small: small operands go through the host (values unchanged), big ones stay."""

import torch

from vllm_torchtpu.megakernel_runtime.operand_hold import host_round_trip_small


def test_small_operands_are_fresh_copies_with_equal_values():
    base = torch.zeros(4, 256)
    base[1:3] = torch.arange(512.0).reshape(2, 256)  # in-place slice write
    view = base.view(torch.int32)  # dtype reinterpretation: a view
    big = torch.randn(64, 64)
    ops = [base, view, big]
    n = host_round_trip_small(ops, limit=8 << 10)
    assert n == 2
    # (on CPU .cpu().to(cpu) is the identity; on TPU it is a host round-trip copy)
    assert torch.equal(ops[0], base) and torch.equal(ops[1], view)
    assert ops[2] is big  # over the limit: untouched


def test_fresh_copy_is_an_op_result_with_equal_values():
    """`own_operand` of a reshaped stack: equal values, a contiguous base tensor (no view), and
    not built by `copy_` (a `copy_` into the result would be caught: its grad_fn-free result
    still has `_base is None`, so the check is that `cat` produced it: version counter 0)."""
    from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_weights import fresh_copy, own_operand

    stack = torch.stack([torch.randn(6, 8, 16) for _ in range(3)])
    view = stack.reshape(-1, 8, 16)
    for t in (view, view[:1], stack.transpose(0, 1)):
        out = own_operand(t)
        assert torch.equal(out, t) and out.shape == t.shape
        assert out._base is None and out.is_contiguous() and out.storage_offset() == 0
        assert out._version == 0  # no in-place write after creation (empty_like().copy_() is 1)
    assert torch.empty_like(view).copy_(view)._version == 1  # positive control of the check
    assert own_operand(stack) is stack  # fresh stack: passed as is
    assert fresh_copy(torch.tensor(3.0)).shape == ()


def test_hbm_summary_counts_only_copies_as_own():
    """Packed operands: a parameter view is shared (total only), a fresh tensor is own; a storage
    counted once even when two operands view it."""
    from vllm_torchtpu.megakernel_runtime.operand_hold import hbm_summary

    model = torch.nn.Linear(256, 256, bias=False)  # 256 KiB fp32 weight
    fresh = torch.zeros(512, 512)  # 1 MiB
    ops = [model.weight.view(-1), fresh, fresh[:10]]
    assert hbm_summary(ops, model) == f"{(256 * 256 * 4 + 512 * 512 * 4) / 2**30:.2f} GiB, {512 * 512 * 4 / 2**30:.2f} GiB not shared with model parameters"
