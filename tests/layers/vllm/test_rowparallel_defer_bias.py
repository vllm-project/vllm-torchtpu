# Copyright 2025 Google LLC
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
"""Equivalence test for the TPU RowParallelLinear bias-defer patch.

``vllm_torchtpu._patch_rowparallel_defer_bias`` rewrites
``RowParallelLinear.forward`` so that, on the TP all-reduce path, the bias is
added *after* the all-reduce on every rank (instead of fused into the rank-0
GEMM). This keeps per-rank graphs identical for the TPU XLA backend while being
numerically identical to vLLM trunk. This test pins that equivalence: for a
simulated TP=2 reduction, the patched forward must produce the SAME output as
the unpatched (trunk) forward, for every (bias, skip_bias_add) combination.
"""

from types import SimpleNamespace

import pytest
import torch
import vllm.distributed as vdist
import vllm.model_executor.layers.linear as linear_mod
from vllm.model_executor.layers.linear import RowParallelLinear

import vllm_torchtpu


def _make_layer(tp_rank,
                tp_size,
                bias,
                weight_block,
                skip_bias_add,
                reduce_results=True,
                return_bias=False):
    """A minimal RowParallelLinear stand-in (avoids real distributed init)."""

    def apply(layer, inp, bias_):
        out = inp @ weight_block.T
        if bias_ is not None:
            out = out + bias_
        return out

    return SimpleNamespace(
        input_is_parallel=True,
        tp_rank=tp_rank,
        tp_size=tp_size,
        reduce_results=reduce_results,
        skip_bias_add=skip_bias_add,
        bias=bias,
        return_bias=return_bias,
        quant_method=SimpleNamespace(apply=apply),
    )


@pytest.mark.parametrize("bias_present", [True, False])
@pytest.mark.parametrize("skip_bias_add", [False, True])
def test_defer_bias_matches_trunk_tp2(monkeypatch, bias_present,
                                      skip_bias_add):
    """Patched forward == trunk forward under a simulated TP=2 all-reduce."""
    torch.manual_seed(0)
    out_f, in_f, tokens = 6, 8, 3
    W = torch.randn(out_f, in_f)
    x = torch.randn(tokens, in_f)
    b = torch.randn(out_f) if bias_present else None

    # RowParallelLinear shards the INPUT dim across ranks: W = [W0 | W1] column
    # blocks, x = [x0 | x1]. We drive rank 0 and fold rank 1's (always
    # bias-free) contribution into the mocked all-reduce -- identical for both
    # the trunk and patched paths, so any output difference is purely the
    # bias-placement change under test.
    half = in_f // 2
    W0, W1 = W[:, :half], W[:, half:]
    x0, x1 = x[:, :half], x[:, half:]
    op1 = x1 @ W1.T  # rank-1 output_parallel (bias-free in both paths)

    def fake_all_reduce(t):
        return t + op1

    # apply_tpu_patches() may already have installed the patch at import time
    # (full-suite run), so the live RowParallelLinear.forward can already be the
    # patched forward. Mock all_reduce in BOTH namespaces -- the trunk forward
    # reads it from the linear module globals, the patched forward closure-binds
    # it via `from vllm.distributed import ...` -- then (re-)apply the patch so
    # its closure captures the mock.
    monkeypatch.setattr(linear_mod, "tensor_model_parallel_all_reduce",
                        fake_all_reduce)
    monkeypatch.setattr(vdist, "tensor_model_parallel_all_reduce",
                        fake_all_reduce)
    saved_forward = RowParallelLinear.forward
    try:
        vllm_torchtpu._patch_rowparallel_defer_bias()
        patched_forward = RowParallelLinear.forward
        # Genuine upstream forward, preserved by the patch regardless of whether
        # it was already applied at import.
        trunk_forward = RowParallelLinear._tpu_upstream_forward
        # Guard against a vacuous comparison (patched vs patched).
        assert trunk_forward is not patched_forward

        out_trunk = trunk_forward(_make_layer(0, 2, b, W0, skip_bias_add), x0)
        out_patched = patched_forward(_make_layer(0, 2, b, W0, skip_bias_add),
                                      x0)
    finally:
        RowParallelLinear.forward = saved_forward  # restore pre-test state

    torch.testing.assert_close(out_patched, out_trunk)

    # And it is the mathematically correct row-parallel result.
    expected = x @ W.T
    if b is not None and not skip_bias_add:
        expected = expected + b
    torch.testing.assert_close(out_patched, expected)


def test_defer_bias_noop_at_tp1(monkeypatch):
    """At TP=1 there is no all-reduce; the patch must be a pure no-op vs trunk."""
    torch.manual_seed(0)
    out_f, in_f, tokens = 5, 4, 2
    W = torch.randn(out_f, in_f)
    x = torch.randn(tokens, in_f)
    b = torch.randn(out_f)

    saved_forward = RowParallelLinear.forward
    try:
        vllm_torchtpu._patch_rowparallel_defer_bias()
        patched_forward = RowParallelLinear.forward
        trunk_forward = RowParallelLinear._tpu_upstream_forward
        out_trunk = trunk_forward(_make_layer(0, 1, b, W, False), x)
        out_patched = patched_forward(_make_layer(0, 1, b, W, False), x)
    finally:
        RowParallelLinear.forward = saved_forward

    torch.testing.assert_close(out_patched, out_trunk)
    torch.testing.assert_close(out_patched, x @ W.T + b)
