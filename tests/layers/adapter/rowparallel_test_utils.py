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
"""A simulated tensor-parallel group for row-parallel layer tests."""

import torch

import vllm_torchtpu.layers.adapter.linear as tpu_linear_mod


def run_tp_group(monkeypatch, upstream_mod, tp_size, run_rank):
    """Returns ``run_rank(rank)`` for every rank of a simulated TP group.

    A real all-reduce hands every rank the same sum, which takes two passes to
    imitate: capture each rank's pre-reduce output, then replay every rank with
    the all-reduce returning their sum.

    ``upstream_mod`` is the vLLM module whose forward is under comparison: it
    reads all_reduce from its own globals, the TPU forward from ours.
    """

    def run_all_ranks(all_reduce):
        for mod in (upstream_mod, tpu_linear_mod):
            monkeypatch.setattr(mod, "tensor_model_parallel_all_reduce", all_reduce)
        return [run_rank(rank) for rank in range(tp_size)]

    pre_reduce = []

    def capture(tensor):
        pre_reduce.append(tensor)
        return tensor

    run_all_ranks(capture)
    reduced = sum(pre_reduce) if pre_reduce else None
    return run_all_ranks(lambda tensor: reduced)


def as_pair(ret):
    return ret if isinstance(ret, tuple) else (ret, None)


def assert_same(ours, upstream, msg):
    out, bias = as_pair(ours)
    want_out, want_bias = as_pair(upstream)
    torch.testing.assert_close(out, want_out, msg=msg)
    assert (bias is None) == (want_bias is None), msg
    if bias is not None:
        torch.testing.assert_close(bias, want_bias, msg=msg)
