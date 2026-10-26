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
"""The whole-step torch op traces with a symbolic token count.

vLLM traces the whole-step model once, with the token count symbolic, and compiles that graph
for every bucket. Anything at the torch level that branches on the token count (a guard on it)
makes every bucket but the traced one fail with ShapeSpecializationError: w06g / w07g
(j-b1a76c09, j-5100e473) lost the megakernel on every rank at the first call because the union
stats output's fake shape was computed from the token count. These tests trace the op the way
vLLM does (symbolic first dimension, fake tensors) with every option that adds an output, and
fail on any guard on the token count. Run them before building a whole-step image.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm_torchtpu.megakernel_runtime import qwen38_wholestep as ws
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

N_ATT, R = 11, 8


def _kind(tag: str) -> "ws._WholeKind":
    kind = ws._WholeKind(
        SimpleNamespace(size=R), None, None, [None] * N_ATT,
        budget=512, max_groups=8192, norm_eps=1e-6, rotary_dim=64, write=True, tag=tag,
    )
    kind.pool_index = 1  # args: [weight, pool]
    return kind


@pytest.fixture(params=["plain", "union_stats"])
def options(request, monkeypatch):
    stats = request.param == "union_stats"
    monkeypatch.setattr(wsc, "UNION_ATTN", stats)
    monkeypatch.setattr(wsc, "UNION_STATS_EVERY", 200 if stats else 0)
    monkeypatch.setattr(wsc, "SHARD_INDEX", True)
    monkeypatch.setattr(wsc, "LOOP_REQUESTS", True)
    return request.param


def _trace_guards(kind) -> tuple[list[str], list]:
    """Guards on the symbolic token count, and the op's outputs, from tracing it like vLLM."""

    def step(hidden, weight, pool):
        return kind.op(hidden, [weight, pool])

    gm = make_fx(step, tracing_mode="symbolic")(
        torch.zeros(4, 2560, dtype=torch.bfloat16),
        torch.zeros(16, dtype=torch.bfloat16),
        torch.zeros(8, 4, dtype=torch.float32),
    )
    for node in gm.graph.nodes:
        val = node.meta.get("val")
        if isinstance(val, (list, tuple)) and val and hasattr(val[0], "fake_mode"):
            return [str(g.expr) for g in val[0].fake_mode.shape_env.guards], list(val)
    raise AssertionError("the op's outputs were not traced")


def test_op_traces_without_guards_on_the_token_count(options):
    guards, outs = _trace_guards(_kind(f"trace_{options}"))
    assert not guards, f"the whole-step op guards on the token count: {guards}"
    assert len(outs) == (3 if options == "union_stats" else 2)
    for v in outs[2:]:  # the stats: static, whatever the token count (the pool follows its input)
        assert all(isinstance(d, int) for d in v.shape), v.shape
        # Torch sees this rank's shard: one row block per QSA layer, not R of them (w08).
        assert v.shape[0] == N_ATT, v.shape


def test_a_token_count_dependent_fake_shape_is_caught(options):
    """Positive control: the pre-7341892cb fake (stats rows from the token count) guards."""
    if options != "union_stats":
        pytest.skip("only the stats output had the dependent shape")
    kind = _kind("trace_control")

    def bad_fake(hidden, args):
        rows = (int(hidden.shape[0]) + 3) // 4  # what ExecSpec(B).M // T amounted to
        return [torch.empty_like(hidden), torch.empty_like(args[1]),
                hidden.new_empty((R * N_ATT, rows, 1, 128), dtype=torch.int32)]

    kind.op.register_fake(bad_fake)
    guards, _ = _trace_guards(kind)
    assert guards, "a token-count-dependent output shape must show up as a guard"
