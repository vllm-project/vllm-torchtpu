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
"""The MK draft programs compile only at the kernel's rows, whatever config is current.

za9j c4 j-d4040629: the step-0 program was constructed under the stock draft config;
vLLM's compile wrapper builds its backend from the *current* config, so the step-0
draft kernel was compiled at every stock size up to max_num_batched_tokens (3072 rows,
VMEM 678 MiB) and the spec-decode warmup hung. Here the stock config carries stock
sizes up to 3072 plus the open range [.., max_num_batched_tokens], and every range the
MK programs' backends hold (and every row count their kernel is traced at) must stay
within the kernel's rows.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

MAX_TOKENS = 3072


@pytest.fixture
def cfg(monkeypatch):
    import vllm.envs
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig
    from vllm.config.compilation import CompilationMode
    from vllm.config.utils import Range

    monkeypatch.setattr(vllm.envs, "VLLM_USE_AOT_COMPILE", False, raising=False)

    # vLLM's own range logic (the TPU plugin overrides it with []): the open range
    # [last endpoint + 1, max_num_batched_tokens] is back in play.
    def vllm_ranges(self):
        if self.compile_ranges_endpoints is None:
            return []
        ends = sorted(set(self.compile_ranges_endpoints))
        return [Range(s + 1, e) for s, e in zip([0] + ends[:-1], ends)]

    monkeypatch.setattr(CompilationConfig, "get_compile_ranges", vllm_ranges)
    c = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, backend="eager",
            compile_sizes=[4, 8, 16, 32, 64, 1024, MAX_TOKENS],
        ),
    )
    c.compilation_config.compile_ranges_endpoints = [MAX_TOKENS]
    assert any(r.end == MAX_TOKENS and r.start < r.end for r in vllm_ranges(c.compilation_config))
    return c


@pytest.fixture
def ranges_seen(monkeypatch):
    from vllm.compilation import piecewise_backend as pb

    seen: list = []
    init = pb.PiecewiseBackend.__init__

    def record(self, *a, **k):
        init(self, *a, **k)
        seen.extend(self.range_entries)

    monkeypatch.setattr(pb.PiecewiseBackend, "__init__", record)
    return seen


class _StubMtp(nn.Module):
    def __init__(self, max_rows):
        super().__init__()
        self.model = NS(mk_draft=NS(max_rows=max_rows))
        self.traced_rows: list = []

    def mk_draft_step(self, input_ids, positions, hidden, rows_per_request):
        self.traced_rows.append(hidden.shape[0])
        return hidden + 1, input_ids

    def forward(self, input_ids, positions, hidden, *a, **k):
        return hidden * 2

    def compute_logits(self, hidden):
        return hidden


def _ends(ranges):
    return sorted({(r.start, r.end) for r in ranges})


def test_step0_program_compiles_only_kernel_rows(cfg, ranges_seen):
    from vllm.config import set_current_vllm_config

    from vllm_torchtpu.spec_decode.mtp_fused_loop import (
        build_mk_draft_step0,
        mk_draft_compile_config,
    )

    mtp = _StubMtp({4: 32, 1: 16})
    mk_cfg, sizes = mk_draft_compile_config(cfg, mtp.model, 4)
    assert sizes == [4, 8, 16, 32]
    assert mk_cfg.compilation_config.get_compile_ranges() == []
    # The caller's context is the STOCK config (as eagle3 had it): the builder must
    # still compile with the MK config.
    with set_current_vllm_config(cfg):
        step0 = build_mk_draft_step0(
            vllm_config=mk_cfg, prefix="mk_draft_step0", mtp=mtp, rows_per_request=4,
            dynamic_arg_dims={"input_ids": 0, "positions": -1, "hidden_states": 0,
                              "draft_rows": 0},
        )
        rows = 8
        out = step0(torch.zeros(rows, dtype=torch.int32), torch.arange(rows),
                    torch.ones(rows, 4), torch.tensor([3, 7]))
    torch.testing.assert_close(out[0], torch.full((2, 4), 2.0))
    assert ranges_seen, "the program never reached a PiecewiseBackend"
    assert max(e for _, e in _ends(ranges_seen)) <= 32, _ends(ranges_seen)
    assert all(not isinstance(r, int) or r <= 32 for r in mtp.traced_rows), mtp.traced_rows


def test_fused_loop_program_compiles_only_kernel_rows(cfg, ranges_seen):
    from vllm.config import set_current_vllm_config

    from vllm_torchtpu.spec_decode.mtp_fused_loop import (
        build_mtp_fused_loop,
        mk_draft_compile_config,
    )

    mtp = _StubMtp({4: 32, 1: 16})
    mk_cfg, sizes = mk_draft_compile_config(cfg, mtp.model, 1)
    assert sizes == [4, 8, 16]
    with set_current_vllm_config(cfg):
        loop = build_mtp_fused_loop(
            vllm_config=mk_cfg, prefix="mk_mtp_fused_loop_k2", mtp=mtp, num_steps=1,
            dynamic_arg_dims={"input_ids": 0, "positions": -1, "hidden_states": 0},
            use_mk=True,
        )
        rows = 4
        loop(torch.zeros(rows, dtype=torch.int32), torch.arange(rows),
             torch.ones(rows, 4), torch.zeros(rows))
    assert ranges_seen
    assert max(e for _, e in _ends(ranges_seen)) <= 16, _ends(ranges_seen)


def test_stock_context_alone_would_compile_the_open_range(cfg, ranges_seen):
    """Control: the same module built from the stock config holds the open range up to
    max_num_batched_tokens, i.e. the fixture really exercises it."""
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import set_current_vllm_config

    class M(nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()

        def forward(self, x):
            return x + 1

    cls = support_torch_compile(dynamic_arg_dims={"x": 0})(type("StockM", (M,), {}))
    with set_current_vllm_config(cfg):
        m = cls(vllm_config=cfg, prefix="stock")
        m(torch.ones(4))
    assert max(e for _, e in _ends(ranges_seen)) == MAX_TOKENS
