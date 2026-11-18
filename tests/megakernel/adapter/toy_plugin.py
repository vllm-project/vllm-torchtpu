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
"""The adapter tests' toy plugin: `y = x * scale + B` in pure torch, as the
single-file module `toy_mk_plugin`, and the `mk_toy_plugin` fixture that puts it
on `sys.path`."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from vllm_torchtpu.megakernel.adapter import fingerprint

TOY_PLUGIN = '''
"""A toy megakernel plugin: y = x * scale + B, in pure torch."""
from vllm_torchtpu.megakernel.core.runtime.contract import (
    Bucket,
    Decline,
    TpuMonoSpec,
    WarmupCase,
)


class ToyProgram:
    display_name = "Toy"

    def __init__(self, view, options):
        self.scale = float(options.get("scale", 2.0))
        self.mesh = "unloaded"
        self.warmed = []
        self.events = []

    def buckets(self, token_paddings):
        self.events.append(("buckets", tuple(token_paddings)))
        return [Bucket(1, 4), Bucket(1, 8, out_rows=8)]

    def load(self, mesh):
        self.mesh = mesh
        self.events.append("load")

    def bind(self, view):
        self.events.append("bind")
        return None

    def build(self, bucket):
        scale, b = self.scale, bucket.B

        def step(x):
            return x * scale + b

        return step

    def warmup_inputs(self, bucket, shape):
        import torch

        def check(out):
            self.warmed.append((bucket.key, shape, float(out[0])))

        return WarmupCase((torch.ones(shape.tok_pad),), check)

    def decide(self, ctx, fwd):
        from vllm_torchtpu.megakernel.core.runtime.contract import StepDecision

        n = int(ctx.input_ids.shape[0])
        b = next((b for b in (4, 8) if b >= n), None)
        if b is None:
            return StepDecision(False, "stock_big", n, 1)
        return StepDecision(True, f"mk_t1_b{b}", n, 1, b)

    def forward(self, decision, ctx, fwd):
        step = fwd.steps[(1, decision.bucket_t)]
        return step(ctx.input_ids.float())[: fwd.num_tokens]

    def on_warmup_done(self):
        self.events.append("warm")


def create(view, options):
    rank = view.tp_rank
    if options.get("decline") or options.get("decline_rank") == rank:
        return Decline("toy_declined", "toy says no")
    if options.get("raise_rank") == rank:
        raise RuntimeError("toy create bug")
    return ToyProgram(view, options)


SPEC = TpuMonoSpec(
    "toy", "1", create=create, decline_codes=("toy_declined",)
)
'''
TOY_MODULE = "toy_mk_plugin"


@pytest.fixture
def mk_toy_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """`TOY_PLUGIN` as the single-file module `toy_mk_plugin` on `sys.path`;
    yields its path (edit it to change the plugin source)."""
    path = tmp_path / f"{TOY_MODULE}.py"
    path.write_text(TOY_PLUGIN, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(TOY_MODULE, None)
    importlib.invalidate_caches()
    fingerprint.clear_caches()
    yield path
    sys.modules.pop(TOY_MODULE, None)
    fingerprint.clear_caches()
