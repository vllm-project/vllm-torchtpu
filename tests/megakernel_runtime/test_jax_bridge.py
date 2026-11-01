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
"""`build_jax_step_op`: a JAX step as a torch callable, run by JAX on CPU,
and a toy plugin whose `StepFn` is one, serving through the runtime."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.megakernel_runtime.agreement as agreement_mod
import vllm_torchtpu.megakernel_runtime.driver as driver_mod
from vllm_torchtpu.megakernel_runtime import fingerprint as fp
from vllm_torchtpu.megakernel_runtime import testing
from vllm_torchtpu.megakernel_runtime.hooks import StepContext
from vllm_torchtpu.megakernel_runtime.jax_bridge import build_jax_step_op


def _mesh() -> jax.sharding.Mesh:
    return jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("tp",))


def test_cpu_round_trip_copies_donated_outputs_back_in_place() -> None:
    def fn(x: jax.Array, state: jax.Array, w: jax.Array) -> tuple[Any, Any]:
        y = (x.astype(jnp.float32) * w).astype(jnp.bfloat16)
        return y, state + 1

    step = build_jax_step_op(
        fn,
        mesh=_mesh(),
        in_specs=(P(), P(), P()),
        out_specs=(P(), P()),
        donate={1: 1},
    )
    x = torch.tensor([1.5, -2.0, 3.25], dtype=torch.bfloat16)
    state = torch.arange(4, dtype=torch.int32)
    ptr = state.data_ptr()
    w = torch.tensor(2.0)
    y, new_state = step(x, state, w)
    assert y.dtype == torch.bfloat16 and torch.equal(y, x * 2)
    assert new_state is state and state.data_ptr() == ptr
    assert state.tolist() == [1, 2, 3, 4]
    step(x, state, w)
    assert state.tolist() == [2, 3, 4, 5]


def test_single_output_spec_returns_a_tensor_and_traces_with_a_fake() -> None:
    step = build_jax_step_op(
        lambda a: a.sum(axis=0), mesh=_mesh(), in_specs=(P(),), out_specs=P()
    )
    a = torch.ones(3, 5)
    out = step(a)
    assert isinstance(out, torch.Tensor) and torch.equal(out, torch.full((5,), 3.0))
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        fake = step.op([torch.empty(7, 2, dtype=torch.float32)])
    assert [tuple(t.shape) for t in fake] == [(2,)]


def test_bad_donate_and_cpu_interpret_off() -> None:
    with pytest.raises(ValueError, match="out of range"):
        build_jax_step_op(
            lambda a: a, mesh=_mesh(), in_specs=(P(),), out_specs=P(), donate={0: 1}
        )
    step = build_jax_step_op(
        lambda a: a,
        mesh=_mesh(),
        in_specs=(P(),),
        out_specs=P(),
        interpret_on_cpu=False,
    )
    with pytest.raises(RuntimeError, match="interpret_on_cpu=False"):
        step(torch.ones(2))


JAX_TOY_PLUGIN = '''
"""A toy megakernel plugin whose steps are JAX: y = x * scale + B."""
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel_runtime.jax_bridge import build_jax_step_op
from vllm_torchtpu.megakernel_runtime.plugin import (
    Bucket, StepDecision, TpuMonoSpec, WarmupCase,
)


class JaxToyProgram:
    display_name = "JaxToy"

    def __init__(self, options):
        self.scale = float(options.get("scale", 2.0))

    def buckets(self, token_paddings):
        return [Bucket(1, 4), Bucket(1, 8, out_rows=8)]

    def load(self, mesh):
        self.mesh = mesh

    def bind(self, view):
        return None

    def build(self, bucket):
        scale, b = self.scale, float(bucket.B)
        return build_jax_step_op(
            lambda x: x * scale + b, mesh=self.mesh, in_specs=(P(),), out_specs=P()
        )

    def warmup_inputs(self, bucket, shape):
        import torch

        return WarmupCase((torch.ones(shape.tok_pad),))

    def decide(self, ctx, fwd):
        n = int(ctx.input_ids.shape[0])
        b = next((b for b in (4, 8) if b >= n), None)
        if b is None:
            return StepDecision(False, "stock_big", n, 1)
        return StepDecision(True, f"mk_t1_b{b}", n, 1, b)

    def forward(self, decision, ctx, fwd):
        step = fwd.steps[(1, decision.bucket_t)]
        return step(ctx.input_ids.float())[: fwd.num_tokens]


def create(view, options):
    return JaxToyProgram(options)


SPEC = TpuMonoSpec("jax_toy", "1", create=create)
'''


@pytest.fixture
def jax_toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    (tmp_path / "jax_toy_mk_plugin.py").write_text(JAX_TOY_PLUGIN, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)
    for lazy in (driver_mod.logger, agreement_mod.logger):
        try:
            lazy._resolve()
        except ModuleNotFoundError:
            monkeypatch.setattr(lazy, "_logger", logging.getLogger(lazy._name))
    sys.modules.pop("jax_toy_mk_plugin", None)
    importlib.invalidate_caches()
    fp.clear_caches()
    yield
    sys.modules.pop("jax_toy_mk_plugin", None)
    fp.clear_caches()


def test_jax_toy_plugin_serves_through_try_forward(
    jax_toy: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: None)
    ctx = SimpleNamespace(attn_metadata={})
    monkeypatch.setattr(driver_mod, "get_forward_context", lambda: ctx)
    runner = testing.fake_runner(
        megakernel={"plugin": "jax_toy_mk_plugin", "options": {"scale": 3.0}}
    )
    drv = driver_mod.ProgramDriver.maybe_create(runner)
    assert isinstance(drv, driver_mod.ProgramDriver) and drv.serving
    runner._megakernel = drv
    drv.bind_and_warmup(runner)
    assert runner._megakernel is drv and sorted(drv._steps) == [(1, 4), (1, 8)]
    ids = torch.arange(6, dtype=torch.int32)
    hit, out = drv.try_forward(runner, StepContext(ids, torch.zeros(6)))
    assert hit and torch.equal(out, ids.float() * 3.0 + 8)
    assert drv.step_mix_tracker.counts == {"mk_t1_b8": 1}
