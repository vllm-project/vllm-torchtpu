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
"""`build_jax_step_op`: a JAX step as a torch callable, run by JAX on CPU;
its fake kernel's shapes with sharded specs; its operand checks; and a
toy plugin whose `StepFn` is one, serving through the adapter."""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import PartitionSpec as P
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu.megakernel.adapter import testkit
from vllm_torchtpu.megakernel.adapter.jax_bridge import (
    build_jax_step_op,
    jax_to_torch_cpu,
    sharded_jax_callable,
    torch_cpu_to_jax,
)


def _mesh(n: int = 1) -> jax.sharding.Mesh:
    return jax.sharding.Mesh(np.asarray(jax.devices()[:n]), ("tp",))


@pytest.fixture
def tp2() -> jax.sharding.Mesh:
    """A 2-device `("tp",)` mesh (the suite's CPU devices)."""
    if len(jax.devices()) < 2:
        pytest.skip(f"needs 2 JAX devices, found {len(jax.devices())}")
    return _mesh(2)


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
    with FakeTensorMode():
        fake = step.op([torch.empty(7, 2, dtype=torch.float32)])
    assert [tuple(t.shape) for t in fake] == [(2,)]


def test_bad_donate_and_cpu_interpret_off() -> None:
    with pytest.raises(ValueError, match="output 0 to input 1: out of range"):
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


def _shapes(outs: Any) -> list[tuple[int, ...]]:
    return [tuple(t.shape) for t in outs]


@pytest.mark.parametrize(
    "in_spec, out_spec, local_out, global_out",
    [(P(), P("tp"), (4,), (8,)), (P("tp"), P(), (16,), (8,))],
    ids=["replicated_in_sharded_out", "sharded_in_replicated_out"],
)
def test_fake_kernel_follows_the_specs(
    tp2: jax.sharding.Mesh,
    in_spec: P,
    out_spec: P,
    local_out: tuple[int, ...],
    global_out: tuple[int, ...],
) -> None:
    """At TP=2 on an (8,) operand: a TPU rank passes and gets its shards, so
    the fake kernel scales the operand to the global array by `in_specs` and
    shrinks the output by `out_specs`; with CPU tensors the op runs over the
    whole mesh in one process, so the operands and outputs are global."""
    step = build_jax_step_op(
        lambda x: x, mesh=tp2, in_specs=(in_spec,), out_specs=out_spec
    )
    with FakeTensorMode():
        rank = step.op([torch.empty(8, device="meta")])
        whole = step.op([torch.empty(8)])
    assert _shapes(rank) == [local_out]
    assert _shapes(whole) == [global_out]
    # The CPU path returns what its fake promises.
    out = step(torch.arange(8, dtype=torch.float32))
    assert tuple(out.shape) == global_out
    assert torch.equal(out, torch.arange(8, dtype=torch.float32))


def test_fake_kernel_with_multi_axis_and_trailing_dims() -> None:
    if len(jax.devices()) < 4:
        pytest.skip(f"needs 4 JAX devices, found {len(jax.devices())}")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("a", "b"))

    def fn(x: jax.Array, w: jax.Array) -> tuple[Any, Any]:
        return x.sum(axis=1), jnp.concatenate([w, w])

    step = build_jax_step_op(
        fn,
        mesh=mesh,
        in_specs=(P(("a", "b")), P("a")),
        out_specs=(P("b"), P(None)),
    )
    with FakeTensorMode():
        out = step.op(
            [torch.empty(3, 5, device="meta"), torch.empty(2, 6, device="meta")]
        )
    # Global x (12, 5), w (4, 6): sum (12,) over "b" -> 6; concat (8, 6).
    assert _shapes(out) == [(6,), (8, 6)]
    # An output this rank can't hold an equal shard of names the mismatch.
    odd = build_jax_step_op(
        lambda x: x[:3], mesh=mesh, in_specs=(P(),), out_specs=P("b")
    )
    with (
        pytest.raises(
            ValueError, match="dimension 3 is not divisible by mesh factor 2"
        ),
        FakeTensorMode(),
    ):
        odd.op([torch.empty(8, device="meta")])


def test_every_build_registers_its_own_op() -> None:
    """The driver builds a step per bucket and again on each bind, so even a
    full `namespace::op` name gets a per-op suffix."""
    names = set()
    for name in ("mk_tests::same_op", "mk_tests::same_op", "plain", None):
        step = build_jax_step_op(
            lambda a: a + 1, mesh=_mesh(), in_specs=(P(),), out_specs=P(), name=name
        )
        assert torch.equal(step(torch.zeros(2)), torch.ones(2))
        names.add(step.op_name)
    assert len(names) == 4
    assert {n.rsplit("_", 1)[0] for n in names} == {
        "mk_tests::same_op",
        "megakernel::plain",
        "megakernel::jax_step",
    }


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn, torch.int32]
)
def test_cpu_conversions_round_trip_bit_exactly(dtype: torch.dtype) -> None:
    src = torch.linspace(-3, 3, 12).reshape(3, 4).to(dtype)
    arr = torch_cpu_to_jax(src)
    assert arr.shape == (3, 4)
    back = jax_to_torch_cpu(arr, src)
    assert back.dtype == dtype
    assert torch.equal(back.view(torch.uint8), src.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_cpu_conversion_of_64_bit_dtypes(dtype: torch.dtype) -> None:
    """Without `jax_enable_x64`, a 64-bit tensor raises instead of narrowing
    silently; with it, the round trip is bit-exact."""
    src = torch.tensor([[2**40 + 3, -(2**33)], [7, -1]]).to(dtype)
    match = f"{dtype} tensor needs jax_enable_x64"
    with jax.enable_x64(False), pytest.raises(ValueError, match=match):
        torch_cpu_to_jax(src)
    with jax.enable_x64(True):
        back = jax_to_torch_cpu(torch_cpu_to_jax(src), src)
    assert back.dtype == dtype and torch.equal(back, src)


def test_jax_to_torch_cpu_rejects_an_array_of_another_dtype() -> None:
    """The conversion reinterprets bytes: a float32 array read as bfloat16
    would come back with a doubled last dimension, so it raises."""
    with pytest.raises(
        ValueError, match="float32 array can't be read as torch.bfloat16"
    ):
        jax_to_torch_cpu(
            jnp.ones((2, 3), jnp.float32), torch.empty(0, dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="int32 array can't be read as torch.float32"):
        jax_to_torch_cpu(jnp.ones((2,), jnp.int32), torch.empty(0))


def test_float64_steps_and_complex64_conversion() -> None:
    """float64 operands and results go through the op with `jax_enable_x64`;
    complex64 (two 32-bit floats) converts without it."""
    step = build_jax_step_op(
        lambda a: a * 2.0, mesh=_mesh(), in_specs=(P(),), out_specs=P()
    )
    src = torch.tensor([0.1, 2.0**60 + 1.0], dtype=torch.float64)
    with jax.enable_x64(True):
        out = step(src)
    assert out.dtype == torch.float64 and torch.equal(out, src * 2.0)
    c = torch.tensor([1 + 2j, -3.5j], dtype=torch.complex64)
    with jax.enable_x64(False):
        arr = torch_cpu_to_jax(c)
    assert arr.dtype == jnp.complex64
    assert torch.equal(jax_to_torch_cpu(arr, c), c)


@pytest.mark.parametrize(
    ("fn", "match"),
    [
        (lambda s: s.astype(jnp.float32), "donated output 0 is torch.float32"),
        (lambda s: jnp.concatenate([s, s]), r"donated output 0 is torch.int32\[8\]"),
    ],
    ids=["dtype", "shape"],
)
def test_donated_output_must_match_its_input(fn: Any, match: str) -> None:
    """`copy_` would cast or broadcast a mismatched donated output silently."""
    step = build_jax_step_op(
        fn, mesh=_mesh(), in_specs=(P(),), out_specs=P(), donate={0: 0}
    )
    state = torch.arange(4, dtype=torch.int32)
    with pytest.raises(ValueError, match=match):
        step(state)
    assert state.tolist() == [0, 1, 2, 3]


def test_operands_must_share_one_device_in_the_op_and_its_fake() -> None:
    """A CPU operand among non-CPU ones would take the CPU path and copy the
    others to the host; the fake kernel makes the op's checks too."""
    step = build_jax_step_op(
        lambda a, b: a + b, mesh=_mesh(), in_specs=(P(), P()), out_specs=P()
    )
    mixed = [torch.ones(2), torch.ones(2, device="meta")]
    with pytest.raises(ValueError, match="operands on more than one device"):
        step.op(mixed)
    with FakeTensorMode():
        fake_mixed = [torch.ones(2), torch.ones(2, device="meta")]
        with pytest.raises(ValueError, match="operands on more than one device"):
            step.op(fake_mixed)
        with pytest.raises(ValueError, match="1 operands, in_specs has 2"):
            step.op([torch.ones(2)])
    strict = build_jax_step_op(
        lambda a: a,
        mesh=_mesh(),
        in_specs=(P(),),
        out_specs=P(),
        interpret_on_cpu=False,
    )
    with FakeTensorMode(), pytest.raises(RuntimeError, match="interpret_on_cpu=False"):
        strict.op([torch.ones(2)])


def test_eager_step_with_donation_and_device_tensors_raises() -> None:
    """Outside `torch.compile`, a donating step with device (non-CPU) tensors
    would copy each donated input every call; it raises instead. CPU
    operands (the interpret path) and steps without donation are allowed."""
    step = build_jax_step_op(
        lambda s: s + 1, mesh=_mesh(), in_specs=(P(),), out_specs=P(), donate={0: 0}
    )
    with pytest.raises(RuntimeError, match="must run under torch.compile"):
        step(torch.zeros(4, device="meta"))
    state = torch.zeros(4)
    step(state)
    assert state.tolist() == [1.0] * 4


def test_sharded_jax_callable_groups_operands_by_spec_and_donation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`pallas.jax_op` gets one `list[jax.Array]` argument per (spec, donated)
    group, the groups' specs and the donated groups' indices; the step sees
    its operands in their own order."""
    seen: dict[str, Any] = {}

    def jax_op(name: str, fn: Any, **kw: Any) -> Any:
        seen.update(kw, name=name, sig=inspect.signature(fn))

        def run(*groups: list[torch.Tensor]) -> list[torch.Tensor]:
            arrays = [[torch_cpu_to_jax(t) for t in g] for g in groups]
            return [jax_to_torch_cpu(a, torch.empty(0)) for a in fn(*arrays)]

        return run

    pallas = types.SimpleNamespace(jax_op=jax_op)
    internal = types.ModuleType("torch_tpu._internal")
    internal.pallas = pallas  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_tpu", types.ModuleType("torch_tpu"))
    monkeypatch.setitem(sys.modules, "torch_tpu._internal", internal)
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.pallas", pallas)

    def fn(a: Any, b: Any, pool: Any, c: Any) -> tuple[Any, Any]:
        return a + 10 * b + 100 * c, pool + 1

    specs = (P(), P("tp"), P("tp"), P())
    call = sharded_jax_callable(
        "mk_tests::grouped",
        fn,
        mesh=_mesh(),
        input_partition_specs=specs,
        output_partition_specs=(P(), P("tp")),
        donate_argnums=(2,),
    )
    assert seen["name"] == "mk_tests::grouped"
    assert seen["input_partition_specs"] == (P(), P("tp"), P("tp"))
    assert seen["output_partition_specs"] == (P(), P("tp"))
    assert seen["donate_argnums"] == (2,)
    assert [p.annotation for p in seen["sig"].parameters.values()] == [
        list[jax.Array]
    ] * 3
    assert seen["sig"].return_annotation == list[jax.Array]
    a, b, pool, c = (torch.full((2,), float(v)) for v in (1, 2, 3, 4))
    out, new_pool = call(a, b, pool, c)
    assert out.tolist() == [421.0, 421.0] and new_pool.tolist() == [4.0, 4.0]
    with pytest.raises(ValueError, match="one '::'"):
        sharded_jax_callable(
            "bad",
            fn,
            mesh=_mesh(),
            input_partition_specs=specs,
            output_partition_specs=(P(), P()),
        )


JAX_TOY_PLUGIN = '''
"""A toy megakernel plugin whose steps are JAX: y = x * scale + B."""
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.adapter.jax_bridge import build_jax_step_op
from vllm_torchtpu.megakernel.core.runtime.contract import (
    Bucket,
    StepDecision,
    TpuMonoSpec,
    WarmupCase,
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


def test_jax_toy_plugin_serves_through_try_forward(
    tmp_path: Path, mk_fake_runner: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    testkit.write_plugin(tmp_path, JAX_TOY_PLUGIN, "jax_toy_mk_plugin")
    monkeypatch.syspath_prepend(str(tmp_path))
    ids = torch.arange(6, dtype=torch.int32)
    res = testkit.run_cpu_e2e(
        "jax_toy_mk_plugin",
        mk_fake_runner,
        [(ids, torch.zeros(6))],
        options={"scale": 3.0},
    )
    assert sorted(res.driver._steps) == [(1, 4), (1, 8)]
    assert res.hits == [True] and torch.equal(res.outputs[0], ids.float() * 3.0 + 8)
    assert res.step_mix == {"mk_t1_b8": 1}


mk_fake_runner = testkit.mk_fake_runner
