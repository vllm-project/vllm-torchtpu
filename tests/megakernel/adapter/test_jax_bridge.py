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
its fake kernel's shapes with sharded specs; the typed step wrapper; and a
toy plugin whose `StepFn` is one, serving through the adapter."""

from __future__ import annotations

import inspect
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
    make_typed_step_wrapper,
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


def test_typed_step_wrapper_signature() -> None:
    """The wrapper `JaxCallable` inspects: fixed positional `jax.Array`
    parameters and a tuple of `jax.Array` results, calling the step."""
    seen: list[tuple[Any, ...]] = []

    def step(*args: Any) -> tuple[Any, ...]:
        seen.append(args)
        return args[::-1]

    typed = make_typed_step_wrapper("mk_step_t1_b8", step, 3, 2)
    sig = inspect.signature(typed, eval_str=True)
    assert typed.__name__ == "mk_step_t1_b8"
    assert list(sig.parameters) == ["x0", "x1", "x2"]
    for param in sig.parameters.values():
        assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert param.annotation is jax.Array
        assert param.default is inspect.Parameter.empty
    assert sig.return_annotation == tuple[jax.Array, jax.Array]
    assert typed(1, 2, 3) == (3, 2, 1) and seen == [(1, 2, 3)]
    one = make_typed_step_wrapper("f", step, 1, 1)
    assert inspect.signature(one, eval_str=True).return_annotation == tuple[jax.Array]


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
