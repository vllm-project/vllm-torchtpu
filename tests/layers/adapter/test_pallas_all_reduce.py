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
"""pallas_all_reduce (TPU_PALLAS_SMALL_ALL_REDUCE_MAX_ROWS): the op body on 16
CPU devices, the communicator's routing and arming.

Only the armed row counts reach the op. Every other row count keeps the
communicator's XLA all-reduce (DeviceCommunicatorBase.all_reduce).

The 16-device check runs this file as a script in a subprocess, because the
host device count (XLA_FLAGS) must be set before JAX starts. It uses TPU
interpret mode with the race detector, so it checks the path choice, the DMA
and semaphore protocol and the arithmetic, not Mosaic lowering, VMEM limits
or TPU timing. Routing and arming run in this process with the mesh, the
TP-group gather and the self-test replaced by stand-ins.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

from vllm_torchtpu.distributed.tpu_communicator import TpuDeviceCommunicator
from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.layers.adapter import pallas_all_reduce

N_RANKS = 16
# GLM-5.2's hidden size; the decode TP all-reduce is bf16 [16 or 32, 6144].
HIDDEN = 6144
MAX_ROWS = 32
# Row counts the kernel serves at MAX_ROWS. 48 (above the cap), 8 (below one
# bf16 row tile) and 24 (not a multiple of 16) are not served: the op body
# refuses them, and the communicator keeps the XLA all-reduce for them.
KERNEL_ROWS = (16, 32)
UNSERVED_ROWS = (48, 8, 24)
ENV = "TPU_PALLAS_SMALL_ALL_REDUCE_MAX_ROWS"


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


# ---------------------------------------------------------------------------
# The op body on 16 devices.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def on_16_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        # bf16 rounded after every op, as on the TPU and in the reference.
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N_RANKS}"
        " --xla_allow_excess_precision=false",
    )
    proc = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


def test_buckets_at_the_cap(on_16_devices):
    assert on_16_devices["buckets"] == list(KERNEL_ROWS)


@pytest.mark.parametrize("rows", KERNEL_ROWS)
def test_kernel_rows_equal_the_rank_order_f32_sum(on_16_devices, rows):
    case = on_16_devices["cases"][str(rows)]
    assert case["pallas_call"], case
    assert not case["races_found"], case
    assert case["ranks_identical"], case
    # Random partials: the kernel's f32 sum in rank order, rounded to bf16.
    assert case["random_bit_exact"], case
    # The start-up self-test's partials and check.
    assert case["selftest_ok"], case


@pytest.mark.parametrize("rows", UNSERVED_ROWS)
def test_op_body_refuses_unserved_rows(on_16_devices, rows):
    case = on_16_devices["cases"][str(rows)]
    assert case["refused"], case


def _interpret(fn, *args):
    """fn(*args) with fresh interpreter state; (out, whether races were found)."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    pltpu.reset_tpu_interpret_mode_state()
    with contextlib.redirect_stdout(io.StringIO()):
        out = np.asarray(jax.block_until_ready(fn(*args)))
    return out, bool(interpret_pallas_call.races.races_found)


def _worker():
    from jax._src.pallas.mosaic import tpu_info
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    pltpu.get_tpu_info = _tpu7x
    tpu_info.get_tpu_info = _tpu7x
    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devices) == N_RANKS, devices
    mesh = Mesh(np.asarray(devices), ("d",))
    buckets = pallas_all_reduce.row_buckets(N_RANKS, HIDDEN, MAX_ROWS, _tpu7x())
    run = jax.jit(
        jax.shard_map(
            pallas_all_reduce._op_body(N_RANKS, buckets),
            mesh=mesh,
            in_specs=(P("d"), P("d")),
            out_specs=P("d"),
            check_vma=False,
        )
    )
    rank_ids = jax.device_put(
        jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1),
        NamedSharding(mesh, P("d")),
    )

    def call(partials):
        """partials: f32 [n, rows, h], exact in bf16 -> (out [n, rows, h] f32,
        races found)."""
        rows = partials.shape[1]
        x = jax.device_put(
            jnp.asarray(partials.reshape(N_RANKS * rows, HIDDEN), jnp.bfloat16),
            NamedSharding(mesh, P("d")),
        )
        out, races = _interpret(run, x, rank_ids)
        return out.astype(np.float32).reshape(N_RANKS, rows, HIDDEN), races

    def identical(out):
        bits = out.view(np.uint32)
        return bool((bits == bits[0]).all())

    result = {"buckets": list(buckets), "cases": {}}
    rng = np.random.default_rng(0)
    for rows in UNSERVED_ROWS:
        probe = jnp.zeros((N_RANKS * rows, HIDDEN), jnp.bfloat16)
        try:
            jax.make_jaxpr(run)(probe, rank_ids)
            refused = False
        except ValueError as e:
            refused = "reached the op" in str(e)
        result["cases"][str(rows)] = {"refused": refused}
    for rows in KERNEL_ROWS:
        probe = jnp.zeros((N_RANKS * rows, HIDDEN), jnp.bfloat16)
        case = {
            "pallas_call": "pallas_call" in str(jax.make_jaxpr(run)(probe, rank_ids))
        }
        partials = np.asarray(
            jnp.asarray(rng.standard_normal((N_RANKS, rows, HIDDEN)), jnp.bfloat16),
            np.float32,
        )
        want = partials[0].copy()
        for r in range(1, N_RANKS):
            want = want + partials[r]
        want = np.asarray(jnp.asarray(want, jnp.bfloat16), np.float32)
        out, races = call(partials)
        spec = all_reduce.make_spec(N_RANKS, rows, HIDDEN)
        exact = all_reduce.selftest_inputs(spec)
        out_exact, races_exact = call(exact)
        case.update(
            races_found=races or races_exact,
            ranks_identical=identical(out) and identical(out_exact),
            random_bit_exact=bool(
                np.array_equal(
                    out.view(np.uint32),
                    np.broadcast_to(want, out.shape).view(np.uint32),
                )
            ),
            selftest_ok=all(
                all_reduce.selftest_ok(spec, exact, out_exact[r], rows)
                for r in range(N_RANKS)
            ),
        )
        result["cases"][str(rows)] = case
    print("RESULT " + json.dumps(result))


# ---------------------------------------------------------------------------
# Routing in TpuDeviceCommunicator.all_reduce.
# ---------------------------------------------------------------------------
class _Op:
    """Stand-in for the armed torch op: records its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x, rank):
        self.calls += 1
        return x * N_RANKS


@pytest.fixture
def armed(monkeypatch):
    op = _Op()
    monkeypatch.setattr(
        pallas_all_reduce,
        "_ARMED",
        pallas_all_reduce._Armed(
            op=op,
            rank=torch.zeros(1, 1, dtype=torch.int32),
            ranks=tuple(range(N_RANKS)),
            hidden=HIDDEN,
            buckets=KERNEL_ROWS,
        ),
    )
    return op


@pytest.fixture
def communicator(monkeypatch):
    """A TpuDeviceCommunicator over the 16 ranks whose XLA all-reduce is a
    marker: it returns x + 1."""
    comm = object.__new__(TpuDeviceCommunicator)
    comm.ranks = list(range(N_RANKS))
    monkeypatch.setattr(DeviceCommunicatorBase, "all_reduce", lambda self, x: x + 1)
    return comm


def _ones(*shape, dtype=torch.bfloat16):
    return torch.ones(*shape, dtype=dtype)


def test_armed_row_counts_route_to_the_op(armed, communicator):
    for rows in KERNEL_ROWS:
        out = communicator.all_reduce(_ones(rows, HIDDEN))
        assert torch.equal(out, _ones(rows, HIDDEN) * N_RANKS)
    assert armed.calls == len(KERNEL_ROWS)


# 4096: a prefill step's token count.
@pytest.mark.parametrize("rows", (*UNSERVED_ROWS, 64, 4096))
def test_other_row_counts_keep_the_xla_all_reduce(armed, communicator, rows):
    x = _ones(rows, HIDDEN)
    assert torch.equal(communicator.all_reduce(x), x + 1)
    assert armed.calls == 0


def test_symbolic_row_count_keeps_the_xla_all_reduce(armed, communicator):
    """A backed SymInt with hint 16 (an armed row count): reading its value
    would add a guard, so the communicator must keep the XLA all-reduce."""
    from torch._dynamo.source import ConstantSource
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    symbol = shape_env.create_symbol(16, ConstantSource("rows"))
    rows = shape_env.create_symintnode(symbol, hint=16)
    with FakeTensorMode(shape_env=shape_env):
        x = torch.empty(rows, HIDDEN, dtype=torch.bfloat16)
        assert isinstance(x.shape[0], torch.SymInt)
        communicator.all_reduce(x)
    assert armed.calls == 0
    assert not shape_env.guards


@pytest.mark.parametrize(
    "x",
    [
        _ones(16, HIDDEN, dtype=torch.float32),
        _ones(16, HIDDEN // 2),
        _ones(2, 8, HIDDEN),
    ],
    ids=["f32", "other-hidden", "3d"],
)
def test_other_shapes_and_dtypes_keep_the_xla_all_reduce(armed, communicator, x):
    assert torch.equal(communicator.all_reduce(x), x + 1)
    assert armed.calls == 0


def test_another_group_keeps_the_xla_all_reduce(armed, communicator):
    communicator.ranks = list(range(8))
    x = _ones(16, HIDDEN)
    assert torch.equal(communicator.all_reduce(x), x + 1)
    assert armed.calls == 0


def test_unarmed_keeps_the_xla_all_reduce(monkeypatch, communicator):
    monkeypatch.setattr(pallas_all_reduce, "_ARMED", None)
    x = _ones(16, HIDDEN)
    assert torch.equal(communicator.all_reduce(x), x + 1)


# ---------------------------------------------------------------------------
# The env var and arming.
# ---------------------------------------------------------------------------
def test_env_default_is_off(monkeypatch):
    import vllm_torchtpu.envs as envs

    monkeypatch.delenv(ENV, raising=False)
    assert envs.TPU_PALLAS_SMALL_ALL_REDUCE_MAX_ROWS == 0
    monkeypatch.setenv(ENV, "-1")
    with pytest.raises(ValueError, match=ENV):
        _ = envs.TPU_PALLAS_SMALL_ALL_REDUCE_MAX_ROWS


@pytest.mark.parametrize(
    "max_rows, hidden, want",
    [
        (0, HIDDEN, ()),
        (8, HIDDEN, ()),
        (16, HIDDEN, (16,)),
        (40, HIDDEN, (16, 32)),
        (64, HIDDEN, (16, 32, 48, 64)),
        # 6000 / 16 = 375 columns per rank: not a multiple of 128 lanes.
        (32, 6000, ()),
    ],
)
def test_row_buckets(max_rows, hidden, want):
    assert pallas_all_reduce.row_buckets(N_RANKS, hidden, max_rows, _tpu7x()) == want


@pytest.fixture
def arming(monkeypatch):
    """arm() with a 16-rank topology, a gather that returns every rank's copy
    of the entry (mesh indices 0..15), and a self-test stand-in."""
    from vllm_torchtpu.layers.adapter import moe_tp_decode

    h = SimpleNamespace(selftest_rows=None, failures=[], disagree=False)
    monkeypatch.setenv(ENV, str(MAX_ROWS))
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    monkeypatch.setattr(
        pallas_all_reduce,
        "_topology",
        lambda: moe_tp_decode._Topology(
            mesh=SimpleNamespace(size=N_RANKS), mesh_index=0
        ),
    )

    def gather(obj):
        seen = [dict(obj, mesh_index=i) for i in range(N_RANKS)]
        if h.disagree and "plan" in obj:
            seen[5]["plan"] = dict(obj["plan"], buckets=(16,))
        return seen

    def selftest(op, rank, mesh_index, n_ranks, hidden, rows, device):
        h.selftest_rows = rows
        return h.failures

    monkeypatch.setattr(pallas_all_reduce, "_gather", gather)
    monkeypatch.setattr(pallas_all_reduce, "_build_op", lambda mesh, buckets: _Op())
    monkeypatch.setattr(pallas_all_reduce, "_selftest", selftest)
    monkeypatch.setattr(
        pallas_all_reduce,
        "_rank_tensor",
        lambda device, index: torch.tensor([[index]], dtype=torch.int32),
    )
    monkeypatch.setattr(
        pallas_all_reduce.parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(ranks=list(range(N_RANKS))),
    )
    monkeypatch.setattr(pallas_all_reduce, "_ARMED", None)
    return h


def test_arm_self_tests_the_buckets(arming):
    assert pallas_all_reduce.arm(HIDDEN, torch.device("cpu"))
    assert arming.selftest_rows == KERNEL_ROWS
    armed = pallas_all_reduce._ARMED
    assert armed is not None
    assert armed.hidden == HIDDEN and armed.ranks == tuple(range(N_RANKS))
    assert armed.buckets == KERNEL_ROWS


def test_arm_refuses_a_topology_it_cannot_serve(arming, monkeypatch):
    monkeypatch.setattr(pallas_all_reduce, "_topology", lambda: "no EP group")
    assert not pallas_all_reduce.arm(HIDDEN, torch.device("cpu"))
    assert pallas_all_reduce._ARMED is None and arming.selftest_rows is None


def test_arm_refuses_a_hidden_size_the_kernel_cannot_split(arming):
    assert not pallas_all_reduce.arm(6000, torch.device("cpu"))
    assert pallas_all_reduce._ARMED is None


def test_failed_selftest_raises_and_arms_nothing(arming):
    arming.failures = [32]
    with pytest.raises(RuntimeError, match="self-test failed"):
        pallas_all_reduce.arm(HIDDEN, torch.device("cpu"))
    assert pallas_all_reduce._ARMED is None


def test_ranks_must_agree(arming):
    arming.disagree = True
    with pytest.raises(RuntimeError, match="disagree"):
        pallas_all_reduce.arm(HIDDEN, torch.device("cpu"))
    assert pallas_all_reduce._ARMED is None and arming.selftest_rows is None


if __name__ == "__main__":
    # Around everything: pallas_call reads the mode when it is built.
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
