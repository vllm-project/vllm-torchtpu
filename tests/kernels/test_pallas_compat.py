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
"""kernels/pallas_compat.py: kernels written against the internal Pallas DMA
hint API trace and run (TPU interpret mode on CPU) on the installed JAX, and
with the internal build's signatures installed the hints reach Pallas."""
import contextlib
import dataclasses
import importlib
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels import pallas_compat as pc

pytestmark = pytest.mark.cpu_test

INTERPRET = pltpu.InterpretParams()
ROWS, LANES = 8, 128
BLOCK = pl.BlockSpec((ROWS, LANES), lambda i: (i, 0))
ANY = pl.BlockSpec(memory_space=pl.ANY)


@pytest.fixture(autouse=True)
def _on_the_host():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@contextlib.contextmanager
def _tpu_info_on_cpu():
    """The pipeline's DMA slicing asks for the TPU generation; answer v6e for
    the CPU device (as tests/kernels/test_moe_onehot_unpermute.py does)."""
    tpu_info.get_tpu_info.cache_clear()
    jax.clear_caches()
    try:
        with mock.patch.dict(
                tpu_info.registry, {
                    "cpu":
                    lambda: tpu_info.get_tpu_info_for_chip(
                        tpu_info.ChipVersion.TPU_V6E, 1)
                }):
            yield
    finally:
        tpu_info.get_tpu_info.cache_clear()
        jax.clear_caches()


def _rows(n_blocks=1, dtype=jnp.float32):
    return jnp.arange(n_blocks * ROWS * LANES,
                      dtype=dtype).reshape(n_blocks * ROWS, LANES)


def _hbm_call(kernel, x, scratch_shapes=()):
    return pl.pallas_call(kernel,
                          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                          in_specs=[ANY],
                          out_specs=ANY,
                          scratch_shapes=scratch_shapes,
                          interpret=INTERPRET)


# ---- the installed JAX ----


@pytest.mark.parametrize("flag, call", [
    ("HAS_DMA_STRICT_ORDERING",
     lambda: pltpu.make_async_copy(None, None, None, strict_ordering=False)),
    ("HAS_REMOTE_DMA_STRICT_ORDERING", lambda: pltpu.make_async_remote_copy(
        None, None, None, None, (0, ), strict_ordering=False)),
    ("HAS_EMIT_PIPELINE_PRIORITY",
     lambda: pltpu.emit_pipeline(lambda: None, grid=(1, ), priority=1)),
    ("HAS_BUFFERED_REF_PRIORITY",
     lambda: pltpu.BufferedRef.create(None, None, None, None, priority=1)),
    ("HAS_REPEAT", lambda: pltpu.repeat),
])
def test_installed_jax_without_the_api_rejects_it(flag, call):
    """What the shims are for: calling the internal API directly fails."""
    if getattr(pc, flag):
        pytest.skip(f"the installed JAX has it ({flag})")
    with pytest.raises((TypeError, AttributeError)):
        call()


def test_local_copies_with_hints_run():

    def kernel(x_hbm, o_hbm, buf, sems):
        copy_in = pc.make_async_copy(x_hbm,
                                     buf,
                                     sems.at[0],
                                     priority=1,
                                     strict_ordering=False,
                                     wait_threshold=None)
        copy_in.start()
        copy_in.wait(strict_ordering=False, wait_threshold=None)
        buf[...] = buf[...] * 2
        pc.make_async_copy(buf, o_hbm, sems.at[1]).start(priority=1,
                                                         strict_ordering=False)
        # The byte-counted self-copy wait the GDN/RPA BufferedRefs use.
        pc.make_async_copy(o_hbm, o_hbm, sems.at[1],
                           strict_ordering=False).wait()

    x = _rows()
    out = _hbm_call(kernel, x, [
        pltpu.VMEM(x.shape, x.dtype),
        pltpu.SemaphoreType.DMA((2, )),
    ])(x)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x) * 2)


def test_priority_reaches_dma_start():
    """`priority` on `start` is public, so a descriptor's priority (given to
    `make_async_copy`, as fused_moe v3 does) must still be issued with it."""

    def kernel(x_hbm, o_hbm, sems):
        hinted = pc.make_async_copy(x_hbm, o_hbm, sems.at[0], priority=1)
        hinted.start()
        hinted.wait()
        plain = pc.make_async_copy(x_hbm, o_hbm, sems.at[0])
        plain.start()
        plain.wait()
        plain.start(priority=2, strict_ordering=False)
        plain.wait(strict_ordering=False)

    x = _rows()
    call = pl.pallas_call(kernel,
                          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                          in_specs=[ANY],
                          out_specs=ANY,
                          scratch_shapes=[pltpu.SemaphoreType.DMA((1, ))])
    (pallas_eqn, ) = jax.make_jaxpr(call)(x).jaxpr.eqns
    starts = [
        eqn.params["priority"] for eqn in pallas_eqn.params["jaxpr"].eqns
        if eqn.primitive.name == "dma_start"
    ]
    assert starts == [1, 0, 2]


@pytest.mark.parametrize("flag", [0, 1])
def test_start_when_issues_only_where_it_holds(flag):

    def kernel(flag_ref, x_hbm, o_ref, buf, sem):
        buf[...] = jnp.zeros_like(buf)
        copy = pc.make_async_copy(x_hbm, buf, sem.at[0], priority=1)
        on = flag_ref[0] == 1
        copy.start(when=on)

        @pl.when(on)
        def _():
            copy.wait()

        o_ref[...] = buf[...]

    x = _rows()
    out = pl.pallas_call(kernel,
                         out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                         in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM), ANY],
                         scratch_shapes=[
                             pltpu.VMEM(x.shape, x.dtype),
                             pltpu.SemaphoreType.DMA((1, ))
                         ],
                         interpret=INTERPRET)(jnp.array([flag], jnp.int32), x)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x) * flag)


def test_remote_copy_with_hints_runs():

    def kernel(x_ref, o_ref, send_sem, recv_sem):
        copy = pc.make_async_remote_copy(src_ref=x_ref,
                                         dst_ref=o_ref,
                                         send_sem=send_sem,
                                         recv_sem=recv_sem,
                                         device_id=(jax.lax.axis_index("x"), ),
                                         device_id_type=pl.DeviceIdType.MESH,
                                         strict_ordering=False)
        copy.start()
        copy.wait_send(strict_ordering=False)
        copy.wait_recv(strict_ordering=False)

    x = _rows()
    call = _hbm_call(kernel, x,
                     [pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA])
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("x", ))
    out = jax.jit(
        jax.shard_map(call,
                      mesh=mesh,
                      in_specs=P(),
                      out_specs=P(),
                      check_vma=False))(x)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


@pytest.mark.skipif(pc.HAS_DMA_WAIT_THRESHOLD,
                    reason="the installed JAX has DMA wait thresholds")
def test_wait_threshold_without_support_is_refused():
    """A threshold changes what is waited for, so it is never dropped."""

    def kernel(x_hbm, o_hbm, sems):
        copy = pc.make_async_copy(x_hbm, o_hbm, sems.at[0])
        copy.start()
        copy.wait(wait_threshold=64)

    x = _rows()
    with pytest.raises(NotImplementedError, match="wait_threshold=64"):
        jax.make_jaxpr(_hbm_call(kernel, x, [pltpu.SemaphoreType.DMA(
            (1, ))]))(x)


def test_emit_pipeline_with_priority_runs():

    def kernel(x_hbm, o_hbm):

        def body(x_ref, o_ref):
            o_ref[...] = x_ref[...] + 1

        pc.emit_pipeline(body,
                         grid=(4, ),
                         in_specs=[BLOCK],
                         out_specs=[BLOCK],
                         priority=1,
                         strict_ordering=False)(x_hbm, o_hbm)

    x = _rows(4)
    with _tpu_info_on_cpu():
        out = _hbm_call(kernel, x)(x)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x) + 1)


def test_buffered_ref_subclass_with_hints_runs_in_a_pipeline():
    """The pattern of the GDN v3 / batched RPA BufferedRefs: build through the
    base `create` with hints, copy the base fields into the subclass, and
    issue the DMAs with `self.priority` / `self.strict_ordering`."""
    issued = []

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class RowsInRef(pc.BufferedRef):
        rows: int = dataclasses.field(metadata=dict(static=True))

        @classmethod
        def create(cls, spec, dtype_or_type, buffer_type, buffer_count, rows,
                   **fields):
            standard_ref = pc.BufferedRef.create(
                spec=spec,
                dtype_or_type=dtype_or_type,
                buffer_type=buffer_type,
                buffer_count=buffer_count,
                grid_rank=1,
                use_lookahead=False,
                priority=fields.pop("priority", 0),
                strict_ordering=fields.pop("strict_ordering", False),
            )
            return cls(rows=rows,
                       **fields,
                       **{
                           f.name: getattr(standard_ref, f.name)
                           for f in dataclasses.fields(pc.BufferedRef)
                       })

        def copy_in(self, src_ref, grid_indices):
            issued.append(("copy_in", self.priority, self.strict_ordering))
            slot = self.current_copy_in_slot
            start = pl.multiple_of(grid_indices[0] * self.rows, self.rows)
            pc.make_async_copy(src_ref.at[pl.ds(start, self.rows)],
                               self.window_ref.at[slot],
                               self.sem_recvs.at[slot]).start(
                                   priority=self.priority,
                                   strict_ordering=self.strict_ordering)

        def wait_in(self, src_ref, grid_indices):
            issued.append(("wait_in", self.priority, self.strict_ordering))
            window = self.window_ref.at[self.current_wait_in_slot]
            pc.make_async_copy(
                window, window,
                self.sem_recvs.at[self.current_wait_in_slot]).wait(
                    strict_ordering=self.strict_ordering)

    ref = RowsInRef.input(BLOCK,
                          jnp.float32,
                          rows=ROWS,
                          priority=1,
                          strict_ordering=True)
    assert isinstance(ref, pltpu.BufferedRef)
    assert (ref.priority, ref.strict_ordering, ref.rows) == (1, True, ROWS)
    leaves, treedef = jax.tree.flatten(ref)
    assert jax.tree.unflatten(treedef, leaves) == ref  # hints are static

    def kernel(x_hbm, o_hbm):

        def body(x_ref, o_ref):
            o_ref[...] = x_ref[...] * 3

        allocations = [
            RowsInRef.input(BLOCK, jnp.float32, rows=ROWS, priority=1),
            pc.BufferedRef.output(BLOCK, jnp.float32, priority=1),
        ]
        pipeline = pc.emit_pipeline(body,
                                    grid=(4, ),
                                    in_specs=[BLOCK],
                                    out_specs=[BLOCK],
                                    priority=1)
        pl.run_scoped(
            lambda allocs: pipeline(x_hbm, o_hbm, allocations=allocs),
            allocations)

    x = _rows(4)
    with _tpu_info_on_cpu():
        out = _hbm_call(kernel, x)(x)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x) * 3)
    assert ("copy_in", 1, False) in issued and ("wait_in", 1, False) in issued


@pytest.mark.parametrize("shape, repeats, axis", [
    ((8, ), 4, 0),
    ((2, 3), 3, 0),
    ((2, 3), 2, 1),
    ((2, 3), 2, -1),
    ((2, 3, 4), 2, -2),
])
def test_repeat_tiles_like_the_internal_definition(shape, repeats, axis):
    x = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    reps = [1] * x.ndim
    reps[axis] = repeats
    np.testing.assert_array_equal(np.asarray(pc.repeat(x, repeats, axis)),
                                  np.tile(x, reps))


def test_repeat_rejects_an_out_of_range_axis():
    with pytest.raises(ValueError):
        pc.repeat(np.zeros((2, 3)), 2, 2)


def test_repeat_in_a_kernel():
    """The batched RPA use: a lane iota repeated down the rows."""

    def kernel(o_ref):
        o_ref[...] = pc.repeat(jnp.arange(LANES, dtype=jnp.int32),
                               ROWS,
                               axis=0).reshape(ROWS, LANES)

    out = pl.pallas_call(kernel,
                         out_shape=jax.ShapeDtypeStruct((ROWS, LANES),
                                                        jnp.int32),
                         interpret=INTERPRET)()
    np.testing.assert_array_equal(
        np.asarray(out), np.broadcast_to(np.arange(LANES), (ROWS, LANES)))


# ---- the internal build's signatures ----


class _InternalDescriptor:
    """`start` / `wait*` with the internal build's signatures, recording the
    arguments they are given."""

    def __init__(self, calls, src_ref):
        self.calls = calls
        self.src_ref = src_ref

    def start(self, priority=None, *, add=False, strict_ordering=None):
        self.calls.append(("start", priority, strict_ordering, add))

    def wait(self, *, strict_ordering=None, wait_threshold=None):
        self.calls.append(("wait", strict_ordering, wait_threshold))

    def wait_send(self, *, strict_ordering=None, wait_threshold=None):
        self.calls.append(("wait_send", strict_ordering, wait_threshold))

    def wait_recv(self, *, strict_ordering=None, wait_threshold=None):
        self.calls.append(("wait_recv", strict_ordering, wait_threshold))


class _PublicDescriptor:
    """`start` / `wait` with the public JAX signatures."""

    def __init__(self, calls):
        self.calls = calls

    def start(self, priority=0, *, add=False):
        self.calls.append(("start", priority, add))

    def wait(self):
        self.calls.append(("wait", ))


@pytest.fixture
def internal_build(monkeypatch):
    """`pallas_compat` re-imported over `pltpu` stand-ins that have the
    internal build's signatures."""
    calls = []

    def make_async_copy(src_ref,
                        dst_ref,
                        sem=None,
                        *,
                        src_sem=None,
                        _unsafe_no_write_effect=False,
                        priority=0,
                        strict_ordering=False,
                        wait_threshold=None):
        calls.append(
            ("make_async_copy", priority, strict_ordering, wait_threshold))
        return _InternalDescriptor(calls, src_ref)

    def make_async_remote_copy(src_ref,
                               dst_ref,
                               send_sem,
                               recv_sem,
                               device_id,
                               device_id_type=None,
                               *,
                               priority=0,
                               strict_ordering=False):
        calls.append(("make_async_remote_copy", device_id_type, priority,
                      strict_ordering))
        return _InternalDescriptor(calls, src_ref)

    def emit_pipeline(body,
                      *,
                      grid,
                      in_specs=(),
                      out_specs=(),
                      priority=0,
                      strict_ordering=False):
        calls.append(("emit_pipeline", grid, priority, strict_ordering))
        return body

    def repeat(x, repeats, axis):
        calls.append(("repeat", repeats, axis))
        return x

    public_buffered_ref = pltpu.BufferedRef

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass(frozen=True)
    class InternalBufferedRef(public_buffered_ref):
        priority: int = dataclasses.field(default=0,
                                          metadata=dict(static=True))
        strict_ordering: bool = dataclasses.field(default=False,
                                                  metadata=dict(static=True))

        @classmethod
        def create(cls, *args, priority=0, strict_ordering=False, **kwargs):
            calls.append(("BufferedRef.create", priority, strict_ordering))
            return dataclasses.replace(super().create(*args, **kwargs),
                                       priority=priority,
                                       strict_ordering=strict_ordering)

    monkeypatch.setattr(pltpu, "make_async_copy", make_async_copy)
    monkeypatch.setattr(pltpu, "make_async_remote_copy",
                        make_async_remote_copy)
    monkeypatch.setattr(pltpu, "emit_pipeline", emit_pipeline)
    monkeypatch.setattr(pltpu, "BufferedRef", InternalBufferedRef)
    monkeypatch.setattr(pltpu, "repeat", repeat, raising=False)
    try:
        yield importlib.reload(pc), calls
    finally:
        monkeypatch.undo()
        importlib.reload(pc)


def test_internal_build_is_detected(internal_build):
    compat, _ = internal_build
    flags = {
        name: value
        for name, value in vars(compat).items() if name.startswith("HAS_")
    }
    assert flags and all(flags.values()), flags


def test_local_dma_hints_are_forwarded(internal_build):
    compat, calls = internal_build
    copy = compat.make_async_copy("src",
                                  "dst",
                                  "sem",
                                  priority=1,
                                  strict_ordering=True,
                                  wait_threshold=64)
    copy.start()
    copy.start(2, strict_ordering=False, add=True)
    copy.wait()
    copy.wait(strict_ordering=False, wait_threshold=128)
    copy.start(when=False)
    copy.start(when=True)
    assert calls == [
        ("make_async_copy", 1, True, 64),
        ("start", 1, True, False),
        ("start", 2, False, True),
        ("wait", True, 64),
        ("wait", False, 128),
        ("start", 1, True, False),
    ]
    assert copy.src_ref == "src"  # other attributes come from the descriptor


def test_remote_dma_hints_are_forwarded(internal_build):
    compat, calls = internal_build
    copy = compat.make_async_remote_copy(src_ref="src",
                                         dst_ref="dst",
                                         send_sem="send",
                                         recv_sem="recv",
                                         device_id=(0, ),
                                         device_id_type="mesh",
                                         priority=1,
                                         strict_ordering=True)
    copy.start()
    copy.wait_send()
    copy.wait_recv(strict_ordering=False)
    assert calls == [
        ("make_async_remote_copy", "mesh", 1, True),
        ("start", 1, True, False),
        ("wait_send", True, None),
        ("wait_recv", False, None),
    ]


def test_emit_pipeline_and_buffered_ref_hints_are_forwarded(internal_build):
    compat, calls = internal_build
    compat.emit_pipeline(lambda: None,
                         grid=(2, ),
                         priority=1,
                         strict_ordering=True)
    ref = compat.BufferedRef.input(BLOCK, jnp.float32, priority=1)
    assert isinstance(ref, pltpu.BufferedRef)  # the stand-in
    assert (ref.priority, ref.strict_ordering) == (1, False)
    assert calls == [("emit_pipeline", (2, ), 1, True),
                     ("BufferedRef.create", 1, False)]


def test_native_repeat_is_used(internal_build):
    compat, calls = internal_build
    compat.repeat(np.zeros(3), 2, 0)
    assert calls == [("repeat", 2, 0)]


def test_public_descriptor_gets_only_what_it_takes():
    calls = []
    copy = pc.AsyncCopyDescriptor(_PublicDescriptor(calls),
                                  priority=1,
                                  strict_ordering=True)
    copy.start()
    copy.start(priority=3, strict_ordering=False)
    copy.wait(strict_ordering=True)
    assert calls == [("start", 1, False), ("start", 3, False), ("wait", )]
    with pytest.raises(NotImplementedError):
        copy.wait(wait_threshold=64)
