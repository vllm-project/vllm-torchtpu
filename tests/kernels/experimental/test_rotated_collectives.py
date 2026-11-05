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
"""The rotated halving-doubling collectives (rotated_collectives,
fused_moe_prefill.row_all_gather and fused_moe_prefill.combine_scatter) on 16
CPU devices in the Pallas TPU interpreter with the race detector, over a mesh
that lists the cores of a 2x2x2 slice of 2-core chips out of order.

Every kernel is checked bit for bit: the gathers against
``lax.all_gather(tiled=True)``; the reduce-scatter and the fused combine against
the exact sums of integer-valued blocks, which bf16 holds at every partial sum,
so no order of the adds can change a bit; the fused projection against the
projection followed by ``reduce_scatter``. combine_scatter with two routes per
token runs without the race detector, which takes minutes on its 1024 row DMAs
per core; with one route it runs with it. Positive control: the peer table's
rows rotated by one core fail the reduce-scatter. The peer tables themselves are
checked on the host.
"""

import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental import rotated_collectives as rc
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    combine_scatter,
    row_all_gather,
)

pytestmark = pytest.mark.cpu_test

N = rc.CORES
AXIS = "d"
PARTS = len(rc.ROTATIONS)
F8 = jnp.float8_e4m3fn


def _places(cores):
    """(chip coords, core on chip) per mesh index: mesh index m is core
    cores[m] of the slice, core c on chip (c >> 1 & 1, c >> 2 & 1, c >> 3 & 1)."""
    return [(((c >> 1) & 1, (c >> 2) & 1, (c >> 3) & 1), c & 1) for c in cores]


SHUFFLED = _places(np.random.default_rng(0).permutation(N))


def _rows(kind):
    return rc.peer_table(SHUFFLED, kind).reshape(N, rc.TABLE_WIDTH)


@pytest.mark.parametrize(
    "places",
    [
        SHUFFLED[:-1],
        SHUFFLED[:-1] + SHUFFLED[:1],
        [((c % 4, c // 4 % 2, 0), c // 8) for c in range(N)],
    ],
    ids=["15_cores", "core_twice", "4x2x1_chips"],
)
def test_layout_rejects_cores_that_are_not_a_2x2x2_slice(places):
    """A mesh the schedule cannot run over must not get a peer table."""
    with pytest.raises(ValueError, match="not a 2x2x2 slice"):
        rc.Layout(places)


def test_peers_pair_up_across_one_axis():
    """A peer on the wrong axis or core, or a one-way pairing, would deadlock or
    misroute a phase."""
    layout = rc.Layout(SHUFFLED)
    for m, (coords, core) in enumerate(SHUFFLED):
        for axis in (*rc.AXES, "c"):
            p = layout.peer(m, axis)
            assert layout.peer(p, axis) == m
            p_coords, p_core = SHUFFLED[p]
            if axis == "c":
                assert (p_coords, p_core) == (coords, 1 - core)
            else:
                moved = [a != b for a, b in zip(coords, p_coords, strict=True)]
                assert p_core == core and moved == [a == axis for a in rc.AXES]


def test_all_gather_sends_are_the_partners_receives():
    """Each phase sends what the core holds into the slots its partner waits on,
    and a core and its chip-mate end up with all 16 blocks of every part."""
    t = _rows("all_gather")
    held = {}
    for m in range(N):
        for j in range(PARTS):
            blocks = {m}
            for k in range(3):
                partner = t[m, rc._PARTNER + 3 * j + k]
                assert t[partner, rc._PARTNER + 3 * j + k] == m
                n = [j * 7 + 2**k - 1 + i for i in range(2**k)]
                sends = list(t[m, rc._AG_SEND + np.array(n)])
                assert sends == list(t[partner, rc._AG_RECV + np.array(n)])
                assert set(sends) == blocks
                blocks |= set(t[m, rc._AG_RECV + np.array(n)])
            held[m, j] = blocks
    for m in range(N):
        mate = t[m, rc._MATE]
        for j in range(PARTS):
            assert len(held[m, j]) == 8
            assert held[m, j] | held[mate, j] == set(range(N))


def test_reduce_scatter_slots_name_the_same_block_on_both_ends():
    """The mate sends each core the blocks its slots sum, every chip phase adds
    the partner's partial of the same block, and slot 0 ends on the core's own
    row block."""
    t = _rows("reduce_scatter")
    for m in range(N):
        mate = t[m, rc._MATE]
        for j in range(PARTS):
            own = t[m, rc._RS_OWN + 8 * j : rc._RS_OWN + 8 * j + 8]
            to_me = t[mate, rc._RS_TO_MATE + 8 * j : rc._RS_TO_MATE + 8 * j + 8]
            assert list(to_me) == list(own) and own[0] == m
            assert len(set(own)) == 8
            assert {SHUFFLED[b][1] for b in own} == {SHUFFLED[m][1]}
            for k in range(3):
                partner = t[m, rc._PARTNER + 3 * j + k]
                theirs = t[partner, rc._RS_OWN + 8 * j : rc._RS_OWN + 8 * j + 8]
                for i in range(8):
                    if i >> k & 1:
                        assert own[i] == theirs[i - 2**k]


# combine_scatter's routes per token, by worker part and result key.
COMBINE_PARTS = {"combine_scatter_2_routes": 2, "combine_scatter_1_route": 1}
WORKER_PARTS = ("collectives", *COMBINE_PARTS)


def _start_worker(part):
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N}"
        " --xla_allow_excess_precision=false",
    )
    return subprocess.Popen(
        [sys.executable, __file__, part],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _result(proc):
    out, err = proc.communicate(timeout=900)
    lines = [ln for ln in out.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, out[-3000:] + err[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def on_16_devices():
    """The worker parts side by side: combine_scatter, with a row DMA per token
    and route, takes about as long as every other case together."""
    procs = [_start_worker(part) for part in WORKER_PARTS]
    try:
        return {k: v for proc in procs for k, v in _result(proc).items()}
    finally:
        for proc in procs:
            proc.kill()


KERNEL_CASES = [
    "reduce_scatter",
    "reduce_scatter_chunks",
    "all_gather",
    "all_gather_many_chunks",
    "matmul_reduce_scatter",
    "row_all_gather_fp8",
    "row_all_gather_bf16",
    *COMBINE_PARTS,
]


@pytest.mark.parametrize("case", KERNEL_CASES)
def test_kernel_is_its_reference_bit_for_bit(on_16_devices, case):
    """A block sent to the wrong core or slot, added twice or not at all, or
    read before its DMA lands changes bits or trips the race detector."""
    got = on_16_devices[case]
    assert got["equal"] and not got["races"], got


def test_rotated_peer_table_fails_the_reference(on_16_devices):
    """Every core reading the previous core's table row (each semaphore still
    gets its bytes, so nothing hangs) misroutes blocks; the reference sees it."""
    assert not on_16_devices["rotated_rows"]["equal"], on_16_devices


def _same(a, b):
    return a.dtype == b.dtype and np.array_equal(
        a.view(np.uint8), np.asarray(b).view(np.uint8)
    )


def _exact_scatter(x, chunks):
    """Each core's rows of the per-chunk sums over the cores of x [N, rows, h]."""
    total = np.asarray(x, np.float32).sum(0)
    blocks = total.reshape(chunks, N, -1, total.shape[1]).transpose(1, 0, 2, 3)
    return blocks.reshape(N, -1, total.shape[1]).astype(jnp.bfloat16)


def _gathered(x, chunk):
    """`lax.all_gather(tiled=True)` of each `chunk`-row chunk of x, in order."""
    return jnp.concatenate(
        [
            lax.all_gather(x[c : c + chunk], AXIS, axis=0, tiled=True)
            for c in range(0, x.shape[0], chunk)
        ]
    )


def _interpreter():
    """run(body, *blocks): body(rank [1], *this core's blocks) on every core of a
    16-device mesh in the interpreter; its outputs stacked by core, and whether
    the race detector fired."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devices) == N, devices
    mesh = Mesh(np.asarray(devices), (AXIS,))
    sharding = NamedSharding(mesh, P(AXIS))

    def run(body, *blocks, detect_races=True):
        def per_core(rank, *xs):
            out = body(rank, *(x[0] for x in xs))
            return jax.tree.map(lambda y: y[None], out)

        args = [np.arange(N, dtype=np.int32), *blocks]
        args = [jax.device_put(jnp.asarray(a), sharding) for a in args]
        fn = jax.jit(
            jax.shard_map(
                per_core,
                mesh=mesh,
                in_specs=(P(AXIS),) * len(args),
                out_specs=P(AXIS),
                check_vma=False,
            )
        )
        pltpu.reset_tpu_interpret_mode_state()
        params = pltpu.InterpretParams(detect_races=detect_races)
        with pltpu.force_tpu_interpret_mode(params):
            out = jax.tree.map(np.asarray, jax.block_until_ready(fn(*args)))
        return out, detect_races and bool(interpret_pallas_call.races.races_found)

    return run


def _ints(rng, bound, shape, dtype):
    return rng.integers(-bound, bound + 1, shape).astype(dtype)


def _checked(out, races, want):
    return dict(equal=_same(out, want), races=races)


def _collective_cases(run, rng):
    rs_table = jnp.asarray(rc.peer_table(SHUFFLED, "reduce_scatter"))
    ag_table = jnp.asarray(rc.peer_table(SHUFFLED, "all_gather"))

    def gather_case(kernel, *blocks, chunk):
        """kernel(rank, *xs) against _gathered of each x, run in the same body."""

        def body(rank, *xs):
            return kernel(rank, *xs), tuple(_gathered(x, chunk) for x in xs)

        (got, want), races = run(body, *blocks)
        got = got if isinstance(got, tuple) else (got,)
        equal = all(_same(g, w) for g, w in zip(got, want, strict=True))
        return dict(equal=equal, races=races)

    result = {}
    x = _ints(rng, 8, (N, 256, 384), jnp.bfloat16)
    result["reduce_scatter"] = _checked(
        *run(lambda r, x: rc.reduce_scatter(x, r, rs_table, axis=AXIS), x),
        _exact_scatter(x, 1),
    )
    rotated = jnp.roll(rs_table, rc.TABLE_WIDTH)
    out, _ = run(lambda r, x: rc.reduce_scatter(x, r, rotated, axis=AXIS), x)
    result["rotated_rows"] = dict(equal=_same(out, _exact_scatter(x, 1)))

    x = _ints(rng, 8, (N, 512, 1536), jnp.bfloat16)
    result["reduce_scatter_chunks"] = _checked(
        *run(
            lambda r, x: rc.reduce_scatter_chunks(x, r, rs_table, axis=AXIS, chunk=256),
            x,
        ),
        _exact_scatter(x, 2),
    )

    result["all_gather"] = gather_case(
        lambda r, x: rc.all_gather(x, r, ag_table, axis=AXIS),
        rng.standard_normal((N, 16, 384)).astype(jnp.bfloat16),
        chunk=16,
    )
    result["all_gather_many_chunks"] = gather_case(
        lambda r, *xs: rc.all_gather_many_chunks(xs, r, ag_table, axis=AXIS, chunk=32),
        rng.standard_normal((N, 64, 512)).astype(F8),
        rng.standard_normal((N, 64, 384)).astype(jnp.bfloat16),
        chunk=32,
    )
    result["row_all_gather_fp8"] = gather_case(
        lambda r, x: row_all_gather.all_gather(x, r, ag_table, axis=AXIS),
        rng.standard_normal((N, 24, 3, 128)).astype(F8),
        chunk=24,
    )
    result["row_all_gather_bf16"] = gather_case(
        lambda r, x: row_all_gather.all_gather(x, r, ag_table, axis=AXIS),
        rng.standard_normal((N, 64, 256)).astype(jnp.bfloat16),
        chunk=64,
    )

    def projection(r, x_q, x_s, w_q, w_s):
        fused = rc.matmul_reduce_scatter(x_q, x_s, w_q, w_s, r, rs_table, axis=AXIS)
        y = jnp.dot(x_q.astype(jnp.float32), w_q.astype(jnp.float32))
        partial = (y * x_s * w_s).astype(jnp.bfloat16)
        return fused, rc.reduce_scatter(partial, r, rs_table, axis=AXIS)

    w_q = np.broadcast_to(_ints(rng, 3, (256, 384), F8), (N, 256, 384))
    w_s = np.broadcast_to(2.0 ** rng.integers(-4, 1, 384), (N, 384))
    (fused, separate), races = run(
        projection,
        _ints(rng, 3, (N, 256, 256), F8),
        (2.0 ** rng.integers(-4, 1, (N, 256, 1))).astype(np.float32),
        w_q,
        w_s.astype(np.float32),
    )
    result["matmul_reduce_scatter"] = _checked(fused, races, separate)
    return result


def _combine_case(run, rng, topk):
    """combine_scatter of the smallest chunk it takes (512 tokens, 384 lanes),
    the race detector on for a single route only (see the module docstring)."""
    tokens, hidden, slab_rows = 512, 384, 64
    table = jnp.asarray(combine_scatter.scatter_table(SHUFFLED))
    slab = _ints(rng, 2, (N, slab_rows, hidden // rc.LANES, rc.LANES), jnp.bfloat16)
    coeff = _ints(rng, 2, (N, tokens, topk), np.float32)
    slab_row = rng.integers(0, slab_rows, (N, tokens, topk)).astype(np.int32)
    picked = np.take_along_axis(
        np.asarray(slab, np.float32).reshape(N, slab_rows, 1, hidden),
        slab_row[..., None],
        axis=1,
    )
    combined = (picked * coeff[..., None]).sum(2)
    out, races = run(
        lambda r, s, c, i: combine_scatter.combine_scatter(
            s, c, i, r, table, axis=AXIS, interpret=True
        ),
        slab,
        coeff,
        slab_row,
        detect_races=topk == 1,
    )
    return _checked(out, races, _exact_scatter(combined, 1))


def _worker(part):
    rng = np.random.default_rng(WORKER_PARTS.index(part))
    run = _interpreter()
    if part in COMBINE_PARTS:
        result = {part: _combine_case(run, rng, COMBINE_PARTS[part])}
    else:
        result = _collective_cases(run, rng)
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    _worker(sys.argv[1])
