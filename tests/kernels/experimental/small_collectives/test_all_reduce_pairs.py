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
"""The chip-pair two-shot (TPU_SMALL_ALL_REDUCE_CHIP_PAIRS) on 16 CPU devices
in the Pallas TPU interpreter with the race detector, at GLM-5.2's hidden size
and the decode buckets' rows.

Its sum is the documented one bit for bit: each chip's two cores (mesh indices
2i, 2i + 1) added in float32 and rounded to bf16, the chip sums added in
float32 in chip order and rounded once; every core gets the same bits, and on
the self-test blocks the exact sum. The flat two-shot is checked against its
own order (float32 over the cores in core order) the same way. Positive
controls: a forward that sends the wrong chunk's bytes fails the oracle, a
forward issued before its chunk lands is a race, and the race detector flags a
read before a DMA wait.
"""

import contextlib
import inspect
import io
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce as ar

pytestmark = pytest.mark.cpu_test

N = 16
AXIS = "d"
HIDDEN = 6144
ROWS = (16, 32, 64)


def reference_pairs(blocks):
    """The pairs form's sum of bf16 blocks [n, rows, hidden]."""
    f32 = np.asarray(blocks, np.float32)
    chips = (f32[0::2] + f32[1::2]).astype(jnp.bfloat16).astype(np.float32)
    acc = chips[0]
    for chip_sum in chips[1:]:
        acc = acc + chip_sum
    return acc.astype(jnp.bfloat16)


def reference_flat(blocks):
    """The two-shot's sum: float32 over the cores in core order, rounded once."""
    f32 = np.asarray(blocks, np.float32)
    acc = f32[0]
    for block in f32[1:]:
        acc = acc + block
    return acc.astype(jnp.bfloat16)


def test_reference_pairs_rounds_each_chip_sum():
    """The two orders differ where a chip sum is not exact in bf16."""
    small = 3 * 2.0**-10
    blocks = np.array([1.0, small, -1.0, small], np.float32).reshape(4, 1, 1)
    blocks = blocks.astype(jnp.bfloat16)
    assert float(reference_flat(blocks)[0, 0]) == 2 * small
    assert float(reference_pairs(blocks)[0, 0]) == 2.0**-8


def test_form_for_takes_pairs_from_pair_rows():
    assert ar.form_for(64, HIDDEN, 2, pair_rows=64) == "pairs"
    assert ar.form_for(32, HIDDEN, 2, pair_rows=64) == "twoshot"
    assert ar.form_for(64, HIDDEN, 2) == "twoshot"
    assert ar.form_for(2, HIDDEN, 2, pair_rows=0) == "oneshot"


def test_pairs_needs_an_even_core_count():
    with pytest.raises(ValueError, match="even"):
        ar.small_all_reduce(
            jnp.zeros((64, 3 * 128 * 3), jnp.bfloat16),
            num_cores=3,
            axis=AXIS,
            form="pairs",
        )


def test_selftest_blocks_have_exact_sums_that_expose_a_wrong_chunk():
    blocks = ar.selftest_blocks(N, 64, HIDDEN)
    as_bf16 = blocks.astype(jnp.bfloat16)
    assert np.array_equal(as_bf16.astype(np.float32), blocks)
    assert np.array_equal(reference_pairs(as_bf16).astype(np.float32), blocks.sum(0))
    assert ar.selftest_ok(blocks, blocks.sum(0))
    own = HIDDEN // N
    moved = blocks.sum(0)
    moved[:, :own] = moved[:, own : 2 * own]
    assert not ar.selftest_ok(blocks, moved)


def _run_worker():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N}"
        " --xla_allow_excess_precision=false",
    )
    proc = subprocess.run(
        [sys.executable, __file__, "worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def on_16_devices():
    return _run_worker()


def test_race_detector_is_live(on_16_devices):
    assert on_16_devices["race_control"] == {"racy": True, "clean": False}


@pytest.mark.parametrize("rows", ROWS)
def test_pairs_is_its_reference_bit_for_bit(on_16_devices, rows):
    case = on_16_devices["cases"][str(rows)]
    for mode in ("on_wait", "eager"):
        got = case[mode]
        assert got["pairs_equal_reference"] and got["ranks_identical"], got
        assert not got["races_found"], got
    assert case["flat_equal_reference"], case
    assert case["selftest_exact"], case


def test_pairs_without_rank_operand(on_16_devices):
    got = on_16_devices["axis_index"]
    assert got["pairs_equal_reference"] and not got["races_found"], got


def test_a_wrong_forward_fails_the_oracle(on_16_devices):
    got = on_16_devices["wrong_forward"]
    assert not got["selftest_exact"] and not got["pairs_equal_reference"], got


def test_an_early_forward_is_a_race(on_16_devices):
    assert on_16_devices["early_forward"]["races_found"], on_16_devices


def _interpret(fn, *args, dma_execution_mode="on_wait"):
    """fn(*args) in the interpreter with the race detector and fresh state;
    (out, whether races were found)."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    params = pltpu.InterpretParams(
        detect_races=True, dma_execution_mode=dma_execution_mode
    )
    pltpu.reset_tpu_interpret_mode_state()
    with (
        pltpu.force_tpu_interpret_mode(params),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        out = np.asarray(jax.block_until_ready(fn(*args)))
    return out, bool(interpret_pallas_call.races.races_found)


def _race_control():
    """The detector on a kernel that reads a DMA target before its wait."""
    from jax.experimental import pallas as pl

    def make(read_early):
        def body(x_hbm, o_ref, buf, sem):
            copy = pltpu.make_async_copy(x_hbm, buf, sem)
            copy.start()
            if read_early:
                o_ref[...] = buf[...]
            copy.wait()
            if not read_early:
                o_ref[...] = buf[...]

        return pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((8, 128), jnp.float32),
                pltpu.SemaphoreType.DMA(()),
            ],
        )

    x = jnp.ones((8, 128), jnp.float32)
    return {
        "racy": _interpret(lambda x: make(True)(x), x)[1],
        "clean": _interpret(lambda x: make(False)(x), x)[1],
    }


FORWARD = """            landed(c)
            send(c, mate)
"""


def _mutated_pairs(replacement):
    """ar._twoshot_pairs with its forward step replaced (positive controls)."""
    source = inspect.getsource(ar._twoshot_pairs)
    assert FORWARD in source
    namespace = {}
    exec(source.replace(FORWARD, replacement), vars(ar), namespace)
    return namespace["_twoshot_pairs"]


def _worker():
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devices) == N, devices
    mesh = Mesh(np.asarray(devices), (AXIS,))
    spec = NamedSharding(mesh, P(AXIS))

    def runner(body, rank_operand=True):
        sharded = jax.jit(
            jax.shard_map(
                lambda x, r: body(x, r if rank_operand else None),
                mesh=mesh,
                in_specs=(P(AXIS), P(AXIS)),
                out_specs=P(AXIS),
                check_vma=False,
            )
        )

        def run(blocks, **kwargs):
            n, rows, hidden = blocks.shape
            x = jax.device_put(jnp.asarray(blocks).reshape(n * rows, hidden), spec)
            rank = jax.device_put(jnp.arange(n, dtype=jnp.int32), spec)
            out, races = _interpret(sharded, x, rank, **kwargs)
            return out.reshape(n, rows, hidden), races

        return run

    def form(name):
        return lambda x, r: ar.small_all_reduce(
            x, num_cores=N, axis=AXIS, rank=r, form=name
        )

    def checked(out, races, want):
        bits = out.view(np.uint16)
        return dict(
            pairs_equal_reference=bool((bits == want.view(np.uint16)).all()),
            ranks_identical=bool((bits == bits[0]).all()),
            races_found=races,
        )

    def rms(out, blocks):
        exact = np.asarray(blocks, np.float64).sum(0)
        return float(np.sqrt(np.mean((out[0].astype(np.float64) - exact) ** 2)))

    pairs, flat = runner(form("pairs")), runner(form("twoshot"))
    rng = np.random.default_rng(0)
    cases = {}
    for rows in ROWS:
        blocks = rng.standard_normal((N, rows, HIDDEN), np.float32).astype(jnp.bfloat16)
        want = reference_pairs(blocks)
        case = {}
        for mode in ("on_wait", "eager"):
            out, races = pairs(blocks, dma_execution_mode=mode)
            case[mode] = checked(out, races, want)
        flat_out, flat_races = flat(blocks)
        case["flat_equal_reference"] = bool(
            np.array_equal(
                flat_out.view(np.uint16),
                np.broadcast_to(reference_flat(blocks), flat_out.shape).view(np.uint16),
            )
            and not flat_races
        )
        case["vs_flat"] = dict(
            differing=int((out.view(np.uint16) != flat_out.view(np.uint16)).sum()),
            elements=int(out.size),
            max_abs=float(np.abs(out.astype(np.float32) - flat_out).max()),
        )
        case["rms_vs_exact"] = dict(pairs=rms(out, blocks), flat=rms(flat_out, blocks))
        exact_blocks = ar.selftest_blocks(N, rows, HIDDEN)
        out, races = pairs(exact_blocks.astype(jnp.bfloat16))
        case["selftest_exact"] = not races and all(
            ar.selftest_ok(exact_blocks, out[r]) for r in range(N)
        )
        cases[str(rows)] = case
        print(rows, json.dumps(case), file=sys.stderr, flush=True)

    blocks = rng.standard_normal((N, 16, HIDDEN), np.float32).astype(jnp.bfloat16)
    out, races = runner(form("pairs"), rank_operand=False)(blocks)
    axis_index = checked(out, races, reference_pairs(blocks))

    exact_blocks = ar.selftest_blocks(N, 16, HIDDEN)
    wrong = _mutated_pairs(
        """            landed(c)
            push(gathered.at[mine], gathered.at[c], send2, recv2.at[c], mate,
                 axis, dict_ids)
"""
    )
    out, races = runner(lambda x, r: wrong(x, N, AXIS, r, ar.COLLECTIVE_ID))(
        exact_blocks.astype(jnp.bfloat16)
    )
    wrong_forward = dict(
        selftest_exact=all(ar.selftest_ok(exact_blocks, out[r]) for r in range(N)),
        pairs_equal_reference=checked(
            out, races, reference_pairs(exact_blocks.astype(jnp.bfloat16))
        )["pairs_equal_reference"],
        races_found=races,
    )
    early = _mutated_pairs(
        """            send(c, mate)
            landed(c)
"""
    )
    out, races = runner(lambda x, r: early(x, N, AXIS, r, ar.COLLECTIVE_ID))(
        exact_blocks.astype(jnp.bfloat16)
    )
    early_forward = dict(
        races_found=races,
        selftest_exact=all(ar.selftest_ok(exact_blocks, out[r]) for r in range(N)),
    )
    result = dict(
        race_control=_race_control(),
        cases=cases,
        axis_index=axis_index,
        wrong_forward=wrong_forward,
        early_forward=early_forward,
    )
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__" and sys.argv[1:] == ["worker"]:
    _worker()
