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
"""The small all-reduce and reduce-scatter kernels against XLA's collectives.

Each case runs the torch op's JAX function (layers/adapter/
small_all_reduce_op.py) over an 8-core mesh, so the shard_map wiring and the
rank tables are covered along with the kernels. On CPU the cases run in a
subprocess with 8 host devices and Pallas TPU interpret mode, which models the
remote DMAs and semaphores; on a TPU host they run on the 8 cores directly.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")

N = 8
HIDDEN = N * 128
# A non-identity rank at each mesh index, as when the mesh is in device-id
# order and the ranks are not.
RANK_OF_CORE = (0, 1, 6, 7, 2, 3, 4, 5)


def _cases():
    cases = [
        ("all_reduce", form, rows, None)
        for form in ("oneshot", "twoshot", None)
        for rows in (4, 16)
    ]
    cases += [
        ("reduce_scatter", None, rows, order)
        for rows in (8, 5)  # 5 pads each rank's block to a row tile
        for order in (tuple(range(N)), RANK_OF_CORE)
    ]
    return cases


def _run_case(kind, form, rows, rank_of_core, interpret):
    """Max |kernel - reference| in bf16 steps, and the same against XLA."""
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    op = pytest.importorskip("vllm_torchtpu.layers.adapter.small_all_reduce_op")
    captured = {}

    def capture(name, fn, **kwargs):
        captured["fn"] = fn
        return type("FakeOp", (), {"register_fake": lambda self, f: None})()

    orig_jax_op = op.pallas.jax_op
    op.pallas.jax_op = capture
    mesh = Mesh(np.array(jax.devices()[:N]), ("d",))
    rng = np.random.default_rng(rows)
    spec = NamedSharding(mesh, P("d"))
    put = lambda a: jax.device_put(jnp.asarray(a), spec)  # noqa: E731
    mesh_index = put(np.arange(N, dtype=np.int32))

    try:
        if kind == "all_reduce":
            op._build_all_reduce(mesh, form)
            blocks = rng.standard_normal((N, rows, HIDDEN)).astype(jnp.bfloat16)
            args = (put(blocks.reshape(N * rows, HIDDEN)), mesh_index)
            want = np.sum(blocks.astype(np.float32), axis=0)
            want = np.broadcast_to(want, (N, rows, HIDDEN))
            xla = jax.jit(
                jax.shard_map(
                    lambda x: jax.lax.psum(x, "d"),
                    mesh=mesh,
                    in_specs=P("d"),
                    out_specs=P("d"),
                    check_vma=False,
                )
            )(args[0])
        else:
            op._build_reduce_scatter(mesh)
            core_of_rank = np.argsort(rank_of_core).astype(np.int32)
            # blocks[i, r]: the partial sums mesh core i holds for rank r's rows
            blocks = rng.standard_normal((N, N, rows, HIDDEN)).astype(jnp.bfloat16)
            args = (
                put(blocks.reshape(N * N * rows, HIDDEN)),
                mesh_index,
                put(np.asarray(rank_of_core, np.int32)),
                put(np.tile(core_of_rank, N)),
            )
            sums = np.sum(blocks.astype(np.float32), axis=0)
            want = np.stack([sums[rank_of_core[i]] for i in range(N)])

            def xla_body(x, my_rank):
                # vLLM's path: the full sum, then this rank's block
                total = jax.lax.psum(x, "d").reshape(N, rows, HIDDEN)
                return jax.lax.dynamic_index_in_dim(total, my_rank[0], keepdims=False)

            xla = jax.jit(
                jax.shard_map(
                    xla_body,
                    mesh=mesh,
                    in_specs=(P("d"), P("d")),
                    out_specs=P("d"),
                    check_vma=False,
                )
            )(args[0], args[2])
    finally:
        op.pallas.jax_op = orig_jax_op

    run = jax.jit(captured["fn"])
    if interpret:
        pltpu.reset_tpu_interpret_mode_state()
        with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
            got = run(*args)
    else:
        got = run(*args)
    got = np.asarray(got).astype(np.float32).reshape(N, rows, HIDDEN)
    want = want.astype(jnp.bfloat16).astype(np.float32)
    xla = np.asarray(xla).astype(np.float32).reshape(N, rows, HIDDEN)
    step = float(np.spacing(np.float32(np.abs(want).max()))) * 2**16
    return {
        "vs_reference_steps": float(np.abs(got - want).max()) / step,
        "vs_xla_max": float(np.abs(got - xla).max()),
    }


def _run_all(interpret):
    return {repr(case): _run_case(*case, interpret=interpret) for case in _cases()}


@pytest.fixture(scope="module")
def cpu_results():
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{os.environ.get('XLA_FLAGS', '')} "
        f"--xla_force_host_platform_device_count={N}",
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


def _check(result):
    # the float32 sum rounded once, up to one bf16 step for summation order
    assert result["vs_reference_steps"] <= 1.0, result
    # XLA's bf16 collective rounds at every hop; the kernel rounds once
    assert result["vs_xla_max"] <= 0.125, result


@pytest.mark.parametrize("case", _cases(), ids=repr)
def test_matches_xla_collective_in_interpret_mode(cpu_results, case):
    _check(cpu_results[repr(case)])


@pytest.mark.parametrize("case", _cases(), ids=repr)
def test_matches_xla_collective_on_tpu(case):
    if jax.default_backend() != "tpu" or len(jax.devices()) < N:
        pytest.skip("needs the 8 TPU cores of one host")
    _check(_run_case(*case, interpret=False))


def test_form_by_bytes():
    from vllm_torchtpu.kernels.collectives import small_all_reduce as sar

    assert sar.form_for(4, 4096, 2) == "oneshot"
    assert sar.form_for(8, 4096, 2) == "twoshot"


def test_refuses_a_width_that_does_not_split():
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.collectives import small_all_reduce as sar

    with pytest.raises(ValueError):
        sar.small_all_reduce(jnp.zeros((16, 4000), jnp.bfloat16), num_cores=N)


if __name__ == "__main__":
    print("RESULT " + json.dumps(_run_all(interpret=True)))
