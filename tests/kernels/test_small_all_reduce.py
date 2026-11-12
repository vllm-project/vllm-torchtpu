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
small_all_reduce_op.py) over an 8-core (or 4-core) mesh, so the shard_map
wiring and the rank tables are covered along with the kernels. On CPU the cases
run in a subprocess with 8 host devices and Pallas TPU interpret mode (with
race detection), which models the remote DMAs and semaphores; on a multi-chip
TPU host they run on the 8 cores directly.
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

N = 8
HIDDEN = N * 128
# A non-identity rank at each mesh index, as when the mesh is in device-id
# order and the ranks are not.
RANK_OF_CORE = (0, 1, 6, 7, 2, 3, 4, 5)
# A permutation that also swaps ranks across physical chips.
CROSS_CHIP_RANK_OF_CORE = (0, 3, 6, 1, 2, 7, 4, 5)
FOUR_CORE_PERMUTED = (2, 0, 3, 1)


def _cases():
    cases = [
        ("all_reduce", form, rows, None, N)
        for form in ("oneshot", "twoshot", None)
        for rows in (4, 16)
    ]
    cases += [
        ("reduce_scatter", None, rows, order, N)
        for rows in (8, 5)  # 5 pads each rank's block to a row tile
        for order in (tuple(range(N)), RANK_OF_CORE)
    ]
    cases += [
        ("reduce_scatter_hier", mode, rows, order, N)
        for mode in ("permuted", "direct")
        for rows in (8, 5)
        for order in (tuple(range(N)), RANK_OF_CORE, CROSS_CHIP_RANK_OF_CORE)
        if mode == "permuted" or order == tuple(range(N))
    ]
    cases += [
        ("reduce_scatter_hier", mode, 8, order, 4)
        for mode, order in (
            ("direct", tuple(range(4))),
            ("permuted", FOUR_CORE_PERMUTED),
        )
    ]
    return cases


def _run_case(kind, form, rows, rank_of_core, num_cores, interpret):
    """Max |kernel - reference| in bf16 steps, and the same against XLA."""
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.collectives import small_gather_scatter as sgs

    op = pytest.importorskip("vllm_torchtpu.layers.adapter.small_all_reduce_op")
    captured = {}

    def capture(name, fn, **kwargs):
        captured["fn"] = fn
        return type("FakeOp", (), {"register_fake": lambda self, f: None})()

    orig_jax_op = op.pallas.jax_op
    orig_mates_check = op.mesh_has_adjacent_chip_mates
    op.pallas.jax_op = capture
    mesh = Mesh(np.array(jax.devices()[:num_cores]), ("d",))
    rng = np.random.default_rng(rows + 17 * num_cores)
    spec = NamedSharding(mesh, P("d"))
    put = lambda a: jax.device_put(jnp.asarray(a), spec)  # noqa: E731
    mesh_index = put(np.arange(num_cores, dtype=np.int32))

    try:
        if kind == "all_reduce":
            op._build_all_reduce(mesh, form)
            blocks = rng.standard_normal((num_cores, rows, HIDDEN)).astype(jnp.bfloat16)
            args = (put(blocks.reshape(num_cores * rows, HIDDEN)), mesh_index)
            want = np.sum(blocks.astype(np.float32), axis=0)
            want = np.broadcast_to(want, (num_cores, rows, HIDDEN))
            xla = jax.jit(
                jax.shard_map(
                    lambda x: jax.lax.psum(x, "d"),
                    mesh=mesh,
                    in_specs=P("d"),
                    out_specs=P("d"),
                    check_vma=False,
                )
            )(args[0])
        elif kind == "reduce_scatter_hier" and form == "direct":
            blocks = rng.standard_normal((num_cores, num_cores, rows, HIDDEN)).astype(
                jnp.bfloat16
            )
            args = (
                put(blocks.reshape(num_cores * num_cores * rows, HIDDEN)),
                mesh_index,
            )
            want = np.sum(blocks.astype(np.float32), axis=0)
            captured["fn"] = lambda x, mi: jax.shard_map(
                lambda b, r: sgs.small_reduce_scatter(
                    b, num_cores=num_cores, axis="d", rank=r, hierarchical=True
                ),
                mesh=mesh,
                in_specs=(P("d"), P("d")),
                out_specs=P("d"),
                check_vma=False,
            )(x, mi)
            xla = jax.jit(
                jax.shard_map(
                    lambda x: jax.lax.psum_scatter(
                        x, "d", scatter_dimension=0, tiled=True
                    ),
                    mesh=mesh,
                    in_specs=P("d"),
                    out_specs=P("d"),
                    check_vma=False,
                )
            )(args[0])
        else:
            is_hier = kind == "reduce_scatter_hier"
            if is_hier and interpret:
                # Host CPU devices lack dual-core chip coords; force the
                # two-stage kernel on so interpret mode exercises it.
                op.mesh_has_adjacent_chip_mates = lambda _devs: True
            op._build_reduce_scatter(mesh, hierarchical=is_hier)
            core_of_rank = np.argsort(rank_of_core).astype(np.int32)
            # blocks[i, r]: the partial sums mesh core i holds for rank r's rows
            blocks = rng.standard_normal((num_cores, num_cores, rows, HIDDEN)).astype(
                jnp.bfloat16
            )
            args = (
                put(blocks.reshape(num_cores * num_cores * rows, HIDDEN)),
                mesh_index,
                put(np.asarray(rank_of_core, np.int32)),
                put(np.tile(core_of_rank, num_cores)),
            )
            sums = np.sum(blocks.astype(np.float32), axis=0)
            want = np.stack([sums[rank_of_core[i]] for i in range(num_cores)])

            def xla_body(x, my_rank):
                # vLLM's path: the full sum, then this rank's block
                total = jax.lax.psum(x, "d").reshape(num_cores, rows, HIDDEN)
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
        op.mesh_has_adjacent_chip_mates = orig_mates_check

    run = jax.jit(captured["fn"])
    if kind.startswith("reduce_scatter") and not interpret:
        hlo_text = run.lower(*args).as_text()
        if kind == "reduce_scatter_hier":
            assert "small_reduce_scatter_hier_" in hlo_text
        else:
            assert "small_reduce_scatter_hier_" not in hlo_text
    if interpret:
        pltpu.reset_tpu_interpret_mode_state()
        with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
            got = run(*args)
    else:
        got = run(*args)
    got = np.asarray(got).astype(np.float32).reshape(num_cores, rows, HIDDEN)
    want = want.astype(jnp.bfloat16).astype(np.float32)
    xla = np.asarray(xla).astype(np.float32).reshape(num_cores, rows, HIDDEN)
    step = float(np.spacing(np.float32(np.abs(want).max()))) * 2**16
    return {
        "vs_reference_steps": float(np.abs(got - want).max()) / step,
        "vs_xla_max": float(np.abs(got - xla).max()),
        "hierarchical": kind == "reduce_scatter_hier",
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
    # One-stage rounds once; two-stage rounds once after the intra-chip pair
    # sum and once at the final output, so allow up to 2 bf16 steps.
    max_steps = 2.0 if result.get("hierarchical") else 1.0
    assert result["vs_reference_steps"] <= max_steps, result
    # XLA's bf16 collective rounds at every hop
    assert result["vs_xla_max"] <= 0.125, result


@pytest.mark.parametrize("case", _cases(), ids=repr)
def test_matches_xla_collective_in_interpret_mode(cpu_results, case):
    _check(cpu_results[repr(case)])


@pytest.mark.multichip
@pytest.mark.parametrize("case", _cases(), ids=repr)
def test_matches_xla_collective_on_tpu(case):
    if jax.default_backend() != "tpu" or len(jax.devices()) < N:
        pytest.skip("needs at least 8 TPU cores across 4 dual-core chips")
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


def test_mesh_has_adjacent_chip_mates_checks_dual_core_pairing():
    from vllm_torchtpu.distributed.ep_mesh import mesh_has_adjacent_chip_mates

    # Live PJRT devices on a 4-chip and 2-chip dual-core slice.
    coords_paired = [
        SimpleNamespace(id=2 * c + s, coords=(c % 2, c // 2, 0), slice_index=0)
        for c in range(4)
        for s in range(2)
    ]
    assert mesh_has_adjacent_chip_mates(coords_paired)
    assert mesh_has_adjacent_chip_mates(coords_paired[:4])

    # Compile-only Pallas devices across PJRT device_kind names and PCI rank order.
    for kind in ("TPU7x", "TPU7", "TPU8i"):
        pallas_sorted = [SimpleNamespace(id=i, device_kind=kind) for i in range(8)]
        pallas_pci_order = [
            SimpleNamespace(id=i, device_kind=kind) for i in (0, 1, 4, 5, 6, 7, 2, 3)
        ]
        assert mesh_has_adjacent_chip_mates(pallas_sorted), kind
        assert mesh_has_adjacent_chip_mates(pallas_pci_order), kind

    # Non-adjacent mates, single-core-per-chip (v6e, 8t), Megacore (v5p),
    # cross-slice coords collision, missing id, CPU, 6 cores, or 2 cores -> False.
    strided = [
        SimpleNamespace(id=i, device_kind="TPU7x") for i in (0, 2, 1, 3, 4, 6, 5, 7)
    ]
    v6e_single_core = [
        SimpleNamespace(id=i, coords=(i % 4, i // 4, 0), device_kind="TPU v6e")
        for i in range(8)
    ]
    v6e_pallas = [SimpleNamespace(id=i, device_kind="TPU v6e") for i in range(8)]
    tpu8t_pallas = [SimpleNamespace(id=i, device_kind="TPU8t") for i in range(8)]
    v5p_megacore = [SimpleNamespace(id=i, device_kind="TPU v5p") for i in range(8)]
    cross_slice_coords = [
        SimpleNamespace(id=2 * c + s, coords=(c % 2, c // 2, 0), slice_index=s)
        for c in range(4)
        for s in range(2)
    ]
    missing_id_coords = [
        SimpleNamespace(
            id=None if (c == 0 and s == 1) else 2 * c + s,
            coords=(c % 2, c // 2, 0),
        )
        for c in range(4)
        for s in range(2)
    ]
    cpu_devs = [SimpleNamespace(id=i, device_kind="cpu") for i in range(8)]
    six_cores = coords_paired[:6]
    two_cores = coords_paired[:2]
    for invalid in (
        strided,
        v6e_single_core,
        v6e_pallas,
        tpu8t_pallas,
        v5p_megacore,
        cross_slice_coords,
        missing_id_coords,
        cpu_devs,
        six_cores,
        two_cores,
    ):
        assert not mesh_has_adjacent_chip_mates(invalid)


def test_hierarchical_refuses_invalid_core_counts():
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.collectives import small_gather_scatter as sgs

    for bad_cores in (2, 6):
        with pytest.raises(ValueError, match="power-of-two chip count"):
            sgs.small_reduce_scatter(
                jnp.zeros((bad_cores * 8, HIDDEN), jnp.bfloat16),
                num_cores=bad_cores,
                hierarchical=True,
            )


def test_hierarchical_falls_back_to_one_stage_when_vmem_exceeded(monkeypatch):
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.collectives import small_gather_scatter as sgs

    # 8 cores * 8 rows * 1024 * 2 B * 2.5 = 327,680 B; 256 KiB VMEM forces fallback.
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(vmem_capacity_bytes=256 * 1024),
    )
    recorded_names = []

    def fake_pallas_call(kernel, *, out_shape, name, **kwargs):
        recorded_names.append(name)
        return lambda *args: jnp.zeros(out_shape.shape, out_shape.dtype)

    monkeypatch.setattr(sgs.pl, "pallas_call", fake_pallas_call)
    out = sgs.small_reduce_scatter(
        jnp.zeros((N * 8, HIDDEN), jnp.bfloat16),
        num_cores=N,
        rank=jnp.zeros((1,), jnp.int32),
        hierarchical=True,
    )
    assert out.shape == (8, HIDDEN)
    assert recorded_names == ["small_reduce_scatter_r8"]


if __name__ == "__main__":
    print("RESULT " + json.dumps(_run_all(interpret=True)))
