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
"""kernels/collectives/qar_reduce_scatter.py on 8 CPU devices: the XLA
reference against an f64 sum and XLA's bf16 reduce-scatter (SNR, chunk
placement), the fused Pallas call in interpret mode bit for bit against the
reference, the op's size routing, and a deviceless tpu7x 2x2x1 compile."""

import dataclasses
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.collectives import qar_all_reduce as qar  # noqa: E402
from vllm_torchtpu.kernels.collectives import qar_reduce_scatter as qrs  # noqa: E402


@dataclasses.dataclass(frozen=True)
class _Dev:
    id: int
    coords: tuple
    core_on_chip: int


def _devices(order=tuple(range(8))):
    """tpu7x id layout (id = 2 * chip + core, chips x-major), in mesh order."""
    return [_Dev(i, ((i // 2) % 2, (i // 2) // 2, 0), i % 2) for i in order]


def _mesh():
    if len(jax.devices()) < 8:
        pytest.skip("needs 8 devices (XLA_FLAGS host device count)")
    return Mesh(np.array(jax.devices()[:8]), ("x",))


def _run(body, x):
    mesh = _mesh()
    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("x"), P("x")),
                              out_specs=P("x"), check_vma=False))
    rank = jnp.arange(8, dtype=jnp.int32).reshape(8, 1)
    return np.asarray(f(x, rank).astype(jnp.float32))


def _inputs(rows, h, seed=0):
    """Per-rank partials [8 rows, h] (rank-major), heavy-tailed row scales."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((8, rows, h)) * np.exp(0.5 * rng.standard_normal((8, rows, 1)))
    xb = jnp.asarray(x.reshape(8 * rows, h), jnp.bfloat16)
    exact = np.asarray(xb.astype(jnp.float32)).reshape(8, rows, h).astype(np.float64).sum(0)
    return xb, exact


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _ref_body(tab):
    return lambda x, r: qrs.reduce_scatter_ref(x, r[0], tab, qar.ppermute_exchange("x", tab))


def _interp():
    # Mosaic tiling queries the TPU generation even in interpret mode.
    from jax._src import tpu_info

    return tpu_info


@pytest.fixture
def tpu7x(monkeypatch):
    info = _interp()
    monkeypatch.setattr(info, "get_device_kind", lambda: "TPU7x")
    monkeypatch.setattr(info, "get_num_device_cores", lambda: 1)


def test_layout_table_owner_row():
    tab = qrs.layout_table(_devices())
    for i in range(8):
        assert tab[qrs.OWNER, tab[qrs.POS, i] * 2 + tab[qrs.CORE, i]] == i
    assert sorted(tab[qrs.OWNER]) == list(range(8))
    np.testing.assert_array_equal(tab[:5], qar.layout_table(_devices()))


@pytest.mark.parametrize("rows,h", [(4096, 512), (3072, 512), (1024, 1536)])
def test_reference_snr_and_placement(rows, h):
    """Rank r gets the sum of chunk r; int8 hops cost ~0.7% rel L2 (bf16 RS
    ~0.17%); every chunk is in place (per-chunk error stays small)."""
    tab = qrs.layout_table(_devices())
    x, exact = _inputs(rows, h)
    got = _run(_ref_body(tab), x)
    bf16 = _run(lambda x, r: lax.psum_scatter(x, "x", scatter_dimension=0, tiled=True), x)
    rel_q, rel_b = _rel(got, exact), _rel(bf16, exact)
    assert got.shape == (rows, h)
    assert rel_q < 1.2e-2, rel_q
    assert rel_b < rel_q < 8 * rel_b, (rel_q, rel_b)
    c = rows // 8
    for r in range(8):
        assert _rel(got[r * c:(r + 1) * c], exact[r * c:(r + 1) * c]) < 1.5e-2, r
    # SNR in dB, for the record (>= ~38 dB).
    assert -20 * np.log10(rel_q) > 38


def test_reference_any_mesh_order():
    """The table describes the mesh order: a shuffled device order still
    gives mesh index r the rows of chunk r."""
    order = (5, 2, 7, 0, 3, 6, 1, 4)
    tab = qrs.layout_table(_devices(order))
    x, exact = _inputs(2048, 512, seed=3)
    got = _run(_ref_body(tab), x)
    assert _rel(got, exact) < 1.2e-2


@pytest.mark.parametrize(
    "rows,h",
    [
        (4096, 512),   # chunk 512: two interleaved halves
        (2048, 512),   # chunk 256: one half
        (3072, 512),   # chunk 384: padded to 512
        (1024, 1536),  # chunk 128 padded to 256; 768-lane column tiles
        (2048, 6144),  # MiniMax hidden: 4 column tiles per direction
    ],
)
def test_fused_interpret_matches_reference_bitwise(tpu7x, rows, h):
    tab = qrs.layout_table(_devices())
    x, exact = _inputs(rows, h, seed=rows + h)
    ref = _run(_ref_body(tab), x)
    got = _run(lambda x, r: qrs.reduce_scatter(x, r[0], tab, interpret=pltpu.InterpretParams()), x)
    np.testing.assert_array_equal(got, ref)
    assert _rel(got, exact) < 1.2e-2


def test_chunk_geometry():
    assert qrs.chunk_rows(8192) == 1024 and qrs.chunk_rows(3072) == 512
    assert qrs.chunk_rows(1024) == 256 and qrs.chunk_rows(2048) == 256
    assert qrs.col_block(3072) == 768 and qrs.col_block(256) == 256
    with pytest.raises(ValueError):
        qrs.col_block(200)


@pytest.mark.parametrize(
    "rows,min_rows,want",
    [(8192, 4096, "qrs"), (4096, 4096, "qrs"), (3072, 4096, "psum"), (8192, 0, "psum"),
     (1024, 1024, "qrs"), (1028, 1024, "psum")],
)
def test_op_size_routing(monkeypatch, rows, min_rows, want):
    """The op's body: qar_reduce_scatter at >= the minimum (rows a multiple of
    8), else psum_scatter."""
    from vllm_torchtpu.distributed import pallas_all_reduce as par

    calls = []
    monkeypatch.setattr(par.qrs, "reduce_scatter", lambda x, r, t: calls.append("qrs") or x[:8])
    monkeypatch.setattr(par.lax, "psum_scatter", lambda x, *a, **k: calls.append("psum") or x[:8])
    body = par._rs_body("x", min_rows, object())
    body(jnp.zeros((rows, 6144), jnp.bfloat16), jnp.zeros((1,), jnp.int32))
    assert calls == [want]


@pytest.mark.parametrize("rows", [8192, 3072])
def test_compiles_for_tpu7x(rows):
    """Mosaic lowers the fused call for 8 TensorCores (deviceless tpu7x
    2x2x1, the serving shard_map, hidden 6144)."""
    from jax.experimental import topologies
    from jax.sharding import NamedSharding

    try:
        topo = topologies.get_topology_desc("tpu7x:2x2x1", "tpu")
    except Exception as e:  # no libtpu
        pytest.skip(f"no deviceless tpu7x topology: {e}")
    devs = topo.devices
    tab = qrs.layout_table(devs)
    mesh = Mesh(np.array(devs), ("x",))
    f = jax.jit(jax.shard_map(lambda x, r: qrs.reduce_scatter(x, r[0], tab), mesh=mesh,
                              in_specs=(P("x"), P("x")), out_specs=P("x"), check_vma=False))
    sh = NamedSharding(mesh, P("x"))
    x = jax.ShapeDtypeStruct((8 * rows, 6144), jnp.bfloat16, sharding=sh)
    r = jax.ShapeDtypeStruct((8, 1), jnp.int32, sharding=sh)
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        f.lower(x, r).compile()


def _rs_op(hier=False):
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.distributed import pallas_all_reduce as par

    return par._make_rs_op(_mesh(), 4096, hier=hier)


@pytest.mark.parametrize("rows", [8192, 3072])
def test_hier_compiles_for_tpu7x(rows):
    """hierrs_tc (TPU_MINIMAX_M3_SP_HIER_RS) on the served rank-ordered mesh
    (chip_perm) compiles deviceless for tpu7x 2x2x1 at hidden 6144."""
    from jax.experimental import topologies
    from jax.sharding import NamedSharding

    from vllm_torchtpu.distributed import pallas_all_reduce as par

    try:
        topo = topologies.get_topology_desc("tpu7x:2x2x1", "tpu")
    except Exception as e:  # no libtpu
        pytest.skip(f"no deviceless tpu7x topology: {e}")
    byid = {d.id: d for d in topo.devices}
    devs = [byid[i] for i in (0, 1, 4, 5, 6, 7, 2, 3)]
    mesh = Mesh(np.array(devs), ("x",))
    cores = [par._ChipCore(int(d.id), tuple(int(c) for c in d.coords[:3]), int(d.core_on_chip))
             for d in devs]
    body = par._hier_rs_body("x", 1024, par._chip_perm(cores))
    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("x"), P("x")), out_specs=P("x"),
                              check_vma=False))
    sh = NamedSharding(mesh, P("x"))
    x = jax.ShapeDtypeStruct((8 * rows, 6144), jnp.bfloat16, sharding=sh)
    r = jax.ShapeDtypeStruct((8, 1), jnp.int32, sharding=sh)
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        f.lower(x, r).compile()


@pytest.mark.parametrize("hier", [False, True])
def test_rs_op_fake_symbolic(hier):
    """The torch op's fake: [T, H] -> [T / 8, H] with a symbolic T under
    FakeTensorMode (nothing lowers)."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        ShapeEnv,
        StatelessSymbolicContext,
    )

    op = _rs_op(hier)
    mode = FakeTensorMode(shape_env=ShapeEnv())
    seed = mode.from_tensor(
        torch.empty(64),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    )
    with mode:
        t = seed.shape[0]
        out = op(torch.empty((8 * t, 512), dtype=torch.bfloat16),
                 torch.empty((1, 1), dtype=torch.int32))
    assert tuple(out.shape) == (t, 512) and out.dtype == torch.bfloat16
    assert isinstance(out.shape[0], torch.SymInt)


@pytest.mark.parametrize("hier", [False, True])
def test_rs_op_dynamo_meta(hier):
    """torch.compile(fullgraph=True) over meta tensors with a dynamic token
    dim, as Dynamo traces the SP step in warmup."""
    import torch

    op = _rs_op(hier)
    x = torch.empty((8 * 512, 512), dtype=torch.bfloat16, device="meta")
    r = torch.empty((1, 1), dtype=torch.int32, device="meta")
    torch._dynamo.mark_dynamic(x, 0)
    torch._dynamo.reset()
    out = torch.compile(lambda x, r: op(x, r) + 1, backend="eager", fullgraph=True)(x, r)
    assert tuple(out.shape) == (512, 512) and out.dtype == torch.bfloat16
