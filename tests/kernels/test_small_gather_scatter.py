"""Small reduce-scatter on eight TPU cores, compared with a float32 sum.

Cover core indices read from the mesh and supplied as bridge operands,
including a non-identity mapping between ranks and mesh cores.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives import small_gather_scatter as sgs

N = 8
AXIS = "d"


@pytest.fixture(autouse=True)
def _needs_the_eight_cores():
    if jax.default_backend() != "tpu" or len(jax.devices()) < N:
        pytest.skip("needs the 8 TPU cores of one host")


def _mesh():
    return Mesh(np.array(jax.devices()[:N]), (AXIS, ))


def _blocks(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=np.float32).astype(jnp.bfloat16)


def _run(mesh, per_core, fn_name, rank_operand):
    """`per_core` is [N, rows_in, hidden]: each core's input block."""
    n, rows_in, hidden = per_core.shape
    x = jax.device_put(
        jnp.asarray(per_core).reshape(n * rows_in, hidden),
        NamedSharding(mesh, P(AXIS)))
    rank = jax.device_put(jnp.arange(n, dtype=jnp.int32),
                          NamedSharding(mesh, P(AXIS)))
    kernel = getattr(sgs, fn_name)

    def body(x, r):
        return kernel(x,
                      num_cores=n,
                      axis=AXIS,
                      rank=r if rank_operand else None)

    fn = jax.jit(
        jax.shard_map(body,
                      mesh=mesh,
                      in_specs=(P(AXIS), P(AXIS)),
                      out_specs=P(AXIS),
                      check_vma=False))
    out = np.asarray(fn(x, rank))
    return out.reshape(n, out.shape[0] // n, hidden)


@pytest.mark.parametrize("rows", [4, 16, 20, 24, 64])
@pytest.mark.parametrize("rank_operand", [False, True])
def test_reduce_scatter_is_the_float32_sum_of_my_block(rows, rank_operand):
    blocks = _blocks((N, N * rows, 4096), seed=100 + rows)
    got = _run(_mesh(), blocks, "small_reduce_scatter", rank_operand)
    for core in range(N):
        want = np.asarray(sgs.reference_scatter(jnp.asarray(blocks),
                                                core)).astype(np.float32)
        step = np.spacing(np.abs(want).max().astype(np.float32)) * 2**16
        np.testing.assert_allclose(got[core].astype(np.float32),
                                   want,
                                   atol=step,
                                   rtol=0)


def test_refuses_a_width_off_the_lane_tile():
    with pytest.raises(ValueError):
        sgs.small_reduce_scatter(jnp.zeros((8, 4000), jnp.bfloat16),
                                 num_cores=N)


@pytest.mark.parametrize("rank_operand", [False, True])
@pytest.mark.parametrize("rows", [4, 8, 20, 24])
def test_reduce_scatter_with_a_permuted_rank_order(_needs_the_eight_cores,
                                                   rank_operand, rows):
    """Core m serves rank rank_of_core[m], while blocks remain in rank order.
    Each core must receive the float32 sum of its own rank's block."""
    mesh = _mesh()
    n = 8
    hidden = 4096
    blocks = _blocks((n, n * rows, hidden), seed=7)
    rank_of_core = np.array([0, 1, 6, 7, 2, 3, 4, 5],
                            np.int32)  # a non-identity rank order
    core_of_rank = np.argsort(rank_of_core).astype(np.int32)
    x = jax.device_put(
        jnp.asarray(blocks).reshape(n * n * rows, hidden),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d",
                                                                    None)))
    rank = jax.device_put(
        jnp.arange(n, dtype=jnp.int32),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d")))
    my_rank = jax.device_put(
        jnp.asarray(rank_of_core),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d")))
    table = jax.device_put(
        jnp.tile(jnp.asarray(core_of_rank), n),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d")))

    def body(x, rank, my_rank, table):
        return sgs.small_reduce_scatter(x,
                                        num_cores=n,
                                        axis="d",
                                        rank=rank if rank_operand else None,
                                        my_rank=my_rank,
                                        core_of_rank=table)

    P = jax.sharding.PartitionSpec
    fn = jax.jit(
        jax.shard_map(body,
                      mesh=mesh,
                      in_specs=(P("d", None), P("d"), P("d"), P("d")),
                      out_specs=P("d", None),
                      check_vma=False))
    out = np.asarray(fn(x, rank, my_rank, table)).reshape(n, rows, hidden)
    for m in range(n):
        want = np.asarray(
            sgs.reference_scatter(jnp.asarray(blocks), int(rank_of_core[m])))
        np.testing.assert_array_equal(out[m], want)
