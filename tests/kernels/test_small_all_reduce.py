"""The small all-reduce kernel (kernels/collectives/small_all_reduce.py)
against the float32 sum and jax.lax.psum on the 8 cores of one host:
collective forms at 4 to 256 rows, the core's index read from the axis and handed in as
data (the torch bridge's way), and the served widths."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives import small_all_reduce as sar

N = 8
AXIS = "d"


@pytest.fixture(autouse=True)
def _needs_the_eight_cores():
    if jax.default_backend() != "tpu" or len(jax.devices()) < N:
        pytest.skip("needs the 8 TPU cores of one host")


def _mesh():
    return Mesh(np.array(jax.devices()[:N]), (AXIS, ))


def _blocks(rows, hidden, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, rows, hidden),
                               dtype=np.float32).astype(jnp.bfloat16)


def _run(mesh, blocks, rank_operand, form=None):
    n, rows, hidden = blocks.shape
    x = jax.device_put(
        jnp.asarray(blocks).reshape(n * rows, hidden),
        NamedSharding(mesh, P(AXIS)))
    rank = jax.device_put(jnp.arange(n, dtype=jnp.int32),
                          NamedSharding(mesh, P(AXIS)))

    def body(x, r):
        return sar.small_all_reduce(x,
                                    num_cores=n,
                                    axis=AXIS,
                                    rank=r if rank_operand else None,
                                    form=form)

    fn = jax.jit(
        jax.shard_map(body,
                      mesh=mesh,
                      in_specs=(P(AXIS), P(AXIS)),
                      out_specs=P(AXIS),
                      check_vma=False))
    return np.asarray(fn(x, rank)).reshape(n, rows, hidden)


def _check(got, blocks):
    want = np.asarray(sar.reference(jnp.asarray(blocks))).astype(np.float32)
    got = got.astype(np.float32)
    # every core holds the same sum, within one bf16 step of the float32 sum
    # rounded once (the summation orders differ)
    step = np.spacing(np.abs(want).max().astype(np.float32)) * 2**16
    for core in range(got.shape[0]):
        np.testing.assert_allclose(got[core], want, atol=step, rtol=0)


@pytest.mark.parametrize("rows", [4, 8, 16, 32, 64, 256])
@pytest.mark.parametrize("rank_operand", [False, True])
def test_matches_the_float32_sum(rows, rank_operand):
    blocks = _blocks(rows, 4096, seed=rows)
    _check(_run(_mesh(), blocks, rank_operand), blocks)


@pytest.mark.parametrize("form", ["oneshot", "twoshot"])
@pytest.mark.parametrize("rows", [4, 16])
def test_both_forms_at_both_sizes(form, rows):
    blocks = _blocks(rows, 4096, seed=100 + rows)
    _check(_run(_mesh(), blocks, rank_operand=True, form=form), blocks)


@pytest.mark.parametrize("hidden", [2048, 5120])
def test_other_widths(hidden):
    blocks = _blocks(16, hidden, seed=hidden)
    _check(_run(_mesh(), blocks, rank_operand=True), blocks)


def test_agrees_with_psum_within_its_rounding():
    mesh = _mesh()
    blocks = _blocks(16, 4096, seed=7)
    x = jax.device_put(
        jnp.asarray(blocks).reshape(N * 16, 4096),
        NamedSharding(mesh, P(AXIS)))
    psum = jax.jit(
        jax.shard_map(lambda x: jax.lax.psum(x, AXIS),
                      mesh=mesh,
                      in_specs=P(AXIS),
                      out_specs=P(AXIS),
                      check_vma=False))
    want = np.asarray(psum(x)).reshape(N, 16, 4096).astype(np.float32)
    got = _run(mesh, blocks, rank_operand=True).astype(np.float32)
    # XLA's ring rounds to bf16 at every hop; the kernel rounds once
    assert np.abs(got - want).max() <= 0.125


def test_refuses_a_width_that_does_not_split():
    with pytest.raises(ValueError):
        sar.small_all_reduce(jnp.zeros((16, 4000), jnp.bfloat16), num_cores=N)


def test_form_by_bytes():
    assert sar.form_for(4, 4096, 2) == "oneshot"
    assert sar.form_for(8, 4096, 2) == "twoshot"
    assert sar.form_for(8, 2048, 2) == "oneshot"
