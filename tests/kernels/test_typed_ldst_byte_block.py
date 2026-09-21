"""The typed views of an fp8-typed pool block (kernels/typed_ldst.py) round-trip on the device.

The block has the pool's own shape, ``[rows, 1, 4, 256]`` bytes (one KV-cache block of this model:
``f8e4m3fn[blocks, 128, 1, 4, 256]``), because the legality of Mosaic's ref reshape depends on the tiling
that shape gets: the 32-bit view of it is (1, 128)-tiled and reshapes to 128 lanes, a 2-D ``[64, 256]``
block gets the large (32, 128) tiling and does not. The bf16 state stored through store_typed with a lane
split and read back through load_typed must be identical, and the f32 routes unchanged. This is the case
the fp8-typed unified pool puts the bf16 GDN recurrent state through (--mamba-ssm-cache-dtype bfloat16)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import typed_ldst

BLOCK = (128, 1, 4, 256)  # one pool block: 128 tokens x (1 x 4 x 256) bytes
LANES = BLOCK[-1]


def _roundtrip(view_dtype, lane_split, block):
    nbytes = int(np.prod(block))
    lanes = block[-1] // lane_split
    n = nbytes // (jnp.dtype(view_dtype).itemsize * lanes)
    values = jax.random.normal(jax.random.PRNGKey(0), (n, lanes),
                               jnp.float32).astype(view_dtype)

    def kernel(v_ref, block_ref, out_ref):
        typed_ldst.store_typed(block_ref, v_ref[...], lane_split=lane_split)
        out_ref[...] = typed_ldst.load_typed(block_ref,
                                             view_dtype=view_dtype,
                                             lane_split=lane_split)

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(values.shape, view_dtype),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM)
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
    )(values, jnp.zeros(block, jnp.float8_e4m3fn))
    return values, out


@pytest.mark.parametrize("view_dtype,lane_split", [(jnp.bfloat16, 2),
                                                   (jnp.float32, 2),
                                                   (jnp.float32, 1),
                                                   (jnp.bfloat16, 1)])
def test_typed_view_roundtrips_on_a_pool_block(view_dtype, lane_split):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("device test")
    values, out = _roundtrip(view_dtype, lane_split, BLOCK)
    np.testing.assert_array_equal(np.asarray(out, np.float32),
                                  np.asarray(values, np.float32))


def test_f32_view_without_a_lane_split_on_a_flat_block():
    """The 2-D block (a 3-D pool) keeps the large tiling; only the leading-dim merge is legal there."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("device test")
    values, out = _roundtrip(jnp.float32, 1, (64, 256))
    np.testing.assert_array_equal(np.asarray(out, np.float32),
                                  np.asarray(values, np.float32))
