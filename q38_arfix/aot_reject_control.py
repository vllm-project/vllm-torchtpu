# SPDX-License-Identifier: Apache-2.0
"""q38 AR-FIX deviceless reject control: a Pallas TPU kernel that Mosaic must refuse (a VMEM scratch far larger than
any TensorCore's VMEM) is compiled in the same deviceless tpu7x:2x2x1 harness as ARB1's --aot legs. It must FAIL to
compile; if it compiles, the harness is not really compiling Pallas kernels and the AOT legs prove nothing.
Prints 'Q38_ARFIX_REJECT_CONTROL PASS' (the compile failed) or '... FAIL' (it compiled). Exit 0 only on PASS."""
import sys

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental import topologies
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

devs = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1").devices
mesh = Mesh(np.array(devs[:1]), ("d",))


def kernel(x_ref, o_ref, big):
    big[...] = jnp.zeros(big.shape, big.dtype)
    o_ref[...] = x_ref[...] + big[0:16, 0:128]


def f(x):
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                          scratch_shapes=[pltpu.VMEM((262144, 4096), jnp.float32)])(x)  # 4 GiB of VMEM


arg = jax.ShapeDtypeStruct((16, 128), jnp.float32, sharding=NamedSharding(mesh, P()))
try:
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        jax.jit(f).lower(arg).compile()
    print("Q38_ARFIX_REJECT_CONTROL FAIL (an over-VMEM Pallas kernel compiled deviceless)", flush=True)
    sys.exit(1)
except Exception as e:  # noqa: BLE001
    print("Q38_ARFIX_REJECT_CONTROL PASS (compile refused: %s)" % str(e).replace("\n", " ")[:300], flush=True)
    sys.exit(0)
