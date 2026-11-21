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
"""Correctness bisect for stage L1 on TPU: quantize-once x wire rows x wire dtype.

Prints one ``BISECT`` line per variant with the max abs error against the
single-device oracle on every local device.
"""

import itertools
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.vmem_resident_moe import kernel as K
from vllm_torchtpu.kernels.experimental.vmem_resident_moe import layer as L


def main():
    experts = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    vmem_mib = int(sys.argv[2]) if len(sys.argv) > 2 else 61
    mesh = jax.make_mesh((jax.device_count(),), (L.AXIS,))
    tokens, steps = 16, 4
    for rows in (16, 32):
        dims = dict(experts=experts, hidden=2816, inter=704, rows=rows)
        sched, x0, gate, ws = K.make_inputs(
            jax.random.PRNGKey(0), tokens=tokens, steps=steps, top_k=8, **dims
        )
        ref = np.asarray(K.reference(sched, x0, gate, ws))[:tokens]
        starts, flat = L.device_schedules(sched, steps, experts, mesh.size)
        args = L.place(mesh, starts, flat, x0, gate, ws)
        for hoist, wire, dt in itertools.product(
            (False, True), (16, rows), (jnp.float32, jnp.bfloat16)
        ):
            if hoist and rows % 32:
                continue  # fp8 scratch needs rows % 32 == 0
            if wire > rows:
                continue
            f = L.build_layer(
                mesh,
                weights="resident",
                allreduce="ar",
                steps=steps,
                vmem_limit_bytes=vmem_mib * 2**20,
                hoist=hoist,
                wire_rows=wire,
                ar_dtype=dt,
                **dims,
            )
            out = f(*args)
            errs = [
                round(float(np.max(np.abs(np.asarray(s.data)[:tokens] - ref))), 6)
                for s in out.addressable_shards
            ]
            rec = {
                "rows": rows,
                "hoist": hoist,
                "wire": wire,
                "dtype": jnp.dtype(dt).name,
                "err": errs,
            }
            sys.stdout.write(f"P{jax.process_index()} BISECT {json.dumps(rec)}\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
