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
"""Run single-mesh SparseCore kernels through ``pl.kernel``.

Delegates directly to ``pl.kernel`` (``mpmd_map`` with explicit ``out_type``)
wrapped in ``@jax.jit`` rather than pre-allocating output buffers via
``lax.empty`` + ``jax_core.new_ref`` + ``core_map``.

Pre-allocating output refs via ``lax.empty`` emits zero-operand
``stablehlo.custom_call @AllocateBuffer()`` ops with ``has_side_effect = false``
and aliases kernel outputs to those inputs via ``output_operand_aliases``. When
``torch_tpu`` inlines custom-kernel functions across chunks/layers and runs
``mlir::createCSEPass()``, all identical ``AllocateBuffer`` ops are merged into
a single shared SSA value, forcing XLA ``CopyInsertion`` to insert hundreds of
HBM-to-HBM ``copy(%empty)`` operations per step. Passing ``out_type`` directly
to ``pl.kernel`` produces pure ``tpu_custom_call`` outputs with zero
``AllocateBuffer`` calls.
"""

from jax._src import api
from jax.experimental import pallas as pl


def kernel(
    body,
    *,
    out_type,
    mesh,
    scratch_types=(),
    compiler_params=None,
    interpret=False,
    cost_estimate=None,
    debug=False,
    name=None,
    metadata=None,
):
    """JIT-wrapped ``pl.kernel`` for single-mesh SparseCore kernels."""
    return api.jit(
        pl.kernel(
            body,
            out_type=out_type,
            mesh=mesh,
            scratch_types=scratch_types,
            compiler_params=compiler_params,
            interpret=interpret,
            cost_estimate=cost_estimate,
            debug=debug,
            name=name,
            metadata=metadata,
        )
    )
