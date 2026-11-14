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
"""Run single-mesh SparseCore kernels through ``core_map``.

``pl.kernel`` now dispatches to ``mpmd_map``; ``kernel`` here bypasses that
and calls ``core_map`` directly on the given mesh. Near drop-in for
``pl.kernel`` -- note ``out_shape`` becomes ``out_type`` and ``scratch_shapes``
(list) becomes ``scratch_types`` (dict).
"""

import functools

import numpy as np
from jax._src import api
from jax._src import core as jax_core
from jax._src import dispatch, dtypes, lax, tree_util
from jax._src.interpreters import mlir
from jax._src.lib.mlir.dialects import hlo
from jax._src.pallas import core as pl_core
from jax._src.sharding_impls import canonicalize_sharding

_dependent_empty_p = jax_core.Primitive("dependent_empty")
_dependent_empty_p.def_impl(
    functools.partial(dispatch.apply_primitive, _dependent_empty_p))


def _dependent_empty_abstract_eval(*deps, shape, dtype, out_sharding):
    del deps
    return jax_core.ShapedArray(shape, dtype, sharding=out_sharding)


_dependent_empty_p.def_abstract_eval(_dependent_empty_abstract_eval)


def _dependent_empty_lower(ctx, *deps, shape, dtype, out_sharding):
    del deps
    return lax._empty_lower(ctx,
                            shape=shape,
                            dtype=dtype,
                            out_sharding=out_sharding)


def _dependent_empty_custom_call_lower(ctx, *deps, shape, dtype, out_sharding):
    if not jax_core.is_constant_shape(shape):
        return _dependent_empty_lower(ctx,
                                      *deps,
                                      shape=shape,
                                      dtype=dtype,
                                      out_sharding=out_sharding)
    dtype = (dtype
             if dtypes.issubdtype(dtype, dtypes.extended) else np.dtype(dtype))
    aval_out = jax_core.ShapedArray(shape, dtype, sharding=out_sharding)
    phys_aval = jax_core.physical_aval(aval_out)
    custom_call_op = hlo.CustomCallOp(
        [
            mlir.ir.RankedTensorType.get(
                list(phys_aval.shape), mlir.dtype_to_ir_type(phys_aval.dtype))
        ],
        list(deps),
        call_target_name=mlir.ir.StringAttr.get("AllocateBuffer"),
        has_side_effect=mlir.ir.BoolAttr.get(False),
    )
    assert len(custom_call_op.results) == 1
    res = custom_call_op.results[0]
    return [mlir.lower_with_sharding_in_types(ctx, res, phys_aval)]


mlir.register_lowering(_dependent_empty_p, _dependent_empty_lower)
mlir.register_lowering(_dependent_empty_p, _dependent_empty_custom_call_lower,
                       "tpu")


def _empty_out_ref(out_type, deps=()):
    # Mirrors internal `pl.kernel` logic; may need updating on JAX bumps.
    aval = pl_core._convert_out_shape_to_aval(out_type)
    memory_space = (None if isinstance(aval.memory_space, jax_core.MemorySpace)
                    else aval.memory_space)
    if deps:
        out_sharding = canonicalize_sharding(aval.sharding, "dependent_empty")
        value = _dependent_empty_p.bind(
            *deps,
            shape=aval.shape,
            dtype=aval.dtype,
            out_sharding=out_sharding,
        )
    else:
        value = lax.empty(aval.shape, aval.dtype, out_sharding=aval.sharding)
    return jax_core.new_ref(value, memory_space=memory_space)


def kernel(body,
           *,
           out_type,
           mesh,
           scratch_types=(),
           compiler_params=None,
           interpret=False,
           cost_estimate=None,
           debug=False,
           name=None,
           metadata=None):
    """Near drop-in for ``pl.kernel`` that lowers via ``core_map``.

    Differs in that ``out_shape`` is renamed to ``out_type`` and
    ``scratch_shapes`` (list) is renamed to ``scratch_types`` (dict).
    """
    single_output = not isinstance(out_type, (tuple, list))
    out_types = (out_type, ) if single_output else out_type

    @api.jit
    def run(*operands):
        arg_refs = tree_util.tree_map(jax_core.new_ref, operands)
        dep_arrays = tuple(tree_util.tree_leaves(operands))
        out_refs = tree_util.tree_map(
            functools.partial(_empty_out_ref, deps=dep_arrays), out_types)

        @pl_core.core_map(
            mesh,
            scratch_shapes=scratch_types,
            compiler_params=compiler_params,
            interpret=interpret,
            cost_estimate=cost_estimate,
            debug=debug,
            name=name,
            metadata=metadata,
        )
        def _(*scratch_refs, **scratch_kwrefs):
            return body(*arg_refs, *out_refs, *scratch_refs, **scratch_kwrefs)

        outs = tree_util.tree_map(lambda ref: ref[...], out_refs)
        return outs[0] if single_output else outs

    return run
