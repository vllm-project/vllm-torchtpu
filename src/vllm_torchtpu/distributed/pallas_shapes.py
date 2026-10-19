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
"""Shape conversion for native Pallas ops with multi-axis PartitionSpecs.

TorchTPU 20260917 handles explicit output specs, but its shape conversion
assumes one mesh axis per dimension. GDN state is sharded on (tp, pcp) on
one dimension. Install the same conversion for native input and output
placeholders; keep JaxCallable, packed dtype handling and dispatch upstream.
"""

import math


def _convert_partitioned_shape(shape, mesh, partition_spec, *, to_global):
    if mesh is None or partition_spec is None:
        return tuple(shape)
    result = []
    for index, size in enumerate(shape):
        axes = partition_spec[index] if index < len(partition_spec) else None
        axes = () if axes is None else axes if isinstance(axes, tuple) else (axes,)
        factor = math.prod(int(mesh.shape[axis]) for axis in axes)
        if not to_global and size % factor:
            raise ValueError(
                f"Global dimension {size} is not divisible by mesh factor {factor}."
            )
        result.append(size * factor if to_global else size // factor)
    return tuple(result)


def get_global_shape(local_shape, mesh, partition_spec):
    return _convert_partitioned_shape(local_shape, mesh, partition_spec, to_global=True)


def get_local_shape(global_shape, mesh, partition_spec):
    return _convert_partitioned_shape(
        global_shape, mesh, partition_spec, to_global=False
    )


def install_pallas_shape_conversion():
    """Apply the shared conversion before custom ops are registered."""
    from torch_tpu._internal.pallas import pallas

    pallas.get_global_shape = get_global_shape
    pallas.get_local_shape = get_local_shape
