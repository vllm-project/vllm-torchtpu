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
# ==============================================================================

"""Helper function on sharding related logics."""


class ShardingAxisName:
  SEQUENCE = 'data'
  ATTN_DATA = 'data'
  MLP_DATA = 'data'
  ATTN_HEAD = 'model'
  ATTN_TENSOR = None
  MLP_TENSOR = 'model'
  MOE_TENSOR = 'model'
  EXPERT = 'model'
  EXPERT_DATA = ('data', 'model')
  VOCAB = ('data', 'model')


def get_mesh_shape_product(mesh, axes) -> int:
  if axes is None:
    return 1

  if isinstance(axes, str):
    axes = [axes]

  product = 1
  for axis in axes:
    product *= mesh.shape.get(axis, 1)

  return product
