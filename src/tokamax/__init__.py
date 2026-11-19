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
"""Minimal vendored tokamax subtree for the De-Dup MoE kernel.

This is NOT the full tokamax package. The upstream `tokamax/__init__.py`
imports every op (attention, ragged_dot, triton, mosaic_gpu, ...); this stub
intentionally imports nothing so that only
`tokamax._src.ops.experimental.{dedup_moe,fused_moe}` and
`tokamax._src.ops.gmm_v2` (plus `tokamax._src.config`) are loaded.
"""

__version__ = "0.0.0+dedup_moe_vendor"
