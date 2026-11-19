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
"""Hyper-connection package: `HCSublayer` and its pure-JAX reference."""

from .block import RESIDUAL_OUT, HCSublayer
from .reference import hc_combine, hc_mix, hc_sublayer_reference

__all__ = [
    "RESIDUAL_OUT",
    "HCSublayer",
    "hc_combine",
    "hc_mix",
    "hc_sublayer_reference",
]
