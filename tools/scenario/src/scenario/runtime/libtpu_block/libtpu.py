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
"""A libtpu that cannot be imported, for hosts without TPU chips.

The runtime puts this directory first on PYTHONPATH for a chipless driver's
tasks and router (P-18). The TPU image ships libtpu, and vLLM's platform
detection takes an importable libtpu to mean the host has TPUs, then fails
on a CPU-only host. Importing this module fails as an absent one would, so
detection finds no TPU, and nothing installed in the image has to change.
"""

raise ModuleNotFoundError(
    "libtpu is blocked on this host, which has no TPU chips", name="libtpu"
)
