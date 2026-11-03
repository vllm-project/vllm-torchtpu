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
"""The ``collective_id`` of every DSv4 Pallas kernel that uses a barrier.

Kernels that can run in one program need distinct ids: the id picks the
barrier semaphore, so two kernels sharing one could consume each other's
signals. Add new DSv4 ids here.
"""

# The fused EP MoE kernel (kernels/fused_moe/v2/kernel.py).
FUSED_MOE_COLLECTIVE_ID = 0
# Chip-pair o fetch inside wo_a (o_projection.py).
O_FETCH_COLLECTIVE_ID = 3
# Chip-pair q send start (q_send.py).
Q_SEND_COLLECTIVE_ID = 4
