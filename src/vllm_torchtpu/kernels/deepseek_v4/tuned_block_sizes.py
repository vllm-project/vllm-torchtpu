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

# (TPU generation, tokens, heads, head dim, reduction, output features)
# -> (DMA generations, compiler VMEM budget). v7 P7 complete-call XPlane A/B:
# T240 22.630 -> 20.399 us; T288 25.391 -> 21.573 us. Three generations hide
# input latency; a fourth did not improve the complete call. The 56 MiB budget
# is a measured compiler setting, not a hardware-capacity constant.
OUTPUT_PROJECTION_TUNING = {
    (7, 240, 64, 512, 4096, 8192): (3, 56 * 1024**2),
    (7, 288, 64, 512, 4096, 8192): (3, 56 * 1024**2),
}
