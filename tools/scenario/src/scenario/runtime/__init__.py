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
"""The runtime: runs one host's part of a Plan.

It reads the Plan and the host's placement, fills the Plan's Refs, and runs
what the Plan says: the server, then the driver's tasks. It imports the Plan
types and codec and the target's data types, never the scenario model, so a
pod needs nothing but its own env to run its part.
"""
