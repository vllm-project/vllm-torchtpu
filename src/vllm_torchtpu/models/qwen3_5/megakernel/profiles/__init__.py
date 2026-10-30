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
"""The Qwen3.5 execution profiles, by name.

- `qwen35-baseline` (`baseline.py`): the model's reference profile, exact
  numerics, conservative tiles, split pool I/O.
- `qwen35-candidate30` / `qwen35-candidate30-fused` (`candidate30.py`): the
  tuned schedule, approx numerics; `-fused` adds 1-launch fused pool I/O and
  is the template's serving default (`DEFAULT_PROFILE`).

Importing this package registers nothing (`model.register()` does).
"""

from __future__ import annotations

from vllm_torchtpu.megakernel.profile import Profile

from . import baseline, candidate30

PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (baseline.PROFILE, candidate30.PROFILE, candidate30.FUSED_PROFILE)
}
DEFAULT_PROFILE: Profile = candidate30.FUSED_PROFILE

__all__ = ["DEFAULT_PROFILE", "PROFILES", "baseline", "candidate30"]
