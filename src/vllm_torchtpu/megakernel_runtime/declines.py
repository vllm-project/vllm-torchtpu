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
"""`LayoutDecline`: the decline a program's `bind` raises (imports the library)."""

from __future__ import annotations

from vllm_torchtpu.megakernel.errors import EligibilityResult, MegakernelLayoutError


class LayoutDecline(MegakernelLayoutError):
    """The runner's cache layout isn't the one the megakernel was built for.

    Carries a named `EligibilityResult`; the runtime makes it rank-uniform.
    """

    def __init__(self, result: EligibilityResult) -> None:
        super().__init__(f"Megakernel: {result.message}")
        self.result = result


__all__ = ["LayoutDecline"]
