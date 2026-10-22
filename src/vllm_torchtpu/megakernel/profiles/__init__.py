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
"""Checked-in model-agnostic execution profiles.

`default` holds the `default` profile's model-agnostic tile rule (imported
eagerly; it depends only on `exec_spec`). Model-specific profiles (the model's
default profile and tuned profiles targeting that model) live under `models/`
(e.g. `models/qwen35_profile.py`, `models/qwen35_candidate30.py`) and register
when `models` is imported.
"""

from __future__ import annotations

from . import default


def load_builtin_profiles() -> None:
    """No-op extension hook: `default` is registered when `profile` imports, and
    model-specific profiles register when `models` imports."""


__all__ = ["default", "load_builtin_profiles"]
