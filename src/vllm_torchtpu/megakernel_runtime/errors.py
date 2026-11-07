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
"""Errors of the megakernel runtime (stdlib only). Config errors are the
library's `megakernel.errors.MegakernelConfigError`."""

from __future__ import annotations


class MegakernelPluginError(RuntimeError):
    """A plugin can't be fingerprinted (e.g. `plugin_source_unhashable:<why>`)."""


class MegakernelUnsupported(RuntimeError):
    """A framework feature was used that a serving megakernel can't follow
    (e.g. `weight_update`: the packed weights would go stale)."""


__all__ = ["MegakernelPluginError", "MegakernelUnsupported"]
