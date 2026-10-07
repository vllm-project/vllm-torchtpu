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
"""Problems found in scenario data."""

from dataclasses import dataclass
from typing import Literal

from scenario.paths import KeyPath

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    """A problem at a key path of scenario data.

    `message` reads after the dotted path ("server.multihost: ..."). A
    warning is reported but does not fail validation.
    """

    path: KeyPath
    message: str
    hint: str | None = None
    rule: str | None = None
    severity: Severity = "error"


def errors_only(issues: list[Issue]) -> list[Issue]:
    return [issue for issue in issues if issue.severity == "error"]
