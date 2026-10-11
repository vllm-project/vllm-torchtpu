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
"""Ray's Python API, as far as the head's node count uses it.

The nodes are the ones the stub `ray start` registered in $STUB_RAY_STATE.
"""

import json
import os
import pathlib
from typing import Any


def init(address: str | None = None, logging_level: str | None = None) -> None:
    # Ray floods stdout once a driver attaches; the count must not rely on it.
    print("(stub) Connecting to existing Ray cluster at address: " + str(address))
    if not pathlib.Path(os.environ["STUB_RAY_STATE"]).exists():
        raise ConnectionError("Could not find any running Ray instance.")


def nodes() -> list[dict[str, Any]]:
    path = pathlib.Path(os.environ["STUB_RAY_STATE"])
    return [
        {"Alive": node["Alive"], "Resources": node["Resources"]}
        for node in json.loads(path.read_text())
    ]
