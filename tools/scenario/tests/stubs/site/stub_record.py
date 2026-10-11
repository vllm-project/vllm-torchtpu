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
"""Records a stand-in module's call as stubs.py records a program's.

One JSON file under $STUB_RECORDS per call, with the same keys, so
runtime_helpers.records() reads them in order with the programs'.
"""

import json
import os
import pathlib
import time
from typing import Any


def record(program: str, argv: list[str], **extra: Any) -> None:
    records = os.environ.get("STUB_RECORDS")
    if not records:
        return
    data = {
        "program": program,
        "argv": argv,
        "env": dict(os.environ),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        **extra,
    }
    path = pathlib.Path(records) / f"{time.time_ns()}-{os.getpid()}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(path)
