# Copyright 2025 Google LLC
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
"""Host-side run directory and environment metadata for the archived experiments."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path


def prepare_run() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("--output must be empty; use a new directory for each run")
    required = [
        "--xla_tpu_profile_traceme_level=10",
        "--xla_enable_custom_call_region_trace=true",
    ]
    flags = os.environ.get("LIBTPU_INIT_ARGS", "").split()
    if not all(flag in flags for flag in required):
        parser.error("Set LIBTPU_INIT_ARGS as documented in README.md before running")
    metadata = {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["jax", "jaxlib", "libtpu", "numpy"]
        },
        "libtpu_init_args": flags,
    }
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return args.output
