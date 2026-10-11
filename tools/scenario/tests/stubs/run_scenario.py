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
"""`scenario` with some runtime constants replaced, for runs of several hosts
on one machine.

    python run_scenario.py run --role ... (the CLI's own arguments)

STUB_PATCH is JSON {"module.NAME": value}; each is set before the CLI runs,
so the hosts of a test can use free ports and short waits. A name the module
does not have is an error, so a renamed constant cannot go unnoticed.
"""

import importlib
import json
import os
import sys


def main() -> None:
    for target, value in json.loads(os.environ.get("STUB_PATCH", "{}")).items():
        name, _, attribute = target.rpartition(".")
        module = importlib.import_module(name)
        if not hasattr(module, attribute):
            sys.exit(f"run_scenario.py: {name} has no {attribute}")
        setattr(module, attribute, value)
    from scenario.cli import main as cli

    sys.argv[0] = "scenario"
    cli()


if __name__ == "__main__":
    main()
