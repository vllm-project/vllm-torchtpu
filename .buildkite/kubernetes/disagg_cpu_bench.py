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
"""Run the HTTP benchmark on CPU without uninstalling the image's TPU runtime."""

import os


def main() -> None:
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    from vllm import platforms
    from vllm.platforms.cpu import CpuPlatform

    # The upstream bench CLI uses this same public platform assignment for CPU.
    platforms.current_platform = CpuPlatform()
    from vllm.entrypoints.cli.main import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
