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
"""Pytest conftest for experimental Rust launcher tests."""

import contextlib

# vLLM's `build_arg_parser()` / `make_arg_parser()` constructs DeviceConfig, which
# validates and initializes `torch.device("tpu")`. When running CPU unit tests
# without a live TPU hardware device (where `libtpu` does not initialize the device),
# PyTorch's custom backend must be renamed to 'tpu' so vLLM CLI parser construction
# succeeds instead of raising `RuntimeError: Expected one of ... device type: tpu`.
with contextlib.suppress(Exception):
    import torch

    torch.utils.rename_privateuse1_backend("tpu")
