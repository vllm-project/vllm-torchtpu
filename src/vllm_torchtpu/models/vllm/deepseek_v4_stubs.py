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
"""CUDA compatibility mocks and stubs for DeepSeek-V4 TPU execution."""

# TODO(patemotter): revisit alongside the dedicated DSv4 model.py; a model that
# does not call into torch.cuda needs fewer of these.

import torch
from vllm.platforms import current_platform


class DummyDeviceCapability:
    major = 9
    minor = 0

    def to_int(self):
        return self.major * 10 + self.minor

    def as_version_str(self):
        return f"{self.major}.{self.minor}"

    def __getitem__(self, idx):
        return (self.major, self.minor)[idx]


# TODO(upstream): Delete when PyTorch or vLLM provides native TPU CUDA stream mocking.
class DummyCudaStream:

    def __init__(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass

    def wait(self, *args, **kwargs):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass


class DummyCudaEvent:

    def __init__(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass

    def wait(self, *args, **kwargs):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass


def patch_cuda_stubs() -> None:
    """Mock CUDA device capability, Stream, and Event allocation since they are unavailable on TPU."""
    current_platform.get_device_capability = (
        lambda *args, **kwargs: DummyDeviceCapability())
    torch.cuda.Stream = DummyCudaStream
    torch.cuda.Event = DummyCudaEvent
