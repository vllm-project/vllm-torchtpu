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

import sys
from unittest.mock import MagicMock

# Mock torch_tpu only if running on host machines without torch_tpu installed
if "torch_tpu" not in sys.modules:
    try:
        import torch_tpu  # noqa: F401
    except ImportError:
        sys.modules["torch_tpu"] = MagicMock()
        sys.modules["torch_tpu._internal"] = MagicMock()
        sys.modules["torch_tpu._internal.compile"] = MagicMock()
        sys.modules["torch_tpu._internal.compile._backend"] = MagicMock()

import pytest

from vllm_torchtpu.compilation import tpu_compiler


def test_signed_cache_save_load(tmp_path):
    cache_path = str(tmp_path / "test_artifact.bin")
    dummy_payload = {"op": "matmul", "dims": [128, 128], "weights": [1, 2, 3]}

    tpu_compiler._save_signed_cache(dummy_payload, cache_path)
    loaded = tpu_compiler._load_signed_cache(cache_path)
    assert loaded == dummy_payload


def test_signed_cache_tampering_rejected(tmp_path):
    cache_path = str(tmp_path / "test_artifact_tampered.bin")
    dummy_payload = {"op": "matmul", "dims": [128, 128]}

    tpu_compiler._save_signed_cache(dummy_payload, cache_path)

    # Modify cached file content after the 32-byte signature
    with open(cache_path, "r+b") as f:
        f.seek(35)
        byte = f.read(1)
        f.seek(35)
        f.write(bytes([ord(byte) ^ 0xFF]))

    with pytest.raises(ValueError, match="HMAC authentication failed"):
        tpu_compiler._load_signed_cache(cache_path)


def test_signed_cache_short_file(tmp_path):
    cache_path = str(tmp_path / "short.bin")
    with open(cache_path, "wb") as f:
        f.write(b"too short")

    with pytest.raises(ValueError, match="too short"):
        tpu_compiler._load_signed_cache(cache_path)
