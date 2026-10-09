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

import pytest

from vllm_torchtpu.distributed.kv_transfer import zmq_shm_base


def test_secure_dumps_loads():
    payload = ("STAGE_NOTIFY", 12345, [0, 1, 2], {"key": "val"})
    encoded = zmq_shm_base._secure_dumps(payload)
    decoded = zmq_shm_base._secure_loads(encoded)
    assert decoded == payload


def test_secure_loads_tampering_rejected():
    payload = ("STAGE_NOTIFY", 12345)
    encoded = zmq_shm_base._secure_dumps(payload)

    # Tamper with payload bytes after the 32-byte signature
    tampered_bytes = bytearray(encoded)
    tampered_bytes[-1] ^= 0xFF
    with pytest.raises(ValueError, match="HMAC authentication failed"):
        zmq_shm_base._secure_loads(bytes(tampered_bytes))

    # Tamper with the signature bytes
    tampered_sig = bytearray(encoded)
    tampered_sig[0] ^= 0xFF
    with pytest.raises(ValueError, match="HMAC authentication failed"):
        zmq_shm_base._secure_loads(bytes(tampered_sig))


def test_secure_loads_short_payload():
    with pytest.raises(ValueError, match="too short"):
        zmq_shm_base._secure_loads(b"short")
