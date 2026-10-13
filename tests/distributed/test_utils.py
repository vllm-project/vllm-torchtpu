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
from vllm_torchtpu.distributed import utils


def test_kv_transfer_names_default(monkeypatch):
    monkeypatch.delenv("TPU_KV_TRANSFER_NAMESPACE", raising=False)

    assert utils.get_ipc_socket_path(0) == "ipc:///tmp/tpu_conn_node0_0.sock"
    assert utils.get_shm_name(0) == "tpu_conn_kv_node0_0"


def test_kv_transfer_names_are_namespaced(monkeypatch):
    monkeypatch.setenv("TPU_KV_TRANSFER_NAMESPACE", "prefill")

    assert utils.get_ipc_socket_path(0) == "ipc:///tmp/tpu_conn_prefill_node0_0.sock"
    assert utils.get_shm_name(0) == "tpu_conn_kv_prefill_node0_0"


def test_kv_transfer_namespace_is_sanitized(monkeypatch):
    monkeypatch.setenv("TPU_KV_TRANSFER_NAMESPACE", "decode:9400")

    assert (
        utils.get_ipc_socket_path(2) == "ipc:///tmp/tpu_conn_decode_9400_node2_0.sock"
    )
    assert utils.get_shm_name(2) == "tpu_conn_kv_decode_9400_node2_0"
