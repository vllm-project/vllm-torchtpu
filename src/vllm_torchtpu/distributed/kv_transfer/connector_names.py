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
"""The KV connectors a KV transfer config puts in effect (no vLLM imports)."""

from collections.abc import Container
from typing import Any

# Connectors that list their child connectors in
# `kv_connector_extra_config["connectors"]`: vLLM's and the TPU one.
MULTI_CONNECTORS = ("MultiConnector", "TPUMultiConnector")


def kv_connector_names(
    kv_transfer_config: Any, multi_connectors: Container[str] = MULTI_CONNECTORS
) -> list[str]:
    """The connector a config names, or the children of a multi-connector (a
    name in `multi_connectors`); empty when the config names no connector."""
    name = kv_transfer_config.kv_connector
    if not name:
        return []
    if name not in multi_connectors:
        return [name]
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    return [c for child in extra.get("connectors", []) if (c := child["kv_connector"])]
