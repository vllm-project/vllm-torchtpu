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

# Names that resolve to the Raiden Stage-3 P/D connector. `TPUConnector` is a
# compatibility alias of `TPURaidenConnector`; every check that asks "is this
# the Raiden P/D connector?" must accept both.
RAIDEN_PD_CONNECTOR_NAMES: tuple[str, ...] = ("TPURaidenConnector", "TPUConnector")

# Startup rejections shared by the early platform config check and the
# connector itself, so both fail with the same text.
USE_RAIDEN_CONNECTOR_REMOVED_MESSAGE = (
    "kv_connector_extra_config['use_raiden_connector'] has been removed; "
    "TPURaidenConnector always uses the Raiden Stage-3 backend (remove "
    "'use_raiden_connector' from kv_connector_extra_config)."
)
RAIDEN_PD_ADMISSION_REQUIRED_MESSAGE = (
    "TPURaidenConnector requires TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 and one of "
    "TPU_RAIDEN_QWEN35_ADMISSION=1, TPU_RAIDEN_KIMIK3_ADMISSION=1, "
    "TPU_RAIDEN_GLM_ADMISSION=1, or TPU_RAIDEN_DSV4_ADMISSION=1 (the legacy "
    "ZMQ transport has been removed)."
)


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
