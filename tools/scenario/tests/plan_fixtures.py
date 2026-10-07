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
"""Plans built from the conftest scenarios and variants of them."""

import copy
from collections.abc import Callable
from typing import Any

from scenario.model.rules import check_scenario
from scenario.model.scenario import Scenario
from scenario.plan.build import build_plan
from scenario.plan.types import Plan

Data = dict[str, Any]


def scenario(data: Data) -> Scenario:
    model = Scenario.model_validate(data)
    errors = [issue for issue in check_scenario(model) if issue.severity == "error"]
    assert errors == [], errors
    return model


def build(data: Data, **kwargs: Any) -> Plan:
    return build_plan(scenario(data), ref=data["name"], **kwargs)


def mp(single: Data) -> Data:
    """GLM over vLLM's own multi-node DP, with remote code (P-04, P-05, P-19)."""
    data = copy.deepcopy(single)
    data["name"] = "glm-mp"
    data["server"]["multihost"] = "mp"
    data["server"]["args"]["trust-remote-code"] = True
    return data


def single_host(single: Data) -> Data:
    data = copy.deepcopy(single)
    data["name"] = "glm-one-host"
    data["server"]["slice"] = "tpu7x/2x2x1"
    del data["server"]["multihost"]
    data["server"]["args"]["data-parallel-size"] = 8
    return data


def k3_pd(pd: Data) -> Data:
    """Two four-host mp roles, the router and tasks on the prefill head (P15)."""
    data = copy.deepcopy(pd)
    data["name"] = "k3-pd"
    server = data["server"]
    server["multihost"] = "mp"
    server["scratch"] = "64Gi"
    server["args"] = {
        "max-model-len": 10240,
        "tensor-parallel-size": 8,
        "data-parallel-size": 4,
        "api-server-count": 1,
        "enable-prefix-caching": True,
        "compilation-config": {"compile_sizes": [1, 8, 1296]},
        "reasoning-parser": "kimi_k3",
    }
    for role in server["roles"].values():
        role["slice"] = "tpu7x/2x2x4"
        role["args"] = {}
    data["tasks"] = {
        "driver": "prefill",
        "custom": copy.deepcopy(pd["tasks"]["custom"]),
    }
    return data


def agentx_point(agentx: Data) -> Data:
    data = copy.deepcopy(agentx)
    del data["sweep"]
    data["server"]["args"]["max-num-seqs"] = 32
    return data


VARIANTS: dict[str, Callable[[dict[str, Data]], Data]] = {
    "single": lambda f: f["single"],
    "single_host": lambda f: single_host(f["single"]),
    "mp": lambda f: mp(f["single"]),
    "pd": lambda f: f["pd"],
    "k3_pd": lambda f: k3_pd(f["pd"]),
    "agentx": lambda f: agentx_point(f["agentx"]),
}
