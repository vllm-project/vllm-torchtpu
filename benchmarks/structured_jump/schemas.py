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
"""Workload schemas and regular expressions for structured jump decoding benchmark.

Ported from SGLang benchmark/json_jump_forward (Apache-2.0) with whitespace
strictly fixed to support zero-branch deterministic jump paths.
"""

from __future__ import annotations

from typing import Any

# Character generation regex from SGLang: 10 fixed fields, decode-heavy workload.
# Whitespace and indentation are deterministic.
CHARACTER_REGEX = (
    r"""\{\n""" + r"""    "name": "[\w\d\s]{1,16}",\n""" +
    r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n""" +
    r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n""" +
    r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n""" + r"""        "wood": "[\w\d\s]{1,16}",\n""" +
    r"""        "core": "[\w\d\s]{1,16}",\n""" +
    r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n""" + r"""    \},\n""" +
    r"""    "alive": "(Alive|Deceased)",\n""" +
    r"""    "patronus": "[\w\d\s]{1,16}",\n""" +
    r"""    "bogart": "[\w\d\s]{1,16}"\n""" + r"""\}""")

# City information extraction regex from SGLang: 5 fields, long document prefill-heavy workload.
CITY_REGEX = (
    r"""\{\n""" + r"""  "name": "[\w\d\s]{1,16}",\n""" +
    r"""  "country": "[\w\d\s]{1,16}",\n""" +
    r"""  "latitude": [-+]?[0-9]*\.?[0-9]{0,2},\n""" +
    r"""  "population": [-+]?[0-9]{1,9},\n""" +
    r"""  "top 3 landmarks": \["[\w\d\s]{1,16}", "[\w\d\s]{1,16}", "[\w\d\s]{1,16}"\]\n"""
    + r"""\}""")

# Telemetry report schema from vllm-torchtpu test suite: high-density enum fields.
TELEMETRY_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "CloudServiceTelemetry",
    "type": "object",
    "properties": {
        "api_version": {
            "type": "string",
            "enum": ["v2.1"],
        },
        "service_name": {
            "type": "string",
            "enum": ["inference-gateway"],
        },
        "status": {
            "type": "string",
            "enum": ["HEALTHY", "DEGRADED", "UNHEALTHY"],
        },
        "metrics": {
            "type": "object",
            "properties": {
                "cluster_region": {
                    "type": "string",
                    "enum": ["us-central1", "us-east4", "europe-west4"],
                },
                "active_instances": {
                    "type": "integer",
                    "enum": [8, 16, 32, 64],
                },
                "is_healthy": {
                    "type": "boolean",
                },
            },
            "required": ["cluster_region", "active_instances", "is_healthy"],
        },
    },
    "required": ["api_version", "service_name", "status", "metrics"],
}
