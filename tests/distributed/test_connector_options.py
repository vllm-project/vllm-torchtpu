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
"""The connector options reader: names, defaults, types, misplaced options."""

import pytest

from vllm_torchtpu.distributed.kv_transfer.connector_options import (
    CONNECTOR_OPTIONS,
    MultiConnectorOptions,
    RaidenOffloadingOptions,
    RaidenTransferOptions,
    option_names,
    read_options,
)


@pytest.mark.parametrize("options_cls", list(CONNECTOR_OPTIONS.values()))
@pytest.mark.parametrize("extra_config", [None, {}, {"cpu_bytes_to_use": 8}])
def test_unset_options_are_the_defaults(options_cls, extra_config):
    assert read_options(options_cls, extra_config) == options_cls()


def test_set_options_are_read():
    assert read_options(
        RaidenTransferOptions, {"async_complete": True, "late_poll": True}
    ) == RaidenTransferOptions(async_complete=True, late_poll=True)
    assert read_options(
        RaidenOffloadingOptions,
        {"host_tier_recency": True, "max_async_loads": 16, "cpu_bytes_to_use": 1},
    ) == RaidenOffloadingOptions(host_tier_recency=True, max_async_loads=16)
    assert read_options(
        MultiConnectorOptions, {"producer_divergent_hits": True, "connectors": []}
    ) == MultiConnectorOptions(producer_divergent_hits=True)


@pytest.mark.parametrize("value", ["1", 1, "true", None, 0])
def test_a_bool_option_must_be_true_or_false(value):
    with pytest.raises(ValueError, match=r"'late_poll'\] must be true or false"):
        read_options(RaidenTransferOptions, {"late_poll": value})


@pytest.mark.parametrize("value", ["16", True, -1, 1.5, None])
def test_an_int_option_must_be_a_non_negative_integer(value):
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        read_options(RaidenOffloadingOptions, {"max_async_loads": value})


@pytest.mark.parametrize(
    ("options_cls", "key", "owner"),
    [
        (RaidenOffloadingOptions, "late_poll", "TPURaidenConnector"),
        (RaidenTransferOptions, "host_tier_recency", "TPURaidenOffloadingConnector"),
        (RaidenTransferOptions, "producer_divergent_hits", "TPUMultiConnector"),
        (MultiConnectorOptions, "max_async_loads", "TPURaidenOffloadingConnector"),
    ],
)
def test_another_connectors_option_is_refused(options_cls, key, owner):
    with pytest.raises(ValueError, match=f"options of {owner}"):
        read_options(options_cls, {key: True})


def test_no_option_name_belongs_to_two_connectors():
    names = [name for cls in CONNECTOR_OPTIONS.values() for name in option_names(cls)]
    assert len(names) == len(set(names))
