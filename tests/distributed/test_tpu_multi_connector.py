# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TPUMultiConnector.

TPUMultiConnector adds TPU-specific method fan-out on top of upstream
MultiConnector. Children come entirely from the explicit
``kv_connector_extra_config["connectors"]`` list. The tests here run the
real upstream __init__, with fake child connector classes resolved from this
module via kv_connector_module_path (the same mechanism the recipe uses for
the real children). No TPU or network access is required.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector


class _FakeRaidenChild:
    """Stands in for TPURaidenConnector: accepts a runner."""

    def __init__(self, vllm_config, role, kv_cache_config):
        self.vllm_config = vllm_config
        self.role = role
        self.kv_cache_config = kv_cache_config
        self.runner = None

    def register_runner(self, runner):
        self.runner = runner

    def get_block_ids_with_load_errors_group_index(self):
        return 2


class _FakeOffloadChild:
    """Stands in for OffloadingConnector: no register_runner, like
    upstream connectors."""

    def __init__(self, vllm_config, role, kv_cache_config):
        self.vllm_config = vllm_config
        self.role = role
        self.kv_cache_config = kv_cache_config


def _make_vllm_config(kv_role="kv_consumer"):
    # Mirrors the disagg-offloading recipe shape (raiden child inheriting the
    # outer role + offloading child as kv_both), with the child classes
    # pointed at this module instead of the real connectors.
    return SimpleNamespace(
        kv_transfer_config=KVTransferConfig(
            kv_connector="TPUMultiConnector",
            kv_connector_module_path=(
                "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector"
            ),
            kv_role=kv_role,
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "_FakeRaidenChild",
                        "kv_connector_module_path": __name__,
                        "kv_role": kv_role,
                    },
                    {
                        "kv_connector": "_FakeOffloadChild",
                        "kv_connector_module_path": __name__,
                        "kv_role": "kv_both",
                    },
                ],
            },
        ),
        # The fake children don't implement SupportsHMA, so hybrid KV cache
        # management must be off for upstream __init__'s HMA assert.
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
    )


def _build(role=KVConnectorRole.WORKER):
    return TPUMultiConnector(_make_vllm_config(), role, MagicMock())


class TestTPUMultiConnectorConstruction:
    def test_children_built_from_explicit_connectors_config(self):
        connector = _build()

        raiden, offload = connector._connectors
        assert isinstance(raiden, _FakeRaidenChild)
        assert isinstance(offload, _FakeOffloadChild)

    def test_each_child_gets_its_own_kv_transfer_config(self):
        connector = _build()

        raiden, offload = connector._connectors
        assert raiden.vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        assert offload.vllm_config.kv_transfer_config.kv_role == "kv_both"

    def test_role_and_kv_cache_config_forwarded_to_children(self):
        kv_cache_config = MagicMock()
        connector = TPUMultiConnector(
            _make_vllm_config(), KVConnectorRole.WORKER, kv_cache_config
        )

        for child in connector._connectors:
            assert child.role == KVConnectorRole.WORKER
            assert child.kv_cache_config is kv_cache_config


class TestTPUMultiConnectorRegisterRunner:
    def test_runner_reaches_children_that_accept_it(self):
        # tpu_runner dispatches register_runner via hasattr; upstream
        # MultiConnector doesn't forward it, so without this fan-out the
        # Raiden child fails with "register_runner must be called before
        # transfer" on the first load. The offload child has no
        # register_runner and must be skipped without raising.
        connector = _build()
        runner = MagicMock()

        connector.register_runner(runner)

        raiden, _ = connector._connectors
        assert raiden.runner is runner

    def test_upstream_still_lacks_register_runner(self):
        # The override exists only because upstream MultiConnector doesn't
        # forward register_runner; if this starts failing, upstream grew
        # native support and the override should be dropped.
        assert not hasattr(MultiConnector, "register_runner")


def test_failed_load_group_index_reaches_child_connector():
    assert _build().get_block_ids_with_load_errors_group_index() == 2


def test_conflicting_group_indices_across_children_raise():
    # A single scalar can't scope errors from two groups; mis-scoping would
    # make the scheduler recompute the wrong group's blocks, so it must fail
    # loudly instead.
    connector = _build()
    conflicting = MagicMock()
    conflicting.get_block_ids_with_load_errors_group_index.return_value = 3
    connector._connectors.append(conflicting)

    with pytest.raises(RuntimeError, match="Conflicting"):
        connector.get_block_ids_with_load_errors_group_index()
