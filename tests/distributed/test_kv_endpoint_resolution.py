# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the P/D KV endpoint advertise/resolve pair:
"""

import os
import unittest
from unittest.mock import patch

from vllm_torchtpu.distributed import utils as dist_utils
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import LoadMeta
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnectorWorker


def _load_meta(remote_host, remote_port) -> LoadMeta:
    return LoadMeta(
        uuid=1,
        local_block_ids=[0],
        remote_block_ids=[0],
        remote_host=remote_host,
        remote_port=remote_port,
    )


class _ResolverHarness:
    _resolve_remote_endpoint = TPURaidenConnectorWorker._resolve_remote_endpoint
    _rank_control_port = TPURaidenConnectorWorker._rank_control_port

    def __init__(self, *, node_id: int, tp_rank: int, tp_size: int):
        self.node_id = node_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size


class TestGetKvIpsPorts(unittest.TestCase):
    """Producer side: what address list an engine advertises."""

    def setUp(self):
        dist_utils._NODES_KV_IP_PORT.clear()
        self._env = patch.dict(os.environ, {"TPU_MULTIHOST_BACKEND": "ray"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        dist_utils._NODES_KV_IP_PORT.clear()

    def test_single_worker_engine_on_nonzero_node(self):
        # A ray executor-DP engine whose only worker landed on node 1
        # registers exactly one entry, keyed by that node id.
        dist_utils.set_node_kv_ip_port((1, "10.0.0.2", 9100))
        self.assertEqual(dist_utils.get_kv_ips(), ["10.0.0.2"])
        self.assertEqual(dist_utils.get_kv_ports(), [9100])

    def test_multihost_engine_dense_nodes(self):
        # A multihost-TP engine registers every node it spans; output is
        # positional (index == node id), same as the pre-fix behavior.
        dist_utils.set_node_kv_ip_port((0, "10.0.0.1", 9100))
        dist_utils.set_node_kv_ip_port((1, "10.0.0.2", 9200))
        self.assertEqual(dist_utils.get_kv_ips(), ["10.0.0.1", "10.0.0.2"])
        self.assertEqual(dist_utils.get_kv_ports(), [9100, 9200])

    def test_mp_backend(self):
        # mp takes the scalar branch: host ip string and the env-configured
        # transfer port, registry ignored.
        with patch.dict(
            os.environ, {"TPU_MULTIHOST_BACKEND": "mp", "TPU_KV_TRANSFER_PORT": "9345"}
        ):
            self.assertIsInstance(dist_utils.get_kv_ips(), str)
            self.assertEqual(dist_utils.get_kv_ports(), "9345")


class TestResolveRemoteEndpoint(unittest.TestCase):
    """Consumer side: which producer endpoint a worker dials."""

    def test_single_endpoint_dialed_from_any_node(self):
        # A single-node (ray executor-DP) producer advertises one endpoint.
        worker = _ResolverHarness(node_id=1, tp_rank=0, tp_size=1)
        meta = _load_meta(["10.0.0.2"], [9100])
        self.assertEqual(worker._resolve_remote_endpoint(meta), "10.0.0.2:9100")

    def test_multihost_producer_indexed_by_consumer_node(self):
        # Multihost-TP producer (2 nodes): consumer node i pulls from
        # producer node i.
        worker = _ResolverHarness(node_id=1, tp_rank=9, tp_size=16)
        meta = _load_meta(["10.0.0.1", "10.0.0.2"], [9100, 9200])
        self.assertEqual(worker._resolve_remote_endpoint(meta), "10.0.0.2:9202")

    def test_scalar_host_mp(self):
        worker = _ResolverHarness(node_id=1, tp_rank=2, tp_size=8)
        meta = _load_meta("10.0.0.9", 9100)
        self.assertEqual(worker._resolve_remote_endpoint(meta), "10.0.0.9:9104")


if __name__ == "__main__":
    unittest.main()
