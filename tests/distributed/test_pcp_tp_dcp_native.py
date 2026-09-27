# SPDX-License-Identifier: Apache-2.0
"""Native C++ registry smoke test on CPU, without TPU tensors or DMA.

Requires an importable tpu_sync extension and its C++ runtime libraries.
Uses ephemeral loopback ports, independent of any serving controller.
"""

import socket
from types import SimpleNamespace

import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol

from .test_pcp_tp_dcp_connector import _manifest


@pytest.mark.parametrize("pcp,heads", [(4, 1), (16, 4)])
def test_native_registry_accepts_pcp_tp_dcp_spans_and_cancellation(pcp, heads):
    store_api = pytest.importorskip(
        "tpu_sync.api.torch.reshard_store", exc_type=ImportError
    )
    from tpu_sync.api.torch import reshard_client as client_api

    sockets = [socket.socket(), socket.socket()]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        ports = [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()
    store = store_api.ReshardStore(
        store_api.RaidenId(
            job_name="pd-cpu-ut",
            job_replica_id="store",
            data_name="kv",
            data_replica_idx=0,
        ),
        "127.0.0.1",
        ports[0],
        ports[1],
    )
    client = client_api.ReshardClient(f"127.0.0.1:{store.reshard_service_port}")
    degree = pcp * 2
    manifest = _manifest(max(1, heads // 2))
    geometry = sol.sol_fa_geometry_from_manifest(manifest)
    registrations = []
    for rank in range(degree):
        p, t = divmod(rank, 2)
        unit = SimpleNamespace(
            job_name="pd-cpu-ut",
            job_replica_id=f"rank{rank}",
            data_name="kv.fa",
            data_replica_idx=0,
        )
        client.register_work_unit(
            unit,
            shards=["127.0.0.1:1"],
            pool_manifest=manifest.pools,
            layout_fingerprint="pd-cpu-hnd128",
            page_tokens=384,
            transfer_parallelism=degree,
            transfer_rank=rank,
        )
        # A 127-token prefix: seven workers send no FA bytes, but all eight
        # workers own a distinct recurrent state and must register it.
        ids = [3] if p == 0 else []
        fa = sol.lower_fa_spans_sol(
            num_tokens=127,
            transfer_rank=p,
            parallelism=pcp,
            producer_tp_rank=t,
            producer_tp_size=2,
            total_kv_heads=heads,
            interleave_tokens=128,
            page_tokens=384,
            geometry=geometry,
            dst_shards=degree,
            dst_dcp_size=8,
            block_ids=ids,
        )
        spans = [fa] if fa.spans else []
        for pool in manifest.pools[1:]:
            spans.append(
                sol.lower_gdn_state_shard_spans_sol(
                    tag=pool.tag,
                    block_id=11,
                    transfer_rank=p,
                    parallelism=pcp,
                    producer_tp_rank=t,
                    producer_tp_size=2,
                    dst_shards=degree,
                    regions=pool.regions,
                    physical_granule_bytes=512,
                )
            )
        registrations.append(
            dict(
                req_id="short-prefix",
                uuid=123,
                unit=unit,
                block_ids=ids,
                pool_spans=spans,
            )
        )

    assert client.get_request_block_status([("short-prefix", 123)]) == [
        client_api.REQUEST_BLOCK_STATUS_UNKNOWN
    ]
    for registration in registrations:
        client.register_request_blocks(**registration)
    assert client.get_request_block_status([("short-prefix", 123)]) == [
        client_api.REQUEST_BLOCK_STATUS_REGISTERED
    ]
    # Idempotent replay and cancellation are exercised through actual RPCs.
    client.register_request_blocks(**registrations[0])
    assert client.cancel_request_blocks_if_unclaimed("short-prefix", 123)
    assert client.get_request_block_status([("short-prefix", 123)]) == [
        client_api.REQUEST_BLOCK_STATUS_CANCELLED
    ]
    with pytest.raises((RuntimeError, ValueError), match="[Cc]ancell"):
        client.register_request_blocks(**registrations[-1])
