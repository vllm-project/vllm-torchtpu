# SPDX-License-Identifier: Apache-2.0
"""Native planner and real HBM transfer, with virtual workers on one TPU.

Each logical worker owns a separate native manager and materialized TPU pool.
This validates bytes and lifecycle, not distributed worker placement or speed.
"""

import dataclasses
import functools
import math
import os
import socket
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol

from .raiden_test_utils import FakeTensor
from .test_pcp_tp_dcp_connector import _manifest
from .test_pcp_tp_dcp_spans import _page
from .test_raiden_fa_layout_calibration_tpu import (
    _apply_xla_tiled_layout,
    _logical_probe,
    _require_e0_runtime,
)
from .tpu_test_utils import run_in_isolated_process


def _store(name):
    from tpu_sync.api.torch import reshard_store as rs
    from tpu_sync.api.torch.reshard_client import ReshardClient

    sockets = [socket.socket(), socket.socket()]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        ports = [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()
    store = rs.ReshardStore(
        rs.RaidenId(
            job_name="pd-tpu-ut",
            job_replica_id=name,
            data_name="kv",
            data_replica_idx=0,
        ),
        "127.0.0.1",
        *ports,
    )
    address = f"127.0.0.1:{store.reshard_service_port}"
    return store, ReshardClient(address), address


def _run_transfer(pcp, heads, dcp):
    # Production has one manager per process; this test hosts all of them.
    os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"
    torch, sync, raw_d2h, layout_api = _require_e0_runtime()
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

    tp, degree = 2, pcp * 2
    manifests = [_manifest(max(1, heads // tp)), _manifest(1)]
    for manifest in manifests:
        shape = manifest.storages[0].shape
        manifest.storages[0] = FakeTensor((384,) + shape[1:], 1)
        manifest.pools[:] = [
            dataclasses.replace(p, num_blocks=128) for p in manifest.pools
        ]
    geometry = sol.sol_fa_geometry_from_manifest(manifests[0])
    stores = [_store("producer"), _store("consumer")]
    managers, units, tensors = [[], []], [[], []], []
    for role, (store, client, _) in enumerate(stores):
        manifest = manifests[role]
        shape = manifest.storages[0].shape
        for rank in range(degree):
            logical = _logical_probe(shape)
            tensor = torch.from_numpy(logical).view(torch.float8_e4m3fn).to("tpu")
            sync.synchronize([tensor], wait=True)
            if rank == 0:
                minor, tiles, bits = layout_api.get_device_layout_if_materialized(
                    tensor
                )
                assert bits in (0, 8)
                host = torch.empty(tensor.nbytes, dtype=torch.uint8)
                raw_d2h([tensor], [host])
                np.testing.assert_array_equal(
                    host.numpy(), _apply_xla_tiled_layout(logical, minor, tiles)
                )
            tensors.append(tensor)  # Keep the registered buffers alive.
            manager = KVCacheManager(
                kv_caches=[tensor],
                local_control_port=0,
                max_blocks=128,
                num_slots=1,
                node_id=rank if role == 0 else 0,
                listener_port=0,
                parallelism=1,
                raiden_worker_port=0,
                raiden_controller_address=store.raiden_controller_address,
                worker_id=f"worker_{rank}",
            )
            manager.register_pools(manifest.pool_dicts())
            unit = SimpleNamespace(
                job_name="pd-tpu-ut",
                job_replica_id=f"{role}-rank{rank}",
                data_name="kv.fa",
                data_replica_idx=0,
            )
            client.register_work_unit(
                unit,
                shards=[manager.transfer_address],
                control_plane_rpc_address=manager.listener_address,
                pool_manifest=manifest.pools,
                layout_fingerprint="hnd128",
                page_tokens=384,
                transfer_parallelism=degree,
                transfer_rank=rank,
            )
            managers[role].append(manager)
            units[role].append(unit)

    for request_index, (tokens, skip) in enumerate(
        ((8191, 0), (16383, 0), (32767, 384 * dcp))
    ):
        request = f"request-{request_index}"
        uuid = 123 + request_index
        for rank, manager in enumerate(managers[0]):
            p, t = divmod(rank, tp)
            pages = list(range(p, math.ceil(tokens / 128), pcp))
            ids = [2 + 2 * i for i in range(math.ceil(len(pages) / 3))]
            stride = manifests[0].pools[0].block_stride_bytes
            blocks = {i: bytearray(stride) for i in ids}
            head_ids = (
                list(range(t * (heads // tp), (t + 1) * (heads // tp)))
                if heads >= tp
                else [0]
            )
            for local, page in enumerate(pages):
                b, slot = divmod(local, 3)
                size = geometry.kernel_page_bytes
                blocks[ids[b]][slot * size : (slot + 1) * size] = _page(
                    head_ids, page, geometry.slab_bytes
                )
            for block, data in blocks.items():
                manager.write_block_bytes(0, block, bytes(data))
            gdn_rank = t * pcp + p
            state = bytes([1 + gdn_rank]) * 131072 + bytes([21 + gdn_rank]) * 4608
            manager.write_block_bytes(0, 127, state + bytes(stride - len(state)))
            manager.h2d_pool_blocks(0, ids + [127]).wait()
            fa = sol.lower_fa_spans_sol(
                num_tokens=tokens,
                transfer_rank=p,
                parallelism=pcp,
                producer_tp_rank=t,
                producer_tp_size=tp,
                total_kv_heads=heads,
                dst_shards=degree,
                dst_dcp_size=dcp,
                geometry=geometry,
                interleave_tokens=128,
                page_tokens=384,
                block_ids=ids,
            )
            spans = [fa] if fa.spans else []
            for pool in manifests[0].pools[1:]:
                spans.append(
                    sol.lower_gdn_state_shard_spans_sol(
                        tag=pool.tag,
                        block_id=127,
                        transfer_rank=p,
                        parallelism=pcp,
                        producer_tp_rank=t,
                        producer_tp_size=tp,
                        dst_shards=degree,
                        regions=pool.regions,
                        physical_granule_bytes=512,
                    )
                )
            stores[0][1].register_request_blocks(
                request, uuid, units[0][rank], ids, spans
            )
        dst_ids = [3 + 2 * i for i in range(math.ceil((tokens - skip) / (384 * dcp)))]
        assert stores[0][1].coordinate_transfer(
            src_units=units[0],
            dst_units=units[1],
            req_id=request,
            uuid=uuid,
            use_block_chunks=True,
            is_sender=True,
            dst_mem_type=2,
            src_controller_address=stores[0][2],
            dst_controller_address=stores[1][2],
            dst_device_block_ids=dst_ids + [127, 127],
            dst_block_counts=[len(dst_ids), 1, 1],
            transfer_pool_tags=["fa", "gdn.conv.g0", "gdn.ssm.g0"],
            dst_skip_bytes=[skip // dcp * 512, 0, 0],
        )
        sent, received = set(), set()
        deadline = time.monotonic() + 30
        while (
            len(sent) < degree or len(received) < degree
        ) and time.monotonic() < deadline:
            for role in range(2):
                for rank, manager in enumerate(managers[role]):
                    done_send, done_recv, failed = manager.poll_stats()
                    assert not failed, (role, rank, failed)
                    if request in done_send:
                        sent.add(rank)
                    if request in done_recv:
                        received.add(rank)
            time.sleep(0.01)
        assert len(sent) == len(received) == degree, (sent, received)
        for rank, manager in enumerate(managers[1]):
            manager.d2h_pool_blocks(0, dst_ids + [127]).wait()
            data = b"".join(manager.read_block_bytes(0, block) for block in dst_ids)
            head = rank // (degree // heads)
            expected = b"".join(
                _page([head], page, geometry.slab_bytes)
                for page in range(
                    skip // 128 + rank % dcp, math.ceil(tokens / 128), dcp
                )
            )
            assert data[: len(expected)] == expected, ("FA", rank, tokens, skip)
            state = manager.read_block_bytes(0, 127)
            assert state[:131072] == bytes([rank + 1]) * 131072, ("SSM", rank)
            assert state[131072:135680] == bytes([rank + 21]) * 4608, ("CONV", rank)
        print("PD_HBM_PASS", pcp, heads, dcp, tokens, skip, flush=True)


@pytest.mark.parametrize("pcp,heads,dcp", [(4, 1, 8), (4, 4, 2), (16, 4, 8)])
def test_pcp_tp_to_dcp_native_hbm_transfer(pcp, heads, dcp):
    run_in_isolated_process(
        functools.partial(_run_transfer, pcp, heads, dcp), timeout=180
    )
