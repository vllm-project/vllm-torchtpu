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
"""Independent native P/D processes; synthetic page bytes, no model decode."""

import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.tpu_test, pytest.mark.spawns_tpu_workers]


def _publish(path: Path, value: dict) -> None:
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value))
    pending.replace(path)


def _receive(path: Path, seconds: float = 90) -> dict:
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Peer did not publish {path.name}")
        time.sleep(0.05)
    return json.loads(path.read_text())


def _run_role(role: str, exchange: Path) -> None:
    import numpy as np
    import torch
    from torch_tpu._internal import sync
    from torch_tpu._internal.batch_transfer import batch_transfer_d2h_sync
    from torch_tpu._internal.compile import tpu_torch_compile
    from tpu_sync.api.common import RaidenId
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager
    from tpu_sync.api.torch.reshard_client import ReshardClient
    from tpu_sync.api.torch.reshard_store import RaidenId as StoreId
    from tpu_sync.api.torch.reshard_store import ReshardStore
    from tpu_sync.rpc.raiden_controller import RaidenMemoryType

    from tests.distributed.test_raiden_fa_layout_calibration_tpu import (
        _apply_xla_tiled_layout,
        _logical_probe,
    )
    from tests.runner.test_qwen4_exp_pd_manifest import live_pool
    from vllm_torchtpu.distributed.kv_transfer.connector_metadata import SendMeta
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import _Stage3LoadMeta
    from vllm_torchtpu.models.vllm.qwen4_exp.pd import (
        TransferExtent,
        build_transfer_groups,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import (
        Qwen4ExpRaidenWorker,
        QwenPagePlan,
    )
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_manifest import (
        build_transfer_manifest,
        measure_transfer_layout,
    )

    def read_raw(raw):
        host = torch.empty(raw.nbytes, dtype=torch.uint8)
        batch_transfer_d2h_sync([raw], [host])
        return host.numpy()

    ssm_dtype = _receive(exchange / "test-config.json")["ssm_dtype"]
    capacity = 128 if role == "prefill" else 160
    config, materialized = live_pool(device="tpu", num_blocks=capacity, ssm=ssm_dtype)
    manifest = build_transfer_manifest(config, materialized)
    source_bytes, before = [], []
    for raw in manifest.storages:
        # Byte patterns are viewed, never numerically cast (including BF16 NaNs).
        logical = (
            _logical_probe((raw.nbytes,))
            .view(np.uint8 if raw.element_size() == 1 else np.uint16)
            .reshape(raw.shape)
        )
        initial = logical if role == "prefill" else np.full_like(logical, 3)
        raw.copy_(
            torch.from_numpy(initial.view(np.uint8).reshape(-1))
            .view(raw.dtype)
            .reshape(raw.shape)
            .to("tpu")
        )
        sync.synchronize([raw], wait=True)
        order, tiles, _ = tpu_torch_compile.get_device_layout_if_materialized(raw)
        expected = _apply_xla_tiled_layout(logical, order, tiles).view(np.uint8)
        source_bytes.append(expected)
        observed = read_raw(raw)
        np.testing.assert_array_equal(
            observed,
            expected if role == "prefill" else initial.reshape(-1).view(np.uint8),
        )
        before.append(observed)
    fingerprint, layout = measure_transfer_layout(manifest)
    unit = RaidenId(role, "qwen-page-test", "worker_0", 0)
    store = ReshardStore(
        StoreId(role, "qwen-page-test", "reshard_store", 0), "127.0.0.1"
    )
    address = f"127.0.0.1:{store.reshard_service_port}"
    manager = KVCacheManager(
        kv_caches=manifest.storages,
        local_control_port=0,
        max_blocks=config.num_blocks,
        num_slots=1,
        timeout_s=60,
        parallelism=1,
        listener_port=0,
        raiden_controller_address=store.raiden_controller_address,
        worker_id="worker_0",
    )
    manager.register_pools(manifest.pool_dicts())
    client = ReshardClient(address)
    groups = build_transfer_groups(config.kv_cache_groups)
    page_tokens = {
        g.block_size for g in groups if g.extent is TransferExtent.TOKEN_PAGES
    }
    assert len(page_tokens) == 1
    page_tokens = page_tokens.pop()
    client.register_work_unit(
        unit=unit,
        shards=[manager.transfer_address],
        control_plane_rpc_address=manager.listener_address,
        pool_manifest=manifest.pool_dicts(),
        layout_fingerprint=fingerprint,
        page_tokens=page_tokens,
        transfer_parallelism=1,
        transfer_rank=0,
    )
    # Exercise production page lowering, without imitating loaded-weight admission.
    worker = object.__new__(Qwen4ExpRaidenWorker)
    worker.tp_rank = 0
    worker.tp_size = 1
    worker._qwen_page_plan = QwenPagePlan(groups, manifest)
    num_tokens = 129
    source_tables, dest_tables = [], []
    cursor = 1
    for group in worker._qwen_page_plan.groups:
        count = group.page_count(num_tokens)
        pages = list(range(cursor, cursor + count))
        source_tables.append(pages)
        dest_tables.append([128 - page for page in pages])
        cursor += count
    assert cursor <= config.num_blocks
    load = _Stage3LoadMeta(
        uuid=1,
        source_req_id="qwen-pages",
        num_tokens=num_tokens,
        src_controller_address="",
        src_job_name="prefill",
        src_engine_id="qwen-page-test",
        src_data_replica_idx=0,
        local_block_ids=dest_tables[0],
        group_block_ids=dest_tables,
        skip_tokens=0,
        src_parallelism=1,
    )
    tags, blocks, counts = worker._stage3_grouped_destination_pages(load, num_tokens)
    assert len(tags) == len(manifest.pools) == len(materialized.kv_caches)
    if role == "prefill":
        send = SendMeta(
            uuid=1,
            expiration_time=time.time() + 180,
            num_tokens=num_tokens,
            local_block_ids=source_tables[0],
            group_block_ids=source_tables,
        )
        pages, spans = worker._stage3_grouped_spans(send, num_tokens)
        assert len(spans) == len(tags)
        client.register_request_blocks(
            req_id="qwen-pages",
            uuid=1,
            unit=unit,
            block_ids=list(pages[0]),
            pool_spans=spans,
        )
    _publish(
        exchange / f"{role}-ready.json",
        dict(
            pid=os.getpid(),
            runtime_rank=int(os.environ["RANK"]),
            runtime_topology=os.environ["TORCH_TPU_TOPOLOGY"],
            address=address,
            fingerprint=fingerprint,
            source_sha256=[
                hashlib.sha256(x[: 128 * (x.size // capacity)]).hexdigest()
                for x in source_bytes
            ],
            layout=layout,
            page_tokens=page_tokens,
            tags=tags,
            blocks=blocks,
            counts=counts,
        ),
    )
    peer_role = "decode" if role == "prefill" else "prefill"
    peer = _receive(exchange / f"{peer_role}-ready.json")
    assert peer["pid"] != os.getpid()
    assert peer["runtime_rank"] != int(os.environ["RANK"])
    assert peer["runtime_topology"] == os.environ["TORCH_TPU_TOPOLOGY"]
    assert peer["fingerprint"] == fingerprint
    assert peer["page_tokens"] == page_tokens
    assert peer["source_sha256"] == [
        hashlib.sha256(x[: 128 * (x.size // capacity)]).hexdigest()
        for x in source_bytes
    ]
    assert (
        peer["tags"] == tags and peer["blocks"] == blocks and peer["counts"] == counts
    )
    if role == "prefill":
        assert (
            client.start_transfer(
                src_units=[unit],
                dst_units=[RaidenId("decode", "qwen-page-test", "worker_0", 0)],
                req_id="qwen-pages",
                uuid=1,
                is_sender=True,
                use_block_chunks=True,
                src_controller_address=address,
                dst_controller_address=peer["address"],
                dst_mem_type=RaidenMemoryType.HBM,
                dst_device_block_ids=blocks,
                num_tokens=num_tokens,
                transfer_pool_tags=tags,
                dst_block_counts=counts,
            )
            is True
        )
    deadline = time.monotonic() + 60
    while True:
        sent, received, failed = manager.poll_stats()
        assert not failed, failed
        if "qwen-pages" in (sent if role == "prefill" else received):
            break
        assert time.monotonic() < deadline, (role, sent, received)
        time.sleep(0.01)
    expected = [x.copy() for x in before]
    moved = 0
    for pool, group_index in worker._qwen_page_plan.pool_groups():
        stride = pool.block_stride_bytes
        for src_page, dst_page in zip(
            source_tables[group_index], dest_tables[group_index]
        ):
            if role == "decode":
                expected[pool.storage_index][
                    dst_page * stride : (dst_page + 1) * stride
                ] = source_bytes[pool.storage_index][
                    src_page * stride : (src_page + 1) * stride
                ]
            moved += stride
    after = [read_raw(raw) for raw in manifest.storages]
    for observed, want in zip(after, expected):
        np.testing.assert_array_equal(observed, want)
    if role == "prefill":
        client.complete_request_blocks(req_id="qwen-pages", uuid=1, unit=unit)
    _publish(
        exchange / f"{role}-done.json",
        dict(
            pid=os.getpid(),
            owners=len(tags),
            pages=cursor - 1,
            ssm_dtype=ssm_dtype,
            transferred_bytes=moved,
            compared_bytes=sum(x.size for x in after),
            capacity=capacity,
            live_carriers=len(after),
            raw_sha256=[hashlib.sha256(x).hexdigest() for x in after],
            fingerprint=fingerprint,
        ),
    )
    _receive(exchange / f"{peer_role}-done.json")


@pytest.mark.parametrize("ssm_dtype", ["bfloat16", "float32"])
def test_qwen_all_owner_pages_over_native_reshard(tmp_path, ssm_dtype):
    from tests.distributed.tpu_test_utils import prepare_tpu_environment

    _publish(tmp_path / "test-config.json", dict(ssm_dtype=ssm_dtype))
    roles = ("prefill", "decode")
    with patch.dict(os.environ):
        for key in (
            "TORCH_TPU_SLICEBUILDER_ADDRESSES",
            "TORCH_TPU_TOPOLOGY",
            "TORCH_TPU_XPROF_SESSION_ID",
            "MASTER_ADDR",
            "MASTER_PORT",
            "JOB_COMPLETION_INDEX",
            "TPU_VISIBLE_CHIPS",
            "TPU_VISIBLE_DEVICES",
        ):
            os.environ.pop(key, None)
        # Initialize the full native slice, including its keep-alive ranks.
        prepare_tpu_environment(world_size=8)
        launch_env = os.environ.copy()
    launch_env.update(
        ALLOW_MULTIPLE_LIBTPU_LOAD="1", TPU_RAIDEN_TRANSFER_PARALLELISM="1"
    )
    root = Path(__file__).resolve().parents[2]
    launch_env["PYTHONPATH"] = str(root) + os.pathsep + launch_env.get("PYTHONPATH", "")
    log_path = tmp_path / "torchrun.log"
    with log_path.open("w") as log:
        child = subprocess.Popen(
            _launcher_command(tmp_path),
            env=launch_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 240
            while child.poll() is None:
                assert time.monotonic() < deadline, "Native P/D process deadline"
                assert log_path.stat().st_size <= 32 * 1024**2, "Launcher log limit"
                time.sleep(0.1)
            assert child.wait() == 0
            records = [_receive(tmp_path / f"{role}-done.json") for role in roles]
            assert all(record["ssm_dtype"] == ssm_dtype for record in records)
            print("QWEN_NATIVE_PAGE_TRANSFER " + json.dumps(records))
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            print("TORCHRUN LOG TAIL\n" + log_path.read_text()[-32000:])


def _launcher_command(exchange: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        "--max-restarts=0",
        str(Path(__file__).resolve()),
        str(exchange),
    ]


def _run_runtime_worker(rank: int, exchange: Path) -> None:
    import torch
    from torch import distributed as dist
    from vllm.plugins import load_general_plugins

    stat = Path(f"/proc/{os.getpid()}/stat").read_text()
    _publish(
        exchange / f"process-{rank}.json",
        dict(
            pid=os.getpid(),
            pgid=os.getpgrp(),
            birth_ticks=stat[stat.rfind(")") + 2 :].split()[19],
        ),
    )
    # Runtime participants need the same plugin bootstrap as the P/D workers.
    load_general_plugins()
    dist.init_process_group(backend="tpu_dist", timeout=datetime.timedelta(seconds=120))
    try:
        torch.empty(1, device="tpu").cpu()
        import jax

        devices = jax.local_devices(backend="tpu")
        _publish(
            exchange / f"hardware-{rank}.json",
            dict(
                devices=[
                    dict(
                        id=d.id,
                        kind=d.device_kind,
                        coords=list(d.coords),
                        core_on_chip=d.core_on_chip,
                    )
                    for d in devices
                ]
            ),
        )
        _publish(exchange / f"runtime-{rank}.json", dict(rank=rank, pid=os.getpid()))
        # Keep every native runtime participant alive until both roles finish.
        for peer in range(8):
            _receive(exchange / f"runtime-{peer}.json")
        if rank < 2:
            _run_role(("prefill", "decode")[rank], exchange)
        for role in ("prefill", "decode"):
            _receive(exchange / f"{role}-done.json", seconds=150)
        _publish(
            exchange / f"sources-{rank}.json",
            dict(
                modules=[
                    dict(
                        module=name,
                        path=module.__file__,
                        sha256=hashlib.sha256(
                            Path(module.__file__).read_bytes()
                        ).hexdigest(),
                    )
                    for name, module in list(sys.modules.items())
                    if name.startswith(("vllm_torchtpu", "tpu_sync"))
                    and getattr(module, "__file__", None)
                    and Path(module.__file__).is_file()
                ]
            ),
        )
        _publish(exchange / f"finished-{rank}.json", dict(rank=rank))
        for peer in range(8):
            _receive(exchange / f"finished-{peer}.json")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8 and rank in range(8)
    exchange = Path(sys.argv[1])
    if sys.argv[2:] == ["--check-launch"]:
        _publish(
            exchange / f"rank{rank}-launch.json",
            dict(
                role=("prefill", "decode")[rank] if rank < 2 else "runtime",
                rank=rank,
                pid=os.getpid(),
                local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
            ),
        )
    else:
        assert len(sys.argv) == 2
        _run_runtime_worker(rank, exchange)
