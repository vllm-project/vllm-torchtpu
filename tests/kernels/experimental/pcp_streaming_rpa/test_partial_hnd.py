# SPDX-License-Identifier: Apache-2.0
"""Regression: HND partial PCP prefill/decode must match causal attention.

Eight TorchTPU processes, no model weights, GDN or transfer. The reference
also checks exact cache contents, including untouched sentinel values.
"""

import faulthandler
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]


def _runtime_mesh_and_partition(torch, dist):
    import jax
    import numpy as np
    from torch_tpu._internal.distributed import tpu_distributed

    device_id = int(tpu_distributed.global_device_id())
    local = torch.tensor([device_id], dtype=torch.int32, device="tpu")
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    device_ids = sorted(int(value.cpu().item()) for value in gathered)
    if len(set(device_ids)) != 8:
        raise RuntimeError(f"Expected eight distinct TPU devices: {device_ids}")
    devices = {int(device.id): device for device in jax.devices()}
    if set(device_ids) != set(devices):
        raise RuntimeError(
            "Worker devices must cover the JAX runtime: "
            f"workers={device_ids}, JAX={sorted(devices)}"
        )
    # Main's TorchTPU assigns custom-op partitions in device-id order, not
    # process-rank order. Index inputs and the oracle in that same space.
    mesh = jax.sharding.Mesh(
        np.asarray([devices[i] for i in device_ids]), axis_names=("pcp",)
    )
    return mesh, device_ids.index(device_id)


def _worker(result_dir):
    out = Path(result_dir)
    rank = int(os.environ["RANK"])
    rows = []
    try:
        import torch
        import torch.distributed as dist
        import torch_tpu  # noqa: F401
        from torch_tpu._internal import sync

        from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
            PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
            make_pcp_streaming_rpa_kernel,
            pcp_streaming_jax_op,
        )

        dist.init_process_group("tpu_dist")
        torch.empty((1,), device="tpu").cpu()
        print("BOOTED", rank, flush=True)
        mesh, partition = _runtime_mesh_and_partition(torch, dist)
        print("PARTITION", rank, partition, flush=True)
        page = 128
        local = 128
        hd = 128
        ops = {}
        for layout in [KVLayout.HEAD_ALONG_SUBLANE, KVLayout.SEQ_ALONG_LANE]:
            entry = make_pcp_streaming_rpa_kernel(
                q_scale=None,
                k_scale=1.0,
                v_scale=1.0,
                mesh=mesh,
                sliding_window=None,
                sm_scale=1.0,
                skip_kv_update=False,
                cp_kv_cache_interleave_size=128,
                q_block_size=128,
                q_compute_size=64,
                kv_layout=layout,
            )
            ops[layout] = pcp_streaming_jax_op(
                "pcp_partial::" + layout.name,
                entry,
                donate_argnums=(0,),
                mesh=mesh,
                input_partition_specs=PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
            )
        for length in [128, 53, 1, 129, 1024]:
            for layout, op in ops.items():
                shape = (
                    (2, 4, 32, 4, page)
                    if layout == KVLayout.SEQ_ALONG_LANE
                    else (2, page, 1, 4, hd)
                )
                # Sentinel detects writes outside logical token ranges.
                cache = torch.full(shape, -2.0, dtype=torch.float8_e4m3fn, device="tpu")
                history = []
                for step, n in enumerate([length, 1, 1]):
                    start = len(history)
                    values = [float(1 + (start + i) // 128 % 8) for i in range(n)]
                    history.extend(values)
                    q = torch.zeros((local, 2, hd), dtype=torch.bfloat16, device="tpu")
                    k = torch.zeros_like(q)
                    v_cpu = torch.zeros((local, 2, hd), dtype=torch.bfloat16)
                    mine = max(0, min(local, n - partition * local))
                    if mine:
                        v_cpu[:mine] = torch.tensor(
                            values[partition * local : partition * local + mine],
                            dtype=torch.bfloat16,
                        )[:, None, None].expand(mine, 2, hd)
                    v = v_cpu.to("tpu")
                    seq = torch.tensor([len(history)], dtype=torch.int32, device="tpu")
                    blocks = torch.tensor([0, 1], dtype=torch.int32, device="tpu")
                    cu = torch.tensor([0, n], dtype=torch.int32, device="tpu")
                    distribution = torch.tensor(
                        [0, 0, 1], dtype=torch.int32, device="tpu"
                    )
                    sync.synchronize(
                        [cache, q, k, v, seq, blocks, cu, distribution], wait=True
                    )
                    print("CALL", rank, length, step, layout.name, flush=True)
                    cache, result = op(cache, q, k, v, seq, blocks, cu, distribution)
                    sync.synchronize([cache, result], wait=True)
                    observed = result.cpu().float()
                    expected = torch.zeros_like(observed)
                    for i in range(mine):
                        pos = start + partition * local + i
                        expected[i] = sum(history[: pos + 1]) / (pos + 1)
                    finite = bool(torch.isfinite(observed).all())
                    err = float((observed - expected).abs().max())
                    raw = cache.cpu().float()
                    logical = (
                        raw.reshape(2, 4, hd, page).permute(0, 3, 1, 2)
                        if layout == KVLayout.SEQ_ALONG_LANE
                        else raw.reshape(2, page, 4, hd)
                    )
                    expected_cache = torch.full_like(logical, -2.0)
                    for pos, value in enumerate(history):
                        if (pos // 128) % 8 != partition:
                            continue
                        block = pos // 1024
                        offset = pos % 128
                        expected_cache[block, offset, 0::2] = 0.0
                        expected_cache[block, offset, 1::2] = value
                    cache_finite = bool(torch.isfinite(logical).all())
                    cache_err = float((logical - expected_cache).abs().max())
                    row = dict(
                        rank=rank,
                        partition=partition,
                        prefill_length=length,
                        step=step,
                        layout=layout.name,
                        active_local_tokens=mine,
                        finite=finite,
                        output_max_abs=err,
                        cache_finite=cache_finite,
                        cache_max_abs=cache_err,
                        pass_=finite and err < 0.04 and cache_finite and cache_err == 0,
                    )
                    rows.append(row)
                    (out / f"rank-{rank}.json").write_text(json.dumps(rows, indent=2))
                    print(json.dumps(row), flush=True)
        _extended_cases(out, rank, partition, mesh, rows)
        dist.barrier()
        dist.destroy_process_group()
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        (out / f"rank-{rank}-error.txt").write_text(traceback.format_exc())
        os._exit(1)


def _extended_cases(out, rank, partition, mesh, rows):
    """Exercise real page sizes, mixed requests and cache-owner boundaries."""
    import torch
    from torch_tpu._internal import sync

    from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
        PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
        make_pcp_streaming_rpa_kernel,
        pcp_streaming_jax_op,
    )

    cases = [
        (128, 128, 128, [0, 0, 0], [[53, 1, 129], [1, 3, 2], [2, 1, 1]]),
        (2304, 256, 512, [0], [[2049], [1], [255]]),
        (2304, 256, 512, [18431], [[3], [129], [1]]),
        (2304, 256, 512, [53, 255, 2303], [[1, 3, 2], [129, 1, 255], [1, 2, 1]]),
    ]
    cases.extend(
        [
            (2304, 256, 512, [0], [[257], [1], [1]]),
            (2304, 256, 512, [11868], [[129], [1], [1]]),
        ]
    )
    hd, heads = 128, 2

    def values(seq, positions):
        # All FP8-exact; vary tokens, heads and dimensions to detect wrong
        # source slices and bit-packing, not only missing writes.
        return (
            (
                positions[:, None, None]
                + seq * 3
                + torch.arange(heads)[None, :, None] * 2
                + torch.arange(hd)[None, None, :] % 5
            )
            % 8
        ).float()

    def keys(seq, positions):
        return (
            (
                positions[:, None, None] * 3
                + seq * 2
                + torch.arange(heads)[None, :, None] * 7
                + torch.arange(hd)[None, None, :] * 5
            )
            % 17
            - 8
        ).float() / 8

    def queries(seq, positions):
        return (
            (
                positions[:, None, None] * 5
                + seq
                + torch.arange(heads)[None, :, None] * 3
                + torch.arange(hd)[None, None, :] * 7
            )
            % 13
            - 6
        ).float() / 8

    def attention_reference(seq, length, added):
        positions = torch.arange(length + added)
        q = queries(seq, positions[length:]).transpose(0, 1)
        k = keys(seq, positions).transpose(0, 1)
        v = values(seq, positions).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(1, 2)) * hd**-0.5
        mask = positions[None, :] > positions[length:, None]
        scores.masked_fill_(mask[None, :, :], float("-inf"))
        return torch.matmul(scores.softmax(-1), v).transpose(0, 1)

    for case_id, (page, interleave, local, initial, steps) in enumerate(cases):
        nonzero_qk = case_id >= 4
        nseq = len(initial)
        blocks_per_seq = 2
        nblocks = nseq * blocks_per_seq
        # Non-identity block tables catch accidental logical/physical mixing.
        table = torch.arange(nblocks).reshape(nseq, blocks_per_seq).flip(1)

        def logical_cache(
            lengths,
            *,
            case_id=case_id,
            nblocks=nblocks,
            page=page,
            interleave=interleave,
            table=table,
            nonzero_qk=nonzero_qk,
        ):
            sentinel = float("nan") if case_id == 3 else -2.0
            logical = torch.full((nblocks, page, heads * 2, hd), sentinel)
            for seq, length in enumerate(lengths):
                pos = torch.arange(length)
                mine = (pos // interleave) % 8 == partition
                owned = pos[mine]
                offsets = (
                    (owned // (8 * interleave)) * interleave + owned % interleave
                ) % page
                physical = table[seq, owned // (8 * page)]
                logical[physical, offsets, 0::2, :] = (
                    keys(seq, owned) if nonzero_qk else 0.0
                )
                logical[physical, offsets, 1::2, :] = values(seq, owned)
            return logical

        for layout in [KVLayout.HEAD_ALONG_SUBLANE, KVLayout.SEQ_ALONG_LANE]:
            entry = make_pcp_streaming_rpa_kernel(
                q_scale=None,
                k_scale=1.0,
                v_scale=1.0,
                mesh=mesh,
                sliding_window=None,
                sm_scale=hd**-0.5 if nonzero_qk else 1.0,
                skip_kv_update=False,
                cp_kv_cache_interleave_size=interleave,
                q_block_size=interleave,
                q_compute_size=64,
                kv_layout=layout,
            )
            op = pcp_streaming_jax_op(
                f"pcp_partial_extended::case{case_id}_{layout.name}",
                entry,
                donate_argnums=(0,),
                mesh=mesh,
                input_partition_specs=PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
            )
            lengths = list(initial)
            logical = logical_cache(lengths)
            if layout == KVLayout.SEQ_ALONG_LANE:
                packed = logical.reshape(nblocks, page, heads * 2, 32, 4)
                packed = packed.permute(0, 2, 3, 4, 1).contiguous()
            else:
                packed = logical.reshape(nblocks, page, 1, 4, hd)
            cache = packed.to(torch.float8_e4m3fn).to("tpu")
            for step, added in enumerate(steps):
                new_values = torch.cat(
                    [
                        values(seq, torch.arange(length, length + n))
                        for seq, (length, n) in enumerate(zip(lengths, added))
                    ]
                )
                targets = torch.cat(
                    [
                        values(seq, torch.arange(length + n)).cumsum(0)[length:]
                        / torch.arange(length + 1, length + n + 1)[:, None, None]
                        for seq, (length, n) in enumerate(zip(lengths, added))
                    ]
                )
                new_keys = torch.cat(
                    [
                        keys(seq, torch.arange(length, length + n))
                        for seq, (length, n) in enumerate(zip(lengths, added))
                    ]
                )
                new_queries = torch.cat(
                    [
                        queries(seq, torch.arange(length, length + n))
                        for seq, (length, n) in enumerate(zip(lengths, added))
                    ]
                )
                if nonzero_qk:
                    targets = torch.cat(
                        [
                            attention_reference(seq, length, n)
                            for seq, (length, n) in enumerate(zip(lengths, added))
                        ]
                    )
                lengths = [a + b for a, b in zip(lengths, added)]
                n = sum(added)
                owned_rows = (torch.arange(n) // interleave) % 8 == partition
                mine = int(owned_rows.sum())
                assert mine <= local
                input_v = torch.zeros((local, heads, hd), dtype=torch.bfloat16)
                expected = torch.zeros_like(input_v).float()
                input_v[:mine] = new_values[owned_rows]
                expected[:mine] = targets[owned_rows]
                q = torch.zeros((local, heads, hd), dtype=torch.bfloat16, device="tpu")
                k = torch.zeros_like(q)
                if nonzero_qk:
                    input_q = torch.zeros_like(input_v)
                    input_k = torch.zeros_like(input_v)
                    input_q[:mine] = new_queries[owned_rows]
                    input_k[:mine] = new_keys[owned_rows]
                    q, k = input_q.to("tpu"), input_k.to("tpu")
                v = input_v.to("tpu")
                seq_lens = torch.tensor(lengths, dtype=torch.int32, device="tpu")
                blocks = table.flatten().to(torch.int32).to("tpu")
                cu = torch.tensor(
                    [0] + list(torch.tensor(added).cumsum(0)),
                    dtype=torch.int32,
                    device="tpu",
                )
                distribution = torch.tensor(
                    [0, 0, nseq], dtype=torch.int32, device="tpu"
                )
                sync.synchronize(
                    [cache, q, k, v, seq_lens, blocks, cu, distribution], wait=True
                )
                cache, result = op(cache, q, k, v, seq_lens, blocks, cu, distribution)
                sync.synchronize([cache, result], wait=True)
                observed = result.cpu().float()
                raw_cpu = cache.cpu()
                raw = raw_cpu.float()
                bits = raw_cpu.view(torch.uint8)
                if layout == KVLayout.SEQ_ALONG_LANE:
                    logical = raw.reshape(nblocks, heads * 2, hd, page)
                    logical = logical.permute(0, 3, 1, 2)
                    bits = bits.reshape(nblocks, heads * 2, hd, page).permute(
                        0, 3, 1, 2
                    )
                else:
                    logical = raw.reshape(nblocks, page, heads * 2, hd)
                    bits = bits.reshape(nblocks, page, heads * 2, hd)
                ref = logical_cache(lengths)
                ref_bits = ref.to(torch.float8_e4m3fn).view(torch.uint8)
                bits_equal = torch.equal(bits, ref_bits)
                differences = torch.where(
                    torch.isnan(logical) & torch.isnan(ref), 0.0, (logical - ref).abs()
                )
                cache_err = float(differences.max())
                err = float((observed - expected).abs().max())
                finite = bool(torch.isfinite(observed).all())
                row = dict(
                    rank=rank,
                    partition=partition,
                    case=f"extended-{case_id}",
                    nonzero_qk=nonzero_qk,
                    page_size=page,
                    interleave=interleave,
                    step=step,
                    layout=layout.name,
                    output_max_abs=err,
                    cache_max_abs=cache_err,
                    cache_bits_equal=bits_equal,
                    nan_sentinel=case_id == 3,
                    pass_=finite and err < 0.04 and bits_equal,
                )
                if not bits_equal:
                    bad = (bits != ref_bits).nonzero()[:8]
                    row["cache_mismatches"] = [
                        dict(
                            index=idx.tolist(),
                            actual=float(logical[tuple(idx)]),
                            expected=float(ref[tuple(idx)]),
                        )
                        for idx in bad
                    ]
                rows.append(row)
                (out / f"rank-{rank}.json").write_text(json.dumps(rows, indent=2))
                print(json.dumps(row), flush=True)


@pytest.mark.nightly
def test_hnd_partial_prefill_and_decode_preserve_attention_and_cache(tmp_path):
    from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
        prepare_tpu_environment,
    )

    keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    )
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        prepare_tpu_environment(world_size=8)
        env = os.environ.copy()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    env["TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS"] = "false"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        str(Path(__file__).resolve()),
        "--worker",
        str(tmp_path),
    ]
    with (tmp_path / "workers.log").open("w") as log:
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=420)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
            pytest.fail(f"PCP workers timed out; see {tmp_path / 'workers.log'}")
    assert returncode == 0, (tmp_path / "workers.log").read_text()[-12000:]
    rows = []
    for rank in range(8):
        rows.extend(json.loads((tmp_path / f"rank-{rank}.json").read_text()))
    assert len(rows) == 528
    failures = [row for row in rows if not row["pass_"]]
    assert not failures, json.dumps(failures, indent=2)


if __name__ == "__main__":
    assert sys.argv[1] == "--worker"
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    _worker(sys.argv[2])
