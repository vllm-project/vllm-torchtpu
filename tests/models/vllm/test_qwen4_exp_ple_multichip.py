# SPDX-License-Identifier: Apache-2.0
"""PLE bytes must survive compiled TP-by-DP transport, including odd buckets.

CPU group mocks cannot catch the narrow-view/packed-reduction corruption this
test reproduces. Use the multichip test selection on an idle eight-device host.
Workers borrow TP and DP groups without creating an EP group or loading the full model.
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.tpu_test,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
]


@pytest.mark.parametrize("tp", [2, 4])
def test_compiled_ple_transport_preserves_raw_bits(tp, tmp_path):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--tp",
        str(tp),
        "--out",
        str(tmp_path),
    ]
    # Bootstrap in a subprocess: pytest must not retain a device client or
    # topology environment after these workers exit.
    with (tmp_path / "workers.log").open("w") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = child.wait(timeout=600)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    assert code == 0, (tmp_path / "workers.log").read_text()[-12000:]
    for rank in range(8):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result["checks"] == 96
        assert not result["errors"], result


def _worker(tp, out):
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    dp = 8 // tp
    dp_rank = rank // tp
    dist.init_process_group(backend="tpu_dist", timeout=datetime.timedelta(seconds=120))
    torch.empty(1, device="tpu").cpu()
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as ps
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbeddingMethod
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    from vllm_torchtpu.models.vllm.qwen4_exp.parallel import PLEParallel
    from vllm_torchtpu.models.vllm.qwen4_exp.ple_storage import ResidentPLETable

    ps.init_distributed_environment(
        world_size=8,
        rank=rank,
        local_rank=rank,
        distributed_init_method="env://",
        backend="tpu_dist",
    )
    ps._TP = ps.init_model_parallel_group(
        [list(range(d * tp, (d + 1) * tp)) for d in range(dp)],
        rank,
        "tpu_dist",
        group_name="tp",
    )
    ps._DP = ps.init_model_parallel_group(
        [list(range(t, 8, tp)) for t in range(tp)], rank, "tpu_dist", group_name="dp"
    )
    etp_group = ps.init_model_parallel_group(
        [list(range(8))], rank, "tpu_dist", group_name="ple_table"
    )
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(cfg, [["LBNHC", "LBHNC"]])
    checks, errors = 0, []
    for dtype, carrier in [
        (torch.bfloat16, torch.int16),
        (torch.float8_e4m3fn, torch.uint8),
    ]:
        # All bit patterns, including signed carrier words. No floating-point
        # operation is part of this raw transport contract.
        source = (torch.arange(8192)[:, None] * 71 + torch.arange(160)[None, :]).to(
            carrier
        )
        for across in (False, True):
            parallel = PLEParallel(
                ps.get_tp_group(), ps.get_dp_group(), embedding_across_dp=across
            )
            method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
                None, "ple", "float8_e4m3fn" if dtype == torch.float8_e4m3fn else None
            )
            with set_current_vllm_config(cfg), set_default_torch_dtype(torch.bfloat16):
                table = ResidentPLETable(
                    8192,
                    160,
                    embedding_method=method,
                    padding_size=64,
                    prefix="ple",
                    parallel_group=etp_group if across else ps.get_tp_group(),
                )
            assert (table.tp_rank, table.tp_size) == (
                parallel.rank,
                parallel.world_size,
            )
            shard_rows = (8192 + 12) // 13
            for shard in reversed(range(13)):
                weight = source[shard * shard_rows : (shard + 1) * shard_rows]
                table.weight_loader(
                    table.weight, weight.view(dtype), shard * shard_rows
                )
            if dtype == torch.float8_e4m3fn:
                table.get_parameter("weight_scale").data.fill_(0.00019931793212890625)
            table.to("tpu")
            table.finalize_weights()

            def forward(ids, parallel=parallel, table=table):
                gathered = parallel.gather_ids(ids)
                raw = table.lookup_raw(gathered)
                return parallel.combine(raw, ids.shape[0])

            for compiled in (False, True):
                fn = (
                    torch.compile(forward, backend="tpu", fullgraph=True, dynamic=False)
                    if compiled
                    else forward
                )
                for slots in (7, 8, 64):
                    counts = (
                        [3, 0, 7, 1][:dp],
                        [slots] + [0] * (dp - 1),
                        [0] * dp,
                        [slots] * dp,
                    )
                    for scenario, live in enumerate(counts):
                        ids = (
                            (
                                torch.arange(slots * 16).reshape(slots, 16) * 193
                                + dp_rank * 1023
                            )
                            .remainder(8192)
                            .int()
                        )
                        if scenario == 3:
                            ids = 7168 + ids.remainder(1024)
                        ids[live[dp_rank] :] = -1
                        expected = torch.where(
                            (ids >= 0)[:, :, None], source[ids.clamp(min=0)], 0
                        ).reshape(slots, 2560)
                        actual = fn(ids.to("tpu")).cpu().view(carrier)
                        checks += 1
                        if not torch.equal(actual, expected):
                            errors.append(
                                dict(
                                    dtype=str(dtype),
                                    across=across,
                                    compiled=compiled,
                                    slots=slots,
                                    scenario=scenario,
                                )
                            )
                torch._dynamo.reset()
    (out / f"rank{rank}.json").write_text(
        json.dumps(dict(rank=rank, checks=checks, errors=errors))
    )
    etp_group.destroy()
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()
    assert not errors, errors


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        _worker(args.tp, args.out)
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tests.distributed.tpu_test_utils import prepare_tpu_environment

    prepare_tpu_environment(world_size=8)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        str(Path(__file__).resolve()),
        "--worker",
        "--tp",
        str(args.tp),
        "--out",
        str(args.out),
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    _main()
