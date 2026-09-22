# SPDX-License-Identifier: Apache-2.0
"""Host PLE transport preserves every bit across compiled TP/DP boundaries.

The CPU consumer test guards input ownership. This test additionally catches
backend corruption in the H2D -> cloned graph argument -> collective path,
which CPU group mocks cannot exercise. No model checkpoint is required.
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

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]


@pytest.mark.skipif(os.environ.get("Q38_TEST_TPU") != "1",
                    reason="Requires an exclusive eight-TPU window")
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_host_ple_compiled_transport_preserves_checkpoint_bits(tp, tmp_path):
    command = [
        sys.executable,
        str(Path(__file__).resolve()), "--tp",
        str(tp), "--out",
        str(tmp_path)
    ]
    with (tmp_path / "workers.log").open("w") as log:
        child = subprocess.Popen(command,
                                 stdout=log,
                                 stderr=subprocess.STDOUT,
                                 start_new_session=True)
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
    dp, dp_rank = 8 // tp, rank // tp
    dist.init_process_group(backend="tpu_dist",
                            timeout=datetime.timedelta(seconds=120))
    torch.empty(1, device="tpu").cpu()
    from vllm.distributed import parallel_state as ps

    from vllm_torchtpu.models.vllm.qwen4_exp.parallel import PLEParallel
    from vllm_torchtpu.models.vllm.qwen4_exp.ple import HostPLEEmbedding

    ps.init_distributed_environment(world_size=8,
                                    rank=rank,
                                    local_rank=rank,
                                    distributed_init_method="env://",
                                    backend="tpu_dist")
    ps._TP = ps.init_model_parallel_group(
        [list(range(d * tp, (d + 1) * tp)) for d in range(dp)],
        rank,
        "tpu_dist",
        group_name="tp")
    ps._DP = ps.init_model_parallel_group(
        [list(range(t, 8, tp)) for t in range(tp)],
        rank,
        "tpu_dist",
        group_name="dp")
    checks, errors = 0, []
    for dtype, carrier in [(torch.bfloat16, torch.int16),
                           (torch.float8_e4m3fn, torch.uint8)]:
        source = (torch.arange(8192, device="cpu")[:, None] * 71 +
                  torch.arange(160, device="cpu")[None, :]).to(carrier)
        for shared in (False, True):
            parallel = PLEParallel(ps.get_tp_group(),
                                   ps.get_dp_group(),
                                   embedding_across_dp=not shared)
            options = (dict(directory=str(out / "control"),
                            key=f"{dtype}/tp{rank % tp}",
                            identity={"dtype": str(dtype)},
                            participants=dp) if shared else None)
            table = HostPLEEmbedding(tp_rank=parallel.rank,
                                     tp_size=parallel.world_size,
                                     total_rows=8192,
                                     checkpoint_shards=13,
                                     device="tpu",
                                     shared=options)
            shard_rows = (8192 + 12) // 13
            try:
                for shard in reversed(range(13)):
                    weight = source[shard * shard_rows:(shard + 1) *
                                    shard_rows]
                    table.load_weights([(f"shard_{shard}.weight",
                                         weight.view(dtype))])
                if dtype == torch.float8_e4m3fn:
                    table.load_weights([("weight_scale",
                                         torch.tensor([0.25],
                                                      dtype=torch.bfloat16,
                                                      device="cpu"))])
                table.finalize_weights()
                assert table.table.numel() == 0
                gather = torch.compile(parallel.gather_ids,
                                       backend="tpu",
                                       fullgraph=True,
                                       dynamic=False)
                for slots in (1, 7, 8, 16, 64, 128):

                    def consume(raw, parallel=parallel, slots=slots):
                        # Same ownership boundary as Qwen4ExpPLE.forward;
                        # the CPU test also checks that actual model entry.
                        return parallel.combine(
                            raw.reshape(-1, 2560).clone(), slots)

                    compiled = torch.compile(consume,
                                             backend="tpu",
                                             fullgraph=True,
                                             dynamic=False)
                    counts = ([min(3, slots), 0, slots, 1, 0, slots, 1,
                               0][:dp], [slots] + [0] * (dp - 1), [0] * dp,
                              [slots] * dp)
                    for scenario, live in enumerate(counts):
                        ids = (torch.arange(slots * 16, device="cpu").reshape(
                            slots, 16) * 193 +
                               dp_rank * 1023).remainder(8192).int()
                        if scenario == 3:
                            ids = 7168 + ids.remainder(1024)
                        ids[live[dp_rank]:] = -1
                        expected = torch.where((ids >= 0)[:, :, None],
                                               source[ids.clamp(min=0)], 0)
                        raw = table.lookup_host(gather(ids.to("tpu")).cpu())
                        prepared_cpu = raw.reshape(-1, slots, 2560)
                        prepared = prepared_cpu.to("tpu")
                        actual = compiled(prepared).cpu().view(carrier)
                        checks += 1
                        if (not torch.equal(actual,
                                            expected.reshape(slots, 2560)) or
                                not torch.equal(prepared.cpu(), prepared_cpu)):
                            errors.append(
                                dict(dtype=str(dtype),
                                     shared=shared,
                                     slots=slots,
                                     scenario=scenario))
                    torch._dynamo.reset()
            finally:
                table.close()
    (out / f"rank{rank}.json").write_text(
        json.dumps(dict(rank=rank, checks=checks, errors=errors)))
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()
    assert not errors, errors


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        _worker(args.tp, args.out)
        return
    from torch_tpu._internal.distributed.launchers.singlehost_wrapper import \
        prepare_tpu_environment

    prepare_tpu_environment(world_size=8)
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nnodes=1", "--nproc-per-node=8",
        str(Path(__file__).resolve()), "--worker", "--tp",
        str(args.tp), "--out",
        str(args.out)
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    _main()
