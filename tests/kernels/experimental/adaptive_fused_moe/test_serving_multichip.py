# SPDX-License-Identifier: Apache-2.0
"""Exercise the serving adapter with real EP workers, in eager and compiled mode.

Unlike the pure-JAX kernel tests and mocked adapter tests, this checks that each
TorchTPU worker receives its own token rows through the production bridge.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
TOKENS = 32
HIDDEN = 512
INTER = 256
TOPK = 2
LAUNCH_TIMEOUT_SECONDS = 480


def _prepare_worker_env():
    from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
        prepare_tpu_environment,
    )
    from torch_tpu._internal.utils import hardware

    if hardware.get_tpu_device_count() < WORLD_SIZE:
        pytest.skip("requires eight TPU devices")
    keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    )
    original = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        prepare_tpu_environment(world_size=WORLD_SIZE)
        env = os.environ.copy()
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    env["MOE_FUSED_EP_KERNEL_IMPL"] = "adaptive"
    env["MOE_FUSED_EP_KERNEL_MIN_TOKENS"] = "1024"
    env["OMP_NUM_THREADS"] = "1"
    return env


def _inputs_and_reference(rank):
    import torch

    generator = torch.Generator().manual_seed(123)
    weights = []
    for shape in ((WORLD_SIZE, HIDDEN, 2 * INTER), (WORLD_SIZE, INTER, HIDDEN)):
        raw = torch.randn(shape, generator=generator) / 10
        scale = raw.abs().amax(dim=1, keepdim=True) / 448
        quantized = (raw / scale).to(torch.float8_e4m3fn)
        weights.append((quantized, scale))
    (w1, s1), (w2, s2) = weights

    generator.manual_seed(1000 + rank)
    x = (torch.randn((TOKENS, HIDDEN), generator=generator) / 10).bfloat16()
    # Distinct, exactly representable logits avoid CPU/TPU top-k tie-breaking.
    logits = (
        torch.stack(
            [torch.randperm(WORLD_SIZE, generator=generator) for _ in range(TOKENS)]
        ).bfloat16()
        / 4
    )
    scores, experts = logits.float().softmax(-1).topk(TOPK, dim=-1)
    scores /= scores.sum(dim=-1, keepdim=True)
    reference = torch.zeros((TOKENS, HIDDEN))
    dequant1, dequant2 = w1.float() * s1, w2.float() * s2
    for expert in range(WORLD_SIZE):
        gate, up = (x.float() @ dequant1[expert]).chunk(2, dim=-1)
        contribution = (torch.nn.functional.silu(gate) * up) @ dequant2[expert]
        routing = ((experts == expert) * scores).sum(-1, keepdim=True)
        reference += contribution * routing
    # Each worker owns one contiguous expert; inputs differ on every rank.
    local = (
        x,
        w1[rank : rank + 1].contiguous(),
        w2[rank : rank + 1].contiguous(),
        s1[rank : rank + 1].unsqueeze(2).contiguous(),
        s2[rank : rank + 1].unsqueeze(2).contiguous(),
        logits,
    )
    return local, reference


def _measure(rank):
    import torch
    from torch_tpu._internal import compile as tpu_compile
    from torch_tpu._internal import sync
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
        vllm_adapter as adapter,
    )

    cpu_inputs, reference = _inputs_and_reference(rank)
    x, w1, w2, s1, s2, logits = (tensor.to("tpu") for tensor in cpu_inputs)
    layer = SimpleNamespace(
        w13_weight=w1,
        w2_weight=w2,
        w13_weight_scale_inv=s1,
        w2_weight_scale_inv=s2,
        global_num_experts=WORLD_SIZE,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        use_grouped_topk=False,
        routed_scaling_factor=1.0,
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(
                use_ep=True, pcp_size=WORLD_SIZE, is_sequence_parallel=False
            )
        ),
    )
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.scheduler_config.max_num_batched_tokens = 1024
    with set_current_vllm_config(config):
        op = adapter.prebuild_adaptive_fused_moe(layer, TOPK, True, "silu")
    owner = SimpleNamespace()
    setattr(owner, adapter.FUSED_MOE_EP_OP_ATTR, op)

    def forward(hidden_states, router_logits):
        return adapter.run_adaptive_fused_moe(
            owner, hidden_states, w1, w2, s1, s2, router_logits
        )

    sync.synchronize([x, w1, w2, s1, s2, logits], wait=True)
    compiled = torch.compile(
        forward, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
    )
    results = {}
    for mode, run in (("eager", forward), ("compiled", compiled)):
        output = run(x, logits)
        assert tuple(output.shape) == (TOKENS, HIDDEN), (
            f"rank={rank} {mode}: expected local token rows, got {output.shape}"
        )
        assert output.dtype == x.dtype
        actual = output.cpu().float()
        assert torch.isfinite(actual).all(), f"rank={rank} {mode}: nonfinite output"
        error = torch.linalg.vector_norm(actual - reference)
        relative = float(error / torch.linalg.vector_norm(reference))
        per_token = torch.linalg.vector_norm(actual - reference, dim=-1)
        worst = float((per_token / torch.linalg.vector_norm(reference, dim=-1)).max())
        # Same FP8 error budget as the dense-reference kernel tests.
        assert relative < 0.06 and worst < 0.075, (
            f"rank={rank} {mode}: relative_l2={relative}, worst_token={worst}"
        )
        results[mode] = {
            "relative_l2": relative,
            "worst_token": worst,
            "shape": list(actual.shape),
        }
    return results


def _run_worker(result_dir):
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    result = {"rank": rank}
    try:
        dist.init_process_group(backend="tpu_dist")
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed import parallel_state

        parallel_state.init_distributed_environment(
            world_size=WORLD_SIZE,
            rank=rank,
            local_rank=rank,
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        # Initialize real EP and TP groups without loading a model checkpoint.
        parallel_state._EP = parallel_state.init_model_parallel_group(
            [list(range(WORLD_SIZE))],
            rank,
            "tpu_dist",
            group_name="ep",
            use_device_communicator=False,
        )
        parallel_state._TP = parallel_state.init_model_parallel_group(
            [[r] for r in range(WORLD_SIZE)],
            rank,
            "tpu_dist",
            group_name="tp",
            use_device_communicator=False,
        )
        result.update(_measure(rank))
    except BaseException:
        result["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        (result_dir / f"rank_{rank}.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        if dist.is_initialized():
            dist.destroy_process_group()
    if "error" in result:
        raise SystemExit(1)


def test_serving_preserves_local_rows_and_matches_dense_reference(tmp_path):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--result-dir",
        str(tmp_path),
    ]
    log_path = tmp_path / "workers.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=_prepare_worker_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=LAUNCH_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            pytest.fail(
                f"adaptive MoE workers timed out; see {log_path}\n"
                + log_path.read_text()[-12000:]
            )
    results = []
    for rank in range(WORLD_SIZE):
        path = tmp_path / f"rank_{rank}.json"
        results.append(
            json.loads(path.read_text())
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    errors = [result for result in results if "error" in result]
    assert process.returncode == 0 and not errors, (
        f"worker exit={process.returncode}, errors={errors}\n"
        + log_path.read_text()[-12000:]
    )
    for result in results:
        for mode in ("eager", "compiled"):
            assert result[mode]["shape"] == [TOKENS, HIDDEN]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    _run_worker(parser.parse_args().result_dir)
