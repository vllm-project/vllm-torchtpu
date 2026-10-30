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
"""TPU_DP_VOCAB_PARALLEL over real collectives (DP 4 x TP 2 on one host).

`test_dp_vocab_parallel.py` pins the arithmetic against threaded stand-ins for
the groups; this pins what only hardware can answer: the embedding and the
lm_head sharded over all 8 ranks, through vLLM groups built like the serving
DP / TP / EP groups, match an unsharded CPU reference on every rank, eager and
under `torch.compile`.

Run on one host (4 chips x 2 cores = 8 ranks).
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
DP, TP = 4, 2
VOCAB = 1000  # pads to 1024: 8 shards of 128
HIDDEN = 256
TOKENS = 24
ROWS = 16
LAUNCH_TIMEOUT_SECONDS = 900


def _worker_command(result_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        "--result-dir",
        str(result_dir),
    ]


def _prepare_worker_env() -> dict[str, str]:
    try:
        from tests.distributed.tpu_test_utils import prepare_tpu_environment
    except ImportError as exc:
        pytest.skip(f"TorchTPU is unavailable: {exc}")
    keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    )
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        prepare_tpu_environment(world_size=WORLD_SIZE)
        env = os.environ.copy()
    except ValueError as exc:
        pytest.skip(f"Eight TPU devices are unavailable: {exc}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    return env


def _check(rank: int) -> dict:
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        get_world_group,
        init_model_parallel_group,
    )
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    from vllm_torchtpu import _patch_vllm_vocab_parallel_embedding
    from vllm_torchtpu.layers.adapter import dp_vocab_parallel as dvp

    _patch_vllm_vocab_parallel_embedding()
    dvp.install_logits_hook()
    world = get_world_group()
    backend = torch.distributed.get_backend(world.device_group)

    def group(ranks, name):
        return init_model_parallel_group(
            ranks, world.local_rank, backend, group_name=name
        )

    # The serving layout: EP rank = dp_rank * TP + tp_rank.
    ep = group([list(range(WORLD_SIZE))], "dp_vocab_ep")
    dp = group([[d * TP + t for d in range(DP)] for t in range(TP)], "dp_vocab_dp")
    tp = group([[d * TP + t for t in range(TP)] for d in range(DP)], "dp_vocab_tp")
    dp_rank, tp_rank = rank // TP, rank % TP
    assert (ep.rank_in_group, dp.rank_in_group, tp.rank_in_group) == (
        rank,
        dp_rank,
        tp_rank,
    )
    dvp._groups = lambda: (dp, tp)

    gen = torch.Generator().manual_seed(20261029)
    table = torch.randn(VOCAB, HIDDEN, generator=gen).to(torch.bfloat16)
    ids = [torch.randint(0, VOCAB, (TOKENS,), generator=gen) for _ in range(DP)]
    # Every rank passes the fixed row count (the runner pads before the call).
    hidden = [
        torch.randn(ROWS, HIDDEN, generator=gen).to(torch.bfloat16) for _ in range(DP)
    ]

    class _Head(ParallelLMHead):
        def __init__(self, parallel_group):
            VocabParallelEmbedding.__init__(
                self,
                VOCAB,
                HIDDEN,
                torch.bfloat16,
                parallel_group=parallel_group,
            )
            self.register_parameter("bias", None)

    out: dict = {}
    with set_current_vllm_config(VllmConfig()):
        embed = VocabParallelEmbedding(VOCAB, HIDDEN, torch.bfloat16, parallel_group=ep)
        head = _Head(ep)
        processor = LogitsProcessor(VOCAB)
    for layer in (embed, head):
        layer.weight_loader(layer.weight, table)
        setattr(layer, dvp.DP_VOCAB_ATTR, True)
        layer.to("tpu")
    dvp.set_logits_rows(head, ROWS)

    def run_embed(x):
        return dvp.embedding_forward(embed, x)

    def run_logits(h):
        return processor(head, h)

    want_embed = table[ids[dp_rank]].float()
    want_logits = hidden[dp_rank].float() @ table.float().T
    for how, (fe, fl) in {
        "eager": (run_embed, run_logits),
        "compiled": (
            torch.compile(run_embed, backend="tpu", fullgraph=True, dynamic=False),
            torch.compile(run_logits, backend="tpu", fullgraph=True, dynamic=False),
        ),
    }.items():
        try:
            got = fe(ids[dp_rank].to("tpu")).cpu().float()
            out[f"embed/{how}"] = {
                "shape": list(got.shape),
                "max_abs_err": float((got - want_embed).abs().max()),
            }
            got = fl(hidden[dp_rank].to("tpu")).cpu().float()
            out[f"logits/{how}"] = {
                "shape": list(got.shape),
                "max_abs_err": float((got - want_logits).abs().max()),
                "ref_max_abs": float(want_logits.abs().max()),
                "argmax_match": bool((got.argmax(-1) == want_logits.argmax(-1)).all()),
            }
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            traceback.print_exc()
            out[f"{how}/error"] = repr(exc)
    return out


def _run_worker(result_dir: Path) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result: dict[str, object] = {"rank": rank, "world": world}
    initialized = False
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed.parallel_state import init_distributed_environment

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        result["check"] = _check(rank)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=2))
        if rank == 0:
            print("RESULT rank0", json.dumps(result), flush=True)
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


def _launch(result_dir: Path) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        _worker_command(result_dir),
        cwd=Path(__file__).resolve().parents[3],
        env=_prepare_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        output = exc.output or ""
        process.wait(timeout=60)
    return subprocess.CompletedProcess(process.args, process.returncode, output, "")


def test_dp_vocab_parallel_on_real_collectives(tmp_path):
    result_dir = tmp_path / "dp_vocab"
    result_dir.mkdir()
    completed = _launch(result_dir)
    results = []
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank_{rank}.json"
        results.append(
            json.loads(path.read_text())
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    errors = [r for r in results if "error" in r]
    assert not errors, (errors, (completed.stdout or "")[-4000:])
    for r in results:
        check = r["check"]
        for how in ("eager", "compiled"):
            assert f"{how}/error" not in check, (r["rank"], check)
            assert check[f"embed/{how}"]["max_abs_err"] == 0.0, (r["rank"], check)
            logits = check[f"logits/{how}"]
            # bf16 matmul against an f32 reference: within bf16 rounding of the
            # largest logit. A misrouted shard or row block is off by the
            # logits themselves. (argmax is reported, not asserted: random
            # bf16 data has near ties.)
            assert logits["max_abs_err"] <= 2**-7 * logits["ref_max_abs"], (
                r["rank"],
                how,
                logits,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        _run_worker(args.result_dir)
