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
import os
import sys
import traceback
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

REPO_ROOT = Path(__file__).resolve().parents[3]

WORLD_SIZE = 8
DP, TP = 4, 2
VOCAB = 1000  # pads to 1024: 8 shards of 128
HIDDEN = 256
TOKENS = 24
ROWS = 16
# Under the per-PR multichip lane's pytest --timeout=600.
LAUNCH_TIMEOUT_SECONDS = 480


def _check(rank: int) -> dict:
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        get_world_group,
        init_model_parallel_group,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    from vllm_torchtpu import _patch_vllm_vocab_parallel_embedding
    from vllm_torchtpu.layers.adapter import dp_vocab_parallel as dvp

    _patch_vllm_vocab_parallel_embedding()
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
    # Every rank passes the same row count (the runner pads before the call).
    hidden = [
        torch.randn(ROWS, HIDDEN, generator=gen).to(torch.bfloat16) for _ in range(DP)
    ]

    out: dict = {}
    # Stock layers through the TPU vocab patch, handed the serving-layout EP group.
    os.environ["TPU_DP_VOCAB_PARALLEL"] = "1"
    dvp.group_for_new_layer = lambda *_args: ep
    with set_current_vllm_config(VllmConfig()):
        embed = VocabParallelEmbedding(VOCAB, HIDDEN, params_dtype=torch.bfloat16)
        head = ParallelLMHead(VOCAB, HIDDEN, params_dtype=torch.bfloat16)
        processor = dvp.TpuLogitsProcessor(VOCAB)
    for layer in (embed, head):
        assert getattr(layer, dvp.DP_VOCAB_ATTR, False)
        layer.weight_loader(layer.weight, table)
        layer.to("tpu")

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


def test_dp_vocab_parallel_on_real_collectives(tmp_path):
    result_dir = tmp_path / "dp_vocab"
    result_dir.mkdir()
    from tests.distributed.tpu_test_utils import (
        launch_torchrun_workers,
        read_worker_results,
        torchrun_worker_env,
    )

    env = torchrun_worker_env(WORLD_SIZE)
    completed = launch_torchrun_workers(
        Path(__file__).resolve(),
        result_dir,
        world_size=WORLD_SIZE,
        env=env,
        cwd=REPO_ROOT,
        timeout=LAUNCH_TIMEOUT_SECONDS,
    )
    results = read_worker_results(result_dir, WORLD_SIZE, completed)
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
        # Run as a script: the repo root is not on sys.path.
        sys.path.insert(0, str(REPO_ROOT))
        from tests.distributed.tpu_test_utils import run_torchrun_worker

        run_torchrun_worker(args.result_dir, _check)
