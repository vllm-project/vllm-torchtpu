# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Real DP8 eager embedding with independent Qwen metadata and lazy idle IDs."""

import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]
CASES = ("uploaded", "fresh_zeros", "fresh_arange")


def _harness():
    path = Path(__file__).with_name("test_dp_vocab_parallel_multichip.py")
    spec = importlib.util.spec_from_file_location("dp_vocab_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(rank, tp_size):
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        get_world_group,
        init_model_parallel_group,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    from vllm_torchtpu import _patch_vllm_vocab_parallel_embedding
    from vllm_torchtpu.layers.adapter import dp_vocab_parallel as dvp
    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter
    from vllm_torchtpu.utils import synchronize_tensors

    assert tp_size == 1
    _patch_vllm_vocab_parallel_embedding()
    world = get_world_group()
    backend = torch.distributed.get_backend(world.device_group)

    def group(ranks, name):
        return init_model_parallel_group(
            ranks, world.local_rank, backend, group_name=name
        )

    ep = group([list(range(8))], "metadata_ep")
    dp = group([list(range(8))], "metadata_dp")
    tp = group([[i] for i in range(8)], "metadata_tp")
    dvp._groups = lambda: (dp, tp)
    vocab, hidden, tokens = 248320, 2560, 8192
    table = (
        (torch.arange(vocab).remainder(97).to(torch.bfloat16) - 48)[:, None]
        * (torch.arange(hidden).remainder(31).to(torch.bfloat16) - 15)[None, :]
    ) * 0.0002
    with set_current_vllm_config(VllmConfig()):
        embed = VocabParallelEmbedding(vocab, hidden, torch.bfloat16, parallel_group=ep)
    embed.weight_loader(embed.weight, table)
    setattr(embed, dvp.DP_VOCAB_ATTR, True)
    embed.to("tpu")

    starts = torch.tensor([0] + [tokens] * 16, dtype=torch.int32, device="tpu")
    lengths = torch.tensor([tokens] + [0] * 15, dtype=torch.int32, device="tpu")
    history = torch.tensor([[100, 101, 102, 103, 104]], dtype=torch.int64, device="tpu")
    predicted = torch.tensor([0], dtype=torch.int64, device="tpu")
    idle = dict(
        context_lens=torch.zeros(16, dtype=torch.int64, device="tpu"),
        ngram_context=torch.full((16, 2), 151645, dtype=torch.int64, device="tpu"),
        valid_tokens=torch.zeros(tokens, dtype=torch.bool, device="tpu"),
    )
    adapter = NS(
        runner=NS(
            device=torch.device("tpu"),
            _attn_metadata_builder_ctx=NS(query_start_loc=starts, seq_lens=lengths),
            speculative_config=NS(num_speculative_tokens=3),
        ),
        config=NS(ngram_size=3, eos_token_id=151645),
    )
    snapshot = NS(
        step_context=(history, predicted),
        row_start=0,
        num_requests=1,
        num_tokens=tokens,
    )
    synchronize_tensors([starts, lengths, history, predicted, *idle.values()])

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def trunk(embedding, context, ngrams, valid):
        # Consume all four roots in one compiled call, like the target model.
        return torch.tanh(
            embedding * 0.75
            + valid[:, None].to(torch.bfloat16) * 0.125
            + context.sum().to(torch.bfloat16) * 0.001
            + ngrams.sum().to(torch.bfloat16) * 0.000001
        )

    ids_cpu = (torch.arange(tokens, dtype=torch.int32) + rank + 37) % vocab
    uploaded = ids_cpu.to("tpu")
    synchronize_tensors(trunk(dvp.embedding_forward(embed, uploaded), *idle.values()))
    torch.distributed.barrier(group=world.cpu_group)
    fences, passed = [], []
    for case in CASES:
        expected_ids, ids = ids_cpu, uploaded
        if rank and case == "fresh_zeros":
            ids = torch.zeros(tokens, dtype=torch.int32, device="tpu")
            expected_ids = torch.zeros_like(ids_cpu)
        elif rank and case == "fresh_arange":
            ids = torch.arange(tokens, dtype=torch.int32, device="tpu")
            expected_ids = torch.arange(tokens, dtype=torch.int32)
        embedding = dvp.embedding_forward(embed, ids)
        metadata = (
            Qwen4ExpAdapter.prepare_inputs(adapter, snapshot) if rank == 0 else idle
        )
        fence = torch.tpu.Event()
        fence.record()
        fences.append(fence)
        synchronize_tensors(
            trunk(
                embedding,
                metadata["context_lens"],
                metadata["ngram_context"],
                metadata["valid_tokens"],
            )
        )
        assert torch.equal(embedding.cpu(), table[expected_ids.long()])
        torch.distributed.barrier(group=world.cpu_group)
        passed.append(case)
    return {"cases": passed, "exact_embedding": True}


def test_dp_vocab_eager_embedding_with_independent_metadata(tmp_path):
    harness = _harness()
    env = harness._prepare_worker_env()
    # The harness initializes libtpu before plugin import. Set the service's
    # TC defaults before process creation, rather than testing the SC backend.
    defaults = {
        "--xla_tpu_use_dynamic_smem_negotiation": "true",
        "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes": "67108864",
        "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes": "67108864",
    }
    flags = [
        flag
        for flag in shlex.split(env.get("LIBTPU_INIT_ARGS", ""))
        if flag.split("=", 1)[0] not in defaults
    ]
    env["LIBTPU_INIT_ARGS"] = " ".join(
        [*flags, *(f"{k}={v}" for k, v in defaults.items())]
    )
    env["TPU_DP_VOCAB_PARALLEL"] = "1"
    command = harness._worker_command(tmp_path, 1)
    script_index = command.index(str(Path(harness.__file__).resolve()))
    command[script_index:] = [str(Path(__file__).resolve()), "--worker", str(tmp_path)]
    harness._worker_command = lambda *_: command
    harness._prepare_worker_env = lambda: env
    harness.LAUNCH_TIMEOUT_SECONDS = 180
    result = harness._launch(tmp_path, 1)
    assert result.returncode == 0, (result.stdout or "")[-4000:]
    for rank in range(8):
        result = json.loads((tmp_path / f"rank_{rank}.json").read_text())
        assert "error" not in result, result
        assert result["check"] == {"cases": list(CASES), "exact_embedding": True}


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "--worker"
    harness = _harness()
    harness._check = _check
    harness._run_worker(Path(sys.argv[2]), 1)
