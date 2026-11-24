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
"""TPU regression for Qwen4ExpBackboneSP live QSA selection on the merge path.

Runs two real pooled Qwen4ExpQSA layers inside Qwen4ExpBackboneSP at TP=2 over
one TPU chip (2 devices) with `VLLM_TPU_QSA_INDEXER_TOKEN_SHARD=True` and a
mixed speculative-verify + prefill step whose verify context enters the
multi-shard merge branch (`max(lengths) // 4 > 512`). Asserts finite non-zero
verify outputs and an updated int32 `_qsa_sel_buf` under `torch.compile`.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

pytestmark = [pytest.mark.tpu_test, pytest.mark.spawns_tpu_workers]

REPO_ROOT = Path(__file__).resolve().parents[4]

WORLD_SIZE = 2
NUM_QSA_LAYERS = 2
HIDDEN = 256
HC = 4
LOWRANK = 64
VOCAB = 256
MAX_MODEL_LEN = 8192
TOKENS = 2048
VERIFY_TOKENS = 4
PREFILL_TOKENS = 2040
ACTIVE_TOKENS = VERIFY_TOKENS + PREFILL_TOKENS
VERIFY_CONTEXT = 2176  # (2176 + 4) // 4 = 545 > 512 (multi-shard merge branch)
SLOTS = 4
PAGE_TOKENS = 128
TABLE_COLS = 20
POOL_PAGES = 160
SEL_SENTINEL = -999999
# Under the per-PR TPU lane's pytest --timeout=600.
LAUNCH_TIMEOUT_SECONDS = 480


def _check(rank: int) -> dict:
    import torch
    from torch import nn
    from torch_tpu._internal import sync
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationMode
    from vllm.distributed import parallel_state as ps
    from vllm.forward_context import override_forward_context
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    from tests.models.qwen4_exp.experimental.qwen38_test_utils import (
        q4exp_text_config,
    )
    from vllm_torchtpu.models.qwen4_exp.experimental import model as qwen4_model
    from vllm_torchtpu.models.qwen4_exp.experimental import plan
    from vllm_torchtpu.models.qwen4_exp.experimental.attention import Qwen4ExpQSA
    from vllm_torchtpu.models.qwen4_exp.experimental.hyperconnection import (
        GatedResidual,
        HyperConnectionConfig,
    )

    world = ps.get_world_group()
    backend = torch.distributed.get_backend(world.device_group)
    ps._TP = ps.init_model_parallel_group(
        [list(range(WORLD_SIZE))], world.local_rank, backend, group_name="tp"
    )

    orig_plan_value = plan.value
    feats = plan.features(plan.UNBOUND_PLAN, plain=True)
    feats.update(
        QWEN4_EXP_PREFILL_SP=True,
        QWEN4_EXP_PREFILL_SP_MOE=False,
        QWEN4_EXP_PREFILL_SP_SHARED_LOCAL=False,
        VLLM_TPU_QSA_INDEXER_TOKEN_SHARD=True,
        VLLM_TPU_QSA_MAIN_KV_DCP=0,
        VLLM_TPU_QSA_DRAFT_SHARED_SELECTION=False,
        VLLM_TPU_GDN_HOIST_METADATA=False,
    )
    plan.configure_kernels(plan.UNBOUND_PLAN)
    plan.value = lambda name: feats[name]
    try:
        cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
        resolve_kv_cache_layout(cfg, [["LBNHC", "LBHNC"]])
        text = q4exp_text_config(
            hidden_size=HIDDEN,
            vocab_size=VOCAB,
            hc_count=HC,
            hc_lowrank=LOWRANK,
            num_hidden_layers=NUM_QSA_LAYERS,
            layer_types=["full_attention"] * NUM_QSA_LAYERS,
            max_position_embeddings=MAX_MODEL_LEN,
        )
        gen = torch.Generator(device="cpu").manual_seed(20261126)

        pool_cpu = torch.zeros(
            (POOL_PAGES, PAGE_TOKENS, 1, 2, text.head_dim), dtype=torch.bfloat16
        )
        pool_cpu[1:].copy_(
            (torch.randn(pool_cpu[1:].shape, generator=gen) * 0.05).to(torch.bfloat16)
        )
        ring_pages: list[int] = []

        hc_cfg = HyperConnectionConfig(
            hc_count=text.hc_count,
            hidden_size=text.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=text.hc_lowrank,
            rms_norm_eps=text.rms_norm_eps,
            hc_per_branch_norm=True,
        )

        with set_current_vllm_config(cfg), set_default_torch_dtype(torch.bfloat16):
            backbone = qwen4_model.Qwen4ExpBackbone.__new__(
                qwen4_model.Qwen4ExpBackbone
            )
            nn.Module.__init__(backbone)
            object.__setattr__(backbone, "do_not_compile", True)
            backbone.hc_count = text.hc_count
            backbone.host_ple_index = None
            backbone.emit_mtp_hidden = False
            backbone.embed_tokens = nn.Embedding(text.vocab_size, text.hidden_size)
            backbone.layers = nn.ModuleList()

            attn_metadata: dict[str, object] = {}
            next_page = 1

            def _alloc_table(req0_pages: int, req1_pages: int) -> torch.Tensor:
                nonlocal next_page
                table = torch.zeros((SLOTS, TABLE_COLS), dtype=torch.int32)
                table[0, :req0_pages] = torch.arange(
                    next_page, next_page + req0_pages, dtype=torch.int32
                )
                next_page += req0_pages
                table[1, :req1_pages] = torch.arange(
                    next_page, next_page + req1_pages, dtype=torch.int32
                )
                next_page += req1_pages
                return table.to("tpu")

            def _alloc_ring() -> torch.Tensor:
                nonlocal next_page
                rings = torch.zeros((SLOTS, 1), dtype=torch.int32)
                rings[0, 0] = next_page
                rings[1, 0] = next_page + 1
                ring_pages.extend((next_page, next_page + 1))
                next_page += 2
                return rings.to("tpu")

            req0_pages = (VERIFY_CONTEXT + VERIFY_TOKENS + PAGE_TOKENS - 1) // (
                PAGE_TOKENS
            )
            req1_pages = (PREFILL_TOKENS + PAGE_TOKENS - 1) // PAGE_TOKENS

            for idx in range(NUM_QSA_LAYERS):
                layer = qwen4_model.Qwen4ExpDecoderLayer.__new__(
                    qwen4_model.Qwen4ExpDecoderLayer
                )
                nn.Module.__init__(layer)
                layer.ple = None
                layer.layer_type = "full_attention"
                prefix = f"model.layers.{idx}.self_attn"
                layer.self_attn = Qwen4ExpQSA(
                    config=text,
                    max_model_len=MAX_MODEL_LEN,
                    tp_rank=rank,
                    tp_size=WORLD_SIZE,
                    device="cpu",
                    cache_dtype="auto",
                    prefix=prefix,
                    pooled=True,
                    max_query_len=VERIFY_TOKENS,
                    max_num_seqs=SLOTS,
                )
                assert layer.self_attn.token_shard is True
                layer.mlp = nn.Linear(text.hidden_size, text.hidden_size, bias=False)
                layer.attn_hyper_connection = GatedResidual(hc_cfg)
                layer.mlp_hyper_connection = GatedResidual(hc_cfg)
                backbone.layers.append(layer)

                attn_metadata[prefix] = NS(
                    block_tables=_alloc_table(req0_pages, req1_pages)
                )
                attn_metadata[prefix + ".compressed_k"] = NS(
                    block_tables=_alloc_table(req0_pages, req1_pages)
                )
                attn_metadata[prefix + ".raw_ring"] = NS(block_tables=_alloc_ring())

            backbone.hyper_connection_mixer = GatedResidual(hc_cfg, use_combine=False)

        assert next_page <= POOL_PAGES
        for page in ring_pages:
            pool_cpu[page].zero_()
        pool = pool_cpu.to("tpu")
        with torch.no_grad():
            for param in backbone.parameters():
                param.copy_(
                    (torch.randn(param.shape, generator=gen) * 0.05).to(param.dtype)
                )
        backbone.to("tpu")
        for layer in backbone.layers:
            layer.self_attn.bind_pool(pool)
            assert layer.self_attn._token_shard_op is not None

        sp_cfg = NS(compilation_config=NS(mode=CompilationMode.NONE))
        model_sp = qwen4_model.Qwen4ExpBackboneSP(vllm_config=sp_cfg, backbone=backbone)
        assert model_sp._qsa_sel_buf is not None
        assert model_sp._qsa_sel_buf.device.type == "tpu"
        assert model_sp._qsa_sel_buf.dtype == torch.int32
        model_sp._qsa_sel_buf.fill_(SEL_SENTINEL)

        input_ids = (torch.arange(TOKENS, dtype=torch.int64) % VOCAB).to("tpu")
        positions_cpu = torch.zeros(TOKENS, dtype=torch.int64)
        positions_cpu[:VERIFY_TOKENS] = VERIFY_CONTEXT + torch.arange(
            VERIFY_TOKENS, dtype=torch.int64
        )
        positions_cpu[VERIFY_TOKENS:ACTIVE_TOKENS] = torch.arange(
            PREFILL_TOKENS, dtype=torch.int64
        )
        positions = positions_cpu.to("tpu")
        query_start_loc = torch.tensor(
            [0, VERIFY_TOKENS, ACTIVE_TOKENS, ACTIVE_TOKENS, ACTIVE_TOKENS],
            dtype=torch.int32,
        ).to("tpu")
        ngram_context = torch.zeros((SLOTS, 2), dtype=torch.int64).to("tpu")
        state_slots = torch.arange(SLOTS, dtype=torch.int32).to("tpu")
        context_lens = torch.tensor([VERIFY_CONTEXT, 0, 0, 0], dtype=torch.int32).to(
            "tpu"
        )
        valid_tokens = (torch.arange(TOKENS) < ACTIVE_TOKENS).to("tpu")
        placeholder_table = torch.zeros((SLOTS, TABLE_COLS), dtype=torch.int32).to(
            "tpu"
        )

        compiled_sp = torch.compile(
            model_sp.forward, backend="tpu", fullgraph=True, dynamic=False
        )
        with override_forward_context(NS(attn_metadata=attn_metadata)):
            out = compiled_sp(
                input_ids,
                positions,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                state_slots=state_slots,
                context_lens=context_lens,
                valid_tokens=valid_tokens,
                main_block_table=placeholder_table,
                compressed_block_table=placeholder_table,
            )
        sync.synchronize([out, model_sp._qsa_sel_buf, pool], wait=True)
        out_cpu = out.cpu().float()
        sel_cpu = model_sp._qsa_sel_buf.cpu()
        return {
            "out_shape": list(out_cpu.shape),
            "verify_finite": bool(torch.isfinite(out_cpu[:VERIFY_TOKENS]).all()),
            "active_finite": bool(torch.isfinite(out_cpu[:ACTIVE_TOKENS]).all()),
            "verify_max_abs": float(out_cpu[:VERIFY_TOKENS].abs().max()),
            "sel_buf_dtype": str(sel_cpu.dtype),
            "sel_buf_shape": list(sel_cpu.shape),
            "sel_buf_values": sel_cpu.tolist(),
            "sel_buf_updated": bool((sel_cpu != SEL_SENTINEL).all()),
            "sel_buf_valid": bool((sel_cpu >= -1).all()),
        }
    finally:
        plan.value = orig_plan_value
        torch._dynamo.reset()
        ps.destroy_model_parallel()


def test_backbone_sp_real_qsa_merge_branch_keeps_verify_rows_finite(tmp_path):
    result_dir = tmp_path / "sp_qsa_live_sel"
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
    for result in results:
        check = result["check"]
        assert check["out_shape"] == [TOKENS, HIDDEN], (result["rank"], check)
        assert check["verify_finite"], (result["rank"], check)
        assert check["active_finite"], (result["rank"], check)
        assert check["verify_max_abs"] > 0.0, (result["rank"], check)
        assert check["sel_buf_dtype"] == "torch.int32", (result["rank"], check)
        assert check["sel_buf_shape"] == [NUM_QSA_LAYERS, 1], (result["rank"], check)
        assert check["sel_buf_updated"], (result["rank"], check)
        assert check["sel_buf_valid"], (result["rank"], check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        sys.path.insert(0, str(REPO_ROOT))
        from tests.distributed.tpu_test_utils import run_torchrun_worker

        run_torchrun_worker(args.result_dir, _check)
