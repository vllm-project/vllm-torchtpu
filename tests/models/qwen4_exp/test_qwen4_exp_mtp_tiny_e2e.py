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
"""End-to-end TPU tests for Qwen4Exp MTP speculative decoding.

Builds a deterministic tiny Qwen4Exp checkpoint (3 GDN + 1 QSA, 4 HC streams,
n-gram PLE on layer 2, 1-layer QSA MTP draft) whose greedy predictions have
wide margins and whose draft agrees with the target only part of the time.

Verifies on TPU:
  - K=3 MTP greedy outputs match K=0 token-for-token across dense and sparse
    QSA prompts (7, 33, 61, 1000 tokens) under async scheduling and
    `VLLM_TPU_FUSED_DRAFT_LOOP=1`.
  - Every acceptance count 0..K occurs, so verification reads back every GDN
    and PLE checkpoint slot.
  - Prefix-cache hits under K=3 reproduce uncached misses token-for-token.
  - TP=2 with `VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1` matches K=0 and preserves
    prefix-cache hit parity.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

pytestmark = [pytest.mark.tpu_test, pytest.mark.spawns_tpu_workers]

DECODE_STEPS = 24
PROMPT_LENGTHS = (7, 33, 61, 1000)
LM_HEAD_SCALE = 4.0
NUM_SPEC = 3
# Noise on mtp.fc_embedding. At 0.28 the draft accepted 0, 1, 2 and 3 tokens in
# 15, 10, 9 and 9 of 43 verify steps on TPU8i (the same at TP=1 and TP=2).
PARTIAL_ACCEPT_NOISE = 0.28


def _tiny_mtp_config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        architectures=["Qwen4ExpForCausalLM"],
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        hc_count=4,
        hc_lowrank=16,
        ple_layer_ids=[2],
        ple_embed_dim=256,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=997,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
        ple_conv_kernel_size=4,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=16,
        indexer_compress_ratio=4,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
        max_position_embeddings=4096,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        tie_word_embeddings=False,
        output_gate_type="sigmoid",
        norm_topk_prob=True,
        mamba_ssm_dtype="float32",
        full_attention_interval=4,
        seed=1234,
        bos_token_id=1,
        eos_token_id=2,
        dtype="bfloat16",
    )


def _metric_sum(llm: LLM, name: str) -> float:
    return sum(
        float(m.value)
        for m in llm.get_metrics()
        if m.name == name and hasattr(m, "value")
    )


def _accepted_per_pos(llm: LLM) -> list[int]:
    totals: list[int] = []
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_accepted_tokens_per_pos" and hasattr(
            m, "values"
        ):
            vals = [int(v) for v in m.values]
            totals = vals if not totals else [a + b for a, b in zip(totals, vals)]
    return totals


def _make_checkpoint(path: Path) -> None:
    """Writes a tiny Qwen4Exp checkpoint with synthetic mtp.* weights tuned
    for mixed partial acceptance across verification steps."""
    torch.default_generator.manual_seed(0)
    cfg = _tiny_mtp_config()
    model = Qwen4ExpForCausalLM(cfg).eval()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn(param.shape, generator=gen) * 0.02)
        perm = torch.randperm(cfg.vocab_size, generator=gen)
        model.lm_head.weight.copy_(
            LM_HEAD_SCALE * model.model.embed_tokens.weight[perm]
        )
    model.to(torch.bfloat16)
    model.save_pretrained(path, safe_serialization=True)

    # Patch config.json with the mtp sub-dict matching release checkpoints.
    cfg_path = path / "config.json"
    raw_cfg = json.loads(cfg_path.read_text())
    raw_cfg["mtp_num_hidden_layers"] = 1
    raw_cfg["mtp_use_dedicated_embeddings"] = False
    raw_cfg["mtp"] = {
        "hybrid": True,
        "layer_types": ["full_attention"],
        "mtp_use_hidden_state_from_layer": None,
        "num_hidden_layers": 1,
        "rope_theta": 10000000,
    }
    cfg_path.write_text(json.dumps(raw_cfg, indent=2))

    # Append mtp.* tensors derived from the target's QSA layer (layer 3).
    st_path = path / "model.safetensors"
    tensors = load_file(str(st_path))
    hidden = cfg.hidden_size
    hc = cfg.hc_count
    fc_emb = (
        torch.eye(hidden, dtype=torch.float32)
        + torch.randn(hidden, hidden, generator=gen) * PARTIAL_ACCEPT_NOISE
    ).to(torch.bfloat16)
    mtp_tensors: dict[str, torch.Tensor] = {
        "mtp.pre_fc_norm_embedding.weight": torch.zeros(hidden, dtype=torch.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": torch.zeros(hidden * hc, dtype=torch.bfloat16),
        "mtp.fc_embedding.weight": fc_emb,
        "mtp.fc_hidden.weight": (
            torch.eye(hidden, dtype=torch.float32) * 0.04
            + torch.randn(hidden, hidden, generator=gen) * 0.02
        ).to(torch.bfloat16),
    }
    for name, val in tensors.items():
        if name.startswith("model.layers.3."):
            dst = name.replace("model.layers.3.", "mtp.layers.0.", 1)
            mtp_tensors[dst] = val.clone()
        elif name.startswith("model.hyper_connection_mixer.") and (
            "block_inject_weight" not in name
        ):
            dst = name.replace(
                "model.hyper_connection_mixer.", "mtp.hyper_connection_mixer.", 1
            )
            mtp_tensors[dst] = val.clone()
    tensors.update(mtp_tensors)
    save_file(tensors, str(st_path))


def _drafter_state(worker) -> dict[str, bool]:
    drafter = worker.model_runner.drafter
    return {
        "fused_loop_built": drafter._mtp_fused_loop is not None,
        "fused_loop_warm": bool(drafter._mtp_fused_loop_warm),
    }


def _run_llm(
    model_path: str,
    prompts: list[list[int]],
    *,
    num_spec: int,
    tp: int = 1,
    async_scheduling: bool = False,
    check_prefix_cache: bool = False,
    check_drafter_state: bool = False,
) -> SimpleNamespace:
    kwargs: dict[str, object] = dict(
        model=model_path,
        skip_tokenizer_init=True,
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        compilation_config={"compile_sizes": [16, 64, 256]},
        tensor_parallel_size=tp,
        enable_prefix_caching=True,
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.5,
        disable_log_stats=False,
        async_scheduling=async_scheduling,
    )
    if num_spec > 0:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": num_spec,
        }
    llm = LLM(**kwargs)  # type: ignore[arg-type]
    try:
        params = SamplingParams(
            temperature=0.0, max_tokens=DECODE_STEPS, ignore_eos=True
        )
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
        tokens = [list(o.outputs[0].token_ids) for o in outs]
        num_drafts = _metric_sum(llm, "vllm:spec_decode_num_drafts")
        draft_toks = _metric_sum(llm, "vllm:spec_decode_num_draft_tokens")
        acc_toks = _metric_sum(llm, "vllm:spec_decode_num_accepted_tokens")
        per_pos = _accepted_per_pos(llm)

        prefix_ok = None
        if check_prefix_cache:
            fresh = [
                torch.randint(
                    3, 512, (976,), generator=torch.Generator().manual_seed(999)
                ).tolist()
            ]
            hits0 = _metric_sum(llm, "vllm:prefix_cache_hits")
            first = llm.generate([TokensPrompt(prompt_token_ids=fresh[0])], params)
            hits1 = _metric_sum(llm, "vllm:prefix_cache_hits")
            second = llm.generate([TokensPrompt(prompt_token_ids=fresh[0])], params)
            hits2 = _metric_sum(llm, "vllm:prefix_cache_hits")
            prefix_ok = (
                hits1 == hits0
                and hits2 > hits1
                and first[0].outputs[0].token_ids == second[0].outputs[0].token_ids
            )
        drafter_state = (
            llm.collective_rpc(_drafter_state) if check_drafter_state else None
        )
        return SimpleNamespace(
            tokens=tokens,
            num_drafts=num_drafts,
            draft_toks=draft_toks,
            acc_toks=acc_toks,
            per_pos=per_pos,
            prefix_ok=prefix_ok,
            drafter_state=drafter_state,
        )
    finally:
        engine_core = getattr(llm.llm_engine, "engine_core", None)
        if engine_core is not None:
            engine_core.shutdown()


@pytest.fixture(scope="module")
def mtp_baseline(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen4_exp_mtp_tiny")
    _make_checkpoint(path)
    gen = torch.Generator().manual_seed(2)
    prompts = [
        torch.randint(3, 512, (n,), generator=gen).tolist() for n in PROMPT_LENGTHS
    ]
    k0 = _run_llm(str(path), prompts, num_spec=0, tp=1)
    return SimpleNamespace(
        path=str(path),
        prompts=prompts,
        k0_tokens=k0.tokens,
    )


def test_mtp_tp1_fused_async_partial_accept_and_prefix_cache(
    mtp_baseline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_TPU_FUSED_DRAFT_LOOP", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    res = _run_llm(
        mtp_baseline.path,
        mtp_baseline.prompts,
        num_spec=NUM_SPEC,
        tp=1,
        async_scheduling=True,
        check_prefix_cache=True,
        check_drafter_state=True,
    )
    assert res.tokens == mtp_baseline.k0_tokens
    # per_pos[i] counts the verify steps that accepted more than i drafts. A
    # strictly decreasing chain means each acceptance count 0..K occurred, so
    # each checkpoint slot was the one read back at least once.
    assert len(res.per_pos) == NUM_SPEC
    assert res.num_drafts > res.per_pos[0] > res.per_pos[1] > res.per_pos[2] > 0, (
        f"drafts={res.num_drafts} per_pos={res.per_pos}"
    )
    assert res.prefix_ok is True
    assert res.drafter_state == [{"fused_loop_built": True, "fused_loop_warm": True}]


@pytest.mark.multichip
def test_mtp_tp2_disable_block_drop_matches_k0(
    mtp_baseline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", "1")
    res = _run_llm(
        mtp_baseline.path,
        mtp_baseline.prompts,
        num_spec=NUM_SPEC,
        tp=2,
        check_prefix_cache=True,
    )
    assert res.tokens == mtp_baseline.k0_tokens
    assert 0 < res.acc_toks < res.draft_toks
    assert res.prefix_ok is True
