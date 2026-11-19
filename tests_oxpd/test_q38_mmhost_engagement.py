# SPDX-License-Identifier: Apache-2.0
"""q38 DEC-HOST-4 (MM host bundle row, precondition P1): the two engagement lines.

`q38 TOKEN_PADDING_REUSE: writes W skips S` and `q38 QWEN4_FUSED_PREP_FUNCTIONAL: adopted N` are
log-only counters printed every _Q38_ENG_EVERY events, and only with their flag on. The served row
reads a flag as engaged only if its line's counter grows inside the level's window."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config


class _Rec:
    """Stands in for the module logger: keeps the q38 engagement lines, ignores everything else."""

    def __init__(self):
        self.lines = []

    def _log(self, fmt, *args, **kwargs):
        msg = fmt % args if args else str(fmt)
        if msg.startswith("q38 "):
            self.lines.append(msg)

    info = warning = debug = error = warning_once = info_once = _log


@pytest.mark.parametrize("reuse", [False, True])
def test_token_padding_reuse_engagement_line(monkeypatch, reuse):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import token_padding

    rec = _Rec()
    monkeypatch.setattr(token_padding, "logger", rec)
    monkeypatch.setattr(token_padding, "_Q38_ENG_EVERY", 2)
    monkeypatch.setattr(envs, "VLLM_TPU_TOKEN_PADDING_REUSE", reuse, raising=False)
    state = token_padding.TokenPaddingState.create(8, "cpu")
    for valid in (3, 3, 2, 2, 2, 5):
        state.update(valid, 8)
    if reuse:
        # writes at 3, 2, 5; skips at the repeated 3 and the two repeated 2s
        assert rec.lines == [
            "q38 TOKEN_PADDING_REUSE: writes 1 skips 1",
            "q38 TOKEN_PADDING_REUSE: writes 2 skips 2",
            "q38 TOKEN_PADDING_REUSE: writes 3 skips 3",
        ], rec.lines
    else:
        assert rec.lines == [], rec.lines
    assert state.local_padding_mask.tolist() == [False] * 5 + [True] * 3


@pytest.mark.parametrize("functional", [False, True])
def test_fused_prep_functional_engagement_line(monkeypatch, functional):
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp import inputs as inputs_mod
    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter
    from vllm_torchtpu.runner.model_adapters import ModelInputSnapshot

    rec = _Rec()
    monkeypatch.setattr(inputs_mod, "logger", rec)
    monkeypatch.setattr(inputs_mod, "_Q38_ENG_EVERY", 1)
    rows, max_reqs, max_len, num_tokens = 4, 6, 64, 16
    rng = np.random.default_rng(7)
    runner = NS(device="cpu", max_num_reqs=max_reqs)
    runner.model_config = NS(hf_text_config=q4exp_text_config())
    runner.cache_config = NS(block_size=128)
    runner.vllm_config = NS(model_config=runner.model_config, parallel_config=NS(pipeline_parallel_size=1))
    runner.input_batch = NS(
        token_ids_cpu_tensor=torch.from_numpy(rng.integers(100, 999, (max_reqs, max_len)).astype(np.int32)),
        num_computed_tokens_cpu=np.zeros(max_reqs, dtype=np.int64),
        req_ids=[],
    )
    adapter = Qwen4ExpAdapter(runner)
    adapter.pooled = False
    adapter.spec_window = 4
    monkeypatch.setattr(envs, "VLLM_TPU_QWEN4_FUSED_PREP", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_QWEN4_NGRAM_WINDOW", "", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_QWEN4_FUSED_PREP_CHECK", "", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL", functional, raising=False)
    monkeypatch.setattr(adapter, "_fused_fn", lambda: inputs_mod.fused_step_inputs)
    attention = f"{adapter.attention_prefix}.self_attn"
    tables = {
        adapter._main_owner(attention): NS(block_tables=torch.arange(rows * 3).reshape(rows, 3)),
        adapter._compressed_owner(attention): NS(block_tables=torch.arange(rows * 2).reshape(rows, 2)),
    }
    for req_ids, lengths in [(["a", "b", "c"], [4, 1, 4]), (["c", "a"], [4, 4]), (["c", "d", "a"], [1, 4, 4])]:
        n = len(req_ids)
        ctx_lens = rng.integers(5, max_len - 8, n)
        starts = torch.zeros(rows + 1, dtype=torch.int32)
        starts[1 : n + 1] = torch.tensor(np.cumsum(lengths), dtype=torch.int32)
        starts[n + 1 :] = starts[n]
        seq_lens = torch.zeros(rows, dtype=torch.int32)
        seq_lens[:n] = torch.tensor(ctx_lens + np.array(lengths), dtype=torch.int32)
        ids = torch.from_numpy(rng.integers(100, 999, num_tokens).astype(np.int32))
        runner.input_batch.req_ids = req_ids
        runner.input_batch.num_computed_tokens_cpu[:n] = ctx_lens
        runner._attn_metadata_builder_ctx = NS(
            query_start_loc=starts, seq_lens=seq_lens, mamba_state_indices=torch.tensor([5, 3, 1, 0], dtype=torch.int32)
        )
        store = adapter.begin_step()
        adapter.prepare_inputs(ModelInputSnapshot(store, 0, n, num_tokens, tables, ids))
    want = [f"q38 QWEN4_FUSED_PREP_FUNCTIONAL: adopted {i}" for i in (1, 2, 3)] if functional else []
    assert rec.lines == want, rec.lines
