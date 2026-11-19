# SPDX-License-Identifier: Apache-2.0
"""q38 DEC-HOST-4 (MM host bundle port): VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL engages.

MM's test_fused_prepare_inputs_matches_eager checks correctness with the flag on and off, so it
passes whether or not the flag is honoured. This test checks the flag's contract itself: with
FUNCTIONAL on, fused_step_inputs leaves the record tensors it is given untouched and returns the
updated record as a fifth item, which the adapter adopts as its current record. With FUNCTIONAL
off, the same steps update the given record in place (the control half)."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter
from vllm_torchtpu.runner.model_adapters import ModelInputSnapshot


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("pooled", [False, True])
def test_functional_fused_prep_leaves_given_record_untouched(monkeypatch, pooled, functional):
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp import inputs as inputs_mod

    spec = 3
    rows, max_reqs, max_len, num_tokens = 4, 6, 64, 16
    rng = np.random.default_rng(38 + pooled)
    runner = NS(device="cpu", max_num_reqs=max_reqs)
    runner.model_config = NS(hf_text_config=q4exp_text_config())
    runner.cache_config = NS(block_size=128)
    # Qwen4ExpAdapter reads its pipeline stage from runner.vllm_config (since 5870deb65).
    runner.vllm_config = NS(model_config=runner.model_config, parallel_config=NS(pipeline_parallel_size=1))
    runner.input_batch = NS(
        token_ids_cpu_tensor=torch.from_numpy(
            rng.integers(100, 999, (max_reqs, max_len)).astype(np.int32)
        ),
        num_computed_tokens_cpu=np.zeros(max_reqs, dtype=np.int64),
        req_ids=[],
    )
    adapter = Qwen4ExpAdapter(runner)
    adapter.pooled = pooled
    adapter.spec_window = spec + 1

    calls = []

    def spy(*args, **kwargs):
        current = args[6]
        before = tuple(t.clone() for t in current) if current is not None else None
        out = inputs_mod.fused_step_inputs(*args, **kwargs)
        calls.append((current, before, out, kwargs.get("functional", False)))
        return out

    monkeypatch.setattr(envs, "VLLM_TPU_QWEN4_FUSED_PREP", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_QWEN4_NGRAM_WINDOW", "", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_QWEN4_FUSED_PREP_CHECK", "", raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL", functional, raising=False)
    monkeypatch.setattr(adapter, "_fused_fn", lambda: spy)
    attention = f"{adapter.attention_prefix}.self_attn"
    tables = {
        adapter._main_owner(attention): NS(block_tables=torch.arange(rows * 3).reshape(rows, 3)),
        adapter._compressed_owner(attention): NS(block_tables=torch.arange(rows * 2).reshape(rows, 2)),
    }
    changed = 0
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
            query_start_loc=starts,
            seq_lens=seq_lens,
            mamba_state_indices=torch.tensor([5, 3, 1, 0], dtype=torch.int32),
        )
        store = adapter.begin_step()
        adapter.prepare_inputs(ModelInputSnapshot(store, 0, n, num_tokens, tables, ids))
        current, before, out, flag = calls[-1]
        assert current is not None and before is not None
        assert flag is functional, (flag, functional)
        untouched = all(torch.equal(t, b) for t, b in zip(current, before))
        if functional:
            assert len(out) == 5 and out[4] is not None
            assert untouched, "FUNCTIONAL on: the given record was mutated in place"
            assert all(a is b for a, b in zip(adapter._recent[1], out[4])), "adapter did not adopt the returned record"
            changed += not all(torch.equal(r, b) for r, b in zip(out[4], before))
        else:
            assert len(out) == 4
            changed += not untouched
    assert changed > 0, "no step changed the record, so the test would be vacuous"
