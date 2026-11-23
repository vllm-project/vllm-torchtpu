# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_DRAFT_LOOP_MAX_COMPILE_T: the fused draft loop compiles only the
token buckets it can run (<= the bucket of max_num_seqs), and the runtime
path is unchanged for every reachable size."""

import types

import pytest
import torch

from vllm_torchtpu.spec_decode.mtp_fused_loop import loop_compile_config

PADDINGS = [4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 384, 512, 1024, 2048, 3072, 4096, 8192]


def _cfg(sizes=PADDINGS):
    cc = types.SimpleNamespace(compile_sizes=list(sizes), compile_ranges_endpoints=[8192])
    return types.SimpleNamespace(compilation_config=cc)


@pytest.mark.parametrize(
    "max_num_reqs,max_t,want_cap",
    [(32, 32, 32), (16, 32, 32), (16, 16, 16), (24, 16, 24), (48, 32, 48), (96, 32, 96), (32, 1, 32)],
)
def test_sizes_capped_at_reachable_bucket(max_num_reqs, max_t, want_cap):
    base = _cfg()
    cfg, sizes, cap = loop_compile_config(base, max_num_reqs, max_t)
    assert cap == want_cap
    assert sizes == [s for s in PADDINGS if s <= want_cap]
    assert cfg.compilation_config.compile_sizes == sizes
    assert cfg.compilation_config.compile_ranges_endpoints == []
    # The draft's own config is untouched (step 0 and the per-step loop keep
    # every bucket).
    assert base.compilation_config.compile_sizes == PADDINGS
    assert base.compilation_config.compile_ranges_endpoints == [8192]
    # Every loop size the runtime can produce (bucket of 1..max_num_reqs) is kept.
    from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len

    reachable = {_get_padded_token_len(PADDINGS, n) for n in range(1, max_num_reqs + 1)}
    assert reachable <= set(sizes)


def test_flag_off_keeps_every_size(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_LOOP_MAX_COMPILE_T", 0, raising=False)
    assert envs.VLLM_TPU_DRAFT_LOOP_MAX_COMPILE_T == 0  # build path not taken


def _proposer(max_t, calls):
    from vllm_torchtpu.spec_decode import eagle3

    p = types.SimpleNamespace(_mtp_fused_loop_max_t=max_t)

    def md(**kw):
        calls.append("metadata")
        raise RuntimeError("stop after the guard")  # the program path is the stock one

    p._build_draft_attn_metadata = md
    return eagle3, p


@pytest.mark.parametrize("max_t", [0, 32])
@pytest.mark.parametrize("n", [4, 8, 16, 24, 32])
def test_reachable_sizes_take_the_program_path(max_t, n):
    """Flag on or off, a reachable size goes on to the unchanged program path
    (metadata build, then the fused program)."""
    calls = []
    eagle3, p = _proposer(max_t, calls)
    with pytest.raises(RuntimeError, match="stop after the guard"):
        eagle3.Eagle3Proposer._run_mtp_fused_loop(
            p, None, torch.zeros(n, dtype=torch.int32), None, None, None, None, None, None
        )
    assert calls == ["metadata"]


def test_size_past_cap_falls_back_to_per_step():
    calls = []
    eagle3, p = _proposer(32, calls)
    out = eagle3.Eagle3Proposer._run_mtp_fused_loop(
        p, None, torch.zeros(48, dtype=torch.int32), None, None, None, None, None, None
    )
    assert out is None and calls == []
