# SPDX-License-Identifier: Apache-2.0
import logging
from types import SimpleNamespace

import torch

from vllm_torchtpu.runner import compile_watch


def test_post_warmup_compile_logs_one_warning(caplog, monkeypatch):
    monkeypatch.setattr(compile_watch, "_installed", False)
    monkeypatch.setenv("VLLM_TPU_LOG_POST_WARMUP_COMPILES", "1")
    compile_watch.install_post_warmup_compile_log()

    @torch.compile(backend="eager", fullgraph=True, dynamic=False)
    def f(x):
        return x + 1

    with caplog.at_level(logging.WARNING):
        f(torch.ones(3))
        f(torch.ones(3))  # cached: no second compile
        f(torch.ones(5))  # new shape: a recompile
    lines = [r.getMessage() for r in caplog.records if "Post-warmup compile" in r.getMessage()]
    assert len(lines) == 2, lines
    assert "#1" in lines[0] and "#2" in lines[1]


def test_warmed_next_tokens_pads_to_a_warmed_length():
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    runner = SimpleNamespace(
        num_reqs_paddings=[8, 16, 32],
        _is_async_drafter=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        max_num_reqs=32,
    )
    runner._substitute_next_lens = lambda: TPUModelRunner._substitute_next_lens(runner)
    warm = lambda t: TPUModelRunner._warmed_next_tokens(runner, t)  # noqa: E731
    t = torch.arange(24, dtype=torch.int64)
    out = warm(t)
    assert out.shape == (32,) and torch.equal(out[:24], t) and not out[24:].any()
    same = torch.arange(16, dtype=torch.int64)
    assert warm(same) is same
    big = torch.arange(200, dtype=torch.int64)
    assert warm(big) is big  # no warmed length fits: unchanged
    two_d = torch.arange(20, dtype=torch.int32).view(5, 4)
    out = warm(two_d)
    assert out.dtype == torch.int64 and out.shape == (32,)
    assert torch.equal(out[:20], two_d.reshape(-1).long())
