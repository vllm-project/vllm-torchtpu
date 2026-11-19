# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_POSTFWD_FUSED (q38 DEC-FB-3 P-a): the fused program is called with
dynamo's mark_dynamic stamps stripped by an owned alias (runner/owned_args.py,
port of dongruizi's 1e044e7d7), so a stamped input cannot make it trace a
dynamic dim and compile again mid-serve. CPU only: a probe stands in for the
compiled program and records what reaches it.
"""

import torch

from test_postfwd_runner_wiring import _call, _runner
from vllm_torchtpu.runner import owned_args, tpu_runner
from vllm_torchtpu.runner.tpu_runner_async_output import subtract_substitute_extract


class _Probe:
    """Stands in for `_postfwd_subst_compiled`: records, per call, whether any
    argument carries a dynamo stamp and which storages it received."""

    def __init__(self):
        self.stamped = []
        self.ptrs = []
        self.owned = []

    def __call__(self, *args):
        self.stamped.append([owned_args.is_stamped(a) for a in args])
        self.ptrs.append([a.data_ptr() for a in args])
        self.owned.append([owned_args.is_owned(a) for a in args])
        return subtract_substitute_extract(*args)


def _g2_args():
    g = torch.Generator().manual_seed(3)
    T, R, K = 16, 8, 3
    seq_lens = torch.randint(10, 5000, (R,), generator=g, dtype=torch.int32)
    positions = torch.randint(0, 5000, (T,), generator=g, dtype=torch.int64)
    num_rejected = torch.tensor([1, 2, 0, 3], dtype=torch.int32)
    seq_idx = torch.tensor([2, 0, 3, 1, -1, -1, -1, -1], dtype=torch.int32)
    pos_idx = torch.tensor([2] * 4 + [0] * 4 + [3] * 4 + [1] * 4, dtype=torch.int32)
    input_ids = torch.randint(1, 150000, (T,), generator=g, dtype=torch.int32)
    cur = torch.arange(T, dtype=torch.int32)
    pre = torch.tensor([2 * 4 + j for j in range(4)] + [j for j in range(4)] + [3 * 4 + j for j in range(4)]
                       + [1 * 4 + j for j in range(4)], dtype=torch.int32)
    next_tokens = torch.randint(1, 150000, (R * (1 + K),), generator=g, dtype=torch.int32)
    logits = torch.arange(T, dtype=torch.int32)
    target = torch.tensor([i * 4 + j for i in range(4) for j in range(K)] + [0] * 4, dtype=torch.int32)
    return [seq_lens, positions, num_rejected, seq_idx, pos_idx, input_ids, cur, pre, next_tokens, logits, target]


def test_fused_entry_strips_stamps_with_an_owned_alias(monkeypatch):
    probe = _Probe()
    monkeypatch.setattr(tpu_runner, "_postfwd_subst_compiled", probe)
    args = _g2_args()
    ref = subtract_substitute_extract(*[a.clone() for a in args])
    torch._dynamo.mark_dynamic(args[1], 0)  # positions, as the forward stamps them
    torch._dynamo.mark_dynamic(args[5], 0)  # input_ids
    assert owned_args.is_stamped(args[1]) and owned_args.is_stamped(args[5])
    out = tpu_runner._postfwd_subst_fused(*args)
    assert probe.stamped == [[False] * len(args)], "a stamped argument reached the compiled program"
    # The owned alias shares the caller's storage (no copy, no view program) ...
    assert probe.ptrs[0] == [a.data_ptr() for a in args]
    assert all(probe.owned[0])
    # ... and leaves the caller's own tensors stamped as they were.
    assert owned_args.is_stamped(args[1]) and owned_args.is_stamped(args[5])
    for a, b in zip(out, ref):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_fused_prepare_calls_the_stripping_entry(monkeypatch):
    """`_postfwd_fused_prepare` reaches the compiled program only through the
    stripping entry: stamped runner inputs arrive unstamped."""
    probe = _Probe()
    monkeypatch.setattr(tpu_runner, "_postfwd_subst_compiled", probe)
    runner = _runner(warmup_done=False)
    torch._dynamo.mark_dynamic(runner.input_ids, 0)
    torch._dynamo.mark_dynamic(runner.position_ids, 0)
    _call(runner)
    assert len(probe.stamped) == 1, "the fused program was not called"
    assert probe.stamped[0] == [False] * len(probe.stamped[0]), "a stamped runner input reached the program"
    assert runner._postfwd_stats == [1, 0]


def test_stamped_view_reaches_the_program_as_an_owned_copy(monkeypatch):
    probe = _Probe()
    monkeypatch.setattr(tpu_runner, "_postfwd_subst_compiled", probe)
    args = _g2_args()
    big = torch.cat([args[5], args[5]])
    view = big[: args[5].shape[0]]
    torch._dynamo.mark_dynamic(view, 0)
    assert not owned_args.is_owned(view)
    args[5] = view
    ref = subtract_substitute_extract(*[a.clone() for a in args])
    out = tpu_runner._postfwd_subst_fused(*args)
    assert probe.owned[0][5] and not probe.stamped[0][5]
    assert probe.ptrs[0][5] != big.data_ptr(), "the view's base buffer reached the program"
    for a, b in zip(out, ref):
        assert torch.equal(a, b)
