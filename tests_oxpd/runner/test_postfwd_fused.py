# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_POSTFWD_FUSED (q38 DEC-FB-2 G2): `subtract_substitute_extract` must
give the same integers, bit for bit, as the separate calls it replaces:
`subtract_num_rejected_tokens`, then the runner's async placeholder
substitution (`_substitute_placeholder_token` on the `_warmed_next_tokens`
source, or the 1:1 fast path), then `extract_draft_token_ids`. CPU only.
"""

import pathlib
import re

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner.tpu_runner_async_output import (
    extract_draft_token_ids,
    subtract_num_rejected_tokens,
    subtract_substitute_extract,
)

K = 3
STRIDE = 1 + K
MAX_REQS = 64
BOUND = MAX_REQS * STRIDE  # the async-spec source length sample_tokens pads to
REQS = [1, 2, 4, 8, 16, 24, 32, 48, 64]
TOKEN_BUCKETS = [16, 32, 64, 128, 256, 512]
RUNNER = pathlib.Path(__file__).resolve().parents[2] / "src/vllm_torchtpu/runner/tpu_runner.py"


def _ref_substitute(input_ids, cur, pre, next_tokens):
    """`tpu_runner._substitute_placeholder_token`'s body (checked against the
    source text in test_reference_matches_runner_source)."""
    input_ids = input_ids.clone()
    token_in_tpu_cur_input_indices = cur
    token_in_tpu_pre_next_tokens_indices = pre
    mask = token_in_tpu_pre_next_tokens_indices > -1
    # clamp_min(0) gives a safe in-range gather index for the -1 sentinel
    # slots; their gathered values are discarded by `mask` in `where`.
    safe_idx = torch.clamp_min(token_in_tpu_pre_next_tokens_indices, 0)
    new_token_values = next_tokens[safe_idx].to(input_ids.dtype)
    original_values = input_ids[token_in_tpu_cur_input_indices]
    update_values = torch.where(mask, new_token_values, original_values)
    input_ids.scatter_(0, token_in_tpu_cur_input_indices, update_values)
    return input_ids


def _warmed(next_tokens, warm_len):
    """`TPUModelRunner._warmed_next_tokens` for a target length `warm_len`."""
    n = next_tokens.numel()
    if next_tokens.dim() == 1 and n == warm_len and next_tokens.dtype == torch.int64:
        return next_tokens
    flat = next_tokens.reshape(-1).to(torch.int64)
    if warm_len > n:
        flat = torch.nn.functional.pad(flat, (0, warm_len - n))
    return flat


def _padded_indices(input_len, cur_np, pre_np):
    """`_apply_async_token_substitution`'s padded device indices."""
    missing = np.ones(input_len, dtype=bool)
    missing[cur_np] = False
    cur = np.concatenate((cur_np, np.flatnonzero(missing).astype(np.int32)))
    pre = np.pad(pre_np, (0, input_len - len(pre_np)), mode="constant", constant_values=-1)
    return torch.from_numpy(cur), torch.from_numpy(pre)


def _ref(case, path):
    seq_lens, positions = subtract_num_rejected_tokens(
        case["seq_lens"], case["positions"], case["num_rejected"], case["seq_idx"], case["pos_idx"]
    )
    ids, nt = case["input_ids"], case["next_tokens"]
    if path == "fast1":
        # Fast path 1: identity mapping over every slot.
        assert np.array_equal(case["cur_np"], np.arange(len(ids))) and np.array_equal(
            case["pre_np"], np.arange(len(ids))
        )
        sub = nt.flatten().to(ids.dtype)[: len(ids)]
    else:
        cur, pre = _padded_indices(len(ids), case["cur_np"], case["pre_np"])
        src = _warmed(nt, case["warm_len"]) if path == "warmed" else nt
        sub = _ref_substitute(ids, cur, pre, src)
    drafts = extract_draft_token_ids(sub, case["logits_idx"], case["target_idx"])
    return seq_lens, positions, sub, drafts


def _fused(case):
    cur, pre = _padded_indices(len(case["input_ids"]), case["cur_np"], case["pre_np"])
    return subtract_substitute_extract(
        case["seq_lens"], case["positions"], case["num_rejected"], case["seq_idx"], case["pos_idx"],
        case["input_ids"], cur, pre, case["next_tokens"], case["logits_idx"], case["target_idx"],
    )


def _case(seed, padded_reqs, n_real, mapped, identity, pos_dtype, nt_dtype, warm_extra):
    """One async-spec chunk: n_real requests x STRIDE tokens in a token bucket;
    `mapped` of them have a source in the previous step (the rest keep their
    host tokens); rejected counts 0..K."""
    g = np.random.default_rng(seed)
    tokens = n_real * STRIDE
    T = next(b for b in TOKEN_BUCKETS if b >= tokens)
    if identity:
        assert n_real * STRIDE == T and mapped == n_real
    target_reqs = MAX_REQS
    seq_lens = torch.from_numpy(g.integers(10, 5000, target_reqs).astype(np.int32))
    positions = torch.from_numpy(g.integers(0, 5000, T).astype(np.int64)).to(pos_dtype)
    num_rejected = torch.from_numpy(g.integers(0, K + 1, padded_reqs).astype(np.int32))
    num_rejected[0] = max(1, int(num_rejected[0]))  # at least one subtraction is visible
    # Request i of this chunk sits at source position src_pos[i] in the previous step.
    src_pos = g.permutation(padded_reqs)[:n_real] if not identity else np.arange(n_real)
    is_mapped = np.zeros(n_real, dtype=bool)
    is_mapped[g.permutation(n_real)[:mapped] if not identity else np.arange(n_real)] = True
    seq_idx = np.full(target_reqs, -1, dtype=np.int32)
    pos_idx = np.full(T, -1, dtype=np.int32)
    cur_l, pre_l = [], []
    for i in range(n_real):
        if not is_mapped[i]:
            continue
        seq_idx[i] = src_pos[i]
        pos_idx[i * STRIDE : (i + 1) * STRIDE] = src_pos[i]
        for j in range(STRIDE):
            cur_l.append(i * STRIDE + j)
            pre_l.append(int(src_pos[i]) * STRIDE + j)
    input_ids = torch.from_numpy(g.integers(1, 150000, T).astype(np.int32))
    n_src = padded_reqs * STRIDE
    next_tokens = torch.from_numpy(g.integers(1, 150000, n_src).astype(np.int64)).to(nt_dtype)
    next_tokens = torch.nn.functional.pad(next_tokens, (0, BOUND - n_src))  # sample_tokens' bound pad
    # Logits rows: every token of a real request; target rows = the K draft
    # positions (their +1 is in range); the padding repeats row 0.
    L = T
    logits_idx = np.zeros(L, dtype=np.int32)
    logits_idx[:tokens] = np.arange(tokens, dtype=np.int32)
    target_idx = np.zeros(L, dtype=np.int32)
    tl = [i * STRIDE + j for i in range(n_real) for j in range(K)]
    target_idx[: len(tl)] = tl
    return {
        "seq_lens": seq_lens,
        "positions": positions,
        "num_rejected": num_rejected,
        "seq_idx": torch.from_numpy(seq_idx),
        "pos_idx": torch.from_numpy(pos_idx),
        "input_ids": input_ids,
        "cur_np": np.array(cur_l, dtype=np.int32),
        "pre_np": np.array(pre_l, dtype=np.int32),
        "next_tokens": next_tokens,
        "warm_len": BOUND + warm_extra,
        "logits_idx": torch.from_numpy(logits_idx),
        "target_idx": torch.from_numpy(target_idx),
    }


def _cases():
    out = []
    seed = 0
    for r in REQS:
        for n_real in sorted({1, max(1, r // 2), r}):
            if n_real * STRIDE > TOKEN_BUCKETS[-1]:
                continue
            for mapped in sorted({1, n_real}):
                for pos_dtype in (torch.int32, torch.int64):
                    for nt_dtype in (torch.int32, torch.int64):
                        for warm_extra in (0, 16):  # no-op pad and a real pad
                            seed += 1
                            out.append((seed, r, n_real, mapped, False, pos_dtype, nt_dtype, warm_extra))
    # Fast path 1: identity over the whole bucket (n_real * STRIDE == bucket).
    for r, n_real in ((4, 4), (8, 8), (16, 16), (32, 32), (64, 64)):
        seed += 1
        out.append((seed, r, n_real, n_real, True, torch.int32, torch.int32, 0))
    return out


CASES = _cases()


@pytest.mark.parametrize("args", CASES, ids=[f"s{c[0]}r{c[1]}n{c[2]}m{c[3]}{'id' if c[4] else ''}" for c in CASES])
def test_fused_equals_separate(args):
    case = _case(*args)
    fused = _fused(case)
    paths = ("fast1",) if args[4] else ("warmed", "raw")
    for path in paths:
        ref = _ref(case, path)
        for name, a, b in zip(("seq_lens", "positions", "input_ids", "drafts"), fused, ref):
            assert a.dtype == b.dtype and a.shape == b.shape, (path, name, a.dtype, b.dtype, a.shape, b.shape)
            assert torch.equal(a, b), (path, name)


def test_sweep_exercises_every_term():
    """The sweep must make each term visible (else a mutant could survive):
    a nonzero subtraction, a kept (-1) slot whose value differs from the
    source's first token, a substituted slot read from source index 0."""
    sub_seen = keep_seen = src0_seen = False
    for args in CASES:
        case = _case(*args)
        s, _, _, _ = _fused(case)
        sub_seen |= not torch.equal(s, case["seq_lens"])
        cur, pre = _padded_indices(len(case["input_ids"]), case["cur_np"], case["pre_np"])
        kept = pre < 0
        keep_seen |= bool((case["input_ids"][cur[kept]] != case["next_tokens"][0].to(torch.int32)).any())
        src0_seen |= bool((pre == 0).any())
    assert sub_seen and keep_seen and src0_seen


def test_reference_matches_runner_source():
    """`_ref_substitute` is the runner's `_substitute_placeholder_token` body."""
    src = RUNNER.read_text()
    m = re.search(r"def _substitute_placeholder_token\(.*?\n(    mask = .*?input_ids\.scatter_\(0, token_in_tpu_cur_input_indices, update_values\)\n)", src, re.S)
    assert m, "the runner's _substitute_placeholder_token body was not found"
    body = [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]
    import inspect

    ref = inspect.getsource(_ref_substitute)
    ref_body = ref[ref.index("    mask = "): ref.index("    return input_ids")]
    assert body == [ln.strip() for ln in ref_body.splitlines() if ln.strip()]
