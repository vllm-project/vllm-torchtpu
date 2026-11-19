# SPDX-License-Identifier: Apache-2.0
"""Draft-carries port (q38 DEC-FB-3 P-c): dongruizi's 81af97b0d (VLLM_TPU_VERIFY_FUSED_READ_OFFSETS,
VLLM_TPU_DRAFT_CARRY_FUSED), a29f195d3 + 9efe59db6 (precompiles, read-offset fold follow-up), a678097bd (check mode) and
7086efb53 (VLLM_TPU_DRAFT_SEED_EXTRAS), cherry-picked as MM serves them. CPU only; the compiled bodies run eagerly.

1. Read-offset fold: the fused verify's per-row offsets and the runner's eager group scatter (both executed from the
   runner's own source text) leave the slot buffer exactly as the separate `_update_read_offsets_impl` does.
2. Carry fusion: `_draft_rows_carry_and_token` equals `_draft_gather_positions` + `_draft_propose_token`.
3. Seed extras: `_async_seed_inputs_impl(..., num_rows)` returns the first three outputs unchanged, plus
   `_padded_draft_rows(gather_indices, num_rows)` and `_maybe_pad_dim0(num_rejected[:num_rows], num_rows)`.
4. Flags off: the seed impl equals 240e25b41's; the fused verify body is 240e25b41's up to the new `num_reqs` branch; the
   three gates read their flags, which default off.
5. The two new precompiles (counted by a recorder) cover the decode call-site shape keys for 1..64 requests.
"""

import contextlib
import inspect
import json
import os
import pathlib
import textwrap
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import postfwd_ref_dc_240e25b41 as ref
from vllm_torchtpu import envs
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import (
    _get_padded_num_reqs_with_upper_limit,
    _get_padded_token_len,
    _update_read_offsets_impl,
)
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID
from vllm_torchtpu.spec_decode import eagle3
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer, _async_seed_inputs_impl, _maybe_pad_dim0

K = 3
PADDINGS = [16, 32, 64, 128, 256, 512]
MAX_REQS = 64
RUNNER_SRC = pathlib.Path(tpu_runner.__file__).read_text()
EAGLE3_SRC = pathlib.Path(eagle3.__file__).read_text()


def _block(src, start, end):
    assert src.count(start) == 1, (src.count(start), start[:60])
    i = src.index(start)
    j = src.index(end, i) + len(end)
    return textwrap.dedent(src[i:j])


TAIL = _block(
    RUNNER_SRC,
    "    num_valid = (next_tokens != INVALID_TOKEN_ID).sum(dim=1)\n    valid = torch.arange(next_tokens.shape[0]",
    "    row_offsets = torch.where(valid, (num_valid - 1).clamp(min=0), 0)\n",
)
FOLD = _block(RUNNER_SRC, "            groups = list(ro_groups)\n",
              "            self.mamba_slot_read_offsets.index_put_((idx,), vals)\n")


def _next_tokens(g, rows, R):
    nt = np.full((rows, K + 1), INVALID_TOKEN_ID, dtype=np.int32)
    acc = g.integers(0, K + 1, rows)
    for i in range(rows):
        nt[i, : acc[i] + 1] = g.integers(1, 150000, acc[i] + 1)
    return torch.from_numpy(nt)


# ---------------------------------------------------------------- 1. read-offset fold


@pytest.mark.parametrize("rows,width,R,ngroups", [(8, 8, 5, 1), (16, 8, 8, 1), (8, 16, 3, 3), (32, 64, 20, 2),
                                                  (64, 32, 32, 3), (16, 16, 16, 1)])
def test_read_offset_fold_equals_separate_scatter(rows, width, R, ngroups):
    g = np.random.default_rng(rows * 7 + width + R + ngroups)
    nt = _next_tokens(g, rows, R)
    num_reqs = torch.tensor(R)
    ns = {"torch": torch, "INVALID_TOKEN_ID": INVALID_TOKEN_ID, "next_tokens": nt, "num_reqs": num_reqs}
    exec(TAIL, ns)
    row_offsets = ns["row_offsets"]
    slots = g.permutation(np.arange(1, 4000))[: ngroups * width].reshape(ngroups, width).astype(np.int32)
    slots[:, R:] = 0  # padded rows hold the null block, as the runner's slot tensors do
    groups = [torch.from_numpy(s) for s in slots]
    buf0 = torch.from_numpy(g.integers(0, 9, 4000).astype(np.int32))
    stub = SimpleNamespace(mamba_slot_read_offsets=buf0.clone(), _verify_fused_checks_left=0,
                           _fused_read_offsets_applied=False)
    exec(FOLD, {"torch": torch, "self": stub, "ro_groups": groups, "row_offsets": row_offsets})
    sep = buf0.clone()
    _update_read_offsets_impl(sep, groups, nt, num_reqs)
    assert torch.equal(stub.mamba_slot_read_offsets, sep)
    assert not torch.equal(sep, buf0)  # the scatter is visible


# ---------------------------------------------------------------- 2. carry fusion


def _draft_model(with_d2t, H, V):
    g = torch.Generator().manual_seed(5)
    W = torch.randn(H, V, generator=g)
    if not with_d2t:
        return SimpleNamespace(compute_logits=lambda h: h.float() @ W)
    d2t = torch.randint(0, 50, (V,), generator=g)
    return SimpleNamespace(draft_id_to_target_id=d2t, lm_head=W, logits_processor=lambda lm_head, h: h.float() @ lm_head)


@pytest.mark.parametrize("with_d2t", [False, True])
@pytest.mark.parametrize("pos_ndim", [1, 2])
def test_carry_fused_equals_gather_plus_token(with_d2t, pos_ndim):
    H, V, T, p = 16, 40, 64, 16
    stub = SimpleNamespace(draft_model=_draft_model(with_d2t, H, V))
    # dongruizi's 48745d9b0 (lever 3) makes _draft_propose_token call self._propose_token_body.
    if hasattr(Eagle3Proposer, "_propose_token_body"):
        stub._propose_token_body = types.MethodType(Eagle3Proposer._propose_token_body, stub)
    g = torch.Generator().manual_seed(9)
    positions = torch.randint(0, 5000, (T,) if pos_ndim == 1 else (3, T), generator=g, dtype=torch.int32)
    indices = torch.randint(0, T, (p,), generator=g, dtype=torch.int64)
    indices[-3:] = -1  # rows this rank does not own / padding
    last_hidden = torch.randn(p, H, generator=g)
    fused_pos, fused_tok = inspect.unwrap(Eagle3Proposer._draft_rows_carry_and_token)(stub, positions, indices, last_hidden)
    sep_pos = inspect.unwrap(Eagle3Proposer._draft_gather_positions)(stub, positions, indices)
    sep_tok = inspect.unwrap(Eagle3Proposer._draft_propose_token)(stub, last_hidden)
    assert fused_pos.dtype == sep_pos.dtype and torch.equal(fused_pos, sep_pos)
    assert fused_tok.dtype == sep_tok.dtype and torch.equal(fused_tok, sep_tok)


# ---------------------------------------------------------------- 3. seed extras


def _seed_inputs(g, width, R, q, T):
    nt = _next_tokens(g, width, R)
    nd = np.zeros(width, dtype=np.int32)
    nd[:R] = K
    qsl = np.zeros(q, dtype=np.int32)
    for i in range(1, q):
        qsl[i] = min(i, R) * (K + 1)
    ids = torch.from_numpy(g.integers(1, 150000, T).astype(np.int32))
    return nt, torch.from_numpy(nd), torch.from_numpy(qsl), torch.tensor(R, dtype=torch.int64), ids


@pytest.mark.parametrize("width,R,p,num_rows", [(8, 5, 16, 64), (16, 12, 16, 64), (32, 20, 32, 16), (64, 40, 64, 64),
                                                (8, 8, 16, 8)])
def test_seed_extras_equal_separate_pads(width, R, p, num_rows):
    g = np.random.default_rng(width * 3 + R + num_rows)
    T = _get_padded_token_len(PADDINGS, R * (K + 1))
    args = _seed_inputs(g, width, R, MAX_REQS + 1, T)
    base = _async_seed_inputs_impl(*args, p)
    ext = _async_seed_inputs_impl(*args, p, num_rows)
    assert len(base) == 3 and len(ext) == 5
    for a, b in zip(ext[:3], base):
        assert a.dtype == b.dtype and torch.equal(a, b)
    rows_ref = Eagle3Proposer._padded_draft_rows(object(), base[1], num_rows)
    rej_ref = _maybe_pad_dim0(base[2][:num_rows], num_rows)
    assert ext[3].dtype == rows_ref.dtype and torch.equal(ext[3], rows_ref)
    assert ext[4].dtype == rej_ref.dtype and torch.equal(ext[4], rej_ref)


# ---------------------------------------------------------------- 4. flags off


def test_flag_off_seed_impl_is_240e25b41():
    g = np.random.default_rng(4)
    for width, R, p in ((8, 3, 16), (16, 16, 16), (64, 33, 64)):
        T = _get_padded_token_len(PADDINGS, R * (K + 1))
        args = _seed_inputs(g, width, R, MAX_REQS + 1, T)
        for a, b in zip(_async_seed_inputs_impl(*args, p), ref.old_async_seed_inputs_impl(*args, p)):
            assert a.dtype == b.dtype and torch.equal(a, b)


def test_flag_off_verify_body_is_240e25b41():
    old = ref.OLD_VERIFY_FUSED_IMPL_SRC
    new = _block(RUNNER_SRC, "def _verify_fused_impl(", "    return next_tokens, row_offsets\n")
    # 240e25b41 returned the rejection sampler's output directly; the port names it next_tokens and returns it when
    # num_reqs is None (flag off), so every line before that branch must be the old body with that one rename.
    old_body = old[old.index('    """'):].rstrip("\n").splitlines()
    assert old_body.count("    return _random_rejection_sample_impl(") == 1
    head = [("    next_tokens = _random_rejection_sample_impl(" if ln == "    return _random_rejection_sample_impl(" else ln)
            for ln in old_body]
    new_body = new[new.index('    """'):].splitlines()
    assert new_body[: len(head)] == head, "the fused verify body before the num_reqs branch changed"
    assert new_body[len(head): len(head) + 2] == ["    if num_reqs is None:", "        return next_tokens"]


def test_gates_read_their_flags_which_default_off(monkeypatch):
    for name in ("VLLM_TPU_VERIFY_FUSED_READ_OFFSETS", "VLLM_TPU_DRAFT_CARRY_FUSED", "VLLM_TPU_DRAFT_SEED_EXTRAS"):
        monkeypatch.delenv(name, raising=False)
        assert getattr(envs, name) is False, name
    assert "        fold_ro = (\n            envs.VLLM_TPU_VERIFY_FUSED_READ_OFFSETS\n" in RUNNER_SRC
    assert "                self._draft_rows\n                and envs.VLLM_TPU_DRAFT_CARRY_FUSED\n" in EAGLE3_SRC
    assert "                    self._draft_rows\n                    and envs.VLLM_TPU_DRAFT_SEED_EXTRAS\n" in EAGLE3_SRC


# ---------------------------------------------------------------- 5. precompile grids


def _eagle3_stub():
    e = object.__new__(Eagle3Proposer)
    e.runner = SimpleNamespace(num_tokens_paddings=list(PADDINGS), num_reqs_paddings=[8, 16, 32, 64],
                               num_reqs_max_model_len=MAX_REQS, num_reqs_most_model_len=None, max_num_reqs=MAX_REQS,
                               device=torch.device("cpu"), _hidden_states_dtype=torch.bfloat16,
                               _precompile_timed=lambda name: contextlib.nullcontext())
    e.speculative_config = SimpleNamespace(num_speculative_tokens=K)
    e._draft_rows = True
    e._draft_hidden_size = lambda: 16
    return e


def test_precompiles_cover_decode_call_sites(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_DRAFT_SEED_EXTRAS", "1")
    seed, carry = set(), set()
    monkeypatch.setattr(eagle3, "synchronize_tensors", lambda *a, **k: None)
    monkeypatch.setattr(eagle3, "_async_seed_inputs_compiled",
                        lambda nt, nd, qsl, n, ids, p, num_rows=0: seed.add(
                            (nt.shape[0], ids.shape[0], p, qsl.shape[0], num_rows)) or (None,) * 5)
    e = _eagle3_stub()
    e._draft_rows_carry_and_token = lambda pos, idx, hid: carry.add((pos.shape[0], idx.shape[0])) or (None, None)
    Eagle3Proposer._precompile_async_seed_inputs(e)
    Eagle3Proposer._precompile_draft_rows_carry_and_token(e)
    want_seed, want_carry = set(), set()
    for R in range(1, MAX_REQS + 1):
        width = _get_padded_num_reqs_with_upper_limit(R, MAX_REQS)
        T = _get_padded_token_len(PADDINGS, R * (K + 1))
        p = _get_padded_token_len(PADDINGS, R)
        want_seed.add((width, T, p, MAX_REQS + 1, MAX_REQS))
        want_carry.add((T, p))
    counts = {"seed_programs": len(seed), "carry_programs": len(carry), "seed_decode_keys": len(want_seed),
              "carry_decode_keys": len(want_carry)}
    out = os.environ.get("Q38_DC_GRID_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(counts, f)
    assert want_seed <= seed, (counts, sorted(want_seed - seed)[:3])
    assert want_carry <= carry, (counts, sorted(want_carry - carry)[:3])


def test_fold_precompile_warms_the_num_reqs_variant_only():
    """q38 DEC-FB-3 (DECODE-LEAD 11-12, COV51 U4): with VERIFY_FUSED_READ_OFFSETS and a mamba read-offset buffer, the
    verify warmup compiles `_verify_fused_compiled` with the trailing num_reqs tensor (the variant the fold calls) and
    skips the plain variant; otherwise only the plain variant, with the same leading arguments."""
    import ast

    tree = ast.parse(RUNNER_SRC)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_precompile_verify_fused")
    assign = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                  and getattr(n.targets[0], "id", None) == "fold_ro")
    assert ast.unparse(assign.value) == (
        "bool(envs.VLLM_TPU_VERIFY_FUSED_READ_OFFSETS and self.mamba_slot_read_offsets is not None)")
    loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and ast.unparse(n.target) == "topk_candidates")
    branch = loop.body[0]
    assert isinstance(branch, ast.If) and ast.unparse(branch.test) == "fold_ro"
    assert isinstance(branch.body[-1], ast.Continue)
    calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_verify_fused_compiled"]
    assert len(calls) == 2
    fold_call = next(c for c in calls if any(c is n for n in ast.walk(branch)))
    plain_call = next(c for c in calls if c is not fold_call)
    fa = [ast.unparse(a) for a in fold_call.args]
    pa = [ast.unparse(a) for a in plain_call.args]
    assert fa[:-1] == pa and fa[-1] == "self._num_reqs_tensor(1)", (fa, pa)
