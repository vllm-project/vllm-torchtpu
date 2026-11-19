"""Carry-rows-fresh port (q38 DEC-FB-3, lever 3): dongruizi's 48745d9b0 (VLLM_TPU_DRAFT_ROWS_FRESH) and 7a0a3958e
(VLLM_TPU_DRAFT_CARRY_ROWS_FRESH), cherry-picked onto the P-c chain (68e6e31c9). CPU only; compiled bodies run eagerly.
1. The fresh carry program on step 0's untrimmed [capacity, h] outputs equals `_draft_rows_carry_and_token` on the
   [:p] views, and its hidden carry equals hidden[:p] in a buffer of its own (not a view of the untrimmed output).
2. The same for ROWS_FRESH's `_draft_token_and_rows` against `_draft_propose_token` on the view.
3. Flags off: eagle3.py is 68e6e31c9's, structurally (crf_flagoff_ast.check: every flag-guarded branch resolved to its
   flags-off arm, plus the one `_propose_token_body` refactor).
4. The gates read their flags, which default off, and the step-0 wiring passes the untrimmed outputs in order.
5. The new precompiles (counted by a recorder) cover the decode call-site shape keys for 1..64 requests.
"""
import contextlib
import inspect
import json
import os
import pathlib
import types
from types import SimpleNamespace

import pytest
import torch

import crf_flagoff_ast
import importlib.util as _ilu
from vllm_torchtpu import envs
from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len
from vllm_torchtpu.spec_decode import eagle3
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

K = 3
PADDINGS = [16, 32, 64, 128, 256, 512]
MAX_REQS = 64
HERE = pathlib.Path(__file__).parent
EAGLE3_SRC = pathlib.Path(eagle3.__file__).read_text()
REF_SRC = (HERE / "eagle3_ref_68e6e31c9.py.txt").read_text()
# q38 DEC-FB-3 lever 2 (DRAFT_TOP_TOKENS) sits on top of lever 3: resolve its switch to off first
# (tests/spec_decode/dtt_flagoff_ast.py), then lever 3's flags.
_spec = _ilu.spec_from_file_location("dtt_flagoff_ast", HERE.parent / "spec_decode" / "dtt_flagoff_ast.py")
dtt_flagoff_ast = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(dtt_flagoff_ast)
# q38 L4 DRAFT-ASM sits on top of both: resolve its flag to off first (tests/q38_draftasm/draftasm_flagoff_ast.py).
_spec_a = _ilu.spec_from_file_location("draftasm_flagoff_ast", HERE.parent / "q38_draftasm" / "draftasm_flagoff_ast.py")
draftasm_flagoff_ast = _ilu.module_from_spec(_spec_a)
_spec_a.loader.exec_module(draftasm_flagoff_ast)


def _draft_model(with_d2t, H, V):
    g = torch.Generator().manual_seed(5)
    W = torch.randn(H, V, generator=g)
    if not with_d2t:
        return SimpleNamespace(compute_logits=lambda h: h.float() @ W)
    d2t = torch.randint(0, 50, (V,), generator=g)
    return SimpleNamespace(draft_id_to_target_id=d2t, lm_head=W, logits_processor=lambda lm_head, h: h.float() @ lm_head)


def _stub(with_d2t, H, V):
    s = SimpleNamespace(draft_model=_draft_model(with_d2t, H, V))
    s._carry_and_token_body = types.MethodType(Eagle3Proposer._carry_and_token_body, s)
    s._propose_token_body = types.MethodType(Eagle3Proposer._propose_token_body, s)
    return s


def _own_buffer(x, of):
    return x.untyped_storage().data_ptr() != of.untyped_storage().data_ptr()


# ---------------------------------------------------------------- 1. fresh carry


@pytest.mark.parametrize("with_d2t", [False, True])
@pytest.mark.parametrize("pos_ndim", [1, 2])
@pytest.mark.parametrize("cap,p", [(64, 16), (64, 32), (32, 16), (16, 8)])
def test_carry_rows_fresh_equals_carry_on_views(with_d2t, pos_ndim, cap, p):
    H, V, T = 16, 40, 128
    s = _stub(with_d2t, H, V)
    g = torch.Generator().manual_seed(cap * 31 + p + pos_ndim + int(with_d2t))
    positions = torch.randint(0, 5000, (T,) if pos_ndim == 1 else (3, T), generator=g, dtype=torch.int32)
    indices = torch.randint(0, T, (p,), generator=g, dtype=torch.int64)
    indices[-3:] = -1  # rows this rank does not own / padding
    last_full = torch.randn(cap, H, generator=g)
    hidden_full = torch.randn(cap, H, generator=g)
    f_pos, f_tok, f_hid = inspect.unwrap(Eagle3Proposer._draft_rows_carry_token_fresh)(
        s, positions, indices, last_full, hidden_full)
    o_pos, o_tok = inspect.unwrap(Eagle3Proposer._draft_rows_carry_and_token)(s, positions, indices, last_full[:p])
    assert f_pos.dtype == o_pos.dtype and torch.equal(f_pos, o_pos)
    assert f_tok.dtype == o_tok.dtype and torch.equal(f_tok, o_tok)
    assert f_hid.shape == (p, H) and torch.equal(f_hid, hidden_full[:p])
    assert _own_buffer(f_hid, hidden_full), "the hidden carry must leave as its own buffer, not a view"


# ---------------------------------------------------------------- 2. rows fresh (propose path)


@pytest.mark.parametrize("with_d2t", [False, True])
@pytest.mark.parametrize("cap,rows", [(64, 16), (32, 8), (16, 8)])
def test_rows_fresh_equals_propose_on_views(with_d2t, cap, rows):
    H, V = 16, 40
    s = _stub(with_d2t, H, V)
    g = torch.Generator().manual_seed(cap * 7 + rows + int(with_d2t))
    last_full, hidden_full = torch.randn(cap, H, generator=g), torch.randn(cap, H, generator=g)
    tok, hid = inspect.unwrap(Eagle3Proposer._draft_token_and_rows)(s, last_full, hidden_full, rows)
    ref = inspect.unwrap(Eagle3Proposer._draft_propose_token)(s, last_full[:rows])
    assert tok.dtype == ref.dtype and torch.equal(tok, ref)
    assert torch.equal(hid, hidden_full[:rows]) and _own_buffer(hid, hidden_full)


# ---------------------------------------------------------------- 3. flags off


def test_flags_off_eagle3_is_68e6e31c9():
    diffs = crf_flagoff_ast.check(
        dtt_flagoff_ast.off_source(draftasm_flagoff_ast.off_source(EAGLE3_SRC)), REF_SRC
    )
    assert not diffs, diffs


# ---------------------------------------------------------------- 4. gates and wiring


def test_gates_read_their_flags_which_default_off(monkeypatch):
    for name in ("VLLM_TPU_DRAFT_ROWS_FRESH", "VLLM_TPU_DRAFT_CARRY_ROWS_FRESH"):
        monkeypatch.delenv(name, raising=False)
        assert getattr(envs, name) is False, name
    # untrimmed is captured (either flag) only when step 0's outputs are wider than the carry rows
    assert ("                    (\n                        envs.VLLM_TPU_DRAFT_ROWS_FRESH\n"
            "                        or envs.VLLM_TPU_DRAFT_CARRY_ROWS_FRESH\n                    )\n"
            "                    and last_hidden.shape[0] > rows\n                    and hidden.shape[0] > rows\n"
            "                ):\n                    untrimmed = (last_hidden, hidden, rows)\n") in EAGLE3_SRC
    # the fresh carry runs only with its flag, a capture, and matching carry rows; it gets (positions, indices,
    # untrimmed last_hidden, untrimmed hidden) in that order and its hidden carry replaces the view
    assert ("                    envs.VLLM_TPU_DRAFT_CARRY_ROWS_FRESH\n                    and untrimmed is not None\n"
            "                    and untrimmed[2] == last_token_indices.shape[0]\n                ):\n") in EAGLE3_SRC
    assert ("                        positions_carry, fused_token0, hidden_carry = (\n"
            "                            self._draft_rows_carry_token_fresh(\n"
            "                                positions,\n                                last_token_indices,\n"
            "                                untrimmed[0],\n                                untrimmed[1],\n") in EAGLE3_SRC


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
    monkeypatch.setenv("VLLM_TPU_DRAFT_CARRY_ROWS_FRESH", "1")
    monkeypatch.setenv("VLLM_TPU_DRAFT_ROWS_FRESH", "1")
    monkeypatch.setattr(eagle3, "synchronize_tensors", lambda *a, **k: None)
    fresh, rows_fresh = set(), set()
    e = _eagle3_stub()
    e._draft_rows_carry_and_token = lambda pos, idx, hid: (None, None)
    e._draft_rows_carry_token_fresh = lambda pos, idx, lh, h: fresh.add(
        (pos.shape[0], idx.shape[0], lh.shape[0], h.shape[0])) or (None, None, None)
    e._draft_propose_token = lambda h: None
    e._draft_token_and_rows = lambda lh, h, rows: rows_fresh.add((lh.shape[0], rows)) or (None, None)
    Eagle3Proposer._precompile_draft_rows_carry_and_token(e)
    Eagle3Proposer._precompile_compute_logits(e)
    want_fresh, want_rows = set(), set()
    for R in range(1, MAX_REQS + 1):
        T = _get_padded_token_len(PADDINGS, R * (K + 1))
        p = _get_padded_token_len(PADDINGS, R)
        if p < MAX_REQS:  # step 0's outputs are [MAX_REQS, h]; untrimmed only when wider than the carry rows
            want_fresh.add((T, p, MAX_REQS, MAX_REQS))
            want_rows.add((MAX_REQS, p))
    counts = {"fresh_programs": len(fresh), "fresh_decode_keys": len(want_fresh), "rows_fresh_programs": len(rows_fresh),
              "rows_fresh_decode_keys": len(want_rows)}
    out = os.environ.get("Q38_CRF_GRID_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(counts, f)
    assert want_fresh <= fresh, (counts, sorted(want_fresh - fresh)[:3])
    assert want_rows <= rows_fresh, (counts, sorted(want_rows - rows_fresh)[:3])


# ---------------------------------------------------------------- 6. ROWS_FRESH propose branch (COV51 U2)


def test_rows_fresh_branch_wiring():
    """q38 DEC-FB-3 (DECODE-LEAD 11-12, COV51 U2): without the carry fusion, a row-selected step 0 with an untrimmed
    capture and the default sequence layout takes its first token and hidden carry from `_draft_token_and_rows` on
    (untrimmed last_hidden, untrimmed hidden, rows), in that order; otherwise the views stay as they were."""
    want = (
        "            elif self._draft_rows:\n"
        "                # The draft already returned one row per last token.\n"
        "                hidden_carry, last_hidden_carry = hidden, last_hidden\n"
        "                if untrimmed is not None and not has_non_default_layout:\n"
    )
    assert EAGLE3_SRC.count(want) == 1
    i = EAGLE3_SRC.index(want) + len(want)
    seg = EAGLE3_SRC[i:EAGLE3_SRC.index("                positions_carry = self._draft_gather_positions(", i)]
    assert ("                    with set_model_tag(\"eagle_head\"):\n"
            "                        fused_token0, hidden_carry = self._draft_token_and_rows(\n"
            "                            *untrimmed\n"
            "                        )\n") in seg
    # untrimmed is (last_hidden, hidden, rows), as _draft_token_and_rows takes them
    assert "                    untrimmed = (last_hidden, hidden, rows)\n" in EAGLE3_SRC
    sig = inspect.signature(inspect.unwrap(Eagle3Proposer._draft_token_and_rows))
    assert list(sig.parameters) == ["self", "last_hidden", "hidden", "rows"]
