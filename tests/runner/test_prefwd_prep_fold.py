# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PREFWD_PREP_FOLD: the folded pre-forward prep program (rejected-count
correction + 1:1 substitution + verified-draft gather + qwen4 fused step
inputs) is bitwise the unfolded sequence, mutates nothing, and runs only on
eligible steps whose shapes warmup compiled."""

from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.qwen4_exp import inputs as inputs_mod
from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.model_adapters import ModelInputSnapshot
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _get_padded_num_reqs_with_upper_limit,
    _prep_fold_impl,
    _prep_fold_key,
    _prep_fold_reference,
    _prep_fold_scatter_shapes,
    _prep_fold_shapes,
    _subst_prefix_impl,
    _substitute_placeholder_token_impl,
)
from vllm_torchtpu.runner.tpu_runner_async_output import (
    extract_draft_token_ids,
    subtract_num_rejected_tokens,
)

NGRAM = q4exp_text_config().ngram_size
EOS = q4exp_text_config().eos_token_id
MAX_REQS = 16


def _fold_args(n, k, src_dtype, pattern, window, pooled, seed=0, slots=MAX_REQS):
    """One steady verify step's folded-program inputs: `n` live requests of
    `k + 1` tokens filling the token bucket exactly, `slots` seq_lens slots."""
    g = torch.Generator().manual_seed(seed * 1000 + n * 10 + k)
    width = k + 1
    tokens = n * width
    spec_window = width
    seq_lens = torch.zeros(slots, dtype=torch.int32)
    seq_lens[:n] = torch.randint(20, 200, (n,), generator=g, dtype=torch.int32)
    positions = torch.zeros(tokens, dtype=torch.int64)
    for i in range(n):
        positions[i * width : (i + 1) * width] = (
            seq_lens[i] - width + torch.arange(width)
        )
    rows = (
        n
        if pattern != "padded_rows"
        else _get_padded_num_reqs_with_upper_limit(max(n - 1, 1), MAX_REQS)
    )
    if pattern == "all_accepted":
        rejected = torch.zeros(rows, dtype=torch.int32)
    elif pattern == "all_rejected":
        rejected = torch.full((rows,), k, dtype=torch.int32)
    else:
        rejected = torch.randint(0, k + 1, (rows,), generator=g, dtype=torch.int32)
    seq_idx = torch.full((slots,), -1, dtype=torch.int32)
    pos_idx = torch.full((tokens,), -1, dtype=torch.int32)
    for i in range(n):
        p = i if pattern != "padded_rows" else min(i, rows - 1)
        if pattern == "neg_slots" and i % 2:
            continue
        seq_idx[i] = p
        pos_idx[i * width : (i + 1) * width] = p
    bound = MAX_REQS * width
    src = torch.randint(100, 50000, (bound,), generator=g).to(src_dtype)
    final = torch.arange(tokens, dtype=torch.int32)
    target = torch.zeros(tokens, dtype=torch.int32)
    draft_pos = [i * width + j for i in range(n) for j in range(k)]
    target[: len(draft_pos)] = torch.tensor(draft_pos, dtype=torch.int32)
    starts = torch.zeros(slots + 1, dtype=torch.int32)
    starts[1 : n + 1] = torch.arange(1, n + 1, dtype=torch.int32) * width
    starts[n + 1 :] = starts[n]
    state = torch.randint(1, 50, (slots,), generator=g, dtype=torch.int32)
    if window:
        w = k + NGRAM - 1
        store = torch.randint(100, 999, (slots, w + 1), generator=g)
        store[:, w] = (seq_lens.to(torch.int64) - w).clamp(min=-3)
    else:
        store = torch.randint(100, 999, (MAX_REQS, 256), generator=g).to(torch.int32)
    scalars = torch.full((2 + slots,), -1, dtype=torch.int64)
    scalars[0], scalars[1] = 0, n
    scalars[2 : 2 + n] = torch.randperm(n, generator=g)
    rec_w = NGRAM - 1 + spec_window
    previous = (
        torch.randint(100, 999, (MAX_REQS + 1, rec_w), generator=g),
        torch.randint(0, 200, (MAX_REQS + 1,), generator=g),
        torch.randint(0, spec_window + 1, (MAX_REQS + 1,), generator=g),
    )
    current = tuple(t.clone() for t in previous)
    args = (
        seq_lens,
        positions,
        rejected,
        seq_idx,
        pos_idx,
        src,
        final,
        target,
        starts,
        starts if pooled else state,
        store,
        scalars,
        previous,
        current,
    )
    static = dict(
        num_tokens=tokens,
        ids_dtype=torch.int32,
        window=window,
        pooled=pooled,
        spec_window=spec_window,
        eos_token_id=EOS,
        ngram_size=NGRAM,
    )
    return args, static


def _unfolded(args, static):
    """The unfolded step: correction, 1:1 substitution, draft gather, fused
    step inputs, each as its own (existing) function."""
    seq_lens, positions = subtract_num_rejected_tokens(*args[:5])
    ids = _subst_prefix_impl(args[5], static["num_tokens"], static["ids_dtype"])
    drafts = extract_draft_token_ids(ids, args[6], args[7])
    starts, state, store, scalars, previous, current = args[8:]
    prepped = inputs_mod.fused_step_inputs(
        starts,
        seq_lens,
        state,
        store,
        scalars,
        previous,
        current,
        ids,
        window=static["window"],
        pooled=static["pooled"],
        spec_window=static["spec_window"],
        eos_token_id=static["eos_token_id"],
        ngram_size=static["ngram_size"],
        functional=True,
    )
    return (seq_lens, positions, ids, drafts, *prepped)


def _assert_bitwise(got, ref):
    assert len(got) == len(ref) == 9
    for i, (g, r) in enumerate(zip(got[:8], ref[:8])):
        assert g.dtype == r.dtype and g.shape == r.shape, i
        assert torch.equal(g, r), i
    for g, r in zip(got[8], ref[8]):
        assert g.dtype == r.dtype and g.shape == r.shape
        assert torch.equal(g, r)


@pytest.mark.parametrize(
    "pattern", ["all_accepted", "all_rejected", "mixed", "neg_slots", "padded_rows"]
)
@pytest.mark.parametrize("src_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("n", [1, 4, 8, 16])
@pytest.mark.parametrize("window,pooled", [(False, False), (True, True)])
def test_fold_impl_bitwise_vs_unfolded(n, k, src_dtype, pattern, window, pooled):
    args, static = _fold_args(n, k, src_dtype, pattern, window, pooled)
    ref = _unfolded(args, static)
    _assert_bitwise(_prep_fold_impl(*args, **static), ref)
    # The CHECK reference is the same sequence; `fault` moves seq_lens.
    _assert_bitwise(_prep_fold_reference(*args, **static), ref)
    faulted = _prep_fold_reference(*args, fault=True, **static)
    assert not torch.equal(faulted[0], ref[0])


def test_fold_no_input_mutation():
    args, static = _fold_args(4, 3, torch.int64, "mixed", True, False)
    leaves = [t for a in args for t in (a if isinstance(a, tuple) else (a,))]
    before = [t.clone() for t in leaves]
    out = _prep_fold_impl(*args, **static)
    for t, b in zip(leaves, before):
        assert torch.equal(t, b)
    in_ptrs = {t.data_ptr() for t in leaves}
    for t in [*out[:8], *out[8]]:
        assert t.data_ptr() not in in_ptrs


def test_prep_fold_shapes_cover_steady_keys():
    """Every steady (rejected rows, token bucket) a c16 MTP K=3 server can fold
    (incl. the small buckets 4 and 8) is warmed; requests counts whose verify
    tokens do not fill a bucket exactly are not (their substitution is not 1:1,
    so they never fold)."""
    from vllm_torchtpu.platforms.tpu_platform import _get_token_paddings

    paddings = _get_token_paddings(4, 2048)
    width = 4
    shapes = _prep_fold_shapes(MAX_REQS, width, paddings)
    for n in range(1, MAX_REQS + 1):
        tokens = n * width
        exact = tokens in paddings
        for rows in (n, _get_padded_num_reqs_with_upper_limit(n, MAX_REQS)):
            assert ((rows, tokens) in shapes) == exact, (n, rows)
    assert {(1, 4), (2, 8), (4, 16), (16, 64)} <= shapes
    # The rejected counts carry the previous step's request count: served c16
    # logged rows n+1 and n+2 (requests finished in between) as unwarmed keys.
    assert {(2, 4), (3, 4), (3, 8), (4, 8), (5, 16), (7, 16)} <= shapes
    # The first verify step after a one-request prefill carries one rejected
    # row at any request count: served za11 logged rows 1 at full-batch buckets.
    assert {(1, 32), (1, 64)} <= shapes
    for rows, tokens in shapes:
        n = tokens // width
        assert rows in tpu_runner.PREP_FOLD_BOOTSTRAP_ROWS or n <= rows <= min(
            max(n + tpu_runner.PREP_FOLD_PREV_ROWS,
                _get_padded_num_reqs_with_upper_limit(n, MAX_REQS)),
            MAX_REQS,
        ), (rows, tokens)


def test_env_parse(monkeypatch):
    get = envs.environment_variables["VLLM_TPU_PREFWD_PREP_FOLD"]
    check = envs.environment_variables["VLLM_TPU_PREFWD_PREP_FOLD_CHECK"]
    monkeypatch.delenv("VLLM_TPU_PREFWD_PREP_FOLD", raising=False)
    monkeypatch.delenv("VLLM_TPU_PREFWD_PREP_FOLD_CHECK", raising=False)
    assert get() == 0 and check() == ""
    for value, want in (("0", 0), ("1", 1), ("2", 2), ("", 0)):
        monkeypatch.setenv("VLLM_TPU_PREFWD_PREP_FOLD", value)
        assert get() == want
    monkeypatch.setenv("VLLM_TPU_PREFWD_PREP_FOLD", "3")
    with pytest.raises(ValueError):
        get()
    monkeypatch.setenv("VLLM_TPU_PREFWD_PREP_FOLD_CHECK", "fault")
    assert check() == "fault"
    monkeypatch.setenv("VLLM_TPU_PREFWD_PREP_FOLD_CHECK", "bogus")
    with pytest.raises(ValueError):
        check()


# --- runner level -----------------------------------------------------------

K = 3
WIDTH = K + 1


def _fold_envs(monkeypatch, window="1", level=1):
    for name, value in (
        ("VLLM_TPU_PREFWD_PREP_FOLD", level),
        ("VLLM_TPU_PREFWD_PREP_FOLD_CHECK", ""),
        ("VLLM_TPU_QWEN4_FUSED_PREP", True),
        ("VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL", True),
        ("VLLM_TPU_QWEN4_FUSED_PREP_CHECK", ""),
        ("VLLM_TPU_QWEN4_NGRAM_WINDOW", window),
        ("VLLM_TPU_DRAFT_SHARD_ARGMAX", False),
        ("VLLM_TPU_SUBTRACT_REJECTED_PAD_ROWS", False),
        ("VLLM_TPU_SUBTRACT_REJECTED_COMPILED", False),
        ("VLLM_TPU_SUBST_FRESH_IDS", False),
        ("VLLM_TPU_SUBST_STATIC_ARGS", False),
        ("VLLM_TPU_WARMED_SUBSTITUTE_SOURCE", True),
    ):
        monkeypatch.setattr(envs, name, value, raising=False)
    monkeypatch.setattr(tpu_runner, "_prep_fold_compiled", _prep_fold_impl)
    # The unfolded scatter, eagerly (the compiled one needs a TPU).
    monkeypatch.setattr(
        tpu_runner, "_substitute_placeholder_token", _substitute_placeholder_token_impl
    )
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **k: None)


def _make_runner(pooled, slots=8, max_len=256, seed=0):
    rng = np.random.default_rng(seed)
    runner = NS(
        device=torch.device("cpu"),
        max_num_reqs=slots,
        num_reqs_max_model_len=slots,
        num_reqs_most_model_len=None,
        num_tokens_paddings=[4, 8, 16, 32, 64],
        speculative_config=NS(num_speculative_tokens=K),
        scheduler_config=NS(async_scheduling=True),
        query_start_loc_cpu=torch.zeros(slots + 1, dtype=torch.int32),
        seq_lens_cpu=torch.zeros(slots, dtype=torch.int32),
        positions_cpu=torch.zeros(64, dtype=torch.int64),
        input_ids_cpu=torch.zeros(64, dtype=torch.int32),
        _pcp_enabled=False,
        _pp_wave=None,
        supports_mm_inputs=False,
        uses_mrope=False,
        lora_config=None,
        _last_sequence_layout_plan=None,
        _fast_token_substitution=True,
        num_reqs_paddings=[1, 2, 4, 8],
        _is_async_drafter=True,
        _unfenced_h2d_sources=[],
        _registered_h2d_sources=set(),
    )
    runner.model_config = NS(hf_text_config=q4exp_text_config())
    runner.cache_config = NS(block_size=128)
    runner.vllm_config = None
    runner.input_batch = NS(
        token_ids_cpu_tensor=torch.from_numpy(
            rng.integers(100, 999, (slots, max_len)).astype(np.int32)
        ),
        num_computed_tokens_cpu=np.zeros(slots, dtype=np.int64),
        req_ids=[],
        num_reqs=0,
    )
    adapter = Qwen4ExpAdapter(runner)
    adapter.pooled = pooled
    adapter.spec_window = WIDTH
    adapter._fused_fn = lambda: inputs_mod.fused_step_inputs
    runner.model_adapter = adapter
    for name in (
        "_prepare_async_token_substitution_indices",
        "_prep_fold_check",
        "_cached_substitution_indices",
        "_warmed_next_tokens",
        "_substitute_next_lens",
    ):
        setattr(runner, name, MethodType(getattr(TPUModelRunner, name), runner))
    return runner, adapter


def _tables(adapter, slots):
    attention = f"{adapter.attention_prefix}.self_attn"
    return {
        adapter._main_owner(attention): NS(
            block_tables=torch.arange(slots * 3).reshape(slots, 3)
        ),
        adapter._compressed_owner(attention): NS(
            block_tables=torch.arange(slots * 2).reshape(slots, 2)
        ),
    }


def _step_inputs(runner, req_ids, prev_index, seed, slots=8, bucket=None):
    """Host and device inputs of one steady verify step (all requests verify
    K drafts; the token bucket is n * (K + 1), or `bucket`). Requests missing
    from `prev_index` keep their host tokens."""
    g = torch.Generator().manual_seed(seed)
    n = len(req_ids)
    tokens = bucket or n * WIDTH
    runner.input_batch.req_ids = list(req_ids)
    runner.input_batch.num_reqs = n
    ctx = torch.randint(30, 200, (n,), generator=g).numpy()
    runner.input_batch.num_computed_tokens_cpu[:n] = ctx
    seq_lens = torch.zeros(slots, dtype=torch.int32)
    seq_lens[:n] = torch.from_numpy(ctx + WIDTH).to(torch.int32)
    positions = torch.zeros(tokens, dtype=torch.int64)
    positions[: n * WIDTH] = torch.cat(
        [torch.arange(c, c + WIDTH, dtype=torch.int64) for c in ctx.tolist()]
    )
    seq_idx = torch.full((slots,), -1, dtype=torch.int32)
    pos_idx = torch.full((tokens,), -1, dtype=torch.int32)
    for i, r in enumerate(req_ids):
        if r in prev_index:
            seq_idx[i] = prev_index[r]
            pos_idx[i * WIDTH : (i + 1) * WIDTH] = prev_index[r]
    input_ids = torch.zeros(tokens, dtype=torch.int32)
    input_ids[: n * WIDTH] = torch.randint(
        1, 99, (n * WIDTH,), generator=g, dtype=torch.int32
    )
    starts = torch.zeros(slots + 1, dtype=torch.int32)
    starts[1 : n + 1] = torch.arange(1, n + 1, dtype=torch.int32) * WIDTH
    starts[n + 1 :] = starts[n]
    target = torch.zeros(tokens, dtype=torch.int32)
    draft_pos = [i * WIDTH + j for i in range(n) for j in range(K)]
    target[: len(draft_pos)] = torch.tensor(draft_pos, dtype=torch.int32)
    final = torch.zeros(tokens, dtype=torch.int32)
    final[: n * WIDTH] = torch.arange(n * WIDTH, dtype=torch.int32)
    md = NS(
        final_logits_indices=final,
        target_logits_indices=target,
        draft_token_ids=None,
    )
    pre = NS(
        next_tokens_tpu=torch.randint(100, 900, (slots * WIDTH,), generator=g),
        spec_decode_num_rejected_tokens=torch.randint(
            0, K + 1, (max(prev_index.values()) + 1,), generator=g, dtype=torch.int32
        ),
        req_id_to_index_copy=dict(prev_index),
    )
    return dict(
        n=n,
        tokens=tokens,
        seq_lens=seq_lens,
        positions=positions,
        subtract_idx=(seq_idx, pos_idx),
        starts=starts,
        state=torch.arange(1, slots + 1, dtype=torch.int32),
        md=md,
        pre=pre,
        sched=np.full(n, WIDTH, dtype=np.int32),
        input_ids=input_ids,
    )


def _run_unfolded(runner, adapter, s, slots=8):
    runner._pre_async_results = s["pre"]
    runner.position_ids = s["positions"].clone()
    runner.input_ids = s["input_ids"].clone()
    seq_lens, runner.position_ids = TPUModelRunner._subtract_rejected(
        runner, s["seq_lens"], s["subtract_idx"]
    )
    cur, src = TPUModelRunner._prepare_async_token_substitution_indices(
        runner, 0, s["n"], s["sched"]
    )
    ids = TPUModelRunner._apply_async_token_substitution(
        runner, runner.input_ids, cur, src
    )
    drafts = extract_draft_token_ids(
        ids, s["md"].final_logits_indices, s["md"].target_logits_indices
    )
    runner._attn_metadata_builder_ctx = NS(
        query_start_loc=s["starts"], seq_lens=seq_lens, mamba_state_indices=s["state"]
    )
    store = adapter.token_store
    kwargs = adapter.prepare_inputs(
        ModelInputSnapshot(store, 0, s["n"], s["tokens"], _tables(adapter, slots), ids)
    )
    return seq_lens, runner.position_ids, ids, drafts, kwargs


def _run_folded(runner, adapter, s, slots=8):
    runner._pre_async_results = s["pre"]
    runner.position_ids = s["positions"].clone()
    runner.input_ids = s["input_ids"].clone()
    host_ids = runner.input_ids.clone()
    runner._prep_fold_out = None
    layout = NS(kind=SequenceLayoutKind.ALL)
    subst = TPUModelRunner._prep_fold_precheck(
        runner, 0, s["n"], s["n"], s["sched"], s["tokens"], layout
    )
    assert subst is not None
    seq_lens = TPUModelRunner._try_prep_fold(
        runner,
        s["seq_lens"],
        s["subtract_idx"],
        s["md"],
        s["starts"],
        s["state"],
        0,
        s["n"],
        subst,
    )
    assert seq_lens is not None, "step did not fold"
    assert torch.equal(runner.input_ids, host_ids)  # no input mutation
    folded = runner._prep_fold_out
    runner._attn_metadata_builder_ctx = NS(
        query_start_loc=s["starts"], seq_lens=seq_lens, mamba_state_indices=s["state"]
    )
    kwargs = adapter.prepare_inputs(
        ModelInputSnapshot(
            adapter.token_store,
            0,
            s["n"],
            s["tokens"],
            _tables(adapter, slots),
            folded["ids"],
        )
    )
    runner._prep_fold_out = None
    return seq_lens, runner.position_ids, folded["ids"], folded["drafts"], kwargs


@pytest.mark.parametrize("warmed_src", [True, False])
@pytest.mark.parametrize("window", ["1", ""])
@pytest.mark.parametrize("pooled", [False, True])
def test_runner_fold_matches_unfolded_step(monkeypatch, pooled, window, warmed_src):
    """Steady steps, flag 0 vs flag 1: identical forward inputs (ids,
    positions, seq_lens, adapter kwargs, verified drafts) and identical
    speculative records and row bookkeeping, step after step."""
    _fold_envs(monkeypatch, window=window)
    monkeypatch.setattr(envs, "VLLM_TPU_WARMED_SUBSTITUTE_SOURCE", warmed_src)
    r_ref, a_ref = _make_runner(pooled)
    r_fold, a_fold = _make_runner(pooled)
    TPUModelRunner._precompile_prep_fold(r_fold)
    assert r_fold._prep_fold_keys
    steps = [
        # Only 1:1 substitutions fold: request i sat at result row i. Rows
        # still move between steps (previous-row bookkeeping, records).
        (["a", "b", "c", "d"], {"a": 0, "b": 1, "c": 2, "d": 3}),
        (["b", "a", "d", "c"], {"b": 0, "a": 1, "d": 2, "c": 3}),
        (["c", "b"], {"c": 0, "b": 1}),
        (["c", "b"], {"c": 0, "b": 1}),
        (["a", "b", "c", "d"], {"a": 0, "b": 1, "c": 2, "d": 3}),
        # Scatter substitutions: rows moved, a partial bucket (3 * 4 tokens
        # in 16, 5 * 4 in 32), one request with host tokens only.
        (["b", "a", "d", "c"], {"a": 0, "b": 1, "c": 2, "d": 3}),
        (["a", "b", "c"], {"c": 0, "a": 1, "b": 2}),
        (["a", "b", "c", "d", "e"], {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}),
        (["a", "b", "c", "x"], {"a": 0, "b": 1, "c": 2, "d": 3}),
    ]
    for step, (req_ids, prev_index, *bucket) in enumerate(steps):
        outs = []
        for runner, adapter, fold in ((r_ref, a_ref, 0), (r_fold, a_fold, 1)):
            monkeypatch.setattr(envs, "VLLM_TPU_PREFWD_PREP_FOLD", fold)
            tokens = bucket[0] if bucket else None
            if tokens is None and len(req_ids) * WIDTH not in (4, 8, 16, 32, 64):
                tokens = next(b for b in (4, 8, 16, 32, 64) if b >= len(req_ids) * 4)
            s = _step_inputs(runner, req_ids, prev_index, seed=step, bucket=tokens)
            adapter.begin_step()
            run = _run_folded if fold else _run_unfolded
            outs.append(run(runner, adapter, s))
        ref, got = outs
        for i in range(4):
            assert ref[i].dtype == got[i].dtype, (step, i)
            assert torch.equal(ref[i], got[i]), (step, i)
        assert ref[4].keys() == got[4].keys()
        for key in ref[4]:
            assert torch.equal(ref[4][key], got[4][key]), (step, key)
        for g, r in zip(a_fold._recent[1], a_ref._recent[1]):
            assert torch.equal(g, r), step
        assert a_fold._current_rows == a_ref._current_rows
    # Both forms folded: the scatter steps went through the cached indices.
    assert getattr(r_fold, "_cached_subst_key", None) is not None


def test_fold_gated_on_warmed_keys(monkeypatch):
    """Unwarmed key: today's path, nothing run, logged once per key. Warmed
    key: the folded program runs."""
    _fold_envs(monkeypatch)
    runner, adapter = _make_runner(True)
    log = Mock()
    monkeypatch.setattr(tpu_runner, "logger", log)
    compiled = Mock(side_effect=_prep_fold_impl)
    monkeypatch.setattr(tpu_runner, "_prep_fold_compiled", compiled)
    adapter.begin_step()
    s = _step_inputs(runner, ["a", "b"], {"a": 0, "b": 1}, seed=1)
    runner._pre_async_results = s["pre"]
    runner.position_ids = s["positions"]
    runner.input_ids = torch.zeros(s["tokens"], dtype=torch.int32)
    runner._prep_fold_keys = frozenset()
    subst = (np.arange(8, dtype=np.int32), np.arange(8, dtype=np.int32))

    def attempt():
        return TPUModelRunner._try_prep_fold(
            runner,
            s["seq_lens"],
            s["subtract_idx"],
            s["md"],
            s["starts"],
            s["state"],
            0,
            s["n"],
            subst,
        )

    assert attempt() is None and attempt() is None
    assert not compiled.called
    misses = [c for c in log.info.call_args_list if "not warmed" in c.args[0]]
    assert len(misses) == 1
    assert runner.position_ids is s["positions"]
    assert getattr(runner, "_prep_fold_out", None) is None
    # A different source dtype is a different key: not warmed either.
    TPUModelRunner._precompile_prep_fold(runner)
    assert attempt() is not None
    assert compiled.call_count == 1 + len(runner._prep_fold_keys)
    assert runner._prep_fold_out is not None


def test_fold_ineligible_steps_fall_back(monkeypatch):
    _fold_envs(monkeypatch)
    runner, adapter = _make_runner(True)
    s = _step_inputs(runner, ["a", "b"], {"a": 0, "b": 1}, seed=2)
    runner._pre_async_results = s["pre"]
    layout = NS(kind=SequenceLayoutKind.ALL)

    def precheck(start=0, end=2, sched=None, tokens=8, plan=layout):
        sched = s["sched"] if sched is None else sched
        return TPUModelRunner._prep_fold_precheck(
            runner, start, 2, end, sched, tokens, plan
        )

    assert precheck() is not None
    # Multi-chunk.
    runner.input_batch.num_reqs = 3
    assert precheck() is None
    runner.input_batch.num_reqs = 2
    assert precheck(start=1) is None
    # Partial bucket or a request that did not run last step: the scatter
    # substitution, still eligible. Nothing to substitute: not eligible.
    assert precheck(tokens=16) is not None
    runner._pre_async_results.req_id_to_index_copy = {"a": 0}
    assert precheck() is not None
    runner._pre_async_results.req_id_to_index_copy = {}
    assert precheck() is None
    runner._pre_async_results.req_id_to_index_copy = {"a": 0, "b": 1}
    # Non-ALL sequence layout (PCP streaming), PCP, multimodal, sync scheduler.
    assert precheck(plan=NS(kind=object())) is None
    for attr, value in (
        ("_pcp_enabled", True),
        ("supports_mm_inputs", True),
        ("_pp_wave", object()),
        ("scheduler_config", NS(async_scheduling=False)),
    ):
        old = getattr(runner, attr)
        setattr(runner, attr, value)
        assert precheck() is None, attr
        setattr(runner, attr, old)
    # Adapter check modes and the non-functional fused program.
    for name, value in (
        ("VLLM_TPU_QWEN4_FUSED_PREP_CHECK", "check"),
        ("VLLM_TPU_QWEN4_NGRAM_WINDOW", "check"),
        ("VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL", False),
        ("VLLM_TPU_QWEN4_FUSED_PREP", False),
    ):
        old = getattr(envs, name)
        monkeypatch.setattr(envs, name, value)
        assert precheck() is None, name
        monkeypatch.setattr(envs, name, old)
    assert precheck() is not None
    # How the drafts were proposed (shard argmax or not) does not reach the
    # prep inputs: the served c32 env runs DRAFT_SHARD_ARGMAX and still folds.
    monkeypatch.setattr(envs, "VLLM_TPU_DRAFT_SHARD_ARGMAX", True)
    assert precheck() is not None
    # No drafts verified this step (spec metadata None) or a padded logits
    # length that differs from the bucket: no fold, nothing run.
    runner.position_ids = s["positions"]
    runner.input_ids = torch.zeros(s["tokens"], dtype=torch.int32)
    runner._prep_fold_keys = frozenset()
    for md in (
        None,
        NS(
            final_logits_indices=torch.zeros(4, dtype=torch.int32),
            target_logits_indices=torch.zeros(4, dtype=torch.int32),
        ),
    ):
        assert (
            TPUModelRunner._try_prep_fold(
                runner,
                s["seq_lens"],
                s["subtract_idx"],
                md,
                s["starts"],
                None,
                0,
                2,
                (np.arange(8, dtype=np.int32),) * 2,
            )
            is None
        )


def test_flag_off_adapter_ignores_stash(monkeypatch):
    """Flag 0: the adapter never reads a stale fold stash."""
    _fold_envs(monkeypatch, level=0)
    runner, adapter = _make_runner(True)
    runner._prep_fold_out = {"bogus": True}
    s = _step_inputs(runner, ["a"], {"a": 0}, seed=3)
    adapter.begin_step()
    out = _run_unfolded(runner, adapter, s)
    assert "ngram_context" in out[4]


def test_prep_fold_key_tracks_aliasing():
    a = torch.zeros(4, dtype=torch.int32)
    b = torch.zeros(4, dtype=torch.int32)
    assert _prep_fold_key((a, a), {}) != _prep_fold_key((a, b), {})
    assert _prep_fold_key((a, (b,)), {"x": 1}) == _prep_fold_key(
        (b.clone(), (a.clone(),)), {"x": 1}
    )


@pytest.mark.parametrize("check", ["1", "fault"])
def test_fold_check_counts_mismatches(monkeypatch, check):
    """PREFWD_PREP_FOLD_CHECK: clean folds count 0 mismatches; `fault` (the
    reference on seq_lens - 1) is caught."""
    _fold_envs(monkeypatch)
    monkeypatch.setattr(envs, "VLLM_TPU_PREFWD_PREP_FOLD_CHECK", check)
    runner, adapter = _make_runner(False)
    TPUModelRunner._precompile_prep_fold(runner)
    adapter.begin_step()
    s = _step_inputs(runner, ["a", "b"], {"a": 0, "b": 1}, seed=4)
    _run_folded(runner, adapter, s)
    bad = int(runner._prep_fold_bad)
    assert (bad == 0) if check == "1" else (bad > 0), bad


def _scatter_args(n, k, src_dtype, pattern, seed):
    """`_fold_args` for a scatter substitution: the bucket above `n * (k + 1)`
    verify tokens (or the exact one, rows moved), random real slots/sources,
    padded as fast substitution pads them."""
    width = k + 1
    real = n * width
    buckets = [4, 8, 16, 32, 64, 128]
    tokens = next(b for b in buckets if b >= real)
    if pattern == "partial" and tokens == real:
        tokens = next(b for b in buckets if b > real)
    args, static = _fold_args(n, k, src_dtype, "mixed", True, False, seed=seed)
    g = torch.Generator().manual_seed(seed + 7)
    if tokens != real:
        seq_lens, positions, rej, seq_idx, pos_idx = args[:5]
        positions = torch.cat(
            [positions, torch.zeros(tokens - real, dtype=torch.int64)]
        )
        pos_idx = torch.cat(
            [pos_idx, torch.full((tokens - real,), -1, dtype=torch.int32)]
        )
        final = torch.zeros(tokens, dtype=torch.int32)
        final[:real] = args[6]
        target = torch.zeros(tokens, dtype=torch.int32)
        target[:real] = args[7]
        args = (
            seq_lens,
            positions,
            rej,
            seq_idx,
            pos_idx,
            args[5],
            final,
            target,
            *args[8:],
        )
        static["num_tokens"] = tokens
    bound = args[5].shape[0]
    m = real if pattern != "subset" else max(1, real // 2)
    cur = np.sort(torch.randperm(real, generator=g)[:m].numpy()).astype(np.int32)
    if pattern == "moved":
        cur = torch.randperm(real, generator=g)[:m].numpy().astype(np.int32)
    src = torch.randint(0, bound, (m,), generator=g).numpy().astype(np.int32)
    runner = NS(
        device=torch.device("cpu"),
        _unfenced_h2d_sources=[],
        _registered_h2d_sources=set(),
    )
    cur_p, src_p = TPUModelRunner._cached_substitution_indices(runner, tokens, cur, src)
    input_ids = torch.zeros(tokens, dtype=torch.int32)
    input_ids[:real] = torch.randint(1, 99, (real,), generator=g, dtype=torch.int32)
    return args, static, (input_ids, cur_p, src_p)


def _unfolded_scatter(args, static, subst):
    seq_lens, positions = subtract_num_rejected_tokens(*args[:5])
    input_ids, cur_p, src_p = subst
    ids = _substitute_placeholder_token_impl(input_ids.clone(), cur_p, src_p, args[5])
    drafts = extract_draft_token_ids(ids, args[6], args[7])
    starts, state, store, scalars, previous, current = args[8:]
    prepped = inputs_mod.fused_step_inputs(
        starts,
        seq_lens,
        state,
        store,
        scalars,
        previous,
        current,
        ids,
        window=static["window"],
        pooled=static["pooled"],
        spec_window=static["spec_window"],
        eos_token_id=static["eos_token_id"],
        ngram_size=static["ngram_size"],
        functional=True,
    )
    return (seq_lens, positions, ids, drafts, *prepped)


@pytest.mark.parametrize("pattern", ["partial", "moved", "subset"])
@pytest.mark.parametrize("src_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("n", [1, 3, 4, 8, 13, 16])
def test_fold_scatter_bitwise_vs_unfolded(n, k, src_dtype, pattern):
    """Non-1:1 substitutions (partial bucket, moved rows, a subset of slots
    from the previous source): the folded scatter is bitwise the unfolded
    `_substitute_placeholder_token` + gather + fused inputs, and leaves its
    inputs (incl. the host input ids) untouched."""
    args, static, subst = _scatter_args(n, k, src_dtype, pattern, seed=n * 3 + k)
    ref = _unfolded_scatter(args, static, subst)
    leaves = [t for a in (*args, *subst) for t in (a if isinstance(a, tuple) else (a,))]
    before = [t.clone() for t in leaves]
    got = _prep_fold_impl(*args, *subst, **static)
    _assert_bitwise(got, ref)
    _assert_bitwise(_prep_fold_reference(*args, *subst, **static), ref)
    for t, b in zip(leaves, before):
        assert torch.equal(t, b)
    assert got[2].data_ptr() != subst[0].data_ptr()


def test_prep_fold_scatter_shapes_cover_steady_keys():
    """Every steady request count (c16 and c32, MTP K=3, buckets incl. 4/8)
    has its (rows, bucket) warmed for the scatter, exact and bucketed rows."""
    from vllm_torchtpu.platforms.tpu_platform import _get_token_paddings

    paddings = _get_token_paddings(4, 2048)
    for max_reqs in (16, 32):
        shapes = _prep_fold_scatter_shapes(max_reqs, 4, paddings)
        for n in range(1, max_reqs + 1):
            tokens = tpu_runner._get_padded_token_len(paddings, n * 4)
            for rows in (n, _get_padded_num_reqs_with_upper_limit(n, max_reqs)):
                assert (rows, tokens) in shapes, (max_reqs, n, rows)
        assert {(1, 4), (2, 8), (3, 16), (16, 64)} <= shapes
        # Rows 1 (after a one-request prefill) at every steady bucket.
        assert {(1, 32), (1, 64)} <= shapes
    assert {(30, 128), (1, 128)} <= _prep_fold_scatter_shapes(32, 4, paddings)


# The za10y B arm's served flags around the fold (c16/c32): metadata reuse, the
# prebuilt forward contexts, the stacked spec tail, the in-graph verify scatter.
# None of them feeds an operand of the folded program; the source and rejected
# counts the fold reads come from the stacked spec tail.
B_ARM_FLAGS = (
    ("VLLM_TPU_ATTN_METADATA_REUSE", True),
    ("VLLM_TPU_FORWARD_CTX_PREBUILT", True),
    ("VLLM_TPU_SPEC_TAIL_COMPILED", True),
    ("VLLM_TPU_SPEC_TAIL_STACK", True),
    ("VLLM_TPU_VERIFY_RO_INGRAPH", True),
    ("VLLM_TPU_MK_WHOLE_DIRECT_EXEC", True),
    ("VLLM_TPU_OUTPUT_PATH_LEAN", True),
    ("VLLM_TPU_SUBST_FRESH_IDS", True),
    ("VLLM_TPU_SUBTRACT_REJECTED_COMPILED", True),
    # Served on too; off here only because its owned-alias wrapper compiles the
    # unfolded reference's scatter, which needs a TPU.
    ("VLLM_TPU_SUBST_STATIC_ARGS", False),
)


def _stacked_tail_pre(prev_index, seed, slots=8, step_dtype=torch.int64):
    """The previous step's async results as the stacked spec tail
    (`_spec_tail_stack`, VLLM_TPU_SPEC_TAIL_STACK) leaves them for a one-chunk
    steady step of `len(prev_index)` requests: the `[bonus, drafts]` source
    padded to the bound and the per-request rejected counts."""
    from vllm_torchtpu.runner.tpu_runner import _spec_tail_stack
    from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID

    g = torch.Generator().manual_seed(1000 + seed)
    n = len(prev_index)
    accepted = torch.randint(0, K + 1, (n,), generator=g)
    next_tokens = torch.randint(100, 900, (n, K + 1), generator=g, dtype=torch.int32)
    cols = torch.arange(K + 1).expand(n, K + 1)
    next_tokens[cols > accepted[:, None]] = INVALID_TOKEN_ID
    steps = [
        torch.randint(100, 900, (slots,), generator=g, dtype=step_dtype)
        for _ in range(K)
    ]
    draft_lengths = torch.full((n,), K, dtype=torch.int32)
    src, num_rejected = _spec_tail_stack(
        next_tokens, draft_lengths, n, n, slots * WIDTH, n, *steps
    )
    return NS(
        next_tokens_tpu=src,
        spec_decode_num_rejected_tokens=num_rejected,
        req_id_to_index_copy=dict(prev_index),
    )


@pytest.mark.parametrize("step_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("pooled", [False, True])
def test_b_arm_flags_steady_steps_fold_after_warmup(monkeypatch, pooled, step_dtype):
    """With the B arm's flags and the stacked spec tail's source / rejected
    counts, every steady verify step after warmup folds (its key is a warmed
    one: no aliasing or shape the warmup lacks) and the folded forward inputs
    are bitwise the unfolded ones."""
    if not hasattr(tpu_runner, "_spec_tail_stack"):
        pytest.skip("no VLLM_TPU_SPEC_TAIL_STACK on this branch")
    _fold_envs(monkeypatch, window="1")
    for name, value in B_ARM_FLAGS:
        monkeypatch.setattr(envs, name, value, raising=False)
    r_ref, a_ref = _make_runner(pooled)
    r_fold, a_fold = _make_runner(pooled)
    TPUModelRunner._precompile_prep_fold(r_fold)
    misses = []
    monkeypatch.setattr(
        tpu_runner.logger,
        "info",
        lambda msg, *a, **k: misses.append(a) if "not warmed" in msg else None,
    )
    steps = [
        (["a", "b", "c", "d"], {"a": 0, "b": 1, "c": 2, "d": 3}),
        (["a", "b", "c", "d"], {"a": 0, "b": 1, "c": 2, "d": 3}),
        (["b", "a", "d", "c"], {"b": 0, "a": 1, "d": 2, "c": 3}),
        (["c", "b"], {"c": 0, "b": 1}),
        # Scatter: rows moved, a partial bucket, a request with host tokens.
        (["b", "a", "d", "c"], {"a": 0, "b": 1, "c": 2, "d": 3}),
        (["a", "b", "c"], {"c": 0, "a": 1, "b": 2}),
        (["a", "b", "c", "x"], {"a": 0, "b": 1, "c": 2}),
    ]
    for step, (req_ids, prev_index) in enumerate(steps):
        outs = []
        for runner, adapter, fold in ((r_ref, a_ref, 0), (r_fold, a_fold, 1)):
            monkeypatch.setattr(envs, "VLLM_TPU_PREFWD_PREP_FOLD", fold)
            tokens = next(b for b in (4, 8, 16, 32, 64) if b >= len(req_ids) * WIDTH)
            s = _step_inputs(runner, req_ids, prev_index, seed=step, bucket=tokens)
            s["pre"] = _stacked_tail_pre(prev_index, step, step_dtype=step_dtype)
            adapter.begin_step()
            run = _run_folded if fold else _run_unfolded
            outs.append(run(runner, adapter, s))  # _run_folded asserts it folded
        ref, got = outs
        for i in range(4):
            assert ref[i].dtype == got[i].dtype, (step, i)
            assert torch.equal(ref[i], got[i]), (step, i)
        assert ref[4].keys() == got[4].keys()
        for key in ref[4]:
            assert torch.equal(ref[4][key], got[4][key]), (step, key)
    assert not misses


@pytest.mark.parametrize("pooled", [False, True])
def test_full_batch_after_one_request_prefill_folds(monkeypatch, pooled):
    """The first verify step of a full batch after a one-request prefill (the
    previous step's rejected counts have one row, `_assemble_async_prefill_bootstrap`)
    folds after warmup -- the self-check counts it folded, not unwarmed -- and
    its forward inputs are bitwise the unfolded ones. Without the rows-1 keys
    (PREP_FOLD_BOOTSTRAP_ROWS) the same step is an unwarmed miss."""
    _fold_envs(monkeypatch, window="1")
    for name, value in B_ARM_FLAGS:
        monkeypatch.setattr(envs, name, value, raising=False)
    req_ids = list("abcdefgh")  # 8 requests x 4 tokens = the 32 bucket, full
    prev_index = {"h": 0}  # one prefilled request: one rejected-count row

    def run(bootstrap_rows):
        monkeypatch.setattr(tpu_runner, "PREP_FOLD_BOOTSTRAP_ROWS", bootstrap_rows)
        r_ref, a_ref = _make_runner(pooled)
        r_fold, a_fold = _make_runner(pooled)
        TPUModelRunner._precompile_prep_fold(r_fold)
        outs = []
        for runner, adapter, fold in ((r_ref, a_ref, 0), (r_fold, a_fold, 1)):
            monkeypatch.setattr(envs, "VLLM_TPU_PREFWD_PREP_FOLD", fold)
            s = _step_inputs(runner, req_ids, prev_index, seed=7, bucket=32)
            assert s["pre"].spec_decode_num_rejected_tokens.shape == (1,)
            adapter.begin_step()
            if not fold:
                outs.append(_run_unfolded(runner, adapter, s))
                continue
            runner._pre_async_results = s["pre"]
            runner.position_ids = s["positions"].clone()
            runner.input_ids = s["input_ids"].clone()
            subst = TPUModelRunner._prep_fold_precheck(
                runner, 0, s["n"], s["n"], s["sched"], s["tokens"],
                NS(kind=SequenceLayoutKind.ALL),
            )
            assert subst is not None  # eligible: the miss is the warmup's
            # As the serving call site: try, count the outcome, fall back.
            seq_lens = TPUModelRunner._try_prep_fold(
                runner, s["seq_lens"], s["subtract_idx"], s["md"], s["starts"],
                s["state"], 0, s["n"], subst,
            )
            TPUModelRunner._note_prep_fold(
                runner, "unwarmed" if seq_lens is None else "folded"
            )
            if seq_lens is None:
                return runner, None
            folded = runner._prep_fold_out
            runner._attn_metadata_builder_ctx = NS(
                query_start_loc=s["starts"], seq_lens=seq_lens,
                mamba_state_indices=s["state"],
            )
            kwargs = adapter.prepare_inputs(
                ModelInputSnapshot(
                    adapter.token_store, 0, s["n"], s["tokens"],
                    _tables(adapter, 8), folded["ids"],
                )
            )
            outs.append(
                (seq_lens, runner.position_ids, folded["ids"], folded["drafts"], kwargs)
            )
        return r_fold, outs

    lines = []
    monkeypatch.setattr(
        tpu_runner.logger, "info", lambda msg, *a, **k: lines.append(msg % a if a else msg)
    )
    # Negative control: without the rows-1 keys the step misses.
    runner, outs = run(())
    assert outs is None and runner._prep_fold_counts == {"unwarmed": 1}
    assert any("not warmed" in line for line in lines)
    runner, (ref, got) = run((1,))
    assert runner._prep_fold_counts == {"folded": 1}
    assert tpu_runner.prep_fold_recorded_line(runner._prep_fold_counts) == (
        "PREFWD_PREP_FOLD recorded: folded 1/1 steps (100.0%), unfolded: "
        "precheck 0, unwarmed 0"
    )
    for i in range(4):
        assert ref[i].dtype == got[i].dtype and torch.equal(ref[i], got[i]), i
    assert ref[4].keys() == got[4].keys()
    for key in ref[4]:
        assert torch.equal(ref[4][key], got[4][key]), key


def test_prep_fold_recorded_self_check(monkeypatch):
    """The rank-0 self-check line counts folded / unfolded correction steps and
    is logged at PREP_FOLD_LOG_FIRST steps and every PREP_FOLD_LOG_EVERY."""
    monkeypatch.setattr(tpu_runner, "PREP_FOLD_LOG_FIRST", 4)
    monkeypatch.setattr(tpu_runner, "PREP_FOLD_LOG_EVERY", 8)
    lines = []
    monkeypatch.setattr(
        tpu_runner.logger, "info", lambda msg, *a, **k: lines.append(msg)
    )
    runner = NS()
    for outcome in ["folded"] * 3 + ["unwarmed"] + ["folded"] * 7 + ["precheck"]:
        TPUModelRunner._note_prep_fold(runner, outcome)
    assert lines == [
        "PREFWD_PREP_FOLD recorded: folded 3/4 steps (75.0%), unfolded: "
        "precheck 0, unwarmed 1",
        "PREFWD_PREP_FOLD recorded: folded 7/8 steps (87.5%), unfolded: "
        "precheck 0, unwarmed 1",
    ]
    assert tpu_runner.prep_fold_recorded_line(runner._prep_fold_counts) == (
        "PREFWD_PREP_FOLD recorded: folded 10/12 steps (83.3%), unfolded: "
        "precheck 1, unwarmed 1"
    )
    # Other TP ranks count but do not log.
    monkeypatch.setattr(tpu_runner, "_prep_fold_rank0", lambda: False)
    lines.clear()
    for _ in range(12):
        TPUModelRunner._note_prep_fold(runner, "folded")
    assert lines == [] and runner._prep_fold_counts["folded"] == 22
