# SPDX-License-Identifier: Apache-2.0
"""q38 X-D14-DRAFT-1 (VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE): the MTP draft's
shared-selection steps without indexer state.

Under VLLM_TPU_QSA_DRAFT_SHARED_SELECTION the draft attends the target's carried
selection on every pass once bound, so nothing reads its compressed keys, raw ring
or tags. The ``_noidx`` steps keep only the main K/V write: no index projection
use, no ring / tag read, no ``_indexer_updates``, no compressed / ring / tag write.

1. Bitwise over an 8-step closed-loop draft, flag off (``pooled_qsa_shared*``) vs
   on (``pooled_qsa_shared*_noidx``), on both DCP-2 ranks (the served
   VLLM_TPU_QSA_MTP_KV_DCP ops) and on the replicated ops. Step 0 is the rows
   step over a verify chunk (the target's selection at each request's last row),
   steps 1-7 one token per request reusing that carried selection; each step's
   draft token (argmax of a fixed head over the attention output) feeds the next
   step's projections. With the flag on the index operand is the step's hidden
   state, as the torch layer passes it. Draft tokens, per-rank outputs / LSEs,
   selected tokens and every pool byte outside the draft's index pages must be
   identical at every step; with the flag on the index pages (compressed keys,
   ring + tags, the null page) keep their initial bytes, with it off the ring
   pages change (the control: the comparison sees index-state writes).
2. Census: the traced (make_jaxpr; real Pallas kernels, not run) 3-pass draft
   holds 3 ring_tags_batched_read + 3 ring_tags_batched_write kernels with the
   flag off and none with it on; the main K/V writers and every other kernel
   are unchanged.

Same harness as ``test_dcp2_draft`` (CPU, Pallas interpret, RPA stand-in).
"""

import collections
import dataclasses
import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dcp2_harness import D, build_layout, from_raw, project_rank_pool, random_fp8, rank_operand, raw
from test_dcp2_op import (
    B,
    DH,
    H,
    INDEX_DIM,
    INDEX_HEADS,
    K_SCALE,
    RATIO,
    ROTARY,
    TOPK,
    V_SCALE,
    partial_kernels,  # noqa: F401 (fixture)
    stub_select,
)

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import selected_tokens

# Step 0: verify windows of K + 1 = 4; the last rows cross B and 2B (both DCP owners);
# the last seat is idle. Steps 1-7: one token per live request.
QUERY_LENS = [4, 4, 4, 4, 0]
CONTEXTS = [37, 509, 1022, 1100, 0]
MAX_Q = 4
STEPS = 8
HID, VOCAB = 64, 97
OPTS = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
            k_scale=K_SCALE, v_scale=V_SCALE, max_query_len=MAX_Q)

# flag -> (rows step, all-token step), replicated and DCP.
REPLICATED = {
    False: (pooled_qsa.pooled_qsa_shared_rows_step, pooled_qsa.pooled_qsa_shared_step),
    True: (pooled_qsa.pooled_qsa_shared_rows_noidx_step, pooled_qsa.pooled_qsa_shared_noidx_step),
}
DCP = {
    False: (pooled_qsa.pooled_qsa_shared_rows_dcp_step, pooled_qsa.pooled_qsa_shared_dcp_step),
    True: (pooled_qsa.pooled_qsa_shared_rows_dcp_noidx_step,
           pooled_qsa.pooled_qsa_shared_dcp_noidx_step),
}


@dataclasses.dataclass
class Draft:
    layout: object
    full: jax.Array
    comp: np.ndarray
    ring: np.ndarray
    index_pages: np.ndarray
    w: dict
    hidden0: jax.Array
    target_sel: np.ndarray


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    q_lens, ctx = np.asarray(QUERY_LENS), np.asarray(CONTEXTS)
    requests = len(ctx)
    live = q_lens > 0
    alloc = np.maximum(np.where(live, ctx + q_lens + STEPS - 1, 0), 1)
    cols = int(-(-alloc.max() // B))
    num_pages = int(sum(-(-alloc // B)) * 2 + requests * (cols + 1) + 4)
    layout = build_layout(rng, alloc, B, num_pages)
    free = iter(layout.free_pages)
    comp = np.array([[next(free) for _ in range(cols)] for _ in range(requests)], np.int32)
    ring = np.array([next(free) for _ in range(requests)], np.int32)
    index_pages = np.asarray(sorted({0, *comp.ravel().tolist(), *ring.tolist()}), np.int64)
    full = random_fp8(rng, (num_pages, B, 1, 4, DH))

    def normal(*shape, scale=1.0):
        return jnp.asarray(rng.normal(0, scale, shape), jnp.bfloat16)

    w = dict(
        embed=normal(VOCAB, HID),
        wq=[normal(HID, 2 * H * DH, scale=0.3) for _ in range(D)],  # per TP rank of the pair
        wk=normal(HID, DH, scale=0.3),
        wv=normal(HID, DH, scale=0.3),
        wi=normal(HID, (INDEX_HEADS + 1) * INDEX_DIM, scale=0.3),
        head=normal(D * H * DH, VOCAB, scale=0.1),
        nq=normal(DH, scale=0.1), nk=normal(DH, scale=0.1),
        niq=normal(INDEX_DIM, scale=0.1), nik=normal(INDEX_DIM, scale=0.1),
        trig=jnp.pad(normal(16384, ROTARY), ((0, 0), (0, 128 - ROTARY))),
    )
    hidden0 = normal(max(8, -(-int(q_lens.sum()) // 8) * 8), HID)
    # The target's selection at each request's last verified row: random groups, not
    # what stub_select (the draft's own indexer stand-in) picks, so a draft that
    # selected from its own state cannot match it by accident.
    budget = TOPK // RATIO
    lengths = np.where(live, ctx + q_lens, 0).astype(np.int32)
    blocks = np.full((requests, budget), -1, np.int32)
    for r in range(requests):
        groups = int(lengths[r]) // RATIO
        take = min(groups, budget)
        if take:
            blocks[r, :take] = rng.permutation(groups)[:take]
    target_sel = np.asarray(selected_tokens(jnp.asarray(blocks), jnp.asarray(lengths), TOPK, RATIO))
    return Draft(layout, full, comp, ring, index_pages, w, hidden0, target_sel)


def _proj(x, w):
    return (x.astype(jnp.float32) @ w.astype(jnp.float32)).astype(jnp.bfloat16)


def _step_inputs(d, i, tokens):
    """Step ``i``'s operands; ``tokens`` = the previous step's draft token per request."""
    q_lens, ctx = np.asarray(QUERY_LENS), np.asarray(CONTEXTS)
    live = q_lens > 0
    requests = len(ctx)
    if i == 0:
        lens, context = q_lens, ctx
    else:
        lens = live.astype(np.int64)
        context = np.where(live, ctx + q_lens + i - 1, 0)
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    count = max(8, -(-int(starts[-1]) // 8) * 8)
    positions = np.zeros(count, np.int32)
    for r in range(requests):
        positions[starts[r]:starts[r + 1]] = context[r] + np.arange(lens[r])
    if i == 0:
        hidden = d.hidden0
    else:
        rows = [d.w["embed"][int(tokens[r])] for r in range(requests) if live[r]]
        hidden = jnp.zeros((count, HID), jnp.bfloat16).at[: len(rows)].set(jnp.stack(rows))
    # The target sink row of each request's last verified token (an idle seat's
    # clamped row reads row 0, request 0's).
    first = d.target_sel[np.where(live, np.arange(requests), 0)]
    if i == 0:
        rows_np = np.where(live, starts[1:] - 1, -1).astype(np.int32)
        rows = jnp.asarray(rows_np)
        shared = jnp.asarray(first)  # source[rows.clamp_min(0)] (qsa.attend_with_indices)
        picked = np.maximum(rows_np, 0)
    else:
        rows = None
        picked = np.arange(count)
        # The carried selection (_selection_rows): step 0's rows, then -1.
        carried = np.full((max(count, requests), first.shape[1]), -1, np.int32)
        carried[:requests] = first
        shared = jnp.asarray(carried[:count])
    return dict(
        hidden=hidden,
        qgate=[_proj(hidden, d.w["wq"][r]) for r in range(D)],
        key=_proj(hidden, d.w["wk"]),
        value=_proj(hidden, d.w["wv"]),
        index_qk=_proj(hidden, d.w["wi"]),
        positions=jnp.asarray(positions),
        starts=jnp.asarray(starts),
        context=jnp.asarray(context, jnp.int32),
        rows=rows,
        shared=shared,
        picked=picked,
    )


def _args(d, inp, qgate, pool, table, placeholder):
    # With the flag on the torch layer passes ``hidden`` for the (ignored) index operand.
    index_qk = inp["hidden"] if placeholder else inp["index_qk"]
    w = d.w
    return (qgate, inp["key"], inp["value"], index_qk, w["nq"], w["nk"], w["niq"], w["nik"],
            inp["positions"], w["trig"], pool, inp["starts"], inp["context"],
            jnp.asarray(d.ring), jnp.asarray(table), jnp.asarray(d.comp))


def _operands(inp):
    return (inp["shared"],) if inp["rows"] is None else (inp["rows"], inp["shared"])


def _one_step(d, variant, flag, inp, pools):
    """One draft pass; returns (new pools, per-rank outputs, tokens per request)."""
    first = inp["rows"] is not None
    if variant == "replicated":
        step = REPLICATED[flag][0 if first else 1]
        pool, out, selected = step(*_args(d, inp, inp["qgate"][0], pools[0], d.layout.full_table,
                                          flag), *_operands(inp), **OPTS)
        attn = jnp.asarray(out, jnp.float32).reshape(len(out), -1)
        head = d.w["head"][: attn.shape[1]]
        new_pools, outputs = [pool], [dict(out=out, selected=selected)]
    else:
        step = DCP[flag][0 if first else 1]
        pair_qgate = jnp.concatenate(inp["qgate"], axis=1)
        results = [step(*_args(d, inp, pair_qgate, pools[r], d.layout.dcp_table, flag),
                        *_operands(inp), rank_operand(r), **OPTS) for r in range(D)]
        new_pools = [res[0] for res in results]
        outputs = [dict(out=res[1], lse=res[2], selected=res[3]) for res in results]
        out_pair = jnp.stack([jnp.asarray(res[1], jnp.float32) for res in results])
        lse_pair = jnp.stack([jnp.asarray(res[2], jnp.float32) for res in results])
        attended = out_pair.shape[1]
        merged = []
        for r in range(D):
            gate = inp["qgate"][r][inp["picked"]].reshape(attended, H, 2 * DH)[..., DH:]
            merged.append(qsa_rpa.dcp_combine(out_pair, lse_pair, gate, rank_operand(r)))
        attn = jnp.concatenate([jnp.asarray(m, jnp.float32).reshape(attended, -1) for m in merged], 1)
        head = d.w["head"]
    logits = attn @ head.astype(jnp.float32)
    tokens = np.asarray(jnp.argmax(logits, axis=1))
    live = np.asarray(QUERY_LENS) > 0
    # Step 0 attends row r = request r; later steps token j = request j (the idle seat is last).
    per_request = [int(tokens[r]) if live[r] else -1 for r in range(len(live))]
    return new_pools, outputs, per_request


def _initial_pools(d, variant):
    if variant == "replicated":
        return [d.full]
    return [from_raw(project_rank_pool(raw(d.full), d.layout, r, keep_pages=d.index_pages))
            for r in range(D)]


def _bytes(x):
    return np.ascontiguousarray(np.asarray(x)).view(np.uint8)


def run_lockstep(variant, seed=0):
    d = _setup(seed)
    start = _initial_pools(d, variant)
    pools = {False: list(start), True: list(start)}
    initial = [raw(p) for p in start]
    outside = np.ones(len(initial[0]), bool)
    outside[d.index_pages] = False
    tokens, history = None, []
    for i in range(STEPS):
        inp = _step_inputs(d, i, tokens)
        got = {flag: _one_step(d, variant, flag, inp, pools[flag]) for flag in (False, True)}
        assert got[False][2] == got[True][2], f"step {i}: draft tokens differ"
        for r, (off, on) in enumerate(zip(got[False][1], got[True][1])):
            for name in off:
                np.testing.assert_array_equal(_bytes(on[name]), _bytes(off[name]),
                                              err_msg=f"step {i} rank {r}: {name} differs")
        for r in range(len(initial)):
            p_off, p_on = raw(got[False][0][r]), raw(got[True][0][r])
            np.testing.assert_array_equal(p_on[outside], p_off[outside],
                                          err_msg=f"step {i} rank {r}: main K/V / pool bytes differ")
            np.testing.assert_array_equal(p_on[d.index_pages], initial[r][d.index_pages],
                                          err_msg=f"step {i} rank {r}: flag on wrote index state")
        pools = {flag: got[flag][0] for flag in (False, True)}
        tokens = got[False][2]
        history.append(tokens)
    for r in range(len(initial)):
        final_off, final_on = raw(pools[False][r]), raw(pools[True][r])
        # Controls: the flag-off pipeline wrote the ring; both wrote main K/V.
        assert not np.array_equal(final_off[d.ring], initial[r][d.ring]), "flag off never wrote the ring"
        assert not np.array_equal(final_on[outside], initial[r][outside]), "no main K/V was written"
    flat = [t for step in history for t in step if t >= 0]
    assert len(set(flat)) > 1, f"degenerate draft tokens {history}"
    return history


@pytest.mark.parametrize("variant", ["dcp"])
def test_eight_step_draft_bitwise_dcp(interpret, rpa_stand_in, monkeypatch, partial_kernels,
                                      variant):
    monkeypatch.setattr(pooled_qsa, "select_blocks", stub_select)
    history = run_lockstep(variant)
    print("DRAFT1_BITWISE", variant, "steps", len(history), "tokens", history)


def test_eight_step_draft_bitwise_replicated(interpret, rpa_stand_in, monkeypatch):
    monkeypatch.setattr(pooled_qsa, "select_blocks", stub_select)
    history = run_lockstep("replicated")
    print("DRAFT1_BITWISE replicated steps", len(history), "tokens", history)


# ----------------------------------------------------------------------------
# Census


def _subjaxprs(value):
    if hasattr(value, "eqns"):
        yield value
    elif hasattr(value, "jaxpr") and hasattr(value, "consts"):
        yield value.jaxpr
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _subjaxprs(item)


def pallas_kernel_census(jaxpr, counts=None):
    """Static count of pallas_call kernels by name, through every nested jaxpr."""
    counts = collections.Counter() if counts is None else counts
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            info = eqn.params.get("name_and_src_info")
            name = getattr(info, "name", None) or eqn.params.get("name")
            if not name:
                # An unnamed call: "for <kernel function> at <file>:<line>".
                found = re.search(r"for (\w+)", str(info))
                name = f"<{found.group(1)}>" if found else "<unnamed>"
            counts[name] += 1
            continue
        for value in eqn.params.values():
            for sub in _subjaxprs(value):
                pallas_kernel_census(sub, counts)
    return counts


@pytest.fixture
def tpu7x_info(monkeypatch):
    """Trace (not run) the real Mosaic kernels on CPU: the TPU7x chip info they size by."""
    from jax.experimental.pallas import tpu as pltpu

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def _three_passes(d, variant, flag, pool):
    """Draft step 0 (rows) and two loop steps, chained through rank 0's pool. Both
    flags take the real index projection here (the census counts kernels, and a
    mutant that writes index state again must trace, not fail on the placeholder)."""
    tokens = [0] * len(QUERY_LENS)
    outs = []
    for i in range(3):
        inp = _step_inputs(d, i, tokens)
        first = inp["rows"] is not None
        if variant == "replicated":
            step = REPLICATED[flag][0 if first else 1]
            res = step(*_args(d, inp, inp["qgate"][0], pool, d.layout.full_table, False),
                       *_operands(inp), **OPTS)
        else:
            step = DCP[flag][0 if first else 1]
            res = step(*_args(d, inp, jnp.concatenate(inp["qgate"], axis=1), pool,
                              d.layout.dcp_table, False), *_operands(inp), rank_operand(0), **OPTS)
        pool = res[0]
        outs.append(res[1:])
    return pool, outs


RING_KERNELS = ("qwen4_exp_ring_tags_batched_read", "qwen4_exp_ring_tags_batched_write")


@pytest.mark.parametrize("variant", ["dcp", "replicated"])
def test_census_index_state_kernels_6_to_0(tpu7x_info, rpa_stand_in, monkeypatch, variant):
    monkeypatch.setattr(pooled_qsa, "select_blocks", stub_select)
    calls = collections.Counter()
    for name in ("_indexer_updates", "produce", "produce_main", "write_main_cache",
                 "write_compressed", "write_ring_and_tags"):
        original = getattr(pooled_qsa, name)

        def counted(*args, _original=original, _name=name, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(pooled_qsa, name, counted)
    d = _setup()
    pool = _initial_pools(d, variant)[0]
    census = {}
    for flag in (False, True):
        calls.clear()
        jaxpr = jax.make_jaxpr(functools.partial(_three_passes, d, variant, flag))(pool)
        census[flag] = (pallas_kernel_census(jaxpr.jaxpr), dict(calls))
        kernels = ", ".join(f"{k}={v}" for k, v in sorted(census[flag][0].items()))
        print(f"DRAFT1_CENSUS variant={variant} flag={'on' if flag else 'off'} "
              f"ring={sum(census[flag][0][k] for k in RING_KERNELS)} calls={dict(sorted(calls.items()))} "
              f"kernels: {kernels}")
    off, off_calls = census[False]
    on, on_calls = census[True]
    # The card's census: the draft's ring/tag kernels go 6 -> 0 over three passes.
    assert [off[k] for k in RING_KERNELS] == [3, 3]
    assert [on[k] for k in RING_KERNELS] == [0, 0]
    assert off_calls.get("_indexer_updates", 0) == 3 and on_calls.get("_indexer_updates", 0) == 0
    assert off_calls.get("produce", 0) == 3 and on_calls.get("produce", 0) == 0
    assert off_calls.get("produce_main", 0) == 0 and on_calls.get("produce_main", 0) == 3
    # Main K/V: one write per pass either way (8-token steps take its XLA scatter).
    assert off_calls.get("write_main_cache", 0) == 3 and on_calls.get("write_main_cache", 0) == 3
    # The compressed-key and ring/tag writers run once per pass with the flag off, never on.
    assert off_calls.get("write_compressed", 0) == 3 and on_calls.get("write_compressed", 0) == 0
    assert off_calls.get("write_ring_and_tags", 0) == 3 and on_calls.get("write_ring_and_tags", 0) == 0
    gone = {k for k in off if on[k] == 0}
    compressed = gone - set(RING_KERNELS)
    # Only index-state kernels disappear: the ring/tag pair and the compressed-key writer
    # (write_compressed's unnamed per-pair kernel at these 8-token steps, one per pass).
    assert compressed and all(off[k] == 3 for k in compressed), compressed
    assert all(on[k] == off[k] for k in off if k not in gone), "a non-index kernel changed"
    assert set(on) <= set(off), f"new kernels with the flag on: {set(on) - set(off)}"
