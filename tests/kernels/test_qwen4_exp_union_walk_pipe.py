# SPDX-License-Identifier: Apache-2.0
"""q38 UNION-BUILD: the gathered union walk's software-pipelined tile loop (walk_pipe, VLLM_TPU_QSA_PREFILL_UNION_WALK_PIPE).

Checks, with the Pallas kernels in interpret mode off TPU (bitwise claims with XLA's excess precision off, pinned before jax
creates its CPU client):
  - the env key reaches union_walk_prefill only when set (unset: no new kwarg) and reaches the gathered walk kernel (the
    kernel name gains _pipe); the kernel refuses walk_pipe off the gathered walk;
  - walk_pipe is bitwise the served walk on packed unaligned steps with the served option set (both forms: the B1 '_gathered'
    form and the in-place / gathered-lean form), at tile counts 1, 2, 3 and more per group (odd and even), with QA-1 on, with
    qk16, on a peaked softmax, and on the over-cap fallback;
  - a mutation of the pipeline order bites: stage B run on the wrong slot (the pipeline's one ordering invariant) changes
    the output (positive control), so the bitwise checks above can fail.
"""

import os

_STRICT = "--xla_allow_excess_precision=false"
if _STRICT not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _STRICT).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax._src import xla_bridge  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

_PINNED_BEFORE_INIT = not xla_bridge.backends_are_initialized()

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud  # noqa: E402

RATIO, TOPK, PAGE = 4, 512, 3200
CAP = 8192
SPARE = ((37, 300, 129, 201, 333), (5000, 1500, 4000, 700, 2900))
FOUR = ((37, 300, 129, 201), (5000, 1500, 4000, 700))
SERVED_GL = dict(group=64, kv_tile=1024, union_cap=CAP, min_context=0, lean=True, listed="reference",
                 gathered="reference", gather_chunks=3, in_place=True, gathered_lean=True, compact_reqs=4,
                 share_bits=True, lazy_share=True)
SERVED_G = dict(group=64, kv_tile=1024, union_cap=CAP, min_context=0, lean=True, listed="reference",
                gathered="reference", gather_chunks=3, compact_reqs=4, share_bits=True, lazy_share=True)
ENV = "VLLM_TPU_QSA_PREFILL_UNION_WALK_PIPE"


@pytest.fixture
def kernels(monkeypatch):
    jax.clear_caches()
    if jax.default_backend() != "tpu":
        from jax._src import tpu_info

        monkeypatch.setitem(
            tpu_info.registry,
            jax.devices()[0].device_kind,
            lambda: tpu_info._get_tpu_info_impl(tpu_info.ChipVersion.TPU_7X, 1),
        )
        original = pl.pallas_call

        def interpreted(*args, **kwargs):
            kwargs["interpret"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(bsp.pl, "pallas_call", interpreted)
    monkeypatch.delenv(ENV, raising=False)
    yield
    jax.clear_caches()


def _packed_step(seed, rows, cached, tokens=1024, q_std=0.5, local=32):
    """A ``tokens``-row step of len(rows) requests on their own pages of a permuted, finite FP8 pool (as the QA-1 tests).
    ``local``: each row selects its last ``local`` groups plus random others (a small ``local`` and short contexts give
    unions of a few tiles; long contexts give many)."""
    rng = np.random.default_rng(seed)
    pages_per_seq = max(-(-(c + n) // PAGE) for n, c in zip(rows, cached))
    pool_pages = pages_per_seq * len(rows)
    assert sum(rows) <= tokens
    starts = np.concatenate([[0], np.cumsum(rows)]).astype(np.int32)
    owner = rng.permutation(pool_pages)
    lengths = np.zeros(tokens, np.int32)
    blocks = np.full((tokens, TOPK), -1, np.int32)
    table = np.zeros((tokens, pages_per_seq), np.int32)
    raw = rng.integers(0, 0x40, size=(pool_pages, PAGE, 1, 4, 256), dtype=np.uint8)  # exponent < 15: finite
    raw |= rng.integers(0, 2, size=raw.shape, dtype=np.uint8) << 7
    for r, (n, ctx) in enumerate(zip(rows, cached)):
        pages = owner[r * pages_per_seq:(r + 1) * pages_per_seq]
        for i in range(n):
            t = starts[r] + i
            length = ctx + i + 1
            lengths[t] = length
            table[t] = pages
            complete = length // RATIO
            if complete:
                loc = list(range(max(0, complete - local), complete))
                rest = rng.permutation(complete)[: TOPK - len(loc)]
                pick = np.unique(np.concatenate([loc, rest]))[:TOPK]
                blocks[t, : len(pick)] = rng.permutation(pick)
    q = jnp.asarray(rng.normal(0, q_std, (tokens, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, (tokens, 24, 256)), jnp.bfloat16)
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    return q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), pool, jnp.asarray(starts)


def _bits(x):
    return np.asarray(x).view(np.uint16)


def _walk(step, base, **options):
    q, gate, blocks, lengths, table, pool, starts = step
    params = dict(base)
    params.update(options)
    out, cache = ud.union_walk_prefill(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO,
                                       **params)
    assert cache is pool
    return np.asarray(out)


def _kernel_names(jaxpr):
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            found.append(str(eqn.params.get("name_and_src_info", eqn.params.get("name"))).split(" ")[0])
        for param in eqn.params.values():
            for item in param if isinstance(param, (tuple, list)) else (param,):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    found += _kernel_names(inner)
    return found


def _walk_names(step, base, **options):
    q, gate, blocks, lengths, table, pool, starts = step
    params = dict(base)
    params.update(options)
    jaxpr = jax.make_jaxpr(lambda: ud.union_walk_prefill(q, gate, blocks, lengths, table, pool, starts, kv_heads=2,
                                                         ratio=RATIO, **params))().jaxpr
    return [n for n in _kernel_names(jaxpr) if n.startswith("qsa_union_walk_g")]


def _tiles(step, group=64):
    """Per live query group, the walk's tile count (host plan census, as union_bits_plan counts the union)."""
    _, _, blocks, lengths, _, _, starts = step
    blocks, lengths, starts = np.asarray(blocks), np.asarray(lengths), np.asarray(starts)
    out = []
    for r in range(len(starts) - 1):
        lo, hi = int(starts[r]), int(starts[r + 1])
        for g0 in range(lo, hi, group):
            sel = set()
            for i in range(g0, min(g0 + group, hi)):
                sel.update(int(x) for x in blocks[i] if x >= 0)
                if lengths[i] % RATIO:
                    sel.add(int(lengths[i]) // RATIO)
            out.append(-(-len(sel) * RATIO // 1024))
    return out


def test_excess_precision_is_off_in_this_process():
    assert _STRICT in os.environ["XLA_FLAGS"] and _PINNED_BEFORE_INIT, (os.environ.get("XLA_FLAGS"), _PINNED_BEFORE_INIT)


def test_kernel_refuses_walk_pipe_off_the_gathered_walk(kernels):
    step = _packed_step(1, (128,), (2000,), tokens=128)
    q, gate, blocks, lengths, table, pool, starts = step
    with pytest.raises(ValueError, match="walk_pipe"):
        ud.union_walk_prefill(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO, group=64,
                              kv_tile=1024, union_cap=CAP, min_context=0, lean=True, listed="reference",
                              walk_pipe=True)


@pytest.mark.parametrize("base", [SERVED_GL, SERVED_G], ids=["glean", "gathered"])
def test_names_and_bitwise_packed_unaligned(kernels, base):
    step = _packed_step(2, SPARE[0], SPARE[1])
    assert any(t >= 3 for t in _tiles(step)) and any(t <= 2 for t in _tiles(step)), _tiles(step)
    names = _walk_names(step, base, walk_pipe=True)
    assert names and all(n.endswith("_pipe") for n in names), names
    assert not any(n.endswith("_pipe") for n in _walk_names(step, base)), "flag off must not build the pipe kernel"
    off = _walk(step, base)
    on = _walk(step, base, walk_pipe=True)
    assert np.array_equal(_bits(off), _bits(on))


@pytest.mark.parametrize("ctx", [600, 1500, 2500, 3500, 7300, 13000], ids=lambda c: "ctx%d" % c)
def test_bitwise_tile_counts(kernels, ctx):
    # one request: tile counts 1, 2, 3, 4-5, 6+ per group as the context grows (odd and even counts, the remainder half and
    # both final-stage-B branches)
    step = _packed_step(3 + ctx, (192,), (ctx,), tokens=256, local=48)
    off = _walk(step, SERVED_GL)
    on = _walk(step, SERVED_GL, walk_pipe=True)
    assert np.array_equal(_bits(off), _bits(on)), _tiles(step)


def test_tile_counts_cover_odd_and_even(kernels):
    seen = set()
    for ctx in (600, 1500, 2500, 3500, 7300, 13000):
        seen.update(_tiles(_packed_step(3 + ctx, (192,), (ctx,), tokens=256, local=48)))
    assert {1, 2, 3} <= seen and any(t % 2 == 0 and t >= 4 for t in seen) and any(t % 2 == 1 and t >= 5 for t in seen), seen


def test_bitwise_with_qa1_and_qk16(kernels):
    step = _packed_step(4, FOUR[0], FOUR[1])
    for extra in (dict(mask_view=True, vguard_lean=True), dict(qk16=True), dict(mask_view=True, vguard_lean=True,
                                                                               align_gather=True)):
        off = _walk(step, SERVED_GL, **extra)
        on = _walk(step, SERVED_GL, walk_pipe=True, **extra)
        assert np.array_equal(_bits(off), _bits(on)), extra


def test_bitwise_peaked_softmax(kernels):
    step = _packed_step(5, FOUR[0], FOUR[1], q_std=4.0)
    assert np.array_equal(_bits(_walk(step, SERVED_GL)), _bits(_walk(step, SERVED_GL, walk_pipe=True)))


def test_bitwise_over_cap_fallback(kernels):
    step = _packed_step(6, (128, 64), (12000, 9000), tokens=256, local=64)
    base = dict(SERVED_GL, union_cap=256)  # every group's union is over the cap: the served fallback runs either way
    assert np.array_equal(_bits(_walk(step, base)), _bits(_walk(step, base, walk_pipe=True)))


def test_env_key_reaches_the_walk(kernels, monkeypatch):
    """Unset: union_walk_prefill gets no walk_pipe kwarg. Set: it arrives as True, and the output is the same."""
    calls = []
    original = ud.union_walk_prefill

    def spy(*args, **kw):
        calls.append(dict(kw))
        return original(*args, **kw)

    for name, value in (("VLLM_TPU_QSA_PREFILL_UNION_WALK", "64"), ("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", "0"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_CAP", str(CAP)), ("VLLM_TPU_QSA_PREFILL_LEAN", "1"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_LISTED", "reference"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_GATHERED", "reference"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_IN_PLACE", "1"), ("VLLM_TPU_QSA_PREFILL_UNION_GATHERED_LEAN", "1")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ud, "union_walk_prefill", spy)
    step = _packed_step(7, (37, 300, 129), (5000, 1500, 4000), tokens=1024)
    q, gate, blocks, lengths, table, pool, starts = step
    want, _ = qsa_rpa._prefill_attention(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO)
    assert calls and "walk_pipe" not in calls[-1], calls
    monkeypatch.setenv(ENV, "1")
    jax.clear_caches()
    got, _ = qsa_rpa._prefill_attention(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO)
    assert calls[-1].get("walk_pipe") is True, calls[-1]
    assert np.array_equal(_bits(want), _bits(got))


def test_mutation_wrong_slot_bites(kernels, monkeypatch):
    # Positive control: stage B of the remainder / final half reading the other slot is a real pipeline-order bug; the
    # bitwise comparison must see it (else the checks above could never fail).
    src = open(ud.__file__).read()
    bad = src.replace("            stage_b(take(0))\n", "            stage_b(take(1))\n", 1)
    assert bad != src
    ns = {}
    code = compile(bad, ud.__file__, "exec")
    exec(code, ns)  # noqa: S102 - the mutated module, in its own namespace
    monkeypatch.setattr(ns["pl"], "pallas_call", bsp.pl.pallas_call)
    step = _packed_step(8, (192,), (2500,), tokens=256, local=48)
    assert any(t % 2 == 1 for t in _tiles(step)), _tiles(step)  # pipe_last_even runs for odd tile counts
    q, gate, blocks, lengths, table, pool, starts = step
    good = _walk(step, SERVED_GL, walk_pipe=True)
    mut, _ = ns["union_walk_prefill"](q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO,
                                      walk_pipe=True, **SERVED_GL)
    assert not np.array_equal(_bits(good), _bits(np.asarray(mut))), "the wrong-slot mutant was not caught"
