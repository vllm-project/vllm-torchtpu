# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_MK_IO_LEAN=ple / pleconv and VLLM_TPU_MK_PLE_TRIM: every lean form gives the
bits of the code it replaces, and the merged collective never branches on the token count.

Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest --noconftest <this file>
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import ple as kple
from vllm_torchtpu.models.vllm.qwen4_exp.ple import (
    head_vocab_sizes,
    layer_multipliers,
    merged_embed_all_reduce,
    ngram_ids,
    token_requests,
)

BUCKETS = (4, 8, 16, 24, 32, 48, 64)


def _qsl_cases(gen):
    """Chunk boundaries as the runner builds them, plus the edge cases: padded tail
    requests (repeated starts), starts at or past the token count, empty chunks."""
    yield 8, [0, 2, 4, 6, 8]
    yield 8, [0, 1, 5, 5, 5]  # padded requests repeat the last start
    yield 16, [0, 4, 8, 12, 16]
    yield 16, [0, 3, 7, 7, 30, 40]  # starts past the token count
    yield 4, [0, 0, 0, 4]  # empty leading chunks
    yield 1, [0, 1]
    for T in BUCKETS:
        for reqs in (1, 2, T // 4, T // 2, T):
            reqs = max(reqs, 1)
            cuts = np.sort(gen.integers(0, T + 3, reqs - 1))
            yield T, [0, *cuts.tolist(), int(min(T, cuts[-1] if len(cuts) else T))]


def test_token_requests_compare_form_matches_cumsum():
    gen = np.random.default_rng(0)
    n = 0
    for T, qsl in _qsl_cases(gen):
        q = torch.tensor(qsl, dtype=torch.int64)
        ref = token_requests(q, T)
        got = token_requests(q, T, compare=True)
        for a, b in zip(ref, got, strict=True):
            assert a.dtype == b.dtype and torch.equal(a, b), (T, qsl, ref, got)
        n += 1
    assert n > 30


def test_token_requests_compare_check_catches_strict_compare():
    """Negative control: `<` instead of `<=` moves the first token of every chunk."""
    q = torch.tensor([0, 2, 4, 6, 8])
    ref, _ = token_requests(q, 8)
    positions = torch.arange(8, dtype=torch.int32)
    wrong = (q[1:4].unsqueeze(0) < positions.unsqueeze(1)).sum(-1, dtype=torch.int32)
    assert not torch.equal(ref, wrong)


def _ngram_case(gen, num_tokens, qsl, vocab_size=248320, eos=248044):
    input_ids = torch.from_numpy(gen.integers(0, vocab_size, num_tokens))
    input_ids[min(3, num_tokens - 1)] = eos
    context = torch.from_numpy(gen.integers(0, vocab_size, (len(qsl) - 1, 2)))
    context[0, 1] = eos
    context[-1, 0] = vocab_size - 1
    return input_ids, context


@pytest.mark.parametrize("layer", [0, 1])
def test_ngram_ids_lean_matches_stock(layer):
    """int32 context gathered then widened + trace-time constants == int64 buffers."""
    vocab_size, eos, heads_per_ngram = 248320, 248044, 8
    multipliers = layer_multipliers(vocab_size, 3, layer, 1234)
    sizes = head_vocab_sizes(20_000_000, 2 * heads_per_ngram, layer)
    offsets = [sum(sizes[:h]) for h in range(len(sizes))]
    gen = np.random.default_rng(layer)
    kw = dict(eos_token_id=eos, heads_per_ngram=heads_per_ngram)
    for T, qsl in _qsl_cases(gen):
        ids, ctx = _ngram_case(gen, T, qsl)
        req, pos = token_requests(torch.tensor(qsl), T)
        req_c, pos_c = token_requests(torch.tensor(qsl), T, compare=True)
        ref = ngram_ids(
            ids, ctx, req, pos, torch.tensor(multipliers), torch.tensor(sizes),
            torch.tensor(offsets), **kw,
        )
        got = ngram_ids(
            ids.to(torch.int32).long(), ctx.to(torch.int32), req_c, pos_c, multipliers,
            sizes, offsets, narrow_context=True, **kw,
        )
        assert got.dtype == ref.dtype == torch.int64 and torch.equal(got, ref), (T, qsl)
        # Negative control: one changed context token changes some id.
        if T >= 1 and len(qsl) > 1:
            bad = ctx.clone()
            bad[0, 1] = (bad[0, 1] + 1) % vocab_size
            moved = ngram_ids(
                ids, bad.to(torch.int32), req_c, pos_c, multipliers, sizes, offsets,
                narrow_context=True, **kw,
            )
            if qsl[1] > 0:  # request 0 has a token that reads its context
                assert not torch.equal(moved, ref)


def test_context_lens_int32_sign():
    lens = torch.tensor([0, 1, 7, 102400, 204800, 262144, 2**31 - 1], dtype=torch.int64)
    assert torch.equal(lens.to(torch.int32) > 0, lens > 0)


# --------------------------------------------------------------------------- merge


def _simulate_ranks(per_rank, world):
    """Run `per_rank[r](all_reduce)` for every rank against a summing all-reduce (the
    float sum in fp32 then cast back, the integer sum wrapping); two passes."""
    seen = []

    def record(x):
        seen.append(x.clone())
        return x

    for r in range(world):
        per_rank[r](record)
    total = seen[0].float() if seen[0].is_floating_point() else seen[0].to(torch.int64)
    for x in seen[1:]:
        total = total + (x.float() if x.is_floating_point() else x.to(torch.int64))
    if seen[0].is_floating_point():
        total = total.to(seen[0].dtype)
    else:  # two's-complement wrap of the integer collective
        total = ((total + 2**31) % 2**32 - 2**31).to(seen[0].dtype)
    return [per_rank[r](lambda x: total) for r in range(world)]


def _merge_case(gen, tokens, width, world, raw_specs):
    full = torch.from_numpy(gen.standard_normal((tokens, width))).to(torch.bfloat16)
    full[0, :5] = -0.0
    full[min(1, tokens - 1), 7] = 0.0
    owner = torch.from_numpy(gen.integers(0, world, tokens))
    embeds = [torch.where((owner == r)[:, None], full, 0) for r in range(world)]
    raw_full = []
    for cols, dtype in raw_specs:
        info = torch.iinfo(dtype)
        raw_full.append(
            torch.from_numpy(gen.integers(info.min, info.max, (tokens, cols))).to(dtype)
        )
    raw_owner = [torch.from_numpy(gen.integers(0, world, (tokens, 1))) for _ in raw_full]
    raws = [
        [torch.where(o == r, x, 0) for x, o in zip(raw_full, raw_owner, strict=True)]
        for r in range(world)
    ]
    return full, embeds, raw_full, raws


@pytest.mark.parametrize("tokens", BUCKETS)
@pytest.mark.parametrize(
    "raw_specs",
    [
        [(2560, torch.int8)],  # FP8 table: 16 heads x 160 bytes
        [(1280, torch.int16)],  # BF16 table
        [(2560, torch.int8), (1280, torch.int16)],  # two PLE layers
        [(7, torch.int8)],  # a row that is not a whole number of words
    ],
    ids=["fp8", "bf16", "two", "odd"],
)
def test_merged_embed_all_reduce_matches_the_separate_collectives(tokens, raw_specs):
    world, width = 8, 2560
    gen = np.random.default_rng(tokens)
    _, embeds, raw_full, raws = _merge_case(gen, tokens, width, world, raw_specs)
    ref_embed = _simulate_ranks([lambda ar, r=r: ar(embeds[r]) for r in range(world)], world)[0]
    assert (ref_embed.view(torch.int16) != -32768).all()  # the float sum has no -0.0
    outs = _simulate_ranks(
        [lambda ar, r=r: merged_embed_all_reduce(embeds[r], raws[r], ar) for r in range(world)],
        world,
    )
    for embed, combined in outs:
        assert embed.dtype == torch.bfloat16 and embed.shape == ref_embed.shape
        assert torch.equal(embed.view(torch.int16), ref_embed.view(torch.int16))
        for got, want in zip(combined, raw_full, strict=True):
            assert got.dtype == want.dtype and got.shape == want.shape
            assert torch.equal(got, want)


def test_merged_check_catches_missing_negative_zero_map(monkeypatch):
    """Negative control: without the -0.0 -> +0.0 map the int32 sum keeps the owner's
    -0.0 and differs from the float all-reduce."""
    world, tokens, width = 8, 8, 2560
    gen = np.random.default_rng(5)
    _, embeds, _, raws = _merge_case(gen, tokens, width, world, [(2560, torch.int8)])
    ref = _simulate_ranks([lambda ar, r=r: ar(embeds[r]) for r in range(world)], world)[0]
    real_where = torch.where
    monkeypatch.setattr(
        torch, "where",
        lambda c, a, b: b if a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16 else real_where(c, a, b),
    )
    embed, _ = _simulate_ranks(
        [lambda ar, r=r: merged_embed_all_reduce(embeds[r], raws[r], ar) for r in range(world)],
        world,
    )[0]
    assert not torch.equal(embed.view(torch.int16), ref.view(torch.int16))


# ------------------------------------------------------- backbone: embed and merge


def _patched_vpe():
    import vllm.model_executor.layers.vocab_parallel_embedding as vpe

    from vllm_torchtpu import _patch_vllm_vocab_parallel_embedding

    _patch_vllm_vocab_parallel_embedding()
    return vpe


class _FakePLE:
    def __init__(self, raw):
        self.raw = raw
        self.cpu_offload = False
        self.parallel = SimpleNamespace(across_dp=False, group=None)
        self.calls = []

    def lookup_partial(self, inputs, valid_tokens, *, lean=False):
        self.calls.append(lean)
        return self.raw[: inputs.input_ids.shape[0]]


def _backbone(rank, world, vocab, width, table, raw, monkeypatch, tp):
    from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model

    per = vocab // world
    si = SimpleNamespace(
        org_vocab_start_index=rank * per,
        org_vocab_end_index=(rank + 1) * per,
        num_org_vocab_padding=0,
        added_vocab_start_index=world * per,
        added_vocab_end_index=world * per,
    )
    weight = table[rank * per : (rank + 1) * per]
    emb = SimpleNamespace(
        tp_size=world,
        shard_indices=si,
        weight=weight,
        quant_method=SimpleNamespace(embedding=lambda e, ids: torch.nn.functional.embedding(ids, e.weight)),
        org_vocab_start_index=torch.tensor(si.org_vocab_start_index),
        org_vocab_end_index=torch.tensor(si.org_vocab_end_index),
        num_org_vocab_padding=torch.tensor(0),
        added_vocab_start_index=torch.tensor(si.added_vocab_start_index),
        added_vocab_end_index=torch.tensor(si.added_vocab_end_index),
    )
    ple = _FakePLE(raw)
    layers = [SimpleNamespace(ple=None), SimpleNamespace(ple=ple), SimpleNamespace(ple=None)]
    ple.parallel.group = tp
    bb = SimpleNamespace(embed_tokens=emb, layers=layers)
    bb.embed_bounds_i32 = torch.tensor(
        [si.org_vocab_start_index, si.org_vocab_end_index, 0, world * per, world * per],
        dtype=torch.int32,
    )
    for name in ("_embed_layout_plain", "_embed_part", "_mk_embed", "_merged_head_embed"):
        fn = getattr(qwen4_model.Qwen4ExpBackbone, name)
        setattr(bb, name, fn.__get__(bb))
    return bb, ple


@pytest.mark.parametrize("tokens", BUCKETS)
def test_backbone_merged_and_lean_embed_match_stock(tokens, monkeypatch):
    """Per rank: `_mk_embed(lean)` == the patched embedding forward's rows; the merged
    collective (PLE_TRIM, lean and not) == the embedding all-reduce + the raw sum."""
    from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model

    vpe = _patched_vpe()
    monkeypatch.setattr(qwen4_model, "get_masked_input_and_mask", vpe.get_masked_input_and_mask)
    tp = object()
    monkeypatch.setattr(qwen4_model, "get_tp_group", lambda: tp)
    world, vocab, width = 8, 8 * 64, 2560
    gen = np.random.default_rng(tokens)
    table = torch.from_numpy(gen.standard_normal((vocab, width))).to(torch.bfloat16)
    table[3, :4] = -0.0
    ids = torch.from_numpy(gen.integers(0, vocab, tokens)).to(torch.int32)
    ids[0] = 3
    raw_full = torch.from_numpy(gen.integers(-128, 127, (tokens, 2560))).to(torch.int8)
    raw_owner = torch.from_numpy(gen.integers(0, world, (tokens, 1)))
    qsl = torch.tensor([0, tokens])
    groups = [[0], [1, 2]]

    def stock_embed(bb, ar):
        # VocabParallelEmbedding.forward as patched (int64 buffers), then the all-reduce.
        e = bb.embed_tokens
        masked, mask = vpe.get_masked_input_and_mask(
            ids, e.org_vocab_start_index, e.org_vocab_end_index, e.num_org_vocab_padding,
            e.added_vocab_start_index, e.added_vocab_end_index,
        )
        part = e.quant_method.embedding(e, masked.long())
        part.masked_fill_(mask.unsqueeze(-1), 0)
        return ar(part)

    bbs = [
        _backbone(r, world, vocab, width, table, torch.where(raw_owner == r, raw_full, 0),
                  monkeypatch, tp)
        for r in range(world)
    ]
    ref = _simulate_ranks([lambda ar, r=r: stock_embed(bbs[r][0], ar) for r in range(world)], world)
    for lean in (False, True):
        monkeypatch.setattr(qwen4_model, "tensor_model_parallel_all_reduce", None)
        got = _simulate_ranks(
            [
                lambda ar, r=r: (
                    monkeypatch.setattr(qwen4_model, "tensor_model_parallel_all_reduce", ar),
                    bbs[r][0]._mk_embed(ids, True) if lean else stock_embed(bbs[r][0], ar),
                )[1]
                for r in range(world)
            ],
            world,
        )
        for a, b in zip(ref, got, strict=True):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), lean
        monkeypatch.setattr(envs, "VLLM_TPU_MK_PLE_TRIM", True)
        merged = _simulate_ranks(
            [
                lambda ar, r=r: (
                    monkeypatch.setattr(qwen4_model, "tensor_model_parallel_all_reduce", ar),
                    bbs[r][0]._merged_head_embed(ids, ids.long(), qsl, None, groups, lean=lean),
                )[1]
                for r in range(world)
            ],
            world,
        )
        monkeypatch.setattr(envs, "VLLM_TPU_MK_PLE_TRIM", False)
        for (embedding, combined), want in zip(merged, ref, strict=True):
            assert torch.equal(embedding.view(torch.int16), want.view(torch.int16)), lean
            assert list(combined) == [1] and torch.equal(combined[1], raw_full), lean
        assert bbs[0][1].calls[-1] is lean


def test_merged_head_embed_trace_serves_every_bucket(monkeypatch):
    """The decision reads no token count: one dynamic trace of the merged path leaves no
    shape guard that refuses an MK bucket (za9g: `shape[0] > 8` refused bucket 32).
    Negative control: the old row cap is refused at 16 and above."""
    from vllm_torchtpu.compilation.shape_variants import trace_shape_env, unsupported_reason
    from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model

    vpe = _patched_vpe()
    monkeypatch.setattr(qwen4_model, "get_masked_input_and_mask", vpe.get_masked_input_and_mask)
    tp = object()
    monkeypatch.setattr(qwen4_model, "get_tp_group", lambda: tp)
    monkeypatch.setattr(qwen4_model, "tensor_model_parallel_all_reduce", lambda x: x * 1)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_PLE_TRIM", True)
    world, vocab, width = 8, 8 * 64, 256
    gen = np.random.default_rng(0)
    table = torch.from_numpy(gen.standard_normal((vocab, width))).to(torch.bfloat16)
    raw = torch.from_numpy(gen.integers(-128, 127, (128, 160))).to(torch.int8)
    bb, _ = _backbone(0, world, vocab, width, table, raw, monkeypatch, tp)
    verdicts = {}

    def backend(name):
        def compile_fn(gm, example_inputs):
            env = trace_shape_env(gm)
            verdicts[name] = {b: unsupported_reason(env, b) for b in BUCKETS}
            return gm.forward

        return compile_fn

    def merged(ids, qsl):
        return bb._merged_head_embed(ids, ids.long(), qsl, None, [[0], [1, 2]], lean=True)

    def capped(ids, qsl):
        if ids.shape[0] > 8:
            return bb._mk_embed(ids, True), {}
        return merged(ids, qsl)

    for name, fn in (("merged", merged), ("capped", capped)):
        torch._dynamo.reset()
        ids = torch.zeros(4, dtype=torch.int32)
        torch._dynamo.mark_dynamic(ids, 0)
        torch.compile(fn, backend=backend(name), dynamic=False)(ids, torch.tensor([0, 4]))
    assert all(v is None for v in verdicts["merged"].values()), verdicts["merged"]
    assert verdicts["capped"][4] is None and verdicts["capped"][8] is None
    assert all(verdicts["capped"][b] is not None for b in BUCKETS if b > 8), verdicts


# ------------------------------------------------------------------ conv (pleconv)


@pytest.mark.parametrize("T", [1, 3, 4, 8, 13, 16, 24, 32, 48, 64])
def test_shifted_rows_matches_clipped_gather(T):
    x = jnp.asarray(np.random.default_rng(T).standard_normal((T, 24)), jnp.bfloat16)
    t = jnp.arange(T)
    for d in range(0, 14):
        ref = x[jnp.clip(t - d, 0, T - 1)]
        got = kple.shifted_rows(x, d)
        assert got.shape == ref.shape and np.array_equal(
            np.asarray(got).view(np.uint16), np.asarray(ref).view(np.uint16)
        ), d
    if T > 1:  # negative control: one row too far
        assert not np.array_equal(np.asarray(kple.shifted_rows(x, 2)), np.asarray(x[jnp.clip(t - 1, 0, T - 1)]))


def _conv_inputs(gen, T, R, C=256, dilation=3, kernel=4, slots=40):
    state_len = (kernel - 1) * dilation
    x = jnp.asarray(gen.standard_normal((T, C)), jnp.bfloat16)
    w = jnp.asarray(gen.standard_normal((C, kernel)), jnp.bfloat16)
    hist = jnp.asarray(gen.standard_normal((slots, state_len, C)), jnp.bfloat16)
    cuts = np.sort(gen.integers(0, T + 1, R - 1)) if R > 1 else np.array([], int)
    qsl = jnp.asarray([0, *cuts.tolist(), T], jnp.int32)
    ids = gen.permutation(slots)[:R].astype(np.int32)
    ids[-1] = -1 if R > 2 else ids[-1]
    initial = jnp.asarray(gen.integers(0, 2, R).astype(bool))
    return x, w, hist, qsl, jnp.asarray(ids), initial, dilation


@pytest.mark.parametrize("fast", [False, True], ids=["plain", "hist_fast"])
@pytest.mark.parametrize("T,R", [(4, 4), (8, 2), (8, 8), (16, 4), (24, 6), (32, 16), (64, 16)])
def test_short_conv_pleconv_bitwise(monkeypatch, T, R, fast):
    """`short_conv` lever on vs off, both history forms: the served op
    (`pooled_short_conv_speculative`) runs it with request-owned history."""
    monkeypatch.setattr(envs, "VLLM_TPU_PLE_CONV_HISTORY_FAST", fast)
    gen = np.random.default_rng(T * 100 + R)
    x, w, hist, qsl, ids, initial, dil = _conv_inputs(gen, T, R)
    outs = {}
    for lean in ("", "pleconv"):
        monkeypatch.setattr(envs, "VLLM_TPU_MK_IO_LEAN", lean)
        for owned in (False, True):
            outs[(lean, owned)] = kple.short_conv(
                x, w, hist, qsl, ids, initial, dilation=dil, request_owned_history=owned
            )
    for owned in (False, True):
        for a, b in zip(outs[("", owned)], outs[("pleconv", owned)], strict=True):
            assert np.array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16))
