# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_MK_PLE_TRIM: each trim gives the same bytes as the path it replaces."""

import torch

from vllm_torchtpu.models.vllm.qwen4_exp.ple import (
    head_vocab_sizes,
    layer_multipliers,
    merged_embed_all_reduce,
    ngram_ids,
    token_requests,
)


def test_ngram_ids_with_constant_hash_parameters_match_buffers() -> None:
    vocab_size, eos, heads_per_ngram = 248320, 248044, 8
    multipliers = layer_multipliers(vocab_size, 3, 1, 1234)
    sizes = head_vocab_sizes(20_000_000, 2 * heads_per_ngram, 1)
    offsets = [sum(sizes[:h]) for h in range(len(sizes))]
    gen = torch.Generator().manual_seed(0)
    for num_tokens, qsl in ((16, [0, 4, 8, 12, 16]), (7, [0, 1, 5, 7])):
        input_ids = torch.randint(0, vocab_size, (num_tokens,), generator=gen)
        input_ids[3] = eos
        context = torch.randint(0, vocab_size, (len(qsl) - 1, 2), generator=gen)
        context[0, 1] = eos
        request, chunk_pos = token_requests(torch.tensor(qsl), num_tokens)
        args = (input_ids, context, request, chunk_pos)
        kw = dict(eos_token_id=eos, heads_per_ngram=heads_per_ngram)
        ref = ngram_ids(
            *args, torch.tensor(multipliers), torch.tensor(sizes), torch.tensor(offsets), **kw
        )
        got = ngram_ids(*args, multipliers, sizes, offsets, **kw)
        assert got.dtype == ref.dtype and torch.equal(got, ref)


def _all_reduce_over_ranks(per_rank_inputs):
    """Run `fn(rank, all_reduce)` per rank against a summing all-reduce: two passes,
    the first records each rank's operand."""
    seen = []

    def record(x):
        seen.append(x.clone())
        return x

    for rank in range(len(per_rank_inputs)):
        per_rank_inputs[rank](record)
    total = seen[0].float() if seen[0].is_floating_point() else seen[0].clone()
    for x in seen[1:]:
        total = total + (x.float() if x.is_floating_point() else x)
    total = total.to(seen[0].dtype)
    return [per_rank_inputs[r](lambda x: total) for r in range(len(per_rank_inputs))]


def test_merged_embed_all_reduce_matches_the_separate_collectives() -> None:
    """Embedding lanes reduce as before; raw bytes come out as the owner's bits
    (including -0.0 embedding entries and every byte pattern of the raw words)."""
    world, tokens, width = 8, 16, 256
    gen = torch.Generator().manual_seed(1)
    full = torch.randn(tokens, width, generator=gen).to(torch.bfloat16)
    full[0, :5] = -0.0
    owner = torch.randint(0, world, (tokens,), generator=gen)
    embeds = [torch.where((owner == r)[:, None], full, 0) for r in range(world)]
    raw_full = [
        torch.randint(-(2**31), 2**31 - 1, (tokens, 160), generator=gen, dtype=torch.int32),
        torch.randint(-128, 127, (tokens, 64), generator=gen, dtype=torch.int8),
    ]
    raw_owner = [torch.randint(0, world, (tokens, 1), generator=gen) for _ in raw_full]
    raws = [
        [torch.where(o == r, x, 0) for x, o in zip(raw_full, raw_owner, strict=True)]
        for r in range(world)
    ]
    # Reference: the embedding all-reduce alone, and integer sums of the raw rows.
    ref_embed = _all_reduce_over_ranks(
        [lambda ar, r=r: ar(embeds[r]) for r in range(world)]
    )[0]
    outs = _all_reduce_over_ranks(
        [lambda ar, r=r: merged_embed_all_reduce(embeds[r], raws[r], ar) for r in range(world)]
    )
    for embed, combined in outs:
        assert torch.equal(embed.view(torch.int16), ref_embed.view(torch.int16))
        for got, want in zip(combined, raw_full, strict=True):
            assert got.dtype == want.dtype and torch.equal(got, want)


def test_merged_head_embed_reads_only_fields_the_installed_vllm_embedding_has() -> None:
    """za8d: `_merged_head_embed` read `emb.reduce_results`, which the pinned vLLM's
    VocabParallelEmbedding does not set, and the AttributeError inside the whole-step
    trace disabled the megakernel. Every field it reads directly must be set by the
    installed class (optional ones go through getattr with the pinned default)."""
    import inspect
    import re

    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model

    # Plain and annotated assignments (`self.quant_method: QuantizeMethodBase = ...`).
    # Over the MRO: the served build may register a subclass under this name.
    have = set()
    for cls in VocabParallelEmbedding.__mro__:
        try:
            src_cls = inspect.getsource(cls)
        except (OSError, TypeError):
            continue
        have |= set(re.findall(r"self\.([a-z_]+)\s*(?::[^=\n]+)?=", src_cls))
    have |= {n for n, _ in inspect.getmembers(VocabParallelEmbedding)}
    have.add("weight")  # registered by the quant method's create_weights
    src = inspect.getsource(qwen4_model.Qwen4ExpBackbone._merged_head_embed)
    direct = set(re.findall(r"\bemb\.([a-z_]+)", src))
    optional = set(re.findall(r'getattr\(emb, "([a-z_]+)"', src))
    assert not (direct - have), (sorted(direct - have), [c.__module__ + "." + c.__qualname__ for c in VocabParallelEmbedding.__mro__])
    assert optional  # reduce_results / parallel_group / use_fused_embedding


def test_merged_head_embed_tolerates_an_embedding_without_reduce_results(monkeypatch) -> None:
    """The guard treats a missing `reduce_results` as the pinned vLLM's always-reduce
    and returns before any collective when no PLE layer is in the head groups."""
    from types import SimpleNamespace

    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model

    monkeypatch.setattr(envs, "VLLM_TPU_MK_PLE_TRIM", True)
    emb = SimpleNamespace(
        tp_size=8, parallel_group=None, use_fused_embedding=False,
        weight=torch.zeros(4, 8, dtype=torch.bfloat16),
    )
    layer = SimpleNamespace(ple=None)
    backbone = SimpleNamespace(embed_tokens=emb, layers=[layer, layer, layer])
    monkeypatch.setattr(qwen4_model, "get_tp_group", lambda: object())
    ids = torch.zeros(4, dtype=torch.int32)
    out = qwen4_model.Qwen4ExpBackbone._merged_head_embed(
        backbone, ids, ids.long(), torch.tensor([0, 4]), None, [[0], [1, 2]]
    )
    assert out is None  # no PLE layer: separate collectives, no AttributeError
