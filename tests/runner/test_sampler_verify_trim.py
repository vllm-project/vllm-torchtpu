# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU_SAMPLER_VERIFY_TRIM: the fused verify chain on 8 target rows.

Exactness: for every case the trimmed chain's selected tokens equal the
untrimmed chain's, bit for bit, on the same logits and the same uniforms
(drawn once at the full shape from one seeded generator, as the runner draws
them). The metadata comes from the real get_spec_decode_metadata, so the
"real slots first" layout the trim relies on is the production one.

Positive controls (each must change some output, else the equality above
proves nothing):
  * trimming below the real slot count (drops a real draft slot);
  * drawing the uniforms at the trimmed shape (a different generator stream).

Runs compiled on a TPU when one is attached, else eager on CPU (the `dev`
fixture; conftest's `device` always demands a TPU).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch._dynamo

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

K = 3
PADDED_NUM_REQS = 8
TOKEN_PADDINGS = [16, 32, 64, 128]
# Stack sampler kwargs (TPU_SAMPLER_VOCAB_PARALLEL + FAST_TOP_P, top_k off)
# and the flags-off default.
VARIANTS = [
    {"need_top_k": False, "need_top_p": True, "radix_bits": 1, "fast_top_p": True},
    {},
]
# draft counts per request; every sum <= 8 at padded logits length 16.
CASES = [[3], [3, 3], [2, 1], [3, 0], [1, 1, 3], [3, 3, 2]]


@pytest.fixture
def dev():
    try:
        import torch_tpu  # noqa: F401

        d = torch.device("tpu")
        torch.zeros(1, device=d)
        return d
    except Exception:
        return torch.device("cpu")


@pytest.fixture(autouse=True)
def _raise_dynamo_recompile_limit():
    cfg = torch._dynamo.config
    names = [n for n in ("recompile_limit", "cache_size_limit") if hasattr(cfg, n)]
    saved = {n: getattr(cfg, n) for n in names}
    for n in names:
        setattr(cfg, n, 1024)
    yield
    for n, v in saved.items():
        setattr(cfg, n, v)


def _metadata(num_draft_tokens, input_ids, device):
    nd = np.asarray(num_draft_tokens, dtype=np.int32)
    cu = np.cumsum(nd + 1).astype(np.int32)
    runner = SimpleNamespace(
        input_ids_cpu=input_ids,
        arange_np=np.arange(4096, dtype=np.int64),
        num_tokens_paddings=TOKEN_PADDINGS,
        device=device,
    )
    mgr = SpeculativeDecodingManager(runner)
    return mgr.get_spec_decode_metadata(nd, cu, PADDED_NUM_REQS)


def _fused(device):
    fn = TPUModelRunner.spec_sample_verify_fused
    if device.type != "tpu":
        fn = getattr(fn, "_torchdynamo_orig_callable", fn)
    return fn


def _inputs(case, seed, vocab, device):
    """Logits, metadata and the runner's three draws for one verify chunk."""
    gen = torch.Generator().manual_seed(seed)
    rows = int(np.sum(np.asarray(case) + 1))
    padded = next(p for p in TOKEN_PADDINGS if p >= rows)
    # Peaked rows (scale up to 8) so drafts are often accepted and the later
    # slots matter; flat rows so top-p keeps many tokens and recovery happens.
    scale = torch.rand(padded, 1, generator=gen) * 8.0
    logits = (torch.randn(padded, vocab, generator=gen) * scale).to(torch.bfloat16)
    # The draft at slot j of a request is the token of input position j+1:
    # half the time the target argmax of that slot (accepted often).
    input_ids = torch.randint(0, vocab, (4096,), dtype=torch.int32, generator=gen)
    am = logits.float().argmax(dim=-1)
    pos = 0
    for n in case:
        for j in range(n):
            if torch.rand(1, generator=gen).item() < 0.7:
                input_ids[pos + j + 1] = int(am[pos + j])
        pos += n + 1
    md = _metadata(case, input_ids, device)
    assert md.target_logits_indices.shape[0] == padded
    temps = torch.ones((PADDED_NUM_REQS, 1), dtype=logits.dtype)
    top_k = torch.zeros((PADDED_NUM_REQS, 1), dtype=torch.int32)
    top_p = torch.ones((PADDED_NUM_REQS, 1), dtype=torch.float32)
    top_p[: len(case)] = 0.95
    return gen, logits, md, temps, top_k, top_p, padded


def _draw(gen, num_reqs, num_tokens, vocab, dtype):
    # Same shapes, dtypes and order as _sample_spec_verify_chunk's fused branch.
    bonus_u = torch.rand((num_reqs, vocab), dtype=dtype, generator=gen)
    accept_u = torch.rand((num_tokens,), dtype=torch.float32, generator=gen)
    recover_u = torch.rand((num_tokens, vocab), dtype=torch.float32, generator=gen)
    return bonus_u, accept_u, recover_u


def _run(fn, device, logits, md, temps, top_k, top_p, draws, n_tok, **kw):
    self_ = SimpleNamespace(rejection_sampler=RejectionSampler())
    bonus_u, accept_u, recover_u = draws
    out = fn(
        self_,
        logits.to(device),
        md.bonus_logits_indices,
        md.target_logits_indices[:n_tok] if n_tok else md.target_logits_indices,
        temps.to(device),
        top_k.to(device),
        top_p.to(device),
        md.draft_token_ids[:n_tok] if n_tok else md.draft_token_ids,
        md.draft_lengths,
        md.segment_ids[:n_tok] if n_tok else md.segment_ids,
        md.group_indices[:n_tok] if n_tok else md.group_indices,
        bonus_u.to(device),
        accept_u.to(device),
        recover_u.to(device),
        K,
        **kw,
    )
    return out.cpu()


def _vocab(device):
    return 154880 if device.type == "tpu" else 128 * 160


@pytest.mark.parametrize("variant", VARIANTS, ids=["stack", "default"])
@pytest.mark.parametrize("case", CASES, ids=[str(c) for c in CASES])
def test_trim_is_exact(dev, case, variant):
    vocab, fn = _vocab(dev), _fused(dev)
    for seed in range(6):
        gen, logits, md, temps, top_k, top_p, padded = _inputs(case, seed, vocab, dev)
        draws = _draw(gen, PADDED_NUM_REQS, padded, vocab, logits.dtype)
        full = _run(fn, dev, logits, md, temps, top_k, top_p, draws, 0, **variant)
        trim = _run(
            fn,
            dev,
            logits,
            md,
            temps,
            top_k,
            top_p,
            draws,
            0,
            verify_rows=8,
            **variant,
        )
        assert torch.equal(full, trim), (case, seed, full, trim)


def test_positive_control_dropped_real_slot_is_caught(dev):
    """verify_rows below the real slot count drops a real draft slot: the
    comparison must see it."""
    vocab, fn = _vocab(dev), _fused(dev)
    case, differs = [3, 3, 2], 0
    for seed in range(6):
        gen, logits, md, temps, top_k, top_p, padded = _inputs(case, seed, vocab, dev)
        draws = _draw(gen, PADDED_NUM_REQS, padded, vocab, logits.dtype)
        full = _run(fn, dev, logits, md, temps, top_k, top_p, draws, 0, **VARIANTS[0])
        bad = _run(
            fn,
            dev,
            logits,
            md,
            temps,
            top_k,
            top_p,
            draws,
            0,
            verify_rows=6,
            **VARIANTS[0],
        )
        differs += int(not torch.equal(full, bad))
    assert differs > 0


def test_positive_control_trimmed_draw_shape_is_caught(dev):
    """Drawing accept_u / recover_u at the trimmed shape moves the generator
    stream: the comparison must see it (why the trim slices full draws)."""
    vocab, fn = _vocab(dev), _fused(dev)
    case, differs = [3, 3], 0
    for seed in range(6):
        gen, logits, md, temps, top_k, top_p, padded = _inputs(case, seed, vocab, dev)
        state = gen.get_state()
        draws = _draw(gen, PADDED_NUM_REQS, padded, vocab, logits.dtype)
        full = _run(fn, dev, logits, md, temps, top_k, top_p, draws, 0, **VARIANTS[0])
        gen.set_state(state)
        small = _draw(gen, PADDED_NUM_REQS, 8, vocab, logits.dtype)
        bad = _run(fn, dev, logits, md, temps, top_k, top_p, small, 8, **VARIANTS[0])
        differs += int(not torch.equal(full, bad))
    assert differs > 0


@pytest.mark.parametrize(
    "padded,drafts,flag,expect",
    [
        (16, [3], True, 8),
        (16, [3, 3, 2], True, 8),
        (16, [3, 3, 3], True, 0),  # 9 real slots
        (8, [1], True, 0),  # nothing to trim
        (32, [2, 2], True, 8),
        (64, [1], True, 0),  # above _VERIFY_TRIM_MAX_LEN: not precompiled
        (16, [3], False, 0),
    ],
)
def test_verify_trim_rows_gate(padded, drafts, flag, expect):
    r = SimpleNamespace(_sampler_verify_trim=flag)
    got = TPUModelRunner._verify_trim_rows(r, padded, np.asarray(drafts))
    assert got == expect
    assert tpu_runner._VERIFY_TRIM_ROWS == 8
