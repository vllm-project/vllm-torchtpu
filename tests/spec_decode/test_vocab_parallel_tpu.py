# SPDX-License-Identifier: Apache-2.0
"""Per-shard greedy selection on a TPU picks the token of the gathered argmax.

Run on a TPU: `PYTHONPATH=src python -m pytest -v -s <this file>`.

Emulates one vocab-parallel LM head at the served Qwen3.x shape (hidden 8192,
vocabulary 248320 over 4 shards) on one core. The logits path compiles the
shard projections and their concatenation (the all-gather) into one program
and takes the argmax in another, as compute_selected_logits and the sampler
do. The token-id path compiles each shard's projection with its local
(max, first index) pair into one program, as compute_selected_greedy_token_ids
does before its pair all-gather. Projected values, pairs and token ids must
match bitwise, including constructed ties within and across shards.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_torchtpu.layers.adapter.sample.vocab_parallel import (
    local_argmax_pair,
    select_argmax_pair,
)

HIDDEN = 8192
VOCAB = 248320
TP = 4
SHARD = VOCAB // TP


@pytest.fixture(scope="module")
def tpu():
    torch_tpu = pytest.importorskip("torch_tpu")
    del torch_tpu
    try:
        torch.zeros(1, device="tpu").cpu()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no TPU: {e}")
    return torch.device("tpu")


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _gathered_logits(h, w0, w1, w2, w3):
    return torch.cat([F.linear(h, w) for w in (w0, w1, w2, w3)], dim=-1)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _argmax(logits):
    return logits.argmax(dim=-1)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _shard_pair(h, w, start, end):
    return local_argmax_pair(F.linear(h, w), start, end)


@pytest.fixture(scope="module")
def head(tpu):
    torch.manual_seed(0)
    shards = [
        (torch.randn(SHARD, HIDDEN, device=tpu) * 0.02).to(torch.bfloat16)
        for _ in range(TP)
    ]
    # Exact ties: one row repeated in shards 0, 1 and 3 (the first shard must
    # win), and one row repeated inside shard 2 (the first index must win).
    across = (torch.randn(HIDDEN, device=tpu) * 0.02).to(torch.bfloat16)
    for r, i in ((0, 5), (1, 7), (3, 11)):
        shards[r][i] = across
    within = (torch.randn(HIDDEN, device=tpu) * 0.02).to(torch.bfloat16)
    for i in (100, 200):
        shards[2][i] = within
    return shards, across, within


@pytest.mark.parametrize("rows", [12, 48, 256])
def test_shard_pairs_pick_the_gathered_argmax(tpu, head, rows):
    shards, across, within = head
    torch.manual_seed(rows)
    h = torch.randn(rows, HIDDEN, device=tpu).to(torch.bfloat16)
    # Rows 0 and 1 align with the repeated rows so the ties are the maxima.
    h[0] = (across.float() * 400).to(torch.bfloat16)
    h[1] = (within.float() * 400).to(torch.bfloat16)

    logits = _gathered_logits(h, *shards)
    want = _argmax(logits).cpu()
    host = logits.float().cpu()
    pairs = []
    for r, w in enumerate(shards):
        start = torch.tensor(r * SHARD, dtype=torch.int32, device=tpu)
        end = torch.tensor((r + 1) * SHARD, dtype=torch.int32, device=tpu)
        pair = _shard_pair(h, w, start, end)
        local = host[:, r * SHARD : (r + 1) * SHARD]
        got = pair.cpu()
        # Same projected values (no extra precision from fusing the max).
        assert torch.equal(got[:, 0], local.max(dim=-1).values)
        assert torch.equal(got[:, 1].long(), local.argmax(dim=-1) + r * SHARD)
        pairs.append(pair)
    ids = select_argmax_pair(torch.stack(pairs, dim=1)).cpu()
    assert ids.dtype == torch.int64
    assert torch.equal(ids, want)
    assert ids[0].item() == 5 and ids[1].item() == 2 * SHARD + 100

    top2 = host.topk(2, dim=-1).values
    ties = int((top2[:, 0] == top2[:, 1]).sum())
    print(f"rows {rows}: {ties} rows with a tied maximum")
    assert ties >= 2
