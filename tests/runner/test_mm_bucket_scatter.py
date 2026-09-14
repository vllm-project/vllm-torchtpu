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
"""The bucketed mm scatter must place exactly what masked-scatter would.

``_build_mm_scatter_plan`` + ``_mm_chunk_scatter_ops`` + ``_scatter_mm_embeds``
replace vLLM's exact-length slice + concat + masked-scatter so that the
device-side shapes are bucketed instead of image-size-dependent. The shapes are
the point of the change, but an off-by-one in the index arithmetic would feed
the model shifted embeddings and still run happily, so these compare the
placement row by row against a direct reference.

Kimi K3 wraps its pads in ``<|media_begin|>...<|media_end|>`` and marks only the
pad positions in ``PlaceholderRange.is_embed``, so the masked cases are the ones
that run in production; the unmasked case keeps the generic path honest.
"""

import random
import types

import pytest
import torch
from vllm.multimodal.inputs import PlaceholderRange

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

BUCKET_MIN = 64
HIDDEN = 8
TRIALS = 100


def bucket(num_tokens):
    """Stand-in for the model's mm_embed_bucket hook."""
    size = BUCKET_MIN
    while size < num_tokens:
        size *= 2
    return size


def make_mask(length, n_embed, rng, contiguous):
    """A K3-style solid pad run, or a scattered mask as a stress case."""
    mask = torch.zeros(length, dtype=torch.bool)
    if contiguous:
        lo = rng.randint(0, length - n_embed)
        mask[lo:lo + n_embed] = True
    else:
        for i in rng.sample(range(length), n_embed):
            mask[i] = True
    return mask


def build_request(rng, req_id, masked, contiguous):
    """One request: prompt tokens with images at K3-style placeholder ranges."""
    features = []
    encoder_cache = {}
    cursor = rng.randint(0, 8)
    for k in range(rng.randint(1, 3)):
        n_embed = rng.randint(1, 300)
        # K3's range also covers the begin/content/end wrapper tokens.
        length = n_embed + (rng.randint(1, 6) if masked else 0)
        offset = cursor + rng.randint(0, 10)
        is_embed = (make_mask(length, n_embed, rng, contiguous)
                    if masked else None)
        pos = PlaceholderRange(offset=offset, length=length, is_embed=is_embed)
        mm_hash = f"{req_id}:{k}"
        rows = torch.randn(n_embed, HIDDEN)
        # postprocess_encoder_output pads each cached item up to its bucket.
        pad = bucket(n_embed) - n_embed
        if pad:
            rows = torch.cat([rows, rows[-1:].expand(pad, HIDDEN)])
        encoder_cache[mm_hash] = rows
        features.append(
            types.SimpleNamespace(mm_position=pos,
                                  identifier=mm_hash,
                                  modality="image"))
        cursor = offset + length
    return features, encoder_cache, cursor + rng.randint(1, 10)


def reference_rows(features, encoder_cache, num_computed, num_scheduled,
                   req_start_idx, writes):
    """What _gather_mm_embeddings + masked-scatter would place, row by row."""
    for feat in features:
        pos = feat.mm_position
        start_pos, length = pos.offset, pos.length
        if start_pos >= num_computed + num_scheduled:
            continue
        if start_pos + length <= num_computed:
            continue
        start_idx = max(num_computed - start_pos, 0)
        end_idx = min(num_computed - start_pos + num_scheduled, length)
        if start_idx >= end_idx:
            continue
        rows = encoder_cache[feat.identifier]
        req_start_pos = req_start_idx + start_pos - num_computed
        if pos.is_embed is None:
            for j in range(start_idx, end_idx):
                writes[req_start_pos + j] = rows[j]
        else:
            src, _ = pos.get_embeds_indices_in_range(start_idx, end_idx)
            for j in range(start_idx, end_idx):
                if pos.is_embed[j]:
                    writes[req_start_pos + j] = rows[src]
                    src += 1


def run_case(rng, masked, contiguous):
    """Drive the real runner methods over one randomized step."""
    runner = object.__new__(TPUModelRunner)
    runner.model = types.SimpleNamespace(mm_embed_bucket=staticmethod(bucket))
    runner.encoder_cache = {}
    requests = {}
    req_ids = []
    num_scheduled = {}
    writes = {}  # destination row -> embedding row that belongs there
    total = 0

    for r in range(rng.randint(1, 3)):
        req_id = f"r{r}"
        features, cache, prompt_len = build_request(rng, req_id, masked,
                                                    contiguous)
        runner.encoder_cache.update(cache)
        # Either a fresh prefill or a chunk resuming mid-prompt.
        num_computed = rng.choice([0, rng.randint(0, max(0, prompt_len - 1))])
        sched = rng.randint(1, prompt_len - num_computed)
        requests[req_id] = types.SimpleNamespace(
            num_computed_tokens=num_computed, mm_features=features)
        req_ids.append(req_id)
        num_scheduled[req_id] = sched
        reference_rows(features, cache, num_computed, sched, total, writes)
        total += sched

    runner.requests = requests
    runner.input_batch = types.SimpleNamespace(req_ids=req_ids)
    scheduler_output = types.SimpleNamespace(
        num_scheduled_tokens=num_scheduled)

    plan = TPUModelRunner._build_mm_scatter_plan(runner, scheduler_output)
    assert plan is not None, "plan must not fall back for these inputs"

    num_padded = total + rng.randint(0, 32)
    base = torch.randn(num_padded, HIDDEN)
    expected = base.clone()
    for dst_row, src_row in writes.items():
        expected[dst_row] = src_row

    got = base.clone()
    # Sometimes split the step across two chunks, as the runner's chunk loop
    # does, so the per-chunk clipping is exercised too.
    chunks = [(0, num_padded)]
    if num_padded > 4 and rng.random() < 0.5:
        cut = rng.randint(1, num_padded - 1)
        chunks = [(0, cut), (cut, num_padded)]
    for tok_start, tok_end in chunks:
        ops = TPUModelRunner._mm_chunk_scatter_ops(plan, tok_start, tok_end,
                                                   tok_end - tok_start, bucket,
                                                   torch.device("cpu"))
        got[tok_start:tok_end] = TPUModelRunner._scatter_mm_embeds(
            got[tok_start:tok_end].clone(), ops)
    return got, expected


@pytest.mark.parametrize("masked,contiguous", [
    pytest.param(False, True, id="no_is_embed_mask"),
    pytest.param(True, True, id="k3_solid_pad_run"),
    pytest.param(True, False, id="scattered_mask"),
])
def test_bucketed_scatter_matches_masked_scatter(masked, contiguous):
    rng = random.Random(0)
    for trial in range(TRIALS):
        got, expected = run_case(rng, masked, contiguous)
        if not torch.equal(got, expected):
            diff = (got != expected).any(dim=1).nonzero().flatten()
            pytest.fail(f"trial {trial}: {len(diff)} rows misplaced, "
                        f"first={diff[:10].tolist()}")


def test_scatter_ops_shapes_are_bucketed():
    """Two different image sizes in the same bucket must give equal shapes."""
    device = torch.device("cpu")
    shapes = set()
    for count in (100, 120, 128):  # all land in the 128 bucket
        plan = [(torch.randn(128, HIDDEN), 0, count, 0)]
        ops = TPUModelRunner._mm_chunk_scatter_ops(plan, 0, 256, 256, bucket,
                                                   device)
        shapes.add(tuple(op[1].shape[0] for op in ops))
    assert shapes == {(128, )}
