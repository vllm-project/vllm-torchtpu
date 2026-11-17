# Copyright 2026 Google LLC
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
"""TPU_SAMPLER_TOPP_CANDIDATES: the candidate-tier top-p threshold keeps the
same tokens as `topp_mask`'s full-vocabulary bisection, covered or not."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.sample import top_k_top_p as tk
from vllm_torchtpu.layers.adapter.sample.topp_candidates import (
    topp_threshold,
    topp_threshold_traced,
)

VOCAB = 160 * 128


def _logits(kind: str, rows: int, rng) -> np.ndarray:
    if kind == "flat":
        return rng.standard_normal((rows, VOCAB)).astype(np.float32)
    if kind == "peaked":
        return (rng.gumbel(size=(rows, VOCAB)) * 6).astype(np.float32)
    if kind == "dominant":
        x = rng.standard_normal((rows, VOCAB)).astype(np.float32)
        x[:, :40] += 12
        return x
    if kind == "ties":
        return rng.integers(-3, 4, size=(rows, VOCAB)).astype(np.float32)
    if kind == "zipf":
        base = -1.6 * np.log(np.arange(1, VOCAB + 1))
        return np.stack([rng.permutation(base) for _ in range(rows)]).astype(np.float32)
    raise AssertionError(kind)


def _check(kind, rows, tiers, seed):
    rng = np.random.default_rng(seed)
    logits = torch.from_numpy(_logits(kind, rows, rng))
    p = rng.uniform(0.3, 0.99, size=(rows,)).astype(np.float32)
    p[0] = 0.95
    if rows > 2:
        p[1] = 1.0  # top-p off: unspecified threshold, not compared
        p[2] = 0.0
    ref = tk.topp_mask(logits, torch.from_numpy(p)[:, None]) > tk.MASKED_LOGIT_VALUE
    probs = torch.softmax(logits, dim=-1)
    thr, covered = topp_threshold_traced(jnp.asarray(probs.numpy()), jnp.asarray(p), tiers)
    got = probs >= torch.from_numpy(np.asarray(thr))[:, None]
    act = torch.from_numpy(p < 1.0)
    assert torch.equal(got[act], ref[act])
    return bool(covered)


@pytest.mark.parametrize("kind", ["flat", "peaked", "dominant", "ties", "zipf"])
@pytest.mark.parametrize("rows", [1, 8, 13])
@pytest.mark.parametrize("tiers", [(8,), (4, 32), (160,)])
def test_candidates_keep_topp_mask_tokens(kind, rows, tiers):
    _check(kind, rows, tiers, rows * 31 + len(kind) + sum(tiers))


def test_both_paths_are_exercised():
    assert _check("peaked", 8, (8,), 1)
    assert not _check("flat", 8, (8,), 2)


def test_tiers_parse_and_cpu_stays_on_bisection(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "TPU_SAMPLER_TOPP_CANDIDATES", "64,512", raising=False)
    assert tk.topp_candidate_tiers() == (64, 512)
    monkeypatch.setattr(envs, "TPU_SAMPLER_TOPP_CANDIDATES", "512,64", raising=False)
    with pytest.raises(ValueError):
        tk.topp_candidate_tiers()
    monkeypatch.setattr(envs, "TPU_SAMPLER_TOPP_CANDIDATES", "", raising=False)
    assert not tk.arm_topp_kernel()
    monkeypatch.setattr(tk, "_topp_threshold_op", object())
    assert not tk._topp_kernel_applies(torch.zeros(8, VOCAB))
