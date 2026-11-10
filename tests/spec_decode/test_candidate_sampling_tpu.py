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
"""Candidate verify sampling on TPU at the served Qwen3.x shape
(VLLM_TPU_CANDIDATE_SAMPLING; port of PR #2821 049bb029b).

Run on a TPU host with at least TP (default 8) cores:
`PYTHONPATH=src python -m pytest -v -s <this file>`.

Vocabulary 248320 over TP shards, 16 requests, 64 verify rows. The candidate
program runs on a 4-core mesh; given the same uniform per (row, token) its
bonus tokens and per-slot decisions must equal the in-program full-vocabulary
path and the served torch sampler (``sample_from_logits`` +
``_random_rejection_sample_with_segment``) bitwise.
"""

import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.layers.core import candidate_sampling as cs

VOCAB = 248320
# Our served geometry is TP8 (8 cores on a v7x 2x2x1); CANDIDATE_TP=4 for
# a 4-core host.
TP = int(__import__("os").environ.get("CANDIDATE_TP", "8"))
SHARD = VOCAB // TP
ACTIVE = 15  # requests; one padding row in the 16-row request bucket
NUM_REQS = 16
ROWS = 64  # padded logits length
KC = 64
K = 3
NUM_DRAFTS = ACTIVE * K


def _devices():
    devices = [d for d in jax.devices() if d.platform == "tpu"]
    if len(devices) < TP:
        pytest.skip(f"needs {TP} TPU cores, found {len(devices)}")
    return devices[:TP]


def _inputs(seed, kind, temp, top_k, top_p, greedy_frac=0.0):
    """One verify chunk laid out as get_spec_decode_metadata pads it."""
    rng = np.random.default_rng(seed)
    if kind == "normal":
        logits = rng.normal(size=(ROWS, VOCAB)) * 3
    else:  # a peaked, LM-like row: a few dozen strong tokens over a flat bulk
        logits = rng.normal(size=(ROWS, VOCAB)) * 1.5
        for r in range(ROWS):
            hot = rng.choice(VOCAB, 40, replace=False)
            logits[r, hot] += rng.uniform(4, 12, size=40)
    logits = jnp.asarray(logits, jnp.float32).astype(jnp.bfloat16)
    reqs = np.arange(ACTIVE)
    bonus_idx = np.zeros(NUM_REQS, np.int32)
    bonus_idx[:ACTIVE] = reqs * (K + 1) + K
    target_idx = np.zeros(ROWS, np.int32)
    target_idx[:NUM_DRAFTS] = (reqs[:, None] * (K + 1) + np.arange(K)).reshape(-1)
    segment = np.full(ROWS, ACTIVE, np.int64)  # padding: sentinel num_reqs
    segment[:NUM_DRAFTS] = np.repeat(reqs, K)
    group = np.zeros(ROWS, np.int32)
    group[:NUM_DRAFTS] = np.tile(np.arange(K), ACTIVE)
    greedy = np.asarray(jnp.argmax(logits, axis=-1))[target_idx]
    draft = np.zeros(ROWS, np.int32)
    real = np.where(
        rng.random(NUM_DRAFTS) < 0.5,
        greedy[:NUM_DRAFTS],
        rng.integers(0, VOCAB, NUM_DRAFTS),
    )
    draft[:NUM_DRAFTS] = real
    temps = np.ones((NUM_REQS, 1), np.float32)
    temps[:ACTIVE] = temp
    temps[:ACTIVE][rng.random(ACTIVE) < greedy_frac] = 0.0
    top_p_r = np.ones((NUM_REQS, 1), np.float32)
    top_p_r[:ACTIVE] = top_p
    key = jax.random.key(seed)
    k_b, k_r, k_a = jax.random.split(key, 3)
    return dict(
        logits=logits,
        bonus_idx=jnp.asarray(bonus_idx),
        target_idx=jnp.asarray(target_idx),
        draft=jnp.asarray(draft),
        temps_b=jnp.asarray(temps).astype(jnp.bfloat16),
        top_k=top_k,
        top_p_b=jnp.asarray(top_p_r),
        segment=segment,
        group=jnp.asarray(group),
        u_b=jax.random.uniform(k_b, (NUM_REQS, VOCAB), dtype=jnp.bfloat16),
        u_r=jax.random.uniform(k_r, (ROWS, VOCAB), dtype=jnp.float32),
        accept_u=jax.random.uniform(k_a, (ROWS,), dtype=jnp.float32),
    )


def _top_k(x, pad):
    """The request bucket's top_k with `pad` in the padding row."""
    top_k = np.full((NUM_REQS, 1), pad, np.int32)
    top_k[:ACTIVE] = x["top_k"]
    return jnp.asarray(top_k)


def _run_candidates(x, *, force_full):
    """Two calls: the first finds the candidate ids, the second gives each
    candidate the uniform the full-vocabulary path draws for its token."""
    mesh = Mesh(np.asarray(_devices()), ("tp",))
    seg = jnp.asarray(x["segment"])
    top_k_b = _top_k(x, pad=1)  # as _sample_spec_verify_candidates pads
    width = KC * TP

    def shards(rows):
        return rows.reshape(rows.shape[0], TP, SHARD).transpose(1, 0, 2)

    bounds = jnp.asarray([[r * SHARD, (r + 1) * SHARD] for r in range(TP)], jnp.int32)
    local = functools.partial(
        cs.verify_local,
        axis="tp",
        kc=KC,
        vocab_size=VOCAB,
        synthetic=False,
        force_full=force_full,
        full_uniforms=(x["u_b"], x["u_r"]),
    )
    spec = P("tp")
    fn = jax.jit(
        jax.shard_map(
            lambda *a: tuple(o[None] for o in local(*(t[0] for t in a))),
            mesh=mesh,
            in_specs=(spec,) * 15,
            out_specs=(spec,) * 6,
            check_vma=False,
        )
    )

    def call(bonus_u, recover_u):
        replicated = [
            x["draft"],
            x["temps_b"],
            top_k_b,
            x["top_p_b"],
            x["temps_b"][seg],
            top_k_b[seg],
            x["top_p_b"][seg],
            bonus_u,
            x["accept_u"],
            recover_u,
            x["accept_u"],
            jnp.zeros((2,), jnp.int32),
        ]
        operands = [
            shards(x["logits"][x["bonus_idx"]]),
            shards(x["logits"][x["target_idx"]]),
            bounds,
        ] + [jnp.broadcast_to(a[None], (TP, *a.shape)) for a in replicated]
        return [np.asarray(o) for o in fn(*operands)]

    *_, ids_b, ids_t = call(
        jnp.zeros((NUM_REQS, width), jnp.bfloat16), jnp.zeros((ROWS, width))
    )
    bonus_u = jnp.take_along_axis(x["u_b"], jnp.asarray(ids_b[0]), axis=-1)
    recover_u = jnp.take_along_axis(x["u_r"], jnp.asarray(ids_t[0]), axis=-1)
    bonus, accepted, recovered, exact, _, _ = call(bonus_u, recover_u)
    for r in range(1, TP):
        assert np.array_equal(bonus[r], bonus[0])
        assert np.array_equal(accepted[r], accepted[0])
        assert np.array_equal(recovered[r], recovered[0])
    return bonus[0], accepted[0], recovered[0], int(exact[0])


CASES = [
    ("normal", 0.7, 20, 0.8, 0.0),
    ("peaked", 0.7, 20, 0.8, 0.0),
    ("peaked", 0.6, 20, 0.95, 0.0),
    ("peaked", 1.0, 20, 1.0, 0.0),
    ("peaked", 0.7, 20, 0.8, 0.5),
    ("normal", 0.7, 48, 0.95, 0.0),
]


@pytest.mark.multichip
@pytest.mark.parametrize("kind,temp,top_k,top_p,greedy_frac", CASES)
def test_candidates_match_full_path_on_tpu(kind, temp, top_k, top_p, greedy_frac):
    x = _inputs(0, kind, temp, top_k, top_p, greedy_frac)
    cand = _run_candidates(x, force_full=False)
    full = _run_candidates(x, force_full=True)
    assert cand[3] == 1 and full[3] == 0
    for a, b in zip(cand[:3], full[:3]):
        np.testing.assert_array_equal(a, b)


@pytest.mark.multichip
@pytest.mark.parametrize("kind,temp,top_k,top_p,greedy_frac", CASES)
def test_candidates_match_served_torch_sampler(kind, temp, top_k, top_p, greedy_frac):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    x = _inputs(0, kind, temp, top_k, top_p, greedy_frac)
    bonus, accepted, recovered, exact = _run_candidates(x, force_full=False)
    assert exact == 1

    def to_torch(a, dtype=None):
        a = np.asarray(jnp.asarray(a).astype(jnp.float32)) if dtype else np.asarray(a)
        t = torch.from_numpy(a.copy())
        return (t.to(dtype) if dtype else t).to("tpu")

    logits = to_torch(x["logits"], torch.bfloat16)
    temps = to_torch(x["temps_b"], torch.bfloat16)
    top_k_r = to_torch(np.asarray(_top_k(x, pad=0)))  # as the served path pads
    top_p_r = to_torch(x["top_p_b"])
    seg = to_torch(np.asarray(x["segment"], np.int64))
    bidx = to_torch(np.asarray(x["bonus_idx"]).astype(np.int64))
    runner = types.SimpleNamespace(
        _apply_temperature=functools.partial(TPUModelRunner._apply_temperature, None)
    )
    ref_bonus = TPUModelRunner.sample_from_logits(
        runner,
        logits[bidx],
        temps,
        to_torch(x["u_b"], torch.bfloat16),
        top_k_r,
        top_p_r,
        all_greedy=False,
    ).view(-1)
    num_draft = torch.zeros(NUM_REQS, dtype=torch.int32)
    num_draft[:ACTIVE] = K
    num_draft = num_draft.to("tpu")
    tidx = to_torch(np.asarray(x["target_idx"]).astype(np.int64))
    draft = to_torch(x["draft"])
    group = to_torch(x["group"])
    ref_out = rs._random_rejection_sample_impl(
        draft,
        logits[tidx],
        num_draft,
        ref_bonus,
        seg,
        group,
        K,
        temps[seg],
        top_k_r[seg],
        top_p_r[seg],
        to_torch(x["accept_u"]),
        to_torch(x["u_r"]),
        None,
        False,
    )
    out = rs.candidate_rejection_output(
        to_torch(accepted),
        to_torch(recovered),
        draft,
        num_draft,
        to_torch(bonus),
        seg,
        group,
        K,
    )
    # The padding request's row differs by design (top_k 1 vs 0).
    ref_bonus = ref_bonus.cpu().numpy()[:ACTIVE]
    ref_out, out = ref_out.cpu().numpy()[:ACTIVE], out.cpu().numpy()[:ACTIVE]
    print(
        f"{kind} T={temp} k={top_k} p={top_p}: bonus diffs "
        f"{int((ref_bonus != bonus[:ACTIVE]).sum())}/{ACTIVE}, output diffs "
        f"{int((ref_out != out).sum())}/{out.size}"
    )
    np.testing.assert_array_equal(bonus[:ACTIVE], ref_bonus)
    np.testing.assert_array_equal(out, ref_out)
