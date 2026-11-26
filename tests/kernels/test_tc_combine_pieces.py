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
"""TPU_MOE_TC_COMBINE_PIECES: the TensorCore MoE combine run on consecutive
row pieces of the tokens (one call each, no gather fallback) equals one call
over all of them bit for bit, alone and through `fused_moe_gmm.moe_gmm`
(interpret mode)."""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.megablox import tc_combine as K
from vllm_torchtpu.layers.core import fused_moe_gmm

TOPK = 4
E = 64
# The kernels themselves (fused_moe_gmm calls them through the same module,
# which _run_moe patches).
_TC, _TC_SHARED = K.tc_combine, K.tc_combine_shared


def _inputs(t, local, h, seed, ks=256):
    rng = np.random.default_rng(seed)
    ids = np.argsort(rng.random((t, E)), axis=1)[:, :TOPK]
    valid = ids < local
    keys = np.where(valid.reshape(-1), ids.reshape(-1), local)
    order = np.argsort(keys, kind="stable")
    pos = np.empty_like(order)
    pos[order] = np.arange(order.size)
    group_sizes = np.bincount(ids[valid], minlength=local)[:local].astype(np.int32)
    count = int(valid.sum())
    y = (rng.standard_normal((t * TOPK, h)) * 3).astype(np.float32)
    y[count:] = np.nan  # rows past the computed prefix are never written
    w = rng.uniform(0.05, 1, (t, TOPK)).astype(np.float32)
    act = (rng.standard_normal((t, ks)) * 2).astype(np.float32)
    ws = rng.standard_normal((ks, h)).astype(np.float32) * 0.05
    scale = 2.0 ** rng.integers(-3, 3, size=(h,)).astype(np.float32)
    return dict(
        y=jnp.asarray(y).astype(jnp.bfloat16),
        pos=jnp.asarray(pos.astype(np.int32)),
        w=jnp.asarray(w.reshape(-1)).astype(jnp.bfloat16),
        valid=jnp.asarray(valid.reshape(-1)),
        gs=jnp.asarray(group_sizes),
        act=jnp.asarray(act).astype(jnp.bfloat16),
        ws=jnp.asarray(ws).astype(jnp.float8_e4m3fn),
        scale=jnp.asarray(scale).astype(jnp.bfloat16),
    )


def _bits(x):
    return np.asarray(x).view(np.uint16)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("pieces", [2, 4])
def test_pieces_concat_equals_one_call(pieces, shared):
    t, h = 1024, 512
    a = _inputs(t, 16, h, seed=pieces)
    per = t // pieces

    def call(sl, fallback):
        routes = slice(sl.start * TOPK, sl.stop * TOPK)
        common = (a["y"], a["pos"][routes], a["w"][routes], a["valid"][routes], a["gs"])
        if not shared:
            return K.tc_combine(*common, topk=TOPK, fallback=fallback, interpret=True)
        return K.tc_combine_shared(
            *common,
            a["act"][sl],
            a["ws"],
            a["scale"],
            topk=TOPK,
            routed_scale=2.0,
            fallback=fallback,
            interpret=True,
        )

    want = call(slice(0, t), True)
    got = jnp.concatenate(
        [call(slice(i * per, (i + 1) * per), False) for i in range(pieces)]
    )
    assert np.isfinite(np.asarray(got.astype(jnp.float32))).all()
    np.testing.assert_array_equal(_bits(got), _bits(want))


def test_no_fallback_matches_fallback():
    a = _inputs(512, 8, 256, seed=3)
    args = (a["y"], a["pos"], a["w"], a["valid"], a["gs"])
    with_cond = K.tc_combine(*args, topk=TOPK, interpret=True)
    plain = K.tc_combine(*args, topk=TOPK, fallback=False, interpret=True)
    np.testing.assert_array_equal(_bits(plain), _bits(with_cond))


def test_combine_pieces_gate(monkeypatch):
    args = (6144, 4, 16, jnp.bfloat16)
    monkeypatch.setattr(envs, "TPU_MOE_TC_COMBINE_PIECES", 2, raising=False)
    assert fused_moe_gmm._combine_pieces(8192, args) == 2
    assert fused_moe_gmm._combine_pieces(4096, args) == 2
    assert fused_moe_gmm._combine_pieces(2048, args) == 1  # below 4096 tokens
    assert fused_moe_gmm._combine_pieces(4096 + 256, args) == 1  # 2176: not blocks
    monkeypatch.setattr(envs, "TPU_MOE_TC_COMBINE_PIECES", 1, raising=False)
    assert fused_moe_gmm._combine_pieces(8192, args) == 1


def test_env_default(monkeypatch):
    monkeypatch.delenv("TPU_MOE_TC_COMBINE_PIECES", raising=False)
    assert envs.environment_variables["TPU_MOE_TC_COMBINE_PIECES"]() == 1


def _run_moe(monkeypatch, pieces, shared):
    t, h, local = 1024, 256, 16
    a = _inputs(t, local, h, seed=11)
    rows = t * TOPK
    monkeypatch.setattr(envs, "TPU_MOE_TC_COMBINE", True, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_TC_COMBINE_PIECES", pieces, raising=False)
    monkeypatch.setattr(fused_moe_gmm, "_COMBINE_PIECES_MIN_TOKENS", 512)
    monkeypatch.setattr(fused_moe_gmm, "get_packing_factor", lambda *a, **k: 1)
    outputs = iter([a["y"], a["y"]])
    monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper", lambda *a, **k: next(outputs))
    calls = []

    def interpret(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kw):
            calls.append((fn.__name__, args[1].shape[0], kw.get("fallback", True)))
            return fn(*args, interpret=True, **kw)

        return wrapped

    monkeypatch.setattr(fused_moe_gmm.tc_combine_mod, "tc_combine", interpret(_TC))
    monkeypatch.setattr(
        fused_moe_gmm.tc_combine_mod,
        "tc_combine_shared",
        interpret(_TC_SHARED),
    )
    out = fused_moe_gmm.moe_gmm(
        x=jnp.ones((rows, h), jnp.bfloat16),
        w1=jnp.ones((local, h, h), jnp.bfloat16),
        w1_scale=None,
        w1_bias=None,
        w2=jnp.ones((local, h, h), jnp.bfloat16),
        w2_scale=None,
        w2_bias=None,
        group_sizes=a["gs"],
        argsort_revert_indices=a["pos"],
        token_indices_sorted=jnp.zeros((rows,), jnp.int32),
        topk_weights_flat=a["w"],
        valid_mask_flat=a["valid"],
        activation="silu",
        num_tokens=t,
        topk=TOPK,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=0,
        shared=(a["act"], a["ws"], a["scale"], 1.0) if shared else None,
    )
    return out, calls


@pytest.mark.parametrize("shared", [False, True])
def test_moe_gmm_combine_pieces_bitwise(monkeypatch, shared):
    want, calls1 = _run_moe(monkeypatch, 1, shared)
    got, calls2 = _run_moe(monkeypatch, 2, shared)
    name = "tc_combine_shared" if shared else "tc_combine"
    assert calls1 == [(name, 1024 * TOPK, True)]
    assert calls2 == [(name, 512 * TOPK, False)] * 2
    np.testing.assert_array_equal(_bits(got), _bits(want))
