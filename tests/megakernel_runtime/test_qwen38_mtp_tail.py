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
"""The MTP draft tail past the largest bucket runs in bucket-row chunks and
matches the L=1 segment call built at the full rows count.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest --noconftest <this file>
"""

from __future__ import annotations

import os
import sys
import types

try:
    import vllm.envs  # noqa: F401
except ImportError:
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _vllm.envs = _envs
    sys.modules.setdefault("vllm", _vllm)
    sys.modules.setdefault("vllm.envs", _envs)

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel import kernel  # noqa: E402
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles  # noqa: E402
from vllm_torchtpu.megakernel.models import qwen38  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen38_inputs import (  # noqa: E402
    pack_segment,
    random_segment_inputs,
)
from vllm_torchtpu.megakernel_runtime.exported_bridge import (  # noqa: E402
    make_typed_step_wrapper,
)
from vllm_torchtpu.megakernel_runtime.qwen38_mtp_tail import (
    build_tail_call,  # noqa: E402
)
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import (  # noqa: E402
    T_VERIFY,
    build_segment_call,
)

R = 2
CHUNK = 4
G = qwen38.TINY
TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)


def _run(fn, in_specs, out_specs, args):
    typed = make_typed_step_wrapper("mtp_tail_test", fn, len(in_specs), len(out_specs))
    (res,) = jax.jit(typed)(*args)
    return np.asarray(res, np.float32)


@pytest.mark.parametrize("B", [12, 14])
def test_chunked_tail_matches_full_call(B):
    T = T_VERIFY if B % T_VERIFY == 0 else 1
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=TILES)
    spec = qwen38.segment_spec(G, R, L=1)
    inp = random_segment_inputs(jax.random.PRNGKey(B), G, ex, 1, gain=0.25)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    w_mk, _, _ = pack_segment(spec, G, ex, inp)
    flat = []
    for name in kernel.build(spec, ex)[1]["plan"].w_names:
        v = w_mk[name]
        flat.extend(v if isinstance(v, tuple) else (v,))
    flat = [jax.device_put(kernel.fold(x), tp) for x in flat]
    hidden = jax.device_put(jnp.tile(inp.x, (R, 1)), tp)
    attn = jax.device_put(inp.attn.reshape(R * B, -1), tp)
    args = (hidden, attn, *flat)

    full = build_segment_call(mesh, G, 1, B, None, interpret=True, tiles=TILES)
    chunked = build_tail_call(mesh, G, B, None, CHUNK, interpret=True, tiles=TILES)
    assert chunked[1:] == full[1:]  # same specs, so the torch op signature is unchanged
    want = _run(*full[:3], args)
    got = _run(*chunked[:3], args)
    assert got.shape == want.shape == (R * B, hidden.shape[-1])
    scale = float(np.abs(want).mean())
    assert float(np.abs(got - want).max()) < 1e-2 * scale


def test_small_counts_build_at_their_own_rows():
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    fn, in_specs, out_specs, donate = build_tail_call(
        mesh, G, CHUNK, None, CHUNK, interpret=True, tiles=TILES
    )
    assert donate == () and len(out_specs) == 1 and len(in_specs) > 2


def _greedy_tail_case(fp8, B, monkeypatch):
    """`(tail options, ex, plain-tail args, greedy-tail args, hc params, reference
    lm_head)` of one random greedy-tail case at `B` rows."""
    from vllm_torchtpu.megakernel.models import qwen38_inputs
    from vllm_torchtpu.megakernel.models.qwen38_inputs import (
        _hc_params,
        hc_weight_arrays,
    )
    from vllm_torchtpu.megakernel.ops import BF16, F32, FP8
    from vllm_torchtpu.megakernel.wholestep.tail import (
        LM_SCALE,
        vocab_chunk_width,
        vocab_chunks,
    )

    V = 384
    T = T_VERIFY if B % T_VERIFY == 0 else 1
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=TILES)
    tail = dict(vocab_shard=V, fp8=fp8)
    spec = qwen38.segment_spec(G, R, L=1)
    gspec = qwen38.segment_spec(G, R, L=1, tail=tail)
    inp = random_segment_inputs(jax.random.PRNGKey(3), G, ex, 1, gain=0.25)
    real_pack = qwen38_inputs.pack_weights
    raw = {}

    def grab(spec_, ex_, weights, smalls, final_norm=None):
        raw.update(weights=dict(weights), smalls=dict(smalls))
        return real_pack(spec_, ex_, weights, smalls, final_norm)

    monkeypatch.setattr(qwen38_inputs, "pack_weights", grab)
    w_mk, _, _ = pack_segment(spec, G, ex, inp)

    k = jax.random.split(jax.random.PRNGKey(7), 2)
    hc = _hc_params(k[0], G, 1)
    lm = (jax.random.normal(k[1], (R, G.H, V), F32) * 0.05).astype(BF16)
    weights = {**raw["weights"], **hc_weight_arrays("tail", hc, G, R, mix=True)}
    smalls = {
        **raw["smalls"],
        "tail_hcn": jnp.broadcast_to(hc["norm"][None], (R, 1, hc["norm"].shape[-1])),
    }
    ref_lm = lm
    if fp8:
        amax = jnp.abs(lm.astype(F32)).max(axis=-2, keepdims=True)
        scale = jnp.where(amax > 0, amax / 448.0, 1.0)
        lm = (lm.astype(F32) / scale).astype(FP8)
        ref_lm = lm.astype(F32) * scale
        cols = sum(width for _, _, width in vocab_chunks(V, vocab_chunk_width(ex)))
        smalls[LM_SCALE] = jnp.pad(
            scale[:, 0], ((0, 0), (0, cols - V)), constant_values=1.0
        )[:, None]
    for name, c0, width in vocab_chunks(V, vocab_chunk_width(ex)):
        blk = lm[:, :, c0 : c0 + width]
        pad = ((0, 0), (0, 0), (0, width - blk.shape[-1]))
        weights[name] = jnp.pad(blk, pad)[:, None]
    gw_mk = real_pack(gspec, ex, weights, smalls)

    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))

    def flat(s, w):
        out = []
        for name in kernel.build(s, ex)[1]["plan"].w_names:
            v = w[name]
            out.extend(v if isinstance(v, tuple) else (v,))
        return [jax.device_put(kernel.fold(x), tp) for x in out]

    hidden = jax.device_put(jnp.tile(inp.x, (R, 1)), tp)
    attn = jax.device_put(inp.attn.reshape(R * B, -1), tp)
    return (
        tail,
        mesh,
        (hidden, attn, *flat(spec, w_mk)),
        (hidden, attn, *flat(gspec, gw_mk)),
        hc,
        ref_lm,
    )


def _assert_reference_picks(tokens, res, B, hc, ref_lm):
    """`tokens` are the stock greedy pick on `res`: exact where the reference's
    top-2 gap exceeds one BF16 ulp of its maximum, within that ulp everywhere."""
    from vllm_torchtpu.megakernel.ops import BF16
    from vllm_torchtpu.megakernel.wholestep.tail import tail_reference

    x = jnp.asarray(res[:B]).astype(BF16)  # every rank holds the residual
    want, logits = tail_reference(
        x, {n: v[0] for n, v in hc.items()}, ref_lm, nhc=G.NHC, eps=G.eps
    )
    got = np.asarray(tokens)
    assert got.shape == (B,) and got.dtype == np.int32
    ref = np.asarray(logits)
    top2 = np.sort(ref, axis=-1)[:, -2:]
    ulp = 2.0 ** (np.floor(np.log2(np.abs(top2[:, 1]))) - 7)
    clear = (top2[:, 1] - top2[:, 0]) > ulp
    assert clear.sum() >= B // 2
    np.testing.assert_array_equal(got[clear], np.asarray(want)[clear])
    assert (ref[np.arange(B), got] >= top2[:, 1] - ulp).all()


def _run_greedy(fn, in_specs, args):
    typed = make_typed_step_wrapper("mtp_greedy_test", fn, len(in_specs), 2)
    res, tokens = jax.jit(typed)(*args)
    return np.asarray(res, np.float32), tokens


@pytest.mark.parametrize("fp8", [False, True])
def test_greedy_tail_keeps_residual_and_picks_reference_tokens(fp8, monkeypatch):
    """`tail=...` (VLLM_TPU_MEGAKERNEL=...,mtpgreedy=1): the residual is the plain
    tail's, and the tokens are the stock greedy pick (final mixer, lm_head, argmax)
    on that residual; `fp8` over the same dequantized head."""
    B = 8
    tail, mesh, plain_args, greedy_args, hc, ref_lm = _greedy_tail_case(
        fp8, B, monkeypatch
    )
    plain = build_segment_call(mesh, G, 1, B, None, interpret=True, tiles=TILES)
    want_res = _run(*plain[:3], plain_args)
    fn, in_specs, out_specs, donate = build_segment_call(
        mesh, G, 1, B, None, interpret=True, tiles=TILES, tail=tail
    )
    assert donate == () and len(out_specs) == 2
    res, tokens = _run_greedy(fn, in_specs, greedy_args)
    np.testing.assert_array_equal(res, want_res)
    _assert_reference_picks(tokens, want_res, B, hc, ref_lm)


def test_chunked_greedy_tail_matches_its_residual(monkeypatch):
    """Past the largest bucket the greedy tail runs in chunks like the plain tail:
    residuals as the full call's, and each chunk's tokens in row order."""
    B = 12
    tail, mesh, _, greedy_args, hc, ref_lm = _greedy_tail_case(False, B, monkeypatch)
    full = build_segment_call(
        mesh, G, 1, B, None, interpret=True, tiles=TILES, tail=tail
    )
    chunked = build_tail_call(
        mesh, G, B, None, CHUNK, interpret=True, tiles=TILES, tail=tail
    )
    assert chunked[1:] == full[1:]
    want_res, _ = _run_greedy(full[0], full[1], greedy_args)
    res, tokens = _run_greedy(chunked[0], chunked[1], greedy_args)
    assert float(np.abs(res - want_res).max()) < 1e-2 * float(np.abs(want_res).mean())
    _assert_reference_picks(tokens, res, B, hc, ref_lm)
