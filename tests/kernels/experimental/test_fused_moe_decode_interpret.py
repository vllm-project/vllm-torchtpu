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
"""The fused MoE decode kernel in the Pallas interpreter on 8 host devices, at
the Qwen3.8-Flash EP layout's awkward shapes: a hidden size whose 8-way
all-reduce chunk is not lane aligned (1536 / 8 = 192, padded to 256 lanes), a
routed intermediate that is not a multiple of the 512-column activation block
(640 = 512 + 128) and a BF16 shared expert, gated by sigmoid(x . w_g), its
80-wide TP slice zero-padded to 128 as the adapter arms it. Checked against the
decomposed layer in f32 (routed experts + gated shared expert, psum).

Runs this file as a worker subprocess with JAX_PLATFORMS=cpu and
XLA_FLAGS=--xla_force_host_platform_device_count=8 so the 8-device checks
execute unconditionally in CPU CI regardless of the parent process's JAX state.
"""

import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
    fused_moe_decode_kernel as kmod,
)
from vllm_torchtpu.layers.adapter.fused_moe_decode import pad_shared_intermediate

N, H, INTER, INTER_S, INTER_S_PAD, E, TOPK, M = 8, 1536, 640, 80, 128, 4, 2, 16
F8 = jnp.float8_e4m3fn
FOLD_TOKENS = (M, M - 4)
FOLD_GATE_MODES = ("xla", "kernel_row", "kernel_col", "ungated")
FOLD_SHARED_FP8 = (False, True)


def _run_worker() -> dict:
    flags = " ".join(
        tok
        for tok in os.environ.get("XLA_FLAGS", "").split()
        if not tok.startswith("--xla_force_host_platform_device_count")
    )
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N}".strip(),
    )
    proc = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=540,
        check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def on_8_devices() -> dict:
    return _run_worker()


def test_qwen38_flash_shapes_match_decomposed_layer(on_8_devices):
    q38 = on_8_devices["qwen38"]
    assert q38["shape"] == [N, M, H]
    assert q38["ranks_agree"], q38
    # FP8 routed matmuls and a BF16 all-reduce against f32 math.
    assert q38["max_rel"] < 0.08, q38
    # Must-fail controls: one routed expert missing, or the shared gate dropped.
    assert q38["skip_expert_rel"] > 0.08, q38
    assert q38["ungated_rel"] > 0.08, q38


@pytest.mark.parametrize("shared_fp8", FOLD_SHARED_FP8)
@pytest.mark.parametrize("gate_mode", FOLD_GATE_MODES)
@pytest.mark.parametrize("tokens", FOLD_TOKENS)
def test_folded_prologue_matches_unfolded_call(
    on_8_devices, tokens, gate_mode, shared_fp8
):
    """fold=True (routing prologue inside the kernel, operands as stored) matches
    the unfolded call across padded/unpadded rows, xla/in-kernel ([1, H] and
    [H, 1])/ungated shared gates, and BF16 + 1-D-scaled FP8 shared experts."""
    key = f"{tokens}:{gate_mode}:{int(shared_fp8)}"
    case = on_8_devices["fold"][key]
    assert case["ranks_agree"], case
    if gate_mode in ("xla", "ungated"):
        assert case["bitwise_equal"], case
    else:
        assert case["bitwise_equal"] or case["max_ulp"] <= 1, case


def _q(w, axis):
    s = np.max(np.abs(w), axis=axis, keepdims=True) / 448.0
    s = np.where(s == 0, 1.0, s)
    return jnp.asarray(w / s).astype(F8), s.astype(np.float32)


def _inputs(seed):
    rng = np.random.default_rng(seed)
    w1, w1_s = _q(rng.normal(size=(N, E, H, 2 * INTER)) * 0.02, 2)
    w2, w2_s = _q(rng.normal(size=(N, E, INTER, H)) * 0.02, 2)
    # The shared expert's per-rank TP slice in the unquantized layout:
    # gate_up [H, 2 i_s] (gate | up), down [i_s, H].
    sw1 = jnp.asarray(rng.normal(size=(N, H, 2 * INTER_S)) * 0.02, jnp.bfloat16)
    sw2 = jnp.asarray(rng.normal(size=(N, INTER_S, H)) * 0.02, jnp.bfloat16)
    gate = jnp.asarray(rng.normal(size=(H, 1)) * 0.02, jnp.bfloat16)
    x = jnp.asarray(rng.normal(size=(M, H)), jnp.bfloat16)
    ids = np.stack([rng.choice(N * E, TOPK, replace=False) for _ in range(M)])
    tw = jnp.asarray(rng.dirichlet(np.ones(TOPK), size=M), jnp.bfloat16)
    return x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, jnp.asarray(ids, jnp.int32)


def _reference(
    x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, ids, *, skip=None, gated=True
):
    """Sum over ranks of each rank's routed experts and shared-expert slice."""
    xf = x.astype(jnp.float32)
    out = jnp.zeros((M, H), jnp.float32)
    g = jax.nn.sigmoid(jnp.dot(xf, gate.astype(jnp.float32))) if gated else 1.0
    for r in range(N):
        gu = xf @ sw1[r].astype(jnp.float32)
        out += g * ((jax.nn.silu(gu[:, :INTER_S]) * gu[:, INTER_S:]) @ sw2[r])
        for e in range(E):
            if r * E + e == skip:
                continue
            coeff = jnp.sum(
                jnp.where(ids == r * E + e, tw.astype(jnp.float32), 0.0),
                1,
                keepdims=True,
            )
            a = xf @ (w1[r, e].astype(jnp.float32) * w1_s[r, e])
            mid = jax.nn.silu(a[:, :INTER]) * a[:, INTER:]
            out += coeff * (mid @ (w2[r, e].astype(jnp.float32) * w2_s[r, e]))
    return np.asarray(out)


def _torch(a):
    return torch.from_numpy(np.asarray(a.astype(jnp.float32))).to(torch.bfloat16)


def _jax(t):
    return jnp.asarray(t.float().numpy(), jnp.bfloat16)


def _max_rel(a, ref):
    return float(
        (
            np.linalg.norm(a - ref, axis=1)
            / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
        ).max()
    )


def _worker() -> None:
    kmod._vmem_capacity = lambda: 64 << 20
    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devices) >= N, devices
    mesh = Mesh(np.asarray(devices[:N]), ("d",))
    sh, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())

    # 1. Qwen3.8-Flash awkward shapes vs f32 decomposed reference.
    x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, ids = _inputs(0)
    padded = [
        pad_shared_intermediate(_torch(sw1[r]), _torch(sw2[r]), INTER_S_PAD)
        for r in range(N)
    ]
    psw1 = jnp.stack([_jax(p[0]) for p in padded])
    psw2 = jnp.stack([_jax(p[1]) for p in padded])
    assert psw1.shape == (N, H, 2 * INTER_S_PAD) and psw1.dtype == jnp.bfloat16

    def q38_body(rank, w1, w2, w1_s, w2_s, sw1, sw2, x, gate, tw, ids):
        return kmod.fused_moe_decode(
            rank.reshape(-1),
            x,
            w1[0],
            w2[0],
            w1_s[0][:, :, None, :],
            w2_s[0][:, :, None, :],
            sw1[0],
            jnp.ones((1, 2 * INTER_S_PAD), jnp.float32),
            sw2[0],
            jnp.ones((1, H), jnp.float32),
            gate,
            tw,
            ids,
            n_ranks=N,
            expert_start=rank.reshape(()) * E,
            interpret=True,
        )[None]

    q38_args = [
        jax.device_put(np.arange(N, dtype=np.int32).reshape(N, 1), sh),
        *(jax.device_put(a, sh) for a in (w1, w2, w1_s, w2_s, psw1, psw2)),
        *(jax.device_put(a, rep) for a in (x, gate, tw, ids)),
    ]
    got = jax.jit(
        jax.shard_map(
            q38_body,
            mesh=mesh,
            in_specs=(P("d"),) * 7 + (P(),) * 4,
            out_specs=P("d"),
            check_vma=False,
        )
    )(*q38_args)
    got_f32 = np.asarray(got.astype(jnp.float32))
    ref = _reference(x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, ids)
    qwen38_result = {
        "shape": list(got_f32.shape),
        "ranks_agree": bool(
            all(np.array_equal(got_f32[0], got_f32[r]) for r in range(N))
        ),
        "max_rel": _max_rel(got_f32[0], ref),
        "skip_expert_rel": _max_rel(
            got_f32[0], _reference(*_inputs(0), skip=int(ids[0, 0]))
        ),
        "ungated_rel": _max_rel(
            got_f32[0],
            _reference(x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, ids, gated=False),
        ),
    }

    # 2. Folded prologue vs unfolded call across row padding, gate modes, and
    # BF16 + 1-D-scaled FP8 shared experts.
    x1, w1_1, w1_s1, w2_1, w2_s1, sw1_1, sw2_1, gate1, tw1, ids1 = _inputs(1)
    padded1 = [
        pad_shared_intermediate(_torch(sw1_1[r]), _torch(sw2_1[r]), INTER_S_PAD)
        for r in range(N)
    ]
    bf16_sw1 = jnp.stack([_jax(p[0]) for p in padded1])
    bf16_sw2 = jnp.stack([_jax(p[1]) for p in padded1])
    bf16_s1 = jnp.ones((N, 2 * INTER_S_PAD), jnp.float32)
    bf16_s2 = jnp.ones((N, H), jnp.float32)

    rng_fp8 = np.random.default_rng(2)
    fp8_sw1, fp8_s1_nd = _q(rng_fp8.normal(size=(N, H, 2 * INTER_S_PAD)) * 0.02, 1)
    fp8_sw2, fp8_s2_nd = _q(rng_fp8.normal(size=(N, INTER_S_PAD, H)) * 0.02, 1)
    # 1-D per-rank scales [2 * INTER_S_PAD] and [H], as stored on FP8 linears.
    fp8_s1 = jnp.asarray(fp8_s1_nd[:, 0, :])
    fp8_s2 = jnp.asarray(fp8_s2_nd[:, 0, :])

    rank = jax.device_put(
        np.stack([np.arange(N), np.arange(N) * E], 1).astype(np.int32), sh
    )
    routed_sh = [jax.device_put(a, sh) for a in (w1_1, w2_1, w1_s1, w2_s1)]
    shared_sets = {
        False: [jax.device_put(a, sh) for a in (bf16_sw1, bf16_s1, bf16_sw2, bf16_s2)],
        True: [jax.device_put(a, sh) for a in (fp8_sw1, fp8_s1, fp8_sw2, fp8_s2)],
    }
    gate_row = jax.device_put(gate1.reshape(1, H), rep)
    gate_col = jax.device_put(gate1.reshape(H, 1), rep)

    def run_variant(
        *,
        fold: bool,
        gated: bool,
        gate_in_kernel: bool,
        col_gate: bool,
        tokens: int,
        shared_fp8: bool,
    ) -> np.ndarray:
        xt = jax.device_put(x1[:tokens], rep)
        twt = jax.device_put(tw1[:tokens], rep)
        idst = jax.device_put(ids1[:tokens], rep)
        gw = gate_col if col_gate else gate_row

        def body(
            rank_b,
            w1_b,
            w2_b,
            w1_sb,
            w2_sb,
            sw1_b,
            s1_b,
            sw2_b,
            s2_b,
            x_b,
            g_b,
            tw_b,
            ids_b,
        ):
            common = (
                x_b,
                w1_b[0],
                w2_b[0],
                w1_sb[0][:, :, None, :],
                w2_sb[0][:, :, None, :],
                sw1_b[0],
            )
            if fold:
                return kmod.fused_moe_decode(
                    rank_b,
                    *common,
                    s1_b[0],
                    sw2_b[0],
                    s2_b[0],
                    g_b if gated else None,
                    tw_b,
                    ids_b,
                    n_ranks=N,
                    expert_start="rank",
                    fold=True,
                    gate_in_kernel=gate_in_kernel,
                    interpret=True,
                )[None]
            return kmod.fused_moe_decode(
                rank_b[0, :1],
                *common,
                s1_b[0].reshape(1, -1),
                sw2_b[0],
                s2_b[0].reshape(1, -1),
                g_b.reshape(-1, 1) if gated else None,
                tw_b,
                ids_b,
                n_ranks=N,
                expert_start=rank_b[0, 1],
                interpret=True,
            )[None]

        out = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P("d"),) * 9 + (P(),) * 4,
                out_specs=P("d"),
                check_vma=False,
            )
        )(rank, *routed_sh, *shared_sets[shared_fp8], xt, gw, twt, idst)
        return np.asarray(out.view(jnp.uint16))

    unfolded_cache: dict[tuple[int, bool, bool], np.ndarray] = {}
    fold_results: dict[str, dict] = {}
    for shared_fp8 in FOLD_SHARED_FP8:
        for tokens in FOLD_TOKENS:
            for gated in (True, False):
                unfolded_cache[(tokens, gated, shared_fp8)] = run_variant(
                    fold=False,
                    gated=gated,
                    gate_in_kernel=False,
                    col_gate=False,
                    tokens=tokens,
                    shared_fp8=shared_fp8,
                )
            for gate_mode in FOLD_GATE_MODES:
                gated = gate_mode != "ungated"
                got_u16 = run_variant(
                    fold=True,
                    gated=gated,
                    gate_in_kernel=gate_mode in ("kernel_row", "kernel_col", "ungated"),
                    col_gate=gate_mode == "kernel_col",
                    tokens=tokens,
                    shared_fp8=shared_fp8,
                )
                ref_u16 = unfolded_cache[(tokens, gated, shared_fp8)]
                ulp = int(
                    np.max(np.abs(got_u16.astype(np.int32) - ref_u16.astype(np.int32)))
                )
                fold_results[f"{tokens}:{gate_mode}:{int(shared_fp8)}"] = {
                    "ranks_agree": bool(
                        all(np.array_equal(got_u16[0], got_u16[r]) for r in range(N))
                    ),
                    "bitwise_equal": bool(np.array_equal(got_u16, ref_u16)),
                    "max_ulp": ulp,
                }

    print("RESULT " + json.dumps({"qwen38": qwen38_result, "fold": fold_results}))


if __name__ == "__main__":
    _worker()
