# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL MOE: FUSED_MOE_DECODE_EXACT_WIDTH (ported from Benliang Wang <benliangw@google.com>, commit b0c1453ba on
benliangw/q38fh-integ-pin-5b44702aea; exact width only) is bitwise the padded split on v8's served MoE path.

Harness (CPU, Pallas interpreter, 8 host devices): the REAL adapter. ``layers/adapter/fused_moe_decode._build_op`` builds
the op's JAX function exactly as arming does (torch_tpu's jax_op is replaced by a recorder that keeps the function: the real
wrapper needs a TPU to execute); that function's shard_map over the 8-device EP mesh runs the real ``_layer`` (v8's served
MoE keys: kernel up to 256 tokens, FOLD=xla, COMPACT_PIPE=160 with 32 rows, COMPACT_ROWS=32), which calls the real kernel
``fused_moe_decode`` in the interpreter (the in-kernel all-to-all + all-gather included). Served per-rank shapes:
64 local experts of 512, hidden 2560 (20 lane tiles: 3, 3, 3, 3, 2, 2, 2, 2 per rank with exact width; padded to 3072
without), routed intermediate 640, a gated BF16 shared expert whose 80-wide TP slice is zero-padded to 128, top-10,
EP over 8 ranks. Routing: Benliang's recorded AgentX routing (``scripts/data/agentx_routing.npz`` on his branch, 4 rows a
verify window, distinct replies; read-only copy passed in Q38_PORTBL_ROUTING_NPZ) when present, else a skewed synthetic one
(each BW line prints which).

  BW  at 80 / 96 / 112 / 128 (dense expert loop) and 160 / 192 / 224 / 256 rows (D60 compact pipe): the layer output with
      the flag on equals the flag-off output bit for bit on every rank (the MoE layer writes no state), ranks agree, the
      output is nonzero.
  ENG the flag-on trace records ``q38 PORT-BL MOE exact width traced for <rows> tokens`` and its kernel writes [m, 2560]
      under a ``-xw`` name; the flag-off trace has neither and writes [m, 3072]. Also the expert-loop mode per bucket.
  W   arming: the flag reaches the target model's op only (the draft's op is today's); off builds today's key / closure;
      on extends key, name and closure; the env key is registered and hashed.
  J   the flag-on op builds through torch_tpu's REAL jax_op whenever the flag-off op does.

Needs JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 (skipped otherwise).
"""

from __future__ import annotations

import functools
import os
import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("torch_tpu")
jnp = jax.numpy
import torch  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME  # noqa: E402
from vllm_torchtpu.kernels.experimental.fused_moe_decode import (  # noqa: E402
    fused_moe_decode_kernel as kmod,
)
from vllm_torchtpu.layers.adapter import fused_moe_decode as bridge  # noqa: E402

N, E, H, INTER, INTER_S, ISP, TOPK = 8, 64, 2560, 640, 80, 128, 10
F8 = jnp.float8_e4m3fn
FLAG = "FUSED_MOE_DECODE_EXACT_WIDTH"
# v8's served MoE keys (scratch/decL1/s11/v8nt/cd/record_envs/v8nt_record_20261116T021053Z.env).
V8_MOE_ENV = {
    "FUSED_MOE_DECODE_KERNEL": "1",
    "FUSED_MOE_DECODE_KERNEL_MAX_TOKENS": "256",
    "FUSED_MOE_DECODE_KERNEL_FOLD": "xla",
    "FUSED_MOE_DECODE_COMPACT_PIPE": "160",
    "FUSED_MOE_DECODE_COMPACT_PIPE_ROWS": "32",
    "FUSED_MOE_DECODE_COMPACT_ROWS": "32",
}
BW_ROWS = (80, 96, 112, 128, 160, 192, 224, 256)
ENG_ROWS = (16, 64, 80, 96, 112, 128, 160, 192, 224, 256)
_KOUT = re.compile(r":bf16\[(\d+),(\d+)\] = pallas_call\[")
_KNAME = re.compile(r"fused_moe_decode[\w\-]*-n_8-buffers_\d+[\w\-]*")


@pytest.fixture(scope="module")
def mesh():
    if jax.default_backend() != "cpu" or jax.device_count() < N:
        pytest.skip(f"needs {N} host devices (see module docstring)")
    return Mesh(np.asarray(jax.devices()[:N]), (EP_AXIS_NAME,))


class _Op:
    """Stands in for torch_tpu's jax_op: keeps the op's name and JAX function."""

    def __init__(self, name, fn, **kwargs):
        self.name, self.fn, self.kwargs = name, fn, kwargs

    def register_fake(self, fake):
        self.fake = fake


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args if args else msg)

    debug = warning = info


def _set_env(monkeypatch, exact: bool) -> None:
    for name in list(os.environ):
        if name.startswith("FUSED_MOE_DECODE_"):
            monkeypatch.delenv(name)
    for name, value in V8_MOE_ENV.items():
        monkeypatch.setenv(name, value)
    if exact:
        monkeypatch.setenv(FLAG, "1")


def _build(monkeypatch, mesh, exact: bool, interpret: bool = True):
    """The op arming builds for the target model (arm_runner_models passes the flag as exact_width=True)."""
    _set_env(monkeypatch, exact)
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    body = functools.partial(kmod.fused_moe_decode, interpret=True) if interpret else kmod.fused_moe_decode
    monkeypatch.setattr(bridge, "_kernel_body", body)
    monkeypatch.setattr(bridge, "jax_op", _Op)
    monkeypatch.setattr(bridge, "_OPS", {})
    log = _Log()
    monkeypatch.setattr(bridge, "logger", log)
    kw = {"exact_width": True} if bridge.envs.FUSED_MOE_DECODE_EXACT_WIDTH else {}
    op = bridge._build_op(
        mesh, N, TOPK, True, bridge.envs.FUSED_MOE_DECODE_KERNEL_MAX_TOKENS, True, 1.0, **kw
    )
    return op, log


@functools.lru_cache(maxsize=1)
def _weights():
    """Per-rank operands as the armed layer stores them, laid out as the op's global (rank-major) arrays."""
    rng = np.random.default_rng(0)
    w1, w1s, w2, w2s, sw1, sw2 = [], [], [], [], [], []
    for _ in range(N):
        a = rng.standard_normal((E, H, 2 * INTER), dtype=np.float32) * np.float32(0.02)
        s = np.max(np.abs(a), axis=1, keepdims=True) / np.float32(448.0)
        w1.append((a / s).astype(F8))
        w1s.append(s[:, :, None, :].astype(np.float32))
        del a
        b = rng.standard_normal((E, INTER, H), dtype=np.float32) * np.float32(0.02)
        t = np.max(np.abs(b), axis=1, keepdims=True) / np.float32(448.0)
        w2.append((b / t).astype(F8))
        w2s.append(t[:, :, None, :].astype(np.float32))
        del b
        # The shared expert's TP slice, gate | up, zero-padded to 128 as arming pads it (pad_shared_intermediate).
        g = rng.standard_normal((H, 2 * INTER_S), dtype=np.float32) * np.float32(0.02)
        z = np.zeros((H, ISP - INTER_S), np.float32)
        sw1.append(np.concatenate([g[:, :INTER_S], z, g[:, INTER_S:], z], axis=1).astype(jnp.bfloat16))
        d = rng.standard_normal((INTER_S, H), dtype=np.float32) * np.float32(0.02)
        sw2.append(np.concatenate([d, np.zeros((ISP - INTER_S, H), np.float32)], axis=0).astype(jnp.bfloat16))
    gate = (rng.standard_normal((H, 1), dtype=np.float32) * np.float32(0.02)).astype(jnp.bfloat16)
    return dict(
        w1=np.concatenate(w1),
        w2=np.concatenate(w2),
        w1s=np.concatenate(w1s),
        w2s=np.concatenate(w2s),
        sw1=np.concatenate(sw1),
        s1=np.ones((N * 2 * ISP,), np.float32),
        sw2=np.concatenate(sw2),
        s2=np.ones((N * H,), np.float32),
        gate=gate,
    )


@functools.lru_cache(maxsize=1)
def _device_weights(mesh_key):
    mesh = _MESH[mesh_key]
    w = _weights()
    sh, rep = NamedSharding(mesh, P(EP_AXIS_NAME)), NamedSharding(mesh, P())
    names = ("w1", "w2", "w1s", "w2s", "sw1", "s1", "sw2", "s2")
    out = {k: jax.device_put(w[k], sh) for k in names}
    out["gate"] = jax.device_put(w["gate"], rep)
    return out


_MESH: dict = {}


def _routing(rows: int, seed: int):
    """tw [rows, 10] f32, ti [rows, 10] int32 global expert ids, and the routing's source."""
    rng = np.random.default_rng(seed)
    tw = np.exp(rng.normal(size=(rows, TOPK)))
    tw = (tw / tw.sum(-1, keepdims=True)).astype(np.float32)
    path = os.environ.get("Q38_PORTBL_ROUTING_NPZ", "")
    if path and os.path.exists(path):
        z = np.load(path)
        experts, groups = z["experts"], z["groups"]  # [windows, 4, layers, 10], reply id per window
        replies = np.unique(groups)
        pick = rng.choice(len(replies), rows // 4, replace=False)
        wins = [int(rng.choice(np.nonzero(groups == g)[0])) for g in replies[pick]]
        layer = seed % experts.shape[2]
        ti = experts[wins, :, layer, :].reshape(rows, TOPK).astype(np.int32)
        return tw, ti, f"agentx(layer {layer})"
    p = np.exp(-3.0 * np.arange(N * E) / (N * E))
    p /= p.sum()
    ti = np.stack([rng.choice(N * E, TOPK, replace=False, p=p) for _ in range(rows)]).astype(np.int32)
    return tw, ti, "synthetic-skewed"


def _args(mesh, rows: int):
    _MESH[id(mesh)] = mesh
    dw = _device_weights(id(mesh))
    rng = np.random.default_rng(1000 + rows)
    x = rng.standard_normal((rows, H), dtype=np.float32).astype(jnp.bfloat16)
    tw, ti, source = _routing(rows, 2000 + rows)
    sh = NamedSharding(mesh, P(EP_AXIS_NAME))
    rank = np.stack([np.arange(N), np.arange(N) * E], 1).astype(np.int32)
    # The op's operands are per-rank tensors concatenated on axis 0 (replicated hidden states and routing).
    args = [
        jax.device_put(np.tile(x, (N, 1)), sh),
        dw["w1"],
        dw["w2"],
        dw["w1s"],
        dw["w2s"],
        dw["sw1"],
        dw["s1"],
        dw["sw2"],
        dw["s2"],
        dw["gate"],
        jax.device_put(np.tile(tw, (N, 1)), sh),
        jax.device_put(np.tile(ti, (N, 1)), sh),
        jax.device_put(rank, sh),
    ]
    return args, source


def _kernel_outs(text: str):
    return sorted({(int(a), int(b)) for a, b in _KOUT.findall(text)}), sorted(set(_KNAME.findall(text)))


def _eng(log: _Log, rows: int) -> int:
    return sum(f"q38 PORT-BL MOE exact width traced for {rows} tokens" in s for s in log.lines)


def _mode(log: _Log, rows: int) -> str:
    lines = [s for s in log.lines if s.startswith(f"Fused MoE decode kernel traced for {rows} tokens")]
    assert lines, f"no kernel trace record for {rows} tokens: the kernel did not serve this bucket"
    return "pipe" if "pipelined" in lines[-1] else ("dense" if "dense experts" in lines[-1] else lines[-1])


@pytest.mark.parametrize("rows", BW_ROWS, ids=[f"m{r}" for r in BW_ROWS])
def test_bw_exact_width_on_is_bitwise_off(mesh, monkeypatch, rows):
    args, source = _args(mesh, rows)
    outs, info = {}, {}
    for exact in (False, True):
        op, log = _build(monkeypatch, mesh, exact)
        shapes, names = _kernel_outs(str(jax.make_jaxpr(op.fn)(*args)))
        got = np.asarray(jax.jit(op.fn)(*args))
        outs[exact] = got.view(np.uint16).reshape(N, rows, H)
        info[exact] = (shapes, names, _eng(log, rows), _mode(log, rows))
    off, on = outs[False], outs[True]
    agree = [all(np.array_equal(o[0], o[r]) for r in range(N)) for o in (off, on)]
    bad = int((on != off).sum())
    nonzero = float((off[0] != 0).mean())
    want_mode = "pipe" if rows >= 160 else "dense"
    print(
        f"PBL_MOE_BW m{rows} mode {info[True][3]} routing {source} mismatches {bad} of {off.size} ranks_agree "
        f"off {agree[0]} on {agree[1]} nonzero {nonzero:.4f} kernel_off {info[False][0]} {info[False][1]} "
        f"kernel_on {info[True][0]} {info[True][1]} eng_on {info[True][2]} eng_off {info[False][2]}"
    )
    # ENG: the flag-on program is the exact-width one, the flag-off program today's.
    assert info[False][0] == [(rows, 3072)] and info[True][0] == [(rows, H)], (info[False][0], info[True][0])
    assert not any(n.endswith("-xw") for n in info[False][1]) and all(n.endswith("-xw") for n in info[True][1])
    assert info[True][2] >= 1 and info[False][2] == 0
    assert info[False][3] == info[True][3] == want_mode
    # BW
    assert all(agree), "ranks disagree"
    assert nonzero > 0.9, "output is (mostly) zero"
    assert bad == 0, f"{bad} of {off.size} output elements differ (exact width vs padded split)"


@pytest.mark.parametrize("rows", ENG_ROWS, ids=[f"m{r}" for r in ENG_ROWS])
def test_eng_trace_records_and_kernel_width(mesh, monkeypatch, rows):
    """Trace only: per bucket, the engagement record and the kernel's output width / name with the flag on and off."""
    args, _ = _args(mesh, rows)
    got = {}
    for exact in (False, True):
        op, log = _build(monkeypatch, mesh, exact, interpret=False)
        shapes, names = _kernel_outs(str(jax.make_jaxpr(op.fn)(*args)))
        got[exact] = (shapes, names, _eng(log, rows))
    print(f"PBL_MOE_ENG m{rows} off {got[False]} on {got[True]}")
    assert got[False] == ([(rows, 3072)], got[False][1], 0) and got[False][1]
    assert not any(n.endswith("-xw") for n in got[False][1])
    assert got[True][0] == [(rows, H)] and got[True][2] == 1
    assert got[True][1] and all(n.endswith("-xw") for n in got[True][1])
    assert [n[: -len("-xw")] for n in got[True][1]] == got[False][1], "exact width must change only the -xw suffix"


def test_not_applied_when_fewer_lane_tiles_than_ranks(monkeypatch):
    """H = 512 has 4 lane tiles for 8 ranks: exact_width=True keeps the padded split (no -xw, [m, 1024] sliced)."""
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    S = jax.ShapeDtypeStruct
    e, h, i = 4, 512, 256

    def call(exact):
        def f(rank, x, w1, w2, s1, s2, sw1, sw2, gate, tw, ti):
            return kmod.fused_moe_decode(
                rank, x, w1, w2, s1, s2, sw1, jnp.ones((2 * 128,), jnp.float32), sw2,
                jnp.ones((h,), jnp.float32), gate, tw, ti, n_ranks=8, expert_start="rank", fold=True,
                exact_width=exact)

        return str(jax.make_jaxpr(f)(
            S((1, 2), jnp.int32), S((32, h), jnp.bfloat16), S((e, h, 2 * i), F8), S((e, i, h), F8),
            S((e, 1, 1, 2 * i), jnp.float32), S((e, 1, 1, h), jnp.float32), S((h, 256), jnp.bfloat16),
            S((128, h), jnp.bfloat16), S((1, h), jnp.bfloat16), S((32, 2), jnp.float32), S((32, 2), jnp.int32)))

    off, on = call(False), call(True)
    assert "-xw" not in on and _KOUT.findall(on) == _KOUT.findall(off) == [("32", "1024")]


def test_arming_passes_the_flag_to_the_target_model_only(monkeypatch):
    calls = []

    def arm(model, label, **kw):
        calls.append((label, kw))
        return 1

    class Runner:
        model = torch.nn.Linear(1, 1)
        drafter = type("D", (), {"draft_model": torch.nn.Linear(1, 1)})()

    monkeypatch.setattr(bridge, "arm_fused_moe_decode", arm)
    monkeypatch.delenv(FLAG, raising=False)
    assert bridge.arm_runner_models(Runner()) == 2
    monkeypatch.setenv(FLAG, "1")
    assert bridge.arm_runner_models(Runner()) == 2
    print(f"PBL_MOE_W arming calls {calls}")
    assert calls == [("target model", {}), ("draft model", {}), ("target model", {"exact_width": True}),
                     ("draft model", {})]


def _partial(op):
    cells = op.fn.__closure__ or ()
    parts = [c.cell_contents for c in cells if isinstance(c.cell_contents, functools.partial)]
    assert parts
    return parts[0]


def test_build_op_key_name_and_closure(mesh, monkeypatch):
    op_off, _ = _build(monkeypatch, mesh, False)
    keys_off = list(bridge._OPS)
    # Today's call (no exact_width argument at all) builds the same key.
    monkeypatch.setattr(bridge, "_OPS", {})
    today = bridge._build_op(mesh, N, TOPK, True, 256, True, 1.0)
    assert list(bridge._OPS) == keys_off and today.name == op_off.name
    assert "exact_width" not in _partial(op_off).keywords
    assert len(keys_off[0]) == 12 and keys_off[0][-1] == ("compact_pipe", 160, 32)
    # On: one more key element, its own op name, exact_width in the closure; both ops coexist.
    on = bridge._build_op(mesh, N, TOPK, True, 256, True, 1.0, exact_width=True)
    keys = list(bridge._OPS)
    print(f"PBL_MOE_W op names off {op_off.name} on {on.name}")
    assert len(keys) == 2 and keys[1] == keys[0] + (("exact_width", True),)
    assert on is not today and on.name != today.name
    assert _partial(on).keywords["exact_width"] is True


def test_env_key_registered_and_hashed(monkeypatch):
    import vllm_torchtpu.envs as envs
    from vllm_torchtpu.compilation import tpu_compiler

    monkeypatch.delenv(FLAG, raising=False)
    assert envs.FUSED_MOE_DECODE_EXACT_WIDTH is False
    monkeypatch.setenv(FLAG, "1")
    assert envs.FUSED_MOE_DECODE_EXACT_WIDTH is True
    assert FLAG in envs.environment_variables
    assert FLAG not in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_j_flag_on_op_builds_through_the_real_jax_op_like_off(mesh, monkeypatch):
    """J: the real torch_tpu jax_op (signature check, schema, registration) builds the flag-on op whenever it builds
    the flag-off one; their names differ."""
    from torch_tpu._internal.pallas import jax_op as real_jax_op

    _set_env(monkeypatch, False)
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    monkeypatch.setattr(bridge, "jax_op", real_jax_op)
    monkeypatch.setattr(bridge, "_OPS", {})
    result = {}
    for exact in (False, True):
        try:
            op = bridge._build_op(mesh, N, TOPK, True, 255, True, 1.0, **({"exact_width": True} if exact else {}))
            result[exact] = ("built", getattr(op, "__name__", None) or str(op)[:80])
        except Exception as exc:  # noqa: BLE001 - compared below
            result[exact] = ("raised", type(exc).__name__, str(exc)[:200])
    print(f"PBL_MOE_J off {result[False]} on {result[True]}")
    assert result[False][0] == result[True][0] == "built", result
