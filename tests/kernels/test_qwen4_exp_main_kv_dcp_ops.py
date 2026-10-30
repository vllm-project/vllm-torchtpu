# SPDX-License-Identifier: Apache-2.0
"""DCP-2 op smoke (VLLM_TPU_QSA_MAIN_KV_DCP): build every DCP jax_op and call
each once through torch on tiny shapes, real and fake.

Kernel tests call the JAX steps directly and miss what only the torch binding
sees: jax_op signature checks (an unannotated argument fails at op build),
schemas, and fakes whose shapes disagree with the real outputs. Run before any
served DCP arm; needs torch_tpu, the real calls a TPU chip.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_qwen4_exp_mtp_draft_rows as mtp_rows
import torch

pytest.importorskip("torch_tpu")

from vllm_torchtpu.layers.adapter.custom_ops import qsa_op  # noqa: E402

OPTIONS = dict(mtp_rows.OPTIONS, token_topk=256)
RATIO, TOPK, KV_HEADS, SHARDS = 4, 256, 2, 2


def _torch(x, device):
    """jax array -> torch tensor (bf16 / fp8 through same-width int views)."""
    views = {
        "bfloat16": (np.uint16, torch.int16, torch.bfloat16),
        "float8_e4m3fn": (np.uint8, torch.uint8, torch.float8_e4m3fn),
    }
    host = np.asarray(x)
    if x.dtype.name in views:
        raw, carrier, dtype = views[x.dtype.name]
        return torch.from_numpy(host.view(raw)).view(carrier).view(dtype).to(device)
    return torch.from_numpy(host.copy()).to(device)


def _inputs(device):
    head, pool, tail = mtp_rows._inputs(True, mtp_rows.STARTS, mtp_rows.CONTEXT)
    return [_torch(x, device) for x in head], _torch(pool, device), [_torch(x, device) for x in tail]


def _run(step, select, attend, combine, head, pool, tail, rank):
    """Every DCP op once, chained the way qsa_op wires them; returns outputs."""
    tokens = head[0].shape[0]
    head_dim = head[4].shape[0]
    rank = rank.to(torch.int32)
    out = {}
    out["step"] = step(*head, pool.clone(), *tail, rank, dcp_size=2, **OPTIONS)
    shards = [
        select(*head, pool.clone(), *tail, torch.tensor([s], dtype=torch.int32, device=rank.device),
               rank, shard_size=SHARDS, dcp_size=2, **OPTIONS)
        for s in range(SHARDS)
    ]
    out["select"] = shards[0]
    new_pool, q, gate, lengths, req, starts, _ = shards[0]
    gathered = torch.cat([shard[-1] for shard in shards])
    out["attend"] = attend(q, gate, lengths, req, starts, tail[3], tail[1], new_pool, gathered,
                           rank, kv_heads=KV_HEADS, compress_ratio=RATIO, token_topk=TOPK,
                           dcp_size=2, merged=True)
    _, partial, lse, _ = out["step"]
    heads = partial.shape[1] // 2
    out_pair = torch.stack([partial, partial])
    lse_pair = torch.stack([lse, lse])
    gate = head[0].reshape(tokens, 2 * heads, 2 * head_dim)[:, :heads, head_dim:]
    out["combine"] = combine(out_pair, lse_pair, gate.contiguous(), rank)
    return out


def _shapes(out):
    return {k: [tuple(t.shape) + (t.dtype,) for t in (v if isinstance(v, (tuple, list)) else (v,))]
            for k, v in out.items()}


def test_dcp_ops_build_and_run_on_tpu():
    """Op build + one real call each (both DCP ranks); outputs finite where
    expected and shaped exactly like the fakes say."""
    try:
        device = torch.device("tpu")
        torch.empty(1, device=device)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no TPU device: {exc}")
    if jax.default_backend() == "cpu":
        pytest.skip("needs the TPU backend")
    ops = qsa_op._dcp_ops()
    head, pool, tail = _inputs(device)
    for r in (0, 1):
        real = _run(*ops, head, pool, tail, torch.tensor([r], device=device))
        for name, value in real.items():
            for t in value if isinstance(value, (tuple, list)) else (value,):
                assert t.device.type == device.type, name
        assert torch.isfinite(real["combine"].float()).all()
        assert not torch.isnan(real["step"][1]).any()

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            fake = _run(*ops, *[[mode.from_tensor(t) for t in x] if isinstance(x, list)
                                else mode.from_tensor(x) for x in (head, pool, tail)],
                        mode.from_tensor(torch.tensor([r], device=device)))
        assert _shapes(fake) == _shapes(real)
