# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_DECODE_BOUNDED wiring in Qwen4ExpPooledPLE (q38 L1 D07-b2; CPU).

The speculative conv op is replaced by a builder that runs the real JAX
``pooled_short_conv_speculative`` (pool region read / write swapped for plain
XLA stand-ins, as in tests/kernels/test_qwen4_exp_ple_decode_bounded.py) on
host copies of the torch tensors, and writes the updated pool back. A decode
step with uniform 4-token verify windows then runs forward_with_history with
and without ``uniform_windows``: the PLE addend, the hidden state after PLE
(``hidden + addend``, as Qwen4ExpDecoderLayer.apply_ple returns it) and the
pool are compared bitwise. With the flag unset the bounded op is never built
and ``uniform_windows`` changes nothing.
"""

import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tests.models.vllm.test_qwen4_exp import (
    _PLE,
    HC,
    HIDDEN,
    _randomize_ple,
    _set_tp,
    _text_config,
)
from vllm_torchtpu.kernels.qwen4_exp import pool as pool_mod
from vllm_torchtpu.models.vllm.qwen4_exp.ple import PLEInputs

FLAG = "VLLM_TPU_PLE_DECODE_BOUNDED"
STATE_LEN, WINDOW = 9, 4
CHANNELS = HC * HIDDEN


class _Group:
    rank_in_group = 0
    world_size = 1

    def all_reduce(self, rows):
        return rows


def _to_jax(t: torch.Tensor):
    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.view(torch.int16).numpy()).view(jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _to_torch(a) -> torch.Tensor:
    if a.dtype == jnp.bfloat16:
        return torch.from_numpy(np.asarray(a.view(jnp.int16)).copy()).view(torch.bfloat16)
    return torch.from_numpy(np.asarray(a).copy())


def _read_region_ref(pool, ids, *, elements, dtype, tok0=0):
    return pool[jnp.maximum(ids, 0)][:, :elements]


def _write_region_ref(pool, values, ids, *, tok0=0):
    destination = jnp.where(ids > 0, ids, pool.shape[0])
    return pool.at[destination, : values.shape[1]].set(values, mode="drop")


def _adapter(monkeypatch, env):
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_torchtpu.models.vllm.qwen4_exp import ple_storage, pooled_ple
    from vllm_torchtpu.models.vllm.qwen4_exp.parallel import PLEParallel

    for name in (FLAG, "VLLM_TPU_PLE_CONV_ONEHOT", "Q38_PROBE_PLE_ONEHOT_MUTANT", "VLLM_TPU_PLE_FUSED"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _set_tp(monkeypatch, 0, 1)
    monkeypatch.setattr(pool_mod, "read_region", _read_region_ref)
    monkeypatch.setattr(pool_mod, "write_region", _write_region_ref)
    group = _Group()

    def raw_lookup(rows, width, chunk_tokens=8):
        def run(table, ids, vocab_start):
            result = table.reshape(rows, width)[ids.clamp(0, rows - 1)].clone()
            result[(ids < 0) | (ids >= rows)] = 0
            return result.flatten(-2)

        return run

    monkeypatch.setattr(ple_storage, "build_raw_lookup_op", raw_lookup)
    monkeypatch.setattr(
        ple_storage, "build_decode_op", lambda: lambda raw, scale: raw.view(torch.bfloat16)
    )
    log = SimpleNamespace(built=[], calls=[])

    def conv_builder(dilation, state_len, past_matmul=False):
        def run(x, weight, pool, starts, ids, initial):
            raise AssertionError("the non-speculative conv must not run in this test")

        return run

    def spec_builder(dilation, state_len, bounded=0):
        log.built.append(bounded)
        fn = jax.jit(
            functools.partial(
                pool_mod.pooled_short_conv_speculative,
                dilation=dilation,
                state_len=state_len,
                bounded=bounded,
            )
        )

        def run(x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed):
            log.calls.append(bounded)
            new_pool, output = fn(
                *map(_to_jax, (x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed))
            )
            pool.copy_(_to_torch(new_pool))
            return _to_torch(output)

        return run

    monkeypatch.setattr(pooled_ple, "build_pooled_short_conv_op", conv_builder)
    monkeypatch.setattr(pooled_ple, "build_pooled_speculative_short_conv_op", spec_builder)
    with set_default_torch_dtype(torch.bfloat16):
        reference = pooled_ple.Qwen4ExpPLELayer(_text_config(**_PLE), 0, None, "ple")
        _randomize_ple(reference)
        adapted = pooled_ple.Qwen4ExpPooledPLE(
            _text_config(**_PLE), 0, None, "ple", parallel=PLEParallel(group, group,
            embedding_across_dp=False), table_group=group,
        )
    weights = dict(reference.state_dict())
    table = weights.pop("ple_embedding.ngram_embedding.weight")
    for i, rows in enumerate(table.split(22)):
        weights[f"ple_embedding.ngram_embedding.shard_{i}.weight"] = rows
    adapted.load_checkpoint(iter(weights.items()))
    adapted.finalize_weights()
    for proj in (adapted.key_proj, adapted.value_proj):
        # vLLM's CPU platform gemm reads `cpu_linear`, which its
        # process_weights_after_loading would set; this test loads the PLE
        # layer directly, so give it the plain linear (same math in both arms).
        if not hasattr(proj, "cpu_linear"):
            proj.cpu_linear = lambda x, weight, bias=None: torch.nn.functional.linear(x, weight, bias)
    pages = 1 + 8 * WINDOW + 3
    pool = torch.zeros(pages, STATE_LEN * CHANNELS, dtype=torch.bfloat16)
    adapted.bind_pool(pool)
    return adapted, pool, log


def _decode_step(seed, *, count, tokens, live):
    """``live`` 4-token verify windows first, then empty rows up to ``count``
    (max-num-seqs), in a ``tokens``-row bucket."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.arange(count)
    ckpt = torch.zeros(count, WINDOW, dtype=torch.int32)
    ckpt[:live] = (1 + torch.randperm(8 * WINDOW + 2, generator=g)[: live * WINDOW]).to(
        torch.int32
    ).reshape(live, WINDOW)
    offsets = torch.randint(0, WINDOW, (count,), generator=g)
    read_ids = torch.where(rows < live, ckpt[rows, offsets], 0).to(torch.int32)
    slots = ckpt[:, 0].clone()
    starts = torch.clamp(torch.arange(count + 1) * WINDOW, max=live * WINDOW).to(torch.int32)
    initial = rows < live
    windowed = rows < live
    ids = torch.randint(4, 30, (tokens,), generator=g, dtype=torch.int64)
    contexts = torch.randint(4, 30, (count, 2), generator=g, dtype=torch.int64)
    hidden = torch.randn(tokens, HC * HIDDEN, generator=g).to(torch.bfloat16)
    pool = torch.randn(1 + 8 * WINDOW + 3, STATE_LEN * CHANNELS, generator=g).to(torch.bfloat16)
    return dict(
        inputs=PLEInputs(ids, starts, contexts),
        slots=slots,
        initial=initial,
        valid=starts[-1:],
        speculative=(read_ids, ckpt, windowed),
        hidden=hidden,
        pool=pool,
    )


def _forward(adapted, pool, step, uniform_windows):
    pool.copy_(step["pool"])
    with torch.no_grad():
        addend = adapted.forward_with_history(
            step["hidden"],
            step["inputs"],
            step["slots"],
            step["initial"],
            step["valid"],
            None,
            step["speculative"],
            uniform_windows=uniform_windows,
        )
    return addend, step["hidden"] + addend, pool.clone()


def _same(a, b, what):
    assert a.dtype == b.dtype and a.shape == b.shape, what
    assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"{what} differs"


@pytest.mark.parametrize("flag", ["1", "2"])
@pytest.mark.parametrize("tokens,live", [(16, 4), (16, 3), (32, 8), (32, 5), (8, 1)])
def test_uniform_windows_run_the_bounded_op_bitwise(monkeypatch, flag, tokens, live):
    adapted, pool, log = _adapter(monkeypatch, {FLAG: flag})
    assert log.built == [0, int(flag)]
    step = _decode_step(tokens * 10 + live, count=8, tokens=tokens, live=live)
    base = _forward(adapted, pool, step, uniform_windows=False)
    bounded = _forward(adapted, pool, step, uniform_windows=True)
    assert log.calls == [0, int(flag)]
    assert torch.count_nonzero(base[0][: live * WINDOW]) > 0
    assert torch.count_nonzero(base[0][live * WINDOW :]) == 0
    assert not torch.equal(base[2].view(torch.int16), step["pool"].view(torch.int16)), "no checkpoint written"
    for got, want, what in zip(bounded, base, ("PLE addend", "hidden after PLE", "pool")):
        _same(got, want, what)


def test_flag_unset_never_builds_or_runs_the_bounded_op(monkeypatch):
    adapted, pool, log = _adapter(monkeypatch, {})
    assert log.built == [0]
    assert adapted._speculative_conv_bounded is None
    step = _decode_step(7, count=8, tokens=16, live=4)
    base = _forward(adapted, pool, step, uniform_windows=False)
    flagged = _forward(adapted, pool, step, uniform_windows=True)
    assert log.calls == [0, 0]
    for got, want, what in zip(flagged, base, ("PLE addend", "hidden after PLE", "pool")):
        _same(got, want, what)


def test_bounded_spec_ops_build_through_jax_op():
    """The bounded ops build through torch_tpu's real pallas.jax_op next to
    the default one (distinct op names)."""
    from vllm_torchtpu.layers.adapter.custom_ops import ple_op

    ops = [ple_op.build_pooled_speculative_short_conv_op(3, 9, *b) for b in ((), (1,), (2,))]
    assert all(callable(op) for op in ops)
    assert len({id(op) for op in ops}) == 3
