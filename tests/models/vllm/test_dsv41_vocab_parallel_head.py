# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-parallel LM head: logits against the plain matmul, and the setups that
keep the whole head.

The SPMD head needs 8 devices; without them the check reruns itself in a
subprocess on 8 forced CPU devices, so it also runs on the CPU lane.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.vllm.deepseek_v41.common import vocab_parallel_head

WORLD, VOCAB, WIDTH = 8, 8 * 48, 64


def test_logits_equal_plain_matmul():
    if len(jax.devices()) >= WORLD:
        _check_logits()
        return
    code = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('t', {__file__!r});"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "m._check_logits()"
    )
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"--xla_force_host_platform_device_count={WORLD}",
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


def _check_logits():
    mesh = Mesh(np.array(jax.devices()[:WORLD]), (vocab_parallel_head._AXIS,))
    sharded = NamedSharding(mesh, P(vocab_parallel_head._AXIS, None))
    head = jax.jit(vocab_parallel_head.make_head(mesh, WORLD, VOCAB))
    rng = np.random.default_rng(0)
    # The target and draft heads' rows per rank at decode batch 2.
    for rows in (16, 40):
        x = jnp.asarray(rng.standard_normal((WORLD * rows, WIDTH)), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((VOCAB, WIDTH)), jnp.bfloat16)
        want = jax.lax.dot_general(
            x, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        xs = jax.device_put(x, sharded)
        got = head(xs, jax.device_put(w, sharded))
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        # Each rank must score its own vocab slice.
        shifted = jax.device_put(jnp.roll(w, VOCAB // WORLD, axis=0), sharded)
        assert not np.array_equal(np.asarray(head(xs, shifted)), np.asarray(want))


def _head_setup(monkeypatch, **changes):
    """An LM head, logits processor and parallel config as DeepSeek-V4.1 Flash
    serves them on 8 EP ranks with DP lockstep, with ``changes`` applied."""
    import torch
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    from vllm_torchtpu.distributed import ep_mesh

    monkeypatch.setattr(ep_mesh, "ep_device_ids", lambda: list(range(WORLD)))
    monkeypatch.setattr(ep_mesh, "ep_mesh_index", lambda: 0)
    monkeypatch.delenv("TPU_GREEDY_HEAD_BACKEND", raising=False)
    lm_head = torch.nn.Module()
    lm_head.weight = torch.nn.Parameter(
        torch.zeros(VOCAB, WIDTH, dtype=torch.bfloat16), requires_grad=False
    )
    lm_head.quant_method = UnquantizedEmbeddingMethod()
    processor = SimpleNamespace(
        logits_as_input=False,
        soft_cap=None,
        scale=1.0,
        org_vocab_size=VOCAB,
        head_dtype=None,
    )
    parallel = SimpleNamespace(data_parallel_size=WORLD, enable_expert_parallel=True)
    for name, value in changes.items():
        if name == "greedy_head_backend":
            monkeypatch.setenv("TPU_GREEDY_HEAD_BACKEND", value)
        elif name == "ep_ranks":
            ranks = list(range(value))
            monkeypatch.setattr(ep_mesh, "ep_device_ids", lambda ranks=ranks: ranks)
        elif name == "quantized":
            lm_head.quant_method = object()
        elif name == "bias":
            lm_head.bias = torch.nn.Parameter(torch.zeros(VOCAB), requires_grad=False)
        elif hasattr(parallel, name):
            setattr(parallel, name, value)
        else:
            setattr(processor, name, value)
    return lm_head, processor, parallel


def test_measured_setup_splits_the_head(monkeypatch):
    head = _head_setup(monkeypatch)
    assert vocab_parallel_head.whole_head_reason(*head) is None


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"data_parallel_size": 1}, "DP lockstep"),
        ({"enable_expert_parallel": False}, "DP lockstep"),
        ({"greedy_head_backend": "pallas"}, "TPU_GREEDY_HEAD_BACKEND=pallas"),
        ({"org_vocab_size": VOCAB - 16}, "padded"),
        ({"scale": 0.5}, "scaled"),
        ({"soft_cap": 30.0}, "soft-capped"),
        ({"bias": True}, "bias"),
        ({"quantized": True}, "quantized"),
        ({"ep_ranks": 1}, "EP over 2 or more ranks"),
        ({"ep_ranks": 7}, "divisible by 7"),
    ],
)
def test_unsupported_setups_keep_the_whole_head(monkeypatch, changes, reason):
    lm_head, processor, parallel = _head_setup(monkeypatch, **changes)
    weight = lm_head.weight
    assert reason in vocab_parallel_head.whole_head_reason(lm_head, processor, parallel)
    assert not vocab_parallel_head.shard_lm_head(lm_head, processor, parallel)
    assert lm_head.weight is weight
    assert not vocab_parallel_head.is_vocab_parallel(lm_head)
