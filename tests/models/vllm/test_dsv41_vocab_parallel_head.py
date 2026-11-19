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


def _head_setup(monkeypatch, vocab=VOCAB, width=WIDTH, device="cpu", **changes):
    """An LM head, logits processor and vLLM config as DeepSeek-V4.1 Flash
    serves them on 8 EP ranks with DP lockstep, 64 requests and DSpark K=5,
    with ``changes`` applied."""
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
        torch.zeros(vocab, width, dtype=torch.bfloat16, device=device),
        requires_grad=False,
    )
    lm_head.quant_method = UnquantizedEmbeddingMethod()
    processor = SimpleNamespace(
        logits_as_input=False,
        soft_cap=None,
        scale=1.0,
        org_vocab_size=vocab,
        head_dtype=None,
    )
    parallel = SimpleNamespace(
        data_parallel_size=WORLD,
        enable_expert_parallel=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        speculative_config=SimpleNamespace(method="dspark", num_speculative_tokens=5),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
        lora_config=None,
    )
    for name, value in changes.items():
        if name == "greedy_head_backend":
            monkeypatch.setenv("TPU_GREEDY_HEAD_BACKEND", value)
        elif name == "ep_ranks":
            ranks = list(range(value))
            monkeypatch.setattr(ep_mesh, "ep_device_ids", lambda ranks=ranks: ranks)
        elif name == "quantized":
            lm_head.quant_method = object()
        elif name == "bias":
            lm_head.bias = torch.nn.Parameter(torch.zeros(vocab), requires_grad=False)
        elif name == "max_num_seqs":
            config.scheduler_config.max_num_seqs = value
        elif hasattr(parallel, name):
            setattr(parallel, name, value)
        else:
            setattr(processor, name, value)
    return lm_head, processor, config


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
    lm_head, processor, config = _head_setup(monkeypatch, **changes)
    weight = lm_head.weight
    assert reason in vocab_parallel_head.whole_head_reason(lm_head, processor, config)
    assert not vocab_parallel_head.shard_lm_head(lm_head, processor, config)
    assert lm_head.weight is weight
    assert not vocab_parallel_head.is_vocab_parallel(lm_head)


@pytest.mark.parametrize("tpu_v7", [False, True])
def test_head_stays_whole_where_a_fused_head_reads_the_whole_vocab(monkeypatch, tpu_v7):
    """With 48 requests and K=5 on DeepSeek-V4.1 Flash's head, the fused
    greedy head (288 rows, TPU v7 only) or the fused DSpark draft sampler
    serves the step and needs the whole vocab; with 64 requests neither does,
    and the head is split."""
    from vllm_torchtpu.kernels import fused_greedy_head

    monkeypatch.setattr(
        fused_greedy_head, "get_device_name", lambda: "TPU v7" if tpu_v7 else "cpu"
    )
    flash = dict(vocab=129280, width=5120, device="meta")
    head = _head_setup(monkeypatch, max_num_seqs=48, **flash)
    reason = vocab_parallel_head.whole_head_reason(*head)
    assert ("greedy head" if tpu_v7 else "draft sampler") in reason

    head = _head_setup(monkeypatch, **flash)
    assert vocab_parallel_head.whole_head_reason(*head) is None
