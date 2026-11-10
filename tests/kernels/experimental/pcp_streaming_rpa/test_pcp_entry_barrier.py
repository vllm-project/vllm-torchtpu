# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""TPU_PCP_ENTRY_BARRIER: the PCP kernels that copy into the peer at entry
barrier on the runtime barrier semaphore first, under a collective id.

Traced on CPU (make_jaxpr; no Mosaic lowering) in a subprocess with two host
devices: with the flag, each kernel's pallas_call carries its collective id and
its body takes the barrier semaphore; without it, neither appears.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src")

_PROGRAM = textwrap.dedent(
    r"""
    import os, sys
    import jax, jax.numpy as jnp, numpy as np
    from jax.experimental.pallas import tpu as pltpu
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    pltpu.get_tpu_info = lambda: info
    import importlib
    for name in ("jax._src.pallas.mosaic.tpu_info", "jax._src.tpu_info"):
        try:
            m = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(m, "get_tpu_info"):
            m.get_tpu_info = lambda: info
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    which = sys.argv[1]
    devs = jax.devices()[:2]
    if which == "attention":
        from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import wrapper as w
        mesh = Mesh(np.asarray(devs), ("pcp",))
        lt, nq, nkv, d, page, seqs = 512, 4, 2, 256, 128, 2
        q = jnp.zeros((2 * lt, nq, d), jnp.bfloat16)
        k = jnp.zeros((2 * lt, nkv, d), jnp.bfloat16)
        cache = jnp.zeros((2 * 16, 2 * nkv, d // 4, 4, page), jnp.float8_e4m3fn)
        f = lambda c, q, k: w.sharded_pcp_ragged_paged_attention(
            mesh, q, k, k, c, jnp.array([700, 0], jnp.int32),
            jnp.zeros((seqs * 8,), jnp.int32), jnp.array([0, 600, 600], jnp.int32),
            jnp.array([0, 0, 1], jnp.int32), None, 0.0625, k_scale=1.0, v_scale=1.0,
            cp_kv_cache_interleave_size=128, q_block_size=512,
            kv_layout=KVLayout.SEQ_ALONG_LANE)
        text = str(jax.make_jaxpr(f)(cache, q, k))
        ids = (14, 15)
    else:
        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
            build_pcp_rank_major_token_order)
        from vllm_torchtpu.layers.core.gdn_attention import (
            run_jax_gdn_attention_pooled_pcp_prefill_projection)
        mesh = Mesh(np.asarray(devs), ("pcp",))
        tokens, H, N_KQ, N_V, DK, DV, KS, IL = 2048, 512, 4, 16, 128, 128, 4, 128
        qkv_dim = 2 * N_KQ * DK + N_V * DV
        qkvz_dim = qkv_dim + N_V * DV
        sds = jax.ShapeDtypeStruct
        args = (
            sds((tokens, H), jnp.bfloat16), sds((qkvz_dim, H), jnp.bfloat16), None,
            sds((tokens, N_V), jnp.bfloat16), sds((tokens, N_V), jnp.bfloat16),
            sds((2 * 3 * 4, 256, 1, 16, 128), jnp.float8_e4m3fn),
            sds((qkv_dim, 1, KS), jnp.bfloat16), sds((qkv_dim,), jnp.bfloat16),
            sds((N_V,), jnp.float32), sds((N_V,), jnp.float32),
            sds((2,), jnp.int32), sds((3,), jnp.int32), sds((3,), jnp.int32), sds((2,), jnp.int32),
        )
        f = lambda *a: run_jax_gdn_attention_pooled_pcp_prefill_projection(
            *a, n_kq=N_KQ, n_v=N_V, d_k=DK, d_v=DV, kernel_size=KS, pool_block_tokens=1024,
            pcp_size=2, interleave_size=IL, mesh=mesh, recurrent_state_dtype=jnp.float32)
        text = str(jax.make_jaxpr(f)(*args))
        ids = (13,)
    for cid in ids:
        print("HAS_ID", cid, f"collective_id={cid}" in text)
    print("HAS_BARRIER", "get_barrier_semaphore" in text)
    """
)


def _run(which, flag):
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.path.abspath(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["TPU_PCP_ENTRY_BARRIER"] = "1" if flag else "0"
    out = subprocess.run(
        [sys.executable, "-c", _PROGRAM, which],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return [line for line in out.stdout.splitlines() if line.startswith("HAS_")]


@pytest.mark.parametrize("which", ["attention", "gdn"])
def test_entry_barrier_flag_adds_barrier_and_collective_id(which):
    on = _run(which, True)
    off = _run(which, False)
    assert on and all(line.endswith("True") for line in on), on
    assert off and all(line.endswith("False") for line in off), off
