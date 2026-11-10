# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""The all-RPA PCP path with the PCP2 pair exchange must not add a lax.axis_index
outside Pallas kernels: under the torch_tpu export it lowers to partition-id,
which the SPMD partitioner refuses (DRA2 run03 j-fdeba0e9-0998-4215-bdb8).

Traced on CPU (make_jaxpr) in a subprocess with two host devices.
"""

import os
import subprocess
import sys
import textwrap

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src")

_PROGRAM = textwrap.dedent(
    r"""
    import importlib
    import jax, jax.numpy as jnp, numpy as np
    from jax.experimental.pallas import tpu as pltpu
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    pltpu.get_tpu_info = lambda: info
    for name in ("jax._src.pallas.mosaic.tpu_info", "jax._src.tpu_info"):
        try:
            m = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(m, "get_tpu_info"):
            m.get_tpu_info = lambda: info
    from jax.sharding import Mesh
    from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import wrapper as w
    mesh = Mesh(np.asarray(jax.devices()[:2]), ("pcp",))
    lt, nq, nkv, d, page, seqs = 512, 4, 2, 256, 128, 2
    q = jnp.zeros((2 * lt, nq, d), jnp.bfloat16)
    k = jnp.zeros((2 * lt, nkv, d), jnp.bfloat16)
    cache = jnp.zeros((2 * 16, 2 * nkv, d // 4, 4, page), jnp.float8_e4m3fn)
    f = lambda c, q, k: w.sharded_pcp_ragged_paged_attention(
        mesh, q, k, k, c, jnp.array([700, 0], jnp.int32),
        jnp.zeros((seqs * 8,), jnp.int32), jnp.array([0, 600, 600], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32), None, 0.0625, k_scale=1.0, v_scale=1.0,
        cp_kv_cache_interleave_size=128, q_block_size=512,
        kv_layout=KVLayout.SEQ_ALONG_LANE, history_rpa_mode=True, current_rpa_mode=True)
    text = str(jax.make_jaxpr(f)(cache, q, k))
    print("HAS_PPERMUTE", "ppermute" in text)
    import re
    # The primitive only (collectives print an unrelated axis_index_groups=... parameter).
    print("AXIS_INDEX_COUNT", len(re.findall(r"axis_index\[", text)))
    """
)


def _trace(pair_exchange):
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.path.abspath(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["TPU_PCP_RPA_PAIR_EXCHANGE"] = pair_exchange
    out = subprocess.run([sys.executable, "-c", _PROGRAM], env=env, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    return dict(line.split(" ", 1) for line in out.stdout.splitlines() if line.startswith(("HAS_", "AXIS_")))


def test_pair_exchange_adds_no_axis_index():
    """The Pallas kernels' own axis_index uses (Mosaic, allowed) appear in both
    traces; the pair-exchange path must not add one at the XLA level."""
    base = _trace("0")
    pair = _trace("1")
    assert pair.get("HAS_PPERMUTE") == "True" and base.get("HAS_PPERMUTE") == "False", (base, pair)
    assert pair["AXIS_INDEX_COUNT"] == base["AXIS_INDEX_COUNT"], (base, pair)
