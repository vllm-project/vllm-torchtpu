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
"""MK draft greedy pick (VLLM_TPU_MK_DRAFT): no partition id after SPMD partitioning.

The served compile (torch_tpu export, SPMD) refuses `partitionId`; `greedy_tokens` on
the tp-sharded per-rank pairs outside shard_map partitions to one (exos j-760b499c: the
MK fused loop's first compile failed there and every retry then tripped vLLM's
"pre_grad_custom_pass" assert on the half-configured backend).
Run: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest <this>
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel.wholestep.tail import greedy_tokens  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_draft import make_draft_greedy  # noqa: E402


def test_draft_greedy_has_no_partition_id_and_matches_greedy_tokens():
    R, M = 8, 16
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    rng = np.random.default_rng(0)
    top = np.zeros((R * M, 128), np.float32)
    top[:, 0] = rng.integers(0, 4, R * M)  # small ints: ties across ranks
    top[:, 1] = np.arange(R * M) % 997
    x = jax.device_put(jnp.asarray(top), NamedSharding(mesh, P("tp")))
    f = jax.jit(make_draft_greedy(mesh))
    hlo = f.lower(x).compile().as_text()
    assert "partition-id" not in hlo
    np.testing.assert_array_equal(
        np.asarray(f(x)), np.asarray(greedy_tokens(jnp.asarray(top), R, M))
    )


def test_stock_and_mk_draft_configs_compile_two_modules(monkeypatch):
    """The MK draft programs are built after the stock draft programs compiled, from
    `mk_draft_compile_config(draft config)` (compile sizes cut to the draft kernel's
    rows, own inductor config): both support_torch_compile modules compile and run,
    each twice (the fused loop's warm call), the stock config unchanged."""
    from types import SimpleNamespace as NS

    import torch
    import vllm.envs
    from torch import nn
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import (
        CompilationConfig,
        DeviceConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.compilation import CompilationMode

    from vllm_torchtpu.spec_decode.mtp_fused_loop import mk_draft_compile_config

    monkeypatch.setattr(vllm.envs, "VLLM_USE_AOT_COMPILE", False, raising=False)

    class Loop(nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.prefix = prefix

        def forward(self, x):
            return x * 2 + 1

    cfg = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, backend="eager", compile_sizes=[4, 8, 8192]
        ),
    )
    mk_cfg, sizes = mk_draft_compile_config(cfg, NS(mk_draft=NS(max_rows={1: 16})), 1)
    assert sizes == [4, 8] and mk_cfg.compilation_config.compile_sizes == [4, 8]
    assert cfg.compilation_config.compile_sizes == [4, 8, 8192]
    assert "pre_grad_custom_pass" not in mk_cfg.compilation_config.inductor_compile_config
    deco = support_torch_compile(dynamic_arg_dims={"x": 0})
    stock_cls = deco(type("StockLoop", (Loop,), {}))
    mk_cls = deco(type("MkLoop", (Loop,), {}))
    x = torch.ones(4)
    with set_current_vllm_config(cfg):
        stock = stock_cls(vllm_config=cfg, prefix="mtp_fused_loop_k3")
        for _ in range(2):
            torch.testing.assert_close(stock(x), x * 3)
    with set_current_vllm_config(mk_cfg):
        mk = mk_cls(vllm_config=mk_cfg, prefix="mk_mtp_fused_loop_k3")
        for _ in range(2):
            torch.testing.assert_close(mk(x), x * 3)
