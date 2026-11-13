# SPDX-License-Identifier: Apache-2.0
"""Phase scopes of the fused prefill MoE: none unless KernelConfig.phase_scopes,
and with it the documented set of each window class (traced, not compiled,
over a deviceless tpu7x mesh)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import topologies
from jax.extend import core as jax_core
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_moe import fused

pytestmark = pytest.mark.cpu_test

# One layer of each weight family a fused MoE launch reads.
_FAMILIES = {
    "router": ((7168, 896), jnp.bfloat16),
    "router_bias": ((1, 896), jnp.float32),
    "latent_down": ((7168, 4096), jnp.bfloat16),
    "latent_up": ((3584, 7168), jnp.bfloat16),
    "latent_norm": ((1, 4096), jnp.float32),
    "shared_gu": ((7168, 6144), jnp.bfloat16),
    "shared_down": ((3072, 7168), jnp.bfloat16),
    "expert_gu": ((56, 448, 3072), jnp.uint32),
    "expert_gus": ((56, 112, 3072), jnp.uint8),
    "expert_down": ((56, 192, 3584), jnp.uint32),
    "expert_ds": ((56, 48, 3584), jnp.uint8),
}
_PHASES = {
    "m_mixture",
    "m_prologue",
    "m_enter_mesh",
    "m_input",
    "m_router",
    "m_latent",
    "m_peer_wait",
    "m_shared_gu",
    "m_shared_down",
    "m_metadata_wait",
    "m_experts",
    "m_expert_setup",
    "m_ring_enter",
    "m_first_column",
    "m_first_init",
    "m_act_gather",
    "m_act_wait",
    "m_consume_tail",
    "m_consume_last",
    "m_remaining_columns",
    "m_emit_subtotals",
    "m_ring_poll",
    "m_ring_drain",
    "m_ring_final_drain",
    "m_epilogue",
    "m_epi_norm",
    "m_epi_up",
}
# Up to 4K the route sort and the latent ring finish between the shared-expert
# panels; from 8K their tails follow the last panel, and the expert tiles have
# 256 rows.
_WINDOW_PHASES = {
    512: {"m_gu_bm64", "m_gu_bm128", "m_down_bm64", "m_down_bm128"},
    8192: {
        "m_metadata_tail",
        "m_latent_drain",
        "m_gu_bm128",
        "m_gu_bm192",
        "m_gu_bm256",
        "m_down_bm64",
        "m_down_bm128",
        "m_down_bm192",
        "m_down_bm256",
    },
}


def _scope_names(jaxpr):
    """The named scopes on the equations of ``jaxpr`` and of its sub-jaxprs."""
    names = set()
    for eqn in jaxpr.eqns:
        names.update(str(eqn.source_info.name_stack).split("/"))
        for param in eqn.params.values():
            for sub in param if isinstance(param, tuple | list) else (param,):
                if isinstance(sub, jax_core.ClosedJaxpr):
                    sub = sub.jaxpr
                if isinstance(sub, jax_core.Jaxpr):
                    names |= _scope_names(sub)
    return names


@pytest.mark.parametrize(
    ("window", "scopes"), [(512, False), (512, True), (8192, True)]
)
def test_fused_moe_phase_scopes_follow_the_kernel_config(monkeypatch, window, scopes):
    monkeypatch.setattr(fused, "PHASE_SCOPES", scopes)
    devices = topologies.get_topology_desc("tpu7x:2x2x4", platform="tpu").devices
    mesh = Mesh(np.array(sorted(devices, key=lambda d: d.id)), ("k3_sp",))

    def shard(shape, dtype, spec=None):
        spec = P("k3_sp") if spec is None else spec
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))

    rows, scalar = window // 32, shard((1,), jnp.int32, P())
    families = {
        name: shard((32, 1, *shape), dtype)
        for name, (shape, dtype) in _FAMILIES.items()
    }
    args = (
        shard((32, rows, 7168), jnp.bfloat16),  # prefix
        shard((32, 8, rows, 7168), jnp.bfloat16),  # residual blocks
        shard((32, 1, 7168), jnp.float32),  # residual-mixture scores (mlp_res)
        shard((32, 1, 7168), jnp.bfloat16),  # FFN norm
        scalar,  # active blocks
        families,
        scalar,  # valid length
        scalar,  # family slot
    )
    # A fresh op: no trace cached under the other setting is reused.
    op = fused.make_indexed_mixture_moe_sp_ring(mesh, seq_len=window)
    names = {n for n in _scope_names(op.trace(*args).jaxpr.jaxpr) if n.startswith("m_")}
    assert names == (_PHASES | _WINDOW_PHASES[window] if scopes else set())
