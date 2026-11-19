# SPDX-License-Identifier: Apache-2.0
"""Bounded PD staging composed with the victim host tier on the served PD prefill (CPU).

With ``QWEN4_EXP_PD_STAGING_LEASES=N`` the PD producer registers its pools with
Raiden's ``register_pools(..., staging_leases=N, staging_blocks_per_pool=...)``:
a bounded host arena instead of tpu_sync's full host mirror (about 515 GB at
the 60 GiB pool). With a host tier the same stage pools are also registered by
the Raiden offload connector (tpu_sync's KVCacheManager over the region
tensors). The two native registrations over one storage are a TPU question
(the victim bytes-identity run engages both); these tests pin the Python
composition the bounded arena depends on, so a host tier can never silently
bring the full mirror back (515 GB mirror + 320 GB tier > the 864 GiB guard):

- the multi connector forwards ``register_runner`` to every child, so the PD
  child's leased registration runs;
- the composed producer (multi + victim policy + leases) is admitted, its PD
  child is a producer, and the staging step passes the leases and one hint per
  pool (not ``{}``, the full mirror) with the composed config in force;
- the offload child registers exactly the stage's own region storages, the
  storages the PD manifest describes.
"""

from types import SimpleNamespace as NS

import torch

from tests.runner.test_qwen4_exp_host_tier import (  # noqa: F401  (pooled: autouse env fixture)
    _pd_child,
    _register,
    _stage_caches,
    pooled,
    pp8_pool,
    served_prefill,
)
from tests.runner.test_qwen4_exp_pd_staging_leases import FakeEngine, manifest, register
from vllm_torchtpu.models.vllm.qwen4_exp import config as qconfig
from vllm_torchtpu.models.vllm.qwen4_exp.offload import pool_regions


def test_multi_forwards_register_runner_to_every_child():
    from vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector import TPUMultiConnector

    seen = []
    pd = NS(register_runner=lambda r: seen.append(("pd", r)))
    offload = NS(register_runner=lambda r: seen.append(("offload", r)))
    runner = object()
    TPUMultiConnector.register_runner(NS(_connectors=[pd, offload]), runner)
    assert seen == [("pd", runner), ("offload", runner)]


def test_composed_producer_keeps_bounded_staging(monkeypatch):
    from vllm.config import KVTransferConfig

    monkeypatch.setenv("QWEN4_EXP_PD_STAGING_LEASES", "16")
    monkeypatch.setenv("QWEN4_EXP_HOST_TIER_POLICY", "victim")
    cfg = served_prefill()
    qconfig.validate_config(cfg)
    assert qconfig.host_offload_enabled(cfg) and qconfig.pd_transfer_enabled(cfg)
    # The PD worker's self.config is its own child KVTransferConfig (upstream
    # MultiConnector builds one per child), so its role is the PD child's.
    child = KVTransferConfig(**_pd_child())
    assert child.is_kv_producer
    worker = NS(is_producer=child.is_kv_producer, _staging=None, tp_rank=0, vllm_config=cfg)
    engine, m = FakeEngine(), manifest(stage=0)
    register(worker, engine, m)
    hints = engine.calls[0][1].get("staging_blocks_per_pool")
    assert len(engine.calls) == 1
    assert engine.calls[0][1].get("staging_leases") == 16
    assert hints is not None and len(hints) == len(m.pools) and min(hints) > 0


def test_offload_registers_the_stage_region_storages(monkeypatch):
    _, _, stages, _ = pp8_pool(monkeypatch, num_blocks=4)
    native = next(s for s in stages if pool_regions(s) == 2)
    shape = (4, 3200, 1, 4, 256)
    regions = [torch.zeros(shape, dtype=torch.float8_e4m3fn) for _ in range(2)]
    caches, _ = _stage_caches(native, regions)
    (canonical,) = _register(native, caches)
    registered = sorted(t.tensor.untyped_storage().data_ptr() for t in canonical.tensors)
    assert registered == sorted(r.untyped_storage().data_ptr() for r in regions)
