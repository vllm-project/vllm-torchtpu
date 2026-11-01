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
"""CPU end-to-end test of this megakernel folder (fake runner, no TPU).

The runtime loads `SPEC` by this folder's package name, runs its startup
checks on a fake runner, and `create` builds a program for a tiny geometry;
the program loads, binds the runner's raw pools, warms up and serves a decode
step through the CPU-interpret kernel (the runtime test kit's `run_cpu_e2e`).
Only relative imports of the folder, so a copy under another package name runs
it unchanged.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel.runtime import (
    EligibilityResult,
    facts_from_pool_layout,
)
from vllm_torchtpu.megakernel_runtime.testkit import check_spec, run_cpu_e2e

from .. import adapter, model, profiles
from .tiny import (
    allocate_aliased_pools,
    cpu_mesh,
    make_synthetic_backbone,
    tiny_exec,
    tiny_geom,
)

FOLDER = __package__.rpartition(".")[0]


@pytest.fixture
def tiny_runner(mk_fake_runner: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A fake TP=1 runner over a tiny synthetic backbone, with this folder as
    the megakernel plugin (the exact-numerics profile, one `(1, 2)` bucket)."""
    g = tiny_geom(L=4, att_every=4)
    ex = tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    folder = sys.modules[FOLDER]
    # The tiny geometry stands in for the checkpoint's (TP=1 on CPU).
    monkeypatch.setattr(folder, "SPEC", dataclasses.replace(folder.SPEC, tp_sizes=(1,)))
    monkeypatch.setattr(
        model.Qwen35, "derive_geometry", classmethod(lambda cls, *a, **k: g)
    )
    monkeypatch.setattr(
        adapter, "validate_hf_config", lambda *_a, **_k: EligibilityResult(True)
    )
    real_cls = adapter.Qwen35Program
    monkeypatch.setattr(
        adapter,
        "Qwen35Program",
        lambda m, **kw: real_cls(m, geom=g, base_exec=ex, **kw),
    )

    # The fake runner has no vLLM GDN state pools to read slot facts from.
    def facts(self: Any, view: Any, linear_attn: Any, pool: Any) -> Any:
        return facts_from_pool_layout(self.spec.cache_backend.layout, self.base_exec.R)

    monkeypatch.setattr(real_cls, "_framework_slot_facts", facts)

    torch.manual_seed(0)
    backbone = make_synthetic_backbone(g, ex)
    allocate_aliased_pools(backbone, g, 96)
    qsl = np.zeros(16, dtype=np.int32)
    qsl[:3] = [0, 1, 2]
    options = {
        "profile": profiles.baseline.PROFILE.name,
        "buckets": "2",
        "verify_buckets": "0",
    }
    return mk_fake_runner(
        model=SimpleNamespace(model=backbone),
        hf_config=SimpleNamespace(architectures=list(model.Qwen35.hf_architectures)),
        megakernel={"plugin": FOLDER, "options": options, "stats_period_s": 3600},
        mesh=cpu_mesh(1),
        mamba_ssm_cache_dtype="bfloat16",
        token_paddings=(2, 4, 8),
        req_block_pairs=((4, 2),),
        max_pages_per_seq=2,
        _attn_metadata_builder_ctx=SimpleNamespace(num_reqs=2),
        query_start_loc_np=qsl,
        _cached_comb_key=None,
        _combined_request_distribution_cpu=None,
    )


def _metadata(prog: Any) -> dict[str, Any]:
    """Attention metadata of a two-request decode step."""
    md: dict[str, Any] = {
        prog.attn_layer_name: SimpleNamespace(
            block_tables=torch.tensor([10, 11, 12, 13, 0, 0, 0, 0], dtype=torch.int32),
            seq_lens=torch.tensor([6, 10, 0, 0], dtype=torch.int32),
        )
    }
    for sub, name in enumerate(prog.mamba_layer_names):
        md[name] = SimpleNamespace(
            mamba_state_indices=torch.tensor(
                [1 + 2 * sub, 2 + 2 * sub, 0, 0], dtype=torch.int32
            ),
            mamba_ckpt_indices=None,
            mamba_slot_read_offsets=None,
        )
    return md


def test_spec_loads_without_registering_anything() -> None:
    spec = check_spec(FOLDER)
    assert spec is importlib.import_module(FOLDER).SPEC
    assert spec.create == ".adapter:create"
    assert profiles.DEFAULT_PROFILE.name in profiles.PROFILES
    assert model.register() is model.Qwen35
    assert model.register() is model.Qwen35  # idempotent


def test_serves_a_decode_step_on_cpu(tiny_runner: Any) -> None:
    ids = torch.tensor([5, 12], dtype=torch.int32)
    pos = torch.tensor([5, 9], dtype=torch.int32)
    res = run_cpu_e2e(None, tiny_runner, [(ids, pos)], metadata=_metadata)
    prog = res.program
    assert prog.profile.name == profiles.baseline.PROFILE.name
    assert prog.profile.numerics == "exact"
    assert sorted(res.driver._steps) == [(1, 2)]
    (out,) = res.outputs
    assert res.hits == [True] and out.shape == (2, 256)
    assert bool(torch.isfinite(out.float()).all())
    assert res.step_mix == {"mk_t1_b2": 1}
