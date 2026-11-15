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
"""`Qwen35Program.attach_draft` through the adapter: the `mtp_draft*` options,
the drafter seam, named declines, and the default (no draft) leaving the
drafter untouched. A tiny TP=1 template on CPU, as `test_qwen35_template`."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import vllm_torchtpu.megakernel.adapter.agreement as agreement_mod
import vllm_torchtpu.megakernel.adapter.driver as driver_mod
from vllm_torchtpu.megakernel.adapter import testing
from vllm_torchtpu.megakernel.core import Profile, register_profile
from vllm_torchtpu.megakernel.core import profile as profile_mod
from vllm_torchtpu.megakernel.core.runtime import (
    EligibilityResult,
    facts_from_pool_layout,
)
from vllm_torchtpu.models.qwen3_5 import megakernel as qwen_plugin
from vllm_torchtpu.models.qwen3_5.megakernel import adapter, draft
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35
from vllm_torchtpu.models.qwen3_5.megakernel.tests.tiny import (
    allocate_aliased_pools,
    cpu_mesh,
    make_synthetic_backbone,
    tiny_exec,
    tiny_geom,
)

_QWEN_HF = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
NUM_PAGES = 40


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)
    monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: None)


class _Drafter:
    """An MTP drafter with the draft megakernel seam; records its calls."""

    def __init__(self, layers: list[Any], norm: Any, accept: bool = True) -> None:
        self.draft_model = SimpleNamespace(
            model=SimpleNamespace(layers=layers, norm=norm)
        )
        self.accept = accept
        self.attached: list[Any] = []
        self.detached = 0
        self.checked = 0

    def attach_draft_megakernel(self, prog: Any) -> bool:
        self.attached.append(prog)
        return self.accept

    def detach_draft_megakernel(self) -> None:
        self.detached += 1

    def check_draft_megakernel_warm(self) -> None:
        self.checked += 1


class _Untouchable:
    """A drafter any attribute read of fails: the default must not look."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the drafter was read ({name})")


def _draft_layer() -> tuple[list[Any], Any]:
    """The MTP predictor's `(layers, norm)`: one tiny attention layer."""
    g = tiny_geom(L=1, att_every=1)
    ex = tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=NUM_PAGES)
    torch.manual_seed(1)
    bb = make_synthetic_backbone(g, ex)
    layer = bb.layers[0]
    layer.self_attn.attn.layer_name = "mtp.layers.0.self_attn.attn"
    pool = torch.zeros((NUM_PAGES, 2, 64, 4, 128), dtype=torch.uint8)
    layer.self_attn.attn.kv_cache = [pool.view(torch.float8_e4m3fn)]
    return [layer], bb.norm


@pytest.fixture
def draft_template(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Serve the template on a tiny CPU geometry (TP=1, one (1, 2) bucket) with
    an MTP drafter (`speculative_config.method == "mtp"`, K = 3)."""
    g = tiny_geom(L=4, att_every=4)
    ex = tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    real_derive = qwen35.Qwen35.derive_geometry

    def derive(cls: Any, *a: Any, **k: Any) -> Any:
        # The checkpoint's geometry is the tiny one; a geometry passed in (the
        # draft's one-layer stack) derives as itself.
        if a and isinstance(a[0], qwen35.Qwen35Geometry):
            return real_derive(*a, **k)
        return g

    monkeypatch.setattr(qwen35.Qwen35, "derive_geometry", classmethod(derive))
    monkeypatch.setattr(
        qwen_plugin, "SPEC", dataclasses.replace(qwen_plugin.SPEC, tp_sizes=(1,))
    )
    monkeypatch.setattr(
        adapter, "validate_hf_config", lambda *_a, **_k: EligibilityResult(True)
    )
    real_cls = adapter.Qwen35Program
    monkeypatch.setattr(
        adapter,
        "Qwen35Program",
        lambda model, **kw: real_cls(model, geom=g, base_exec=ex, **kw),
    )
    monkeypatch.setattr(profile_mod, "_REGISTRY", dict(profile_mod._REGISTRY))
    monkeypatch.setattr(
        profile_mod, "_MODEL_DEFAULTS", dict(profile_mod._MODEL_DEFAULTS)
    )
    register_profile(
        Profile(
            name="qwen35-test-tiny", model="qwen3_5", buckets=(2,), verify_buckets=()
        ),
        replace=True,
    )
    monkeypatch.setattr(adapter, "DEFAULT_PROFILE", "qwen35-test-tiny")

    def _facts(self: Any, view: Any, linear_attn: Any, pool: Any) -> Any:
        return facts_from_pool_layout(self.spec.cache_backend.layout, self.base_exec.R)

    monkeypatch.setattr(real_cls, "_framework_slot_facts", _facts)

    def make(
        options: dict[str, Any], drafter: Any, *, method: str = "mtp", k: int = 3
    ) -> tuple[Any, Any]:
        torch.manual_seed(0)
        backbone = make_synthetic_backbone(g, ex)
        allocate_aliased_pools(backbone, g, 96)
        qsl = np.zeros(16, dtype=np.int32)
        qsl[:3] = [0, 1, 2]
        runner = testing.fake_runner(
            model=SimpleNamespace(model=backbone),
            hf_config=_QWEN_HF,
            # With K = 3 the decode buckets need setting explicitly (verify
            # steps only, otherwise); the tiny profile has no verify buckets.
            megakernel={
                "plugin": "qwen35",
                "options": {"buckets": "2", "verify_buckets": "0", **options},
            },
            mesh=cpu_mesh(1),
            mamba_ssm_cache_dtype="bfloat16",
            token_paddings=(2, 4, 8),
            req_block_pairs=((4, 2),),
            speculative_tokens=k,
            ckpt_window=1,
            _attn_metadata_builder_ctx=SimpleNamespace(num_reqs=2),
            query_start_loc_np=qsl,
            _cached_comb_key=None,
            _combined_request_distribution_cpu=None,
            drafter=drafter,
        )
        spec = SimpleNamespace(num_speculative_tokens=k, method=method)
        runner.vllm_config.speculative_config = spec
        runner.speculative_config = spec
        runner.vllm_config.scheduler_config = SimpleNamespace(max_num_seqs=4)
        drv = driver_mod.ProgramDriver.maybe_create(runner)
        assert isinstance(drv, driver_mod.ProgramDriver), getattr(drv, "decline", drv)
        runner._megakernel = drv
        return drv, runner

    return make


_ON = {
    "mtp_draft": True,
    "mtp_draft_buckets": "4:8",
    "mtp_draft_verify_buckets": "8:16",
}


def test_parse_options_reads_the_draft_keys() -> None:
    cfg = adapter.parse_options({})
    assert cfg["mtp_draft"] is False
    assert cfg["mtp_draft_buckets"] == draft.DEFAULT_LOOP_BUCKETS
    assert cfg["mtp_draft_verify_buckets"] == draft.DEFAULT_VERIFY_BUCKETS
    cfg = adapter.parse_options(
        {
            "mtp_draft": "1",
            "mtp_draft_buckets": "4:8",
            "mtp_draft_verify_buckets": [12, 4],
        }
    )
    assert cfg["mtp_draft"] is True
    assert cfg["mtp_draft_buckets"] == (4, 8)
    assert cfg["mtp_draft_verify_buckets"] == (4, 12)
    assert adapter.parse_options({"mtp_draft": "off"})["mtp_draft"] is False
    with pytest.raises(ValueError, match="invalid mtp_draft"):
        adapter.parse_options({"mtp_draft": "maybe"})
    with pytest.raises(ValueError, match="multiple of 4"):
        adapter.parse_options({"mtp_draft_verify_buckets": "6"})
    assert adapter.DRAFT_OPTION_KEYS.issubset(adapter.OPTION_KEYS)
    assert adapter.DRAFT_OPTION_KEYS.issubset(qwen_plugin.SPEC.top_level_options)


def test_draft_off_by_default_leaves_the_drafter_untouched(draft_template: Any) -> None:
    drv, runner = draft_template({}, _Untouchable())
    assert drv.program.mtp_draft is False
    drv.bind_and_warmup(runner)
    assert runner._megakernel is drv and drv._bound
    assert drv.program.draft is None
    assert drv.program.attach_draft(_Untouchable()) is None
    drv.on_warmup_done(runner)  # no drafter check either


def test_attach_draft_hands_a_bound_program_to_the_drafter(draft_template: Any) -> None:
    drafter = _Drafter(*_draft_layer())
    drv, runner = draft_template(_ON, drafter)
    drv.bind_and_warmup(runner)
    assert runner._megakernel is drv and drv._bound
    (prog,) = drafter.attached
    assert isinstance(prog, draft.Qwen35DraftProgram)
    assert drv.program.draft is prog and prog.bound
    assert prog.layer_name == "mtp.layers.0.self_attn.attn"
    assert (prog.num_pages, prog.page_size) == (NUM_PAGES, 128)
    # max_num_seqs 4 over paddings (2, 4, 8): 4 loop rows, 16 step-0 rows
    assert prog.loop_buckets == (4,) and prog.verify_buckets == (8, 16)
    assert prog.geom.L == 1 and prog.geom.NGDN == 0
    assert prog.profile is drv.program.profile
    assert prog.base_exec.debug.modules["moe"].row_mask is True
    drv.on_warmup_done(runner)
    assert drafter.checked == 1 and drafter.detached == 0
    drv.program.detach_draft()
    assert drafter.detached == 1 and drv.program.draft is None


def _two_layers() -> _Drafter:
    layers, norm = _draft_layer()
    return _Drafter(layers * 2, norm)


def _bf16_experts() -> _Drafter:
    layers, norm = _draft_layer()
    ex = layers[0].mlp.experts
    ex.w13_weight = torch.nn.Parameter(
        torch.zeros(ex.w13_weight.shape, dtype=torch.bfloat16), requires_grad=False
    )
    return _Drafter(layers, norm)


def _flagged() -> _Drafter:
    d = _Drafter(*_draft_layer())
    d._draft_kv_shared = True
    return d


def _no_seam() -> Any:
    d = _Drafter(*_draft_layer())
    return SimpleNamespace(draft_model=d.draft_model)


@pytest.mark.parametrize(
    "make_drafter, k, why",
    [
        (_no_seam, 3, "no attach_draft_megakernel seam"),
        (lambda: _Drafter(*_draft_layer(), accept=False), 3, "built no draft program"),
        (_flagged, 3, "drafter._draft_kv_shared is set"),
        (_two_layers, 3, "one-layer MTP predictor"),
        (_bf16_experts, 3, "are not FP8"),
        (lambda: _Drafter(*_draft_layer()), 2, "num_speculative_tokens=2"),
    ],
)
def test_attach_draft_declines_the_megakernel_by_name(
    draft_template: Any, make_drafter: Any, k: int, why: str
) -> None:
    drafter = make_drafter()
    drv, runner = draft_template(_ON, drafter, k=k)
    drv.bind_and_warmup(runner)
    declined = runner._megakernel
    assert isinstance(declined, driver_mod.DeclinedDriver)
    assert declined.decline.code == "draft_declined"
    assert why in declined.decline.message
    assert drv.program.draft is None and not drv._bound


@pytest.mark.parametrize("method, drafter", [("eagle3", "drafter"), ("mtp", None)])
def test_no_mtp_drafter_serves_without_a_draft(
    draft_template: Any, method: str, drafter: Any
) -> None:
    d = _Drafter(*_draft_layer()) if drafter else None
    drv, runner = draft_template(_ON, d, method=method)
    drv.bind_and_warmup(runner)
    assert runner._megakernel is drv and drv._bound
    assert drv.program.draft is None
    assert d is None or d.attached == []
