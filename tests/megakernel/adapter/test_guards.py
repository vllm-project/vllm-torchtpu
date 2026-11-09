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
"""Startup guards: the spec checks (chips, TP sizes, features, parallelism,
server settings, collective domain), KV connectors, multimodal inputs and
the decoder backbone lookup."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_torchtpu.megakernel.adapter.guards as guards_mod
from vllm_torchtpu.distributed.kv_transfer import connector_names
from vllm_torchtpu.megakernel.adapter import testing
from vllm_torchtpu.megakernel.adapter.backbone import find_backbone
from vllm_torchtpu.megakernel.adapter.plugin import RunnerView
from vllm_torchtpu.megakernel.core.runtime.contract import TpuMonoSpec

_QWEN_CAPS = frozenset({"spec_decode", "mrope", "prefix_caching", "kv_offload"})


def _spec(**kw: Any) -> TpuMonoSpec:
    return TpuMonoSpec("t", "1", create="m:c", **kw)


def _check(runner: Any, **kw: Any) -> Any:
    return guards_mod.check_spec(
        runner, _spec(**kw), view=RunnerView(runner, tp_rank=0)
    )


def _kv(name: str, members: list[str] | None = None) -> Any:
    extra = {"connectors": [{"kv_connector": m} for m in members or []]}
    return SimpleNamespace(kv_connector=name, kv_connector_extra_config=extra)


@pytest.mark.parametrize(
    "kv, code",
    [
        (None, None),
        (_kv("OffloadingConnector"), None),
        (_kv("TPURaidenOffloadingConnector"), None),
        (
            _kv(
                "MultiConnector",
                ["OffloadingConnector", "TPURaidenOffloadingConnector"],
            ),
            None,
        ),
        (
            _kv("MultiConnector", ["OffloadingConnector", "NixlConnector"]),
            "kv_connector_unsupported:NixlConnector",
        ),
        (_kv("TPURaidenConnector"), "kv_connector_unsupported:TPURaidenConnector"),
        (_kv("DecodeBenchConnector"), "kv_connector_unsupported:DecodeBenchConnector"),
        (_kv("MultiConnector"), "kv_connector_unsupported:MultiConnector"),
        # The TPU multi-connector is expanded like vLLM's.
        (
            _kv(
                "TPUMultiConnector",
                ["TPURaidenOffloadingConnector", "OffloadingConnector"],
            ),
            None,
        ),
        (
            _kv(
                "TPUMultiConnector",
                ["TPURaidenOffloadingConnector", "TPURaidenConnector"],
            ),
            "kv_connector_unsupported:TPURaidenConnector",
        ),
    ],
)
def test_kv_connector_guard(kv: Any, code: str | None) -> None:
    runner = testing.fake_runner(kv_transfer_config=kv)
    res = guards_mod.check_runner_features(runner, _QWEN_CAPS)
    assert (res.code if res else None) == code
    if code is None and kv is not None:
        res = guards_mod.check_runner_features(runner, frozenset())
        assert res is not None and res.code == "kv_offload_unsupported"
    if code is not None:
        caps = _QWEN_CAPS | {"kv_transfer"}
        assert guards_mod.check_runner_features(runner, caps) is None


@pytest.mark.parametrize("multi", ["MultiConnector", "TPUMultiConnector"])
def test_mixed_kv_connector_needs_offload_and_transfer(multi: str) -> None:
    """Offloading and transfer are checked separately: a multi-connector with
    an offloading child and a transfer child needs both features."""
    kv = _kv(multi, ["TPURaidenOffloadingConnector", "NixlConnector"])
    assert guards_mod.kv_connector_names(kv) == (
        "TPURaidenOffloadingConnector",
        "NixlConnector",
    )
    runner = testing.fake_runner(kv_transfer_config=kv)
    res = guards_mod.check_runner_features(runner, frozenset({"kv_transfer"}))
    assert res is not None and res.code == "kv_offload_unsupported"
    res = guards_mod.check_runner_features(runner, frozenset({"kv_offload"}))
    assert res is not None and res.code == "kv_connector_unsupported:NixlConnector"
    caps = frozenset({"kv_offload", "kv_transfer"})
    assert guards_mod.check_runner_features(runner, caps) is None


def test_kv_connector_names_share_one_helper() -> None:
    """The guard and the pipeline-parallel check expand connectors through one
    helper; the pipeline check expands only the TPU multi-connector."""
    vllm_multi = _kv("MultiConnector", ["OffloadingConnector"])
    tpu_multi = _kv("TPUMultiConnector", ["TPURaidenConnector"])
    tpu_only = ("TPUMultiConnector",)
    assert connector_names.kv_connector_names(vllm_multi, tpu_only) == [
        "MultiConnector"
    ]
    assert connector_names.kv_connector_names(tpu_multi, tpu_only) == [
        "TPURaidenConnector"
    ]
    assert guards_mod.kv_connector_names(vllm_multi) == ("OffloadingConnector",)
    assert guards_mod.kv_connector_names(tpu_multi) == ("TPURaidenConnector",)
    assert guards_mod.kv_connector_names(None) == ()


def _mm_cfg(lm_only: bool = False, limits: dict[str, Any] | None = None) -> Any:
    return SimpleNamespace(language_model_only=lm_only, limit_per_prompt=limits or {})


@pytest.mark.parametrize(
    "supports, mm, possible",
    [
        (False, None, False),
        (True, None, True),
        (True, _mm_cfg(), True),
        (True, _mm_cfg(lm_only=True), False),
        (True, _mm_cfg(limits={"image": 0, "video": 0}), False),
        (
            True,
            _mm_cfg(limits={"image": {"count": 0}, "video": SimpleNamespace(count=0)}),
            False,
        ),
        (True, _mm_cfg(limits={"image": 1, "video": 0}), True),
    ],
)
def test_multimodal_inputs_possible(supports: bool, mm: Any, possible: bool) -> None:
    runner = testing.fake_runner(supports_mm_inputs=supports, multimodal_config=mm)
    assert guards_mod._multimodal_inputs_possible(runner) is possible


def test_kv_offload_on_a_text_only_server_is_not_a_multimodal_decline() -> None:
    """vLLM 0.32 turns on `enable_mm_embeds` for a `kv_both` offload
    connector, so a language-model-only server reports
    `supports_mm_inputs=True`; the megakernel still serves it. A server that
    really accepts multimodal inputs still declines."""
    offload = _kv("TPURaidenOffloadingConnector")
    text_only = _mm_cfg(lm_only=True, limits={"image": 0, "video": 0})
    runner = testing.fake_runner(
        kv_transfer_config=offload, supports_mm_inputs=True, multimodal_config=text_only
    )
    assert guards_mod.check_runner_features(runner, _QWEN_CAPS) is None
    runner = testing.fake_runner(
        kv_transfer_config=offload,
        supports_mm_inputs=True,
        multimodal_config=_mm_cfg(limits={"image": 1, "video": 0}),
    )
    res = guards_mod.check_runner_features(runner, _QWEN_CAPS)
    assert res is not None and res.code == "multimodal_inputs"


@pytest.mark.parametrize(
    "kw, code, cap",
    [
        ({"speculative_tokens": 3}, "spec_decode_unsupported", "spec_decode"),
        ({"uses_mrope": True}, "mrope_unsupported", "mrope"),
        (
            {"enable_prefix_caching": True},
            "prefix_caching_unsupported",
            "prefix_caching",
        ),
        ({"enable_eplb": True}, "eplb_dynamic_unsupported", "eplb_dynamic"),
        ({"supports_mm_inputs": True}, "multimodal_inputs", "multimodal"),
        ({"lora_config": object()}, "lora_enabled", "lora"),
    ],
)
def test_undeclared_feature_is_a_named_decline(
    kw: dict[str, Any], code: str, cap: str
) -> None:
    runner = testing.fake_runner(**kw)
    res = _check(runner)
    assert res is not None and res.code == code and res.message
    assert _check(runner, features={cap}) is None


@pytest.mark.parametrize(
    "kw, par, kind",
    [
        ({"_pp_is_last": False}, {}, "pp"),
        ({}, {"decode_context_parallel_size": 2}, "dcp"),
        ({"_pcp_enabled": True}, {}, "pcp"),
        ({}, {"prefill_context_parallel_size": 2}, "pcp"),
        ({"_dp_lockstep_enabled": lambda: True}, {}, "dp_lockstep"),
        ({"enable_expert_parallel": True}, {}, "ep_in_tp"),
    ],
)
def test_undeclared_parallelism_is_a_named_decline(
    kw: dict[str, Any], par: dict[str, Any], kind: str
) -> None:
    runner = testing.fake_runner(**kw)
    for k, v in par.items():
        setattr(runner.parallel_config, k, v)
    assert guards_mod.detect_parallelism(runner, 1) == [kind]
    res = _check(runner)
    assert res is not None and res.code == f"parallelism_unsupported:{kind}"
    assert res.message.startswith("Megakernel ") and kind in res.message
    assert _check(runner, parallelism={"tp", kind}) is None


def test_dp_and_wide_ep_parallelism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guards_mod, "_dp_size", lambda _p_cfg: 2)
    runner = testing.fake_runner()
    res = _check(runner, features=_QWEN_CAPS)
    assert res is not None and res.code == "parallelism_unsupported:dp"
    assert "data_parallel_size" in res.message
    monkeypatch.setattr(guards_mod, "_dp_size", lambda _p_cfg: 1)
    # An EP group wider than TP is its own kind (EP inside TP is not); the
    # EP group counts only with expert parallelism on.
    monkeypatch.setattr(guards_mod, "_ep_group_size", lambda: 2)
    assert guards_mod.detect_parallelism(testing.fake_runner(), 1) == []
    runner = testing.fake_runner(enable_expert_parallel=True)
    assert guards_mod.detect_parallelism(runner, 1) == ["ep_wider_than_tp"]
    assert guards_mod.detect_parallelism(runner, 2) == ["ep_in_tp"]
    res = _check(runner, parallelism={"tp", "ep_in_tp"})
    assert res is not None and res.code == "parallelism_unsupported:ep_wider_than_tp"


def test_dp_and_ep_sizes_come_from_the_runner_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DP size is `utils.get_dp_size`'s and the EP group is
    `ep_mesh.get_ep_group`'s (both need vLLM)."""
    pytest.importorskip("vllm")
    from vllm_torchtpu.distributed import ep_mesh

    p_cfg = SimpleNamespace(data_parallel_size=1)
    monkeypatch.setenv("TORCH_TPU_DP_SIZE", "4")
    assert guards_mod._dp_size(p_cfg) == 4
    monkeypatch.delenv("TORCH_TPU_DP_SIZE")
    p_cfg.data_parallel_size = 2
    assert guards_mod._dp_size(p_cfg) == 2
    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    assert guards_mod._ep_group_size() == 8
    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: None)
    assert guards_mod._ep_group_size() == 1


def test_dp_and_ep_without_vllm_are_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without vLLM (a CPU test of a plugin) there are no DP engines and no EP
    group; a broken import of anything else still raises."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("vllm_torchtpu.utils", "vllm_torchtpu.distributed.ep_mesh"):
            raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    missing = "vllm.distributed"
    p_cfg = SimpleNamespace(data_parallel_size=2)
    assert guards_mod._dp_size(p_cfg) == 1 and guards_mod._ep_group_size() == 1
    missing = "jax"
    with pytest.raises(ModuleNotFoundError, match="jax"):
        guards_mod._dp_size(p_cfg)
    with pytest.raises(ModuleNotFoundError, match="jax"):
        guards_mod._ep_group_size()


def test_sleep_mode_is_a_named_decline() -> None:
    runner = testing.fake_runner()
    assert _check(runner, features=_QWEN_CAPS) is None
    runner.vllm_config.model_config.enable_sleep_mode = True
    res = _check(runner, features=_QWEN_CAPS)
    assert res is not None and res.code == "sleep_mode_unsupported"


def test_chip_and_tp_size_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = testing.fake_runner()
    # A CPU (interpret mode) device isn't chip-checked.
    assert _check(runner, chips=("TPU7x",)) is None
    monkeypatch.setattr(RunnerView, "device_kind", property(lambda _s: "TPU v5 lite"))
    res = _check(runner, chips=("TPU7x",))
    assert res is not None and res.code == "unsupported_tpu_chip"
    assert "TPU v5 lite" in res.message
    assert _check(runner, chips=("TPU v5",)) is None
    assert _check(runner, chips=("tpu V5 LITE",)) is None
    assert _check(runner) is None
    res = _check(runner, tp_sizes=(8,))
    assert res is not None and res.code == "unsupported_tp_size"
    assert "(8,)" in res.message
    assert _check(runner, tp_sizes=(1, 8)) is None
    # Chip before TP size before parallelism.
    runner.parallel_config.data_parallel_size = 2
    assert _check(runner, chips=("TPU7x",), tp_sizes=(8,)).code == (
        "unsupported_tpu_chip"
    )
    assert _check(runner, tp_sizes=(8,)).code == "unsupported_tp_size"


@pytest.mark.parametrize(
    "requires, kw, code",
    [
        ({"mamba_ssm_cache_dtype": "bfloat16"}, {}, "mamba_ssm_cache_dtype"),
        (
            {"mamba_ssm_cache_dtype": "bfloat16"},
            {"mamba_ssm_cache_dtype": "bfloat16"},
            None,
        ),
        (
            {"mamba_ssm_cache_dtype": "bfloat16"},
            {"mamba_ssm_cache_dtype": torch.bfloat16},
            None,
        ),
        ({"kv_cache_dtype": "auto"}, {}, None),
        ({"kv_cache_dtype": "fp8"}, {}, "kv_cache_dtype"),
        ({"mamba_cache_mode": "none"}, {}, None),
        ({"mamba_cache_mode": "align"}, {}, "mamba_cache_mode"),
        ({"enable_prefix_caching": "False"}, {}, None),
        ({"enable_prefix_caching": "true"}, {}, "enable_prefix_caching"),
    ],
)
def test_requires(requires: dict[str, str], kw: dict[str, Any], code: Any) -> None:
    runner = testing.fake_runner(**kw)
    res = _check(runner, requires=requires)
    if code is None:
        assert res is None
    else:
        assert res is not None and res.code == f"requires_unmet:{code}"
        assert f"--{code.replace('_', '-')}" in res.message


def test_features_before_requires() -> None:
    runner = testing.fake_runner(speculative_tokens=3)
    res = _check(runner, requires={"kv_cache_dtype": "fp8"})
    assert res is not None and res.code == "spec_decode_unsupported"


@pytest.mark.parametrize("domain", ["tp_x_dcp", "kernel_mesh"])
def test_unsupported_collective_domain_is_a_named_decline(domain: str) -> None:
    res = _check(testing.fake_runner(), collective_domain=domain)
    assert res is not None and res.code == f"parallelism_unsupported:{domain}"
    assert _check(testing.fake_runner(), collective_domain="tp") is None


def test_find_backbone_and_aux_hidden_state_layers() -> None:
    backbone = SimpleNamespace(layers=[], norm=None)
    for model in (
        SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(model=backbone))
        ),
        SimpleNamespace(language_model=SimpleNamespace(model=backbone)),
        SimpleNamespace(model=SimpleNamespace(model=backbone)),
        SimpleNamespace(model=backbone),
        backbone,
    ):
        assert find_backbone(model) is backbone
    with pytest.raises(ValueError, match="Cannot locate decoder backbone"):
        find_backbone(SimpleNamespace(model=SimpleNamespace(layers=[])))
    # A backbone that captures auxiliary hidden states (EAGLE-3) declines
    # unless the spec serves them.
    aux = SimpleNamespace(layers=[], norm=None, aux_hidden_state_layers=(1, 2))
    runner = testing.fake_runner(model=SimpleNamespace(model=aux))
    res = guards_mod.check_runner_features(runner, _QWEN_CAPS)
    assert res is not None and res.code == "aux_hidden_states"
    assert (
        guards_mod.check_runner_features(runner, _QWEN_CAPS | {"aux_hidden_states"})
        is None
    )
