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
"""The Qwen3.5 rule for the batched RPA tuned tiles and decode lanes: on for
the legs of the validation entries (Qwen3.5-397B-A17B on TPU7x, their KV role
and parallel sizes, the seq-along-lane cache they ran), off for every other
engine: main's nightly configurations, other PD layouts, other checkpoints,
chips and models."""

import json
import pathlib
from types import SimpleNamespace

import pytest
import torch
import yaml
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.layers.adapter import attention
from vllm_torchtpu.models import qwen3_5
from vllm_torchtpu.models.qwen3_5 import hooks

QWEN3_5 = "Qwen3_5MoeForConditionalGeneration"
OTHERS = [
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen2ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "LlamaForCausalLM",
]
# Main's CI checkpoint, Qwen3.5-35B-A3B: its text config differs from the
# measured one in all three keys the rule reads.
CI_35B_SHAPE = {"num_experts": 256, "hidden_size": 2048, "num_hidden_layers": 40}
# HND, the seq-along-lane layout every leg of the records ran
# (VLLM_KV_CACHE_LAYOUT; the entries list only this package's variables).
SEQ = KVCacheLayout.LBHNC
HEAD = KVCacheLayout.LBNHC
VALIDATION = pathlib.Path(qwen3_5.__file__).parent / "validation"
POINTS = sorted(VALIDATION.glob("p[0-9][0-9]-*.yaml"))
GATES = sorted(VALIDATION.glob("gate-*.yaml"))


@pytest.fixture(autouse=True)
def _tpu7x(monkeypatch):
    monkeypatch.setattr(hooks, "_chip_family", lambda: "7x")


def _engine(
    arch=QWEN3_5,
    shape=None,
    role="kv_consumer",
    tp=2,
    pcp=1,
    dcp=1,
    layout=SEQ,
    max_model_len=262144,
):
    text = SimpleNamespace(**(hooks._CALIBRATED_SHAPE if shape is None else shape))
    return SimpleNamespace(
        speculative_config=None,
        model_config=SimpleNamespace(
            architectures=[arch], max_model_len=max_model_len, hf_text_config=text
        ),
        kv_transfer_config=None if role is None else SimpleNamespace(kv_role=role),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
        ),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: layout),
        additional_config={},
    )


def _flag(args, name, default=None):
    for i, arg in enumerate(args):
        if arg == name:
            return args[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return default


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _leg_geometry(leg):
    """The leg's KV role and tensor, prefill context and decode context
    parallel sizes, from its serve arguments."""
    args = [str(a) for a in leg["serve_args"]]
    role = json.loads(_flag(args, "--kv-transfer-config"))["kv_role"]
    sizes = [
        int(_flag(args, f"--{name}-parallel-size", 1))
        for name in ("tensor", "prefill-context", "decode-context")
    ]
    return dict(role=role, tp=sizes[0], pcp=sizes[1], dcp=sizes[2])


RECORD_LEGS = [
    pytest.param(_leg_geometry(leg), id=f"{path.stem}-{name}")
    for path in POINTS
    for name, leg in _load(path)["legs"].items()
]


def test_the_records_have_legs():
    assert len(POINTS) == 5
    assert len(RECORD_LEGS) == 10


@pytest.mark.parametrize("geometry", RECORD_LEGS)
def test_every_leg_of_the_records_takes_the_tuned_tiles(geometry):
    assert attention.qwen3_5_tuned_tiles(_engine(**geometry)) is True


@pytest.mark.parametrize("path", GATES, ids=lambda p: p.stem)
def test_the_gates_run_their_base_points_legs(path):
    """A gate's legs are its base point's, with only the speculative config
    replaced, so they take the tiles of the base point's legs."""
    gate = _load(path)
    base = [p for p in POINTS if _load(p)["point"] == gate["base_point"]]
    assert len(base) == 1
    base_legs = _load(base[0])["legs"]
    for name, leg in gate["legs"].items():
        assert leg["from"] == f"the {name} leg of base_point"
        assert set(leg.get("serve_args_override", {})) <= {"--speculative-config"}
        geometry = _leg_geometry(base_legs[name])
        assert attention.qwen3_5_tuned_tiles(_engine(**geometry)) is True


@pytest.mark.parametrize("geometry", RECORD_LEGS)
def test_a_record_leg_on_the_head_along_sublane_cache_keeps_mains_tiles(geometry):
    assert attention.qwen3_5_tuned_tiles(_engine(**geometry, layout=HEAD)) is False


@pytest.mark.parametrize("tp", [1, 2, 8])
@pytest.mark.parametrize("layout", [HEAD, SEQ])
def test_mains_nightly_configs_keep_mains_tiles(tp, layout):
    # No KV transfer: main's Qwen3.5-397B TP1 x DP8, TP2 x DP4 and TP8 nightlies.
    engine = _engine(role=None, tp=tp, layout=layout, max_model_len=10240)
    assert attention.qwen3_5_tuned_tiles(engine) is False


@pytest.mark.parametrize(
    "role, tp, pcp, dcp",
    [
        ("kv_consumer", 1, 1, 1),  # a TP1 decode leg (two KV heads per device)
        ("kv_consumer", 4, 1, 1),
        ("kv_consumer", 8, 1, 1),  # TP8 decode without DCP4
        ("kv_consumer", 8, 1, 2),
        ("kv_consumer", 2, 1, 2),
        ("kv_producer", 2, 1, 1),
        ("kv_producer", 8, 1, 1),
        ("kv_producer", 1, 8, 1),  # a PCP8 prefill leg
        ("kv_producer", 1, 1, 4),
        ("kv_both", 1, 1, 1),
        ("kv_both", 2, 1, 1),
    ],
)
def test_pd_layouts_the_records_did_not_run_keep_mains_tiles(role, tp, pcp, dcp):
    engine = _engine(role=role, tp=tp, pcp=pcp, dcp=dcp)
    assert attention.qwen3_5_tuned_tiles(engine) is False


@pytest.mark.parametrize("geometry", RECORD_LEGS)
def test_main_ci_qwen3_5_checkpoint_keeps_mains_tiles(geometry):
    """Qwen3.5-35B-A3B, which the integration lane serves, keeps main's tiles
    even at the records' layouts: at its TP1 decode shape the tuned decode
    tiles (8 lanes of 2048 keys) need more VMEM than TPU7x has."""
    engine = _engine(shape=CI_35B_SHAPE, **geometry)
    assert attention.qwen3_5_tuned_tiles(engine) is False


@pytest.mark.parametrize("chip", ["v6e", "v5e", None])
def test_the_measured_checkpoint_on_another_chip_keeps_mains_tiles(monkeypatch, chip):
    monkeypatch.setattr(hooks, "_chip_family", lambda: chip)
    for param in RECORD_LEGS:
        assert attention.qwen3_5_tuned_tiles(_engine(**param.values[0])) is False


def test_other_models_and_no_config_keep_mains_tiles():
    for arch in OTHERS:
        assert attention.qwen3_5_tuned_tiles(_engine(arch=arch)) is False, arch
    assert attention.qwen3_5_tuned_tiles(None) is False
    assert attention.qwen3_5_tuned_tiles(SimpleNamespace(model_config=None)) is False


def _batched_layer(monkeypatch, engine, num_kv_heads=1):
    """A batched RPA layer with `num_kv_heads` fp8 KV heads of 256 per device
    (one: a DP4 x TP2 decode device of Qwen3.5-397B) built under `engine`."""

    def base_init(self, *args, **kwargs):
        self.head_size = 256
        self.num_kv_heads = num_kv_heads
        self.kv_cache_quantized_dtype = torch.float8_e4m3fn

    monkeypatch.setattr(attention.PallasAttentionBackendImpl, "__init__", base_init)
    monkeypatch.setattr(attention, "get_current_vllm_config_or_none", lambda: engine)
    return attention.PallasBatchedRPAAttentionBackendImpl()


def test_a_record_decode_layer_takes_the_tuned_tiles_and_lanes(monkeypatch):
    layer = _batched_layer(monkeypatch, _engine())
    assert layer.tuned_tiles is True
    assert layer.skip_empty_mixed is True
    assert layer.decode_lanes == 4


def test_a_tp1_x_dp8_nightly_layer_keeps_mains_tiles_and_lanes(monkeypatch):
    engine = _engine(role=None, tp=1, layout=HEAD, max_model_len=10240)
    layer = _batched_layer(monkeypatch, engine, num_kv_heads=2)
    assert layer.tuned_tiles is False
    assert layer.skip_empty_mixed is False
    assert layer.decode_lanes is None


@pytest.mark.parametrize("arch", OTHERS)
def test_other_models_keep_mains_tiles_and_lanes(monkeypatch, arch):
    layer = _batched_layer(monkeypatch, _engine(arch=arch))
    assert layer.tuned_tiles is False
    assert layer.skip_empty_mixed is False
    assert layer.decode_lanes is None


def test_main_ci_qwen3_5_layers_keep_mains_tiles_and_lanes(monkeypatch):
    engine = _engine(shape=CI_35B_SHAPE, tp=1, layout=HEAD)
    layer = _batched_layer(monkeypatch, engine, num_kv_heads=2)
    assert layer.tuned_tiles is False
    assert layer.skip_empty_mixed is False
    assert layer.decode_lanes is None
