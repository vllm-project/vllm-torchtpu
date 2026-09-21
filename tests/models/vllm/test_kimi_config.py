# SPDX-License-Identifier: Apache-2.0
"""Kimi KDA layer counts, supplied through vLLM's per-architecture config hook.

The hook writes ``layers_block_type`` on the HF text config, so the counts come
from upstream's own ``ModelConfig.get_num_layers_by_block_type``. These tests
call that upstream function unbound on a stand-in config: no checkpoint, no
model, and no TPU runtime. The module under test is loaded by path because
importing ``vllm_torchtpu.layers`` pulls in the TPU attention kernels.
"""

import importlib.util
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm.config import ModelConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP

pytestmark = pytest.mark.cpu_test

_SRC = Path(__file__).resolve().parents[3] / "src/vllm_torchtpu"
_MODELS_PATH = _SRC / "models/vllm/__init__.py"
_REGISTRY_PATH = _SRC / "patch_registry.py"
_KIMI_ARCHITECTURES = ("KimiK3ForConditionalGeneration",
                       "KimiLinearForCausalLM")
_NUM_LAYERS = 6
_BLOCK_TYPES = [
    "linear_attention", "attention", "linear_attention", "attention",
    "attention", "linear_attention"
]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture
def config_hook(monkeypatch):
    module = _load("_test_tpu_models", _MODELS_PATH)
    # register_models() writes to two process-wide registries; hand both back
    # exactly as they were, including an entry that did not exist before.
    monkeypatch.setattr(ModelRegistry, "models", ModelRegistry.models.copy())
    previous = {
        name: MODELS_CONFIG_MAP.get(name)
        for name in _KIMI_ARCHITECTURES
    }
    module.register_models()
    try:
        yield module._TPUKimiConfig
    finally:
        for name, handler in previous.items():
            if handler is None:
                MODELS_CONFIG_MAP.pop(name, None)
            else:
                MODELS_CONFIG_MAP[name] = handler


def _config(kda_layers=(1, 3, 6), **text_fields):
    """A text-only stand-in: ``hf_config`` is its own text config."""
    text = SimpleNamespace(num_hidden_layers=_NUM_LAYERS,
                           linear_attn_config={"kda_layers": list(kda_layers)},
                           **text_fields)
    return SimpleNamespace(
        hf_config=text,
        hf_text_config=text,
        get_total_num_hidden_layers=lambda: text.num_hidden_layers)


def _count(config, start, end, block_type):
    """Upstream's counter for one pipeline stage, on the stand-in config."""
    config.is_hybrid = True
    config.has_noops = False
    config.is_attention_free = False
    config.model_arch_config = SimpleNamespace(text_model_type="kimi_linear")
    config.get_layers_start_end_indices = lambda _: (start, end)
    return ModelConfig.get_num_layers_by_block_type(config, None, block_type)


def _removed_patch_attention_count(kda_layers, start, end):
    """The arithmetic of the removed ``_patch_vllm_kimi_kda_layer_counts``."""
    kda_layers = set(kda_layers)
    return sum((idx + 1) not in kda_layers for idx in range(start, end))


def test_both_kimi_architectures_use_the_hook(config_hook):
    for architecture in _KIMI_ARCHITECTURES:
        assert MODELS_CONFIG_MAP[architecture] is config_hook


def test_hook_is_reached_through_model_config_lookup(config_hook):
    """Drive upstream's own dispatch, keyed by architecture name."""
    for architecture in _KIMI_ARCHITECTURES:
        config = _config()
        config.architecture = architecture
        ModelConfig._try_verify_and_update_model_config(config)
        assert _count(config, 0, _NUM_LAYERS, "attention") == 3


def test_upstream_raises_without_the_hook():
    """The gap being filled: a bare Kimi config has nothing to count."""
    with pytest.raises(ValueError, match="cannot determine the num of"):
        _count(_config(), 0, _NUM_LAYERS, "attention")


@pytest.mark.parametrize("start,end,attention,linear", [(0, 6, 3, 3),
                                                        (0, 3, 1, 2),
                                                        (3, 6, 2, 1),
                                                        (0, 0, 0, 0)])
def test_upstream_counts_kda_layers(config_hook, start, end, attention,
                                    linear):
    config = _config()
    config_hook.verify_and_update_model_config(config)
    assert _count(config, start, end, "attention") == attention
    assert _count(config, start, end, "linear_attention") == linear


_KDA_LAYER_SETS = {
    "empty": (),
    "all": tuple(range(1, _NUM_LAYERS + 1)),
    "non_contiguous": (1, 3, 6),
    "first_only": (1, ),
    "last_only": (_NUM_LAYERS, ),
}
# Every contiguous pipeline-stage slice of the model, empty ones included.
_STAGE_SLICES = [
    (start, end)
    for start, end in itertools.product(range(_NUM_LAYERS + 1), repeat=2)
    if start <= end
]


@pytest.mark.parametrize("kda_layers",
                         _KDA_LAYER_SETS.values(),
                         ids=_KDA_LAYER_SETS.keys())
def test_parity_with_removed_patch(config_hook, kda_layers):
    original = ModelConfig.get_num_layers_by_block_type
    config = _config(kda_layers)
    config_hook.verify_and_update_model_config(config)
    for start, end in _STAGE_SLICES:
        expected = _removed_patch_attention_count(kda_layers, start, end)
        assert _count(config, start, end,
                      "attention") == expected, (start, end)
        # The removed patch raised for this block type; the two now partition
        # the stage.
        assert _count(config, start, end,
                      "linear_attention") == end - start - expected
    # The counts come from metadata alone: upstream's method is untouched.
    assert ModelConfig.get_num_layers_by_block_type is original
    assert not hasattr(ModelConfig, "_tpu_kimi_kda_layer_counts_patched")


def test_hook_is_idempotent(config_hook):
    """ModelConfig can be rebuilt around an already-updated HF config."""
    config = _config()
    config_hook.verify_and_update_model_config(config)
    first = config.hf_text_config.layers_block_type
    assert first == _BLOCK_TYPES
    # A second run keeps the first answer rather than deriving a new one.
    config.hf_text_config.linear_attn_config["kda_layers"] = []
    config_hook.verify_and_update_model_config(config)
    assert config.hf_text_config.layers_block_type is first
    assert first == _BLOCK_TYPES


@pytest.mark.parametrize(
    "field", ["layers_block_type", "layer_types", "attn_type_list"])
def test_preserves_existing_layer_metadata(config_hook, field):
    metadata = ["attention"] * _NUM_LAYERS
    config = _config(**{field: metadata})
    config_hook.verify_and_update_model_config(config)
    assert getattr(config.hf_text_config, field) is metadata
    if field != "layers_block_type":
        assert not hasattr(config.hf_text_config, "layers_block_type")


def test_attn_type_list_guard_reads_the_top_level_config(config_hook):
    """Upstream reads ``attn_type_list`` from ``hf_config``, not from the text
    config, so that is where the guard has to look."""
    config = _config()
    config.hf_config = SimpleNamespace(attn_type_list=[1] * _NUM_LAYERS)
    config_hook.verify_and_update_model_config(config)
    assert not hasattr(config.hf_text_config, "layers_block_type")


def test_multimodal_wrapper_updates_only_the_text_config(config_hook):
    """K3 nests the KDA text config under a multimodal ``hf_config``, and its
    MXFP4 quantization config must stay on compressed-tensors."""
    quantization = {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized"
    }
    config = _config(quantization_config=quantization.copy())
    config.hf_config = SimpleNamespace(quantization_config=quantization.copy())
    config.model_arch_config = SimpleNamespace(
        quantization_config=quantization.copy())
    config_hook.verify_and_update_model_config(config)
    assert not hasattr(config.hf_config, "layers_block_type")
    assert config.hf_text_config.layers_block_type == _BLOCK_TYPES
    for cfg in (config.hf_config, config.hf_text_config,
                config.model_arch_config):
        assert cfg.quantization_config == quantization


def test_missing_linear_attn_config_is_left_alone(config_hook):
    config = _config()
    config.hf_text_config.linear_attn_config = None
    config_hook.verify_and_update_model_config(config)
    assert not hasattr(config.hf_text_config, "layers_block_type")
    del config.hf_text_config.linear_attn_config
    config_hook.verify_and_update_model_config(config)
    assert not hasattr(config.hf_text_config, "layers_block_type")


@pytest.mark.parametrize("linear_attn_config", [{}, {"kda_layers": None}])
def test_absent_kda_list_counts_all_attention(config_hook, linear_attn_config):
    config = _config()
    config.hf_text_config.linear_attn_config = linear_attn_config
    config_hook.verify_and_update_model_config(config)
    assert _count(config, 0, _NUM_LAYERS, "attention") == _NUM_LAYERS
    assert _count(config, 0, _NUM_LAYERS, "linear_attention") == 0


def test_missing_num_hidden_layers_counts_nothing(config_hook):
    """Upstream reports zero total layers for a config without
    ``num_hidden_layers``. The list follows that total, as the removed patch
    did through the resulting (0, 0) stage bounds."""
    config = _config()
    del config.hf_text_config.num_hidden_layers
    config.get_total_num_hidden_layers = lambda: 0
    config_hook.verify_and_update_model_config(config)
    assert config.hf_text_config.layers_block_type == []
    assert _count(config, 0, 0, "attention") == 0


def test_real_kimi_configs_count_through_the_hook(config_hook):
    """The same path on upstream's HF config classes, including the K3 wrapper
    whose text config is nested, checked against ``is_kda_layer``."""
    from vllm.transformers_utils.config import get_hf_text_config
    from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    def text_kwargs():
        return {
            "num_hidden_layers": 4,
            "linear_attn_config": {
                "kda_layers": [1, 2, 3],
                "full_attn_layers": [4]
            },
        }

    for hf_config in (KimiLinearConfig(**text_kwargs()),
                      KimiK3Config(text_config=text_kwargs())):
        text = get_hf_text_config(hf_config)
        config = SimpleNamespace(hf_config=hf_config,
                                 hf_text_config=text,
                                 get_total_num_hidden_layers=lambda text=text:
                                 text.num_hidden_layers)
        config_hook.verify_and_update_model_config(config)
        assert text.layers_block_type == [
            "linear_attention" if text.is_kda_layer(idx) else "attention"
            for idx in range(4)
        ]
        assert _count(config, 0, 4, "attention") == 1
        assert _count(config, 2, 4, "linear_attention") == 1
        if hf_config is not text:
            assert not hasattr(hf_config, "layers_block_type")


def test_patch_registry_no_longer_lists_the_layer_count_patch():
    registry = _load("_test_kimi_tpu_patch_registry", _REGISTRY_PATH)
    targets = [patch.target for patch in registry.PATCHES]
    assert targets
    assert not any("kimi_kda_layer_counts" in target for target in targets)
