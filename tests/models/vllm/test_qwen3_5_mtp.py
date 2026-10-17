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

"""The Qwen3.5 MTP drafts come from upstream; the sharded loader reads them."""

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP, Qwen3_5MTP

from vllm_torchtpu.model_loader_patches import _unused_checkpoint_tensor_predicate
from vllm_torchtpu.models.vllm import register_models


def test_qwen35_mtp_drafts_resolve_to_upstream():
    """No local override: upstream supports PP and the checkpoint filter."""
    register_models()
    for arch, cls in (("Qwen3_5MTP", Qwen3_5MTP), ("Qwen3_5MoeMTP", Qwen3_5MoeMTP)):
        assert ModelRegistry.models[arch].load_model_cls() is cls


def test_sharded_loader_fetches_what_upstream_load_weights_reads(monkeypatch):
    """The fetch plan built from upstream's ``is_unused_checkpoint_weight``
    keeps exactly the tensors upstream's ``load_weights`` reads."""
    import torch
    from vllm.model_executor.models import qwen3_5_mtp as upstream

    names = (
        "mtp.fc.weight",
        "mtp.layers.0.mlp.experts.3.up_proj.weight",
        "mtp.layers.0.mlp.experts.3.up_proj.weight_scale_inv",
        "mtp.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.91.mlp.gate.weight",
        "model.norm.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    )
    read: list[str] = []

    class RecordingLoader:
        def __init__(self, module):
            pass

        def load_weights(self, weights, mapper=None):
            read.extend(name for name, _ in weights)

    monkeypatch.setattr(upstream, "AutoWeightsLoader", RecordingLoader)
    for cls in (Qwen3_5MTP, Qwen3_5MoeMTP):
        draft = cls.__new__(cls)
        torch.nn.Module.__init__(draft)
        is_unused = _unused_checkpoint_tensor_predicate(draft)
        assert is_unused is not None
        for name in names:
            read.clear()
            draft.load_weights([(name, None)])
            assert is_unused(name) == (not read), (cls.__name__, name)
