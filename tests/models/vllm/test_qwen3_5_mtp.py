# SPDX-License-Identifier: Apache-2.0

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.interfaces import supports_pp

from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.qwen3_5_mtp import Qwen3_5MoeMTP, Qwen3_5MTP


def test_qwen35_mtp_drafts_declare_pipeline_support():
    assert supports_pp(Qwen3_5MTP)
    assert supports_pp(Qwen3_5MoeMTP)


def test_qwen35_mtp_drafts_are_registered_over_upstream():
    register_models()
    for arch, cls in (("Qwen3_5MTP", Qwen3_5MTP), ("Qwen3_5MoeMTP", Qwen3_5MoeMTP)):
        assert ModelRegistry.models[arch].load_model_cls() is cls


def test_predictor_forward_embeds_on_a_non_first_pipeline_rank(monkeypatch):
    """The embedding branch runs whenever no intermediate tensors are given,
    on a stage whose PP group says it is not the first rank."""
    from types import SimpleNamespace

    import torch
    from vllm.model_executor.models import qwen3_5_mtp as upstream

    import vllm_torchtpu.models.vllm.qwen3_5_mtp as ours

    monkeypatch.setattr(
        ours,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    calls = []

    class Layer:
        use_attn_reduce_scatter_for_moe = False

        def __call__(self, *, positions, hidden_states, residual):
            calls.append(("layer", residual))
            return hidden_states, hidden_states

    predictor = SimpleNamespace(
        embed_input_ids=lambda ids: torch.ones(ids.shape[0], 2),
        pre_fc_norm_embedding=lambda x: x,
        pre_fc_norm_hidden=lambda x: x,
        fc=lambda x: x[:, :2] + x[:, 2:],
        num_mtp_layers=1,
        layers=[Layer()],
        norm=lambda h, r: (h + r, None),
    )
    out = upstream.Qwen3_5MultiTokenPredictor.forward(
        predictor,
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3),
        hidden_states=torch.full((3, 2), 2.0),
    )
    assert calls == [("layer", None)]
    # fc adds embedding (1) and hidden (2); norm adds the residual again.
    assert torch.equal(out, torch.full((3, 2), 6.0))
