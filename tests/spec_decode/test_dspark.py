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
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from vllm_torchtpu.spec_decode import dflash as dflash_mod
from vllm_torchtpu.spec_decode import dspark as dspark_mod
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.dspark import DSparkProposer


class _FakeHFConfig:
    """to_dict() reflects attribute writes, like a transformers config."""

    def __init__(self, d):
        self._d = dict(d)

    def to_dict(self):
        out = dict(self._d)
        if hasattr(self, "dflash_config"):
            out["dflash_config"] = self.dflash_config
        return out


def _make_proposer(hf_dict, num_speculative_tokens=4, **spec_fields):
    # The vLLM 0.29 drafting fields at their defaults; tests override them.
    fields = dict(
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
        dspark_draft_topk=None,
    )
    fields.update(spec_fields)
    speculative_config = SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        draft_tensor_parallel_size=1,
        draft_model_config=SimpleNamespace(hf_config=_FakeHFConfig(hf_dict)),
        **fields,
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        model_config=SimpleNamespace(max_model_len=2048),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "model.layers.0.self_attn": mock.MagicMock(),
            },
            fast_moe_cold_start=False,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=1,
            is_moe_model=False,
            use_sequence_parallel_moe=False,
        ),
    )
    runner = mock.MagicMock()
    runner.device = torch.device("cpu")
    return DSparkProposer(runner=runner, vllm_config=vllm_config)


_LAYER_IDS = {"dspark_target_layer_ids": [1, 15, 30]}


def test_noise_resolution_and_dense_block():
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "dspark_block_size": 4,
            **_LAYER_IDS,
        }
    )
    assert p.mask_token_id == 7
    assert p.sample_from_anchor
    assert p._query_block_size(4) == 4
    # propose() sizes the query block via this property, so the dense
    # override must reach it (K slots, not DFlash's 1 + K).
    assert p.block_size == 4


def test_bonus_anchor_keeps_dflash_layout():
    p = _make_proposer(
        {
            "mask_token_id": 5,
            "sample_from_anchor": False,
            **_LAYER_IDS,
        }
    )
    assert not p.sample_from_anchor
    assert p._query_block_size(4) == 5
    assert p.block_size == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("draft_sample_method", "probabilistic"),
        ("dspark_draft_topk", 8),
    ],
)
def test_warns_on_unsupported_drafting_options(field, value):
    with mock.patch("vllm_torchtpu.spec_decode.dspark.logger") as log:
        p = _make_proposer(
            {
                "dspark_noise_token_id": 7,
                "dspark_block_size": 4,
                **_LAYER_IDS,
            },
            **{field: value},
        )
    # Drafting falls back to greedy over the fixed K-token block.
    assert p.block_size == 4
    log.warning.assert_called_once()
    assert "does not support" in log.warning.call_args.args[0]


@pytest.mark.parametrize(
    "sample_from_anchor,trained_block,K,query_block",
    [
        (True, 5, 6, 6),  # DeepSeek-V4-Pro-0813 served at K=6
        (True, 7, 4, 4),
        (False, 7, 4, 5),  # bonus-anchor layout keeps 1 + K
    ],
)
def test_k_need_not_match_checkpoint_block(
    sample_from_anchor, trained_block, K, query_block
):
    # The query block follows K whatever block the checkpoint was trained
    # at; the mismatch is allowed but warned about.
    hf = {
        "dspark_noise_token_id": 7,
        "dspark_block_size": trained_block,
        "sample_from_anchor": sample_from_anchor,
        **_LAYER_IDS,
    }
    with mock.patch("vllm_torchtpu.spec_decode.dspark.logger") as log:
        p = _make_proposer(hf, num_speculative_tokens=K)
    assert p.block_size == query_block
    log.warning.assert_called_once()
    assert "trained with block size" in log.warning.call_args.args[0]


@pytest.mark.parametrize(
    "block_key,sample_from_anchor,trained_block,K",
    [
        ("dspark_block_size", True, 4, 4),
        # Speculators-format checkpoints count the bonus anchor in block_size
        # and draft block_size - 1 tokens (e.g. dspark_qwen3_8b_block7, K=6).
        ("block_size", False, 7, 6),
    ],
)
def test_matching_checkpoint_block_does_not_warn(
    block_key, sample_from_anchor, trained_block, K
):
    hf = {
        "dspark_noise_token_id": 7,
        block_key: trained_block,
        "sample_from_anchor": sample_from_anchor,
        **_LAYER_IDS,
    }
    with mock.patch("vllm_torchtpu.spec_decode.dspark.logger") as log:
        _make_proposer(hf, num_speculative_tokens=K)
    log.warning.assert_not_called()


def test_missing_noise_token():
    with pytest.raises(ValueError, match="noise/mask token"):
        _make_proposer({})


def test_missing_target_layer_ids():
    with pytest.raises(ValueError, match="target layer ids"):
        _make_proposer({"dspark_noise_token_id": 7})


def test_target_layer_id_resolution_chain():
    # Aux ids (target + 1) take precedence and are shifted back down.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "eagle_aux_hidden_state_layer_ids": [2, 16, 31],
            "dspark_target_layer_ids": [9, 9, 9],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [1, 15, 30]

    # Dense-checkpoint spelling.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "target_layer_ids": [3, 7],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [3, 7]

    # An explicit dflash_config wins unchanged.
    p = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "dflash_config": {"target_layer_ids": [4, 8]},
            "dspark_target_layer_ids": [9, 9],
        }
    )
    assert p.dflash_config["target_layer_ids"] == [4, 8]


def test_no_forced_embedding_sharing():
    # A reduced-vocab DSpark draft owns its lm_head; load_model must honor
    # has_own_embed_tokens / has_own_lm_head instead of force-sharing.
    p = _make_proposer({"dspark_noise_token_id": 7, **_LAYER_IDS})
    assert p._force_share_target_embeddings is False


class _FakeBackbone:
    def __init__(self, confidence_head: bool):
        self.confidence_head = object() if confidence_head else None

    def __call__(self, input_ids, positions):
        rows = torch.arange(input_ids.shape[0], dtype=torch.float32)
        return torch.stack([rows, torch.zeros_like(rows)], dim=1)


class _FakeDraft:
    """A DSpark draft whose outputs name their inputs: hidden row t holds t,
    row t drafts token t + 1, and a confidence is 100 * row + previous token."""

    has_own_lm_head = False
    has_own_embed_tokens = False

    def __init__(self, vocab=16, confidence_head=True):
        self.vocab = vocab
        self.model = _FakeBackbone(confidence_head)

    def compute_draft_logits(self, hidden):
        rows = hidden[:, 0].long()
        return torch.nn.functional.one_hot((rows + 1) % self.vocab, self.vocab).float()

    def markov_embed(self, token_ids):
        return token_ids.float().unsqueeze(1)

    def markov_bias(self, markov_embed):
        return torch.zeros(markov_embed.shape[0], self.vocab)

    def map_draft_to_target(self, draft_ids):
        return draft_ids

    def compute_confidence(self, head_hidden, markov_embed):
        return 100 * head_hidden[:, 0] + markov_embed[:, 0]


@pytest.mark.parametrize(
    "enabled,confidence_head,uses_head,warns",
    [
        (True, True, True, False),
        (True, False, False, True),
        (False, True, False, False),
    ],
)
def test_adaptive_verification_needs_a_confidence_head(
    monkeypatch, enabled, confidence_head, uses_head, warns
):
    with mock.patch("vllm_torchtpu.spec_decode.dspark.logger") as log:
        p = _make_proposer(
            {"dspark_noise_token_id": 7, **_LAYER_IDS},
            enable_adaptive_verification=enabled,
        )
        draft = _FakeDraft(confidence_head=confidence_head)
        monkeypatch.setattr(
            DFlashProposer,
            "load_model",
            lambda self, target_model: setattr(self, "draft_model", draft),
        )
        p.load_model(None)
    assert p.use_confidence_head is uses_head
    assert log.warning.called is warns


def test_draft_step_scores_each_slot_from_its_hidden_and_previous_token():
    p = _make_proposer(
        {"dspark_noise_token_id": 7, **_LAYER_IDS},
        num_speculative_tokens=3,
    )
    p.draft_model = _FakeDraft()
    forward = DSparkProposer._dflash_forward_and_sample.__wrapped__
    # Two requests of 3 dense slots: anchors 5 and 9, noise 7.
    input_ids = torch.tensor([5, 7, 7, 9, 7, 7])
    positions = torch.zeros(6, dtype=torch.int32)

    _, hidden = forward(p, input_ids, positions, 3)
    assert hidden.shape == (6, 2), "without the head the hidden states pass through"

    p.use_confidence_head = True
    drafts, confidence = forward(p, input_ids, positions, 3)
    assert drafts.tolist() == [[1, 2, 3], [4, 5, 6]]
    # Slot 0 follows the anchor, each later slot the token drafted before it.
    assert confidence.tolist() == [[5, 101, 202], [309, 404, 505]]


def test_proposal_confidences_are_taken_once():
    p = _make_proposer({"dspark_noise_token_id": 7, **_LAYER_IDS})
    p.use_confidence_head = True
    first, second = torch.ones(8, 4), torch.zeros(4, 4)
    p._observe_draft_aux(first, 0, SimpleNamespace(start_index=2, num_reqs=3))
    p._observe_draft_aux(second, 1, SimpleNamespace(start_index=5, num_reqs=1))
    taken = p.take_draft_confidences()
    assert taken is not None
    assert taken.chunks == [first, second]
    assert taken.spans == [(2, 3), (5, 1)]
    assert p.take_draft_confidences() is None


def test_a_new_proposal_replaces_untaken_confidences():
    p = _make_proposer({"dspark_noise_token_id": 7, **_LAYER_IDS})
    p.use_confidence_head = True
    p._observe_draft_aux(
        torch.ones(8, 4), 0, SimpleNamespace(start_index=0, num_reqs=8)
    )
    p._observe_draft_aux(
        torch.zeros(4, 4), 0, SimpleNamespace(start_index=1, num_reqs=2)
    )
    taken = p.take_draft_confidences()
    assert taken is not None
    assert taken.spans == [(1, 2)]


def test_draft_aux_is_dropped_without_the_confidence_head():
    p = _make_proposer({"dspark_noise_token_id": 7, **_LAYER_IDS})
    p._observe_draft_aux(
        torch.ones(8, 4), 0, SimpleNamespace(start_index=0, num_reqs=8)
    )
    assert p.take_draft_confidences() is None


class _FakeTpuTensor(torch.Tensor):
    """CPU tensor subclass reporting ``device.type == 'tpu'``."""

    @property
    def device(self):  # type: ignore[override]
        return SimpleNamespace(type="tpu")


def _as_fake_tpu(tensor: torch.Tensor) -> _FakeTpuTensor:
    return tensor.as_subclass(_FakeTpuTensor)


def _build_minimal_dspark_draft_model(vocab_size: int = 32, hidden_size: int = 16):
    gen = torch.Generator().manual_seed(1234)
    embed_weight = torch.randn(
        vocab_size, hidden_size, generator=gen, dtype=torch.bfloat16
    )
    lm_weight = torch.randn(
        vocab_size, hidden_size, generator=gen, dtype=torch.bfloat16
    )
    markov_w = torch.randn(vocab_size, vocab_size, generator=gen, dtype=torch.bfloat16)
    call_log: list[dict[str, object]] = []
    state = {"in_compiled": False}

    def embed_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
        plain_ids = input_ids.as_subclass(torch.Tensor)
        out = embed_weight.index_select(0, plain_ids.long())
        if state["in_compiled"] and out.shape[0] >= 3:
            # Simulate libtpu < 0.0.52 custom-gather sublane row scrambling.
            out = out.clone()
            out[[0, 2]] = out[[2, 0]]
        if isinstance(input_ids, _FakeTpuTensor):
            out = _as_fake_tpu(out)
        call_log.append(
            {"op": "embed_input_ids", "in_compiled": state["in_compiled"], "out": out}
        )
        return out

    class _InnerModel:
        def __init__(self) -> None:
            self.embed_input_ids = embed_input_ids
            self.layers: list[object] = []

        def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
            call_log.append({"op": "model_forward", "args": args, "kwargs": kwargs})
            if len(args) >= 3 and args[2] is not None:
                embeds = args[2]
            else:
                req_ids = args[0] if args else kwargs["input_ids"]
                embeds = self.embed_input_ids(req_ids)  # type: ignore[arg-type]
            assert isinstance(embeds, torch.Tensor)
            plain_embeds = embeds.as_subclass(torch.Tensor)
            call_log.append({"op": "s0_embeds", "tensor": plain_embeds.clone()})
            return plain_embeds

    draft_model = SimpleNamespace(
        model=_InnerModel(),
        compute_draft_logits=lambda h: h.float() @ lm_weight.float().T,
        markov_embed=lambda ids: markov_w.index_select(
            0, ids.as_subclass(torch.Tensor).long()
        ).float(),
        markov_bias=lambda emb: emb,
        map_draft_to_target=lambda ids: ids.to(torch.int32),
        has_own_lm_head=False,
        has_own_embed_tokens=False,
    )
    return draft_model, call_log, state


def test_eager_embed_outside_compile_on_dp_gt_1(monkeypatch):
    """Materialize draft embeddings outside compile across propose and dummy."""
    proposer = _make_proposer(
        {
            "dspark_noise_token_id": 7,
            "dspark_block_size": 5,
            **_LAYER_IDS,
        },
        num_speculative_tokens=5,
    )
    proposer.vllm_config.parallel_config.data_parallel_size = 8
    draft_model, call_log, state = _build_minimal_dspark_draft_model()
    proposer.draft_model = draft_model

    sync_calls: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        dspark_mod,
        "synchronize_tensors",
        lambda tensors=None, wait=True: sync_calls.append((tensors, wait)),
    )
    monkeypatch.setattr(
        dflash_mod,
        "synchronize_tensors",
        lambda tensors=None, wait=True: None,
    )
    eager_fn = torch._dynamo.disable(DSparkProposer._dflash_forward_and_sample)

    def wrapped_compiled(*args: object, **kwargs: object):
        state["in_compiled"] = True
        try:
            return eager_fn(proposer, *args, **kwargs)
        finally:
            state["in_compiled"] = False

    proposer._dflash_forward_and_sample = wrapped_compiled  # type: ignore[method-assign]
    proposer.runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(num_reqs=2),
        mesh=None,
        max_num_reqs=8,
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda num_tokens: None,
        _dp_step_num_chunks=0,
    )
    proposer.draft_chunks = [
        SimpleNamespace(
            num_reqs=2,
            start_index=0,
            aux_hidden_states=[torch.zeros((2, 16), dtype=torch.bfloat16)],
            attn_ctx=SimpleNamespace(
                request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32)
            ),
        )
    ]
    raw_ids = torch.tensor([14, 7, 7, 7, 7, 21, 7, 7, 7, 7], dtype=torch.int32)
    eager_ref = draft_model.model.embed_input_ids(raw_ids).as_subclass(torch.Tensor)
    call_log.clear()

    fake_tpu_ids = _as_fake_tpu(raw_ids)
    pos = torch.arange(10, dtype=torch.int32)
    proposer._update_draft_kv_cache_from_target = mock.MagicMock()
    proposer._prepare_dflash_inputs = mock.MagicMock(
        return_value=(fake_tpu_ids, pos, torch.tensor([5, 5], dtype=torch.int32))
    )
    proposer._build_draft_attn_metadata = mock.MagicMock(return_value={})

    draft_tokens = proposer.propose(
        sampled_token_ids=[[14], [21]],
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=np.array([0, 0]),
        scheduler_output=None,
        return_device=True,
    )
    assert len(draft_tokens) == 1
    embed_calls = [c for c in call_log if c["op"] == "embed_input_ids"]
    fwd_calls = [c for c in call_log if c["op"] == "model_forward"]
    s0_entries = [c for c in call_log if c["op"] == "s0_embeds"]
    assert len(embed_calls) == 1
    assert embed_calls[0]["in_compiled"] is False
    assert len(sync_calls) == 1 and sync_calls[0][1] is False
    assert fwd_calls[0]["args"] == (fake_tpu_ids, pos, embed_calls[0]["out"])
    s0_embeds = s0_entries[0]["tensor"]
    assert isinstance(s0_embeds, torch.Tensor)
    assert torch.equal(s0_embeds, eager_ref)
    for row_idx in (2, 3, 4):
        assert torch.equal(s0_embeds[1], s0_embeds[row_idx])

    # Verify _dummy_draft_forward() also dispatches eager embedding on DP > 1.
    call_log.clear()
    sync_calls.clear()
    proposer._build_attn_metadata_for_draft = mock.MagicMock(return_value={})
    orig_zeros = dflash_mod.torch.zeros

    def fake_tpu_zeros(*args: object, **kwargs: object) -> torch.Tensor:
        tensor = orig_zeros(*args, **kwargs)
        if tensor.ndim == 1 and tensor.shape[0] == 10 and tensor.dtype == torch.int32:
            return _as_fake_tpu(tensor)
        return tensor

    with mock.patch.object(dflash_mod.torch, "zeros", side_effect=fake_tpu_zeros):
        proposer._dummy_draft_forward(num_tokens=10, use_max_model_len=True, sync=False)
    embed_dummy = [c for c in call_log if c["op"] == "embed_input_ids"]
    fwd_dummy = [c for c in call_log if c["op"] == "model_forward"]
    assert len(embed_dummy) == 1 and embed_dummy[0]["in_compiled"] is False
    assert len(sync_calls) == 1 and sync_calls[0][1] is False
    assert fwd_dummy[0]["args"][2] is embed_dummy[0]["out"]


@pytest.mark.parametrize(
    ("dp_size", "on_tpu", "has_draft"),
    [
        (1, True, True),
        (8, False, True),
        (8, True, False),
    ],
    ids=["dp_eq_1", "cpu_tensor", "missing_draft_model"],
)
def test_eager_embed_negative_controls_and_base_dispatch(
    dp_size: int,
    on_tpu: bool,
    has_draft: bool,
):
    """Keep in-graph embedding when DP == 1, on CPU, or on base DFlash."""
    proposer = _make_proposer(
        {"dspark_noise_token_id": 7, "dspark_block_size": 5, **_LAYER_IDS},
        num_speculative_tokens=5,
    )
    proposer.vllm_config.parallel_config.data_parallel_size = dp_size
    draft_model, call_log, _ = _build_minimal_dspark_draft_model()
    proposer.draft_model = draft_model if has_draft else None

    raw_ids = torch.tensor([10, 7, 7, 7, 7, 20, 7, 7, 7, 7], dtype=torch.int32)
    ids = _as_fake_tpu(raw_ids) if on_tpu else raw_ids
    pos = torch.arange(10, dtype=torch.int32)

    assert proposer._maybe_eager_embed_input_ids(ids) is None
    if has_draft:
        eager_fn = torch._dynamo.disable(DSparkProposer._dflash_forward_and_sample)
        proposer._dflash_forward_and_sample = (  # type: ignore[method-assign]
            lambda *a, **k: eager_fn(proposer, *a, **k)
        )
        proposer._forward_and_sample_draft_block(ids, pos, 5)
        fwd_calls = [c for c in call_log if c["op"] == "model_forward"]
        assert len(fwd_calls) == 1
        assert fwd_calls[0]["kwargs"] == {"input_ids": ids, "positions": pos}

    base_dflash = DFlashProposer.__new__(DFlashProposer)
    base_fwd = mock.MagicMock(
        return_value=(torch.zeros((2, 4), dtype=torch.int32), torch.zeros((10, 16)))
    )
    base_dflash._dflash_forward_and_sample = base_fwd  # type: ignore[method-assign]
    base_dflash._forward_and_sample_draft_block(ids, pos, 5)
    base_fwd.assert_called_once_with(ids, pos, 5)
