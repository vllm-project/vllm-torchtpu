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

import pytest
import torch

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
    def __call__(self, input_ids, positions):
        rows = torch.arange(input_ids.shape[0], dtype=torch.float32)
        return torch.stack([rows, torch.zeros_like(rows)], dim=1)


class _FakeDraft:
    """A DSpark draft whose outputs name their inputs: hidden row t holds t,
    row t drafts token t + 1, and a confidence is 100 * row + previous token."""

    has_own_lm_head = False
    has_own_embed_tokens = False

    def __init__(self, vocab=16):
        self.vocab = vocab
        self.model = _FakeBackbone()

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


@pytest.mark.parametrize("enabled", [True, False])
def test_adaptive_verification_is_ignored_with_a_warning(monkeypatch, enabled):
    with mock.patch("vllm_torchtpu.spec_decode.dspark.logger") as log:
        p = _make_proposer(
            {"dspark_noise_token_id": 7, **_LAYER_IDS},
            enable_adaptive_verification=enabled,
        )
    monkeypatch.setattr(
        DFlashProposer,
        "load_model",
        lambda self, target_model: setattr(self, "draft_model", _FakeDraft()),
    )
    p.load_model(None)
    assert p.use_confidence_head is False
    assert log.warning.called is enabled


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


class _ShardedFakeDraft(_FakeDraft):
    """_FakeDraft with a token-dependent Markov bias and the vocab-parallel
    interface, simulating two ranks of half the vocab each."""

    def __init__(self, enable=True, vocab=16):
        super().__init__(vocab)
        self.enable = enable
        self.gathered_calls = 0

    def compute_draft_logits(self, hidden):
        self.gathered_calls += 1
        rows = hidden[:, 0]
        return torch.cos(rows[:, None] * torch.arange(self.vocab)).round(decimals=1)

    def markov_bias(self, markov_embed):
        self.gathered_calls += 1
        return self._bias(markov_embed)

    def _bias(self, markov_embed):
        return torch.sin(markov_embed * torch.arange(self.vocab)).round(decimals=1)

    def enable_vocab_parallel_markov_argmax(self):
        return self.enable

    def compute_draft_logits_shard(self, hidden):
        logits = _ShardedFakeDraft.compute_draft_logits(self, hidden)
        self.gathered_calls -= 1
        return logits

    def markov_argmax_shard(self, logits_shard, markov_embed):
        from vllm_torchtpu.spec_decode.vocab_parallel_argmax import (
            reduce_shard_argmax_pairs,
            shard_argmax_pair,
        )

        scores = logits_shard + self._bias(markov_embed)
        half = self.vocab // 2
        pairs = [
            shard_argmax_pair(scores[:, r * half : (r + 1) * half], r * half)
            for r in range(2)
        ]
        return reduce_shard_argmax_pairs(torch.cat(pairs, dim=-1), 2)


def _draft_with(monkeypatch, draft, replicated=False):
    p = _make_proposer(
        {"dspark_noise_token_id": 7, **_LAYER_IDS}, num_speculative_tokens=4
    )
    p._draft_replicated = replicated
    monkeypatch.setattr(
        DFlashProposer,
        "load_model",
        lambda self, target_model: setattr(self, "draft_model", draft),
    )
    p.load_model(None)
    return p


@pytest.mark.parametrize(
    "enable,replicated,uses_shards",
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_vocab_parallel_markov_argmax_is_opt_in_and_exact(
    monkeypatch, enable, replicated, uses_shards
):
    forward = DSparkProposer._dflash_forward_and_sample.__wrapped__
    input_ids = torch.tensor([5, 7, 7, 7, 9, 7, 7, 7, 3, 7, 7, 7])
    positions = torch.zeros(12, dtype=torch.int32)

    reference = _draft_with(monkeypatch, _FakeDraft(), replicated)
    reference.draft_model = _ShardedFakeDraft(enable=False)
    expected, _ = forward(reference, input_ids, positions, 4)

    draft = _ShardedFakeDraft(enable=enable)
    p = _draft_with(monkeypatch, draft, replicated)
    assert p._vocab_parallel_markov is uses_shards
    drafts, _ = forward(p, input_ids, positions, 4)
    assert drafts.tolist() == expected.tolist()
    # The shard path never builds the gathered logits or the full bias.
    assert (draft.gathered_calls == 0) is uses_shards


def test_drafts_without_the_interface_keep_gathered_logits(monkeypatch):
    p = _draft_with(monkeypatch, _FakeDraft())
    assert p._vocab_parallel_markov is False
