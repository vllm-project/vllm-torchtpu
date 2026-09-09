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
"""Tests for the per-rank Run:AI aux-file pull (runai_repull)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu.worker.runai_repull import (COMPLETE_SENTINEL,
                                               PULLING_SENTINEL,
                                               ensure_runai_aux_files)

MODEL_URI = "gs://bucket/models/kimi/k3"
TOK_URI = "gs://bucket/models/kimi/tok"


class FakeModelConfig:
    """Stands in for vllm's ModelConfig: carries the three fields the
    repull reads/writes and a maybe_pull_model_tokenizer_for_runai that
    behaves like upstream's (pull into a URI-keyed dir, rewrite the
    fields), against a tmp cache root."""

    def __init__(self, cache_root, model, tokenizer=None, model_weights=""):
        self.cache_root = str(cache_root)
        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else model
        self.model_weights = model_weights
        self.pull_calls = []
        self.crash_after_config = False

    def dir_for(self, uri):
        return os.path.join(self.cache_root, uri.rsplit("/", 1)[1])

    def maybe_pull_model_tokenizer_for_runai(self, model, tokenizer):
        self.pull_calls.append((model, tokenizer))
        if self.model_weights:
            return
        if not model.startswith("gs://"):
            return
        d = self.dir_for(model)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            f.write("{}")
        if self.crash_after_config:
            raise RuntimeError("simulated SIGKILL mid-pull")
        with open(os.path.join(d, "modeling_custom.py"), "w") as f:
            f.write("")
        self.model_weights = model
        self.model = d
        if tokenizer == model:
            self.tokenizer = d
        elif tokenizer.startswith("gs://"):
            td = self.dir_for(tokenizer)
            os.makedirs(td, exist_ok=True)
            self.tokenizer = td


@pytest.fixture
def cache_root(tmp_path):
    return tmp_path / "assets"


def _v2_config(cache_root, **kw):
    """Ray V2 state: head's rewritten config (local dir + URI)."""
    local = os.path.join(str(cache_root), "k3")
    return FakeModelConfig(cache_root,
                           model=local,
                           model_weights=MODEL_URI,
                           **kw)


def test_noop_for_non_object_storage_model(cache_root):
    cfg = FakeModelConfig(cache_root, model="Qwen/Qwen3-0.6B")
    ensure_runai_aux_files(cfg)
    assert cfg.pull_calls == []


def test_v2_pulls_when_dir_missing_and_restores_fields(cache_root):
    cfg = _v2_config(cache_root)
    local = cfg.model
    ensure_runai_aux_files(cfg)
    # Upstream's pull was asked for model AND tokenizer from the model URI
    # (tokenizer shared the model dir on the head).
    assert cfg.pull_calls == [(MODEL_URI, MODEL_URI)]
    assert cfg.model == local and cfg.tokenizer == local
    assert cfg.model_weights == MODEL_URI
    assert os.path.exists(os.path.join(local, "config.json"))
    assert os.path.exists(os.path.join(local, PULLING_SENTINEL))
    assert os.path.exists(os.path.join(local, COMPLETE_SENTINEL))


def test_v2_skips_when_sibling_rank_completed(cache_root):
    cfg = _v2_config(cache_root)
    ensure_runai_aux_files(cfg)
    again = _v2_config(cache_root)
    ensure_runai_aux_files(again)
    assert again.pull_calls == []
    assert again.model_weights == MODEL_URI


def test_v2_skips_on_head_host_dir_pulled_by_api_server(cache_root):
    # The API server's own pull leaves config.json and no worker sentinel.
    cfg = _v2_config(cache_root)
    os.makedirs(cfg.model)
    with open(os.path.join(cfg.model, "config.json"), "w") as f:
        f.write("{}")
    ensure_runai_aux_files(cfg)
    assert cfg.pull_calls == []


def test_v2_repulls_after_interrupted_worker_pull(cache_root):
    crashed = _v2_config(cache_root)
    crashed.crash_after_config = True
    with pytest.raises(RuntimeError):
        ensure_runai_aux_files(crashed)
    local = crashed.model
    # config.json landed but the pull never completed.
    assert os.path.exists(os.path.join(local, "config.json"))
    assert os.path.exists(os.path.join(local, PULLING_SENTINEL))
    assert not os.path.exists(os.path.join(local, COMPLETE_SENTINEL))

    retry = _v2_config(cache_root)
    ensure_runai_aux_files(retry)
    assert retry.pull_calls == [(MODEL_URI, MODEL_URI)]
    assert os.path.exists(os.path.join(local, COMPLETE_SENTINEL))


def test_v1_pre_pull_state_pulls_and_rewrites(cache_root):
    # Ray V1 restores the URI into model and clears model_weights.
    cfg = FakeModelConfig(cache_root, model=MODEL_URI, model_weights=None)
    ensure_runai_aux_files(cfg)
    assert cfg.pull_calls == [(MODEL_URI, MODEL_URI)]
    assert cfg.model == cfg.dir_for(MODEL_URI)
    assert cfg.tokenizer == cfg.model
    assert cfg.model_weights == MODEL_URI
    assert os.path.exists(os.path.join(cfg.model, COMPLETE_SENTINEL))


def test_v1_separate_tokenizer_uri_passes_through(cache_root):
    cfg = FakeModelConfig(cache_root,
                          model=MODEL_URI,
                          tokenizer=TOK_URI,
                          model_weights=None)
    ensure_runai_aux_files(cfg)
    assert cfg.pull_calls == [(MODEL_URI, TOK_URI)]
    assert cfg.tokenizer == cfg.dir_for(TOK_URI)


def test_v2_separate_tokenizer_dir_missing_is_reported(cache_root):
    # Head pulled a distinct tokenizer URI into its own dir; the URI is not
    # retained, so the worker can only flag the gap.
    missing_tok = os.path.join(str(cache_root), "tok-only-on-head")
    cfg = _v2_config(cache_root, tokenizer=missing_tok)
    with patch("vllm_torchtpu.worker.runai_repull.logger") as log:
        ensure_runai_aux_files(cfg)
    assert cfg.pull_calls == [(MODEL_URI, missing_tok)]
    assert cfg.tokenizer == missing_tok
    messages = [str(c.args[0]) % c.args[1:] for c in log.error.call_args_list]
    assert any("[runai-repull] tokenizer dir" in m and missing_tok in m
               for m in messages), messages


def test_noop_for_mock_model_config():
    # TPUWorker unit tests construct the worker with MagicMock configs.
    cfg = MagicMock()
    ensure_runai_aux_files(cfg)
    cfg.maybe_pull_model_tokenizer_for_runai.assert_not_called()
