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
"""TPU test for Speech & Speaker LRU Cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils import speaker_cache as speaker_cache_mod

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

_RESOLVE_MAX_ENTRIES = 2
_WAVEFORM_SAMPLES = 4
_RESOLVE_MAX_BYTES = _WAVEFORM_SAMPLES * 4 * 2  # 2 float32 waveforms (32 B)
_EMBED_DIM = 8
_SPEAKER_MAX_BYTES = _EMBED_DIM * 4 * 2  # 2 float32 embeddings (64 B)


def test_speech_and_speaker_lru_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speech & Speaker LRU Cache

    - TPU path: parse ``speech_cache`` deploy YAML via ``load_deploy_config``,
      initialize ``OmniOpenAIServingSpeech._init_speaker_storage()``, and exercise
      ``_ref_audio_resolve_cache`` and ``_speaker_cache`` with ``"tpu"`` speaker
      embeddings (direct and ``.detach().cpu()``).
    - Setup check: ``current_omni_platform.device_type == "tpu"`` and resolved
      cache limits on ``server`` match ``SpeechCacheConfig``.
    - Outcome check: LRU promotion/eviction order, hit/miss counters, byte bounds
      (``<= max_bytes``), and exact tensor parity against CPU reference embeddings.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"Expected tpu platform, got {current_omni_platform.device_type!r}"
    )

    deploy_yaml = tmp_path / "deploy.yaml"
    deploy_yaml.write_text(
        f"speech_cache:\n"
        f"  resolve_max_bytes: {_RESOLVE_MAX_BYTES}\n"
        f"  resolve_max_entries: {_RESOLVE_MAX_ENTRIES}\n"
        f"  speaker_max_bytes: {_SPEAKER_MAX_BYTES}\n",
        encoding="utf-8",
    )
    deploy_cfg = load_deploy_config(deploy_yaml)

    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    monkeypatch.setattr(speaker_cache_mod, "_SINGLETON", None)

    server = object.__new__(OmniOpenAIServingSpeech)
    server.speech_cache_config = deploy_cfg.speech_cache
    server._init_speaker_storage()

    assert server._ref_audio_resolve_cache_max_bytes == _RESOLVE_MAX_BYTES, (
        f"resolve_max_bytes {server._ref_audio_resolve_cache_max_bytes} != "
        f"{_RESOLVE_MAX_BYTES}"
    )
    assert server._ref_audio_resolve_cache_max_entries == _RESOLVE_MAX_ENTRIES, (
        f"resolve_max_entries {server._ref_audio_resolve_cache_max_entries} != "
        f"{_RESOLVE_MAX_ENTRIES}"
    )
    assert server._speaker_cache.stats()["max_bytes"] == _SPEAKER_MAX_BYTES, (
        f"speaker_max_bytes {server._speaker_cache.stats()['max_bytes']} != "
        f"{_SPEAKER_MAX_BYTES}"
    )

    # Exercise _ref_audio_resolve_cache LRU promotion and entry/byte eviction.
    wav = np.zeros(_WAVEFORM_SAMPLES, dtype=np.float32)
    server._put_resolved_ref_audio("k1", wav, 16000, "a1")
    server._put_resolved_ref_audio("k2", wav, 16000, "a2")
    assert server._get_resolved_ref_audio_artifact_key("k1") == "a1", (
        "k1 was not promoted on resolve cache lookup"
    )
    server._put_resolved_ref_audio("k3", wav, 16000, "a3")

    assert list(server._ref_audio_resolve_cache.keys()) == ["k1", "k3"], (
        f"Expected resolve cache keys ['k1', 'k3'], got "
        f"{list(server._ref_audio_resolve_cache.keys())}"
    )
    assert len(server._ref_audio_resolve_cache) <= _RESOLVE_MAX_ENTRIES, (
        f"Resolve cache entries {len(server._ref_audio_resolve_cache)} > "
        f"{_RESOLVE_MAX_ENTRIES}"
    )
    assert server._ref_audio_resolve_cache_bytes <= _RESOLVE_MAX_BYTES, (
        f"Resolve cache bytes {server._ref_audio_resolve_cache_bytes} > "
        f"{_RESOLVE_MAX_BYTES}"
    )

    # Exercise SpeakerEmbeddingCache with "tpu" tensors (direct + .detach().cpu()).
    torch.manual_seed(0)
    ref_a = torch.randn(_EMBED_DIM, dtype=torch.float32)
    ref_b = torch.randn(_EMBED_DIM, dtype=torch.float32)
    ref_c = torch.randn(_EMBED_DIM, dtype=torch.float32)

    key_a = server._speaker_cache.make_cache_key("Alice", "qwen3_tts", 1)
    key_b = server._speaker_cache.make_cache_key("Bob", "qwen3_tts", 2)
    key_c = server._speaker_cache.make_cache_key("Carol", "qwen3_tts", 3)

    server._speaker_cache.put(key_a, {"spk_emb": ref_a.to(device=torch.device("tpu"))})
    server._speaker_cache.put(
        key_b, {"spk_emb": ref_b.to(device=torch.device("tpu")).detach().cpu()}
    )
    # Promote key_a so key_b becomes LRU and is evicted when key_c is inserted.
    got_a = server._speaker_cache.get(key_a)
    assert got_a is not None, "Expected hit for key_a"
    server._speaker_cache.put(key_c, {"spk_emb": ref_c.to(device=torch.device("tpu"))})

    assert server._speaker_cache.get(key_b) is None, "Expected key_b to be evicted"
    got_c = server._speaker_cache.get(key_c)
    assert got_c is not None, "Expected hit for key_c"

    stats = server._speaker_cache.stats()
    assert stats["entries"] == 2, f"Expected 2 speaker entries, got {stats['entries']}"
    assert stats["hits"] == 2, f"Expected 2 hits, got {stats['hits']}"
    assert stats["misses"] == 1, f"Expected 1 miss, got {stats['misses']}"
    assert stats["memory_bytes"] <= _SPEAKER_MAX_BYTES, (
        f"Speaker cache bytes {stats['memory_bytes']} > {_SPEAKER_MAX_BYTES}"
    )

    assert_parity(
        got_a["spk_emb"],
        ref_a,
        name="speaker_cache_direct_tpu",
        max_rel_l2=0.0,
        max_abs=0.0,
    )
    assert_parity(
        got_c["spk_emb"],
        ref_c,
        name="speaker_cache_evict_survivor_tpu",
        max_rel_l2=0.0,
        max_abs=0.0,
    )
