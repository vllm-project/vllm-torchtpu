#!/usr/bin/env python3
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

"""Pull a model's aux files and import its remote code once on this host.

Usage: warm_remote_code.py <object-storage model URI>

For a pod of a multi-host engine other than index 0, run before the engine
starts its workers there. On index 0 the API server pulls the aux files
and imports the remote code before any worker exists. On the other hosts
every worker would otherwise copy the same modules into an empty HF modules
cache at once, and one can import a half-written file - for Kimi-K3,
"tokenization_kimi has no attribute 'TikTokenTokenizer'".

The pull is the plugin's own (runai_repull, which runs upstream's pull under
its lock and sentinels), so the workers find the directory complete and the
modules cached, and only import.
"""

import sys
import types

from transformers import AutoConfig, AutoTokenizer
from vllm.config import ModelConfig

from vllm_torchtpu.worker.runai_repull import ensure_runai_aux_files


def main(uri: str) -> None:
    # The three fields the pull reads and rewrites, as a worker's config has
    # them before its own pull.
    config = types.SimpleNamespace(model=uri, model_weights=None, tokenizer=uri)
    config.maybe_pull_model_tokenizer_for_runai = types.MethodType(
        ModelConfig.maybe_pull_model_tokenizer_for_runai, config
    )
    ensure_runai_aux_files(config)
    AutoConfig.from_pretrained(config.model, trust_remote_code=True)
    AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    print(f"remote code for {uri} imported from {config.model}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <object-storage model URI>")
    main(sys.argv[1])
