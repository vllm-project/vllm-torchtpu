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

    python3 -m scenario.runtime.warm <object-storage model URI>

A Plan runs this before the server on every host of an mp role but the
first, when the model trusts remote code and lives in object storage
(P-19). On the first host the API server pulls the aux files and imports
the remote code before any engine worker exists. On the other hosts every
worker would otherwise copy the same modules into an empty HF modules cache
at once, and one can import a half-written file: for Kimi-K3,
"tokenization_kimi has no attribute 'TikTokenTokenizer'".

The pull is vllm_torchtpu's own (runai_repull, which runs vLLM's pull under
its lock and sentinels), so the workers find the directory complete and the
modules cached, and only import.

This is the one runtime module that imports vLLM and transformers, and only
when it runs, so importing it costs nothing.
"""

import sys
import types

USAGE = "usage: python3 -m scenario.runtime.warm <object-storage model URI>"


def warm(uri: str) -> str:
    """Pull `uri`'s aux files and import its config and tokenizer; the pulled dir."""
    from transformers import AutoConfig, AutoTokenizer
    from vllm.config import ModelConfig

    from vllm_torchtpu.worker.runai_repull import ensure_runai_aux_files

    # The three fields the pull reads and rewrites, as an engine worker's
    # config has them before its own pull.
    config = types.SimpleNamespace(model=uri, model_weights=None, tokenizer=uri)
    config.maybe_pull_model_tokenizer_for_runai = types.MethodType(
        ModelConfig.maybe_pull_model_tokenizer_for_runai, config
    )
    ensure_runai_aux_files(config)
    AutoConfig.from_pretrained(config.model, trust_remote_code=True)
    AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    return str(config.model)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(USAGE, file=sys.stderr)
        return 2
    pulled = warm(argv[0])
    print(f"remote code for {argv[0]} imported from {pulled}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
