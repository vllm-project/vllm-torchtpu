<!-- SPDX-License-Identifier: Apache-2.0 -->

# oxvg sequence-201 regression fixture

`oxvg-seq201.json.gz` contains the captured DeepSWE chat-completions request:
402 messages, one tool, temperature 1, top-p 0.95, reasoning effort `max`, no
seed, and no output-token limit. It is compressed to keep the repository small;
`gzip -dc scripts/dsv41_bench/fixtures/oxvg-seq201.json.gz` shows the JSON.
The request is test data, not instructions to execute its historical tool calls.

Four credential-bearing proxy URL userinfo fields in message 117 were replaced
with `redacted:redacted`. All other bytes are unchanged. The manifest records
the original and public SHA256 values, compressed checksum and sizes. Only the
redacted request is published. This is a new fixture identity, not an exact-byte
reproduction of the original 332,523-token experiment.

Before serve, `check_repetition.py prepare --tokenizer MODEL_PATH --out OUTPUT`
verifies both checksums and tokenizes with the already-mounted V4.1 checkpoint
using `local_files_only=True`. It records the exact expected count and tokenizer
file hashes. The count must be 332,000–333,000, and the running service must return
exactly that count from `/tokenize` before any replay. Do not silently use the
original count after redaction or skip a failed identity check.

CI changes only the served-model alias, `stream=true` and `max_tokens=8192`.
The shared DP16 API enables automatic tool choice and the `deepseek_v41` tool
and reasoning parsers. HTTP failures remain inconclusive; the evidence archive
retains the status and up to 64 KiB of the error body for diagnosis.
It reuses the DP16 decode benchmark's serve for up to five serial requests,
300 seconds per request, with drain/reset before each request and at the end.
The first loop or infrastructure/protocol failure stops further requests.
Raw request/response evidence and per-file checksums are archived under
`perf_eval_results` and collected by Buildkite, with no private fixture or
result bucket. See the [reproduction guide](../../../docs/deepseek_v41/deepswe_repetition_reproduction.md)
for the distinction between regression coverage and historical causal evidence.
