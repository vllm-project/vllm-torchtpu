<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSWE long-context repetition: reproduction and CI handoff

The frozen oxvg sequence-201 request exposes sustained repetition after
[#2566](https://github.com/vllm-project/vllm-torchtpu/pull/2566), commit
`8421301cf61f9e9ef2b08881cbd11832710136b1`. The November 6 investigation completed
an adjacent-source comparison and a full rollback/restoration experiment.
Those bounded observations support reverting the PR; they do **not** identify
a particular kernel defect or prove all historical DeepSWE failures share it.

This change reverts #2566 on the current epic branch and adds a regression check
to the existing DP16 decode benchmark server. Conflict resolution keeps later
SWA-only attention, steady/drain pipelining, HBM output placement and blocked
inverse-RoPE gathering; it removes fused QA/QB/compressor preprocessing, CSA2
in-place output/KV owners and the other changes introduced by #2566, including
its Engram hash-download deferral and MoE metadata-wait scheduling. It is an
adapted revert on the current epic, not a checkout of the historical parent tree.
No production numerical fix is claimed.

## Shared-server CI regression

The `dsv41_flash_dp16_c768_decode` Buildkite step now runs, in order:

1. Before the two-host serve, decompress the repository fixture, verify both
   compressed and request SHA256, and tokenize locally with the mounted V4.1
   checkpoint. Missing or changed data fails the job; no Falcon bucket is used.
2. Start the existing decode-benchmark serve once, run its original C768 formal
   performance measurement, and keep both hosts alive.
3. Drain/reset all 16 engines, then issue up to **five serial** seq201 requests.
   Each performs the full roughly 332K-token prefill plus bounded decode. Reset
   between requests and after the suite. Never execute generated tool calls.
4. Stop sending requests on the first sustained-loop candidate, HTTP/protocol
   failure, deadline or unsuccessful drain/reset. Archive evidence under the
   existing `perf_eval_results` Buildkite artifact mount;
   then stop both hosts through the existing coordinator. The final CI audit
   requires all five checks, successful resets and the local archive SHA256.
   Buildkite then uploads that archive through its existing artifact collection.

The implementation is
[`check_repetition.py`](../../scripts/dsv41_bench/server_steps/check_repetition.py).
It changes only the fixture's served-model alias, `stream=true`, and
`max_tokens=8192`. Temperature 1, top-p 0.95 and the absence of a seed are retained.
The redacted fixture SHA is fixed; every effective request and received SSE
body has its own recorded SHA. The exact token count is computed with the CI
checkpoint before serve, recorded with both tokenizer file hashes, and must
match the service's `/tokenize` result. It must also lie in 332,000–333,000: the
redaction changes tokenization slightly, but must not shorten the long prefill.

| Setting | Regression policy |
| --- | --- |
| Attempts | 5 serial requests, no retries or extra samples |
| Per-request bound | 300 seconds wall time, including prefill, and 8,192 generated tokens |
| Drain/reset | 60 seconds each; all 16 engines must be idle; no forced cancellation |
| Whole checker | 2,100 seconds, including resets and local evidence archiving |
| Loop detection | 1–256 word periods, at least 8 repeats/32 words; unseparated character loops; sustained punctuation loops |
| Valid bounded result | Supported finish reason plus `[DONE]`, no detected loop, valid complete tool JSON where applicable |
| Failure classification | A loop fails the regression; HTTP, malformed/incomplete SSE, timeout, identity mismatch or reset failure is inconclusive and also fails CI |

Five attempts reflect the observed third-request failures in the adjacent,
restored and instrumented sessions. Normal historical completions took roughly
37–254 seconds, so 300 seconds retains that observed range. Five full time
budgets are 25 minutes; normal early completions shorten this. The checker has
35 minutes total including reset/archive overhead. Neither an additional
30–40 minutes nor the entire CI wall time is guaranteed: loading, compilation,
the formal benchmark and the existing job deadline still apply.

The current CI recipe stays TP1/DP16/EP16, 8 physical chips/2 hosts, push SC2/TC1,
**dynamic EPLB enabled**, first-layer Engram in HBM, DSpark K5, 48 sequences per
rank and compile buckets 288/2048. Both hosts' model window increases from
98,816 to **524,288**, because the existing 96K benchmark window cannot accept
the 333K fixture. Formal benchmark inputs remain 96K prefix/512 output and C768;
its timing window ends before these checks. The larger model window can affect
memory, compilation and performance: record it with results rather than treating
old 98K-window measurements as an identical baseline. No throughput threshold
is relaxed by this change.

The current EPLB-on recipe is intentionally different from the historical
EPLB-off triage. This latest-config mitigation plus CI still requires actual
TPU validation, including resource fit at 512K. A five-request pass is bounded
coverage, not proof that repetition is absent. A `length` finish is a completed
bounded observation, not a complete tool invocation or a correct oxvg solution.
The 8,192-token cap is a new CI limit, not part of the historical causal replay;
later-onset failures remain outside its coverage.

### Fixture access and artifacts

The fixture is included as
[`oxvg-seq201.json.gz`](../../scripts/dsv41_bench/fixtures/oxvg-seq201.json.gz),
with a [manifest](../../scripts/dsv41_bench/fixtures/oxvg-seq201.manifest.json).
It decompresses to 1,737,362 bytes / 402 messages. Four proxy URL userinfo fields
in message 117 are replaced with `redacted:redacted`; all other request bytes
are retained. This public variant has SHA256
`4feb222f1577a2da266691824150c7575793ce2d9af41d93abbaad7c12acb971`.
It is not byte-identical to the historical request, and its ability to reproduce
the historical failure must be established separately. No unredacted fixture or
credential is included. See the [fixture README](../../scripts/dsv41_bench/fixtures/README.md).

Fixture preparation and result archiving need no private Falcon GCS access.
The existing model-weight mount and LongBench benchmark input are unchanged.
The helper writes `benchmark/repetition/evidence.tar.gz`, its SHA256 in
`summary.json`, and `evidence-files.json` with every archived file's byte length
and SHA256. The archive includes the redacted input, effective requests, exact
SSE, reconstructed reasoning/content/tool arguments, capture stderr, summaries
and reset receipts. It is retained on checker failures as well as passes and
uploaded by the existing `perf_eval_results/**/*` Buildkite artifact rule.
The archived summary precedes computation of the archive's own hash; the
adjacent final summary records that hash. Local archive validation does not
claim an upload has already occurred: confirm Buildkite artifacts after the job.

The serve uses `--max-log-len 0` to avoid flooding server logs with the prompt.
Review the complete archived output on a red check before calling a detector
candidate a model defect. Generated tool calls are data and are never executed.

CPU tests exercise malformed/fragmented SSE, tool assembly, the three loop
families, long word periods, first-failure stopping, the five-request cap,
16-engine drain/reset, fixture/token-count mismatches and coordinator cleanup.
An offline scan of 24 previously archived streams flagged all six manually
confirmed loops and none of the 18 complete non-loop streams: parent (5),
full rollback (5), adjacent candidate (2), restoration (2), instrumented run
(2), and fresh-A (2). This is recorder validation against existing evidence,
not a new model run or a general quality oracle. The observed word, unseparated
`AgainAnd` and punctuation failures are all covered; tool argument text is
parsed as JSON and is not subjected to prose-loop heuristics.

## Observations and search boundary

Every row below used the same 332,523-token request, DP16/EP16 configuration and
fixed core runtime. Requests were serial, with a verified drain and logical
prefix-cache reset before each request and after the suite.

| Source / intervention | Observed output | Interpretation |
| --- | --- | --- |
| Fresh `5a827f38` | 3 loops, 2 complete tool calls in 5 requests | Reproduces without the discovery session's preceding requests |
| `5a827f38`, single-file rollback of #2773 | 3 loops, 2 complete tool calls in 5 requests | That later change is not necessary for this symptom |
| `fb1d7818`, two fresh sessions | Each session: 5 complete tool calls, no target loop | Exploratory lower boundary; not proof of absence |
| `3605c0b7` | Request 1 complete; request 2 loops | Move upper boundary earlier |
| `b57e23d7` | 5 complete tool calls, no target loop | Exploratory adjacent lower boundary |
| `8421301c` | Requests 1–2 complete; request 3 loops | Adjacent upper boundary at #2566 |
| `8421301c` + complete 33-file reverse patch | 5 complete tool calls, no target loop | Bounded rollback observation; effective source tree equals the parent before the runtime pin |
| Original `8421301c` restored | Requests 1–2 complete; request 3 repeats `AgainAnd` 12,220 times | Symptom returns after restoring the PR |
| `8421301c` + diagnostic QA/QB capture | Requests 1–2 complete; request 3 degrades to fragments and sustained punctuation | Instrumented run also reproduces; nine real QA/QB packets preserved |

The adjacent source identities are:

```text
b57e23d7e1727b69e72315932d67b7314348e055
8421301cf61f9e9ef2b08881cbd11832710136b1
```

At `8421301c`, the first two requests completed in 253.82 and 63.76 seconds.
The third degraded from relevant analysis to unrelated fragments and repeated
variants of "Then/Than you've got more". Capture stopped at 62.48 seconds;
the final two-word period repeated at least 16 times. The positive finding came
from reviewing the complete received text, not from a timeout or detector flag.
Requests 4–5 were not sent. Raw received SSE SHA256 for that third request:

```text
749b0e1fc5ae34c4ec388071a06ad7520c86adcf442eb53327e0ae73fa5ec619
```

Later SWA #2720 and dynamic EPLB #2523 changes are not necessary for the
`8421301c` symptom. This does not exclude existing SWA, MoE, CSA or their
interactions. #2566 changes 33 files, including integration and cache ownership;
its title does not identify which mechanism is responsible.

Two earlier rollback launches failed before requests (oversized shell argument
and patch-text assertion). The third verified the effective source tree on both
hosts and completed five requests without the loop. The restored PR reproduced
it on request 3. All associated request/SSE hashes, complete text, tool JSON,
per-request/final resets and archives were reviewed before resource release.
The failed launches remain infrastructure failures, not model-quality evidence.

## Fixture and runtime identity

The fixture is a captured chat-completions request from oxvg, sequence 201:
402 messages, one tool definition, `temperature=1`, `top_p=0.95`,
`reasoning_effort=max`, and no explicit seed or output-token limit. The
original JSON has no `stream` field. The replay changes only `stream=true`.
The prompt token count is from the recorded service; identical text with a
different tokenizer or chat template is a different experiment.

The identities below describe the **historical unredacted experiment**, not
this PR's public CI fixture. CI uses the credential-redacted variant described
above. The unredacted request and old generated trajectories remain outside
the repository; their hashes retain the provenance of the causal experiments.
Do not treat the public variant's first CI result as a rerun of those exact bytes.

| Input | SHA256 |
| --- | --- |
| Original request bytes | `45871ebc0e5c9adb23e877b82bb605749a0124c3bd4b5f998412ee6e2adb4f1f` |
| Effective streaming bytes | `db69ae0dfa85b65605b34b7636902e72ea3f19d4d27cb4bda4b0395f90bc49ad` |
| Model `config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `tokenizer_config.json` | `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547` |
| `tokenizer.json` | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |

The model was DeepSeek-V4.1-Flash. These metadata hashes are not a full weight
checksum; also record the checkpoint revision and weight manifest in any new
run. Recorded runtime on both hosts:

| Component | Version / source |
| --- | --- |
| vLLM | `e37e51dd246cf421c554e7d6f53e178e3fd29085` (`0.1.dev1+ge37e51dd2.empty`) |
| torch / torch-tpu | `2.13.0+cpu` / `0.2.0rc1` |
| JAX / jaxlib / libtpu | `0.11.2` / `0.11.2` / `0.0.50` |
| tpu-sync-torch | `0.0.1.dev20261031081414` |
| NumPy / transformers | `2.3.5` / `5.18.0` |

For older sources, the `pyproject.toml` tpu-sync dependency metadata was
explicitly pinned from `20261012084241` to `20261031081414`. Preserve that diff
separately from source interventions. Verify installed versions on both hosts;
matching manifests alone is insufficient. This is a matched-runtime source
comparison, not a recreation of an earlier evaluation image.

## Historical server configuration

Use a dedicated service: **8 physical TPU7x chips across 2 hosts**, with
8 JAX devices per host, TP1/DP16/EP16. This was push SC2/TC1, dynamic EPLB off,
first-layer Engram in HBM, FP8 KV, CED and DSpark K5, with a 524,288-token
window. A single-host DP8 run or the throughput CI's dynamic-EPLB configuration
does not reproduce this configuration.

In the pinned environment, set `MODEL` to the approved checkpoint path on both
hosts, `HEAD_IP` to the first host's address, and `HOST_RANK` to 0 or 1. Use
fresh writable cache directories for the run. The following is the recorded
launch with deployment-specific paths parameterized:

```bash
set -euo pipefail
: "${MODEL:?checkpoint path}"
: "${HEAD_IP:?head host address}"
: "${HOST_RANK:?0 or 1}"
: "${RUN_CACHE:?fresh writable cache directory}"
[[ "$HOST_RANK" == 0 || "$HOST_RANK" == 1 ]]

unset MEGASCALE_COORDINATOR_ADDRESS MEGASCALE_NUM_SLICES MEGASCALE_PORT MEGASCALE_SLICE_ID
unset TPU_PROCESS_ADDRESSES TPU_PROCESS_PORT TPU_WORKER_HOSTNAMES
unset TORCH_TPU_SLICEBUILDER_ADDRESSES TORCH_DEVICE_BACKEND_AUTOLOAD LD_PRELOAD
unset TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD
export TPU_MULTIHOST_BACKEND=mp TPU_SKIP_MDS_QUERY=1 TORCH_TPU_BASE_PORT=19000
export MASTER_ADDR="$HEAD_IP" MASTER_PORT=19700
export TORCH_TPU_DP_MASTER_ADDR="$HEAD_IP" TORCH_TPU_DP_MASTER_PORT=19700
export VLLM_TARGET_DEVICE=tpu OMP_NUM_THREADS=1 NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_USE_AOT_COMPILE=0 TPU_RECLAIM_COMPILE_MEMORY=1
export VLLM_DISABLE_COMPILE_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export RUNAI_STREAMER_MEMORY_LIMIT=34359738368 TPU_SHARDED_LOAD_SYNC_EVERY=512
export VLLM_ENGINE_READY_TIMEOUT_S=7200 VLLM_SERVER_DEV_MODE=1 TPU_ACCELERATOR_TYPE=tpu7x
export TPU_DEEPSEEK_V41_ENGRAM_HBM=first TPU_MOE_EPLB_DYNAMIC=0
export MOE_FUSED_EP_KERNEL_IMPL=push MOE_FUSED_EP_PREPROCESS_BACKEND=topology
export MOE_FUSED_EP_PUSH_SC_PREPUSH=2 MOE_FUSED_EP_PUSH_TC_EXCHANGE=1
export LIBTPU_INIT_ARGS='--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0 --xla_tpu_dvfs_p_state=7 --xla_mosaic_unsafe_allow_multicore_remote_dma=true'
export VLLM_CACHE_ROOT="$RUN_CACHE/vllm" TORCHINDUCTOR_CACHE_DIR="$RUN_CACHE/inductor"

host_args=(--api-server-count 1)
if [[ "$HOST_RANK" == 1 ]]; then
  host_args=(--headless --data-parallel-start-rank 8)
fi
vllm serve "$MODEL" --tokenizer "$MODEL" --load-format runai_streamer \
  --served-model-name deepseek-v4.1-flash deepseek-v41 --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --data-parallel-size 16 --data-parallel-size-local 8 \
  --data-parallel-address "$HEAD_IP" --data-parallel-rpc-port 19500 \
  --enable-expert-parallel --enable-ep-weight-filter --language-model-only \
  --async-scheduling --enable-prefix-caching --swa-bounded-replay \
  --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype":"fp8"}' \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
  --additional-config '{"deepseek_v41":{"execution":"ced","moe":"fused_ep_w4a8"}}' \
  --block-size 64 --max-model-len 524288 --max-num-batched-tokens 2048 --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 --compilation-config '{"compile_sizes":[288,2048]}' \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"model\":\"$MODEL\"}" \
  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice --default-chat-template-kwargs '{"reasoning_effort":"max"}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --trust-remote-code "${host_args[@]}"
```

Record actual argv, relevant environment, source HEAD and tracked diff, installed
runtime, checkpoint metadata and startup logs on **both** hosts. The headless
worker does not expose an independent port-8000 health endpoint.

## Historical replay protocol

Use the head's root URL as `SERVER_URL`, without `/v1`. Do not replay on a
shared service: resetting its prefix cache would affect other users.

1. Require `/health` HTTP 200 and `/v1/models` reporting `max_model_len=524288`.
   Inspect `/metrics`: each of `vllm:num_requests_running`,
   `vllm:num_requests_waiting` and `vllm:kv_cache_usage_perc` must contain exactly
   the `engine` labels `0` through `15`. Running Pods do not establish readiness.
2. Before **each** request, wait at most 180 seconds for all three metrics to
   be zero on all 16 engines. Save the metrics, then POST `{}` to
   `/reset_prefix_cache?reset_running_requests=false&reset_external=false`.
   Require JSON `success: true`, then save and recheck all three metrics.
   Stop if any engine is missing or the reset fails. This is a logical cache
   reset, not HBM zeroing or a model restart; never force-cancel other requests.
3. Verify and serialize the approved fixture as below. Keep all original
   fields, tool definitions and history; do not add a seed, change a sampling
   parameter, truncate history or execute returned tool calls.
4. Send one streaming request, bounded to 300 seconds. Preserve every received
   SSE line. Stop further requests at the first repetition candidate and review
   the full received output. Otherwise send at most five requests, repeating
   the drain/reset between them. HTTP/protocol errors or incomplete output
   without a confirmed loop are inconclusive; stop and investigate them.
5. Drain and reset after the last request, then archive the results. Leave
   enough session budget for that drain and archive; do not extend the resource
   deadline to finish the suite.

Prepare the effective request in a new result directory:

```bash
python3 - "$REQUEST_JSON" "$RESULTS" <<'PY'
import hashlib
import json
import pathlib
import sys

original = pathlib.Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(original).hexdigest() == "45871ebc0e5c9adb23e877b82bb605749a0124c3bd4b5f998412ee6e2adb4f1f"
payload = json.loads(original)
assert "stream" not in payload
payload["stream"] = True
effective = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
assert hashlib.sha256(effective).hexdigest() == "db69ae0dfa85b65605b34b7636902e72ea3f19d4d27cb4bda4b0395f90bc49ad"
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=False)
(out / "original-request.json").write_bytes(original)
(out / "effective-request.json").write_bytes(effective)
PY
```

After the mandatory readiness/drain/reset checks, this command captures one
bounded request. It does not implement the checks or the early-stop detector:

```bash
set +e
curl --silent --show-error --no-buffer --connect-timeout 10 --max-time 300 \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${OPENAI_API_KEY:-EMPTY}" \
  --data-binary "@$RESULTS/effective-request.json" \
  --dump-header "$RESULTS/response.headers" \
  --output "$RESULTS/response.sse" \
  "$SERVER_URL/v1/chat/completions" 2>"$RESULTS/curl.stderr"
capture_rc=$?
set -e
printf '%s\n' "$capture_rc" > "$RESULTS/curl.exit"
```

The original stop detector inspected the last 3,000 whitespace-separated words
in each `content`, `reasoning` and `reasoning_content` stream. It flagged a
period of 1–256 words repeated at least eight times and spanning at least
32 words, checked approximately every five seconds. This is a stopping aid,
not an automatic quality oracle. A 16-word maximum missed an observed
176-word loop; reaching 300 seconds was not the evidence for that loop.

Reconstruct text from the raw SSE by concatenating each choice's deltas in
order, **including `delta.reasoning`**. Compare it byte-for-byte with any
incremental text snapshot. Require `[DONE]`, valid events and complete tool
arguments before calling an output complete. Parse tool arguments as data
only. A complete tool call is not a successful oxvg solution. A five-request
suite with no observed loop is a bounded observation, not an exclusion or five
independent statistical trials.

## Evidence to attach when contributing a result

Use a stable result identifier and preserve the following files together:

```text
identity/                 # both hosts: source, patches, runtime, argv, model
startup/                  # both hosts: complete loading/compile/serve logs
run-01/                   # repeat only for requests actually sent
  original-request.json
  effective-request.json
  reset.json              # before/after metrics and reset response
  response.headers
  response.sse
  reconstructed-text.json
  result.json             # time, status, finish reason, hashes, review verdict
final-reset.json
review.md                 # full-output review and exact reason for classification
manifest.json             # path, byte length and SHA256 of every archived file
```

Verify the archive checksum and every file checksum after download, before
releasing the service. The adjacent parent and candidate archives contained
66 and 52 files respectively; raw request/SSE hashes, reconstructed text,
tool-argument JSON and per-request/final resets were checked. Retain failed
launches as separate records with **zero requests**, never as quality failures.

For public discussion, attach a reviewed summary with source/runtime identity,
request hash, full versus partial completion, request index, loop evidence and
artifact checksums. This PR publishes only the reviewed, credential-redacted
fixture and its new CI outputs. Other historical raw traces require a separate
review before redistribution. Distinguish new measurements from this snapshot.

## Remaining kernel investigation

Single-kernel tests on exact `8421301c` did not isolate the defect. T288/T2048
QA/QB/compressor, ragged CSA2/chunk/owner reuse, two-step Torch bridge, KV-only
and draft-T40 checks completed. QB matched the old 512K fallback bitwise. QA
had small BF16 differences. The long compressor case differed at three effective
FP8 bytes, each by one adjacent FP8 value; scale was equal, and CPU reconstruction
crossed quantization midpoints. FP32 ring max absolute difference was
`4.76837158203125e-7`; near-zero ULP distances were larger. These observations
do not establish the cause of the generation loop.

The final instrumented DP16 run preserved nine real QA/QB packets: a final
T2048 prefill bucket with 1,395 real rows and two early T288 decode buckets with
six real rows for each of three requests. Packet arrays, metadata and provenance
were verified and archived. These are early-step samples from sessions including
a failing request, not proof they contain the first divergent operation.
Old/new/independent-reference replay of these actual packets has not run.
Compressor/ring/cache state and the full later failure trajectory remain gaps.
All remote investigation resources were released; no further experiment is
started by this documentation change.

A future kernel fix should compare the previous implementation, candidate and
independent reference on the same real inputs and state, then return to the
frozen failing request. Preserve dtype, positions, scales, page/ring mappings
and untouched-cache sentinels. For alias suspicions, include the compiled Torch
bridge and consecutive state updates. Once a localized failing case exists,
add it to the TPU parity lane as well as retaining this end-to-end guard.
