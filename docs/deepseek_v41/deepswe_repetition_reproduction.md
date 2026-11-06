<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSWE long-context repetition: reproduction and CI handoff

This is an investigation snapshot from 2026-11-06 UTC. A frozen oxvg request
reproduced sustained repetition in a fresh DeepSeek-V4.1-Flash service. Source
bisection reached the adjacent commits `b57e23d7` and `8421301c`
([#2566](https://github.com/vllm-project/vllm-torchtpu/pull/2566)).
The specific defect and a production fix are **not established**. Full rollback
and restoration validation remain incomplete. This document provides the
protocol and evidence boundaries so another contributor can reproduce, review,
or reduce the case without repeating the entire agent evaluation.

The commands below describe the historical experiment, not the defaults of the
current epic branch. Keep the source, runtime, model, request and topology
identities together when comparing results.

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

The complete #2566 rollback attempts failed before model requests: first an
oversized shell argument, then a source-patch text assertion. They provide no
model-quality result. A future rollback must verify the resulting tree rather
than assume that equal HEADs or differently serialized patches imply equal
effective source.

## Fixture and runtime identity

The fixture is a captured chat-completions request from oxvg, sequence 201:
402 messages, one tool definition, `temperature=1`, `top_p=0.95`,
`reasoning_effort=max`, and no explicit seed or output-token limit. The
original JSON has no `stream` field. The replay changes only `stream=true`.
The prompt token count is from the recorded service; identical text with a
different tokenizer or chat template is a different experiment.

**The raw request and generated trajectory are not included in this PR.**
They contain a full agent conversation and require a redistribution review.
Contributors who already have an approved copy can use the hashes below.
Otherwise request a reviewed fixture through this PR before claiming an exact
reproduction. A hash identifies data; it does not make the case self-contained.
Do not upload private paths, credentials, bucket locations or unreviewed agent
traces into public CI artifacts.

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

## Replay protocol

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
artifact checksums. Keep the raw trace in an approved location until it is
cleared for redistribution. Distinguish new measurements from this snapshot.

## Kernel reduction and CI proposal

Single-kernel checks on exact `8421301c` have not yet isolated the model defect.
The completed synthetic checks include T288 compressor, QA/QB, and CSA2
chunk/direct and owner-reuse comparisons; a T2048 CSA2 case covered ragged
request boundaries and all-padding chunks. Those pipeline comparisons matched
the direct path, with independent mathematical references also checked.
The T2048 short compressor's updated FP32 ring differed by at most
`4.76837158203125e-7`, while effective latent/FP8 outputs matched. A longer
T2048 case produced three differing packed FP8 words and is under review.
Neither these differences nor the passing cases establish an E2E cause.
Actual failing intermediate model tensors have not been captured.

Reuse the existing correctness suites when reducing a concrete failure:

```bash
python -m pytest -q tests/kernels/attention_preprocessing/test_compress_rope.py
python -m pytest -q tests/kernels/attention_preprocessing/test_qa_rope.py
python -m pytest -q tests/kernels/attention_preprocessing/test_qb_rope.py
python -m pytest -q tests/kernels/deepseek_v41/test_csa2_pipeline.py
```

These commands are entry points for the checked-out source, not a claim that
the current epic tests reproduce the historical loop. Require actual TPU
execution for DMA/kernel cases and inspect skipped tests. For a new numerical
case, preserve inputs and outputs, masks, positions, ring/page mappings,
scales and untouched-cache sentinels. Compare the previous implementation,
candidate and independent mathematical reference on the **same** inputs.
For a bridge/alias suspicion, include the actual compiled Torch bridge and
consecutive state updates, not only a stateless JAX call.

At epic base `9311c49ab`, the current
[pipeline](https://github.com/vllm-project/vllm-torchtpu/blob/9311c49ab8f0a02b02afc62d37857967ba50a764/.buildkite/pipeline_dsv41.yml) has CPU contracts, a TPU parity
step, DP8 quality, DP16 C768 throughput with dynamic EPLB, and cached-extend
performance. The
[TPU parity runner](https://github.com/vllm-project/vllm-torchtpu/blob/9311c49ab8f0a02b02afc62d37857967ba50a764/.buildkite/scripts/run_dsv41_decode_kernel_checks.sh)
selects scorer, indexer-route and MoE-scale checks. It does not run the frozen
seq201 protocol above. Green throughput or kernel CI is not this regression's
quality result.

Proposed integration, with no new CI lane enabled by this document:

1. **CPU capture-contract tests:** add fixtures for SSE reasoning/tool assembly,
   malformed events, incomplete streams, the 176-word repeat pattern and
   missing/non-idle DP engines. Wire these into the existing CPU benchmark
   harness step; they validate the recorder, not model quality.
2. **TPU regression unit test:** once a real failing tensor/state is isolated,
   reduce it into the existing preprocessing or CSA2 suite, prove it fails
   without the fix, and explicitly select it in the TPU parity runner. Preserve
   the real dtype, shape and state boundary; a synthetic pass is not a substitute.
3. **Opt-in DP16 diagnostic lane:** first publish an approved, immutable fixture
   with licensing/access and hashes, then implement the bounded replay protocol
   on a dedicated two-host allocation. Always upload reviewed diagnostics for
   pass, candidate and inconclusive outcomes. Start as a diagnostic lane, not a
   probabilistic merge gate. Define the reviewed oracle and baseline behavior
   before promoting it to a required check.

Useful contributions now are a matched replay of the adjacent sources, a
minimal failing tensor/state capture, review of the 33-file integration change,
or a fixture/CI implementation following this protocol. Label changes to seed,
runtime, topology, checkpoint, CED, Engram or MoE mode as separate experiments.
Finish with the smallest rollback/restoration check needed to demonstrate
causality; do not expand the matrix simply to accumulate passing runs.
