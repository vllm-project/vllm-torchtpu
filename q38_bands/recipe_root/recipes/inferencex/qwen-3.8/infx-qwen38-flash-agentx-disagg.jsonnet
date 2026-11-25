// infx-qwen38-flash-agentx-disagg — Qwen3.8-Flash-Next-FP8 P/D disaggregation (prefill and
// decode on two tpu7x hosts, Raiden stage-3 KV transfer), measured with the AgentX trace replay.
//
//   prefill : one 2x2x1 host, TP x DP = 8 with EP8 (TP1 / DP8 by default), kv_producer.
//   decode  : a second 2x2x1 host, the same TP, kv_consumer.
//   front   : ROUTER=proxy, examples/disagg/toy_proxy_server.py in a CPU pod of its own (:10000);
//             ROUTER=llmd, the llm-d Router (Envoy :8081 + EPP) plus the routing sidecar on each
//             decode pod. PREFILL_REPLICAS / DECODE_REPLICAS pods per leg (xPyD).
//   bench   : cdk.agentx, the client, dataset and parser of infx-qwen38-flash-agentx. The replay
//             goes to the front end; $VLLM_URL, its /health wait and the request-timing scrape
//             point at prefill pod 0.
//
//     cdk recipe render infx-qwen38-flash-agentx-disagg
//     cdk job create infx-qwen38-flash-agentx-disagg CLIENT_CONCURRENCIES=8 FAST=true
//     cdk job create infx-qwen38-flash-agentx-disagg CLIENT_CONCURRENCIES="16 4" DURATION=1800
//     cdk job create infx-qwen38-flash-agentx-disagg ENABLE_XPROF=true CLIENT_CONCURRENCIES=4
//     cdk job create infx-qwen38-flash-agentx-disagg PIPELINE_PARALLEL_SIZE_PREFILL=8 \
//       NUM_PREFILL_SERVERS=2 NUM_DECODE_SERVERS=4 ROUTER=llmd EPP_PROFILE=pd-soft-affinity ...
//     cdk job create infx-qwen38-flash-agentx-disagg TENSOR_PARALLEL_SIZE=8 ...  # see "Serving topology"
//     ... --map-dir <vllm-torchtpu tree>:prefill:/root/cloud-devkit/vllm-torchtpu \
//         --map-dir <vllm-torchtpu tree>:decode:/root/cloud-devkit/vllm-torchtpu
//
// The structure follows qwen3.5-397b-pd-agentx (the two legs as data, one server function
// called twice) and glm52-fp8-pd-agentx (cdk.agentx's benchJob driven through the toy proxy).
//
// What the image allows. qwen4_exp disaggregates only through its own connector,
// Qwen4ExpRaidenConnector (vllm_torchtpu/models/vllm/qwen4_exp/pd_connector.py, vllm-torchtpu
// 0329c4722, in the default image's f1161cd65), and models/vllm/qwen4_exp/config.py admits a P/D
// engine only at TP1 / DP8 with expert parallelism (vllm-torchtpu e7c69fe71 widens that to any
// TP x DP = 8; see "Serving topology" below), the unified FP8 pool, prefix caching off (the
// branch below lifts this) and none of the TPU_RAIDEN_*_ADMISSION flags; the connector also
// refuses MTP, PP and CP. So this
// is NOT the single-host shape (TP8 / DP1, MTP3, prefix caching on), and none of
// infx-qwen38-flash-agentx's tuned stack carries over by default: those flags were measured at
// TP8 with MTP and the concurrency-dependent auto rules assume it. What is carried over is the
// model's base server env (MoE kernels, QSA pool layout and decode tiling, RoPE and runner
// flags). Stack flags go in through EXTRA_SERVER_ENV (both legs) or
// PREFILL_EXTRA_SERVER_ENV / DECODE_EXTRA_SERVER_ENV.
//
// Prefix caching is off on both legs by default, so every AgentX turn re-prefills its whole
// context: the numbers are not comparable with the single-host curve, which runs at ~93% prefix
// hits. It is a per-leg switch on vllm-torchtpu caojx/qwen38-flash-agentx-disagg after
// "qwen4_exp PD: prefix caching on either leg" (the default image refuses it):
//   PREFILL_ENABLE_PREFIX_CACHING=true  the prefill leg reuses cached prefixes and ships the
//                                       running state of its align-mode state tables.
//   DECODE_ENABLE_PREFIX_CACHING=true   the decode leg adopts its local prefix and pulls only the
//                                       suffix (TPU_RAIDEN_PREFIX_AWARE_LOAD=1, set with it; needs a
//                                       tpu-sync reshard client with SUPPORTS_DST_SKIP_BYTES, or
//                                       every partially hit load fails and is recomputed locally).
//   ENABLE_PREFIX_CACHING=true          both.
//
// llm-d routing (ROUTER=llmd), as qwen3.5-397b-pd-agentx runs it (q38 copy, 2026-11-06). Every
// prefill pod is an endpoint of the EPP's prefill pool and every decode pod (behind its routing
// sidecar) one of the decode pool. Any prefill pairs with any decode per request: the producer's
// kv_transfer_params name its own Raiden store (src_controller_address, src_job_name,
// src_engine_id; tpu_connector.py _stage3_publish_send) and the sidecar (nixlv2 dialect) relays
// them to the decode verbatim, as the toy proxy does. Additions in this copy:
//   EPP_PATCH=band-a   IX's Qwen3.5 Band A routing (j-b271144e-a7a2-4a1d-8ba2: sessaff-spill plus
//                      an EPP_CONFIG_PATCH whose lists replace the preset's). Prefill: the
//                      prefix-cache affinity filter, then prefix-cache x2 + token load x1. Decode: a
//                      fail-open headroom filter (KV <= EPP_DECODE_HEADROOM_KV_UTIL, 0.80; running
//                      <= EPP_DECODE_HEADROOM_RUNNING, DECODE_MAX_NUM_SEQS - 2 as IX's 26 of 28),
//                      then x-dynamo-session-id affinity, then prefix x2 / active x3 / running x2 /
//                      KV x1. EPP_CONFIG_PATCH, if also given, is merged on top of it.
//   LLMD_EPP_IMAGE / LLMD_SIDECAR_IMAGE default to the digests IX's band runs pinned, not
//   cdk.llmd's mutable :main tags.
//   With PREFILL_REPLICAS > 1 the bench's per-level prefix-cache and request-timing deltas are
//   summed over every prefill pod (cdk.agentx's vllm_metric reads prefill pod 0 only); each read
//   also logs the per-pod values (Q38_PREFILL_METRIC lines in the bench log). Under ROUTER=llmd
//   read prefix_cache_hit_rate, not kv_cache_hit_rate: aiperf scrapes server metrics through
//   Envoy, which lands on a decode pod and reports its transferred (external) hits; that is the
//   0.99999 j-709f9b42-e4b2-4283-a1d0 published with prefix caching off.
// Decode prefix caching cannot be combined with DECODE_MTP=true: qwen4_exp/config.py refuses MTP
// on a P/D decode with prefix caching on.

local agentx = import 'cdk.agentx';
local cdk = import 'cdk.common';
local llmd = import 'cdk.llmd';

// --- General & container images ---------------------------------------------
local jobName = cdk.var('JOB_NAME', 'qwen3.8-flash-agentx-disagg');
// The single-host default image: vllm-torchtpu f1161cd65, which carries the qwen4_exp P/D
// connector. All three server containers (prefill, decode, proxy) run it, so the Raiden wire
// format cannot mismatch between the legs.
local serverImage = cdk.var('SERVER_IMAGE', 'us-central1-docker.pkg.dev/cloud-tpu-inference-test/inferencex/vllm:dongruizi-qwen38-flash-hostfinal-f1161cd65');
local clientImage = cdk.var('CLIENT_IMAGE', 'us-central1-docker.pkg.dev/cloud-tpu-inference-test/inferencex/inferencex-agentx:inferencex_bench_sep22');
local modelName = cdk.var('MODEL_NAME', 'Qwen/Qwen3.8-Flash-Next-FP8');

// q38 RECIPE-STD (2026-11-24; user, ~12:0x PM PST: one team branch and one standard recipe, so mainline work stops getting
// lost between iterations). Q38_STD=q38-1124 (the default) turns on the standard defaults marked "Q38_STD" below;
// Q38_STD=legacy renders exactly as recipe 9c615374 did. Every standard value is a DEFAULT only: a car that passes the
// variable renders as before, so per-band tuning stays an override. Most defaults apply only to the served P/D shape
// (a PIPELINE_PARALLEL_SIZE_PREFILL > 1 prefill feeding a TP > 1 decode); see reports/recipe_std_1124.md.
local q38StdName = cdk.var('Q38_STD', 'q38-1124');
assert std.member(['q38-1124', 'legacy'], q38StdName) : 'Q38_STD must be q38-1124 or legacy, got "%s"' % q38StdName;
local q38Std = q38StdName != 'legacy';
// The served P/D shape, read from the same variables as ppPrefill and tpFor('decode') further down (jsonnet locals
// cannot look ahead); the shape asserts below still police the real shape.
local q38StdPd = q38Std && cdk.numVar('PIPELINE_PARALLEL_SIZE_PREFILL', 1) > 1
  && std.parseJson(cdk.var('TENSOR_PARALLEL_SIZE_DECODE', cdk.var('TENSOR_PARALLEL_SIZE', '1'))) > 1;
// Q38_STD pools (user: "HBM KV 60GB", "KV offload 500GB"). The decode pool is counted in 512 KiB blocks (the TP8 decode's
// page_bytes=524288). Default 108544 = 53 GiB = 56.91 GB: the largest 1024-multiple that keeps >= 1 GiB of RUNTIME HBM
// headroom on the B1 mainline form with SB1-3's V1b (measured: B1 decode at 110592 blocks + 8 GiB staging peaks with
// 0.858 GiB strict headroom per rank, 0.164 GiB after V1b; colldcpg-hbm1 j-324ef168). The user's 60 GB (114688 blocks =
// 56 GiB = 60.13 GB, or 114441 = 60.00 GB) does NOT fit that form: B1 runtime headroom would be -1.14 GiB solo. A band
// that has the room passes Q38_STD_DECODE_POOL_BLOCKS (e.g. 114688) or DECODE_QWEN4_EXP_MAX_POOL_BYTES as an override.
local q38StdDecodePoolBlocks = cdk.numVar('Q38_STD_DECODE_POOL_BLOCKS', 108544);
assert q38StdDecodePoolBlocks % 1024 == 0 && q38StdDecodePoolBlocks >= 65536 && q38StdDecodePoolBlocks <= 122880 :
  'Q38_STD_DECODE_POOL_BLOCKS must be a multiple of 1024 in [65536, 122880] (108544 standard; 110592 posted B1 / B3; 114688 = the 60 GB reading), got %d' % q38StdDecodePoolBlocks;
local q38StdPrefillPoolBytes = '64424509440';  // 60 GiB with QWEN4_EXP_STATIC_POOL_FILL=1: every posted point (28 of 28)
local q38StdDecodeStagingBytes = '8589934592';  // 8 GiB carrier, the code default, rendered explicitly
local q38StdHostTierBytes = '500000000000';  // prefill host KV tier (TPURaidenOffloadingConnector), victim policy

// Refused by the image's P/D path (see the header); failing here beats failing at engine init.
assert !cdk.boolVar('MTP', false) : 'MTP=true: Qwen4ExpRaidenConnector requires speculative decoding off on the producer; DECODE_MTP=true runs it on the decode leg';
// DECODE_MTP=true: MTP on the decode leg only (vllm-torchtpu "decode-only MTP behind a staged
// decode"). The producer ships no draft KV, so the draft's prompt KV starts empty: meant for
// synthetic rejection sampling, which accepts drafts at a fixed mean length whatever they hold,
// as infx-qwen38-flash-agentx reports its curve (SYNTHETIC_ACCEPTANCE_LENGTH 3.32,
// MTP_NUM_SPECULATIVE_TOKENS 3; SYNTHETIC_ACCEPTANCE_LENGTH=0 keeps real acceptance).
local decodeMtp = cdk.boolVar('DECODE_MTP', q38StdPd);  // Q38_STD: on in the served P/D shape
// q38 MTP-AL (user ruling 11-10 ~01:00Z): SERVE_ENABLE_THINKING sets the served chat template's default thinking mode
// (--default-chat-template-kwargs, every leg). InferenceX serves this model thinking ON. Default false keeps today's
// renders byte-identical; new goal/dashboard runs and new served A/Bs set SERVE_ENABLE_THINKING=true. Per-request
// chat_template_kwargs (LongBench's enable_thinking) still win over this default.
local q38ServeThinking = cdk.boolVar('SERVE_ENABLE_THINKING', q38Std);  // Q38_STD: thinking ON (dash basis)
// q38 F2 (D-MK, reports/lane_dmk.md s29-s36): PREFILL_MTP_DRAFT_KV=true runs upstream prefill-only MTP (K=1) on the
// producer's last stage so it ships the draft layer's KV; both legs get QWEN4_EXP_PD_DRAFT_KV=1. Needs DECODE_MTP=true
// and a vllm-torchtpu tree with F2. Unset: nothing renders differently.
local prefillMtpDraftKv = cdk.boolVar('PREFILL_MTP_DRAFT_KV', q38StdPd && decodeMtp && !cdk.boolVar('PREFILL_TP2_STAGES', false));  // Q38_STD
assert !prefillMtpDraftKv || decodeMtp : 'PREFILL_MTP_DRAFT_KV=true needs DECODE_MTP=true';
local decodeMtpTokens = cdk.numVar('MTP_NUM_SPECULATIVE_TOKENS', 3);
// q38 (INT eval patch, q38_int_eval_patch.py): EVAL_ONLY / RUN_EVAL run cdk.agentx's built-in GSM8K or LongBench-v2 eval.
// Under eval: real MTP acceptance by default, the eval client goes to the front end (not prefill pod 0), MTP acceptance
// is read on the decode, and EVAL_MAX_MODEL_LEN reaches the client. Nothing renders differently when neither is set.
local q38EvalOn = cdk.boolVar('EVAL_ONLY', false) || cdk.boolVar('RUN_EVAL', false);
// q38 (INT, q38_int_evallimit_patch.py): EVAL_LIMIT=<n> under eval -> the bench env -> run_lm_eval's --limit n. Unset: nothing renders.
local q38EvalLimit = cdk.var('EVAL_LIMIT', '');
// Q38_STD: the InferenceX golden acceptance length for the chosen k (counts the bonus token): 2.32 at k=3 (as upstream
// 7b73226), 2.06 at k=2; any other k must pass SYNTHETIC_ACCEPTANCE_LENGTH. Legacy keeps 3.32.
local q38GoldenAl = if decodeMtpTokens == 3 then '2.32' else if decodeMtpTokens == 2 then '2.06' else '';
assert !q38Std || !decodeMtp || q38EvalOn || q38GoldenAl != '' || cdk.rawVar('SYNTHETIC_ACCEPTANCE_LENGTH', '') != '' :
  'Q38_STD: no golden acceptance length for MTP_NUM_SPECULATIVE_TOKENS=%d; pass SYNTHETIC_ACCEPTANCE_LENGTH' % decodeMtpTokens;
local decodeSyntheticAcceptance =
  local v = cdk.rawVar('SYNTHETIC_ACCEPTANCE_LENGTH', if q38EvalOn then '0' else if q38Std then q38GoldenAl else '3.32');
  if v == '0' then '' else v;
local decodeSpeculativeConfig =
  if !decodeMtp then ''
  else if decodeSyntheticAcceptance != '' && std.parseJson(decodeSyntheticAcceptance) > decodeMtpTokens + 1 then
    error 'SYNTHETIC_ACCEPTANCE_LENGTH=%s exceeds MTP_NUM_SPECULATIVE_TOKENS+1=%d' % [decodeSyntheticAcceptance, decodeMtpTokens + 1]
  else '{"method":"mtp","num_speculative_tokens":%d%s}' % [
    decodeMtpTokens,
    if decodeSyntheticAcceptance == '' then ''
    else ',"rejection_sample_method":"synthetic","synthetic_acceptance_length":%s' % decodeSyntheticAcceptance,
  ];

// --- TPU shape ---------------------------------------------------------------
local chipsPerHost = 4;
local accelerator = 'tpu7x';
local topology = '2x2x1';

// --- Request path and pool sizes (as qwen3.5-397b-pd-agentx) ------------------
// PREFILL_REPLICAS / DECODE_REPLICAS (or NUM_PREFILL_SERVERS / NUM_DECODE_SERVERS) pods per leg,
// one host each: xPyD. ROUTER=proxy (default): the toy proxy in a CPU pod of its own spreads
// requests over every pod of each leg (PROXY_* variables reach it as env). ROUTER=llmd: the llm-d
// Router (Envoy + EPP, file-discovery) as its own job and the llm-d routing sidecar on every decode
// pod; EPP_PROFILE picks a cdk.llmd preset (default pd-soft-affinity, as ix-tpu-pd-cand4-llmd-oct28)
// with its EPP_* knobs, EPP_CONFIG_PATCH / EPP_CONFIG_JSON as escape hatches. Replica i > 0 of a leg
// registers with Raiden as <role>-<i>-engine.
local router = cdk.var('ROUTER', 'proxy');
// q38 RECIPE-STD (user 11-23 / 11-24): decode routing sticky by a request header is not allowed going forward. Under
// Q38_STD, PROXY_DECODE_ROUTING=sticky needs the header-free opener key (PROXY_AFFINITY_KEY=opener; on the merged branch
// also Q38_PROXY_FORM=opener, since Benliang's proxy keys on PROXY_AFFINITY_HEADER only). Q38_STD=legacy re-runs a
// historic header-sticky point.
assert !q38Std || cdk.var('PROXY_DECODE_ROUTING', '') != 'sticky' || cdk.var('PROXY_AFFINITY_KEY', '') == 'opener' :
  'Q38_STD: PROXY_DECODE_ROUTING=sticky routes by a request header unless PROXY_AFFINITY_KEY=opener (header-sticky routing is not allowed); pass PROXY_AFFINITY_KEY=opener, or Q38_STD=legacy for a historic re-run';
local useLlmd =
  if router == 'llmd' then true
  else if router == 'proxy' then false
  else error 'ROUTER must be proxy or llmd, got "%s"' % router;
local prefillReplicas = cdk.numVar('PREFILL_REPLICAS', cdk.numVar('NUM_PREFILL_SERVERS', 1));
local decodeReplicas = cdk.numVar('DECODE_REPLICAS', cdk.numVar('NUM_DECODE_SERVERS', 1));
assert prefillReplicas >= 1 && decodeReplicas >= 1 : 'PREFILL_REPLICAS and DECODE_REPLICAS must be at least 1';
local eppProfile = cdk.var('EPP_PROFILE', 'pd-soft-affinity');
// EPP_PATCH: a named merge patch over EPP_PROFILE's preset (see the header); '' is none.
local eppPatchName = cdk.var('EPP_PATCH', '');
assert eppPatchName == '' || useLlmd : 'EPP_PATCH=%s needs ROUTER=llmd' % eppPatchName;
assert std.member(['', 'band-a'], eppPatchName) : 'EPP_PATCH must be band-a or empty, got "%s"' % eppPatchName;
// Per-chip throughput is divided by every host's chips.
local tpuChips = cdk.numVar('TPU_CHIPS', chipsPerHost * (prefillReplicas + decodeReplicas));
// q38 X222-SERVE (reports/x222_serve.md): PD_SLICE=2x2x2 serves the same engines on standard 2x2x2 slices. CDK refuses a
// 2x2x2 replicatedJob whose pods do not fill each slice (cloud-devkit-admin 8c38cf7 lib/kueue_tas.go:363), so the engine
// hosts, in role order (prefill-0..P-1, then decode-0..D-1), are packed two per slice: slice k is ONE replicatedJob
// pd<k> (replicas 2 x 4 chips) whose container vllm-tpu-pd runs the bootstrap below, which picks the host's role by
// JOBSET_JOB_INDEX, applies that role's own env and argv, forces IX's single-host TPU env (X222 v3) and sets
// JOBSET_JOB_INDEX to the role's original replica index. An odd tail host idles. TPU_CHIPS stays the used chips
// (4 x engine hosts). ROUTER=proxy only. Unset: nothing renders differently.
local pdSlice = cdk.var('PD_SLICE', '');
assert std.member(['', '2x2x2'], pdSlice) : 'PD_SLICE must be empty or 2x2x2, got "%s"' % pdSlice;
local pdSliceOn = pdSlice != '';
assert !pdSliceOn || !useLlmd : 'PD_SLICE=2x2x2 runs ROUTER=proxy only (the llm-d decode sidecar would share a slice pod with a prefill server)';
local pdRoles = [{ role: 'prefill', index: i } for i in std.range(0, prefillReplicas - 1)]
                + [{ role: 'decode', index: j } for j in std.range(0, decodeReplicas - 1)];
local pdNumSlices = std.ceil(std.length(pdRoles) / 2);
local pdPos(role, i) = if role == 'prefill' then i else prefillReplicas + i;
local pdSliceName(k) = 'pd%d' % k;
local pdHost(role, i) = cdk.host(pdSliceName(std.floor(pdPos(role, i) / 2)), replica=pdPos(role, i) % 2);
// In the leg env (and so in the compile-cache key: a slice host's archive is its own) and forced again by the bootstrap.
local pdSliceEnv = { TPU_ACCELERATOR_TYPE: 'tpu7x-8', TPU_WORKER_HOSTNAMES: 'localhost', TPU_WORKER_ID: '0' };
// PD_SLICE_DEVICE_PROBE (default true): before exec the bootstrap counts this host's JAX devices with the role's env and
// refuses to start the engine unless it sees 8 (16 = the host joined the slice).
local pdSliceProbe = cdk.boolVar('PD_SLICE_DEVICE_PROBE', true);

// --- Serving topology, per leg (as qwen3.5-397b-pd-agentx) ---------------------
// TENSOR_PARALLEL_SIZE_<ROLE> / DATA_PARALLEL_SIZE_<ROLE> (falling back to TENSOR_PARALLEL_SIZE /
// DATA_PARALLEL_SIZE) and PIPELINE_PARALLEL_SIZE_PREFILL, the names qwen3.5-397b-pd-agentx uses.
// One host per leg: prefill TP x DP x PP = 8, decode TP x DP = 8 (one core each). Two ways to
// move the KV, both whole pages:
//
//   Matching TP (PIPELINE_PARALLEL_SIZE_PREFILL=1): producer TP rank k ships its pages to decode
//     TP rank k; both legs run the same TP, and the Raiden transfer parallelism is that TP.
//     TP1 / DP8 is the default and the only P/D layout the image's config.py admits. TP8 / DP1 is
//     the single-host shape (PLE tables and paired main KV sharded over TP).
//   Pipeline prefill (PIPELINE_PARALLEL_SIZE_PREFILL=8), Qwen3.5's prefill: PP8 x TP1 stages, EP
//     off, one Raiden transfer unit per stage (transfer parallelism = the producer's PCP x TP x PP,
//     as in qwen3.5-397b-pd-agentx); each stage ships its own layers' pages and the TP1 decode
//     (DP8) pairs them by per-layer tag (TPU_RAIDEN_POOL_TAGS_PER_LAYER=1 on both legs). The decode
//     must be TP1: a TP1 stage's pages are the per-layer encoding a TP1 decode holds.
//     PREFILL_VLLM_PP_LAYER_PARTITION sets the split (default even, 6 layers a stage).
//
// Anything but TP1 / DP8 needs vllm-torchtpu branch caojx/qwen38-flash-agentx-disagg (488bbb5fc
// on: qwen4_exp pipeline stages, any TP x DP x PP = 8 P/D engine, the pipeline producer),
// shipped with --map-dir to both legs until it is in an image:
//   --map-dir <tree>:prefill:/root/cloud-devkit/vllm-torchtpu --map-dir <tree>:decode:/root/cloud-devkit/vllm-torchtpu
local ppPrefill = cdk.numVar('PIPELINE_PARALLEL_SIZE_PREFILL', 1);
local pipelinePrefill = ppPrefill > 1;
// q38 P-PIPE (J1, 2026-11-07): PREFILL_PARTIAL_HOST=true runs the prefill leg as ONE PP4 x TP1 engine on 4 of the
// host's 8 cores (TensorCores 0-3; chips 2-3 idle), the served proxy for one engine of PP4xDP2, instead of
// TP x DP x PP = 8. The prefill ships 4 transfer units; the decode keeps its own rank bound (its TP) and is told the
// producer's unit count in TPU_RAIDEN_PRODUCER_PARALLELISM (XFER's source-count split). Unset: no render changes.
local prefillPartialHost = cdk.boolVar('PREFILL_PARTIAL_HOST', false);
// q38 TP2-RT (reports/lane_tp2rt.md): PREFILL_SHADOW_RUNTIME=true runs the prefill leg on the q38 shadow runtime, torch_tpu
// 0.2.1.dev20261021050749 + libtpu 0.0.51 carried in the prefill's mapped tree (<tree>/q38_shadow_rt, staged by
// q38_tp2rt_stage.sh), whose per-group collectives PP x TP2 prefill stages need. The decode stays on the image runtime.
// Prefill env only: Q38_SHADOW_RUNTIME=1 (fail-closed activation, exit 86), Q38_SHADOW_CACHE_MOVE=0 (this recipe's
// compile-cache archive is keyed on the leg env, the switch included, and pulls/uploads the leg's own directories),
// VLLM_TPU_H2D_KEEPALIVE=1 (the runtime frees non-blocking upload sources) and TPU_RAIDEN_FINGERPRINT_VERSION_OVERRIDE =
// SHADOW_FP_VERSION_OVERRIDE (default the image's torch_tpu=0.2.0,libtpu=0.0.50; 'none' omits it), so the producer's layout
// fingerprint pairs with the image decode. PREFILL_HOSTMEM_LOG (default: on with the switch) logs the prefill container's
// memory cgroup every 60 s (Q38_HOSTMEM lines). Map the shadow tree onto the prefill and the same commit without
// q38_shadow_rt onto the decode. Unset: nothing renders differently.
local prefillShadowRuntime = cdk.boolVar('PREFILL_SHADOW_RUNTIME', false);
local shadowFpVersionOverride = cdk.var('SHADOW_FP_VERSION_OVERRIDE', 'torch_tpu=0.2.0,libtpu=0.0.50');
local prefillHostmemLog = cdk.boolVar('PREFILL_HOSTMEM_LOG', prefillShadowRuntime);
// q38 (SV-XPROFA round 3): DECODE_HOSTMEM_LOG=true renders the same Q38_HOSTMEM loop on the decode leg (decode host-RAM headroom;
// the decode leg has no host-memory telemetry). Its lines go to the decode server container's log, as the prefill's go to the
// prefill container's log. Default off: nothing renders differently.
local decodeHostmemLog = cdk.boolVar('DECODE_HOSTMEM_LOG', false);
assert !prefillShadowRuntime || shadowFpVersionOverride == 'none'
       || (std.length(std.findSubstr('torch_tpu=', shadowFpVersionOverride)) == 1
           && std.length(std.findSubstr('libtpu=', shadowFpVersionOverride)) == 1) :
  'SHADOW_FP_VERSION_OVERRIDE must be torch_tpu=<v>,libtpu=<v> (the decode runtime) or none, got "%s"' % shadowFpVersionOverride;
// q38 T10 (SHARD-TP, reports/t10_plan.md): PREFILL_TP2_STAGES=true runs the pipeline prefill as PP4 x TP2 stages (one chip
// per stage, one KV head per rank: PIPELINE_PARALLEL_SIZE_PREFILL=4 TENSOR_PARALLEL_SIZE_PREFILL=2) on the shadow runtime,
// whose per-group collectives TP stages need (PREFILL_SHADOW_RUNTIME=true is required). Prefill env: TPU_PP_TP_STAGES=1,
// QWEN4_EXP_PP_TP_STAGES=1, EP2 MoE on path D (TPU_MOE_EP_ARGSORT_PERMUTE=1) with the fused EP kernel OFF (the EP-leg rule
// below leaves USE_MOE_FUSED_EP_KERNEL=0 and no MOE_FUSED_EP_KERNEL_IMPL for this leg), the dense one-KV-head page
// (QWEN4_EXP_DENSE_MAIN_KV=1), the ring hand-off (TPU_PP_HANDOFF=ring), TPU_PP_SPREAD_NET_HITS=1. Both legs:
// QWEN4_EXP_PD_PRODUCER_TP=2 (the TP2 producer's 8 units, unit = 2 x stage + TP rank). Unset: nothing renders differently.
local prefillTp2Stages = cdk.boolVar('PREFILL_TP2_STAGES', false);
assert !prefillTp2Stages || prefillShadowRuntime :
  'PREFILL_TP2_STAGES=true needs PREFILL_SHADOW_RUNTIME=true (TP stage collectives need the shadow torch_tpu)';
local q38PrefillExtraEnv = cdk.var('EXTRA_SERVER_ENV', '') + ',' + cdk.var('PREFILL_EXTRA_SERVER_ENV', '');
assert !prefillTp2Stages || (std.length(std.findSubstr('USE_MOE_FUSED_EP_KERNEL=', q38PrefillExtraEnv)) == 0
                             && std.length(std.findSubstr('MOE_FUSED_EP_KERNEL_IMPL=', q38PrefillExtraEnv)) == 0) :
  'PREFILL_TP2_STAGES=true: the extra env may not set USE_MOE_FUSED_EP_KERNEL or MOE_FUSED_EP_KERNEL_IMPL (the TP2 prefill runs MoE path D, fused EP kernel off)';
// F2 (QWEN4_EXP_PD_DRAFT_KV) and S4 (QWEN4_EXP_PD_SHARED_COMPRESSED) are refused behind TP-P stages on both legs (XFER-2,
// q38-tp2xfer-t10 cdf38589e: built for TP1 stages only), so the extra env of either leg may not turn them on.
local q38Tp2ExtraEnvAll = q38PrefillExtraEnv + ',' + cdk.var('DECODE_EXTRA_SERVER_ENV', '');
assert !prefillTp2Stages || (std.length(std.findSubstr('QWEN4_EXP_PD_DRAFT_KV=', q38Tp2ExtraEnvAll)) == 0
                             && std.length(std.findSubstr('QWEN4_EXP_PD_SHARED_COMPRESSED=', q38Tp2ExtraEnvAll)) == 0) :
  'PREFILL_TP2_STAGES=true: F2 (QWEN4_EXP_PD_DRAFT_KV) and S4 (QWEN4_EXP_PD_SHARED_COMPRESSED) are refused behind TP stages; the extra env may not set them';
local coresFor(role) = if role == 'prefill' && prefillPartialHost then 4 else 8;
local tpFor(role) = std.parseJson(cdk.var('TENSOR_PARALLEL_SIZE_%s' % std.asciiUpper(role), cdk.var('TENSOR_PARALLEL_SIZE', '1')));
local ppFor(role) = if role == 'prefill' then ppPrefill else 1;
local dpFor(role) = std.parseJson(cdk.var(
  'DATA_PARALLEL_SIZE_%s' % std.asciiUpper(role),
  cdk.var('DATA_PARALLEL_SIZE', std.toString(coresFor(role) / (tpFor(role) * ppFor(role))))
));
local shapeFor(role) =
  local s = { tp: tpFor(role), dp: dpFor(role), pp: ppFor(role) };
  // 24 attention heads, 8 cores.
  if !std.member([1, 2, 4, 8], s.tp) then error '%s TP must be 1, 2, 4 or 8, got %d' % [role, s.tp]
  else if s.tp * s.dp * s.pp != coresFor(role) then
    error 'each leg is one 8-core host: %s TP x DP x PP must be %d, got %d x %d x %d' % [role, coresFor(role), s.tp, s.dp, s.pp]
  else s;
local prefillShape = shapeFor('prefill');
local decodeShape =
  local d = shapeFor('decode');
  if !pipelinePrefill && d.tp != prefillShape.tp then
    error 'matching-TP transfer ships TP rank k to TP rank k: TENSOR_PARALLEL_SIZE_PREFILL=%d and _DECODE=%d must match' % [prefillShape.tp, d.tp]
  else if pipelinePrefill && prefillShape.tp != 1 && !(prefillTp2Stages && prefillShape.tp == 2) then
    error 'pipeline stages run TP1 (as Qwen3.5), got TENSOR_PARALLEL_SIZE_PREFILL=%d' % prefillShape.tp
  else d;
local raidenTransferParallelism = prefillShape.tp * prefillShape.pp;
assert !prefillTp2Stages || (pipelinePrefill && prefillShape.pp == 4 && prefillShape.tp == 2 && prefillShape.dp == 1 && !prefillPartialHost) :
  'PREFILL_TP2_STAGES=true is one PP4 x TP2 x DP1 prefill engine (PIPELINE_PARALLEL_SIZE_PREFILL=4 TENSOR_PARALLEL_SIZE_PREFILL=2)';
assert !prefillTp2Stages || !prefillMtpDraftKv :
  'PREFILL_TP2_STAGES=true: F2 (PREFILL_MTP_DRAFT_KV=true) is refused behind TP stages';
assert !prefillPartialHost || (pipelinePrefill && prefillShape.pp == 4 && prefillShape.tp == 1 && prefillShape.dp == 1 && decodeShape.tp > 1) :
  'PREFILL_PARTIAL_HOST=true is one PP4 x TP1 engine (PIPELINE_PARALLEL_SIZE_PREFILL=4) feeding a TP > 1 decode';
// DECODE_CONTEXT_PARALLEL_SIZE_DECODE (qwen3.5-397b-pd-agentx's name): the decode's KV split. For
// Qwen3.8 that is the model's own main-KV DCP (VLLM_TPU_QSA_MAIN_KV_DCP): the TP ranks sharing a
// KV head split each request's main KV by token; vLLM's decode_context_parallel_size stays 1.
// 2 or 4, within the TP ranks sharing one of the 2 KV heads (TP / 2): TP8 takes up to 4 (Qwen3.5's
// TP8 + DCP4), TP4 up to 2. DCP4 needs vllm-torchtpu e1581678c (main-KV DCP groups of 4).
local dcpDecode =
  local d = cdk.numVar('DECODE_CONTEXT_PARALLEL_SIZE_DECODE', if q38Std && pipelinePrefill && decodeShape.tp == 8 then 4 else 1);  // Q38_STD
  local headRanks = std.max(1, decodeShape.tp / 2);
  if d == 1 then 1
  else if !std.member([2, 4], d) then error 'Qwen3.8 decode DCP supports 1, 2 or 4, got DECODE_CONTEXT_PARALLEL_SIZE_DECODE=%d' % d
  else if headRanks % d != 0 then error 'decode DCP %d needs %d TP ranks per KV head (decode TP %d gives %d)' % [d, d, decodeShape.tp, headRanks]
  else d;
// A TP>1 decode behind TP1 pipeline stages receives whole TP1 pages into a staging carrier and
// repacks its slice (vllm-torchtpu qwen4_exp/pd_staging.py): Qwen3.5's TP8 + DCP and DP x TP2
// decodes. QWEN4_EXP_PD_STAGING_BYTES (DECODE_QWEN4_EXP_PD_STAGING_BYTES) sizes that carrier.
local stagedDecode = pipelinePrefill && decodeShape.tp > 1;

// --- Sizing, shared by both legs ----------------------------------------------
local maxModelLen = cdk.numVar('MAX_MODEL_LEN', 262144);
// QSA's compressed-key and raw-ring layout is built around 128-token pages.
local blockSize = 128;
// The decode's prefix-hit alignment, for the aligned handoff of a caching decode (see serverEnv):
// its main-KV table entry, 512-token pages at TP8 times VLLM_TPU_QSA_MAIN_KV_DCP. Other decode TP
// sizes page differently; pass HANDOFF_ALIGN_TOKENS for them (a wrong value only costs hits).
local handoffAlignTokens = std.parseJson(cdk.var(
  'HANDOFF_ALIGN_TOKENS',
  std.toString(if q38Std && stagedDecode then 1 else if decodeShape.tp == 8 then 512 * std.max(dcpDecode, 1) else 0),  // Q38_STD: 1
));
local legVar(role, name, default) = cdk.var('%s_%s' % [std.asciiUpper(role), name], cdk.var(name, default));
local legNumVar(role, name, default) = std.parseJson(legVar(role, name, std.toString(default)));
local concurrencies = agentx.parseConcurrencies(cdk.var('CLIENT_CONCURRENCIES', '8'));

// --- qwen4_exp model config ---------------------------------------------------
// Per leg; <ROLE>_<NAME> first, as everywhere. The PLE (engram) tables stay on HBM, sharded over
// all 8 ranks of a leg: over TP at DP1, as on the single host, and across DP otherwise, since at
// TP1 with embedding_across_dp=false every DP engine stages the whole ~95 GiB table (rows x 160 x
// BF16), 8 x 95 GiB of host RAM per leg (both legs OOMKilled at weight load in
// j-7f4deae7-8a69-4f74-ac47) and more than one core's HBM. A pipeline stage cannot shard it: the
// stage holding the PLE layer keeps it in host memory instead (PLE_CPU_OFFLOAD).
local qwen4ExpConfig(role, shape) = {
  engram_config: {
    cpu_offload: std.asciiLower(legVar(role, 'PLE_CPU_OFFLOAD', if shape.pp > 1 then 'true' else 'false')) == 'true',
    dp_shared_memory: false,
    embedding_across_dp: std.asciiLower(legVar(role, 'PLE_EMBEDDING_ACROSS_DP', if shape.dp > 1 then 'true' else 'false')) == 'true',
  },
  hc_prefill_full_tile: cdk.numVar('HC_PREFILL_FULL_TILE', 128),
  hc_prefill_norm_tile: cdk.numVar('HC_PREFILL_NORM_TILE', 128),
  hc_prefill_up_tile: cdk.numVar('HC_PREFILL_UP_TILE', 128),
};
// Escape hatch: replaces the whole JSON above, verbatim (ADDITIONAL_CONFIG or <ROLE>_ADDITIONAL_CONFIG).
local additionalConfig(role, shape) = legVar(role, 'ADDITIONAL_CONFIG', std.manifestJsonMinified({ qwen4_exp: qwen4ExpConfig(role, shape) }));
// Pool layout. Part of the Raiden layout fingerprint, so one value for both legs.
// Main-KV pairing (the single-host default) packs two layers into the padding head slots of an
// FP8 row, which needs one KV head per device: on at TP >= 2 (2 KV heads), off at TP1, where a
// device holds both (pool_layout.py; both legs died on it at engine config in
// j-2ef10a7b-9cdd-43ec-9eb8). Compressed-key packing has no such bound, but a packed group spans
// consecutive layers, which a pipeline split would cut, so it is off with a pipeline prefill.
local qwen4ExpPackCompressedK = cdk.var('QWEN4_EXP_PACK_COMPRESSED_K', if pipelinePrefill then '0' else '1');
// The staged decode repacks into unpaired main-KV pages (the pipeline producer's layout).
// A staged decode behind a pipeline prefill repacks every load into its own pool, so its compressed
// keys may pack (vllm-torchtpu "packed compressed keys on a staged decode"): 1/8-filled pages become
// whole ones. Not part of the fingerprint there, which the producer-geometry staging carrier sets.
local decodePackCompressedK = cdk.var('DECODE_QWEN4_EXP_PACK_COMPRESSED_K', if q38Std && pipelinePrefill then '1' else qwen4ExpPackCompressedK);  // Q38_STD
assert decodePackCompressedK == qwen4ExpPackCompressedK || pipelinePrefill :
  'DECODE_QWEN4_EXP_PACK_COMPRESSED_K differs from the prefill leg only behind a pipeline prefill';
assert !decodeMtp || (pipelinePrefill && decodeShape.tp >= 2) :
  'DECODE_MTP=true needs a pipeline prefill and a staged TP>=2 decode';
local qwen4ExpPackMainKv = cdk.var('QWEN4_EXP_PACK_MAIN_KV', if decodeShape.tp >= 2 && !pipelinePrefill then '1' else '0');

// --- Startup: ROX PVC weights, per-role GCS compile cache ----------------------
// MODEL_SOURCE: 'pvc' symlinks the pre-staged weights from the read-only claim (both pods mount
// it; ReadOnlyMany), 'gcs' syncs gs://cloud-devkit/ckpt/hf/$MODEL_NAME, 'auto' prefers the PVC.
local modelSource =
  local s = std.asciiLower(cdk.var('MODEL_SOURCE', 'pvc'));
  if std.member(['pvc', 'gcs', 'auto'], s) then s
  else error 'MODEL_SOURCE must be pvc, gcs, or auto, got "%s"' % s;
// q38 FASTSTART Phase A (lead 11-10, reports/lane_faststart.md): WEIGHTS_HDML=true mounts the Hyperdisk ML copy of the
// weights (cdk-qwen38-flash-fp8-weights-hdml-rox-pvc, upstream FAST_STARTUP's claim) and starts upstream's page-cache
// prefetch in the server script (16 readers, stopped at the first "Loading weights took"). Script and claim only: the leg
// env and serve args, and so the compile-cache key, are unchanged. Unset/false renders byte-identical to before. A
// -hdml-rox-pvc claim passed as SHARED_CACHE_PVC also prefetches (upstream 6bd9f15b).
local q38WeightsHdml = cdk.boolVar('WEIGHTS_HDML', false);
local sharedCachePvc = cdk.var('SHARED_CACHE_PVC', if q38WeightsHdml then 'cdk-qwen38-flash-fp8-weights-hdml-rox-pvc' else 'cdk-qwen38-flash-fp8-weights-rox-pvc');
local q38HdmlPrefetch = modelSource != 'gcs' && std.endsWith(sharedCachePvc, '-hdml-rox-pvc');
assert !q38WeightsHdml || std.endsWith(sharedCachePvc, '-hdml-rox-pvc') : 'WEIGHTS_HDML=true needs a -hdml-rox-pvc claim, got SHARED_CACHE_PVC=%s' % sharedCachePvc;
// The base helper's env-gated 8-reader block (cdk.agentx resolveCheckpointScript), as it renders, and upstream's
// replacement (master 85bf2f63 cdk.agentx:880-897), verbatim apart from the two q38 echo markers.
local q38PrefetchBase = std.join('\n', [
  '  if [ "${WEIGHT_PREFETCH:-false}" = "true" ]; then',
  '    python3 /tmp/cdk_gcs.py prefetch_weights "${_found_pvc_dir}" "${WEIGHT_PREFETCH_WORKERS:-8}" &',
  '    export CDK_WEIGHT_PREFETCH_PID=$!',
  '  fi',
  '',
]);
local q38PrefetchHdml = std.join('\n', [
  '  if [ "${WEIGHT_PREFETCH:-true}" = "true" ]; then',
  '    echo "[Q38WeightPrefetch] start claim=hdml dir=${_found_pvc_dir} workers=${WEIGHT_PREFETCH_WORKERS:-16} ts=$(date +%s.%N)"',
  '    TPU_EVICT_WEIGHTS_PAGE_CACHE=0 python3 /tmp/cdk_gcs.py prefetch_weights "${_found_pvc_dir}" "${WEIGHT_PREFETCH_WORKERS:-16}" &',
  '    export CDK_WEIGHT_PREFETCH_PID=$!',
  '    ( set +x; while kill -0 "${CDK_WEIGHT_PREFETCH_PID}" 2>/dev/null; do',
  '        if grep -a -q -E "Loading weights took|Model weights loaded|Page cache eviction" /tmp/cdk_*server.log* /tmp/cdk_vllm.log* "${CDK_OUTPUT_DIR:-/tmp/cdk_output}/q38f-bench/pod.log" 2>/dev/null; then',
  '          kill -TERM "${CDK_WEIGHT_PREFETCH_PID}" 2>/dev/null || true',
  '          echo "[Q38WeightPrefetch] stopped at weight load ts=$(date +%s.%N)"',
  '          break',
  '        fi',
  '        sleep 2',
  '      done ) &',
  '  fi',
  '',
]);
local q38Ckpt(s) =
  if !q38HdmlPrefetch then s
  else
    local parts = std.split(s, q38PrefetchBase);
    assert std.length(parts) == 2 : 'FASTSTART: the base prefetch block occurs %d times in the checkpoint script (want 1)' % (std.length(parts) - 1);
    std.join(q38PrefetchHdml, parts);
// q38 FASTSTART B0 + identity (lead 11-10): Q38_STARTUP_DIAG=true appends ONE background block to each leg's server script.
// After the leg's own /health it prints (marker [Q38Diag]) every vLLM torch.compile cache_key_factors.json under
// VLLM_CACHE_ROOT (why a restored compile cache is or is not reused); the prefill leg (index 0) also prints a sampled weights
// manifest of MODEL_DIR (size of every file; sha256 of every non-safetensors file and of each safetensors header plus its
// first and last 64 MiB) and sends 8 fixed greedy probes (max_tokens 1, temperature 0, logprobs 5) through the proxy once it
// answers. Script only: leg env, serve args and the compile-cache key are unchanged. Diagnostic runs only (the probes are 8
// requests at sweep start). Unset/false renders byte-identical.
local q38StartupDiag = cdk.boolVar('Q38_STARTUP_DIAG', false);
local q38DiagBlock(role, port, front) = std.strReplace(std.strReplace(std.strReplace(|||
  # q38 STARTUP DIAG begin
  ( set +x
    _q38d_t0=$(date +%s)
    until curl -sf "http://127.0.0.1:@PORT@/health" >/dev/null 2>&1; do
      sleep 5; [ $(( $(date +%s) - _q38d_t0 )) -gt 7200 ] && exit 0
    done
    echo "[Q38Diag] role=@ROLE@ healthy ts=$(date +%s.%N)"
    python3 - "@ROLE@" <<'Q38DIAG_FACTORS'
  import glob, json, os, sys
  root = os.environ.get("VLLM_CACHE_ROOT", "")
  fs = sorted(glob.glob(os.path.join(root, "torch_compile_cache", "*", "*", "*", "cache_key_factors.json")))
  print("[Q38Diag] role=%s factors_files=%d root=%s" % (sys.argv[1], len(fs), root), flush=True)
  for f in fs:
      try:
          d = json.load(open(f))
      except Exception as e:
          d = {"error": repr(e)}
      print("[Q38Diag] role=%s factors %s %s" % (sys.argv[1], os.path.relpath(os.path.dirname(f), root),
                                                 json.dumps(d, sort_keys=True, separators=(",", ":"))), flush=True)
  Q38DIAG_FACTORS
    if [ "@ROLE@" = prefill ] && [ "${JOBSET_JOB_INDEX:-0}" = 0 ]; then
      nice -n 19 python3 - "${MODEL_DIR:-}" <<'Q38DIAG_MANIFEST'
  import hashlib, json, os, sys
  d = os.path.realpath(sys.argv[1]) if sys.argv[1] else ""
  out = []
  for n in (sorted(os.listdir(d)) if d and os.path.isdir(d) else []):
      p = os.path.join(d, n)
      if not os.path.isfile(p):
          continue
      sz = os.path.getsize(p)
      h = hashlib.sha256()
      with open(p, "rb") as f:
          if n.endswith(".safetensors"):
              hl = f.read(8); h.update(hl); m = int.from_bytes(hl, "little")
              h.update(f.read(min(m, 100 << 20)))
              f.seek(8 + m); h.update(f.read(64 << 20))
              if sz > 8 + m + (64 << 20):
                  f.seek(max(8 + m + (64 << 20), sz - (64 << 20))); h.update(f.read(64 << 20))
          else:
              for b in iter(lambda: f.read(1 << 24), b""):
                  h.update(b)
          try:
              os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
          except Exception:
              pass
      out.append([n, sz, h.hexdigest()[:16]])
  print("[Q38Diag] manifest dir=%s files=%d bytes=%d sha=%s" % (d, len(out), sum(x[1] for x in out),
        hashlib.sha256(json.dumps(out).encode()).hexdigest()[:16]), flush=True)
  for x in out:
      print("[Q38Diag] mfile %s %d %s" % tuple(x), flush=True)
  Q38DIAG_MANIFEST
      if [ -n "@FRONT@" ]; then
        python3 - "http://@FRONT@" "${MODEL_NAME:-}" <<'Q38DIAG_PROBE'
  import hashlib, json, sys, time, urllib.request
  url, model = sys.argv[1] + "/v1/completions", sys.argv[2]
  P = ["The capital of France is",
       "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
       "Q: What is 17 * 23?\nA:",
       "Translate to German: The weather is nice today.\nGerman:",
       "In 1969, the first humans to walk on the Moon were",
       "SELECT name, COUNT(*) FROM orders GROUP BY",
       " ".join("Item %d: the value is %d." % (i, i * i % 97) for i in range(1200)) + " The sum of all values is",
       "Once upon a time, a fox and a crow argued about cheese. " * 300 + "The moral of the story is"]
  res = []
  t0 = time.time()
  for i, p in enumerate(P):
      body = json.dumps({"model": model, "prompt": p, "max_tokens": 1, "temperature": 0, "logprobs": 5}).encode()
      r = None
      while r is None and time.time() - t0 < 7200:
          try:
              req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
              r = json.loads(urllib.request.urlopen(req, timeout=600).read())
          except Exception as e:
              print("[Q38Diag] probe i=%d retry (%s)" % (i, str(e)[:120]), flush=True)
              time.sleep(15)
      c = (r or {}).get("choices", [{}])[0]
      lp = c.get("logprobs") or {}
      rec = {"i": i, "text": c.get("text"), "top": lp.get("top_logprobs"), "tokens": lp.get("tokens")}
      res.append(rec)
      print("[Q38Diag] probe %s" % json.dumps(rec, sort_keys=True), flush=True)
  print("[Q38Diag] probe_digest n=%d sha=%s" % (len(res), hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()[:16]), flush=True)
  Q38DIAG_PROBE
      fi
    fi
  ) &
  # q38 STARTUP DIAG end
|||, '@ROLE@', role), '@PORT@', std.toString(port)), '@FRONT@', front);
local mountPvc = modelSource != 'gcs';
// Archived per user and per role (gs://$COMPILE_CACHE_GCS/.../infx-qwen38-flash-agentx-disagg/),
// keyed by the leg's serve arguments and env, so the two legs never share an archive.
local compileCacheEnabled = cdk.boolVar('COMPILE_CACHE_ENABLED', true);
local compileCacheGcs = cdk.var('COMPILE_CACHE_GCS', agentx.defaultCompileCacheGcs);
local compileCacheTag = cdk.var('COMPILE_CACHE_TAG', 'v1');
local compileCacheRefresh = cdk.boolVar('COMPILE_CACHE_REFRESH', false);
// DEV_SINGLE_BUCKET=true compiles [256, MAX_NUM_BATCHED_TOKENS] only; COMPILE_SIZES[_<ROLE>]
// sets a custom CSV list.
local devSingleBucket = cdk.boolVar('DEV_SINGLE_BUCKET', false);
local ephemeralStorage = cdk.var('EPHEMERAL_STORAGE', '64Gi');
local ephemeralStorageLimit = cdk.var('EPHEMERAL_STORAGE_LIMIT', '200Gi');
local useGcsTokenizer = cdk.boolVar('USE_GCS_TOKENIZER', true);
local gcsDatasetCache = cdk.var('GCS_DATASET_CACHE', agentx.defaultGcsDatasetCache);

// --- Profiling ----------------------------------------------------------------
local enableXprof = cdk.boolVar('ENABLE_XPROF', false);
// q38 (INT, q38_int_xprofshm_patch.py): XPROF_SHM_CLEANUP=true frees each /dev/shm/xprof capture once the mirror copied it
// (tmpfs counts against the container memory limit; J0 prefill OOM 11-07). Default off: nothing renders differently.
local q38XprofShmCleanup = cdk.boolVar('XPROF_SHM_CLEANUP', false);
local enablePerfetto = cdk.boolVar('ENABLE_PERFETTO', !q38StdPd);  // Q38_STD: off in the served P/D shape
local profileDurationSeconds = cdk.numVar('PROFILE_DURATION_SECONDS', 10);
local profileRanks = cdk.var('PROFILE_RANKS', std.join(',', [std.toString(i) for i in std.range(0, chipsPerHost - 1)]));
// q38 (SV-XPROFA, reports/lane_sv_xprof.md s2 Option A): a bounded steady-window capture on the prefill leg only.
//   XPROF_LEGS=prefill   the profiler flags, the /dev/shm mirror and TPU_PJRT_EVENT_TRACE_ENABLE go on the prefill leg
//                        only (ENABLE_PERFETTO=true still sets the trace flag on every leg for Perfetto), XPROF_URLS is
//                        the prefill URL alone, and the PROFILE_DECODE_SECONDS window is not rendered.
//   PROFILE_OFFSETS_S=a[,b...]  whole seconds after 'Phase profiling (profiling) started': one capture of
//                        PROFILE_DURATION_SECONDS at each offset (replaces the lib's fixed 180 / 330 / 120 s schedule).
// Both need ENABLE_XPROF=true. Unset (XPROF_LEGS=both, PROFILE_OFFSETS_S empty): nothing renders differently.
local q38XprofLegs = cdk.var('XPROF_LEGS', 'both');
assert std.member(['both', 'prefill'], q38XprofLegs) : 'XPROF_LEGS must be both or prefill, got %s' % q38XprofLegs;
assert q38XprofLegs == 'both' || enableXprof : 'XPROF_LEGS=prefill needs ENABLE_XPROF=true';
local q38XprofOn(leg) = enableXprof && (q38XprofLegs == 'both' || leg.role == 'prefill');
local q38ProfileOffsetsRaw = cdk.var('PROFILE_OFFSETS_S', '');
local q38ProfileOffsets =
  if q38ProfileOffsetsRaw == '' then []
  else [
    if std.length(p) > 0 && std.length(p) <= 5 && std.all([c >= '0' && c <= '9' for c in std.stringChars(p)])
    then std.parseInt(p)
    else error 'PROFILE_OFFSETS_S: "%s" is not a whole number of seconds (comma-separated, e.g. 1800,3000)' % p
    for p in std.split(q38ProfileOffsetsRaw, ',')
  ];
assert std.length(q38ProfileOffsets) == 0 || enableXprof : 'PROFILE_OFFSETS_S needs ENABLE_XPROF=true';
assert std.all([o > 0 for o in q38ProfileOffsets]) : 'PROFILE_OFFSETS_S: every offset must be > 0, got %s' % q38ProfileOffsetsRaw;
assert std.all([q38ProfileOffsets[i] > q38ProfileOffsets[i - 1] for i in std.range(1, std.length(q38ProfileOffsets) - 1)])
       : 'PROFILE_OFFSETS_S must strictly increase, got %s' % q38ProfileOffsetsRaw;

// --- Benchmark client & replay --------------------------------------------------
local fastMode = cdk.boolVar('FAST', false) || cdk.boolVar('FAST_MODE', false);
// DURATION, or BENCH_DURATION_S (qwen3.5-397b-pd-agentx's name): a custom replay length in seconds;
// under 900 the replay adds --unsafe-override. FAST=true BENCH_DURATION_S=180 for quick checks.
local durationVar = if cdk.rawVar('BENCH_DURATION_S', '') != '' then 'BENCH_DURATION_S' else 'DURATION';
local customDuration = cdk.rawVar(durationVar, '') != '';
local duration =
  if customDuration then cdk.numVar(durationVar)
  else if fastMode then 1200
  else 3600;
// Startup on both legs (weight load, DP8 compile) can pass an hour cold.
local healthTimeoutSeconds = cdk.numVar('HEALTH_TIMEOUT_S', 7200);
local benchName = cdk.var('BENCH_NAME', 'qwen3.8-flash-agentx-disagg-%dp%dd%s' % [
  prefillReplicas, decodeReplicas,
  if useLlmd then '-llmd-' + eppProfile + (if eppPatchName != '' then '-' + eppPatchName else '') else '',
]);
// The legs pair at startup, so a restart has to recreate the whole gang, and every restart
// re-runs the cold start. 0 by default: this path has no green run yet, and a real failure
// should stop with its logs rather than burn another start. job_manager still retries node
// preemptions on its own.
local maxRestarts = cdk.numVar('MAX_RESTARTS', 0);

local proxyPort = 10000;
// Model hidden size, for the hierarchical all-reduce threshold.
local hiddenSize = 2560;

// --- The two legs, as data ------------------------------------------------------
// Everything that differs between prefill and decode. Every <NAME> knob here also reads
// <ROLE>_<NAME> first (PREFILL_MAX_NUM_SEQS, DECODE_GPU_MEMORY_UTILIZATION, ...).
local leg(role, shape, port, kvRole, raidenPortBase, maxNumSeqs, maxNumBatchedTokens, replicas) = {
  role: role,
  replicas: replicas,
  tp: shape.tp,
  dp: shape.dp,
  pp: shape.pp,
  // A pipeline stage's experts stay whole (Qwen3.5's PP prefill runs EP off); TP and DP engines
  // spread them.
  expertParallel: shape.tp * shape.dp > 1,
  additionalConfig: additionalConfig(role, shape),
  port: port,
  kvRole: kvRole,
  raidenPortBase: raidenPortBase,
  // Per DP engine: eight of them share each leg's requests.
  maxNumSeqs: legNumVar(role, 'MAX_NUM_SEQS', maxNumSeqs),
  // The model caps a step at 8192 tokens.
  maxNumBatchedTokens:
    local t = legNumVar(role, 'MAX_NUM_BATCHED_TOKENS', maxNumBatchedTokens);
    if t < blockSize || t > 16384 then error '%s MAX_NUM_BATCHED_TOKENS must be in [%d, 16384] for qwen4_exp (above 8192 needs a tree that lifts the qwen4_exp S1 step check; upstream 0bdd35a), got %d' % [role, blockSize, t]
    else t,
  gpuMemoryUtilization: legVar(role, 'GPU_MEMORY_UTILIZATION', '0.92'),
  // <ROLE>_ENABLE_PREFIX_CACHING / ENABLE_PREFIX_CACHING (see the header).
  prefixCaching:
    local v = legVar(role, 'ENABLE_PREFIX_CACHING', if q38Std && stagedDecode then 'true' else 'false');  // Q38_STD
    if !std.member(['true', 'false'], v) then error '%s ENABLE_PREFIX_CACHING must be true or false, got "%s"' % [role, v]
    else v == 'true',
  // 0 is "no cap": the pool is sized from free HBM. Emitted as 0 rather than left unset because
  // the runner's static sizing path reads an unset variable as a 48 GiB cap.
  // A staged decode (TP>1 behind pipeline stages) keeps unpaired main KV, so its static pool for
  // max-num-seqs full-context requests outgrows HBM (77.7 GB + reserve vs 69.3 GB free at TP8,
  // j-e692942c-0a86-477e-aec2); cap it at 48 GiB, which also leaves room for the staging carrier.
  // Q38_STD: decode Q38_STD_DECODE_POOL_BLOCKS x 512 KiB on a TP8 staged decode; prefill 60 GiB (with
  // QWEN4_EXP_STATIC_POOL_FILL=1, serverEnv) behind it.
  maxPoolBytes: legVar(role, 'QWEN4_EXP_MAX_POOL_BYTES',
    if role == 'decode' && stagedDecode then (if q38Std && shape.tp == 8 then std.toString(q38StdDecodePoolBlocks * 524288) else '51539607552')
    else if role == 'prefill' && q38Std && stagedDecode then q38StdPrefillPoolBytes
    else '0'),
  // Pipeline stages before the last never see the sampled tokens async scheduling substitutes
  // (platforms/pp_validation.py), so a pipeline leg runs without it.
  asyncScheduling:
    local v = legVar(role, 'ASYNC_SCHEDULING', if shape.pp > 1 then 'false' else 'true') != 'false';
    if v && shape.pp > 1 then error '%s: pipeline stages need ASYNC_SCHEDULING=false' % role else v,
  ppLayerPartition: legVar(role, 'VLLM_PP_LAYER_PARTITION', ''),
  compileSizes: legVar(role, 'COMPILE_SIZES', ''),
  tokenBucketExtra: legVar(role, 'TPU_TOKEN_BUCKET_EXTRA', ''),
  extraVllmArgs: legVar(role, 'EXTRA_VLLM_ARGS', ''),
  // <ROLE>_KV_CACHE_MEMORY_BYTES / KV_CACHE_MEMORY_BYTES: --kv-cache-memory-bytes, which skips the memory-profiling run
  // (upstream 4f42a7f; needs a vllm-torchtpu with "qwen4_exp: kv_cache_memory_bytes skips profiling for a static pool").
  kvCacheMemoryBytes: legVar(role, 'KV_CACHE_MEMORY_BYTES', ''),
  // q38 RECIPE-STD: PREFILL_HOST_TIER_BYTES / HOST_TIER_BYTES, the prefill's host KV tier (TPURaidenOffloadingConnector's
  // cpu_bytes_to_use, per engine) with QWEN4_EXP_HOST_TIER_POLICY (default victim). Q38_STD default 500e9 in the served
  // shape; '' or 0 turns it off. A car whose PREFILL_EXTRA_VLLM_ARGS passes its own --kv-transfer-config keeps that
  // config untouched (every posted point does: B3 530e9, B1 500e9, B2 498e9, the rest 320e9).
  hostTierBytes:
    local v = if role == 'prefill' then legVar(role, 'HOST_TIER_BYTES', if q38Std && stagedDecode then q38StdHostTierBytes else '') else '';
    if v == '0' then '' else v,
  libtpuExtraArgs: legVar(role, 'LIBTPU_EXTRA_ARGS', std.join(' ', [
    '--xla_tpu_dvfs_p_state=7',
    '--xla_tpu_use_enhanced_launch_barrier=true',
    '--xla_enable_async_all_gather=true',
    '--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true',
    '--xla_tpu_enable_sparse_core_collective_offload_all_gather=false',
  ])),
  // EXTRA_SERVER_ENV=NAME=VALUE[,NAME=VALUE...] for both legs, then <ROLE>_EXTRA_SERVER_ENV.
  extraServerEnv:
    local parse(s) = [
      std.splitLimit(kv, '=', 1)
      for kv in std.filter(function(x) x != '', std.split(s, ','))
    ];
    local pairs = parse(cdk.var('EXTRA_SERVER_ENV', '')) + parse(cdk.var('%s_EXTRA_SERVER_ENV' % std.asciiUpper(role), ''));
    assert std.length([p for p in pairs if std.length(p) != 2 || p[0] == '']) == 0
           : 'EXTRA_SERVER_ENV / %s_EXTRA_SERVER_ENV entries must look like NAME=VALUE' % std.asciiUpper(role);
    { [p[0]]: p[1] for p in pairs },
};

// Prefill: 8192-token chunks, the model's ceiling and the single-host default.
local prefill = leg('prefill', prefillShape, 8000, 'kv_producer', 27000, maxNumSeqs=8, maxNumBatchedTokens=8192, replicas=prefillReplicas);
// Decode: no prompt is prefilled here, so the step budget only has to hold the running set.
// At DP8, 8 seqs per engine, as prefill: the profile run sizes the unified pool for max_num_seqs
// full 262144-token requests, and at 16 that is 62.76 GB + an 8 GiB reserve against 62.05 GB of
// free HBM per core (j-1ed43c05-22ab-4e28-9d02; prefill's 8 sized 31.38 GB and went on). Raise it
// only with a shorter MAX_MODEL_LEN. At TP8 one engine spreads the pool over 8 cores; 16 is the
// single-host recipe's c8 value.
local decode = leg('decode', decodeShape, 7200, 'kv_consumer', 28000, maxNumSeqs=if decodeShape.dp == 1 then 16 else 8, maxNumBatchedTokens=2048, replicas=decodeReplicas);

// ---------------------------------------------------------------------------
// The servers
// ---------------------------------------------------------------------------

local q38HostTierOn(leg) =
  leg.hostTierBytes != '' && std.strReplace(leg.extraVllmArgs, '--kv-transfer-config', '') == leg.extraVllmArgs;  // contains-test via strReplace: std.findSubstr here cost ~12 s per render (cdk's limit is 30 s)
local kvTransferConfig(leg) =
  local pd = {
    kv_connector: 'Qwen4ExpRaidenConnector',
    kv_connector_module_path: 'vllm_torchtpu.models.vllm.qwen4_exp.pd_connector',
    kv_role: leg.kvRole,
    kv_port: 14579,
  };
  if !q38HostTierOn(leg) then std.manifestJsonMinified(pd)
  else std.manifestJsonMinified({
    kv_connector: 'TPUMultiConnector',
    kv_connector_module_path: 'vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector',
    kv_role: leg.kvRole,
    kv_port: 14579,
    kv_connector_extra_config: { connectors: [pd, {
      kv_connector: 'TPURaidenOffloadingConnector',
      kv_connector_module_path: 'vllm_torchtpu.offload.raiden_connector',
      kv_role: 'kv_both',
      kv_connector_extra_config: { cpu_bytes_to_use: std.parseJson(leg.hostTierBytes) },
    }] },
  });

local serveArgs(leg) = [
  '"${MODEL_DIR}"',
  '--served-model-name="$MODEL_NAME"',
  '--tensor-parallel-size=%d' % leg.tp,
  '--data-parallel-size=%d' % leg.dp,
  '--data-parallel-size-local=%d' % leg.dp,
  '--max-model-len=%d' % maxModelLen,
  '--quantization=fp8',
  '--max-num-batched-tokens=%d' % leg.maxNumBatchedTokens,
  '--max-num-seqs=%d' % leg.maxNumSeqs,
  '--host 0.0.0.0',
  '--port %d' % leg.port,
  if leg.asyncScheduling then '--async-scheduling' else '--no-async-scheduling',
  '--prefill-schedule-interval=%d' % cdk.numVar('PREFILL_SCHEDULE_INTERVAL', 1),
  if leg.prefixCaching then '--enable-prefix-caching' else '--no-enable-prefix-caching',
] + (if leg.role == 'decode' && decodeSpeculativeConfig != '' then ["--speculative-config '%s'" % decodeSpeculativeConfig] else [])
+ (if leg.role == 'prefill' && prefillMtpDraftKv then ["--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}'"] else []) + [
  '--gpu-memory-utilization=%s' % leg.gpuMemoryUtilization,
] + (if leg.kvCacheMemoryBytes != '' then ['--kv-cache-memory-bytes=%s' % leg.kvCacheMemoryBytes] else []) + [
  '--kv-cache-dtype=fp8',
  '--language-model-only',
  '--attention-backend CUSTOM',
  '--block-size %d' % blockSize,
  "--limit-mm-per-prompt '{\"image\":0,\"video\":0}'",
  "--default-chat-template-kwargs '{\"enable_thinking\":%s}'" % (if q38ServeThinking then 'true' else 'false'),
  '--tool-call-parser=qwen3_coder',
  '--reasoning-parser=qwen3',
  '--mamba-cache-mode align',
  '--trust-remote-code',
  '--enable-prompt-tokens-details',
  "--additional-config '%s'" % leg.additionalConfig,
  '--renderer-num-workers %d' % cdk.numVar('RENDERER_NUM_WORKERS', 4),
  "--kv-transfer-config '%s'" % kvTransferConfig(leg),
]
+ (if leg.pp > 1 then ['--pipeline-parallel-size=%d' % leg.pp] else [])
// A caching decode keeps a GDN checkpoint every alignment unit: its prefix hits must land on its
// main-KV table entry (handoffAlignTokens), and the default retention (0) keeps only the replay
// boundary, aligned to the finer state page, which drops the loaded state at the aligned handoff.
+ (if leg.role == 'decode' && leg.prefixCaching && handoffAlignTokens > 0
   then ['--prefix-cache-retention-interval=%d' % handoffAlignTokens] else [])
+ (if leg.expertParallel then ['--enable-expert-parallel'] else [])
+ agentx.compilationConfigArg(devSingleBucket, leg.compileSizes, leg.maxNumBatchedTokens)
+ (if q38XprofOn(leg) then agentx.profilerArgs('/dev/shm/xprof') else [])
+ (if leg.extraVllmArgs != '' then [leg.extraVllmArgs] else []);

// LIBTPU_INIT_ARGS as infx-qwen38-flash-agentx builds it with the fused EP kernel on (its async
// row gather turns on the async all-gather offload), then the leg's extra args; later flags win.
local libtpuInitArgs(leg) = std.join(' ', [
  '--xla_tpu_use_minor_sharding_for_major_trivial_input=true',
  '--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false',
  '--xla_tpu_ars_combiner_threshold_in_bytes=0',
  '--xla_tpu_enable_async_collective_merger=false',
  '--xla_enable_async_all_gather=true',
  '--xla_tpu_enable_sparse_core_collective_offload_all_gather=true',
  '--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144',
] + (if leg.libtpuExtraArgs != '' then [leg.libtpuExtraArgs] else []));

local serverEnv(leg) =
  { MODEL_NAME: modelName }
  + agentx.megascaleOff
  + cdk.jobSetNameEnv
  // TPU_PARALLEL_PRECOMPILE only; peer replay and biggest-first stay at the image default
  // unless passed (glm52-fp8-pd-agentx found peer replay refusing buckets on a P/D leg).
  + agentx.parallelPrecompileEnv(true, fullSuite=false)
  + {
    VLLM_LOGGING_LEVEL: cdk.var('VLLM_LOGGING_LEVEL', 'INFO'),
    LIBTPU_INIT_ARGS: cdk.var('LIBTPU_INIT_ARGS', libtpuInitArgs(leg)),
    TPU_ACCELERATOR_TYPE: accelerator,

    // --- device / topology (8 cores on one 2x2x1 tpu7x host, DP8 as qwen3.5-397b-pd-agentx) ---
    JAX_PLATFORMS: 'tpu,cpu',
    PJRT_DEVICE: 'TPU',
    VLLM_TARGET_DEVICE: 'tpu',
    TPU_TOPOLOGY: topology,
    TPU_CHIPS_PER_HOST_BOUNDS: '2,2,1',
    TPU_HOST_BOUNDS: '1,1,1',
    TPU_PROCESS_ADDRESSES: '',
    TPU_WORKER_HOSTNAMES: '',
    TPU_PROCESS_PORT: '',
    VLLM_DP_MASTER_IP: '127.0.0.1',
    DP_SCHED_ENABLED: cdk.var('DP_SCHED_ENABLED', '0'),
    DP_SCHED_BATCH_PREFILL: cdk.var('DP_SCHED_BATCH_PREFILL', '1'),
    TPU_TOKEN_BUCKET_LINEAR_UNTIL: cdk.numVar('TPU_TOKEN_BUCKET_LINEAR_UNTIL', 64),
    TPU_TOKEN_BUCKET_LINEAR_INTERVAL: cdk.numVar('TPU_TOKEN_BUCKET_LINEAR_INTERVAL', 16),
    [if leg.tokenBucketExtra != '' then 'TPU_TOKEN_BUCKET_EXTRA']: leg.tokenBucketExtra,

    // --- qwen4_exp pool layout and sizing ---
    // The P/D path needs the unified pool, which infx-qwen38-flash-agentx turns on through
    // prefix caching; here it is on with prefix caching off.
    TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL: '1',
    QWEN4_EXP_PACK_COMPRESSED_K: if leg.role == 'decode' then decodePackCompressedK else qwen4ExpPackCompressedK,
    QWEN4_EXP_PACK_MAIN_KV: qwen4ExpPackMainKv,
    [if leg.maxPoolBytes != 'unset' then 'QWEN4_EXP_MAX_POOL_BYTES']: leg.maxPoolBytes,
    TPU_KV_BYTES_EXPLICIT: '0',
    VLLM_TPU_QSA_DECODE_QUERY_BLOCK: cdk.var('VLLM_TPU_QSA_DECODE_QUERY_BLOCK', '4'),
    [if leg.role == 'decode' && dcpDecode > 1 then 'VLLM_TPU_QSA_MAIN_KV_DCP']: std.toString(dcpDecode),
    [if prefillMtpDraftKv then 'QWEN4_EXP_PD_DRAFT_KV']: '1',
    // q38 TP2-RT: the shadow runtime on the prefill leg only (PREFILL_SHADOW_RUNTIME=true; see the header of the switch).
    [if leg.role == 'prefill' && prefillShadowRuntime then 'Q38_SHADOW_RUNTIME']: '1',
    [if leg.role == 'prefill' && prefillShadowRuntime then 'Q38_SHADOW_CACHE_MOVE']: '0',
    [if leg.role == 'prefill' && prefillShadowRuntime then 'VLLM_TPU_H2D_KEEPALIVE']: '1',
    [if leg.role == 'prefill' && prefillShadowRuntime && shadowFpVersionOverride != 'none' then 'TPU_RAIDEN_FINGERPRINT_VERSION_OVERRIDE']:
      shadowFpVersionOverride,
    // q38 T10: PP4 x TP2 prefill stages (PREFILL_TP2_STAGES=true; see the header of the switch).
    [if leg.role == 'prefill' && prefillTp2Stages then 'TPU_PP_TP_STAGES']: '1',
    [if leg.role == 'prefill' && prefillTp2Stages then 'QWEN4_EXP_PP_TP_STAGES']: '1',
    [if leg.role == 'prefill' && prefillTp2Stages then 'TPU_MOE_EP_ARGSORT_PERMUTE']: '1',
    [if leg.role == 'prefill' && prefillTp2Stages then 'QWEN4_EXP_DENSE_MAIN_KV']: '1',
    [if leg.role == 'prefill' && prefillTp2Stages then 'TPU_PP_HANDOFF']: 'ring',
    [if leg.role == 'prefill' && prefillTp2Stages then 'TPU_PP_SPREAD_NET_HITS']: '1',
    [if prefillTp2Stages then 'QWEN4_EXP_PD_PRODUCER_TP']: '2',
    // The decode models the TP2 producer's dense (TD) page while its own DCP4 pool stays padded (XFER-2 c3964d9d2).
    [if leg.role == 'decode' && prefillTp2Stages then 'QWEN4_EXP_PD_PRODUCER_DENSE_MAIN_KV']: '1',
    [if leg.role == 'decode' && stagedDecode && legVar(leg.role, 'QWEN4_EXP_PD_STAGING_BYTES', if q38Std then q38StdDecodeStagingBytes else '') != '' then 'QWEN4_EXP_PD_STAGING_BYTES']:
      legVar(leg.role, 'QWEN4_EXP_PD_STAGING_BYTES', if q38Std then q38StdDecodeStagingBytes else ''),
    // q38 RECIPE-STD members (Q38_STD, served shape): the prefill's static pool fill, the host tier policy when the
    // recipe owns the tier, no JAX compilation cache (the >2 GiB executable abort), and the merged branch's form keys.
    [if leg.role == 'prefill' && q38Std && stagedDecode then 'QWEN4_EXP_STATIC_POOL_FILL']: legVar(leg.role, 'QWEN4_EXP_STATIC_POOL_FILL', '1'),
    [if q38HostTierOn(leg) then 'QWEN4_EXP_HOST_TIER_POLICY']: legVar(leg.role, 'QWEN4_EXP_HOST_TIER_POLICY', 'victim'),
    // Q38_STD: bounded host staging (6 leases, as B1 at 500e9 and B3 at 530e9); 0, the code default, keeps a full host
    // mirror of every pool (about 8 x 60 GiB), which with a 500e9 tier would overrun the ~900 GiB prefill host.
    [if leg.role == 'prefill' && q38Std && stagedDecode then 'QWEN4_EXP_PD_STAGING_LEASES']: legVar(leg.role, 'QWEN4_EXP_PD_STAGING_LEASES', '6'),
    [if q38Std && stagedDecode then 'JAX_ENABLE_COMPILATION_CACHE']: legVar(leg.role, 'JAX_ENABLE_COMPILATION_CACHE', 'false'),
    // Q38_FORM / Q38_PROXY_FORM (origin/caojx/qwen38-flash-1124 merge, MERGE-1124): run a posted point's own
    // vllm_torchtpu package / proxy, stored byte for byte in the branch. Unset: nothing renders.
    [if cdk.var('Q38_FORM', '') != '' then 'Q38_FORM']: cdk.var('Q38_FORM', ''),
    [if cdk.var('Q38_PROXY_FORM', '') != '' then 'Q38_PROXY_FORM']: cdk.var('Q38_PROXY_FORM', ''),
    VLLM_TPU_QSA_DECODE_KV_BLOCK: cdk.var('VLLM_TPU_QSA_DECODE_KV_BLOCK', '3328'),
    // QSA sparse prefill kernel switches (vllm-torchtpu envs.py), off unless asked for:
    // PREFILL_LEAN skips the FP8 key guard the score mask already covers (bit-identical);
    // PREFILL_BF16_MXU feeds the MXU BF16 operands (exact QK, probabilities as two BF16 terms).
    [if cdk.boolVar('PREFILL_LEAN', false) then 'VLLM_TPU_QSA_PREFILL_LEAN']: '1',
    [if cdk.boolVar('PREFILL_BF16_MXU', false) then 'VLLM_TPU_QSA_PREFILL_BF16_MXU']: '1',
    VLLM_KV_CACHE_LAYOUT: cdk.var('VLLM_KV_CACHE_LAYOUT', 'NHD'),
    // config.py rejects block-major KV for qwen4_exp.
    VLLM_TPU_BLOCK_MAJOR_KV: '0',

    // --- model / kernel config: infx-qwen38-flash-agentx's base server env ---
    MODEL_IMPL_TYPE: 'vllm',
    HF_HUB_OFFLINE: '1',
    HF_HOME: if mountPvc then '/vllm-cache' else '/root/.cache/huggingface',
    TF_CPP_MIN_LOG_LEVEL: cdk.var('TF_CPP_MIN_LOG_LEVEL', '3'),
    VLLM_XLA_CHECK_RECOMPILATION: '0',
    VLLM_ENGINE_READY_TIMEOUT_S: cdk.numVar('VLLM_ENGINE_READY_TIMEOUT_S', 7200),
    // uvicorn closes an idle keep-alive connection after this many seconds (vLLM's default is 5);
    // the proxy's backend pool expires its own connections well inside it.
    VLLM_HTTP_TIMEOUT_KEEP_ALIVE: cdk.numVar('VLLM_HTTP_TIMEOUT_KEEP_ALIVE', 600),
    FUSED_MOE_DECODE_KERNEL: cdk.var('FUSED_MOE_DECODE_KERNEL', '1'),
    FUSED_MOE_DECODE_KERNEL_MAX_TOKENS: cdk.var('FUSED_MOE_DECODE_KERNEL_MAX_TOKENS', '256'),
    [if cdk.var('FUSED_MOE_DECODE_KERNEL_FOLD', 'xla') != 'off' then 'FUSED_MOE_DECODE_KERNEL_FOLD']: cdk.var('FUSED_MOE_DECODE_KERNEL_FOLD', 'xla'),
    USE_MOE_FUSED_EP_KERNEL: if leg.expertParallel && !(leg.role == 'prefill' && prefillTp2Stages) then '1' else '0',
    // Selecting an implementation turns the fused EP kernel on, so an EP-less leg leaves it unset.
    [if leg.expertParallel && !(leg.role == 'prefill' && prefillTp2Stages) then 'MOE_FUSED_EP_KERNEL_IMPL']: cdk.var('MOE_FUSED_EP_KERNEL_IMPL', 'adaptive'),
    MOE_FUSED_EP_KERNEL_MIN_TOKENS: cdk.var('MOE_FUSED_EP_KERNEL_MIN_TOKENS', '1024'),
    TPU_MOE_FUSED_EP_STEP_MIN_TOKENS: cdk.var('TPU_MOE_FUSED_EP_STEP_MIN_TOKENS', '64'),
    TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER: cdk.var('TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER', '1'),
    // Off with the fused EP kernel, as in infx-qwen38-flash-agentx.
    TPU_ATTN_MOE_OVERLAP: '0',
    TPU_MOE_SEQUENCE_PARALLEL: '0',
    TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS: '0',
    TPU_MOE_ROUTER_TOPK: cdk.var('TPU_MOE_ROUTER_TOPK', 'rowmax'),
    TPU_MOE_SKIP_PADDED_TOKENS: cdk.var('TPU_MOE_SKIP_PADDED_TOKENS', '1'),
    TPU_MOE_SINGLE_SORT: cdk.var('TPU_MOE_SINGLE_SORT', '1'),
    TPU_MOE_DENSE_EXPERT_THRESHOLD: cdk.var('TPU_MOE_DENSE_EXPERT_THRESHOLD', '0'),
    USE_MOE_SPARSE_CORE: cdk.var('USE_MOE_SPARSE_CORE', '1'),
    ONEHOT_MOE_PERMUTE_THRESHOLD: cdk.var('ONEHOT_MOE_PERMUTE_THRESHOLD', '2048'),
    RAGGED_GATHER_REDUCE_VERSION: cdk.var('RAGGED_GATHER_REDUCE_VERSION', 'v3'),
    USE_BATCHED_RPA_LONGCTX: cdk.var('USE_BATCHED_RPA_LONGCTX', '1'),
    TPU_RPA_DECODE_BATCHED: cdk.var('TPU_RPA_DECODE_BATCHED', '1'),
    TPU_ROPE_CACHE_TRUNCATE: cdk.var('TPU_ROPE_CACHE_TRUNCATE', '1'),
    TPU_ROPE_CACHE_ROW_MAJOR: cdk.var('TPU_ROPE_CACHE_ROW_MAJOR', '1'),
    TPU_ENABLE_GDN_DYNAMIC_TILING: cdk.var('TPU_ENABLE_GDN_DYNAMIC_TILING', '1'),
    TPU_SAMPLER_TOPK_CANDIDATES: cdk.var('TPU_SAMPLER_TOPK_CANDIDATES', '64'),
    VLLM_TPU_FAST_TOKEN_SUBSTITUTION: cdk.var('VLLM_TPU_FAST_TOKEN_SUBSTITUTION', '1'),
    TPU_EVICT_WEIGHTS_PAGE_CACHE: cdk.var('TPU_EVICT_WEIGHTS_PAGE_CACHE', 'true'),
    RAIDEN_NUMA_POLICY: cdk.var('RAIDEN_NUMA_POLICY', 'preferred'),
    BASE_RESULT_DIR: '/root/cloud-devkit/results',

    // --- Raiden stage-3 KV transfer, store mode (as qwen3.5-397b-pd-agentx's DP8 legs) ---
    TPU_KV_RESHARD_TRANSPORT: 'raiden',
    TPU_RAIDEN_RESHARD_IMPL: 'store',
    TPU_RAIDEN_CLIENT_IMPL: 'cpp',
    // The producer's TP ranks are its transfer units, and the consumer expects the same count.
    TPU_RAIDEN_TRANSFER_PARALLELISM: std.toString(if prefillPartialHost && leg.role == 'decode' then decodeShape.tp else raidenTransferParallelism),
    [if prefillPartialHost && leg.role == 'decode' then 'TPU_RAIDEN_PRODUCER_PARALLELISM']: std.toString(raidenTransferParallelism),
    // The decode TP; emitted only when it is not the default of 1, as qwen3.5-397b-pd-agentx does.
    [if decodeShape.tp != 1 then 'TPU_RAIDEN_DST_SHARDS']: std.toString(decodeShape.tp),
    // A pipeline producer's stages each register their own layers' pools; both peers pair them by
    // layer (qwen3.5-397b-pd-agentx's TPU_RAIDEN_POOL_TAGS_PER_LAYER=1).
    [if pipelinePrefill then 'TPU_RAIDEN_POOL_TAGS_PER_LAYER']: '1',
    [if leg.ppLayerPartition != '' then 'VLLM_PP_LAYER_PARTITION']: leg.ppLayerPartition,
    // Pinned, so the GDN conv layout is the single-host one at every TP. env_override.py would
    // pick the pair layout at a transfer degree >= 8; that layout exists for Qwen3.5's head-sharded
    // spans, and this connector moves whole pages either way.
    TPU_GDN_CONV_QK_PAIR_LAYOUT: cdk.var('TPU_GDN_CONV_QK_PAIR_LAYOUT', '0'),
    TPU_RAIDEN_ADVERTISE_HOST: { valueFrom: { fieldRef: { fieldPath: 'status.podIP' } } },
    TPU_RAIDEN_RESHARD_PORT_BASE: leg.raidenPortBase,
    TPU_RAIDEN_STORE_DISPATCH_PORT_BASE: leg.raidenPortBase + 100,
    CONTROLLER_PORT: leg.raidenPortBase,
    TPU_RAIDEN_JOB_NAME: leg.role,
    // Replica i > 0 renames itself <role>-<i>-engine in the server script.
    TPU_RAIDEN_ENGINE_ID: '%s-engine' % leg.role,
    [if leg.replicas > 1 then 'JOBSET_JOB_INDEX']: { valueFrom: { fieldRef: { fieldPath: "metadata.labels['jobset.sigs.k8s.io/job-index']" } } },
    TPU_KV_TRANSFER_NAMESPACE: leg.role,
    // A tripwire, not a knob: over it a prompt silently skips the transfer.
    TPU_RAIDEN_MAX_TRANSFER_TOKENS: maxModelLen,
    TPU_RAIDEN_ASYNC_COMPLETE: legVar(leg.role, 'TPU_RAIDEN_ASYNC_COMPLETE', '1'),
    // A caching decode must pull only the suffix of its local prefix: without suffix loads a
    // partially hit request's load fails and the decode recomputes the request.
    [if leg.role == 'decode' && leg.prefixCaching then 'TPU_RAIDEN_PREFIX_AWARE_LOAD']: '1',
    // A caching decode can only hit where it holds a GDN checkpoint, and a loaded request leaves
    // one only if the handoff ends on its block boundary: the prefill stops on a multiple of the
    // decode's prefix-hit alignment and the decode recomputes the rest (<= that many tokens).
    [if leg.role == 'prefill' && decode.prefixCaching then 'TPU_RAIDEN_HANDOFF_ALIGN_TOKENS']: std.toString(handoffAlignTokens),
    TPU_ENABLE_D2H_TRANSFER: '1',
    TPU_HMA_D2H_COPY_IMPL: 'device_put',
    TPU_HMA_MIN_DONE_RECVING_BATCH: '1',
    TPU_HMA_MAX_DONE_RECVING_BATCH: '1',
    TPU_HMA_MAX_INFLIGHT_D2H_SENDS: '1',
    TPU_HMA_MAX_INFLIGHT_PULLS: '8',
    TPU_MAX_HOST_KV_BUFFER_SIZE: '64',
    TPU_P2P_WAIT_PULL_TIMEOUT: cdk.var('TPU_P2P_WAIT_PULL_TIMEOUT', '300'),
    TPU_KV_SHM_POOL_GB: '8',
    TPU_KV_TRANSFER_PORT: '9100',
    TPU_SIDE_CHANNEL_PORT: '9600',
    // 16 GiB, the size env_override.py picks for Raiden when unset, rather than the single-host 4.
    // Q38_STD: the prefill of the served shape takes 4 GiB (upstream 5b46cc7; every posted point); the decode keeps 16 GiB
    // (B1 and B3 run it; upstream 3b321fe's decode 4 GiB is NOT taken: pass DECODE_TPU_PREMAPPED_BUFFER_SIZE=4294967296).
    local premap = if q38Std && stagedDecode && leg.role == 'prefill' then '4294967296' else '17179869184',
    TPU_PREMAPPED_BUFFER_SIZE: legVar(leg.role, 'TPU_PREMAPPED_BUFFER_SIZE', premap),
    TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES: legVar(leg.role, 'TPU_PREMAPPED_BUFFER_SIZE', premap),

    // --- startup: checkpoint and compile cache (cdk.agentx) ---
    MODEL_SOURCE: modelSource,
    COMPILE_CACHE_GCS: compileCacheGcs,
    COMPILE_CACHE_TAG: compileCacheTag,
    COMPILE_CACHE_REFRESH: if compileCacheRefresh then 'true' else 'false',
  }
  + agentx.compileCacheEnv('/dev/shm/compile-cache/' + leg.role, compileCacheEnabled)
  // Perfetto reads the PJRT client's own log lines, which are off by default.
  + (if enablePerfetto then {
       TPU_VMODULE: cdk.var('TPU_VMODULE', 'tpu_pjrt_client=1,pjrt_stream_executor_client=1,tpu_pjrt_compiler_utils=1'),
       TPU_STDERR_LOG_LEVEL: '0',
     } else {})
  + (if enablePerfetto || q38XprofOn(leg) then { TPU_PJRT_EVENT_TRACE_ENABLE: '1' } else {})
  + (if pdSliceOn then pdSliceEnv else {})  // q38 X222-SERVE
  + leg.extraServerEnv;

local q38ShmCleanupBlock = '\n' + std.rstripChars(|||
          # q38 XPROF_SHM_CLEANUP: free /dev/shm (tmpfs counts against the container's memory limit; J0 OOM 11-07). A capture
          # directory goes once it has been quiet for 120 s and every merged rank file in it is in xprof_samples with the same
          # size; raw rank_N files go after 5 min (the flat cp -n copies above ran first).
          find /dev/shm/xprof -path '*/plugins/profile/*' -type d -not -path '*/rank_[0-9]*' 2>/dev/null |
            while IFS= read -r d; do
              case "$(basename "$d")" in [0-9]*_[0-9]*) ;; *) continue ;; esac
              [ -n "$(find "$d" -type f -newermt '-120 seconds' 2>/dev/null | head -1)" ] && continue
              s="$XPROF_SAMPLES/$(basename "$d")"; ok=1; n=0
              for f in "$d"/rank*_*.xplane.pb; do
                [ -e "$f" ] || continue
                n=$((n + 1))
                [ -e "$s/$(basename "$f")" ] && [ "$(stat -c %s "$f")" = "$(stat -c %s "$s/$(basename "$f")")" ] || ok=0
              done
              [ "$ok" = 1 ] && [ "$n" -gt 0 ] && rm -rf "$d" && echo "q38 xprof shm cleanup: removed $d ($n rank files)"
            done
          find /dev/shm/xprof -path '*/rank_[0-9]*/*' -type f -mmin +5 -delete 2>/dev/null || true
|||, '\n');

local xprofMirror(leg) = agentx.fill(
  |||
    XPROF_OUT="/cdk-outputs/outputs/xprof"
    XPROF_JOB_OUT="/cdk-outputs/outputs/${CDK_JOB_ID}-@ROLE@"
    # q38: every merged sample also lands in xprof_samples/<role>/<capture ts>/; the flat cp -n copies keep only
    # the first sample of each rank file name (q38_int_fr1_mirror_patch.py).
    XPROF_SAMPLES="/cdk-outputs/outputs/xprof_samples/@ROLE@"
    mkdir -p "$XPROF_OUT" "$XPROF_JOB_OUT" "$XPROF_SAMPLES"
    (
      while true; do
        sleep 15
        find /dev/shm/xprof -name '*.xplane.pb' -exec cp -n {} "$XPROF_OUT/" \; 2>/dev/null || true
        find /dev/shm/xprof -name '*.xplane.pb' -exec cp -n {} "$XPROF_JOB_OUT/" \; 2>/dev/null || true
        find /dev/shm/xprof -path '*/plugins/profile/*' -not -path '*/rank_[0-9]*/*' -name 'rank*_*.xplane.pb' 2>/dev/null |
          while IFS= read -r f; do
            s="$XPROF_SAMPLES/$(basename "$(dirname "$f")")"
            [ -e "$s/$(basename "$f")" ] || { mkdir -p "$s" && cp "$f" "$s/"; } 2>/dev/null || true
          done@SHM_CLEANUP@
      done
    ) &
  |||,
  { ROLE: leg.role, SHM_CLEANUP: if q38XprofShmCleanup then q38ShmCleanupBlock else '' }
);

// Which code is serving, published once per job from the prefill leg (both legs run one image).
local provenanceBlock = |||
  if command -v result-updater >/dev/null 2>&1; then
    PROV_SRC="/root/cloud-devkit/vllm-torchtpu"
    PROV_VLLM="$(pip show vllm 2>/dev/null | awk '/^Version:/{print $2}')"
    PROV_RAIDEN="$(pip show tpu-sync-torch 2>/dev/null | awk '/^Version:/{print $2}')"
    PROV_COMMIT="$(git -C "$PROV_SRC" rev-parse HEAD 2>/dev/null)"
    PROV_PENDING="$(git -C "$PROV_SRC" status --porcelain 2>/dev/null | wc -l)"
    PROV_JSON="$(python3 -c "import json,sys; c=sys.argv[1]; n=sys.argv[4]; print(json.dumps({'provenance':{'vllm_version':sys.argv[2] or 'unknown','raiden_version':sys.argv[3] or 'unknown','vllm_torchtpu_commit':c or 'unknown','vllm_torchtpu_dirty':(int(n)>0) if c else None}}))" "$PROV_COMMIT" "$PROV_VLLM" "$PROV_RAIDEN" "$PROV_PENDING")"
    echo "Provenance: ${PROV_JSON}"
    result-updater -a "${PROV_JSON}" || echo "WARNING: result-updater failed for provenance"
  fi
|||;

// The compile-cache key: everything that changes what the leg compiles.
local compileCacheKeySpec(leg) = std.manifestJsonMinified({
  model: modelName,
  role: leg.role,
  args: serveArgs(leg),
  env: {
    [k]: serverEnv(leg)[k]
    for k in std.objectFields(serverEnv(leg))
    if std.isString(serverEnv(leg)[k]) || std.isNumber(serverEnv(leg)[k])
  },
});

// Perfetto slices for the server's startup, from vLLM's own log lines (there is no other hook into
// the engine's phases). cdk's trace generator turns `trace-start-event` / `trace-end-event` pairs
// and `trace-event: ... ts=<us> dur=<us>` lines in any archived container log into slices. Each leg
// is its own process (pid=vllm-<role>); its threads:
//   server  : vllm-<role>, the whole server script.
//   startup : copy-ckpt, compile-cache-pull (cdk.agentx), then server-startup, from `vllm serve` to
//             "Application startup complete".
//   engine  : weight-load, then engine-init-compile-cache-<hit|miss> (profile run, KV cache, warmup),
//             back-dated from the durations vLLM logs when each phase ends.
//   compile : compile, the compilation share of engine init, placed at its end.
// Only the first engine's lines are used: at DP8 every engine prints its own, and overlapping
// slices on one thread do not render.
local serverTraceWatcher = |||
  (
    { set +x; set +e; } 2>/dev/null
    _cc=miss; [ "${CDK_COMPILE_CACHE_HIT:-0}" = "1" ] && _cc=hit
    # A complete slice ending now, back-dated by the duration (seconds) vLLM reports.
    _tx() {
      local now_us dur_us
      now_us=$(date +%s%6N)
      dur_us=$(awk -v s="$3" 'BEGIN { printf "%d", s * 1000000 }')
      echo "trace-event: name=$1 ts=$(( now_us - dur_us )) dur=${dur_us} pid=vllm-@ROLE@ tid=$2"
    }
    for _ in $(seq 1 120); do [ -f "$SERVER_LOG" ] && break; sleep 1; done
    _w=0; _e=0
    tail -F -n +1 "$SERVER_LOG" 2>/dev/null | while IFS= read -r line; do
      case "$line" in
        *"Loading weights took "*" seconds"*)
          [ "$_w" = 1 ] && continue; _w=1
          s=${line##*Loading weights took }; _tx weight-load engine "${s%% seconds*}" ;;
        *"init engine (profile, create kv cache, warmup model) took "*)
          [ "$_e" = 1 ] && continue; _e=1
          s=${line##*warmup model) took }; c=${line##*compilation: }
          _tx "engine-init-compile-cache-${_cc}" engine "${s%% s*}"
          case "$line" in *"compilation: "*) _tx compile compile "${c%% s*}" ;; esac ;;
        *"Application startup complete"*)
          echo "trace-end-event: name=server-startup ts=$(date +%s.%N) pid=vllm-@ROLE@ tid=startup"
          break ;;
      esac
    done
  ) &
|||;

// q38 TP2-RT: prefill-script additions, '' unless PREFILL_SHADOW_RUNTIME / PREFILL_HOSTMEM_LOG (renders unchanged then).
// The presence check runs before the first python3 of the script, so a missing shadow names itself.
local q38PrefillRuntimeBlock =
  (if prefillShadowRuntime then |||
     # q38 TP2-RT PREFILL_SHADOW_RUNTIME: this leg runs the shadow runtime carried in the mapped tree.
     Q38_SR_DIR="${Q38_SHADOW_ROOT:-/root/cloud-devkit/vllm-torchtpu/q38_shadow_rt}"
     for f in "$Q38_SR_DIR/expect.json" "$Q38_SR_DIR/site/libtpu/libtpu.so" /root/cloud-devkit/vllm-torchtpu/src/q38_shadow_runtime.py; do
       [ -f "$f" ] || { echo "FATAL: PREFILL_SHADOW_RUNTIME=true but $f is missing: map a shadow tree (q38_tp2rt_stage.sh) onto the prefill leg" >&2; exit 86; }
     done
     echo "Q38_SHADOW_RUNTIME recipe: $(tr -d ' \n' < "$Q38_SR_DIR/expect.json" | head -c 400)"
   ||| else '')
  + (if prefillHostmemLog then q38HostmemBlock else ''),
// The Q38_HOSTMEM loop: one text for both legs (SV-XPROFA round 3 named it, unchanged, so prefill renders stay byte-identical).
q38HostmemBlock = |||
       # q38 PREFILL_HOSTMEM_LOG: this container's memory cgroup every 60 s (anon, file, shmem = tmpfs incl. /dev/shm).
       # q38 SV-XPROFA: the cgroup is found as the engine's tier telemetry finds it (raiden_connector._cgroup_memory): v2 at this
       # process's own path from /proc/self/cgroup ('0::<path>'), then v2 at the root, then v1 at the memory controller's own path,
       # then v1 at the root; the first readable usage file wins. Q38_HOSTMEM_SRC names the source whenever it changes. Nothing
       # readable: one 'Q38_HOSTMEM UNAVAILABLE <stamp> <reason>' line per tick, never silence.
       (
         { set +x; } 2>/dev/null
         q38_hm_last=""
         while true; do
           q38_v2=""; q38_v1=""; q38_pc=read
           if [ -r /proc/self/cgroup ]; then
             while IFS=: read -r q38_h q38_c q38_p || [ -n "$q38_h" ]; do
               if [ "$q38_h" = 0 ] && [ -z "$q38_c" ]; then q38_v2=$q38_p; else case ",$q38_c," in *,memory,*) q38_v1=$q38_p ;; esac; fi
             done < /proc/self/cgroup
           else
             q38_pc=unreadable
           fi
           q38_src=""; q38_base=""
           if [ -n "$q38_v2" ] && [ "$q38_v2" != / ] && [ -r "/sys/fs/cgroup$q38_v2/memory.current" ]; then q38_src=v2:own; q38_base="/sys/fs/cgroup$q38_v2"
           elif [ -r /sys/fs/cgroup/memory.current ]; then q38_src=v2:root; q38_base=/sys/fs/cgroup
           elif [ -n "$q38_v1" ] && [ "$q38_v1" != / ] && [ -r "/sys/fs/cgroup/memory$q38_v1/memory.usage_in_bytes" ]; then q38_src=v1:own; q38_base="/sys/fs/cgroup/memory$q38_v1"
           elif [ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then q38_src=v1:root; q38_base=/sys/fs/cgroup/memory
           fi
           if [ -n "$q38_src" ] && [ "$q38_src $q38_base" != "$q38_hm_last" ]; then echo "Q38_HOSTMEM_SRC $q38_src $q38_base"; fi
           q38_hm_last="$q38_src $q38_base"
           case "$q38_src" in
             v2:*) echo "Q38_HOSTMEM $(date -u +%FT%TZ) current=$(cat "$q38_base/memory.current") max=$(cat "$q38_base/memory.max") $(grep -E '^(anon|file|shmem) ' "$q38_base/memory.stat" | tr '\n' ' ')" ;;
             v1:*) echo "Q38_HOSTMEM $(date -u +%FT%TZ) usage=$(cat "$q38_base/memory.usage_in_bytes") limit=$(cat "$q38_base/memory.limit_in_bytes") $(grep -E '^(rss|cache|shmem) ' "$q38_base/memory.stat" | tr '\n' ' ')" ;;
             *) echo "Q38_HOSTMEM UNAVAILABLE $(date -u +%FT%TZ) no readable memory.current or memory.usage_in_bytes (/proc/self/cgroup $q38_pc; v2 path '${q38_v2:--}', v1 memory path '${q38_v1:--}'; tried v2 own, v2 root, v1 own, v1 root under /sys/fs/cgroup)" ;;
           esac
           sleep 60
         done
       ) &
     |||,
q38DecodeHostmemBlock = '# q38 DECODE_HOSTMEM_LOG: the PREFILL_HOSTMEM_LOG loop below, on the decode leg (Q38_HOSTMEM lines in this container\'s log).\n'
                        + q38HostmemBlock;

local serverScript(leg) = std.strReplace(agentx.fill(
  |||
    set -x
    if [ "${JOBSET_JOB_INDEX:-0}" != "0" ]; then
      export TPU_RAIDEN_ENGINE_ID="@ROLE@-${JOBSET_JOB_INDEX}-engine"
    fi
    mkdir -p /dev/shm/xprof || exit 1
    rm -f /tmp/libtpu* /dev/shm/libtpu* 2>/dev/null || true
    @XPROF_MIRROR@
    echo "trace-start-event: name=vllm-@ROLE@ ts=$(date +'%s.%N') pid=vllm-@ROLE@ tid=server"
    # Closed on every way out, the TERM the JobSet sends when the bench finishes included: an
    # open slice shows as [UNFINISHED] and hides when the server actually stopped.
    _vllm_trace_end() {
      [ -n "${_VLLM_TRACE_ENDED:-}" ] && return 0
      _VLLM_TRACE_ENDED=1
      echo "trace-end-event: name=vllm-@ROLE@ ts=$(date +'%s.%N') pid=vllm-@ROLE@ tid=server"
    }
    echo "LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS}"
    @PROVENANCE@
    @CKPT_AND_COMPILE_CACHE@

    SERVER_LOG="/tmp/cdk_server.log"
    FATAL_LOG="${CDK_OUTPUT_DIR:-/tmp/cdk_output}/@ROLE@.log.fatal"
    rm -f "$SERVER_LOG" "${SERVER_LOG}.1"
    echo "trace-start-event: name=server-startup ts=$(date +'%s.%N') pid=vllm-@ROLE@ tid=startup"
    @TRACE_WATCHER@
    vllm serve \
      @SERVE_ARGS@ > >(python3 /tmp/cdk_gcs.py ring_tee "$SERVER_LOG") 2>&1 &
    VLLM_PID=$!
    trap 'kill -TERM "$VLLM_PID" 2>/dev/null || true; sleep 1; kill -KILL "$VLLM_PID" 2>/dev/null || true; _vllm_trace_end; exit 0' TERM INT
    @WATCHDOG@
    VLLM_EXIT_CODE=0
    wait "$VLLM_PID" || VLLM_EXIT_CODE=$?
    @FATAL_ON_EXIT@

    _vllm_trace_end
    exit ${VLLM_EXIT_CODE}
  |||,
  {
    ROLE: leg.role,
    XPROF_MIRROR: if q38XprofOn(leg) then std.rstripChars(xprofMirror(leg), '\n') else '',
    PROVENANCE: if leg.role == 'prefill' then q38PrefillRuntimeBlock + std.rstripChars(provenanceBlock, '\n')
                else if decodeHostmemLog then std.rstripChars(q38DecodeHostmemBlock, '\n') else '',
    CKPT_AND_COMPILE_CACHE: std.rstripChars(
      std.rstripChars(q38Ckpt(agentx.resolveCheckpointScript(modelSource=modelSource, fallbackHf=true, allowPvcRoot=true)), '\n') + '\n'
      + std.rstripChars(agentx.compileCacheInitScript('infx-qwen38-flash-agentx-disagg', serverImage, leg.role, compileCacheKeySpec(leg)), '\n') + '\n'
      + agentx.compileCacheUploadScript('http://127.0.0.1:%d/health' % leg.port)
      + (if q38StartupDiag then '\n' + q38DiagBlock(leg.role, leg.port, if useLlmd then '' else '%s:%d' % [cdk.host('proxy'), proxyPort]) else ''),
      '\n'
    ),
    SERVE_ARGS: std.join(' \\\n  ', serveArgs(leg)),
    TRACE_WATCHER: if enablePerfetto then std.rstripChars(agentx.fill(serverTraceWatcher, { ROLE: leg.role }), '\n') else '',
    WATCHDOG: std.rstripChars(cdk.serverFatalWatchdog('VLLM_PID', '$SERVER_LOG', '$FATAL_LOG'), '\n'),
    FATAL_ON_EXIT: std.rstripChars(agentx.snapshotFatalOnExit('VLLM_EXIT_CODE', '$SERVER_LOG', '$FATAL_LOG'), '\n'),
  }
// cdk.agentx's own startup markers (copy-ckpt, compile-cache-pull) say pid=vllm, which puts both
// legs on one process track; move them onto this leg's.
), 'pid=vllm tid=vllm', 'pid=vllm-%s tid=startup' % leg.role);

// vm.max_map_count and the vfio DMA entry limit, which Raiden's pinned host buffers exceed at
// their node defaults (as in qwen3.5-397b-pd-agentx).
local tpuNodeSetup = {
  initContainers: [{
    name: 'tpu-node-setup',
    image: 'busybox',
    securityContext: { privileged: true },
    command: ['/bin/sh', '-c'],
    args: [|||
      sysctl -w vm.max_map_count=8388608
      if [ -f /sys/module/vfio_iommu_type1/parameters/dma_entry_limit ]; then
        echo 2000000 > /sys/module/vfio_iommu_type1/parameters/dma_entry_limit
      fi
    |||],
  }],
};

local serverJob(leg, sidecars=[]) = cdk.job(
  name=leg.role,
  replicas=leg.replicas,
  ttlSeconds=900,
  podSpec=cdk.tpuNodeSelector(accelerator, topology)
          + tpuNodeSetup
          + { terminationGracePeriodSeconds: 180 },
  volumes=[
    agentx.hfTokenVolume(),
    agentx.memoryVolume('huggingface-storage'),
    agentx.memoryVolume('dshm'),
    agentx.memoryVolume('tmp-volume'),
    agentx.memoryVolume('cache-volume'),
    agentx.memoryVolume('triton-volume'),
    agentx.memoryVolume('var-tmp-volume'),
  ] + (if mountPvc then [agentx.sharedCacheVolume(sharedCachePvc)] else []),
  containers=[
    cdk.container(
      name='vllm-tpu-' + leg.role,
      image=serverImage,
      script=agentx.dedupGcsHelper(serverScript(leg), viaEnv=true),
      env=serverEnv(leg) + { _CDK_GCS_B64: agentx.cdkGcsB64 },
      resources=cdk.tpuResources(
        chipsPerHost,
        cpu='200',
        memory='900Gi',
        ephemeralStorage=ephemeralStorage,
        ephemeralStorageLimit=ephemeralStorageLimit,
      ),
      volumeMounts=[
        agentx.mount('huggingface-storage', '/root/.cache/huggingface/'),
        agentx.hfTokenMount(),
        agentx.mount('tmp-volume', '/tmp'),
        agentx.mount('cache-volume', '/root/.cache'),
        agentx.mount('triton-volume', '/root/.triton'),
        agentx.mount('var-tmp-volume', '/var/tmp'),
        agentx.mount('dshm', '/dev/shm'),
      ] + (if mountPvc then [agentx.sharedCacheMount()] else []),
    ) + {
      // The TPU driver needs direct device access, and Raiden pins its premapped host
      // buffers, which needs IPC_LOCK.
      securityContext: { privileged: true, capabilities: { add: ['IPC_LOCK'] } },
      ports: [{ containerPort: leg.port }],
    },
  ] + sidecars,
);

// ---------------------------------------------------------------------------
// The proxy
// ---------------------------------------------------------------------------

// Every pod of each leg as the JobSet's headless service names them ('${JS_NAME}' expands in the
// pods' shells).
// q38 X222-SERVE: under PD_SLICE each role's pod is host (position % 2) of slice pd<position / 2>.
local prefillHosts = if pdSliceOn then [pdHost('prefill', i) for i in std.range(0, prefillReplicas - 1)]
                     else [cdk.host('prefill', replica=i) for i in std.range(0, prefillReplicas - 1)];
local decodeHosts = if pdSliceOn then [pdHost('decode', i) for i in std.range(0, decodeReplicas - 1)]
                    else [cdk.host('decode', replica=i) for i in std.range(0, decodeReplicas - 1)];

local proxyJob = cdk.job(
  name='proxy',
  containers=[
    cdk.container(
      name='vllm-tpu-proxy',
      image=serverImage,
      script=agentx.fill(
        |||
          ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
          set -e
          # The image's own tree, or a --map-dir overlay of it; the SSE content type for
          # aiperf's streaming requests is already in this source.
          for PROXY_SRC in /root/cloud-devkit/vllm-torchtpu/examples/disagg/toy_proxy_server.py \
                           /root/torchtpu-vllm/examples/disagg/toy_proxy_server.py \
                           /workspace/vllm-torchtpu/examples/disagg/toy_proxy_server.py; do
            [ -f "$PROXY_SRC" ] && break
          done
          [ -f "$PROXY_SRC" ] || { echo "FATAL: toy_proxy_server.py not found in the image" >&2; exit 1; }
          echo "proxy: $PROXY_SRC"
          # uvicorn's 5 s default keep-alive closes idle client connections the replay reuses
          # after its gated waits ("Can not write request body"; see glm52-fp8-pd-agentx), so the
          # proxy runs with PROXY_CLIENT_KEEPALIVE_S instead.
          cat > /tmp/run_proxy.py <<'RUN_PROXY_EOF'
          import functools, os, runpy, sys
          import uvicorn
          uvicorn.run = functools.partial(uvicorn.run, timeout_keep_alive=int(sys.argv[1]))
          sys.argv = sys.argv[2:]
          sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
          runpy.run_path(sys.argv[0], run_name="__main__")
          RUN_PROXY_EOF
          exec python3 /tmp/run_proxy.py @KEEPALIVE_S@ "$PROXY_SRC" \
            --host 0.0.0.0 --port @PROXY_PORT@ \
            --prefiller-hosts @PREFILL@ --prefiller-ports @PREFILL_PORT@ \
            --decoder-hosts @DECODE@ --decoder-ports @DECODE_PORT@
        |||,
        {
          PROXY_PORT: proxyPort,
          PREFILL: std.join(' ', prefillHosts),
          PREFILL_PORT: std.join(' ', [std.toString(prefill.port) for _ in prefillHosts]),
          DECODE: std.join(' ', decodeHosts),
          DECODE_PORT: std.join(' ', [std.toString(decode.port) for _ in decodeHosts]),
          KEEPALIVE_S: cdk.numVar('PROXY_CLIENT_KEEPALIVE_S', 3600),
        }
      ),
      // Every PROXY_* recipe variable reaches the proxy as an env var of the same name (routing,
      // admission caps, keep-alive expiry: see toy_proxy_server.py).
      env=cdk.jobSetNameEnv + {
        [k]: cdk.var(k)
        for k in std.objectFields(cdk.vars)
        if (std.startsWith(k, 'PROXY_') && k != 'PROXY_CLIENT_KEEPALIVE_S' || std.member(['Q38_FORM', 'Q38_PROXY_FORM'], k)) && cdk.var(k, '') != ''
      },
      resources={ requests: { cpu: '8', memory: '16Gi' }, limits: { cpu: '16', memory: '32Gi' } },
    ) + { ports: [{ containerPort: proxyPort }] },
  ],
);

// --- llm-d Router (ROUTER=llmd), as qwen3.5-397b-pd-agentx -------------------------
// Envoy :8081 + EPP + endpoint resolver, CPU only. The resolver admits a prefill pod once its vLLM
// /health answers and a decode pod once vLLM :7200 does, registering the decode at the routing
// sidecar's :8000. TPURaidenConnector (and so Qwen4ExpRaidenConnector) answers the sidecar's nixlv2
// prefill handshake.
assert !useLlmd || (llmd.assertEngineKeepAlive(prefill.role, cdk.numVar('VLLM_HTTP_TIMEOUT_KEEP_ALIVE', 600))
                    && llmd.assertEngineKeepAlive(decode.role, cdk.numVar('VLLM_HTTP_TIMEOUT_KEEP_ALIVE', 600)));
local eppKnobs = llmd.knobs({ approxMaxPrefixTokens: cdk.numVar('EPP_APPROX_MAX_PREFIX_TOKENS', maxModelLen) }, cdk=cdk);
// EPP_PATCH=band-a: the plugins and schedulingProfiles of IX's Band A EPP_CONFIG_PATCH
// (reports/ix_qwen35_llmd_configs.md), entry for entry, with the cdk.llmd knobs in place of its
// literals (their defaults are Band A's values) and the decode headroom scaled to this decode's
// cap. As a merge patch its two lists replace the preset's entirely.
local bandAPatch(k) = {
  plugins: [
    { type: 'always-disagg-pd-decider' },
    { type: 'disagg-profile-handler', parameters: { deciders: { prefill: 'always-disagg-pd-decider' } } },
    { type: 'prefill-filter' },
    { type: 'decode-filter' },
    {
      type: 'approx-prefix-cache-producer',
      parameters: {
        autoTune: false,
        blockSizeTokens: k.approxBlockSizeTokens,
        lruCapacityPerServer: k.approxLruCapacity,
        maxPrefixTokensToMatch: k.approxMaxPrefixTokens,
      },
    },
    { type: 'inflight-load-producer' },
    {
      type: 'prefix-cache-affinity-filter',
      parameters: { maxTTFTPenaltyMs: k.approxMaxTtftPenaltyMs, peakPrefillThroughput: k.approxPeakPrefillThroughput },
    },
    {
      type: 'utilization-detector',
      name: 'utilization-detector',
      parameters: { kvCacheUtilThreshold: k.prefillSpillKvUtil, queueDepthThreshold: k.prefillSpillQueueDepth },
    },
    {
      type: 'utilization-filter',
      name: 'decode-headroom',
      parameters: {
        fallbackOnEmpty: true,
        conditions: [
          { metric: 'kv-cache-utilization', maxValue: cdk.numVar('EPP_DECODE_HEADROOM_KV_UTIL', 0.8) },
          { metric: 'running-requests', maxValue: cdk.numVar('EPP_DECODE_HEADROOM_RUNNING', std.max(1, decode.maxNumSeqs - 2)) },
        ],
      },
    },
    {
      type: 'session-affinity-filter',
      name: 'session-affinity-decode',
      parameters: {
        strategy: 'session_id',
        sessionIdConfig: {
          evictionSweepSeconds: 30,
          evictionTtlSeconds: k.sessionTtlSeconds,
          sources: [{ header: 'x-dynamo-session-id' }],
        },
      },
    },
    { type: 'token-load-scorer', parameters: { queueThresholdTokens: 67108864 } },
    { type: 'prefix-cache-scorer' },
    { type: 'active-request-scorer' },
    { type: 'running-requests-size-scorer' },
    { type: 'kv-cache-utilization-scorer' },
    { type: 'max-score-picker' },
  ],
  schedulingProfiles: [
    {
      name: 'prefill',
      plugins: [
        { pluginRef: 'prefill-filter' },
        { pluginRef: 'prefix-cache-affinity-filter' },
        { pluginRef: 'prefix-cache-scorer', weight: 2 },
        { pluginRef: 'token-load-scorer', weight: 1 },
        { pluginRef: 'max-score-picker' },
      ],
    },
    {
      name: 'decode',
      plugins: [
        { pluginRef: 'decode-filter' },
        { pluginRef: 'decode-headroom' },
        { pluginRef: 'session-affinity-decode' },
        { pluginRef: 'prefix-cache-scorer', weight: 2 },
        { pluginRef: 'active-request-scorer', weight: 3 },
        { pluginRef: 'running-requests-size-scorer', weight: 2 },
        { pluginRef: 'kv-cache-utilization-scorer', weight: 1 },
        { pluginRef: 'max-score-picker' },
      ],
    },
  ],
};
// The named patch, then EPP_CONFIG_PATCH over it; without EPP_PATCH, EPP_CONFIG_PATCH alone as before.
local eppPatchJson =
  local raw = cdk.rawVar('EPP_CONFIG_PATCH', '');
  if eppPatchName == '' then raw
  else
    local named = bandAPatch(eppKnobs);
    std.manifestJsonMinified(if raw == '' then named else std.mergePatch(named, std.parseJson(raw)));
local routerJob = llmd.routerJob(
  eppConfig=llmd.eppConfig(
    eppProfile,
    eppKnobs,
    customJson=cdk.rawVar('EPP_CONFIG_JSON', ''),
    patchJson=eppPatchJson,
  ),
  prefillHosts=llmd.jobsetHosts(prefill.role, prefill.replicas, placeholder='{JS_NAME}'),
  decodeHosts=llmd.jobsetHosts(decode.role, decode.replicas, placeholder='{JS_NAME}'),
  prefillPort=prefill.port,
  decodePort=llmd.sidecarPort,
  decodeHealthPort=decode.port,
  // IX's band runs (ix-tpu-pd-bands-llmd-nov04) pinned these digests; the EPP logged llm-d-router
  // build 6e5f3f0 (ix_qwen35_llmd_configs.md 3.5). cdk.llmd's own defaults are :main tags.
  eppImage=cdk.var('LLMD_EPP_IMAGE', 'ghcr.io/llm-d/llm-d-router-endpoint-picker@sha256:d79fb878aeaf85a08fda41f6641d9522b2d066144aa618413e9759681ebf6779'),
  envoyImage=cdk.var('LLMD_ENVOY_IMAGE', llmd.defaultEnvoyImage),
  resolverImage=cdk.var('LLMD_RESOLVER_IMAGE', llmd.defaultResolverImage),
  eppLogLevel=cdk.numVar('LLMD_EPP_V', 2),
);
local decodeSidecars = if !useLlmd then [] else [
  llmd.decodeSidecar(
    modelServerPort=decode.port,
    image=cdk.var('LLMD_SIDECAR_IMAGE', 'ghcr.io/llm-d/llm-d-router-disagg-sidecar@sha256:259638b2ddaeb50b97c2cfd670050d1d747886a2d7c7c0715dc5354abf9df4ca'),
    kvConnector=cdk.var('LLMD_SIDECAR_KV_CONNECTOR', 'nixlv2'),
    prefillMaxRetries=cdk.numVar('LLMD_SIDECAR_PREFILL_MAX_RETRIES', 2),
    prefillRetryBackoff=cdk.var('LLMD_SIDECAR_PREFILL_RETRY_BACKOFF', '200ms'),
  ),
];

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

local modeLine =
  if customDuration then
    'Bench: Mode = CUSTOM DURATION (AIPERF_EXPERIMENTAL_FAST=%s, duration=%ds)' % [if fastMode then '1' else '0', duration]
  else if fastMode then
    'Bench: Mode = FAST (AIPERF_EXPERIMENTAL_FAST=1, duration=1200s, warmup=1 req/lane)'
  else
    'Bench: Mode = CANONICAL (AIPERF_EXPERIMENTAL_FAST=0, duration=3600s, warmup=10 req/lane)';

local prelude = agentx.fill(
  |||
    # Perfetto: from the client's start until the front end answers the probe (closed in
    # datasetInit). A raw echo: the prelude runs before cdk.agentx defines emit_trace_event.
    { set +x; } 2>/dev/null
    echo "trace-start-event: name=wait-servers-ready ts=$(date +%s.%N) pid=bench tid=bench"
    set -x
    export TPU_PJRT_EVENT_TRACE_ENABLE="0"
    echo "Qwen3.8-Flash P/D AgentX: prefill @PREFILL_SHAPE@ (MNBT @PREFILL_MNBT@, seqs @PREFILL_SEQS@) -> decode @DECODE_SHAPE@ (MNBT @DECODE_MNBT@, seqs @DECODE_SEQS@), @CHIPS@ chips"
    PREFILL_HOSTS="@PREFILL_HOSTS@"
    DECODE_HOSTS="@DECODE_HOSTS@"
    # The first pod of each leg, for the paths that address one pod.
    PREFILL_URL="http://${PREFILL_HOSTS%% *}:@PREFILL_PORT@"
    DECODE_URL="http://${DECODE_HOSTS%% *}:@DECODE_PORT@"
    # The front end the replay goes to: the toy proxy, or the llm-d Router's Envoy listener.
    FRONT_URL="http://@FRONT@:@FRONT_PORT@"

    # Paths baked into the client image.
    export INFMAX_CONTAINER_WORKSPACE="/root/cloud-devkit/InferenceX"
    export AIPERF_DIR="/root/cloud-devkit/InferenceX/utils/aiperf"
    if [ -f "${INFMAX_CONTAINER_WORKSPACE}/benchmarks/runtime_settings.sh" ]; then
      source "${INFMAX_CONTAINER_WORKSPACE}/benchmarks/runtime_settings.sh"
    fi

    # Fast vs canonical mode; benchmark_lib.sh turns these into the replay's own flags.
    export AIPERF_EXPERIMENTAL_FAST="@FAST@"
    export DURATION="@DURATION@"
    echo "@MODE_LINE@"

    # No shared-cache PVC on the cpu-only pool, so the tokenizer is fast-synced from GCS
    # (falling back to HuggingFace).
    mkdir -p "$HF_HOME"
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    if [ -f /root/.cache/huggingface/token ]; then
      export HF_TOKEN="$(cat /root/.cache/huggingface/token | tr -d '\r\n')"
      export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    fi

    export MODEL="@MODEL@"
    source "${INFMAX_CONTAINER_WORKSPACE}/benchmarks/benchmark_lib.sh"
    export AIPERF_HF_CLI="$(which hf 2>/dev/null || echo '/usr/local/bin/hf')"
    @SESSION_HEADERS@
    @TOKENIZER_SYNC@
  |||,
  {
    PREFILL_HOSTS: std.join(' ', prefillHosts),
    PREFILL_PORT: prefill.port,
    PREFILL_SHAPE: 'TP%d/DP%d/PP%d' % [prefill.tp, prefill.dp, prefill.pp],
    DECODE_SHAPE: 'TP%d/DP%d' % [decode.tp, decode.dp],
    PREFILL_MNBT: prefill.maxNumBatchedTokens,
    PREFILL_SEQS: prefill.maxNumSeqs,
    DECODE_HOSTS: std.join(' ', decodeHosts),
    DECODE_PORT: decode.port,
    DECODE_MNBT: decode.maxNumBatchedTokens,
    DECODE_SEQS: decode.maxNumSeqs,
    FRONT: if useLlmd then cdk.host('router') else cdk.host('proxy'),
    FRONT_PORT: if useLlmd then llmd.listenerPort else proxyPort,
    // aiperf's session headers, which the EPP's session-affinity filters pin on (and its socket
    // timeouts for Envoy's paused large-body reads), as qwen3.5-397b-pd-agentx sets them.
    SESSION_HEADERS: if !useLlmd then '' else std.rstripChars(|||
      export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID="true"
      export AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING="0"
      export AIPERF_HTTP_TCP_USER_TIMEOUT="${AIPERF_HTTP_TCP_USER_TIMEOUT:-1000000}"
      export AIPERF_HTTP_TCP_KEEPIDLE="${AIPERF_HTTP_TCP_KEEPIDLE:-3600}"
      export AIPERF_HTTP_TCP_KEEPINTVL="${AIPERF_HTTP_TCP_KEEPINTVL:-60}"
      export AIPERF_HTTP_TCP_KEEPCNT="${AIPERF_HTTP_TCP_KEEPCNT:-100}"
    |||, '\n'),
    CHIPS: tpuChips,
    FAST: if fastMode then '1' else '0',
    DURATION: duration,
    MODE_LINE: modeLine,
    MODEL: modelName,
    TOKENIZER_SYNC: std.rstripChars(agentx.fastTokenizerSyncScript(enabled=useGcsTokenizer), '\n'),
  }
);

// With several prefill pods: cdk.agentx's vllm_metric / vllm_metric_ms (defined before
// datasetInit runs) read $VLLM_URL, prefill pod 0, alone, so the per-level prefix-cache and
// request-timing deltas would cover one pod's share of the traffic. Redefined here to sum over
// every prefill pod and to log the per-pod values; a caller that points VLLM_URL elsewhere
// (cdk.agentx's spec_metric) still reads just that URL. Rendered only for PREFILL_REPLICAS > 1.
local prefillMetricsSum = agentx.fill(
  |||
    _Q38_PREFILL0_URL="${VLLM_URL}"
    _Q38_PREFILL_URLS=""
    for h in $PREFILL_HOSTS; do _Q38_PREFILL_URLS="${_Q38_PREFILL_URLS} http://${h}:@PREFILL_PORT@"; done
    echo "Q38_PREFILL_METRICS summing vllm_metric over $(echo ${_Q38_PREFILL_URLS} | wc -w) prefill pods (expect @REPLICAS@):${_Q38_PREFILL_URLS}"
    _q38_prefill_metric() {
      local name="$1" scale="$2" urls u v total=0 per=""
      if [ "${VLLM_URL}" = "${_Q38_PREFILL0_URL}" ]; then urls="${_Q38_PREFILL_URLS}"; else urls="${VLLM_URL}"; fi
      for u in ${urls}; do
        v=$(curl -sf "${u}/metrics" 2>/dev/null | awk -v n="vllm:${name}" -v m="${scale}" \
          '{k=$1; sub(/\{.*/, "", k); if (k==n) s+=$NF} END {printf "%.0f", (s+0)*m}') || v=""
        case "$v" in ''|*[!0-9]*) v=0 ;; esac
        total=$(( total + v ))
        per="${per} ${u#http://}=${v}"
      done
      echo "Q38_PREFILL_METRIC ${name} total=${total}${per}" >&2
      echo "${total}"
    }
    vllm_metric() { _q38_prefill_metric "$1" 1; }
    vllm_metric_ms() { _q38_prefill_metric "$1" 1000; }
  |||,
  { PREFILL_PORT: prefill.port, REPLICAS: prefillReplicas }
);

// Runs once $VLLM_URL (the prefill) answers /health: wait for the decode and the proxy, then
// send one request the way aiperf does, so a broken transfer fails here and not as a replay
// with zero successful requests.
// q38 eval: cdk.agentx's eval calls ${VLLM_HOST}:${PORT}, i.e. prefill pod 0 (a producer). Under eval, point it at the
// front end; VLLM_URL stays on prefill pod 0 so the prefill metrics are still read there.
local q38EvalTarget = if !q38EvalOn then '' else agentx.fill(
  |||
    VLLM_HOST="@FRONT@"; PORT=@FRONT_PORT@
    echo "Q38_EVAL_TARGET ${VLLM_HOST}:${PORT} (VLLM_URL stays ${VLLM_URL})"
  |||,
  { FRONT: if useLlmd then cdk.host('router') else cdk.host('proxy'), FRONT_PORT: if useLlmd then llmd.listenerPort else proxyPort }
);
// q38 (INT; q38_int_decode_window_patch.py): PROFILE_DECODE_SECONDS=<s> stops the decode profiler after <s> s while the
// prefill keeps PROFILE_DURATION_SECONDS (the decode device trace buffer overflows after ~3 s). Empty: nothing rendered.
local q38DecodeWindow =
  local s = cdk.var('PROFILE_DECODE_SECONDS', '');
  if s == '' || !enableXprof || q38XprofLegs == 'prefill' then '' else agentx.fill(
    |||
      capture_xprof_sample() {
        local sample_name="$1"
        local prof_dur="${PROFILE_DURATION_SECONDS:-10}"
        local dec_dur="@DECODE_S@"
        echo "=== [${sample_name}] Starting xprof sampling (prefill ${prof_dur}s, decode ${dec_dur}s; q38 per-leg window) ==="
        emit_trace_event start "${sample_name}"
        curl -sf -X POST "${PREFILL_URL}/start_profile" || true
        curl -sf -X POST "${DECODE_URL}/start_profile" || true
        emit_trace_event start "${sample_name}-decode"
        sleep "${dec_dur}"
        ( curl -sf -X POST "${DECODE_URL}/stop_profile" || true ) &
        local q38_dec_stop=$!
        emit_trace_event end "${sample_name}-decode"
        sleep "$(awk -v a="${prof_dur}" -v b="${dec_dur}" 'BEGIN { r = a - b; if (r < 0) r = 0; print r }')"
        curl -sf -X POST "${PREFILL_URL}/stop_profile" || true
        wait "${q38_dec_stop}" 2>/dev/null || true
        emit_trace_event end "${sample_name}"
        echo "=== [${sample_name}] Stopped xprof sampling ==="
        filter_profile_ranks
      }
      echo "Q38 xprof: per-leg windows, decode @DECODE_S@ s, prefill ${PROFILE_DURATION_SECONDS:-10} s"
    |||,
    { DECODE_S: s }
  );
// q38 (SV-XPROFA): PROFILE_OFFSETS_S=<a>[,<b>...] redefines start_xprof_profiler (datasetInit renders after the lib's
// xprofFns, so this definition wins, as capture_xprof_sample's above does): after 'Phase profiling (profiling) started'
// it sleeps to each absolute offset and calls capture_xprof_sample once. No 180 / 330 / 120 s samples. Empty: nothing rendered.
assert std.all([o + profileDurationSeconds < duration for o in q38ProfileOffsets])
       : 'PROFILE_OFFSETS_S: every offset + PROFILE_DURATION_SECONDS (%s) must end before the replay duration (%d s), got %s'
         % [profileDurationSeconds, duration, q38ProfileOffsetsRaw];
local q38OffsetsProfiler = if std.length(q38ProfileOffsets) == 0 then '' else agentx.fill(
  |||
    start_xprof_profiler() {
      local conc="$1" res_dir="$2"
      local log="${res_dir}/benchmark.log"
      (
        echo "Q38 xprof offsets: waiting for 'Phase profiling (profiling) started' (then captures at @OFFSETS@ s after it, ${PROFILE_DURATION_SECONDS:-10} s each, urls ${XPROF_URLS:-$VLLM_URL})"
        while true; do
          if [ -f "${res_dir}/.steady_started" ] || { [ -f "$log" ] && grep -q "Phase profiling (profiling) started" "$log" 2>/dev/null; }; then
            break
          fi
          sleep 2
        done
        q38_t0=$(date +%s)
        echo "Q38 xprof offsets: profiling phase seen at $(date -u +%FT%TZ) (t0 ${q38_t0}, 2 s poll)"
        q38_k=0
        for q38_off in @OFFSETS@; do
          q38_k=$((q38_k + 1))
          q38_now=$(date +%s)
          if [ "${q38_now}" -lt $((q38_t0 + q38_off)) ]; then sleep $((q38_t0 + q38_off - q38_now)); fi
          echo "Q38 xprof offsets: s${q38_k} start t0+$(( $(date +%s) - q38_t0 )) s (offset ${q38_off} s) $(date -u +%FT%TZ)"
          capture_xprof_sample "xprof-sample-c${conc}-s${q38_k}"
          echo "Q38 xprof offsets: s${q38_k} returned t0+$(( $(date +%s) - q38_t0 )) s $(date -u +%FT%TZ)"
        done
        echo "Staged Profiler: staged captures completed for c${conc}."
      ) &
      PROFILER_PID=$!
    }
    echo "Q38 xprof: absolute offsets @OFFSETS@ s after the profiling phase starts (PROFILE_OFFSETS_S), ${PROFILE_DURATION_SECONDS:-10} s each"
  |||,
  { OFFSETS: std.join(' ', [std.toString(o) for o in q38ProfileOffsets]) }
);
local datasetInit = (if prefillReplicas > 1 then prefillMetricsSum else '') + q38EvalTarget + q38DecodeWindow + q38OffsetsProfiler + agentx.fill(
  |||
    # Each wait is its own slice, inside wait-servers-ready (opened in the prelude).
    wait_up() {
      local name="$1" url="$2" slice="$3" deadline=$(( $(date +%s) + @WAIT_S@ ))
      emit_trace_event start "$slice"
      until curl -sf "$url" > /dev/null; do
        [ "$(date +%s)" -ge "$deadline" ] && { emit_trace_event end "$slice"; emit_trace_event end wait-servers-ready; echo "RESULT: FAIL ($name not up after @WAIT_S@s)"; exit 1; }
        sleep 10
      done
      emit_trace_event end "$slice"
      echo "READY $name $(date -u +%FT%TZ)"
    }
    _i=0; for h in $PREFILL_HOSTS; do wait_up "prefill $h" "http://${h}:@PREFILL_PORT@/health" "wait-prefill-${_i}"; _i=$(( _i + 1 )); done
    _i=0; for h in $DECODE_HOSTS; do wait_up "decode $h" "http://${h}:@DECODE_PORT@/health" "wait-decode-${_i}"; _i=$(( _i + 1 )); done
    @FRONT_WAIT@
    PROBE_BODY='{"model":"@MODEL@","messages":[{"role":"user","content":"What is 2+3? Answer with one number."}],"max_tokens":8,"temperature":0,"stream":false}'
    PROXY_CODE=000
    emit_trace_event start front-probe
    for _ in $(seq 1 60); do
      PROXY_CODE=$(curl -s -m 300 -o /tmp/proxy_probe.out -w '%{http_code}' -H 'Content-Type: application/json' \
        -d "$PROBE_BODY" "${FRONT_URL}/v1/chat/completions" 2>/dev/null || echo 000)
      case "$PROXY_CODE" in 2??) break ;; esac
      sleep 10
    done
    emit_trace_event end front-probe
    emit_trace_event end wait-servers-ready
    echo "proxy probe http_code=${PROXY_CODE}: $(head -c 600 /tmp/proxy_probe.out 2>/dev/null)"
    case "$PROXY_CODE" in 2??) ;; *) echo "RESULT: FAIL (proxy probe: prefill -> decode)"; exit 1 ;; esac

  ||| + agentx.fastDatasetSyncScript(gcsDatasetCache, enabled=true) + |||
    export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
    resolve_trace_source 2>/dev/null || true
  |||,
  {
    WAIT_S: healthTimeoutSeconds,
    MODEL: modelName,
    PREFILL_PORT: prefill.port,
    DECODE_PORT: decode.port,
    // The Router admits the endpoints itself; the probe below waits for it to route.
    FRONT_WAIT: if useLlmd then '' else 'wait_up proxy "${FRONT_URL}/healthcheck" wait-proxy',
  }
);

// build_replay_cmd points the replay at the loopback address, which cdk.agentx rewrites to
// $VLLM_HOST:$PORT, the prefill. Send it to the proxy instead, and refuse to run if anything
// still points elsewhere: a replay against the prefill alone would publish prefill-only numbers
// under this recipe's name.
local replayRewrites = [
  |||
    REPLAY_CMD="${REPLAY_CMD//http:\/\/${VLLM_HOST}:${PORT}/${FRONT_URL}}"
    case "$REPLAY_CMD" in
      *"${FRONT_URL}"*) ;;
      *) echo "RESULT: FAIL (replay command does not target the front end ${FRONT_URL})"; echo "REPLAY_CMD=$REPLAY_CMD"; exit 1 ;;
    esac
  |||,
] + (if !enablePerfetto then [] else [
  // Perfetto: aiperf's own sub-phases, which the benchmark-agentx / warmup / steady-eval slices
  // leave as gaps: its setup (dataset load, mmap store) before warmup, and the record processing
  // and export after the measured window (19.5 min at c600 in j-b9ebbdbd-00b4-4351-8556). Tails
  // this level's benchmark.log, as cdk.agentx's phase monitor does, on a thread of its own so its
  // slices never interleave with the bench thread's.
  |||
    (
      { set +x; set +e; } 2>/dev/null
      _log="${RESULT_DIR}/benchmark.log"; _c="${conc}"
      _ev() { echo "trace-$1-event: name=$2-c${_c} ts=$(date +%s.%N) pid=bench tid=aiperf"; }
      for _ in $(seq 1 600); do [ -f "$_log" ] && break; sleep 1; done
      [ -f "$_log" ] || exit 0
      _st=0
      timeout $(( ${DURATION:-3600} + 43200 )) tail -F -n +1 "$_log" 2>/dev/null | while IFS= read -r line; do
        case "${_st}:${line}" in
          0:*"Starting AIPerf System"*) _ev start aiperf-setup; _st=1 ;;
          1:*"Phase warmup (warmup) started"*|1:*"Phase profiling (profiling) started"*) _ev end aiperf-setup; _st=2 ;;
          2:*"Phase profiling (profiling) complete"*) _ev start records-processing; _st=3 ;;
          3:*"All results received"*) _ev end records-processing; _st=4 ;;
          4:*"Exporting all records"*) _ev start aiperf-export; _st=5 ;;
          5:*"Exporting console data"*) _ev end aiperf-export; break ;;
        esac
      done
    ) &
  |||,
]) + (if !customDuration then [] else [
  agentx.fill(
    |||
      REPLAY_CMD=$(echo "$REPLAY_CMD" | sed -E 's/--benchmark-duration [0-9]+/--benchmark-duration @DURATION@/')
      if [ "@DURATION@" -lt 900 ]; then
        echo "Bench: non-standard duration (@DURATION@s < 900s); adding --unsafe-override to bypass the AIPerf scenario check"
        if [[ "$REPLAY_CMD" != *"--unsafe-override"* ]]; then
          REPLAY_CMD="$REPLAY_CMD --unsafe-override"
        fi
      fi
    |||,
    { DURATION: duration }
  ),
]);

local benchJob = agentx.benchJob(
  name='bench',
  host=if pdSliceOn then pdHost('prefill', 0) else cdk.host(prefill.role),  // q38 X222-SERVE
  port=prefill.port,
  image=clientImage,
  concurrencies=concurrencies,
  resultFilePrefix='qwen3.8_flash_tpu_v7x_pd_conc_',
  benchName=benchName,
  tpuChips=tpuChips,
  prelude=prelude,
  datasetInit=datasetInit,
  exports={
    // For client images whose benchmark_lib.sh reads it; the rewrite above covers the rest.
    AIPERF_SERVER_URL: '${FRONT_URL}',
    // Profile both legs: the start/stop calls go to every URL here. An export rather than pod
    // env, so the shell expands the hosts' ${JS_NAME}. q38 XPROF_LEGS=prefill: the prefill URL only.
    [if enableXprof then 'XPROF_URLS']: if q38XprofLegs == 'prefill' then '${PREFILL_URL}' else '${PREFILL_URL} ${DECODE_URL}',
    // q38 eval: MTP acceptance is counted where drafts are verified (decode pod 0).
    [if q38EvalOn then 'SPEC_DECODE_METRICS_URL']: '${DECODE_URL}',
    KV_OFFLOADING: 'none',
    PRECISION: 'fp8',
    FRAMEWORK: 'vllm',
    IS_MULTINODE: 'true',
    TP: decodeShape.tp,
    EP_SIZE: decodeShape.tp * decodeShape.dp,
    REQUIRE_POWER: '0',
    PP_SIZE: std.toString(prefillShape.pp),
    PCP_SIZE: '1',
    DCP: dcpDecode,
  },
  // The prefill leg ($VLLM_URL) reports prefix hits only when it caches; MTP is off.
  scrapePrefixCache=prefill.prefixCaching,
  scrapeSpecDecode=q38EvalOn && decodeMtp,
  // Queue / prefill / inference timing from the prefill leg ($VLLM_URL).
  scrapeRequestTiming=true,
  enableXprof=enableXprof,
  phaseTrace=enablePerfetto,
  profileDirs=['${CDK_OUTPUT_DIR:-/tmp/cdk_output}/xprof', '/cdk-outputs'],
  replayRewrites=replayRewrites,
  waitTimeoutSeconds=healthTimeoutSeconds,
  cooldownSeconds=cdk.numVar('COOLDOWN_SECONDS', 60),
  startupOpt=true,
  // The dataset sync runs after the probe (datasetInit), not alongside the servers' startup.
  preinitDataset=false,
  env={
    HF_HOME: '/tmp/hf_cache',
    CLIENT_CONCURRENCIES: std.join(' ', [std.toString(c) for c in concurrencies]),
    AIPERF_EXPERIMENTAL_FAST: if fastMode then '1' else '0',
    DURATION: duration,
    GCS_DATASET_CACHE: gcsDatasetCache,
    USE_GCS_TOKENIZER: if useGcsTokenizer then 'true' else 'false',
    CLIENT_PARALLEL_DATASET: 'false',
    [if enableXprof then 'PROFILE_RANKS']: profileRanks,
    [if q38EvalOn then 'EVAL_MAX_MODEL_LEN']: cdk.var('EVAL_MAX_MODEL_LEN', std.toString(maxModelLen)),
    [if q38EvalOn && q38EvalLimit != '' then 'EVAL_LIMIT']: q38EvalLimit,
    [if enableXprof then 'PROFILE_DURATION_SECONDS']: profileDurationSeconds,
  },
  resources={
    local disk = { 'ephemeral-storage': '64Gi' },
    requests: disk,
    limits: disk,
  },
  volumes=[
    agentx.hfTokenVolume(),
    agentx.memoryVolume('dshm'),
    agentx.memoryVolume('sidecar-tmp'),
  ],
  volumeMounts=[
    agentx.hfTokenMount(),
    agentx.mount('sidecar-tmp', '/tmp'),
    agentx.mount('dshm', '/dev/shm'),
  ],
);

// q38 X222-SERVE: the PD_SLICE slice jobs (see the switch). The bootstrap's forced env is IX's X222 v3 single-host block
// (reports/ix_2x2x2_microbench.md s3): GKE injects tpu7x-16 / 2x2x2 / both pods' addresses on a 2x2x2 host, and each key
// below was found by a failure there. TPU_LIBRARY_PATH is dropped only when it does not name a file (IX: blank -> warning).
local pdForce = {
  TPU_ACCELERATOR_TYPE: 'tpu7x-8', TPU_TOPOLOGY: '2x2x1', TPU_CHIPS_PER_HOST_BOUNDS: '2,2,1', TPU_HOST_BOUNDS: '1,1,1',
  TPU_WORKER_ID: '0', TPU_WORKER_HOSTNAMES: 'localhost', TPU_PROCESS_ADDRESSES: '', TPU_PROCESS_PORT: '',
  MEGASCALE_NUM_SLICES: '', MEGASCALE_SLICE_ID: '', MEGASCALE_PORT: '', MEGASCALE_COORDINATOR_ADDRESS: '',
};
local pdUnset = ['TPU_NODE_ID', 'CLOUD_TPU_TASK_ID', 'TPU_MULTIHOST_BACKEND', 'TPU_PROCESS_BOUNDS', 'TPU_CHIPS_PER_PROCESS_BOUNDS'];
local pdBoot = |||
  import base64, json, os, signal, sys, time
  idx = os.environ.get("JOBSET_JOB_INDEX", "")
  sl = os.environ.get("_Q38_PD_SLICE", "?")
  if idx not in ("0", "1"):
      sys.exit("Q38_PD_SLICE FATAL: JOBSET_JOB_INDEX=%r on slice %s, expected 0 or 1 (job-index fieldRef missing?)" % (idx, sl))
  blob = os.environ.get("_Q38_PD_HOST_" + idx, "")
  if not blob:
      sys.exit("Q38_PD_SLICE FATAL: no role blob _Q38_PD_HOST_%s on slice %s" % (idx, sl))
  spec = json.loads(base64.b64decode(blob))
  probe = os.environ.get("_Q38_PD_PROBE", "") == "1"
  for k in [k for k in os.environ if k.startswith("_Q38_PD_")]:
      os.environ.pop(k)
  watch = sorted(set(list(spec.get("force", {})) + list(spec.get("unset", [])) + ["TPU_LIBRARY_PATH", "TPU_SKIP_MDS_QUERY", "JOBSET_JOB_INDEX"]))
  print("Q38_PD_SLICE slice=%s host=%s role=%s index=%s pod=%s" % (sl, idx, spec["role"], spec["index"], os.environ.get("HOSTNAME", "?")), flush=True)
  print("Q38_PD_SLICE env-before " + json.dumps({k: os.environ.get(k) for k in watch}, sort_keys=True), flush=True)
  if spec["role"] == "idle":
      print("Q38_PD_SLICE idle host: holding the slice's second host until the JobSet ends", flush=True)
      signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
      while True:
          time.sleep(3600)
  for e in spec["env"]:
      os.environ[e["name"]] = e["value"]
  for k in spec["unset"]:
      os.environ.pop(k, None)
  lp = os.environ.get("TPU_LIBRARY_PATH")
  if lp is not None and not os.path.isfile(lp):
      os.environ.pop("TPU_LIBRARY_PATH")
  for k, v in spec["force"].items():
      os.environ[k] = v
  os.environ["JOBSET_JOB_INDEX"] = str(spec["index"])
  print("Q38_PD_SLICE env-after " + json.dumps({k: os.environ.get(k) for k in watch}, sort_keys=True), flush=True)
  if probe:
      import re, subprocess
      code = "import jax; d = jax.devices(); print('Q38_PD_SLICE_PROBE devices=%d kind=%s' % (len(d), d[0].device_kind if d else '-'))"
      try:
          r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=600)
          m = re.search(r'Q38_PD_SLICE_PROBE devices=(\d+)', r.stdout or '')
          n, rc, tail = (int(m.group(1)) if m else -1), r.returncode, (r.stdout or '')[-1500:] + (r.stderr or '')[-3000:]
      except subprocess.TimeoutExpired:
          n, rc, tail = -1, -9, 'probe timed out after 600 s'
      print("Q38_PD_SLICE devices=%d probe_rc=%d (want 8: this host's 4 chips x 2 TensorCores; 16 = joined the slice)" % (n, rc), flush=True)
      if n != 8:
          sys.stderr.write(tail + '\n')
          sys.exit("Q38_PD_SLICE FATAL: %d devices visible on this host, want 8; not starting the %s engine" % (n, spec["role"]))
  os.execv(spec["argv"][0], spec["argv"])
|||;
local pdSliceJob(k) =
  local P = serverJob(prefill);
  local D = serverJob(decode, sidecars=decodeSidecars);
  local ps = P.template.spec.template.spec;
  local ds = D.template.spec.template.spec;
  local cp = ps.containers[0];
  local cd = ds.containers[0];
  local envMap(c) = { [e.name]: e for e in std.get(c, 'env', []) };
  local mp = envMap(cp);
  local md = envMap(cd);
  local same(n) = std.objectHas(mp, n) && std.objectHas(md, n) && mp[n] == md[n];
  // JOBSET_JOB_INDEX (a leg with > 1 replica) is replaced: the bootstrap sets each role's original index.
  local roleOnly(m) = [m[n] for n in std.objectFields(m) if !same(n) && n != 'JOBSET_JOB_INDEX'];
  local refs = [e.name for e in roleOnly(mp) + roleOnly(md) if !std.objectHas(e, 'value')];
  assert std.length(refs) == 0 : 'PD_SLICE: role-specific env entries use valueFrom: %s' % std.join(',', refs);
  assert std.length(ps.containers) == 1 && std.length(ds.containers) == 1 : 'PD_SLICE: a leg carries a sidecar container';
  assert cp.image == cd.image && std.get(cp, 'resources') == std.get(cd, 'resources')
         && std.get(cp, 'volumeMounts') == std.get(cd, 'volumeMounts') && std.get(cp, 'securityContext') == std.get(cd, 'securityContext') :
    'PD_SLICE: the prefill and decode containers differ in image, resources, mounts or securityContext';
  assert std.get(ps, 'volumes') == std.get(ds, 'volumes') && std.get(ps, 'initContainers') == std.get(ds, 'initContainers')
         && ps.nodeSelector == ds.nodeSelector && std.get(ps, 'terminationGracePeriodSeconds') == std.get(ds, 'terminationGracePeriodSeconds') :
    'PD_SLICE: the prefill and decode pod specs differ';
  local spec(r) =
    if r == null then { role: 'idle', index: -1 }
    else
      local c = if r.role == 'prefill' then cp else cd;
      local m = if r.role == 'prefill' then mp else md;
      { role: r.role, index: r.index, argv: c.command + c.args, env: [{ name: e.name, value: e.value } for e in roleOnly(m)],
        unset: pdUnset, force: pdForce };
  local hostRole(h) = if 2 * k + h < std.length(pdRoles) then pdRoles[2 * k + h] else null;
  P + {
    name: pdSliceName(k),
    replicas: 2,
    template+: { spec+: { template+: { spec+: {
      nodeSelector+: { 'cloud.google.com/gke-tpu-topology': pdSlice },
      containers: [cp + {
        name: 'vllm-tpu-pd',
        command: ['python3', '-c'],
        args: [pdBoot],
        env: [mp[n] for n in std.objectFields(mp) if same(n) && n != 'JOBSET_JOB_INDEX']
        + [{ name: 'JOBSET_JOB_INDEX', valueFrom: { fieldRef: { fieldPath: "metadata.labels['jobset.sigs.k8s.io/job-index']" } } }]
        + [{ name: '_Q38_PD_SLICE', value: pdSliceName(k) }]
        + (if pdSliceProbe then [{ name: '_Q38_PD_PROBE', value: '1' }] else [])
        + [{ name: '_Q38_PD_HOST_%d' % h, value: std.base64(std.manifestJsonMinified(spec(hostRole(h)))) } for h in [0, 1]],
        ports: std.get(cp, 'ports', []) + std.get(cd, 'ports', []),
      }],
    } } } },
  };

cdk.jobSet(
  jobName,
  // The benchmark decides the outcome: when it finishes, both legs and the proxy go with it.
  successPolicy=cdk.succeedWhen(['bench']),
  // See MAX_RESTARTS above. BlockingRecreate tears the whole gang down before recreating it, so
  // a restarted leg never pairs with a survivor of the previous generation.
  failurePolicy={ restartStrategy: 'BlockingRecreate', maxRestarts: maxRestarts },
  jobs=(if pdSliceOn then [pdSliceJob(k) for k in std.range(0, pdNumSlices - 1)]  // q38 X222-SERVE
        else [serverJob(prefill), serverJob(decode, sidecars=decodeSidecars)])
       + (if useLlmd then [routerJob] else [proxyJob])
       + [benchJob],
) + {
  // The prefill leg as configured by this file and its variables, for infx-qwen38-flash-bench
  // (ENGINE=disagg-prefill), which runs it as one offline engine: the PP8 pipeline prefill of
  // j-7a160f28-2fe2-46ac-b1d3 is PIPELINE_PARALLEL_SIZE_PREFILL=8 on both recipes. Hidden, so it
  // never reaches the rendered JobSet. Same fields as infx-qwen38-flash-agentx's `bench`.
  //   - No --kv-transfer-config: there is no decode to pull the pages, and a producer holds each
  //     finished prompt's KV until a pull or TPU_P2P_WAIT_PULL_TIMEOUT. qwen4_exp's config.py
  //     admits a pipeline engine without the connector (TP1 stages, unified pool,
  //     QWEN4_EXP_PACK_COMPRESSED_K=0, no MTP), which is what the leg already runs.
  //   - The Raiden env stays: env_override.py reserves its premapped host buffer from it and
  //     tpu_compiler.py keys compiles on TPU_RAIDEN_TRANSFER_PARALLELISM, so the engine matches the
  //     leg's even with nothing to transfer.
  //   - The PJRT per-execution logging ENABLE_PERFETTO turns on is dropped (millions of lines into
  //     the bench's pod.log, and no trace generator reads it there); BENCH_PJRT_LOGS=true keeps it.
  bench:: {
    local env = serverEnv(prefill),
    local pjrtLogKeys = ['TPU_VMODULE', 'TPU_STDERR_LOG_LEVEL', 'TPU_PJRT_EVENT_TRACE_ENABLE'],
    image: serverImage,
    modelName: modelName,
    serveArgs: std.filter(function(a) !std.startsWith(a, '--kv-transfer-config'), serveArgs(prefill)),
    serverEnv:
      if cdk.boolVar('BENCH_PJRT_LOGS', false) then env
      else { [k]: env[k] for k in std.objectFields(env) if !std.member(pjrtLogKeys, k) },
    // LIBTPU_INIT_ARGS is in the server env here; the bench script only needs a line to run.
    libtpuScript: '# LIBTPU_INIT_ARGS: from the prefill leg env\n',
    gpuMemoryUtilization: prefill.gpuMemoryUtilization,
    maxNumBatchedTokens: prefill.maxNumBatchedTokens,
    attentionBackend: 'CUSTOM',
    dataParallelSize: prefill.dp,
    ckptPvc: if mountPvc then sharedCachePvc else '',
  },
  // q38 (decode microbench, 2026-11-06): the DECODE leg as configured by this file and its variables, for the q38
  // copy of infx-qwen38-flash-bench (ENGINE=disagg-decode), which runs it as one offline engine. Same fields as
  // `bench` above and built the same way, from serverEnv(decode) and serveArgs(decode):
  //   - No --kv-transfer-config: there is no producer, so the engine prefills each request's context itself
  //     (MBT 2048 chunks) before the timed decode window; no P->D landing happens during a bench step.
  //   - Everything else of the served decode leg stays: TP8 + VLLM_TPU_QSA_MAIN_KV_DCP + EP8, the MTP speculative
  //     config (synthetic acceptance as served), QWEN4_EXP_PACK_MAIN_KV / _PACK_COMPRESSED_K / _MAX_POOL_BYTES, the
  //     DECODE_EXTRA_SERVER_ENV list and the Raiden env (tpu_compiler.py keys compiles on part of it).
  //   - The PJRT per-execution logging keys are dropped as in `bench` (BENCH_PJRT_LOGS=true keeps them).
  // Hidden, so it never reaches a rendered JobSet of this recipe.
  benchDecode:: {
    local env = serverEnv(decode),
    local pjrtLogKeys = ['TPU_VMODULE', 'TPU_STDERR_LOG_LEVEL', 'TPU_PJRT_EVENT_TRACE_ENABLE'],
    image: serverImage,
    modelName: modelName,
    serveArgs: std.filter(function(a) !std.startsWith(a, '--kv-transfer-config'), serveArgs(decode)),
    serverEnv:
      if cdk.boolVar('BENCH_PJRT_LOGS', false) then env
      else { [k]: env[k] for k in std.objectFields(env) if !std.member(pjrtLogKeys, k) },
    libtpuScript: '# LIBTPU_INIT_ARGS: from the decode leg env\n',
    gpuMemoryUtilization: decode.gpuMemoryUtilization,
    maxNumBatchedTokens: decode.maxNumBatchedTokens,
    attentionBackend: 'CUSTOM',
    dataParallelSize: decode.dp,
    ckptPvc: if mountPvc then sharedCachePvc else '',
  },
}
