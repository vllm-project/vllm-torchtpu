// SYNC HEADER (exos-infx-qwen38-flash-agentx-disagg v1.6, q38 copy) -----------------------------------------------------------------
// The Exos (--cluster cdk-tpu-exos) sibling of THIS directory's infx-qwen38-flash-agentx-disagg.jsonnet (the q38 copy in
// /tmp/claude-q38/q38-recipes, synced from md5 a420f95d4ad67bc75866feed9f582baf, 1811 lines). Everything below this header down to
// "The JobSet on cdk-tpu-exos" is that file's lines 1-1742 VERBATIM (all of it before its final cdk.jobSet(...) expression), and its
// bare imports (cdk.agentx, cdk.common, cdk.llmd) resolve to OUR recipes/lib, so every leg's serveArgs / serverEnv /
// kvTransferConfig and the bench script are computed by the same code and libs as on Cloud. The tail is v1.5's (sibling
// infx-qwen38-flash-agentx-disagg.exos15.jsonnet md5 6722969d, kept on disk) plus the v1.6 block before its final expression
// (HIGHC-SIB16, 2026-11-18): RAIDEN_WHEEL=<file name under gs://cloud-devkit/wheels/> (the Cloud knob; Cloud: cdk.agentx
// q38RaidenWheelScript, BLSTACK) installs that Raiden wheel on every vllm-pd leg before vllm serve with the Cloud block's commands, fetched
// from the Exos GCS mount, checked before the image's copy is uninstalled and verified after (FATAL on any failure; no fallback to the
// image wheel). RAIDEN_WHEEL unset (default): the render equals v1.5's byte for byte. v1.5 = v1.4's tail (sibling
// infx-qwen38-flash-agentx-disagg.exos14.jsonnet md5 604c0d6c, kept on disk; v1.3 3b02ae29, v1.2 da5ad007 and v1 ba9e05bc also kept;
// vendored upstream exos_vllm 25a37532 in ../../lib_exos_q38/, unchanged) plus the v1.5 block before its final expression (X1-WATCH,
// 2026-11-17; design credit MM's AGEN sibling 646a3639 and mm/reports/exos_howto_q38.md section 5): optional compile-cache SAVE after
// /health and before the client is released (Q38X_CC_SAVE + Q38X_CC_KEY; chunked tar into the job's own outputs, a state port per leg and
// a sidecar gate), optional FROM warm restore (Q38X_CC_FROM; same leg / key / post src only), and a cachedirs du line
// (Q38X_CC_DIRS_EVERY_S). Every v1.5 knob defaults off, and then the render equals v1.4's byte for byte. v1.4 = v1.3 + the sidecar
// no-pypatch block and the optional Q38X_EXPECT_POST_HASH post pin; v1.3 = v1.2 + per-leg Q38X_MEMHALT self-halt + live Q38X_OUT lines;
// v1.2 = v1.1's guards (Q38X_GUARD, src pin, Q38X_PYPATCH, Q38X_MEMSTAT) + P1 runner-leftover cleanup, P2 no tier-3 cache when the
// Cloud legs have none, P3 rotated vllm_serve.log. No prefetch; filter 0 (v1).
// `diff -u infx-qwen38-flash-agentx-disagg.jsonnet infx-qwen38-flash-agentx-disagg.exos16.jsonnet` must show only this header
// and that tail; parity: unset renders = v1.5's byte for byte; set renders = v1.5's + one wheel block per vllm-pd (sib16_blockcheck.py),
// whose steps map onto the Cloud lib block line by line (sib16_parity.py); v1.5's own parity: served_parity_compare_v6.py (X1-WATCH).
// ---------------------------------------------------------------------------------------------------------------------------------
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

// Refused by the image's P/D path (see the header); failing here beats failing at engine init.
assert !cdk.boolVar('MTP', false) : 'MTP=true: Qwen4ExpRaidenConnector requires speculative decoding off on the producer; DECODE_MTP=true runs it on the decode leg';
// DECODE_MTP=true: MTP on the decode leg only (vllm-torchtpu "decode-only MTP behind a staged
// decode"). The producer ships no draft KV, so the draft's prompt KV starts empty: meant for
// synthetic rejection sampling, which accepts drafts at a fixed mean length whatever they hold,
// as infx-qwen38-flash-agentx reports its curve (SYNTHETIC_ACCEPTANCE_LENGTH 3.32,
// MTP_NUM_SPECULATIVE_TOKENS 3; SYNTHETIC_ACCEPTANCE_LENGTH=0 keeps real acceptance).
local decodeMtp = cdk.boolVar('DECODE_MTP', false);
// q38 MTP-AL (user ruling 11-10 ~01:00Z): SERVE_ENABLE_THINKING sets the served chat template's default thinking mode
// (--default-chat-template-kwargs, every leg). InferenceX serves this model thinking ON. Default false keeps today's
// renders byte-identical; new goal/dashboard runs and new served A/Bs set SERVE_ENABLE_THINKING=true. Per-request
// chat_template_kwargs (LongBench's enable_thinking) still win over this default.
local q38ServeThinking = cdk.boolVar('SERVE_ENABLE_THINKING', false);
// q38 F2 (D-MK, reports/lane_dmk.md s29-s36): PREFILL_MTP_DRAFT_KV=true runs upstream prefill-only MTP (K=1) on the
// producer's last stage so it ships the draft layer's KV; both legs get QWEN4_EXP_PD_DRAFT_KV=1. Needs DECODE_MTP=true
// and a vllm-torchtpu tree with F2. Unset: nothing renders differently.
local prefillMtpDraftKv = cdk.boolVar('PREFILL_MTP_DRAFT_KV', false);
assert !prefillMtpDraftKv || decodeMtp : 'PREFILL_MTP_DRAFT_KV=true needs DECODE_MTP=true';
local decodeMtpTokens = cdk.numVar('MTP_NUM_SPECULATIVE_TOKENS', 3);
// q38 (INT eval patch, q38_int_eval_patch.py): EVAL_ONLY / RUN_EVAL run cdk.agentx's built-in GSM8K or LongBench-v2 eval.
// Under eval: real MTP acceptance by default, the eval client goes to the front end (not prefill pod 0), MTP acceptance
// is read on the decode, and EVAL_MAX_MODEL_LEN reaches the client. Nothing renders differently when neither is set.
local q38EvalOn = cdk.boolVar('EVAL_ONLY', false) || cdk.boolVar('RUN_EVAL', false);
// q38 (INT, q38_int_evallimit_patch.py): EVAL_LIMIT=<n> under eval -> the bench env -> run_lm_eval's --limit n. Unset: nothing renders.
local q38EvalLimit = cdk.var('EVAL_LIMIT', '');
local decodeSyntheticAcceptance =
  local v = cdk.rawVar('SYNTHETIC_ACCEPTANCE_LENGTH', if q38EvalOn then '0' else '3.32');
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
  local d = cdk.numVar('DECODE_CONTEXT_PARALLEL_SIZE_DECODE', 1);
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
  std.toString(if decodeShape.tp == 8 then 512 * std.max(dcpDecode, 1) else 0),
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
local decodePackCompressedK = cdk.var('DECODE_QWEN4_EXP_PACK_COMPRESSED_K', qwen4ExpPackCompressedK);
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
local enablePerfetto = cdk.boolVar('ENABLE_PERFETTO', true);
local profileDurationSeconds = cdk.numVar('PROFILE_DURATION_SECONDS', 10);
local profileRanks = cdk.var('PROFILE_RANKS', std.join(',', [std.toString(i) for i in std.range(0, chipsPerHost - 1)]));

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
    if t < blockSize || t > 8192 then error '%s MAX_NUM_BATCHED_TOKENS must be in [%d, 8192] for qwen4_exp, got %d' % [role, blockSize, t]
    else t,
  gpuMemoryUtilization: legVar(role, 'GPU_MEMORY_UTILIZATION', '0.92'),
  // <ROLE>_ENABLE_PREFIX_CACHING / ENABLE_PREFIX_CACHING (see the header).
  prefixCaching:
    local v = legVar(role, 'ENABLE_PREFIX_CACHING', 'false');
    if !std.member(['true', 'false'], v) then error '%s ENABLE_PREFIX_CACHING must be true or false, got "%s"' % [role, v]
    else v == 'true',
  // 0 is "no cap": the pool is sized from free HBM. Emitted as 0 rather than left unset because
  // the runner's static sizing path reads an unset variable as a 48 GiB cap.
  // A staged decode (TP>1 behind pipeline stages) keeps unpaired main KV, so its static pool for
  // max-num-seqs full-context requests outgrows HBM (77.7 GB + reserve vs 69.3 GB free at TP8,
  // j-e692942c-0a86-477e-aec2); cap it at 48 GiB, which also leaves room for the staging carrier.
  maxPoolBytes: legVar(role, 'QWEN4_EXP_MAX_POOL_BYTES', if role == 'decode' && stagedDecode then '51539607552' else '0'),
  // Pipeline stages before the last never see the sampled tokens async scheduling substitutes
  // (platforms/pp_validation.py), so a pipeline leg runs without it.
  asyncScheduling:
    local v = legVar(role, 'ASYNC_SCHEDULING', if shape.pp > 1 then 'false' else 'true') != 'false';
    if v && shape.pp > 1 then error '%s: pipeline stages need ASYNC_SCHEDULING=false' % role else v,
  ppLayerPartition: legVar(role, 'VLLM_PP_LAYER_PARTITION', ''),
  compileSizes: legVar(role, 'COMPILE_SIZES', ''),
  tokenBucketExtra: legVar(role, 'TPU_TOKEN_BUCKET_EXTRA', ''),
  extraVllmArgs: legVar(role, 'EXTRA_VLLM_ARGS', ''),
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

local kvTransferConfig(leg) = std.manifestJsonMinified({
  kv_connector: 'Qwen4ExpRaidenConnector',
  kv_connector_module_path: 'vllm_torchtpu.models.vllm.qwen4_exp.pd_connector',
  kv_role: leg.kvRole,
  kv_port: 14579,
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
+ (if enableXprof then agentx.profilerArgs('/dev/shm/xprof') else [])
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
    [if leg.role == 'decode' && stagedDecode && legVar(leg.role, 'QWEN4_EXP_PD_STAGING_BYTES', '') != '' then 'QWEN4_EXP_PD_STAGING_BYTES']:
      legVar(leg.role, 'QWEN4_EXP_PD_STAGING_BYTES', ''),
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
    TPU_PREMAPPED_BUFFER_SIZE: legVar(leg.role, 'TPU_PREMAPPED_BUFFER_SIZE', '17179869184'),
    TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES: legVar(leg.role, 'TPU_PREMAPPED_BUFFER_SIZE', '17179869184'),

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
  + (if enablePerfetto || enableXprof then { TPU_PJRT_EVENT_TRACE_ENABLE: '1' } else {})
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
  + (if prefillHostmemLog then |||
       # q38 PREFILL_HOSTMEM_LOG: this container's memory cgroup every 60 s (anon, file, shmem = tmpfs incl. /dev/shm).
       (
         { set +x; } 2>/dev/null
         while true; do
           if [ -r /sys/fs/cgroup/memory.current ]; then
             echo "Q38_HOSTMEM $(date -u +%FT%TZ) current=$(cat /sys/fs/cgroup/memory.current) max=$(cat /sys/fs/cgroup/memory.max) $(grep -E '^(anon|file|shmem) ' /sys/fs/cgroup/memory.stat | tr '\n' ' ')"
           elif [ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
             echo "Q38_HOSTMEM $(date -u +%FT%TZ) usage=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes) limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes) $(grep -E '^(rss|cache|shmem) ' /sys/fs/cgroup/memory/memory.stat | tr '\n' ' ')"
           fi
           sleep 60
         done
       ) &
     ||| else '');

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
    XPROF_MIRROR: if enableXprof then std.rstripChars(xprofMirror(leg), '\n') else '',
    PROVENANCE: if leg.role == 'prefill' then q38PrefillRuntimeBlock + std.rstripChars(provenanceBlock, '\n') else '',
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
        if std.startsWith(k, 'PROXY_') && k != 'PROXY_CLIENT_KEEPALIVE_S' && cdk.var(k, '') != ''
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
  if s == '' || !enableXprof then '' else agentx.fill(
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
local datasetInit = (if prefillReplicas > 1 then prefillMetricsSum else '') + q38EvalTarget + q38DecodeWindow + agentx.fill(
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
    // env, so the shell expands the hosts' ${JS_NAME}.
    [if enableXprof then 'XPROF_URLS']: '${PREFILL_URL} ${DECODE_URL}',
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

// ---------------------------------------------------------------------------
// The JobSet on cdk-tpu-exos (q38 Exos sibling; replaces the Cloud cdk.jobSet(...) call above)
// ---------------------------------------------------------------------------
// Ported from upstream cloud-devkit-recipes 25a37532 infx-qwen38-flash-agentx-disagg.jsonnet, the
// block after "The JobSet: Cloud (GKE) or Exos" (`local exosJobSet = ... exosVllm.fromPdSpec({...})`),
// adapted to this file's locals. Borg linked runners are IPv6-only, diskless and airgapped, have no
// PVCs, CSI drivers, privileged init containers or CPU-only pods, so exos_vllm.pdRecipe owns the
// pods, each leg's `vllm serve` wrapper, weight streaming from GCS, the toy proxy and the client
// container (a sidecar of the prefill pod), while this file contributes each leg's serveArgs and
// serverEnv, computed above exactly as on Cloud. One prefill (1PnD): DECODE_REPLICAS=n runs n decode
// jobs named decode, decode1, ..., decode<n-1> (--map-dir needs '*' or one line per decode job).
// v1.7 (q38 TUNING-KIT5, 2026-11-18; = v1.6 51ed7a64 + this block of counted edits): PREFILL_REPLICAS=m > 1 (2PnD) runs m prefill
// jobs named prefill, prefill1, ..., prefill<m-1> (exos_vllm PdBase.prefillReplicas; --map-dir needs one line per prefill job too);
// prefill 0 runs the proxy (round robin over every prefill and every decode) and the sidecar bench, whose PREFILL_HOSTS (wait loop,
// per-pod prefill metrics) then name the Exos prefill jobs at exos_vllm's port. PREFILL_REPLICAS=1 (default): v1.6's render, byte for byte.
//
// The client is NOT exos_vllm's single-level AgentX bench: the sidecar runs THIS file's cdk.agentx
// bench script (the `benchJob` container above, the same CLIENT_CONCURRENCIES sweep with the same
// per-level replay rewrites, warmup rule, scrapes, parser, cooldown and BENCH_NAME publishing as on
// Cloud), with only its endpoints and airgap handling changed (exosBenchSubs / exosAirgapShim below).
//
// Every difference from the Cloud JobSet is a platform delta listed in scratch/exos/port/PORT_NOTES.md
// and in parity_compare.py's ALLOWED_DELTAS, and each one is commented where it is made.
local exosVllm = import '../../lib_exos_q38/exos_vllm.libsonnet';
local cdkx = import '../../lib_exos_q38/cdk.common';
local exosx = import '../../lib_exos_q38/cdk.exos';

assert cdkx.isTpuExos : 'exos-infx-qwen38-flash-agentx-disagg needs --cluster cdk-tpu-exos; use infx-qwen38-flash-agentx-disagg for Cloud';

// --- Fail closed: modes and knobs that cannot take effect on cdk-tpu-exos ---------------------------
// (1) Variables exos_vllm.pdRecipe would read on its own and that would override the Cloud-derived argv,
// env, image, model or placement behind this file's back. The Cloud recipe has none of them; each concept
// has exactly one knob here, the Cloud one named in the message.
local exosRefusedVars = {
  IMAGE: 'use SERVER_IMAGE',
  MODEL: 'MODEL_NAME picks the checkpoint (streamed from gs://cloud-devkit/ckpt/hf/<MODEL_NAME>)',
  SERVED_MODEL_NAME: 'use MODEL_NAME',
  PREFILL_EXTRA_ARGS: 'use PREFILL_EXTRA_VLLM_ARGS',
  DECODE_EXTRA_ARGS: 'use DECODE_EXTRA_VLLM_ARGS',
  PREFILL_KV_CONFIG: "use PREFILL_EXTRA_VLLM_ARGS=--kv-transfer-config '<json>'",
  DECODE_KV_CONFIG: "use DECODE_EXTRA_VLLM_ARGS=--kv-transfer-config '<json>'",
  EXTRA_ENV_PREFILL: 'use PREFILL_EXTRA_SERVER_ENV',
  EXTRA_ENV_DECODE: 'use DECODE_EXTRA_SERVER_ENV',
  BENCH_MODE: "the client is this recipe's cdk.agentx AgentX sweep",
  CONCURRENCY: 'use CLIENT_CONCURRENCIES',
  CONCURRENCIES: 'use CLIENT_CONCURRENCIES',
  ISL: 'an exos_vllm 8k1k/smoke knob; the AgentX sweep does not read it',
  OSL: 'an exos_vllm 8k1k/smoke knob; the AgentX sweep does not read it',
  RANGE_RATIO: 'an exos_vllm 8k1k/smoke knob; the AgentX sweep does not read it',
  NUM_PROMPTS: 'an exos_vllm 8k1k/smoke knob; the AgentX sweep does not read it',
  NUM_WARMUPS: 'an exos_vllm 8k1k/smoke knob; the AgentX sweep does not read it',
  PROMPT: 'an exos_vllm smoke knob',
  MAX_TOKENS: 'an exos_vllm smoke knob',
  LOAD_FORMAT: 'the legs load the real checkpoint, as on Cloud (streamed from GCS)',
  USE_RUNAI_STREAMER: 'the legs stream the checkpoint from GCS (runai_streamer)',
  GLM_RAIDEN_DEFAULTS: 'GLM-only exos_vllm defaults; this recipe sets its own Raiden env',
  MLA_LAYOUT: 'GLM-only',
  DEST_DCP_SIZE: 'GLM-only (default TPURaidenConnector config, unused here)',
  DEST_CP_INTERLEAVE: 'GLM-only (default TPURaidenConnector config, unused here)',
  HOSTS_PER_LEG: 'one tpu7x 2x2x1 host per leg, as on Cloud',
  TWO_SLICE_PD: 'one tpu7x 2x2x1 slice per leg, as on Cloud',
  PLATFORM: 'tpu7x 2x2x1 per leg, as on Cloud',
  SLICE: 'tpu7x 2x2x1 per leg, as on Cloud',
  PREFILL_HOSTNAMES: 'leg addressing belongs to exos_vllm',
  PREFILL_HOST_IPS: 'leg addressing belongs to exos_vllm',
  DECODE_HOSTNAMES: 'leg addressing belongs to exos_vllm',
  DECODE_HOST_IPS: 'leg addressing belongs to exos_vllm',
  STARTUP_TIMEOUT: 'use HEALTH_TIMEOUT_S',
  PROXY_PROBE: 'always on (the Cloud bench probes the proxy too)',
  RUNNER_GROUP_ID: 'Q38 runs on the shared prewarm Kueue pool only (no BYO pools)',
  RUNNER_GROUP: 'Q38 runs on the shared prewarm Kueue pool only (no BYO pools)',
  PREFILL_RUNNER_GROUP_ID: 'Q38 runs on the shared prewarm Kueue pool only (no BYO pools)',
  DECODE_RUNNER_GROUP_ID: 'Q38 runs on the shared prewarm Kueue pool only (no BYO pools)',
  DECODE_RUNNER_GROUP: 'Q38 runs on the shared prewarm Kueue pool only (no BYO pools)',
  RUNNER_NAME: 'Q38 runs on the shared prewarm Kueue pool only (no runner pinning)',
  NODE_NAME: 'Q38 runs on the shared prewarm Kueue pool only (no node pinning)',
  KUEUE_QUEUE_NAME: 'never steer the queue',
  KUEUE_PRIORITY_CLASS: 'never steer the queue',
  PULL_POLICY: 'linked runners need imagePullPolicy Always',
  GCS_BUCKET: 'outputs stay in gs://cloud-devkit',
};
// (2) Cloud-only modes and knobs of this file that would be silently ignored on Exos.
local exosRefusals =
  [k + ' (' + exosRefusedVars[k] + ')' for k in std.objectFields(exosRefusedVars) if cdk.rawVar(k, '') != '']
  + (if useLlmd then ['ROUTER=llmd (the llm-d Router is a CPU job; Exos has no CPU-only pods: use ROUTER=proxy)'] else [])
  + (if pdSliceOn then ['PD_SLICE=%s (Cloud 2x2x2 slice packing; Exos legs are one 2x2x1 slice each)' % pdSlice] else [])
  + (if enableXprof then ['ENABLE_XPROF=true (no xprof mirror on Exos: the captures would stay in the runner\'s /dev/shm and be lost)'] else [])
  + (if q38EvalOn then ['EVAL_ONLY / RUN_EVAL (the eval client is not ported: it targets the Cloud proxy host and installs lm-eval from PyPI)'] else [])
  + (if q38WeightsHdml then ['WEIGHTS_HDML=true (no Hyperdisk ML PVC on Exos: weights stream from gs://cloud-devkit/ckpt/hf/<MODEL_NAME>; omit it)'] else [])
  + (if cdk.rawVar('SHARED_CACHE_PVC', '') != '' then ['SHARED_CACHE_PVC (no PVCs on Exos)'] else [])
  + (if cdk.rawVar('MODEL_SOURCE', '') != '' then ['MODEL_SOURCE (Exos always streams the checkpoint from GCS)'] else [])
  + (if cdk.rawVar('COMPILE_CACHE_ENABLED', '') != '' && compileCacheEnabled then ['COMPILE_CACHE_ENABLED=true (no GCS compile-cache archive on Exos; pass false or omit it)'] else [])
  + [k + ' (no GCS compile-cache archive on Exos)' for k in ['COMPILE_CACHE_GCS', 'COMPILE_CACHE_TAG', 'COMPILE_CACHE_REFRESH'] if cdk.rawVar(k, '') != '']
  + (if q38StartupDiag then ['Q38_STARTUP_DIAG=true (a block of the Cloud server script, which Exos does not run)'] else [])
  + (if prefillHostmemLog then ['PREFILL_HOSTMEM_LOG (a block of the Cloud server script, which Exos does not run; PREFILL_SHADOW_RUNTIME=true needs PREFILL_HOSTMEM_LOG=false here)'] else [])
  + (if maxRestarts != 0 then ['MAX_RESTARTS=%d (exos_vllm runs neverRestart and linked-runner losses are not retried: pass 0 or omit it)' % maxRestarts] else [])
  + [k + ' (linked runners take no pod resources)' for k in ['EPHEMERAL_STORAGE', 'EPHEMERAL_STORAGE_LIMIT'] if cdk.rawVar(k, '') != '']
  + (if std.startsWith(serverImage, 'us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/') then
       ['SERVER_IMAGE in cloud-ullm-inference-ci-cd (the runner service account cannot pull it; upstream silently swaps the image, this copy refuses)'] else []);
assert std.length(exosRefusals) == 0 :
  'exos-infx-qwen38-flash-agentx-disagg cannot honour on cdk-tpu-exos: ' + std.join('; ', exosRefusals);

// --- Fixed by exos_vllm.pdRecipe (one-host legs, TWO_SLICE_PD): the endpoints and Raiden bases ------
// Platform delta: the leg ports are exos_vllm's (vllm serve --host <leg IPv6> --port 25200 / 25100,
// the toy proxy on [::1]:25000 in the prefill pod), not this file's 8000 / 7200 / proxy :10000.
local exosPrefillPort = 25200;
local exosDecodePort = 25100;
local exosProxyUrl = 'http://[::1]:25000';
// exos_vllm exports TPU_RAIDEN_RESHARD_PORT_BASE 27000 / 28000 (+100 dispatch) per leg; the Cloud
// CONTROLLER_PORT (raidenPortBase), which reaches the legs unchanged, must agree with it.
assert prefill.raidenPortBase == 27000 && decode.raidenPortBase == 28000 :
  'exos_vllm hard-codes the Raiden port bases 27000 (prefill) / 28000 (decode)';
// Pod DNS names of the legs (exos_vllm legHostnames: <jobset>-<job>-0-0.<jobset>, decode job i > 0 is decode<i>).
local exosDecodeHosts = [cdk.host(if i == 0 then 'decode' else 'decode%d' % i) for i in std.range(0, decodeReplicas - 1)];
// v1.7: prefill job i > 0 is prefill<i> on Exos (Cloud: replica i of job prefill).
local exosPrefillHosts = [cdk.host(if i == 0 then 'prefill' else 'prefill%d' % i) for i in std.range(0, prefillReplicas - 1)];

// --- KV transfer config (as upstream) ---------------------------------------------------------------
// vLLM keeps the last --kv-transfer-config, so on Cloud a leg's <ROLE>_EXTRA_VLLM_ARGS
// --kv-transfer-config '<json>' (e.g. TPUMultiConnector + TPURaidenOffloadingConnector) replaces the
// recipe's. sanitizeServeArgs strips every --kv-transfer-config here, so that JSON is carried over as the
// leg's KV config. Platform delta: TPURaidenOffloadingConnector's workers dial their controller at
// store_server_ip, falling back to 127.0.0.1, which an IPv6-only linked runner cannot reach, so it
// defaults to [::1] (bracketed: gRPC parses "<ip>:<port>"; j-65d0d1ed-3271-4741-b84c), in a top-level
// connector or one nested in a TPUMultiConnector. Nothing else in the JSON changes.
local exosOffloadIpv6(c) =
  local extra = std.get(c, 'kv_connector_extra_config', {});
  c + (
    if c.kv_connector == 'TPURaidenOffloadingConnector' then
      { kv_connector_extra_config: { store_server_ip: '[::1]' } + extra }
    else if std.objectHas(extra, 'connectors') then
      { kv_connector_extra_config: extra { connectors: std.map(exosOffloadIpv6, extra.connectors) } }
    else {}
  );
local exosKvConfig(leg) =
  local x = std.stripChars(leg.extraVllmArgs, ' \t\n');
  local flag = "--kv-transfer-config '";
  if std.startsWith(x, flag) && std.endsWith(x, "'") then
    std.manifestJsonMinified(exosOffloadIpv6(std.parseJson(x[std.length(flag):std.length(x) - 1])))
  else if std.length(std.findSubstr('--kv-transfer-config', x)) > 0 then
    error '%s EXTRA_VLLM_ARGS: on cdk-tpu-exos --kv-transfer-config must be the whole value, as --kv-transfer-config \'<json>\'' % leg.role
  else kvTransferConfig(leg);

// --- Leg argv and env (as upstream) -----------------------------------------------------------------
// sanitizeServeArgs drops what exos_vllm sets itself: the model path, --served-model-name, --host,
// --port, --max-model-len (re-added with the same MAX_MODEL_LEN) and --kv-transfer-config (above);
// keepTp=true keeps --tensor-parallel-size, as the legs differ. Every other flag passes in order.
// Platform delta: --data-parallel-address=::1 for a DP leg (IPv6 loopback).
local exosLegArgs(leg) =
  exosVllm.sanitizeServeArgs(serveArgs(leg), keepTp=true)
  + (if leg.dp > 1 then ['--data-parallel-address=::1'] else []);
// sanitizeServerEnv drops the pod identity, TPU topology and accelerator keys tpu_runner injects, the
// checkpoint / compile-cache / cache-path keys, the Raiden advertise host, job name and port bases and the
// KV transfer / side-channel ports, all of which exos_vllm sets itself (and object-valued fieldRefs).
// Every other key reaches the leg unchanged: exos_vllm exports them in the leg's branch of its script,
// after its own defaults, so this file's values win.
local exosLegEnv(leg) =
  exosVllm.sanitizeServerEnv(serverEnv(leg)) + {
    // Platform delta (upstream j-cc718d5f-c894-47dd-8b9d): Qwen PD admits a transfer only when both legs
    // digest the same loaded checkpoint tensors; exos_vllm's runai loader otherwise skips other PP stages'
    // layers, MTP tensors and non-local experts per rank. Stream the full checkpoint to every rank, as
    // Cloud loads it.
    EXOS_WEIGHT_LOAD_FILTER: cdk.var('EXOS_WEIGHT_LOAD_FILTER', '0'),
    // Platform delta: IPv6 loopback (Cloud 127.0.0.1); exos_vllm exports the same again for one-host legs.
    VLLM_DP_MASTER_IP: '::1',
  };
// Every PROXY_* recipe variable reaches the toy proxy as env, as on Cloud (where it is the proxy pod's
// env). Platform delta: the proxy runs inside the prefill pod's vllm-pd container, so the legs see them too.
local exosProxyEnv = {
  [k]: cdk.var(k)
  for k in std.objectFields(cdk.vars)
  if std.startsWith(k, 'PROXY_') && cdk.var(k, '') != ''
};

local exosPd = exosVllm.fromPdSpec({
  name: jobName,
  slice: '2x4',
  hostsPerLeg: 1,
  decodeReplicas: decodeReplicas,
  prefillReplicas: prefillReplicas,
  glmRaidenDefaults: false,
  proxyProbe: true,
  startupTimeout: healthTimeoutSeconds,
  serverImage: serverImage,
  clientImage: clientImage,
  modelName: modelName,
  maxModelLen: maxModelLen,
  // 'agentx' only makes exos_vllm build the sidecar-bench container; its script is replaced below.
  benchMode: 'agentx',
  concurrency: concurrencies[0],
  fast: fastMode,
  duration: duration,
  prefillKvConfig: exosKvConfig(prefill),
  decodeKvConfig: exosKvConfig(decode),
  prefillArgs: exosLegArgs(prefill),
  decodeArgs: exosLegArgs(decode),
  prefillEnv: exosLegEnv(prefill),
  decodeEnv: exosLegEnv(decode),
  serverEnv: {
    VLLM_ENGINE_READY_TIMEOUT_S: std.toString(cdk.numVar('VLLM_ENGINE_READY_TIMEOUT_S', 7200)),
    // exos_vllm reads ENABLE_PERFETTO with its own default (false) and unsets TPU_VMODULE /
    // TPU_PJRT_EVENT_TRACE_ENABLE on the decode when it is not true; this file defaults it to true.
    // Pass this file's value so the decode keeps exactly the Cloud PJRT logging env.
    ENABLE_PERFETTO: if enablePerfetto then 'true' else 'false',
  } + exosProxyEnv,
});

// --- The client: this file's cdk.agentx bench, inside exos_vllm's sidecar --------------------------
// The `benchJob` container above is the Cloud client, built by the same code and libs. Its script runs
// unchanged apart from these substitutions, each counted (a Cloud edit that moves one fails the render):
local exosReplaceN(s, old, new, n) =
  local c = (std.length(s) - std.length(std.strReplace(s, old, ''))) / std.length(old);
  assert c == n : 'exos bench port: expected %d of "%s" in the Cloud bench script, found %d' % [n, old, c];
  std.strReplace(s, old, new);
local exosCloudBench =
  local cs = benchJob.template.spec.template.spec.containers;
  assert std.length(cs) == 1 : 'exos bench port: the Cloud bench job has %d containers, expected 1' % std.length(cs);
  cs[0];
// Platform delta (airgap; Exos only, after cdk.agentx's own aiperf venv binding): linked runners reach
// Google APIs but not huggingface.co or PyPI. Keep the HF libraries offline once the prelude has staged the
// tokenizer and dataset from GCS, apply exos_vllm's two aiperf offline-resolution patches (its
// benchAgentxScript, 25a37532, verbatim) and keep benchmark_lib.sh from reinstalling aiperf with uv when
// the prebaked venv was bound (AIPERF_DEPS_READY=1). None of this touches the replay command.
local exosAirgapShim = |||
  # q38 exos airgap shim (cdk-tpu-exos only; see the recipe tail): no huggingface.co / PyPI on linked runners.
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  if [ "${AIPERF_DEPS_READY:-0}" = "1" ]; then
    ensure_aiperf_venv() { return 0; }
    setup_aiperf_venv() { return 0; }
  fi
  python3 - <<'Q38_EXOS_PYTOK'
  p = '/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/common/tokenizer.py'
  try:
    with open(p) as f:
      s = f.read()
    target = '    def _resolve_local_snapshot(cls, name: str, revision: str) -> str:\n'
    patch = target + '        if __import__("os").path.isdir(name):\n            return name\n'
    if target in s and 'isdir(name)' not in s:
      with open(p, 'w') as f:
        f.write(s.replace(target, patch, 1))
  except Exception:
    pass
  p2 = '/root/cloud-devkit/InferenceX/utils/aiperf/src/aiperf/dataset/loader/base_hf_dataset.py'
  try:
    with open(p2) as f:
      s2 = f.read()
    t2 = '        return hf_load_dataset(\n            self.hf_dataset_name,'
    r2 = (
        '        ds_path = self.hf_dataset_name\n'
        '        try:\n'
        '            from huggingface_hub import snapshot_download\n'
        '            ds_path = snapshot_download(self.hf_dataset_name, repo_type="dataset", local_files_only=True, **revision)\n'
        '        except Exception:\n'
        '            pass\n'
        '        return hf_load_dataset(\n'
        '            ds_path,'
    )
    if t2 in s2:
      with open(p2, 'w') as f:
        f.write(s2.replace(t2, r2, 1))
  except Exception:
    pass
  Q38_EXOS_PYTOK
|||;
local exosBenchSubs = [
  // [old, new, expected count]
  // Platform delta (ports): the bench's $VLLM_URL is the prefill pod at exos_vllm's port. The pod DNS name
  // is the same on both clusters (<jobset>-prefill-0-0.<jobset>), so VLLM_HOST, and with it the per-level
  // replay rewrite to the front end, are unchanged.
  ['export PORT=%d\nVLLM_HOST="%s"\n' % [prefill.port, cdk.host(prefill.role)],
   'export PORT=%d\nVLLM_HOST="%s"\n' % [exosPrefillPort, cdk.host(prefill.role)], 1],
  // The same port in cdk.agentx's startupOpt PORT restore after the prelude.
  ['export PORT=%d\nVLLM_URL="http://${VLLM_HOST}:${PORT}"\n' % prefill.port,
   'export PORT=%d\nVLLM_URL="http://${VLLM_HOST}:${PORT}"\n' % exosPrefillPort, 1],
  ['PREFILL_URL="http://${PREFILL_HOSTS%% *}:' + std.toString(prefill.port) + '"',
   'PREFILL_URL="http://${PREFILL_HOSTS%% *}:' + std.toString(exosPrefillPort) + '"', 1],
  ['DECODE_URL="http://${DECODE_HOSTS%% *}:' + std.toString(decode.port) + '"',
   'DECODE_URL="http://${DECODE_HOSTS%% *}:' + std.toString(exosDecodePort) + '"', 1],
  // Platform delta: the front end is exos_vllm's toy proxy on the prefill host's loopback (no proxy pod).
  ['FRONT_URL="http://' + cdk.host('proxy') + ':' + std.toString(proxyPort) + '"',
   'FRONT_URL="' + exosProxyUrl + '"', 1],
  // Platform delta: decode i > 0 is its own job decode<i> on Exos (Cloud: replica i of job decode).
  ['DECODE_HOSTS="' + std.join(' ', decodeHosts) + '"', 'DECODE_HOSTS="' + std.join(' ', exosDecodeHosts) + '"', 1],
  ['"http://${h}:' + std.toString(prefill.port) + '/health" "wait-prefill-${_i}"',
   '"http://${h}:' + std.toString(exosPrefillPort) + '/health" "wait-prefill-${_i}"', 1],
  ['"http://${h}:' + std.toString(decode.port) + '/health" "wait-decode-${_i}"',
   '"http://${h}:' + std.toString(exosDecodePort) + '/health" "wait-decode-${_i}"', 1],
  // The airgap shim, right after cdk.agentx's aiperf venv binding.
  [agentx.ensureAiperfVenvScript, agentx.ensureAiperfVenvScript + exosAirgapShim, 1],
] + (if prefillReplicas == 1 then [] else [
  // v1.7 (2PnD): the bench's prefill host list (the wait-prefill loop and the per-pod prefill metrics) names the Exos prefill jobs,
  // and the per-pod prefill metrics read exos_vllm's prefill port.
  ['PREFILL_HOSTS="' + std.join(' ', prefillHosts) + '"', 'PREFILL_HOSTS="' + std.join(' ', exosPrefillHosts) + '"', 1],
  ['_Q38_PREFILL_URLS="${_Q38_PREFILL_URLS} http://${h}:' + std.toString(prefill.port) + '"',
   '_Q38_PREFILL_URLS="${_Q38_PREFILL_URLS} http://${h}:' + std.toString(exosPrefillPort) + '"', 1],
]);
local exosBenchScript = std.foldl(function(s, p) exosReplaceN(s, p[0], p[1], p[2]), exosBenchSubs, exosCloudBench.args[0]);
// Platform delta (GCS auth): cdk_gcs.py, which the bench uses for the tokenizer and dataset sync and its
// result-updater path, fetches its token from metadata.google.internal; linked runners serve it at
// $GCE_METADATA_HOST ([::1]:18081), as exos_vllm's exos_gcs_cp.py reads it. Same helper, that one line.
local exosCdkGcsB64 = std.base64(exosReplaceN(
  agentx.cdkGcsPy,
  '"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",',
  '"http://%s/computeMetadata/v1/instance/service-accounts/default/token" % (os.environ.get("GCE_METADATA_HOST") or "metadata.google.internal"),',
  1,
));
// The Cloud bench env, with PORT and the helper swapped, plus exos_vllm's sidecar plumbing: the IPv6 env
// (AIPERF_HTTP_IP_VERSION=auto among it), the Python bootstrap its prologue runs, EXOS_RECIPE for its log lines.
local exosSidecarEnv =
  local cloudEnv = [e for e in exosCloudBench.env if !std.member(['PORT', '_CDK_GCS_B64'], e.name)];
  local env = cloudEnv + cdkx.envList(
    { PORT: std.toString(exosPrefillPort), _CDK_GCS_B64: exosCdkGcsB64, EXOS_RECIPE: jobName }
    + exosx.ipv6Env + exosVllm.pyPatchEnv
  );
  assert std.length(std.set([e.name for e in env])) == std.length(env) : 'exos bench port: duplicate sidecar env names';
  exosx.orderContainerEnv(env);
// exos_vllm.sidecarBenchContainer's prologue (25a37532): non-head exit, the done-file EXIT trap the prefill
// waits on, result-updater staging, the IPv6 shim build, the wait for the legs + proxy probe (SYNC_DIR/ready)
// and LD_PRELOAD. Its own tokenizer / dataset pre-staging is dropped (the Cloud prelude stages them), and
// the ready file is read for the startup time only, so its METRICS_URL / SERVE_* never reach the client.
local exosSidecarScript = |||
  set -uo pipefail
  WORKER_ID="${TPU_WORKER_ID:-0}"
  if [ "${PD_LEG:-}" = "decode" ] || [ "${WORKER_ID}" != "0" ]; then
    echo "=== ${EXOS_RECIPE} sidecar-bench: worker=${WORKER_ID} leg=${PD_LEG:-worker} is non-head; exiting ==="
    exit 0
  fi
  trap 'rc=$?; if [ -n "${SYNC_DIR:-}" ] && [ ! -f "${SYNC_DIR}/done" ]; then echo "$rc" > "${SYNC_DIR}/done" 2>/dev/null || true; fi; echo "=== ${EXOS_RECIPE} sidecar-bench: exiting rc=$rc ==="; sleep ${EXIT_LINGER_S:-60}' EXIT
||| + exosx.stageResultUpdaterSh + exosVllm.pyPatchSnippet + |||
  SYNC_DIR="/tmp/workspace/sync_${JS_NAME:-run}"
  mkdir -p "${SYNC_DIR}"
  echo "=== ${EXOS_RECIPE} sidecar-bench: waiting for ${SYNC_DIR}/ready ==="
  while [ ! -f "${SYNC_DIR}/ready" ] && [ ! -f "${SYNC_DIR}/failed" ]; do
    sleep 5
  done
  if [ -f "${SYNC_DIR}/failed" ]; then
    echo "=== ${EXOS_RECIPE} sidecar-bench: vLLM server failed before ready ==="
    echo "1" > "${SYNC_DIR}/done"
    exit 1
  fi
  _q38_exos_startup_s="$(sed -n 's/^EXOS_STARTUP_S="\(.*\)"$/\1/p' "${SYNC_DIR}/ready")"
  echo "=== ${EXOS_RECIPE} sidecar-bench: legs and proxy ready (startup=${_q38_exos_startup_s}s); running the q38 cdk.agentx sweep CLIENT_CONCURRENCIES='${CLIENT_CONCURRENCIES}' ==="
||| + exosx.ipv6PreloadSnippet + exosBenchScript;
local exosSwapSidecar(job) =
  if job.name != 'prefill' then job
  else
    local cs = job.template.spec.template.spec.containers;
    assert std.length([c for c in cs if c.name == 'sidecar-bench']) == 1 : 'exos bench port: no single sidecar-bench in the prefill job';
    job + { template+: { spec+: { template+: { spec+: {
      containers: [
        if c.name == 'sidecar-bench' then c + { args: [exosSidecarScript], env: exosSidecarEnv } else c
        for c in cs
      ],
    } } } } };
local exosJobSet = exosPd + { spec+: { replicatedJobs: [exosSwapSidecar(j) for j in super.replicatedJobs] } };

// --- v1.2 (EXOS-SERVED, 2026-11-16): v1.1's per-leg Q38X prelude + the RAM-fit platform cuts P1 / P2 / P3 (lead rulings 22:5xZ) ---
// v1.2 = v1 (ba9e05bc) + three shell blocks in each leg's vllm-pd script (prefill, decode, decode<i>) + two counted edits of
// exos_vllm 25a37532's leg script; the sidecar is unchanged. The blocks export / unset nothing the engine reads; served_parity_compare_v3.py
// checks every piece, reverts the two edits, strips the blocks, and requires the rest of every container to equal v1's render byte for byte.
//   q38x init   (before exos_vllm's python bootstrap) the Q38X_V12 banner; Q38X_MEMSTAT start / outdir / workspace-at-start, then
//               P1 (EXOS_WORKSPACE_CLEAN, default true; mirrors MM's sibling v3 (e)): a runner's /tmp/workspace outlives jobs (M: MM X48
//               started with 29 GiB of an earlier job's caches; PD T93d / T93e / T93f started with Shmem 40 / 63 / 34 GiB). Keep the runner's
//               own entries (bin etc images models), this job's sync dir and every hidden entry (exos_vllm's .exos_* files); EMPTY the
//               compile-cache dirs (tier3_cache inductor_cache jax_cache vllm_cache torch_tpu_cache vllm_pd_cache); REMOVE anything else
//               (other jobs' sync dirs, model_local, ...). Each action is logged with its size, then a Q38X_MEMSTAT after-clean line.
//               A background Q38X_MEMSTAT line every EXOS_MEMSTAT_EVERY_S; Q38X_TREE: the mapped tree's src hash before the bootstrap
//               (FATAL against Q38X_EXPECT_SRC_HASH); and q38x_logcap for P3 (below).
//   q38x post   (after the bootstrap) Q38X_PYPATCH: exos_vllm 25a37532's _EXOS_PYPATCH rewrites 7 vllm_torchtpu files of the mapped tree
//               in place (M: 11-14 j-b8f3be5e decode log; CPU proof j-28bf9561 on trees e2511c6df / 359b43536); lists them, FATAL outside.
//   q38x guard  (before each leg's `trace-start-event: name=vllm-<role>`) Q38X_GUARD: FATAL unless vllm_torchtpu resolves to the mapped
//               tree and the tree still has its post-bootstrap hash.
//   P2 (edit)   when the Cloud legs set no tier-3 root (COMPILE_CACHE_ENABLED=false, as v8 / S0 / HCCAR: M, rendered Cloud manifests),
//               exos_vllm's three TORCH_TPU_*_TIER3_* exports and the matching mkdir argument are removed, so the Exos legs carry the Cloud
//               cache env exactly (no tier-3 executables accumulating in tmpfs). Cloud renders with the tier-3 root set (the cache on)
//               keep exos_vllm's exports unchanged.
//   P3 (edit)   exos_vllm tees each leg's `vllm serve` output into tmpfs /tmp/vllm_serve.log with `tee -a`, never rotated (v7's prefill
//               container log: 3.28 GiB in 3.05 h, 95.6 % per-execution TPU_VMODULE lines that PD readers count, so TPU_VMODULE stays the
//               Cloud value). Both tees become q38x_logcap: the same pass-through to the container log, plus /tmp/vllm_serve.log rotated
//               at Q38X_LOG_SEGMENT_MB (default 1024) on a line boundary; each full segment is renamed, copied to
//               $CDK_OUTPUT_DIR/q38x/vllm_serve.<leg>.NNN.log in the background and deleted locally. exos_vllm still copies the live
//               /tmp/vllm_serve.log (the last segment) to $CDK_OUTPUT_DIR/<leg>.log at exit, so the full log = the q38x segments + that
//               file. If python3 fails, the wrapper falls back to `tee -a /tmp/vllm_serve.log` for the rest of the stream.
// Readers read outputs/: <job>-<prefill|decode>-0/<pod>/vllm-pd/q38x/q38x.log (every Q38X line, refreshed at every MEMSTAT tick, at the
// guard and on a FATAL), q38x/vllm_serve.<leg>.NNN.log and <leg>.log. Never a non-follow pod log (truncates at ~4.2-4.5 MB, MM M).
// Knobs (read only by this tail; none reaches the engine env or argv):
//   EXOS_MEMSTAT_EVERY_S=<n>       default 30 (integer 10..3600).
//   Q38X_EXPECT_SRC_HASH=<10 hex>  the car's tree src hash (its launcher's SRC pin). Empty (default): the hash is logged, not pinned.
//   EXOS_WORKSPACE_CLEAN=true|false  default true (P1).
//   Q38X_LOG_SEGMENT_MB=<n>        default 1024 (integer 64..8192; P3).
// Refused: Q38X_WORKSPACE_SYNC as a recipe variable or in either leg's server env (MM's opt-in for exos_vllm f1e83d94's /workspace
// source overlay; this tail builds on 25a37532, which has no overlay, so the key can only be a mistake here).
local q38xMemEvery = cdk.numVar('EXOS_MEMSTAT_EVERY_S', 30);
local q38xExpectSrc = cdk.var('Q38X_EXPECT_SRC_HASH', '');
local q38xWsClean = cdk.boolVar('EXOS_WORKSPACE_CLEAN', true);
local q38xSegMb = cdk.numVar('Q38X_LOG_SEGMENT_MB', 1024);
assert std.isNumber(q38xSegMb) && q38xSegMb == std.floor(q38xSegMb) && q38xSegMb >= 64 && q38xSegMb <= 8192
       : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_LOG_SEGMENT_MB must be an integer from 64 to 8192';
local q38xTier3Cloud = std.objectHas(serverEnv(prefill), 'TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT');
assert q38xTier3Cloud == std.objectHas(serverEnv(decode), 'TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT')
       : 'exos-infx-qwen38-flash-agentx-disagg v1.2 P2: the Cloud legs disagree on the tier-3 root, and the legs share one Exos script';
assert std.isNumber(q38xMemEvery) && q38xMemEvery == std.floor(q38xMemEvery) && q38xMemEvery >= 10 && q38xMemEvery <= 3600
       : 'exos-infx-qwen38-flash-agentx-disagg: EXOS_MEMSTAT_EVERY_S must be an integer from 10 to 3600';
assert q38xExpectSrc == '' || (std.length(q38xExpectSrc) == 10 && std.all([std.member('0123456789abcdef', ch) for ch in std.stringChars(q38xExpectSrc)]))
       : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_EXPECT_SRC_HASH must be 10 lowercase hex characters';
local q38xSyncRefusals =
  (if cdk.rawVar('Q38X_WORKSPACE_SYNC', '') != '' then ['the recipe variable'] else [])
  + (if std.objectHas(serverEnv(prefill), 'Q38X_WORKSPACE_SYNC') then ['the prefill server env'] else [])
  + (if std.objectHas(serverEnv(decode), 'Q38X_WORKSPACE_SYNC') then ['the decode server env'] else []);
assert std.length(q38xSyncRefusals) == 0
       : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_WORKSPACE_SYNC is refused (%s): it is the opt-in for exos_vllm f1e83d94\'s /workspace source overlay, and this recipe builds on 25a37532, which has none' % std.join(', ', q38xSyncRefusals);
// The seven mapped-tree files exos_vllm 25a37532's bootstrap may rewrite (pyPatchBody: model_loader_patches, kv_block_override,
// connector_metadata, tpu_connector, tpu_runner, the two ray executors), as paths relative to the tree root.
local q38xPypatchAllowed = std.join(' ', ['src/vllm_torchtpu/' + p for p in [
  'model_loader_patches.py',
  'executors/kv_block_override.py',
  'distributed/kv_transfer/connector_metadata.py',
  'distributed/kv_transfer/tpu_connector.py',
  'runner/tpu_runner.py',
  'executors/ray_distributed_executor.py',
  'executors/ray_distributed_executor_v2.py',
]]);
local q38xFill(s) = std.strReplace(std.strReplace(std.strReplace(std.strReplace(std.strReplace(s,
  '@MEMEVERY@', std.toString(q38xMemEvery)), '@EXPECTSRC@', q38xExpectSrc), '@PYPATCH_ALLOWED@', q38xPypatchAllowed),
  '@WSCLEAN@', if q38xWsClean then 'true' else 'false'), '@SEGBYTES@', std.toString(q38xSegMb * 1048576));
local q38xInit = q38xFill(|||
  # ---- q38x v1.2 init BEGIN (EXOS-SERVED; the exos-infx-qwen38-flash-agentx-disagg recipe tail): no export / unset; see the tail
  q38x_say() { echo "$*"; echo "$*" >> /tmp/q38x.log 2>/dev/null || true; }
  q38x_sync() { mkdir -p "${CDK_OUTPUT_DIR:-/tmp/cdk_output}/q38x" 2>/dev/null; cp -f /tmp/q38x.log "${CDK_OUTPUT_DIR:-/tmp/cdk_output}/q38x/q38x.log" 2>/dev/null || true; }
  q38x_fatal() { q38x_say "FATAL: $*"; q38x_sync; exit 1; }
  q38x_mem() { awk '/^(MemTotal|MemAvailable|MemFree|AnonPages|Shmem|Cached):/{printf "%s=%dGiB ", $1, $2/1048576}' /proc/meminfo 2>/dev/null; df -BG --output=target,used,avail /dev/shm /tmp /tmp/workspace 2>/dev/null | awk 'NR>1{printf "%s:used=%s,avail=%s ", $1, $2, $3}'; }
  q38x_srclist() { (cd /root/cloud-devkit/vllm-torchtpu 2>/dev/null && find src -type f -name '*.py' -print0 | LC_ALL=C sort -z | xargs -0 -r sha1sum) 2>/dev/null; }
  q38x_clean() { for _q38x_p in /tmp/workspace/*; do [ -e "${_q38x_p}" ] || [ -L "${_q38x_p}" ] || continue; _q38x_b="${_q38x_p##*/}"; case "${_q38x_b}" in bin|etc|images|models|"sync_${JS_NAME:-run}") continue ;; esac; _q38x_s="$(du -sh "${_q38x_p}" 2>/dev/null | cut -f1)"; case "${_q38x_b}" in tier3_cache|inductor_cache|jax_cache|vllm_cache|torch_tpu_cache|vllm_pd_cache) if [ -d "${_q38x_p}" ] && [ ! -L "${_q38x_p}" ] && [ -n "$(ls -A "${_q38x_p}" 2>/dev/null)" ]; then find "${_q38x_p}" -mindepth 1 -delete 2>/dev/null || true; q38x_say "Q38X_MEMSTAT cleaned ${_q38x_s:-?} ${_q38x_p} (emptied: runner leftover)"; fi ;; *) rm -rf "${_q38x_p}" 2>/dev/null || true; q38x_say "Q38X_MEMSTAT cleaned ${_q38x_s:-?} ${_q38x_p} (removed: not this job's)" ;; esac; done; }
  cat > /tmp/q38x_logcap.py <<'Q38X_LOGCAP_PY'
  import os, subprocess, sys
  path, outdir, leg, cap = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
  note = '/tmp/q38x.log'


  def put(fd, b):
      while b:
          try:
              n = os.write(fd, b)
          except InterruptedError:
              continue
          except OSError:
              return False
          b = b[n:]
      return True


  def say(msg):
      put(1, (msg + '\n').encode())
      try:
          with open(note, 'a') as fh:
              fh.write(msg + '\n')
      except OSError:
          pass


  try:
      os.makedirs(outdir, exist_ok=True)
  except OSError:
      pass
  f = open(path, 'ab', buffering=0)
  size, seg, out_ok = f.tell(), 0, True
  say('Q38X_LOGCAP leg=%s %s rotated at %d bytes; segments -> %s/vllm_serve.%s.NNN.log' % (leg, path, cap, outdir, leg))
  while True:
      try:
          b = os.read(0, 1 << 16)
      except InterruptedError:
          continue
      if not b:
          break
      if out_ok:
          out_ok = put(1, b)
      put(f.fileno(), b)
      size += len(b)
      if size >= cap and (b.endswith(b'\n') or size >= 2 * cap):
          seg += 1
          part = '%s.q38x%03d' % (path, seg)
          dst = os.path.join(outdir, 'vllm_serve.%s.%03d.log' % (leg, seg))
          try:
              f.close()
              os.rename(path, part)
              subprocess.Popen(['sh', '-c', 'cp -f "$1" "$2" && rm -f "$1"', 'q38x-logcap', part, dst],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
              say('Q38X_LOGCAP leg=%s segment %03d (%d bytes) -> %s' % (leg, seg, size, dst))
          except OSError as e:
              say('Q38X_LOGCAP leg=%s rotation %03d failed: %s' % (leg, seg, e))
          f = open(path, 'ab', buffering=0)
          size = 0
  f.close()
  Q38X_LOGCAP_PY
  q38x_logcap() { python3 /tmp/q38x_logcap.py /tmp/vllm_serve.log "${CDK_OUTPUT_DIR:-/tmp/cdk_output}/q38x" "${PD_LEG:-leg}${PD_DECODE_INDEX:+${PD_DECODE_INDEX}}" @SEGBYTES@ || tee -a /tmp/vllm_serve.log; }
  q38x_guard() {
    _q38x_imp="$(cd /tmp && VLLM_PLUGINS="" VLLM_TARGET_DEVICE="empty" JAX_PLATFORMS="cpu" python3 -c 'import importlib.util as u; s = u.find_spec("vllm_torchtpu"); print(list(s.submodule_search_locations)[0] if s else "")' 2>/dev/null || true)"
    case "${_q38x_imp}" in
      /root/cloud-devkit/vllm-torchtpu/src/*) ;;
      *) q38x_fatal "Q38X_GUARD leg=$1 vllm_torchtpu imports from '${_q38x_imp}', not the mapped tree /root/cloud-devkit/vllm-torchtpu/src" ;;
    esac
    _q38x_now="$(q38x_srclist | sha1sum | cut -c1-10)"
    [ "${_q38x_now}" = "${_q38x_post:-}" ] || q38x_fatal "Q38X_GUARD leg=$1 the mapped tree changed after the bootstrap (src_hash ${_q38x_now}, post-bootstrap ${_q38x_post:-unset})"
    q38x_say "Q38X_GUARD leg=$1 vllm_torchtpu=${_q38x_imp} pre=${_q38x_pre:-} post=${_q38x_now} pin=@EXPECTSRC@: vllm serve next $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    q38x_sync
  }
  q38x_init() {
    : > /tmp/q38x.log
    q38x_say "Q38X_V12 exos-infx-qwen38-flash-agentx-disagg v1.2 leg=${PD_LEG:-?}${PD_DECODE_INDEX:+-${PD_DECODE_INDEX}} EXOS_MEMSTAT_EVERY_S=@MEMEVERY@ Q38X_EXPECT_SRC_HASH=@EXPECTSRC@ EXOS_WORKSPACE_CLEAN=@WSCLEAN@ Q38X_LOG_SEGMENT_BYTES=@SEGBYTES@ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    q38x_say "Q38X_MEMSTAT start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(q38x_mem)nproc=$(nproc)"
    q38x_say "Q38X_MEMSTAT outdir $(df -BG --output=fstype,used,avail,target "${CDK_OUTPUT_DIR:-/tmp}" 2>/dev/null | tail -n 1)"
    du -sh /tmp/workspace/* 2>/dev/null | while IFS= read -r _q38x_ln; do q38x_say "Q38X_MEMSTAT workspace-at-start ${_q38x_ln}"; done
    if [ "@WSCLEAN@" = true ]; then
      q38x_clean
      q38x_say "Q38X_MEMSTAT after-clean $(date -u +%Y-%m-%dT%H:%M:%SZ) $(q38x_mem)"
    fi
    ( _q38x_min=999999; while sleep @MEMEVERY@; do _q38x_a="$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo 2>/dev/null)"; [ -n "${_q38x_a}" ] && [ "${_q38x_a}" -lt "${_q38x_min}" ] && _q38x_min="${_q38x_a}"; q38x_say "Q38X_MEMSTAT $(date -u +%Y-%m-%dT%H:%M:%SZ) $(q38x_mem)min_avail=${_q38x_min}GiB"; q38x_sync; done ) &
    q38x_srclist > /tmp/q38x_tree_pre.sha1
    [ -s /tmp/q38x_tree_pre.sha1 ] || q38x_fatal "Q38X_TREE no src/*.py under /root/cloud-devkit/vllm-torchtpu (no --map-dir tree for this leg?)"
    _q38x_pre="$(sha1sum < /tmp/q38x_tree_pre.sha1 | cut -c1-10)"
    q38x_say "Q38X_TREE pre-bootstrap src_hash=${_q38x_pre} py_files=$(wc -l < /tmp/q38x_tree_pre.sha1) files=$(cd /root/cloud-devkit/vllm-torchtpu && find . -type f | wc -l) expect=@EXPECTSRC@"
    if [ -n "@EXPECTSRC@" ] && [ "${_q38x_pre}" != "@EXPECTSRC@" ]; then
      q38x_fatal "Q38X_GUARD the mapped tree's src hash is '${_q38x_pre}', not the car's pin @EXPECTSRC@"
    fi
    q38x_sync
  }
  q38x_init
  # ---- q38x v1.2 init END
|||);
local q38xPost = q38xFill(|||
  # ---- q38x v1.2 post BEGIN: what exos_vllm's bootstrap just rewrote under the mapped tree
  q38x_post() {
    q38x_srclist > /tmp/q38x_tree_post.sha1
    _q38x_post="$(sha1sum < /tmp/q38x_tree_post.sha1 | cut -c1-10)"
    _q38x_chg="$(awk 'NR==FNR{a[$2]=$1; next} {if (!($2 in a)) print "+" $2; else if (a[$2] != $1) print $2; delete a[$2]} END{for (k in a) print "-" k}' /tmp/q38x_tree_pre.sha1 /tmp/q38x_tree_post.sha1 | LC_ALL=C sort | tr '\n' ' ')"
    _q38x_bad=""
    for _q38x_f in ${_q38x_chg}; do
      case " @PYPATCH_ALLOWED@ " in *" ${_q38x_f} "*) ;; *) _q38x_bad="${_q38x_bad} ${_q38x_f}" ;; esac
    done
    q38x_say "Q38X_PYPATCH exos_vllm 25a37532 bootstrap rewrote $(echo ${_q38x_chg} | wc -w) mapped-tree files: ${_q38x_chg:-none}; post-bootstrap src_hash=${_q38x_post}"
    [ -z "${_q38x_bad}" ] || q38x_fatal "Q38X_PYPATCH the bootstrap changed mapped-tree files outside the known 25a37532 set:${_q38x_bad}"
    q38x_sync
  }
  q38x_post
  # ---- q38x v1.2 post END
|||);
// The four anchors in each leg's vllm-pd script (exos_vllm 25a37532 pdRecipe; one each, counted: a moved anchor fails the render).
local q38xAnchorPy = 'python3 -c \'import os; exec(compile(os.environ["_EXOS_PYPATCH"], "<exos_pypatch>", "exec"), {"__name__": "__main__"})\' || true\n';
local q38xAnchorUnset = 'unset _EXOS_PYPATCH\n';
local q38xAnchorDecode = '\n  echo "trace-start-event: name=vllm-decode ts=';
local q38xAnchorPrefill = '\necho "trace-start-event: name=vllm-prefill ts=';
// P2 / P3: counted edits of exos_vllm's leg script, applied BEFORE the blocks go in (the init block's fallback carries the old tee text).
local q38xP2Old = 'export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/tmp/workspace/tier3_cache"\nexport TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="/tmp/workspace/tier3_cache"\nexport TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK=0\n';
local q38xP2New = '# exos-served v1.2 P2: exos_vllm\'s three tier-3 compile-cache exports removed here (the Cloud legs set no tier-3 root)\n';
local q38xP2MkdirOld = '"${TORCH_TPU_CACHE_DIR}" "${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}"';
local q38xP2MkdirNew = '"${TORCH_TPU_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"';
local q38xP3Old = '> >(tee -a /tmp/vllm_serve.log) 2>&1 &';
local q38xP3New = '> >(q38x_logcap) 2>&1 &';
local q38xEdits(s) =
  local s3 = exosReplaceN(s, q38xP3Old, q38xP3New, 2);
  if q38xTier3Cloud then s3 else exosReplaceN(exosReplaceN(s3, q38xP2Old, q38xP2New, 1), q38xP2MkdirOld, q38xP2MkdirNew, 1);
local q38xLegScript(s0) =
  local s = q38xEdits(s0);
  exosReplaceN(exosReplaceN(exosReplaceN(exosReplaceN(s,
    q38xAnchorPy, q38xInit + q38xAnchorPy, 1),
    q38xAnchorUnset, q38xAnchorUnset + q38xPost, 1),
    q38xAnchorDecode, '\n  q38x_guard decode || exit 1' + q38xAnchorDecode, 1),
    q38xAnchorPrefill, '\nq38x_guard prefill || exit 1' + q38xAnchorPrefill, 1);
local q38xJob(job) =
  local cs = job.template.spec.template.spec.containers;
  assert std.length([c for c in cs if c.name == 'vllm-pd']) == 1 : 'exos v1.2: job %s has no single vllm-pd container' % job.name;
  job + { template+: { spec+: { template+: { spec+: {
    containers: [if c.name == 'vllm-pd' then c + { args: [q38xLegScript(c.args[0])] } else c for c in cs],
  } } } } };
local exosJobSetV12 = exosJobSet + { spec+: { replicatedJobs: [q38xJob(j) for j in super.replicatedJobs] } };

// --- v1.3 (EXOS-SERVED-2, 2026-11-17): per-leg MemAvailable self-halt (Q38X_MEMHALT) + per-line live q38x lines (Q38X_OUT) ---
// v1.3 = v1.2 (da5ad007) + five delimited blocks in each leg's vllm-pd script, inserted at counted anchors of v1.2's leg script; no v1.2
// text is edited, so stripping every '# ---- q38x v1.3 <name> BEGIN' .. 'END' block gives v1.2's render byte for byte (served_parity_compare_v4.py
// checks that, then runs v3's whole Cloud-vs-Exos comparison on the stripped render). The blocks export / unset nothing the engine reads.
//   memhalt  (before v1.2's `q38x_init` call) Q38X_OUT: q38x_say also appends each line to outputs/.../q38x/q38x.log (open-append-close,
//            bench v3's live mode); the halt functions and the watcher (Q38X_MEMHALT; see the block's own comment).
//   arm      (after `q38x_init`) starts the watcher: MemAvailable every 5 s against this leg's floor.
//   gate     (before each leg's `q38x_guard <role>`) no vllm serve once this leg has halted.
//   pid      (after each leg's `vllm serve ... &`, i.e. after `DECODE_PID=$!` / `export SERVER_PID=$!`) pins that pid + its /proc start
//            time for the watcher.
// On a halt the leg's own exos_vllm exit path runs (it watches that pid): prefill -> SYNC_DIR/failed, prefill.log copy, exit 1; decode ->
// decode.log copy, exit 1. The other legs are not signalled (a decode leg leaves by itself when the prefill port closes; the prefill pod's
// sidecar runs its sweep out against the dead leg, every level failing).
// Readers: outputs/<job>-<leg>-0/<pod>/vllm-pd/q38x/{q38x.log, MEMHALT.<leg>, vllm_serve.<leg>.halt.log, vllm_serve.<leg>.NNN.log}.
// Knobs (read only by this tail; none reaches the engine env or argv):
//   EXOS_MEM_FLOOR_GIB=<n>          default 64 (integer 16..900): the MemAvailable floor of every leg.
//   PREFILL_EXOS_MEM_FLOOR_GIB=<n>  the prefill leg's floor (default EXOS_MEM_FLOOR_GIB); DECODE_EXOS_MEM_FLOOR_GIB=<n> every decode leg's.
//   EXOS_MEMHALT_GRACE_S=<n>        default 20 (integer 5..120): SIGTERM -> SIGKILL of the vllm serve tree (SIGKILL at once under half the floor).
local q38x13FloorAll = cdk.numVar('EXOS_MEM_FLOOR_GIB', 64);
local q38x13FloorP = cdk.numVar('PREFILL_EXOS_MEM_FLOOR_GIB', q38x13FloorAll);
local q38x13FloorD = cdk.numVar('DECODE_EXOS_MEM_FLOOR_GIB', q38x13FloorAll);
local q38x13Grace = cdk.numVar('EXOS_MEMHALT_GRACE_S', 20);
local q38x13IntIn(x, lo, hi) = std.isNumber(x) && x == std.floor(x) && x >= lo && x <= hi;
assert q38x13IntIn(q38x13FloorAll, 16, 900) : 'exos-infx-qwen38-flash-agentx-disagg: EXOS_MEM_FLOOR_GIB must be an integer from 16 to 900 (GiB of MemAvailable)';
assert q38x13IntIn(q38x13FloorP, 16, 900) : 'exos-infx-qwen38-flash-agentx-disagg: PREFILL_EXOS_MEM_FLOOR_GIB must be an integer from 16 to 900 (GiB of MemAvailable)';
assert q38x13IntIn(q38x13FloorD, 16, 900) : 'exos-infx-qwen38-flash-agentx-disagg: DECODE_EXOS_MEM_FLOOR_GIB must be an integer from 16 to 900 (GiB of MemAvailable)';
assert q38x13IntIn(q38x13Grace, 5, 120) : 'exos-infx-qwen38-flash-agentx-disagg: EXOS_MEMHALT_GRACE_S must be an integer from 5 to 120 (seconds)';
local q38x13Defs = std.strReplace(std.strReplace(std.strReplace(|||
  # ---- q38x v1.3 memhalt BEGIN (EXOS-SERVED-2; the exos-infx-qwen38-flash-agentx-disagg recipe tail): no export / unset; see the tail
  # Q38X_OUT (v1.3): every q38x_say line is also appended to outputs/.../q38x/q38x.log, one open-append-close per line (bench v3's write
  # mode, live within ~1 s on T93j); v1.2's q38x_sync still rewrites that file from the complete local /tmp/q38x.log at every MEMSTAT tick.
  _q38x_outd="${CDK_OUTPUT_DIR:-/tmp/cdk_output}/q38x"
  mkdir -p "${_q38x_outd}" 2>/dev/null || true
  q38x_say() { echo "$*"; echo "$*" >> /tmp/q38x.log 2>/dev/null || true; echo "$*" >> "${_q38x_outd}/q38x.log" 2>/dev/null || true; }
  # Q38X_MEMHALT (v1.3, methodology rule 17 'Exos RAM wall'): a linked runner that runs out of host RAM is lost without an OOM line and
  # skips the exit copy (MM X48, M). A watcher reads MemAvailable every 5 s; below this leg's floor it sets /tmp/q38x_memhalt and stops
  # THIS leg's vllm serve: SIGTERM to that one process, then SIGKILL to every process of its tree still alive after the grace (at once if
  # MemAvailable falls under half the floor). The leg script then leaves through exos_vllm's own exit path (it watches that pid), which
  # copies /tmp/vllm_serve.log to outputs; the watcher also copies the live segment itself (q38x/vllm_serve.<leg>.halt.log) and writes
  # q38x/MEMHALT.<leg>. The flag is never cleared: a vllm serve found while it is set is stopped too, and the gate refuses to start one.
  # The target is the pid the leg recorded from `vllm serve ... &` ($!), pinned by its /proc start time and checked against the exact
  # argv (argv[0] vllm, or a python interpreter then vllm, then 'serve'); with no live pin, an exact-argv scan. Never `pgrep -f`: the leg
  # shell's own command line (/bin/bash -c <this script>) carries the text 'vllm serve'. Processes are signalled one by one, never by
  # process group (the leg's background jobs share the leg shell's group); pid 1, the leg shell and the watcher's own lineage are skipped.
  _q38x_leg="${PD_LEG:-leg}${PD_DECODE_INDEX:+${PD_DECODE_INDEX}}"
  case "${PD_LEG:-}" in prefill) _q38x_floor=@FLOOR_P@ ;; *) _q38x_floor=@FLOOR_D@ ;; esac
  _q38x_hard=$(( _q38x_floor / 2 ))
  _q38x_avail() { awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo 2>/dev/null; }
  _q38x_stat() {  # $1 pid -> _q38x_S state, _q38x_PP ppid, _q38x_T start time (fields 3, 4, 22 of /proc/<pid>/stat); 1 if unreadable
    local _s=""; _q38x_S=""; _q38x_PP=""; _q38x_T=""
    IFS= read -r _s 2>/dev/null < "/proc/$1/stat" || return 1
    set -- ${_s##*) }
    _q38x_S="${1:-}"; _q38x_PP="${2:-}"; _q38x_T="${20:-}"
    [ -n "${_q38x_T}" ]; }
  _q38x_argv3() {  # $1 pid -> _q38x_A0 _q38x_A1 _q38x_A2: its first three argv entries, exact (NUL-separated /proc/<pid>/cmdline)
    _q38x_A0=""; _q38x_A1=""; _q38x_A2=""
    { IFS= read -r -d '' _q38x_A0; IFS= read -r -d '' _q38x_A1; IFS= read -r -d '' _q38x_A2; } 2>/dev/null < "/proc/$1/cmdline"
    return 0; }
  _q38x_is_serve() {  # $1 pid: 0 if its argv is exactly vllm|.../vllm serve ..., or python*|.../python* vllm|.../vllm serve ...
    _q38x_argv3 "$1"
    case "${_q38x_A0##*/}" in
      vllm) [ "${_q38x_A1}" = serve ] ;;
      python*) [ "${_q38x_A1##*/}" = vllm ] && [ "${_q38x_A2}" = serve ] ;;
      *) return 1 ;;
    esac; }
  _q38x_is_shell() { _q38x_argv3 "$1"; case "${_q38x_A0##*/}" in bash|sh|dash|-bash|-sh) return 0 ;; *) return 1 ;; esac; }
  _q38x_alive() { _q38x_stat "${1%%:*}" && [ "${_q38x_S}" != Z ] && [ "${_q38x_T}" = "${1#*:}" ]; }  # $1 pid:start
  _q38x_targets() {  # -> _q38x_TG: 'pid:start ...' of this leg's vllm serve, _q38x_WHY: how it was identified; 1 if none is running
    local _d="" _p="" _pp="" _pt="" _scan=""
    _q38x_TG=""; _q38x_WHY=""
    [ -r /tmp/q38x_serve.pid ] && read -r _pp _pt < /tmp/q38x_serve.pid 2>/dev/null
    if [ -n "${_pp}" ] && [ "${_pp}" -gt 1 ] 2>/dev/null && _q38x_alive "${_pp}:${_pt}"; then
      if _q38x_is_serve "${_pp}"; then
        _q38x_TG="${_pp}:${_pt}"; _q38x_WHY="the pinned pid ${_pp} (start ${_pt}), exact argv '${_q38x_A0} ${_q38x_A1}${_q38x_A2:+ ${_q38x_A2}}'"; return 0
      fi
      if ! _q38x_is_shell "${_pp}"; then
        _q38x_TG="${_pp}:${_pt}"; _q38x_WHY="the pinned pid ${_pp} (start ${_pt}); its argv[0] '${_q38x_A0}' is not 'vllm serve' (process title rewritten)"; return 0
      fi
    fi
    for _d in /proc/[0-9]*; do
      _p="${_d#/proc/}"
      [ "${_p}" -gt 1 ] 2>/dev/null && [ "${_p}" != "$$" ] && [ "${_p}" != "${BASHPID}" ] || continue
      _q38x_is_serve "${_p}" && _q38x_stat "${_p}" && [ "${_q38x_S}" != Z ] && _scan="${_scan} ${_p}:${_q38x_T}:${_q38x_PP}"
    done
    for _d in ${_scan}; do  # top-level matches only: a forked child keeps its parent's argv until it renames itself
      case "${_scan} " in *" ${_d##*:}:"*) continue ;; esac
      _q38x_TG="${_q38x_TG} ${_d%:*}"
    done
    _q38x_TG="${_q38x_TG# }"
    [ -n "${_q38x_TG}" ] || return 1
    _q38x_WHY="the exact argv 'vllm serve' (no live pinned pid)"; return 0; }
  _q38x_tree() {  # $@ 'pid:start' roots -> stdout: 'pid:start' of each live root and every live descendant (by ppid; one /proc pass),
    # except the subtree of a shell that is a root's direct child: the q38x_logcap process substitution of `vllm serve ... > >(q38x_logcap)`
    # is forked by the job that then execs vllm, and it must outlive the engine to flush the log and finish its segment copies.
    local _d="" _p="" _r=""
    for _d in "$@"; do _r="${_r} ${_d%%:*}"; done
    for _d in /proc/[0-9]*; do
      _p="${_d#/proc/}"; _q38x_stat "${_p}" && [ "${_q38x_S}" != Z ] || continue
      _q38x_argv3 "${_p}"; echo "${_p} ${_q38x_PP} ${_q38x_T} ${_q38x_A0##*/}"
    done | awk -v r="${_r}" 'BEGIN { n = split(r, a, " "); for (i = 1; i <= n; i++) { s[a[i]] = 1; root[a[i]] = 1 } }
        { p[NR] = $1; q[NR] = $2; t[NR] = $3; sh[NR] = ($4 ~ /^-?(ba|da)?sh$/) }
        END { c = 1; while (c) { c = 0; for (i = 1; i <= NR; i++) if (!(p[i] in s) && (q[i] in s) && !(sh[i] && (q[i] in root))) { s[p[i]] = 1; c = 1 } }
              for (i = 1; i <= NR; i++) if (p[i] in s) print p[i] ":" t[i] }'; }
  _q38x_sig() {  # $1 signal, $2... 'pid:start' -> stdout: how many were signalled (only processes still at that start time, never our lineage)
    local _s="$1" _e="" _n=0; shift
    for _e in "$@"; do
      _q38x_alive "${_e}" || continue
      case " ${_q38x_mine:-1} " in *" ${_e%%:*} "*) continue ;; esac
      kill "-${_s}" "${_e%%:*}" 2>/dev/null && _n=$((_n + 1))
    done
    echo "${_n}"; }
  _q38x_left() { local _e="" _l=""; for _e in "$@"; do _q38x_alive "${_e}" && _l="${_l} ${_e}"; done; echo "${_l# }"; }
  _q38x_halt_stop() {  # $1 why: stop this leg's vllm serve, then copy the live log segment and the q38x lines; 1 if none is running
    local _tree="" _tree2="" _left="" _a="" _i=0 _n="" _k=0 _lc="" _d=""
    _q38x_targets || return 1
    _tree="$(_q38x_tree ${_q38x_TG} | tr '\n' ' ')"
    q38x_say "Q38X_MEMHALT $(date -u +%Y-%m-%dT%H:%M:%SZ) leg=${_q38x_leg} SIGTERM vllm serve ${_q38x_TG} ($1; identified by ${_q38x_WHY}); tree $(echo ${_tree} | wc -w) processes $(q38x_mem)"
    q38x_sync
    _q38x_sig TERM ${_q38x_TG} > /dev/null
    while [ "${_i}" -lt @GRACE@ ]; do
      sleep 1; _i=$((_i + 1))
      _left="$(_q38x_left ${_tree})"; [ -n "${_left}" ] || break
      _a="$(_q38x_avail)"
      if [ -n "${_a}" ] && [ "${_a}" -lt "${_q38x_hard}" ]; then
        q38x_say "Q38X_MEMHALT $(date -u +%Y-%m-%dT%H:%M:%SZ) leg=${_q38x_leg} MemAvailable=${_a}GiB < ${_q38x_hard} GiB (half the floor) ${_i} s after SIGTERM: SIGKILL now"; break
      fi
    done
    _left="$(_q38x_left ${_tree})"
    if [ -n "${_left}" ]; then
      _tree2="$(_q38x_tree ${_left} | tr '\n' ' ')"
      _k="$(_q38x_sig KILL $(echo ${_left} ${_tree2} | tr ' ' '\n' | sort -u))"
      q38x_say "Q38X_MEMHALT $(date -u +%Y-%m-%dT%H:%M:%SZ) leg=${_q38x_leg} SIGKILL ${_k} process(es) of the vllm serve tree still alive ${_i} s after SIGTERM"
      _i=0; while [ "${_i}" -lt 10 ] && [ -n "$(_q38x_left ${_left} ${_tree2})" ]; do sleep 1; _i=$((_i + 1)); done
    fi
    _left="$(_q38x_left ${_tree} ${_tree2})"
    q38x_say "Q38X_MEMHALT $(date -u +%Y-%m-%dT%H:%M:%SZ) leg=${_q38x_leg} vllm serve tree $(if [ -n "${_left}" ]; then echo "NOT fully stopped: ${_left}"; else echo stopped; fi) $(q38x_mem)"
    # the q38x_logcap writer ends at EOF once no process of the tree holds its pipe; give it up to 10 s, then copy the live segment
    _i=0
    while [ "${_i}" -lt 10 ]; do
      _lc=""; for _d in /proc/[0-9]*; do _q38x_argv3 "${_d#/proc/}"; case "${_q38x_A0##*/}" in python*) [ "${_q38x_A1}" = /tmp/q38x_logcap.py ] && _lc=1 ;; esac; done
      [ -z "${_lc}" ] && break; sleep 1; _i=$((_i + 1))
    done
    _n="$(wc -c < /tmp/vllm_serve.log 2>/dev/null || echo 0)"
    if cp -f /tmp/vllm_serve.log "${_q38x_outd}/vllm_serve.${_q38x_leg}.halt.log" 2>/dev/null; then
      q38x_say "Q38X_MEMHALT $(date -u +%Y-%m-%dT%H:%M:%SZ) leg=${_q38x_leg} live serving-log segment (${_n} bytes) -> ${_q38x_outd}/vllm_serve.${_q38x_leg}.halt.log"
    else
      q38x_say "Q38X_MEMHALT $(date -u +%Y-%m-%dT%H:%M:%SZ) leg=${_q38x_leg} copy of /tmp/vllm_serve.log (${_n} bytes) to outputs FAILED"
    fi
    { echo "Q38X_MEMHALT leg=${_q38x_leg} floor=${_q38x_floor}GiB why=$1 at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "target=${_q38x_TG} identified_by=${_q38x_WHY}"
      echo "not_stopped=${_left:-none}"; } > "${_q38x_outd}/MEMHALT.${_q38x_leg}" 2>/dev/null || true
    q38x_sync
    return 0; }
  q38x_memhalt_gate() {  # $1 leg role: 1 (and a line) if this leg already halted, so vllm serve is not started
    [ -e /tmp/q38x_memhalt ] || return 0
    q38x_say "Q38X_MEMHALT $(date -u +%Y-%m-%dT%H:%M:%SZ) leg=$1: the halt flag is set before vllm serve: not starting it"
    q38x_sync; return 1; }
  q38x_memhalt_pid() {  # $1 leg role, $2 the $! of `vllm serve ... &`: pin pid + /proc start time for the watcher
    if _q38x_stat "$2"; then
      echo "$2 ${_q38x_T}" > /tmp/q38x_serve.pid.tmp 2>/dev/null && mv -f /tmp/q38x_serve.pid.tmp /tmp/q38x_serve.pid 2>/dev/null
      q38x_say "Q38X_MEMHALT leg=$1 watching vllm serve pid $2 (start ${_q38x_T}); floor ${_q38x_floor} GiB $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    else
      q38x_say "Q38X_MEMHALT leg=$1 vllm serve pid $2 exited before it was pinned; the watcher uses the exact-argv scan $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    fi
    q38x_sync; }
  q38x_memhalt_arm() {  # start the watcher (after q38x_init, so the banner lines come first)
    rm -f /tmp/q38x_memhalt /tmp/q38x_serve.pid /tmp/q38x_serve.pid.tmp
    q38x_say "Q38X_MEMFLOOR armed: leg=${_q38x_leg} floor ${_q38x_floor} GiB (SIGKILL at once under ${_q38x_hard} GiB), SIGTERM grace @GRACE@ s, MemAvailable every 5 s (now $(_q38x_avail) GiB) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    q38x_sync
    ( _q38x_mine="1 $$ ${BASHPID}"; _q38x_p="${BASHPID}"
      while _q38x_stat "${_q38x_p}" && [ "${_q38x_PP}" -gt 1 ] 2>/dev/null; do _q38x_p="${_q38x_PP}"; _q38x_mine="${_q38x_mine} ${_q38x_p}"; done
      while sleep 5; do
        if [ -e /tmp/q38x_memhalt ]; then _q38x_halt_stop "the halt flag is set: a vllm serve is running after Q38X_MEMHALT" || true; continue; fi
        _a="$(_q38x_avail)"
        [ -n "${_a}" ] && [ "${_a}" -lt "${_q38x_floor}" ] || continue
        : > /tmp/q38x_memhalt
        q38x_say "Q38X_MEMHALT $(date -u +%Y-%m-%dT%H:%M:%SZ) leg=${_q38x_leg} MemAvailable=${_a}GiB is below the floor ${_q38x_floor} GiB: halt (flag /tmp/q38x_memhalt set; this leg's vllm serve is stopped and the leg leaves through its own exit path) $(q38x_mem)"
        q38x_sync
        _q38x_halt_stop "MemAvailable=${_a}GiB < ${_q38x_floor} GiB" \
          || { q38x_say "Q38X_MEMHALT $(date -u +%Y-%m-%dT%H:%M:%SZ) leg=${_q38x_leg} no vllm serve is running now; the flag keeps one from starting"; q38x_sync; }
      done ) &
  }
  # ---- q38x v1.3 memhalt END
|||, '@FLOOR_P@', std.toString(q38x13FloorP)), '@FLOOR_D@', std.toString(q38x13FloorD)), '@GRACE@', std.toString(q38x13Grace));
local q38x13Arm = '# ---- q38x v1.3 arm BEGIN\nq38x_memhalt_arm\n# ---- q38x v1.3 arm END\n';
local q38x13Gate(ind, role) = '%s# ---- q38x v1.3 gate BEGIN\n%sq38x_memhalt_gate %s || exit 1\n%s# ---- q38x v1.3 gate END\n' % [ind, ind, role, ind];
local q38x13Pid(ind, role, var) = '%s# ---- q38x v1.3 pid BEGIN\n%sq38x_memhalt_pid %s "${%s}"\n%s# ---- q38x v1.3 pid END\n' % [ind, ind, role, var, ind];
// The five anchors in v1.2's leg script (one each, counted: a moved anchor fails the render).
local q38x13AnchorInit = 'q38x_init\n# ---- q38x v1.2 init END\n';
local q38x13AnchorGateD = '\n  q38x_guard decode || exit 1\n';
local q38x13AnchorGateP = '\nq38x_guard prefill || exit 1\n';
local q38x13AnchorPidD = '  DECODE_PID=$!\n';
local q38x13AnchorPidP = 'export SERVER_PID=$!\n';
local q38x13LegScript(s) =
  exosReplaceN(exosReplaceN(exosReplaceN(exosReplaceN(exosReplaceN(s,
    q38x13AnchorInit, q38x13Defs + 'q38x_init\n' + q38x13Arm + '# ---- q38x v1.2 init END\n', 1),
    q38x13AnchorGateD, '\n' + q38x13Gate('  ', 'decode') + '  q38x_guard decode || exit 1\n', 1),
    q38x13AnchorGateP, '\n' + q38x13Gate('', 'prefill') + 'q38x_guard prefill || exit 1\n', 1),
    q38x13AnchorPidD, q38x13AnchorPidD + q38x13Pid('  ', 'decode', 'DECODE_PID'), 1),
    q38x13AnchorPidP, q38x13AnchorPidP + q38x13Pid('', 'prefill', 'SERVER_PID'), 1);
local q38x13Job(job) =
  local cs = job.template.spec.template.spec.containers;
  assert std.length([c for c in cs if c.name == 'vllm-pd']) == 1 : 'exos v1.3: job %s has no single vllm-pd container' % job.name;
  job + { template+: { spec+: { template+: { spec+: {
    containers: [if c.name == 'vllm-pd' then c + { args: [q38x13LegScript(c.args[0])] } else c for c in cs],
  } } } } };
local exosJobSetV13 = exosJobSetV12 + { spec+: { replicatedJobs: [q38x13Job(j) for j in super.replicatedJobs] } };

// --- v1.4 (X1-WATCH for the lead, 2026-11-17): the prefill pod's sidecar-bench no longer runs exos_vllm's pypatch on the shared mapped
// tree; optional post-bootstrap src pin (Q38X_EXPECT_POST_HASH) ---
// Why (T93r q38-exos-x1 j-520fee59-38a0-4e27-a30d, runs.md X1-WATCH 2026-11-17T02:45Z): exos_vllm 25a37532 gives every main container the
// same preamble, so the prefill pod's sidecar-bench ran `python3 -c ... _EXOS_PYPATCH ...` about 1.5 s after its start on the mapped tree
// that it shares with the prefill leg (volume cdk-map-0 is mounted by both containers), before the leg's Q38X_TREE pre-bootstrap hash:
// the leg saw the post-patch hash 7f5f092244 and Q38X_GUARD failed closed against the pin f10b76478f. The sidecar image (inferencex-agentx)
// has no vllm / vllm_torchtpu install, so its patch served nothing; the leg patches the same tree itself (idempotent markers), exactly as
// the decode leg (no sidecar) did on T93r (pre f10b76478f -> 7 files -> post 7f5f092244).
// v1.4 = v1.3 (3b02ae29) + two delimited blocks; no v1.3 text is edited except the one sidecar line the first block replaces:
//   sidecar-nopatch  (the sidecar-bench script only) the exec line becomes a Q38X_SIDECAR_NOPATCH echo; 'unset _EXOS_PYPATCH' stays.
//   postpin          (each leg's vllm-pd script, only with Q38X_EXPECT_POST_HASH set) right after v1.2's `q38x_post` call: FATAL unless
//                    the post-bootstrap src hash equals the pin, else a Q38X_POSTPIN line.
// Restoring the exec line and removing the postpin blocks gives v1.3's render byte for byte (served_parity_compare_v5.py checks that, then
// runs v4's whole comparison on the restored render).
// Knob (read only by this tail; never reaches the engine env or argv):
//   Q38X_EXPECT_POST_HASH=<10 hex>  the tree's post-bootstrap src hash under exos_vllm 25a37532's pypatch (v8 tree f10b76478f -> 7f5f092244,
//                                   M: T93r decode leg). Needs Q38X_EXPECT_SRC_HASH. Empty (default): no postpin block.
local q38x14PostPin = cdk.var('Q38X_EXPECT_POST_HASH', '');
assert q38x14PostPin == '' || (std.length(q38x14PostPin) == 10 && std.all([std.member('0123456789abcdef', ch) for ch in std.stringChars(q38x14PostPin)]))
       : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_EXPECT_POST_HASH must be 10 lowercase hex characters';
assert q38x14PostPin == '' || q38xExpectSrc != ''
       : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_EXPECT_POST_HASH needs Q38X_EXPECT_SRC_HASH (the pre-bootstrap pin)';
local q38x14Exec = 'python3 -c \'import os; exec(compile(os.environ["_EXOS_PYPATCH"], "<exos_pypatch>", "exec"), {"__name__": "__main__"})\' || true\n';
local q38x14SideOld = q38x14Exec + 'unset _EXOS_PYPATCH\n';
local q38x14SideNew = '# ---- q38x v1.4 sidecar-nopatch BEGIN\n'
                      + 'echo "Q38X_SIDECAR_NOPATCH sidecar-bench: exos_vllm 25a37532 pypatch not run here; the prefill leg patches the shared mapped tree (v1.4)"\n'
                      + '# ---- q38x v1.4 sidecar-nopatch END\n'
                      + 'unset _EXOS_PYPATCH\n';
local q38x14PostAnchor = 'q38x_post\n# ---- q38x v1.2 post END\n';
local q38x14PostBlk = '# ---- q38x v1.4 postpin BEGIN\n'
                      + ('if [ "${_q38x_post:-}" != "%s" ]; then q38x_fatal "Q38X_GUARD leg=${PD_LEG:-?}${PD_DECODE_INDEX:+-${PD_DECODE_INDEX}} '
                         + "the post-bootstrap src hash is '${_q38x_post:-unset}', not the car's post pin %s\"; fi\n") % [q38x14PostPin, q38x14PostPin]
                      + 'q38x_say "Q38X_POSTPIN leg=${PD_LEG:-?}${PD_DECODE_INDEX:+-${PD_DECODE_INDEX}} post-bootstrap src_hash=${_q38x_post} = pin %s"\n' % q38x14PostPin
                      + '# ---- q38x v1.4 postpin END\n';
local q38x14Count(s, x) = (std.length(s) - std.length(std.strReplace(s, x, ''))) / std.length(x);
local q38x14Side(c) =
  local s = exosReplaceN(c.args[0], q38x14SideOld, q38x14SideNew, 1);
  assert q38x14Count(s, '_EXOS_PYPATCH"]') == 0 : 'exos v1.4: the sidecar-bench script still runs the pypatch';
  c + { args: [s] };
local q38x14Leg(jobName, c) =
  assert q38x14Count(c.args[0], q38x14Exec) == 1 && q38x14Count(c.args[0], '_EXOS_PYPATCH"]') == 1
         : 'exos v1.4: job %s vllm-pd does not run the pypatch exactly once' % jobName;
  if q38x14PostPin == '' then c else c + { args: [exosReplaceN(c.args[0], q38x14PostAnchor, q38x14PostAnchor + q38x14PostBlk, 1)] };
local q38x14Job(job) =
  local cs = job.template.spec.template.spec.containers;
  job + { template+: { spec+: { template+: { spec+: {
    containers: [if c.name == 'sidecar-bench' then q38x14Side(c) else if c.name == 'vllm-pd' then q38x14Leg(job.name, c) else c for c in cs],
  } } } } };
local q38x14NSide = std.length([c for j in exosJobSetV13.spec.replicatedJobs for c in j.template.spec.template.spec.containers if c.name == 'sidecar-bench']);
assert q38x14NSide == 1 : 'exos v1.4: expected exactly one sidecar-bench container in the JobSet, found %d' % q38x14NSide;
local exosJobSetV14 = exosJobSetV13 + { spec+: { replicatedJobs: [q38x14Job(j) for j in super.replicatedJobs] } };

// --- v1.5 (X1-WATCH for the lead, 2026-11-17): compile-cache SAVE after /health and before the client is released, optional FROM, and a
// cachedirs du line. Design credit: MM's AGEN sibling 646a3639 (EXOS_COMPILE_CACHE save-before-client + EXOS_COMPILE_CACHE_FROM) and
// mm/reports/exos_howto_q38.md sections 5 and 7 (MM, read-only) ---
// Why (x1b T92b j-f90b0a7d-bb1c-4b3f-9ed0, runs.md X1-WATCH 2026-11-17T05:4xZ, M): on Exos '/', /dev/shm and /tmp/workspace are ONE in-RAM
// tmpfs, and every compile cache of the legs lives there (VLLM_CACHE_ROOT=/tmp/workspace/vllm_pd_cache incl. VLLM_XLA_CACHE_PATH,
// TORCH_TPU tier-2 /tmp/workspace/torch_tpu_cache, inductor_cache incl. AOTAutogradCache's aotautograd copies, jax_cache; tier-3 is off on
// these legs since v1.2 P2). x1b decode: Shmem 12 -> 125 GiB and AnonPages 49 -> 489 GiB over a 59-min cold compile + warmup (MemAvailable
// minimum 247 GiB); prefill: Shmem 9 -> 44 GiB. A cold compile is the startup wall on Exos; a saved cache lets later twins start warm.
// v1.5 = v1.4 (604c0d6c) + delimited blocks, each present only when its knob is on; no v1.4 text is edited:
//   cc defs     (each vllm-pd script, after v1.3's arm): helpers + /tmp/q38x15_split.py, /tmp/q38x15_cat.py; with Q38X_CC_DIRS_EVERY_S > 0
//               a background loop logs 'Q38X_MEMSTAT cachedirs <ISO>Z leg=<leg> <dir>=<MiB>M ... tmpfs_used=<G>G' (du -sm; the totals
//               vllm_pd_cache / inductor_cache include their listed sub-dirs).
//   cc from     (each leg, between v1.3's gate and q38x_guard; only with Q38X_CC_FROM): restore the saver job's compile cache into
//               /tmp/workspace before vllm serve. Refuses (cold start, a 'Q38X_CC restore SKIPPED' line) unless the saver's manifest has
//               .done, the same leg, the same Q38X_CC_KEY and the same post-bootstrap src hash as this leg: compile caches are keyed by
//               program, so a warm start helps only twins on the SAME tree and config (lead ruling 2026-11-17: EXTWIN's pfg / WF / HC2
//               trees each need their own saver; nobody should expect a cross-tree warm start). Every chunk's size and sha256 are checked
//               while it streams; any failure wipes the restored dirs and the leg compiles cold ('Q38X_CC restore FAILED').
//   cc save     (each leg, after v1.3's pid lines; only with Q38X_CC_SAVE=true): a background job waits for this leg's own /health, then
//               streams tar of /tmp/workspace/{vllm_pd_cache,torch_tpu_cache,inductor_cache,jax_cache} | zstd (else pigz, else raw tar:
//               the legs' image f1161cd65 has neither, M) into ${CDK_OUTPUT_DIR}/compile-cache/<leg>.tar.<ext>.NNNN chunks of
//               Q38X_CC_CHUNK_MB (default 1024; up to 4 written at once, each closed when done, so gcsfuse staging stays bounded), then
//               <leg>.manifest.json (per-chunk bytes + sha256, src hashes, key,
//               dirs) and <leg>.done; on any failure <leg>.failed. A 2 s sampler logs 'Q38X_CC save-peak tmpfs_used_max=<G>G
//               memavail_min=<G>GiB' and aborts the save if MemAvailable < Q38X_CC_ABORT_AVAIL_GIB (default 96, above the 64 GiB floors).
//               The leg serves a one-line state (off|pending|saving|done|failed) on TCP Q38X_CC_STATUS_PORT (default 25199).
//   cc gate     (the prefill pod's sidecar-bench, after its proxy wait; only with Q38X_CC_SAVE=true): before the first level the sidecar
//               polls every leg's state port until each says done or failed (or Q38X_CC_GATE_S passes), and logs 'Q38X_CC gate ...' to
//               its log and to ${CDK_OUTPUT_DIR}/q38x_ccgate.log. So a tar never runs inside a measured window (MM's rule), and a later
//               requeue still finds the cache.
//   vcc         (each vllm-pd script, right after the leg's cache exports + mkdir + zero-size cleanup; only with
//               Q38X_VLLM_CACHE_ON_OUTPUTS=true; credit MM sibling v4's EXOS_VLLM_CACHE_ON_OUTPUTS): VLLM_CACHE_ROOT (vLLM's torch.compile /
//               profile cache, the bulk of the legs' tmpfs growth: x1b decode Shmem 12 -> 125 GiB) moves to ${CDK_OUTPUT_DIR}/vllm_cache_live
//               on the job's own gcsfuse outputs; 'Q38X_CC vllm cache root on outputs ...'. MM (M, lead relay 2026-11-17): served c32 cold
//               minimum 36 -> 163 GiB, c48 6 -> 132 GiB, vllm_cache_live 136-146 GiB on outputs with no gcsfuse errors. With it on, FROM skips
//               the restored vllm_pd_cache/torch_compile_cache subtree (it would sit unused in tmpfs).
// Removing the v1.5 blocks gives v1.4's render byte for byte (served_parity_compare_v6.py restores them, then runs v5 unchanged).
// Knobs (read only by this tail; none reaches the engine env or argv):
//   Q38X_CC_SAVE=true|false        default false: SAVE + state port on every leg + the sidecar gate. Needs Q38X_CC_KEY.
//   Q38X_CC_FROM=<job-id>          default '': the saver job whose outputs/<job>-<leg>-0/*/vllm-pd/compile-cache/ to restore. Needs Q38X_CC_KEY.
//   Q38X_CC_KEY=<[A-Za-z0-9._-]{1,64}>  the car's cache key (launchers set the car's params md5); FROM restores only an equal key.
//   Q38X_CC_DIRS_EVERY_S=<n>       default 0 (off); 30..3600: the cachedirs du line period.
//   Q38X_CC_CHUNK_MB=<n>           default 1024 (64..4096).        Q38X_CC_SAVE_MAX_S=<n>   default 3600 (120..7200): the tar's timeout.
//   Q38X_CC_GATE_S=<n>             default 4200 (60..7200).        Q38X_CC_ABORT_AVAIL_GIB=<n> default 96 (16..900).
//   Q38X_CC_STATUS_PORT=<n>        default 25199 (1024..65535).
//   Q38X_VLLM_CACHE_ON_OUTPUTS=true|false  default false: the vcc block (VLLM_CACHE_ROOT on the job's own outputs).
local q38x15Save = cdk.boolVar('Q38X_CC_SAVE', false);
local q38x15From = cdk.var('Q38X_CC_FROM', '');
local q38x15Key = cdk.var('Q38X_CC_KEY', '');
local q38x15DirsEvery = cdk.numVar('Q38X_CC_DIRS_EVERY_S', 0);
local q38x15ChunkMb = cdk.numVar('Q38X_CC_CHUNK_MB', 1024);
local q38x15SaveMax = cdk.numVar('Q38X_CC_SAVE_MAX_S', 3600);
local q38x15GateS = cdk.numVar('Q38X_CC_GATE_S', 4200);
local q38x15Abort = cdk.numVar('Q38X_CC_ABORT_AVAIL_GIB', 96);
local q38x15Port = cdk.numVar('Q38X_CC_STATUS_PORT', 25199);
local q38x15Vcc = cdk.boolVar('Q38X_VLLM_CACHE_ON_OUTPUTS', false);
local q38x15Int(x, lo, hi) = std.isNumber(x) && x == std.floor(x) && x >= lo && x <= hi;
local q38x15KeyOk(k) = std.length(k) >= 1 && std.length(k) <= 64 && std.all([std.member('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-', ch) for ch in std.stringChars(k)]);
local q38x15JobOk(j) = std.length(j) == 25 && std.startsWith(j, 'j-') && std.all([std.member('0123456789abcdef-', ch) for ch in std.stringChars(std.substr(j, 2, 23))])
                       && std.join('', [std.substr(j, i, 1) for i in [10, 15, 20]]) == '---';
assert q38x15Key == '' || q38x15KeyOk(q38x15Key) : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_CC_KEY must be 1-64 characters of [A-Za-z0-9._-]';
assert !q38x15Save || q38x15Key != '' : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_CC_SAVE=true needs Q38X_CC_KEY (the manifest records it)';
assert q38x15From == '' || q38x15JobOk(q38x15From) : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_CC_FROM must be a CDK job id j-xxxxxxxx-xxxx-xxxx-xxxx';
assert q38x15From == '' || q38x15Key != '' : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_CC_FROM needs Q38X_CC_KEY (restore only an equal key)';
assert q38x15DirsEvery == 0 || q38x15Int(q38x15DirsEvery, 30, 3600) : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_CC_DIRS_EVERY_S must be 0 or an integer from 30 to 3600';
assert q38x15Int(q38x15ChunkMb, 64, 4096) : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_CC_CHUNK_MB must be an integer from 64 to 4096';
assert q38x15Int(q38x15SaveMax, 120, 7200) : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_CC_SAVE_MAX_S must be an integer from 120 to 7200';
assert q38x15Int(q38x15GateS, 60, 7200) : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_CC_GATE_S must be an integer from 60 to 7200';
assert q38x15Int(q38x15Abort, 16, 900) : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_CC_ABORT_AVAIL_GIB must be an integer from 16 to 900';
assert q38x15Int(q38x15Port, 1024, 65535) && !std.member([25000, 25100, 25200, 26379, 14579], q38x15Port)
       : 'exos-infx-qwen38-flash-agentx-disagg: Q38X_CC_STATUS_PORT must be a free port from 1024 to 65535';
local q38x15On = q38x15Save || q38x15From != '' || q38x15DirsEvery > 0;  // the cc defs block; the vcc block has its own knob
local q38x15Defs = std.strReplace(std.strReplace(std.strReplace(std.strReplace(std.strReplace(std.strReplace(std.strReplace(|||
  # ---- q38x v1.5 cc BEGIN (X1-WATCH; credit MM AGEN sibling 646a3639 + exos_howto_q38.md section 5): compile-cache SAVE / FROM / cachedirs
  _q38x15_leg="${PD_LEG:-leg}${PD_DECODE_INDEX:+${PD_DECODE_INDEX}}"
  _q38x15_ccd="${CDK_OUTPUT_DIR:-/tmp/cdk_output}/compile-cache"
  _q38x15_key='@KEY@'
  q38x15_state() { echo "$1" > /tmp/q38x_cc_state.tmp 2>/dev/null && mv -f /tmp/q38x_cc_state.tmp /tmp/q38x_cc_state 2>/dev/null || true; }
  q38x15_avail() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null; }
  q38x15_tmpfs() { df -BG --output=used /tmp/workspace 2>/dev/null | tail -n 1 | tr -dc '0-9'; }
  q38x15_du() { (cd /tmp/workspace 2>/dev/null && for _p in vllm_pd_cache/torch_compile_cache vllm_pd_cache/xla_cache vllm_pd_cache torch_tpu_cache inductor_cache/aotautograd inductor_cache jax_cache tier3_cache; do [ -e "${_p}" ] && printf '%s=%sM ' "${_p}" "$(du -sm "${_p}" 2>/dev/null | cut -f1)"; done) 2>/dev/null; }
  q38x15_comp() {  # -> _q38x15_c (compressor), _q38x15_d (decompressor), _q38x15_x (extension); no single-thread gzip (too slow for ~100 GiB)
    if command -v zstd >/dev/null 2>&1; then _q38x15_c="zstd -T8 -3 -q -c"; _q38x15_d="zstd -d -q -c"; _q38x15_x=zst
    elif command -v pigz >/dev/null 2>&1; then _q38x15_c="pigz -1 -c"; _q38x15_d="pigz -d -c"; _q38x15_x=gz
    else _q38x15_c="cat"; _q38x15_d="cat"; _q38x15_x=tar; fi; }
  cat > /tmp/q38x15_split.py <<'Q38X15_SPLIT'
  # stdin -> <prefix>NNNN chunks of <bytes> + a manifest of bytes and sha256 per chunk. Each chunk is read whole, then written and closed by
  # one of 4 writer threads (gcsfuse uploads a file on close, so 4 uploads overlap; at most 5 chunks are in memory). Stops at the abort flag.
  # argv: prefix chunk_bytes manifest_tmp abort_flag
  import concurrent.futures as cf, hashlib, json, os, sys
  prefix, size, man, abort = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
  inp = sys.stdin.buffer
  def put(name, data):
      with open(name, 'wb') as f:
          f.write(data)
      return os.path.basename(name), len(data), hashlib.sha256(data).hexdigest()
  futs, n, total = [], 0, 0
  with cf.ThreadPoolExecutor(max_workers=4) as ex:
      while True:
          if os.path.exists(abort):
              os._exit(5)
          parts, got = [], 0
          while got < size:
              b = inp.read(min(8 << 20, size - got))
              if not b:
                  break
              parts.append(b); got += len(b)
          if got == 0:
              break
          while sum(not f.done() for f in futs) >= 4:
              cf.wait(futs, return_when=cf.FIRST_COMPLETED)
          futs.append(ex.submit(put, '%s%04d' % (prefix, n), b''.join(parts))); n += 1; total += got
          if got < size:
              break
  chunks = [dict(zip(('name', 'bytes', 'sha256'), f.result())) for f in futs]
  if os.path.exists(abort):
      sys.exit(5)
  json.dump({'chunks': chunks, 'total_bytes': total}, open(man, 'w'))
  Q38X15_SPLIT
  cat > /tmp/q38x15_cat.py <<'Q38X15_CAT'
  # stream a saved cache's chunks to stdout in order, checking each chunk's bytes and sha256 against the manifest. argv: manifest.json
  import hashlib, json, os, sys
  m = json.load(open(sys.argv[1])); d = os.path.dirname(sys.argv[1]); out = sys.stdout.buffer
  for c in m['chunks']:
      h, got = hashlib.sha256(), 0
      with open(os.path.join(d, c['name']), 'rb') as f:
          while True:
              b = f.read(8 << 20)
              if not b:
                  break
              out.write(b); h.update(b); got += len(b)
      if got != c['bytes'] or h.hexdigest() != c['sha256']:
          sys.stderr.write('chunk %s: %d of %d bytes, sha256 %s\n' % (c['name'], got, c['bytes'], 'ok' if h.hexdigest() == c['sha256'] else 'MISMATCH'))
          sys.exit(3)
  out.flush()
  Q38X15_CAT
  if [ "@DIRS_EVERY@" != "0" ]; then
    ( while :; do sleep @DIRS_EVERY@; q38x_say "Q38X_MEMSTAT cachedirs $(date -u +%Y-%m-%dT%H:%M:%SZ) leg=${_q38x15_leg} $(q38x15_du)tmpfs_used=$(q38x15_tmpfs)G"; done ) > /dev/null 2>&1 &
  fi
  q38x15_status_server() {  # one line per connection: off|pending|saving|done|failed
    python3 - "@PORT@" > /dev/null 2>&1 <<'Q38X15_STATUS' &
  import socket, sys
  try:
      s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0); s.bind(('::', int(sys.argv[1]))); fam = 'inet6-dualstack'
  except OSError:
      s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      s.bind(('0.0.0.0', int(sys.argv[1]))); fam = 'inet'
  s.listen(16)
  open('/tmp/q38x_cc_status.bind', 'w').write(fam + '\n')
  while True:
      c, _ = s.accept()
      try:
          st = open('/tmp/q38x_cc_state').read().strip() or 'unknown'
      except OSError:
          st = 'unknown'
      try:
          c.sendall((st + '\n').encode())
      except OSError:
          pass
      c.close()
  Q38X15_STATUS
  }
  q38x15_cc_from() {  # $1 leg role: restore the saver's cache before vllm serve, or log why not (cold start)
    local _f='@FROM@' _m _t0 _rc
    [ -n "${_f}" ] || return 0
    _m="$(ls -1 /exos/gcs/cloud-devkit/jobs/${_f}/outputs/${_f}-$1-0/*/vllm-pd/compile-cache/${_q38x15_leg}.manifest.json 2>/dev/null | head -n 1)"
    if [ -z "${_m}" ] || [ ! -e "${_m%.manifest.json}.done" ]; then
      q38x_say "Q38X_CC restore SKIPPED leg=${_q38x15_leg}: no saver manifest with .done under /exos/gcs/cloud-devkit/jobs/${_f}/outputs/${_f}-$1-0/ (cold start)"; return 0; fi
    if ! python3 -c 'import json, sys; m = json.load(open(sys.argv[1])); sys.exit(0 if (m.get("leg"), m.get("key"), m.get("src_post")) == (sys.argv[2], sys.argv[3], sys.argv[4]) else 1)' \
         "${_m}" "${_q38x15_leg}" "${_q38x15_key}" "${_q38x_post:-}"; then
      q38x_say "Q38X_CC restore SKIPPED leg=${_q38x15_leg}: ${_m} is not this leg / key ${_q38x15_key} / post src ${_q38x_post:-unset} (caches are keyed by program; cold start)"; return 0; fi
    q38x15_comp; _t0=$(date +%s)
    case "${_m}" in *) [ "$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("ext", ""))' "${_m}")" = "${_q38x15_x}" ] \
      || { q38x_say "Q38X_CC restore SKIPPED leg=${_q38x15_leg}: ${_m} was saved with another compressor (cold start)"; return 0; } ;; esac
    q38x_say "Q38X_CC restoring leg=${_q38x15_leg} from ${_m} $(date -u +%Y-%m-%dT%H:%M:%SZ) tmpfs_used=$(q38x15_tmpfs)G MemAvailable=$(q38x15_avail)GiB"
    local _ex=""; [ "${VLLM_CACHE_ROOT:-/tmp/workspace/vllm_pd_cache}" = /tmp/workspace/vllm_pd_cache ] || _ex="--exclude=vllm_pd_cache/torch_compile_cache"
    [ -n "${_ex}" ] && q38x_say "Q38X_CC restore leg=${_q38x15_leg}: VLLM_CACHE_ROOT is ${VLLM_CACHE_ROOT} (on outputs): skipping vllm_pd_cache/torch_compile_cache"
    ( set -o pipefail; python3 /tmp/q38x15_cat.py "${_m}" | ${_q38x15_d} | tar -x -C /tmp/workspace ${_ex} ) > /tmp/q38x_cc_restore.err 2>&1; _rc=$?
    if [ "${_rc}" != 0 ]; then
      (cd /tmp/workspace && rm -rf vllm_pd_cache torch_tpu_cache inductor_cache jax_cache) 2>/dev/null
      mkdir -p "${VLLM_CACHE_ROOT:-/tmp/workspace/vllm_pd_cache}" "${VLLM_XLA_CACHE_PATH:-/tmp/workspace/vllm_pd_cache/xla_cache}" "${TORCH_TPU_CACHE_DIR:-/tmp/workspace/torch_tpu_cache}" "${TORCHINDUCTOR_CACHE_DIR:-/tmp/workspace/inductor_cache}" "${JAX_COMPILATION_CACHE_DIR:-/tmp/workspace/jax_cache}" 2>/dev/null
      q38x_say "Q38X_CC restore FAILED leg=${_q38x15_leg} rc=${_rc} after $(( $(date +%s) - _t0 )) s ($(tail -c 300 /tmp/q38x_cc_restore.err | tr '\n' ' ')); restored dirs wiped, cold compile"; return 0; fi
    q38x_say "Q38X_CC restored leg=${_q38x15_leg} in $(( $(date +%s) - _t0 )) s: $(q38x15_du)tmpfs_used=$(q38x15_tmpfs)G MemAvailable=$(q38x15_avail)GiB"
  }
  q38x15_cc_save() {  # $1 leg role (runs in the background job of q38x15_cc_save_bg)
    local _t0 _rc _ab=/tmp/q38x_cc_abort _dn=/tmp/q38x_cc_done _dirs="" _p _peak
    q38x15_comp; _t0=$(date +%s); rm -f "${_ab}" "${_dn}" /tmp/q38x_cc_peak
    for _p in vllm_pd_cache torch_tpu_cache inductor_cache jax_cache; do [ -d "/tmp/workspace/${_p}" ] && _dirs="${_dirs} ${_p}"; done
    mkdir -p "${_q38x15_ccd}" 2>/dev/null
    if [ -z "${_dirs}" ] || [ ! -d "${_q38x15_ccd}" ]; then
      echo "no cache dirs or no ${_q38x15_ccd}" > "${_q38x15_ccd}/${_q38x15_leg}.failed" 2>/dev/null; q38x15_state failed
      q38x_say "Q38X_CC save FAILED leg=${_q38x15_leg}: nothing to save or no output dir"; return 0; fi
    q38x15_state saving
    q38x_say "Q38X_CC saving leg=${_q38x15_leg} $(date -u +%Y-%m-%dT%H:%M:%SZ) key=${_q38x15_key} dirs:${_dirs} $(q38x15_du)tmpfs_used=$(q38x15_tmpfs)G MemAvailable=$(q38x15_avail)GiB comp=${_q38x15_x} chunk=@CHUNK_MB@M -> ${_q38x15_ccd}/${_q38x15_leg}.tar.${_q38x15_x}.NNNN"
    ( _mx=0; _mn=999999
      while [ ! -e "${_dn}" ]; do
        _u=$(q38x15_tmpfs); _a=$(q38x15_avail)
        [ -n "${_u}" ] && [ "${_u}" -gt "${_mx}" ] && _mx=${_u}
        [ -n "${_a}" ] && [ "${_a}" -lt "${_mn}" ] && _mn=${_a}
        echo "${_mx} ${_mn}" > /tmp/q38x_cc_peak
        if [ -n "${_a}" ] && [ "${_a}" -lt @ABORT@ ] && [ ! -e "${_ab}" ]; then : > "${_ab}"; q38x_say "Q38X_CC save ABORT leg=${_q38x15_leg}: MemAvailable ${_a} GiB < @ABORT@ GiB"; fi
        sleep 2
      done ) > /dev/null 2>&1 &
    ( set -o pipefail; timeout @SAVE_MAX@ tar -C /tmp/workspace -cf - ${_dirs} | ${_q38x15_c} \
        | python3 /tmp/q38x15_split.py "${_q38x15_ccd}/${_q38x15_leg}.tar.${_q38x15_x}." $(( @CHUNK_MB@ * 1048576 )) /tmp/q38x_cc_man.json "${_ab}" ) > /tmp/q38x_cc_save.err 2>&1; _rc=$?
    : > "${_dn}"; sleep 3; _peak="$(cat /tmp/q38x_cc_peak 2>/dev/null)"
    q38x_say "Q38X_CC save-peak leg=${_q38x15_leg} tmpfs_used_max=${_peak%% *}G memavail_min=${_peak##* }GiB (2 s samples during the save)"
    if [ "${_rc}" = 0 ] && [ ! -e "${_ab}" ] && python3 - "${_q38x15_ccd}/${_q38x15_leg}.manifest.json" /tmp/q38x_cc_man.json "${_q38x15_leg}" "${_q38x15_key}" \
         "${_q38x_pre:-}" "${_q38x_post:-}" "${_q38x15_x}" "${JS_NAME:-}" "${_dirs}" "$(q38x15_du)" "$(( $(date +%s) - _t0 ))" <<'Q38X15_MAN'
  import json, sys
  out, tmp, leg, key, pre, post, ext, job, dirs, du, secs = sys.argv[1:12]
  m = json.load(open(tmp))
  m.update(leg=leg, key=key, src_pre=pre, src_post=post, ext=ext, job=job, dirs=dirs.split(), du=du.strip(), seconds=int(secs), format='q38x15-v1')
  json.dump(m, open(out, 'w'), indent=1)
  Q38X15_MAN
    then
      : > "${_q38x15_ccd}/${_q38x15_leg}.done"; q38x15_state done
      q38x_say "Q38X_CC saved leg=${_q38x15_leg} $(python3 -c 'import json, sys; m = json.load(open(sys.argv[1])); print(len(m["chunks"]), "chunks", m["total_bytes"], "bytes")' "${_q38x15_ccd}/${_q38x15_leg}.manifest.json") in $(( $(date +%s) - _t0 )) s -> ${_q38x15_ccd}/${_q38x15_leg}.manifest.json"
    else
      { echo "rc=${_rc} abort=$([ -e "${_ab}" ] && echo yes || echo no)"; tail -c 600 /tmp/q38x_cc_save.err; } > "${_q38x15_ccd}/${_q38x15_leg}.failed" 2>/dev/null; q38x15_state failed
      q38x_say "Q38X_CC save FAILED leg=${_q38x15_leg} rc=${_rc} abort=$([ -e "${_ab}" ] && echo yes || echo no) after $(( $(date +%s) - _t0 )) s ($(tail -c 200 /tmp/q38x_cc_save.err | tr '\n' ' '))"
    fi
  }
  q38x15_cc_save_bg() {  # $1 leg role, $2 this leg's port, $3 the vllm serve pid: SAVE after this leg's /health (background)
    [ "@SAVE@" = "true" ] || return 0
    q38x15_state pending; q38x15_status_server
    ( until python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)' "http://[${MY_IP}]:$2/health" > /dev/null 2>&1; do
        if [ ! -d "/proc/$3" ] || [ -e /tmp/q38x_memhalt ]; then
          q38x15_state failed; mkdir -p "${_q38x15_ccd}" 2>/dev/null; echo "vllm serve gone or halted before /health" > "${_q38x15_ccd}/${_q38x15_leg}.failed" 2>/dev/null
          q38x_say "Q38X_CC save FAILED leg=${_q38x15_leg}: vllm serve exited or halted before /health"; exit 0; fi
        sleep 10
      done
      q38x15_cc_save "$1" ) > /dev/null 2>&1 &
  }
  # ---- q38x v1.5 cc END
|||, '@KEY@', q38x15Key), '@DIRS_EVERY@', std.toString(q38x15DirsEvery)), '@PORT@', std.toString(q38x15Port)), '@FROM@', q38x15From),
   '@CHUNK_MB@', std.toString(q38x15ChunkMb)), '@ABORT@', std.toString(q38x15Abort)), '@SAVE_MAX@', std.toString(q38x15SaveMax));
local q38x15DefsAll = std.strReplace(q38x15Defs, '@SAVE@', if q38x15Save then 'true' else 'false');
local q38x15Gate = std.strReplace(std.strReplace(|||
  # ---- q38x v1.5 ccgate BEGIN (X1-WATCH): the client starts only after every leg's compile-cache SAVE is done or failed
  _q38x15_gt0=$(date +%s); _q38x15_gl="${CDK_OUTPUT_DIR:-/tmp}/q38x_ccgate.log"
  echo "Q38X_CC gate $(date -u +%Y-%m-%dT%H:%M:%SZ): waiting for the compile-cache SAVE on ${PREFILL_HOSTS} ${DECODE_HOSTS} (port @PORT@, up to @GATE@ s)" | tee -a "${_q38x15_gl}"
  while :; do
    _q38x15_all=1; _q38x15_line=""
    for _q38x15_h in ${PREFILL_HOSTS} ${DECODE_HOSTS}; do
      _q38x15_st="$(python3 -c 'import socket, sys
  try:
      c = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=5); print(c.recv(64).decode().strip() or "empty")
  except Exception:
      print("unreachable")' "${_q38x15_h}" @PORT@ 2>/dev/null)"
      _q38x15_line="${_q38x15_line} ${_q38x15_h}=${_q38x15_st:-none}"
      case "${_q38x15_st}" in done|failed) ;; *) _q38x15_all=0 ;; esac
    done
    if [ "${_q38x15_all}" = 1 ]; then echo "Q38X_CC gate $(date -u +%Y-%m-%dT%H:%M:%SZ): released after $(( $(date +%s) - _q38x15_gt0 )) s:${_q38x15_line}" | tee -a "${_q38x15_gl}"; break; fi
    if [ $(( $(date +%s) - _q38x15_gt0 )) -ge @GATE@ ]; then echo "Q38X_CC gate $(date -u +%Y-%m-%dT%H:%M:%SZ): TIMEOUT after @GATE@ s:${_q38x15_line}; the client starts now" | tee -a "${_q38x15_gl}"; break; fi
    sleep 10
  done
  # ---- q38x v1.5 ccgate END
|||, '@PORT@', std.toString(q38x15Port)), '@GATE@', std.toString(q38x15GateS));
local q38x15VccBlk = |||
  # ---- q38x v1.5 vcc BEGIN (X1-WATCH; credit MM sibling v4 EXOS_VLLM_CACHE_ON_OUTPUTS): VLLM_CACHE_ROOT on the job's own outputs, not tmpfs
  if [ -n "${CDK_OUTPUT_DIR:-}" ]; then
    export VLLM_CACHE_ROOT="${CDK_OUTPUT_DIR}/vllm_cache_live"
    mkdir -p "${VLLM_CACHE_ROOT}" 2>/dev/null || true
    q38x_say "Q38X_CC vllm cache root on outputs ${VLLM_CACHE_ROOT} ($(df -BG --output=fstype,used,avail "${CDK_OUTPUT_DIR}" 2>/dev/null | tail -n 1 | tr -s ' '))"
  else
    q38x_say "Q38X_CC vllm cache root on outputs NOT applied: no CDK_OUTPUT_DIR (VLLM_CACHE_ROOT stays ${VLLM_CACHE_ROOT:-unset})"
  fi
  # ---- q38x v1.5 vcc END
|||;
local q38x15AnchorVcc = 'find -L "${VLLM_CACHE_ROOT}" -type f -size 0 -delete 2>/dev/null || true\n';
local q38x15From1(ind, role) = '%s# ---- q38x v1.5 ccfrom BEGIN\n%sq38x15_cc_from %s\n%s# ---- q38x v1.5 ccfrom END\n' % [ind, ind, role, ind];
local q38x15Save1(ind, role, port, var) = '%s# ---- q38x v1.5 ccsave BEGIN\n%sq38x15_cc_save_bg %s %d "${%s}"\n%s# ---- q38x v1.5 ccsave END\n' % [ind, ind, role, port, var, ind];
// The anchors in v1.4's leg script (one each, counted: a moved anchor fails the render).
local q38x15AnchorArm = '# ---- q38x v1.3 arm END\n';
local q38x15AnchorFromD = '  # ---- q38x v1.3 gate END\n  q38x_guard decode || exit 1\n';
local q38x15AnchorFromP = '# ---- q38x v1.3 gate END\nq38x_guard prefill || exit 1\n';
local q38x15AnchorSaveD = '  q38x_memhalt_pid decode "${DECODE_PID}"\n  # ---- q38x v1.3 pid END\n';
local q38x15AnchorSaveP = 'q38x_memhalt_pid prefill "${SERVER_PID}"\n# ---- q38x v1.3 pid END\n';
local q38x15AnchorGate = 'wait_up proxy "${FRONT_URL}/healthcheck" wait-proxy\n';
local q38x15LegScript(s0) =
  local s = if q38x15Vcc then exosReplaceN(s0, q38x15AnchorVcc, q38x15AnchorVcc + q38x15VccBlk, 1) else s0;
  if !q38x15On then s else
  local s1 = exosReplaceN(s, q38x15AnchorArm, q38x15AnchorArm + q38x15DefsAll, 1);
  local s2 = if q38x15From == '' then s1 else
    exosReplaceN(exosReplaceN(s1,
      q38x15AnchorFromD, '  # ---- q38x v1.3 gate END\n' + q38x15From1('  ', 'decode') + '  q38x_guard decode || exit 1\n', 1),
      q38x15AnchorFromP, '# ---- q38x v1.3 gate END\n' + q38x15From1('', 'prefill') + 'q38x_guard prefill || exit 1\n', 1);
  if !q38x15Save then s2 else
    exosReplaceN(exosReplaceN(s2,
      q38x15AnchorSaveD, q38x15AnchorSaveD + q38x15Save1('  ', 'decode', 25100, 'DECODE_PID'), 1),
      q38x15AnchorSaveP, q38x15AnchorSaveP + q38x15Save1('', 'prefill', 25200, 'SERVER_PID'), 1);
local q38x15Side(c) = if !q38x15Save then c else c + { args: [exosReplaceN(c.args[0], q38x15AnchorGate, q38x15AnchorGate + q38x15Gate, 1)] };
local q38x15Job(job) =
  local cs = job.template.spec.template.spec.containers;
  job + { template+: { spec+: { template+: { spec+: {
    containers: [if c.name == 'sidecar-bench' then q38x15Side(c) else if c.name == 'vllm-pd' then c + { args: [q38x15LegScript(c.args[0])] } else c for c in cs],
  } } } } };
local exosJobSetV15 = exosJobSetV14 + { spec+: { replicatedJobs: [q38x15Job(j) for j in super.replicatedJobs] } };

// --- v1.6 (HIGHC-SIB16 for HIGHC-LEAD, 2026-11-18): RAIDEN_WHEEL on cdk-tpu-exos. Cloud parity: cdk.agentx q38RaidenWheelScript (BLSTACK,
// lib ede826e8 -> e00021e2, installed by the lead 2026-11-17T21:02Z), which installs Benliang Wang's Raiden wheel on each Cloud server leg ---
// Why (runs.md BLSTACK-EXOS 2026-11-17T21:56Z D2 + HIGHC-WFX2 2026-11-18T00:57Z, M): v1.5 renders RAIDEN_WHEEL to nothing. The Cloud block sits in
// agentx.compileCacheInitScript, which only the Cloud server script (lines above) calls; on cdk-tpu-exos exos_vllm.pdRecipe writes each leg's
// script, so the knob was read by no rendered text. His R form (prefill tier cpu_bytes_to_use 498e9 + VLLM_TPU_OFFLOAD_TIER_CLASSES=1 +
// VLLM_TPU_OFFLOAD_TIER_KEY_ROWS=1) needs his wheel tier_rows_9d19c961 for KEY_ROWS (BLSTACK, C), so Exos R-form cars had to drop it.
// v1.6 = v1.5 (6722969d) + ONE delimited block per vllm-pd script, present only when RAIDEN_WHEEL is set; no v1.5 text is edited:
//   wheel  (each vllm-pd script, right before exos_vllm's `cat > /tmp/exos_gcs_cp.py <<'PYGCS'`: after v1.2's bootstrap + post and v1.4's
//          postpin, before the leg's exports, compile-cache dirs, gate, guard and vllm serve). Steps, each FATAL through q38x_fatal (a
//          'FATAL: Q38X_WHEEL ...' line, q38x.log synced to outputs, exit 1: the leg never starts vllm serve on the image's tpu_sync):
//          1 fetch: copy gs://cloud-devkit/wheels/<RAIDEN_WHEEL> from the Exos GCS mount /exos/gcs/cloud-devkit (exos_vllm's own read path;
//            the vllm-pd containers have no /tmp/cdk_gcs.py, which the Cloud block uses) to /root/wheels/<RAIDEN_WHEEL> (the Cloud path);
//          2 check, before anything is uninstalled: the file is a wheel whose METADATA Name / Version equal its file name and whose every
//            member matches its RECORD sha256 (so a truncated or renamed copy never replaces the image's tpu_sync);
//          3 the Cloud block's uninstall + install, verbatim (uv pip --system if uv exists, else python3 -m pip; --no-deps);
//          4 the Cloud block's import check, verbatim (from tpu_sync.api.torch.kv_cache_manager import _torch_impl; _torch_impl());
//          5 verify under the interpreter of the `vllm` entry point (python3 if none): exactly one installed distribution of the wheel's name,
//            its version = the file name's, every RECORD file of the wheel on disk with the wheel's sha256, tpu_sync (if a regular package) and
//            tpu_sync.api.torch.kv_cache_manager import from those verified files, and no other installed distribution lists those files;
//          6 lines: 'RAIDEN_WHEEL: installed <file> as <dist> <version>' (the Cloud line, for readers that grep it) and
//            'Q38X_WHEEL installed <file> <sha256> leg=<leg> dist=... version=... files=... root=... tpu_sync=... tpu_sync.__version__=...
//            tpu-raiden-torch=... python=...' (the engagement line), plus 'Q38X_WHEEL ... image before: tpu-sync-torch=<v> tpu-raiden-torch=<v>'.
// Knob (the Cloud one, same name and meaning; read on Cloud by cdk.agentx, on cdk-tpu-exos by this tail; never reaches the engine env):
//   RAIDEN_WHEEL=<file name under gs://cloud-devkit/wheels/>   default '': no wheel block (render = v1.5's). Refused: a '/' (the Cloud message),
//                               anything but [A-Za-z0-9._+-], a name not ending in .whl or without 5-6 dash fields (it is pasted into the script).
local q38x16Whl = cdk.var('RAIDEN_WHEEL', '');
local q38x16Chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._+-';
assert q38x16Whl == '' || std.length(std.findSubstr('/', q38x16Whl)) == 0 : 'RAIDEN_WHEEL is a file name under gs://cloud-devkit/wheels/, got "%s"' % q38x16Whl;
assert q38x16Whl == '' || (std.endsWith(q38x16Whl, '.whl') && std.all([std.member(q38x16Chars, ch) for ch in std.stringChars(q38x16Whl)])
                           && std.member([5, 6], std.length(std.split(std.substr(q38x16Whl, 0, std.length(q38x16Whl) - 4), '-'))))
       : 'exos-infx-qwen38-flash-agentx-disagg: RAIDEN_WHEEL must be a wheel file name <dist>-<version>[-<build>]-<python>-<abi>-<platform>.whl of [A-Za-z0-9._+-], got "%s"' % q38x16Whl;
local q38x16Blk = std.strReplace(|||
  # ---- q38x v1.6 wheel BEGIN (HIGHC-SIB16; Cloud parity: cdk.agentx q38RaidenWheelScript, BLSTACK e00021e2): RAIDEN_WHEEL on this leg before vllm serve
  _q38x16_whl='@WHL@'
  _q38x16_leg="${PD_LEG:-?}${PD_DECODE_INDEX:+-${PD_DECODE_INDEX}}"
  _q38x16_src="/exos/gcs/cloud-devkit/wheels/${_q38x16_whl}"
  RAIDEN_WHL="/root/wheels/${_q38x16_whl}"
  cat > /tmp/q38x16_wheel.py <<'Q38X16_PY'
  # q38x v1.6 wheel helper: check <whl> | dists | verify <whl>   (stdout: one line; exit 1 with the reason on stdout on any failure)
  import base64, csv, hashlib, importlib, io, os, re, sys, zipfile
  import importlib.metadata as md
  def die(msg):
      print(msg); sys.exit(1)
  def norm(n):
      return re.sub(r'[-_.]+', '-', n or '').lower()
  def vkey(v):
      try:
          from packaging.version import Version
          return str(Version(v))
      except Exception:
          return v
  def b64(data):
      return 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
  def fname(path):
      b = os.path.basename(path)
      p = b[:-4].split('-') if b.endswith('.whl') else []
      if len(p) not in (5, 6):
          die('%s is not a wheel file name <dist>-<version>[-<build>]-<python>-<abi>-<platform>.whl' % b)
      return p[0], p[1]
  def record(path):  # -> dist, version, {RECORD path: sha256=...} after checking METADATA against the file name and every member's sha256
      dist, ver = fname(path)
      try:
          z = zipfile.ZipFile(path)
          rec = [n for n in z.namelist() if n.count('/') == 1 and n.endswith('.dist-info/RECORD')]
          if len(rec) != 1:
              die('want one top-level *.dist-info/RECORD, found %d' % len(rec))
          meta = z.read(rec[0][:-len('RECORD')] + 'METADATA').decode('utf-8', 'replace')
          rows = {r[0]: r[1] for r in csv.reader(io.StringIO(z.read(rec[0]).decode('utf-8'))) if len(r) >= 2 and r[1].startswith('sha256=')}
          for n, h in rows.items():
              if b64(z.read(n)) != h:
                  die('member %s does not match its RECORD sha256' % n)
      except (zipfile.BadZipFile, KeyError, OSError, ValueError) as e:
          die('not a readable wheel: %s: %s' % (type(e).__name__, e))
      m_name = re.search(r'^Name: *(.+?)\s*$', meta, re.M)
      m_ver = re.search(r'^Version: *(.+?)\s*$', meta, re.M)
      if not m_name or not m_ver:
          die('METADATA has no Name / Version')
      if norm(m_name.group(1)) != norm(dist) or vkey(m_ver.group(1)) != vkey(ver):
          die('METADATA %s %s does not match the file name (%s %s)' % (m_name.group(1), m_ver.group(1), dist, ver))
      if not rows:
          die('RECORD lists no hashed file')
      return dist, ver, rows
  def dver(n):
      try:
          return md.version(n)
      except md.PackageNotFoundError:
          return 'none'
  cmd = sys.argv[1] if len(sys.argv) > 1 else ''
  if cmd == 'dists':
      print('tpu-sync-torch=%s tpu-raiden-torch=%s' % (dver('tpu-sync-torch'), dver('tpu-raiden-torch')))
  elif cmd == 'check':
      dist, ver, rows = record(sys.argv[2])
      print('wheel %s %s, %d RECORD files, every member matches its sha256' % (norm(dist), ver, len(rows)))
  elif cmd == 'verify':
      dist, ver, rows = record(sys.argv[2])
      ds = [d for d in md.distributions() if norm(d.metadata['Name']) == norm(dist)]
      if len(ds) != 1:
          die('%d installed distributions named %s on this interpreter: %s' % (len(ds), norm(dist), ', '.join('%s at %s' % (d.version, d.locate_file('')) for d in ds) or 'none'))
      d = ds[0]
      if vkey(d.version) != vkey(ver):
          die('installed %s is version %s, the wheel file is %s' % (norm(dist), d.version, ver))
      ok, bad = {}, []
      for p, h in sorted(rows.items()):
          if '.dist-info/' in p or '.data/' in p:
              continue
          f = str(d.locate_file(p))
          try:
              data = open(f, 'rb').read()
          except OSError:
              bad.append('missing ' + p); continue
          if b64(data) != h:
              bad.append('sha256 ' + p); continue
          ok[os.path.realpath(f)] = p
      if bad:
          die('%d wheel files are not on disk as shipped: %s' % (len(bad), '; '.join(bad[:4])))
      if not ok:
          die('the wheel ships no package file to verify')
      mods = {}
      for m in ('tpu_sync', 'tpu_sync.api.torch.kv_cache_manager'):
          mod = importlib.import_module(m)
          f = getattr(mod, '__file__', None)
          if f is None and m == 'tpu_sync':
              mods[m] = 'namespace'; continue
          if not f or os.path.realpath(f) not in ok:
              die('%s imports from %s, not from a file of the installed wheel (root %s)' % (m, f, d.locate_file('')))
          mods[m] = ok[os.path.realpath(f)]
      own = set(v for v in mods.values() if v != 'namespace')
      for od in md.distributions():
          if norm(od.metadata['Name']) == norm(dist):
              continue
          hit = sorted(own & set(str(x) for x in (od.files or [])))
          if hit:
              die('%s %s also lists %s (the image copy was not removed)' % (od.metadata['Name'], od.version, ', '.join(hit)))
      ts = sys.modules['tpu_sync']
      print('dist=%s version=%s files=%d root=%s tpu_sync=%s tpu_sync.__version__=%s tpu-raiden-torch=%s python=%s' % (
          norm(dist), d.version, len(ok), d.locate_file(''), mods['tpu_sync'], getattr(ts, '__version__', 'n/a'), dver('tpu-raiden-torch'), sys.executable))
  else:
      die('usage: q38x16_wheel.py check <whl> | dists | verify <whl>')
  Q38X16_PY
  q38x_say "Q38X_WHEEL leg=${_q38x16_leg} start $(date -u +%Y-%m-%dT%H:%M:%SZ): RAIDEN_WHEEL=${_q38x16_whl} from gs://cloud-devkit/wheels/ (Exos mount ${_q38x16_src}); image before: $(python3 /tmp/q38x16_wheel.py dists 2>&1 | tr '\n' ' ')"
  mkdir -p /root/wheels || q38x_fatal "Q38X_WHEEL leg=${_q38x16_leg} cannot create /root/wheels"
  rm -f "${RAIDEN_WHL}" "${RAIDEN_WHL}.q38x16tmp"
  { [ -f "${_q38x16_src}" ] && cp "${_q38x16_src}" "${RAIDEN_WHL}.q38x16tmp" && mv -f "${RAIDEN_WHL}.q38x16tmp" "${RAIDEN_WHL}"; } \
    || q38x_fatal "Q38X_WHEEL leg=${_q38x16_leg} cannot fetch gs://cloud-devkit/wheels/${_q38x16_whl} (${_q38x16_src} missing or the copy failed); no fallback to the image's tpu_sync"
  _q38x16_sha="$(sha256sum "${RAIDEN_WHL}" | cut -c1-64)"
  _q38x16_chk="$(python3 /tmp/q38x16_wheel.py check "${RAIDEN_WHL}" 2>&1)" \
    || q38x_fatal "Q38X_WHEEL leg=${_q38x16_leg} ${RAIDEN_WHL} (sha256 ${_q38x16_sha}) is not a valid wheel for its file name, nothing uninstalled: ${_q38x16_chk}"
  q38x_say "Q38X_WHEEL leg=${_q38x16_leg} fetched ${RAIDEN_WHL} $(stat -c %s "${RAIDEN_WHL}") B sha256 ${_q38x16_sha}: ${_q38x16_chk}"
  # the Cloud block's install, verbatim: tpu_raiden_torch and tpu_sync_torch both unpack into tpu_sync/, so the image's copy goes first
  if command -v uv >/dev/null 2>&1; then
    uv pip uninstall --system tpu-raiden-torch tpu-sync-torch >/dev/null 2>&1 || true
    uv pip install --system --no-deps "$RAIDEN_WHL" || q38x_fatal "Q38X_WHEEL leg=${_q38x16_leg} install failed (uv pip install --system --no-deps ${RAIDEN_WHL}); vllm serve not started"
  else
    python3 -m pip uninstall -y tpu-raiden-torch tpu-sync-torch >/dev/null 2>&1 || true
    python3 -m pip install --no-deps "$RAIDEN_WHL" || q38x_fatal "Q38X_WHEEL leg=${_q38x16_leg} install failed (python3 -m pip install --no-deps ${RAIDEN_WHL}); vllm serve not started"
  fi
  python3 -c "from tpu_sync.api.torch.kv_cache_manager import _torch_impl; _torch_impl()" \
    || q38x_fatal "Q38X_WHEEL leg=${_q38x16_leg} ${RAIDEN_WHL} does not import against the image's torch_tpu (the Cloud block's check)"
  _q38x16_py=python3
  _q38x16_vb="$(command -v vllm 2>/dev/null || true)"
  if [ -n "${_q38x16_vb}" ]; then
    _q38x16_sb="$(head -n 1 "${_q38x16_vb}" 2>/dev/null | sed -n 's|^#![[:space:]]*||p')"
    case "${_q38x16_sb}" in */env\ *) _q38x16_py="${_q38x16_sb##* }" ;; /*) _q38x16_py="${_q38x16_sb%% *}" ;; esac
    case "${_q38x16_py##*/}" in python*) ;; *) _q38x16_py=python3 ;; esac
  fi
  _q38x16_ver="$("${_q38x16_py}" /tmp/q38x16_wheel.py verify "${RAIDEN_WHL}" 2>&1)" \
    || q38x_fatal "Q38X_WHEEL leg=${_q38x16_leg} the installed wheel does not verify under ${_q38x16_py} (the vllm interpreter): ${_q38x16_ver}"
  echo "RAIDEN_WHEEL: installed ${_q38x16_whl} as $(echo "${_q38x16_ver}" | sed -n 's/^dist=\([^ ]*\) version=\([^ ]*\) .*/\1 \2/p')"
  q38x_say "Q38X_WHEEL installed ${_q38x16_whl} ${_q38x16_sha} leg=${_q38x16_leg} ${_q38x16_ver}"
  q38x_sync
  # ---- q38x v1.6 wheel END
|||, '@WHL@', q38x16Whl);
local q38x16Anchor = "cat > /tmp/exos_gcs_cp.py <<'PYGCS'\n";
local q38x16Job(job) =
  local cs = job.template.spec.template.spec.containers;
  job + { template+: { spec+: { template+: { spec+: {
    containers: [if c.name == 'vllm-pd' then c + { args: [exosReplaceN(c.args[0], q38x16Anchor, q38x16Blk + q38x16Anchor, 1)] } else c for c in cs],
  } } } } };
local q38x16NLeg = std.length([c for j in exosJobSetV15.spec.replicatedJobs for c in j.template.spec.template.spec.containers if c.name == 'vllm-pd']);
assert q38x16Whl == '' || q38x16NLeg >= 2 : 'exos v1.6: expected a vllm-pd container per leg (>= 2), found %d' % q38x16NLeg;
local exosJobSetV16 = if q38x16Whl == '' then exosJobSetV15 else exosJobSetV15 + { spec+: { replicatedJobs: [q38x16Job(j) for j in super.replicatedJobs] } };

exosJobSetV16 + {
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
