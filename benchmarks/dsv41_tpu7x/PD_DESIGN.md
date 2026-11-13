<!-- Source: P/D research subagent b8ef188e, 2026-10-20 (read-only file review). Path keys:
     D = repos/vtt-decode/src/vllm_torchtpu
     P = repos/vtt-prefill/src/vllm_torchtpu
     VL = repos/vllm-decode
     VT = main vllm-torchtpu checkout (../vllm-torchtpu) -->
# DSv4.1-Flash P/D disaggregation on TPU7x: design

## TL;DR

1. **`TPURaidenConnector` "Stage 3" is the only TPU connector that can move DSv4-style multi-group KV.**
   * Transport: host network (DCN), through the tpu_sync Raiden C++ engine. Each engine has a controller and a reshard-store RPC.
   * There is a working DSv4-Flash (V4.0) recipe: DP8 prefill → DP8 decode, TP=1 only. It comes with a JobSet and a correctness gate.
   * The ZMQ-shm `TPUConnector` handles a single KV group only, so it cannot do DSv4.
2. **V4.1 does not work with the Raiden DSv4 manifest as written.** Reasons:
   * The manifest classifies only ratio-4 layers as CSA. V4.1's ratio-1/2 layers become "HCA" and hard-fail on the (nope, rope) pair.
   * TP must be 1.
   * The prefill branch's allocator differs from the decode branch's: it uses compact SWA arrays and aliases state onto the CSA arrays.
   * A PURE_20L prefill has a different layer and group set, so the `.g{group}` tags won't line up.
   * The connector has no V4.1- or CED-specific handling.
3. **SWA bounded replay** exists as an in-model planner: `ced.py plan_decoder_replay` runs the decoder stage only on tokens at positions ≥ prompt_len − 128.
   * It has no P/D integration.
   * With PURE_20L, the prefill's dummy-logit token is discarded by the proxy, which only forwards `kv_transfer_params`. The decode side computes the first token.
4. **Transfer size** is ≈0.16–0.32 GB per 100K-token request, or ≈0.3–0.6 GB/s per node at 197K tok/s/node. Bandwidth is not the bottleneck.
5. **Recommendation.**
   * **Phase 1:** use one tree with an identical KV layout on both nodes.
     * Prefill: DP8/TP1 with `CED=1, PURE_20L=0`, i.e. a full 40-layer prefill with in-engine 128-token replay.
     * Decode: DP8.
     * Connector: Raiden Stage-3 with DSv4 admission, plus a small V4.1 manifest patch.
     * Turn DSpark and prefix caching off for the first run.
   * **Phase 2:** move to the target 4×(TP2) PURE_20L prefill, which needs gaps G5–G9.

## 1. KV connectors (`distributed/kv_transfer/`)

| Connector | Where | Transport | Cross-node | DSv4 multi-group |
|---|---|---|---|---|
| `TPUConnector` (ZMQ-shm) | D/.../tpu_connector.py:730-811, worker 1311-1433; zmq_shm_base.py | Per-host SharedMemory + ZMQ ROUTER/DEALER over TCP | yes | **no**: asserts a single KV group (902-905) |
| `TPURaidenConnector` | tpu_connector.py:1013-1017; scheduler 1436-2322; worker 2325+ | tpu_sync Raiden C++ (D2H, H2H, H2D) plus a ReshardClient RPC to the source controller. Stage 3 is selected by `TPU_KV_RESHARD_TRANSPORT=raiden` (569-570). | yes (DCN, host-staged) | **yes, DSv4-Flash only**, via `TPU_RAIDEN_DSV4_ADMISSION=1` (670-673, 2888-3044) |
| `TPUMultiConnector` | tpu_multi_connector.py | MultiConnector wrapper (disagg plus offload) | – | – |

**Stage-3 flow.**

*Prefill side:*
* It truncates the prompt by 1 token whenever there are windowed groups (1058-1086, 1492-1513).
* `request_finished` publishes `num_tokens = N-1` (2045-2063).
* It skips the send, so decode prefills locally, in two cases:
  * a windowed page was already recycled (2150-2175)
  * the request is larger than `TPU_RAIDEN_MAX_TRANSFER_TOKENS` (2081-2095)
* It returns `kv_transfer_params` (2274-2285).
* The DSv4 path requires `src_parallelism == 1` (2197-2202).

*Decode side:*
* `get_num_new_matched_tokens` returns `num_tokens − computed` (1610-1620).
* It needs the complete destination page set for every group (1724-1806).
* Prefix-aware load is disabled for multi-group plans (1816-1823), so prefix caching must be off.

*DSv4 manifest and fingerprint:*
* `raiden/pool_manifest.py:1584-1692` builds one pool per (layer, array) with tags `dsv4.<role>.g<group>`.
* The layout fingerprint lives in `raiden/layout_fingerprint.py:302-323`.

**Launcher:** `examples/disagg/launch_dsv4_disagg_multihost.sh`. It is in the decode worktree and identical to VT.

*kv-transfer-config:*
* producer: `{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer","kv_port":14579}`
* consumer: the same with `kv_role:"kv_consumer"`

*Common env:*
* `TPU_USE_RAIDEN_KV_CACHE_MANAGER=1`
* `TPU_RAIDEN_DSV4_ADMISSION=1`
* `TPU_KV_RESHARD_TRANSPORT=raiden`
* `TPU_RAIDEN_TRANSFER_PARALLELISM=1`
* `TPU_ENABLE_D2H_TRANSFER=1`
* `TPU_HMA_D2H_COPY_IMPL=device_put`
* `TPU_HMA_{MIN,MAX}_DONE_RECVING_BATCH=1`
* `TPU_HMA_MAX_INFLIGHT_D2H_SENDS=1`
* `TPU_HMA_MAX_INFLIGHT_PULLS=8`
* `TPU_MAX_HOST_KV_BUFFER_SIZE=64`
* `TPU_P2P_WAIT_PULL_TIMEOUT=300`
* `TPU_KV_SHM_POOL_GB=8`
* `TPU_SIDE_CHANNEL_PORT=9600`

*Per-role env:*
* `TPU_RAIDEN_JOB_NAME=prefill|decode`
* `TPU_RAIDEN_ENGINE_ID`
* `TPU_RAIDEN_RESHARD_IMPL=store`
* `TPU_RAIDEN_ADVERTISE_HOST=<pod IP>`
* `TPU_RAIDEN_RESHARD_PORT_BASE=27000|28000` (dispatch is +100)
* `TPU_KV_TRANSFER_PORT=9100|9200`
* `TORCH_TPU_SLICEBUILDER_ADDRESSES`
* `TORCH_TPU_DP_MASTER_PORT`
* `TPU_KV_TRANSFER_NAMESPACE`

*Server flags:*
* `--enable-expert-parallel --no-disable-hybrid-kv-cache-manager --no-enable-prefix-caching --no-async-scheduling`
* `--data-parallel-size 8 --data-parallel-size-local 8 --tensor-parallel-size 1`

**Proxy:** `examples/disagg/toy_proxy_server.py`.
1. It renders the request to token ids.
2. It sends it to the prefiller with `max_tokens=1`.
3. It copies `kv_transfer_params` into the decode request and streams from the decoder.
4. It round-robins across multiple prefiller and decoder host:port pairs, so 4 prefill servers can sit behind one proxy.

Health check is at `/healthcheck`. Decode DP routing is done by the decode API server's internal load balancer.

**K8s:** `VT/.buildkite/kubernetes/manifests/workloads/dsv4-pd-disagg.yaml`.
* It is a JobSet with a `decode` job and a `head` job; the head runs prefill and the probes.
* The gate script is `VT/scripts/vllm/integration/run_dsv4_pd_correctness.sh`. It covers quick_probe, correctness_default/large/concurrent and `check_dsv4_stage3_log.py`. The last one catches the silent "decode prefilled locally" fallback.

## 2. V4.1 KV cache against the connector: what breaks

1. **Ratio classification.** `_dsv4_kv_tag` treats a layer as CSA only if `tokens_per_state==4` (pool_manifest.py:1446-1455). V4.1 ratios are 1 and 2, so its CSA layers become `dsv4.hca` and hit "must bind a single array" (1546-1551). The row checks (1564-1581) also assume ratio 4.
2. **Indexer.** V4.1 uses the seq-along-lane layout `uint8[nb,33,4,T]`. The manifest requires width % 128 (1652-1655) and expects `tokens//4` rows.
3. **Topology.** DSv4 admission accepts only TP=1 and PCP=1 (tpu_connector.py:3031-3044), and producer parallelism must be 1 (2197-2202, 3146-3154).
4. **Allocator mismatch between branches.**
   * Prefill uses compact SWA overflow arrays plus a BlockPool monkey-patch, and aliases state onto CSA (P/runner/kv_cache_deepseek_v41_flash.py:24-70, 271-317).
   * Decode uses nb-sized overlays and separate state arrays.
   * The result is a "pools disagree on num_blocks" error and a fingerprint mismatch. The fingerprint also needs one shape per tag (layout_fingerprint.py:364-368).
5. **Layer and group sets.** PURE_20L builds 21 layers, while decode builds 40 layers plus the drafter. The tags carry the group index.
6. **No V4.1 or CED awareness** in tpu_connector.py, pool_manifest.py or layout_fingerprint.py.

**TP=2 → TP=1.** MLA/CSA KV is a single latent head and is replicated across TP: the specs are `P()` and `fused_wqa_wkv` has `disable_tp=True`. So TP rank 0 holds the full KV.

## 3. SWA bounded replay

`models/vllm/deepseek_v41/ced.py:38-199` `plan_decoder_replay`.

Forward pass (P/model.py:1685-1800):
1. The encoder runs layers 0-19 on all tokens.
2. It writes the layer-20 global KV (1079-1089).
3. The tail is gathered into the DecoderStage (layers 20-39 with `ced_decoder=True`).

Modes:
* `CED=1, PURE_20L=0`: encoder plus bounded decoder in one engine.
* `HYBRID_CED_DECODE=1`: also builds a FullStage for pure-decode steps.
* `PURE_20L=1`: the decoder is skipped and zeros are returned as the hidden state, which produces dummy logits.

For CED P/D, the prefill side must stop the transfer at T ≤ N−128. Decode then recomputes the tail with CED=1 and emits token 1. None of this is implemented yet.

## 4. KV size per 100K request (fp8)

* CSA record: NoPE 512 B + RoPE 128 B = 640 B per entry.
* Indexer: 132 B per entry.
* kv-source layers are [2, 8, 14, 20], so main KV is 4·(772)/r B per token: r=1 gives 309 MB per request, r=2 gives 154 MB.
* SWA window and state add a few MB.
* Total is ≈0.16–0.32 GB per request.

At 197K tok/s/node that is ≈0.3–0.6 GB/s. Per-request transfer takes ≈60–250 ms against ≈2 s of prefill, and it overlaps asynchronously. Size `TPU_KV_SHM_POOL_GB` to ~16.

Still to do: get the real `compress_ratios` from the checkpoint config.json.

## 5. Plan

**Phase 0.** Decode-local gates must pass first. This is the `decodefix` thread.

**Phase 1** (one tree on both nodes, identical layout):
* Prefill: DP8/TP1/EP8 with `CED=1, PURE_20L=0`, as producer.
* Decode: DP8 with `ENABLE_DECODE_BENCH_CONNECTOR` unset, as consumer, with `--no-enable-prefix-caching --no-async-scheduling` and **no DSpark** on the first run.

This should be token-exact against a co-located CED=1 server, because P stops at N−1 and the decoder SWA for the last 128 tokens is computed on P and transferred.

Gaps to fill:
* **G1:** the V4.1 manifest. Ratio 1/2 maps to CSA (nope, rope) with rows T and T/4; handle the V4.1 indexer layout; relax width % 128.
* **G2:** every SWA host array must have one shape, or the tag must be split by host shape.
* **G3:** identical aux partitions and page sizes on both peers, using the same vllm patches.
* **G4:** launcher env for V4.1 (model path, max-model-len 131072, the V4.1 env block).

**Phase 2** (target 4×TP2 PURE_20L → DP8):
* **G5:** a TP=2 producer where rank 0 sends.
* **G6:** a layer-subset producer, so group tags don't depend on group index, and destination-only pools are allowed.
* **G7:** CED handoff extent T = align_down(N−128). Decode then runs `CED=1 HYBRID_CED_DECODE=1` with a 128–256-token tail compile bucket.
* **G8:** unify the prefill allocator with decode's.
* **G9:** unique ports and engine IDs for 4 servers on one host, behind `toy_proxy_server.py --prefiller-ports 8400 8401 8402 8403`.

**Correctness tests (both phases).**
* Greedy first-64-token equality against a co-located reference.
* Gates through the proxy: 2+2 ×16 and USA.
* The four smoke suites.
* `check_dsv4_stage3_log.py` to prove the transfer actually happened.
* NIAH at 8K/32K/64K/100K × needle depths 0–100%. Phase 2 adds needles inside the last 128 tokens and at the T boundary.

## 6. Findings from the Phase-1 bring-up (pd thread, 2026-10-20)

* **Pods must use `hostNetwork: true`.** Raiden's transport server (`RaidenManagerBase::InitTransportServer` in `tpu_sync/.../_tpu_raiden_torch_*.so`) maps each data NIC to its NUMA node through `/sys/class/net/<nic>/device/numa_node`. A pod-network veth has no `device/`, the per-NUMA lookup misses, and every worker aborts with `absl raw_hash_map<>::at` right after `InitTransportServer: target_numa=…`, during KV admission. (`TPU_RAIDEN_DATA_NICS` exists as an override but was not tried.) `scripts/pd/run_pd.py` uses hostNetwork, and the peers advertise node IPs.
* **`libtpu >= 0.0.47` (`0.0.48`) is required for 8-worker sub-chip `DmaMap` on `tpu7x`.** Under `libtpu 0.0.44`, `tpu_sync::TorchKVCacheManager` (inside `KVCacheManager()`) calls `xla::PjRtCApiClient::DmaMap()` → `tpu::System::DmaMap()` (`libtpu.so`), which aborts with `std::__throw_out_of_range("raw_hash_map::at")` whenever 8 sub-chip workers open 1 TensorCore each (`--deepsea_hal_included_devs`). With `libtpu 0.0.48` (and our patched `support/mock_metadata.py` serving all `tpu-env` attributes), all 8 workers pass `DmaMap`, `KVCacheManager`, `register_pools` (55 pools, 26 storages), and `ReshardClient.register_work_unit` cleanly (`scripts/pd/in_pod_setup.sh`).
* **Checkpoint deletion after weight load prevents host RAM OOM.** Because `/dev/shm` is tmpfs backed by host RAM, keeping 476 GiB of `.safetensors` in `/dev/shm` while `KVCacheManager` allocates `8 × 16 GB = 128 GB` of pinned shared memory triggers the Linux OOM-killer (`in_pod_serve.sh` deletes `*.safetensors` after all 8 workers log `Memory statistics |`, preserving `tokenizer.json` and `config.json`).
* **Real V4.1 layout** (decode tree, no DSpark): 26 arrays for 51 layers: mla=8 (4 CSA, 4 indexers), swa=40 in 3 groups sized [14, 13, 13], state=3, block size 1024, `num_blocks=5218` at gpu-mem 0.85. Group page sizes are [128, 128, 128, 1024, 64].
  * CSA: T=512 for ratio 2 (layers 2/8/14) and T=1024 for ratio 1 (layer 20).
  * Indexer: `uint8[nb,33,4,T]`, where T is 512 or 1024, so width % 128 holds.
  * SWA: overlays the 3 T=512 NoPE hosts, plus overflow arrays `int32[nb,256,128]`.
* **CPU-first UE8M0 unit-scale initialization (`c0f38024b`):** In `kv_cache_deepseek_v41_flash._create_cache`, running 4D slice assignment (`cache[(slice(1, main_end), 32, 0, :)] = 0x7F`) on TPU compiled an XLA scatter whose output buffer had `get_device_layout_if_materialized(cache) == None` and padded `num_blocks` (`5218 -> 5224`). Initializing on CPU before `.to(runner.device)` preserves the materialized device layout (`([3, 2, 1, 0], [[4, 128], [4, 1]], 0)`) and exact `num_blocks=5218` across all 55 pools (`26` storages, `13` tags).
* **XLA scheduling annotation flag for `2048`-token prefill under `libtpu 0.0.48` (`b8b0468ea`):** Added `--xla_tpu_scheduling_annotation_deannotate_unsupported_groups=true` to `LIBTPU_INIT_ARGS` in `launch_dsv41_disagg_role.sh` for `StreamIdxTC-m-bq_128-bkvp_16-sal-c4`.
* `support/decode/sitecustomize.py` force-sets `VLLM_DSV41_CED=0` and `ENABLE_DECODE_BENCH_CONNECTOR=1` in every process. P/D uses `scripts/pd/sitecustomize_pd.py`, which has no forced block.
* **Phase-1 1P1D End-to-End Validation (`d5129e7c6`, `hard_gates_passed=True` on both `CED=1` and `CED=0`):**
  * **`CED=1` Prefill → `CED=0` Decode (`runs/pd/20261020-165851_1p1d_ced1_final_pass_a`):** `2+2` **16/16**, `USA` **50/50 states coherent**, greedy 64-token equality + needle `739204` retrieval **5/5 `equal=True` (`first_div=None`) across `64, 700, 1500, 5000, 20000` tokens**, `STAGE3_CHECK_OK 1` (`153/153` sends completed, `0` skipped, window trim `157→20` pages verified), **`bench_pd` (`n=16, c=8, 8K in, 128 out`): TTFT mean `1.445s` (`p50=1.298s`) vs `3.564s` decode-local (`2.47×` faster), TPOT `24.81 ms` vs `43.15 ms` (`-42.5%`), output throughput `200.94 tok/s` vs `109.11 tok/s` (`+84.2%`)**.
  * **`CED=0` Prefill → `CED=0` Decode (`runs/pd/20261020-165853_1p1d_ced0_final_pass_b`):** `2+2` **16/16**, `USA` **50/50 states coherent**, greedy 64-token equality + needle `739204` retrieval **5/5 `equal=True` (`first_div=None`) across `64, 700, 1500, 5000, 20000` tokens**, `STAGE3_CHECK_OK 1` (`171/171` sends completed, `0` skipped, window trim `157→20` pages verified), **`bench_pd`: TTFT mean `2.525s` (`p50=2.089s`), TPOT `25.64 ms` vs `37.89 ms` (`-32.3%`), output throughput `157.13 tok/s` vs `130.57 tok/s` (`+20.3%`)**.
