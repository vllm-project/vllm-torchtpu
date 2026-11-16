# DeepSeek-V4.1-Flash 1P1D（vllm-torchtpu，Stage-3 Raiden）复现手册

更新：2026-11-12。称呼约定：**GCP** = 我们的部署；**Baseline** = 客户内部数据（对比基准，当前用 2026-11-02 版本）。

## 最新结果一览

2026-11-11 在主机 hl-2lxt / hl-6g41 上复测，代码是合并了 epic/dsv41 ef2599d3d 之后的 `cszhz/dsv41-1p1d`（4082db4a2），FP8 Index-K。
本轮只测了 8P TTFT、TPOT、TPS，1P / 4P 没有重测。详细表格见 5.1；11-05/06 的结果（合并前）见 5.2–5.6，旧主机见附录 B。

| 指标 | GCP 11-11（合并后） | GCP 11-05/06（合并前） | Baseline（11-02） |
|---|---|---|---|
| TTFT 8P（1k+1k … 32k+768k） | 0.37 – 45.2 s | 0.39 – 49.0 s | — |
| TPOT（bs 1–256，cache 1k–768k，CLI） | 3.9 – 9.0 ms | 4.7 – 44.5 ms | 2.2 – 8.3 ms |
| P 的 prefill 吞吐（PD 里的 P，8 并发） | 16.5 – 19.1 万 tok/s | 16.4 – 17.2 万 tok/s | — |

- 合并后 TPOT 基本不随 cache 长度增长（bs64 在 1k–768k 都是 5.2–5.7 ms），是 Baseline 的约 1.0–2.2 倍（合并前 1.6–7.8 倍）。
- TTFT 8P 11 个场景里 10 个更快（最多快 17%），168k+32k 慢 2%。

---

## 1. 部署与代码

### 1.1 拓扑

| 角色 | 主机 | 地址 | 说明 |
|---|---|---|---|
| P（prefill） | zzl-tpu7x-slice-mig-hl-2lxt | 10.128.0.100:8100 | 一台 4 芯片 v7x，DP8 / TP1 / EP8 |
| D（decode） | zzl-tpu7x-slice-mig-hl-6g41 | 10.128.0.156:8200 | 同上 |
| proxy | 与 P 同机 | 10.128.0.100:8000 | 客户端只打这个端口 |

- 上一套主机（5.2–5.5 的结果）：P = tpuv7-mig-g45q（10.128.0.145），D = tpuv7-mig-pqs0（10.128.0.28）。
- 原部署（附录 B 的结果）：P = zzl-tpu7x-slice-mig-2z6z（10.128.0.175），D = …-clsk（10.128.0.177），
  另有编译 / 单测机 …-sjd1（10.128.0.176，**不要在 P/D 上跑 pytest，会占 TPU**）。脚本里的默认地址仍是原部署，新主机用环境变量覆盖（第 3 节）。
- 每个 DP rank 是一颗 v7x 芯片的一个 TensorCore（每颗芯片 2 个），device `2k`、`2k+1` 在同一颗芯片上；DP rank 和 device 的对应在每次启动时分配，不一定按顺序。
  EP 下 8 个 DP rank 同步（lockstep）前进，空闲 rank 每步跑 dummy forward。
- 模型：`gs://zzl-bucket/models/DeepSeek-V4.1-Flash`（runai streamer 直接从 GCS 加载）。
- 共享盘 `/zzlfs` 所有机器都挂载，下文路径都在共享盘上。

### 1.2 代码版本

- vllm-torchtpu：`https://github.com/cszhz/vllm-torchtpu`，分支 **`cszhz/dsv41-1p1d`**（本地 `/zzlfs/baseline/src/vllm-torchtpu-cszhz`），
  2026-11-11 合并了上游 `vllm-project/vllm-torchtpu` 的 `epic/dsv41`（ef2599d3d，合并提交 f6693c749），之前基于 6a1ebddb7。
  **所有改动只在 cszhz fork 上，不要推上游**（本地 clone 的 upstream push URL 已设成 `DISABLED-do-not-push-to-vllm-project`）。提交列表见附录 A。
- P、D 用同一份代码，以 editable 方式装进 venv。
- tpu-sync（KV 传输）：
  - **部署实际用的版本**：PyPI nightly `tpu-sync-torch==0.0.1.dev20261031081414`（对应 main 1523b07d，已包含 `GetBlockChunks` O(n²) 修复，
    原 PR google/tpu-sync#1382）+ "允许各 storage 块数不同"的修改 = cszhz/tpu-sync 分支 **`per-layer-major-dim-on-1031`**（f2fa4536），
    本地 `/zzlfs/baseline/src/tpu-sync-1031`。编法见 2.2。
  - 为什么必须自己编：合并后的 epic/dsv41 把 SWA 放进了窗口大小的私有池（每个 rank 241 页，主池约 52 万页），
    原版 tpu-sync 要求所有 storage 块数相同，启动时报 `Layer N has major dimension ... but layer 0 has ...`；
    另外 host 镜像按最大的 storage 分配会 OOM、传输时 staging 槽位越界（`Chunk out of bounds`）、块 id 越界检查不分组（`local block id N is out of range for pool M`）。
  - 上游 PR：**google/tpu-sync#1527**（cszhz/tpu-sync 分支 `pool-heterogeneous-block-counts`，基于 main d2d9d37b）。
    main 在 47e9f200 已经修了 staging 和 host 镜像大小两处，PR 只含另外 3 处。

---

## 2. 环境搭建

### 2.1 venv（`/zzlfs/baseline/.venv-epic`）

epic/dsv41 需要 jax 0.11.2（旧 venv 缺 `pltpu.ACC`），所以单独建一个新 venv。依赖清单见 `scripts/dsv41_pd/requirements-epic.txt`
（torch 2.13.0、torch-tpu 0.2.0、jax/jaxlib 0.11.2、libtpu 0.0.50、vLLM v0.31.1rc0 = e37e51dd、tpu-sync-torch dev20261031、runai-model-streamer[gcs]、xxhash）。

```bash
UV=/zzlfs/.tools/pyenv/bin/uv
$UV venv -q -p /usr/bin/python3.12 /zzlfs/baseline/.venv-epic
# vLLM 用 empty 目标构建（TPU 插件由 vllm_torchtpu 提供）；torch 走 CPU wheel 源
(cd /tmp && VLLM_TARGET_DEVICE=empty CC=gcc CXX=g++ $UV pip install --python /zzlfs/baseline/.venv-epic/bin/python \
    --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /zzlfs/baseline/src/vllm-torchtpu-cszhz/scripts/dsv41_pd/requirements-epic.txt)
$UV pip install --python /zzlfs/baseline/.venv-epic/bin/python orjson
# vllm_torchtpu 以 editable 方式指向 cszhz 这份代码
$UV pip install --python /zzlfs/baseline/.venv-epic/bin/python --no-deps -e /zzlfs/baseline/src/vllm-torchtpu-cszhz
```

- torch-tpu 的私有源（registry）返回 403，所以改用 PyPI 上的 torch-tpu 0.2.0 正式版（pyproject 里写的是 0.2.0rc1）。
- 这个 venv 建在非 PIE 的 `/usr/bin/python3.12` 上，所以 `TPU_RECLAIM_COMPILE_MEMORY` 只能关（见第 7 节）。
- 旧的 `/zzlfs/baseline/.venv` 留给旧 PD 部署；benchmark 客户端脚本仍用 `.venv/bin/python` 跑（只需要 httpx）。

### 2.2 编译并替换 tpu-sync 的 .so

venv 里先装 PyPI 的 `tpu-sync-torch==0.0.1.dev20261031081414`（Python 层用它的），再把 C++ 扩展换成自己编的。
编译机需要 clang-18（P 那台 hl-2lxt 上就能编，约 10 分钟；不要和正在跑的服务抢 CPU 时编）。

```bash
# 取代码（共享盘上已有的话跳过）
git clone -b per-layer-major-dim-on-1031 https://github.com/cszhz/tpu-sync.git /zzlfs/baseline/src/tpu-sync-1031
cat /zzlfs/baseline/src/build-tpusync-1031.sh      # 内容如下
#   cd /zzlfs/baseline/src/tpu-sync-1031
#   SP=/zzlfs/baseline/.venv-epic/lib/python3.12/site-packages
#   export CC=/usr/bin/clang-18 CXX=/usr/bin/clang++-18
#   export PATH=/zzlfs/baseline/tools/nopip:$PATH TORCH_SOURCE=$SP TORCH_TPU_SOURCE=$SP
#   bash build.sh torch
bash /zzlfs/baseline/src/build-tpusync-1031.sh

# 产物改 NEEDED 后放进 venv（原 .so 备份为 .pypi-1031）
D=/zzlfs/baseline/tools/tpu_sync_epic/_tpu_raiden_torch_2_13_0-1031-perlayer.so
cp /zzlfs/baseline/src/tpu-sync-1031/tpu_sync/frameworks/torch/_tpu_raiden_torch.so $D
patchelf --add-needed libpywrap_2_13_0_common.so $D
patchelf --remove-needed libpywrap_torch_tpu_common.so $D
T=/zzlfs/baseline/.venv-epic/lib/python3.12/site-packages/tpu_sync/frameworks/torch
cp $T/_tpu_raiden_torch_2_13_0.so $T/_tpu_raiden_torch_2_13_0.so.pypi-1031   # 只做一次
cp $D $T/_tpu_raiden_torch_2_13_0.so
# 检查能加载（直接 import 模块名会失败，这是正常的）
/zzlfs/baseline/.venv-epic/bin/python -c "from tpu_sync.api.torch import kv_cache_manager as k; k._torch_impl(); print('ok')"
```

- `/zzlfs/baseline/tools/nopip/python3` 是一个包装，用来阻止 build.sh 调用 pip 去改 venv。
- 不替换 .so 时，合并后的代码启动阶段就会失败（1.2 里的几个报错）。
- 跑 tpu-sync 的 C++ 单测：把 `build.sh` 的 `build` 换成 `test`、目标换成测试目标即可（参考 `/zzlfs/baseline/src/tpusync-test.sh`），例如
  `//tpu_sync/kv_cache:reshard_receive_session_test`。

### 2.3 第三方包运行时补丁（必做，可重复执行）

```bash
/zzlfs/baseline/.venv-epic/bin/python /zzlfs/baseline/src/vllm-torchtpu-cszhz/scripts/dsv41_pd/apply_venv_patches.py
```

| 节 | 改动 | 原因 / 效果 |
|---|---|---|
| §3 | runai_model_streamer_gcs：`bucket()` 代替 `get_bucket()` | 不需要 `storage.buckets.get` 权限 |
| §4 | vLLM 调度器：connector 的 `skips_prefix_replay(request)` 为真时，远端加载不取整、不回放 | D 在 `--swa-bounded-replay` 下也能保留前缀缓存（Stage-3 要求完整页集，原逻辑会让 decode 引擎退出） |
| §5 | vLLM detokenizer：`DecodeStream` 只用 prompt 末尾 32 个 token 预热（如果末尾解码出 `�` 则退回整段） | 原来 557k token 时要预热 0.5s，P、D 各一次；512k+32k TTFT 4.27→3.58s |
| §6 | DP busy loop：`VLLM_DP_WAVE_START_DELAY_MS`，新 wave 开始时先等一会再收一次请求 | 让几乎同时到达、分散在不同 rank 的请求赶上同一个 lockstep 步（4P 对齐） |
| §7 | starlette `Request.json` 优先用 orjson | 10^6 个 token id 的请求体解析更快 |

原文件备份在 `/zzlfs/baseline/tools/*-epic.py.orig`。

### 2.4 每台机器的准备（换主机时逐项检查）

- tokenizer：把模型目录里的 `tokenizer.json`、`tokenizer_config.json`、`config.json`、`chat_template.jinja`、`encoding/`
  拷到**每台机器**的 `/tmp/dsv41-tokenizer`（P、D、proxy 都用这个路径）。
- 主机名短于 64 字符：`sudo hostname $(hostname -s)`（GCE 主机名正好 64 字符时 gloo 会报错，重启或 guest agent 会改回去）。
- 根分区小（新主机只有 8.7 GB）时：`TMPDIR`、`VLLM_CACHE_ROOT` 放到 `/dev/shm`（第 3 节的命令已包含）。
- `/dev/shm` 里不用的缓存会直接占主机内存，启动前清掉（`stop_vllm_local.sh` 会清 `psm_*` 和超过 30G 的 `torch_tpu_cache`）。
  `VLLM_CACHE_ROOT/torch_compile_cache/<hash>/` 每换一种配置（Index-K dtype、compile_sizes…）多一份 23–36 GB，不会自动清，
  换配置前把不再用的 hash 目录删掉（对应关系可以在启动日志里 grep `torch_compile_cache/<hash>` 查到）。内存账见第 7 节。
- 各条原因见第 7 节。

---

## 3. 启动与停止

脚本都在 `scripts/dsv41_pd/`（部署时用的是 `/zzlfs/baseline/scripts/` 下的同名副本，内容一致）。下面是新主机的命令；
不设 `P_HOST` / `D_HOST` / `P_ADDR` / `D_ADDR` / `PREFILL_HOST` / `DECODE_HOST` 时就是原部署的默认值。

```bash
cd /zzlfs/baseline/scripts
export P_HOST=10.128.0.100 D_HOST=10.128.0.156 TMPDIR=/dev/shm/epic-tmp VLLM_CACHE_ROOT=/dev/shm/epic-vllm-cache
# P：默认配置（async、compile [128,256,2048]、4 个 API server、起步等待 40ms）；5.2–5.4 的 P 关 async（ASYNC_SCHED=0）
ROLES=prefill bash restart_epic.sh
# D：async + MAX_NUM_SEQS=32 + compile [16,48,96,192,2048]（起步等待默认 20ms）。不设时脚本默认 16 / [16,96,128,2048]，TPOT 和表不可比（4.2）
ROLES=decode MAX_NUM_SEQS=32 DECODE_COMPILE_SIZES='[16,48,96,192,2048]' bash restart_epic.sh
# 等两边 /health 返回 200，并检查日志里有没有致命错误；冷启动约 30–45 分钟（有编译缓存时约 12–15 分钟）
P_ADDR=10.128.0.100 D_ADDR=10.128.0.156 bash wait_ready.sh
# proxy（在 P 那台上）
ssh -n -f 10.128.0.100 'PREFILL_HOST=10.128.0.100 DECODE_HOST=10.128.0.156 setsid nohup bash /zzlfs/baseline/scripts/start_proxy.sh \
    > /zzlfs/baseline/logs/proxy-$(date +%m%d-%H%M).log 2>&1 < /dev/null &'
# 测试脚本用的地址
export PROXY_URL=http://10.128.0.100:8000 PREFILL_METRICS_URL=http://10.128.0.100:8100/metrics D_METRICS_URL=http://10.128.0.156:8200/metrics
```

- Index-K 默认 FP8（`start_1p1d_epic.sh` 里写死 `--attention-config '{"indexer_kv_dtype":"fp8"}'`），合并后 FP8 已能正常传输。
  需要 BF16 时用 `EXTRA_ARGS='--attention-config {"indexer_kv_dtype":"bf16"}'` 覆盖（追加在最后，后者生效）。

- `restart_epic.sh` 通过 ssh 在对应主机上先执行 `stop_vllm_local.sh`（连残留 worker 一起停，等 `/dev/vfio` 释放；根分区超过 75% 时清缓存），
  再启动 `start_1p1d_epic.sh <role>`，并把 `TMPDIR`、`VLLM_CACHE_ROOT` 和下面的可调变量透传过去。
  日志写到 `/zzlfs/baseline/logs/{prefill,decode}-epic-<MMDD-HHMM>.log`，时间戳记录在 `/tmp/epic_T`。
  P、D 分两次启动且跨了分钟时，`wait_ready.sh` 会找不到另一边的日志：用 `ln -sf` 把那边的日志链接成同名。
- `start_1p1d_epic.sh` 的关键参数：`--data-parallel-size 8 --enable-expert-parallel --kv-cache-dtype fp8 --block-size 16
  --enable-prefix-caching --swa-bounded-replay --engram-config '{"cpu_offload":true}'
  --additional-config '{"deepseek_v41":{"execution":"ced","moe":"fused_ep_w4a8"}}'
  --speculative-config dspark K=5 --max-num-batched-tokens 2048`，KV connector 为 `TPURaidenConnector`。
- 可调环境变量：`ASYNC_SCHED`（默认 1 = `--async-scheduling`，P、D 都开；0 = 关）、`DVFS_P_STATE`（默认 7）、`MAX_NUM_SEQS`、`PREFILL_COMPILE_SIZES`、`DECODE_COMPILE_SIZES`、`API_SERVERS`、
  `VLLM_DP_WAVE_START_DELAY_MS`、`TP_SIZE`/`DP_SIZE`、`NO_KV_TRANSFER`、`KV_BYTES`、`EXTRA_ARGS`。
  诊断用：`VLLM_LOGGING_PREFIX='%(created).3f '`（日志带毫秒级时间戳，`pd_timeline.py` 依赖它）、`TPU_DUMMY_STEP_TIMING=1`、`TPU_STAGE3_PROFILE=1`。
- proxy 环境变量：`PROXY_TOKENIZER`（默认 `/tmp/dsv41-tokenizer`，本地分词）、`PROXY_TOKENIZER_PROCS`（默认 8）、
  `PROXY_ALIGN_MS`（默认 0=关，见 6.2）、`PROXY_RENDER_CACHE_MB`（默认 0=关，见 6.2）。

**冒烟测试**

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
/zzlfs/baseline/.venv-epic/bin/python -I /zzlfs/baseline/scripts/pd_logprob_check.py --proxy $PROXY_URL --direct http://10.128.0.156:8200 \
    --tokens 3000 20000 100000 --offset 1000000      # PD 与直接打 D 的首 token logprob 对比（含 D 前缀命中 pd_hit）
/zzlfs/baseline/.venv-epic/bin/python -I /zzlfs/baseline/scripts/local_hit_control.py --direct http://10.128.0.156:8200 --offset 3000000
                                                   # 对照：不走 PD，D 本地前缀命中 vs 另一个 rank 从头算（噪声下限）
```

- logprob 检查的语料约 1489 万字符，`--offset` 加上各档长度不能超出（超出时脚本会断言失败；旧版本会静默只发问题，所有用例都"完全一致"）。
- 2026-11-11（FP8 Index-K）：pd 0.125–0.5、pd_hit 0.5–0.625，对照组 local_hit 0.125–0.75，同一量级，PD 没有引入额外误差。

**停止**：`bash /zzlfs/baseline/scripts/stop_vllm_local.sh`（在 P、D 各执行一次）；proxy 按 PID 停：`kill $(pgrep -f 'oy_proxy_serv[e]r')`。
调用 `stop_vllm_local.sh` 的命令行里不要出现 `VLLM::`、`.venv/bin/vllm serve` 这些字样，否则会连调用者的 shell 一起停掉。

---

## 4. 测试方法

客户端统一打 proxy（:8000），用 `/zzlfs/baseline/.venv/bin/python -I` 运行。

### 4.1 Prefill / TTFT 表（cached + new）

```bash
OUT=/zzlfs/baseline/results/table-$(date +%m%d-%H%M).jsonl PIN=distinct bash run_table.sh        # 11 个场景 × 1P、4P
CONCS="8" bash run_table.sh                                                                 # 8P
ONLY="1024:1024 524288:32768" bash run_table.sh                                             # 只跑部分场景
bash quick_set.sh                                                                           # 快速对比子集
```

- 每个场景先预热前缀（让 P 和 D 都缓存好 cached 部分），再发 cached+new；k = 1024 token，168k 按 172032。
- **Prefill** = P 的 vLLM 指标 `request_prefill_time`；**TTFT** = 客户端从发出请求到收到第一个 token。
- `--pin distinct`：第 i 条并发请求通过 `X-data-parallel-rank` 固定到 rank `i % 8`，4P 时 4 条请求落在 4 个不同的 rank 上。
- 请求体在计时前就编码好（大 prompt 的 JSON 编码不算进 TTFT）。短场景 5 轮、总长 ≥200k 的场景 3 轮，取中位数。

### 4.2 TPOT 表

```bash
/zzlfs/baseline/.venv/bin/python -I tpot_bench.py --out /zzlfs/baseline/results/tpot-$(date +%m%d-%H%M).jsonl
# 默认 cache 1k…768k × bs 1/16/64/128/256，out_len 1024；512k/768k bs256 拉满窗口太短时加 --out-len 3072
```

- 所有请求共享一个长度为 cache 的前缀，各自再接 128 个不同的 token，ignore_eos，temperature 0。语料是代码文本（`/zzlfs/baseline/tools/tpot_corpus_ids.json`）。
- 只统计"batch 拼满"的窗口：从所有请求都收到第一个 token，到第一个请求收到最后一个 token。
  - **CLI** = 每条请求在窗口内的平均 token 间隔，再对请求取平均。
  - **Server** = D 的 `inter_token_latency_seconds_sum` 增量 ÷ `generation_tokens_total` 增量（投机解码一步可能出多个 token）。
- 同时记录 DSpark 的平均接受长度（约 4.3–5.8）。
- **D 的 compile_sizes 要和 bs 对齐**：每个 rank 每步的行数 = 每 rank 请求数 × 6（DSpark K=5 + 1），不足时补到下一个 bucket。
  bs 1/16/64/128/256 在 8 个 rank 上对应 6/12/48/96/192 行，所以用 `[16,48,96,192,2048]` + `MAX_NUM_SEQS=32`（第 3 节）。
  脚本默认的 `[16,96,128,2048]` + 16 会让 bs64 补到 96 行（实测 TPOT 7.41 vs 5.32 ms）、bs256 超过并发上限。
- 合并后 decode 变快，长 cache 的大 batch（128k bs256、256k–768k bs128）在 out_len 1024 时"拼满窗口"为负，
  用 `--caches ... --bss ... --out-len 4096` 补测。注意 out_len 越长接受长度越高（约 4.7→5.4），TPOT 偏低 10–15%。
- 512k / 768k bs256 在合并前后都失败：proxy→P 的 HTTP 连接断开（500 / `RemoteProtocolError`），P、D 本身正常，是接入层扛不住 256 条超长 prompt 同时到达。

### 4.3 P 的 prefill 吞吐（TPS）

**测什么**：P 满载时每秒能算多少个新 token。只测 P，不经过 proxy 和 D。

**P 怎么起**：用和线上 P 相同的配置（chunk 2048、编译桶 `[128,256,2048]`、DP8/EP8），但不挂 KV 传输。5.4 的结果是 P 关 async（`ASYNC_SCHED=0`）时测的。
最好起在一台单独的机器上（原部署用 sjd1）；只有两台机器时，可以先停掉 PD 再在 P 那台上起，
也可以不重启、直接测 PD 里正在运行的 P（`--url http://<P>:8100`，不要和其他测试同时跑；5.1 和 5.4 最后一段就是这样测的，没有开 async 的对比）：

```bash
ssh -n -f <host> "cd /zzlfs/baseline && TMPDIR=/dev/shm/epic-tmp VLLM_CACHE_ROOT=/dev/shm/epic-vllm-cache NO_KV_TRANSFER=1 \
    setsid nohup bash /zzlfs/baseline/scripts/start_1p1d_epic.sh prefill > /zzlfs/baseline/logs/tps-prefill-\$(date +%m%d-%H%M).log 2>&1 < /dev/null &"
# 机器上要先有 /tmp/dsv41-tokenizer；等日志出现 "Application startup complete"。复现 5.4 的数字时加 ASYNC_SCHED=0
```

**怎么测**（`tps_bench.py`）：

```bash
cd /zzlfs/baseline/scripts
OUT=/zzlfs/baseline/results/tps-$(date +%m%d-%H%M).jsonl
PY=/zzlfs/baseline/.venv-epic/bin/python
# 新 token 长度扫描，8 并发（每个 rank 一条）
$PY -I tps_bench.py --url http://<host>:8100 --tag scan --cached 32768 --new 32768 65536 172032 262144 --conc 8 --out $OUT
# 持续满载，每个 rank 排 2 条、3 条
$PY -I tps_bench.py --url http://<host>:8100 --tag sustained --cached 32768 --new 172032 --conc 16 24 --out $OUT
```

- 每个 rank 先预热同一个 32k 前缀；每条请求的新 token 取自然代码文本（`tools/tpot_corpus_ids.json`），各用不同的偏移，互不命中缓存；第 i 条通过 `X-data-parallel-rank` 固定到 rank `i % 8`。
- 请求直接发 token id（不含 API server 分词），`max_tokens=1`。
- **TPS = 新 token 总数 ÷ 墙钟时间**（从第一条发出到最后一条完成）。每组先跑 1 轮预热（不计入），再跑 `--rounds`（默认 2）轮正式。
- 每轮从 P 的 metrics 记录 `cache_hit_tokens`，**正式轮的命中必须正好等于 前缀长度 × 并发数**，多了就说明新 token 命中了缓存，这一轮作废。脚本每次运行用随机起点生成偏移，同一个服务上重复跑也不会撞上。

### 4.4 InferenceX AgentX（aiperf，真实 Claude Code 会话回放）

```bash
# aiperf 用 GitHub main（0.13.0 正式版没有 --scenario）；数据集 semianalysisai/cc-traces-weka-062126 首次运行时自动下载
uv venv -p /usr/bin/python3.12 /zzlfs/baseline/.venv-aiperf && uv pip install --python /zzlfs/baseline/.venv-aiperf/bin/python git+https://github.com/ai-dynamo/aiperf
# proxy 必须开会话亲和（见下），再跑
ssh -n -f <P> 'PROXY_SESSION_AFFINITY_DP=8 PREFILL_HOST=<P> DECODE_HOST=<D> setsid nohup bash start_proxy.sh > proxy.log 2>&1 < /dev/null &'
CONC=256 DURATION=3600 TAG=base bash run_agentx.sh      # 结果和每 30s 的 P/D /metrics 在 results/agentx-c<CONC>-<TAG>-<时间>/
```

- `--scenario inferencex-agentx-mvp` 锁定：ignore_eos、流式、`--cache-bust first_turn_prefix`、轮间延迟按 trace 回放（只截系统空闲间隔到 10s）、至少 900s。
  `--concurrency` 是**会话数**，同时在跑的请求只有其中一部分。
- aiperf 构造数据集约 5 分钟，`run_agentx.sh` 把 `AIPERF_DATASET_CONFIGURATION_TIMEOUT` 调到 1800s（默认 300s 不够）；数据写到 `/tmp/aiperf_mmap_*`（根分区），跑完删掉。
- **前缀路由**（`PROXY_SESSION_AFFINITY_DP=<dp>` 时默认开）：前缀缓存按 DP rank 存，不做亲和时同一会话的后续轮次落到别的 rank，P 命中率只有 26%、TTFT P50 84s。
  proxy 把 messages（和 tools）拼成文本、每 2048 字符一块做链式哈希，每个 rank 记一份发过去的块（LRU，容量按 P 的 KV 大小），
  请求发到已缓存最长前缀的 rank（一样长时选排队少的），子 agent 自然跟着父会话走。`PROXY_PREFIX_MIGRATE_FACTOR` 默认 0 = 不迁移（见 5.6）。
  `PROXY_PREFIX_ROUTING=0` 退回下面的会话 key 路由：
  proxy 用 system prompt 和第一条用户消息的开头做会话 key；新会话分到估计 prefill 工作量最少的 rank，老会话留在原 rank，
  除非在原 rank 等待的代价超过"最闲 rank 重算整个上下文"的 `PROXY_AFFINITY_MIGRATE_FACTOR` 倍（默认 3，0 = 不迁移）；`PROXY_AFFINITY_BALANCE=0` 退回 hash % dp。
- 流式响应的 Content-Type 必须是 `text/event-stream`（aiperf 按它选解析器），proxy 已修。
- **命中诊断** `PROXY_HIT_DIAG=1`（P 加 `EXTRA_ARGS=--enable-prompt-tokens-details`）：每个请求记 P 实际命中、同会话 / 本 rank / 全局能命中多少（按 P 返回的 token id 每 128 个一块哈希），`hitdiag_report.py <proxy.log> <跳过前 N 个>` 汇总、按缺口类型分类。

### 4.5 单请求时间线 / 耗时分解

```bash
/zzlfs/baseline/.venv/bin/python -I d_breakdown.py --cached 524288 --new 32768 --rounds 3   # D 端 vLLM 指标分解
python3 pd_timeline.py <proxy.log> <prefill.log> <decode.log> [req_id 前缀]             # 毫秒级跨机时间线（需 VLLM_LOGGING_PREFIX）
python3 transfer_stats.py ...                                                           # KV 传输耗时统计
```

---

## 5. 结果

### 5.1 2026-11-11：合并 epic/dsv41 之后（主机 hl-2lxt / hl-6g41）

代码 `cszhz/dsv41-1p1d` 4082db4a2（合并 ef2599d3d + 附录 A 的 3 个提交），tpu-sync 用 2.2 的 `per-layer-major-dim-on-1031`，FP8 Index-K，P、D 都开 async。
参数和 5.2–5.4 完全相同，"合并前"一列就是 5.2–5.4 的数字。

**TTFT 8P**（秒，格式 Prefill / TTFT；`results/table-merge-8p-1111-1134.{jsonl,log}`；D 为脚本默认的 16 / `[16,96,128,2048]`，不影响 TTFT）

| 场景 | 合并后 | 合并前（5.2 的 8P） | TTFT 变化 |
|---|---|---|---|
| 1k+1k | 0.199 / 0.366 | 0.180 / 0.385 | −5% |
| 8k+4k | 0.462 / 0.674 | 0.487 / 0.720 | −6% |
| 32k+16k | 0.916 / 1.197 | 0.991 / 1.300 | −8% |
| 64k+32k | 1.632 / 2.003 | 1.794 / 2.245 | −11% |
| 168k+32k | 2.145 / 2.745 | 2.043 / 2.699 | +2% |
| 32k+168k | 8.455 / 9.362 | 8.597 / 9.621 | −3% |
| 64k+64k | 3.186 / 3.831 | 3.417 / 3.970 | −4% |
| 256k+32k | 2.313 / 3.030 | 2.217 / 3.082 | −2% |
| 512k+32k | 2.409 / 3.521 | 2.707 / 4.183 | −16% |
| 768k+32k | 2.776 / 4.348 | 3.185 / 5.237 | −17% |
| 32k+768k | 42.187 / 45.224 | 45.536 / 49.021 | −8% |

**TPOT**（ms，CLI；`results/tpot-merge-seqs32-1111-1455.{jsonl,log}`，D 按第 3 节 `MAX_NUM_SEQS=32` + `[16,48,96,192,2048]`，和 5.3 相同；括号内为合并前）

| bs | 1k | 16k | 32k | 64k | 128k | 256k | 512k | 768k |
|---|---|---|---|---|---|---|---|---|
| 1 | 4.01 (4.91) | 4.17 (4.68) | 4.89 (4.94) | 4.46 (5.40) | 3.91 (5.97) | 5.03 (6.30) | 4.73 (7.74) | 4.54 (9.16) |
| 16 | 4.54 (5.23) | 4.59 (5.15) | 4.58 (5.27) | 4.66 (5.42) | 4.82 (5.85) | 4.70 (6.56) | 4.76 (7.72) | 4.81 (8.66) |
| 64 | 5.32 (7.43) | 5.28 (8.28) | 5.32 (9.18) | 5.44 (10.99) | 5.22 (8.68) | 5.50 (10.27) | 5.32 (14.52) | 5.67 (19.07) |
| 128 | 6.03 (9.93) | 5.80 (12.47) | 5.66 (15.06) | 5.70 (19.18) | 5.96 (14.51) | 5.27\* (21.03) | 5.43\* (31.27) | 5.81\* (44.47) |
| 256 | 8.97 (14.60) | 8.45 (15.04) | 8.37 (15.28) | 7.53 (16.86) | 7.44\* (19.32) | 7.63\* (24.97) | —‡ | —‡ |

- 全部持平或更快：小 batch 快 1–50%，大 batch / 长 cache 快 40–70%；TPOT 基本不再随 cache 长度增长。DSpark 接受长度 4.0–5.1（合并前 4.4–5.1）。
- \* out_len 4096 补测（`results/tpot-merge-seqs32-out4k-1111-1510.*`）；out_len 1024 时"拼满窗口"为负（decode 变快，第一条请求在最后一条拿到首 token 前就生成完了）。接受长度约 5.4–5.5，TPOT 偏低 10–15%，只能粗比。
- ‡ 3 次都是 proxy→P 连接断开（4.2），合并前也失败。

**P 的 prefill 吞吐**（tok/s，直接测 PD 里的 P，8 并发，cached 32k；`results/tps-merge-1111-1134.jsonl`；合并前对应 5.4 最后一段）

| 新 token | 合并后 | 合并前（PD 里的 P） | 变化 |
|---|---|---|---|
| 32k | 19.0–19.1 万 | 16.4–16.5 万 | +16% |
| 64k | 17.6 万 | 16.4–16.7 万 | +6% |
| 168k | 16.6–17.4 万 | 17.1–17.2 万 | −3% |
| 256k | 16.5–16.8 万 | 16.6–16.9 万 | −1% |

- 和 TTFT 一致：new 段短时更快，new 段很长时略慢。

### 5.2–5.5 2026-11-05/06：合并前（主机 g45q / pqs0）

配置同第 3 节，proxy 的对齐窗口和分词缓存都关；代码 54e98d673（= 1dc6984ac + 6cb686ef4）。Baseline 为 2026-11-02 内部数据（`results/baseline-1102.md`）。

### 5.2 Prefill / TTFT（秒，格式 Prefill / TTFT，倍数 = GCP ÷ Baseline）

1P / 4P：`results/table-new-hosts-1105-1647.{jsonl,log}`；8P：`results/table-new-hosts-8p-1106-0106.{jsonl,log}`（2026-11-06，重启后同配置）。Baseline 没有 8P 数据。

| 场景 | GCP 1P | Baseline 1P | 1P 倍数 | GCP 4P | Baseline 4P | 4P 倍数 | GCP 8P |
|---|---|---|---|---|---|---|---|
| 1k+1k | 0.227 / 0.307 | 0.036 / 0.065 | 6.3× / 4.7× | 0.192 / 0.395 | 0.051 / 0.124 | 3.8× / 3.2× | 0.180 / 0.385 |
| 8k+4k | 0.432 / 0.565 | 0.055 / 0.092 | 7.9× / 6.1× | 0.465 / 0.682 | 0.113 / 0.315 | 4.1× / 2.2× | 0.487 / 0.720 |
| 32k+16k | 1.033 / 1.223 | 0.143 / 0.197 | 7.2× / 6.2× | 0.990 / 1.267 | 0.156 / 0.511 | 6.3× / 2.5× | 0.991 / 1.300 |
| 64k+32k | 1.885 / 2.146 | 0.144 / 0.220 | 13.1× / 9.8× | 1.765 / 2.133 | 0.144 / 0.609 | 12.3× / 3.5× | 1.794 / 2.245 |
| 168k+32k | 2.043 / 2.413 | 0.148 / 0.277 | 13.8× / 8.7× | 1.944 / 2.440 | 0.149 / 0.641 | 13.0× / 3.8× | 2.043 / 2.699 |
| 32k+168k | 9.139 / 9.785 | 0.726 / 0.859 | 12.6× / 11.4× | 8.408 / 9.297 | 0.747 / 2.823 | 11.3× / 3.3× | 8.597 / 9.621 |
| 64k+64k | 3.598 / 3.937 | 0.263 / 0.358 | 13.7× / 11.0× | 3.335 / 3.797 | 0.265 / 1.073 | 12.6× / 3.5× | 3.417 / 3.970 |
| 256k+32k | 2.194 / 2.664 | 0.154 / 0.319 | 14.2× / 8.4× | 2.174 / 2.841 | 0.156 / 0.684 | 13.9× / 4.2× | 2.217 / 3.082 |
| 512k+32k | 2.634 / 3.388 | 0.176 / 0.452 | 15.0× / 7.5× | 2.711 / 3.938 | 0.176 / 0.810 | 15.4× / 4.9× | 2.707 / 4.183 |
| 768k+32k | 3.010 / 4.083 | 0.200 / 0.597 | 15.0× / 6.8× | 2.900 / 4.532 | 0.202 / 0.952 | 14.4× / 4.8× | 3.185 / 5.237 |
| 32k+768k | 48.400 / 50.985 | 3.792 / 4.186 | 12.8× / 12.2× | 45.156 / 48.122 | 3.882 / 13.198 | 11.6× / 3.6× | 45.536 / 49.021 |

- 和旧主机（附录 B.1）比：1P 的 Prefill 都在 ±3% 以内，TTFT 普遍快 2–5%（本轮代码含 a233edf04，旧表在它之前测）；
  4P 长场景 TTFT 改善更明显（512k+32k 4.259→3.938s，768k+32k 5.261→4.532s）。1k / 8k 的 4P Prefill 慢约 20%（绝对差 30–80ms，短场景单次波动大）。
- 8P 和旧主机比：短场景一致，256k 以上前缀的场景快 5–10%（512k+32k 4.650→4.183s，768k+32k 5.688→5.237s）。
- Baseline 11-02 比 10 月的数据快，所以倍数比旧表大。

### 5.3 TPOT（ms，格式 CLI / Server，倍数按 CLI 算）

`results/tpot-new-hosts-1105-1748.{jsonl,log}`，默认参数 out_len 1024。

| bs | 引擎 | 1k | 16k | 32k | 64k | 128k | 256k | 512k | 768k |
|---|---|---|---|---|---|---|---|---|---|
| 1 | GCP | 4.91/5.02 | 4.68/4.61 | 4.94/4.78 | 5.40/5.35 | 5.97/5.95 | 6.30/6.35 | 7.74/7.81 | 9.16/9.22 |
| 1 | Baseline | 2.22 | 2.24 | 2.24 | 2.22 | 2.25 | 2.25 | 2.24 | 2.26 |
| 16 | GCP | 5.23/5.14 | 5.15/5.07 | 5.27/5.04 | 5.42/5.26 | 5.85/5.74 | 6.56/6.39 | 7.72/7.58 | 8.66/8.64 |
| 16 | Baseline | 3.10 | 3.14 | 3.15 | 3.15 | 3.15 | 3.15 | 3.19 | 3.21 |
| 64 | GCP | 7.43/7.38 | 8.28/8.08 | 9.18/8.80 | 10.99/10.65 | 8.68/8.37 | 10.27/10.21 | 14.52/13.73 | 19.07/18.31 |
| 64 | Baseline | 4.33 | 4.35 | 4.35 | 4.34 | 4.35 | 4.38 | 4.49 | 4.56 |
| 128 | GCP | 9.93/9.73 | 12.47/12.25 | 15.06/14.87 | 19.18/18.74 | 14.51/14.15 | 21.03/20.55 | 31.27/30.38 | 44.47/43.89 |
| 128 | Baseline | 5.21 | 5.22 | 5.18 | 5.20 | 5.25 | 5.35 | 5.51 | 5.68 |
| 256 | GCP | 14.60/14.19 | 15.04/14.62 | 15.28/14.75 | 16.86/16.33 | 19.32/18.51 | 24.97/24.88 | —‡ | OOM† |
| 256 | Baseline | 7.30 | 7.38 | 7.33 | 7.41 | 7.50 | 7.64 | 7.97 | 8.27 |

- 总体是 Baseline 的 1.6–7.8 倍（按 CLI）；DSpark 平均接受长度 4.36–5.09。长 cache、大 batch 差距最大（768k bs128 44.5 vs 5.7ms）。
- 和旧主机（附录 B.2）基本一致，bs1 / bs16 在 1k–512k 范围内相差 ±10% 以内。
- ‡ 512k bs256：out_len 1024 时"拼满窗口"不足 1s（-0.46s），旧表用 out_len 3072 补测，本轮没有补测。
- † 768k bs256：两次 HTTP 500 后 D 被 OOM killer 杀掉（第 7 节）。

### 5.4 P 的 prefill 吞吐上限（tok/s；方法见 4.3）

2026-11-06 在 g45q 上单独起的 P（`NO_KV_TRANSFER=1`、`ASYNC_SCHED=0`、DVFS 7，刚启动），和旧主机 sjd1 的测法一致；
`results/tps-dvfs7-1106-1635.jsonl`。32k 缓存前缀，每组 1 轮预热 + 3 轮正式，所有正式轮的 `cache_hit_tokens` 都正好等于 前缀 × 并发。

| 组 | 场景 | 整机 tok/s | 每芯片 tok/s | 旧主机（sjd1） |
|---|---|---|---|---|
| 新 token 长度扫描（8 并发） | 32k | 16.1–16.4 万 | 4.03–4.10 万 | 15.8–16.1 万 |
| | **64k** | **17.7 万** | **4.43 万** | 17.7–17.8 万 |
| | 168k | 17.3 万 | 4.31–4.33 万 | 17.3 万 |
| | 256k | 17.1–17.2 万 | 4.28–4.29 万 | 17.2 万 |
| 持续满载（168k 新 token） | 16 并发（每个 rank 2 条） | 16.9–17.4 万 | 4.23–4.34 万 | 17.0–17.4 万 |
| | 24 并发（每个 rank 3 条） | 16.5–17.4 万 | 4.13–4.34 万 | 16.8–17.0 万 |

- **P 的 prefill 吞吐上限约为整机 17.7 万 tok/s，每芯片约 4.4 万 tok/s**，和旧主机一致。新 token ≥64k 不再提升；每个 rank 多排请求没有用（每步预算固定 2048 token）。
- 直接测 PD 里正在运行的 P（挂着 KV connector、已连续跑了几个小时；`results/tps-new-hosts-1106-0224.jsonl`）时低 1–6%：
  32k 16.4–16.5 万、64k 16.4–16.7 万、168k 17.1–17.2 万、256k 16.6–16.9 万、16 并发 17.2 万、24 并发 16.6–17.1 万。

### 5.5 如何复现某一组数字

1. 按第 1、2 节准备代码和 venv，checkout 第 5 节注明的提交。
2. 按第 2.4 节准备机器，按第 3 节启动 P、D、proxy（`PROXY_ALIGN_MS`、`PROXY_RENDER_CACHE_MB` 都不设）。
3. 按第 4 节跑对应脚本，结果写进 `results/*.jsonl`，汇总行在同名 `.log` 里（`== ... (median): Prefill x / TTFT y`）。
4. 附录 B 的旧主机数字是在 a233edf04 之前测的（32842f3d7 + d8e22db82 的内存回收改动即可复现）；a233edf04 只降低 TTFT 中 D 这一跳的开销，不影响 Prefill 和 TPOT。

### 5.6 AgentX（2026-11-12，hl-2lxt / hl-6g41，合并后代码）

`results/agentx-c*-{base,affinity,sweep-balanced,sweep-swafix,p1fix,dtune-A,dtune-A2}-1112-*`。P 生产配置，D `MAX_NUM_SEQS=64` + `[16,48,96,192,384,2048]`（c256 两轮 3600s 时 D 为 32 / `[16,48,96,192,2048]`、P 关 async）；
标 **P 修复** 的行带 CSA2 deep gather 修复（6.1 第 11 条），标 **D 桶** 的行 D 编译桶为 `[16,48,96,144,192,288,384,2048]`（6.1 第 12 条）。

| 运行 | 时长 | req/s | 输出 tok/s | TTFT P50 / P90 | ITL P50 | P 命中率 |
|---|---|---|---|---|---|---|
| c256 无亲和 | 3600s | 1.63 | 1,636 | 84.5 / 135 s | 11.3 ms | 26% |
| c256 hash 亲和 | 3600s | 3.75 | 3,713 | 2.33 / 7.9 s | 17.4 ms | 73.5% |
| c256 负载感知亲和 | 900s | 5.04 | 4,282 | 2.29 / 6.5 s | 16.4 ms | 86.8% |
| c512 负载感知亲和 | 900s | **5.51** | **4,360** | 5.7 / 36 s | 39.8 ms | 75.3% |
| c768 负载感知亲和 | 900s | 4.50 | 3,854 | 44 / 115 s | 41.0 ms | — |
| c256 + P 修复 | 900s | 5.22 | 4,414 | 1.23 / 3.1 s | 16.8 ms | 87.5% |
| c512 + P 修复 | 900s | 6.65 | 5,538 | 2.76 / 18.2 s | 34.0 ms | 78.2% |
| c512 + P 修复 + D 桶 | 900s | **8.58** | **7,947** | 3.82 / 19.2 s | 17.1 ms | 82.0% |
| c512 + P 修复 + 更细 D 桶（加 240/336） | 900s | 8.33 | 7,772 | 4.76 / 20.0 s | 16.7 ms | 81.7% |
| c768 + P 修复 + D 桶 | 900s | 4.89 | 4,794 | 34 / 160 s | 19.4 ms | 60.2% |
| c512 + P 修复 + D 桶，会话亲和（复测） | 900s | 8.38 | 7,775 | 3.86 / 20.8 s | 16.4 ms | 稳态 92.4% |
| c512 前缀路由，迁移阈值 3 | 900s | 8.08 | 7,455 | 6.21 / 24.8 s | 17.4 ms | 稳态 90.4% |
| c512 **前缀路由，不迁移（默认）** | 900s | **8.82** | **8,091** | **2.83** / 18.8 s | 16.8 ms | 稳态 **93.3%** |
| c512 前缀路由，不迁移（默认） | **3600s** | 7.63 | 7,936 | 3.58 / 17.5 s | 15.7 ms | 稳态 94.1% |
| 参考（客户提供，c512） | 3600s | 10.79 | 10,802 | 0.34 / 0.75 s | 5.93 ms | 96.05%（理论 96.8%） |
| 参考（客户提供，c256） | 3600s | 5.22 | 5,218 | 0.22 / 0.48 s | 4.83 ms | 理论 97% |

- 峰值在 c512 附近；c768 时 P 8 个 rank 全忙、平均排队 261，D 只跑 146 个请求、KV 28%：这个负载下 **P 是瓶颈**（每个会话第一轮整段 prefill，单 rank 算）。
- ITL 随 D 并发上升（69 → 175 个请求：16 → 40 ms），DSpark 接受长度约 3.0（greedy bench 约 4.9）。
- c512 起曾反复崩溃：P 的 SWA 私有池耗尽（见第 7 节），已修。
- P 修复后 TTFT 减半（c256 P50 2.29→1.23s、P99 23.6→7.4s）；D 加 144/288 桶后 c512 的 ITL 34→17 ms、吞吐 +43%。再细（240/336）没有额外收益。现在 c512 下 TTFT 又由 P 决定（P50 3.8s）；c768 时 P 过载，吞吐反而下降，峰值仍在 c512。
- **P 命中率的缺口在路由，不在容量**（命中诊断，2026-11-13）：P 的 KV 使用率一直 0–15%、没有抢占，P 实际命中和"本 rank 模拟"完全一致；
  整轮平均偏低主要是 aiperf warmup（前约 12 分钟，512 个会话的首轮全部冷算）。稳态下会话亲和 92.4%，全局理论 95.7%，差的 3.3% 全是前缀在别的 rank：
  子 agent 的 key 和父会话不同、负载触发的迁移。前缀路由跟着缓存走、不迁移后到 93.3%，c512 吞吐 8.82 req/s、TTFT P50 2.83s。
  迁移阈值 3 时 P 一饱和就频繁迁移（483 次），每次整段重算又加重 P，反而更差（8.08）；按加性代价（排队 + 2 × 重算）选 rank 时迁移 1895 次，只有 6.67 req/s。

### 5.7 AgentX c256（2026-11-15，Rocky 主机 1hnv / 304m）

`results/agentx-c256-{base-el9,chunked,chunked-droute2,chunked-droute-eplbperm,best-3600}-1115-*`，汇总用 `agentx_summary.py`。
代码 e6f1f1e17；P 生产配置 + `--enable-prompt-tokens-details`，D `MAX_NUM_SEQS=64` + `[16,48,96,144,192,288,384,2048]`，两边 `TPU_RECLAIM_COMPILE_MEMORY=1`；proxy 前缀路由 + 命中诊断。
aiperf 的统计不含 warmup 阶段的请求。

| 运行 | 时长 | req/s | 输出 tok/s | TTFT P50 / P90 | ITL P50 | D 最忙 / 最闲 rank（运行中请求） |
|---|---|---|---|---|---|---|
| 基线 | 900s | 5.51 | 4,906 | 2.56 / 8.85 s | 11.8 ms | 6.4 / 1.8 |
| + P 分块分词（6.1 第 15 条） | 900s | 5.84 | 5,168 | 0.83 / 2.36 s | 12.3 ms | 10.0 / 2.3 |
| + D 路由（6.1 第 16 条） | 900s | **5.99** | **5,348** | 0.87 / 2.25 s | **11.2 ms** | 6.6 / 4.2 |
| + D 静态 EPLB `dsv41_flash_dp8_c384_perm` | 900s | 5.91 | 5,252 | 0.87 / 2.38 s | 11.7 ms | 7.0 / 4.5 |
| 分块分词 + D 路由 | **3600s** | 4.55 | 4,600 | 0.90 / 2.58 s | 10.4 ms | — |
| 参考（客户提供，c256） | 3600s | 5.22 | 5,218 | 0.22 / 0.48 s | 4.83 ms | — |

- 基线的 TTFT 卡在 P 的 API server：proxy 只对 `/v1/completions` 本地分词，AgentX 的 chat 请求每个都在 P 上整段分词（py-spy：API server 91% 的 CPU 在 `_encode_plus`，25 万 token 约 0.5s）。
  4 个 API server 一忙就排队，P 段 P50 随时间从 1.2s 涨到 7.8s，而 P 引擎内部只要约 0.74s。分块分词后稳态 P 段 P50 约 0.55s（排队 0.45s + prefill 0.51s，均值），D 段约 0.27s。
- D 路由前 D 的请求跟着 P 的 rank（前缀路由），c256 时最忙 rank 平均 10 个、最闲 2.3 个；DP lockstep 下每步等最忙的。均衡后 ITL -9%，换 rank 多拉 KV 的代价看不出来（D 段不变）。
- 静态 EPLB 没有收益：c256 下 D 每个 rank 只有约 40 行，MoE 受权重读取限制，专家负载均衡影响不大。换 layout 后 D 重新编译约 1 小时。
- 3600s 的吞吐低于 900s：会话是闭环的（上一轮结束才发下一轮），吞吐由单请求延迟决定；后段会话更长、长输出也能跑完（平均输出 1011 vs 892）。
  单请求约 0.9s + 1011 × 10.4ms ≈ 11.4s，参考约 5s。剩下的主要差距是 ITL：每步耗时约 34ms（和 greedy bench 相当），但 temperature 1.0 下 DSpark 每步只接受约 3.0 个 token（greedy 约 5.9）。

---

## 6. 改动与效果（按收益排序）

### 6.1 已生效（结果表基于这些）

1. **tpu-sync `GetBlockChunks` O(n²)→线性**：557k token 传输 27.9s→0.95s。
2. **D 开前缀缓存，只拉未命中部分**（b4d7bf2af + 补丁 §4）：可前缀缓存的 group 用 `dst_skip_bytes` 跳过命中页，SWA / 压缩器状态按 P 的长度整段拉；调度器对这类请求不取整、不回放。512k 前缀命中时实际传输只要 30–40ms。
3. **P 的编译桶加 128 / 256**（128 = sliding_window）：CED 在最后一个 chunk 上只对尾部 ≤128 行重算 decoder，原来被补齐成 2048 行，每个请求白跑约 0.37s。
   新主机上单机 DP8 cached-extend（96k+16k，C8）复测 `[128,2048]` → `[128,256,2048]`：最后一个 chunk 256.6→184.5ms，整轮 810→724ms（-10.6%），中间 chunk 不变。
4. **升级到最新 epic/dsv41**：kernel 优化让 Prefill 快 15–26%。需要配套 6b6565b38、dd89c8b2d 和新 venv。
5. **反分词器只用 prompt 末尾预热**（补丁 §5）：512k+32k TTFT 4.27→3.58s。2000 例随机测试输出逐字一致。
6. **proxy 只分词一次**：本地切块并行分词（只在"字母 空格 字母"处切，保证拼接结果和整段分词一致）+ 进程池 + orjson；P、D 不再各分词一次（557k token 每次约 2.7s）。
7. **4P 对齐**：DP wave 起步等待（P 40ms / D 20ms）、4 个 API server、orjson 解析、proxy 进程池。512k+32k 4P TTFT 7.0→约 4.0–4.5s。
8. **空闲 EP-DP rank 复用全零 dummy block table**（4250476f6）：bs1 TPOT 降低约一半。
9. **D 开 async scheduling**：TPOT 降低 10–46%；P 开了对 TTFT 没有帮助（2026-10 的结论）。2026-11-08 起脚本默认 P、D 都开 async。
10. **合并 epic/dsv41 ef2599d3d**（2026-11-11，5.1）：TPOT 不再随 cache 长度增长（大 batch / 长 cache 快 40–70%），TTFT 8P 最多快 17%。
    合并带来的 SWA 私有池、packed FP8 Index-K、45 个 KV storage 需要配套：
    - tpu-sync 允许各 storage 块数不同（2.2，上游 PR google/tpu-sync#1527）；
    - manifest / layout 校验支持 SWA 私有池整页传输、FP8 Index-K 整页传输，块数按 KV cache group 校验（c96b3269e）；
    - LIFO 释放补丁按 `block.pool` 分流（dc4f44ee0）：KV connector 的延迟释放会把 SWA 私有池的块一起交给共享池，
      原补丁把它们塞进共享池的空闲队列，下一个请求就拿到重复的块 id（`dst_device_block_ids ... must not contain duplicates`）。上游 epic/dsv41 也有这个问题，只在 PD 下触发。
11. **CSA2 deep gather 的清零行按 lane 分开**（159e4efe4，2026-11-12）：`_gather_sc_banked_deep` 对无效 id（-1）从共享零表 DMA 第 0 行来清零，
    所有 subcore 的所有 lane 读同一个地址。无效 id 占多数时（DP lockstep 下空闲 rank 的 dummy 前向、短 selection）这些读串行化：
    空闲 rank 的 gather 0.84 ms/次，忙 rank 只要 0.057 ms；MoE 集合通信又让忙 rank 等空闲 rank，结果 P 上单请求 prefill 比 8 并发还慢约 2.4 倍。
    改成每个 subcore lane 读自己的零行。32k token prefill（P DP8）：1 请求 3.32→1.34–1.54s，8 请求 1.25–2.06→1.18–1.72s；
    和 `TPU_DEEPSEEK_V41_CSA2_DEEP_GATHER=0` 一样快，PD logprob 检查不变。AgentX c512 吞吐 +21%、TTFT P50 减半（5.6）。
12. **D 编译桶加 144 / 288**（DSpark K=5，每请求 6 行；`MAX_NUM_SEQS=64` 时最多 384 行）：`[16,48,96,192,384,2048]` → `[16,48,96,144,192,288,384,2048]`。
    原来 96→192→384 之间最多补齐 2 倍，AgentX c512 的 ITL 34→17 ms、输出吞吐 5,538→7,947 tok/s。D 编译缓存多约 10 GB（42 GB）。
14. **proxy 前缀路由**（2026-11-13）：按文本前缀哈希把请求发到已缓存最长前缀的 P rank、不迁移，AgentX c512 稳态命中 92.4→93.3%、吞吐 8.38→8.82 req/s、TTFT P50 3.86→2.83s（5.6）。
13. **proxy 复用 P 的 prompt token ids**（159e4efe4）：chat 请求让 P 返回 `prompt_token_ids`，放进 `kv_transfer_params` 交给 D，D 不再渲染模板、重新分词（`PROXY_REUSE_PROMPT_IDS=0` 关）。100k token 命中缓存的 chat TTFT 873→678–774ms，输出一致。
15. **P 的 API server 分块分词**（`apply_venv_patches.py` 第 8 节，2026-11-15）：vLLM `_tokenize_prompt` 对长 prompt（≥128K 字符）按 proxy 同样的规则切块
   （只在两个 ASCII 字母之间的空格处切），用 `tokenizers.encode_batch` 并行编码再拼接。只用于不要 offset、参数只含 `add_special_tokens` / 截断的情况，
   截断方向按 tokenizer 的 `truncation_side`（vLLM 默认 left）；用一份关掉截断 / padding 的 backend 副本（HF 的 `__call__` 会把 `enable_truncation` 留在共享 backend 上）。
   离线随机测试和 AgentX 真实流量 1414 条长请求（`VLLM_CHUNKED_TOKENIZE_VERIFY=1` 逐条和整段分词比对）全部一致；单条约快 9 倍。`VLLM_CHUNKED_TOKENIZE=0` 关。
   AgentX c256 TTFT P50 2.56→0.83s、P90 8.85→2.36s（5.7）。
16. **proxy 给 D 单独选 rank**（`PROXY_DECODE_ROUTING=1`，`PROXY_DECODE_SLACK` 默认 2）：D 也按前缀块哈希选缓存了最长前缀的 rank（保留 D 的前缀缓存），
   但该 rank 在跑的 decode 流比最少的 rank 多出 slack 以上时改发最闲的 rank。日志 `PERF PROXY decode_route`。AgentX c256 ITL 12.3→11.2 ms（5.7）。

### 6.2 D 这一跳的控制路径优化（2026-10-29，已部署）

512k+32k 1P 毫秒级时间线（`pd_timeline.py`）显示，D 这一跳的 0.58s 里 KV 传输只有约 35ms，其余全是控制路径开销：

| 时刻（ms） | 事件 |
|---|---|
| 0 → 137 | proxy 分词 |
| 2950 | P 请求完成 |
| 2957 | proxy 转发给 D |
| 3161 | D 调度器第一次看到请求（D 接收请求约 200ms：JSON 解析 23ms、token 校验 8ms、前缀哈希 20–40ms、msgpack/ZMQ、起步等待 20ms…） |
| 3255 | **P worker 登记 KV block**（比请求完成晚 305ms，当时的关键路径） |
| 3290 / 3328 | D 提交传输 / 传输完成 |
| 3514 | 第一个 token（KV 到齐后约 170ms） |

- **P 端登记加速**（a233edf04）：`lower_dsv4_row_spans` 按 (每页字节数, rank, 并行度) 缓存整页 span，每次切片 + 补尾页；同时缓存 ReshardClient 的 wire 元组，`_register_request_blocks_prelowered` 直接交给 C++ client（非 C++ client 时退回公开 API）。
  效果：登记 305ms→138ms（lowering 192→72ms，RPC 72→36ms）。与原实现做了 3000 组随机等价性测试。
- **sliding-window trim 加速**（a233edf04，P、D 都部署）：cProfile（`TPU_STAGE3_PROFILE=1`）显示 P 剩下的 72ms 和 D 准备接收计划的 67ms，都花在 `_stage3_trim_window_pages` 上：
  51 个 cache group，每个约 1.7 万个已出窗口的页，逐个 `int()` 加生成器 `any(...)` 检查。改为只转换保留的页，丢弃的页用 `list.count` 检查空块（报错语义不变）。
  效果：51 个 group 74ms→3.4ms；P lowering 72→10.5ms（登记只比请求完成晚 81ms），D stage 67→5ms；**D 这一跳 581→约 500ms**。
  `test_tpu_connector` / `test_pool_byte_spans` / `test_dsv41_pool_manifest` 共 399 个测试通过。
- **proxy 对齐窗口 `PROXY_ALIGN_MS`**（原部署上已部署，当时 8ms；脚本默认 0=关，第 5 节的复测没开）：窗口内先后到达的请求归为一组，组内全部分词完成后再一起发给 P，避免分词耗时不同导致错过同一个 lockstep 步。单请求的对齐等待实测 0.01ms。4P 效果待测。
- **proxy 分词缓存 `PROXY_RENDER_CACHE_MB`**（已实现，默认关）：每个 worker 按 64KB 切块做 LRU 缓存，前缀相同的 prompt 只对尾部重新分词（约省 120ms）。随机测试与整段分词逐 token 一致。
  注意：只在文本前缀重复时有效（多轮对话 / 共享文档，即表里的 cached+new 场景）；报告结果时开、关要分开列。

### 6.3 试过但没用 / 不可行

- **D 静态 EPLB**（`TPU_EXPERT_PLACEMENT=dsv41_flash_dp8_c384_perm`，2026-11-15）：AgentX c256 无收益（ITL 11.2→11.7 ms，在误差内），换 layout 后 D 重新编译约 1 小时（5.7）。
- async scheduling 用在 P 上：TTFT 无改善。
- **DSpark 的 K 不能改**：`num_speculative_tokens` 必须等于 checkpoint 的 block size 5（`block/Markov-head machinery is trained for exactly that block length`），K=3/4 启动即报错。
- dummy 前向不放 token（让空闲 rank 的 selection 全为 -1）：没用，热点在 deep gather 的共享零行（6.1 第 11 条）。
- chunk 4096 / 8192：XLA SparseCore RET_CHECK，编译失败。
- prefix 哈希换 xxhash：82 万 token 只省 35ms。
- 8 个 API server：对 4P 没有额外帮助；P 的起步等待 60ms 也没有帮助。
- **TP8**：TP>1 时 CSA2 关闭，退回原生注意力 + RoPE 附属池，block 数和主池不同，tpu-sync KVCacheManager 拒绝；不带 KV 传输单独起 P 时输出是乱码（虽然 Prefill 约快 2 倍）。维持 DP8。

---

## 7. 已知问题与踩坑

**Rocky 9 主机（2026-11-14，sunnyfang-tpu7x-dws-3-1hnv / -4-304m）**

venv 和自编 tpu-sync 都是在 Ubuntu 24.04 上建的，换到 Rocky Linux 9 要额外处理：
- 自编的 `_tpu_raiden_torch_2_13_0.so` 要 glibc 2.38（Rocky 9 是 2.34）：在 Rocky 上用 clang 19 重编（`/zzlfs/temu/src/build-tpusync-1031-el9.sh`，
  libzmq 要强制 `CMAKE_INSTALL_LIBDIR=lib`），产物放 `/zzlfs/temu/overlay-el9/tpu_sync`，`start_1p1d_epic.sh` 在 RHEL 上自动加进 PYTHONPATH。
- 要装 `python3.12`（venv 的 python 指向 `/usr/bin/python3.12`）、`libibverbs`；runai streamer 找 Ubuntu 的 CA 路径，建软链接 `/etc/ssl/certs/ca-certificates.crt -> /etc/pki/tls/certs/ca-bundle.crt`，否则读 GCS 报 `File access error`。
- Rocky 的 python3.12 是 PIE，**必须开 `TPU_RECLAIM_COMPILE_MEMORY=1`**：不开时 P、D 编译期都把 944 GB 吃光，整机卡死（OOM 后 worker 报 `RuntimeUnexpectedCoreHalt`）。
- `/etc/environment` 设了 `LD_PRELOAD=libtcmalloc`（ssh 登录时由 PAM 带进来），和 reclaim 冲突（`requires Linux with glibc malloc`）；`start_1p1d_epic.sh` 开 reclaim 时会 unset。
- `stop_vllm_local.sh` 原来只匹配 `.venv/bin/vllm serve`，停不掉 `.venv-epic` 的主进程（已修）。

**主机内存（新主机 944 GB）**

- **`TPU_RECLAIM_COMPILE_MEMORY` 只能关**：`host_memory.py` 用 `dladdr` 检查 malloc 是否来自 glibc。`/usr/bin/python3.12`（以及 uv 的 python-build-standalone）
  是非 PIE 可执行文件，`dlsym(RTLD_DEFAULT,"malloc")` 解析到可执行文件里的 PLT 地址而误判失败（实际 malloc 仍是 glibc）。
  用 PIE 的 Python（`--enable-shared --enable-optimizations --with-lto` 自编 CPython，与 CI 的 `python:3.12-slim` 一致）建 venv 后可以开。
- **关着回收时内存余量很小**：8 个 worker 编译完不归还堆内存，新主机上 P 启动后剩约 140 GB，D（编译桶更多）只剩约 37 GB；
  单机 DP8 的编译阶段曾因此被 OOM killer 杀（`program continuator has halted unexpectedly` / `cancelled`）。
- **D 跑不了 TPOT 768k bs256**（2026-11-05）：D 的 worker 共享内存涨到每个 150 GB 以上，先返回 HTTP 500，随后 worker 被 OOM killer 杀、整个 D 退出。
  长时间测试后 P 的可用内存也会从约 140 GB 降到约 74 GB。要跑这一格需要 PIE Python + 回收，或更大内存的机器。
- 进程被杀时 `/dev/shm/psm_*` 不会回收，`/dev/shm/torch_tpu_cache` 只增不减，都直接占主机内存；`stop_vllm_local.sh` 会清。
- **D 换新 compile_sizes 编译时差点 OOM**（2026-11-11，hl-6g41）：内存账是 8 个 worker 编译期峰值约 640 GB（每个 76–109 GB）
  + Engram CPU offload 共享表 189 GB（`/dev/shm/vllm_engram_*`，2 × 94 GB）+ `/dev/shm` 编译缓存 242 GB（`epic-vllm-cache` 136 G + `torch_tpu_cache` 106 G），
  超过 944 GB，可用内存降到 15 GB。删掉不再用的 `torch_compile_cache/<hash>`（2.4）后正常。`df /dev/shm` 比 `du` 大时，差额是已删除但仍被进程打开的文件（`lsof +L1`）。
- 换配置启动时建议盯着可用内存（`free -g` 的 available），低于 20 GB 就停掉；停的时候要确认 `VLLM::Worker` 都退出了，第一次 `stop_vllm_local.sh` 偶尔停不干净。
- **合并前 TPOT 768k bs256 时 D 被 OOM killer 杀**（2026-11-05）的问题，合并后没有再出现（这一格现在在 proxy→P 就失败，见 4.2）。

- **P 在 AgentX c512 下崩溃：`DeepSeek-V4.1 SWA pool exhausted: ... needs N pages, M free of 241`**（2026-11-12）。SWA 私有池按 `max_num_seqs` 个运行中请求定容量，
  但 PD 下 P 做完 prefill 的请求要留着窗口页等 D 拉走，D 跟不上时积压越来越多；原代码页不够直接抛异常，引擎退出、其余引擎随后报 gloo `Connection closed by peer`
  （看起来像 worker 被杀，其实第一现场是某个 EngineCore 的这条异常）。已修：KV connector 下按 `4 × max_num_seqs` 预留待传请求的窗口页（`TPU_DSV41_SWA_HELD_SEQS` 可调），
  超出时延后分配而不是退出（`TPU_DSV41_SWA_EXHAUSTED_RAISE=1` 恢复报错）。

**磁盘与路径**

- **根分区被编译缓存写满**（原部署 TorchInductor + vLLM torch_compile_cache 一度 484G，新主机根分区只有 8.7 GB）：编译时报 ENOSPC 退出。
  `TMPDIR`、`VLLM_CACHE_ROOT` 放 `/dev/shm`；stop 脚本会在根分区超过 75% 时清缓存。
- 根分区满时 /tmp 也写不了。

**进程与端口**

- **pkill / pgrep 会匹配到自己的 shell**：停进程的匹配模式只写在 `stop_vllm_local.sh` 文件里；手动查进程用 `pgrep -f "[t]oy_proxy_server.py"` 这种写法；
  调用 `stop_vllm_local.sh` 的命令行里不要出现它匹配的字样。
- **`Failed to acquire a TPU device node '/dev/vfio/N'`**、**D 的 gRPC / Raiden 端口被占用**：上一轮的 `VLLM::Worker` 残留进程还在。一定要用 `restart_epic.sh`（会清掉残留 worker、等 `/dev/vfio` 释放）。
- **建 DP 组时 `DistBackendError: File name too long`**：GCE 主机名正好 64 字符，gloo 用 `HOST_NAME_MAX` 大小的缓冲读 `gethostname()`，放不下结尾的 `\0`。
  `GLOO_SOCKET_IFNAME` 不影响 vLLM 这条 `ProcessGroupGloo(store, rank, size)` 路径；只能改短主机名（2.4 节）。
- ssh 远程起后台进程时用 `ssh -n -f ... > /dev/null 2>&1`，否则管道会卡住。

**测试**

- `wait_ready.sh` 以 HTTP 状态码判断就绪（`curl -w ok` 会误判）。
- 首次遇到新的形状（batch / 长度）时会临时编译，那一轮的数字会偏大；要先预热，或者重跑确认。
- TPOT 512k / 768k bs256 时 proxy→P 的 `httpx.ReadError`（500）或 `RemoteProtocolError`，P、D 日志里没有报错；`tpot_bench.py` 会自动重试 3 次，合并前后都过不了。
- TPOT 的"拼满窗口"为负（`full-batch window shorter than 1 s`）不是服务出错，是 decode 比 P 把所有请求送进来更快，加大 `--out-len` 补测（4.2）。
- logprob 检查的 `--offset` 超出语料长度会让 prompt 只剩问题（见第 3 节冒烟测试下方）；当前脚本已加断言。
- **测吞吐时新 token 命中缓存会让数字虚高**（曾出现一轮 230 万 tok/s）：同一个服务上重复跑，如果请求内容和之前一样，P 的前缀缓存会直接命中。要核对每轮的 `cache_hit_tokens`（见 4.3）。
- 在 sjd1 上直接跑 JAX/Pallas 脚本时，要先设好 TPU 环境变量（`/zzlfs/baseline/tools/sc_repro/env.sh`，和启动脚本里的一样），否则 JAX 初始化会卡住。

---

## 8. 下一步

512k+32k 1P 当前的关键路径（P 端登记、trim 修复后；TTFT 约 3.39–3.45s，其中 P-hop 约 2.76–2.82s）：

| 段 | 耗时 | 计划 |
|---|---|---|
| proxy 分词 | 130ms | 分词缓存（已实现，默认关） |
| D 接收请求（proxy 发出 → D 调度器） | ~195ms | 根治办法是让 proxy 在 P 计算的同时就把请求发给 D（需要提前确定 P 的 request id、uuid、DP rank，改动大） |
| D 调度器 → 传输提交完成 | ~64ms | 其中接收计划 5ms、RPC 35ms |
| 传输 | 35–66ms | — |
| KV 到齐 → 第一个 token | ~173ms | 检测→调度→前向 53ms + draft 42ms，待分析 |

其他候选：4P 对齐窗口的效果验证；动态投机 token 数；空步剩余约 20ms；共享前缀注意力 kernel；venv 换 PIE Python 以打开编译内存回收。

2026-11-12 新增（来自和 B200 的对比表，见对话记录）：
- 生产采样（temperature 1.0、top_p 0.95）下 DSpark 的接受长度没有测过；按 TPOT 反推只有约 1.2–1.5（greedy 时约 4.9），是低并发 ITL 比 bench 慢约 3 倍的主要嫌疑。用 `tpot_bench.py` 加采样参数实测。
- 每个请求独立长上下文（不共享前缀）时 D 的 KV 容量：c=64 起吞吐下降、ITL 翻倍，怀疑抢占；需要记录 KV 使用率和 `num_preemptions`。
- 编译缓存从 `/dev/shm` 挪到本地 NVMe（根分区约 206 GB 空闲），避免每换一次配置就多占 23–36 GB 内存。

---

## 附录 A. 提交列表（合并前的提交在 epic/dsv41 6a1ebddb7 之上，11-11 合并到 ef2599d3d）

| 提交 | 内容 |
|---|---|
| 92b47e58c … 2f41443be | 旧 PD 分支的 9 个 cherry-pick（Stage-3 Raiden PD 基础） |
| b4d7bf2af | connector：D 保留前缀缓存，只拉命中之后的部分（per-tag `dst_skip_bytes`，`skips_prefix_replay`）；拉取失败按请求上报（failed_recving），不再杀引擎 |
| 0a0e11633 | producer 只构建 draft 上下文 |
| 6b6565b38 | halfplane Index-K 以 512 字节 owner 行为单位传输（pool_manifest）；去掉"禁止 kv_transfer"的检查 |
| dd89c8b2d | cache.py 补 `from vllm_torchtpu import envs` |
| f1b67377b | async scheduling 下 producer 保留 draft 提议（否则断言失败） |
| 4250476f6 | 空闲 EP-DP rank 复用全零 dummy block table（bs1 TPOT 减半）；`TPU_DUMMY_STEP_TIMING` 计时 |
| 32842f3d7 | `scripts/dsv41_pd/`：部署、venv 补丁、benchmark 脚本 |
| a233edf04 | Stage-3 DSv4：P 端 KV 登记加速（span 缓存 + 预建 wire 元组）、sliding-window trim 加速；诊断计时与 `TPU_STAGE3_PROFILE` 开关（默认关） |
| d8e22db82 | 加载权重时回收 host 内存（`_device_sync` 里 sync + `gc.collect()` + `malloc_trim`，`flush_all` 后清空 pool）；没有这项时加载阶段 host 内存可能不够 |
| 2d5d94f28 / b20ea9162 / 58dce5ae1 | 本文档、`pd_timeline.py`、proxy 对齐窗口 / 分词缓存、`restart_epic.sh` 透传诊断变量 |
| 6cb686ef4 | `tps_bench.py` 和 TPS 测试方法 / 结果 |
| 1dc6984ac | 新主机的已知问题；修正 `TPU_RECLAIM_COMPILE_MEMORY` 注释 |
| 2289534bc | 主机 / 地址可用环境变量覆盖（默认值不变）；新主机复测 1P / 4P、TPOT |
| 058073801 | 重新整理本文档结构；新主机 8P、TPS 结果 |
| df88c0c57 | 脚本默认 P、D 都开 async；`DVFS_P_STATE`；TPS 与 DVFS 结果 |
| f6693c749 | 合并上游 epic/dsv41 ef2599d3d（SWA 私有池、packed FP8 Index-K 等）；indexer_cache 去掉"禁止 kv_transfer"的检查随合并移植 |
| dc4f44ee0 | LIFO free-block 补丁按 `block.pool` 把块还给所属的池（修 PD 下重复块 id）；`tests/test_block_pool_lifo_free.py` |
| c96b3269e | Raiden manifest：SWA 私有池整页传输、packed FP8 Index-K 整页传输、块数按 KV cache group 校验；layout 校验放宽；新增 3 个测试 |
| 4082db4a2 | `pd_logprob_check.py` 语料越界断言；新增 `local_hit_control.py` |
| （本次） | 本文档更新到 11-11 合并后的状态 |

---

## 附录 B. 旧主机结果（2026-10-28/29，2z6z / clsk，Baseline 为 10 月数据）

### B.1 Prefill / TTFT（秒）

1P：`results/table-epic-1028-2309`；4P/8P：`results/table-4p8p-1029-0059`；16P（前 5 个场景）：`results/table-16p-1029-0333`。

| 场景 | GCP 1P | Baseline 1P | 1P 倍数 | GCP 4P | Baseline 4P | 4P 倍数 | GCP 8P | GCP 16P |
|---|---|---|---|---|---|---|---|---|
| 1k+1k | 0.213 / 0.317 | 0.039 / 0.090 | 5.5× / 3.5× | 0.161 / 0.383 | 0.060 / 0.140 | 2.7× / 2.7× | 0.185 / 0.406 | 0.219 / 0.438 |
| 8k+4k | 0.419 / 0.583 | 0.171 / 0.258 | 2.5× / 2.3× | 0.383 / 0.651 | 0.173 / 0.520 | 2.2× / 1.3× | 0.469 / 0.744 | 0.500 / 0.835 |
| 32k+16k | 1.069 / 1.286 | 0.171 / 0.289 | 6.3× / 4.4× | 0.978 / 1.280 | 0.182 / 0.596 | 5.4× / 2.1× | 0.991 / 1.358 | 1.132 / 1.831 |
| 64k+32k | 1.912 / 2.240 | 0.179 / 0.341 | 10.7× / 6.6× | 1.770 / 2.182 | 0.179 / 0.759 | 9.9× / 2.9× | 1.795 / 2.253 | 2.039 / 3.299 |
| 168k+32k | 2.069 / 2.504 | 0.185 / 0.407 | 11.2× / 6.2× | 1.926 / 2.496 | 0.185 / 0.827 | 10.4× / 3.0× | 2.071 / 2.805 | 3.139 / 4.915 |
| 32k+168k | 9.172 / 9.893 | 0.892 / 1.109 | 10.3× / 8.9× | 8.372 / 9.296 | 0.915 / 3.471 | 9.1× / 2.7× | 8.650 / 9.692 | |
| 64k+64k | 3.595 / 4.032 | 0.322 / 0.505 | 11.2× / 8.0× | 3.298 / 3.801 | 0.323 / 1.334 | 10.2× / 2.8× | 3.382 / 3.975 | |
| 256k+32k | 2.189 / 2.784 | 0.189 / 0.459 | 11.6× / 6.1× | 2.069 / 3.008 | 0.189 / 0.882 | 10.9× / 3.4× | 2.338 / 3.286 | |
| 512k+32k | 2.628 / 3.508 | 0.205 / 0.609 | 12.8× / 5.8× | 2.858 / 4.259 | 0.206 / 1.055 | 13.9× / 4.0× | 2.896 / 4.650 | |
| 768k+32k | 2.997 / 4.227 | 0.225 / 0.763 | 13.3× / 5.5× | 3.401 / 5.261 | 0.225 / 1.272 | 15.1× / 4.1× | 3.472 / 5.688 | |
| 32k+768k | 48.466 / 50.967 | 4.618 / 5.151 | 10.5× / 9.9× | 45.273 / 48.958 | 4.746 / 16.275 | 9.5× / 3.0× | 45.741 / 49.432 | |

单点（2026-10-29，含反分词器修复与空闲 rank 修复）：512k+32k 1P TTFT 约 3.40–3.45s。

### B.2 TPOT（ms，CLI / Server；`results/tpot-final-1029.md`）

| bs | 引擎 | 1k | 16k | 32k | 64k | 128k | 256k | 512k | 768k |
|---|---|---|---|---|---|---|---|---|---|
| 1 | GCP | 5.24/5.29 | 4.87/4.74 | 4.54/4.35 | 5.74/5.63 | 5.27/5.22 | 6.30/6.20 | 7.96/8.06 | 7.68/7.54 |
| 1 | Baseline | 2.81 | 2.77 | 2.76 | 2.76 | 2.80 | 2.76 | 2.80 | 2.81 |
| 16 | GCP | 5.24/5.17 | 5.06/4.82 | 5.33/5.10 | 5.67/5.43 | 5.80/5.69 | 6.15/5.89 | 7.62/7.51 | 8.57/8.75 |
| 16 | Baseline | 3.52 | 3.51 | 3.50 | 3.56 | 3.52 | 3.56 | 3.62 | 3.72 |
| 64 | GCP | 7.21/7.05 | 8.16/8.05 | 9.49/9.21 | 10.57/10.42 | 8.36/8.13 | 10.40/10.09 | 14.10/13.92 | 19.50/19.22 |
| 64 | Baseline | 5.39 | 5.41 | 5.40 | 5.41 | 5.41 | 5.39 | 5.59 | 5.81 |
| 128 | GCP | 9.66/9.44 | 12.50/12.30 | 14.70/14.29 | 18.66/18.31 | 14.43/13.99 | 21.54/21.19 | 31.32/30.11 | 45.11/44.69 |
| 128 | Baseline | 6.60 | 6.64 | 6.61 | 6.65 | 6.70 | 6.82 | 7.16 | 7.53 |
| 256 | GCP | 14.51/13.97 | 12.82/12.65* | 15.68/14.78 | 16.34/15.84 | 18.31/17.51 | 23.63/22.83 | 21.72/21.35* | 27.87/27.18* |
| 256 | Baseline | 9.56 | 9.60 | 9.58 | 9.61 | 9.92 | 10.05 | 10.69 | 11.28 |

\* 用 out_len 3072（out_len 1024 时拉满窗口太短）。总体是 Baseline 的 1.3–6.0 倍。

### B.3 P 的 prefill 吞吐上限（tok/s；sjd1，`results/tps-sjd1-1029-1600.jsonl`）

32k 缓存前缀，每组 2 轮正式。整机 = 4 芯片 / 8 个 rank。

| 组 | 场景 | 不开 async | 开 async |
|---|---|---|---|
| 新 token 长度扫描（8 并发） | 32k | 15.8–16.1 万 | — |
| | **64k** | **17.7–17.8 万** | 16.9 万 |
| | 168k | 17.3 万 | 17.4 万 |
| | 256k | 17.2 万 | — |
| 持续满载（168k 新 token） | 16 并发（每个 rank 2 条） | 17.0–17.4 万 | 17.2–17.4 万 |
| | 24 并发（每个 rank 3 条） | 16.8–17.0 万 | — |

- 上限约为整机 17.5 万 tok/s，每芯片约 4.4 万 tok/s，条件是 8 个 rank 各有一条长请求（新 token ≥64k、缓存上下文中等长度）。
- 新 token 超过 64k 不再提升；每个 rank 多排请求没有用（每个 rank 每步的预算固定是 2048 个 token）；P 开 async 对吞吐没有帮助。
- 单请求（1P）时只有一个 rank 在算，约 1.9 万 tok/s。
- profile（32k+168k 1P，每步约 105ms 设备时间）：MoE（20 个 encoder 层）44.5ms（42%），CSA 注意力 12.8ms，indexer top-k 8.7ms，量化融合 6.4ms，all-gather 5.1ms；host 侧空隙约 4%。
