<!-- SPDX-License-Identifier: Apache-2.0 -->

# vLLM TorchTPU 原生服务启动与 Decode 性能测试

本记录覆盖 2026 年 10 月 30 日使用原生 `vllm serve`、dummy 权重、DP8TP4 和 MTP K=3 进行的长上下文 decode 测试，并提供无需 prompt 预热的合成 KV 测试模式。

合成 KV 模式的 P131072/D1024、全局 96 并发实测稳态吞吐为 **879.89 tok/s**，稳态平均 TPOT 为 **109.10 ms**。正常 prefill 后截取共同 decode 窗口的结果为 **877.83 tok/s / 109.36 ms**。两次均为单轮结果，不能把 0.23% 的差异视为显著性能提升。

## 1. 硬件、软件和模型条件

| 项目 | 本次测试条件 |
|---|---|
| TPU | TPU7X，四 host，拓扑 `2x2x4` |
| 并行部署 | DP=8，TP=4，PP=1；每 host 两个 DP、八个 worker；总计 32 个 worker |
| 软件 | `vllm==0.29.0+cpu`，`vllm-torchtpu==0.0.0+gdc73b217.native8` |
| TorchTPU | `torch_tpu==0.1.1.dev20260928182719` |
| 模型 | Qwen3.8；配置中 `model_type=qwen3_next`，92 层，hidden size 8192 |
| 权重与计算 | `load_format=dummy`，主计算 dtype BF16；权重量化方案由模型配置解析，实际启用 NVFP4 GMM W4A16、block-16 |
| KV / Mamba | attention KV FP8，Mamba SSM BF16，Mamba cache mode `align` |
| KV 预算 | 每 worker `25769803776` bytes，即 24 GiB；显式预算优先于 `gpu_memory_utilization=0.9` |
| 模型最大长度 | 1048576；实际请求输入 131072 token |
| 并发限制 | 每 DP `max_num_seqs=12`，全局 96 |
| 编译档位 | `[12,48,1024]`；12 请求 × (1+3) 个 token 对应 48 档 |
| 请求采样 | temperature=0，由请求提供，不是服务启动参数 |

以下命令针对上述 TPU 形态和软件接口。其他拓扑需要重新设置拓扑与并行参数。dummy 模式仍需提供匹配的模型配置、量化元数据和 tokenizer；无需加载真实权重内容。

代码基线为 `alibaba/release-0.0.1` 的提交 `dc73b2174626728600110e3b55285d47e7e522c3`。本目录从先前复现资料迁入，仅补充许可证声明、按仓库规范格式化并清理未使用变量；不改动服务或压测逻辑。文中 connector 的 SHA256 指格式化前的实测文件。

## 2. 文件与参数准备

将本目录放在四个服务运行环境中。激活各自安装了上述软件的 Python 环境，准备 `numactl`。四端使用相同的模型配置和代码。

| 文件 | 作用 |
|---|---|
| [env.sh](env.sh) | 全部显式配置的性能与分布式环境变量，地址与依赖路径参数化 |
| [serve.sh](serve.sh) | 完整原生启动命令，支持 `normal` / `synthetic` |
| [engine-config.json](engine-config.json) | 参数清单；含路径占位符，供核对，不直接交给 vLLM 解析 |
| [synthetic_decode_connector.py](synthetic_decode_connector.py) | 当前合成 KV connector |
| [synthetic_decode_connector_measured.py](synthetic_decode_connector_measured.py) | 产生本次实测数据的 connector 快照 |
| [test_synthetic_decode.py](test_synthetic_decode.py) | 合成 KV 的调度契约、初始化与日志测试 |
| [benchmark.py](benchmark.py) | P/D 固定长度、按 DP 分配请求的流式压测客户端 |
| [benchmark_window.py](benchmark_window.py) | 正常 prefill 基线的共同 decode 窗口测试 |
| [make_prompts.py](make_prompts.py) | 生成格式兼容的替代 token-ID 数据集 |

四个环境都填写相同顺序的地址列表；它表示 host rank 0、1、2、3。地址必须能用于进程间通信及 HTTP 请求。每个环境使用自身的模型目录和 Python 环境路径：

```bash
export NATIVE_HOST_IPS='<host0-address>,<host1-address>,<host2-address>,<host3-address>'
export MODEL_PATH='/path/to/model-config-and-tokenizer'
export PYTHON_BIN='/path/to/venv/bin/python'
export VLLM_BIN='/path/to/venv/bin/vllm'
# 依赖库路径按本地安装选择；下面是常见 Linux 路径。
export TCMALLOC_LIBRARY='/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4'
# 可选：覆盖自动发现的 libtpu 路径与编译缓存目录。
# export TPU_LIBRARY_PATH='/path/to/site-packages/libtpu/libtpu.so'
# export VLLM_CACHE_ROOT='/path/to/compile-cache'
```

使用干净的运行环境，避免继承与本配置冲突的旧服务变量。无需手工设置 worker 的 `RANK/LOCAL_RANK/WORLD_SIZE`，由 vLLM 分配。

| host rank | DP start rank | 承载的 DP |
|---:|---:|---|
| 0 | 0 | 0、1 |
| 1 | 2 | 2、3 |
| 2 | 4 | 4、5 |
| 3 | 6 | 6、7 |

四端共同启动同一种模式，各端仅改变第一个参数：

```bash
# 各 host 分别将 0 替换为自己的 host rank：0、1、2、3。
bash serve.sh 0 synthetic
# 正常 prefill 模式则使用：
# bash serve.sh 0 normal
```

`nnodes=1` 是本次使用的每 host 本地 executor 配置；跨 host 的 DP 联合由 DP 地址、端口和 start rank 配置完成，不要把这里机械改成 4。端口沿用本次配置，复现时保证在所选环境中可用。

## 3. 完整环境变量

下面是 `env.sh` 的完整内容。硬件与性能变量保持本次取值；地址、模型目录、解释器、库目录和缓存目录参数化，未照搬任何特定环境的文件路径。`PATH` 使用已激活的环境，`LD_LIBRARY_PATH` 补入当前 torch 的库目录并保留环境原有依赖目录。

启动父进程设置了 `LD_PRELOAD` 和 `OMP_NUM_THREADS`；实际 worker 的环境审计中没有这两个键，因此不能据此宣称所有 worker 均使用了相同 allocator 或 OpenMP 配置。

```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Source in the container: source ./env.sh HOST_RANK
HOST_RANK=${1:?Pass host rank 0, 1, 2 or 3}
: "${NATIVE_HOST_IPS:?Set four comma-separated host addresses in rank order}"
: "${MODEL_PATH:?Set the model configuration and tokenizer directory}"
IFS=',' read -r -a HOST_IPS <<< "$NATIVE_HOST_IPS"
if [[ ${#HOST_IPS[@]} != 4 || ! "$HOST_RANK" =~ ^[0-3]$ ]]; then
  echo 'Expected four host addresses and a host rank in 0..3' >&2
  return 2
fi
export POD_IP=${HOST_IPS[$HOST_RANK]}
export MASTER_ADDRESS=${HOST_IPS[0]}
export VLLM_HOST_IP="$POD_IP"
export TORCH_TPU_DP_MASTER_ADDR="$MASTER_ADDRESS"
export PYTHON_BIN=${PYTHON_BIN:-python3}
export VLLM_BIN=${VLLM_BIN:-$(command -v vllm)}
export DP_START_RANK=$((HOST_RANK * 2))
NATIVE_RUN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NATIVE_RUN_DIR
# Let the worker launcher assign distributed ranks.
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export LD_PRELOAD=${TCMALLOC_LIBRARY:-/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4}
export TZ=Asia/Shanghai
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=224
export MALLOC_ARENA_MAX=16
export LIBTPU_INIT_ARGS='--xla_tpu_use_dynamic_smem_negotiation=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=67108864 --xla_tpu_enable_diagnostic_coordinator=false --tpu_core_dump_directory=/dev/shm/tpu_core_dumps --tpu_core_dump_arch_states=TPU_CORE_DUMP_ARCH_STATE_MINIMAL,TPU_CORE_DUMP_ARCH_STATE_TC,TPU_CORE_DUMP_ARCH_STATE_SCS,TPU_CORE_DUMP_ARCH_STATE_SC_TEC,TPU_CORE_DUMP_ARCH_STATE_SC_SGE --xla_tpu_report_hlo_details_in_core_dump=true --xla_tpu_use_enhanced_launch_barrier=true'
export TPU_ACCELERATOR_TYPE=tpu7x-32
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,4
export TPU_TOPOLOGY=2x2x4
export TPU_TOPOLOGY_WRAP=false,false,false
export TPU_SKIP_MDS_QUERY=true
export TPU_VISIBLE_CHIPS=0,1,2,3,4,5,6,7
export TPU_MULTIHOST_BACKEND=mp
export TPU_PROCESS_PORT=8471
export TPU_SIDE_CHANNEL_PORT=19600
export TPU_PREMAPPED_BUFFER_SIZE=4294967296
export TPU_VLLM_SKIP_DYNAMIC_SMEM_NEGOTIATION_FLAG=1
export TPU_GDN_CONV_STATE_TILE_PAD=1
export TPU_GDN_CONV_QK_PAIR_LAYOUT=1
export TPU_PARALLEL_PRECOMPILE=1
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export XLA_FLAGS=--xla_enable_hlo_sharding_v3=false
export USE_BATCHED_RPA_LONGCTX=1
export TPU_ROPE_CACHE_TRUNCATE=1
export VLLM_TARGET_DEVICE=tpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0
export VLLM_PLUGINS=torchtpu,torchtpu_layers
export TORCH_TPU_DP_MASTER_PORT=29750
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_HOST_IP="$POD_IP"
export PYTHONPATH="$NATIVE_RUN_DIR"
# Resolve package locations from the active Python environment.
export TPU_LIBRARY_PATH=${TPU_LIBRARY_PATH:-$("$PYTHON_BIN" -c 'from importlib.metadata import distribution; print(distribution("libtpu").locate_file("libtpu/libtpu.so"))')}
TORCH_LIBRARY_DIR=$("$PYTHON_BIN" -c 'from importlib.metadata import distribution; print(distribution("torch").locate_file("torch/lib"))')
export LD_LIBRARY_PATH="$TORCH_LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$NATIVE_RUN_DIR/compile-cache}
export TORCH_TPU_TIER2_COMPILATION_CACHE="native_decode_${POD_IP}"
```

## 4. 完整服务启动命令

下面是 `serve.sh` 的完整内容，与参数清单逐项对应。`quantization=null` 表示不显式传 `--quantization`，由模型配置解析。传入 `block_size=384` 后，当前 hybrid 模型实际解析出的 scheduler block size 为 2176；不要把启动参数直接当成最终物理布局。

```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
# shellcheck source-path=SCRIPTDIR
source "$(dirname -- "$0")/env.sh" "${1:?Pass host rank 0..3}"
MODE=${2:-synthetic}
case "$MODE" in
  synthetic)
    CACHE_ARGS=(--no-enable-prefix-caching --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_connector_module_path":"synthetic_decode_connector","kv_role":"kv_both","kv_connector_extra_config":{"synthetic_decode_benchmark":true}}') ;;
  normal)
    CACHE_ARGS=(--enable-prefix-caching) ;;
  *) echo "Mode must be synthetic or normal" >&2; exit 2 ;;
esac
cd "$NATIVE_RUN_DIR"
exec numactl --interleave=all "$VLLM_BIN" serve \
  --pipeline-parallel-size 1 \
  --data-parallel-size 8 \
  --data-parallel-size-local 2 \
  --nnodes 1 \
  --distributed-executor-backend mp \
  --data-parallel-backend mp \
  --cpu-distributed-timeout-seconds 14400 \
  --master-port 29500 \
  --safetensors-load-strategy lazy \
  --enable-ep-weight-filter \
  --dtype bfloat16 \
  --language-model-only \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.9 \
  --cp-kv-cache-interleave-size 256 \
  --attention-backend CUSTOM \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype bfloat16 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --no-disable-hybrid-kv-cache-manager \
  --async-scheduling \
  --no-enforce-eager \
  --additional-config '{"tpu_online_fp8":{"enabled":true,"weight_scheme":"per_channel","targets":["model.layers.*.self_attn.qkv_proj","model.layers.*.self_attn.o_proj","model.layers.*.linear_attn.in_proj_qkvz","model.layers.*.linear_attn.in_proj_ba","model.layers.*.linear_attn.out_proj","model.layers.*.mlp.shared_expert.gate_up_proj","model.layers.*.mlp.shared_expert.down_proj"],"exclude":["model.layers.*.mlp.experts*"]}}' \
  --seed 42 \
  --generation-config vllm \
  --mamba-cache-mode align \
  --load-format dummy \
  --hf-overrides '{"max_position_embeddings":1048576,"rope_parameters":{"partial_rotary_factor":0.25,"rope_theta":10000000,"type":"yarn","rope_type":"yarn","factor":4,"original_max_position_embeddings":262144}}' \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --block-size 384 \
  --tensor-parallel-size 4 \
  --data-parallel-hybrid-lb \
  --prefill-context-parallel-size 1 \
  --decode-context-parallel-size 1 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 12 \
  --compilation-config '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[12,48,1024],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --data-parallel-address "$MASTER_ADDRESS" \
  --model "$MODEL_PATH" \
  --served-model-name qwen38-native-dummy \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-rpc-port 29650 \
  --enable-prompt-tokens-details \
  --kv-cache-memory-bytes 25769803776 \
  --data-parallel-start-rank "$DP_START_RANK" \
  "${CACHE_ARGS[@]}"
```

## 5. 实际新增的代码

没有修改已安装的 `vllm` 或 `vllm-torchtpu` 源码。新增的是独立 Python 模块，通过 `kv_connector_module_path` 扩展接口加载。

合成模式增加以下配置，同时关闭真实 prefix caching：

```json
{
  "kv_connector": "DecodeBenchConnector",
  "kv_connector_module_path": "synthetic_decode_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "synthetic_decode_benchmark": true
  }
}
```

`SyntheticTPUDecodeConnector` 继承内置 `DecodeBenchConnector`。模块将该类导出为 `DecodeBenchConnector`，使平台仍走已有的 benchmark connector 分支；外部模块路径的解析优先于内置 registry。

| 代码位置 | 实际行为 |
|---|---|
| `__init__` | 要求 `load_format=dummy` 且显式设置 `synthetic_decode_benchmark=true` |
| 继承的 scheduler 逻辑 | 首次调度返回 `num_tokens - num_computed_tokens - 1` 个外部已计算 token；对 P131072 即 131071，只留下最后一个 prompt token |
| 正常 hybrid KV allocator | 为请求真实分配各缓存组的 block，并保留 MTP scratch state 的正常生命周期；不同请求没有被强制指向同一小块缓存 |
| `register_kv_caches` / `register_runner` | 跳过通用逐层填充，改为取得 TPU unified FA/Mamba raw pool；不支持该布局时直接报错 |
| `start_load_kv` | 首批请求执行前逐个 raw tensor 清零并同步，仅执行一次；后续请求使用已有的任意数值，不再清零，以免覆盖活跃请求的共享底层存储 |
| `update_state_after_alloc` | 记录请求 prompt 长度、合成 token 数和各组 block 数 |
| `build_connector_meta` | 记录前若干步和新请求加入时的请求数、scheduled token 数和每请求最大 scheduled token 数 |

这是“模拟已经存在历史 KV”，不是把真实 prefix-cache 查找强行改为成功。服务仍计算完整逻辑上下文上的 attention；首个 prompt token 的尾部计算、后续 decode 和 MTP 正常执行。合成 KV 不保留输入语义，并可能改变专家路由及 MTP 接受情况，因此结果属于合成负载性能数据。

### 实测快照与日志修正

实测快照使用 `init_logger(__name__)`，INFO 消息被运行环境过滤。完成压测后，仅将 logger 名称改为 `vllm.synthetic_decode_connector`，让它继承 vLLM 的 INFO 配置；调度和 KV 操作没有变化。当前模块包含该修正，实测快照单独保留。该日志修正未重新跑一轮吞吐测试；已在四个运行环境分别通过 6 项单元测试，覆盖日志可见性、dummy 限制、显式启用限制、长 prompt token 数、四缓存组元数据和只清零一次。

```bash
"$PYTHON_BIN" test_synthetic_decode.py
```

修正后的启动应能看到 `SYNTHETIC_DECODE_POOL_READY`、`SYNTHETIC_DECODE_ADMIT` 和 `SYNTHETIC_DECODE_STEP`。本次是否跳过 prefill 的实测证据来自请求的 cached token 数和 TTFT，而非这些被过滤的日志。

## 6. 压测数据与请求构造

实测 token-ID 数据集共有 128 条，每条 `token_ids` 长度为 262144，另有 `suffix_ids`。其 SHA-256 为：

```text
c48fdd64aeada4a9f9a3baf9a7e9783b9640abca0a51e32ba523728102f4e7d5
```

原数据集不随本目录提交。若持有该文件，可直接指定；否则用下列命令生成格式兼容的替代数据。替代数据不是上述 SHA 对应的数据，路由与吞吐不保证逐数值一致。

```bash
"$PYTHON_BIN" make_prompts.py --model "$MODEL_PATH" --output ./prompts.json
export NATIVE_BENCH_DATASET="$PWD/prompts.json"
export NATIVE_BENCH_OUTPUT_DIR="$PWD/benchmark-results/run-001"
```

客户端依赖 `aiohttp`；窗口测试另用 `pyzmq`、`msgspec` 和 Linux `ss`。生成替代数据需要 `transformers` 和对应 tokenizer。

每个 DP 取 12 条不同请求。全局 DP `dp` 的 slot `s` 使用数据集索引 `dp*16+s`，其中 `dp=0..7`、`s=0..11`。实际 prompt 构造为：

```python
prompt = item['token_ids'][:131072-len(item['suffix_ids'])] + item['suffix_ids']
```

请求发往 `NATIVE_HOST_IPS[dp//2]:8000/v1/completions`，HTTP header `X-data-parallel-rank` 为本 host 的本地索引 `dp%2`，不是全局 DP 编号。每 host 的 24 条请求按 0.25 秒间隔送入；四个 host 并行。

请求字段如下，`prompt` 为上述整数 token-ID 数组：

```json
{
  "model": "qwen38-native-dummy",
  "prompt": ["此处使用整数 token ID 数组"],
  "max_tokens": 1024,
  "min_tokens": 1024,
  "ignore_eos": true,
  "temperature": 0.0,
  "stream": true,
  "stream_options": {"include_usage": true},
  "return_token_ids": true
}
```

上面 JSON 用文字标记了 prompt 的位置，仅解释协议；实际可执行脚本发送的是整数数组。统计只累加生成输出的 `choices[].token_ids`，不把 `prompt_token_ids` 计入吞吐。

## 7. 合成 KV 模式的执行步骤

1. 四端以 `synthetic` 模式启动，等待所有 `/health` 返回 200。
2. 不发送预热 prompt；直接提交 96 条 P131072/D1024 请求。
3. 检查 96 条请求均成功，且每条恰好生成 1024 token。
4. 检查 `usage.prompt_tokens_details.cached_tokens` 为 131071，确认前 P−1 个 token 被合成提供。
5. 使用所有请求共同生成的时间窗口计算稳态性能。

```bash
# 在可访问四个服务端点、持有数据集的客户端环境执行。
"$PYTHON_BIN" benchmark.py \
  --stage synthetic-c12-r1 \
  --per-dp 12 --slot 0 \
  --input 131072 --output 1024 \
  --arrival-interval 0.25
```

`benchmark.py` 保存每条请求的 token 到达时间、usage 和汇总 JSON。它不自动启动服务，也不自动执行预热。请为每轮选择独立输出目录与 stage 名称。

这里省略的是长 prompt 预热和实际 prefill，模型初始化、图编译、启动 dummy forward 以及首批请求前的 pool 清零仍会执行。

## 8. 正常 prefill 基线的测量方式

正常模式启用 prefix caching，不配置 connector。此前按每 DP 4 条请求分三批对 P131072/D1 做过预热，但后续校验仍出现缓存未命中，因此没有把整轮请求耗时作为纯 decode 数据。

有效基线保留 96 条不同的 P131072 请求，将生成上限设为 8192。所有 96 条请求开始生成后再等待 2 秒，随后累计到每条在共同窗口内至少输出 1024 token，记录结果并主动结束请求。

```bash
# 四端均先使用 bash serve.sh <host-rank> normal。
# 窗口脚本需在 host rank 0 的服务环境运行，以只读订阅本地 DP coordinator。
export NATIVE_BENCH_OUTPUT_DIR="$PWD/benchmark-results/normal-window-001"
"$PYTHON_BIN" benchmark_window.py
```

该基线是“约 P128K 的共同 decode 窗口”，不是 96 条完整 P128K/D1024 请求的耗时统计。窗口开始时实际上下文长度为 131128–132079 token，窗口内各请求又生成 1043–1044 token。脚本的 coordinator 订阅只读，不发送调度指令。

本目录的客户端与实测版本相比，只把数据路径、输出路径和服务地址参数化；窗口脚本的 stage 名称也改为通用名称。请求构造、到达间隔、计数和统计公式保持一致。上述可移植脚本未重新发起一轮 TPU 压测。

## 9. 统计公式与校验口径

固定 D1024 测试：令每条请求首、末 token 到达时间为 `first_i`、`last_i`，共同窗口为：

```text
lo = max(first_i) + 2 秒
hi = min(last_i) - 2 秒
T = hi - lo
N_i = 请求 i 在 (lo, hi] 内收到的输出 token 数
全局稳态吞吐 = sum(N_i) / T
稳态请求平均 TPOT = mean(1000 * T / N_i)
摊销 TPOT = 96 * 1000 * T / sum(N_i)
整条请求 TPOT_i = 1000 * (last_i - first_i) / (输出 token 数 - 首个 chunk 的 token 数)
TTFT_i = first_i - 请求发送时间
```

整轮 elapsed 包含请求到达间隔、TTFT、完整输出和最后的 2 秒 metrics 等待，不能直接解释为纯 decode step 时间。这里的 TPOT 是客户端输出 token 的时间指标，不是设备 main forward 或单步 kernel 耗时。

共同窗口的并发另外通过 DP coordinator 的只读采样核验：各 DP 均为 12 running、0 waiting。固定长度客户端本身不包含该订阅器；已保存的结果 JSON 包含本次采样审计。

本次没有获得可用的 MTP 接受率计数，也没有采集新的设备 profiling；不要据此反推精确 main forward、完整 step 或按真实权重接受率外推吞吐。

## 10. 实测结果

| 指标 | 正常 prefill 后共同窗口 | 合成 KV，P131072/D1024 |
|---|---:|---:|
| 全局稳态吞吐 | 877.8286 tok/s | 879.8899 tok/s |
| 稳态请求平均 TPOT | 109.3608 ms | 109.1046 ms |
| 稳态 TPOT P95 | 未统计 | 109.1069 ms |
| 整条请求平均 TPOT | 不作固定 D1024 口径统计 | 109.7752 ms |
| 共同窗口长度 | 114.0644 s | 104.3062 s |
| 窗口内总输出 token | 100129 | 91778 |
| 每请求窗口输出 token | 1043–1044 | 956–957 |
| 完整请求输出 | 达到窗口目标后主动结束 | 96/96 条各输出 1024 token |
| 每 DP 并发 / waiting | 12 / 0 | 12 / 0 |
| coordinator 有效采样数 | 23 | 22 |
| 每条请求 cached token | 不作为该窗口的前提 | 131071，96/96 条 |
| TTFT 范围 | 不采用此口径比较 | 0.3450–1.8590 s |
| 整轮客户端 elapsed | 不采用此口径比较 | 120.1035 s |

合成 KV 结果相对基线吞吐增加 **0.2348%**。跳过 prefill 显著缩短了进入 decode 的准备时间，但没有显著改变测得的稳态吞吐。这一结果不支持把之前稳态窗口的低吞吐主要归因于缓存未命中；合成值可能影响路由与接受率，因而也不能把它视为严格控制所有变量的性能消融。

结果摘要保存在 [results/synthetic.json](results/synthetic.json) 和 [results/normal-window.json](results/normal-window.json)。为便于分享，摘要未包含具体地址、机器标识、运行目录或原始服务日志。
