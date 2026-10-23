# Raiden PD 完成与错误终态

## TL;DR

Raiden 的响应生命周期现通过 init patch 接入官方 `Scheduler` / `AsyncScheduler`。
四个临时 Raiden scheduler 类已删除，不再配置它们的 `scheduler_cls`。
P 侧 Raiden Stage-3 producer 的 DELTA / CUMULATIVE 请求自动等待发送完成后才结束响应，
无需配置 `additional_config.raiden_pd_send_completion`；D 侧 Raiden consumer 设置
`kv_load_failure_policy="fail"`，
保留按请求身份终止加载失败、输出 `model.KVTDecodeLoadFailure` 的契约。
调度、异步 token 账目、跨 rank 屏障及 KV 回收仍由 vLLM 负责。

## 配置迁移

移除旧 `scheduler_cls`，由 `async_scheduling` 选择官方类：

```python
config.pop("scheduler_cls", None)
config["async_scheduling"] = True  # False 选择官方同步 Scheduler

# D：沿用显式加载失败策略。
config["kv_transfer_config"]["kv_load_failure_policy"] = "fail"
```

P 完成桥由 Raiden Stage-3、`KVTransferConfig.is_kv_producer` 和恰好一个
`TPURaidenConnector` 自动启用，可位于 `TPUMultiConnector` 中。沿用 vLLM 的
producer 能力属性，`kv_producer` 与 `kv_both` 均可启用，包含外层角色为 `kv_both`、
Raiden 子 connector 为 producer 的默认组合；旧配置字段不再读取，可删除。
不具备 producer 能力、其他 transport 或 connector 保持原路径。
完成桥不再额外限制 TP/DP/PP 为 1；拓扑仍须通过
平台与 connector 原有校验，不代表任意 PP×TP×DP 组合均受支持。
每个 DP engine 独立处理请求，沿用 connector 的 `get_finished_count()` 和 vLLM
`KVOutputAggregator` 汇聚该 engine 内的发送完成，不增加跨 DP engine 的屏障。

P 前端必须在收到首个 token/KV handle 后启动 D，并继续读 P 的响应流。
沿用当前 DashLLM 契约：`n=1`、`max_tokens=1`、DELTA 输出，
`DS_LLM_ENABLE_DISAGGREGATED_PREFILL_ABORT=0`。不能先等 P 整条响应结束再启动 D。
完成桥延后 vLLM `RequestOutputKind.DELTA` / `CUMULATIVE` 请求，两者都能提前交付
pending 输出和 KV handle。CUMULATIVE 的调用方应在中间输出到达时启动 D，
不能等待 P 的最终输出；其最终 `RequestOutput` 保留累计 token，
引擎完成通知本身不追加 token。`FINAL_ONLY` 保留原模型终态和 client finished ID，
不生成 Pkg2。OpenAI 非流式请求使用
`FINAL_ONLY`，因此示例代理和容量验收仍可先获取 P 的 KV handle，再启动 D。

D 处理只对 Raiden Stage-3 consumer、`fail` 策略和一个 `TPURaidenConnector`
启用，要求 `TPU_RAIDEN_INLINE_LOAD=0`。其他 connector、transport、role 或
`recompute` 策略保留原路径。启用响应处理时，入口拒绝自定义 scheduler 子类；
官方同步和异步类均支持。

KV consumer 的 `kv_load_failure_policy="fail"` 同时启用容量抢占终止，
无需额外配置，见[容量策略](abort_on_preemption.md)。D 加载失败处理与容量终止
各自管理数据，不共用错误状态。

## 保留的生命周期

- P 首个成功输出保留 token、KV handle 和缓存统计，只清除终止标记。
  `RaidenSendResults` 跨 rank / step 按
  `SEND_FAILURE > UNCLAIMED_TIMEOUT > SUCCESS` 合并；仅在 `finished_sending`
  屏障后发送一次无新增 token 的引擎终态，累计输出由 vLLM OutputProcessor 保留。
  成功为 STOP + `kv_transfer_done=True`，
  失败为 ERROR + `kv_transfer_error`。
  DELTA / CUMULATIVE 的 pending 输出即使命中前端 stop strings 也保持未结束，
  避免前端提前 abort 并取消 KV 注册；`FINAL_ONLY` 仍直接返回原模型终态。
- DP 的 `EngineCoreOutputs.finished_requests` 对等待发送的 P 请求延后到发送终态，
  保留 vLLM DP client 的 abort 路由与 inflight 记账；成功、失败及缺失发送结果均只归还一次。
  worker 使用的 `SchedulerOutput.finished_req_ids` 保持原时序。
- P 收到 abort 时清除等待响应的数据，并通知 worker 将对应注册的到期时间设为 `-1`，
  在下一次轮询复用过期回收。仅取消尚未 claim 的注册；已 claim 的 KV 仍等待 native
  terminal，跨 rank 回收屏障保持不变。
  取消时暂存的 client finished ID 归还原 scheduler 队列；迟到结果不再生成 Pkg2。
- D 先读取 `RaidenLoadFailures`，按 request ID 调用 `finish_requests(FINISHED_ERROR)`，
  再处理本步模型输出，避免迟到 token 泄漏。`WAITING_FOR_REMOTE_KVS` 的 buffer
  仍等 `finished_recving` 后才释放；重复失败、已取消请求不再输出终态。
- 其他 connector 的原生 invalid-block 错误仍经上游处理，只为对应 ERROR 追加原有
  `model.KVTDecodeLoadFailure` metadata。普通 grammar/model 错误不被重新分类。

`core/raiden_pd.py` 包装 `Scheduler.__init__`，按配置给实例绑定必要方法，
并在前端 `OutputProcessor` 的 stop 检查之后保留上述 pending 响应。
官方 `_update_after_schedule()`、`_update_request_with_output()` 和调度算法保持原实现。
worker 与 scheduler 使用同一个配置解析函数决定是否收集/消费 Raiden metadata，
不再通过类名推断能力。不符合上述角色与 connector 条件的实例不绑定响应适配器。

registry 先安装 Raiden init patch，再安装 capacity init patch。D 输出顺序为：

```text
capacity 包装器 → Raiden 失败预处理 → 官方 update
               → Raiden 补加载错误 → capacity 按 finished IDs 补容量终态
```

结构化错误保留到 `RequestOutput.kv_transfer_params`。原生 OpenAI serving 对 ERROR
返回通用推理错误；DashLLM 的错误透传仍依赖 tpu-misc 中已有的可选错误 patch。
本次不增加另一种载荷格式或 adapter 解析路径。

错误响应不是 DMA 已停止的证明。此实现沿用 Raiden 已有终态/超时契约，不根据经过
一段时间推导 buffer 可安全复用，也不增加 controller 轮询。

### 操作与原因分层

对齐解压后的 `pai-vllm/vllm/v1/hybrid_connector/errors.py`：其
`KVTransferFailure.to_request_status()` 保留操作错误码，将原因类别以
`Category: internal.` 放入公开的 `status_message`。当前仅复用这个公开格式，
不引入 pai-vllm 的内部错误对象或额外协议字段。

- D：Raiden 或基类确认 KV 加载失败，但当前通道没有更细原因，返回
  `model.KVTDecodeLoadFailure`，消息为
  `D-side failed to load KV cache data. Category: internal.`。
- P：当前 `SEND_FAILURE` 只表明发送失败，不包含更细原因，返回
  `model.KVTPrefillWriteFailure`，消息为
  `P-side failed to write KV cache data. Category: internal.`。
- P 缺失发送结果：沿用 `model.KVTPrefillWriteFailure`，保留
  `P-side KV cache transfer finished without a sender result.` 的具体描述，
  后接 `Category: internal.`。这是无法确认完成的内部错误，不证明 native 写入失败。
- 已明确的未 claim 超时仍为 `model.KVTPrefillTransferTimeout`；成功、普通模型错误、
  取消及 recompute 均不因原因分类改变行为。

以上只基于已有失败/完成事件产生终态，不把仍在进行的传输直接判为失败，
也不由 `internal` 推导故障一定可重试。原始内部异常保留在日志中，不直接暴露给 client。

## 验证与复现

长期测试保护响应生命周期、缓存统计、结构化错误、跨 rank / step 失败优先、
DP 路由与终态记账、取消后无迟到终态，以及未启用时的旧行为；不依赖 TPU。
使用与服务一致的 vLLM / Raiden 依赖；connector 环境变量测试单独启动进程，
避免 scheduler 配置初始化后 vLLM 环境缓存干扰 monkeypatch：

```bash
python -m pytest tests/test_raiden_pd_scheduler.py \
  tests/test_raiden_pd_decode_scheduler.py tests/test_raiden_pd_async_scheduler.py -q
python -m pytest tests/distributed/test_tpu_connector.py -q
```

2026-09-20 的本机环境为 vLLM `0.27.0+empty`，可直接复现：

```bash
export PATH=/mnt/data/wister_zwz/.cache/tpu-misc-pd-disagg/conda_env_py3.12/bin:$PATH
export LD_LIBRARY_PATH=/mnt/data/wister_zwz/.cache/tpu-misc-pd-disagg/conda_env_py3.12/lib:${LD_LIBRARY_PATH:-}
PYTHONPATH=src python -m pytest \
  tests/test_raiden_pd_scheduler.py \
  tests/test_raiden_pd_decode_scheduler.py \
  tests/test_raiden_pd_async_scheduler.py \
  tests/distributed/test_tpu_connector.py -q
```

同步/异步共用相同的响应契约用例，覆盖同轮及跨轮完成、失败优先、取消、重复终态、
缓存统计和 D 错误分类。入口用例检查官方类选择、按需安装响应适配器，以及保留官方
异步 token 处理方法。
测试不执行 TPU；这些证据不能替代异步 PCP4/DP4 的模型端到端验收。

## 历史验证（旧 scheduler 版本）

下述记录对应各自标注的旧版本，不作为本次 init patch 迁移的端到端证据。

### 2026-09-08：同步/异步入口 CPU 验证

基于 vllm-torchtpu `2721e908`、本机 vLLM `0.27.0`，上述四个测试文件共
192 项通过、无跳过；16 条依赖弃用/环境警告与修改前基线一致（基线 177 项通过）。
异步入口及 connector 接入用例先在未实现版本失败，再通过新增类与最小接入修改通过。
ruff、isort、修改的四个生产文件 pyrefly 检查及 `git diff --check` 通过。

在本机独立 worktree 中复现，无需启动 TPU 或修改运行中的镜像：

```bash
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PJRT_DEVICE=CPU \
TPU_SKIP_MDS_QUERY=true \
PYTHONPATH=/mnt/data/yanzecheng_yzc/dashllm-pkg-20260906/pinned-runtime:src \
/mnt/data/miniconda3/envs/torch_tpu/bin/python -B -m pytest \
  tests/test_raiden_pd_scheduler.py tests/test_raiden_pd_decode_scheduler.py \
  tests/test_raiden_pd_async_scheduler.py tests/distributed/test_tpu_connector.py \
  -q -p no:cacheprovider
```

这次结果仅覆盖调度器与输出契约；未声称完成异步模型端到端或真实传输故障注入。

错误验收必须检查 `RequestOutput.finished=True`、`new_token_ids=[]`、存在且仅存在 `kv_transfer_error`，不能只检查 `FinishReason.ERROR`。当前离线用例已覆盖 native failure、未 claim 超时、缺失结果、跨 rank / step 优先级，以及真实 OutputProcessor 的保留行为；历史CPU验收未注入真实TPU传输故障；本次hybrid加载失败的实机验证见下节。

2026-09-07 `internal` 修订基于 `ecc5980a`，按上面固定依赖命令验证：172 项通过，
无跳过；ruff、pyrefly 和 `git diff --check` 通过。tpu-misc 使用原版错误 patch
重新生成测试 overlay，错误透传及拆包回归 40 项通过。本次未重启服务，不构成
新增原因消息的 TPU 在线端到端验收。

真实 PCP4 / DP4 验收必须在收到 pkg1 后启动 D，同时继续读 P；不能先等 P 流结束再启动 D。最小检查是一冷两热三个相同的 20000-token 请求：每次收到一个握手 pkg0、一个 prefill pkg1、一个 `kvt_transfer_done=True` 的 STOP pkg2，且 D 正常生成 8 tokens。当前 block geometry 下 P 缓存命中预期为 0 / 16384 / 16384。

当前机器的独立实验目录：

```text
/mnt/data/yanzecheng_yzc/dashllm-logical-20260906/pkg2-vllm-only-Ikmc4R
```

该目录的 `launch.py` 复用既有 canonical 配置及未修改的 DashLLM overlay，排除旧 vllm-torchtpu 观测 overlay，仅 P 选择新 scheduler。`evidence/*-engine.json`、`*-environment.json` 记录实际启动参数，`probe_pkg2.py` 记录逐包 JSON 和验收断言。环境为单机 8 TPU、Qwen3.5-35B-A3B-FP8，P PCP4 / D DP4、max model len 32768、block size 4096、每角色 offload 64 GiB；基线 vllm-torchtpu `10565640`、vLLM 0.27.0。

服务已就绪时，在该实验目录执行：

```bash
/mnt/data/miniconda3/envs/torch_tpu/bin/python probe_pkg2.py
/mnt/data/miniconda3/envs/torch_tpu/bin/python probe_pkg2.py --highlevel
```

第一条要求 P 尚未处理脚本中的固定前缀；复测请使用新前缀（同样用于后续热请求）或重新启动服务清空缓存。第二条使用标准 Rust PD client，需紧接第一条运行以复用热缓存。逐包证据和摘要位于 `evidence/`；测试脚本是本次开发验收工具，不要求在其他机器按绝对路径部署。

### 本次结果（2026-09-07，Asia/Shanghai）

- 10:57:30–10:57:34：冷请求及两次热请求全部通过，P 各返回三个包且只有一个成功 pkg2；pkg1 缓存命中为 0 / 16384 / 16384；D 各输出 8 tokens，以 LENGTH 结束。证据：`evidence/pkg2-summary.json` 及其中列出的逐包文件。
- 10:57:50–10:57:51：标准 Rust PD client 请求 `pkg2-standard-warm-59295d29aabe4ef7a7024d271f07edd7` 正常完成，客户端日志明确出现 `[PD] KVT-CONFIRMED ... prefill_first=true`，随后报告 `tokens=8`、`cached=16384`。标准 API 隐藏内部控制包，因此其响应文件只记录用户输出，不能用该文件的包数代替原始 P 三包计数。证据：`evidence/standard-summary.json`、对应响应文件，以及 P 日志中的同 request id `Raiden P send terminal ... success=True`。
- CPU 回归 167 项通过；ruff、pyrefly 和 `git diff --check` 通过。此次只验证功能，不将单次请求耗时作为性能结论。

这证明了当前同步 P PCP4 / D DP4 部署的成功完成链路；不代表失败、已 claim 后 hang、完整取消清理或 D 全缓存命中场景已经端到端验收。新增错误终态目前只有 CPU 契约验证，后续应在应用 tpu-misc 现有错误 patch 的独立 overlay 中注入真实失败，并验证到 client。

### 2026-09-11 hybrid 加载失败回归

基于aios/dev加本地修复、R14仅替换vllm-torchtpu，在PCP4/DP4、异步scheduler、
非inline加载上验证：3次不存在的P request_id注入均只让对应D请求返回
`model.KVTDecodeLoadFailure`（0个token），每次随后正常请求成功生成8个token。
随后99个请求协议完成，27项短前缀/STOP检查通过；服务PID未改变，8卡已释放。
这是明确缺失注册的pre-arm故障验证，不等同于所有uncertain超时的DMA安全验证。

本次新增25个长期case，覆盖请求身份上报、无FA block、native失败与同轮取消清理、
旧runner接口和聚合、真实scheduler延迟释放、重复终态、迟到token及入口能力边界。
源码相关测试344 passed；R14镜像内PD/connector测试245 passed。
原有并发输出差异仍存在（40个目标请求中4个在第10个token多空格），不计作一致性通过。

本机复现脚本、固定镜像/补丁摘要、原始错误包及完整验证记录：
`/var/tmp/r14-hybrid-load-failure-20260911/README.md`。
该目录的`run_matrix.py`先执行3轮错误/恢复检查，再运行99请求与27项STOP/短前缀；
重新运行需独立输出目录、8个空闲TPU及对应R14模型/镜像环境。
