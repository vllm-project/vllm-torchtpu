# Qwen3.5 CI 配置与 MTP 精度检查

## TL;DR

两组 CI 从同一份 `scripts/vllm/integration/configs/qwen35_disagg.sh` 读取模型配置，
默认拓扑为 PCP4→TP2/DP2 和 PCP8→TP2/DP4，分别保留 8/16 devices 的资源规模。
配置以 R11 发布记录为基准；当前源码的 CUSTOM attention 已合并 longctx batched RPA，
无需恢复已删除的 `USE_BATCHED_RPA_LONGCTX`。

本次仅保留部署配置对齐。PR 新增的 Qwen3.5-4B dense 逐 draft logits、K3 和
长上下文测试已撤回，main 原有测试保持不变。PCP8 的新增用例、logits 采集和
验证暂不实施。

已在本机独立 venv 中以 `Qwen/Qwen3.5-35B-A3B-FP8` 跑通 PCP4→TP2/DP2，
完成两轮各 8 条自然语言请求并冻结第一轮实际逐 draft logits。4 条 512-token 输入
用例（含 1024-token 长输出）的两轮 P/D logits 逐字节一致，已单独登记固定回归
参考子集；4 条 64K 输入用例复跑输出不同，保留采集证据，暂不作为稳定回归基准。
这不是完整 CI 验收，也没有与独立高精度 reference 对齐。

## 配置与证据

初始实现基线：`1c25edbb1`；PR 移植到 `origin/main` 的 `250958522`。
移植仅包含本次 CI 和测试差异，保留 main 新增的 MTP logprobs/DSpark 用例，
并保留单机 CI 原有 `MIN_MTP_ACCEPTANCE_RATE=0.70`；新 TP2/DP2 geometry
仍需实机重新测量此阈值。直接读取的 R9 镜像 ID 为 `dc4bd2ba1e61`，配置路径为
`/opt/dashllm-venv/lib/python3.12/site-packages/dashgen/configs/qwen3.7-plus/0522-thinking-re-nvfp4-inputscale/l20g{,_prefill,_decode}.config`。
R11 依据工作区 `tpu-misc/dashllm/docs/docker-native-r11-release-20260928.md`，
本次没有直接读取 R11 镜像；发布记录将 decode token budget 和最大编译档位提高到 768。

| 项目 | Prefill | Decode |
| --- | --- | --- |
| MTP | K1 | K3 |
| TP | 1 | 2 |
| PCP / DP | PCP4 或 PCP8 | DP2 或 DP4 |
| block size | 768 | 2304 |
| max batched tokens | 32768 | 768 |
| max sequences | 128 | 192 |
| memory utilization | 0.75 | 0.9 |
| compile sizes | 512,1024,2048,4096；PCP4 加 8192 | 32,64,128,256,384,512,768 |

共同设置包括 HND、FP8 KV、BF16 SSM state、CUSTOM attention、async scheduling、
prefix caching、align mamba cache 和 unified block pool。MoE 使用 fused EP、
sharded plan、128 requantize block、32768 onehot threshold 和 gather reduce v3。
Prefill 的 Raiden destination shards 为 2。

PCP4 的 8192 档位是本拓扑的必要差异：单请求 32768 tokens、interleave=256 时，
`pcp_local_token_counts` 得到 `[8192] * 4`，仅有 4096 bucket 会无法选择编译档位；
PCP8 则得到 `[4096] * 8`。这不是所有 PCP 输入的通用均分公式。

入口只保留拓扑、端口、connector 和测试场景差异。Pipeline step key 和脚本名保留，
因此旧文件名中的 `p8d8` 不再描述默认 decode DP。角色环境变量可覆盖公共默认值，
显式 `PREFILL_SPECULATIVE_CONFIG=''` / `DECODE_SPECULATIVE_CONFIG=''` 可关闭 MTP。
独立使用 launcher 时保持原有默认值；其他多机消费者（如 DSv4）保留原外层默认值。

## 本机基准的执行范围

使用本 PR 的 PCP4→TP2/DP2 配置，模型为 `Qwen3.5-35B-A3B-FP8`，P K1 / D K3。
测试输入使用自然语言。冻结该轮每个 draft 的全词表 logits，并记录原文、精确
input token IDs、request/round/draft step、draft 输入与 rejection 元信息、模型文件
SHA256、运行库版本、代码 commit、实际启动参数和数据 SHA256。

采集必须来自实际 TPU 模型运行；历史接受率、最终输出文本、CPU 参考重算或
合成数据均不能代替该轮 logits。只有真实运行完成且数据完整性检查通过后，
才生成候选基准 manifest；未运行或中断的采集不能标记为基准完成。
冻结一次结果可用于后续回归，但数值容差和基准质量仍需单独验证。

PCP8 因设备不足，本次不增加采集与 logits 检查，不套用 PCP4 的基准。
撤回的 4B 测试结果不能作为 35B MoE 的验证证据。

## 2026-09-29 本机实测

代码为 PR commit `b66556fae1ef46905b7763009890d20c31c41b88`，生产 Python 未修改。
运行在宿主机独立 Python 3.12.14 venv，按该分支 `pyproject.toml` 安装：

- vLLM `0.29.0+empty`，源码 revision `98dff2a81d747d1dba01a47f939f48c3526d4206`。
- torch `2.13.0+cpu`、torch-tpu `0.1.1.dev20260924085621`。
- JAX/JAXLIB `0.10.2`、libtpu `0.0.47`、tpu-sync-torch `0.0.1.dev20260923040928`。
- 模型本地目录 `/mnt/data/yanzecheng_yzc/models/Qwen3.5-35B-A3B-FP8`。
- 上表完整默认配置，未缩小 max length、batch 或编译档位。

采用测试专属 worker extension 在服务预热完成后开启采集；compiled head 返回该次
实际 draft hidden states 对应的完整 logits 和原 greedy token，包含 D 的 K3 每一步。
所有 worker（含 DP dummy）使用同一 head，dummy 不写入记录；生产源码没有采集开关。
单机 logical-client 实验工具将 8 个物理设备分为 P=`0,1,6,7`、D=`2,3,4,5`；
八 rank 映射和 compiled collective 探针通过。这不等价于多机 PCP8 的验证。

| 自然语言场景 | 请求数/轮 | 输入 tokens | 输出 tokens | 两轮结果 |
| --- | --- | --- | --- | --- |
| 短输入/短输出 | 2 | 512 | 64 | P/D 完整 logits、draft token、最终文本一致 |
| 长 prefill | 2 | 65536 | 128 | 请求与采集完成；最终文本复跑不同 |
| 长 decode | 2 | 512 | 1024 | P/D 完整 logits、draft token、最终文本一致 |
| 长 prefill + 长 decode | 2 | 65536 | 1024 | 请求与采集完成；最终文本复跑不同 |

请求使用 temperature=0、seed=42、ignore_eos=true，同组两条并发，按表中顺序执行；
prefix cache 保持开启，第二轮复用同一服务及缓存历史。首先运行了现有 launcher 的
quick probe（HTTP 200、输出 `4`），因此这不是干净缓存的重复独立实验。

第一轮保存 9,946 个 tensor 文件，含副本共 9,954 条 request-step，约 13 GiB。
去除 PCP/TP 副本后，P 为 18 个 K1 step、D 为 4,941 个 K3 step；完整词表为 248,320。
两轮各自均通过真实 tensor 加载、shape、finite、argmax、K-step 完整性、文件 SHA256
以及同一请求的 PCP/TP 副本逐字节一致性检查。
4 条短输入用例跨轮的 P 4 step + D 2,364 step 也逐字节一致，登记为固定参考子集。
跨轮通过 response ID 关联请求，按请求内 round/step 排序比较，不比较随机 request ID。

### 已确认事实与未闭环问题

64K 用例在相同请求 payload 的第二轮中输出不同，最早分别在输出字符 14/33 处出现
差异；第一轮 128/1024 输出长度的两组同输入请求也有前缀差异。现有证据尚不能区分
缓存历史、调度或数值路径的影响，不能将其中任一项认定为根因。下一步需要固定请求
路由/批次，并对比清缓存与复用缓存的独立重复实验；本 PR 不顺带修改运行时。

本轮只是实际运行数据的固定参考，没有独立高精度 oracle。采集 D2H 会改变时序，
不用于性能或无竞态结论；rank-local draft 快照也不是完整 target/KV replay 契约。
原 CI 的 0.70 acceptance gate 保留，本轮自然语言采集未运行整套 CI，不能宣称该门槛
已实测通过。PCP8 未实测，未新增相关 logits 逻辑。

### 环境排查证据

R9 镜像旧 torch-tpu 缺少 main 所需 `get_pallas_mesh`，该尝试停止后改用上述 venv；
新版本导入检查和 `pip check` 通过。registry 返回 401 时使用本机精确版本原始 wheel，
记录 SHA256。宿主机共享编译缓存权限及 Python 基础环境 libstdc++ 搜索路径修正后，
探针和服务启动通过。完整首次编译超过 launcher 30 分钟等待上限，保留服务进程继续
等待后启动 proxy，未通过缩减配置或跳过预热绕过。上述环境接入调整仅在实验脚本中。根据此次超时证据，PR 另将公共 CI 的
`STARTUP_TIMEOUT_S` 默认设为 3600 秒，单机 launcher 改为按该 deadline 等待；
显式覆盖（包括 Kubernetes 既有 5400 秒）仍生效，独立 launcher 保留 1800 秒默认。
这部分只有健康等待变化，不改变本轮已采集模型的推理参数。

## 固定数据与复现

实验目录（下文 `EXP`）为：

```text
/mnt/data/yanzecheng_yzc/tpu-misc/tmp/pcp4-mtp-venv-20260929
```

第一轮只读数据位于 `results/capture-03/golden-01`；第二轮证据为同级 `repeat-02`。
固定参考子集为 `results/capture-03/REGRESSION_BASELINE.json`，其 SHA256：

```text
567fe728f4a4d8aaa1172cfe8c72948dd15211b4a2526f2c37aefa524202d1b8
```

第一轮全量 `golden-01/BASELINE.json` 的 SHA256：

```text
c31e15f72f187a9fc1289c63716167502548eda98e48b6440ec651b6c99b3865
```

全量 manifest 标记 `frozen_candidate`，显式记录长输入复跑差异。参考子集只引用其中
4 条短输入用例，不将 64K 数据当作稳定 golden。数据保存在本机，未上传制品库，
本 PR 没有新增自动下载、重录或 logits 比较的 CI 步骤；后续接入必须按 manifest 固定
模型、依赖、输入和基准版本，禁止在测试运行时覆盖基准。

`golden-01/provenance/` 含采集、校验、冻结工具、logical-client 工具、启动命令、
模型文件清单、pip freeze 和依赖清单。安装入口 `install-venv.sh`、启动入口
`run-servers.sh` 及超时后续接入口 `resume-proxy.sh` 位于实验目录。
新环境先复制实验工具与精确 wheel/模型，按路径调整脚本，并核验对应 SHA256；
不能仅根据最终文本重新构造 logits。使用本机环境复核：

```bash
EXP=/mnt/data/yanzecheng_yzc/tpu-misc/tmp/pcp4-mtp-venv-20260929
cd "$EXP/results/capture-03"
sha256sum -c REGRESSION_BASELINE.sha256
(cd golden-01 && sha256sum -c BASELINE.sha256)
# 读取真实 tensor，复查全部逐步记录和副本；写到可写的外部报告。
PYTHONPATH="$EXP/tools" "$EXP/venv/bin/python" - <<'PYTHON'
import json
from pathlib import Path
from validate_capture import validate_run
report = validate_run(Path("golden-01/ranks").resolve())
Path("revalidated.json").write_text(json.dumps(report, indent=2))
PYTHON
```

重跑时先确认设备空闲，复制 `run-servers.sh` 并修改副本内硬编码的 `RUN_DIR`
为新目录，再运行副本；仅在外部设置环境变量不会覆盖原脚本的目录。若使用
`resume-proxy.sh`，也须在副本中修改硬编码 `run_dir`。为冷启动预留超过 30 分钟
的等待。服务就绪后用新的输出目录调用
`bash "$EXP/run-capture.sh" /absolute/path/to/new-capture`，不复用只读基准目录。

## 其他验证

配置此前通过 6 组 launcher 参数检查和 4 组 Docker/SSH 环境转发检查。
撤回 dense 测试后，`tests/` 与基线 main 完全一致，原 57 个设备测试收集通过；
适用 pre-commit hooks 通过。这些静态检查与上述本机实验分别记录，均不代表完整 CI
或 PCP8 精度验收。
