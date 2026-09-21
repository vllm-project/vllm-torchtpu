# Dense expert MoE：设计与性能验证

本文记录 2026-09-21 在 TPU7x 上的实验实现、布局优化与实际模型压测结果。
基于 main 提交 `eaf9756f357564f886d8597528383a57e09fb3bf`，
模型为 `Qwen3.5-397B-A17B-FP8`。实现保持原 `fused_moe_func` 入口，默认关闭。

## 动机与边界

小 batch decode 的 routed MoE 需要访问大量专家权重，但每个专家实际接收的 token 很少。
原流程包含 top-k、排序及 token 重排、GMM1、SiLU、GMM2、输出还原与通信。
此前真实服务 trace 中，两个 GMM 合计约 140.94 μs，而整层 MoE 约 311.12 μs。
非 GMM 部分包含路由、重排、shared expert 和通信，不能全部归为排序开销。

实验思路是让每个命中的本地专家计算全部 token，用额外 MXU 运算换取更简单的数据流。
关键约束是：同一个专家只处理一次，且不保存所有专家的输出 tensor。
这并不意味着当前整个 decode 已被证明是纯 HBM 带宽瓶颈；
布局实验表明，VPU 数据重排和 VMEM 临时值也会显著影响性能。

## 算法与数据流

对于本机配置，每个 DP 输入 64 个 token，TP2 持有副本。
DP all-gather 后，每个 EP rank 处理 256 个 token；512 个 routed expert
按 EP8 分配，每 rank 64 个本地专家，top-k 为 10。

1. 保留外部 router projection、top-k 和 EP 专家 ID 映射。
2. 将 top-k 展开为 `[token, local_expert]` 路由系数与 active bitmap。
   非本地专家和零权重路由被屏蔽；重复 expert ID 的系数相加。
3. 在 SMEM 中把 active bitmap 压缩为去重专家列表。
   实现按本地 expert ID 遍历，语义等价于逐 token 遍历时记录已处理的专家。
4. hidden 加载到 VMEM，按 512-wide block 动态量化为 FP8，在专家间复用。
5. 每个命中专家的 gate/up、down 权重及 scale DMA 一次，计算全部 token。
   双缓冲提前发起下一个专家的权重搬运。
6. 专家输出按路由系数加权，直接累加到一个 FP32 `[token, hidden]`
   VMEM accumulator；零系数显式屏蔽，避免 `NaN * 0` 污染输出。
7. 全部专家完成后，将 accumulator 转为 BF16，写回 HBM 一次。

```text
coeff, active = local_routing(topk_ids, topk_weights)
output = zeros([tokens, hidden], fp32, VMEM)
x_quantized = quantize(hidden)
for expert in unique_active_local_experts(active):
    prefetch_next_expert()
    intermediate = silu(x_quantized @ gate) * (x_quantized @ up)
    expert_output = quantize(intermediate) @ down
    output += masked_weighted(expert_output, coeff[:, expert])
return bf16(output)
```

伪代码省略了 block scale 与舍入步骤；实际实现保留原 GMM 的动态 FP8
activation quantization 和 BF16 中间舍入，最终专家加权累加使用 FP32。

新 kernel 内不再需要 token 排序、permute 或独立 unpermute。
路由系数构造、shared expert、DP reduce-scatter 和 TP all-reduce 仍在外部执行。
没有创建 `[expert, token, hidden]` 的 HBM tensor；单专家 intermediate 复用 VMEM。
因此不会因多个 token 命中同一专家而重复搬运该专家权重。

## 存储量与布局修复

本机 hidden=4096、expert intermediate=1024，FP8 权重：

| 单专家数据 | 大小 |
| --- | ---: |
| gate + up | 8 MiB |
| down | 4 MiB |
| FP32 权重 scale | 24 KiB |
| 总计 | 12.0234 MiB |

若本 rank 命中 32 个专家，一次完整权重读取约 384.75 MiB。
这只是权重字节量，不是整层 MoE 的实测 HBM 流量，也不能直接据此给出 MBU。
`[256,4096]` FP32 accumulator 为 4 MiB，显式 VMEM scratch 合计约 34.6 MiB，
包含双缓冲权重；kernel 配置上限为 60 MiB，入口要求硬件至少有 64 MiB VMEM。

第一版 activation scale 使用 `[tokens, Kblocks, 128]`。
LLO 与源码对应分析发现，读取固定 K block 的非连续平面产生大量
`vrot`、`vunpack`、`vsel` 和临时值。改为 `[Kblocks, tokens, 128]` 后，
固定 K block 对应连续的 `[tokens,128]` 平面，算术顺序保持不变。

| pool32 隔离实验 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 融合 kernel 时间 | 490.57 μs | 158.65 μs |
| MoE 整层 XLA Ops 时间之和 | 595.29 μs | 264.03 μs |
| 专家循环静态 LLO bundles | 25,818 | 7,159 |
| 专家循环静态 instructions | 150,127 | 44,811 |

同一次配对实验的原 MoE 路径为 305.64 μs，布局修复后整层降低约 13.6%。
布局修复前后输出逐位相同。静态 LLO issue 比例不是运行时利用率或 MBU；
VRF 与 VMEM 间的 spill/fill 也不能算作 HBM 权重流量。
双缓冲的存在不能证明 DMA 等待完全隐藏。

## 开关、支持范围与回退

在服务进程启动前设置，重启后生效：

```bash
export TPU_MOE_DECODE_IMPL=dense_expert
# 默认及回退对照：export TPU_MOE_DECODE_IMPL=standard
```

入口和输出 shape/dtype 不变。静态条件不满足时保留原 GMM 路径：

- BF16 hidden、原生 `float8_e4m3fn` 权重、每输出通道 FP32 scale。
- SiLU、无 expert bias、无 packed quantization，`skip_padded_tokens=False`。
- token 数为 16 的倍数，范围 16–256；本地专家 1–64。
- hidden 为 512/1024/2048/4096，expert intermediate 为 512/1024。
- 每 core VMEM 至少 64 MiB。

长 prefill 继续走原路径。应从日志确认
`Selected experimental dense-expert decode MoE`，不能仅根据环境变量判断生效。
新旧实现应使用独立编译缓存；配套 benchmark 启动脚本已经隔离。

## 正确性验证

- 集成后 203 项 TPU 测试通过，含 13 项新 kernel 测试和 190 项原 MoE core 回归。
- 覆盖同入口切换、EP 屏蔽、单专家、空 rank、零权重、重复专家和长 prefill 回退。
- 全尺寸 DP4/TP2/EP8、seed=20260922、专家池 16/32/64 的数值校验通过。
  相对同 GMM 算术的独立路由参考，整体 L2 误差约 0.264%–0.267%，
  最坏 token 约 0.445%–0.470%，低于既有 0.8% / 1.5% 门限。
  TP 副本逐位一致。
- 另一个独立 dense contraction 参考因中间舍入和量化差异，整体 L2 约 1.53%–1.56%；
  这与上面的同算术路由参考是不同口径，不能混用。
- 新算子无缓存、有缓存服务以及有缓存原实现，各回答 8 道简单问题，覆盖 4 个 DP rank。
  中英文、算术、翻译、Python、JSON 和四季问答文本逐条一致，未发现乱码或异常重复。
  原实现无缓存基线在追加问答检查前已完成。

简单问答和算子 UT 不等于完整模型精度评测，也不保证任意 prompt 输出完全一致。

## 端到端压测

实测日期 2026-09-21，TPU7x、DP4/TP2/EP8、全局并发 256。
每请求输入 65,536 token，输出 1,024 token；SSM 为 FP32，KV 为 FP8。
每 DP 最多 64 个请求，max model length 为 66,560。未开启 profiler 采集。

两个协议分别比较 standard 与 dense_expert，不跨协议直接归因算子收益：

| 配置 | 无 prefix cache | 有 prefix cache |
| --- | --- | --- |
| 客户端 | 原 `bench_decode_sliding_window.py` | 共享前缀复测客户端 |
| compile sizes | `[64,4096]` | `[64,4224]` |
| 有效 block size | 4224（请求下限 2304，由 FP32 SSM 调整） | 4224 |
| Mamba cache mode | none | align |
| 每实现测量轮数 | 1 | 3 |
| 请求前缀 | 各请求独立 prompt 和 salt | 各 DP 共享 63,360 token，独立 2,176-token 后缀 |

缓存组先用 4 个请求在各 DP 填充共享前缀。新旧 prompt manifest 哈希相同。
每轮 256/256 请求成功，零失败，逐请求命中 63,360 token，完整收到 262,144 输出 token。
每轮均有约 39–41 秒持续满 256 路 decode 区间。

| 指标 | standard | dense_expert | 变化 |
| --- | ---: | ---: | ---: |
| 无缓存，峰值 1 秒吞吐（tok/s） | 5739 | 5995 | +4.46% |
| 无缓存，峰值窗口 TPOT p50（ms） | 44.911 | 42.790 | −4.72% |
| 无缓存，峰值窗口活跃请求 | 248 | 248 | 相同 |
| 无缓存，含 prefill 整轮吞吐（tok/s） | 524.38 | 518.95 | −1.04% |
| 有缓存，满 256 路区间平均吞吐（tok/s） | 5917.03 | 6294.73 | +6.38% |
| 有缓存，满 batch 窗口 TPOT p50（ms） | 43.294 | 40.682 | −6.03% |
| 有缓存，含剩余 prefill 整轮吞吐（tok/s） | 3294.63 | 3435.62 | +4.28% |

缓存指标取三轮中位数。满 batch 平均吞吐逐轮为：

| 轮次 | standard（tok/s） | dense_expert（tok/s） |
| --- | ---: | ---: |
| 1 | 5917.03 | 6294.73 |
| 2 | 5913.68 | 6296.06 |
| 3 | 5918.26 | 6293.40 |

TPOT 来自客户端接收时间，不是设备 kernel 时间。
满 batch TPOT 使用原脚本 `peak_active_tpot_ms`：合并所有最大活跃数窗口后，
统计结束时间位于这些区间内的 token 间隔；缓存组最大活跃数均为 256。

无缓存组均无持续一秒满 256 路的窗口，全部请求重叠仅 0.214 / 0.170 秒；
不能拿该极短区间的吞吐代表稳定满 batch 性能。
原客户端记录输出 token 时间戳数为 262142 / 262143，保留原始统计结果。
无缓存单轮包含 prefill 的总体性能没有提升，不应只凭峰值宣称整轮请求更快。

缓存组非约束峰值吞吐的三轮中位数为 6144 / 6510 tok/s（+5.96%），
但峰值窗口活跃数分别为 256/256/254 与 251/253/256。
因此以满 256 路区间平均吞吐的 **+6.38%** 作为稳定 decode 对照结论。
单层 UT 的 13.6% 改善不能直接等同于完整模型收益。

## 代码、复测与原始记录

- [Kernel 实现](../../src/vllm_torchtpu/kernels/megablox/moe_dense_expert.py)
- [原入口与分支选择](../../src/vllm_torchtpu/layers/core/fused_moe_gmm.py)
- [Kernel 测试](../../tests/kernels/test_moe_dense_expert.py)
- [配套 UT、分析工具与服务启动脚本][benchmark]

[benchmark]: https://github.com/aios-tpu-infra/tpu_benchmark_daily/tree/handsomehow/dense_expert_moe

配套项目的 `MOE_DECODE_DENSE_EXPERT.md` 和 `MOE_DECODE_UT.md` 包含 UT 命令及环境。
硬件测试需要独占 TPU，应在停止服务后执行。已有验证使用 JAX/JAXLIB 0.10.2、
libtpu 0.0.47、torch 2.13.0+cpu、torch-tpu 0.1.1.dev20260912075415。

原始 E2E 数据保存在 benchmark 项目的本机
`runs/moe-e2e-20260921T065541Z/`，含配置、源码哈希、问答、逐请求时间线、
三轮缓存验证和 `comparison.json`；`runs/` 不纳入 Git，本文保留完整汇总口径。
已上传的历史 LLO/源码分析包（不包含本轮 E2E）：

```text
gs://xiaohao-tran/xiaohao_ignore/moe-llo-source-analysis-20260921T060341Z.tar.gz
SHA256: bf4435e16efc62468294ca06062e3d037b41916e89b657a53a695d54efe8ad49
```
