# DSV4.1 Indexer 优化记录

保存日期：2026-10-29。本文件记录本次提交保留的实现、测量口径、实验结论与未完成项。

## 流水线开销去除（最新保留版本）

针对上一轮的准备 15.2 µs、块间间隙 6.9 µs、输出/结束 9.2 µs，继续在
Falcon `exp-mj4tmkos8z` / `art-yz8s2t5ir2` 做同机 A/B。形状、seed、P7、
依赖版本及计时口径与下节一致；本轮 baseline 是已有三段 16K/无序 Top-K
版本，不是原始 32K 版本。

保留 `overhead-v1`：16K 按 8K 扫描只有两个块，静态使用 `[0, 1]`，删除
HBM 遍历表、VMEM scratch 和 DMA；默认请求偏移零不再传输；长度保持一维，
由 SparseCore 读取对应 lane，删除 TC 的 `[rows, 16]` 广播。较长输入保留
原来的固定 shuffle；非零动态请求偏移保留并增加回归覆盖。

| 阶段 | 本轮 baseline µs | 保留版本 µs |
| --- | ---: | ---: |
| logits | 156.955 | 156.506 |
| Top-K | 106.246 | 101.408 |
| logits + Top-K | 237.783 | 228.374 |
| gather + logits + Top-K | 719.492 | 711.471 |
| metadata + gather + logits + Top-K | 783.849 | 767.010 |

组合计时减少 9.409 µs（3.96%）。完整合成层相对本轮 16K 前版减少 2.15%，
但这不证明超过原始 32K 版本，也不是完整 serving/model 吞吐收益。

同一次中位样本的 TC/SC 时间线分解：

| 区间 | baseline µs | 保留版本 µs | 含义 |
| --- | ---: | ---: | --- |
| 首块 logits 前 | 15.209 | 15.482 | 入口、候选窗口计算、切片和布局转换 |
| TC logits 合计 | 171.606 | 171.171 | 三个 96-query chunk 的实际计算 |
| TC 块间间隙 | 7.021 | 2.333 | 元数据准备、复制、异步调用衔接 |
| 末块 TC 结束至最后 SC TEC 结束 | 34.737 | 30.473 | 剩余 Top-K 执行与依赖等待 |
| 最后 SC TEC 至模块结束 | 9.210 | 8.915 | 输出拼接及 trace 未细分的结束时间 |

三个 order-table copy 和长度广播已从 trace 消失。长度对齐 pad 与布局复制
仍需约 2 µs，不能称为已零开销。准备阶段首个可见 XLA op 之前约 7.4 µs；
最后输出 fusion 约 1.67 µs，之后约 7.1 µs 未被逐算子事件细分。因此不能
将准备和结束区间全部解释成可删除的计算，也不能假定都属于 host 开销。

`overhead-v2` 将逐块更新输出改成静态 list/一次 concatenate，组合为
229.566 µs，结束区间 9.258 µs，未改善，已撤回；最终保留原输出流水线。

验证：保留 kernel 的 84 项 Top-K 回归通过；拼接实验的 69 项 metadata、
logits 和 CPU reference 回归通过；同一保留 kernel 的 8 项无序/动态偏移
回归通过（其中 4 项为新增非零偏移）。共覆盖 157 个不同用例，三个变体
的完整合成层数值检查均通过。最终源码与 `overhead-v1` 的可执行 AST 一致；
并发 docstring 修改保留。未运行完整模型评测；全环境类型检查仍受私有
Artifact Registry 镜像认证阻塞，相关文件的 pre-commit 检查通过。

归档包括三个变体的五次设备样本、XPlane、HLO、源码快照、测试日志和
`trace-timeline.json`。实验已成功结束，TPU 已释放。
分析 `an-ee71o1z08k`（24 行算子指标）与 `an-1t7x3t9gg7`（A/B、数值验证和
实际 TC/SC 时间线）均成功，已读取声明的报告。标准算子插件仍报告 traces=0；
本节的时间线结论来自自定义分析的 `summary.json`，不能把标准插件的扫描
计数当作未捕获 profile。下节保留上一轮数据，避免混用不同实验的 baseline。

## 三段 union 与固定 16K 输出（本轮更新）

本轮代码将每对候选排成 `[A-only | shared | B-only]`，每段内部按逻辑 ID
升序。输出 logits 从 `[T, 32768]` 缩为 `[T, 16384]`，逻辑 block 表从
`[T, 4096]` 缩为 `[T, 2048]`，另带 `[T, 2]` 的窗口起点和段分界。
不足容量的列置 `-inf`。A 直接发布 union 前缀；B 在 MXU 静态展开结束后，
连续读取窗口并进行 128 列内旋转与相邻页拼接，不逐 block gather logits。
K 的物理 fragment gather 保留原流水线，输入物理表按新 union 顺序排列。

每个 member 的两个有序段分别计算因果前缀，避免前段不可见而后段可见时
错误截断。按用户确认的契约，候选 Top-K 同分可任取、输出无序；不执行 ID
同分 radix 和最终 ID 排序，仅在选出最多 512 个槽位后映射 record ID。
16K 使用已测静态展开配置。

同机 v7x、P-state 7、单逻辑设备，48×6 queries、context 98816、2048×8
候选、物理页 128、overlap 0.7、topk 512；输入 seed 826/827。
每项预热后 50 次 host 测量，并取 5 次 XPlane `XLA Modules` 设备耗时中位数：

| 阶段 | 基线 µs | 三段布局、无序 Top-K µs | 变化 |
| --- | ---: | ---: | ---: |
| prepare_metadata | 71.917 | 93.695 | +30.28% |
| gather | 478.445 | 501.917 | +4.91% |
| metadata_gather | 528.441 | 573.727 | +8.57% |
| logits | 165.702 | 156.254 | -5.70% |
| topk | 145.474 | 105.747 | -27.31% |
| logits_topk | 256.842 | 236.561 | -7.90% |
| gather_logits_topk | 710.096 | 722.507 | +1.75% |
| metadata_gather_logits_topk | 759.142 | 786.370 | +3.59% |

结论：logits 约快 5.7%，Top-K 约快 27.3%，logits+Top-K 约快 7.9%，
但包含 metadata/gather 的完整单层约慢 3.6%。固定 16K 达成，不能宣称整层
加速。Top-K 的 105.747 µs 是整个 288×16K batch，并非单行。
block 表重排与 gather 局部性有实测代价；跨层复用阶段的收益不可外推成完整模型吞吐收益。

失败/中间原型已移出工作区实现：逐页移位/carry 的 logits 为 527.763 µs；
整段旋转为 265.369 µs；128 列旋转/拼接为 266.306 µs；K 提前预取两组后
降到 157.056 µs。保留严格 ID 同分/排序时，Top-K 为 135.092 µs，整层
791.668 µs；按用户确认去掉两个约束后，最终 Top-K 为 105.747 µs，整层
786.370 µs。独立映射 buffer 的中间版为 99.693 µs，但复用原 ID scratch 后
代码更短、少一个独立 VMEM buffer，组合 logits+Top-K 分别为 236.190/236.561 µs。
动态旋转使用非负量，避免负值在非二次幂窗口上产生错误偏移。

无序路径的中位 trace 说明 TC/SC 已实际重叠：三个 logits chunk 分别占
15.222–70.564、76.147–133.669、135.027–193.263 µs；前两个 SparseCore
Top-K 执行约 79.745–115.120、138.817–173.889 µs，全部被下一块 logits
覆盖。末块约 198.413–227.392 µs。236.561 µs 的分解为准备 15.222、
logits 合计 171.100、块间 6.941、末块收尾 34.129、输出/结束 9.169 µs。
不能用两个独立 kernel 的耗时之和减组合耗时，直接解释成实际重叠长度。
原始跨 TC/SC 时间线见 `trace-timeline.json`，SC offload 包含等待，以上使用
真实 TEC 执行区间。压缩阈值 512/1024/2048/4096/8192 的 Top-K 分别为
113.106/106.475/106.040/105.986/105.805 µs；后四档接近，保留已验 2048。
显式 scheduling group 的组合为 238.146 µs，未优于自由调度的 236.561 µs，
48-query 分块为 260.902 µs，亦未改善，因此保留原 96-query 静态展开流水线。

LLO 证据：编译输出 HBM shape 为 `288x16384`，保留静态 MXU 前缀分支；
两组预取实验的计算指令主体与一组预取一致。三段布局去掉原 ownership
查询所需的大量 vector permute。原始 LLO 含互斥分支，静态指令数不能相加
当作实际执行周期；本轮不声称测得 register spill 的改善。

Falcon 实验 `exp-a4kg1c0vs9`，artifact `art-q4x201l93r`；同一 artifact 的
`baseline/` 与 `unordered-final/` 是最终 A/B；`candidate/`、`candidate-v5/`、
`candidate-v6/`、`candidate-v7/`、`unordered/` 保留中间实验。
前轮 `exp-1e5xq5er8x` 保存了首版回退数据。
归档包含源码 patch/快照、harness、全部设备样本、XPlane、HLO、Mosaic LLO。
Falcon 已成功结束并释放 TPU。分析：`an-1p9nbnl6le`（算子指标）、
`an-z888g8aj3x`（LLO）、`an-wu3m8i8cbr`（A/B 与实际时间线）均成功。
工作区同步的 `candidate_block_ids` / `physical_block_ids` 命名整理已保留；
与实测版相比，相关 producer/consumer/test 的 AST 除标识符和文档外一致，
旧字段引用已清空。

验证：metadata/scorer/模型参考组合 67 项通过；最终实际 TPU scorer 5 项
通过，包含 singleton 和单 token；最终无序 Top-K、原有严格 Top-K 和模型
参考 110 项通过。合计覆盖 153 个不同用例，详见归档测试日志。
测试 harness 跳过 vLLM serving 初始化及 cache allocator import，使用真实数值函数；没有执行完整模型端到端。
changed-files pre-commit（含固定版本 Ruff/Pyrefly）通过。完整依赖类型检查
因 CI 镜像 Artifact Registry 下载鉴权失败而未完成，本机 Docker 可用。

## Candidate 实现清理

`fp8/candidates.py` 从 1382 行缩至 908 行，减少 474 行：删除 8-token 整页的
独立特化及配置，将通用与 128-token 页 gather 合并为一套 DMA 流水线，
删除重复 canonical 包装和单 query 特判。8-token 页保留正确性支持，
不再保留其独立性能特化。跨页块、CPU/大 context fallback、缺页与 padding
处理仍有调用方，继续保留。

38 个真实 TPU candidate 回归通过，包括补充的 16/32/64/128-token 块、
双 bank、非零偏移/步长、空 tile 和 metadata 换批的逐 word 检查。
真实 128-token 页完整单层检查通过：ownership、gather readback、拆分与组合
Top-K、NumPy 阈值与候选归属均通过。相关 pre-commit 通过。

| 路径 | 清理后设备中位数 µs |
|---|---:|
| Metadata | 76.506 |
| Gather | 481.596 |
| Metadata + gather | 536.633 |
| Logits | 174.955 |
| Top-K | 157.167 |
| Logits + Top-K | 268.644 |
| Gather + logits + Top-K | 724.582 |
| 完整单层 | 779.639 |

仍采用标准 seed826、overlap=0.7、48×6 queries、page=128、block=8 的测量口径。
上轮完整单层为 777.291 µs；本轮没有新的整模性能结论。

## 追加保留：block max 对齐写回

在已提交基线 `052fe1135086254da58ce313ac47862eb596dbd9` 上，仅保留本轮
block max 写回优化：累计最多 8 页的 16-block 结果，再按 128 个元素写回；
尾段单独处理，并合并最新可见块置为 +inf 的操作。仍使用 3×96 分批，
每个 query 独立选择候选，不恢复旧版代表 query 共享候选的语义。

同机同输入，48 requests、288 queries、98816 keys、page=128、双 bank、P7：

| 范围 | 修改前 µs | 修改后 µs | 减少 |
|---|---:|---:|---:|
| Logits + block max 调用，含 3×96 输出拼接 | 570.7825 | 535.7965 | 6.13% |
| 完整 token/block Top-K 选择流水线 | 632.3040 | 598.6655 | 5.32% |

设备时间取 20 次 XPlane 样本中位数；预热 5 次，host 测量另取 100 次。
Logits、block maxima 与基线逐项一致，完整 token/block 选择集合一致。
535.8 µs 包含 wrapper、布局处理及输出拼接，不是纯 scorer 时间。
两行输出形式和流水重叠不同，不能相减得到独立 Top-K 耗时。
整模验证已提交，尚无本轮端到端收益结论；下文的 3.437–3.448 ms 属于此前基线。

同轮旧 dense 对照复现了原性能：纯 scorer 原版 288 行为 299.858 µs，
当前未融合 288 行为 309.424 µs，当前未融合 3×96 为 335.813 µs，
当前融合 3×96 为 460.913 µs。完整选择路径中，旧 dense + 当前逐 query 选择
为 698.576 µs，当前未融合为 707.746 µs，当前融合为 631.250 µs。
因此不能仅凭纯 scorer 耗时决定是否融合。
2×144 融合/未融合分别为 692.522/904.748 µs，同机 3×96 融合基线为
630.821 µs；分批变更不采用。本轮未继续增加其他优化。

## 上一已提交版本：端到端基线

该基线是按完整 Indexer 成本选择的已验证实现：每逻辑设备、每次模型 decode
forward 的 Indexer 中位数为 **3.437–3.448 ms**，目标 **≤2.6 ms 尚未达到**。
全模型正式测量的 decode 步间隔中位数为 **46.104778 ms**。
后者来自独立 formal 窗口，不是将 Indexer 各项相加所得的设备执行时间。

保存前已逐字核对：当前生产代码、单层 benchmark 和新增测试与成功端到端实验的
源码快照一致。相对于实验基线 `1890e7644c80bd213fecaaab4a96b96fdf682054`，
已跟踪代码的实验 patch SHA256 为：
`1d611c0dbe65331292b4bd484ca45e5bcd5c7a7532f259f5fa5424c2457e65c7`。

“最好”指当前已验证的 Indexer 路径，不表示全模型步间隔的历史最小值。
六 query adaptive 原型曾测到 45.904778 ms 的全模型步间隔，但 Indexer 回退到
3.587–3.608 ms，因此没有采用。不同轮次的小幅全模型差异需要重复匹配 A/B 验证。

| 验收项目 | 当前值 | 目标 | 状态 |
|---|---:|---:|---|
| 完整 forward 的 Indexer 累计 | 3.437–3.448 ms | ≤2.6 ms | 未达到；六设备可归属 |
| 单层 logits | 175.632 µs | ≤235 µs | 达到 |
| 单层 gather | 479.193 µs | ≤200 µs | 未达到 |
| 单层 metadata + gather | 538.924 µs | ≤500 µs | 未达到 |

## 测量口径与环境

- 模型：DeepSeek-V4.1-Flash，40 个目标模型层；draft 模型不计入 Indexer 总计。
- 端到端：v7x，4 个物理 chip、8 个逻辑设备，TP1/DP8/EP8/PP1。
  并发 384，输入 98304 token、输出 512 token，DSpark K5；编译 bucket 288/2048。
  KV 自动分配、memory utilization 0.85，Engram CPU raw DMA，禁用持久编译缓存。
- 运行环境：JAX/jaxlib 0.11.2、libtpu 0.0.50、NumPy 2.3.5、Transformers 5.18.0、
  Torch 2.13 CPU、torch-tpu 0.2.0rc1；该 torch-tpu 实验 wheel 的 SHA256 为
  `915bf99d453e33b46933ad9baa37af24b930dcafc5e015a75f9f2b7943087d38`。
  vLLM 为 `e37e51dd246cf421c554e7d6f53e178e3fd29085`。
- 单层：单逻辑设备，48×6=288 queries，2048 个候选块/query，block=8，topk=512，
  逻辑宽度 98816，物理 cache page=128 token；seed=826，overlap=0.7。
  两 query 共享候选并集，prepared logits 静态宽度为 32768。
- 单层设备时间取 5 次 XPlane 样本中位数；host 时间为预热后 100 次测量，单独记录。
  组合路径直接编译测量，其时间不能用独立算子时间相加替代。
- Indexer 计入专有 Q/weight 投影、RoPE/量化、K 生成/写入、候选准备、评分和选择。
  排除同时服务主模型的共享 QKV/MTP 祖先节点。
  对 TC 时间线区间取并集，不重复累加 conditional/while 父事件和内部子事件，
  不把 SparseCore 异步活动再加到已计入的 TC 等待上。
  该指标是暴露在 TC 时间线上的 Indexer 成本，不是两类核心活动时间之和。
- 本轮设备 0/2/3/4/6/7 各有 3 个目标程序 × 6 次完整执行；设备 1/5 的部分事件
  缺少 HLO 符号，因此只报告六设备结果。不能按运行顺序猜测未知 region 的归属。
  上一轮完整八设备结果为 3.459–3.484 ms。
- 早期 5.65/4.78 ms 统计包含共享投影，已废弃，不参与性能比较。
  旧 8-token 物理页单测也不能用于验收实际 128-token 页的 gather。

## 当前完整 forward 构成

下表为六个可归属设备、每设备六次 forward 的平均分项。各项互斥、合计等于
同一口径的区间并集；各项均值可相加，各项中位数不保证可相加。

| 部分 | 每 forward 累计 µs | 占比 |
|---|---:|---:|
| Encoder logits + Top-K，3 次 | 936.366 | 27.21% |
| 候选源层 Indexer，全量评分和两种选择 | 647.303 | 18.81% |
| Reindex 候选评分 + Top-K，4 次 | 1045.224 | 30.38% |
| 共享候选准备：metadata + gather，1 次 | 482.251 | 14.02% |
| Q 投影、RoPE、量化 | 114.466 | 3.33% |
| Weight 投影 | 22.191 | 0.64% |
| K 生成与写入 | 193.051 | 5.61% |
| 合计 | 3440.853 | 100% |

此前将候选源层称为“publisher”，容易误解为发布的增量开销。这里的 647.303 µs
包含该层的全量 logits、自身 Top-512 token 选择和 Top-2048 candidate block 选择，
不是额外增加的一个 publisher kernel。后续 union/dedup、地址生成和 K gather
属于单独的共享候选准备，供四次 reindex 复用一次。

候选源层评分输出宽度为 98816，prepared reindex 为 32768。代表性设备 0 的六次
forward 内部 conditional 均值为 644.419 µs，可分为：

| 内部阶段 | 均值 µs |
|---|---:|
| Dense logits，包含融合 block max | 458.866 |
| Token Top-K 暴露的 TC 等待 | 96.384 |
| Candidate block Top-K 暴露的 TC 等待 | 23.029 |
| 其余区间，包括转换、控制和间隙 | 66.140 |

内部 conditional 不含外围准备，不能直接替换上表六设备的 647.303 µs。
两项等待也不是 SparseCore kernel 的完整活动时间。当前该层主要暴露成本是 dense
logits；全范围评分的数据量较大，但不能据此断言当前实现已达到 DMA 极限。

## 完整模型历史对比

所有 Indexer 数字采用排除共享投影后的口径。正式 decode 步间隔来自独立的
formal 数据；当前轮共有 80 个正式步间隔样本。

| 阶段 | Indexer 中位数范围 ms | 可归属设备 | 全模型步间隔中位数 ms | 处理 |
|---|---:|---:|---:|---|
| 首轮完整基线 | 4.804–4.830 | 8 | 47.328705 | 对照 |
| 真实页 gather + Top-K 展开 + 融合候选发布 | 3.929–3.944 | 6 | — | 后续继续优化 |
| 有界候选源层流水线 + 实际 compact 长度 | 3.743–3.756 | 6 | — | 后续继续优化 |
| Refined Top-K | 3.537–3.556 | 6 | 46.138681 | 后续继续优化 |
| Dense logits 融合 block max | 3.459–3.484 | 8 | 46.067056 | 最近完整八设备对照 |
| Matrix/direct metadata，当前保存版本 | 3.437–3.448 | 6 | 46.104778 | 保留 |
| 六 query adaptive 原型 | 3.587–3.608 | 6 | 45.904778 | Indexer 回退，未采用 |

首轮到当前的全模型步间隔减少约 1.224 ms（2.59%）。相对上一轮 46.067056 ms，
当前轮没有显示全模型步间隔改善。不能把 Indexer 时间减少直接等同于整模收益。

最新六设备 Indexer 中位数（µs）：
0=3440.644、2=3446.954、3=3437.011、4=3448.419、6=3444.349、7=3438.075。
最近完整八设备对照（µs）：0=3459.460、1=3470.423、2=3480.455、3=3470.755、
4=3484.473、5=3470.730、6=3478.475、7=3477.921。

## 当前单层构成

| 路径 | 设备中位数 µs |
|---|---:|
| Metadata | 76.753 |
| Gather | 479.193 |
| Metadata + gather | 538.924 |
| Logits | 175.632 |
| Top-K | 157.076 |
| Logits + Top-K | 269.094 |
| Gather + logits + Top-K | 726.049 |
| Metadata + gather + logits + Top-K | 777.291 |

同口径上轮 full 为 834.525 µs。新 metadata 在整模中减少共享准备约 47 µs，
但 reindex 分项增加约 16 µs，净变化约 33 µs；独立单层节省量不能直接外推。
真实候选分布下共享准备 482.251 µs，也不能代替标准 overlap=0.7 的单测验收。

## 保留的实现

1. Prepared logits 按 4096-token 粒度选择静态展开前缀；地址用无符号索引，
   稳态交替 MXU/VPU head 归约，drain 使用 VPU。已测配置 issue depth=16、
   buffers=4；标准输入使用 20480-token 前缀。
2. 真实 128-token 页的 SparseCore gather 采用直接 metadata DMA、融合 live flags、
   轮换尾部 tile、外提 bank 判断；使用 64 fragments/4 buffers。
   8-token 整页路径使用 K/scale 分离读取与 VMEM 重排，其调优结果单独记录。
3. v7、288 queries 的请求分组用矩阵归约替换扫描和二分查找；已测
   288×2048、width=98816、block=8、page=128 路径直接读取原始 query/table，
   将 logical union 写回各 query 行，省去中间 gather/展开。
   公开 canonical API 保持原语义，其他形状保留原路径。
4. Streaming Top-K 对已测 v7、K512、宽度 32768/49408/98816 静态展开 4 次，
   bootstrap=8×K；正常非零阈值用 FP32 比较，零/subnormal 保留位序比较。
   最终 ID 排序用 SparseCore 原生向量排序及静态 bitonic 合并。
5. 选择 prepared scores 时使用实际 compact 长度。候选源层在已测形状使用有界
   评分/选择流水线；dense logits 融合 block=8 的最大值、长度 mask 和最新块处理，
   避免单独扫描整张 score buffer。仅在明确的已测形状与页尺寸上分派。
6. Profile 采集使用两个相邻的六 forward 窗口轮换 chip 内 collector owner。
   此方法曾取得八设备完整符号，但最新轮仍缺两个设备，尚未解决稳定覆盖问题。

## 实验记录与取舍

表中的独立 kernel/组合数字仅在对应行的输入和测量范围内比较；未采用的原型
不在默认生产路径中。正确性通过不代表端到端性能通过。

| 方向 | 测量或发现 | 结论 |
|---|---|---|
| 动态 logits → 静态前缀 | 634.6 →190.3 µs；交错归约后约 175.6 µs | 保留静态前缀和交错归约 |
| 完整展开 32768 列 | 261.7 µs，慢于按实际长度选择前缀 | 未采用 |
| Prepared issue depth | 4/8/24/32/48 均未优于 16 | 保留 16 |
| 原始页布局直接送 logits | gather 可减少，但 logits 约 18152 µs | 未采用 |
| 真实页 gather 调优 | 原约 949.348 →480.929 µs；静态 LLO 3103 →1332 bundles | 保留 |
| 更深 gather 缓冲 | 482.638 →最佳 476.708 µs，收益很小 | 未更改默认值 |
| 多 query 共享源页读取 | 完整 word 对照通过；480.881 →1112.952 µs；CPU 计划成本未计入 | 未采用 |
| 更大 tile、空 tile 地址跳过、bank flags 变体 | 没有稳定组合收益 | 未采用 |
| Matrix 请求分组 | pair metadata 约 143.8 →92.2 µs | 保留 |
| 直接 metadata 单独原型 | 约 143.8 →123.2 µs | 与 matrix 合并 |
| Matrix + direct metadata | metadata 76.753 µs；full 777.291 µs；整模已验证 | 保留 |
| Packed presence counters | pair 约省 3 µs，六 query 路径回退 | 未采用 |
| 六 query 高重合原型 | overlap=0.9、depth=8：logits 131.315 µs，full 556.905 vs 780.692 µs | 仅 synthetic 有效 |
| 六 query adaptive | guard 对照通过；真实模型执行两套 metadata，scorer 走 pair 分支；Indexer 3.587–3.608 ms | 未采用 |
| 候选 block max 独立融合 | 96×98816：148.0 →63.3 µs；与选择组合 183.0 →112.8 µs | 已测形状保留 |
| Dense logits 融合 block max | 候选源层独立组合 845.020 →731.006 µs | 保留，并完成整模验证 |
| Top-K 展开 4 次 | 三种宽度约 241/330/537 →186/258/401 µs；展开 8 次回退 | 保留 4 次 |
| Top-K refined 比较/排序 | 三种宽度约 159/217/315 µs | 保留 |
| Top-K sampled exact pivot | 独立 Top-K 有收益，组合仅约 15–20 µs；未做真实整模验收 | 未提升为默认 |
| Top-K batch scan 8 | 98816 列约 315 →298 µs，其他宽度无益或回退 | 未提升为默认 |
| Top-K light ties | 最大宽度约 315 →311 µs，收益很小 | 未采用 |
| TensorCore 最终 ID 排序 | 三种宽度约 197/257/353 µs，均回退 | 未采用 |
| Builtin Top-K | ordered uint 约 7–18 ms，float 约 4–20 ms | 未采用 |
| Dense query 重排 | 219.296/371.889 →225.317/390.375 µs（raw/pipeline） | 未采用 |
| Dense query + weight 重排 | 编译失败，没有可用性能结果 | 未采用 |
| Dense 提前预取 | 原 219.815/371.476；lead2 为 247.630/381.417，lead4 为 279.837/399.525 µs | 未采用 |
| Dense coalesced DMA、较小 depth、免 mask 静态前缀 | 完整组合无收益 | 未采用 |
| Token-major cache | raw 219.291 →9905.312 µs，编译约 357 s；后续组合编译终止 | 未采用；未改生产 cache 布局 |
| SparseCore cache writer | 全 bank word 对照通过，但实际稀疏写入 stride=4/6 回退 | 未采用 |
| 通用 JAX take/gather | 当前双 bank/scale/packed 输出下约 55–112 ms，trace 已有 gather offload | 未采用；不代表所有 take 场景 |
| 提前准备共享候选 | 140 预检通过，模型启动失败，见下文 | 未采用；没有端到端计时 |

提前准备原型的实际失败为 `RuntimeProgramAllocationFailure`：加载
`tt_jit_layernorm_L21C0_rms_norm_mul` 时需预留 26.60G，可用/可预留仅 20.81G。
这证明该次模型启动内存分配失败；具体是否由候选 buffer 生命周期或其他调度变化
造成尚未证实，不能将预期的重叠收益当作测量结果。

## DMA roofline 与 LLO

真实 128-token 页、标准输入的 gather：358846 个有效 block、5668 个 live tile；
读取 378941376 B，写回 383066112 B，合计 **762007488 B**。
按 profile 给出的单逻辑设备峰值 HBM **3686155681792 B/s**，有效载荷理想下界为
**206.721 µs**。按用户给出的 **1688 B/cycle**，为 **451426 cycles**。
不含 metadata、描述符、启动和等待，也未把重复源页当作硬件 cache 命中。
保持当前搬运量时，200 µs 已低于这个有效载荷下界，需要减少搬运或改变布局。

Prepared logits 的动态版本主循环处理 2048 K token：K 和输出 DMA 预算约
169.86 cycles，原静态调度 634 bundles、无符号地址版本 520 bundles。
该循环明显还有指令发射空间，不能称作纯 DMA bound。

静态前缀版本选中 20480-token 分支后，纯 VPU 核心为 1631 bundles，
交错 MXU/VPU 归约核心为 **1426 bundles**；典型 query pair 的全量 DMA 预算约
1744.22 cycles。核心计算段已低于该预算，但外围控制、spill 和实际等待仍存在。
1426-bundle 核心的 vector spill store/reload 为 180/259；纯 VPU 为 1020/1095。

Dense logits 的互斥核心分支约 6200/6027/6317/6168 bundles；每分支 1250 条
底层 vmatmul。典型 6 queries×49408 的 K 读和 score 写约 7707648 B，
按 1688 B/cycle 为 4566 cycles；核心发射预算仍高于该有效载荷预算。
不能将多个互斥分支的 bundles 相加，也不能将静态指令数当作动态周期或耗时占比。

SIM PJRT 导出限制单 CPU affinity，OMP/BLAS/TF 线程均为 1，只用于编译和 LLO。
SIM 版本警告不影响本记录的真实设备计时来源；没有用模拟耗时验收。
旧记录以 2.2 GHz 换算的微秒值是假设，不是本次测得的频率。

## 验证与复现

当前源码已有两组真实 TPU 验证：单层 110 个回归通过（candidate 30、Top-K 80），
完整单层输出对照通过；端到端 136 个预检通过，formal/profile 均通过。
新增覆盖包括请求奇偶数、padding、全无效、不等长请求、缺页、bank 边界、
部分长度、zero/subnormal/ties 和候选所有权。当前保存的代码不含失败原型。

本次提交前的 CPU benchmark 回归 **44 个通过**，包括两个采集窗口在八个 rank
轮换 owner、保存记录及恢复原执行入口的检查；旧单窗口测试已按现有行为更新。
生产实现未因该测试更新而改变。changed-files pre-commit 覆盖 Ruff、format、
Pyrefly、Markdown、环境读取和 kernel closure 检查。
完整依赖环境 typecheck 未执行：本机无 Docker；
不能把无第三方依赖的 pre-commit Pyrefly 等同于完整环境类型检查。

在上文固定版本的 TPU 环境中运行单层复现：

```bash
python scripts/dsv41_bench/bench_sparse_layer.py \
  --out /path/to/new-output-dir --requests 48 --queries-per-request 6 \
  --width 98816 --candidates 2048 --block 8 --page-size 128 \
  --topk 512 --overlap 0.7 --repeats 100
```

核心回归入口：

```bash
python -m pytest -q \
  tests/kernels/deepseek_v41/test_fp8_candidate_membership.py \
  tests/kernels/deepseek_v41/test_fp8_gathered_mqa_logits.py \
  tests/kernels/deepseek_v41/test_candidate_block_scores.py \
  tests/kernels/deepseek_v41/test_fp8_paged_mqa_publication.py \
  tests/kernels/test_dsv41_streaming_topk.py \
  tests/models/vllm/test_dsv41_fp8_candidate_correctness.py
```

完整模型使用仓库已有的
[server-step benchmark](scripts/dsv41_bench/server_steps/README.md)，准备满足 decode
场景的 384 条固定输入，使用同一模型、版本、KV 配置和独立输出目录：

```bash
export MODEL_PATH=/models/DeepSeek-V4.1-Flash
python scripts/dsv41_bench/server_steps/run.py --scenario decode \
  --input /path/to/fixed-decode-input.json \
  --out /path/to/new-decode-output-dir --rounds 1 --profile
```

成功实验另含 benchmark 启动期的 GC 绕过：仅启动期禁用自动 cyclic GC，
在 `benchmark_begin` 中 collect 并重新启用；八个 worker 均确认恢复。
它用于绕过该 runtime 的 `RecordDataPtrDestroyed → GetEagerMode` 启动异常，
未改生产模型，未改计时窗口内 GC 设置，也未作为本次生产修改提交。
直接运行上述入口若遇到该启动异常，不能认为已复现成功实验。

保留原始 XPlane、HLO、运行版本、输入摘要、源码 patch、正确性输出和 formal
原始记录才能重新审计。内部原始产物索引与旧逐轮日志保存在本地实验归档，
本文件提供不依赖内部链接的结论和参数。

## 后续未完成项

- 将完整 Indexer 从约 3.44 ms 降至 ≤2.6 ms；当前差约 0.84 ms。
- 减少 gather 的实际搬运量与发射开销，达到标准输入上的 200 µs/500 µs 两项目标。
- 优化候选源层 dense logits；继续核对计算必要性和调度，不能仅由分项名称推断。
- 修复八个逻辑设备的稳定符号覆盖，按相同口径完成全部设备验收。
- 提前准备原型先解释启动内存失败，再决定是否重测；未验证的原型继续与默认实现分离。
