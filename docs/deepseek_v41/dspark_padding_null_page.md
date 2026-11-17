# DSpark draft padding 请求读取 null KV 页导致接受率崩塌

本文记录 epic/dsv41 上一个 DSpark 投机解码 bug 的根因、修复和验证结果。
受影响版本至少包括 `epic/dsv41` 11edf50e25（2026-11-13）。

## 现象

TP1/DP8/EP8、DSpark K5、greedy 下，同一个服务进程连续跑三轮，每轮 128 条
相同的请求（128 个输入 token，512 个输出 token），只改变请求在 8 个 DP rank
上的分布。单台 v7x-8、未修复的 11edf50e 上的结果：

| 轮次 | 每个 rank 的请求数 | draft 接受率 | 首 draft 接受率 |
| --- | --- | ---: | ---: |
| uniform（前） | 16×8 | 57.33% | 72.36% |
| skew | 19/16/16/16/16/15/15/15 | 0.09% | 0.24% |
| uniform（后） | 16×8 | 55.14% | 70.54% |

skew 轮的推测轮数是 65,256，约等于 128×512，几乎每一轮都没有接受任何 draft。
skew 结束后 uniform 立即恢复，问题只和当步使用的 draft 桶有关。

## 触发条件

EP-DP lockstep 要求各 rank 的集合通信形状一致。每个 decode 步，runner 用一次
`all_reduce(MAX)` 得到全 DP 组的最大请求数（`tpu_runner.py` 中的
`_dp_step_max_reqs`），draft 按「最大请求数 × 每请求行数」选同一个静态桶
（`dflash.py` 中的 `_dp_draft_bucket`）。DSpark K5 的 dense 布局每个请求占
5 行。

- uniform：最大请求数 16，选 80 行桶，rank 1 的 16 个请求正好填满。
- skew：最大请求数 19，95 行装不进 80 行桶，选下一档 160 行桶（32 个请求槽）。
  rank 1 仍然只有 16 个真实请求，剩下 16 个请求槽、80 行是 padding。

实验中给 `_prepare_dflash_inputs` 加的日志确认了这一点：skew 轮 rank 1 以
`num_reqs=16 dp_max=19 padded_len=160` 连续跑了 506 步，其他 rank 也都在
160 行桶里，各有 13 到 17 个 padding 请求槽。

只要某个 rank 有 padding 请求槽就会走到这条路径，不需要人为构造偏斜。
自由路由的线上流量同样会出现。

## 根因

### 1. padding 请求槽的 query 区间不为空

修复前，`DFlashProposer._get_static_attn_tensors` 按补齐后的请求数生成 draft
attention 元数据：

```python
qsl = torch.arange(padded_num_reqs + 1) * self.block_size
rd = torch.tensor([0, 0, padded_num_reqs])
```

块表只拷贝真实请求的行，padding 行是 0，即 null block。于是每个 padding
请求槽都被当成一个有 5 个 query、块号为 0 的真实请求。以 rank 1 为例，
第一个 padding 请求槽的 query 区间是 `[80, 85)`，正确的应该是 `[80, 80)`。

target forward 一直按另一个约定处理 padding：`tpu_runner.py` 把
`num_reqs` 之后的 `query_start_loc` 都填成最后一个真实请求的结束位置，
padding 请求长度为 0。draft 侧没有遵守这个约定。

### 2. CSA2 SWA 把 null 页当成有效 KV 读取和写入

`csa2/cache.py` 的 `prepare_swa_metadata` 用 `count = starts[1:] - starts[:-1]`
判断页是否可见，用 `arange(tokens) < starts[-1]` 判断哪些行有写记录。
padding 请求槽的 count 是 5，`starts[-1]` 是 160，所以：

- 读：padding 请求的 SWA 窗口覆盖物理页 0，页内容进入 attention 计算。
- 写：padding 行也生成了写记录。SWA 写端（`paged_swa.py` 的 `_pack_native`
  和 `decode_preparation.py` 的 `pack_matrix`）只写 `records >= page_size`
  的记录，所以层偏移为 0 时不会写到页 0；有层偏移的层会把 padding 的 KV
  写进 null block 里属于该层的那一页。

compressor state ring 和 main KV 写记录也用 `starts[-1]` 判断有效行，同样会
把 padding 行当成真实请求。

### 3. null 页里有 NaN

物理页 0 原本是全零的（`v2/cache.py` 用 `torch.zeros` 分配 KV 池）。
最初报告问题时的现场捕获和 PR #3397 的复测都看到页 0 里有非有限值。
NaN 最初由哪个写入方写入，目前没有查清。已知的相关变化：

- PR #2970（2026-11-11）把 CSA2 SWA 从专用池并入统一的 word pool。之前
  SWA 的 null 页只属于 SWA 自己，写端又跳过页 0，所以一直是 0。
- PR #3045（2026-11-11）把 cache group 改成 block-outer 布局。block 0 的
  开头同时是 SWA 层、256-token main KV 层的 null 页，以及部分 compressor
  state ring。main KV、state ring、indexer 的写端只跳过负数记录，没有
  页 0 的下限。

读端 `attention.py` 的 `_shift_swa_metadata` 注释写着 null block
"holds no KV for any layer"。在统一池里这个假设不再成立。

### 4. 融合 mHC 把 NaN 扩散到真实请求

mHC 在数学上逐 token 独立：每个 token 只和自己的门控系数相乘。融合 kernel
`shifted_post_pre_rms.py` 为了用 MXU，把一组 token（160 行桶的首组是 32 行）
的门控乘上单位阵，构成块对角系数矩阵，再做一次 `jax.lax.dot`。

对有限输入，块对角之外的 0 系数让其他行的贡献为零，结果和逐行计算一致。
输入里有 NaN 时，按 IEEE 浮点语义 `0 × NaN = NaN`，NaN 会进入同组所有行：

```text
y0 = 1 × 2 + 0 × NaN = NaN    （逐行计算应为 1 × 2 = 2）
```

现场捕获显示，NaN 先出现在第 40 层 SWA 输出的 padding 行 `[80, 160)`，
之后逐层扩散到 `[60, 160)`、`[30, 160)`，FFN 前的融合 mHC 输出全部 160 行
都是 NaN。

### 5. 首 draft 恒为 0，被全部拒绝

draft head 的输入 hidden 全是 NaN，129,280 个 logits 全是 NaN，argmax 返回
token 0。首 draft 为 0 和 target 的 top-1 不一致，被拒绝，后续 draft 也随之
作废。接受率因此接近 0。

### 代码来源

| 环节 | 代码 | 来源 |
| --- | --- | --- |
| padding 请求的非空 query 区间 | `dflash.py` | main 上的通用 DFlash 代码（b8193963dd，#517） |
| 按 count 判可见、按 `starts[-1]` 判写入 | `csa2/cache.py` | epic/dsv41，#2334 |
| SWA 并入统一 word pool | `paged_swa.py` 等 | epic/dsv41，#2970 |
| block-outer 布局，读写层偏移不对称 | `attention.py` | epic/dsv41，#3045 |
| 块对角密集矩阵乘 | `shifted_post_pre_rms.py` | epic/dsv41，#2062 |

draft padding 读 null block 在 main 上一直存在。对逐行独立的 attention，
null 页是有限值时 padding 行算出的结果没人使用，也影响不到真实行，所以
没有暴露。DSv4.1 上 CSA2 把 null 页当成有效 KV、null 页里有 NaN、mHC 按组
做密集乘，三者同时成立时才导致接受率崩塌。

## 修复

修复在 PR #3397，只改 `dflash.py` 和对应的单测：

- 两个编译好的 input builder（`_tpu_build_dflash_inputs_next_tokens` 和
  `_tpu_build_dflash_inputs_seed`）用 device 上的真实请求数生成 draft 的
  `query_start_loc`、`request_distribution` 和 `seq_lens`：

  ```python
  query_start_loc = torch.minimum(arange(padded_num_reqs + 1), num_reqs) * block_size
  request_distribution = [0, 0, num_reqs]
  seq_lens[num_reqs:] = 0
  ```

  形状仍按桶固定，`num_reqs` 是 device 标量，不进编译 key，不会按请求数
  重新编译。删除了按 `padded_num_reqs` 缓存的 `_get_static_attn_tensors`。
- 预热和 DP 空闲 rank 的 dummy draft forward 改成 0 个真实请求，形状和
  集合通信调用不变。

padding 请求的 query 区间为空后，`prepare_swa_metadata` 给它们的可见页和
写记录都自动为空。compressor state、main KV 等同样按 `starts[-1]` 判断的
路径也一起修好，不需要改各个写入 kernel。

修复不改 mHC kernel，也不全局清理非有限值。

## 验证

- 单测：`tests/spec_decode/test_dflash.py` 新增
  `test_draft_padding_has_empty_attention_windows`，对两个 builder、
  `num_reqs` 为 0/1/15/16/19/32 检查 padding 的 query 区间、
  `request_distribution` 和 `seq_lens`。dummy forward 的测试检查
  `num_reqs == 0`。
- 服务端复测：同样的单台 v7x-8、同一份三轮请求，`epic/dsv41` 11edf50e 加
  PR #3397 的 `dflash.py`。结果见下表。

| 轮次 | 每个 rank 的请求数 | 修复前接受率 | 修复后接受率 | 修复后首 draft 接受率 |
| --- | --- | ---: | ---: | ---: |
| uniform（前） | 16×8 | 57.33% | 56.38% | 71.77% |
| skew | 19/16/16/16/16/15/15/15 | 0.09% | 76.55% | 84.41% |
| uniform（后） | 16×8 | 55.14% | 56.69% | 71.85% |

修复后 skew 轮的桶没有变：8 个 rank 都在 160 行桶里跑了约 98 步，rank 1 仍是
`num_reqs=16 dp_max=19 padded_len=160`。接受率恢复不是因为没走到 padding
路径。

修复前后两组 uniform 的接受率基本一致，说明修复没有影响没有 padding 的情况。
skew 轮修复后的接受率高于 uniform，PR #3397 自己的复测（81.57% 对 59.83%）
也是这样，原因还没查。这组请求用了 `ignore_eos`，接受率只用于三轮之间对比，
不代表模型质量。

## 复现方法

在一台 v7x-8 上用 `epic/dsv41` 启动服务，参数要点：
`--tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel`、
`--max-num-seqs 48 --max-num-batched-tokens 2048 --block-size 256`、
`--num-gpu-blocks-override 45056 --kv-cache-dtype fp8`、
`--additional-config '{"deepseek_v41":{"execution":"ced","moe":"fused_ep_w4a8"}}'`、
DSpark `num_speculative_tokens=5`、greedy。

客户端用 `X-data-parallel-rank` 头把 128 条请求按 uniform（16×8）、
skew（19/16/16/16/16/15/15/15）、uniform 的顺序分三轮发送，每条请求用不同的
`cache_salt`，用 `/metrics` 前后差值计算接受率：

- `vllm:spec_decode_num_accepted_tokens_total` 除以
  `vllm:spec_decode_num_draft_tokens_total`
- 位置 0 的 `vllm:spec_decode_num_accepted_tokens_per_pos_total` 除以
  `vllm:spec_decode_num_drafts_total`

复测期间不能混入其他流量。

## 已知限制和遗留问题

- null 页里 NaN 的写入来源没有查清。修复保证 padding 不再读写 null block，
  不依赖页 0 的内容，但其他路径如果把 null block 当成有效 KV，仍可能遇到
  同样的问题。
- 融合 mHC 的同组跨行耦合仍然存在：同组任何一行出现非有限值，都会污染
  同组其他请求。是否改成逐行隔离需要单独评估性能代价。
- TPU 采样路径上 `min_tokens` 不生效：带 `min_tokens=512` 的请求生成
  EOS 后就结束了（`finish_reason: "stop"`，3 个 token）。这是独立问题。
  上面的复测因此在请求里加了 `ignore_eos: true`，模型在 EOS 之后继续生成，
  接受率的绝对值不能和其他环境直接比较，三轮之间的对比成立。
