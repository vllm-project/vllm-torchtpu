# PCP/TP → TP/DCP：单机开发与验证

日期：2026-09-20。分支：`handsomehow/qwen38_c3_main`。

本轮完成 connector 接线、CPU 回归、原生 controller 注册/取消、真实 TPU
HBM 搬运，以及 8-device SPMD GDN state 布局验证。按用户要求，短请求问题
暂时搁置，最终搬运测试集中在 8K、16K、32K 长请求。本轮没有启动远端服务，
没有执行双机 PD 端到端推理，代码尚未提交。

## 实现范围

- P worker 的传输身份使用 `pcp_rank * TP + tp_rank`，完成数使用 `PCP * TP`。
  GDN head 的顺序独立使用 `tp_rank * PCP + pcp_rank`。
- HND FA lowering 同时处理源 TP head 所有权、PCP token 所有权、目标 TP head
  所有权及 DCP token 所有权。源 FA head 复制时只选一个 canonical TP sender。
  复制与不复制复用同一路径。
- 全局 kernel page `j` 进入目标 DCP lane `j % DCP`，目标局部 page 为
  `j // DCP`。manager block 可以包含多个 kernel page。
- GDN conv/SSM 按 head 搬运，不按 DCP token 切分。QK 复制时的本地 head
  几何复用计算侧 `derive_gdn_head_geometry`。
- BF16 conv 使用实际 HND carrier 的 512-byte 粒度；旧布局调用仍使用其原有
  1024-byte 粒度。没有新增功能开关或替代计算路径。
- producer 使用已有 connector extra config 字段
  `destination_decode_context_parallel_size` 声明目标 DCP；目标 TP 使用已有
  `TPU_RAIDEN_DST_SHARDS`。这些是目的端几何，必须与 D 端启动参数一致。
  scheduler 在跨服务 metadata 中传递 `dst_tp_size`、`dst_dcp_size`，D 端核对。
- D scheduler 的逻辑 block 长度为 `block_size * DCP`。缓存前缀裁剪使用
  `skip_tokens / DCP`，以及目标 manifest 已经包含的本地 KV heads，避免再次除 TP。
- TP>1 的 D 端统一由 TP0 提交一次 transfer，广播提交结果；每个 worker 仍等待
  本地原生搬运完成。覆盖 TP2、TP8、TP32 的 leader/follower UT。
- HND、DP1、PP1 下，准入 PCP+TP producer 和 TP+DCP consumer；其他既有拓扑
  保持原有规则。仅改变 transport/connector，不改变 FA/GDN 计算 kernel。

## 测试配置与结果

| 用例 | P | D | 全局 FA KV heads | 结果 |
| --- | --- | --- | --- | --- |
| 单机缩放目标 | PCP4 TP2 | TP8 DCP8 | 1 | 三组长请求搬运通过 |
| 源 head 拆分 | PCP4 TP2 | TP8 DCP2 | 4 | 三组长请求搬运通过 |
| 完整部署传输几何 | PCP16 TP2 | TP32 DCP8 | 4 | 三组长请求搬运通过 |

每组分别传输 8191、16383、32767 token，对应 P8K/P16K/P32K 的 N−1 handoff。
32K 用例另外裁剪一个 D scheduler block 的已缓存前缀：DCP8 为
3072 token，DCP2 为 768 token。kernel page=128、manager block=384。
使用非连续物理 block ID，conv/SSM 放在不同于 FA 的 block。

原生搬运测试在同一进程中创建两组独立 controller/store 和独立 worker manager，
每个 worker 持有独立、真实物化的 TPU buffer。通过实际原生 planner、D2H、
本机网络传输和 H2D，最后从目标 HBM 读回逐字节核对 FA、conv、SSM，检查全部
source/destination worker 的完成回报。完整几何用例包含 32 个 P manager 和
32 个 D manager。

这里的 worker 是虚拟传输角色，buffer 位于本机 TPU；**不是 64 个实际设备，
也没有验证跨 host 的进程启动、device placement、网络或吞吐。**

### CPU / 原生 registry

- 417 passed，17 warnings，无跳过。
- 包括 span 独立 oracle、connector metadata/block 数、prefix clipping、
  head 复制、完成计数、准入、原生 C++ RPC 注册/取消与旧路径回归。
- 全尺寸 GDN K16/V128、缩放 K4/V32，分别按有效并行度 32、8 分片；
  每 rank K1/V4，QK 复制两份。conv 为 BF16，live bytes=4608；
  SSM 为 BF16，live bytes=131072。
- 不声明支持任意不等 P/D GDN 并行度下的 replicated-QK resharding。
- 日志：`../runs/pytest_pd_single_host_final2.log`。

### 真实 TPU

- HND 原始布局实测：源 KV heads=1/2、目标=1；minor-to-major 为
  `[4,3,2,1,0]`，tiles 为 `[[4,128],[4,1]]`，与 CPU 物理字节 oracle 一致。
- 原生 HBM 搬运：3 个拓扑、每个 3 种长请求，全部通过。
  日志：`../runs/pytest_pd_hbm_long.log`（2 passed）、
  `../runs/pytest_pd_hbm_full.log`（1 passed）。
- GDN state：8 个实际 TPU device 上通过 SPMD 调用生产 Pallas 的
  `store_state_region` / `load_state_region` 和生产 state plan。
  源/目标 carrier heads 为 1→1、2→1，BF16 conv/SSM 及 QK pair row 排列
  精确还原。中间按 span 的 byte copy 在 host 上模拟，不能单独算原生 DMA 证明。
  日志：`../runs/pytest_pd_hbm.log` 中 state 用例 passed；
  `../runs/pd_gdn_state_spmd.log` 有两组成功记录。
- Ruff、Pyrefly、`git diff --check` 通过。

## 已知限制与本轮遇到的问题

1. 既有 PCP 短请求准入限制保持不变。直接绕过 scheduler 测 127-token 原生
   transfer 时，planner 拒绝没有 FA coverage 的目标 lane。
   当前 native planner 还要求各目标 lane 有相同的 coverage extents。
   **请求长不代表一定不受限制**；非均匀末尾仍可能失败。本轮通过的是上表所列
   对齐长请求，不声明任意 token 长度已支持。
   失败日志为 `../runs/pytest_pd_hbm.log`，两个 transfer 用例报
   `Destination block index 0 has no declared coverage ... destination unit 1`。
   依据用户要求，不修复、不以多传无效页绕过。
2. host-only manager 缺少生产控制端点，不能用于完整 controller 驱动的 HBM
   transfer。最终测试改用真实 TPU-backed manager。
3. 同进程多个 manager 必须在测试子进程使用已有
   `RAIDEN_DISABLE_SINGLETON_WORKER=1`，否则 worker service singleton 相互覆盖。
   生产每 worker 一进程，不需要这个测试配置。
4. 本机原生扩展需要较新的 libstdc++。测试进程通过 `LD_PRELOAD` 使用机器已有
   `/opt/compat-12-libstdc++/lib64/libstdc++.so.6.0.30`；未替换系统库或重装 wheel。
5. native manager 异常退出时可能延迟释放 TPU runtime。测试使用独立子进程并
   串行运行 TPU 用例；一次探测因前一进程尚未退出发生 lockfile 错误，清理本轮
   探测进程后恢复。没有修改 slice 配置或删除其他服务资源。
6. 搬运测试成功结束后可能打印 transport `recv eof` 的连接关闭日志；验收依据是
   完成回报、无 failed transfer、及 HBM 字节一致，不依据单独的 RPC accepted。

## 复现命令

从仓库根目录执行，先确认本机没有其他 TPU 服务占用设备：

```bash
export TPU_SKIP_MDS_QUERY=true
export LD_PRELOAD=/opt/compat-12-libstdc++/lib64/libstdc++.so.6.0.30

JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
  tests/distributed/test_pcp_tp_dcp_connector.py \
  tests/distributed/test_tpu_connector.py \
  tests/distributed/test_pcp_tp_dcp_spans.py \
  tests/distributed/test_pcp_tp_dcp_native.py \
  tests/distributed/test_seq_on_lane.py \
  tests/distributed/test_pool_byte_spans.py \
  tests/distributed/test_raiden_pool_manifest.py \
  tests/distributed/test_raiden_layout_fingerprint.py \
  -q --basetemp=../runs/pytest_pd_cpu_replay

JAX_PLATFORMS=tpu .venv/bin/python -m pytest \
  tests/distributed/test_pcp_tp_dcp_transfer_tpu.py \
  tests/distributed/test_pcp_tp_dcp_state_tpu.py \
  -q -s --basetemp=../runs/pytest_pd_tpu_replay
```

TPU 测试包含隔离子进程，不要使用 pytest-xdist 并行执行。测试不依赖远端机器。

## 下一步：双机长请求 PD

单机已验证传输几何、实际 HBM 字节及 GDN state 读写。下一阶段才检查两个独立
8-device runtime 的服务：P PCP4TP2、D TP8DCP8，quarter heads/experts、先 12 层，
使用上述 8K/16K/32K 长请求。模型缩放见 `qwen38_pcp_tp_dcp_pd_plan.md`。

还需验证真实 scheduler 的 block tables、跨 host endpoints、请求从 P 到 D 的
交接、D continuation，以及与相同权重下非 PD 参考的结果对齐。dummy 模型只能
验证请求能跑通；不同 TP 分片下各自初始化的 dummy 权重不能作为 logit 正确性基准。

## 提交前 review 修复（2026-09-20）

- GDN producer 在创建 store、注册 pool 前，用计算链路的 head geometry
  校验目的端映射。当前 byte-span 拼接不能增删 Q/K 副本，因此拒绝两端
  Q/K replication factor 不一致的组合；非全量 producer 还要求目的并行度
  整除源并行度。等并行度的复制模式，以及原有不复制的 fan-in / 全量
  producer fan-out 继续共用原链路，不新增开关或 fallback。
- 反例：GDN K4/V32，PCP4TP2 → TP4，目的 conv live bytes 为 7680，
  原拼接生成 9216 字节的写入范围。现在启动准入直接拒绝该组合。
  PCP4TP2 → TP8 和 PCP16TP2 → TP32 的复制模式保持支持。
- 接收端要求请求元数据显式携带匹配的 `dst_tp_size` / `dst_dcp_size`，
  不再用本地值或 DCP1 补齐缺失字段。P/D 须同时部署此协议版本。
  producer 的 `destination_decode_context_parallel_size` 仍是目的端拓扑配置，
  默认 1 表示实际目的端不使用 DCP；它不是功能启用开关。
- 直接读取正式 `decode_context_parallel_size` 配置字段；共享 GDN geometry
  工具改为顶层 import，并补齐返回类型。
- 三个不支持拓扑的反例测试先失败；修复后 connector、span 和 native
  registry 的 346 项测试全部通过，其中新增 12 个参数化用例。测试确认
  无效拓扑在 store/engine 构造前失败，缺失或不匹配的元数据在读取 block
  table 前失败。未重启现有服务，也未重跑 TPU / 端到端测试。
  日志：`../runs/pd_code_review/pytest-review-fix-red.log`、
  `../runs/pd_code_review/pytest-review-fix.log`。
