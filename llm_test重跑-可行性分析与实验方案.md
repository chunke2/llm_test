# llm_test 重跑:可行性分析与实验方案

> 目标:基于 Karpathy 原版 llm.c,在 vast.ai 租用 GPU 上重新迭代优化,产出一份经得起推敲的实验报告。

## 结论

**完全可行。** 算力成本约 $20-50,旧报告的问题都能在重跑中系统性修复。

---

## 一、旧报告的硬伤清单(reviewer 视角)

1. **核心结论自相矛盾**:报告结尾称"cuBLASLt 换计算路径是 1.21× 加速的来源",但自己的数据表中 cuBLASLt 为 114ms,比手写 V5 的 107ms 更慢。结论与数据打架。
2. **精度对比不诚实**:cuBLASLt 版使用 `CUBLAS_COMPUTE_32F_FAST_TF32`(TF32 Tensor Core),手写 kernel 是纯 FP32 FMA。不同精度下的性能差异不能直接归因于"架构差异"。
3. **数字前后不一致**:baseline 出现 142.7ms/28,680 tok/s 与 138ms/22,800 tok/s 两套;硬件在 3080 Ti 与 5070 Ti 之间混用。
4. **无正确性验证**:V2-V5 只报速度,没有 loss 曲线或数值对比证明数学语义未被破坏。
5. **统计 rigor 缺失**:无方差、无多次运行、无迭代数/warmup 说明、无环境清单(驱动/CUDA/编译 flags/git commit),无法复现。
6. **V2 提速机理讲不通**:occupancy 不变(33.3%)、数据复用率下降,却快 21%,归因含糊。
7. **基线选择未论证**:优化的是 llm.c 的 legacy fp32 教学快照,而 mainline 已有 bf16 + cuDNN attention 完整方案。必须主动声明实验范围。

---

## 二、经得起推敲的重跑设计

### 实验规范
- **单一环境**:一台实例一块 GPU 全程不换;锁频(`nvidia-smi -lgc`);`deviceQuery` + `nvidia-smi -q` 存档。
- **固定配置**:GPT-2 124M、固定 B/T、同一数据集、固定 seed;所有数字出自同一轮实验。
- **正确性门禁**:每版优化先过 llm.c 自带 `test_gpt2` 分层数值对比(starter pack 内含 PyTorch debug state),再验证固定 batch 的 loss 曲线一致。
- **统计规范**:每版 n≥50 次迭代取 mean±std;NCU(默认锁频 base)与 nsys 数据分开对比,不跨工具混用绝对值。
- **精度分栏**:FP32 SGEMM / TF32 / BF16 明确分栏,架构结论只在同精度栏内下。
- **指标升级**:MFU(llmc/mfu.h 现成)、roofline、Amdahl 分析(matmul 占 61.6% → 整体加速上限)。
- **保留失败实验**(如 V3 padding 失败)并给出正确分析(padding 应为 33 而非 36)。
- **一键复现**:run 脚本 + 环境 manifest + 原始数据 + 数据自动生成表格的脚本。

### 实验阶段
| 阶段 | 内容 | 产出 |
| --- | --- | --- |
| P0 | 基线确定:从原版 llm.c 出发,记录 commit hash 与环境 manifest | 可复现基线 |
| P1 | nsys + ncu 全面 profiling,定位瓶颈 kernel | 瓶颈根因报告 |
| P2 | 手写 kernel 迭代(tile/launch_bounds/cp.async…),单变量改动,每版过正确性门禁 | 版本对比表(含 loss 等价性) |
| P3 | cuBLAS FP32 / cuBLASLt TF32 / BF16 分栏对比 | 精度×架构矩阵 |
| P4(可选冲刺) | 真 BF16 全链路 或 Flash Attention | 把"下一步方向"变成"已完成" |
| P5 | 撰写报告:方法、数据、失败分析、局限性 | 最终实验报告 |

---

## 三、vast.ai + Mac 实操要点(已核实官方文档,GPU 已定 4090)

- **实例类型**:必须租 **VM(KVM 全虚拟化)**,不要 docker 容器 —— 容器拿不到 GPU perf counter 权限,ncu 会报权限错误(旧报告在 AutoDL 已踩过)。vast 官方把 VM 的标注用途就写着 "CUDA performance profiling"(支持 ptrace + 硬件计数器)。
- **Template 选择**:`vastai/kvm:ubuntu_terminal`(Ubuntu 22.04 VM,自带 CUDA + Docker)—— VM 模板只能用 `docker.io/vastai/kvm` 仓库的镜像,可选空间本来就小,这个就是正确答案;另一个 Ubuntu Desktop VM 是 GUI 用途,忽略。
- **筛选条件**:GPU = RTX 4090 ×1、Extra Filters 填 `vms_enabled=true`(过滤出支持 VM 的机器)、On-demand(不要 interruptible,全程同一台物理机)、磁盘 ≥80GB(VM 磁盘开销大)、RAM ≥16GB、驱动支持 CUDA ≥12.4、direct SSH。
- **三个坑**:① **必须先在 vast 账号 Keys 页添加 SSH 公钥再租** —— VM 运行中公钥不可修改;② VM 创建启动比容器慢(10-20 分钟);③ 支持 VM 的机器较少,4090 无货时可放宽到 4080/3090(不影响结论,只需全程一致)。
- **开机后清单**:nvidia-smi 验证 → apt 安装 cuda-toolkit-12-4、nsight-compute、nsight-systems-cli、build-essential → 锁频(`nvidia-smi -lgc`)→ clone karpathy/llm.c 原版 → 记录环境 manifest。
- **CUDA 版本**:4090(sm_89)用 CUDA 12.4 即可,2023 年后的 NCU 均支持 Ada,工具链成熟(避开 5070 Ti/sm_120 需 CUDA 12.8+ 的坑)。
- **流程已验证**:ssh/rsync 传代码,`.ncu-rep` 拉回 Mac 用 Nsight Compute GUI 查看(Karpathy 本人同款流程)。
- **成本估算**:4090 约 $0.2-0.4/h,全程 30-50 GPU 小时,总计 $20-50 封顶。

## 四、自动化(我可以直接参与的部分)

- ssh 远程驱动实验、批量跑版本矩阵
- `ncu --csv` 程序化抽取指标,自动生成对比表,杜绝手抄数字出错
- 生成 runbook、环境 manifest、复现脚本

---

## 开工前三个决策点

1. **GPU**:✅ 已定 RTX 4090(省心、工具链成熟)
2. **基线**:legacy fp32 教学版(推荐,教学价值高)还是 mainline?
3. **冲刺项**:是否包含真 BF16 全链路或 Flash Attention?
