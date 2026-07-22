# llm_test 重��：实验设计方案与执行流程

> 目标：从 karpathy 原版 llm.c 出发，在 RTX 4090 上重新迭代优化，产出一份**经得起推敲**的 GPU kernel 优化实验报告。
> 核心方法论转变：从旧报告的"经验驱动 + 手动抄数 + 无验证门禁" → 改为"**指标驱动 + 自动化 + 每版过正确性门禁**"。

---

## 一、旧项目问题诊断（按严重程度分级）

### P0 级 —— 颠覆结论可信度（必须修）
1. **结论与数据自相矛盾**：报告结尾称"cuBLASLt 换计算路径是 1.21× 加速的来源"，但其自身数据表中 cuBLASLt 为 114ms，**比手写 V5 的 107ms 更慢**。最终结论不成立。
2. **全程无正确性验证**：V2→V5 每版只报速度，**没有任何 loss 曲线或分层数值对比**证明优化未破坏数学语义。改 tile/cp.async 极易写出边界 bug，无法自证正确。
3. **精度混比，归因错误**：cuBLASLt 用 `CUBLAS_COMPUTE_32F_FAST_TF32`（TF32 Tensor Core），手写 kernel 是纯 FP32 FMA。**不同数值精度下的性能差异被直接归因于"架构差异"**，这是逻辑硬伤。

### P1 级 —— 无法复现
4. **统计 rigor 缺失**：无方差、无多次运行、无迭代数/warmup 说明、无环境清单（驱动/CUDA/编译 flags/git commit）。
5. **数字前后不一致**：baseline 出现 142.7ms/28,680 tok/s 与 138ms/22,800 tok/s 两套。
6. **硬件混用**：RTX 3080 Ti 与 5070 Ti 之间来回切换，寄存器/共享内存推算基于错误硬件。

### P2 级 —— 分析不严谨
7. **V2 提速机理讲不通**：occupancy 不变（33.3%）、数据复用率下降，却快 21%，归因含糊。
8. **基线选择未论证**：为何优化 legacy fp32 教学快照而非 mainline（已有 bf16+cuDNN），未主动声明实验范围。
9. **V3 padding=36 失败分析不到位**：正确解应是 padding=33（奇数错开 bank），36 是错误方向。
10. **occupancy/共享内存计算张冠李戴**：5070 Ti 标 102400 B/SM，寄存器推算却写"3080 Ti 65536"。

### 根因总���
旧报告的问题不是某个数据错，而是**方法论缺陷**：没有正确性门禁、没有统计规范、没有环境一致性、结论脱离数据。重跑的价值在于用一套可辩护的流程重做，让每一个结论都有数据和验证支撑。

---

## 二、实验设计原则（问题 → 对策）

| 旧问题 | 新对策 |
|---|---|
| 结论矛盾 | 单一硬件 + 单一配置 + 所有数字出自同一轮实验；结论只从最终数据表得出 |
| 无正确性验证 | **每版先过正确性门禁**（test_gpt2 分层数值对比 + loss 曲线一致），不过门禁不进性能对比 |
| 精度混比 | **FP32 / TF32 / BF16 分栏**，架构结论只在同精度栏内下 |
| 无 rigor | n≥50 iter 取 mean±std；锁频 2520MHz + persistence；NCU（锁 base）与 nsys 分开对比，不跨工具混用绝对值 |
| 无法复现 | 环境 manifest + git commit + run 脚本 + 原始数据 + 自动生成表格的脚本 |
| 归因含糊 | **NCU 指标驱动**：每版单变量改动，用硬件指标（occupancy/stall/throughput）验证机理 |

---

## 三、实验设计方案

### 3.1 研究对象与基线
- **基线**：`karpathy/llm.c` @ `f1e2ace`，**legacy fp32 教学版**（`train_gpt2fp32cu`）
- **模型**：GPT-2 124M，数据集 tinyshakespeare
- **选择 legacy fp32 的理由**（回应"基线选择未论证"）：① 与旧报告研究对象一致，对比公平；② 纯 C/CUDA、无 cuDNN 依赖，每个 kernel 可逐行讲清原理；③ mainline 已是高度优化版本（bf16+Tensor Core+cuDNN attention），再"优化"空间小、且偏离教学目的。

### 3.2 固定实验配置（全程不变）
- **GPU**：RTX 4090（sm_89），锁频 **2520 MHz**，persistence on
- **工具链**：CUDA 12.4.1 / ncu 2024.1.1 / nsys 2023.4.4 / gcc 11.4
- **训练配置**：固定 batch_size、seq_len=1024、固定 seed（以 starter pack 默认为准并记录）
- **统计**：前 N 步 warmup 不计，取 n≥50 次迭代��� **mean ± std**

### 3.3 手写优化版本矩阵（V0→V5，单变量递进）
| 版本 | 改动 | 验证的假设 |
|---|---|---|
| **V0** Baseline | 原版 `matmul_forward_kernel4`（8×8 tile, 128 regs） | 参照点 |
| **V1** | tile 8×8 → 4×4 | 降 register → occupancy？（复现旧 V2，但带正确性验证） |
| **V2** | + `__launch_bounds__` 提升 block 驻留 | 突破 occupancy 上限 |
| **V3** | + shared memory padding（**正确的 33**，非 36） | 消除 bank conflict |
| **V4** | + `cp.async` 异步预取 | 隐藏 global memory 延迟 |
| **V5** | 综合最优手写版 | 手写优化的 FMA 天花板 |

### 3.4 精度 × 架构对比矩阵
| 版本 | 计算路径 | 精度 | 说明 |
|---|---|---|---|
| cuBLAS SGEMM | `cublasSgemm` | FP32 | 库的同精度参照 |
| cuBLASLt TF32 | `cublasLtMatmul` + `FAST_TF32` | TF32 | Tensor Core，**单独一栏** |
| cuBLASLt BF16 | 混合精度 | BF16 | 探索精度换性能 |

> 关键纪律：**架构对比（手写 vs 库）只在 FP32 栏内下结论**；TF32/BF16 的加速单独归因于"精度降低换 Tensor Core"，不与 FP32 混谈。

### 3.5 指标与门禁
- **正确性门禁**（每版必过）：`test_gpt2` 分层数值对比（max abs/rel err < 阈值）+ 固定 batch 的 loss 曲线与 baseline 一致
- **性能指标**：ms/iter、tok/s、MFU（`llmc/mfu.h`）、各 kernel duration（nsys）
- **硬件指标**（ncu）：occupancy、warp stall 构成、compute/memory throughput、Tensor Core util、register 数、shared mem、bank conflict
- **分析指标**：Amdahl（matmul 占比 → 整体加速上限）、roofline

---

## 四、执行流程

### P0 — 基线复现（本次执行）
1. 下载 starter pack（GPT-2 124M fp32 权重 + tokenizer + tinyshakespeare + debug state）
2. 编译 `train_gpt2fp32cu`（`-O3`，sm_89）
3. 跑通基线：记录 ms/iter、tok/s、MFU、loss 曲线
4. nsys 粗定位瓶颈 kernel → ncu 深挖（occupancy/stall/throughput/registers）
5. **产出**：可复现 baseline 数值 + 瓶颈根因报告

### P1 — 手写优化迭代（V1→V5）
每版循环：**改代码（单变量）→ 编译 → 过正确性门禁 → 测性能（mean±std）→ ncu 对比 → 记录机理与数据**
- 产出：版本对比表（每版含 loss 等价性证明 + NCU 指标变化）

### P2 — 精度 × 架构对比
- 产出：FP32 / TF32 / BF16 × 手写 / cuBLAS / cuBLASLt 完整矩阵，精度分栏

### P3 — 冲刺项（可选，视进度）
- 真 BF16 全链路重构 或 Flash Attention（把旧报告"下一步方向"变成"已完成"）

### P4 — 报告撰写
- 方法、数据（mean±std）、失败分析（如 V3 初版失败）、局限性；表格由脚本从原始数据自动生成

---

## 五、验收标准（"经得起推敲"的判据）
- [ ] 所有性能数字同环境、同配置、同一轮实验
- [ ] 每个优化版本都有 loss 等价性证明
- [ ] 精度分栏，架构结论只在同精度栏内
- [ ] 环境 manifest + git commit + 一键复现脚本 + 原始数据公开
- [ ] 保留并正确分析失败实验（不再回避）
- [ ] 结论��条可在数据表中找到出处，无超出数据的断言
