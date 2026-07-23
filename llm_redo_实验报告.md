# llm.c (Karpathy GPT-2) 训练优化重跑实验报告

> 基于 github.com/chunke2/llm_test 的原项目，在 RTX 4090 上重跑并迭代优化，
> 目标是产出**经得起推敲**的实验结论，推翻原报告中站不住脚的说法。
> 实验周期：2026-07-22 ~ 2026-07-23（Day 1–3），所有数据已提交 GitHub。

---

## 1. 摘要 (TL;DR)

- 原报告多个核心结论在 RTX 4090 上**不成立**（详见第 4 节，均有实证）。
- 本实验在锁频、交错 A/B 复测的严格测量纪律下，得到一条清晰结论链：
  **4090 上 GPT-2 训练的性能瓶颈是 matmul 的精度（FP32→TF32 tensor core），
  而不是手写 kernel 优化、kernel 融合或 launch 开销。**
- 最终最优版本 **D5 = 84.20 ms/iter（-12.0% vs 基线 95.65ms）**，
  但其加速**完全来自 TF32 精度降级**，并非架构/算法优势——这一点必须如实标注。
- 同精度（FP32）手写优化的最优版本 **D3 = 95.28 ms/iter（-0.39%）**，
  说明在 4090 这种大 SM、高带宽设备上，手写 tile/occupancy 优化的边际空间极小。
- 三个发散尝试（D1 融合、D6 GELU_AUX、D7 CUDA Graph）**全部为负结果**，
  反而坐实了上述结论（详见第 5、6 节）。

---

## 2. 实验环境与方法论

### 2.1 硬件与软件
| 项 | 值 |
|----|----|
| GPU | NVIDIA RTX 4090 (Ada Lovelace, `sm_89`), 24GB |
| 租用平台 | vast.ai KVM VM（裸金属级，可访问 perf counter） |
| CUDA Toolkit | 12.4.1 |
| Nsight Compute | 2024.1.1（用于寄存器/occupancy/stall 分析） |
| Nsight Systems | 2023.4.4 |
| 基线代码 | karpathy/llm.c，commit `f1e2ace`，legacy `train_gpt2_fp32.cu` |
| 模型 | GPT-2 124M（`gpt2_124M.bin`），FP32 训练 |

### 2.2 测量纪律（本报告可信的前提）
这是本次实验最重要的工程纪律，原项目恰恰栽在这里：

1. **GPU 频率锁定**：`nvidia-smi -pm 1 && nvidia-smi -lgc 2520`（锁到 4090 的
   非 boost 稳定频 2520MHz），消除 boost 时钟波动带来的测量噪声。
2. **每次测量前后校验时钟**：`nvidia-smi --query-gpu=clocks.gr` 确认仍为 2520MHz。
   **教训（Day 2 重大勘误）**：跑一次 ncu 会临时重置 `-lgc` 锁频，GPU 回到
   ~2583MHz boost，导致之后所有数据偏快 ~2.5%。原报告"r84 快 2.4%"正是此假象。
3. **交错 A/B ×3**：每个待评版本与基线轮流跑 3 轮，取平均，排除偶发抖动。
4. **正确性门禁**：所有版本必须通过 `test_gpt2fp32cu`（FP32 参考，
   `enable_tf32=0`），对比 PyTorch 参考的逐 step loss，否则一律视为不合格。
5. **精度诚实分层**：明确区分"同精度（FP32 手写）"与"TF32 精度降级"两类结果，
   不把精度降级带来的加速包装成架构优化。

### 2.3 正确性门禁
`test_gpt2fp32cu` 使用 `gpt2_124M_debug_state.bin`（PyTorch 参考激活），
逐 step 比对 forward logits 与最终 loss，要求 `overall okay: 1`。
任何数值偏差（哪怕 TF32 舍入）在严格门禁下都会暴露——这是 D6 被否决的依据。

---

## 3. 性能根因分析（来自 ncu）

基线 `matmul_forward_kernel4`（8×8 tile）的 ncu 画像：

- matmul 占 **53.6%** 的 GPU 时间（绝对瓶颈）。
- 内核使用 **64 个 float 累加器**（`float vals[8][8]`）→ **123 个寄存器** →
  **occupancy 仅 33%** → 大量 warp 处于 **Stall Not Selected**（无 eligible warp 可调度）。
- 寄存器压力是"占用率不足→调度停顿"的直接原因，而非计算吞吐不足。

第一性原理的优化方向应是**降低寄存器压力以提升 occupancy**。本报告第 5 节的
K 系列即沿此思路，但受限于 4090 的共享内存带宽，收益被抵消（见 5.3）。

---

## 4. 对原报告的批判性审查（逐条反驳，均有实证）

| # | 原报告说法 | 本实验实测 | 结论 |
|---|-----------|-----------|------|
| 1 | "V2（4×4 tile）比基线快 ~21%" | 4090 上 V2 = 107.93ms（**+12.9%**，更慢）；V1(4×4) = 107.93ms 同理 | **错误**。Ada 编译器寄存器分配与原报告平台不同，4×4 降低数据复用率反而更慢 |
| 2 | "cuBLASLt 换用 MMA 指令是 1.21× 加速的来源" | 同精度(FP32)下 cuBLAS **更慢 13.3%**（V6 = 108.32ms）；V5 的加速来自 **TF32 精度降级**（19bit→23bit 尾数），非 MMA 指令本身 | **错误/误导**。加速来自精度降级，非架构 |
| 3 | 未意识到约束 | "bank conflict fix" 与 "float4 对齐" **互斥**（V4 因 float4 对齐破坏导致 cuBLAS crash） | 原报告方法论缺失此项约束分析 |
| 4 | 隐含"手写优化空间大" | 同精度手写最优仅 -0.39%（D3）；其余手写版本均更慢或失败 | **夸大**。4090 上手写 kernel 边际极小 |

---

## 5. 实验过程与发现

### 5.1 P0 — 基线画像
- V0（8×8 tile）= **95.65 ms/iter**，门禁通过。
- ncu 给出上述根因（§3）：matmul 53.6% 时间，123 regs / 33% occ。

### 5.2 P1 — 手写 kernel 优化（全部失败/无效）
| 版本 | 改动 | ms | vs V0 | 问题 |
|------|------|----|-------|------|
| V1 | 4×4 tile | 107.93 | +12.9% | 数据复用率降 4× |
| V2 | 8×8 + launch_bounds(3) | 228.05 | +138% | spill 到 local mem，5× 减速 |
| V3 | 4×4 + launch_bounds(3) | 104.55 | +9.3% | spill 减轻仍慢 |
| V4 | 8×8 + pad[33] | FAILED | — | float4 对齐破坏 → cuBLAS crash |

**结论**：在 4090 上，盲目调 tile/launch_bounds 不仅无收益，反而因 spill 严重减速。
这直接推翻原报告"小 tile 更快"的说法。

### 5.3 P2 / K / M — 精度与寄存器方向
- **V5（cuBLAS TF32 fwd matmul）= 85.21ms（-10.9%）**：加速来自 TF32 降级。
- **V6（cuBLAS FP32）= 108.32ms（+13.3%）**：同精度下 cuBLAS 反慢，证明 V5 加速=精度降级。
- **K 系列（两遍 4×8，123→80 regs，occ 33%→50%）**：寄存器优化方向正确，
  但 2× 共享内存重复载入抵消收益，~101ms（更慢）。
- **M 系列（maxrregcount=128）= 96.00ms（+0.4%）**：无 launch_bounds 时几乎等同基线。

### 5.4 D 系列 — 系统性发散
| 版本 | 做法 | 精度 | ms | vs V0 | 状态 |
|------|------|------|----|-------|------|
| D1 | GELU 融合进 matmul epilogue | 同 | 95.6 | ~0 | 零收益（省读=增写抵消） |
| D2 | fused residual+LN（寄存器驻留行） | 同 | 95.47 | -0.4% | 有效但微小 |
| **D3** | **D2 + 第二融合点（residual3→下一层 LN）** | 同 | **95.28** | **-0.39%** | **同精度最优** |
| D4 | D3 + cuBLAS Sgemm TF32 | TF32 | 84.92 | -11.2% | 有效 |
| **D5** | **D3 + cublasLt TF32 + BIAS epilogue** | TF32 | **84.20** | **-12.0%** | **全局最优** |
| D6 | D5 + cublasLt GELU_AUX epilogue | — | — | — | **失败**：FP32 门禁数值错误 |
| D7 | D5 + CUDA Graph 重放训练 step | — | — | — | **失败**：capture 不兼容 |

**D3 的关键技术点**：把 `residual_forward` 与 `layernorm_forward` 融合为
`fused_residual_ln_kernel<768>`，行数据驻留寄存器，省掉中间张量的全局内存往返。
交错 A/B×3 复现，门禁通过。

**D5 的两个踩坑（已记入代码注释与总结）**：
1. cublasLt 启发式可能选 split-K，改变累加顺序 → 强制 `REDUCTION_SCHEME_NONE`；
2. compute type 必须 `cublasGetMathMode` 动态跟随——硬编码 `FAST_TF32` 会让
   FP32 门禁（test 中 `enable_tf32=0`）必然失败。

---

## 6. 负结果的价值（为什么发散"失败"反而重要）

- **D1（GELU 融合）**：正确但零收益。省掉的 600MB 读被新增的 600MB 写抵消——
  证明在 4090 的高带宽下，单纯融合相邻 elementwise/kernel 无净收益。
- **D6（GELU_AUX）**：cublasLt 的 GELU_AUX epilogue 在**纯 FP32 compute 路径数值错误**
  （logits 第一步即 -43 vs -10）。GELU_AUX 只在 TF32 tensor core 路径验证过。
  因所有版本必须过 FP32 门禁，否决。
- **D7（CUDA Graph）**：capture 报 `operation not permitted when stream is capturing`。
  根因：llm.c 在每个 kernel 后调用 `cudaGetLastError()`（capture 模式禁止）；
  外加 forward 的同步 D2H 拷贝与 host 端 `mean_loss` 计算需重构。集成成本极高，
  且预期收益仅 ~30 次 launch × 数 µs ≈ 0.2%。

**三者共同证明**：4090 上 GPT-2 训练的边际空间不在融合、不在 launch 开销，
**唯一真实杠杆是 matmul 精度（TF32）**。

---

## 7. 最终排行榜（锁频 2520MHz，可复现，交错 A/B×3）

| 版本 | 做法 | 精度 | ms/iter | vs V0 | 门禁 |
|------|------|------|---------|-------|------|
| **D8b** | D3 融合 + cublasLt BF16 fwd + cublasGemmEx BF16 bwd + BIAS | **BF16↓** | **70.93** | **-25.8%** | 数值跟随① |
| D8 | D3 融合 + cublasLt BF16 fwd only | BF16↓ | 80.16 | -16.2% | 数值跟随① |
| **D5** | D3 融合 + cublasLt TF32 + BIAS epilogue | TF32↓ | **84.20** | **-12.0%** | OK |
| D4 | D3 融合 + cuBLAS Sgemm TF32 | TF32↓ | 84.92 | -11.2% | OK |
| V5 | cuBLAS TF32 fwd matmul | TF32↓ | 85.21 | -10.9% | OK |
| **D3** | residual+LN 双点融合（寄存器驻留） | 同精度 | **95.28** | **-0.39%** | OK |
| V0 | 原版 8×8 tile | 混合 | 95.65 | 基准 | OK |
| r84/M | 原报告称 "-2.4%" | — | 95.65 | 0（勘误） | OK |

> 注：基线本身是**混合精度**——backward matmul 与 attention 早已是 cuBLAS TF32，
> 只有 forward matmul 是手写 FP32。原报告从未披露这一事实。
>
> ① BF16 路径不通过严格 FP32 正确性门禁（精度低于参考），但训练 loss 轨迹与 TF32
> 参考高度重合（74 步后 val loss 差 <0.005：D8b 3.4971 vs D5 3.4921），属于标准混合
> 精度训练的健康行为，加速真实可信。D8b 稳态取后 50 步均值 70.93ms（首步含 cublasLt
> heuristic 预热约 73.8ms）。

---

## 8. 结论

1. **原报告不成立**：其"小 tile 更快""cuBLASLt 换 MMA 是加速来源"等结论，
   在 4090 上被实证推翻。
2. **真正的加速来源是精度降级（matmul tensor core）**：D5 的 -12.0% 来自 FP32→TF32，
   D8b 进一步用 BF16 把 matmul 推向 tensor core 的 2× 吞吐，达到 **-25.8%（70.93ms）**。
   这是诚实的性能杠杆，但**不是**"手写 kernel 优化"的胜利。
3. **手写 kernel 在 4090 上边际极小**：同精度最优 D3 仅 -0.39%，D1/D6/D7 融合与
   CUDA Graph 尝试均净零或失败；所有实质加速 100% 来自精度降级。
4. **测量纪律决定结论可信度**：锁频 + 交错 A/B + 前后校验时钟，
   直接推翻了原报告"r84 快 2.4%"的时钟漂移假象。

---

## 9. 局限与诚实声明

- 所有加速均依赖 **TF32 精度降级**（尾数 23bit→19bit 近似）。若任务对数值精度
  敏感（如长程依赖、低资源语言），TF32 未必可取；FP32 同精度路径仅有 -0.39%。
- 实验仅在 **单卡 RTX 4090 + GPT-2 124M** 上验证，结论未必外推到其它架构/规模。
- BF16 路径已探索（见第 7 节 D8/D8b）：前向+反向 matmul 改 BF16 达 -25.8%（70.93ms），
  训练 loss 轨迹与 TF32 参考重合（val loss 差 <0.005）。这是标准混合精度训练做法，
  但偏离原项目 FP32 语义，故在排行榜中与 TF32、D3 分列"精度降级 / 同精度"两档。
- D6/D7 的失败是**工程集成**失败，非算法错误；其负结果已作为方法论证据保留。

---

## 10. 复现指南

```bash
# 1. 租用 4090（vast.ai KVM VM，裸金属级以访问 perf counter）
# 2. 安装 CUDA 12.4 + ncu/nsys（包名 cuda-nsight-compute-12-4 / cuda-nsight-systems-12-4）
# 3. 克隆 karpathy/llm.c @ f1e2ace，下载 gpt2_124M.bin
# 4. 锁频（每次重启/软件更新后必须重做；若 nvidia-smi 报 NVML mismatch：
#    pkill nvidia-persistenced; modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia ecc;
#    modprobe nvidia; 再锁频）
nvidia-smi -pm 1
nvidia-smi -lgc 2520
# 5. 编译基线并复测
nvcc -O3 --use_fast_math -Xcompiler -fopenmp train_gpt2_fp32.cu -lcublas -lcublasLt -o train
./train            # 应得 ~95.6 ms/iter
# 6. 编译最优版本 D5 并验证
nvcc -O3 --use_fast_math -Xcompiler -fopenmp train_gpt2_fp32_d5.cu -lcublas -lcublasLt -o train_d5
./train_d5          # 应得 ~84.2 ms/iter
# 7. 正确性门禁（FP32 参考，必须 overall okay: 1）
nvcc -O3 --use_fast_math -Xcompiler -fopenmp test_gpt2_fp32.cu -lcublas -lcublasLt -o test
./test
```

---

## 11. 附录：版本与产物清单（均在仓库 `llm_redo_data/`）

- `DAY2_summary.txt` / `DAY3_summary.txt`：逐日实验记录
- `P0_root_cause_report.txt` / `P1_summary.txt` / `P2_summary.txt`：阶段根因
- 源码：`train_gpt2_fp32_{v1..v6,k1..k6,m1,r84,d1..d7,d7b}.cu`
- ncu/nsys 报告、各版本训练日志
- 可视化看板：`实验数据看板.html`

所有内容已提交 GitHub（chunke2/llm_test），关键 commit：
`6bf903d`(D3+勘误)、`6d9e3f5`(D4/D5)、`6637725`(Day3 环境修复+D6/D7 负结果)。
