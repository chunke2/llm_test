# llm_test Optimization Progress - 2026-07-28

## 1. Current Decision

The project should be split into two explicit tracks:

1. **Legacy FP32 teaching track**: keep the hand-written FP32 implementation for
   kernel analysis and controlled experiments. D5 is the best validated legacy
   implementation.
2. **Practical training track**: use mainline BF16 with cuDNN FlashAttention and
   `recompute=0`. This is the current overall winner.

The current practical command is:

```bash
make PRECISION=BF16 USE_CUDNN=1 train_gpt2cu test_gpt2cu
./test_gpt2cu
./train_gpt2cu -v 1000 -s 0 -g 1 -h 0 -r 0
```

On the rented RTX 4090, the final configuration reaches **34.405 ms/step**,
compared with **95.599 ms/step** for the repository's default legacy baseline.
That is a **64.01% step-time reduction** at the same `B=4, T=1024` workload.
This comparison crosses precision and implementation tracks, so it is a
practical throughput result, not a same-precision kernel claim.

## 2. Fixed Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| Driver | 575.51.03 |
| Locked graphics clock | 2520 MHz |
| Persistence mode | enabled |
| CUDA toolkit | 12.4.131 |
| Nsight Compute | 2024.1.1 |
| Nsight Systems | 2023.4.4 |
| cuDNN | 9.25.0.15 |
| cuDNN Frontend | v1.5.2 (`98ca4e1`) |
| Repository commit | `bc079355a792f4544d22a3dea5da407300ce33aa` |

All starter-pack files were checked against their official byte counts and
SHA-256 hashes. The original download script had reported success even when the
remote machine could not connect to Hugging Face and no files existed.

## 3. Measurement Rules

- Compile each candidate independently; do not overwrite the baseline source.
- Generate a candidate-specific correctness test instead of using the fixed
  `#include "train_gpt2_fp32.cu"` unchanged.
- Lock the GPU to 2520 MHz and enable persistence mode.
- Use `ABBAAB` ordering for pairwise legacy comparisons.
- Discard the first 10 steps of every run.
- Use three runs per candidate and 64 steady-state steps per run.
- Record raw logs, executable/source hashes, environment data, and JSON stats.
- Treat changes below 0.5% as real but not practically important unless they
  enable a larger follow-up change.
- A simple command gets a fixed timeout and up to two retries. A timed-out
  process is terminated before retrying so duplicate GPU workloads cannot run.

## 4. Legacy Results

### 4.1 Default Mixed FP32/TF32 Track

The repository's default legacy baseline is not strict FP32. Its hand-written
forward matmul is FP32, but cuBLAS backward matmuls run with TF32 enabled.

| Comparison | A (ms) | B (ms) | B vs A | Correctness |
|---|---:|---:|---:|---|
| baseline -> D1 | 95.582 | 95.761 | +0.187% | pass |
| baseline -> D2 | 95.609 | 95.437 | -0.179% | pass |
| D2 -> D3 | 95.438 | 95.235 | -0.213% | pass |
| baseline -> D3 | 95.585 | 95.209 | -0.393% | pass |
| baseline -> D5 | 95.599 | 84.148 | -11.978% | pass |
| D4 -> D5 | 84.862 | 84.131 | -0.862% | pass |

Conclusions:

- D1 GELU fusion is a small regression.
- D2 and D3 residual/layernorm fusions are reproducible, but their combined
  0.39% gain is below the practical threshold.
- D5 is the only large compatible legacy improvement.
- D5's cuBLASLt BIAS epilogue contributes a measurable 0.86% over D4.

### 4.2 Strict FP32 Track

Two controls force `enable_tf32=0` in the training program.

| Comparison | A (ms) | B (ms) | B vs A | Correctness |
|---|---:|---:|---:|---|
| V0-strict -> D3-strict | 105.581 | 105.205 | -0.357% | pass |
| D3-strict -> D5-strict | 105.267 | 102.395 | -2.728% | pass |
| V0-strict -> D5-strict | 105.517 | 102.378 | -2.975% | pass |
| V0-strict -> default V0 | 105.631 | 95.620 | -9.477% | pass |
| D5-strict -> default D5 | 102.351 | 84.135 | -17.797% | pass |

This invalidates the previous statement that D5's entire gain came from lower
precision. TF32 is the largest contributor, but cuBLASLt plus BIAS epilogue
still provides about 2.7% at strict FP32 compute relative to D3-strict.

### 4.3 BF16 Legacy Experiment

D8b is not FP32-reference compatible:

- first checked logit differs by about 0.44;
- initial loss is 5.238554 instead of 5.270009;
- the 10-step reference trajectory fails.

It must therefore remain in a separate BF16 track. Its locked performance is
70.613 ms, 16.09% faster than D5. A 745-step repeated-data stress test passed
the exploratory mixed-precision convergence thresholds:

- all losses finite;
- final validation loss: D5 6.564579, D8b 6.574111;
- final validation delta: +0.009532;
- last-100-step mean train-loss delta: -0.004858.

This is evidence of short-run health, not proof of production convergence.
Both runs overfit the repeated tiny dataset after roughly 300 steps.

## 5. Mainline Results

| Configuration | Mean ms/step | Change from previous |
|---|---:|---:|
| Mainline BF16, no cuDNN, recompute=1 | 44.864 | -36.46% vs D8b |
| Mainline BF16 + cuDNN, recompute=1 | 35.019 | -21.94% |
| Mainline BF16 + cuDNN, recompute=0 | **34.405** | **-1.76%** |
| Previous row + GELU fusion | 34.697 | +0.85% regression |

The built-in BF16/cuDNN test reports `overall okay: 1`. Turning recompute off
does not change the measured validation trajectory:

- recompute=1: validation 4.503205 -> 3.487114, 3663 MiB used;
- recompute=0: validation 4.503205 -> 3.487114, 3927 MiB used.

The 264 MiB increase is small on this 24 GB GPU, so `recompute=0` is the better
default for this fixed GPT-2 124M workload.

## 6. Profile Evidence

Legacy baseline:

- `matmul_forward_kernel4`: 31.1% of GPU kernel time, 2.371 seconds total.
- D5 replaces it with Tensor Core cuBLASLt kernels and saves roughly one second
  across the profiled run.
- GPU memcpy time is effectively unchanged, so memcpy is not the main cause of
  D5's speedup.

Current mainline BF16 + cuDNN:

- BF16 GEMM kernels remain the dominant cost.
- AdamW is about 9.4% of GPU kernel time.
- cuDNN FlashAttention forward/backward plus related kernels are roughly 11%.
- steady-state MFU is about 56.5%.

The next profile-driven work should target matmul shape/throughput and optimizer
cost, not another attention rewrite or CUDA Graph integration.

## 7. Falsified Hypotheses

1. **D3 is a major same-precision optimization**: false. It is stable but only
   0.39%.
2. **D5 is 100% a precision gain**: false. Strict-FP32 D5 is 2.98% faster than
   strict-FP32 V0.
3. **Caching cuBLASLt descriptors and heuristics matters**: false for this
   workload. D9 changed D5 by only -0.0018%.
4. **Legacy D8b is reference-compatible**: false. It fails the FP32 reference
   gate and belongs in a BF16 track.
5. **GELU fusion helps the current mainline best path**: false. It regresses
   performance by 0.85%.
6. **CUDA Graph is the next priority**: unsupported by the profile. Kernel work
   dominates, while integration requires removing incompatible synchronization
   and capture behavior.

## 8. Next Iteration

1. Make mainline BF16 + cuDNN + `recompute=0` the reproducible performance
   target and add a documented build/run target.
2. Repeat convergence tests on a realistic non-repeated dataset and for a
   materially longer schedule before making quality claims.
3. Sweep micro-batch size on the 24 GB GPU and report tokens/second, MFU, and
   memory. Treat changed global batch size as a training-policy change.
4. Profile GEMM shapes individually before attempting cublasLt autotuning.
5. Evaluate optimizer changes only behind a convergence gate; disabling FP32
   master weights may affect quality even if it reduces the 9.4% AdamW cost.

## 9. Reproducibility Artifacts

- `OPTIMIZATION_PLAN.md`
- `experiments/scripts/experiment_runner.py`
- `experiments/scripts/download_starter_pack_verified.sh`
- `experiments/scripts/prepare_dataset_shards.py`
- `experiments/smoke/cuda_smoke.cu`
- `experiments/results/*/validation.json`
- `experiments/results/*/comparison.json`
- `experiments/results/*/benchmark.json`
- `experiments/results/*/*.log`
- `experiments/results/nsys_baseline.nsys-rep`
- `experiments/results/nsys_d5.nsys-rep`
- `experiments/results/nsys_mainline_bf16_cudnn.nsys-rep`
