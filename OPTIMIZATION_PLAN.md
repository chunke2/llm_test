# llm.c GPT-2 Optimization Plan

## 1. Objective

Rebuild the optimization study around a reproducible evidence loop:

1. establish an immutable baseline;
2. verify numerical behavior;
3. profile the complete training step;
4. rank falsifiable bottleneck hypotheses;
5. change one variable at a time;
6. run interleaved A/B measurements;
7. keep only changes that pass correctness and statistical gates.

The primary target is single-GPU GPT-2 124M training on an RTX 4090. The
legacy FP32 implementation is retained as the educational kernel-optimization
track. The modern BF16/cuDNN implementation is a separate practical-throughput
track. Results from different precision tracks must not be compared as if they
were equivalent implementations.

## 2. Questions To Answer

1. What consumes wall-clock time in a complete steady-state training step?
2. Which kernels are compute-bound, bandwidth-bound, latency-bound, or launch-bound?
3. How much speedup is possible without changing numerical precision?
4. What speedup comes from TF32 or BF16, and what convergence cost does it carry?
5. Which optimizations remain useful after including conversion, synchronization,
   allocation, heuristic-selection, and host overhead?
6. Does an optimization survive repeated interleaved A/B runs at fixed clocks?
7. Does it generalize beyond one short Tiny Shakespeare pass?

## 3. Fixed Experimental Scope

- GPU: one RTX 4090, same Vast.ai KVM instance for a comparison batch.
- Model: GPT-2 124M.
- Initial workload: Tiny Shakespeare, `B=4`, `T=1024`.
- Extended validation: a fixed longer-running token stream after the fast gate passes.
- Baseline source: repository `main` at a recorded commit.
- Compiler: a pinned CUDA toolkit and complete compile command recorded per binary.
- Randomness: fixed seed, fixed data order, fixed warm-up policy.
- GPU state: persistence mode enabled, graphics clock locked, clock checked before
  and after every run.
- No profiler timings are mixed with normal execution timings.

## 4. Precision Tracks

### Track A: Baseline-Compatible

- Preserve the baseline's actual precision contract.
- Document that the legacy baseline already uses TF32 in backward matmul and
  attention while forward matmul is handwritten FP32.
- Use the strict reference-state test and per-step loss comparison.

### Track B: Explicit FP32

- Disable TF32 everywhere.
- Compare handwritten kernels and cuBLAS only under the same FP32 contract.
- Treat this track as the architecture comparison.

### Track C: TF32

- Enable TF32 explicitly for selected or all GEMMs.
- Attribute gains to precision plus Tensor Core execution, not to library choice alone.
- Require bounded loss divergence against Track A.

### Track D: BF16 Mixed Precision

- Keep FP32 master weights and optimizer state unless an experiment says otherwise.
- Require short deterministic loss-curve checks before performance measurement.
- Require a longer convergence check before a BF16 result is called training-valid.

## 5. Correctness Gates

Every candidate must identify its precision track before it can run.

### Gate 0: Build and Runtime

- clean build;
- no CUDA, cuBLAS, or cuBLASLt errors;
- no invalid memory access under Compute Sanitizer for reduced test shapes;
- deterministic command and artifact naming.

### Gate 1: Reference-State Differential

- build the test harness against the candidate source, not the root baseline;
- compare logits, loss, selected activations, and ten optimization steps;
- use precision-specific absolute and relative tolerances;
- store full output, source hash, binary hash, and pass/fail metadata.

### Gate 2: Short Training Differential

- identical batches and seed;
- compare per-step train loss and scheduled validation loss;
- reject unexplained NaN, Inf, discontinuity, or excessive drift.

### Gate 3: Extended Convergence

- required for TF32/BF16 claims;
- run enough steps to distinguish healthy rounding noise from accumulating divergence;
- compare final validation loss, loss-area-under-curve, parameter norms, and gradient norms.

## 6. Performance Measurement

- Discard initialization and at least ten warm-up steps.
- Measure at least 50 steady-state steps.
- Run baseline and candidate in interleaved order: `A B B A A B`.
- Repeat at least three paired rounds.
- Record mean, median, standard deviation, p50, p95, tokens/s, and paired delta.
- Accept a speedup only when:
  - all correctness gates for the track pass;
  - the paired mean improves by at least 0.5%;
  - the confidence interval does not cross zero;
  - GPU clocks, temperature, and power state remain comparable.
- Re-run any result after NCU, because profiling can reset clock controls.

## 7. Baseline Profiling

### Nsight Systems

Capture the complete steady-state step and quantify:

- GPU kernel time by symbol;
- CUDA API and kernel-launch overhead;
- synchronization and host/device copies;
- cuBLAS/cuBLASLt heuristic or descriptor overhead;
- CPU gaps between GPU launches;
- overlap between conversion kernels and GEMMs.

### Nsight Compute

Profile only the top kernels needed to distinguish hypotheses:

- achieved and theoretical occupancy;
- registers, shared memory, and spills;
- tensor/SIMT pipeline utilization;
- DRAM, L2, and shared-memory throughput;
- warp stall breakdown;
- branch and memory-access efficiency.

Use Amdahl's law to reject directions whose maximum possible end-to-end gain is
smaller than measurement noise or implementation cost.

## 8. Initial Ranked Hypotheses

These are provisional and must be re-ranked after the baseline profile.

1. **Repeated FP32-to-BF16 conversion and per-call cuBLASLt setup limit D8b.**
   Prediction: caching descriptors/algorithms and maintaining BF16 shadow weights
   will reduce CPU gaps and conversion time without changing GEMM output.
2. **The next same-precision opportunity is outside forward matmul.**
   Prediction: after Tensor Core GEMMs, classifier/softmax, attention transforms,
   residual/LayerNorm backward, or optimizer kernels dominate the step.
3. **D3's reported 0.39% gain is near the old measurement floor.**
   Prediction: strict paired A/B testing will either confirm it with a narrow
   interval or reduce it to noise.
4. **CUDA Graphs have low upside unless host synchronization is first removed.**
   Prediction: NSYS will show launch/API overhead below 1% before graph work.
5. **Persistent BF16 dataflow beats repeated casting.**
   Prediction: FP32 master weights plus BF16 shadow weights updated in the optimizer
   reduce conversion traffic while preserving convergence.
6. **Flash Attention is meaningful only on the practical mainline track.**
   Prediction: the legacy attention path is material in Track A, but cuDNN attention
   changes the implementation and should be reported as a separate track.

## 9. Automated Experiment System

Create scripts that:

- capture an environment manifest;
- build any source variant by path;
- generate a candidate-specific correctness test wrapper;
- download and checksum required data;
- lock and verify GPU clocks;
- run paired benchmarks;
- collect GPU telemetry;
- invoke NSYS/NCU with bounded profiles;
- write one JSON record and one raw log directory per run;
- regenerate CSV and Markdown result tables from raw records;
- refuse to benchmark a candidate that failed its correctness gate.

Expected layout:

```text
experiments/
  config/
  scripts/
  raw/<run-id>/
  results.csv
  results.md
  environment.json
```

## 10. Iteration Loop

For each candidate:

1. state one hypothesis and its predicted metric change;
2. create the smallest implementation that tests it;
3. run Gates 0-2;
4. profile only if the prediction needs hardware evidence;
5. run interleaved A/B performance measurements;
6. keep, revise, or reject the candidate;
7. record negative results with the same rigor as positive results;
8. run Gate 3 before promoting a reduced-precision candidate.

No candidate may combine unrelated changes. Combined versions are created only
after individual effects are known.

## 11. Deliverables

- reproducible environment setup;
- immutable baseline record;
- candidate-aware correctness harness;
- automated benchmark and profiler runner;
- raw logs and machine-readable results;
- precision-separated leaderboard;
- accepted and rejected hypothesis ledger;
- final report whose claims can be traced to source, command, and raw data.

## 12. Immediate Execution Order

1. install the pinned CUDA/Nsight toolchain without replacing the working driver;
2. clone this repository onto the GPU machine;
3. download and checksum the starter pack;
4. compile and run the unmodified baseline test;
5. build the automated candidate test and benchmark harness;
6. reproduce V0, D3, D5, and D8b under the new discipline;
7. capture NSYS for V0 and D8b;
8. re-rank the hypotheses from measured evidence;
9. begin single-variable optimization iterations.
