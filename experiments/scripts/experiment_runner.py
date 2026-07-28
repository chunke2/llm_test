#!/usr/bin/env python3
"""Build, validate, and benchmark one llm_test CUDA candidate reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_PATTERN = re.compile(
    r"step\s+\d+/\d+: train loss\s+[-+0-9.eE]+\s+\(([0-9.]+) ms,"
)
MAINLINE_STEP_PATTERN = re.compile(
    r"step\s+\d+/\d+\s+\|.*?\|\s+([0-9.]+) ms\s+\|"
)
TRAIN_STEP_PATTERN = re.compile(
    r"step\s+(\d+)/(\d+): train loss\s+([-+0-9.eE]+)\s+\(([0-9.]+) ms,"
)
VAL_PATTERN = re.compile(r"^val loss\s+([-+0-9.eE]+)$", re.MULTILINE)
INCLUDE_PATTERN = '#include "train_gpt2_fp32.cu"'
NVCC = "/usr/local/cuda/bin/nvcc"


def run_command(
    command: list[str],
    cwd: Path,
    timeout_seconds: int,
    retries: int = 2,
) -> subprocess.CompletedProcess[str]:
    """Run a command, retrying only timeouts and transient SSH-style exits."""
    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, retries + 2):
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            if attempt > retries:
                raise RuntimeError(
                    f"command timed out after {attempt} attempts: {' '.join(command)}"
                ) from error
            time.sleep(attempt)
            continue

        last_result = result
        if result.returncode == 0:
            return result
        if result.returncode not in (124, 137, 255) or attempt > retries:
            break
        time.sleep(attempt)

    assert last_result is not None
    raise RuntimeError(
        f"command failed with exit {last_result.returncode}: {' '.join(command)}\n"
        f"{last_result.stdout[-4000:]}"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_environment(repo_root: Path) -> dict[str, Any]:
    commands = {
        "git_commit": ["git", "rev-parse", "HEAD"],
        "nvcc": [NVCC, "--version"],
        "gpu": [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,clocks.current.graphics,"
            "persistence_mode,power.limit",
            "--format=csv,noheader",
        ],
        "os": ["uname", "-a"],
    }
    environment: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for key, command in commands.items():
        result = run_command(command, repo_root, timeout_seconds=20)
        environment[key] = result.stdout.strip()
    return environment


def candidate_paths(repo_root: Path, name: str) -> tuple[Path, Path, Path]:
    bin_dir = repo_root / "experiments" / "bin"
    generated_dir = repo_root / "experiments" / "generated"
    bin_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    return (
        bin_dir / name,
        generated_dir / f"test_{name}.cu",
        bin_dir / f"test_{name}",
    )


def compile_candidate(repo_root: Path, name: str, source: Path) -> Path:
    executable, _, _ = candidate_paths(repo_root, name)
    command = [
        NVCC,
        "--threads=0",
        "--use_fast_math",
        "--std=c++17",
        "-O3",
        "-lineinfo",
        "-I",
        str(repo_root),
        str(source),
        "-lcublas",
        "-lcublasLt",
        "-lnvidia-ml",
        "-o",
        str(executable),
    ]
    result = run_command(command, repo_root, timeout_seconds=180)
    if result.stdout:
        print(result.stdout, end="")
    return executable


def compile_candidate_test(
    repo_root: Path,
    name: str,
    source: Path,
) -> Path:
    _, generated_test, test_executable = candidate_paths(repo_root, name)
    base_test = repo_root / "test_gpt2_fp32.cu"
    test_text = base_test.read_text(encoding="utf-8")
    if test_text.count(INCLUDE_PATTERN) != 1:
        raise RuntimeError(f"expected exactly one fixed include in {base_test}")
    relative_source = os.path.relpath(source, generated_test.parent).replace("\\", "/")
    generated_test.write_text(
        test_text.replace(
            INCLUDE_PATTERN,
            f'#include "{relative_source}"',
        ),
        encoding="utf-8",
    )
    command = [
        NVCC,
        "--threads=0",
        "--use_fast_math",
        "--std=c++17",
        "-O3",
        "-lineinfo",
        "-I",
        str(repo_root),
        str(generated_test),
        "-lcublas",
        "-lcublasLt",
        "-lnvidia-ml",
        "-o",
        str(test_executable),
    ]
    result = run_command(command, repo_root, timeout_seconds=180)
    if result.stdout:
        print(result.stdout, end="")
    return test_executable


def validate(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    source = (repo_root / args.source).resolve()
    executable = compile_candidate(repo_root, args.name, source)
    test_executable = compile_candidate_test(repo_root, args.name, source)
    test_result = run_command(
        [str(test_executable)],
        repo_root,
        timeout_seconds=args.timeout,
    )

    result_dir = repo_root / "experiments" / "results" / args.name
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "correctness.log").write_text(
        test_result.stdout,
        encoding="utf-8",
    )
    passed = "overall okay: 1" in test_result.stdout
    record = {
        "candidate": args.name,
        "source": str(source.relative_to(repo_root)),
        "source_sha256": sha256(source),
        "executable": str(executable.relative_to(repo_root)),
        "correctness_passed": passed,
        "environment": collect_environment(repo_root),
    }
    (result_dir / "validation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"validation candidate={args.name} correctness_passed={passed}")
    if not passed:
        print(test_result.stdout[-4000:])
        raise SystemExit(2)


def summarize(samples: list[float]) -> dict[str, float | int]:
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        index = fraction * (len(ordered) - 1)
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "stddev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p05_ms": percentile(0.05),
        "p95_ms": percentile(0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def run_benchmark_once(
    repo_root: Path,
    executable: Path,
    timeout_seconds: int,
    warmup: int,
    style: str = "legacy",
    recompute: int | None = None,
    gelu_fusion: int | None = None,
) -> tuple[str, list[float], list[float]]:
    if style == "mainline":
        command = [
            str(executable),
            "-v",
            "1000",
            "-s",
            "0",
            "-g",
            "1",
            "-h",
            "0",
        ]
        if recompute is not None:
            command.extend(["-r", str(recompute)])
        if gelu_fusion is not None:
            command.extend(["-ge", str(gelu_fusion)])
        pattern = MAINLINE_STEP_PATTERN
    else:
        command = [
            str(executable),
            "-v",
            "1000",
            "-s",
            "1000",
            "-g",
            "1",
        ]
        pattern = STEP_PATTERN
    result = run_command(command, repo_root, timeout_seconds=timeout_seconds)
    samples = [float(value) for value in pattern.findall(result.stdout)]
    if len(samples) <= warmup:
        raise RuntimeError(
            f"benchmark produced {len(samples)} steps, not enough for warmup={warmup}"
        )
    return result.stdout, samples, samples[warmup:]


def benchmark(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    executable = (repo_root / args.executable).resolve()
    if not executable.is_file():
        raise RuntimeError(f"missing executable: {executable}")

    result_dir = repo_root / "experiments" / "results" / args.name
    result_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    all_steady_samples: list[float] = []
    for run_index in range(1, args.repeats + 1):
        output, samples, steady = run_benchmark_once(
            repo_root,
            executable,
            args.timeout,
            args.warmup,
            args.style,
            args.recompute,
            args.gelu_fusion,
        )
        (result_dir / f"benchmark_run_{run_index}.log").write_text(
            output,
            encoding="utf-8",
        )
        all_steady_samples.extend(steady)
        runs.append(
            {
                "run": run_index,
                "all_steps": summarize(samples),
                "steady_steps": summarize(steady),
            }
        )
        print(
            f"benchmark candidate={args.name} run={run_index} "
            f"steady_mean_ms={statistics.fmean(steady):.6f}"
        )

    record = {
        "candidate": args.name,
        "executable": str(executable.relative_to(repo_root)),
        "executable_sha256": sha256(executable),
        "warmup_steps_discarded_per_run": args.warmup,
        "runs": runs,
        "aggregate_steady_steps": summarize(all_steady_samples),
        "environment": collect_environment(repo_root),
    }
    output_path = result_dir / "benchmark.json"
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"benchmark_complete candidate={args.name} "
        f"aggregate_mean_ms={record['aggregate_steady_steps']['mean_ms']:.6f} "
        f"result={output_path}"
    )


def compare(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    candidates = {
        "A": {
            "name": args.a_name,
            "executable": (repo_root / args.a_executable).resolve(),
        },
        "B": {
            "name": args.b_name,
            "executable": (repo_root / args.b_executable).resolve(),
        },
    }
    for candidate in candidates.values():
        if not candidate["executable"].is_file():
            raise RuntimeError(f"missing executable: {candidate['executable']}")

    schedule = args.schedule.upper()
    if not schedule or set(schedule) - {"A", "B"}:
        raise RuntimeError("schedule must contain only A and B")

    comparison_name = f"{args.a_name}_vs_{args.b_name}"
    result_dir = repo_root / "experiments" / "results" / comparison_name
    result_dir.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, list[float]] = {"A": [], "B": []}
    run_means: dict[str, list[float]] = {"A": [], "B": []}
    runs: list[dict[str, Any]] = []

    for run_index, label in enumerate(schedule, start=1):
        candidate = candidates[label]
        output, samples, steady = run_benchmark_once(
            repo_root,
            candidate["executable"],
            args.timeout,
            args.warmup,
        )
        log_name = f"{run_index:02d}_{label}_{candidate['name']}.log"
        (result_dir / log_name).write_text(output, encoding="utf-8")
        aggregate[label].extend(steady)
        run_mean = statistics.fmean(steady)
        run_means[label].append(run_mean)
        runs.append(
            {
                "order": run_index,
                "label": label,
                "candidate": candidate["name"],
                "all_steps": summarize(samples),
                "steady_steps": summarize(steady),
            }
        )
        print(
            f"compare order={run_index} label={label} "
            f"candidate={candidate['name']} steady_mean_ms={run_mean:.6f}"
        )

    a_mean = statistics.fmean(run_means["A"])
    b_mean = statistics.fmean(run_means["B"])
    delta_percent = (b_mean / a_mean - 1.0) * 100.0
    record = {
        "comparison": comparison_name,
        "schedule": schedule,
        "warmup_steps_discarded_per_run": args.warmup,
        "candidates": {
            label: {
                "name": candidate["name"],
                "executable": str(candidate["executable"].relative_to(repo_root)),
                "executable_sha256": sha256(candidate["executable"]),
                "run_means_ms": run_means[label],
                "run_mean_summary": summarize(run_means[label]),
                "aggregate_steady_steps": summarize(aggregate[label]),
            }
            for label, candidate in candidates.items()
        },
        "b_vs_a_delta_percent": delta_percent,
        "runs": runs,
        "environment": collect_environment(repo_root),
    }
    output_path = result_dir / "comparison.json"
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"compare_complete comparison={comparison_name} "
        f"a_mean_ms={a_mean:.6f} b_mean_ms={b_mean:.6f} "
        f"b_vs_a_delta_percent={delta_percent:.4f} result={output_path}"
    )


def parse_training_trace(output: str) -> dict[str, Any]:
    steps = [
        {
            "step": int(step),
            "total_steps": int(total),
            "loss": float(loss),
            "time_ms": float(time_ms),
        }
        for step, total, loss, time_ms in TRAIN_STEP_PATTERN.findall(output)
    ]
    validation_losses = [float(value) for value in VAL_PATTERN.findall(output)]
    if not steps or len(validation_losses) < 2:
        raise RuntimeError(
            f"incomplete convergence trace: steps={len(steps)} "
            f"validation_losses={len(validation_losses)}"
        )
    return {
        "steps": steps,
        "validation_losses": validation_losses,
    }


def convergence(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    candidates = [
        (args.a_name, (repo_root / args.a_executable).resolve()),
        (args.b_name, (repo_root / args.b_executable).resolve()),
    ]
    result_dir = (
        repo_root
        / "experiments"
        / "results"
        / f"convergence_{args.a_name}_vs_{args.b_name}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    traces: dict[str, dict[str, Any]] = {}

    for name, executable in candidates:
        if not executable.is_file():
            raise RuntimeError(f"missing executable: {executable}")
        command = [
            str(executable),
            "-i",
            args.train_data,
            "-v",
            str(args.val_every),
            "-s",
            "1000",
            "-g",
            "1",
        ]
        result = run_command(command, repo_root, timeout_seconds=args.timeout)
        (result_dir / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        traces[name] = parse_training_trace(result.stdout)
        print(
            f"convergence candidate={name} "
            f"steps={len(traces[name]['steps'])} "
            f"final_val_loss={traces[name]['validation_losses'][-1]:.6f}"
        )

    a_trace = traces[args.a_name]
    b_trace = traces[args.b_name]
    if len(a_trace["steps"]) != len(b_trace["steps"]):
        raise RuntimeError("candidate traces have different step counts")

    loss_differences = [
        abs(a_step["loss"] - b_step["loss"])
        for a_step, b_step in zip(a_trace["steps"], b_trace["steps"])
    ]
    tail_count = min(args.tail_steps, len(a_trace["steps"]))
    a_tail_mean = statistics.fmean(
        step["loss"] for step in a_trace["steps"][-tail_count:]
    )
    b_tail_mean = statistics.fmean(
        step["loss"] for step in b_trace["steps"][-tail_count:]
    )
    final_val_delta = (
        b_trace["validation_losses"][-1] - a_trace["validation_losses"][-1]
    )
    finite = all(
        math.isfinite(step["loss"])
        for trace in traces.values()
        for step in trace["steps"]
    )
    passed = (
        finite
        and abs(final_val_delta) <= args.val_tolerance
        and abs(b_tail_mean - a_tail_mean) <= args.train_tolerance
    )
    record = {
        "comparison": f"{args.a_name}_vs_{args.b_name}",
        "train_data": args.train_data,
        "steps": len(a_trace["steps"]),
        "tail_steps": tail_count,
        "candidates": {
            args.a_name: {
                "validation_losses": a_trace["validation_losses"],
                "tail_train_loss_mean": a_tail_mean,
            },
            args.b_name: {
                "validation_losses": b_trace["validation_losses"],
                "tail_train_loss_mean": b_tail_mean,
            },
        },
        "pointwise_train_loss_abs_difference": summarize(loss_differences),
        "final_validation_loss_delta_b_minus_a": final_val_delta,
        "tail_train_loss_delta_b_minus_a": b_tail_mean - a_tail_mean,
        "finite_losses": finite,
        "thresholds": {
            "absolute_final_validation_loss_delta": args.val_tolerance,
            "absolute_tail_train_loss_mean_delta": args.train_tolerance,
        },
        "convergence_gate_passed": passed,
        "environment": collect_environment(repo_root),
    }
    output_path = result_dir / "convergence.json"
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"convergence_complete comparison={record['comparison']} "
        f"final_val_delta={final_val_delta:.6f} "
        f"tail_train_delta={b_tail_mean - a_tail_mean:.6f} "
        f"gate_passed={passed} result={output_path}"
    )
    if not passed:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--name", required=True)
    validate_parser.add_argument("--source", required=True)
    validate_parser.add_argument("--timeout", type=int, default=300)
    validate_parser.set_defaults(handler=validate)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--name", required=True)
    benchmark_parser.add_argument("--executable", required=True)
    benchmark_parser.add_argument("--repeats", type=int, default=3)
    benchmark_parser.add_argument("--warmup", type=int, default=10)
    benchmark_parser.add_argument(
        "--style",
        choices=("legacy", "mainline"),
        default="legacy",
    )
    benchmark_parser.add_argument("--recompute", type=int, choices=(0, 1, 2))
    benchmark_parser.add_argument("--gelu-fusion", type=int, choices=(0, 1))
    benchmark_parser.add_argument("--timeout", type=int, default=300)
    benchmark_parser.set_defaults(handler=benchmark)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--a-name", required=True)
    compare_parser.add_argument("--a-executable", required=True)
    compare_parser.add_argument("--b-name", required=True)
    compare_parser.add_argument("--b-executable", required=True)
    compare_parser.add_argument("--schedule", default="ABBAAB")
    compare_parser.add_argument("--warmup", type=int, default=10)
    compare_parser.add_argument("--timeout", type=int, default=300)
    compare_parser.set_defaults(handler=compare)

    convergence_parser = subparsers.add_parser("convergence")
    convergence_parser.add_argument("--a-name", required=True)
    convergence_parser.add_argument("--a-executable", required=True)
    convergence_parser.add_argument("--b-name", required=True)
    convergence_parser.add_argument("--b-executable", required=True)
    convergence_parser.add_argument("--train-data", required=True)
    convergence_parser.add_argument("--tail-steps", type=int, default=100)
    convergence_parser.add_argument("--val-every", type=int, default=74)
    convergence_parser.add_argument("--val-tolerance", type=float, default=0.02)
    convergence_parser.add_argument("--train-tolerance", type=float, default=0.02)
    convergence_parser.add_argument("--timeout", type=int, default=600)
    convergence_parser.set_defaults(handler=convergence)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except RuntimeError as error:
        print(f"experiment_error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
