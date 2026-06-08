#!/usr/bin/env python3
"""Summarize FLIP/APIC/PolyPIC benchmark efficiency runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALGORITHMS = ("flip", "apic", "polypic")
ALGORITHM_LABELS = {"flip": "FLIP", "apic": "APIC", "polypic": "PolyPIC"}
COLORS = {"flip": "#d64f4f", "apic": "#2a9d8f", "polypic": "#4f6edb"}
SCENES = ("dam_break", "liquid_pouring")
SCENE_LABELS = {"dam_break": "Dam Break", "liquid_pouring": "Liquid Pouring"}


def read_energy_csv(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return {
        "frame": np.array([int(r["frame"]) for r in rows]),
        "time": np.array([float(r["time"]) for r in rows]),
        "kinetic_energy": np.array([float(r["kinetic_energy"]) for r in rows]),
        "frame_time_ms": np.array([float(r["frame_time_ms"]) for r in rows]),
    }


def load_rows(raw_dir: Path, warmup_frames: int) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for rep_dir in sorted(raw_dir.glob("rep_*")):
        rep = int(rep_dir.name.split("_", 1)[1])
        for alg in ALGORITHMS:
            for scene in SCENES:
                csv_path = rep_dir / alg / scene / "energy.csv"
                if not csv_path.exists():
                    raise FileNotFoundError(csv_path)
                data = read_energy_csv(csv_path)
                frame_ms = data["frame_time_ms"]
                measured = frame_ms[data["frame"] >= warmup_frames]
                if len(measured) == 0:
                    raise ValueError(f"No frames after warmup in {csv_path}")
                n_particles = estimate_particles(csv_path, scene, warmup_frames)
                mean_ms = float(np.mean(measured))
                summary_rows.append({
                    "repetition": rep,
                    "algorithm": alg,
                    "scene": scene,
                    "rows": int(len(frame_ms)),
                    "warmup_frames": warmup_frames,
                    "mean_ms_per_frame": mean_ms,
                    "std_ms_per_frame": float(np.std(measured, ddof=1)),
                    "median_ms_per_frame": float(np.median(measured)),
                    "p95_ms_per_frame": float(np.percentile(measured, 95)),
                    "min_ms_per_frame": float(np.min(measured)),
                    "max_ms_per_frame": float(np.max(measured)),
                    "total_measured_s": float(np.sum(measured) / 1000.0),
                    "estimated_particles": n_particles,
                    "million_particles_per_s": (
                        float((n_particles / mean_ms) / 1000.0) if n_particles else 0.0
                    ),
                })
    return summary_rows


def estimate_particles(csv_path: Path, scene: str, warmup_frames: int) -> int:
    """Best-effort mean particle count from benchmark stdout log, else 0.

    The per-frame CSV does not record particle count. The benchmark job writes a
    per-algorithm log next to the two scene directories, so we parse the sampled
    ``Frame ... N=...`` lines for the requested scene.
    """
    log_path = csv_path.parent.parent / "run.log"
    if not log_path.exists():
        return 0
    current_scene = None
    counts: list[int] = []
    for line in log_path.read_text(errors="ignore").splitlines():
        if "[run] Scene:" in line:
            try:
                current_scene = line.split("[run] Scene:", 1)[1].split("|", 1)[0].strip()
            except IndexError:
                current_scene = None
            continue
        if current_scene != scene or " Frame " not in line or " N=" not in line:
            continue
        try:
            frame_text = line.split("Frame", 1)[1].split("/", 1)[0].strip()
            frame = int(frame_text)
            n = int(line.rsplit(" N=", 1)[1].split()[0])
        except (IndexError, ValueError):
            continue
        if frame >= warmup_frames:
            counts.append(n)
    return int(round(float(np.mean(counts)))) if counts else 0


def aggregate(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    keys = sorted({(r["algorithm"], r["scene"]) for r in summary_rows})
    for alg, scene in keys:
        rows = [r for r in summary_rows if r["algorithm"] == alg and r["scene"] == scene]
        means = np.array([float(r["mean_ms_per_frame"]) for r in rows])
        medians = np.array([float(r["median_ms_per_frame"]) for r in rows])
        p95s = np.array([float(r["p95_ms_per_frame"]) for r in rows])
        totals = np.array([float(r["total_measured_s"]) for r in rows])
        particles = np.array([float(r["estimated_particles"]) for r in rows])
        mpps = np.array([float(r["million_particles_per_s"]) for r in rows])
        out.append({
            "algorithm": alg,
            "scene": scene,
            "repetitions": len(rows),
            "mean_ms_per_frame": float(np.mean(means)),
            "std_of_run_means_ms": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
            "median_ms_per_frame": float(np.mean(medians)),
            "p95_ms_per_frame": float(np.mean(p95s)),
            "total_measured_s": float(np.mean(totals)),
            "estimated_particles": int(round(float(np.mean(particles)))),
            "million_particles_per_s": float(np.mean(mpps)),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_aggregate(agg_rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        rows = [r for r in agg_rows if r["scene"] == scene]
        rows.sort(key=lambda r: ALGORITHMS.index(str(r["algorithm"])))
        labels = [ALGORITHM_LABELS[str(r["algorithm"])] for r in rows]
        means = np.array([float(r["mean_ms_per_frame"]) for r in rows])
        errs = np.array([float(r["std_of_run_means_ms"]) for r in rows])
        colors = [COLORS[str(r["algorithm"])] for r in rows]

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(labels, means, yerr=errs, capsize=4, color=colors)
        ax.set_title(f"Simulation Time per Frame: {SCENE_LABELS[scene]}")
        ax.set_ylabel("ms / frame, frames >= warmup")
        ax.grid(True, axis="y", alpha=0.28)
        fig.tight_layout()
        fig.savefig(out_dir / f"{scene}_efficiency_ms_per_frame.png", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, scene in zip(axes, SCENES):
        rows = [r for r in agg_rows if r["scene"] == scene]
        rows.sort(key=lambda r: ALGORITHMS.index(str(r["algorithm"])))
        labels = [ALGORITHM_LABELS[str(r["algorithm"])] for r in rows]
        means = np.array([float(r["mean_ms_per_frame"]) for r in rows])
        colors = [COLORS[str(r["algorithm"])] for r in rows]
        speedup = means[0] / means
        ax.bar(labels, speedup, color=colors)
        ax.axhline(1.0, color="#444444", linewidth=0.9)
        ax.set_title(SCENE_LABELS[scene])
        ax.set_ylabel("speedup vs FLIP")
        ax.grid(True, axis="y", alpha=0.28)
    fig.suptitle("Relative Simulation Throughput")
    fig.tight_layout()
    fig.savefig(out_dir / "speedup_vs_flip.png", dpi=220)
    plt.close(fig)


def write_markdown(agg_rows: list[dict[str, object]], out_path: Path) -> None:
    lines = [
        "# Efficiency Benchmark",
        "",
        "This benchmark reruns FLIP, APIC, and PolyPIC under the same Slurm job on an RTX 6000 partition.",
        "Rendering and video export are disabled so the timing focuses on simulation work. The first five frames are discarded to reduce Taichi JIT and initialization effects.",
        "",
        "| Scene | Algorithm | Repetitions | Mean ms/frame | Std of run means | Median ms/frame | P95 ms/frame | M particles/s | Speedup vs FLIP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene in SCENES:
        scene_rows = [r for r in agg_rows if r["scene"] == scene]
        scene_rows.sort(key=lambda r: ALGORITHMS.index(str(r["algorithm"])))
        flip_ms = float(scene_rows[0]["mean_ms_per_frame"])
        for r in scene_rows:
            mean_ms = float(r["mean_ms_per_frame"])
            speedup = flip_ms / mean_ms
            lines.append(
                f"| {SCENE_LABELS[scene]} | {ALGORITHM_LABELS[str(r['algorithm'])]} | "
                f"{int(r['repetitions'])} | {mean_ms:.3f} | "
                f"{float(r['std_of_run_means_ms']):.3f} | "
                f"{float(r['median_ms_per_frame']):.3f} | "
                f"{float(r['p95_ms_per_frame']):.3f} | "
                f"{float(r['million_particles_per_s']):.2f} | {speedup:.2f}x |"
            )
    lines += [
        "",
        "Generated artifacts:",
        "",
        "- `output/benchmark_efficiency/efficiency_runs.csv`",
        "- `output/benchmark_efficiency/efficiency_summary.csv`",
        "- `output/benchmark_efficiency/dam_break_efficiency_ms_per_frame.png`",
        "- `output/benchmark_efficiency/liquid_pouring_efficiency_ms_per_frame.png`",
        "- `output/benchmark_efficiency/speedup_vs_flip.png`",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("output/benchmark_efficiency"))
    parser.add_argument("--doc", type=Path, default=Path("docs/final_report/efficiency_benchmark.md"))
    parser.add_argument("--warmup-frames", type=int, default=5)
    args = parser.parse_args()

    rows = load_rows(args.raw_dir, args.warmup_frames)
    agg = aggregate(rows)
    write_csv(args.out_dir / "efficiency_runs.csv", rows)
    write_csv(args.out_dir / "efficiency_summary.csv", agg)
    plot_aggregate(agg, args.out_dir)
    write_markdown(agg, args.doc)
    print(f"[ok] wrote {args.out_dir}")
    print(f"[ok] wrote {args.doc}")


if __name__ == "__main__":
    main()
