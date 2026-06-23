#!/usr/bin/env python3
"""Build Option 1 comparison artifacts for the final project report.

This script reads the committed FLIP/APIC/PolyPIC energy CSVs, generates
comparison figures and summary tables, and writes a Markdown plus PDF report.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from build_latex_report import build_latex_report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "comparison"
REPORT_DIR = ROOT / "docs" / "final_report"
FIG_DIR = REPORT_DIR / "figures"
SHOT_DIR = FIG_DIR / "screenshots"
BENCHMARK_DIR = ROOT / "output" / "benchmark_efficiency"
FLIP_SWEEP_DIR = ROOT / "output" / "flip"
FLIP_SWEEP_FIG_DIR = ROOT / "output" / "flip_sweep"

# Available FLIP ratios (PIC-FLIP blend: 0.0 = pure PIC, 1.0 = pure FLIP)
FLIP_RATIOS = [0.0, 0.5, 0.8, 0.95, 0.97, 1.0]

ALGORITHMS = {
    "FLIP": {
        "color": "#d64f4f",
        "base": ROOT / "output" / "flip" / "ratio_970",
        "description": "baseline FLIP/PIC blended transfer with flip_ratio=0.97",
    },
    "APIC": {
        "color": "#2a9d8f",
        "base": ROOT / "output" / "apic" / "ratio_970",
        "description": "affine particle-in-cell transfer with stabilized affine matrix",
    },
    "PolyPIC": {
        "color": "#4f6edb",
        "base": ROOT / "output" / "polypic" / "ratio_970",
        "description": "polynomial particle-in-cell transfer using higher-order local moments",
    },
}
SCENES = ["dam_break", "liquid_pouring"]
SCENE_LABELS = {"dam_break": "Dam Break", "liquid_pouring": "Liquid Pouring"}
EFFICIENCY_FIGURES = [
    "dam_break_efficiency_ms_per_frame.png",
    "liquid_pouring_efficiency_ms_per_frame.png",
    "speedup_vs_flip.png",
]


def load_csv(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return {
        "frame": np.array([int(r["frame"]) for r in rows], dtype=int),
        "time": np.array([float(r["time"]) for r in rows], dtype=float),
        "kinetic_energy": np.array([float(r["kinetic_energy"]) for r in rows], dtype=float),
        "frame_time_ms": np.array([float(r["frame_time_ms"]) for r in rows], dtype=float),
        "total_vorticity": np.array([float(r.get("total_vorticity", 0.0)) for r in rows], dtype=float),
    }


def load_flip_sweep_data(scene: str) -> dict[float, dict[str, np.ndarray]]:
    """Load energy.csv for all FLIP ratios for a given scene.

    Returns dict keyed by ratio value (float).
    """
    data: dict[float, dict[str, np.ndarray]] = {}
    for rdir in sorted(FLIP_SWEEP_DIR.glob("ratio_*")):
        try:
            ratio_val = int(rdir.name.split("_")[1]) / 1000.0
        except (IndexError, ValueError):
            continue
        if ratio_val not in FLIP_RATIOS:
            continue
        csv_path = rdir / scene / "energy.csv"
        if csv_path.exists():
            data[ratio_val] = load_csv(csv_path)
    return data


def flip_sweep_summary(scene: str) -> list[dict[str, str]]:
    """Generate per-ratio summary statistics for a scene."""
    data = load_flip_sweep_data(scene)
    rows = []
    for ratio in FLIP_RATIOS:
        if ratio not in data:
            continue
        d = data[ratio]
        ek = d["kinetic_energy"]
        t = d["time"]
        ek0 = ek[0] if ek[0] > 1e-8 else 1.0
        rows.append({
            "ratio": f"{ratio:.2f}",
            "type": "Pure PIC" if ratio == 0.0 else ("Pure FLIP" if ratio == 1.0 else "Blend"),
            "initial_energy": f"{ek[0]:.6f}",
            "peak_energy": f"{ek.max():.6f}",
            "final_energy": f"{ek[-1]:.6f}",
            "energy_auc": f"{np.trapz(ek, t):.4f}",
            "retention_pct": f"{ek[-1] / ek0 * 100:.1f}",
            "mean_ms": f"{np.mean(d['frame_time_ms']):.0f}",
        })
    return rows


def flip_sweep_markdown_table(scene: str) -> str:
    """Generate a Markdown table for the FLIP ratio sweep per scene."""
    rows = flip_sweep_summary(scene)
    lines = [
        "| Ratio | Type | Initial E | Peak E | Final E | Energy AUC | Retention | Mean ms/frame |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['ratio']} | {r['type']} | {float(r['initial_energy']):.4f} | "
            f"{float(r['peak_energy']):.4f} | {float(r['final_energy']):.4f} | "
            f"{float(r['energy_auc']):.4f} | {r['retention_pct']}% | {r['mean_ms']} |"
        )
    return "\n".join(lines)


def copy_flip_sweep_figures() -> None:
    """Copy FLIP sweep figures into the report figures directory."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        "dam_break/dam_break_ratio_sweep_energy.png":      "flip_sweep_dam_break_energy.png",
        "dam_break/dam_break_ratio_sweep_normalized.png":   "flip_sweep_dam_break_normalized.png",
        "dam_break/dam_break_ratio_sweep_vorticity.png":    "flip_sweep_dam_break_vorticity.png",
        "dam_break/dam_break_ratio_retention.png":          "flip_sweep_dam_break_retention.png",
        "liquid_pouring/liquid_pouring_ratio_sweep_energy.png":    "flip_sweep_pouring_energy.png",
        "liquid_pouring/liquid_pouring_ratio_sweep_normalized.png": "flip_sweep_pouring_normalized.png",
        "liquid_pouring/liquid_pouring_ratio_sweep_vorticity.png":  "flip_sweep_pouring_vorticity.png",
        "liquid_pouring/liquid_pouring_ratio_retention.png":        "flip_sweep_pouring_retention.png",
        "flip_sweep_dam_break_visual_comparison.png":              "flip_sweep_dam_break_visual_comparison.png",
        "flip_sweep_liquid_pouring_visual_comparison.png":         "flip_sweep_liquid_pouring_visual_comparison.png",
    }
    for src_rel, dst_name in mapping.items():
        src = FLIP_SWEEP_FIG_DIR / src_rel
        if src.exists():
            shutil.copy2(src, FIG_DIR / dst_name)
            print(f"[flip_sweep] Copied {dst_name}")
        else:
            print(f"[flip_sweep] WARNING: missing {src}")


def dataset() -> dict[str, dict[str, dict[str, np.ndarray]]]:
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for scene in SCENES:
        data[scene] = {}
        for alg, cfg in ALGORITHMS.items():
            path = cfg["base"] / scene / "energy.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            data[scene][alg] = load_csv(path)
    return data


def write_summary(data: dict[str, dict[str, dict[str, np.ndarray]]]) -> list[dict[str, str]]:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for scene, by_alg in data.items():
        for alg, d in by_alg.items():
            e = d["kinetic_energy"]
            t = d["time"]
            row = {
                "scene": scene,
                "algorithm": alg,
                "rows": str(len(e)),
                "finite": str(bool(np.isfinite(e).all())),
                "initial_energy": f"{e[0]:.8f}",
                "peak_energy": f"{e.max():.8f}",
                "final_energy": f"{e[-1]:.8f}",
                "energy_auc": f"{np.trapz(e, t):.8f}",
            }
            summary.append(row)
    with (OUT / "energy_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    for scene, by_alg in data.items():
        frames = by_alg["FLIP"]["frame"]
        times = by_alg["FLIP"]["time"]
        path = OUT / f"{scene}_energy_timeseries.csv"
        with path.open("w", newline="") as f:
            fieldnames = ["frame", "time"] + [f"{alg.lower()}_kinetic_energy" for alg in ALGORITHMS]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(frames)):
                row = {"frame": int(frames[i]), "time": f"{times[i]:.6f}"}
                for alg in ALGORITHMS:
                    row[f"{alg.lower()}_kinetic_energy"] = f"{by_alg[alg]['kinetic_energy'][i]:.8f}"
                writer.writerow(row)
    return summary


def plot_energy(data):
    for scene, by_alg in data.items():
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        for alg, d in by_alg.items():
            ax.plot(d["time"], d["kinetic_energy"], label=alg, color=ALGORITHMS[alg]["color"], linewidth=1.8)
        ax.set_title(f"Kinetic Energy Comparison: {SCENE_LABELS[scene]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Kinetic energy")
        ax.grid(True, alpha=0.28)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / f"{scene}_kinetic_energy.png", dpi=220)
        fig.savefig(FIG_DIR / f"{scene}_kinetic_energy.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        for alg, d in by_alg.items():
            e = d["kinetic_energy"]
            peak = e.max() if e.max() > 1e-9 else 1.0
            ax.plot(d["time"], e / peak, label=alg, color=ALGORITHMS[alg]["color"], linewidth=1.8)
        ax.set_title(f"Peak-Normalized Energy: {SCENE_LABELS[scene]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Kinetic energy / method peak")
        ax.grid(True, alpha=0.28)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / f"{scene}_energy_peak_normalized.png", dpi=220)
        fig.savefig(FIG_DIR / f"{scene}_energy_peak_normalized.png", dpi=220)
        plt.close(fig)

    labels = []
    aucs = []
    colors = []
    for scene, by_alg in data.items():
        for alg, d in by_alg.items():
            labels.append(f"{alg}\n{SCENE_LABELS[scene]}")
            aucs.append(np.trapz(d["kinetic_energy"], d["time"]))
            colors.append(ALGORITHMS[alg]["color"])
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(labels, aucs, color=colors)
    ax.set_title("Integrated Kinetic Energy over the 5s Simulation")
    ax.set_ylabel("Integral of kinetic energy")
    ax.grid(True, axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(OUT / "energy_auc_summary.png", dpi=220)
    fig.savefig(FIG_DIR / "energy_auc_summary.png", dpi=220)
    plt.close(fig)


def make_montages():
    for scene in SCENES:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
        for ax, alg in zip(axes, ALGORITHMS):
            img_path = SHOT_DIR / f"{alg.lower()}_{scene}_t2p5.png"
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_title(alg)
            ax.axis("off")
        fig.suptitle(f"Visual Comparison at t=2.5s: {SCENE_LABELS[scene]}", y=0.98)
        fig.tight_layout()
        fig.savefig(OUT / f"{scene}_visual_comparison.png", dpi=220)
        fig.savefig(FIG_DIR / f"{scene}_visual_comparison.png", dpi=220)
        plt.close(fig)


def load_efficiency_summary() -> list[dict[str, str]]:
    path = BENCHMARK_DIR / "efficiency_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def copy_efficiency_figures():
    for filename in EFFICIENCY_FIGURES:
        src = BENCHMARK_DIR / filename
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, FIG_DIR / filename)


def efficiency_rows(efficiency: list[dict[str, str]], scene: str) -> list[dict[str, str]]:
    by_alg = {r["algorithm"]: r for r in efficiency if r["scene"] == scene}
    return [by_alg[alg.lower()] for alg in ALGORITHMS]


def efficiency_speedup(row: dict[str, str], rows: list[dict[str, str]]) -> float:
    flip_mean = next(float(r["mean_ms_per_frame"]) for r in rows if r["algorithm"] == "flip")
    return flip_mean / float(row["mean_ms_per_frame"])


def efficiency_markdown_table(efficiency: list[dict[str, str]]) -> str:
    lines = [
        "| Scene | Algorithm | Reps | Mean ms/frame | Std | Median | P95 | M particles/s | Speedup vs FLIP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene in SCENES:
        rows = efficiency_rows(efficiency, scene)
        for r in rows:
            alg = r["algorithm"].upper() if r["algorithm"] != "polypic" else "PolyPIC"
            lines.append(
                f"| {SCENE_LABELS[scene]} | {alg} | {int(r['repetitions'])} | "
                f"{float(r['mean_ms_per_frame']):.3f} | "
                f"{float(r['std_of_run_means_ms']):.3f} | "
                f"{float(r['median_ms_per_frame']):.3f} | "
                f"{float(r['p95_ms_per_frame']):.3f} | "
                f"{float(r['million_particles_per_s']):.2f} | "
                f"{efficiency_speedup(r, rows):.2f}x |"
            )
    return "\n".join(lines)


def markdown_table(summary, scene):
    rows = [r for r in summary if r["scene"] == scene]
    lines = ["| Algorithm | Rows | Finite | Initial E | Peak E | Final E | Energy AUC |",
             "|---|---:|:---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['algorithm']} | {r['rows']} | {r['finite']} | {float(r['initial_energy']):.4f} | "
            f"{float(r['peak_energy']):.4f} | {float(r['final_energy']):.4f} | "
            f"{float(r['energy_auc']):.4f} |"
        )
    return "\n".join(lines)


def write_markdown(summary, efficiency):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md = f"""# Comparative Study of FLIP, APIC, and PolyPIC Transfers for Particle-Grid Fluid Simulation

## Abstract

This report compares three particle-grid transfer schemes, FLIP, APIC, and PolyPIC, under a shared Taichi-based simulation framework. The experiments use identical grid resolution, boundary handling, particle generation, rendering style, and two scene configurations: a 3D dam break and a liquid pouring setup. The comparison focuses on kinetic-energy evolution, qualitative visual behavior, and a controlled runtime benchmark.

## Scope

This study is an experimental validation rather than a new fluid model. The goal is to isolate the transfer scheme as much as possible.

The contribution is a controlled comparison pipeline: three transfer schemes are implemented or run from a common framework, their outputs are converted into consistent CSV traces and videos, and the resulting behavior is summarized quantitatively and visually.

## Related Work

The FLIP method was introduced by Brackbill and Ruppel as a low-dissipation particle-in-cell variant for fluid-flow calculation. APIC was later proposed by Jiang et al. to improve particle-grid transfers by carrying locally affine velocity information. PolyPIC generalizes this direction by representing a more general local polynomial function on each particle. Our implementation-level comparison follows the same conceptual ladder: constant particle velocity transfer, affine transfer, and higher-order polynomial transfer.

## Methods

All algorithms were evaluated using the shared 3D grid framework in `framework.py`. The main experimental parameters were fixed across methods: grid resolution `80 x 100 x 80`, cell size `DX = 0.01`, two substeps per rendered frame, 300 output frames, and `ratio_970` for the FLIP/PIC blend convention. The two scenes are:

- `dam_break`: a dense water block initialized near one side of the domain.
- `liquid_pouring`: a source-like initialization producing an incoming stream.

FLIP uses an incremental grid velocity update to reduce dissipation relative to pure PIC. APIC augments particle state with a local affine velocity matrix, allowing first-order velocity variation to survive particle-grid transfers. PolyPIC extends this idea with higher-order local polynomial information, aiming to preserve richer local flow structure during transfers.

Efficiency was benchmarked separately from the rendered runs. FLIP, APIC, and PolyPIC were evaluated in a single batch benchmark on the `rtxp6000` partition with one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU, 8 CPU cores, and 64 GB host memory. Taichi CUDA was enabled with `TI_ARCH=cuda`. Rendering and video export were disabled, each method-scene pair was measured for three repetitions, and the first five frames were discarded to reduce JIT and initialization effects.

## Results: Dam Break

![Dam break kinetic energy](figures/dam_break_kinetic_energy.png)

{markdown_table(summary, 'dam_break')}

In the dam-break scene, all three methods remain finite for the full 300-frame run. PolyPIC has the largest integrated kinetic energy, suggesting the least overall dissipation in this test. APIC reaches the highest peak energy but, because the APIC branch was stabilized with finite-value guards and affine damping, it dissipates more strongly near the end of the run.

## Results: Liquid Pouring

![Liquid pouring kinetic energy](figures/liquid_pouring_kinetic_energy.png)

{markdown_table(summary, 'liquid_pouring')}

In the liquid-pouring scene, FLIP produces substantially higher kinetic energy than APIC and PolyPIC. This should not be interpreted as strictly better energy preservation. In a forced pouring setup, high kinetic energy can also reflect transfer noise or excessive momentum retention. APIC and PolyPIC give lower and smoother energy curves, with PolyPIC slightly above APIC in both peak and integrated energy.

## Results: FLIP PIC-FLIP Ratio Sweep

To understand how the PIC-FLIP blend ratio affects simulation behavior, we evaluated FLIP at six ratios from pure PIC (ratio=0.00) to pure FLIP (ratio=1.00), using the same grid, scenes, and render settings as the main comparison. The FLIP update blends the PIC-interpolated grid velocity with an incremental FLIP correction (Equation 1). At ratio=0 the method reduces to pure PIC; at ratio=1 it becomes pure FLIP; intermediate values blend both contributions.

### Dam Break — Ratio Sweep

![FLIP ratio sweep: dam break energy](figures/flip_sweep_dam_break_energy.png)

![FLIP ratio sweep: dam break normalized energy](figures/flip_sweep_dam_break_normalized.png)

{flip_sweep_markdown_table('dam_break')}

In the dam-break scene, pure PIC (ratio=0.00) dissipates kinetic energy extremely fast: the final retention is only around 2%, and the mean frame time is the highest of all ratios. Pure FLIP (ratio=1.00) produces a dramatic energy spike above 3000% retention, indicating unstable velocity amplification rather than genuine conservation — pure FLIP without any PIC blending accumulates high-frequency velocity noise that grows over time. Moderate blending (ratio=0.50) also shows excessive energy growth but less severely.

The best physical behavior is seen at high FLIP ratios near but below 1.0: ratio=0.95 retains about 14% and ratio=0.97 (used in the main comparison) retains about 26%. Ratio=0.80 sits in between with about 5% retention.

![FLIP ratio sweep: dam break vorticity](figures/flip_sweep_dam_break_vorticity.png)

![FLIP ratio sweep: dam break retention bar chart](figures/flip_sweep_dam_break_retention.png)

### Liquid Pouring — Ratio Sweep

![FLIP ratio sweep: liquid pouring energy](figures/flip_sweep_pouring_energy.png)

![FLIP ratio sweep: liquid pouring normalized energy](figures/flip_sweep_pouring_normalized.png)

{flip_sweep_markdown_table('liquid_pouring')}

In the continuously driven liquid-pouring scene, the ratio effects are even more pronounced. Pure FLIP (ratio=1.00) again shows extreme energy growth (over 20000% retention), confirming the instability pattern. The ratio sweep reveals that the PIC component acts as a crucial stabilizer: pure PIC (ratio=0.00) still retains over 5000% of initial energy in this driven scenario, but the growth is more moderate than at higher ratios.

Notably, ratio=0.80 shows the lowest energy retention (8253%) below the extremes, suggesting an intermediate blend that balances the stabilizing PIC interpolation against FLIP's momentum-preserving deltas. However, frame-time cost is not monotonic with ratio: ratio=0.00 (pure PIC) is approximately 4.5× slower than ratio=1.00 (pure FLIP), because the PIC interpolation path triggers different Taichi kernel code-generation patterns in this implementation.

![FLIP ratio sweep: liquid pouring vorticity](figures/flip_sweep_pouring_vorticity.png)

![FLIP ratio sweep: liquid pouring retention bar chart](figures/flip_sweep_pouring_retention.png)

### Visual Comparison — Ratio Sweep

![FLIP ratio sweep: dam break visual comparison](figures/flip_sweep_dam_break_visual_comparison.png)

![FLIP ratio sweep: liquid pouring visual comparison](figures/flip_sweep_liquid_pouring_visual_comparison.png)

The visual comparison at t=2.5s confirms the quantitative trends. Pure PIC (ratio=0.00) shows the wave barely propagating in the dam-break scene and minimal splash activity in the pouring scene — consistent with its extreme energy dissipation. Pure FLIP (ratio=1.00) displays visible particle scattering and noisy splashing, reflecting the unstable velocity amplification. Ratios 0.95 and 0.97 show well-defined fluid surfaces and natural splash behavior, with ratio=0.97 (used in the main comparison) producing the most visually plausible results.

These results confirm that ratio=0.97 — used in the main FLIP-APIC-PolyPIC comparison — provides a pragmatic balance: it retains meaningful kinetic activity without the catastrophic energy growth seen at ratio=1.00, while avoiding the severe dissipation of pure PIC at ratio=0.00.

## Visual Comparison

![Dam break visual comparison](figures/dam_break_visual_comparison.png)

![Liquid pouring visual comparison](figures/liquid_pouring_visual_comparison.png)

The video outputs are stored with the corresponding algorithm results. The screenshots above were extracted at `t = 2.5s` from the committed MP4 files.

## Efficiency Benchmark

![Dam break efficiency](figures/dam_break_efficiency_ms_per_frame.png)

![Liquid pouring efficiency](figures/liquid_pouring_efficiency_ms_per_frame.png)

![Relative speedup](figures/speedup_vs_flip.png)

{efficiency_markdown_table(efficiency)}

The efficiency benchmark follows the expected method-complexity ordering. FLIP is the fastest method in both scenes because it transfers only the baseline particle velocity state. APIC is slower because it carries and updates an affine velocity matrix. PolyPIC is slowest because its higher-order local transfer stores and evaluates more particle-side information. Relative to FLIP, APIC reaches `0.79x` speed on dam break and `0.64x` speed on liquid pouring, while PolyPIC reaches `0.64x` and `0.60x`, respectively.

## Discussion

The experiments support four practical observations. First, a shared framework is necessary: small changes in grid resolution, particle count, or boundary treatment can dominate the numerical differences between transfer schemes. Second, kinetic energy is informative but not sufficient on its own. Higher energy can mean useful reduced dissipation, but it can also expose noisy transfer or excessive momentum retention. Third, APIC and PolyPIC require more care than baseline FLIP. APIC in particular needed affine-matrix limiting to avoid NaN growth in the full run. Fourth, transfer accuracy has a measurable cost: the richer APIC and PolyPIC particle state reduces throughput in the controlled benchmark.

For the dam-break case, PolyPIC gives the clearest energy-retention advantage by integrated kinetic energy. For the pouring case, APIC and PolyPIC are more restrained than FLIP; PolyPIC retains slightly more energy than APIC while remaining stable. These results are consistent with the motivation behind affine and polynomial particle-grid transfers: additional local velocity information can improve transfer quality, but stability controls remain important in a compact implementation.

The FLIP ratio sweep adds an important nuance to the main comparison. Pure FLIP (ratio=1.00) is catastrophically unstable in both scenes, confirming that the PIC component in the blend acts as an essential stabilizer. Pure PIC (ratio=0.00) is severely dissipative. The ratio=0.97 used in the main FLIP-APIC-PolyPIC comparison sits in a narrow effective window: it retains meaningful kinetic activity without the energy explosion seen at ratio=1.00, and it avoids the near-total dissipation of pure PIC.

## Limitations

The comparison is a small-scale validation rather than a full production benchmark. The rendering is particle-based and does not include a high-quality surface reconstruction stage. The efficiency benchmark is simulation-only: rendering and video export are excluded, and the first five frames are discarded. This makes the timing comparison more controlled than the original rendered frame times, but it still measures this implementation and hardware configuration rather than a hardware-independent property of the algorithms. The APIC implementation also includes damping for robustness, which changes its energy behavior relative to an ideal APIC formulation.

The FLIP ratio sweep uses the same framework but was run as a separate batch; the frame-time values include rendering overhead and were measured under different runtime conditions than the efficiency benchmark, so per-ratio timing should not be directly compared to the benchmark numbers.

## Conclusion

The final pipeline produces comparable outputs for FLIP, APIC, and PolyPIC under common scenes. PolyPIC shows the strongest energy retention in the dam-break test, while APIC and PolyPIC show smoother, lower-energy behavior in the pouring scene than the baseline FLIP run. The efficiency benchmark adds the complementary tradeoff: FLIP is fastest, APIC is moderately slower, and PolyPIC is slowest because it preserves richer transfer state.

## Artifact Index

- `output/comparison/energy_summary.csv`
- `output/comparison/dam_break_energy_timeseries.csv`
- `output/comparison/liquid_pouring_energy_timeseries.csv`
- `output/comparison/dam_break_kinetic_energy.png`
- `output/comparison/liquid_pouring_kinetic_energy.png`
- `output/comparison/energy_auc_summary.png`
- `output/benchmark_efficiency/efficiency_runs.csv`
- `output/benchmark_efficiency/efficiency_summary.csv`
- `output/benchmark_efficiency/dam_break_efficiency_ms_per_frame.png`
- `output/benchmark_efficiency/liquid_pouring_efficiency_ms_per_frame.png`
- `output/benchmark_efficiency/speedup_vs_flip.png`
- `docs/final_report/efficiency_benchmark.md`
- `docs/final_report/option1_comparative_study.tex`
- `docs/final_report/option1_comparative_study.pdf`
- `docs/final_report/option1_comparative_study.md`
- `output/flip_sweep/dam_break/dam_break_ratio_sweep_energy.png`
- `output/flip_sweep/dam_break/dam_break_ratio_sweep_normalized.png`
- `output/flip_sweep/dam_break/dam_break_ratio_sweep_vorticity.png`
- `output/flip_sweep/dam_break/dam_break_ratio_retention.png`
- `output/flip_sweep/liquid_pouring/liquid_pouring_ratio_sweep_energy.png`
- `output/flip_sweep/liquid_pouring/liquid_pouring_ratio_sweep_normalized.png`
- `output/flip_sweep/liquid_pouring/liquid_pouring_ratio_sweep_vorticity.png`
- `output/flip_sweep/liquid_pouring/liquid_pouring_ratio_retention.png`

## References

1. J. U. Brackbill and H. M. Ruppel. FLIP: A method for adaptively zoned, particle-in-cell calculations of fluid flows in two dimensions. Journal of Computational Physics 65(2), 314-343, 1986. DOI: <https://doi.org/10.1016/0021-9991(86)90211-1>.
2. N. Foster and R. Fedkiw. Practical animation of liquids. Proceedings of SIGGRAPH 2001, 23-30, 2001. DOI: <https://doi.org/10.1145/383259.383261>.
3. Y. Zhu and R. Bridson. Animating sand as a fluid. ACM Transactions on Graphics 24(3), 965-972, 2005. DOI: <https://doi.org/10.1145/1073204.1073298>.
4. C. Jiang, C. Schroeder, A. Selle, J. Teran, and A. Stomakhin. The Affine Particle-In-Cell Method. ACM Transactions on Graphics 34(4), Article 51, 2015. DOI: <https://doi.org/10.1145/2766996>.
5. C. Fu, Q. Guo, T. Gast, C. Jiang, and J. Teran. A Polynomial Particle-In-Cell Method. ACM Transactions on Graphics 36(6), Article 222, 2017. DOI: <https://doi.org/10.1145/3130800.3130878>.
6. Y. Hu, T.-M. Li, L. Anderson, J. Ragan-Kelley, and F. Durand. Taichi: A language for high-performance computation on spatially sparse data structures. ACM Transactions on Graphics 38(6), Article 201, 2019. DOI: <https://doi.org/10.1145/3355089.3356506>.
7. R. Bridson. Fluid Simulation for Computer Graphics, 2nd ed. CRC Press, 2015.

## Tool Use Statement

Large language models assisted with implementation debugging, batch-command preparation, plotting code, and language editing. All numerical values in the tables and figures are computed from the committed simulation outputs.
"""
    (REPORT_DIR / "option1_comparative_study.md").write_text(md)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = dataset()
    summary = write_summary(data)
    plot_energy(data)
    make_montages()
    copy_flip_sweep_figures()
    efficiency = load_efficiency_summary()
    copy_efficiency_figures()
    write_markdown(summary, efficiency)
    build_latex_report(compile_pdf=True)
    print("[ok] Wrote comparison artifacts to", OUT)
    print("[ok] Wrote report to", REPORT_DIR)


if __name__ == "__main__":
    main()
