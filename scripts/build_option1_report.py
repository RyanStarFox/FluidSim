#!/usr/bin/env python3
"""Build Option 1 comparison artifacts for the final project report.

This script reads the committed FLIP/APIC/PolyPIC energy CSVs, generates
comparison figures and summary tables, and writes a Markdown plus PDF report.
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "comparison"
REPORT_DIR = ROOT / "docs" / "final_report"
FIG_DIR = REPORT_DIR / "figures"
SHOT_DIR = FIG_DIR / "screenshots"

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
            ms = d["frame_time_ms"]
            row = {
                "scene": scene,
                "algorithm": alg,
                "rows": str(len(e)),
                "finite": str(bool(np.isfinite(e).all())),
                "initial_energy": f"{e[0]:.8f}",
                "peak_energy": f"{e.max():.8f}",
                "final_energy": f"{e[-1]:.8f}",
                "energy_auc": f"{np.trapezoid(e, t):.8f}",
                "mean_frame_time_ms_excluding_first": f"{ms[1:].mean():.6f}",
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
            aucs.append(np.trapezoid(d["kinetic_energy"], d["time"]))
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

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    width = 0.24
    x = np.arange(len(SCENES))
    for offset, alg in zip([-width, 0, width], ALGORITHMS):
        vals = [data[scene][alg]["frame_time_ms"][1:].mean() for scene in SCENES]
        ax.bar(x + offset, vals, width=width, color=ALGORITHMS[alg]["color"], label=alg)
    ax.set_xticks(x, [SCENE_LABELS[s] for s in SCENES])
    ax.set_title("Mean Frame Time (first frame excluded)")
    ax.set_ylabel("ms / frame")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "frame_time_summary.png", dpi=220)
    fig.savefig(FIG_DIR / "frame_time_summary.png", dpi=220)
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


def markdown_table(summary, scene):
    rows = [r for r in summary if r["scene"] == scene]
    lines = ["| Algorithm | Rows | Finite | Initial E | Peak E | Final E | Energy AUC | Mean ms/frame* |",
             "|---|---:|:---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['algorithm']} | {r['rows']} | {r['finite']} | {float(r['initial_energy']):.4f} | "
            f"{float(r['peak_energy']):.4f} | {float(r['final_energy']):.4f} | "
            f"{float(r['energy_auc']):.4f} | {float(r['mean_frame_time_ms_excluding_first']):.3f} |"
        )
    return "\n".join(lines)


def write_markdown(summary):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md = f"""# Comparative Study of FLIP, APIC, and PolyPIC Transfers for Particle-Grid Fluid Simulation

## Abstract

This report presents an Option 1 experimental validation project for CS3511: Physical Simulation of Solids and Fluids. We compare three particle-grid transfer schemes, FLIP, APIC, and PolyPIC, under a shared Taichi-based simulation framework. The experiments use identical grid resolution, boundary handling, particle generation, rendering style, and two scene configurations: a 3D dam break and a liquid pouring setup. The comparison focuses on kinetic-energy evolution, qualitative visual behavior, and reproducibility of generated artifacts.

## Course Option and Scope

The course logistics slides define Option 1 as experimental validation of existing simulators. The same slides specify that the final report should contain introduction, related work, methods, results, and discussion. This project follows that structure. Open-source and AI-assisted implementation were used with explicit acknowledgement in the appendix.

The project contribution is not a new fluid model. Instead, it is a controlled comparison pipeline: three transfer schemes are implemented or run from a common framework, their outputs are converted into consistent CSV traces and videos, and the resulting behavior is summarized quantitatively and visually.

## Related Work

The FLIP method was introduced by Brackbill and Ruppel as a low-dissipation particle-in-cell variant for fluid-flow calculation. APIC was later proposed by Jiang et al. to improve particle-grid transfers by carrying locally affine velocity information. PolyPIC generalizes this direction by representing a more general local polynomial function on each particle. Our implementation-level comparison follows the same conceptual ladder: constant particle velocity transfer, affine transfer, and higher-order polynomial transfer.

## Methods

All algorithms were evaluated using the shared 3D grid framework in `framework.py`. The main experimental parameters were fixed across methods: grid resolution `80 x 100 x 80`, cell size `DX = 0.01`, two substeps per rendered frame, 300 output frames, and `ratio_970` for the FLIP/PIC blend convention. The two scenes are:

- `dam_break`: a dense water block initialized near one side of the domain.
- `liquid_pouring`: a source-like initialization producing an incoming stream.

FLIP uses an incremental grid velocity update to reduce dissipation relative to pure PIC. APIC augments particle state with a local affine velocity matrix, allowing first-order velocity variation to survive particle-grid transfers. PolyPIC extends this idea with higher-order local polynomial information, aiming to preserve richer local flow structure during transfers.

## Results: Dam Break

![Dam break kinetic energy](figures/dam_break_kinetic_energy.png)

{markdown_table(summary, 'dam_break')}

*The first frame includes initialization and compilation overhead, so the timing column excludes frame 0.*

In the dam-break scene, all three methods remain finite for the full 300-frame run. PolyPIC has the largest integrated kinetic energy, suggesting the least overall dissipation in this test. APIC reaches the highest peak energy but, because the APIC branch was stabilized with finite-value guards and affine damping, it dissipates more strongly near the end of the run.

## Results: Liquid Pouring

![Liquid pouring kinetic energy](figures/liquid_pouring_kinetic_energy.png)

{markdown_table(summary, 'liquid_pouring')}

In the liquid-pouring scene, FLIP produces substantially higher kinetic energy than APIC and PolyPIC. This should not be interpreted as strictly better energy preservation. In a forced pouring setup, high kinetic energy can also reflect transfer noise or excessive momentum retention. APIC and PolyPIC give lower and smoother energy curves, with PolyPIC slightly above APIC in both peak and integrated energy.

## Visual Comparison

![Dam break visual comparison](figures/dam_break_visual_comparison.png)

![Liquid pouring visual comparison](figures/liquid_pouring_visual_comparison.png)

The video outputs are stored with the corresponding algorithm results. The screenshots above were extracted at `t = 2.5s` from the committed MP4 files.

## Discussion

The experiments support three practical observations. First, a shared framework is necessary: small changes in grid resolution, particle count, or boundary treatment can dominate the numerical differences between transfer schemes. Second, kinetic energy is informative but not sufficient on its own. Higher energy can mean useful reduced dissipation, but it can also expose noisy transfer or excessive momentum retention. Third, APIC and PolyPIC require more care than baseline FLIP. APIC in particular needed affine-matrix limiting to avoid NaN growth in the full run.

For the dam-break case, PolyPIC gives the clearest energy-retention advantage by integrated kinetic energy. For the pouring case, APIC and PolyPIC are more restrained than FLIP; PolyPIC retains slightly more energy than APIC while remaining stable. These results are consistent with the motivation behind affine and polynomial particle-grid transfers: additional local velocity information can improve transfer quality, but stability controls remain important in a compact course implementation.

## Limitations

The comparison is a course-scale validation rather than a full benchmark. The rendering is particle-based and not a high-quality surface reconstruction. Timing is reported but should be treated cautiously because the FLIP data were produced in a different run context from the APIC/PolyPIC reruns. The APIC implementation also includes damping for robustness, which changes its energy behavior relative to an ideal APIC formulation.

## Conclusion

The final pipeline successfully produces reproducible outputs for FLIP, APIC, and PolyPIC under common scenes. PolyPIC shows the strongest energy retention in the dam-break test, while APIC and PolyPIC show smoother, lower-energy behavior in the pouring scene than the baseline FLIP run. The project therefore satisfies the Option 1 goal: experimental validation and comparison of existing simulator techniques through controlled runs, quantitative plots, videos, and a documented analysis pipeline.

## Artifact Index

- `output/comparison/energy_summary.csv`
- `output/comparison/dam_break_energy_timeseries.csv`
- `output/comparison/liquid_pouring_energy_timeseries.csv`
- `output/comparison/dam_break_kinetic_energy.png`
- `output/comparison/liquid_pouring_kinetic_energy.png`
- `output/comparison/energy_auc_summary.png`
- `docs/final_report/option1_comparative_study.pdf`
- `docs/final_report/option1_comparative_study.md`

## References

1. J. U. Brackbill and H. M. Ruppel. FLIP: A method for adaptively zoned, particle-in-cell calculations of fluid flows in two dimensions. Journal of Computational Physics 65(2), 314-343, 1986. DOI: <https://doi.org/10.1016/0021-9991(86)90211-1>.
2. C. Jiang, C. Schroeder, A. Selle, J. Teran, and A. Stomakhin. The Affine Particle-In-Cell Method. ACM Transactions on Graphics, 2015. Paper: <https://mass.math.ucdavis.edu/~jteran/papers/JSSTS15.pdf>.
3. C. Fu, Q. Guo, T. Gast, C. Jiang, and J. Teran. A Polynomial Particle-In-Cell Method. ACM Transactions on Graphics 36(6), Article 222, 2017. DOI: <https://doi.org/10.1145/3130800.3130878>.
4. R. Bridson. Fluid Simulation for Computer Graphics. A K Peters/CRC Press.

## Appendix: AI Tool Usage

AI tools were used to assist with code repair, Slurm job orchestration, data plotting, and drafting this report. The numerical results were not generated by language-model text; they were computed by running the repository code and were verified from the committed CSV files. The final interpretation was checked against the actual outputs, including finite-value validation for all 300-frame APIC and PolyPIC runs.
"""
    (REPORT_DIR / "option1_comparative_study.md").write_text(md)


def add_text_page(pdf, title, body, footer=None, fontsize=10):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    fig.text(0.08, 0.93, title, fontsize=17, fontweight="bold", va="top")
    wrapped = []
    for para in body.split("\n"):
        if not para.strip():
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(para, 96))
    y = 0.87
    for line in wrapped:
        fig.text(0.08, y, line, fontsize=fontsize, va="top", family="DejaVu Sans")
        y -= 0.026
        if y < 0.08:
            break
    if footer:
        fig.text(0.08, 0.04, footer, fontsize=8, color="#555555")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_figure_page(pdf, title, image_path, caption, table_lines=None):
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.axis("off")
    fig.text(0.08, 0.94, title, fontsize=16, fontweight="bold", va="top")
    img = mpimg.imread(image_path)
    ax = fig.add_axes([0.09, 0.43, 0.82, 0.43])
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.08, 0.39, textwrap.fill(caption, 92), fontsize=9.5, va="top")
    if table_lines:
        y = 0.29
        fig.text(0.08, y, "Summary statistics", fontsize=10, fontweight="bold")
        y -= 0.028
        for line in table_lines:
            fig.text(0.08, y, line, fontsize=8.4, family="DejaVu Sans Mono")
            y -= 0.024
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def summary_lines(summary, scene):
    lines = ["Algorithm  Rows  Finite  InitialE   PeakE    FinalE   AUC"]
    for r in [x for x in summary if x["scene"] == scene]:
        lines.append(f"{r['algorithm']:<9} {r['rows']:>4}  {r['finite']:<6}  {float(r['initial_energy']):>8.4f} {float(r['peak_energy']):>8.4f} {float(r['final_energy']):>8.4f} {float(r['energy_auc']):>8.4f}")
    return lines


def write_pdf(summary):
    path = REPORT_DIR / "option1_comparative_study.pdf"
    with PdfPages(path) as pdf:
        add_text_page(pdf, "Comparative Study of FLIP, APIC, and PolyPIC", """
This Option 1 project validates and compares three existing particle-grid transfer schemes for fluid simulation: FLIP, APIC, and PolyPIC. The experiments are run in a shared Taichi framework with fixed grid resolution, particle initialization, boundary treatment, and rendering settings.

The objective is to produce a controlled comparison rather than a new simulator. The main outputs are energy CSV files, videos, visual screenshots, comparison plots, and this report.

Main finding: PolyPIC has the strongest integrated kinetic-energy retention in the dam-break test. In the liquid-pouring test, FLIP produces much higher kinetic energy, which is treated as possible momentum over-retention or transfer noise rather than an automatic quality improvement. APIC and PolyPIC are more restrained and stable in that setup.
""", footer="CS3511 final project, Option 1 experimental validation")
        add_text_page(pdf, "Course Option, Related Work, and Setup", """
The course logistics slides describe Option 1 as experimental validation of existing simulators and require the final report to include introduction, related work, methods, results, and discussion. This report follows that structure.

Related work: FLIP is a low-dissipation particle-in-cell variant introduced by Brackbill and Ruppel. APIC improves transfer accuracy by augmenting particles with locally affine velocity information. PolyPIC extends the idea to richer local polynomial functions. Our comparison follows this progression from baseline FLIP to affine and polynomial transfers.

Common setup: 80 x 100 x 80 grid, DX = 0.01, two simulation substeps per rendered frame, 300 output frames, and ratio_970 for the FLIP/PIC blending convention. Two scenes are evaluated: a dense 3D dam break and a liquid pouring configuration.

All CSV outputs are checked for finite kinetic energy values. The APIC and PolyPIC full runs each completed 300 frames for both scenes. The FLIP ratio_970 outputs from the existing main branch are used as the baseline.
""")
        add_text_page(pdf, "Methods", """
FLIP transfers particle velocity changes from the grid back to particles, reducing the dissipation of pure PIC but potentially retaining noisy velocity components.

APIC stores a local affine matrix per particle. The P2G transfer evaluates a local velocity model at grid faces, and the G2P transfer reconstructs particle velocity and affine state from the projected grid velocity field. The implementation used here includes finite-value guards and affine damping to keep the full 300-frame runs stable.

PolyPIC generalizes the transfer by carrying higher-order polynomial information. In principle, this can preserve richer local velocity variation than APIC, though it also increases implementation complexity.
""")
        add_figure_page(pdf, "Dam Break: Kinetic Energy", FIG_DIR / "dam_break_kinetic_energy.png", "All three methods remain finite for 300 frames. PolyPIC has the largest integrated kinetic energy in this scene, while APIC reaches the highest peak but dissipates more strongly near the end after stabilization.", summary_lines(summary, "dam_break"))
        add_figure_page(pdf, "Liquid Pouring: Kinetic Energy", FIG_DIR / "liquid_pouring_kinetic_energy.png", "FLIP produces substantially higher kinetic energy in the pouring scene. Because the scene is continuously driven, higher energy is interpreted cautiously: it can indicate reduced dissipation, but also numerical noise or excessive momentum retention.", summary_lines(summary, "liquid_pouring"))
        add_figure_page(pdf, "Visual Comparison", FIG_DIR / "dam_break_visual_comparison.png", "Screenshots extracted at t = 2.5s from the committed dam-break videos. The videos themselves are stored under each algorithm's output directory.")
        add_figure_page(pdf, "Integrated Energy and Timing", FIG_DIR / "energy_auc_summary.png", "Integrated kinetic energy summarizes the total kinetic activity over the five-second run. Timing is included as secondary evidence only, because the baseline FLIP outputs were generated in a different run context from the APIC and PolyPIC reruns.")
        add_text_page(pdf, "Discussion, Limitations, and AI Usage", """
Discussion: The experiments show that a shared framework is essential for meaningful comparison. PolyPIC gives the clearest energy-retention advantage in the dam-break scene. APIC and PolyPIC are smoother than FLIP in liquid pouring, where FLIP's higher kinetic energy should not be read as an unqualified improvement.

Limitations: The renderer is particle-based and not a production surface reconstruction. Timing numbers are not a strict benchmark because run contexts differ. The APIC implementation uses damping for robustness, so it is not an idealized APIC-only measurement.

AI tool usage: AI assistance was used for code repair, Slurm orchestration, plotting, and report drafting. The numerical values in the tables and plots come from repository CSV files generated by simulation runs, not from language-model invention.

References: Brackbill and Ruppel 1986, doi:10.1016/0021-9991(86)90211-1; Jiang et al. 2015; Fu et al. 2017, doi:10.1145/3130800.3130878; Bridson, Fluid Simulation for Computer Graphics.
""")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = dataset()
    summary = write_summary(data)
    plot_energy(data)
    make_montages()
    write_markdown(summary)
    write_pdf(summary)
    print("[ok] Wrote comparison artifacts to", OUT)
    print("[ok] Wrote report to", REPORT_DIR)


if __name__ == "__main__":
    main()
