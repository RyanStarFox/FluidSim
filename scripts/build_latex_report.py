#!/usr/bin/env python3
"""Build the final Option 1 report as a LaTeX conference-style paper."""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "final_report"
OUT = ROOT / "output" / "comparison"
BENCHMARK_DIR = ROOT / "output" / "benchmark_efficiency"
TEX_PATH = REPORT_DIR / "option1_comparative_study.tex"
PDF_PATH = REPORT_DIR / "option1_comparative_study.pdf"

SCENE_LABELS = {
    "dam_break": "Dam Break",
    "liquid_pouring": "Liquid Pouring",
}
SCENE_ORDER = ["dam_break", "liquid_pouring"]
ALGORITHM_ORDER = ["FLIP", "APIC", "PolyPIC"]


class LatexTemplate(Template):
    delimiter = "@@"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def energy_table_rows(rows: list[dict[str, str]]) -> str:
    indexed = {(r["scene"], r["algorithm"]): r for r in rows}
    lines = []
    for scene in SCENE_ORDER:
        for alg in ALGORITHM_ORDER:
            r = indexed[(scene, alg)]
            lines.append(
                f"{SCENE_LABELS[scene]} & {alg} & {int(r['rows'])} & {r['finite']} & "
                f"{float(r['initial_energy']):.4f} & {float(r['peak_energy']):.4f} & "
                f"{float(r['final_energy']):.4f} & {float(r['energy_auc']):.4f} \\\\"
            )
    return "\n".join(lines)


def efficiency_table_rows(rows: list[dict[str, str]]) -> str:
    indexed = {(r["scene"], r["algorithm"]): r for r in rows}
    lines = []
    for scene in SCENE_ORDER:
        scene_rows = [indexed[(scene, alg.lower())] for alg in ALGORITHM_ORDER]
        flip_ms = float(indexed[(scene, "flip")]["mean_ms_per_frame"])
        for alg, r in zip(ALGORITHM_ORDER, scene_rows):
            speedup = flip_ms / float(r["mean_ms_per_frame"])
            lines.append(
                f"{SCENE_LABELS[scene]} & {alg} & {int(r['repetitions'])} & "
                f"{float(r['mean_ms_per_frame']):.3f} & "
                f"{float(r['std_of_run_means_ms']):.3f} & "
                f"{float(r['p95_ms_per_frame']):.3f} & "
                f"{float(r['million_particles_per_s']):.2f} & {speedup:.2f}$\\times$ \\\\"
            )
    return "\n".join(lines)


def locate_tectonic() -> str | None:
    local = ROOT / ".micromamba_fluidsim" / "bin" / "tectonic"
    if local.exists():
        return str(local)
    return shutil.which("tectonic")


def latex_source(energy_rows: str, efficiency_rows: str) -> str:
    template = LatexTemplate(r"""\documentclass[conference]{IEEEtran}

\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{url}

\graphicspath{{figures/}}
\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=black,
  urlcolor=blue
}

\title{Comparative Study of FLIP, APIC, and PolyPIC Transfers for Particle--Grid Fluid Simulation}

\author{
\IEEEauthorblockN{Qinzhe Hu \quad Yan Shao \quad Baihan Deng}
\IEEEauthorblockA{CS3511 Physical Simulation of Solids and Fluids}
}

\begin{document}
\maketitle

\begin{abstract}
This paper presents an Option 1 experimental validation project comparing three particle--grid transfer schemes for fluid simulation: FLIP, APIC, and PolyPIC. All methods are evaluated under a shared Taichi-based framework with identical grid resolution, boundary handling, particle generation, rendering style, and scene definitions. The study uses two representative three-dimensional scenarios, dam break and liquid pouring, and evaluates the methods using kinetic-energy evolution, visual behavior, and a same-job simulation-only efficiency benchmark. The results show that PolyPIC provides the strongest integrated kinetic-energy retention in the dam-break scene, while APIC and PolyPIC produce smoother and lower-energy behavior than FLIP in the continuously driven pouring scene. The efficiency benchmark confirms the expected cost tradeoff: FLIP is fastest, APIC is slower due to affine transfer state, and PolyPIC is slowest due to higher-order polynomial transfer state.
\end{abstract}

\begin{IEEEkeywords}
fluid simulation, FLIP, APIC, PolyPIC, particle-in-cell, Taichi, experimental validation
\end{IEEEkeywords}

\section{Introduction}

Particle--grid hybrid methods are widely used for visually rich fluid simulation because they combine the low numerical diffusion of particle representations with the structured projection and force evaluation available on grids. However, different particle--grid transfer schemes can produce noticeably different dissipation, stability, and runtime behavior even when the outer simulator is otherwise identical.

This project follows the course Option 1 requirement: experimental validation of existing simulators. Rather than proposing a new fluid model, we build a controlled comparison pipeline for FLIP, APIC, and PolyPIC. The key design goal is to isolate the transfer scheme as much as possible. The three algorithms are run with the same grid, scenes, particle initialization, boundaries, frame count, and plotting pipeline. The final artifacts include energy CSV files, rendered videos, visual screenshots, comparison figures, an efficiency benchmark, and this LaTeX report.

The contribution is therefore practical and empirical. We ask three questions: (1) which method preserves kinetic energy most strongly in the shared scenarios; (2) whether the qualitative videos match the quantitative energy trends; and (3) how much runtime overhead APIC and PolyPIC introduce relative to baseline FLIP in this implementation.

\section{Related Work}

The Particle-In-Cell (PIC) family transfers information between Lagrangian particles and an Eulerian grid. FLIP, introduced by Brackbill and Ruppel~\cite{brackbill1986flip}, reduces the excessive dissipation of pure PIC by transferring grid velocity increments back to particles. This makes FLIP attractive for energetic liquids, but it may also retain noisy velocity components.

APIC augments particles with locally affine velocity information~\cite{jiang2015apic}. Instead of storing only a constant velocity per particle, APIC carries an affine matrix that represents first-order local velocity variation. This improves angular momentum behavior and transfer consistency, at the cost of extra particle state and implementation complexity.

PolyPIC generalizes this direction by representing higher-order polynomial local velocity functions on particles~\cite{fu2017polypic}. The method is motivated by preserving richer local flow structure during particle--grid transfers. In a compact course implementation, this additional expressiveness must be balanced against stability controls and runtime cost.

\section{Methods}

\subsection{Shared Simulation Framework}

All algorithms are evaluated using the shared three-dimensional Taichi framework in the repository. The common setup uses a grid resolution of $80 \times 100 \times 80$, cell size $\Delta x = 0.01$, two simulation substeps per rendered frame, and 300 output frames. The experiments use the \texttt{ratio\_970} convention for the FLIP/PIC blend parameter so that the branches can be compared under the same naming and output structure.

Two scenes are used. The dam-break scene initializes a dense water block near one side of the domain and observes the subsequent collapse and spread. The liquid-pouring scene initializes a stream-like incoming fluid source and therefore acts as a continuously driven test case.

\subsection{Transfer Schemes}

FLIP uses the change in grid velocity after force application and projection to update particle velocities. This keeps more kinetic activity than pure PIC but can carry high-frequency velocity noise.

APIC stores an affine matrix for each particle. During particle-to-grid transfer, the local affine velocity model contributes to nearby grid nodes. During grid-to-particle transfer, both the particle velocity and affine state are reconstructed from the projected grid velocity field. The implementation used for the full run includes finite-value guards and affine damping to prevent NaN growth.

PolyPIC stores higher-order local transfer information. In principle, this can preserve richer local flow variation than APIC. In this project, PolyPIC is evaluated as the high-order branch of the same framework and compared using identical output metrics.

\subsection{Metrics and Efficiency Benchmark}

For each rendered frame, the pipeline records kinetic energy and validates that the time series remains finite. We summarize each method using initial kinetic energy, peak kinetic energy, final kinetic energy, and the integral of kinetic energy over the simulated time interval.

Runtime efficiency is measured separately from the visual-output runs. FLIP, APIC, and PolyPIC are rerun inside the same Slurm job on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU with Taichi CUDA enabled. Rendering and video export are disabled so that timing focuses on simulation work. Each method-scene pair is measured for three repetitions, and the first five frames are discarded to reduce JIT compilation and initialization effects.

\section{Results}

\subsection{Kinetic Energy}

\begin{figure*}[t]
  \centering
  \begin{minipage}{0.49\textwidth}
    \centering
    \includegraphics[width=\linewidth]{dam_break_kinetic_energy.png}
    \vspace{-0.6em}
    \centerline{\footnotesize (a) Dam break}
  \end{minipage}
  \hfill
  \begin{minipage}{0.49\textwidth}
    \centering
    \includegraphics[width=\linewidth]{liquid_pouring_kinetic_energy.png}
    \vspace{-0.6em}
    \centerline{\footnotesize (b) Liquid pouring}
  \end{minipage}
  \caption{Kinetic-energy evolution for the two shared scenes. PolyPIC has the largest integrated kinetic energy in dam break, while FLIP produces much higher kinetic energy in the continuously driven pouring scene.}
  \label{fig:energy}
\end{figure*}

\begin{table*}[t]
  \centering
  \caption{Kinetic-energy summary from the 300-frame visual-output runs. AUC denotes the time integral of kinetic energy.}
  \label{tab:energy}
  \begin{tabular}{llrrrrrr}
    \toprule
    Scene & Algorithm & Rows & Finite & Initial E & Peak E & Final E & Energy AUC \\
    \midrule
@@energy_rows
    \bottomrule
  \end{tabular}
\end{table*}

In the dam-break scene, all three methods remain finite for the full run. PolyPIC has the largest integrated kinetic energy, suggesting the strongest energy retention in this test. APIC reaches the highest peak energy, but after stabilization it dissipates more strongly near the end of the run.

In the liquid-pouring scene, FLIP produces substantially higher kinetic energy than APIC and PolyPIC. Because the scene is continuously driven, this should not be interpreted as a simple quality win. High kinetic energy can indicate reduced dissipation, but it can also reflect noisy transfer or excessive momentum retention. APIC and PolyPIC give lower and smoother energy curves, with PolyPIC slightly above APIC in both peak and integrated energy.

\subsection{Visual Comparison}

\begin{figure*}[t]
  \centering
  \begin{minipage}{0.49\textwidth}
    \centering
    \includegraphics[width=\linewidth]{dam_break_visual_comparison.png}
    \vspace{-0.6em}
    \centerline{\footnotesize (a) Dam break at $t=2.5$s}
  \end{minipage}
  \hfill
  \begin{minipage}{0.49\textwidth}
    \centering
    \includegraphics[width=\linewidth]{liquid_pouring_visual_comparison.png}
    \vspace{-0.6em}
    \centerline{\footnotesize (b) Liquid pouring at $t=2.5$s}
  \end{minipage}
  \caption{Visual comparison extracted from the committed videos. The screenshots use the same rendering style and sampling time across algorithms.}
  \label{fig:visual}
\end{figure*}

The visual outputs provide a qualitative check against the CSV traces. The dam-break videos show broadly similar large-scale motion, but the energy summaries indicate that PolyPIC retains more total kinetic activity over time. The pouring screenshots support the quantitative observation that APIC and PolyPIC behave more restrainedly than FLIP in the driven source setup.

\subsection{Efficiency}

\begin{figure*}[t]
  \centering
  \begin{minipage}{0.32\textwidth}
    \centering
    \includegraphics[width=\linewidth]{dam_break_efficiency_ms_per_frame.png}
    \vspace{-0.6em}
    \centerline{\footnotesize (a) Dam break}
  \end{minipage}
  \hfill
  \begin{minipage}{0.32\textwidth}
    \centering
    \includegraphics[width=\linewidth]{liquid_pouring_efficiency_ms_per_frame.png}
    \vspace{-0.6em}
    \centerline{\footnotesize (b) Liquid pouring}
  \end{minipage}
  \hfill
  \begin{minipage}{0.32\textwidth}
    \centering
    \includegraphics[width=\linewidth]{speedup_vs_flip.png}
    \vspace{-0.6em}
    \centerline{\footnotesize (c) Speedup vs. FLIP}
  \end{minipage}
  \caption{Simulation-only efficiency benchmark. Rendering and video export are disabled; each method-scene pair is measured for three repetitions after discarding the first five frames.}
  \label{fig:efficiency}
\end{figure*}

\begin{table*}[t]
  \centering
  \caption{Efficiency benchmark on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU. Speedup is normalized against FLIP within each scene.}
  \label{tab:efficiency}
  \begin{tabular}{llrrrrrr}
    \toprule
    Scene & Algorithm & Reps & Mean ms/frame & Std & P95 & M particles/s & Speedup \\
    \midrule
@@efficiency_rows
    \bottomrule
  \end{tabular}
\end{table*}

The efficiency benchmark shows the expected cost ladder. FLIP is fastest in both scenes because it transfers only the baseline particle velocity state. APIC is slower because it carries and updates affine velocity information. PolyPIC is slowest because its higher-order local transfer stores and evaluates more particle-side information. Relative to FLIP, APIC reaches $0.79\times$ speed on dam break and $0.64\times$ speed on liquid pouring, while PolyPIC reaches $0.64\times$ and $0.60\times$, respectively.

\section{Discussion}

The experiments support four observations. First, a shared framework is essential: small changes in resolution, particle count, or boundary treatment can dominate the numerical differences between transfer schemes. Second, kinetic energy is informative but not sufficient on its own. Higher energy can indicate useful reduced dissipation, but it can also expose transfer noise or excessive momentum retention. Third, APIC and PolyPIC require more implementation care than baseline FLIP; APIC in particular needed affine-matrix limiting to avoid NaN growth in full runs. Fourth, richer transfer state has measurable runtime cost, as shown by the same-job benchmark.

For dam break, PolyPIC gives the clearest energy-retention advantage by integrated kinetic energy. For liquid pouring, APIC and PolyPIC are smoother and more restrained than FLIP, with PolyPIC retaining slightly more energy than APIC while remaining stable. These observations are consistent with the motivation behind affine and polynomial transfers: additional local velocity information can improve transfer quality, but a compact implementation must still manage stability and throughput.

\section{Limitations and Reproducibility}

This is a course-scale validation rather than a production simulator benchmark. The renderer is particle-based and not a high-quality surface reconstruction. The efficiency benchmark is simulation-only; rendering and video export are excluded, and the first five frames are discarded. This makes the timing comparison fairer than the original visual-output frame times, but it still measures this implementation, these scenes, and this hardware configuration rather than a universal algorithmic constant.

The committed artifacts include the energy summaries, efficiency summaries, plots, screenshots, videos, and scripts needed to reproduce the report figures. The main source for this paper is \texttt{docs/final\_report/option1\_comparative\_study.tex}, and the compiled PDF is \texttt{docs/final\_report/option1\_comparative\_study.pdf}.

\section*{Acknowledgment of AI Assistance}

AI tools were used to assist with code repair, Slurm job orchestration, plotting, and report drafting. The numerical values in the tables and figures are computed from repository CSV outputs generated by simulation runs, not invented by language-model text.

\begin{thebibliography}{00}

\bibitem{brackbill1986flip}
J.~U. Brackbill and H.~M. Ruppel, ``FLIP: A method for adaptively zoned, particle-in-cell calculations of fluid flows in two dimensions,'' \emph{Journal of Computational Physics}, vol.~65, no.~2, pp. 314--343, 1986.

\bibitem{jiang2015apic}
C.~Jiang, C.~Schroeder, A.~Selle, J.~Teran, and A.~Stomakhin, ``The affine particle-in-cell method,'' \emph{ACM Transactions on Graphics}, 2015.

\bibitem{fu2017polypic}
C.~Fu, Q.~Guo, T.~Gast, C.~Jiang, and J.~Teran, ``A polynomial particle-in-cell method,'' \emph{ACM Transactions on Graphics}, vol.~36, no.~6, Article 222, 2017.

\bibitem{bridson}
R.~Bridson, \emph{Fluid Simulation for Computer Graphics}. A K Peters/CRC Press.

\end{thebibliography}

\end{document}
""")
    return template.substitute(energy_rows=energy_rows, efficiency_rows=efficiency_rows)


def compile_tex() -> None:
    tectonic = locate_tectonic()
    if tectonic is None:
        raise RuntimeError(
            "No LaTeX compiler found. Install tectonic or run: "
            "micromamba install -y -p ./.micromamba_fluidsim -c conda-forge tectonic"
        )
    subprocess.run([tectonic, "option1_comparative_study.tex"], cwd=REPORT_DIR, check=True)


def build_latex_report(compile_pdf: bool = True) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    energy = read_csv(OUT / "energy_summary.csv")
    efficiency = read_csv(BENCHMARK_DIR / "efficiency_summary.csv")
    TEX_PATH.write_text(latex_source(energy_table_rows(energy), efficiency_table_rows(efficiency)))
    if compile_pdf:
        compile_tex()
        if not PDF_PATH.exists():
            raise FileNotFoundError(PDF_PATH)
    print("[ok] Wrote LaTeX source to", TEX_PATH)
    if compile_pdf:
        print("[ok] Wrote LaTeX PDF to", PDF_PATH)


def main() -> None:
    build_latex_report(compile_pdf=True)


if __name__ == "__main__":
    main()
