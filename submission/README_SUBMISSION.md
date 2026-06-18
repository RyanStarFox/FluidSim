# FluidSim Final Submission Manifest

This branch is the consolidated final submission branch. It keeps the final
report, source code, plots, CSV data, benchmark summaries, and demo videos in
one place.

## Required Report

- Final PDF report:
  - `docs/final_report/option1_comparative_study.pdf`
- LaTeX source used to build the PDF:
  - `docs/final_report/option1_comparative_study.tex`
  - `scripts/build_latex_report.py`

Important: the current PDF already contains the team member names, but the
student IDs are not known in this repository. Before uploading the final zip,
add the student IDs and the verified per-member division of work to the PDF.

## Source Code

The three algorithm branches are preserved as separate framework copies:

| Algorithm | Source file | Branch | Commit used in benchmark |
|---|---|---|---|
| FLIP | `submission/code/flip/framework.py` | `main` | `b1f84b6` |
| APIC | `submission/code/apic/framework.py` | `apic-branch` | `3edde91` |
| PolyPIC | `submission/code/polypic/framework.py` | `polypic` | `0dc4cf1` |

Additional shared scripts:

- `plot_energy.py`
- `scripts/build_option1_report.py`
- `scripts/build_latex_report.py`
- `scripts/run_branch_benchmark.py`
- `scripts/summarize_efficiency.py`
- `scripts/submit_efficiency_benchmark.sbatch`

## Demo Videos

Include these six MP4 files in the final zip:

- `output/flip/ratio_970/dam_break/dam_break.mp4`
- `output/flip/ratio_970/liquid_pouring/liquid_pouring.mp4`
- `output/apic/ratio_970/dam_break/dam_break.mp4`
- `output/apic/ratio_970/liquid_pouring/liquid_pouring.mp4`
- `output/polypic/ratio_970/dam_break/dam_break.mp4`
- `output/polypic/ratio_970/liquid_pouring/liquid_pouring.mp4`

Do not include the `frames/` directories unless the instructor explicitly asks
for raw rendered PNG frames.

## Data and Figures

Primary comparison data:

- `output/comparison/energy_summary.csv`
- `output/comparison/dam_break_energy_timeseries.csv`
- `output/comparison/liquid_pouring_energy_timeseries.csv`

Per-algorithm scene data:

- `output/flip/ratio_970/dam_break/energy.csv`
- `output/flip/ratio_970/liquid_pouring/energy.csv`
- `output/apic/ratio_970/dam_break/energy.csv`
- `output/apic/ratio_970/liquid_pouring/energy.csv`
- `output/polypic/ratio_970/dam_break/energy.csv`
- `output/polypic/ratio_970/liquid_pouring/energy.csv`

Efficiency benchmark:

- `output/benchmark_efficiency/efficiency_summary.csv`
- `output/benchmark_efficiency/efficiency_runs.csv`
- `output/benchmark_efficiency/benchmark_metadata.txt`
- `docs/final_report/efficiency_benchmark.md`

Final report figures:

- `docs/final_report/figures/*.png`
- `docs/final_report/figures/screenshots/*.png`

The raw benchmark logs under `output/benchmark_efficiency/raw/` are intermediate
files and can be omitted from the final zip unless detailed provenance is needed.

## Team and Division of Work

Verify this mapping before final upload:

| Member | Student ID | Main responsibility |
|---|---|---|
| Qinzhe Hu | TODO | PolyPIC branch, data visualization, final report and integration |
| Yan Shao | TODO | APIC branch, APIC output validation, presentation support |
| Baihan Deng | TODO | Shared framework, FLIP baseline, common scene setup |

## Recommended Zip Contents

For a compact submission zip, include:

- `README.md`
- `division_of_work.md`
- `submission/`
- `framework.py`
- `plot_energy.py`
- `render_all.sh`
- `scripts/`
- `docs/`
- `output/flip/ratio_970/`
- `output/apic/ratio_970/`
- `output/polypic/ratio_970/`
- `output/comparison/`
- `output/benchmark_efficiency/efficiency_summary.csv`
- `output/benchmark_efficiency/efficiency_runs.csv`
- `output/benchmark_efficiency/benchmark_metadata.txt`
- `output/benchmark_efficiency/*.png`

Exclude:

- `__pycache__/`
- `.git/`
- `.micromamba_fluidsim/`
- `slurm_logs/`
- `output/**/frames/`
- `output/benchmark_efficiency/raw/`
