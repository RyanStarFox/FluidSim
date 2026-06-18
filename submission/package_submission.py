#!/usr/bin/env python3
"""Create the final FluidSim submission zip using only the Python standard library."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ZIP_PATH = DIST / "fluidsim_final_submission.zip"
ROOT_REPORT_PATH = ROOT / "docs" / "final_report" / "option1_comparative_study.pdf"
ROOT_REPORT_NAME = "option1_comparative_study.pdf"
REVIEW_NOTE_PATH = DIST / "fluidsim_submission_review_note.txt"
REVIEW_NOTE_ARCNAME = "README_FOR_REVIEW.txt"

REVIEW_NOTE_TEXT = """FluidSim final submission package overview

This zip is prepared for the final course project submission and group review.
The report PDF is also copied to the top level of the zip for quick access.

Top-level files
- option1_comparative_study.pdf: final report PDF. The appendix contains team member names, identifiers, student IDs, and division of work.
- README_FOR_REVIEW.txt: this plain-text review note.
- README.md: repository-level usage notes.
- framework.py: framework file from the final branch.
- plot_energy.py, render_all.sh: helper scripts.

Source code
- submission/code/flip/framework.py: FLIP baseline source exported from branch main.
- submission/code/apic/framework.py: APIC source exported from branch apic-branch.
- submission/code/polypic/framework.py: PolyPIC source exported from branch polypic.
- scripts/: report generation, plotting, benchmark, and packaging scripts.

Report and figures
- docs/final_report/option1_comparative_study.pdf: same final report PDF, kept with the LaTeX source and figures.
- docs/final_report/option1_comparative_study.tex: LaTeX source.
- docs/final_report/figures/: final report figures and screenshots.

Demo videos
- output/flip/ratio_970/dam_break/dam_break.mp4
- output/flip/ratio_970/liquid_pouring/liquid_pouring.mp4
- output/apic/ratio_970/dam_break/dam_break.mp4
- output/apic/ratio_970/liquid_pouring/liquid_pouring.mp4
- output/polypic/ratio_970/dam_break/dam_break.mp4
- output/polypic/ratio_970/liquid_pouring/liquid_pouring.mp4

Data and benchmark outputs
- output/comparison/: cross-algorithm energy CSV files and comparison plots.
- output/benchmark_efficiency/: efficiency summary CSV files, metadata, and benchmark plots.
- output/{flip,apic,polypic}/ratio_970/: per-algorithm energy CSV files, energy plots, and demo videos.

Team information in the report appendix
- Qinzhe Hu (hqz, 523030910139): PolyPIC branch, data visualization, final report, integration.
- Yan Shao (sy, 523031910224): shared framework, FLIP baseline, common scenes, baseline results.
- Baihan Deng (dbh, 523031910756): APIC branch, APIC validation, efficiency comparison support.

Not included
- division_of_work.md is intentionally not included in the zip.
- Raw rendered frame directories are excluded.
- Raw benchmark logs are excluded.
- .git, caches, local environments, and dist/ are excluded.
"""

INCLUDE_PATHS = [
    "README.md",
    "framework.py",
    "plot_energy.py",
    "render_all.sh",
    "scripts",
    "docs",
    "submission",
    "output/flip/ratio_970",
    "output/apic/ratio_970",
    "output/polypic/ratio_970",
    "output/comparison",
    "output/benchmark_efficiency/benchmark_metadata.txt",
    "output/benchmark_efficiency/efficiency_runs.csv",
    "output/benchmark_efficiency/efficiency_summary.csv",
    "output/benchmark_efficiency/dam_break_efficiency_ms_per_frame.png",
    "output/benchmark_efficiency/liquid_pouring_efficiency_ms_per_frame.png",
    "output/benchmark_efficiency/speedup_vs_flip.png",
]

EXCLUDE_PARTS = {
    "__pycache__",
    "frames",
    "raw",
    "dist",
    ".git",
    ".micromamba_fluidsim",
    "slurm_logs",
}


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return not (set(rel.parts) & EXCLUDE_PARTS)


def collect_files() -> list[Path]:
    files: list[Path] = []
    for item in INCLUDE_PATHS:
        path = ROOT / item
        if not path.exists():
            raise FileNotFoundError(item)
        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file() and should_include(file_path):
                    files.append(file_path)
        elif should_include(path):
            files.append(path)

    unique: list[Path] = []
    seen: set[str] = set()
    for file_path in files:
        rel = file_path.relative_to(ROOT).as_posix()
        if rel not in seen:
            seen.add(rel)
            unique.append(file_path)
    return unique


def main() -> None:
    DIST.mkdir(exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    REVIEW_NOTE_PATH.write_text(REVIEW_NOTE_TEXT, encoding="utf-8")
    if not ROOT_REPORT_PATH.exists():
        raise FileNotFoundError(ROOT_REPORT_PATH)

    files = collect_files()
    with ZipFile(ZIP_PATH, "w", ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, file_path.relative_to(ROOT).as_posix())
        archive.write(ROOT_REPORT_PATH, ROOT_REPORT_NAME)
        archive.write(REVIEW_NOTE_PATH, REVIEW_NOTE_ARCNAME)

    file_count = len(files) + 2
    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {ZIP_PATH}")
    print(f"Review note: {REVIEW_NOTE_PATH}")
    print(f"Files: {file_count}")
    print(f"Size: {size_mb:.2f} MiB")


if __name__ == "__main__":
    main()
