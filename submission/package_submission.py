#!/usr/bin/env python3
"""Create the final FluidSim submission zip using only the Python standard library."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ZIP_PATH = DIST / "fluidsim_final_submission.zip"

INCLUDE_PATHS = [
    "README.md",
    "division_of_work.md",
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

    files = collect_files()
    with ZipFile(ZIP_PATH, "w", ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, file_path.relative_to(ROOT).as_posix())

    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {ZIP_PATH}")
    print(f"Files: {len(files)}")
    print(f"Size: {size_mb:.2f} MiB")


if __name__ == "__main__":
    main()
