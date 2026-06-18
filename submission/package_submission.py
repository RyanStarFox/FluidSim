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

本说明用于组员快速审阅最终提交压缩包。文件名和目录结构保留英文，解释文字使用中文。
压缩包根目录已经额外放置一份最终报告 PDF，方便打开检查。

Package structure
fluidsim_final_submission.zip
├── option1_comparative_study.pdf
├── README_FOR_REVIEW.txt
├── README.md
├── framework.py
├── plot_energy.py
├── render_all.sh
├── scripts/
├── docs/
│   ├── APIC_RESULTS.md
│   ├── POLYPIC_RESULTS.md
│   └── final_report/
│       ├── option1_comparative_study.pdf
│       ├── option1_comparative_study.tex
│       ├── efficiency_benchmark.md
│       └── figures/
├── submission/
│   ├── README_SUBMISSION.md
│   ├── package_submission.py
│   ├── package_submission.sh
│   └── code/
│       ├── flip/framework.py
│       ├── apic/framework.py
│       └── polypic/framework.py
└── output/
    ├── flip/ratio_970/
    ├── apic/ratio_970/
    ├── polypic/ratio_970/
    ├── comparison/
    └── benchmark_efficiency/

Top-level files
- option1_comparative_study.pdf: 最终报告 PDF。报告附录包含小组成员姓名、缩写、学号和分工。
- README_FOR_REVIEW.txt: 当前这份纯文本审阅说明，方便组员不解压太深也能知道包里有什么。
- README.md: 仓库原始说明，包括运行环境、基本命令和输出结构。
- framework.py: final 分支根目录下的框架文件，仅作仓库入口参考；三算法源码以 submission/code/ 下的副本为准。
- plot_energy.py, render_all.sh: 绘图和批处理辅助脚本。

Source code
- submission/code/flip/framework.py: FLIP baseline 代码，从 main 分支导出。
- submission/code/apic/framework.py: APIC 代码，从 apic-branch 分支导出。
- submission/code/polypic/framework.py: PolyPIC 代码，从 polypic 分支导出。
- scripts/: 包含报告生成、效率汇总、打包和 benchmark 相关脚本。

Report and figures
- docs/final_report/option1_comparative_study.pdf: 与根目录报告相同，保留在报告目录中方便配合 LaTeX 源码查看。
- docs/final_report/option1_comparative_study.tex: 最终报告的 LaTeX 源文件。
- docs/final_report/figures/: 报告中使用的能量曲线、效率图、截图和汇总图。

Demo videos
- output/flip/ratio_970/dam_break/dam_break.mp4: FLIP 溃坝场景 demo。
- output/flip/ratio_970/liquid_pouring/liquid_pouring.mp4: FLIP 倒水场景 demo。
- output/apic/ratio_970/dam_break/dam_break.mp4: APIC 溃坝场景 demo。
- output/apic/ratio_970/liquid_pouring/liquid_pouring.mp4: APIC 倒水场景 demo。
- output/polypic/ratio_970/dam_break/dam_break.mp4: PolyPIC 溃坝场景 demo。
- output/polypic/ratio_970/liquid_pouring/liquid_pouring.mp4: PolyPIC 倒水场景 demo。

Data and benchmark outputs
- output/comparison/: 三个算法的能量曲线对比数据、AUC 汇总和对比图。
- output/benchmark_efficiency/: 效率实验的 summary CSV、逐次运行 CSV、硬件元数据和效率图。
- output/{flip,apic,polypic}/ratio_970/: 每个算法在两个场景下的 energy.csv、energy.png 和 demo 视频。

Team information in the report appendix
- Qinzhe Hu (hqz, 523030910139): PolyPIC 分支实现、对比数据可视化、最终报告撰写、提交材料整合。
- Yan Shao (sy, 523031910224): 统一模拟框架、FLIP baseline、公共实验场景设置、基线结果生成。
- Baihan Deng (dbh, 523031910756): APIC 分支实现、APIC 完整运行验证、效率对比支持。

Not included
- division_of_work.md: 按要求不放入最终 zip。
- frames/: 原始逐帧 PNG 数量较多，属于中间产物，已排除。
- output/benchmark_efficiency/raw/: 原始 benchmark 日志属于中间产物，已排除。
- .git, caches, local environments, dist/: 本地开发和缓存目录，已排除。
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
