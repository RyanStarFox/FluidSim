#!/usr/bin/env python3
"""Run a simulation-only benchmark for one algorithm branch copy.

The framework's ``frame_time_ms`` is recorded before rendering and CSV energy
post-processing, so it is a useful simulation-only timing signal. This runner
loads a branch-specific ``framework.py``, disables frame rendering/video export,
and writes fresh ``energy.csv`` files for benchmark aggregation.
"""

from __future__ import annotations

import argparse
import inspect
import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCENES = ("dam_break", "liquid_pouring")


def load_framework(repo_dir: Path, module_name: str):
    framework_path = repo_dir / "framework.py"
    if not framework_path.exists():
        raise FileNotFoundError(framework_path)

    spec = importlib.util.spec_from_file_location(module_name, framework_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {framework_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True,
                        help="Directory containing the branch-specific framework.py")
    parser.add_argument("--algorithm", required=True,
                        choices=["flip", "apic", "polypic"])
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for this repetition/algorithm")
    parser.add_argument("--ratio", type=float, default=0.97)
    parser.add_argument("--scenes", nargs="+", default=list(SCENES),
                        choices=list(SCENES))
    args = parser.parse_args()

    fw = load_framework(args.repo.resolve(), f"fluidsim_{args.algorithm}_benchmark")

    # Disable work that is outside the simulation timing window.
    fw.export_frame = lambda frame_dir, frame_num: None
    fw._plot_energy = lambda csv_path, output_dir: None
    fw.subprocess.run = lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr="")

    for scene in args.scenes:
        out_dir = args.out / scene
        out_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = out_dir / f"{scene}.mp4"
        if mp4_path.exists():
            mp4_path.unlink()
        print(f"[benchmark] {args.algorithm} scene={scene} out={out_dir}", flush=True)
        kwargs = {}
        if "method" in inspect.signature(fw.run_simulation).parameters:
            kwargs["method"] = args.algorithm
        fw.run_simulation(scene, args.ratio, out_dir, **kwargs)


if __name__ == "__main__":
    main()
