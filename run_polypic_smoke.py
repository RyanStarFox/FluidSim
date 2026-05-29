#!/usr/bin/env python3
import argparse
from pathlib import Path

import framework


def main():
    parser = argparse.ArgumentParser(description="Short PolyPIC smoke run")
    parser.add_argument("--scene", default="dam_break", choices=["dam_break", "liquid_pouring"])
    parser.add_argument("--ratio", type=float, default=0.97)
    parser.add_argument("--method", default="polypic", choices=list(framework.SOLVER_METHODS))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output-root", type=Path, default=Path("output/slurm_smoke/manual"))
    args = parser.parse_args()

    framework.SCENE_FRAMES[args.scene] = args.frames
    ratio_str = f"ratio_{int(args.ratio * 1000):03d}"
    out_dir = args.output_root / args.method / ratio_str / f"{args.scene}_{args.frames}f"
    print(f"[smoke] scene={args.scene} method={args.method} frames={args.frames} out={out_dir}", flush=True)
    framework.run_simulation(args.scene, args.ratio, out_dir, args.method)


if __name__ == "__main__":
    main()
