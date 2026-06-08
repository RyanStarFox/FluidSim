# Efficiency Benchmark

This benchmark reruns FLIP, APIC, and PolyPIC under the same Slurm job on an RTX 6000 partition.
Rendering and video export are disabled so the timing focuses on simulation work. The first five frames are discarded to reduce Taichi JIT and initialization effects.

| Scene | Algorithm | Repetitions | Mean ms/frame | Std of run means | Median ms/frame | P95 ms/frame | M particles/s | Speedup vs FLIP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Dam Break | FLIP | 3 | 27.246 | 2.777 | 28.465 | 35.070 | 75.91 | 1.00x |
| Dam Break | APIC | 3 | 34.417 | 0.386 | 34.643 | 45.385 | 59.67 | 0.79x |
| Dam Break | PolyPIC | 3 | 42.639 | 0.055 | 39.852 | 57.272 | 48.16 | 0.64x |
| Liquid Pouring | FLIP | 3 | 9.053 | 0.046 | 7.413 | 15.317 | 33.36 | 1.00x |
| Liquid Pouring | APIC | 3 | 14.151 | 0.092 | 14.576 | 16.266 | 21.34 | 0.64x |
| Liquid Pouring | PolyPIC | 3 | 14.988 | 0.055 | 15.514 | 16.972 | 20.15 | 0.60x |

Generated artifacts:

- `output/benchmark_efficiency/efficiency_runs.csv`
- `output/benchmark_efficiency/efficiency_summary.csv`
- `output/benchmark_efficiency/dam_break_efficiency_ms_per_frame.png`
- `output/benchmark_efficiency/liquid_pouring_efficiency_ms_per_frame.png`
- `output/benchmark_efficiency/speedup_vs_flip.png`
