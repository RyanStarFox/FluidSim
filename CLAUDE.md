# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Comparative study of three particle-grid transfer methods (FLIP, APIC, PolyPIC) using a shared 3D Taichi-based fluid simulation framework. This is a final-submission snapshot for a university course project — the repo is not under active development.

Team: Qinzhe Hu (PolyPIC + report), Yan Shao (framework + FLIP baseline), Baihan Deng (APIC).

## Python version constraint

**Python 3.9–3.11 only.** Taichi 1.7.x does not support Python 3.12+. On macOS, use the system `/usr/bin/python3` (3.9).

## Quick start

```bash
# Install deps (macOS system Python)
/usr/bin/python3 -m pip install taichi pillow matplotlib numpy --user

# Run a single scene (dam break or liquid pouring)
/usr/bin/python3 framework.py dam_break
/usr/bin/python3 framework.py liquid_pouring 0.97

# Run both scenes
/usr/bin/python3 framework.py all 0.97

# GPU backend selection (macOS M-series auto-selects Metal)
TI_ARCH=metal  python3 framework.py dam_break   # Apple Metal (default on M-series)
TI_ARCH=cpu    python3 framework.py dam_break   # Force CPU
TI_ARCH=cuda   python3 framework.py dam_break   # NVIDIA GPU (Linux only)
```

## Source code layout

The three algorithms exist as separate, full copies of `framework.py` — each branch modified only the `fluid_step()` function:

| Algorithm | Source | Original branch |
|-----------|--------|-----------------|
| FLIP (baseline) | `submission/code/flip/framework.py` | `main` |
| APIC | `submission/code/apic/framework.py` | `apic-branch` |
| PolyPIC | `submission/code/polypic/framework.py` | `polypic` |

The root `framework.py` is the **APIC** variant (final branch copy). When working on a specific algorithm, use `submission/code/<algo>/framework.py`.

Shared utility scripts (in `scripts/`):
- `build_option1_report.py` — generates comparison figures, summary CSVs, and the Markdown/LaTeX/PDF report from energy CSVs
- `build_latex_report.py` — LaTeX → PDF compilation helper (called by build_option1_report.py)
- `summarize_efficiency.py` — aggregates benchmark repetition data into efficiency summaries and plots
- `run_branch_benchmark.py` — loads a branch-specific framework.py, disables rendering/video, runs simulation-only timing
- `submit_efficiency_benchmark.sbatch` — Slurm job script for the GPU efficiency benchmark

## Architecture: simulation pipeline

The framework implements a **MAC (staggered) grid + particle hybrid** in 3D using Taichi:

```
Per-frame loop:
  1. emit_inflow_particles()    # liquid_pouring only: add new particles at top
  2. For each SUBSTEP:
     a. fluid_step()            # THE EXTENSION POINT — P2G → forces → pressure → G2P
     b. advect_particles()      # forward Euler: pos += vel * dt
     c. mark_fluid_cells()      # re-tag cells containing particles as FLUID
  3. remove_ghost_particles()   # periodically cull airborne particles in AIR cells
  4. export_frame()             # GPU perspective rendering → PNG
```

### `fluid_step()` — the extension interface

This function is the **only place** teammates modify to implement their algorithm. It orchestrates:
1. **P2G** — scatter particle velocities (and affine matrices for APIC/PolyPIC) to the MAC grid via trilinear interpolation
2. **External forces** — `add_gravity()` applies g=−9.81 on the y-axis
3. **Boundary conditions** — `enforce_boundary_velocity()` zeros normal velocity at SOLID walls (free-slip)
4. **Pressure projection** — computes divergence, solves ∇²p = ∇·u/dt via Conjugate Gradient, applies pressure gradient correction
5. **G2P** — interpolates grid velocities back to particles (PIC-FLIP blend for FLIP; velocity + affine matrix update for APIC/PolyPIC)

### Key Taichi fields (global, accessible in `fluid_step()`)

| Field | Shape | Description |
|-------|-------|-------------|
| `u`, `v`, `w` | staggered MAC faces | Velocity components (x/y/z) |
| `u_saved`, `v_saved`, `w_saved` | same as u/v/w | Velocity snapshots for FLIP delta |
| `cell_type` | `(NX, NY, NZ)` | FLUID=0, SOLID=1, AIR=2 |
| `pressure` | `(NX, NY, NZ)` | CG solver output |
| `px`, `py`, `pz` | `(MAX_PARTICLES,)` | Particle positions |
| `pu`, `pv`, `pw` | `(MAX_PARTICLES,)` | Particle velocities |
| `c0`…`c8` | `(MAX_PARTICLES,)` | 3×3 affine velocity matrix (APIC/PolyPIC) |
| `num_particles` | scalar | Active particle count |
| `flip_ratio` | scalar | PIC-FLIP blend ratio |

### Key helper functions (callable in `fluid_step()`)

- `p2g_trilinear()` — P2G with standard trilinear weights (FLIP)
- `p2g_apic()` — P2G with affine momentum (APIC/PolyPIC)
- `save_velocities()` — snapshot u,v,w → u_saved,v_saved,w_saved
- `add_gravity()` — v += GRAVITY_Y * DT
- `enforce_boundary_velocity()` — free-slip BC on all 6 domain walls + internal solids
- `compute_divergence()` — ∇·u → div_field
- `solve_pressure_cg()` — CG solve, returns iteration count
- `apply_pressure_gradient(dt)` — subtracts ∇p from MAC velocities
- `g2p_flip()` — PIC-FLIP blend G2P (clears c0…c8)
- `g2p_apic()` — APIC G2P: velocity + affine matrix update

### CG pressure solver details

Solves the positive-definite system (−∇²)p = −∇·u/dt (both sides negated so eigenvalues > 0). Neumann BC at SOLID neighbors (zero normal gradient), Dirichlet BC (p=0) at AIR neighbors. Cold-start every frame (pressure reset to 0) — warm-starting with the previous frame's pressure is counter-productive for rapidly-evolving free surfaces.

### Ghost particle removal

Particles floating in AIR cells above 62% of domain height are periodically removed (every 5 frames starting from frame 5). This prevents the "floating slab" artefact from pressure overshoots at the free surface.

## Grid and simulation parameters

| Parameter | Value |
|-----------|-------|
| Grid resolution | 80 × 100 × 80 (NX × NY × NZ) |
| Cell size (DX) | 0.01 m |
| Particles per cell | 3³ = 27 (PPC=3) |
| Max particles | 2,300,000 |
| Frames | 300 per scene (5 s @ 60 fps) |
| Sub-steps | 2 per frame (dt = 1/120 s) |
| FLIP ratio | 0.97 (default) |
| CG tolerance | 5e-3 (relative) |
| CG max iterations | 100 |
| Rendering | 1080×1080 perspective projection PNG, compiled to 60fps H.264 MP4 |

## Scenes

- **Dam Break** (`dam_break`): Tall water column on the left collapses, wave propagates rightward
- **Liquid Pouring** (`liquid_pouring`): Inflow stream from top-center falls onto a rectangular obstacle, pools on the floor. Uses Weyl-sequence quasi-random particle emission (2000 particles/frame for first 300 frames)

## Output structure

```
output/
  {flip,apic,polypic}/
    ratio_970/
      dam_break/
        frames/         # frame_0000.png … frame_0299.png (1080×1080)
        dam_break.mp4   # 60fps H.264 video
        energy.csv      # per-frame: time, kinetic_energy, frame_time_ms, total_vorticity
        energy.png      # auto-generated self-check plot
      liquid_pouring/   # same structure
  comparison/           # cross-algorithm comparisons (generated by build_option1_report.py)
  benchmark_efficiency/ # GPU benchmark results
```

## Plotting and analysis

```bash
# Single-run self-check
python3 plot_energy.py single output/flip/ratio_970/dam_break/energy.csv -o check.png

# Three-way algorithm comparison
python3 plot_energy.py compare \
    output/flip/ratio_970/dam_break/energy.csv \
    output/apic/ratio_970/dam_break/energy.csv \
    output/polypic/ratio_970/dam_break/energy.csv \
    -s dam_break -o comparison/

# FLIP ratio sweep analysis
python3 plot_energy.py sweep output/flip/ -s dam_break -o sweep/

# Full report generation (comparison figures + LaTeX → PDF)
python3 scripts/build_option1_report.py

# Efficiency benchmark aggregation
python3 scripts/summarize_efficiency.py --raw-dir output/benchmark_efficiency/raw/

# Sequential batch rendering (all FLIP ratios × both scenes)
bash render_all.sh
```

## Report and documentation

- Final report: `docs/final_report/option1_comparative_study.pdf` (and `.tex` source)
- Efficiency benchmark doc: `docs/final_report/efficiency_benchmark.md`
- Per-algorithm result notes: `docs/APIC_RESULTS.md`, `docs/POLYPIC_RESULTS.md`
- Submission packaging: `submission/package_submission.py` / `.sh`

## Important conventions

- **Framework boundary**: The `fluid_step()` function body is the only code teammates should modify. Framework infrastructure (grid setup, rendering, I/O, scene setup, CG solver) must not be changed.
- **Affine matrix damping for APIC**: The APIC implementation includes NaN/inf guards (`_sanitize_limit`) and affine matrix damping (`APIC_C_DAMPING = 0.5`) for stability — this is a practical necessity for long runs and affects energy behavior relative to an ideal APIC formulation.
- **Velocity clamping**: All particle velocities are clamped to `V_MAX_PHYS` (CFL-based: 0.9·DX/DT ≈ 1.08 m/s) each substep to prevent particles from skipping multiple cells.
- **Benchmark mode**: `run_branch_benchmark.py` monkey-patches `export_frame`, `_plot_energy`, and `subprocess.run` to no-ops, isolating simulation timing from rendering/video overhead. First 5 frames are warm-up (discarded for JIT compilation effects).
