# PolyPIC Branch Implementation and Results

## Summary

This branch adds a quadratic PolyPIC transfer mode to the shared 3D MAC-grid fluid
simulation framework. The goal is to keep the same grid, scenes, pressure
projection, boundary conditions, renderer, and output format as the FLIP/APIC
branches, so the final comparison can use aligned videos and energy CSV files.

The implementation is selected with:

```bash
python framework.py all 0.97 polypic
```

On the cluster, the full production run was launched with:

```bash
sbatch slurm_polypic_full.sbatch
```

The completed full run was Slurm job `562`.

## Algorithm Implementation

The PolyPIC branch is implemented in `framework.py` as an additional solver
method named `polypic`. The main simulation loop still calls the shared
`fluid_step()` interface, but that function now dispatches the particle-grid
transfer stage according to `SOLVER_METHOD`:

- `flip`: original trilinear FLIP/PIC transfer.
- `polypic`: quadratic Polynomial PIC transfer.

For each particle, the PolyPIC mode stores:

- Particle velocity: `pu`, `pv`, `pw`.
- Affine APIC-style coefficients: `c0` through `c8`.
- Quadratic polynomial coefficients: `q0` through `q17`.

Each velocity component uses six quadratic basis coefficients:

- `xx`, `yy`, `zz`, `xy`, `xz`, `yz`

The transfer uses quadratic B-spline weights over a 3x3x3 stencil. During P2G,
the particle velocity is evaluated as a local polynomial around the particle:

```text
v(x) = v0 + C d + Q phi2(d)
```

where:

- `v0` is the particle velocity component.
- `C` is the affine velocity matrix.
- `d` is the face-center displacement from the particle.
- `Q` stores the quadratic coefficients.
- `phi2(d)` contains centered quadratic basis terms.

The quadratic terms are centered by local second moments so they do not shift
the constant velocity mode. During G2P, the solver fits velocity, affine terms,
and quadratic terms back from the MAC grid to each particle.

## Stability Notes

The first full PolyPIC run exposed numerical growth in the high-order terms:
the energy became `nan` after the early part of the simulation. The final branch
therefore includes conservative stabilization around the high-order polynomial
state:

- non-finite value cleanup;
- velocity CFL limiting;
- polynomial value limiting during P2G;
- damping and limiting of affine and quadratic coefficients after G2P.

This keeps the run stable for the 300-frame final scenes while preserving a
distinct quadratic PolyPIC transfer path for comparison.

## Output Files

The final PolyPIC outputs are under:

```text
output/polypic/ratio_970/
```

Primary handoff file:

```text
output/polypic/ratio_970/polypic_energy.csv
```

Scene-specific CSV files:

```text
output/polypic/ratio_970/polypic_energy_dam_break.csv
output/polypic/ratio_970/polypic_energy_liquid_pouring.csv
output/polypic/ratio_970/dam_break/energy.csv
output/polypic/ratio_970/liquid_pouring/energy.csv
```

Rendered videos:

```text
output/polypic/ratio_970/dam_break/dam_break.mp4
output/polypic/ratio_970/liquid_pouring/liquid_pouring.mp4
```

Energy plots:

```text
output/polypic/ratio_970/dam_break/energy.png
output/polypic/ratio_970/liquid_pouring/energy.png
```

Frame images are intentionally not committed. They are generated under each
scene's `frames/` directory and ignored by `.gitignore`.

## Result Validation

Final production run:

```text
Slurm job: 562
GPU partition: rtxp6000
Method: polypic
Ratio: 0.97
Frames per scene: 300
Simulated time per scene: 5.0 s
```

CSV validation:

```text
dam_break:
  rows: 300
  finite kinetic energy: true
  time range: 0.016667 s to 5.0 s
  first energy: 0.80742717
  last energy: 0.21173687
  max energy: 34.86706924

liquid_pouring:
  rows: 300
  finite kinetic energy: true
  time range: 0.016667 s to 5.0 s
  first energy: 0.0432
  last energy: 2.6197269
  max energy: 2.63019872
```

The output logs and CSV files were checked for `nan`, `Traceback`, and `ERROR`;
no such failure markers were present in the final run.

## Reproducing the Results

For a short stability check:

```bash
sbatch --export=ALL,FLUIDSIM_SMOKE_FRAMES=120 slurm_polypic_smoke.sbatch
```

For the full result:

```bash
sbatch slurm_polypic_full.sbatch
```

The full Slurm script sets:

```bash
TI_ARCH=cuda
FLUIDSIM_FORCE_RENDER=1
```

`FLUIDSIM_FORCE_RENDER=1` ensures that an existing old video does not cause the
run to skip rendering.

## Use in Final Comparison

After the FLIP and APIC branches produce their CSV files, generate comparison
figures with `plot_energy.py`. For the Dam Break comparison, use:

```bash
python plot_energy.py compare \
  output/flip/ratio_970/dam_break/energy.csv \
  output/apic/ratio_970/dam_break/energy.csv \
  output/polypic/ratio_970/dam_break/energy.csv \
  -s dam_break \
  -o output/comparison/dam_break
```

If the APIC branch uses a different output folder layout, adjust only the APIC
CSV path.
