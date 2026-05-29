# APIC Full-Run Results

Branch: `apic-branch`
Algorithm: Affine Particle-In-Cell (APIC)
Run job: Slurm job `565` on `rtxp6000`
Taichi backend: CUDA
Ratio: `0.97`

## Implementation Summary

The APIC branch keeps the shared experiment setup unchanged and implements the solver inside `fluid_step()`. Each simulation step transfers particle velocities and local affine velocity matrices to the staggered MAC grid, solves incompressibility with the existing pressure projection, then gathers grid velocities back to particles while updating the affine matrix.

Compared with a basic PIC/FLIP transfer, APIC carries first-order local velocity variation through a per-particle 3x3 affine matrix `C`. During P2G, the particle contribution at a grid face is evaluated as `v_p + C_p * (x_grid - x_p)`. During G2P, the new particle velocity is interpolated from grid faces and `C_p` is reconstructed from weighted outer products of sampled grid velocities and offsets.

## Stability Fixes

The original APIC branch failed the smoke test because `fluid_step()` was decorated as a Taichi kernel and then called other kernels. That was fixed by making `fluid_step()` a normal Python driver. The full run also exposed NaN growth in the APIC affine transfer, so the final branch adds finite-value guards, velocity limiting, and damping for the affine matrix. These changes keep the solver stable under the shared high-particle-count scenes without changing the common scene setup.

## Produced Artifacts

- `output/apic/ratio_970/apic_energy.csv`
- `output/apic/ratio_970/apic_energy_dam_break.csv`
- `output/apic/ratio_970/apic_energy_liquid_pouring.csv`
- `output/apic/ratio_970/dam_break/energy.csv`
- `output/apic/ratio_970/dam_break/energy.png`
- `output/apic/ratio_970/dam_break/dam_break.mp4`
- `output/apic/ratio_970/liquid_pouring/energy.csv`
- `output/apic/ratio_970/liquid_pouring/energy.png`
- `output/apic/ratio_970/liquid_pouring/liquid_pouring.mp4`

Intermediate PNG frames are generated during rendering but are intentionally ignored by git.

## Validation

Both full scenes completed 300 frames with finite kinetic energy values.

| Scene | Rows | Finite | First kinetic energy | Final kinetic energy | Max kinetic energy |
|---|---:|:---:|---:|---:|---:|
| dam_break | 300 | yes | 0.83330113 | 0.01778594 | 37.08362961 |
| liquid_pouring | 300 | yes | 0.04320000 | 2.33835316 | 2.36300254 |

## Notes for Comparison

For the class comparison pipeline, use `apic_energy.csv` as the default APIC curve. It mirrors the `dam_break` scene so it is directly comparable with the existing PolyPIC default CSV convention. The scene-specific CSVs are retained for any expanded two-scene comparison.
