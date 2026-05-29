#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
framework.py - Unified 3D fluid simulation framework
=====================================================
3D MAC grid + FLIP implementation for FLIP / APIC / PolyPIC comparison.
Provides trilinear P2G/G2P, CG pressure solver, and perspective 3D rendering.

Course: Final Project - Fluid Simulation Methods Comparison
Phase I (Student A): 3D framework + FLIP implementation

Quick start:
    python framework.py dam_break
    python framework.py liquid_pouring
    python framework.py all
"""

import os
import sys
import time
import csv
import math
from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

import taichi as ti

# =============================================================================
# Configuration
# =============================================================================

# 3D Grid dimensions (Y = height / gravity axis)
# High-quality: 80×100×80, DX=0.01 m.  Combined with PPC=3 this gives
# 27 particles/cell and 640 K cells — richer surface detail and smoother
# free-surface dynamics.
NX = 80
NY = 100
NZ = 80
DX = 1.0 / NY          # uniform cell size; domain = 0.8 x 1.0 x 0.8
INV_DX = float(NY)     # 1 / DX

# Time stepping
# CFL stability: dt < 0.9 * DX / v_max.
# We use SUBSTEPS=4 (dt≈0.0042 s) combined with a per-step velocity clamp
# (V_MAX_PHYS) that keeps CFL≤0.9 without distorting slow-flow frames.
FRAME_DT = 1.0 / 60.0
SUBSTEPS = 2
DT = FRAME_DT / SUBSTEPS

# Hard velocity ceiling: clamp |v| ≤ 0.9*DX/DT each substep (CFL safety).
# DX=0.01, DT=1/120 → V_MAX = 0.9*0.01*120 = 1.08 m/s.
V_MAX_PHYS = 0.9 * DX / DT
NUM_FRAMES = 450
PHYSICS_DURATION = NUM_FRAMES * FRAME_DT

# Particles
PPC = 3                             # particles per cell per dimension (3x3x3=27)
MAX_PARTICLES = 2_300_000
GRAVITY_Y = -9.81
RHO = 1000.0
MASS_PER_PARTICLE = RHO * (DX ** 3) / (PPC ** 3)

# PIC-FLIP blend
DEFAULT_FLIP_RATIO = 0.97
FLIP_RATIOS = [0.97]

# Cell type flags
FLUID = 0
SOLID = 1
AIR   = 2

# CG solver
CG_MAX_ITER = 100   # hard cap; typically converges in 55-70 iters
# Relative tolerance: sqrt(rsnew/rsold) < CG_TOL.
# 5e-3 gives visually accurate pressure (residual drops ×200) in ~60 iterations.
# Tightening beyond 1e-3 requires ×1000 reduction and 200+ iterations
# without meaningfully improving the rendered output.
CG_TOL = 5e-3

# Rendering (perspective 3D)
IMG_W = 1080
IMG_H = 1080

# Per-scene camera configurations.
# dam_break: side-on view from +X side, looking down the channel so the
#   horizontal wave front is clearly visible propagating left→right.
# liquid_pouring: elevated front-left view centered on the obstacle so the
#   stream, impact, and spreading pool are all visible.
SCENE_CAMERAS = {
    # 3/4 view: camera upper-right, looking toward the base of the left wall.
    # The column at x=0 appears on the LEFT; the floor extends to the RIGHT
    # (wave direction).  Z-offset from camera gives clear 3-D depth.
    "dam_break": {
        "pos":    np.array([1.30, 0.85, 1.50], dtype=np.float64),
        "target": np.array([0.10, 0.02, 0.35], dtype=np.float64),
        "up":     np.array([0.00, 1.00, 0.00], dtype=np.float64),
        "fov":    62.0,
    },
    # Same 3/4 perspective as dam_break but aimed at domain centre where the
    # stream lands, so the falling column, obstacle, and spreading pool are all
    # clearly visible with 3-D depth.
    "liquid_pouring": {
        "pos":    np.array([1.30, 0.85, 1.50], dtype=np.float64),
        "target": np.array([0.40, 0.15, 0.40], dtype=np.float64),
        "up":     np.array([0.00, 1.00, 0.00], dtype=np.float64),
        "fov":    62.0,
    },
}
# Default fallback camera (overridden per-scene in run_simulation)
CAM_POS    = np.array([1.40, 0.65, 0.40], dtype=np.float64)
CAM_TARGET = np.array([0.00, 0.02, 0.40], dtype=np.float64)
CAM_UP     = np.array([0.00, 1.00, 0.00], dtype=np.float64)
CAM_FOV_DEG = 58.0

EXPORT_INTERVAL = 1
OUTPUT_BASE = Path("output")

# =============================================================================
# Taichi initialization
# =============================================================================

def init_taichi():
    """Initialize Taichi with the best available backend.

    Priority on macOS (Apple Silicon):
      1. TI_ARCH env var (if set)
      2. Metal  (GPU, fastest for this workload)
      3. CPU    (fallback if Metal unavailable)

    Note on Vulkan (macOS/MoltenVK): functionally correct but shader
    compilation via SPIRV takes 5-10 minutes on first run -- not practical.
    """
    env_arch = os.environ.get("TI_ARCH", "").lower()
    arch_map = {
        "cpu": ti.cpu, "cuda": ti.cuda, "metal": ti.metal,
        "vulkan": ti.vulkan, "opengl": ti.opengl,
    }

    if env_arch:
        arch = arch_map.get(env_arch, ti.cpu)
        ti.init(arch=arch, default_fp=ti.f32)
        print(f"[framework] Taichi backend: {arch} (from TI_ARCH)")
        return

    if sys.platform == "darwin":
        # Try Metal first; fall back to CPU if unavailable
        try:
            ti.init(arch=ti.metal, default_fp=ti.f32)
            print("[framework] Taichi backend: Metal (Apple GPU)")
            return
        except Exception as e:
            print(f"[framework] Metal unavailable ({e}), falling back to CPU")
            ti.init(arch=ti.cpu, default_fp=ti.f32)
            print("[framework] Taichi backend: CPU")
            return

    # Linux / Windows: prefer CUDA, then generic GPU, then CPU
    for arch in (ti.cuda, ti.gpu, ti.cpu):
        try:
            ti.init(arch=arch, default_fp=ti.f32)
            print(f"[framework] Taichi backend: {arch}")
            return
        except Exception:
            continue

init_taichi()

# =============================================================================
# Taichi fields -- 3D MAC grid + particles
# =============================================================================

# MAC grid velocities (staggered)
u = ti.field(ti.f32, shape=(NX + 1, NY,     NZ    ))  # x-faces
v = ti.field(ti.f32, shape=(NX,     NY + 1, NZ    ))  # y-faces
w = ti.field(ti.f32, shape=(NX,     NY,     NZ + 1))  # z-faces

# Saved copies for FLIP delta computation (also used as P2G weight buffers)
u_saved = ti.field(ti.f32, shape=(NX + 1, NY,     NZ    ))
v_saved = ti.field(ti.f32, shape=(NX,     NY + 1, NZ    ))
w_saved = ti.field(ti.f32, shape=(NX,     NY,     NZ + 1))

# Cell-centered fields
cell_type = ti.field(ti.i32, shape=(NX, NY, NZ))
pressure  = ti.field(ti.f32, shape=(NX, NY, NZ))

# CG solver scratch
cg_r  = ti.field(ti.f32, shape=(NX, NY, NZ))
cg_p  = ti.field(ti.f32, shape=(NX, NY, NZ))
cg_Ap = ti.field(ti.f32, shape=(NX, NY, NZ))
div_field = ti.field(ti.f32, shape=(NX, NY, NZ))

# Particle data
px = ti.field(ti.f32, shape=MAX_PARTICLES)
py = ti.field(ti.f32, shape=MAX_PARTICLES)
pz = ti.field(ti.f32, shape=MAX_PARTICLES)
pu = ti.field(ti.f32, shape=MAX_PARTICLES)
pv = ti.field(ti.f32, shape=MAX_PARTICLES)
pw = ti.field(ti.f32, shape=MAX_PARTICLES)

# Affine velocity matrix (3x3 = 9 components) for APIC / PolyPIC
# c0=C[0,0], c1=C[0,1], c2=C[0,2]
# c3=C[1,0], c4=C[1,1], c5=C[1,2]
# c6=C[2,0], c7=C[2,1], c8=C[2,2]
c0 = ti.field(ti.f32, shape=MAX_PARTICLES)
c1 = ti.field(ti.f32, shape=MAX_PARTICLES)
c2 = ti.field(ti.f32, shape=MAX_PARTICLES)
c3 = ti.field(ti.f32, shape=MAX_PARTICLES)
c4 = ti.field(ti.f32, shape=MAX_PARTICLES)
c5 = ti.field(ti.f32, shape=MAX_PARTICLES)
c6 = ti.field(ti.f32, shape=MAX_PARTICLES)
c7 = ti.field(ti.f32, shape=MAX_PARTICLES)
c8 = ti.field(ti.f32, shape=MAX_PARTICLES)

num_particles = ti.field(ti.i32, shape=())
flip_ratio    = ti.field(ti.f32, shape=())

# Render buffers (GPU-side, avoids numpy round-trip for pixels)
zbuf_i   = ti.field(ti.i32, shape=(IMG_H, IMG_W))  # depth (scaled int, for atomic_min)
render_r = ti.field(ti.u8,  shape=(IMG_H, IMG_W))
render_g = ti.field(ti.u8,  shape=(IMG_H, IMG_W))
render_b = ti.field(ti.u8,  shape=(IMG_H, IMG_W))


# =============================================================================
# Helper kernels -- cell marking
# =============================================================================

@ti.kernel
def mark_domain_boundaries(obstacle_mask: ti.types.ndarray()):
    """Mark domain edges and obstacle cells as SOLID; everything else AIR."""
    for i, j, k in ti.ndrange(NX, NY, NZ):
        obs = ti.cast(obstacle_mask[j, i, k], ti.i32)
        if obs == 1:
            cell_type[i, j, k] = SOLID
        elif i == 0 or i == NX - 1 or j == 0 or j == NY - 1 or k == 0 or k == NZ - 1:
            cell_type[i, j, k] = SOLID
        else:
            cell_type[i, j, k] = AIR


@ti.kernel
def mark_fluid_cells():
    """Reset to AIR, then mark cells containing particles as FLUID."""
    for i, j, k in ti.ndrange(NX, NY, NZ):
        if cell_type[i, j, k] != SOLID:
            cell_type[i, j, k] = AIR
    for p in range(num_particles[None]):
        xi = ti.cast(px[p] * INV_DX, ti.i32)
        yi = ti.cast(py[p] * INV_DX, ti.i32)
        zi = ti.cast(pz[p] * INV_DX, ti.i32)
        if 0 <= xi < NX and 0 <= yi < NY and 0 <= zi < NZ:
            if cell_type[xi, yi, zi] != SOLID:
                cell_type[xi, yi, zi] = FLUID


# =============================================================================
# Helper kernels -- divergence
# =============================================================================

@ti.kernel
def compute_divergence():
    for i, j, k in ti.ndrange(NX, NY, NZ):
        if cell_type[i, j, k] == FLUID:
            div_field[i, j, k] = (
                u[i + 1, j,     k    ] - u[i, j, k] +
                v[i,     j + 1, k    ] - v[i, j, k] +
                w[i,     j,     k + 1] - w[i, j, k]
            ) / DX
        else:
            div_field[i, j, k] = 0.0


# =============================================================================
# Helper kernels -- pressure solve (Conjugate Gradient, 3D)
# =============================================================================

@ti.kernel
def compute_Ap(p_fld: ti.template(), Ap_out: ti.template()):
    """3D NEGATIVE-LAPLACIAN operator with 6-neighbor stencil.

    We solve  (-∇²) p = -f  (positive-definite system) so that the standard
    CG assumption <x, Ax> > 0 holds.  The sign flip is undone in
    solve_pressure_cg by also negating the RHS.

    BCs:
      Neumann at SOLID  →  ghost-cell value = p[self]  (zero normal gradient)
      Dirichlet at AIR  →  p[neighbor] = 0  (free surface)
    """
    for i, j, k in ti.ndrange(NX, NY, NZ):
        if cell_type[i, j, k] != FLUID:
            Ap_out[i, j, k] = 0.0
            continue
        # Positive-definite stencil:  (n_active * p_self - sum_active_neighbors)
        lap = 0.0
        diag = 0.0
        for di, dj, dk in ti.static([
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]):
            ni = i + di; nj = j + dj; nk = k + dk
            in_bounds = (0 <= ni < NX) and (0 <= nj < NY) and (0 <= nk < NZ)
            nb = cell_type[ni, nj, nk] if in_bounds else SOLID
            if nb == FLUID:
                lap   -= p_fld[ni, nj, nk]
                diag  += 1.0
            elif nb == SOLID:
                # Neumann: p_ghost = p_self → cancels with diagonal → omit
                pass
            # AIR: p = 0 (Dirichlet), contributes only to diagonal
            else:
                diag += 1.0
        Ap_out[i, j, k] = (diag * p_fld[i, j, k] + lap) / (DX * DX)


@ti.kernel
def vector_add_scaled(x: ti.template(), y: ti.template(), scale: ti.f32):
    """x += scale * y  (fluid cells only)."""
    for i, j, k in ti.ndrange(NX, NY, NZ):
        if cell_type[i, j, k] == FLUID:
            x[i, j, k] += scale * y[i, j, k]


@ti.kernel
def vector_combine_scaled(x: ti.template(), a: ti.template(), b: ti.template(),
                           sa: ti.f32, sb: ti.f32):
    """x = sa*a + sb*b  (fluid cells only)."""
    for i, j, k in ti.ndrange(NX, NY, NZ):
        if cell_type[i, j, k] == FLUID:
            x[i, j, k] = sa * a[i, j, k] + sb * b[i, j, k]


@ti.kernel
def pressure_init():
    for i, j, k in ti.ndrange(NX, NY, NZ):
        pressure[i, j, k] = 0.0


@ti.kernel
def scalar_dot(a: ti.template(), b: ti.template()) -> ti.f32:
    result = 0.0
    for i, j, k in ti.ndrange(NX, NY, NZ):
        if cell_type[i, j, k] == FLUID:
            result += a[i, j, k] * b[i, j, k]
    return result


@ti.kernel
def apply_pressure_gradient(dt: ti.f32):
    """Subtract pressure gradient from MAC grid velocities."""
    inv_dx = 1.0 / DX
    for i, j, k in ti.ndrange((1, NX), NY, NZ):
        if cell_type[i, j, k] == FLUID or cell_type[i - 1, j, k] == FLUID:
            u[i, j, k] -= dt * (pressure[i, j, k] - pressure[i - 1, j, k]) * inv_dx
    for i, j, k in ti.ndrange(NX, (1, NY), NZ):
        if cell_type[i, j, k] == FLUID or cell_type[i, j - 1, k] == FLUID:
            v[i, j, k] -= dt * (pressure[i, j, k] - pressure[i, j - 1, k]) * inv_dx
    for i, j, k in ti.ndrange(NX, NY, (1, NZ)):
        if cell_type[i, j, k] == FLUID or cell_type[i, j, k - 1] == FLUID:
            w[i, j, k] -= dt * (pressure[i, j, k] - pressure[i, j, k - 1]) * inv_dx


def solve_pressure_cg() -> int:
    """CG solver for the positive-definite system (-nabla^2) p = -(nabla.u)/dt.

    We negate both sides so the operator is positive-definite (eigenvalues > 0),
    which is required by the standard conjugate-gradient algorithm.
    The velocity update in apply_pressure_gradient is unchanged because both
    p and -(-p) are the same value.

    Convergence: relative residual  sqrt(rsnew/rsold) < CG_TOL.
    """
    # RHS: -∇·u / dt   (negated to match positive-definite (-∇²) operator)
    # Cold-start: reset pressure to 0 so r = b = -div/DT.
    # Warm-starting with the previous frame's pressure is counter-productive for
    # rapidly-evolving free surfaces (dam break): the stale pressure can produce
    # a residual much larger than the cold-start residual, causing many more
    # CG iterations rather than fewer.
    vector_combine_scaled(cg_r, div_field, div_field, -1.0 / DT, 0.0)
    pressure_init()   # reset pressure field to 0
    vector_combine_scaled(cg_p, cg_r, cg_r, 1.0, 0.0)

    rsold = scalar_dot(cg_r, cg_r)
    if rsold < 1e-20:
        return 0

    tol_sq = CG_TOL * CG_TOL * rsold   # relative convergence threshold

    for k in range(CG_MAX_ITER):
        compute_Ap(cg_p, cg_Ap)
        pAp = scalar_dot(cg_p, cg_Ap)
        if pAp < 1e-30:
            break
        alpha = rsold / pAp
        vector_add_scaled(pressure, cg_p,  alpha)
        vector_add_scaled(cg_r,     cg_Ap, -alpha)
        rsnew = scalar_dot(cg_r, cg_r)
        if rsnew < tol_sq:
            return k + 1
        beta = rsnew / rsold
        vector_combine_scaled(cg_p, cg_r, cg_p, 1.0, beta)
        rsold = rsnew
    return CG_MAX_ITER


# =============================================================================
# Helper kernels -- velocity operations
# =============================================================================

@ti.kernel
def add_gravity():
    """Apply gravitational acceleration to y-velocity component."""
    for i, j, k in ti.ndrange(NX, (1, NY), NZ):
        v[i, j, k] += GRAVITY_Y * DT


@ti.kernel
def save_velocities():
    """Snapshot current grid velocities (needed for FLIP delta)."""
    for i, j, k in u_saved:
        u_saved[i, j, k] = u[i, j, k]
    for i, j, k in v_saved:
        v_saved[i, j, k] = v[i, j, k]
    for i, j, k in w_saved:
        w_saved[i, j, k] = w[i, j, k]


@ti.kernel
def enforce_boundary_velocity():
    """Free-slip BC: zero normal velocity at all solid surfaces."""
    # Domain walls
    for j, k in ti.ndrange(NY, NZ):
        u[0,  j, k] = 0.0
        u[NX, j, k] = 0.0
    for i, k in ti.ndrange(NX, NZ):
        v[i, 0,  k] = 0.0
        v[i, NY, k] = 0.0
    for i, j in ti.ndrange(NX, NY):
        w[i, j, 0 ] = 0.0
        w[i, j, NZ] = 0.0
    # Interior solid surfaces
    for i, j, k in ti.ndrange((1, NX), NY, NZ):
        if cell_type[i - 1, j, k] == SOLID or cell_type[i, j, k] == SOLID:
            u[i, j, k] = 0.0
    for i, j, k in ti.ndrange(NX, (1, NY), NZ):
        if cell_type[i, j - 1, k] == SOLID or cell_type[i, j, k] == SOLID:
            v[i, j, k] = 0.0
    for i, j, k in ti.ndrange(NX, NY, (1, NZ)):
        if cell_type[i, j, k - 1] == SOLID or cell_type[i, j, k] == SOLID:
            w[i, j, k] = 0.0


# =============================================================================
# Particle -> Grid (P2G) -- 3D trilinear
# =============================================================================

@ti.kernel
def p2g_trilinear():
    """Scatter particle velocities onto MAC grid using trilinear weights.
    Uses u_saved/v_saved/w_saved as temporary weight accumulators."""
    # Reset grid + weight buffers
    for i, j, k in u:       u[i, j, k] = 0.0
    for i, j, k in v:       v[i, j, k] = 0.0
    for i, j, k in w:       w[i, j, k] = 0.0
    for i, j, k in u_saved: u_saved[i, j, k] = 0.0
    for i, j, k in v_saved: v_saved[i, j, k] = 0.0
    for i, j, k in w_saved: w_saved[i, j, k] = 0.0

    for p in range(num_particles[None]):
        xp = px[p] * INV_DX
        yp = py[p] * INV_DX
        zp = pz[p] * INV_DX

        # --- u-field: staggered at (i, j+0.5, k+0.5) ---
        iu = ti.cast(xp,       ti.i32)
        ju = ti.cast(yp - 0.5, ti.i32)
        ku = ti.cast(zp - 0.5, ti.i32)
        fxu = xp       - ti.cast(iu, ti.f32)
        fyu = yp - 0.5 - ti.cast(ju, ti.f32)
        fzu = zp - 0.5 - ti.cast(ku, ti.f32)
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxu if di == 1 else (1.0 - fxu)
                    wy = fyu if dj == 1 else (1.0 - fyu)
                    wz = fzu if dk == 1 else (1.0 - fzu)
                    wt = wx * wy * wz
                    ni = iu + di; nj = ju + dj; nk = ku + dk
                    if 0 <= ni <= NX and 0 <= nj < NY and 0 <= nk < NZ:
                        u[ni, nj, nk]       += wt * pu[p]
                        u_saved[ni, nj, nk] += wt

        # --- v-field: staggered at (i+0.5, j, k+0.5) ---
        iv = ti.cast(xp - 0.5, ti.i32)
        jv = ti.cast(yp,       ti.i32)
        kv = ti.cast(zp - 0.5, ti.i32)
        fxv = xp - 0.5 - ti.cast(iv, ti.f32)
        fyv = yp       - ti.cast(jv, ti.f32)
        fzv = zp - 0.5 - ti.cast(kv, ti.f32)
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxv if di == 1 else (1.0 - fxv)
                    wy = fyv if dj == 1 else (1.0 - fyv)
                    wz = fzv if dk == 1 else (1.0 - fzv)
                    wt = wx * wy * wz
                    ni = iv + di; nj = jv + dj; nk = kv + dk
                    if 0 <= ni < NX and 0 <= nj <= NY and 0 <= nk < NZ:
                        v[ni, nj, nk]       += wt * pv[p]
                        v_saved[ni, nj, nk] += wt

        # --- w-field: staggered at (i+0.5, j+0.5, k) ---
        iw = ti.cast(xp - 0.5, ti.i32)
        jw = ti.cast(yp - 0.5, ti.i32)
        kw = ti.cast(zp,       ti.i32)
        fxw = xp - 0.5 - ti.cast(iw, ti.f32)
        fyw = yp - 0.5 - ti.cast(jw, ti.f32)
        fzw = zp       - ti.cast(kw, ti.f32)
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxw if di == 1 else (1.0 - fxw)
                    wy = fyw if dj == 1 else (1.0 - fyw)
                    wz = fzw if dk == 1 else (1.0 - fzw)
                    wt = wx * wy * wz
                    ni = iw + di; nj = jw + dj; nk = kw + dk
                    if 0 <= ni < NX and 0 <= nj < NY and 0 <= nk <= NZ:
                        w[ni, nj, nk]       += wt * pw[p]
                        w_saved[ni, nj, nk] += wt

    # Normalize by accumulated weights
    for i, j, k in u:
        if u_saved[i, j, k] > 1e-8:
            u[i, j, k] /= u_saved[i, j, k]
    for i, j, k in v:
        if v_saved[i, j, k] > 1e-8:
            v[i, j, k] /= v_saved[i, j, k]
    for i, j, k in w:
        if w_saved[i, j, k] > 1e-8:
            w[i, j, k] /= w_saved[i, j, k]


# =============================================================================
# Grid -> Particles (G2P) -- 3D PIC-FLIP blend
# =============================================================================

@ti.kernel
def g2p_flip():
    """Interpolate grid velocity changes back to particles (PIC-FLIP blend).
    PIC:  vel_p = interp(grid_new)
    FLIP: vel_p = vel_p + interp(grid_new - grid_old)
    """
    alpha = flip_ratio[None]

    for p in range(num_particles[None]):
        xp = px[p] * INV_DX
        yp = py[p] * INV_DX
        zp = pz[p] * INV_DX

        # --- Interpolate u (new and old) ---
        iu = ti.cast(xp,       ti.i32)
        ju = ti.cast(yp - 0.5, ti.i32)
        ku = ti.cast(zp - 0.5, ti.i32)
        fxu = xp       - ti.cast(iu, ti.f32)
        fyu = yp - 0.5 - ti.cast(ju, ti.f32)
        fzu = zp - 0.5 - ti.cast(ku, ti.f32)
        u_new = 0.0; u_old = 0.0
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxu if di == 1 else (1.0 - fxu)
                    wy = fyu if dj == 1 else (1.0 - fyu)
                    wz = fzu if dk == 1 else (1.0 - fzu)
                    wt = wx * wy * wz
                    ni = iu + di; nj = ju + dj; nk = ku + dk
                    if 0 <= ni <= NX and 0 <= nj < NY and 0 <= nk < NZ:
                        u_new += wt * u[ni, nj, nk]
                        u_old += wt * u_saved[ni, nj, nk]

        # --- Interpolate v ---
        iv = ti.cast(xp - 0.5, ti.i32)
        jv = ti.cast(yp,       ti.i32)
        kv = ti.cast(zp - 0.5, ti.i32)
        fxv = xp - 0.5 - ti.cast(iv, ti.f32)
        fyv = yp       - ti.cast(jv, ti.f32)
        fzv = zp - 0.5 - ti.cast(kv, ti.f32)
        v_new = 0.0; v_old = 0.0
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxv if di == 1 else (1.0 - fxv)
                    wy = fyv if dj == 1 else (1.0 - fyv)
                    wz = fzv if dk == 1 else (1.0 - fzv)
                    wt = wx * wy * wz
                    ni = iv + di; nj = jv + dj; nk = kv + dk
                    if 0 <= ni < NX and 0 <= nj <= NY and 0 <= nk < NZ:
                        v_new += wt * v[ni, nj, nk]
                        v_old += wt * v_saved[ni, nj, nk]

        # --- Interpolate w ---
        iw = ti.cast(xp - 0.5, ti.i32)
        jw = ti.cast(yp - 0.5, ti.i32)
        kw = ti.cast(zp,       ti.i32)
        fxw = xp - 0.5 - ti.cast(iw, ti.f32)
        fyw = yp - 0.5 - ti.cast(jw, ti.f32)
        fzw = zp       - ti.cast(kw, ti.f32)
        w_new = 0.0; w_old = 0.0
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxw if di == 1 else (1.0 - fxw)
                    wy = fyw if dj == 1 else (1.0 - fyw)
                    wz = fzw if dk == 1 else (1.0 - fzw)
                    wt = wx * wy * wz
                    ni = iw + di; nj = jw + dj; nk = kw + dk
                    if 0 <= ni < NX and 0 <= nj < NY and 0 <= nk <= NZ:
                        w_new += wt * w[ni, nj, nk]
                        w_old += wt * w_saved[ni, nj, nk]

        # PIC-FLIP blend
        pu[p] = (1.0 - alpha) * u_new + alpha * (pu[p] + (u_new - u_old))
        pv[p] = (1.0 - alpha) * v_new + alpha * (pv[p] + (v_new - v_old))
        pw[p] = (1.0 - alpha) * w_new + alpha * (pw[p] + (w_new - w_old))

        # Clear affine matrix (FLIP does not use it; APIC/PolyPIC will set these)
        c0[p] = 0.0; c1[p] = 0.0; c2[p] = 0.0
        c3[p] = 0.0; c4[p] = 0.0; c5[p] = 0.0
        c6[p] = 0.0; c7[p] = 0.0; c8[p] = 0.0


@ti.kernel
def p2g_apic():
    """Scatter particle velocity + affine momentum to MAC grid using trilinear weights."""
    for i, j, k in u:       u[i, j, k] = 0.0
    for i, j, k in v:       v[i, j, k] = 0.0
    for i, j, k in w:       w[i, j, k] = 0.0
    for i, j, k in u_saved: u_saved[i, j, k] = 0.0
    for i, j, k in v_saved: v_saved[i, j, k] = 0.0
    for i, j, k in w_saved: w_saved[i, j, k] = 0.0

    for p in range(num_particles[None]):
        xp = px[p] * INV_DX
        yp = py[p] * INV_DX
        zp = pz[p] * INV_DX
        C = ti.Matrix([[c0[p], c1[p], c2[p]],
                       [c3[p], c4[p], c5[p]],
                       [c6[p], c7[p], c8[p]]])
        vel_p = ti.Vector([pu[p], pv[p], pw[p]])

        # --- u-field: staggered at (i, j+0.5, k+0.5) ---
        iu = ti.cast(xp, ti.i32)
        ju = ti.cast(yp - 0.5, ti.i32)
        ku = ti.cast(zp - 0.5, ti.i32)
        fxu = xp - ti.cast(iu, ti.f32)
        fyu = yp - 0.5 - ti.cast(ju, ti.f32)
        fzu = zp - 0.5 - ti.cast(ku, ti.f32)
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxu if di == 1 else (1.0 - fxu)
                    wy = fyu if dj == 1 else (1.0 - fyu)
                    wz = fzu if dk == 1 else (1.0 - fzu)
                    wt = wx * wy * wz
                    ni = iu + di; nj = ju + dj; nk = ku + dk
                    if 0 <= ni <= NX and 0 <= nj < NY and 0 <= nk < NZ:
                        dpos = ti.Vector([
                            (ti.cast(ni, ti.f32) - xp) * DX,
                            (ti.cast(nj, ti.f32) + 0.5 - yp) * DX,
                            (ti.cast(nk, ti.f32) + 0.5 - zp) * DX,
                        ])
                        g_v = vel_p + C @ dpos
                        u[ni, nj, nk]       += wt * g_v.x
                        u_saved[ni, nj, nk] += wt

        # --- v-field: staggered at (i+0.5, j, k+0.5) ---
        iv = ti.cast(xp - 0.5, ti.i32)
        jv = ti.cast(yp, ti.i32)
        kv = ti.cast(zp - 0.5, ti.i32)
        fxv = xp - 0.5 - ti.cast(iv, ti.f32)
        fyv = yp - ti.cast(jv, ti.f32)
        fzv = zp - 0.5 - ti.cast(kv, ti.f32)
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxv if di == 1 else (1.0 - fxv)
                    wy = fyv if dj == 1 else (1.0 - fyv)
                    wz = fzv if dk == 1 else (1.0 - fzv)
                    wt = wx * wy * wz
                    ni = iv + di; nj = jv + dj; nk = kv + dk
                    if 0 <= ni < NX and 0 <= nj <= NY and 0 <= nk < NZ:
                        dpos = ti.Vector([
                            (ti.cast(ni, ti.f32) + 0.5 - xp) * DX,
                            (ti.cast(nj, ti.f32) - yp) * DX,
                            (ti.cast(nk, ti.f32) + 0.5 - zp) * DX,
                        ])
                        g_v = vel_p + C @ dpos
                        v[ni, nj, nk]       += wt * g_v.y
                        v_saved[ni, nj, nk] += wt

        # --- w-field: staggered at (i+0.5, j+0.5, k) ---
        iw = ti.cast(xp - 0.5, ti.i32)
        jw = ti.cast(yp - 0.5, ti.i32)
        kw = ti.cast(zp, ti.i32)
        fxw = xp - 0.5 - ti.cast(iw, ti.f32)
        fyw = yp - 0.5 - ti.cast(jw, ti.f32)
        fzw = zp - ti.cast(kw, ti.f32)
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxw if di == 1 else (1.0 - fxw)
                    wy = fyw if dj == 1 else (1.0 - fyw)
                    wz = fzw if dk == 1 else (1.0 - fzw)
                    wt = wx * wy * wz
                    ni = iw + di; nj = jw + dj; nk = kw + dk
                    if 0 <= ni < NX and 0 <= nj < NY and 0 <= nk <= NZ:
                        dpos = ti.Vector([
                            (ti.cast(ni, ti.f32) + 0.5 - xp) * DX,
                            (ti.cast(nj, ti.f32) + 0.5 - yp) * DX,
                            (ti.cast(nk, ti.f32) - zp) * DX,
                        ])
                        g_v = vel_p + C @ dpos
                        w[ni, nj, nk]       += wt * g_v.z
                        w_saved[ni, nj, nk] += wt

    for i, j, k in u:
        if u_saved[i, j, k] > 1e-8:
            u[i, j, k] /= u_saved[i, j, k]
    for i, j, k in v:
        if v_saved[i, j, k] > 1e-8:
            v[i, j, k] /= v_saved[i, j, k]
    for i, j, k in w:
        if w_saved[i, j, k] > 1e-8:
            w[i, j, k] /= w_saved[i, j, k]


@ti.kernel
def g2p_apic():
    """Interpolate MAC grid velocities to particles and update affine matrices."""
    for p in range(num_particles[None]):
        xp = px[p] * INV_DX
        yp = py[p] * INV_DX
        zp = pz[p] * INV_DX

        # --- Interpolate u ---
        iu = ti.cast(xp, ti.i32)
        ju = ti.cast(yp - 0.5, ti.i32)
        ku = ti.cast(zp - 0.5, ti.i32)
        fxu = xp - ti.cast(iu, ti.f32)
        fyu = yp - 0.5 - ti.cast(ju, ti.f32)
        fzu = zp - 0.5 - ti.cast(ku, ti.f32)
        u_new = 0.0
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxu if di == 1 else (1.0 - fxu)
                    wy = fyu if dj == 1 else (1.0 - fyu)
                    wz = fzu if dk == 1 else (1.0 - fzu)
                    wt = wx * wy * wz
                    ni = iu + di; nj = ju + dj; nk = ku + dk
                    if 0 <= ni <= NX and 0 <= nj < NY and 0 <= nk < NZ:
                        u_new += wt * u[ni, nj, nk]

        # --- Interpolate v ---
        iv = ti.cast(xp - 0.5, ti.i32)
        jv = ti.cast(yp, ti.i32)
        kv = ti.cast(zp - 0.5, ti.i32)
        fxv = xp - 0.5 - ti.cast(iv, ti.f32)
        fyv = yp - ti.cast(jv, ti.f32)
        fzv = zp - 0.5 - ti.cast(kv, ti.f32)
        v_new = 0.0
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxv if di == 1 else (1.0 - fxv)
                    wy = fyv if dj == 1 else (1.0 - fyv)
                    wz = fzv if dk == 1 else (1.0 - fzv)
                    wt = wx * wy * wz
                    ni = iv + di; nj = jv + dj; nk = kv + dk
                    if 0 <= ni < NX and 0 <= nj <= NY and 0 <= nk < NZ:
                        v_new += wt * v[ni, nj, nk]

        # --- Interpolate w ---
        iw = ti.cast(xp - 0.5, ti.i32)
        jw = ti.cast(yp - 0.5, ti.i32)
        kw = ti.cast(zp, ti.i32)
        fxw = xp - 0.5 - ti.cast(iw, ti.f32)
        fyw = yp - 0.5 - ti.cast(jw, ti.f32)
        fzw = zp - ti.cast(kw, ti.f32)
        w_new = 0.0
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxw if di == 1 else (1.0 - fxw)
                    wy = fyw if dj == 1 else (1.0 - fyw)
                    wz = fzw if dk == 1 else (1.0 - fzw)
                    wt = wx * wy * wz
                    ni = iw + di; nj = jw + dj; nk = kw + dk
                    if 0 <= ni < NX and 0 <= nj < NY and 0 <= nk <= NZ:
                        w_new += wt * w[ni, nj, nk]

        new_v = ti.Vector([u_new, v_new, w_new])
        new_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        # u-node contributions: only x velocity available at u locations
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxu if di == 1 else (1.0 - fxu)
                    wy = fyu if dj == 1 else (1.0 - fyu)
                    wz = fzu if dk == 1 else (1.0 - fzu)
                    wt = wx * wy * wz
                    ni = iu + di; nj = ju + dj; nk = ku + dk
                    if 0 <= ni <= NX and 0 <= nj < NY and 0 <= nk < NZ:
                        dpos = ti.Vector([
                            (ti.cast(ni, ti.f32) - xp) * DX,
                            (ti.cast(nj, ti.f32) + 0.5 - yp) * DX,
                            (ti.cast(nk, ti.f32) + 0.5 - zp) * DX,
                        ])
                        g_v = ti.Vector([u[ni, nj, nk], 0.0, 0.0])
                        new_C += 4.0 * INV_DX * INV_DX * wt * g_v.outer_product(dpos)
        # v-node contributions: only y velocity at v locations
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxv if di == 1 else (1.0 - fxv)
                    wy = fyv if dj == 1 else (1.0 - fyv)
                    wz = fzv if dk == 1 else (1.0 - fzv)
                    wt = wx * wy * wz
                    ni = iv + di; nj = jv + dj; nk = kv + dk
                    if 0 <= ni < NX and 0 <= nj <= NY and 0 <= nk < NZ:
                        dpos = ti.Vector([
                            (ti.cast(ni, ti.f32) + 0.5 - xp) * DX,
                            (ti.cast(nj, ti.f32) - yp) * DX,
                            (ti.cast(nk, ti.f32) + 0.5 - zp) * DX,
                        ])
                        g_v = ti.Vector([0.0, v[ni, nj, nk], 0.0])
                        new_C += 4.0 * INV_DX * INV_DX * wt * g_v.outer_product(dpos)
        # w-node contributions: only z velocity at w locations
        for di in ti.static(range(2)):
            for dj in ti.static(range(2)):
                for dk in ti.static(range(2)):
                    wx = fxw if di == 1 else (1.0 - fxw)
                    wy = fyw if dj == 1 else (1.0 - fyw)
                    wz = fzw if dk == 1 else (1.0 - fzw)
                    wt = wx * wy * wz
                    ni = iw + di; nj = jw + dj; nk = kw + dk
                    if 0 <= ni < NX and 0 <= nj < NY and 0 <= nk <= NZ:
                        dpos = ti.Vector([
                            (ti.cast(ni, ti.f32) + 0.5 - xp) * DX,
                            (ti.cast(nj, ti.f32) + 0.5 - yp) * DX,
                            (ti.cast(nk, ti.f32) - zp) * DX,
                        ])
                        g_v = ti.Vector([0.0, 0.0, w[ni, nj, nk]])
                        new_C += 4.0 * INV_DX * INV_DX * wt * g_v.outer_product(dpos)

        pu[p] = new_v.x; pv[p] = new_v.y; pw[p] = new_v.z
        c0[p] = new_C[0, 0]; c1[p] = new_C[0, 1]; c2[p] = new_C[0, 2]
        c3[p] = new_C[1, 0]; c4[p] = new_C[1, 1]; c5[p] = new_C[1, 2]
        c6[p] = new_C[2, 0]; c7[p] = new_C[2, 1]; c8[p] = new_C[2, 2]


# =============================================================================
# Particle advection -- 3D forward Euler
# =============================================================================

@ti.kernel
def advect_particles():
    """Move particles: pos += vel * dt.  Clamp velocity (CFL safety) then position."""
    dom_x = ti.cast(NX, ti.f32) * DX - 1e-5
    dom_y = ti.cast(NY, ti.f32) * DX - 1e-5
    dom_z = ti.cast(NZ, ti.f32) * DX - 1e-5
    v_lim = ti.cast(V_MAX_PHYS, ti.f32)
    for p in range(num_particles[None]):
        # CFL velocity clamp: prevent particles from skipping multiple cells
        spd = ti.sqrt(pu[p]*pu[p] + pv[p]*pv[p] + pw[p]*pw[p])
        if spd > v_lim:
            scale = v_lim / spd
            pu[p] *= scale;  pv[p] *= scale;  pw[p] *= scale
        px[p] += pu[p] * DT
        py[p] += pv[p] * DT
        pz[p] += pw[p] * DT
        if px[p] < 1e-5:   px[p] = 1e-5;   pu[p] = 0.0
        if px[p] > dom_x:  px[p] = dom_x;  pu[p] = 0.0
        if py[p] < 1e-5:   py[p] = 1e-5;   pv[p] = 0.0
        if py[p] > dom_y:  py[p] = dom_y;  pv[p] = 0.0
        if pz[p] < 1e-5:   pz[p] = 1e-5;   pw[p] = 0.0
        if pz[p] > dom_z:  pz[p] = dom_z;  pw[p] = 0.0


# Scratch buffer for stream-compaction (ghost-particle removal)
_alive = ti.field(ti.i32, shape=MAX_PARTICLES)  # 1 = keep, 0 = remove
_prefix = ti.field(ti.i32, shape=MAX_PARTICLES)


@ti.kernel
def _mark_alive_kernel(max_height: ti.f32) -> ti.i32:
    """Mark particles above max_height that are in AIR cells as dead (0).
    Returns number of particles marked dead."""
    dead = 0
    for p in range(num_particles[None]):
        xi = ti.cast(px[p] * INV_DX, ti.i32)
        yi = ti.cast(py[p] * INV_DX, ti.i32)
        zi = ti.cast(pz[p] * INV_DX, ti.i32)
        in_bounds = (0 <= xi < NX and 0 <= yi < NY and 0 <= zi < NZ)
        is_air = in_bounds and (cell_type[xi, yi, zi] == AIR)
        too_high = py[p] > max_height
        if is_air and too_high:
            _alive[p] = 0
            dead += 1
        else:
            _alive[p] = 1
    return dead


@ti.kernel
def _compact_particles_kernel(new_n: ti.i32):
    """Write surviving particles into a contiguous block (prefix-sum compaction)."""
    for p in range(new_n):
        if _alive[p] == 1:
            dst = _prefix[p]
            px[dst] = px[p]; py[dst] = py[p]; pz[dst] = pz[p]
            pu[dst] = pu[p]; pv[dst] = pv[p]; pw[dst] = pw[p]
            c0[dst] = c0[p]; c1[dst] = c1[p]; c2[dst] = c2[p]
            c3[dst] = c3[p]; c4[dst] = c4[p]; c5[dst] = c5[p]
            c6[dst] = c6[p]; c7[dst] = c7[p]; c8[dst] = c8[p]


def remove_ghost_particles(max_height_frac: float = 0.55) -> int:
    """Remove airborne particles above max_height_frac * domain_height.

    Particles floating in AIR cells above this threshold are ghost particles
    created by pressure overshoots at the free surface.  Removing them after
    the initial splash settles prevents the 'floating slab' artefact.

    Returns the number of particles removed.
    """
    max_h = float(NY) * DX * max_height_frac
    n = num_particles[None]
    dead = _mark_alive_kernel(max_h)
    if dead == 0:
        return 0
    # CPU prefix-sum (n is at most a few million, this is fast enough)
    alive_np = _alive.to_numpy()[:n]
    prefix_np = np.cumsum(alive_np) - 1  # 0-indexed write positions
    _prefix.from_numpy(np.pad(prefix_np.astype(np.int32),
                               (0, MAX_PARTICLES - n)))
    new_n = int(alive_np.sum())
    _compact_particles_kernel(n)
    num_particles[None] = new_n
    return dead


# =============================================================================
# fluid_step() -- THE INTERFACE TEAMMATES IMPLEMENT
# =============================================================================

def fluid_step():
    """Execute one sub-step of the 3D FLIP fluid simulation.

    THIS FUNCTION IS THE ONLY PLACE TEAMMATES SHOULD MODIFY.

    Available global Taichi fields:
        u[NX+1, NY,   NZ  ]  : x-velocity at x-MAC-faces
        v[NX,   NY+1, NZ  ]  : y-velocity at y-MAC-faces
        w[NX,   NY,   NZ+1]  : z-velocity at z-MAC-faces
        u_saved, v_saved, w_saved : grid-velocity snapshots (FLIP delta)
        cell_type[NX, NY, NZ] : FLUID=0, SOLID=1, AIR=2
        pressure [NX, NY, NZ] : pressure (CG output)
        px/py/pz/pu/pv/pw     : particle positions and velocities (MAX_PARTICLES)
        c0..c8                : 3x3 affine velocity matrix (APIC/PolyPIC)
        num_particles[None]   : active particle count
        flip_ratio[None]      : PIC-FLIP blend (1.0 = pure FLIP)
        div_field, cg_r, cg_p, cg_Ap : CG scratch fields

    Available helper functions:
        p2g_trilinear()              : 3D particle->grid trilinear transfer
        save_velocities()            : snapshot u,v,w -> u_saved,v_saved,w_saved
        add_gravity()                : v += GRAVITY_Y * DT
        enforce_boundary_velocity()  : zero normal velocity at solid walls
        compute_divergence()         : store nabla.u in div_field
        solve_pressure_cg() -> int   : solve nabla^2p = div/dt, return iterations
        apply_pressure_gradient(dt) : u,v,w -= dt * nablap
        g2p_flip()                  : 3D grid->particles PIC-FLIP blend

    Replace the body of this function for APIC (Student B) or PolyPIC (Student C).
    """

    # APIC P2G: scatter particle velocity + affine momentum to MAC grid.
    p2g_apic()

    # External forces
    add_gravity()

    # Boundary conditions
    enforce_boundary_velocity()

    # Pressure projection: nabla^2p = (nabla.u)/dt
    compute_divergence()
    solve_pressure_cg()
    apply_pressure_gradient(DT)

    # Re-enforce BC after pressure correction
    enforce_boundary_velocity()

    # APIC G2P: interpolate grid velocity back to particles and update C.
    g2p_apic()


# =============================================================================
# Rendering -- Taichi GPU kernel (perspective projection + integer z-buffer)
# =============================================================================

def _build_camera_basis():
    """Return (fwd, right, up) orthonormal camera basis vectors."""
    fwd = CAM_TARGET - CAM_POS
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, CAM_UP)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fwd, right, up

# Precompute focal length once (mutable globals – updated by _set_camera)
_FOCAL = float(IMG_W / (2.0 * math.tan(math.radians(CAM_FOV_DEG) / 2.0)))
_FWD, _RIGHT, _UP_CAM = _build_camera_basis()


def _set_camera(pos, target, up, fov_deg):
    """Update the global camera state used by export_frame."""
    global CAM_POS, CAM_TARGET, CAM_UP, CAM_FOV_DEG
    global _FOCAL, _FWD, _RIGHT, _UP_CAM
    CAM_POS    = np.asarray(pos,    dtype=np.float64)
    CAM_TARGET = np.asarray(target, dtype=np.float64)
    CAM_UP     = np.asarray(up,     dtype=np.float64)
    CAM_FOV_DEG = fov_deg
    _FOCAL = float(IMG_W / (2.0 * math.tan(math.radians(fov_deg) / 2.0)))
    _FWD, _RIGHT, _UP_CAM = _build_camera_basis()


@ti.kernel
def _render_env_kernel(
    fx: ti.f32, fy: ti.f32, fz: ti.f32,
    rx: ti.f32, ry: ti.f32, rz: ti.f32,
    ux: ti.f32, uy: ti.f32, uz: ti.f32,
    cx: ti.f32, cy: ti.f32, cz: ti.f32,
    focal: ti.f32,
):
    """GPU kernel: clear buffers, ray-cast floor, render obstacle cells.

    For the pinhole camera model used here, the camera-space depth of a point
    hit by ray r(t) = cam_pos + t*rd equals t (since dot(rd, fwd) == 1 for
    rd = fwd + x_ndc*right + y_ndc*up with orthonormal basis).
    """
    # Clear: light sky-blue background
    for r, c in ti.ndrange(IMG_H, IMG_W):
        zbuf_i[r, c]   = 2_000_000_000
        render_r[r, c] = ti.cast(215, ti.u8)
        render_g[r, c] = ti.cast(225, ti.u8)
        render_b[r, c] = ti.cast(235, ti.u8)

    # Ray-cast floor at y = 0 (checkerboard pattern)
    dom_x = ti.cast(NX, ti.f32) * DX
    dom_z = ti.cast(NZ, ti.f32) * DX
    for row, col in ti.ndrange(IMG_H, IMG_W):
        x_ndc = (ti.cast(col, ti.f32) - IMG_W * 0.5) / focal
        y_ndc = -(ti.cast(row, ti.f32) - IMG_H * 0.5) / focal
        rdx = fx + x_ndc * rx + y_ndc * ux
        rdy = fy + x_ndc * ry + y_ndc * uy
        rdz = fz + x_ndc * rz + y_ndc * uz
        if rdy < -1e-5:
            t_hit = -cy / rdy
            if 0.05 < t_hit < 60.0:
                hx = cx + t_hit * rdx
                hz = cz + t_hit * rdz
                if 0.0 <= hx <= dom_x and 0.0 <= hz <= dom_z:
                    z_int = ti.cast(t_hit * 10000.0, ti.i32)
                    old_z = ti.atomic_min(zbuf_i[row, col], z_int)
                    if z_int <= old_z:
                        tile_x = ti.cast(hx / dom_x * 8.0, ti.i32)
                        tile_z = ti.cast(hz / dom_z * 8.0, ti.i32)
                        if (tile_x + tile_z) % 2 == 0:
                            render_r[row, col] = ti.cast(185, ti.u8)
                            render_g[row, col] = ti.cast(185, ti.u8)
                            render_b[row, col] = ti.cast(180, ti.u8)
                        else:
                            render_r[row, col] = ti.cast(205, ti.u8)
                            render_g[row, col] = ti.cast(205, ti.u8)
                            render_b[row, col] = ti.cast(200, ti.u8)

    # Render interior obstacle cells (solid but not domain-boundary walls)
    for i, j, k in cell_type:
        is_bnd = (i == 0 or i == NX - 1 or
                  j == 0 or j == NY - 1 or
                  k == 0 or k == NZ - 1)
        if cell_type[i, j, k] == SOLID and not is_bnd:
            wx = (ti.cast(i, ti.f32) + 0.5) * DX
            wy = (ti.cast(j, ti.f32) + 0.5) * DX
            wz = (ti.cast(k, ti.f32) + 0.5) * DX
            ddx = wx - cx; ddy = wy - cy; ddz = wz - cz
            z_cam = ddx * fx + ddy * fy + ddz * fz
            if z_cam < 0.05: continue
            x_cam = ddx * rx + ddy * ry + ddz * rz
            y_cam = ddx * ux + ddy * uy + ddz * uz
            xi = ti.cast(focal * x_cam / z_cam + IMG_W * 0.5, ti.i32)
            yi = ti.cast(-focal * y_cam / z_cam + IMG_H * 0.5, ti.i32)
            z_int = ti.cast(z_cam * 10000.0, ti.i32)
            for dr in ti.static(range(-2, 3)):
                for dc in ti.static(range(-2, 3)):
                    nr = yi + dr; nc = xi + dc
                    if 0 <= nr < IMG_H and 0 <= nc < IMG_W:
                        old_z = ti.atomic_min(zbuf_i[nr, nc], z_int)
                        if z_int <= old_z:
                            render_r[nr, nc] = ti.cast(90, ti.u8)
                            render_g[nr, nc] = ti.cast(90, ti.u8)
                            render_b[nr, nc] = ti.cast(95, ti.u8)


@ti.kernel
def _render_particles_kernel(
    fx: ti.f32, fy: ti.f32, fz: ti.f32,
    rx: ti.f32, ry: ti.f32, rz: ti.f32,
    ux: ti.f32, uy: ti.f32, uz: ti.f32,
    cx: ti.f32, cy: ti.f32, cz: ti.f32,
    focal: ti.f32,
):
    """GPU-accelerated particle renderer with integer atomic z-buffer.

    Called after _render_env_kernel (buffers already cleared).
    Renders each particle as a 3x3 splat with depth-based shading.
    Race conditions on color writes are visually negligible for dense fluids.
    """
    for p in range(num_particles[None]):
        ddx = px[p] - cx;  ddy = py[p] - cy;  ddz = pz[p] - cz

        # Camera-space depth
        z_cam = ddx * fx + ddy * fy + ddz * fz
        if z_cam < 0.05:
            continue

        x_cam = ddx * rx + ddy * ry + ddz * rz
        y_cam = ddx * ux + ddy * uy + ddz * uz

        # Perspective projection
        xi = ti.cast(focal * x_cam / z_cam + IMG_W * 0.5, ti.i32)
        yi = ti.cast(-focal * y_cam / z_cam + IMG_H * 0.5, ti.i32)

        # Depth-based shading: close=bright blue, far=dark blue
        brt = ti.min(1.0, ti.max(0.35, 1.5 - z_cam * 0.45))
        rc = ti.cast(35.0  * brt, ti.u8)
        gc = ti.cast(100.0 * brt, ti.u8)
        bc = ti.cast(200.0 * brt, ti.u8)

        z_int = ti.cast(z_cam * 10000.0, ti.i32)

        # 3x3 splat with integer atomic z-test
        for dr in ti.static(range(-1, 2)):
            for dc in ti.static(range(-1, 2)):
                nr = yi + dr;  nc = xi + dc
                if 0 <= nr < IMG_H and 0 <= nc < IMG_W:
                    old_z = ti.atomic_min(zbuf_i[nr, nc], z_int)
                    if z_int <= old_z:
                        render_r[nr, nc] = rc
                        render_g[nr, nc] = gc
                        render_b[nr, nc] = bc


def export_frame(frame_dir: Path, frame_num: int):
    """Render current particle state to PNG using Taichi GPU kernels.

    Pipeline: env (clear + floor + obstacles) -> particles -> save PNG.
    """
    cam_args = (
        float(_FWD[0]),    float(_FWD[1]),    float(_FWD[2]),
        float(_RIGHT[0]),  float(_RIGHT[1]),  float(_RIGHT[2]),
        float(_UP_CAM[0]), float(_UP_CAM[1]), float(_UP_CAM[2]),
        float(CAM_POS[0]), float(CAM_POS[1]), float(CAM_POS[2]),
        float(_FOCAL),
    )
    _render_env_kernel(*cam_args)
    _render_particles_kernel(*cam_args)
    r_np = render_r.to_numpy()
    g_np = render_g.to_numpy()
    b_np = render_b.to_numpy()
    img_arr = np.stack([r_np, g_np, b_np], axis=-1)
    Image.fromarray(img_arr, mode="RGB").save(frame_dir / f"frame_{frame_num:04d}.png")


# =============================================================================
# Energy and vorticity
# =============================================================================

def compute_kinetic_energy() -> float:
    n = num_particles[None]
    pu_np = pu.to_numpy()[:n]
    pv_np = pv.to_numpy()[:n]
    pw_np = pw.to_numpy()[:n]
    return float(0.5 * MASS_PER_PARTICLE * np.sum(pu_np**2 + pv_np**2 + pw_np**2))


@ti.kernel
def compute_total_vorticity() -> ti.f32:
    """Sum |curl(v)| over fluid cells as numerical dissipation proxy."""
    total = 0.0
    for i, j, k in ti.ndrange((0, NX - 1), (0, NY - 1), (0, NZ - 1)):
        if cell_type[i, j, k] == FLUID:
            omx = (w[i, j + 1, k] - w[i, j, k] - v[i, j, k + 1] + v[i, j, k]) / DX
            omy = (u[i, j, k + 1] - u[i, j, k] - w[i + 1, j, k] + w[i, j, k]) / DX
            omz = (v[i + 1, j, k] - v[i, j, k] - u[i, j + 1, k] + u[i, j, k]) / DX
            total += ti.sqrt(omx * omx + omy * omy + omz * omz)
    return total


def write_csv_row(writer, frame_num, t, ek, frame_time_ms, vorticity):
    writer.writerow({
        "frame": frame_num,
        "time": round(t, 6),
        "kinetic_energy": round(ek, 8),
        "frame_time_ms": round(frame_time_ms, 3),
        "total_vorticity": round(vorticity, 6),
    })


# =============================================================================
# Scene setup -- 3D Dam Break
# =============================================================================

def setup_dam_break():
    """3D Dam Break: tall, narrow water column on the left side of the domain.

    Tall column (height >> width) so the collapse is dramatic: the column
    topples sideways rather than just draining from the base.

    Water block: x in [0, NX/6], y in [0, NY*3/4], z in [1, NZ-1]
    Full Z-width gives maximum visual impact from the camera.
    """
    x1 = max(1, NX // 6)          # narrow: ~13 cells (0.13 m)
    y1 = NY * 3 // 4              # tall:   ~75 cells (0.75 m)
    z0 = 1
    z1 = NZ - 1                   # full Z width

    nx = x1; ny = y1; nz = z1 - z0
    n_total = nx * ny * nz * PPC * PPC * PPC
    print(f"[setup] Dam Break 3D: block {nx}x{ny}x{nz} cells -> {n_total} particles")

    _init_block_particles(0, x1, 0, y1, z0, z1)

    obstacle = np.zeros((NY, NX, NZ), dtype=np.uint8)
    mark_domain_boundaries(obstacle)
    mark_fluid_cells()               # mark initial water column as FLUID
    u.fill(0.0); v.fill(0.0); w.fill(0.0)


# =============================================================================
# Scene setup -- 3D Liquid Pouring
# =============================================================================

def setup_liquid_pouring():
    """3D Liquid Pouring: stream falls from top, hits a rectangular obstacle.

    Obstacle: centered at (NX/2, NY/4, NZ/2).
    Fluid is emitted each frame from a narrow inflow region near the top.
    """
    print("[setup] Liquid Pouring 3D: inflow stream + obstacle")

    num_particles[None] = 0
    for f in (pu, pv, pw, c0, c1, c2, c3, c4, c5, c6, c7, c8):
        f.fill(0.0)

    obstacle = _make_obstacle_mask_3d()
    mark_domain_boundaries(obstacle)
    mark_fluid_cells()               # ensure FLUID cells are marked from frame 0
    u.fill(0.0); v.fill(0.0); w.fill(0.0)
    print(f"[setup] Obstacle cells: {obstacle.sum()}")


def _make_obstacle_mask_3d() -> np.ndarray:
    """Rectangular obstacle near lower-center of domain (Liquid Pouring)."""
    mask = np.zeros((NY, NX, NZ), dtype=np.uint8)
    cx, cy, cz = NX // 2, NY // 4, NZ // 2
    hx, hy, hz = NX // 8, NY // 10, NZ // 8
    x0, x1 = max(1, cx - hx), min(NX - 1, cx + hx)
    y0, y1 = max(1, cy - hy), min(NY - 1, cy + hy)
    z0, z1 = max(1, cz - hz), min(NZ - 1, cz + hz)
    mask[y0:y1, x0:x1, z0:z1] = 1
    return mask


# =============================================================================
# Particle initialization helper (flat Taichi kernel)
# =============================================================================

@ti.kernel
def _init_block_kernel(x0: ti.i32, x1: ti.i32,
                        y0: ti.i32, y1: ti.i32,
                        z0: ti.i32, z1: ti.i32,
                        ppc: ti.i32, dx: ti.f32):
    """Fill a rectangular block with uniformly spaced particles."""
    nx = x1 - x0; ny = y1 - y0; nz = z1 - z0
    ppc3 = ppc * ppc * ppc
    for flat in range(nx * ny * nz * ppc3):
        # Decode flat index -> (i, j, k, si, sj, sk)
        rem = flat
        sk = rem % ppc;      rem = rem // ppc
        sj = rem % ppc;      rem = rem // ppc
        si = rem % ppc;      rem = rem // ppc
        k  = rem % nz;       rem = rem // nz
        j  = rem % ny;       rem = rem // ny
        i  = rem

        ppc_f = ti.cast(ppc, ti.f32)
        pos_x = (ti.cast(x0 + i, ti.f32) + (ti.cast(si, ti.f32) + 0.5) / ppc_f) * dx
        pos_y = (ti.cast(y0 + j, ti.f32) + (ti.cast(sj, ti.f32) + 0.5) / ppc_f) * dx
        pos_z = (ti.cast(z0 + k, ti.f32) + (ti.cast(sk, ti.f32) + 0.5) / ppc_f) * dx
        px[flat] = pos_x
        py[flat] = pos_y
        pz[flat] = pos_z
        pu[flat] = 0.0; pv[flat] = 0.0; pw[flat] = 0.0
        c0[flat] = 0.0; c1[flat] = 0.0; c2[flat] = 0.0
        c3[flat] = 0.0; c4[flat] = 0.0; c5[flat] = 0.0
        c6[flat] = 0.0; c7[flat] = 0.0; c8[flat] = 0.0
    num_particles[None] = nx * ny * nz * ppc3


def _init_block_particles(x0, x1, y0, y1, z0, z1):
    n = (x1 - x0) * (y1 - y0) * (z1 - z0) * PPC**3
    assert n <= MAX_PARTICLES, f"Too many particles: {n} > {MAX_PARTICLES}"
    _init_block_kernel(x0, x1, y0, y1, z0, z1, PPC, DX)


# =============================================================================
# Inflow emission -- Liquid Pouring
# =============================================================================

INFLOW_START = 0
INFLOW_END   = 300          # emit for first 5 s

# Inflow window: narrower focused beam at top-center.
# Narrower XZ gives a denser, more coherent stream.
_IFX0 = NX // 2 - NX // 16
_IFX1 = NX // 2 + NX // 16
_IFZ0 = NZ // 2 - NZ // 16
_IFZ1 = NZ // 2 + NZ // 16
_IFY0 = NY - 10             # near top wall (physical top)
_IFY1 = NY - 3

# Dense inflow stream: 2000/frame × 600 frames = 1.2 M total < MAX_PARTICLES.
INFLOW_PER_FRAME = 2000


@ti.kernel
def _emit_kernel(x0: ti.f32, x1: ti.f32, y0: ti.f32, y1: ti.f32,
                  z0: ti.f32, z1: ti.f32,
                  frame_num: ti.i32, start: ti.i32, count: ti.i32):
    """Emit particles in [x0,x1]x[y0,y1]x[z0,z1] with downward velocity.

    Uses three independent Weyl sequences (irrational multiples of index)
    so x/y/z positions are uncorrelated -- no horizontal banding artefacts.
    """
    for i in range(count):
        idx = start + i
        # Three genuinely independent Weyl sequences using mutually irrational steps.
        # Bases: phi-1, sqrt(2)-1, sqrt(3)-1 -- no integer-multiple relationships.
        fi = ti.cast(idx, ti.f32)
        ff = ti.cast(frame_num, ti.f32)
        h_x = fi * 0.6180339887 + ff * 0.3819660113   # phi - 1
        h_y = fi * 0.4142135624 + ff * 0.5857864376   # sqrt(2) - 1
        h_z = fi * 0.7320508076 + ff * 0.2679491924   # sqrt(3) - 1
        rx = h_x - ti.floor(h_x)
        ry = h_y - ti.floor(h_y)
        rz = h_z - ti.floor(h_z)
        px[idx] = x0 + rx * (x1 - x0)
        py[idx] = y0 + ry * (y1 - y0)
        pz[idx] = z0 + rz * (z1 - z0)
        pu[idx] = 0.0; pv[idx] = -2.5; pw[idx] = 0.0
        c0[idx] = 0.0; c1[idx] = 0.0; c2[idx] = 0.0
        c3[idx] = 0.0; c4[idx] = 0.0; c5[idx] = 0.0
        c6[idx] = 0.0; c7[idx] = 0.0; c8[idx] = 0.0
    num_particles[None] = start + count


def emit_inflow_particles(frame_num: int):
    if frame_num < INFLOW_START or frame_num >= INFLOW_END:
        return
    n = num_particles[None]
    if n + INFLOW_PER_FRAME > MAX_PARTICLES:
        print(f"[WARNING] Max particles reached at frame {frame_num}")
        return
    _emit_kernel(
        _IFX0 * DX, _IFX1 * DX,
        _IFY0 * DX, _IFY1 * DX,
        _IFZ0 * DX, _IFZ1 * DX,
        frame_num, n, INFLOW_PER_FRAME,
    )


# =============================================================================
# Main simulation loop
# =============================================================================

# Per-scene frame counts: dam_break ends after the wave settles to avoid
# the 'floating ghost slab' artefact; liquid_pouring runs longer to show
# the growing pool after the stream stops.
SCENE_FRAMES = {
    "dam_break":      300,   # 5 s: full collapse + wave spread
    "liquid_pouring": 300,   # 5 s: full stream + growing pool
}

# Ghost-particle removal: after this frame the cleanup runs every N frames.
_GHOST_START_FRAME = 5       # start cleaning early to prevent ghost FLUID-cell inflation
_GHOST_INTERVAL    = 5       # run every 5 frames
# Remove airborne particles above 60 % of domain height (= 0.60 m).
# After dam collapse water level ≤ 0.12 m; above 0.60 m is definitely ghost.
# The initial column (0.75 m) has contiguous FLUID cells → NOT removed.
_GHOST_HEIGHT_FRAC = 0.62


def run_simulation(scene: str, ratio: float, output_dir: Path):
    n_frames = SCENE_FRAMES.get(scene, NUM_FRAMES)

    print(f"\n{'='*65}")
    print(f"[run] Scene: {scene}  |  flip_ratio: {ratio:.2f}"
          f"  |  Frames: {n_frames}  |  Sub-steps: {SUBSTEPS}")
    print(f"[run] Grid: {NX}x{NY}x{NZ}  |  DX: {DX:.4f}"
          f"  |  PPC: {PPC}^3={PPC**3}/cell  |  dt: {DT:.6f}")
    print(f"[run] Output: {output_dir}")
    print(f"{'='*65}")

    # Skip entirely if the compiled video already exists.
    mp4_path = output_dir / f"{scene}.mp4"
    if mp4_path.exists():
        print(f"[skip] {mp4_path} already exists — skipping this render.")
        return

    flip_ratio[None] = ratio

    # Apply per-scene camera before rendering any frame
    cam = SCENE_CAMERAS.get(scene)
    if cam:
        _set_camera(cam["pos"], cam["target"], cam["up"], cam["fov"])

    if scene == "dam_break":
        setup_dam_break()
        is_pouring = False
    elif scene == "liquid_pouring":
        setup_liquid_pouring()
        is_pouring = True
    else:
        raise ValueError(f"Unknown scene: {scene}")

    frames_dir = output_dir / "frames"
    # Wipe stale frames so ffmpeg only compiles frames from THIS run.
    if frames_dir.exists():
        for old in frames_dir.glob("frame_*.png"):
            old.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "energy.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "time", "kinetic_energy", "frame_time_ms", "total_vorticity"
        ])
        writer.writeheader()

        t_phys = 0.0
        for frame_num in range(n_frames):
            t_start = time.perf_counter()

            if is_pouring:
                emit_inflow_particles(frame_num)

            for _ in range(SUBSTEPS):
                fluid_step()
                advect_particles()
                mark_fluid_cells()
                t_phys += DT

            # Periodically remove ghost particles (airborne in AIR cells)
            if frame_num >= _GHOST_START_FRAME and frame_num % _GHOST_INTERVAL == 0:
                removed = remove_ghost_particles(_GHOST_HEIGHT_FRAC)
                if removed > 0:
                    mark_fluid_cells()   # refresh cell types after removal

            frame_ms = (time.perf_counter() - t_start) * 1000.0

            if frame_num % EXPORT_INTERVAL == 0:
                export_frame(frames_dir, frame_num)
                ek   = compute_kinetic_energy()
                # Vorticity is expensive (GPU kernel + readback); compute sparsely
                vort = float(compute_total_vorticity()) if frame_num % 30 == 0 else 0.0
                write_csv_row(writer, frame_num, t_phys, ek, frame_ms, vort)

                if frame_num % 10 == 0:
                    print(f"  Frame {frame_num:4d}/{n_frames}"
                          f"  t={t_phys:.2f}s"
                          f"  E_k={ek:.4f}"
                          f"  dt={frame_ms:.0f}ms"
                          f"  N={num_particles[None]}", flush=True)

    # Compile video
    print(f"\n[ffmpeg] Compiling {scene} video...")
    mp4_path = output_dir / f"{scene}.mp4"
    result = subprocess.run([
        "ffmpeg", "-y", "-framerate", "60",
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(mp4_path),
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[ffmpeg] Video saved: {mp4_path}")
    else:
        print(f"[ffmpeg] WARNING: {result.stderr[:300]}")

    _plot_energy(csv_path, output_dir)


def _plot_energy(csv_path: Path, output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    frames, times, energies = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            frames.append(int(row["frame"]))
            times.append(float(row["time"]))
            energies.append(float(row["kinetic_energy"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(times, energies, "b-", linewidth=0.8)
    axes[0].set(xlabel="Time (s)", ylabel="Kinetic Energy",
                title="Kinetic Energy over Time")
    axes[0].grid(True, alpha=0.3)

    ek0 = energies[0] if energies and energies[0] > 1e-8 else 1.0
    axes[1].plot(times, [e / ek0 for e in energies], "r-", linewidth=0.8)
    axes[1].set(xlabel="Time (s)", ylabel="E_k / E_k(0)",
                title="Normalized Energy Decay")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    png = output_dir / "energy.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[plot] Energy plot: {png}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python framework.py <scene> [flip_ratio]")
        print("  scene: dam_break | liquid_pouring | all")
        print("Examples:")
        print("  python framework.py dam_break")
        print("  python framework.py liquid_pouring 0.97")
        print("  python framework.py all")
        sys.exit(1)

    scene_arg = sys.argv[1]
    ratios = [float(sys.argv[2])] if len(sys.argv) >= 3 else FLIP_RATIOS
    scenes = ["dam_break", "liquid_pouring"] if scene_arg == "all" else [scene_arg]

    print(f"[framework] 3D simulation  Grid: {NX}x{NY}x{NZ}")
    print(f"[framework] Scenes: {scenes}  |  APIC ratios: {ratios}")

    t0 = time.perf_counter()
    for scene in scenes:
        for ratio in ratios:
            ratio_str = f"ratio_{int(ratio * 1000):03d}"
            out_dir = OUTPUT_BASE / "apic" / ratio_str / scene
            run_simulation(scene, ratio, out_dir)

    elapsed = time.perf_counter() - t0
    print(f"\n[framework] Done! Total: {elapsed / 60:.1f} min")
    print(f"[framework] Output: {OUTPUT_BASE.resolve()}")


if __name__ == "__main__":
    main()
