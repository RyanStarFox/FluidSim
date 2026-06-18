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
    python framework.py dam_break polypic
    python framework.py all 0.97 polypic
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
DEFAULT_SOLVER_METHOD = "flip"
SOLVER_METHODS = ("flip", "polypic")
SOLVER_METHOD = DEFAULT_SOLVER_METHOD
POLYPIC_EPS = 1e-12
POLYPIC_VALUE_MAX = V_MAX_PHYS
POLYPIC_C_MAX = 4.0 * V_MAX_PHYS / DX
POLYPIC_Q_MAX = 4.0 * V_MAX_PHYS / (DX * DX)
POLYPIC_C_DAMPING = 0.85
POLYPIC_Q_DAMPING = 0.25

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

# Quadratic PolyPIC coefficients.
# Per velocity component: [xx, yy, zz, xy, xz, yz].
# q0..q5   -> u component, q6..q11 -> v component, q12..q17 -> w component.
q0  = ti.field(ti.f32, shape=MAX_PARTICLES)
q1  = ti.field(ti.f32, shape=MAX_PARTICLES)
q2  = ti.field(ti.f32, shape=MAX_PARTICLES)
q3  = ti.field(ti.f32, shape=MAX_PARTICLES)
q4  = ti.field(ti.f32, shape=MAX_PARTICLES)
q5  = ti.field(ti.f32, shape=MAX_PARTICLES)
q6  = ti.field(ti.f32, shape=MAX_PARTICLES)
q7  = ti.field(ti.f32, shape=MAX_PARTICLES)
q8  = ti.field(ti.f32, shape=MAX_PARTICLES)
q9  = ti.field(ti.f32, shape=MAX_PARTICLES)
q10 = ti.field(ti.f32, shape=MAX_PARTICLES)
q11 = ti.field(ti.f32, shape=MAX_PARTICLES)
q12 = ti.field(ti.f32, shape=MAX_PARTICLES)
q13 = ti.field(ti.f32, shape=MAX_PARTICLES)
q14 = ti.field(ti.f32, shape=MAX_PARTICLES)
q15 = ti.field(ti.f32, shape=MAX_PARTICLES)
q16 = ti.field(ti.f32, shape=MAX_PARTICLES)
q17 = ti.field(ti.f32, shape=MAX_PARTICLES)

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
# Particle <-> Grid -- quadratic PolyPIC transfer
# =============================================================================

@ti.func
def _quad_bspline(f: ti.f32, offset: ti.template()) -> ti.f32:
    """Quadratic B-spline weight for offsets 0, 1, 2."""
    w = 0.0
    if ti.static(offset == 0):
        x = 1.5 - f
        w = 0.5 * x * x
    elif ti.static(offset == 1):
        x = f - 1.0
        w = 0.75 - x * x
    else:
        x = f - 0.5
        w = 0.5 * x * x
    return w


@ti.func
def _sanitize_limit(x: ti.f32, limit: ti.f32) -> ti.f32:
    y = x
    if y != y:
        y = 0.0
    if y > limit:
        y = limit
    if y < -limit:
        y = -limit
    return y


@ti.func
def _damped_limit(x: ti.f32, limit: ti.f32, damping: ti.f32) -> ti.f32:
    return _sanitize_limit(x * damping, limit)


@ti.kernel
def p2g_polypic():
    """Scatter a quadratic per-particle velocity polynomial to the MAC grid.

    This is a practical PolyPIC branch for the existing 3D MAC solver.  Each
    velocity component stores a local polynomial around the particle:
        v(x) = v0 + C d + Q phi_2(d)
    where d is the face position relative to the particle, C is c0..c8, and Q
    is q0..q17.  Quadratic terms are centered by the local stencil moments so
    they do not shift the constant velocity mode.
    """
    # Reset grid + weight buffers
    for i, j, k in u:       u[i, j, k] = 0.0
    for i, j, k in v:       v[i, j, k] = 0.0
    for i, j, k in w:       w[i, j, k] = 0.0
    for i, j, k in u_saved: u_saved[i, j, k] = 0.0
    for i, j, k in v_saved: v_saved[i, j, k] = 0.0
    for i, j, k in w_saved: w_saved[i, j, k] = 0.0

    eps = ti.cast(POLYPIC_EPS, ti.f32)
    val_lim = ti.cast(POLYPIC_VALUE_MAX, ti.f32)

    for p in range(num_particles[None]):
        xp = px[p] * INV_DX
        yp = py[p] * INV_DX
        zp = pz[p] * INV_DX

        # --- u-field: faces at (i, j+0.5, k+0.5) ---
        bu_i = ti.cast(ti.floor(xp - 0.5), ti.i32)
        bu_j = ti.cast(ti.floor(yp - 1.0), ti.i32)
        bu_k = ti.cast(ti.floor(zp - 1.0), ti.i32)
        fu = xp - ti.cast(bu_i, ti.f32)
        gu = (yp - 0.5) - ti.cast(bu_j, ti.f32)
        hu = (zp - 0.5) - ti.cast(bu_k, ti.f32)
        wsum = 0.0; m2x = 0.0; m2y = 0.0; m2z = 0.0
        for di in ti.static(range(3)):
            wx = _quad_bspline(fu, di)
            ni = bu_i + di
            dxp = (ti.cast(ni, ti.f32) - xp) * DX
            for dj in ti.static(range(3)):
                wy = _quad_bspline(gu, dj)
                nj = bu_j + dj
                dyp = (ti.cast(nj, ti.f32) + 0.5 - yp) * DX
                for dk in ti.static(range(3)):
                    wz = _quad_bspline(hu, dk)
                    nk = bu_k + dk
                    dzp = (ti.cast(nk, ti.f32) + 0.5 - zp) * DX
                    if 0 <= ni <= NX and 0 <= nj < NY and 0 <= nk < NZ:
                        wt = wx * wy * wz
                        wsum += wt
                        m2x += wt * dxp * dxp
                        m2y += wt * dyp * dyp
                        m2z += wt * dzp * dzp
        if wsum > eps:
            m2x /= wsum; m2y /= wsum; m2z /= wsum
        for di in ti.static(range(3)):
            wx = _quad_bspline(fu, di)
            ni = bu_i + di
            dxp = (ti.cast(ni, ti.f32) - xp) * DX
            for dj in ti.static(range(3)):
                wy = _quad_bspline(gu, dj)
                nj = bu_j + dj
                dyp = (ti.cast(nj, ti.f32) + 0.5 - yp) * DX
                for dk in ti.static(range(3)):
                    wz = _quad_bspline(hu, dk)
                    nk = bu_k + dk
                    dzp = (ti.cast(nk, ti.f32) + 0.5 - zp) * DX
                    if 0 <= ni <= NX and 0 <= nj < NY and 0 <= nk < NZ:
                        bxx = dxp * dxp - m2x
                        byy = dyp * dyp - m2y
                        bzz = dzp * dzp - m2z
                        val = (pu[p] + c0[p] * dxp + c1[p] * dyp + c2[p] * dzp +
                               q0[p] * bxx + q1[p] * byy + q2[p] * bzz +
                               q3[p] * dxp * dyp + q4[p] * dxp * dzp +
                               q5[p] * dyp * dzp)
                        val = _sanitize_limit(val, val_lim)
                        wt = wx * wy * wz
                        u[ni, nj, nk] += wt * val
                        u_saved[ni, nj, nk] += wt

        # --- v-field: faces at (i+0.5, j, k+0.5) ---
        bv_i = ti.cast(ti.floor(xp - 1.0), ti.i32)
        bv_j = ti.cast(ti.floor(yp - 0.5), ti.i32)
        bv_k = ti.cast(ti.floor(zp - 1.0), ti.i32)
        fv = (xp - 0.5) - ti.cast(bv_i, ti.f32)
        gv = yp - ti.cast(bv_j, ti.f32)
        hv = (zp - 0.5) - ti.cast(bv_k, ti.f32)
        wsum = 0.0; m2x = 0.0; m2y = 0.0; m2z = 0.0
        for di in ti.static(range(3)):
            wx = _quad_bspline(fv, di)
            ni = bv_i + di
            dxp = (ti.cast(ni, ti.f32) + 0.5 - xp) * DX
            for dj in ti.static(range(3)):
                wy = _quad_bspline(gv, dj)
                nj = bv_j + dj
                dyp = (ti.cast(nj, ti.f32) - yp) * DX
                for dk in ti.static(range(3)):
                    wz = _quad_bspline(hv, dk)
                    nk = bv_k + dk
                    dzp = (ti.cast(nk, ti.f32) + 0.5 - zp) * DX
                    if 0 <= ni < NX and 0 <= nj <= NY and 0 <= nk < NZ:
                        wt = wx * wy * wz
                        wsum += wt
                        m2x += wt * dxp * dxp
                        m2y += wt * dyp * dyp
                        m2z += wt * dzp * dzp
        if wsum > eps:
            m2x /= wsum; m2y /= wsum; m2z /= wsum
        for di in ti.static(range(3)):
            wx = _quad_bspline(fv, di)
            ni = bv_i + di
            dxp = (ti.cast(ni, ti.f32) + 0.5 - xp) * DX
            for dj in ti.static(range(3)):
                wy = _quad_bspline(gv, dj)
                nj = bv_j + dj
                dyp = (ti.cast(nj, ti.f32) - yp) * DX
                for dk in ti.static(range(3)):
                    wz = _quad_bspline(hv, dk)
                    nk = bv_k + dk
                    dzp = (ti.cast(nk, ti.f32) + 0.5 - zp) * DX
                    if 0 <= ni < NX and 0 <= nj <= NY and 0 <= nk < NZ:
                        bxx = dxp * dxp - m2x
                        byy = dyp * dyp - m2y
                        bzz = dzp * dzp - m2z
                        val = (pv[p] + c3[p] * dxp + c4[p] * dyp + c5[p] * dzp +
                               q6[p] * bxx + q7[p] * byy + q8[p] * bzz +
                               q9[p] * dxp * dyp + q10[p] * dxp * dzp +
                               q11[p] * dyp * dzp)
                        val = _sanitize_limit(val, val_lim)
                        wt = wx * wy * wz
                        v[ni, nj, nk] += wt * val
                        v_saved[ni, nj, nk] += wt

        # --- w-field: faces at (i+0.5, j+0.5, k) ---
        bw_i = ti.cast(ti.floor(xp - 1.0), ti.i32)
        bw_j = ti.cast(ti.floor(yp - 1.0), ti.i32)
        bw_k = ti.cast(ti.floor(zp - 0.5), ti.i32)
        fw = (xp - 0.5) - ti.cast(bw_i, ti.f32)
        gw = (yp - 0.5) - ti.cast(bw_j, ti.f32)
        hw = zp - ti.cast(bw_k, ti.f32)
        wsum = 0.0; m2x = 0.0; m2y = 0.0; m2z = 0.0
        for di in ti.static(range(3)):
            wx = _quad_bspline(fw, di)
            ni = bw_i + di
            dxp = (ti.cast(ni, ti.f32) + 0.5 - xp) * DX
            for dj in ti.static(range(3)):
                wy = _quad_bspline(gw, dj)
                nj = bw_j + dj
                dyp = (ti.cast(nj, ti.f32) + 0.5 - yp) * DX
                for dk in ti.static(range(3)):
                    wz = _quad_bspline(hw, dk)
                    nk = bw_k + dk
                    dzp = (ti.cast(nk, ti.f32) - zp) * DX
                    if 0 <= ni < NX and 0 <= nj < NY and 0 <= nk <= NZ:
                        wt = wx * wy * wz
                        wsum += wt
                        m2x += wt * dxp * dxp
                        m2y += wt * dyp * dyp
                        m2z += wt * dzp * dzp
        if wsum > eps:
            m2x /= wsum; m2y /= wsum; m2z /= wsum
        for di in ti.static(range(3)):
            wx = _quad_bspline(fw, di)
            ni = bw_i + di
            dxp = (ti.cast(ni, ti.f32) + 0.5 - xp) * DX
            for dj in ti.static(range(3)):
                wy = _quad_bspline(gw, dj)
                nj = bw_j + dj
                dyp = (ti.cast(nj, ti.f32) + 0.5 - yp) * DX
                for dk in ti.static(range(3)):
                    wz = _quad_bspline(hw, dk)
                    nk = bw_k + dk
                    dzp = (ti.cast(nk, ti.f32) - zp) * DX
                    if 0 <= ni < NX and 0 <= nj < NY and 0 <= nk <= NZ:
                        bxx = dxp * dxp - m2x
                        byy = dyp * dyp - m2y
                        bzz = dzp * dzp - m2z
                        val = (pw[p] + c6[p] * dxp + c7[p] * dyp + c8[p] * dzp +
                               q12[p] * bxx + q13[p] * byy + q14[p] * bzz +
                               q15[p] * dxp * dyp + q16[p] * dxp * dzp +
                               q17[p] * dyp * dzp)
                        val = _sanitize_limit(val, val_lim)
                        wt = wx * wy * wz
                        w[ni, nj, nk] += wt * val
                        w_saved[ni, nj, nk] += wt

    # Normalize by accumulated weights
    for i, j, k in u:
        if u_saved[i, j, k] > eps:
            u[i, j, k] /= u_saved[i, j, k]
    for i, j, k in v:
        if v_saved[i, j, k] > eps:
            v[i, j, k] /= v_saved[i, j, k]
    for i, j, k in w:
        if w_saved[i, j, k] > eps:
            w[i, j, k] /= w_saved[i, j, k]


@ti.kernel
def g2p_polypic():
    """Fit quadratic velocity polynomials from the MAC grid back to particles."""
    eps = ti.cast(POLYPIC_EPS, ti.f32)
    v_lim = ti.cast(POLYPIC_VALUE_MAX, ti.f32)
    c_lim = ti.cast(POLYPIC_C_MAX, ti.f32)
    q_lim = ti.cast(POLYPIC_Q_MAX, ti.f32)
    c_damp = ti.cast(POLYPIC_C_DAMPING, ti.f32)
    q_damp = ti.cast(POLYPIC_Q_DAMPING, ti.f32)

    for p in range(num_particles[None]):
        xp = px[p] * INV_DX
        yp = py[p] * INV_DX
        zp = pz[p] * INV_DX

        # --- u component ---
        bu_i = ti.cast(ti.floor(xp - 0.5), ti.i32)
        bu_j = ti.cast(ti.floor(yp - 1.0), ti.i32)
        bu_k = ti.cast(ti.floor(zp - 1.0), ti.i32)
        fu = xp - ti.cast(bu_i, ti.f32)
        gu = (yp - 0.5) - ti.cast(bu_j, ti.f32)
        hu = (zp - 0.5) - ti.cast(bu_k, ti.f32)
        wsum = 0.0; vsum = 0.0; m2x = 0.0; m2y = 0.0; m2z = 0.0
        for di in ti.static(range(3)):
            wx = _quad_bspline(fu, di); ni = bu_i + di
            dxp = (ti.cast(ni, ti.f32) - xp) * DX
            for dj in ti.static(range(3)):
                wy = _quad_bspline(gu, dj); nj = bu_j + dj
                dyp = (ti.cast(nj, ti.f32) + 0.5 - yp) * DX
                for dk in ti.static(range(3)):
                    wz = _quad_bspline(hu, dk); nk = bu_k + dk
                    dzp = (ti.cast(nk, ti.f32) + 0.5 - zp) * DX
                    if 0 <= ni <= NX and 0 <= nj < NY and 0 <= nk < NZ:
                        wt = wx * wy * wz; val = u[ni, nj, nk]
                        wsum += wt; vsum += wt * val
                        m2x += wt * dxp * dxp
                        m2y += wt * dyp * dyp
                        m2z += wt * dzp * dzp
        if wsum > eps:
            pu[p] = vsum / wsum
            m2x /= wsum; m2y /= wsum; m2z /= wsum
        sx = 0.0; sy = 0.0; sz = 0.0
        sxx = 0.0; syy = 0.0; szz = 0.0; sxy = 0.0; sxz = 0.0; syz = 0.0
        dxn = 0.0; dyn = 0.0; dzn = 0.0
        xxn = 0.0; yyn = 0.0; zzn = 0.0; xyn = 0.0; xzn = 0.0; yzn = 0.0
        for di in ti.static(range(3)):
            wx = _quad_bspline(fu, di); ni = bu_i + di
            dxp = (ti.cast(ni, ti.f32) - xp) * DX
            for dj in ti.static(range(3)):
                wy = _quad_bspline(gu, dj); nj = bu_j + dj
                dyp = (ti.cast(nj, ti.f32) + 0.5 - yp) * DX
                for dk in ti.static(range(3)):
                    wz = _quad_bspline(hu, dk); nk = bu_k + dk
                    dzp = (ti.cast(nk, ti.f32) + 0.5 - zp) * DX
                    if 0 <= ni <= NX and 0 <= nj < NY and 0 <= nk < NZ:
                        wt = wx * wy * wz
                        dv = u[ni, nj, nk] - pu[p]
                        bxx = dxp * dxp - m2x
                        byy = dyp * dyp - m2y
                        bzz = dzp * dzp - m2z
                        bxy = dxp * dyp; bxz = dxp * dzp; byz = dyp * dzp
                        sx += wt * dv * dxp; sy += wt * dv * dyp; sz += wt * dv * dzp
                        sxx += wt * dv * bxx; syy += wt * dv * byy; szz += wt * dv * bzz
                        sxy += wt * dv * bxy; sxz += wt * dv * bxz; syz += wt * dv * byz
                        dxn += wt * dxp * dxp; dyn += wt * dyp * dyp; dzn += wt * dzp * dzp
                        xxn += wt * bxx * bxx; yyn += wt * byy * byy; zzn += wt * bzz * bzz
                        xyn += wt * bxy * bxy; xzn += wt * bxz * bxz; yzn += wt * byz * byz
        c0[p] = 0.0
        if dxn > eps: c0[p] = sx / dxn
        c1[p] = 0.0
        if dyn > eps: c1[p] = sy / dyn
        c2[p] = 0.0
        if dzn > eps: c2[p] = sz / dzn
        q0[p] = 0.0
        if xxn > eps: q0[p] = sxx / xxn
        q1[p] = 0.0
        if yyn > eps: q1[p] = syy / yyn
        q2[p] = 0.0
        if zzn > eps: q2[p] = szz / zzn
        q3[p] = 0.0
        if xyn > eps: q3[p] = sxy / xyn
        q4[p] = 0.0
        if xzn > eps: q4[p] = sxz / xzn
        q5[p] = 0.0
        if yzn > eps: q5[p] = syz / yzn

        # --- v component ---
        bv_i = ti.cast(ti.floor(xp - 1.0), ti.i32)
        bv_j = ti.cast(ti.floor(yp - 0.5), ti.i32)
        bv_k = ti.cast(ti.floor(zp - 1.0), ti.i32)
        fv = (xp - 0.5) - ti.cast(bv_i, ti.f32)
        gv = yp - ti.cast(bv_j, ti.f32)
        hv = (zp - 0.5) - ti.cast(bv_k, ti.f32)
        wsum = 0.0; vsum = 0.0; m2x = 0.0; m2y = 0.0; m2z = 0.0
        for di in ti.static(range(3)):
            wx = _quad_bspline(fv, di); ni = bv_i + di
            dxp = (ti.cast(ni, ti.f32) + 0.5 - xp) * DX
            for dj in ti.static(range(3)):
                wy = _quad_bspline(gv, dj); nj = bv_j + dj
                dyp = (ti.cast(nj, ti.f32) - yp) * DX
                for dk in ti.static(range(3)):
                    wz = _quad_bspline(hv, dk); nk = bv_k + dk
                    dzp = (ti.cast(nk, ti.f32) + 0.5 - zp) * DX
                    if 0 <= ni < NX and 0 <= nj <= NY and 0 <= nk < NZ:
                        wt = wx * wy * wz; val = v[ni, nj, nk]
                        wsum += wt; vsum += wt * val
                        m2x += wt * dxp * dxp
                        m2y += wt * dyp * dyp
                        m2z += wt * dzp * dzp
        if wsum > eps:
            pv[p] = vsum / wsum
            m2x /= wsum; m2y /= wsum; m2z /= wsum
        sx = 0.0; sy = 0.0; sz = 0.0
        sxx = 0.0; syy = 0.0; szz = 0.0; sxy = 0.0; sxz = 0.0; syz = 0.0
        dxn = 0.0; dyn = 0.0; dzn = 0.0
        xxn = 0.0; yyn = 0.0; zzn = 0.0; xyn = 0.0; xzn = 0.0; yzn = 0.0
        for di in ti.static(range(3)):
            wx = _quad_bspline(fv, di); ni = bv_i + di
            dxp = (ti.cast(ni, ti.f32) + 0.5 - xp) * DX
            for dj in ti.static(range(3)):
                wy = _quad_bspline(gv, dj); nj = bv_j + dj
                dyp = (ti.cast(nj, ti.f32) - yp) * DX
                for dk in ti.static(range(3)):
                    wz = _quad_bspline(hv, dk); nk = bv_k + dk
                    dzp = (ti.cast(nk, ti.f32) + 0.5 - zp) * DX
                    if 0 <= ni < NX and 0 <= nj <= NY and 0 <= nk < NZ:
                        wt = wx * wy * wz
                        dv = v[ni, nj, nk] - pv[p]
                        bxx = dxp * dxp - m2x
                        byy = dyp * dyp - m2y
                        bzz = dzp * dzp - m2z
                        bxy = dxp * dyp; bxz = dxp * dzp; byz = dyp * dzp
                        sx += wt * dv * dxp; sy += wt * dv * dyp; sz += wt * dv * dzp
                        sxx += wt * dv * bxx; syy += wt * dv * byy; szz += wt * dv * bzz
                        sxy += wt * dv * bxy; sxz += wt * dv * bxz; syz += wt * dv * byz
                        dxn += wt * dxp * dxp; dyn += wt * dyp * dyp; dzn += wt * dzp * dzp
                        xxn += wt * bxx * bxx; yyn += wt * byy * byy; zzn += wt * bzz * bzz
                        xyn += wt * bxy * bxy; xzn += wt * bxz * bxz; yzn += wt * byz * byz
        c3[p] = 0.0
        if dxn > eps: c3[p] = sx / dxn
        c4[p] = 0.0
        if dyn > eps: c4[p] = sy / dyn
        c5[p] = 0.0
        if dzn > eps: c5[p] = sz / dzn
        q6[p] = 0.0
        if xxn > eps: q6[p] = sxx / xxn
        q7[p] = 0.0
        if yyn > eps: q7[p] = syy / yyn
        q8[p] = 0.0
        if zzn > eps: q8[p] = szz / zzn
        q9[p] = 0.0
        if xyn > eps: q9[p] = sxy / xyn
        q10[p] = 0.0
        if xzn > eps: q10[p] = sxz / xzn
        q11[p] = 0.0
        if yzn > eps: q11[p] = syz / yzn

        # --- w component ---
        bw_i = ti.cast(ti.floor(xp - 1.0), ti.i32)
        bw_j = ti.cast(ti.floor(yp - 1.0), ti.i32)
        bw_k = ti.cast(ti.floor(zp - 0.5), ti.i32)
        fw = (xp - 0.5) - ti.cast(bw_i, ti.f32)
        gw = (yp - 0.5) - ti.cast(bw_j, ti.f32)
        hw = zp - ti.cast(bw_k, ti.f32)
        wsum = 0.0; vsum = 0.0; m2x = 0.0; m2y = 0.0; m2z = 0.0
        for di in ti.static(range(3)):
            wx = _quad_bspline(fw, di); ni = bw_i + di
            dxp = (ti.cast(ni, ti.f32) + 0.5 - xp) * DX
            for dj in ti.static(range(3)):
                wy = _quad_bspline(gw, dj); nj = bw_j + dj
                dyp = (ti.cast(nj, ti.f32) + 0.5 - yp) * DX
                for dk in ti.static(range(3)):
                    wz = _quad_bspline(hw, dk); nk = bw_k + dk
                    dzp = (ti.cast(nk, ti.f32) - zp) * DX
                    if 0 <= ni < NX and 0 <= nj < NY and 0 <= nk <= NZ:
                        wt = wx * wy * wz; val = w[ni, nj, nk]
                        wsum += wt; vsum += wt * val
                        m2x += wt * dxp * dxp
                        m2y += wt * dyp * dyp
                        m2z += wt * dzp * dzp
        if wsum > eps:
            pw[p] = vsum / wsum
            m2x /= wsum; m2y /= wsum; m2z /= wsum
        sx = 0.0; sy = 0.0; sz = 0.0
        sxx = 0.0; syy = 0.0; szz = 0.0; sxy = 0.0; sxz = 0.0; syz = 0.0
        dxn = 0.0; dyn = 0.0; dzn = 0.0
        xxn = 0.0; yyn = 0.0; zzn = 0.0; xyn = 0.0; xzn = 0.0; yzn = 0.0
        for di in ti.static(range(3)):
            wx = _quad_bspline(fw, di); ni = bw_i + di
            dxp = (ti.cast(ni, ti.f32) + 0.5 - xp) * DX
            for dj in ti.static(range(3)):
                wy = _quad_bspline(gw, dj); nj = bw_j + dj
                dyp = (ti.cast(nj, ti.f32) + 0.5 - yp) * DX
                for dk in ti.static(range(3)):
                    wz = _quad_bspline(hw, dk); nk = bw_k + dk
                    dzp = (ti.cast(nk, ti.f32) - zp) * DX
                    if 0 <= ni < NX and 0 <= nj < NY and 0 <= nk <= NZ:
                        wt = wx * wy * wz
                        dv = w[ni, nj, nk] - pw[p]
                        bxx = dxp * dxp - m2x
                        byy = dyp * dyp - m2y
                        bzz = dzp * dzp - m2z
                        bxy = dxp * dyp; bxz = dxp * dzp; byz = dyp * dzp
                        sx += wt * dv * dxp; sy += wt * dv * dyp; sz += wt * dv * dzp
                        sxx += wt * dv * bxx; syy += wt * dv * byy; szz += wt * dv * bzz
                        sxy += wt * dv * bxy; sxz += wt * dv * bxz; syz += wt * dv * byz
                        dxn += wt * dxp * dxp; dyn += wt * dyp * dyp; dzn += wt * dzp * dzp
                        xxn += wt * bxx * bxx; yyn += wt * byy * byy; zzn += wt * bzz * bzz
                        xyn += wt * bxy * bxy; xzn += wt * bxz * bxz; yzn += wt * byz * byz
        c6[p] = 0.0
        if dxn > eps: c6[p] = sx / dxn
        c7[p] = 0.0
        if dyn > eps: c7[p] = sy / dyn
        c8[p] = 0.0
        if dzn > eps: c8[p] = sz / dzn
        q12[p] = 0.0
        if xxn > eps: q12[p] = sxx / xxn
        q13[p] = 0.0
        if yyn > eps: q13[p] = syy / yyn
        q14[p] = 0.0
        if zzn > eps: q14[p] = szz / zzn
        q15[p] = 0.0
        if xyn > eps: q15[p] = sxy / xyn
        q16[p] = 0.0
        if xzn > eps: q16[p] = sxz / xzn
        q17[p] = 0.0
        if yzn > eps: q17[p] = syz / yzn

        # Stabilize the fitted high-order modes.  The independent projection
        # above is intentionally cheap, so we bound the modes before they feed
        # the next P2G pass.
        pu[p] = _sanitize_limit(pu[p], v_lim)
        pv[p] = _sanitize_limit(pv[p], v_lim)
        pw[p] = _sanitize_limit(pw[p], v_lim)

        c0[p] = _damped_limit(c0[p], c_lim, c_damp)
        c1[p] = _damped_limit(c1[p], c_lim, c_damp)
        c2[p] = _damped_limit(c2[p], c_lim, c_damp)
        c3[p] = _damped_limit(c3[p], c_lim, c_damp)
        c4[p] = _damped_limit(c4[p], c_lim, c_damp)
        c5[p] = _damped_limit(c5[p], c_lim, c_damp)
        c6[p] = _damped_limit(c6[p], c_lim, c_damp)
        c7[p] = _damped_limit(c7[p], c_lim, c_damp)
        c8[p] = _damped_limit(c8[p], c_lim, c_damp)

        q0[p] = _damped_limit(q0[p], q_lim, q_damp)
        q1[p] = _damped_limit(q1[p], q_lim, q_damp)
        q2[p] = _damped_limit(q2[p], q_lim, q_damp)
        q3[p] = _damped_limit(q3[p], q_lim, q_damp)
        q4[p] = _damped_limit(q4[p], q_lim, q_damp)
        q5[p] = _damped_limit(q5[p], q_lim, q_damp)
        q6[p] = _damped_limit(q6[p], q_lim, q_damp)
        q7[p] = _damped_limit(q7[p], q_lim, q_damp)
        q8[p] = _damped_limit(q8[p], q_lim, q_damp)
        q9[p] = _damped_limit(q9[p], q_lim, q_damp)
        q10[p] = _damped_limit(q10[p], q_lim, q_damp)
        q11[p] = _damped_limit(q11[p], q_lim, q_damp)
        q12[p] = _damped_limit(q12[p], q_lim, q_damp)
        q13[p] = _damped_limit(q13[p], q_lim, q_damp)
        q14[p] = _damped_limit(q14[p], q_lim, q_damp)
        q15[p] = _damped_limit(q15[p], q_lim, q_damp)
        q16[p] = _damped_limit(q16[p], q_lim, q_damp)
        q17[p] = _damped_limit(q17[p], q_lim, q_damp)


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

        # Clear higher-order modes (FLIP does not use APIC/PolyPIC state)
        c0[p] = 0.0; c1[p] = 0.0; c2[p] = 0.0
        c3[p] = 0.0; c4[p] = 0.0; c5[p] = 0.0
        c6[p] = 0.0; c7[p] = 0.0; c8[p] = 0.0
        q0[p] = 0.0; q1[p] = 0.0; q2[p] = 0.0
        q3[p] = 0.0; q4[p] = 0.0; q5[p] = 0.0
        q6[p] = 0.0; q7[p] = 0.0; q8[p] = 0.0
        q9[p] = 0.0; q10[p] = 0.0; q11[p] = 0.0
        q12[p] = 0.0; q13[p] = 0.0; q14[p] = 0.0
        q15[p] = 0.0; q16[p] = 0.0; q17[p] = 0.0


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
        if px[p] != px[p]:
            px[p] = 0.5 * dom_x; pu[p] = 0.0
        if py[p] != py[p]:
            py[p] = 0.5 * dom_y; pv[p] = 0.0
        if pz[p] != pz[p]:
            pz[p] = 0.5 * dom_z; pw[p] = 0.0

        pu[p] = _sanitize_limit(pu[p], v_lim)
        pv[p] = _sanitize_limit(pv[p], v_lim)
        pw[p] = _sanitize_limit(pw[p], v_lim)

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
            q0[dst] = q0[p]; q1[dst] = q1[p]; q2[dst] = q2[p]
            q3[dst] = q3[p]; q4[dst] = q4[p]; q5[dst] = q5[p]
            q6[dst] = q6[p]; q7[dst] = q7[p]; q8[dst] = q8[p]
            q9[dst] = q9[p]; q10[dst] = q10[p]; q11[dst] = q11[p]
            q12[dst] = q12[p]; q13[dst] = q13[p]; q14[dst] = q14[p]
            q15[dst] = q15[p]; q16[dst] = q16[p]; q17[dst] = q17[p]


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
    """Execute one sub-step using the selected particle-grid transfer method.

    SOLVER_METHOD == "flip" uses trilinear PIC/FLIP transfer.
    SOLVER_METHOD == "polypic" uses quadratic Polynomial PIC transfer while
    sharing the same gravity, boundary, pressure solve, advection, and render
    path for apples-to-apples comparison.
    """
    if SOLVER_METHOD == "polypic":
        p2g_polypic()
    else:
        p2g_trilinear()

    # Save grid velocities before forces/projection (needed by FLIP delta).
    save_velocities()

    add_gravity()
    enforce_boundary_velocity()

    compute_divergence()
    solve_pressure_cg()
    apply_pressure_gradient(DT)

    enforce_boundary_velocity()

    if SOLVER_METHOD == "polypic":
        g2p_polypic()
    else:
        g2p_flip()


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
    speed_sq = pu_np**2 + pv_np**2 + pw_np**2
    speed_sq = np.where(np.isfinite(speed_sq), speed_sq, 0.0)
    return float(0.5 * MASS_PER_PARTICLE * np.sum(speed_sq))


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
    for f in (pu, pv, pw, c0, c1, c2, c3, c4, c5, c6, c7, c8,
              q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11,
              q12, q13, q14, q15, q16, q17):
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
        q0[flat] = 0.0; q1[flat] = 0.0; q2[flat] = 0.0
        q3[flat] = 0.0; q4[flat] = 0.0; q5[flat] = 0.0
        q6[flat] = 0.0; q7[flat] = 0.0; q8[flat] = 0.0
        q9[flat] = 0.0; q10[flat] = 0.0; q11[flat] = 0.0
        q12[flat] = 0.0; q13[flat] = 0.0; q14[flat] = 0.0
        q15[flat] = 0.0; q16[flat] = 0.0; q17[flat] = 0.0
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
        q0[idx] = 0.0; q1[idx] = 0.0; q2[idx] = 0.0
        q3[idx] = 0.0; q4[idx] = 0.0; q5[idx] = 0.0
        q6[idx] = 0.0; q7[idx] = 0.0; q8[idx] = 0.0
        q9[idx] = 0.0; q10[idx] = 0.0; q11[idx] = 0.0
        q12[idx] = 0.0; q13[idx] = 0.0; q14[idx] = 0.0
        q15[idx] = 0.0; q16[idx] = 0.0; q17[idx] = 0.0
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


def run_simulation(scene: str, ratio: float, output_dir: Path,
                   method: str = DEFAULT_SOLVER_METHOD):
    global SOLVER_METHOD
    SOLVER_METHOD = method
    n_frames = SCENE_FRAMES.get(scene, NUM_FRAMES)

    print(f"\n{'='*65}")
    print(f"[run] Scene: {scene}  |  method: {method}"
          f"  |  ratio: {ratio:.2f}"
          f"  |  Frames: {n_frames}  |  Sub-steps: {SUBSTEPS}")
    print(f"[run] Grid: {NX}x{NY}x{NZ}  |  DX: {DX:.4f}"
          f"  |  PPC: {PPC}^3={PPC**3}/cell  |  dt: {DT:.6f}")
    print(f"[run] Output: {output_dir}")
    print(f"{'='*65}")

    # Skip existing videos by default; Slurm reruns can opt into overwrite.
    mp4_path = output_dir / f"{scene}.mp4"
    force_render = os.environ.get("FLUIDSIM_FORCE_RENDER", "0") == "1"
    if mp4_path.exists() and not force_render:
        print(f"[skip] {mp4_path} already exists — skipping this render.")
        return
    if mp4_path.exists() and force_render:
        print(f"[rerun] overwriting existing {mp4_path}")

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
        print("Usage: python framework.py <scene> [ratio] [method]")
        print("  scene: dam_break | liquid_pouring | all")
        print("  method: flip | polypic")
        print("Examples:")
        print("  python framework.py dam_break")
        print("  python framework.py liquid_pouring 0.97")
        print("  python framework.py dam_break polypic")
        print("  python framework.py all 0.97 polypic")
        sys.exit(1)

    scene_arg = sys.argv[1]
    method = DEFAULT_SOLVER_METHOD
    ratios = FLIP_RATIOS

    for arg in sys.argv[2:]:
        lowered = arg.lower()
        if lowered in SOLVER_METHODS:
            method = lowered
        else:
            ratios = [float(arg)]

    if scene_arg not in ("dam_break", "liquid_pouring", "all"):
        raise ValueError(f"Unknown scene: {scene_arg}")

    scenes = ["dam_break", "liquid_pouring"] if scene_arg == "all" else [scene_arg]

    print(f"[framework] 3D simulation  Grid: {NX}x{NY}x{NZ}")
    print(f"[framework] Scenes: {scenes}  |  Method: {method}  |  Ratios: {ratios}")

    t0 = time.perf_counter()
    for scene in scenes:
        for ratio in ratios:
            ratio_str = f"ratio_{int(ratio * 1000):03d}"
            out_dir = OUTPUT_BASE / method / ratio_str / scene
            run_simulation(scene, ratio, out_dir, method)

    elapsed = time.perf_counter() - t0
    print(f"\n[framework] Done! Total: {elapsed / 60:.1f} min")
    print(f"[framework] Output: {OUTPUT_BASE.resolve()}")


if __name__ == "__main__":
    main()
