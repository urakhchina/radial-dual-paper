#!/usr/bin/env python3
"""
Composite Linear-Radial Folding Visualization
==============================================
Demonstrates the composite operator ι_r ∘ H4fold which maps high-norm
E8 lattice points through the Moxness projection to 4D, applies
hyperspherical inversion to trap them inside the 600-cell boundary,
and slices the result into a bounded 3D Fibonacci Icosagrid (FIG).

Output: composite_folding_fig.png / .pdf

Reference: Schmidt & Irwin (2026), Section 5.1
"""

import time
import numpy as np
from itertools import combinations, product as iprod

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

Phi = (1 + np.sqrt(5)) / 2
phi = Phi - 1
phi_sq = phi ** 2

H4fold = np.array([
    [ Phi,    0,     0,     0,    phi_sq,  0,     0,     0   ],
    [  0,    phi,    1,     0,     0,    -phi,    1,     0   ],
    [  0,     1,     0,    phi,    0,      1,     0,   -phi  ],
    [  0,     0,    phi,    1,     0,      0,   -phi,    1   ],
    [phi_sq,  0,     0,     0,    Phi,     0,     0,     0   ],
    [  0,   -phi,    1,     0,     0,     phi,    1,     0   ],
    [  0,     1,     0,   -phi,    0,      1,     0,    phi  ],
    [  0,     0,   -phi,    1,     0,      0,    phi,    1   ]
])

Pi_proj = H4fold[:4, :]       # parallel space (4×8)


# ═══════════════════════════════════════════════════════════════════
# Step 1: Generate E8 lattice (large patch)
# ═══════════════════════════════════════════════════════════════════

def generate_e8_lattice(max_norm_sq=20, coord_range=4):
    """Generate E8 lattice points up to given squared norm."""
    t0 = time.time()
    points = []
    r = coord_range
    coords = list(range(-r, r + 1))

    # Integer sublattice (D8): even coordinate sum
    for v in iprod(coords, repeat=8):
        v_arr = np.array(v, dtype=float)
        if np.sum(v_arr ** 2) <= max_norm_sq and sum(v) % 2 == 0:
            points.append(v_arr)

    # Half-integer sublattice
    half_coords = [x + 0.5 for x in range(-r, r)]
    for v in iprod(half_coords, repeat=8):
        v_arr = np.array(v, dtype=float)
        if np.sum(v_arr ** 2) <= max_norm_sq and round(sum(v)) % 2 == 0:
            points.append(v_arr)

    elapsed = time.time() - t0
    pts = np.array(points)
    print(f"  E8 lattice: {len(pts)} points (max N={max_norm_sq}) [{elapsed:.1f}s]")
    return pts


# ═══════════════════════════════════════════════════════════════════
# Step 2: Project 8D → 4D via H4fold (top 4 rows)
# ═══════════════════════════════════════════════════════════════════

def project_to_4d(lattice_8d):
    """Apply Moxness H4fold parallel-space projection."""
    proj_4d = (Pi_proj @ lattice_8d.T).T
    norms_4d = np.linalg.norm(proj_4d, axis=1)
    print(f"  Projected to 4D: {len(proj_4d)} points")
    print(f"    4D norm range: [{norms_4d.min():.4f}, {norms_4d.max():.4f}]")
    return proj_4d, norms_4d


# ═══════════════════════════════════════════════════════════════════
# Step 3: Hyperspherical inversion in 4D
# ═══════════════════════════════════════════════════════════════════

def apply_inversion(proj_4d, norms_4d):
    """Apply ι_r(w) = w · r² / |w|² in 4D.
    r is set to the minimum nonzero 4D norm (the 600-cell boundary)."""
    # Find inversion radius = min nonzero norm
    nonzero_mask = norms_4d > 1e-10
    proj_nz = proj_4d[nonzero_mask]
    norms_nz = norms_4d[nonzero_mask]

    r_boundary = norms_nz.min()
    r_sq = r_boundary ** 2
    print(f"  Inversion radius r = {r_boundary:.6f} (r² = {r_sq:.6f})")

    # Identify zones
    boundary_tol = 1e-6
    boundary_mask = np.abs(norms_nz - r_boundary) < boundary_tol
    outer_mask = norms_nz > r_boundary + boundary_tol
    inner_already = norms_nz < r_boundary - boundary_tol

    n_boundary = np.sum(boundary_mask)
    n_outer = np.sum(outer_mask)
    n_inner_already = np.sum(inner_already)
    print(f"  Zones: {n_boundary} boundary, {n_outer} outer, {n_inner_already} already inner")

    # Apply inversion to outer-zone points → inner dual
    norms_sq_nz = norms_nz ** 2
    inverted = proj_nz * (r_sq / norms_sq_nz[:, None])
    inverted_norms = np.linalg.norm(inverted, axis=1)

    # Verify: all inverted outer points should now be inside the boundary
    outer_inverted_norms = inverted_norms[outer_mask]
    all_inside = np.all(outer_inverted_norms <= r_boundary + 1e-10)
    print(f"  After inversion: all outer points inside boundary? {all_inside}")
    print(f"    Inverted norm range: [{inverted_norms.min():.6f}, {inverted_norms.max():.6f}]")

    return inverted, inverted_norms, r_boundary, nonzero_mask


# ═══════════════════════════════════════════════════════════════════
# Step 4: Slice 4D → 3D (FIG hyperplane slice)
# ═══════════════════════════════════════════════════════════════════

def slice_to_3d(pts_4d, slab_thickness=0.5):
    """Take a 3D hyperplane slice of the 4D point cloud."""
    # Normal vector for the FIG slice
    eta = np.array([1, -1, 1, 1]) / 2.0
    eta = eta / np.linalg.norm(eta)

    # Orthonormal basis for the 3D hyperplane
    def orthonormal_complement(normal, dim=4):
        basis = []
        candidates = np.eye(dim)
        for e in candidates:
            v = e.copy()
            v -= np.dot(v, normal) * normal
            for b in basis:
                v -= np.dot(v, b) * b
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                basis.append(v / norm)
            if len(basis) == dim - 1:
                break
        return np.array(basis)

    basis_3d = orthonormal_complement(eta, 4)

    # Project onto normal to find height
    heights = pts_4d @ eta

    # Keep points within slab
    in_slab = np.abs(heights) < slab_thickness
    pts_3d = pts_4d[in_slab] @ basis_3d.T

    print(f"  3D slice (slab={slab_thickness}): {np.sum(in_slab)} of {len(pts_4d)} points")

    return pts_3d, in_slab


# ═══════════════════════════════════════════════════════════════════
# Step 5: Render
# ═══════════════════════════════════════════════════════════════════

def render_figure(pts_3d, r_boundary, n_total_8d, n_inverted, n_sliced):
    """Create publication-quality 3D scatter plot."""
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Color by distance from origin
    dists = np.linalg.norm(pts_3d, axis=1)

    scatter = ax.scatter(
        pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2],
        c=dists, cmap='magma_r', s=2, alpha=0.5,
        rasterized=True
    )

    cb = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.08)
    cb.set_label('Distance from origin', fontsize=10)

    ax.set_title(
        'Composite Linear-Radial Folding: The Bounded Inner-Zone FIG\n'
        f'$\\iota_r \\circ H_{{4\\mathrm{{fold}}}}$: '
        f'{n_total_8d:,} E8 pts → {n_inverted:,} inverted → {n_sliced:,} in 3D slice',
        fontsize=12, fontweight='bold'
    )

    ax.set_xlabel('$x_1$', fontsize=9)
    ax.set_ylabel('$x_2$', fontsize=9)
    ax.set_zlabel('$x_3$', fontsize=9)
    ax.tick_params(labelsize=7)

    # Clean look: remove grid panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)

    plt.tight_layout()
    plt.savefig('composite_folding_fig.png', dpi=200)
    plt.savefig('composite_folding_fig.pdf')
    print("\nSaved: composite_folding_fig.png / .pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("=" * 60)
    print("COMPOSITE LINEAR-RADIAL FOLDING VISUALIZATION")
    print("ι_r ∘ H4fold: E8 (8D) → H4 (4D) → Inner Dual → FIG (3D)")
    print("=" * 60)

    # Step 1: Generate E8 lattice
    print("\n[1] Generating E8 lattice (max_norm_sq=20)...")
    lattice_8d = generate_e8_lattice(max_norm_sq=20, coord_range=4)
    n_total = len(lattice_8d)

    # Report shell structure
    norms_sq_8d = np.sum(lattice_8d ** 2, axis=1)
    shells_8d = {}
    for nsq in sorted(np.unique(np.round(norms_sq_8d, 1))):
        count = int(np.sum(np.abs(norms_sq_8d - nsq) < 0.1))
        if count > 0:
            shells_8d[int(round(nsq))] = count
    print(f"  Shell structure: {shells_8d}")

    # Step 2: Project to 4D
    print("\n[2] Projecting 8D → 4D via H4fold...")
    proj_4d, norms_4d = project_to_4d(lattice_8d)

    # Step 3: Hyperspherical inversion
    print("\n[3] Applying 4D hyperspherical inversion...")
    inverted_4d, inverted_norms, r_boundary, nonzero_mask = apply_inversion(
        proj_4d, norms_4d)
    n_inverted = len(inverted_4d)

    # Step 4: Slice to 3D
    print("\n[4] Slicing 4D → 3D (FIG hyperplane)...")
    pts_3d, slab_mask = slice_to_3d(inverted_4d, slab_thickness=0.5)
    n_sliced = len(pts_3d)

    # Summary statistics
    dists_3d = np.linalg.norm(pts_3d, axis=1)
    print(f"\n  3D point cloud statistics:")
    print(f"    Points: {n_sliced}")
    print(f"    Bounding radius: {dists_3d.max():.6f}")
    print(f"    Min distance from origin: {dists_3d[dists_3d > 1e-10].min():.6f}")
    print(f"    Mean distance: {dists_3d.mean():.6f}")
    print(f"    Median distance: {np.median(dists_3d):.6f}")

    # Density analysis: how many points within inner radial bands
    for frac in [0.25, 0.50, 0.75]:
        r_cut = frac * dists_3d.max()
        n_inside = np.sum(dists_3d < r_cut)
        print(f"    Within {frac:.0%} of max radius ({r_cut:.4f}): "
              f"{n_inside} pts ({100*n_inside/n_sliced:.1f}%)")

    # Step 5: Render
    print("\n[5] Rendering 3D visualization...")
    render_figure(pts_3d, r_boundary, n_total, n_inverted, n_sliced)

    total_time = time.time() - t_total
    print(f"\nTotal runtime: {total_time:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
