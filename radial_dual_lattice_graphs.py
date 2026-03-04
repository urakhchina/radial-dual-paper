#!/usr/bin/env python3
"""
Radial Dual Lattice Graphs — 2-Panel Figure
============================================
Panel (a): 2D Eisenstein lattice with zone-colored graph edges
Panel (b): 3D E8 radial dual with zone coloring

Output: radial_dual_lattice_graphs.png / .pdf

Reference: Schmidt & Irwin (2026), Sections 2 & 5
"""

import time
import numpy as np
from itertools import product as iprod

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

SQRT3 = np.sqrt(3)


# ═══════════════════════════════════════════════════════════════════
# Panel (a): 2D Eisenstein Lattice Graph
# ═══════════════════════════════════════════════════════════════════

def eisenstein_xy(m, n):
    """Lattice coords (m, n) -> Cartesian (x, y) for z = m + n*omega."""
    return m - n / 2.0, n * SQRT3 / 2.0


def eisenstein_N(m, n):
    """Eisenstein norm squared: N(m + n*omega) = m^2 - m*n + n^2."""
    return m * m - m * n + n * n


# Nearest-neighbor unit differences in (m, n) space
# Full set of 6 Eisenstein units: (1,0),(-1,0),(0,1),(0,-1),(-1,-1),(1,1)
# Half set (avoids double-counting edges):
NN_HALF = [(1, 0), (0, 1), (1, 1)]


def generate_eisenstein(R=4):
    """Generate Eisenstein lattice points classified into zones.

    Returns:
        outer:    dict (m,n) -> (x,y) for 1 < N <= R^2
        boundary: dict (m,n) -> (x,y) for N = 1
        inner:    dict (m,n) -> (ix,iy) inversions of outer points
    """
    R_sq = R * R
    outer, boundary, inner = {}, {}, {}

    for m in range(-8, 9):
        for n in range(-8, 9):
            if m == 0 and n == 0:
                continue
            N = eisenstein_N(m, n)
            if N > R_sq:
                continue
            x, y = eisenstein_xy(m, n)
            if N == 1:
                boundary[(m, n)] = (x, y)
            else:  # N > 1
                outer[(m, n)] = (x, y)
                inner[(m, n)] = (x / N, y / N)  # iota(z) = z/N since r^2 = 1

    print(f"  Eisenstein: {len(boundary)} boundary, {len(outer)} outer, "
          f"{len(inner)} inner")
    return outer, boundary, inner


def zone_edges(pts):
    """Nearest-neighbor edges within a zone. Returns [(seg1), (seg2), ...]."""
    keys = set(pts)
    edges = []
    for (m, n) in keys:
        for dm, dn in NN_HALF:
            nb = (m + dm, n + dn)
            if nb in keys:
                edges.append((pts[(m, n)], pts[nb]))
    return edges


def dual_edges(outer, inner):
    """Inner-zone edges that are images of outer-zone NN edges."""
    keys = set(outer)
    edges = []
    for (m, n) in keys:
        for dm, dn in NN_HALF:
            nb = (m + dm, n + dn)
            if nb in keys:
                edges.append((inner[(m, n)], inner[nb]))
    return edges


def render_panel_a(ax, outer, boundary, inner):
    """Render 2D Eisenstein lattice with graph edges and zone coloring."""

    # ── Decorations ──────────────────────────────────────────────

    # Faint 6-fold sector wedges
    for k in range(6):
        rad = np.radians(k * 60)
        ax.plot([0, 4.6 * np.cos(rad)], [0, 4.6 * np.sin(rad)],
                color='#aaaaaa', lw=0.3, alpha=0.25, zorder=0)

    # Sector labels S_0 -- S_5 at midpoints
    for k in range(6):
        theta = np.radians(k * 60 + 30)
        ax.text(4.3 * np.cos(theta), 4.3 * np.sin(theta),
                f'$S_{k}$', fontsize=6, ha='center', va='center',
                color='#888888')

    # Dashed unit circle (|z| = r = 1)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ls='--', lw=0.7,
                         color='#777777', alpha=0.5, zorder=1))

    # Dashed unit hexagon (boundary outline)
    hx = [np.cos(np.radians(k * 60)) for k in range(7)]
    hy = [np.sin(np.radians(k * 60)) for k in range(7)]
    ax.plot(hx, hy, '--', color='#777777', lw=0.7, alpha=0.5, zorder=1)

    # Faint outer circle at R = 4
    ax.add_patch(Circle((0, 0), 4.0, fill=False, ls=':', lw=0.5,
                         color='#aaaaaa', alpha=0.3, zorder=0))

    # Faint Re / Im axes
    ax.axhline(0, color='#cccccc', lw=0.3, zorder=0)
    ax.axvline(0, color='#cccccc', lw=0.3, zorder=0)

    # ── Graph edges ──────────────────────────────────────────────

    outer_e = zone_edges(outer)
    inner_e = dual_edges(outer, inner)
    bnd_e = zone_edges(boundary)

    if outer_e:
        ax.add_collection(LineCollection(
            outer_e, colors='red', linewidths=0.35, alpha=0.3, zorder=2))
    if inner_e:
        ax.add_collection(LineCollection(
            inner_e, colors='blue', linewidths=0.4, alpha=0.45, zorder=2))
    if bnd_e:
        ax.add_collection(LineCollection(
            bnd_e, colors='green', linewidths=0.6, alpha=0.5, zorder=2))

    # ── Scatter points ───────────────────────────────────────────

    def unzip(d):
        xy = list(d.values())
        return [p[0] for p in xy], [p[1] for p in xy]

    ox, oy = unzip(outer)
    bx, by = unzip(boundary)
    ix, iy = unzip(inner)

    ax.scatter(ox, oy, c='red', s=6, zorder=4, alpha=0.65,
               edgecolors='none', label='Outer ($N > 1$)')
    ax.scatter(bx, by, c='green', s=18, zorder=5, marker='D',
               edgecolors='darkgreen', linewidths=0.4,
               label='Boundary ($N = 1$)')
    ax.scatter(ix, iy, c='blue', s=5, zorder=4, alpha=0.7,
               edgecolors='none', label='Inner (inverted)')

    # ── Twin path ────────────────────────────────────────────────
    # v1 = (m=-2, n=1) -> (-2.5, sqrt3/2),  N=7
    # v2 = (m=-1, n=1) -> (-1.5, sqrt3/2),  N=3
    # v3 = (m=-1, n=2) -> (-2.0, sqrt3),    N=7

    tp_out = [(-2.5, SQRT3 / 2), (-1.5, SQRT3 / 2), (-2.0, SQRT3)]
    tp_inn = [(-5.0 / 14, SQRT3 / 14), (-0.5, SQRT3 / 6),
              (-2.0 / 7, SQRT3 / 7)]

    # Outer path (red thick)
    for i in range(2):
        ax.plot([tp_out[i][0], tp_out[i + 1][0]],
                [tp_out[i][1], tp_out[i + 1][1]],
                color='darkred', lw=2.0, zorder=6, solid_capstyle='round')

    # Inner path (blue thick)
    for i in range(2):
        ax.plot([tp_inn[i][0], tp_inn[i + 1][0]],
                [tp_inn[i][1], tp_inn[i + 1][1]],
                color='darkblue', lw=2.0, zorder=6, solid_capstyle='round')

    # Gray dotted connectors between twin pairs
    for op, ip in zip(tp_out, tp_inn):
        ax.plot([op[0], ip[0]], [op[1], ip[1]],
                ':', color='gray', lw=0.7, alpha=0.5, zorder=3)

    # Mark twin path vertices
    for pt in tp_out:
        ax.plot(*pt, 'o', color='darkred', ms=4, zorder=7)
    for pt in tp_inn:
        ax.plot(*pt, 'o', color='darkblue', ms=4, zorder=7)

    # ── Axes styling ─────────────────────────────────────────────

    ax.set_xlim(-4.8, 4.8)
    ax.set_ylim(-4.8, 4.8)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=5.5, framealpha=0.85,
              handletextpad=0.3, borderpad=0.3)
    ax.set_title(r'(a) Eisenstein lattice graph, $r{=}1$, $R{=}4$',
                 fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Re', fontsize=8)
    ax.set_ylabel('Im', fontsize=8)
    ax.tick_params(labelsize=6)


def add_inner_inset(ax, outer, boundary, inner):
    """Add magnified inset of the inner zone."""
    ax_ins = ax.inset_axes([0.62, 0.62, 0.36, 0.36])

    # Inner edges
    ie = dual_edges(outer, inner)
    if ie:
        ax_ins.add_collection(LineCollection(
            ie, colors='blue', linewidths=0.8, alpha=0.6))

    # Boundary edges
    be = zone_edges(boundary)
    if be:
        ax_ins.add_collection(LineCollection(
            be, colors='green', linewidths=1.0, alpha=0.6))

    # Inner points
    for (x, y) in inner.values():
        ax_ins.plot(x, y, 'o', color='blue', ms=2.5, alpha=0.8)

    # Boundary points
    for (x, y) in boundary.values():
        ax_ins.plot(x, y, 'D', color='green', ms=3.5,
                    markeredgecolor='darkgreen', markeredgewidth=0.3)

    # Twin inner path
    tp_inn = [(-5.0 / 14, SQRT3 / 14), (-0.5, SQRT3 / 6),
              (-2.0 / 7, SQRT3 / 7)]
    for i in range(2):
        ax_ins.plot([tp_inn[i][0], tp_inn[i + 1][0]],
                    [tp_inn[i][1], tp_inn[i + 1][1]],
                    color='darkblue', lw=1.5, zorder=6)
    for pt in tp_inn:
        ax_ins.plot(*pt, 'o', color='darkblue', ms=3, zorder=7)

    # Dashed unit circle
    ax_ins.add_patch(Circle((0, 0), 1.0, fill=False, ls='--', lw=0.4,
                             color='gray', alpha=0.3))

    ax_ins.set_xlim(-1.15, 1.15)
    ax_ins.set_ylim(-1.15, 1.15)
    ax_ins.set_aspect('equal')
    ax_ins.tick_params(labelsize=4)
    ax_ins.set_title('Inner zone (magnified)', fontsize=5.5, pad=2)

    # Connect inset to main plot
    ax.indicate_inset_zoom(ax_ins, edgecolor='blue', alpha=0.3, lw=0.5)


# ═══════════════════════════════════════════════════════════════════
# Panel (b): 3D E8 Radial Dual — Zone-Colored
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

Pi_proj = H4fold[:4, :]  # parallel space (4x8)


def generate_e8(max_norm_sq=10, coord_range=2):
    """Generate E8 lattice points (integer + half-integer sublattices)."""
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

    pts = np.array(points)
    print(f"  E8: {len(pts)} points (max N={max_norm_sq}) [{time.time()-t0:.1f}s]")
    return pts


def slice_4d_to_3d(pts_4d, slab=0.5):
    """FIG hyperplane slice: 4D -> 3D."""
    eta = np.array([1, -1, 1, 1]) / 2.0
    eta /= np.linalg.norm(eta)

    # Orthonormal complement basis for the 3D hyperplane
    basis = []
    for e in np.eye(4):
        v = e - np.dot(e, eta) * eta
        for b in basis:
            v -= np.dot(v, b) * b
        nrm = np.linalg.norm(v)
        if nrm > 1e-10:
            basis.append(v / nrm)
        if len(basis) == 3:
            break
    basis = np.array(basis)

    heights = pts_4d @ eta
    mask = np.abs(heights) < slab
    return (pts_4d[mask] @ basis.T), mask


def render_panel_b(ax):
    """Render E8 radial dual with zone coloring in 3D."""
    print("\n[Panel B] E8 radial dual...")
    lat8 = generate_e8(max_norm_sq=10, coord_range=2)

    # Project to 4D
    print("  Projecting to 4D...")
    p4 = (Pi_proj @ lat8.T).T
    n4 = np.linalg.norm(p4, axis=1)

    # Remove origin
    nz = n4 > 1e-10
    p4, n4 = p4[nz], n4[nz]

    # Zone classification
    r_bnd = n4.min()
    r_sq = r_bnd ** 2
    tol = 1e-6

    m_bnd = np.abs(n4 - r_bnd) < tol
    m_out = n4 > r_bnd + tol

    pts_bnd = p4[m_bnd]
    pts_out = p4[m_out]
    pts_inn = pts_out * (r_sq / (np.linalg.norm(pts_out, axis=1) ** 2)[:, None])

    print(f"  Zones: {len(pts_bnd)} boundary, {len(pts_out)} outer")
    print(f"  r = {r_bnd:.6f}")

    # Slice each zone to 3D independently
    print("  Slicing to 3D...")
    b3, _ = slice_4d_to_3d(pts_bnd)
    o3, _ = slice_4d_to_3d(pts_out)
    i3, _ = slice_4d_to_3d(pts_inn)

    print(f"  3D: {len(b3)} bnd, {len(o3)} out, {len(i3)} inn")

    # Scatter plot with zone colors
    if len(o3):
        ax.scatter(o3[:, 0], o3[:, 1], o3[:, 2],
                   c='red', s=2, alpha=0.35, label='Outer',
                   rasterized=True, edgecolors='none')
    if len(b3):
        ax.scatter(b3[:, 0], b3[:, 1], b3[:, 2],
                   c='green', s=8, alpha=0.7, label='Boundary',
                   rasterized=True, edgecolors='none')
    if len(i3):
        ax.scatter(i3[:, 0], i3[:, 1], i3[:, 2],
                   c='blue', s=2, alpha=0.45, label='Inner (inv.)',
                   rasterized=True, edgecolors='none')

    ax.legend(loc='upper left', fontsize=5.5, framealpha=0.85)
    ax.set_title('(b) E8 radial dual, zone-colored',
                 fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('$x_1$', fontsize=8)
    ax.set_ylabel('$x_2$', fontsize=8)
    ax.set_zlabel('$x_3$', fontsize=8)
    ax.tick_params(labelsize=5)

    # Clean panes
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('w')
    ax.grid(False)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 60)
    print("RADIAL DUAL LATTICE GRAPHS \u2014 2-Panel Figure")
    print("=" * 60)

    fig = plt.figure(figsize=(14, 6.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)

    # Panel (a): 2D Eisenstein
    print("\n[Panel A] Eisenstein lattice graph...")
    outer, boundary, inner = generate_eisenstein()
    ax_a = fig.add_subplot(gs[0])
    render_panel_a(ax_a, outer, boundary, inner)
    add_inner_inset(ax_a, outer, boundary, inner)

    # Panel (b): 3D E8
    ax_b = fig.add_subplot(gs[1], projection='3d')
    render_panel_b(ax_b)

    fig.savefig('radial_dual_lattice_graphs.png', dpi=200, bbox_inches='tight')
    fig.savefig('radial_dual_lattice_graphs.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved: radial_dual_lattice_graphs.png / .pdf")
    print(f"Total: {time.time() - t0:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
