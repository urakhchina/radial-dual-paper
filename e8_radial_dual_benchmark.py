#!/usr/bin/env python3
"""
E8 Radial Dual Verification Benchmark (6-Part Suite)
=====================================================
Comprehensive verification of all major claims from Schmidt & Irwin (2026),
"Radial Dual Lattice Graphs via Admissible Hyperspherical Inversion".

Parts:
  1. Table 1 exact reproduction (4-cycle clock)
  2. H4fold → 600-cell projection (240 roots → two 120-vertex polytopes)
  3. Norm relation N(ι(v)) = 4/N(v)  [exact Fraction arithmetic]
  4. Involution ι(ι(v)) = v           [exact Fraction arithmetic]
  5. Adjacency preservation under ι
  6. Magnitude compression across shells

Usage:
  cd /Users/natasha/Documents/radial-dual-paper
  python3 e8_radial_dual_benchmark.py
"""

import time
import random
from fractions import Fraction
from itertools import combinations, product as iprod

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

Phi = (1 + np.sqrt(5)) / 2       # golden ratio ≈ 1.618
phi = Phi - 1                      # 1/Phi ≈ 0.618
phi_sq = phi ** 2                  # ≈ 0.382

# Moxness H4fold projection matrix (8×8 orthogonal)
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
Omega_proj = H4fold[4:, :]    # perpendicular space (4×8)

# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def squared_norm_frac(v):
    """Squared norm of a tuple of Fractions — exact."""
    return sum(x * x for x in v)


def radial_dual_frac(v):
    """ι(v) = 2v/N(v), exact Fraction arithmetic."""
    n = squared_norm_frac(v)
    return tuple(Fraction(2) * x / n for x in v)


def format_frac(f):
    """Pretty-print a Fraction."""
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def to_frac_tuple(arr):
    """Convert a numpy array (or list) of floats to a tuple of Fractions.
    Handles integers and half-integers exactly."""
    result = []
    for x in arr:
        # Round to nearest half-integer to avoid float noise
        doubled = round(2 * float(x))
        result.append(Fraction(doubled, 2))
    return tuple(result)


# ═══════════════════════════════════════════════════════════════════════
# Generators
# ═══════════════════════════════════════════════════════════════════════

def generate_e8_roots_frac():
    """240 E8 roots as tuples of Fractions. All have N(v) = 2."""
    roots = []
    half = Fraction(1, 2)

    # Type A (112): permutations of (±1, ±1, 0⁶)
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = [Fraction(0)] * 8
                    v[i] = Fraction(si)
                    v[j] = Fraction(sj)
                    roots.append(tuple(v))

    # Type B (128): (±1/2)⁸ with even number of minus signs
    for bits in range(256):
        signs = []
        neg_count = 0
        for k in range(8):
            if bits & (1 << k):
                signs.append(-half)
                neg_count += 1
            else:
                signs.append(half)
        if neg_count % 2 == 0:
            roots.append(tuple(signs))

    assert len(roots) == 240
    return roots


def generate_e8_roots_numpy():
    """240 E8 roots as numpy array (240, 8). All have squared norm 2."""
    # Type A (112)
    type_A = []
    for i, j in combinations(range(8), 2):
        for si in (+1, -1):
            for sj in (+1, -1):
                v = np.zeros(8)
                v[i] = si
                v[j] = sj
                type_A.append(v)
    type_A = np.array(type_A)

    # Type B (128)
    type_B = []
    for bits in range(256):
        signs = np.array([(bits >> k) & 1 for k in range(8)])
        num_minus = np.sum(signs)
        if num_minus % 2 == 0:
            v = np.where(signs, -0.5, +0.5)
            type_B.append(v)
    type_B = np.array(type_B)

    roots = np.vstack([type_A, type_B])
    assert len(roots) == 240
    assert np.allclose(np.sum(roots ** 2, axis=1), 2.0)
    return roots


def generate_e8_lattice(max_norm_sq=10, coord_range=3):
    """Generate E8 lattice points up to given squared norm.
    Returns numpy array of shape (N, 8).
    E8 = D8 ∪ (D8 + [½]⁸): integer coords with even sum,
    plus half-integer coords with even sum."""
    points = []
    r = coord_range
    coords = list(range(-r, r + 1))

    # Integer sublattice (D8): all integer coords with even coordinate sum
    for v in iprod(coords, repeat=8):
        v_arr = np.array(v, dtype=float)
        if np.sum(v_arr ** 2) <= max_norm_sq and sum(v) % 2 == 0:
            points.append(v_arr)

    # Half-integer sublattice: all (n+½) coords with even sum
    half_coords = [x + 0.5 for x in range(-r, r)]
    for v in iprod(half_coords, repeat=8):
        v_arr = np.array(v, dtype=float)
        if np.sum(v_arr ** 2) <= max_norm_sq and round(sum(v)) % 2 == 0:
            points.append(v_arr)

    return np.array(points)


# ═══════════════════════════════════════════════════════════════════════
# Part 1: Table 1 Exact Reproduction
# ═══════════════════════════════════════════════════════════════════════

def verify_table1():
    """Verify the paper's 4-cycle clock: p0→p1→p2→p3→p0.
    Each step differs by an E8 root, and ι maps produce exact fractions."""
    print("\n" + "=" * 60)
    print("PART 1: Table 1 Exact Reproduction (4-cycle clock)")
    print("=" * 60)

    F = Fraction
    Z = F(0)

    # The 4 cycle-clock points from the paper: p0=(4,2,0⁶) etc.
    cycle = [
        (F(4), F(2), Z, Z, Z, Z, Z, Z),   # p0: N=20
        (F(5), F(1), Z, Z, Z, Z, Z, Z),   # p1: N=26
        (F(4), F(2), Z, Z, Z, Z, Z, Z),   # p2 = p0 (cycle wraps)
        (F(5), F(1), Z, Z, Z, Z, Z, Z),   # p3 = p1 (cycle wraps)
    ]
    # Actually, the 4-cycle from the paper is:
    # p0 = (4, 2, 0⁶)  N=20
    # p1 = (5, 1, 0⁶)  N=26
    # p2 = (4, 2, 0⁶)  N=20  (back to p0 after adding root (-1,1,0⁶))
    # p3 = (5, 1, 0⁶)  N=26  (back to p1 after adding root (1,-1,0⁶))
    # Let's use the actual distinct points and verify the cycle structure.

    # Distinct points in the cycle
    p0 = (F(4), F(2), Z, Z, Z, Z, Z, Z)
    p1 = (F(5), F(1), Z, Z, Z, Z, Z, Z)

    # The roots connecting them
    r01 = tuple(p1[i] - p0[i] for i in range(8))  # (1, -1, 0⁶)
    r10 = tuple(p0[i] - p1[i] for i in range(8))  # (-1, 1, 0⁶)

    passed = True
    errors = []

    # Check roots have N=2
    n_r01 = squared_norm_frac(r01)
    n_r10 = squared_norm_frac(r10)
    if n_r01 != 2:
        errors.append(f"N(r01) = {n_r01}, expected 2")
        passed = False
    if n_r10 != 2:
        errors.append(f"N(r10) = {n_r10}, expected 2")
        passed = False

    # Check norms
    n_p0 = squared_norm_frac(p0)
    n_p1 = squared_norm_frac(p1)
    print(f"  p0 = (4,2,0⁶)  N(p0) = {n_p0}")
    print(f"  p1 = (5,1,0⁶)  N(p1) = {n_p1}")

    if n_p0 != 20:
        errors.append(f"N(p0) = {n_p0}, expected 20")
        passed = False
    if n_p1 != 26:
        errors.append(f"N(p1) = {n_p1}, expected 26")
        passed = False

    # Compute duals
    iota_p0 = radial_dual_frac(p0)
    iota_p1 = radial_dual_frac(p1)

    print(f"  ι(p0) = ({', '.join(format_frac(x) for x in iota_p0[:3])}, ...)")
    print(f"  ι(p1) = ({', '.join(format_frac(x) for x in iota_p1[:3])}, ...)")

    # Verify exact fractions: ι(p0) = (2/5, 1/5, 0⁶)
    expected_iota_p0 = (F(2, 5), F(1, 5), Z, Z, Z, Z, Z, Z)
    expected_iota_p1 = (F(5, 13), F(1, 13), Z, Z, Z, Z, Z, Z)

    if iota_p0 != expected_iota_p0:
        errors.append(f"ι(p0) mismatch: got {iota_p0[:2]}, expected {expected_iota_p0[:2]}")
        passed = False
    if iota_p1 != expected_iota_p1:
        errors.append(f"ι(p1) mismatch: got {iota_p1[:2]}, expected {expected_iota_p1[:2]}")
        passed = False

    # Verify involution: ι(ι(pi)) = pi
    for label, p in [("p0", p0), ("p1", p1)]:
        iota_p = radial_dual_frac(p)
        iota_iota_p = radial_dual_frac(iota_p)
        if iota_iota_p != p:
            errors.append(f"ι(ι({label})) ≠ {label}")
            passed = False

    # Verify norm relation: N(ι(v)) = 4/N(v)
    for label, p in [("p0", p0), ("p1", p1)]:
        n_p = squared_norm_frac(p)
        n_iota = squared_norm_frac(radial_dual_frac(p))
        expected = F(4) / n_p
        if n_iota != expected:
            errors.append(f"N(ι({label})) = {n_iota}, expected {expected}")
            passed = False
        print(f"  N(ι({label})) = {format_frac(n_iota)} = 4/{n_p} ✓")

    # Verify cycle closure: p0 + root = p1, p1 + root = p0
    p0_plus_r01 = tuple(p0[i] + r01[i] for i in range(8))
    p1_plus_r10 = tuple(p1[i] + r10[i] for i in range(8))
    if p0_plus_r01 != p1:
        errors.append("p0 + r01 ≠ p1")
        passed = False
    if p1_plus_r10 != p0:
        errors.append("p1 + r10 ≠ p0")
        passed = False

    if passed:
        print("  PART 1: PASS ✓")
    else:
        print(f"  PART 1: FAIL ✗")
        for e in errors:
            print(f"    - {e}")

    return passed, {
        'p0': p0, 'p1': p1,
        'iota_p0': iota_p0, 'iota_p1': iota_p1,
        'n_p0': n_p0, 'n_p1': n_p1,
    }


# ═══════════════════════════════════════════════════════════════════════
# Part 2: H4fold → 600-cell Projection
# ═══════════════════════════════════════════════════════════════════════

def verify_h4fold(roots_np):
    """Project 240 E8 roots through Pi_proj. Verify two clusters of 120
    vectors each, with norm ratio = golden ratio Φ."""
    print("\n" + "=" * 60)
    print("PART 2: H4fold → 600-cell Projection")
    print("=" * 60)

    passed = True
    errors = []

    # Project to 4D parallel space
    proj_4d = (Pi_proj @ roots_np.T).T  # (240, 4)
    norms_4d = np.linalg.norm(proj_4d, axis=1)

    # Find distinct norm clusters
    unique_norms = np.unique(np.round(norms_4d, 6))
    print(f"  Distinct 4D norms (rounded): {unique_norms}")

    if len(unique_norms) != 2:
        errors.append(f"Expected 2 distinct norms, got {len(unique_norms)}")
        passed = False
    else:
        n_small = np.sum(np.abs(norms_4d - unique_norms[0]) < 0.01)
        n_large = np.sum(np.abs(norms_4d - unique_norms[1]) < 0.01)
        ratio = unique_norms[1] / unique_norms[0]

        print(f"  Cluster 1: {n_small} vectors, norm ≈ {unique_norms[0]:.6f}")
        print(f"  Cluster 2: {n_large} vectors, norm ≈ {unique_norms[1]:.6f}")
        print(f"  Norm ratio: {ratio:.6f}  (Φ = {Phi:.6f})")

        if n_small != 120:
            errors.append(f"Small cluster has {n_small} vectors, expected 120")
            passed = False
        if n_large != 120:
            errors.append(f"Large cluster has {n_large} vectors, expected 120")
            passed = False
        if not np.isclose(ratio, Phi, atol=1e-6):
            errors.append(f"Norm ratio {ratio} ≠ Φ = {Phi}")
            passed = False

    if passed:
        print("  PART 2: PASS ✓")
    else:
        print("  PART 2: FAIL ✗")
        for e in errors:
            print(f"    - {e}")

    return passed, {
        'proj_4d': proj_4d,
        'norms_4d': norms_4d,
        'unique_norms': unique_norms,
    }


# ═══════════════════════════════════════════════════════════════════════
# Parts 3+4: Norm Relation & Involution (single pass)
# ═══════════════════════════════════════════════════════════════════════

def verify_norm_and_involution(lattice_np):
    """For every non-origin lattice point, verify:
      Part 3: N(ι(v)) = 4/N(v)   [exact Fraction]
      Part 4: ι(ι(v)) = v         [exact Fraction]
    Combined into a single pass to avoid redundant Fraction conversion."""
    print("\n" + "=" * 60)
    print("PARTS 3+4: Norm Relation & Involution (exact, single pass)")
    print("=" * 60)

    n_points = len(lattice_np)
    print(f"  Lattice points: {n_points}")

    # Remove origin
    norms_sq_np = np.sum(lattice_np ** 2, axis=1)
    nonzero_mask = norms_sq_np > 0.01
    lattice_nz = lattice_np[nonzero_mask]
    norms_sq_nz = norms_sq_np[nonzero_mask]
    n_nz = len(lattice_nz)
    print(f"  Non-origin points: {n_nz}")

    passed_norm = True
    passed_invol = True
    norm_errors = []
    invol_errors = []
    n_checked = 0

    # Collect data for plotting: N(v) and N(ι(v)) for each point
    outer_norms = []
    inner_norms = []

    # Shell breakdown
    shells = {}
    for nsq in sorted(np.unique(np.round(norms_sq_nz, 1))):
        shell_mask = np.abs(norms_sq_nz - nsq) < 0.1
        shells[int(round(nsq))] = int(np.sum(shell_mask))

    print(f"  Shells: {shells}")
    t0 = time.time()

    for idx in range(n_nz):
        v_frac = to_frac_tuple(lattice_nz[idx])
        n_v = squared_norm_frac(v_frac)

        if n_v == 0:
            continue

        # Part 3: N(ι(v)) = 4/N(v)
        iota_v = radial_dual_frac(v_frac)
        n_iota = squared_norm_frac(iota_v)
        expected_n = Fraction(4) / n_v

        if n_iota != expected_n:
            passed_norm = False
            if len(norm_errors) < 5:
                norm_errors.append(
                    f"v={v_frac[:3]}..., N(v)={n_v}, "
                    f"N(ι(v))={n_iota} ≠ 4/{n_v}={expected_n}")

        # Part 4: ι(ι(v)) = v
        iota_iota_v = radial_dual_frac(iota_v)
        if iota_iota_v != v_frac:
            passed_invol = False
            if len(invol_errors) < 5:
                invol_errors.append(
                    f"v={v_frac[:3]}..., ι(ι(v))={iota_iota_v[:3]}...")

        outer_norms.append(float(n_v))
        inner_norms.append(float(n_iota))
        n_checked += 1

        if (idx + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"    ...checked {idx + 1}/{n_nz} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Checked {n_checked} points in {elapsed:.1f}s")

    # Part 3 result
    print(f"\n  PART 3 (Norm relation N(ι(v))=4/N(v)):")
    if passed_norm:
        print(f"    PASS ✓  — all {n_checked} points exact")
    else:
        print(f"    FAIL ✗")
        for e in norm_errors:
            print(f"      - {e}")

    # Part 4 result
    print(f"  PART 4 (Involution ι(ι(v))=v):")
    if passed_invol:
        print(f"    PASS ✓  — all {n_checked} points exact")
    else:
        print(f"    FAIL ✗")
        for e in invol_errors:
            print(f"      - {e}")

    return passed_norm, passed_invol, {
        'outer_norms': np.array(outer_norms),
        'inner_norms': np.array(inner_norms),
        'shells': shells,
        'n_checked': n_checked,
    }


# ═══════════════════════════════════════════════════════════════════════
# Part 5: Adjacency Preservation
# ═══════════════════════════════════════════════════════════════════════

def verify_adjacency(lattice_np, roots_frac):
    """Find all nearest-neighbor edges (N(v-w)=2) and verify:
      N(ι(v) - ι(w)) = 8 / (N(v)·N(w))
    Float check for ALL edges, exact Fraction check for 1000 sample edges.
    Also verify Table 1 cycle clock in inner zone."""
    print("\n" + "=" * 60)
    print("PART 5: Adjacency Preservation")
    print("=" * 60)

    passed = True
    errors = []

    # Remove origin
    norms_sq = np.sum(lattice_np ** 2, axis=1)
    nonzero_mask = norms_sq > 0.01
    lattice_nz = lattice_np[nonzero_mask]
    norms_sq_nz = norms_sq[nonzero_mask]
    n_nz = len(lattice_nz)

    # Build KD-tree and find all nearest-neighbor pairs (distance = sqrt(2))
    print(f"  Building cKDTree for {n_nz} non-origin points...")
    t0 = time.time()
    tree = cKDTree(lattice_nz)
    # N(v-w) = 2 means ||v-w|| = sqrt(2) ≈ 1.4142
    pairs = tree.query_pairs(r=np.sqrt(2) + 0.01)
    # Filter to exact N=2 edges
    edges = []
    for i, j in pairs:
        diff = lattice_nz[i] - lattice_nz[j]
        nsq = np.sum(diff ** 2)
        if abs(nsq - 2.0) < 0.01:
            edges.append((i, j))

    n_edges = len(edges)
    print(f"  Found {n_edges} edges with N(v-w)=2 in {time.time()-t0:.1f}s")

    # ── Vectorized float verification for ALL edges ──
    print("  Vectorized float check (all edges)...")
    t0 = time.time()
    idx_i = np.array([e[0] for e in edges])
    idx_j = np.array([e[1] for e in edges])

    vi = lattice_nz[idx_i]  # (n_edges, 8)
    vj = lattice_nz[idx_j]
    ni = norms_sq_nz[idx_i]
    nj = norms_sq_nz[idx_j]

    # Compute ι(v) = 2v/N(v) for both endpoints
    iota_vi = 2.0 * vi / ni[:, None]
    iota_vj = 2.0 * vj / nj[:, None]

    # N(ι(v) - ι(w))
    diff_iota = iota_vi - iota_vj
    n_diff_iota = np.sum(diff_iota ** 2, axis=1)

    # Expected: 8 / (N(v) · N(w))
    expected = 8.0 / (ni * nj)

    rel_errors = np.abs(n_diff_iota - expected) / expected
    max_rel_err = np.max(rel_errors) if len(rel_errors) > 0 else 0
    float_pass = max_rel_err < 1e-10

    print(f"    Max relative error: {max_rel_err:.2e}")
    print(f"    Float check: {'PASS' if float_pass else 'FAIL'}")
    if not float_pass:
        passed = False
        errors.append(f"Float adjacency check failed, max rel error = {max_rel_err}")

    print(f"    Completed in {time.time()-t0:.1f}s")

    # ── Exact Fraction verification for random sample ──
    n_sample = min(1000, n_edges)
    print(f"  Exact Fraction check ({n_sample} sampled edges)...")
    t0 = time.time()

    random.seed(42)
    sample_indices = random.sample(range(n_edges), n_sample)
    exact_pass = True

    for count, edge_idx in enumerate(sample_indices):
        i, j = edges[edge_idx]
        v_frac = to_frac_tuple(lattice_nz[i])
        w_frac = to_frac_tuple(lattice_nz[j])

        n_v = squared_norm_frac(v_frac)
        n_w = squared_norm_frac(w_frac)

        iota_v = radial_dual_frac(v_frac)
        iota_w = radial_dual_frac(w_frac)

        diff_f = tuple(iota_v[k] - iota_w[k] for k in range(8))
        n_diff_f = squared_norm_frac(diff_f)
        expected_f = Fraction(8) / (n_v * n_w)

        if n_diff_f != expected_f:
            exact_pass = False
            if len(errors) < 5:
                errors.append(
                    f"Edge ({i},{j}): N(ι(v)-ι(w))={n_diff_f} ≠ {expected_f}")

        if (count + 1) % 250 == 0:
            print(f"    ...checked {count + 1}/{n_sample}")

    if not exact_pass:
        passed = False

    elapsed = time.time() - t0
    print(f"    Exact check: {'PASS' if exact_pass else 'FAIL'} ({elapsed:.1f}s)")

    # ── Table 1 cycle clock in inner zone ──
    print("  Table 1 cycle clock adjacency check...")
    F = Fraction
    Z = F(0)
    p0 = (F(4), F(2), Z, Z, Z, Z, Z, Z)
    p1 = (F(5), F(1), Z, Z, Z, Z, Z, Z)

    iota_p0 = radial_dual_frac(p0)
    iota_p1 = radial_dual_frac(p1)
    diff_inner = tuple(iota_p0[k] - iota_p1[k] for k in range(8))
    n_diff_inner = squared_norm_frac(diff_inner)
    n_p0 = squared_norm_frac(p0)
    n_p1 = squared_norm_frac(p1)
    expected_inner = Fraction(8) / (n_p0 * n_p1)

    cycle_pass = (n_diff_inner == expected_inner)
    print(f"    N(ι(p0)-ι(p1)) = {format_frac(n_diff_inner)}")
    print(f"    8/(N(p0)·N(p1)) = 8/({n_p0}·{n_p1}) = {format_frac(expected_inner)}")
    print(f"    Cycle clock check: {'PASS' if cycle_pass else 'FAIL'}")
    if not cycle_pass:
        passed = False
        errors.append("Table 1 cycle clock adjacency mismatch")

    if passed:
        print("  PART 5: PASS ✓")
    else:
        print("  PART 5: FAIL ✗")
        for e in errors:
            print(f"    - {e}")

    return passed, {'n_edges': n_edges, 'max_rel_err': max_rel_err}


# ═══════════════════════════════════════════════════════════════════════
# Part 6: Magnitude Compression Across Shells
# ═══════════════════════════════════════════════════════════════════════

def compute_shell_compression(lattice_np):
    """For each shell N=2,4,6,8,10: compute max|coord| in outer zone
    and in inner zone (after ι). Show outer grows as O(√N), inner
    shrinks as O(1/√N)."""
    print("\n" + "=" * 60)
    print("PART 6: Magnitude Compression Across Shells")
    print("=" * 60)

    passed = True
    errors = []

    norms_sq = np.sum(lattice_np ** 2, axis=1)

    shell_data = {}
    target_shells = [2, 4, 6, 8, 10]

    print(f"  {'Shell N':>8s}  {'Count':>6s}  {'max|outer|':>12s}  "
          f"{'max|inner|':>12s}  {'√N':>6s}  {'1/√N':>8s}")
    print(f"  {'─' * 8}  {'─' * 6}  {'─' * 12}  {'─' * 12}  {'─' * 6}  {'─' * 8}")

    for N in target_shells:
        shell_mask = np.abs(norms_sq - N) < 0.1
        shell_pts = lattice_np[shell_mask]
        count = len(shell_pts)

        if count == 0:
            print(f"  {N:>8d}  {0:>6d}  {'N/A':>12s}  {'N/A':>12s}")
            continue

        # Outer max|coord|
        outer_max = np.max(np.abs(shell_pts))

        # Inner: ι(v) = 2v/N(v), max|coord| = 2*max|coord|/N
        inner_max = 2.0 * outer_max / N

        shell_data[N] = {
            'count': count,
            'outer_max': outer_max,
            'inner_max': inner_max,
        }

        print(f"  {N:>8d}  {count:>6d}  {outer_max:>12.4f}  "
              f"{inner_max:>12.6f}  {np.sqrt(N):>6.3f}  {1/np.sqrt(N):>8.5f}")

    # Verify monotonicity: outer_max should increase, inner_max should decrease
    outer_maxes = [shell_data[N]['outer_max'] for N in target_shells if N in shell_data]
    inner_maxes = [shell_data[N]['inner_max'] for N in target_shells if N in shell_data]

    outer_increasing = all(outer_maxes[i] <= outer_maxes[i + 1]
                          for i in range(len(outer_maxes) - 1))
    inner_decreasing = all(inner_maxes[i] >= inner_maxes[i + 1]
                          for i in range(len(inner_maxes) - 1))

    if not outer_increasing:
        errors.append("Outer max|coord| is not monotonically increasing with N")
        passed = False
    if not inner_decreasing:
        errors.append("Inner max|coord| is not monotonically decreasing with N")
        passed = False

    print(f"\n  Outer max|coord| increasing: {'✓' if outer_increasing else '✗'}")
    print(f"  Inner max|coord| decreasing: {'✓' if inner_decreasing else '✗'}")

    if passed:
        print("  PART 6: PASS ✓")
    else:
        print("  PART 6: FAIL ✗")
        for e in errors:
            print(f"    - {e}")

    return passed, shell_data


# ═══════════════════════════════════════════════════════════════════════
# Figure: 4-panel 2×2 plot
# ═══════════════════════════════════════════════════════════════════════

def make_figure(table1_data, h4fold_data, norm_data, shell_data):
    """Generate 4-panel 2×2 figure summarizing all results."""
    fig = plt.figure(figsize=(14, 11))

    # ── Panel (a): Table 1 grouped bar chart ──
    ax_a = fig.add_subplot(2, 2, 1)

    F = Fraction
    Z = F(0)
    points = {
        'p0': (F(4), F(2), Z, Z, Z, Z, Z, Z),
        'p1': (F(5), F(1), Z, Z, Z, Z, Z, Z),
    }
    labels = ['p0\nN=20', 'p1\nN=26']
    outer_vals = []
    inner_vals = []
    for key in ['p0', 'p1']:
        p = points[key]
        iota_p = radial_dual_frac(p)
        outer_vals.append(float(max(abs(x) for x in p)))
        inner_vals.append(float(max(abs(x) for x in iota_p)))

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax_a.bar(x - width / 2, outer_vals, width, label='Outer max|coord|',
                     color='#cc3333', alpha=0.85)
    bars2 = ax_a.bar(x + width / 2, inner_vals, width, label='Inner max|coord|',
                     color='#3366cc', alpha=0.85)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel('Max |coordinate|')
    ax_a.set_title('(a) Table 1: Cycle Clock Points', fontweight='bold')
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax_a.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                      ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax_a.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                      ha='center', va='bottom', fontsize=8)

    # ── Panel (b): H4fold 3D scatter ──
    ax_b = fig.add_subplot(2, 2, 2, projection='3d')

    proj_4d = h4fold_data['proj_4d']
    norms_4d = h4fold_data['norms_4d']
    unique_norms = h4fold_data['unique_norms']

    if len(unique_norms) == 2:
        mask_small = np.abs(norms_4d - unique_norms[0]) < 0.01
        mask_large = ~mask_small
        # Plot first 3 of 4 dimensions
        ax_b.scatter(proj_4d[mask_small, 0], proj_4d[mask_small, 1],
                     proj_4d[mask_small, 2], c='#3366cc', s=8, alpha=0.6,
                     label=f'H4 (n≈{unique_norms[0]:.2f})')
        ax_b.scatter(proj_4d[mask_large, 0], proj_4d[mask_large, 1],
                     proj_4d[mask_large, 2], c='#daa520', s=8, alpha=0.6,
                     label=f'H4Φ (n≈{unique_norms[1]:.2f})')
        ax_b.legend(fontsize=7, loc='upper left')

    ax_b.set_title('(b) H4fold: 240 E8 Roots → 4D', fontweight='bold')
    ax_b.set_xlabel('x₁', fontsize=8)
    ax_b.set_ylabel('x₂', fontsize=8)
    ax_b.set_zlabel('x₃', fontsize=8)
    ax_b.tick_params(labelsize=7)

    # ── Panel (c): Norm relation scatter (log-log) ──
    ax_c = fig.add_subplot(2, 2, 3)

    outer_n = norm_data['outer_norms']
    inner_n = norm_data['inner_norms']

    ax_c.scatter(outer_n, inner_n, s=1, alpha=0.15, color='#3366cc',
                 rasterized=True)

    # Overlay y = 4/x curve
    x_curve = np.linspace(max(outer_n.min(), 0.1), outer_n.max(), 200)
    ax_c.plot(x_curve, 4.0 / x_curve, 'r-', linewidth=1.5, alpha=0.8,
              label='$N(\\iota(v)) = 4/N(v)$')

    ax_c.set_xscale('log')
    ax_c.set_yscale('log')
    ax_c.set_xlabel('$N(v)$')
    ax_c.set_ylabel('$N(\\iota(v))$')
    ax_c.set_title('(c) Norm Relation (log-log)', fontweight='bold')
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3, which='both')

    # ── Panel (d): Magnitude compression across shells ──
    ax_d = fig.add_subplot(2, 2, 4)

    if shell_data:
        shells_N = sorted(shell_data.keys())
        outer_maxes = [shell_data[N]['outer_max'] for N in shells_N]
        inner_maxes = [shell_data[N]['inner_max'] for N in shells_N]

        ax_d.plot(shells_N, outer_maxes, 'o-', color='#cc3333', linewidth=2,
                  markersize=7, label='Outer max|coord|')
        ax_d.plot(shells_N, inner_maxes, 's-', color='#3366cc', linewidth=2,
                  markersize=7, label='Inner max|coord|')

        # Overlay theoretical curves
        N_fine = np.linspace(2, 10, 100)
        # Normalize to match data at N=2
        outer_scale = outer_maxes[0] / np.sqrt(2)
        inner_scale = inner_maxes[0] * np.sqrt(2)
        ax_d.plot(N_fine, outer_scale * np.sqrt(N_fine), '--',
                  color='#cc3333', alpha=0.4, label='$O(\\sqrt{N})$')
        ax_d.plot(N_fine, inner_scale / np.sqrt(N_fine), '--',
                  color='#3366cc', alpha=0.4, label='$O(1/\\sqrt{N})$')

        ax_d.set_xlabel('Shell $N = \\|v\\|^2$')
        ax_d.set_ylabel('Max |coordinate|')
        ax_d.legend(fontsize=8)

    ax_d.set_title('(d) Magnitude Compression by Shell', fontweight='bold')
    ax_d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('e8_radial_dual_benchmark.png', dpi=180)
    plt.savefig('e8_radial_dual_benchmark.pdf')
    print("\nSaved: e8_radial_dual_benchmark.png / .pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main Orchestration
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()
    print("=" * 60)
    print("E8 RADIAL DUAL VERIFICATION BENCHMARK")
    print("Schmidt & Irwin (2026) — 6-Part Suite")
    print("=" * 60)

    results = {}

    # ── Generate E8 data ──
    print("\n[Setup] Generating E8 roots...")
    t0 = time.time()
    roots_frac = generate_e8_roots_frac()
    roots_np = generate_e8_roots_numpy()
    print(f"  240 roots generated ({time.time()-t0:.2f}s)")

    print("\n[Setup] Generating E8 lattice (max_norm_sq=10, coord_range=3)...")
    t0 = time.time()
    lattice_np = generate_e8_lattice(max_norm_sq=10, coord_range=3)
    print(f"  {len(lattice_np)} lattice points ({time.time()-t0:.1f}s)")

    # ── Part 1 ──
    t0 = time.time()
    p1_pass, table1_data = verify_table1()
    results['Part 1'] = p1_pass
    print(f"  [{time.time()-t0:.1f}s]")

    # ── Part 2 ──
    t0 = time.time()
    p2_pass, h4fold_data = verify_h4fold(roots_np)
    results['Part 2'] = p2_pass
    print(f"  [{time.time()-t0:.1f}s]")

    # ── Parts 3+4 ──
    t0 = time.time()
    p3_pass, p4_pass, norm_data = verify_norm_and_involution(lattice_np)
    results['Part 3'] = p3_pass
    results['Part 4'] = p4_pass
    print(f"  [{time.time()-t0:.1f}s]")

    # ── Part 5 ──
    t0 = time.time()
    p5_pass, adj_data = verify_adjacency(lattice_np, roots_frac)
    results['Part 5'] = p5_pass
    print(f"  [{time.time()-t0:.1f}s]")

    # ── Part 6 ──
    t0 = time.time()
    p6_pass, shell_data = compute_shell_compression(lattice_np)
    results['Part 6'] = p6_pass
    print(f"  [{time.time()-t0:.1f}s]")

    # ── Figure ──
    print("\n[Plot] Generating 4-panel figure...")
    make_figure(table1_data, h4fold_data, norm_data, shell_data)

    # ── Summary ──
    total_time = time.time() - t_total
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for part, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {part}: {status}")
    print(f"\n  OVERALL: {n_pass}/{n_total} PARTS PASSED")
    print(f"  Total runtime: {total_time:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
