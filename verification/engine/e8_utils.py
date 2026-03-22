"""Shared utilities for machine verification of
'Kinematic Spinors on Division-Algebra Root Systems.'

Provides: root system construction, Weyl reflections,
quaternionic Hopf fibration, and permutation algebra.
All arithmetic uses numpy float64 with exact-integer inputs;
correctness is validated by assertion checks throughout.
"""

import numpy as np
from itertools import product as iprod

TOL = 1e-10

# ================================================================
# Root system construction
# ================================================================

def build_e8_roots():
    """Return the 240 roots of E8 as an (240, 8) numpy array.

    Integer-type (112): +-e_i +- e_j, 0 <= i < j <= 7.
    Half-integer-type (128): (+-1/2)^8 with even number of minus signs.
    All roots have norm^2 = 2.
    """
    roots = []
    # Integer type: 112 roots
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1.0, -1.0]:
                for sj in [1.0, -1.0]:
                    r = [0.0] * 8
                    r[i] = si
                    r[j] = sj
                    roots.append(r)
    # Half-integer type: 128 roots
    for signs in iprod([0.5, -0.5], repeat=8):
        if sum(1 for x in signs if x < 0) % 2 == 0:
            roots.append(list(signs))
    roots = np.array(roots)
    assert len(roots) == 240, f"Expected 240 E8 roots, got {len(roots)}"
    norms_sq = np.sum(roots ** 2, axis=1)
    assert np.allclose(norms_sq, 2.0, atol=TOL), "Not all E8 roots have norm^2 = 2"
    return roots


def build_d4_roots():
    """Return the 24 roots of D4 as a (24, 4) numpy array.

    All +-e_i +- e_j for 0 <= i < j <= 3.
    """
    roots = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in [1.0, -1.0]:
                for sj in [1.0, -1.0]:
                    r = [0.0] * 4
                    r[i] = si
                    r[j] = sj
                    roots.append(r)
    roots = np.array(roots)
    assert len(roots) == 24, f"Expected 24 D4 roots, got {len(roots)}"
    norms_sq = np.sum(roots ** 2, axis=1)
    assert np.allclose(norms_sq, 2.0, atol=TOL), "Not all D4 roots have norm^2 = 2"
    return roots


def build_a2_roots():
    """Return the 6 roots of A2 as a (6, 2) numpy array.

    Standard embedding in R^2: the six vectors at 60-degree intervals
    with norm^2 = 2.
    """
    s = np.sqrt(3.0)
    roots = np.array([
        [ 1.0,  s],   #  60 deg direction
        [-1.0,  s],   # 120 deg direction
        [ 2.0, 0.0],  #   0 deg direction (= e1 - e2 in Dynkin basis)
        [-2.0, 0.0],  # 180 deg
        [ 1.0, -s],   # 300 deg
        [-1.0, -s],   # 240 deg
    ]) / np.sqrt(2.0)
    # Normalize: we want norm^2 = 2
    # The vectors above have norm^2 = 4/2 = 2.  Check:
    norms_sq = np.sum(roots ** 2, axis=1)
    # Actually let's use the standard A2 roots in the e1,e2 basis:
    # alpha1 = (1, -1/sqrt(3)), alpha2 = (0, 2/sqrt(3)), etc.
    # Simpler: use the 3D embedding then project to 2D.
    # Standard A2 in R^3: {e_i - e_j : i != j, i,j in {0,1,2}}
    roots_3d = np.array([
        [ 1, -1,  0],
        [-1,  1,  0],
        [ 0,  1, -1],
        [ 0, -1,  1],
        [ 1,  0, -1],
        [-1,  0,  1],
    ], dtype=float)
    # Project to the 2D subspace orthogonal to (1,1,1):
    # Basis: u1 = (1,-1,0)/sqrt(2), u2 = (1,1,-2)/sqrt(6)
    u1 = np.array([1, -1, 0]) / np.sqrt(2)
    u2 = np.array([1, 1, -2]) / np.sqrt(6)
    basis = np.column_stack([u1, u2])  # (3, 2)
    roots = roots_3d @ basis  # (6, 2)
    norms_sq = np.sum(roots ** 2, axis=1)
    assert np.allclose(norms_sq, 2.0, atol=TOL), "Not all A2 roots have norm^2 = 2"
    assert len(roots) == 6
    return roots


def build_a1_roots():
    """Return the 2 roots of A1 as a (2, 1) numpy array. Norm^2 = 2."""
    roots = np.array([[np.sqrt(2.0)], [-np.sqrt(2.0)]])
    norms_sq = np.sum(roots ** 2, axis=1)
    assert np.allclose(norms_sq, 2.0, atol=TOL)
    return roots


# ================================================================
# Root indexing
# ================================================================

def root_index_map(roots, decimals=6):
    """Build a dict mapping rounded-coordinate-tuple -> root index.

    Args:
        roots: (N, d) numpy array
        decimals: rounding precision
    Returns:
        dict of {tuple: int}
    """
    rmap = {}
    for i, r in enumerate(roots):
        key = tuple(np.round(r, decimals))
        assert key not in rmap, f"Duplicate root at index {i}: {key}"
        rmap[key] = i
    return rmap


def find_root(roots, rmap, v, decimals=6):
    """Find the index of vector v in the root system. Returns -1 if not found."""
    key = tuple(np.round(v, decimals))
    return rmap.get(key, -1)


# ================================================================
# Weyl reflections
# ================================================================

def weyl_reflect_vector(v, alpha):
    """Reflect vector v through the hyperplane perpendicular to root alpha.

    Formula: sigma_alpha(v) = v - (v . alpha) * alpha  [since alpha . alpha = 2].
    """
    return v - np.dot(v, alpha) * alpha


def weyl_reflection_perm(roots, alpha_idx, rmap, decimals=6):
    """Compute the Weyl reflection through root alpha as a permutation of root indices.

    Returns: numpy array of length N, where result[i] = index of sigma_alpha(root_i).
    """
    N = len(roots)
    alpha = roots[alpha_idx]
    perm = np.empty(N, dtype=int)
    for i in range(N):
        w = weyl_reflect_vector(roots[i], alpha)
        j = find_root(roots, rmap, w, decimals)
        assert j >= 0, f"Reflected root {i} through {alpha_idx} not found in system"
        perm[i] = j
    return perm


def all_weyl_reflections(roots, rmap=None, decimals=6):
    """Compute all distinct Weyl reflections as permutations.

    For E8 (240 roots), returns ~120 distinct reflections (antipodal roots
    give the same reflection).
    """
    if rmap is None:
        rmap = root_index_map(roots, decimals)
    N = len(roots)
    seen = set()
    refls = []
    for k in range(N):
        perm = weyl_reflection_perm(roots, k, rmap, decimals)
        pkey = tuple(perm)
        if pkey not in seen:
            seen.add(pkey)
            refls.append(perm)
    return refls


# ================================================================
# Quaternionic Hopf map
# ================================================================

def hopf_map_quat(v):
    """Quaternionic Hopf fibration S^7 -> S^4.

    Views R^8 = H^2:  q1 = (a,b,c,d) = v[0:4], q2 = (e,f,g,h) = v[4:8].
    Returns a 5-vector: (|q1|^2 - |q2|^2,  2*q1*conj(q2)).
    """
    a, b, c, d = v[0], v[1], v[2], v[3]
    e, f, g, h = v[4], v[5], v[6], v[7]
    r0 = a*a + b*b + c*c + d*d - (e*e + f*f + g*g + h*h)
    r1 = 2 * (a*e + b*f + c*g + d*h)
    r2 = 2 * (b*e - a*f - d*g + c*h)
    r3 = 2 * (c*e + d*f - a*g - b*h)
    r4 = 2 * (d*e - c*f + b*g - a*h)
    return np.array([r0, r1, r2, r3, r4])


def cluster_by_hopf(roots, tol=0.01):
    """Cluster E8 roots by their quaternionic Hopf image.

    Returns: list of 10 lists, each containing 24 root indices.
    """
    images = np.array([hopf_map_quat(roots[i]) for i in range(len(roots))])
    clusters = {}
    for i in range(len(roots)):
        key = tuple(np.round(images[i] / tol) * tol)
        clusters.setdefault(key, []).append(i)
    shells = [sorted(clusters[k]) for k in sorted(clusters.keys())]
    assert len(shells) == 10, f"Expected 10 Hopf clusters, got {len(shells)}"
    for s in shells:
        assert len(s) == 24, f"Cluster has {len(s)} roots, expected 24"
    all_idx = sorted(idx for s in shells for idx in s)
    assert all_idx == list(range(len(roots))), "Clusters don't partition all roots"
    return shells


# ================================================================
# Permutation algebra
# ================================================================

def compose_perms(p1, p2):
    """Compose two permutations: apply p1 first, then p2.

    result[i] = p2[p1[i]].
    """
    return tuple(p2[p1[i]] for i in range(len(p1)))


def inverse_perm(p):
    """Compute the inverse permutation."""
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(inv)


def identity_perm(n):
    """Return the identity permutation of length n."""
    return tuple(range(n))


def perm_order(p):
    """Compute the order of a permutation (smallest k > 0 with p^k = id)."""
    n = len(p)
    current = tuple(p)
    ident = identity_perm(n)
    for k in range(1, 10000):
        if current == ident:
            return k
        current = compose_perms(current, p)
    raise RuntimeError("Permutation order exceeds 10000")


def perm_sign(p):
    """Compute the sign (parity) of a permutation. Returns +1 or -1."""
    n = len(p)
    visited = [False] * n
    sign = 1
    for i in range(n):
        if visited[i]:
            continue
        cycle_len = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = p[j]
            cycle_len += 1
        if cycle_len % 2 == 0:
            sign *= -1
    return sign


# ================================================================
# Root system verification
# ================================================================

def verify_root_system(roots, expected_count, expected_rank):
    """Verify basic root system properties.

    Checks: count, norm^2 = 2, rank, inner products in {-2,-1,0,1,2},
    closure under negation.
    """
    N, d = roots.shape
    ok = True
    if N != expected_count:
        return False, f"Count {N} != {expected_count}"
    norms_sq = np.sum(roots ** 2, axis=1)
    if not np.allclose(norms_sq, 2.0, atol=TOL):
        return False, "Not all roots have norm^2 = 2"
    rank = np.linalg.matrix_rank(roots, tol=1e-8)
    if rank != expected_rank:
        return False, f"Rank {rank} != {expected_rank}"
    # Check inner products
    G = roots @ roots.T
    for i in range(N):
        for j in range(i + 1, N):
            ip = G[i, j]
            if not any(abs(ip - v) < TOL for v in [-2, -1, 0, 1, 2]):
                return False, f"Inner product {ip} at ({i},{j}) not in {{-2,-1,0,1,2}}"
    # Check closure under negation
    rmap = root_index_map(roots)
    for i in range(N):
        neg = find_root(roots, rmap, -roots[i])
        if neg < 0:
            return False, f"Negation of root {i} not found"
    return True, "OK"


def verify_d4_cartan(sub_roots):
    """Verify that a 24-root system in R^k has D4 Cartan matrix.

    Finds simple roots and checks the Cartan matrix matches D4.
    """
    # D4 Cartan matrix (standard labeling: central node is index 1):
    #    2  -1   0   0
    #   -1   2  -1  -1
    #    0  -1   2   0
    #    0  -1   0   2
    d4_cartan = np.array([
        [ 2, -1,  0,  0],
        [-1,  2, -1, -1],
        [ 0, -1,  2,  0],
        [ 0, -1,  0,  2],
    ])
    # Find simple roots by picking a positive system
    # A simple root: not expressible as sum of two positive roots
    # Use: pick a generic vector c and take roots with positive c-dot-product
    d = sub_roots.shape[1]
    c = np.random.RandomState(42).randn(d)
    dots = sub_roots @ c
    positive = [i for i in range(len(sub_roots)) if dots[i] > TOL]
    # Simple roots: positive roots that cannot be written as sum of two other positive roots
    pos_set = set(positive)
    rmap = root_index_map(sub_roots)
    simple = []
    for i in positive:
        is_simple = True
        for j in positive:
            for k in positive:
                if j >= k:
                    continue
                s = sub_roots[j] + sub_roots[k]
                if np.allclose(s, sub_roots[i], atol=TOL):
                    is_simple = False
                    break
            if not is_simple:
                break
        if is_simple:
            simple.append(i)
    if len(simple) != 4:
        return False, f"Found {len(simple)} simple roots, expected 4"
    # Compute Cartan matrix
    S = sub_roots[simple]
    C = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            C[i, j] = 2 * np.dot(S[i], S[j]) / np.dot(S[j], S[j])
    C = np.round(C).astype(int)
    # The Cartan matrix is unique up to relabeling of simple roots.
    # Check if C is D4 under some permutation of rows/columns.
    from itertools import permutations
    for perm in permutations(range(4)):
        P = C[np.ix_(list(perm), list(perm))]
        if np.array_equal(P, d4_cartan):
            return True, "D4 Cartan matrix verified"
    return False, f"Cartan matrix {C} does not match D4 under any permutation"


# ================================================================
# Perpendicular pair detection
# ================================================================

def find_perpendicular_pairs(shells, roots):
    """Find pairs of shells whose 4D subspaces are orthogonal.

    Returns list of (i, j) pairs with i < j.
    """
    spans = []
    for sh in shells:
        sub = roots[sh]
        _, S, Vt = np.linalg.svd(sub, full_matrices=False)
        spans.append(Vt[:4])  # 4D span
    pairs = []
    for i in range(len(shells)):
        for j in range(i + 1, len(shells)):
            cross = spans[i] @ spans[j].T
            if np.linalg.norm(cross) < 0.01:
                pairs.append((i, j))
    return pairs
