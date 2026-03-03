#!/usr/bin/env python3
"""
E8 Shelling Benchmark: θ-Function Polynomial vs Radial Dual Divisor Sum
========================================================================
Proves that the radial dual's modular substitution q → q⁻¹ reduces
shell-count computation from heavy polynomial expansion to a direct
arithmetic evaluation via the cubic divisor sum c₈(n) = 240 Σ_{d|n} d³.

Convention: "shell n" means E8 vectors of squared norm 2n (half-norm index),
consistent with the Eisenstein series E₄(q) = Σ r(n) qⁿ where q = e^{2πiτ}.
Shell n=1 → 240 roots, shell n=2 → 2160, etc.

Reference: Schmidt & Irwin (2026), Section 6.2 / Chapter 5 of CCT textbook.
"""

import time
import numpy as np
from scipy.signal import fftconvolve


# ═══════════════════════════════════════════════════════════════════
# Method 1: Standard θ-function polynomial expansion
# ═══════════════════════════════════════════════════════════════════
#
# Θ_{E8}(q) = ½(θ₂(p)⁸ + θ₃(p)⁸ + θ₄(p)⁸)   where p = e^{iπτ}, q = p²
#
# The coefficient of p^{2n} gives the shell count for squared norm 2n.
# We expand theta functions as polynomials in p to degree 2n, then
# read off the coefficient at index 2n.

def theta3_poly(max_deg):
    """θ₃(p) = 1 + 2Σ_{k≥1} p^{k²}, polynomial in nome p."""
    coeffs = np.zeros(max_deg + 1, dtype=np.float64)
    coeffs[0] = 1.0
    k = 1
    while k * k <= max_deg:
        coeffs[k * k] += 2.0
        k += 1
    return coeffs


def theta4_poly(max_deg):
    """θ₄(p) = 1 + 2Σ_{k≥1} (-1)^k p^{k²}."""
    coeffs = np.zeros(max_deg + 1, dtype=np.float64)
    coeffs[0] = 1.0
    k = 1
    while k * k <= max_deg:
        coeffs[k * k] += 2.0 * ((-1) ** k)
        k += 1
    return coeffs


def theta2_inner_poly(max_deg):
    """θ₂(p) = 2p^{1/4} Σ_{k≥0} p^{k²+k}.
    We return the inner sum Σ p^{k²+k} as integer-exponent polynomial.
    θ₂⁸ = 256 · p² · (inner)⁸."""
    coeffs = np.zeros(max_deg + 1, dtype=np.float64)
    k = 0
    while k * k + k <= max_deg:
        coeffs[k * k + k] += 1.0
        k += 1
    return coeffs


def polymul_trunc(a, b, max_deg):
    """FFT-based polynomial multiplication, truncated to max_deg."""
    result = fftconvolve(a, b)
    n = min(len(result), max_deg + 1)
    out = np.zeros(max_deg + 1)
    out[:n] = result[:n]
    return out


def power8_trunc(p, max_deg):
    """Raise polynomial to 8th power via repeated squaring (3 muls)."""
    p2 = polymul_trunc(p, p, max_deg)
    p4 = polymul_trunc(p2, p2, max_deg)
    p8 = polymul_trunc(p4, p4, max_deg)
    return p8


def shell_count_theta(n):
    """Compute E8 shell count for shell index n (squared norm 2n)
    via ½(θ₂⁸ + θ₃⁸ + θ₄⁸) expanded as polynomial in the nome p.
    We need coefficient at p^{2n}."""
    max_deg = 2 * n  # need terms up to p^{2n}

    # Build and raise each theta function to the 8th power
    t3_8 = power8_trunc(theta3_poly(max_deg), max_deg)
    t4_8 = power8_trunc(theta4_poly(max_deg), max_deg)

    # θ₂⁸ = 256 · p² · (inner)⁸
    inner = theta2_inner_poly(max_deg)
    inner_8 = power8_trunc(inner, max_deg)
    t2_8 = np.zeros(max_deg + 1)
    if max_deg >= 2:
        end = min(len(inner_8), max_deg - 1)
        t2_8[2:2 + end] = 256.0 * inner_8[:end]

    # Θ_{E8} = ½(θ₂⁸ + θ₃⁸ + θ₄⁸)
    theta_e8 = 0.5 * (t2_8 + t3_8 + t4_8)

    idx = 2 * n  # coefficient of p^{2n}
    return int(round(theta_e8[idx])) if idx < len(theta_e8) else 0


# ═══════════════════════════════════════════════════════════════════
# Method 1b: Exact integer polynomial (arbitrary precision)
# ═══════════════════════════════════════════════════════════════════
# This shows the TRUE cost when exact answers are needed (no float
# rounding), which is the realistic scenario for number-theoretic work.

def poly_mul_exact(a, b, max_deg):
    """Exact integer polynomial multiplication, truncated."""
    la, lb = len(a), len(b)
    out = [0] * (max_deg + 1)
    for i in range(min(la, max_deg + 1)):
        if a[i] == 0:
            continue
        for j in range(min(lb, max_deg + 1 - i)):
            if b[j] == 0:
                continue
            out[i + j] += a[i] * b[j]
    return out


def power8_exact(p, max_deg):
    """Raise integer polynomial to 8th power via repeated squaring."""
    p2 = poly_mul_exact(p, p, max_deg)
    p4 = poly_mul_exact(p2, p2, max_deg)
    p8 = poly_mul_exact(p4, p4, max_deg)
    return p8


def shell_count_theta_exact(n):
    """Exact integer arithmetic version of the θ-function expansion."""
    max_deg = 2 * n

    # θ₃ as exact integer list
    t3 = [0] * (max_deg + 1)
    t3[0] = 1
    k = 1
    while k * k <= max_deg:
        t3[k * k] += 2
        k += 1

    # θ₄
    t4 = [0] * (max_deg + 1)
    t4[0] = 1
    k = 1
    while k * k <= max_deg:
        t4[k * k] += 2 * ((-1) ** k)
        k += 1

    # θ₂ inner: Σ p^{k²+k}
    t2_inner = [0] * (max_deg + 1)
    k = 0
    while k * k + k <= max_deg:
        t2_inner[k * k + k] += 1
        k += 1

    t3_8 = power8_exact(t3, max_deg)
    t4_8 = power8_exact(t4, max_deg)
    t2_inner_8 = power8_exact(t2_inner, max_deg)

    # θ₂⁸ = 256 · p² · (inner)⁸
    t2_8_coeff = 0
    idx = 2 * n
    if idx >= 2:
        inner_idx = idx - 2
        if inner_idx < len(t2_inner_8):
            t2_8_coeff = 256 * t2_inner_8[inner_idx]

    total = t2_8_coeff + t3_8[idx] + t4_8[idx]
    return total // 2


# ═══════════════════════════════════════════════════════════════════
# Method 2: Radial Dual — cubic divisor sum
# ═══════════════════════════════════════════════════════════════════

def sigma3(n):
    """σ₃(n) = Σ_{d|n} d³ via trial division up to √n."""
    total = 0
    d = 1
    while d * d <= n:
        if n % d == 0:
            total += d * d * d
            other = n // d
            if other != d:
                total += other * other * other
        d += 1
    return total


def shell_count_dual(n):
    """E8 shell count for index n: c₈(n) = 240·σ₃(n)."""
    if n == 0:
        return 1
    return 240 * sigma3(n)


# ═══════════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("E8 SHELLING BENCHMARK")
    print("θ-Function Polynomial Expansion vs Radial Dual Divisor Sum")
    print("Convention: shell n = E8 vectors with squared norm 2n")
    print("=" * 72)

    # ── Validation ──
    print("\n[Validation] Comparing all three methods on shells n = 1..10:")
    known = {1: 240, 2: 2160, 3: 6720, 4: 17520, 5: 30240,
             6: 60480, 7: 82560, 8: 140400, 9: 181680, 10: 272160}

    print(f"  {'n':>3s}  {'θ-FFT':>10s}  {'θ-exact':>10s}  {'240·σ₃':>10s}  "
          f"{'Known':>10s}  {'OK':>4s}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*4}")

    all_match = True
    for n in range(1, 11):
        c_fft = shell_count_theta(n)
        c_exact = shell_count_theta_exact(n)
        c_dual = shell_count_dual(n)
        c_known = known[n]
        ok = (c_fft == c_exact == c_dual == c_known)
        if not ok:
            all_match = False
        print(f"  {n:>3d}  {c_fft:>10,d}  {c_exact:>10,d}  {c_dual:>10,d}  "
              f"{c_known:>10,d}  {'✓' if ok else '✗':>4s}")

    print(f"\n  All match known values: {'YES ✓' if all_match else 'NO ✗'}")

    # ── Timing benchmark ──
    test_shells = [100, 500, 1000, 2000, 5000]

    print(f"\n{'='*72}")
    print("BENCHMARK: Shell Multiplicity Computation Time")
    print(f"{'='*72}")
    print(f"\n  {'n':>6s}  {'θ-FFT':>11s}  {'θ-exact':>11s}  {'Divisor':>11s}  "
          f"{'FFT/Div':>9s}  {'Exact/Div':>10s}  {'c₈(n)':>22s}")
    print(f"  {'─'*6}  {'─'*11}  {'─'*11}  {'─'*11}  "
          f"{'─'*9}  {'─'*10}  {'─'*22}")

    results = []
    for n in test_shells:
        # Divisor sum (always instant)
        t0 = time.perf_counter()
        c_dual = shell_count_dual(n)
        t_dual = time.perf_counter() - t0

        # θ-FFT
        t0 = time.perf_counter()
        c_fft = shell_count_theta(n)
        t_fft = time.perf_counter() - t0

        # θ-exact (only up to n=2000, gets very slow beyond)
        if n <= 2000:
            t0 = time.perf_counter()
            c_exact = shell_count_theta_exact(n)
            t_exact = time.perf_counter() - t0
            exact_str = f"{t_exact:>10.4f}s"
            exact_ratio = t_exact / t_dual if t_dual > 0 else float('inf')
            exact_ratio_str = f"{exact_ratio:>9,.0f}×"
            match = (c_fft == c_exact == c_dual)
        else:
            t_exact = None
            exact_str = "(skipped)"
            exact_ratio_str = "—"
            match = (c_fft == c_dual)

        fft_ratio = t_fft / t_dual if t_dual > 0 else float('inf')

        marker = " ✓" if match else " ✗"
        print(f"  {n:>6,d}  {t_fft:>10.4f}s  {exact_str:>11s}  {t_dual:>10.6f}s  "
              f"{fft_ratio:>8.0f}×  {exact_ratio_str:>10s}  {c_dual:>22,d}{marker}")

        results.append({
            'n': n, 't_fft': t_fft, 't_exact': t_exact,
            't_dual': t_dual, 'count': c_dual, 'match': match,
            'fft_ratio': fft_ratio,
            'exact_ratio': exact_ratio if t_exact else None,
        })

    # ── Large-n divisor-only ──
    print(f"\n  Divisor sum at extreme shells (θ-expansion impractical):")
    for n in [10000, 50000, 100000]:
        t0 = time.perf_counter()
        c = shell_count_dual(n)
        t = time.perf_counter() - t0
        print(f"    n = {n:>7,d}:  c₈(n) = {c:>28,d}  ({t:.6f}s)")

    # ── Summary ──
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")

    r1000 = [r for r in results if r['n'] == 1000][0]
    print(f"\n  Shell n = 1000 (squared norm 2000):")
    print(f"    c₈(1000) = 240 · σ₃(1000) = {r1000['count']:,d}")
    print(f"    θ-FFT method:    {r1000['t_fft']:.4f}s   ({r1000['fft_ratio']:,.0f}× slower)")
    if r1000['t_exact']:
        print(f"    θ-exact method:  {r1000['t_exact']:.4f}s   "
              f"({r1000['exact_ratio']:,.0f}× slower)")
    print(f"    Divisor sum:     {r1000['t_dual']:.6f}s")

    all_ok = all(r['match'] for r in results)
    print(f"\n  All computed values agree: {'YES ✓' if all_ok else 'NO ✗'}")
    print(f"{'='*72}")


if __name__ == '__main__':
    main()
