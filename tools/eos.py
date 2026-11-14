#!/usr/bin/env python3
"""Birch-Murnaghan equation-of-state fit (3rd order).

Usage:  eos.py result1.json result2.json ...      (output files of lsda)
        eos.py --data file.txt                    (two columns: volume [bohr^3], energy [Ha])

The energy used is the free energy E-TS. Prints V0, the lattice constant of a cubic cell with
`--atoms-per-cubic-cell` atoms (if given), the bulk modulus B0 and its pressure derivative.
"""
import argparse
import json

HA_BOHR3_TO_GPA = 29421.02648438959
BOHR = 0.529177210903


def solve(a, b):
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    a = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[p] = a[p], a[i]
        for r in range(i + 1, n):
            f = a[r][i] / a[i][i]
            for c in range(i, n + 1):
                a[r][c] -= f * a[i][c]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (a[i][n] - sum(a[i][c] * x[c] for c in range(i + 1, n))) / a[i][i]
    return x


def fit_birch_murnaghan(volumes, energies):
    """E is a cubic polynomial in x = V^(-2/3). Returns (V0, E0, B0 [Ha/bohr^3], B0', rms)."""
    if len(volumes) < 4:
        raise ValueError("need at least 4 points for a Birch-Murnaghan fit")
    vref = sum(volumes) / len(volumes)
    eref = min(energies)
    xs = [(v / vref) ** (-2.0 / 3.0) for v in volumes]
    ys = [e - eref for e in energies]
    ata = [[sum(x ** (i + j) for x in xs) for j in range(4)] for i in range(4)]
    aty = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(4)]
    c = solve(ata, aty)
    e = lambda v: sum(ci * ((v / vref) ** (-2.0 / 3.0)) ** i for i, ci in enumerate(c))
    # minimum by golden-section search inside the sampled range (slightly extended)
    lo, hi = 0.9 * min(volumes), 1.1 * max(volumes)
    g = (5 ** 0.5 - 1) / 2
    for _ in range(200):
        m1, m2 = hi - g * (hi - lo), lo + g * (hi - lo)
        if e(m1) < e(m2):
            hi = m2
        else:
            lo = m1
    v0 = 0.5 * (lo + hi)
    h = 1e-3 * v0
    d2 = (e(v0 + h) - 2 * e(v0) + e(v0 - h)) / h ** 2
    d3 = (e(v0 + 2 * h) - 2 * e(v0 + h) + 2 * e(v0 - h) - e(v0 - 2 * h)) / (2 * h ** 3)
    b0 = v0 * d2
    b0p = -(d2 + v0 * d3) / d2
    rms = (sum((e(v) - y) ** 2 for v, y in zip(volumes, ys)) / len(ys)) ** 0.5
    return v0, e(v0) + eref, b0, b0p, rms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--data")
    ap.add_argument("--atoms-per-cubic-cell", type=float, default=None,
                    help="e.g. 4 (fcc), 2 (bcc, B2): prints the cubic lattice constant")
    args = ap.parse_args()
    pts, natoms = [], None
    if args.data:
        for line in open(args.data):
            t = line.split("#")[0].split()
            if len(t) >= 2:
                pts.append((float(t[0]), float(t[1])))
    for f in args.files:
        r = json.load(open(f))
        pts.append((r["volume"], r["free_energy"]))
        natoms = r["n_atoms"]
    pts.sort()
    v0, e0, b0, b0p, rms = fit_birch_murnaghan([p[0] for p in pts], [p[1] for p in pts])
    print(f"points          : {len(pts)}   (V = {pts[0][0]:.3f} ... {pts[-1][0]:.3f} bohr^3)")
    print(f"V0              : {v0:.5f} bohr^3 = {v0 * BOHR ** 3:.5f} A^3")
    if natoms and args.atoms_per_cubic_cell:
        a0 = (v0 / natoms * args.atoms_per_cubic_cell) ** (1.0 / 3.0)
        print(f"a0 (cubic)      : {a0:.5f} bohr = {a0 * BOHR:.5f} A")
    print(f"E0              : {e0:.8f} Ha")
    print(f"B0              : {b0 * HA_BOHR3_TO_GPA:.2f} GPa")
    print(f"B0'             : {b0p:.3f}")
    print(f"rms fit error   : {rms:.2e} Ha")
    if not (min(p[0] for p in pts) < v0 < max(p[0] for p in pts)):
        print("WARNING: the minimum lies outside of the sampled volume range")


if __name__ == "__main__":
    main()
