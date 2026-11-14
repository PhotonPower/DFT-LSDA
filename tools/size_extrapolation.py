#!/usr/bin/env python3
"""Extrapolation of point-defect energies to infinite supercell size.

The elastic interaction of a defect with its periodic images, and the homogeneous strain of a supercell
kept at the volume of the perfect crystal, both decay as 1/V. For supercells of several sizes the
energies are therefore fitted to
    E(V) = E_inf + a / V
(exact through two sizes, least squares for more).

Usage:  size_extrapolation.py DIR1 DIR2 [DIR3 ...]
Every directory holds the results of examples/defects/run_defects.sh for one supercell size:
none.json, vac_tm.json, vac_al.json, al_tm.json, tm_al.json.
Printed are the composition-conserving reaction energies for each size and their extrapolation; the raw
energies of single defects depend on the reference of the chemical potentials and are not extrapolated.
"""
import json
import os
import sys

HARTREE_EV = 27.211386245988
BOHR3_A3 = 0.529177210903 ** 3


def fit_inverse_volume(volumes, values):
    """Least-squares fit y = y_inf + a/V; returns (y_inf, a)."""
    xs = [1.0 / v for v in volumes]
    n = len(xs)
    sx, sy = sum(xs), sum(values)
    sxx, sxy = sum(x * x for x in xs), sum(x * y for x, y in zip(xs, values))
    det = n * sxx - sx * sx
    if abs(det) < 1e-300:
        raise ValueError("need at least two different supercell volumes")
    a = (n * sxy - sx * sy) / det
    return (sy - a * sx) / n, a


def reactions(directory):
    def load(name):
        r = json.load(open(os.path.join(directory, name + ".json")))
        if not r.get("converged", False):
            print(f"WARNING: {directory}/{name}.json is not converged")
        return r

    perfect = load("none")
    e = lambda r: r.get("energy_sigma0", r["free_energy"]) * HARTREE_EV
    n_cells = perfect["n_atoms"] // 2
    eps0 = e(perfect) / n_cells
    d = {k: e(load(k)) - e(perfect) for k in ("vac_tm", "vac_al", "al_tm", "tm_al")}
    return perfect["volume"] * BOHR3_A3, perfect["n_atoms"], {
        "triple defect 2 V_TM + TM_Al": 2 * d["vac_tm"] + d["tm_al"] + eps0,
        "triple defect 2 V_Al + Al_TM": 2 * d["vac_al"] + d["al_tm"] + eps0,
        "Schottky pair V_TM + V_Al": d["vac_tm"] + d["vac_al"] + eps0,
        "antisite pair Al_TM + TM_Al": d["al_tm"] + d["tm_al"],
    }


def main():
    dirs = sys.argv[1:]
    if len(dirs) < 2 or dirs[0] in ("-h", "--help"):
        raise SystemExit(__doc__)
    data = sorted((reactions(d) for d in dirs), key=lambda t: t[0])
    volumes = [t[0] for t in data]
    print("supercells: " + ", ".join(f"{t[1]} atoms ({t[0]:.1f} A^3)" for t in data))
    print(f"{'reaction energy [eV]':32s}" + "".join(f"{t[1]:>10d}" for t in data) + f"{'V -> inf':>12s}{'a [eV A^3]':>14s}")
    for name in data[0][2]:
        values = [t[2][name] for t in data]
        y_inf, a = fit_inverse_volume(volumes, values)
        print(f"{name:32s}" + "".join(f"{v:10.4f}" for v in values) + f"{y_inf:12.4f}{a:14.2f}")
    if len(data) == 2:
        print("note: two sizes determine the fit exactly; a third size is needed to judge its quality")


if __name__ == "__main__":
    main()
