#!/usr/bin/env python3
"""Quasi-harmonic approximation: thermal expansion and temperature-dependent bulk modulus.

Usage:  qha.py vol1.phmesh.json vol2.phmesh.json ... [--T 0 100 ... ] [--atoms-per-cubic-cell N]

Each file comes from one `task = phonons` run with `phonon_supercell` at a different lattice constant
(at least four volumes around the minimum). For every temperature the free energy per primitive cell
    F(V, T) = E(V) + 1/N_q sum_{q, modes} [ h nu / 2 + kT ln(1 - exp(-h nu / kT)) ]
is fitted with a Birch-Murnaghan equation of state, which gives V0(T), B0(T) and the thermal expansion.
Modes with imaginary or vanishing frequency are left out (and counted).
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eos import fit_birch_murnaghan, HA_BOHR3_TO_GPA, BOHR  # noqa: E402

HARTREE_EV = 27.211386245988
KB_EV = 8.617333262e-5
H_EV_PER_THZ = 4.135667696e-3


def vibrational_free_energy(mesh, T):
    """Per primitive cell, in eV; returns (F, number of skipped modes)."""
    kt = KB_EV * T
    f, skipped = 0.0, 0
    for modes in mesh:
        for nu in modes:
            if nu < 1e-3:
                skipped += 1
                continue
            e = H_EV_PER_THZ * nu
            f += 0.5 * e + (kt * math.log1p(-math.exp(-e / kt)) if T > 0 else 0.0)
    return f / len(mesh), skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--T", type=float, nargs="+", default=[0, 100, 200, 300, 400, 500, 600, 800, 1000])
    ap.add_argument("--atoms-per-cubic-cell", type=float, default=None)
    args = ap.parse_args()
    data = sorted((json.load(open(f)) for f in args.files), key=lambda d: d["volume"])
    if len(data) < 4:
        raise SystemExit("need phonon meshes at four or more volumes")
    volumes = [d["volume"] for d in data]
    natoms = data[0]["n_atoms"]
    skipped = max(vibrational_free_energy(d["frequencies_THz"], 300.0)[1] for d in data)
    if skipped:
        print(f"WARNING: up to {skipped} imaginary/zero modes per volume were left out")

    def equilibrium(T):
        f = [d["free_energy"] + vibrational_free_energy(d["frequencies_THz"], T)[0] / HARTREE_EV for d in data]
        return fit_birch_murnaghan(volumes, f)

    v_static = fit_birch_murnaghan(volumes, [d["free_energy"] for d in data])
    print(f"static lattice: V0 = {v_static[0] * BOHR ** 3:.4f} A^3, B0 = {v_static[2] * HA_BOHR3_TO_GPA:.2f} GPa")
    head = f"{'T (K)':>8s} {'V0 (A^3)':>10s}"
    if args.atoms_per_cubic_cell:
        head += f" {'a (A)':>9s}"
    print(head + f" {'B0 (GPa)':>9s} {'alpha_V (1e-6/K)':>17s} {'F (eV)':>12s}")
    for T in args.T:
        v0, f0, b0, _, _ = equilibrium(T)
        dT = 5.0
        vp, vm = equilibrium(T + dT)[0], equilibrium(max(T - dT, 0.0))[0]
        alpha = (vp - vm) / ((T + dT) - max(T - dT, 0.0)) / v0
        line = f"{T:8.0f} {v0 * BOHR ** 3:10.4f}"
        if args.atoms_per_cubic_cell:
            line += f" {(v0 / natoms * args.atoms_per_cubic_cell) ** (1 / 3) * BOHR:9.4f}"
        print(line + f" {b0 * HA_BOHR3_TO_GPA:9.2f} {alpha * 1e6:17.2f} {f0 * HARTREE_EV:12.5f}")
        if not volumes[0] < v0 < volumes[-1]:
            print("         WARNING: equilibrium volume outside of the sampled range")


if __name__ == "__main__":
    main()
