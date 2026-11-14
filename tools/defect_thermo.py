#!/usr/bin/env python3
"""Point-defect thermodynamics of an ordered B2 compound A-B (A = transition metal on sublattice
alpha, B = Al on sublattice beta) from supercell total energies.

Method: grand-canonical statistical model for non-interacting point defects in ordered compounds
(Mayer, Elsaesser, Faehnle, phys. stat. sol. (b) 191, 283 (1995); Fu, Ye, Yoo, Ho, PRB 48, 6712 (1993);
Korzhavyi et al., PRB 61, 6003 (2000)).  For N unit cells with n_d defects of type
d in {V_alpha, B_alpha, V_beta, A_beta},

    E = N eps0 + sum_d n_d eps_d ,     eps_d = E_supercell(defect d) - E_supercell(perfect),

the free energy with the configurational entropy is minimized at fixed numbers of A and B atoms
(the number of unit cells N is free).  This gives, with x_alpha = 1 - c_Va - c_Ba, x_beta = 1 - c_Vb - c_Ab,

    c_Va/x_alpha = exp[-(eps_Va + mu_A)/kT]            c_Vb/x_beta = exp[-(eps_Vb + mu_B)/kT]
    c_Ba/x_alpha = exp[-(eps_Ba + mu_A - mu_B)/kT]     c_Ab/x_beta = exp[-(eps_Ab - mu_A + mu_B)/kT]
    eps0 - mu_A - mu_B + kT ln(x_alpha x_beta) = 0
    (x_alpha + c_Ab) / (x_alpha + c_Ab + x_beta + c_Ba) = x_A      (composition)

which is solved for the chemical potentials mu_A, mu_B.  Effective formation energies are the
Arrhenius slopes  E_eff = -d ln c_d / d(1/kT)  at fixed composition.

Usage:
  defect_thermo.py --perfect perfect.json --vac-a vac_tm.json --vac-b vac_al.json \
                   --b-on-a al_tm.json --a-on-b tm_al.json [--T 1000 1200 1400] [--x 0.48 0.50 0.52]
All supercells must have the same size and computational parameters (cutoffs, k-points, smearing).
Formation volumes and vibrational entropies are neglected (p = 0, configurational entropy only).
"""
import argparse
import json
import math

HARTREE_EV = 27.211386245988
KB_EV = 8.617333262e-5


def expc(x):
    return math.exp(max(-700.0, min(700.0, x)))


class B2Model:
    def __init__(self, eps0, e_va, e_ba, e_vb, e_ab):
        self.eps0, self.e = eps0, {"V_a": e_va, "B_a": e_ba, "V_b": e_vb, "A_b": e_ab}

    def concentrations(self, mu_a, mu_b, kt):
        e = self.e
        r_va = expc(-(e["V_a"] + mu_a) / kt)
        r_ba = expc(-(e["B_a"] + mu_a - mu_b) / kt)
        r_vb = expc(-(e["V_b"] + mu_b) / kt)
        r_ab = expc(-(e["A_b"] - mu_a + mu_b) / kt)
        xa = 1.0 / (1.0 + r_va + r_ba)
        xb = 1.0 / (1.0 + r_vb + r_ab)
        return {"V_a": xa * r_va, "B_a": xa * r_ba, "V_b": xb * r_vb, "A_b": xb * r_ab, "x_a": xa, "x_b": xb}

    def solve(self, x_a, T):
        """Chemical potentials and defect concentrations (per unit cell) at atomic fraction x_a and temperature T."""
        kt = KB_EV * T

        def mus(d):  # for given mu_A - mu_B = d solve the lattice equation for s = mu_A + mu_B
            lo, hi = self.eps0 - 20.0, self.eps0 + 20.0
            for _ in range(200):
                s = 0.5 * (lo + hi)
                c = self.concentrations(0.5 * (s + d), 0.5 * (s - d), kt)
                g = self.eps0 - s + kt * math.log(c["x_a"] * c["x_b"])
                if g > 0:
                    lo = s
                else:
                    hi = s
            s = 0.5 * (lo + hi)
            return 0.5 * (s + d), 0.5 * (s - d)

        def composition(d):
            ma, mb = mus(d)
            c = self.concentrations(ma, mb, kt)
            na, nb = c["x_a"] + c["A_b"], c["x_b"] + c["B_a"]
            return na / (na + nb)

        # mu_A - mu_B is of the order of the antisite energies (absolute total energies are large numbers)
        d0 = 0.5 * (self.e["A_b"] - self.e["B_a"])
        lo, hi = d0 - 40.0, d0 + 40.0
        for _ in range(200):
            d = 0.5 * (lo + hi)
            if composition(d) < x_a:
                lo = d
            else:
                hi = d
        ma, mb = mus(0.5 * (lo + hi))
        return ma, mb, self.concentrations(ma, mb, kt)

    def effective_energies(self, x_a, T, dT=2.0):
        out = {}
        _, _, c1 = self.solve(x_a, T - dT)
        _, _, c2 = self.solve(x_a, T + dT)
        b1, b2 = 1.0 / (KB_EV * (T - dT)), 1.0 / (KB_EV * (T + dT))
        for d in self.e:
            out[d] = -(math.log(c2[d]) - math.log(c1[d])) / (b2 - b1) if c1[d] > 0 and c2[d] > 0 else float("nan")
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perfect", required=True)
    ap.add_argument("--vac-a", required=True, help="vacancy on the transition-metal sublattice")
    ap.add_argument("--vac-b", required=True, help="vacancy on the Al sublattice")
    ap.add_argument("--b-on-a", required=True, help="Al antisite atom on the transition-metal sublattice")
    ap.add_argument("--a-on-b", required=True, help="transition-metal antisite atom on the Al sublattice")
    ap.add_argument("--T", type=float, nargs="+", default=[800.0, 1000.0, 1200.0, 1400.0])
    ap.add_argument("--x", type=float, nargs="+", default=[0.48, 0.49, 0.50, 0.51, 0.52],
                    help="atomic fraction of the transition metal")
    ap.add_argument("--names", nargs=2, default=["TM", "Al"])
    args = ap.parse_args()

    def load(path):
        r = json.load(open(path))
        if not r.get("converged", False):
            print(f"WARNING: {path} is not converged")
        return r

    # energies extrapolated to zero smearing (identical to the free energy for cold smearing)
    energy = lambda r: r.get("energy_sigma0", r["free_energy"]) * HARTREE_EV
    perfect = load(args.perfect)
    n_cells = perfect["n_atoms"] // 2
    e_perf = energy(perfect)
    eps0 = e_perf / n_cells
    files = {"V_a": args.vac_a, "B_a": args.b_on_a, "V_b": args.vac_b, "A_b": args.a_on_b}
    expected_atoms = {"V_a": -1, "B_a": 0, "V_b": -1, "A_b": 0}
    eps = {}
    for d, path in files.items():
        r = load(path)
        if r["n_atoms"] != perfect["n_atoms"] + expected_atoms[d]:
            raise SystemExit(f"{path}: unexpected number of atoms for defect {d}")
        eps[d] = energy(r) - e_perf
    A, B = args.names
    label = {"V_a": f"V_{A}", "B_a": f"{B}_{A}", "V_b": f"V_{B}", "A_b": f"{A}_{B}"}

    print(f"Supercell: {n_cells} unit cells, eps0 = {eps0:.6f} eV per formula unit")
    print("\nRaw defect energies eps_d = E(defect cell) - E(perfect cell)  [eV]")
    for d in eps:
        print(f"  {label[d]:8s} {eps[d]:14.6f}")

    print("\nComposition-conserving defect reactions (independent of the chemical potentials) [eV]")
    reactions = [
        (f"triple defect  0 -> 2 {label['V_a']} + {label['A_b']}", 2 * eps["V_a"] + eps["A_b"] + eps0),
        (f"triple defect  0 -> 2 {label['V_b']} + {label['B_a']}", 2 * eps["V_b"] + eps["B_a"] + eps0),
        (f"Schottky pair  0 -> {label['V_a']} + {label['V_b']}", eps["V_a"] + eps["V_b"] + eps0),
        (f"antisite pair  0 -> {label['B_a']} + {label['A_b']}", eps["B_a"] + eps["A_b"]),
    ]
    for name, e in reactions:
        print(f"  {name:42s} {e:10.4f}")
    print(f"\nConstitutional defects at T -> 0 (energy per excess atom, relative) [eV]")
    # A-rich: one excess A atom either as antisite A_b (plus 1/2 formula unit removed) or as two V_b... compare per excess atom
    #   antisite:   A_b   changes (N_A, N_B) by (+1, -1)  -> +2 A excess per defect at fixed N: cost eps_Ab / 2 per unit deviation
    #   vacancy :   V_b   changes (N_A, N_B) by (0, -1) and needs eps0/2 ... -> cost (eps_Vb + eps0/2) per unit deviation
    a_rich = {label["A_b"]: eps["A_b"] / 2.0, label["V_b"]: eps["V_b"] + eps0 / 2.0}
    b_rich = {label["B_a"]: eps["B_a"] / 2.0, label["V_a"]: eps["V_a"] + eps0 / 2.0}
    for side, opts in ((f"{A}-rich", a_rich), (f"{B}-rich", b_rich)):
        best = min(opts, key=opts.get)
        ref = opts[best]
        txt = ", ".join(f"{k}: {v - ref:+.4f}" for k, v in opts.items())
        print(f"  {side}: stable constitutional defect = {best}   ({txt})")

    model = B2Model(eps0, eps["V_a"], eps["B_a"], eps["V_b"], eps["A_b"])
    order = ["V_a", "B_a", "V_b", "A_b"]
    for T in args.T:
        print(f"\nT = {T:.0f} K   (concentrations per sublattice site; E_eff in eV)")
        print("   x_%-3s " % A + " ".join(f"{'c(' + label[d] + ')':>12s}" for d in order) + "  " +
              " ".join(f"{'E(' + label[d] + ')':>9s}" for d in order) + f"   mu_{A}-mu_{B}")
        for x in args.x:
            ma, mb, c = model.solve(x, T)
            ee = model.effective_energies(x, T)
            print(f"   {x:.4f} " + " ".join(f"{c[d]:12.4e}" for d in order) + "  " +
                  " ".join(f"{ee[d]:9.4f}" for d in order) + f"   {ma - mb:9.4f}")


if __name__ == "__main__":
    main()
