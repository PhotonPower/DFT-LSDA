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

Pressure and formation volumes.  At pressure p the energies are replaced by enthalpies,
eps_d -> eps_d + p v_d and eps0 -> eps0 + p Omega0, with the raw volume changes
v_d = V(defect cell) - V(perfect cell) and Omega0 = V(perfect cell)/N.  Effective formation volumes are
V_eff = -kT d ln c_d / dp.  The volumes are taken from
  * supercells relaxed to zero pressure (task = cellrelax): v_d directly from the cell volumes, or
  * supercells at the fixed volume of the perfect cell with stresses (stress = true) and --bulk-modulus B:
    v_d = V (p_d - p_0)/B from the residual pressure, and the energies are corrected to zero pressure by
    the elastic relaxation energy -V (p_d - p_0)^2 / (2 B)   (first-order elastic estimate).

Defect complexes (--complex NAME FILE MULTIPLICITY [PARTS]).  A bound complex (e.g. a nearest-neighbour
vacancy-antisite pair or the bound triple defect) is treated as an additional dilute defect species with
its own supercell: eps_k = E(complex cell) - E(perfect cell), the changes dA, dB of the atom numbers are
read from the files, and
    c_k = g_k exp[-(eps_k - dA mu_A - dB mu_B)/kT]
with the number g_k of distinct orientations per unit cell (MULTIPLICITY).  Complexes enter the particle
balance and the lattice equation (- kT sum_k c_k).  PARTS (optional, e.g. V_a,V_a,A_b) names the single
defects the complex consists of; the binding energy eps_k - sum eps_parts is then printed (negative =
bound).  Keys of the single defects: V_a, B_a, V_b, A_b.

Vibrations (--vibrations).  If every supercell has Gamma-point phonon frequencies (task = phonons, file
<prefix>.phonons.json next to <prefix>.json), the harmonic vibrational free energy
    F_vib(T) = sum_modes [ h nu / 2 + kT ln(1 - exp(-h nu / kT)) ]        (three acoustic modes omitted)
is added to every supercell energy, so that eps_d and eps0 become temperature dependent and the
concentrations contain the vibrational formation entropies.  Without this option only the
configurational entropy is taken into account.
"""
import argparse
import json
import math

HARTREE_EV = 27.211386245988
KB_EV = 8.617333262e-5
BOHR3_A3 = 0.529177210903 ** 3
GPA_A3_EV = 6.241509074e-3   # 1 GPa * 1 A^3 in eV
H_EV_PER_THZ = 4.135667696e-3  # h * 1 THz in eV


def expc(x):
    return math.exp(max(-700.0, min(700.0, x)))


def vibrational_free_energy(freqs_thz, T):
    """Harmonic free energy (eV) of the Gamma-point modes; the three modes of smallest |nu| are acoustic."""
    modes = sorted(freqs_thz, key=abs)[3:]
    kt = KB_EV * T
    f = 0.0
    for nu in modes:
        if nu <= 0.0:
            continue  # imaginary mode (reported by the caller)
        e = H_EV_PER_THZ * nu
        f += 0.5 * e + (kt * math.log1p(-math.exp(-e / kt)) if kt > 0 else 0.0)
    return f


class B2Model:
    def __init__(self, eps0, e_va, e_ba, e_vb, e_ab, complexes=()):
        self.eps0, self.e = eps0, {"V_a": e_va, "B_a": e_ba, "V_b": e_vb, "A_b": e_ab}
        self.complexes = list(complexes)  # dicts with name, eps, dA, dB, g, vol

    def at_pressure(self, p_gpa, omega0, vol):
        """Model with enthalpies at pressure p (GPa); omega0 and vol[d] in A^3."""
        f = p_gpa * GPA_A3_EV
        cx = [dict(c, eps=c["eps"] + f * c.get("vol", 0.0)) for c in self.complexes]
        return B2Model(self.eps0 + f * omega0, *(self.e[d] + f * vol[d] for d in ("V_a", "B_a", "V_b", "A_b")), complexes=cx)

    def complex_concentrations(self, mu_a, mu_b, kt):
        return {c["name"]: c["g"] * expc(-(c["eps"] - c["dA"] * mu_a - c["dB"] * mu_b) / kt) for c in self.complexes}

    def concentrations(self, mu_a, mu_b, kt):
        e = self.e
        r_va = expc(-(e["V_a"] + mu_a) / kt)
        r_ba = expc(-(e["B_a"] + mu_a - mu_b) / kt)
        r_vb = expc(-(e["V_b"] + mu_b) / kt)
        r_ab = expc(-(e["A_b"] - mu_a + mu_b) / kt)
        xa = 1.0 / (1.0 + r_va + r_ba)
        xb = 1.0 / (1.0 + r_vb + r_ab)
        out = {"V_a": xa * r_va, "B_a": xa * r_ba, "V_b": xb * r_vb, "A_b": xb * r_ab, "x_a": xa, "x_b": xb,
               "ln_x": -math.log1p(r_va + r_ba) - math.log1p(r_vb + r_ab)}  # ln(x_a x_b) without underflow
        out["complex"] = self.complex_concentrations(mu_a, mu_b, kt)
        return out

    def solve(self, x_a, T):
        """Chemical potentials and defect concentrations (per unit cell) at atomic fraction x_a and temperature T."""
        kt = KB_EV * T

        def lattice(s, d):
            c = self.concentrations(0.5 * (s + d), 0.5 * (s - d), kt)
            return self.eps0 - s + kt * c["ln_x"] - kt * sum(c["complex"].values())

        def mus(d):  # for given mu_A - mu_B = d solve the lattice equation for s = mu_A + mu_B
            # The physical (dilute) solution is the first root below eps0. Without complexes the left-hand
            # side is monotonic in s; complexes that remove formula units grow like exp(-s/kT). The root is
            # therefore bracketed by stepping down from eps0 with growing steps (fine near eps0).
            hi = self.eps0 if lattice(self.eps0, d) <= 0 else self.eps0 + 50.0
            lo, step = hi, 0.05 * kt
            while lattice(lo, d) <= 0 and lo > self.eps0 - 500.0:
                hi = lo
                lo -= step
                step *= 1.5
            for _ in range(100):
                s = 0.5 * (lo + hi)
                if lattice(s, d) > 0:
                    lo = s
                else:
                    hi = s
            s = 0.5 * (lo + hi)
            return 0.5 * (s + d), 0.5 * (s - d)

        def composition(d):
            ma, mb = mus(d)
            c = self.concentrations(ma, mb, kt)
            na, nb = c["x_a"] + c["A_b"], c["x_b"] + c["B_a"]
            for cx in self.complexes:  # atoms added or removed by the complexes
                na += c["complex"][cx["name"]] * cx["dA"]
                nb += c["complex"][cx["name"]] * cx["dB"]
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
        if abs(lattice(ma + mb, ma - mb)) > 1e-6:
            raise SystemExit("no dilute solution of the lattice equation: a defect or complex has a "
                             "negative formation energy at these conditions")
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
    ap.add_argument("--p", type=float, nargs="+", default=[0.0], help="pressures in GPa")
    ap.add_argument("--complex", nargs="+", action="append", default=[], metavar="ARG",
                    help="NAME FILE MULTIPLICITY [PARTS]: bound defect complex with its own supercell (see above)")
    ap.add_argument("--vibrations", action="store_true",
                    help="add harmonic vibrational free energies from <prefix>.phonons.json of every supercell")
    ap.add_argument("--bulk-modulus", type=float, default=None,
                    help="bulk modulus in GPa: estimate relaxation volumes of fixed-volume supercells from their pressure")
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
    eps, vol = {}, {}
    v_perf = perfect["volume"] * BOHR3_A3
    omega0 = v_perf / n_cells
    volume_mode = None
    for d, path in files.items():
        r = load(path)
        if r["n_atoms"] != perfect["n_atoms"] + expected_atoms[d]:
            raise SystemExit(f"{path}: unexpected number of atoms for defect {d}")
        eps[d] = energy(r) - e_perf
        dv = r["volume"] * BOHR3_A3 - v_perf
        if abs(dv) > 1e-6 * v_perf:
            mode, vol[d] = "relaxed cells", dv
        elif args.bulk_modulus and "pressure_GPa" in r and "pressure_GPa" in perfect:
            # fixed volume: relaxation volume and elastic energy correction from the residual pressure
            dp = r["pressure_GPa"] - perfect["pressure_GPa"]
            mode, vol[d] = "residual pressure", v_perf * dp / args.bulk_modulus
            eps[d] -= 0.5 * v_perf * dp * dp / args.bulk_modulus * GPA_A3_EV
        else:
            mode, vol[d] = None, 0.0
        if volume_mode not in (None, mode) and mode != volume_mode:
            raise SystemExit("inconsistent supercells: some are relaxed to zero pressure, others are not")
        volume_mode = mode
    A, B = args.names
    label = {"V_a": f"V_{A}", "B_a": f"{B}_{A}", "V_b": f"V_{B}", "A_b": f"{A}_{B}"}

    def count(r, name):
        return sum(1 for a in r.get("atoms", []) if a["species"].rstrip("*") == name)

    complexes = []
    for spec in args.complex:
        if len(spec) not in (3, 4):
            raise SystemExit("--complex needs NAME FILE MULTIPLICITY [PARTS]")
        r = load(spec[1])
        if "atoms" not in perfect or "atoms" not in r:
            raise SystemExit("--complex: the result files must contain the atom lists")
        cx = {"name": spec[0], "file": spec[1], "g": float(spec[2]), "eps": energy(r) - e_perf,
              "dA": count(r, A) - count(perfect, A), "dB": count(r, B) - count(perfect, B),
              "vol": r["volume"] * BOHR3_A3 - v_perf, "parts": spec[3].split(",") if len(spec) == 4 else None}
        complexes.append(cx)

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
    if complexes:
        print("\nDefect complexes (raw energy eps_k [eV], change of atom numbers, orientations per cell)")
        for cx in complexes:
            line = f"  {cx['name']:12s} {cx['eps']:14.6f}   d{A} = {cx['dA']:+d}, d{B} = {cx['dB']:+d}, g = {cx['g']:g}"
            if cx["parts"]:
                for part in cx["parts"]:
                    if part not in eps:
                        raise SystemExit(f"--complex {cx['name']}: unknown part '{part}' (use V_a, B_a, V_b, A_b)")
                da = sum({"V_a": -1, "B_a": -1, "V_b": 0, "A_b": 1}[q] for q in cx["parts"])
                db = sum({"V_a": 0, "B_a": 1, "V_b": -1, "A_b": -1}[q] for q in cx["parts"])
                if (da, db) != (cx["dA"], cx["dB"]):
                    raise SystemExit(f"--complex {cx['name']}: the parts do not match the atom numbers of {cx['file']}")
                line += f"   binding energy {cx['eps'] - sum(eps[q] for q in cx['parts']):+.4f} eV"
            print(line)
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

    order = ["V_a", "B_a", "V_b", "A_b"]
    # vibrational free energies of the supercells (temperature dependent shifts of all energies)
    freqs = None
    if args.vibrations:
        freqs = {}
        for key, path in dict(files, perfect=args.perfect, **{c["name"]: c["file"] for c in complexes}).items():
            ph = path[:-5] + ".phonons.json" if path.endswith(".json") else path + ".phonons.json"
            try:
                freqs[key] = json.load(open(ph))["frequencies_THz"]
            except OSError:
                raise SystemExit(f"--vibrations: {ph} not found (run task = phonons for every supercell)")
            n_imag = sum(1 for nu in sorted(freqs[key], key=abs)[3:] if nu < 0)
            if n_imag:
                print(f"WARNING: {ph}: {n_imag} imaginary mode(s) ignored")

    def model_at(T):
        """Model with the supercell free energies at temperature T (static energies without --vibrations)."""
        if freqs is None:
            return B2Model(eps0, eps["V_a"], eps["B_a"], eps["V_b"], eps["A_b"], complexes=complexes)
        fp = vibrational_free_energy(freqs["perfect"], T)
        shift = {d: vibrational_free_energy(freqs[d], T) - fp for d in order}
        cx = [dict(c, eps=c["eps"] + vibrational_free_energy(freqs[c["name"]], T) - fp) if c["name"] in freqs else c for c in complexes]
        return B2Model(eps0 + fp / n_cells, *(eps[d] + shift[d] for d in order), complexes=cx)

    if freqs is not None:
        print("\nVibrational contributions to the composition-conserving reactions")
        print("  (Delta F_vib in eV, formation entropy S = -d(Delta F_vib)/dT in k_B; Gamma-point modes of the supercells)")

        def reaction_fvib(T):
            m = model_at(T)
            return [2 * m.e["V_a"] + m.e["A_b"] + m.eps0 - reactions[0][1], 2 * m.e["V_b"] + m.e["B_a"] + m.eps0 - reactions[1][1],
                    m.e["V_a"] + m.e["V_b"] + m.eps0 - reactions[2][1], m.e["B_a"] + m.e["A_b"] - reactions[3][1]]
        for T in args.T:
            f0, fp_, fm_ = reaction_fvib(T), reaction_fvib(T + 1.0), reaction_fvib(T - 1.0)
            print(f"  T = {T:.0f} K")
            for (name, _), df, a_, b_ in zip(reactions, f0, fp_, fm_):
                print(f"    {name:42s} {df:9.4f} {-(a_ - b_) / 2.0 / KB_EV:8.3f}")
    model0 = model_at(args.T[0])
    have_vol = volume_mode is not None
    if have_vol:
        atomic = omega0 / 2.0
        print(f"\nVolumes ({volume_mode}): Omega0 = {omega0:.4f} A^3 per formula unit; raw volume changes v_d [A^3]")
        for d in order:
            print(f"  {label[d]:8s} {vol[d]:12.4f}")
        print("Formation volumes of the composition-conserving reactions [A^3 | atomic volumes]")
        for name, v in ((reactions[0][0], 2 * vol["V_a"] + vol["A_b"] + omega0), (reactions[1][0], 2 * vol["V_b"] + vol["B_a"] + omega0),
                        (reactions[2][0], vol["V_a"] + vol["V_b"] + omega0), (reactions[3][0], vol["B_a"] + vol["A_b"])):
            print(f"  {name:42s} {v:10.4f} {v / atomic:8.3f}")
    elif any(p != 0.0 for p in args.p):
        raise SystemExit("pressure requested, but the supercells carry no volume information "
                         "(use cellrelax cells, or stress = true together with --bulk-modulus)")
    def solve_at(x, T, p):
        m = model_at(T)
        return (m.at_pressure(p, omega0, vol) if have_vol else m).solve(x, T)

    for p in args.p:
        for T in args.T:
            head = f"\nT = {T:.0f} K, p = {p:g} GPa   (concentrations per sublattice site; E_eff in eV"
            print(head + ("; V_eff in atomic volumes)" if have_vol else ")"))
            print("   x_%-3s " % A + " ".join(f"{'c(' + label[d] + ')':>12s}" for d in order) + "  " +
                  " ".join(f"{'E(' + label[d] + ')':>9s}" for d in order) +
                  ("  " + " ".join(f"{'V(' + label[d] + ')':>8s}" for d in order) if have_vol else "") + f"   mu_{A}-mu_{B}")
            for x in args.x:
                ma, mb, c = solve_at(x, T, p)
                # effective formation energies (enthalpies with --vibrations): -d ln c / d(1/kT)
                dT = 2.0
                c1, c2 = solve_at(x, T - dT, p)[2], solve_at(x, T + dT, p)[2]
                b1, b2 = 1.0 / (KB_EV * (T - dT)), 1.0 / (KB_EV * (T + dT))
                ee = {d: -(math.log(c2[d]) - math.log(c1[d])) / (b2 - b1) for d in order}
                line = f"   {x:.4f} " + " ".join(f"{c[d]:12.4e}" for d in order) + "  " + " ".join(f"{ee[d]:9.4f}" for d in order)
                if have_vol:  # V_eff = -kT d ln c / dp by central differences
                    dp = 0.05
                    cp, cm = solve_at(x, T, p + dp)[2], solve_at(x, T, p - dp)[2]
                    kt = KB_EV * T
                    veff = {d: -kt * (math.log(cp[d]) - math.log(cm[d])) / (2 * dp * GPA_A3_EV) / (omega0 / 2.0) for d in order}
                    line += "  " + " ".join(f"{veff[d]:8.3f}" for d in order)
                print(line + f"   {ma - mb:9.4f}")
                if complexes:
                    print("            complexes per unit cell: " + ", ".join(f"c({k}) = {v:.4e}" for k, v in c["complex"].items()))


if __name__ == "__main__":
    main()
