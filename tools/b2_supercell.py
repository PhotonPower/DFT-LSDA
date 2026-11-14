#!/usr/bin/env python3
"""Generate lsda input files for B2 (CsCl-type) TM-Al supercells with a single point defect.

Sublattice alpha (transition metal, TM):  (i, j, k)/n
Sublattice beta  (Al)                  :  (i+1/2, j+1/2, k+1/2)/n

Defects (always placed at a site with full cubic point symmetry, so 48 operations survive):
  none    perfect crystal
  vac_tm  vacancy on the TM sublattice            V_alpha
  vac_al  vacancy on the Al sublattice            V_beta
  al_tm   Al atom on a TM site (antisite)         Al_alpha
  tm_al   TM atom on an Al site (antisite)        TM_beta

Example:
  b2_supercell.py --tm Ni --a 2.834 --n 3 --defect vac_tm --kpoints 4 4 4 --task relax -o NiAl_vac_tm.in
"""
import argparse


def build(tm, n, defect):
    """Returns list of (label, (x, y, z)) in fractional coordinates of the supercell."""
    atoms = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                atoms.append((tm, (i / n, j / n, k / n)))
                atoms.append(("Al", ((i + 0.5) / n, (j + 0.5) / n, (k + 0.5) / n)))
    tm0 = atoms.index((tm, (0.0, 0.0, 0.0)))
    al0 = atoms.index(("Al", (0.5 / n, 0.5 / n, 0.5 / n)))
    if defect == "none":
        pass
    elif defect == "vac_tm":
        del atoms[tm0]
    elif defect == "vac_al":
        del atoms[al0]
    elif defect == "al_tm":
        atoms[tm0] = ("Al", atoms[tm0][1])
    elif defect == "tm_al":
        atoms[al0] = (tm + "*", atoms[al0][1])  # '*' marks the antisite atom (may carry a different moment)
    else:
        raise ValueError(defect)
    return atoms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tm", required=True, help="transition metal, e.g. Ni or Fe")
    ap.add_argument("--a", type=float, required=True, help="B2 lattice constant (Angstrom)")
    ap.add_argument("--n", type=int, default=2, help="supercell n x n x n (2 n^3 sites)")
    ap.add_argument("--defect", default="none", choices=["none", "vac_tm", "vac_al", "al_tm", "tm_al"])
    ap.add_argument("--task", default="scf", choices=["scf", "forces", "relax"])
    ap.add_argument("--ecut", type=float, default=20.0)
    ap.add_argument("--kpoints", type=int, nargs=3, default=[4, 4, 4])
    ap.add_argument("--kshift", type=float, nargs=3, default=[0, 0, 0])
    ap.add_argument("--nspin", type=int, default=1)
    ap.add_argument("--moment", type=float, default=0.0, help="initial moment of regular TM atoms (mu_B)")
    ap.add_argument("--moment-antisite", type=float, default=None, help="initial moment of a TM antisite atom")
    ap.add_argument("--smearing", default="cold")
    ap.add_argument("--width", type=float, default=0.01)
    ap.add_argument("--pseudo-dir", default="../../pseudos")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="additional input keywords")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    atoms = build(args.tm, args.n, args.defect)
    prefix = args.prefix or args.output.rsplit(".", 1)[0].split("/")[-1]
    L = args.a * args.n
    lines = [
        f"# B2 {args.tm}Al {args.n}x{args.n}x{args.n} supercell, defect: {args.defect}, a = {args.a} A",
        f"task       = {args.task}",
        f"prefix     = {prefix}",
        f"pseudo_dir = {args.pseudo_dir}",
        f"ecut       = {args.ecut}",
        f"kpoints    = {args.kpoints[0]} {args.kpoints[1]} {args.kpoints[2]}",
        f"kshift     = {args.kshift[0]} {args.kshift[1]} {args.kshift[2]}",
        f"nspin      = {args.nspin}",
        f"smearing   = {args.smearing}",
        f"width      = {args.width}",
    ]
    for kv in args.set:
        key, value = kv.split("=", 1)
        lines.append(f"{key.strip():10s} = {value.strip()}")
    lines += ["units      = angstrom", "begin lattice", f"  {L:.10f} 0 0", f"  0 {L:.10f} 0", f"  0 0 {L:.10f}", "end",
              "begin species", f"  {args.tm} {args.tm}.LDA_PW-JTH.xml", "  Al Al.LDA_PW-JTH.xml", "end", "begin atoms"]
    for label, (x, y, z) in atoms:
        moment = ""
        if args.nspin == 2 and label.startswith(args.tm):
            m = args.moment
            if label.endswith("*") and args.moment_antisite is not None:
                m = args.moment_antisite
            moment = f"  {m}"
        lines.append(f"  {label.rstrip('*'):3s} {x:.12f} {y:.12f} {z:.12f}{moment}")
    lines.append("end")
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{args.output}: {len(atoms)} atoms")


if __name__ == "__main__":
    main()
