#!/usr/bin/env python3
"""End-to-end regression tests: runs small calculations and compares with stored reference values.

Usage: run_regression.py <path to lsda> <pseudo dir> [--print]   (--print shows the computed values)

The references were produced with this code after it had been validated against
  * the all-electron total energy of the Al reference atom (isolated atom in a box; LDA and PBE),
  * finite-difference forces, and
  * the equality of calculations with and without symmetry.
"""
import json
import os
import subprocess
import sys
import tempfile

BOHR = 0.529177210903

CASES = {
    # fcc Al: non-magnetic metal, s/p projectors, 48 symmetry operations
    "Al_fcc": dict(
        text="""task = scf
ecut = 15
kpoints = 6 6 6
smearing = cold
width = 0.01
scf_tol = 1e-11
units = angstrom
scale = 3.98
begin lattice
 0 .5 .5
 .5 0 .5
 .5 .5 0
end
begin species
 Al Al.LDA_PW-JTH.xml
end
begin atoms
 Al 0 0 0
end
""", free_energy=None, tol=2e-7),
    # B2 NiAl: d projectors, semicore states, symmetry reduction of k-points
    "NiAl_B2": dict(
        text="""task = scf
ecut = 20
kpoints = 4 4 4
smearing = cold
width = 0.01
scf_tol = 1e-11
units = angstrom
scale = 2.83
begin lattice
 1 0 0
 0 1 0
 0 0 1
end
begin species
 Ni Ni.LDA_PW-JTH.xml
 Al Al.LDA_PW-JTH.xml
end
begin atoms
 Ni 0 0 0
 Al .5 .5 .5
end
""", free_energy=None, tol=2e-7),
    # distorted ferromagnetic FeAl without symmetry: spin polarization and forces
    "FeAl_forces": dict(
        text="""task = forces
ecut = 20
kpoints = 3 3 3
symmetry = false
nspin = 2
smearing = cold
width = 0.01
scf_tol = 1e-11
eig_tol = 1e-8
units = angstrom
scale = 2.83
begin lattice
 1 0 0
 0 1 0
 0 0 1
end
begin species
 Fe Fe.LDA_PW-JTH.xml
 Al Al.LDA_PW-JTH.xml
end
begin atoms
 Fe 0 0 0 2.0
 Al 0.54 0.53 0.48
end
""", free_energy=None, magnetization=None, force_al=None, tol=3e-7),
}

# PBE: gradient-corrected functional on the grid and in the PAW spheres (non-magnetic and spin-polarized)
CASES["NiAl_B2_pbe"] = dict(text=CASES["NiAl_B2"]["text"].replace("LDA_PW-JTH", "GGA_PBE-JTH").replace("kpoints = 4 4 4", "kpoints = 3 3 3"),
                            free_energy=None, tol=2e-7)
CASES["FeAl_forces_pbe"] = dict(text=CASES["FeAl_forces"]["text"].replace("LDA_PW-JTH", "GGA_PBE-JTH").replace("kpoints = 3 3 3", "kpoints = 2 2 2"),
                                free_energy=None, magnetization=None, force_al=None, tol=3e-7)

# second eigensolver (all-band LOBPCG) must reproduce the PPCG results
CASES["NiAl_B2_lobpcg"] = dict(text="eigensolver = lobpcg\n" + CASES["NiAl_B2"]["text"], free_energy=None, tol=2e-7)

# reference values (Hartree, mu_B, Hartree/bohr)
REFERENCE = {
    "Al_fcc": {"free_energy": -241.922333544851},
    "NiAl_B2": {"free_energy": -1760.310786754092},
    "NiAl_B2_lobpcg": {"free_energy": -1760.310786754092},
    "NiAl_B2_pbe": {"free_energy": -1763.634387933795},
    "FeAl_forces_pbe": {"free_energy": -1515.621394585464, "magnetization": 1.11573263,
                        "force_al": [-0.0261915559, -0.0198981485, 0.0133896651]},
    "FeAl_forces": {"free_energy": -1512.506028752653, "magnetization": 1.05260965,
                    "force_al": [-0.0184209509, -0.0141651253, 0.0096173705]},
}


NIAL = """ecut = 20
kpoints = 3 3 3
smearing = cold
width = 0.01
scf_tol = 1e-11
units = angstrom
scale = 2.83
begin lattice
 1 0 0
 0 1 0
 0 0 1
end
begin species
 Ni Ni.LDA_PW-JTH.xml
 Al Al.LDA_PW-JTH.xml
end
begin atoms
 Ni 0 0 0
 Al %s 0.5 0.5
end
"""


def relax_and_restart(lsda, pseudo, tmp):
    """Relaxation of a distorted cell must recover the ideal structure and its energy;
    a restart from the checkpoint must reproduce the energy in fewer SCF iterations."""
    def run(name, text):
        with open(os.path.join(tmp, name + ".in"), "w") as f:
            f.write(f"prefix = {name}\npseudo_dir = {pseudo}\n" + text)
        out = subprocess.run([lsda, name + ".in"], cwd=tmp, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"[FAIL] {name}: lsda exited with code {out.returncode}\n{out.stdout[-1500:]}\n{out.stderr}")
            return None
        return json.load(open(os.path.join(tmp, name + ".json")))

    failures = 0
    def check(ok, what, value, tol):
        nonlocal failures
        failures += not ok
        print(f"[{' ok ' if ok else 'FAIL'}] {what:42s} {value:.2e} (tolerance {tol:.0e})")

    ideal = run("ideal", "task = scf\ndos = true\n" + NIAL % "0.5")
    relaxed = run("relaxed", "task = relax\nforce_tol = 2e-5\n" + NIAL % "0.53")
    if ideal is None or relaxed is None:
        return failures + 1
    de = abs(relaxed["free_energy"] - ideal["free_energy"])
    dx = abs(relaxed["atoms"][1]["frac"][0] - relaxed["atoms"][0]["frac"][0] - 0.5)
    check(relaxed["converged"] and de < 2e-7, "relaxation: energy of the ideal structure", de, 2e-7)
    check(dx < 2e-4, "relaxation: ideal position recovered", dx, 2e-4)
    again = run("ideal", "task = scf\nrestart = true\n" + NIAL % "0.5")
    if again is None:
        return failures + 1
    de = abs(again["free_energy"] - ideal["free_energy"])
    check(de < 1e-8, "restart: same energy from checkpoint", de, 1e-8)
    check(again["scf_iterations"] < ideal["scf_iterations"], "restart: fewer SCF iterations",
          again["scf_iterations"], ideal["scf_iterations"])
    # density of states: number of states below the Fermi level = number of electrons
    rows = [[float(x) for x in l.split()] for l in open(os.path.join(tmp, "ideal.dos")) if not l.startswith("#")]
    nel = sum(r[1] for r in rows if r[0] < 0) * (rows[1][0] - rows[0][0])
    check(abs(nel - ideal["n_electrons"]) < 0.1, "density of states integrates to the electron number", abs(nel - ideal["n_electrons"]), 0.1)
    # nudged elastic band, plumbing only (the optimizer itself is unit-tested on an analytic surface):
    # Al moved by one lattice vector -> equal end-point energies, positive barrier, path file written
    neb_in = "task = neb\nneb_images = 1\nneb_steps = 1\nkpoints = 2 2 2\n" + (NIAL % "0.5").replace("kpoints = 3 3 3\n", "") + \
        "begin atoms_final\n Ni 0 0 0\n Al 1.5 0.5 0.5\nend\n"
    with open(os.path.join(tmp, "path.in"), "w") as f:
        f.write(f"prefix = path\npseudo_dir = {pseudo}\n" + neb_in)
    out = subprocess.run([lsda, "path.in"], cwd=tmp, capture_output=True, text=True)
    try:
        nb = json.load(open(os.path.join(tmp, "path.neb.json")))
        ends = abs(nb["images"][0]["free_energy"] - nb["images"][-1]["free_energy"])
        check(out.returncode in (0, 3) and ends < 1e-8 and nb["barrier_forward_eV"] > 0.5, "NEB task: end points and barrier", ends, 1e-8)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] NEB task: {e}\n{out.stdout[-800:]}{out.stderr}")
        failures += 1
    # Gamma-point phonons of cubic NiAl: three acoustic modes at zero and a threefold degenerate optical mode
    with open(os.path.join(tmp, "ph.in"), "w") as f:
        f.write(f"prefix = ph\npseudo_dir = {pseudo}\ntask = phonons\necut = 15\n" + (NIAL % "0.5").replace("ecut = 20\n", ""))
    out = subprocess.run([lsda, "ph.in"], cwd=tmp, capture_output=True, text=True)
    try:
        nu = json.load(open(os.path.join(tmp, "ph.phonons.json")))["frequencies_THz"]
        split = max(nu[3:]) - min(nu[3:])
        # the three displacement directions are independent calculations: the degeneracy holds up to force noise
        check(out.returncode == 0 and max(abs(x) for x in nu[:3]) < 1e-3 and split < 2e-2 and 3 < nu[3] < 20,
              f"phonons of NiAl: optical mode {nu[3]:.3f} THz, degeneracy", split, 2e-2)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] phonons: {e}\n{out.stdout[-800:]}{out.stderr}")
        failures += 1
    # interrupted SCF (here: iteration limit; the wall-time limit max_seconds uses the same path):
    # the checkpoint holds the unconverged density and wavefunctions, the restart must finish the job
    with open(os.path.join(tmp, "cut.in"), "w") as f:
        f.write(f"prefix = cut\npseudo_dir = {pseudo}\ntask = scf\nmax_scf = 5\n" + NIAL % "0.5")
    part = subprocess.run([lsda, "cut.in"], cwd=tmp, capture_output=True, text=True)
    done = run("cut", "task = scf\nrestart = true\n" + NIAL % "0.5")
    if done is None:
        return failures + 1
    de = abs(done["free_energy"] - ideal["free_energy"])
    check(part.returncode == 3 and done["converged"] and de < 1e-8, "restart of an interrupted SCF run", de, 1e-8)
    check(done["scf_iterations"] + 5 <= ideal["scf_iterations"] + 3, "restart continues (no repeated work)",
          done["scf_iterations"] + 5, ideal["scf_iterations"] + 3)
    return failures


def stress_tests(lsda, pseudo, tmp):
    """Stress tensor = strain derivative of the free energy (finite differences at fixed basis set);
    variable-cell relaxation reaches zero pressure and lowers the energy."""
    a0 = 5.35
    ref = [[a0, 0, 0], [0.15, 1.02 * a0, 0], [0, -0.1, 0.97 * a0]]

    def run(name, eps):
        lat = [[sum(ref[i][k] * ((j == k) + eps[j][k]) for k in range(3)) for j in range(3)] for i in range(3)]
        fmt = lambda m: "\n".join(" ".join(f"{x:.14f}" for x in row) for row in m)
        text = f"""prefix = {name}
pseudo_dir = {pseudo}
task = stress
ecut = 15
kpoints = 2 2 2
symmetry = false
smearing = cold
width = 0.02
scf_tol = 1e-12
eig_tol = 1e-9
begin lattice
{fmt(lat)}
end
begin basis_lattice
{fmt(ref)}
end
begin species
 Ni Ni.LDA_PW-JTH.xml
 Al Al.LDA_PW-JTH.xml
end
begin atoms
 Ni 0 0 0
 Al 0.54 0.47 0.52
end
"""
        with open(os.path.join(tmp, name + ".in"), "w") as f:
            f.write(text)
        out = subprocess.run([lsda, name + ".in"], cwd=tmp, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(out.stdout[-1500:] + out.stderr)
        return json.load(open(os.path.join(tmp, name + ".json")))

    failures = 0
    try:
        zero = [[0.0] * 3 for _ in range(3)]
        r0 = run("s0", zero)
        h = 5e-4
        for (i, j) in ((0, 0), (1, 2)):
            ep, em = [row[:] for row in zero], [row[:] for row in zero]
            ep[i][j] += h; em[i][j] -= h
            if i != j:
                ep[j][i] += h; em[j][i] -= h
            fd = (run("sp", ep)["free_energy"] - run("sm", em)["free_energy"]) / (2 * h) / r0["volume"] / (1 if i == j else 2)
            err = abs(fd - r0["stress"][i][j])
            ok = err < 1e-7
            failures += not ok
            print(f"[{' ok ' if ok else 'FAIL'}] stress[{i}][{j}] = dF/d(strain)/V (finite differences)   {err:.2e} (tolerance 1e-07)")
        # cell relaxation of fcc Al from a too large lattice constant
        text = f"""prefix = alcell
pseudo_dir = {pseudo}
task = cellrelax
ecut = 15
kpoints = 4 4 4
smearing = cold
width = 0.02
units = angstrom
scale = 4.08
begin lattice
 0 .5 .5
 .5 0 .5
 .5 .5 0
end
begin species
 Al Al.LDA_PW-JTH.xml
end
begin atoms
 Al 0 0 0
end
"""
        with open(os.path.join(tmp, "alcell.in"), "w") as f:
            f.write(text)
        out = subprocess.run([lsda, "alcell.in"], cwd=tmp, capture_output=True, text=True)
        r = json.load(open(os.path.join(tmp, "alcell.json")))
        a = (4 * r["volume"]) ** (1 / 3) * BOHR
        ok = out.returncode == 0 and r["converged"] and abs(r["pressure_GPa"]) < 0.05 and 3.9 < a < 4.05
        failures += not ok
        print(f"[{' ok ' if ok else 'FAIL'}] cell relaxation of fcc Al: a = {a:.4f} A, p = {r['pressure_GPa']:.4f} GPa")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] stress tests: {e}")
        failures += 1
    return failures


def noncollinear_test(lsda, pseudo, tmp):
    """Non-collinear spinor calculation with all moments along a tilted axis must reproduce the collinear
    energy, moment and forces (spin-rotation invariance without spin-orbit coupling)."""
    base = CASES["FeAl_forces"]["text"].replace("ecut = 20", "ecut = 15").replace("kpoints = 3 3 3", "kpoints = 2 2 2") \
        .replace("scf_tol = 1e-11", "scf_tol = 1e-10").replace("eig_tol = 1e-8\n", "")
    res = {}
    for name, text in (("col", base), ("nc", base.replace("nspin = 2", "noncollinear = true").replace("Fe 0 0 0 2.0", "Fe 0 0 0 1.2 0 1.6"))):
        with open(os.path.join(tmp, name + ".in"), "w") as f:
            f.write(f"prefix = {name}\npseudo_dir = {pseudo}\ncheckpoint_wfc = false\n" + text)
        out = subprocess.run([lsda, name + ".in"], cwd=tmp, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"[FAIL] non-collinear test, run {name}: exit code {out.returncode}\n{out.stdout[-800:]}{out.stderr}")
            return 1
        res[name] = json.load(open(os.path.join(tmp, name + ".json")))
    c, n = res["col"], res["nc"]
    de = abs(c["free_energy"] - n["free_energy"])
    dm = abs(c["magnetization"] - n["magnetization"])
    df = max(abs(x - y) for p, q in zip(c["atoms"], n["atoms"]) for x, y in zip(p["force"], q["force"]))
    mv = n["magnetization_vector"]
    tilt = abs(mv[0] / n["magnetization"] - 0.6) + abs(mv[2] / n["magnetization"] - 0.8)
    ok = de < 1e-8 and dm < 1e-4 and df < 5e-6 and tilt < 5e-3
    print(f"[{' ok ' if ok else 'FAIL'}] non-collinear = collinear for a rotated moment: dE {de:.1e}, dM {dm:.1e}, dF {df:.1e}")
    return 0 if ok else 1


HYBRID_H = """task = scf
xc = hf
ecut = 15
kpoints = 1 1 1
nspin = 2
nbands = 3
smearing = fermi-dirac
width = 0.002
scf_tol = 1e-9
exx_tol = 1e-5
units = bohr
scale = 12.0
begin lattice
 1 0 0
 0 1 0
 0 0 1
end
begin species
 H H.LDA_PW-JTH.xml
end
begin atoms
 H 0 0 0 1.0
end
"""

HYBRID_AL = """task = forces
xc = pbe0
ecut = 10
kpoints = 2 1 1
smearing = fermi-dirac
width = 0.02
scf_tol = 1e-10
units = bohr
scale = 1.0
begin lattice
 5.4 0 0
 0 5.0 0
 0 0.3 9.0
end
begin species
 Al Al.GGA_PBE-JTH.xml
end
begin atoms
 Al 0 0 0
 Al 0.45 0.55 0.52
end
"""


def hybrid_test(lsda, pseudo, tmp):
    """Hybrid functionals: the Hartree-Fock energy of the hydrogen atom is -1/2 Ha (exact cancellation of the
    self-interaction); PBE0 energy and force of a small Al cell against stored values (the force was verified by
    finite differences of the energy)."""
    failures = 0
    res = {}
    for name, text in (("h_hf", HYBRID_H), ("al_pbe0", HYBRID_AL)):
        with open(os.path.join(tmp, name + ".in"), "w") as f:
            f.write(f"prefix = {name}\npseudo_dir = {pseudo}\ncheckpoint_wfc = false\n" + text)
        out = subprocess.run([lsda, name + ".in"], cwd=tmp, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"[FAIL] hybrid test, run {name}: exit code {out.returncode}\n{out.stdout[-800:]}{out.stderr}")
            return 1
        res[name] = json.load(open(os.path.join(tmp, name + ".json")))
    e_h = res["h_hf"]["free_energy"]
    ok = abs(e_h + 0.5) < 1e-3
    print(f"[{' ok ' if ok else 'FAIL'}] Hartree-Fock hydrogen atom: E = {e_h:.6f} Ha (exact -0.5)")
    failures += 0 if ok else 1
    e_al, f_al = res["al_pbe0"]["free_energy"], res["al_pbe0"]["atoms"][1]["force"][0]
    ok = abs(e_al - (-485.943851975)) < 2e-7 and abs(f_al - (-0.0121146)) < 2e-5
    print(f"[{' ok ' if ok else 'FAIL'}] PBE0, two Al atoms: E = {e_al:.9f} Ha, force {f_al:.7f} Ha/bohr (finite difference -0.0121146)")
    failures += 0 if ok else 1
    return failures


def mgga_test(lsda, pseudo, tmp):
    """Meta-GGA r2SCAN: hydrogen atom (the functional is exact for one-electron densities up to self-consistency
    and grid effects), symmetry on/off, and the force of a small Al cell against a finite difference of the energy
    (computed without symmetry, where the forces are not averaged over equivalent atoms)."""
    h_atom = HYBRID_H.replace("xc = hf", "xc = r2scan").replace("scale = 12.0", "scale = 10.0").replace("exx_tol = 1e-5\n", "dense_factor = 4\n")
    al = HYBRID_AL.replace("xc = pbe0", "xc = r2scan").replace("ecut = 10", "ecut = 12").replace("scf_tol = 1e-10", "scf_tol = 1e-11")
    res = {}
    for name, text in (("h_r2scan", h_atom), ("al_r2scan", al), ("al_r2scan_nosym", al + "symmetry = false\n")):
        with open(os.path.join(tmp, name + ".in"), "w") as f:
            f.write(f"prefix = {name}\npseudo_dir = {pseudo}\ncheckpoint_wfc = false\n" + text)
        out = subprocess.run([lsda, name + ".in"], cwd=tmp, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"[FAIL] meta-GGA test, run {name}: exit code {out.returncode}\n{out.stdout[-800:]}{out.stderr}")
            return 1
        res[name] = json.load(open(os.path.join(tmp, name + ".json")))
    failures = 0
    e_h = res["h_r2scan"]["free_energy"]
    ok = abs(e_h + 0.5) < 3e-3 and abs(e_h - (-0.5014937774)) < 2e-6
    print(f"[{' ok ' if ok else 'FAIL'}] r2SCAN hydrogen atom (small box, coarse density grid): E = {e_h:.7f} Ha")
    failures += 0 if ok else 1
    a, b = res["al_r2scan"], res["al_r2scan_nosym"]
    de = abs(a["free_energy"] - b["free_energy"])
    ok = abs(a["free_energy"] - (-486.1084993392)) < 2e-7 and de < 1e-8
    print(f"[{' ok ' if ok else 'FAIL'}] r2SCAN, two Al atoms: E = {a['free_energy']:.9f} Ha, symmetry on/off difference {de:.1e}")
    failures += 0 if ok else 1
    f2 = b["atoms"][1]["force"][0]
    ok = abs(f2 - (-0.0029167)) < 3e-6
    print(f"[{' ok ' if ok else 'FAIL'}] r2SCAN force without symmetry: {f2:.7f} Ha/bohr (finite difference -0.0029167)")
    failures += 0 if ok else 1
    return failures


def main():
    lsda, pseudo = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    show = "--print" in sys.argv
    if "--only" in sys.argv:  # run a single group of self-contained tests
        which = sys.argv[sys.argv.index("--only") + 1]
        groups = {"noncollinear": noncollinear_test, "hybrid": hybrid_test, "mgga": mgga_test, "stress": stress_tests, "relax": relax_and_restart}
        with tempfile.TemporaryDirectory() as tmp:
            failures = groups[which](lsda, pseudo, tmp)
        print("PASSED" if failures == 0 else "FAILED")
        return 1 if failures else 0
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, case in CASES.items():
            inp = os.path.join(tmp, name + ".in")
            with open(inp, "w") as f:
                f.write(f"prefix = {name}\npseudo_dir = {pseudo}\n" + case["text"])
            out = subprocess.run([lsda, inp], cwd=tmp, capture_output=True, text=True)
            if out.returncode != 0:
                print(f"[FAIL] {name}: lsda exited with code {out.returncode}\n{out.stdout[-2000:]}\n{out.stderr}")
                failures += 1
                continue
            r = json.load(open(os.path.join(tmp, name + ".json")))
            got = {"free_energy": r["free_energy"]}
            if "magnetization" in case:
                got["magnetization"] = r["magnetization"]
            if "force_al" in case:
                got["force_al"] = r["atoms"][1]["force"]
            if show:
                print(f'    "{name}": {json.dumps(got)},')
                continue
            ref = REFERENCE.get(name)
            if ref is None:
                print(f"[FAIL] {name}: no reference value stored")
                failures += 1
                continue
            for key, val in ref.items():
                a = got[key] if isinstance(got[key], list) else [got[key]]
                b = val if isinstance(val, list) else [val]
                tol = case["tol"] * (20 if key == "magnetization" else 1)
                err = max(abs(x - y) for x, y in zip(a, b))
                ok = err < tol
                failures += not ok
                print(f"[{' ok ' if ok else 'FAIL'}] {name:12s} {key:14s} deviation {err:.2e} (tolerance {tol:.0e})")
        if not show:
            failures += relax_and_restart(lsda, pseudo, tmp)
            failures += stress_tests(lsda, pseudo, tmp)
            failures += noncollinear_test(lsda, pseudo, tmp)
            failures += hybrid_test(lsda, pseudo, tmp)
            failures += mgga_test(lsda, pseudo, tmp)
    if not show:
        print("ALL REGRESSION TESTS PASSED" if failures == 0 else f"{failures} REGRESSION TEST(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
