#!/usr/bin/env python3
"""End-to-end regression tests: runs small calculations and compares with stored reference values.

Usage: run_regression.py <path to lsda> <pseudo dir> [--print]   (--print shows the computed values)

The references were produced with this code after it had been validated against
  * the all-electron total energy of the Al reference atom (isolated atom in a box, 2e-5 Ha),
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

# second eigensolver (all-band LOBPCG) must reproduce the PPCG results
CASES["NiAl_B2_lobpcg"] = dict(text="eigensolver = lobpcg\n" + CASES["NiAl_B2"]["text"], free_energy=None, tol=2e-7)

# reference values (Hartree, mu_B, Hartree/bohr)
REFERENCE = {
    "Al_fcc": {"free_energy": -241.922333544851},
    "NiAl_B2": {"free_energy": -1760.310786754092},
    "NiAl_B2_lobpcg": {"free_energy": -1760.310786754092},
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

    ideal = run("ideal", "task = scf\n" + NIAL % "0.5")
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


def main():
    lsda, pseudo = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    show = "--print" in sys.argv
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
    if not show:
        print("ALL REGRESSION TESTS PASSED" if failures == 0 else f"{failures} REGRESSION TEST(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
