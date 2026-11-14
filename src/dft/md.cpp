#include "dft/md.hpp"

#include <cmath>
#include <fstream>
#include <random>

#include "base/parallel.hpp"

namespace lsda {

MdResult run_md(Calculation& calc, const std::vector<double>& masses, const MdParams& mp) {
  const int nat = calc.crystal().natoms();
  LSDA_CHECK(static_cast<int>(masses.size()) == nat, "md: one mass per atom is required");
  LSDA_CHECK(calc.n_symmetry_ops() == 1, "md: molecular dynamics needs symmetry = false");
  LSDA_CHECK(mp.thermostat == "none" || mp.thermostat == "berendsen", "md: thermostat must be none or berendsen");
  const double amu = 1822.888486209, fs = 41.341374575751, kb = 1.0 / hartree_in_kelvin;
  const double dt = mp.dt_fs * fs;
  const int ndof = std::max(1, 3 * nat - 3);  // centre of mass at rest
  MdResult res;
  std::vector<double> m(nat);
  for (int a = 0; a < nat; ++a) m[a] = masses[a] * amu;
  std::vector<Vec3> x(nat), v(nat, Vec3{0, 0, 0});
  for (int a = 0; a < nat; ++a) x[a] = calc.crystal().atom_cart(a);
  auto kinetic = [&]() {
    double ek = 0.0;
    for (int a = 0; a < nat; ++a) ek += 0.5 * m[a] * dot(v[a], v[a]);
    return ek;
  };
  auto apply_positions = [&]() {
    std::vector<Vec3> frac(nat);
    for (int a = 0; a < nat; ++a) frac[a] = calc.crystal().frac_from_cart(x[a]);
    calc.move_atoms(frac);
  };

  // ---- start: restart file or Maxwell-Boltzmann velocities ---------------------------------------
  const std::string rfile = mp.prefix + ".md.restart", traj = mp.prefix + ".md.xyz";
  int step0 = 0;
  double e_ref = 0.0, bath = 0.0;  // bath: energy removed by the thermostat (keeps the conserved quantity)
  bool have_ref = false, restarted = false;
  if (mp.restart) {
    std::ifstream in(rfile);
    int n = 0;
    if (in >> n >> step0 >> e_ref >> bath && n == nat) {
      bool ok = true;
      for (int a = 0; a < nat; ++a)
        for (int d = 0; d < 3; ++d) ok = ok && static_cast<bool>(in >> x[a][d]);
      for (int a = 0; a < nat; ++a)
        for (int d = 0; d < 3; ++d) ok = ok && static_cast<bool>(in >> v[a][d]);
      if (ok) {
        restarted = have_ref = true;
        apply_positions();
        logf("  MD restart: step %d read from %s\n", step0, rfile.c_str());
      }
    }
  }
  if (!restarted) {
    std::mt19937_64 rng(mp.seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (int a = 0; a < nat; ++a)
      for (int d = 0; d < 3; ++d) v[a][d] = gauss(rng) * std::sqrt(kb * std::max(mp.temperature, 0.0) / m[a]);
    Vec3 p{0, 0, 0};
    double mtot = 0.0;
    for (int a = 0; a < nat; ++a) {
      for (int d = 0; d < 3; ++d) p[d] += m[a] * v[a][d];
      mtot += m[a];
    }
    for (int a = 0; a < nat; ++a)
      for (int d = 0; d < 3; ++d) v[a][d] -= p[d] / mtot;
    const double t0 = 2.0 * kinetic() / (ndof * kb);
    if (t0 > 0.0 && nat > 1)
      for (auto& va : v)
        for (double& c : va) c *= std::sqrt(mp.temperature / t0);  // exactly the requested temperature
    if (par::root()) std::ofstream(traj, std::ios::trunc);
  }

  auto forces = [&](std::vector<Vec3>& f) {
    calc.scf();
    if (calc.stopped_by_time()) return false;
    f = calc.forces();
    return true;
  };
  std::vector<Vec3> f;
  if (!forces(f)) {
    res.stopped = true;
    return res;
  }
  logf("\n  Molecular dynamics: %d steps of %.2f fs, thermostat %s\n", mp.steps, mp.dt_fs, mp.thermostat.c_str());
  logf("   step   time (fs)     E_pot (Ha)         E_kin (Ha)     conserved (Ha)      T (K)\n");
  double tsum = 0.0;
  for (int step = step0; step <= mp.steps; ++step) {
    const double ek = kinetic(), epot = calc.energies().free, temp = 2.0 * ek / (ndof * kb), cons = epot + ek + bath;
    if (!have_ref) {
      e_ref = cons;
      have_ref = true;
    }
    res.drift = std::max(res.drift, std::abs(cons - e_ref));
    res.steps = step;
    tsum += temp;
    logf("  %5d  %10.2f  %16.8f  %14.8f  %16.8f  %9.2f\n", step, step * mp.dt_fs, epot, ek, cons, temp);
    if (par::root()) {
      std::ofstream out(traj, std::ios::app);  // extended XYZ in Angstrom
      const Crystal& c = calc.crystal();
      out << nat << "\nLattice=\"";
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out << c.A[i][j] * bohr_in_angstrom << (i == 2 && j == 2 ? "" : " ");
      out << "\" Properties=species:S:1:pos:R:3 step=" << step << " energy=" << epot << "\n";
      for (int a = 0; a < nat; ++a)
        out << c.setup(a).symbol << " " << x[a][0] * bohr_in_angstrom << " " << x[a][1] * bohr_in_angstrom << " "
            << x[a][2] * bohr_in_angstrom << "\n";
      std::ofstream rs(rfile);
      rs.precision(17);
      rs << nat << " " << step << " " << e_ref << " " << bath << "\n";
      for (const auto& xa : x) rs << xa[0] << " " << xa[1] << " " << xa[2] << "\n";
      for (const auto& va : v) rs << va[0] << " " << va[1] << " " << va[2] << "\n";
    }
    if (step == mp.steps) break;
    // ---- velocity Verlet ---------------------------------------------------------------------------
    for (int a = 0; a < nat; ++a)
      for (int d = 0; d < 3; ++d) {
        v[a][d] += 0.5 * dt * f[a][d] / m[a];
        x[a][d] += dt * v[a][d];
      }
    apply_positions();
    if (!forces(f)) {
      res.stopped = true;  // the restart file holds the last completed step
      return res;
    }
    for (int a = 0; a < nat; ++a)
      for (int d = 0; d < 3; ++d) v[a][d] += 0.5 * dt * f[a][d] / m[a];
    if (mp.thermostat == "berendsen") {
      const double ek1 = kinetic(), t1 = 2.0 * ek1 / (ndof * kb);
      if (t1 > 0.0) {
        const double lam = std::sqrt(1.0 + mp.dt_fs / mp.tau_fs * (mp.temperature / t1 - 1.0));
        for (auto& va : v)
          for (double& c : va) c *= lam;
        bath += ek1 - kinetic();
      }
    }
  }
  res.mean_temperature = tsum / (res.steps - step0 + 1);
  return res;
}

}  // namespace lsda
