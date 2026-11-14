// lsda - plane-wave PAW density-functional code (local spin density approximation)
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>

#include "dft/calculation.hpp"
#include "dft/relax.hpp"
#include "io/input.hpp"

using namespace lsda;

static void print_forces(const Calculation& calc, const std::vector<Vec3>& f) {
  const Crystal& c = calc.crystal();
  logf("\n  Forces (Ha/bohr)\n");
  Vec3 net{0, 0, 0};
  double fmax = 0.0;
  for (int a = 0; a < c.natoms(); ++a) {
    logf("    %4d %-4s %14.8f %14.8f %14.8f\n", a + 1, c.species[c.atoms[a].species].name.c_str(), f[a][0], f[a][1], f[a][2]);
    for (int d = 0; d < 3; ++d) net[d] += f[a][d];
    fmax = std::max(fmax, norm(f[a]));
  }
  logf("    max |F| = %.8f   net force = (%.2e, %.2e, %.2e)\n", fmax, net[0], net[1], net[2]);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <input file>\n", argv[0]);
    return 1;
  }
  try {
    wall_time();
    logf("\n  lsda - plane-wave PAW code, local spin density approximation\n");
    logf("  input file: %s\n", argv[1]);
    Input inp = read_input(argv[1]);
    logf("\n  PAW datasets\n");
    for (const Species& s : inp.crystal.species)
      logf("    %-4s Z = %-4g valence = %-4g r_paw = %.3f bohr  projectors = %-2d  %s\n", s.name.c_str(), s.setup.Z,
           s.setup.n_valence, s.setup.r_paw, s.setup.ni, s.file.c_str());
    const std::string chk = inp.prefix + ".chk", json = inp.prefix + ".json";
    bool restart_positions = false;
    if (inp.restart && inp.task != "neb") {  // structure of the last checkpoint
      std::vector<Vec3> frac;
      Mat3 lattice{};
      restart_positions = Calculation::read_checkpoint_structure(chk, inp.crystal.natoms(), frac, lattice);
      if (restart_positions) {
        for (int a = 0; a < inp.crystal.natoms(); ++a) inp.crystal.atoms[a].frac = frac[a];
        // continue with the last cell of a cell relaxation; phonons use the relaxed cell of the checkpoint
        if (inp.task == "cellrelax" || inp.task == "phonons") inp.crystal.set_lattice(lattice);
      }
    }
    if (inp.task == "phonons") {
      std::vector<double> masses;
      for (const Atom& a : inp.crystal.atoms) {
        const Species& sp = inp.crystal.species[a.species];
        masses.push_back(sp.mass > 0.0 ? sp.mass : atomic_mass(static_cast<int>(std::lround(sp.setup.Z))));
      }
      const PhononResult r = run_phonons(inp.crystal, inp.params, inp.phonons, masses);
      if (r.stopped) {
        logf("\n  Stopped by the wall-time limit; rerun with 'restart = true' (completed displacements are kept).\n");
        return 4;
      }
      logf("\n  Force constants: acoustic sum rule violated by %.2e, asymmetry %.2e Ha/bohr^2 (both removed)\n", r.asr_violation, r.asymmetry);
      logf("\n  Gamma-point phonons of the cell\n   mode     THz        cm^-1       meV\n");
      int imaginary = 0;
      for (size_t i = 0; i < r.frequency_thz.size(); ++i) {
        const double nu = r.frequency_thz[i];
        logf("  %5zu %10.4f %10.3f %10.4f%s\n", i + 1, nu, nu * 33.35640951982, nu * 4.135667696, nu < -0.05 ? "   imaginary" : "");
        if (nu < -0.05) ++imaginary;
      }
      if (imaginary > 0) logf("  WARNING: %d imaginary modes (structure not at a minimum, or numerical noise)\n", imaginary);
      write_phonon_results(inp.prefix + ".phonons.json", inp.crystal, masses, r);
      logf("\n  Frequencies written to %s.phonons.json\n", inp.prefix.c_str());
      print_timings();
      logf("\n  Total wall time: %.1f s\n", wall_time());
      return 0;
    }
    if (inp.task == "neb") {
      const NebResult r = run_neb(inp.crystal, inp.frac_final, inp.params, inp.neb);
      if (r.stopped) {
        logf("\n  Stopped by the wall-time limit; rerun with 'restart = true' to continue the path optimization.\n");
        return 4;
      }
      logf(r.converged ? "\n  NEB converged after %d steps\n" : "\n  WARNING: NEB NOT converged after %d steps\n", r.steps);
      logf("\n  image   path length (bohr)   free energy (Ha)      relative (eV)\n");
      for (size_t i = 0; i < r.energy.size(); ++i)
        logf("  %4zu   %14.6f   %18.10f   %12.5f%s\n", i, r.distance[i], r.energy[i], (r.energy[i] - r.energy[0]) * hartree_in_ev,
             static_cast<int>(i) == r.saddle ? "   <- highest image" : "");
      logf("\n  Barrier: %.5f eV (forward), %.5f eV (backward)\n", (r.energy[r.saddle] - r.energy[0]) * hartree_in_ev,
           (r.energy[r.saddle] - r.energy.back()) * hartree_in_ev);
      write_neb_results(inp.prefix + ".neb.json", inp.crystal, r);
      logf("  Path written to %s.neb.json\n", inp.prefix.c_str());
      print_timings();
      logf("\n  Total wall time: %.1f s\n", wall_time());
      return r.converged ? 0 : 3;
    }
    auto calc_ptr = std::make_unique<Calculation>(std::move(inp.crystal), inp.params);
    if (inp.restart) {
      const bool ok = restart_positions && calc_ptr->load_checkpoint(chk);
      logf(ok ? "\n  Restart: positions, density and density matrices read from %s\n"
              : "\n  Restart: no compatible checkpoint %s, starting from atomic densities\n", chk.c_str());
    }

    bool converged = false;
    std::vector<Vec3> forces;
    Mat3 sigma{};
    bool have_stress = false;
    if (inp.task == "cellrelax") {
      inp.cell.on_step = [&](const Calculation& c, const std::vector<Vec3>& f, const Mat3& st) {
        write_results(json, c, false, &f, &st);
      };
      const CellRelaxResult r = relax_cell(calc_ptr, inp.relax, inp.cell);
      converged = r.converged;
      forces = r.forces;
      sigma = r.stress;
      have_stress = true;
      logf(r.converged ? "\n  Cell relaxation converged after %d steps\n" : "\n  WARNING: cell relaxation NOT converged after %d steps\n",
           r.steps);
      const Crystal& c = calc_ptr->crystal();
      logf("\n  Final lattice vectors (bohr)\n");
      for (int i = 0; i < 3; ++i) logf("    %16.10f %16.10f %16.10f\n", c.A[i][0], c.A[i][1], c.A[i][2]);
      logf("  Final lattice vectors (Angstrom)\n");
      for (int i = 0; i < 3; ++i)
        logf("    %16.10f %16.10f %16.10f\n", c.A[i][0] * bohr_in_angstrom, c.A[i][1] * bohr_in_angstrom, c.A[i][2] * bohr_in_angstrom);
      logf("  Final fractional coordinates\n");
      for (const Atom& a : c.atoms)
        logf("    %-4s %16.10f %16.10f %16.10f\n", c.species[a.species].name.c_str(), a.frac[0], a.frac[1], a.frac[2]);
      calc_ptr->save_checkpoint(chk);
    }
    Calculation& calc = *calc_ptr;
    if (inp.task == "cellrelax") {
    } else if (inp.task == "relax") {
      inp.relax.on_step = [&](const Calculation& c, const std::vector<Vec3>& f, bool scf_ok) {
        c.save_checkpoint(chk);
        write_results(json, c, false, &f);  // "converged" refers to the relaxation; final state is written below
        (void)scf_ok;
      };
      const RelaxResult r = relax(calc, inp.relax);
      converged = r.converged;
      forces = r.forces;
      logf(r.converged ? "\n  Relaxation converged after %d steps\n" : "\n  WARNING: relaxation NOT converged after %d steps\n",
           r.steps);
      logf("\n  Final fractional coordinates\n");
      for (const Atom& a : calc.crystal().atoms)
        logf("    %-4s %16.10f %16.10f %16.10f\n", calc.crystal().species[a.species].name.c_str(), a.frac[0], a.frac[1], a.frac[2]);
    } else {
      converged = calc.scf();
      if (!calc.stopped_by_time()) {
        if (inp.task == "forces" || inp.task == "stress") forces = calc.forces();
        calc.save_checkpoint(chk);
      }
    }
    if (calc.stopped_by_time()) {
      // interrupted by max_seconds: keep the state and leave; "restart = true" continues the run
      calc.save_checkpoint(chk);
      write_results(json, calc, false, nullptr);
      logf("\n  Stopped by the wall-time limit. Checkpoint written to %s; rerun with 'restart = true' to continue.\n", chk.c_str());
      print_timings();
      logf("\n  Total wall time: %.1f s\n", wall_time());
      return 4;
    }
    if (!forces.empty()) print_forces(calc, forces);
    if (inp.stress) {
      if (!have_stress) sigma = calc.stress();
      const double gpa = 29421.02648438959;
      logf("\n  Stress tensor (1/V) dF/d(strain) at fixed basis set (GPa)\n");
      for (int i = 0; i < 3; ++i) logf("    %14.6f %14.6f %14.6f\n", sigma[i][0] * gpa, sigma[i][1] * gpa, sigma[i][2] * gpa);
      logf("    pressure = %.6f GPa\n", -(sigma[0][0] + sigma[1][1] + sigma[2][2]) / 3.0 * gpa);
    }
    write_results(json, calc, converged, forces.empty() ? nullptr : &forces, inp.stress ? &sigma : nullptr);
    if (inp.dos) {
      const std::string dos_file = inp.prefix + ".dos";
      calc.write_dos(dos_file, inp.dos_width > 0.0 ? inp.dos_width : calc.params().width, inp.dos_points);
      logf("\n  Density of states written to %s\n", dos_file.c_str());
    }
    logf("\n  Results written to %s (checkpoint: %s)\n", json.c_str(), chk.c_str());
    print_timings();
    logf("\n  Total wall time: %.1f s\n", wall_time());
    return converged ? 0 : 3;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "\nERROR: %s\n", e.what());
    return 2;
  }
}
