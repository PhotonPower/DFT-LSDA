// lsda - plane-wave PAW density-functional code (local spin density approximation)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>

#include "base/parallel.hpp"
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

static int run(int argc, char** argv) {
  if (argc != 2) {
    if (par::root()) std::fprintf(stderr, "usage: %s <input file>\n", argv[0]);
    return 1;
  }
  try {
    wall_time();
    logf("\n  lsda - plane-wave PAW code, local spin density approximation\n");
    logf("  input file: %s\n", argv[1]);
    if (par::size() > 1) logf("  MPI processes: %d (k-points are distributed over the processes)\n", par::size());
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
    if (inp.task == "elastic") {
      const ElasticResult r = run_elastic(inp.crystal, inp.params, inp.relax, inp.strain);
      if (r.stopped) {
        logf("\n  Stopped by the wall-time limit (elastic constants cannot be restarted; increase max_seconds).\n");
        return 4;
      }
      const double gpa = 29421.02648438959;
      logf("\n  Elastic constants C_ij (GPa), Voigt order xx yy zz yz xz xy; %d strained cells, asymmetry %.3f GPa\n", r.n_strained,
           r.asymmetry * gpa);
      for (int i = 0; i < 6; ++i) {
        std::string line;
        for (int j = 0; j < 6; ++j) line += strf(" %10.3f", r.C(i, j) * gpa);
        logf("   %s\n", line.c_str());
      }
      const RMatrix& C = r.C;
      logf("  Bulk modulus (Voigt average): %.3f GPa\n", (C(0, 0) + C(1, 1) + C(2, 2) + 2.0 * (C(0, 1) + C(0, 2) + C(1, 2))) / 9.0 * gpa);
      logf("  C11 = %.3f, C12 = %.3f, C44 = %.3f GPa;  C' = (C11 - C12)/2 = %.3f GPa\n", C(0, 0) * gpa, C(0, 1) * gpa, C(3, 3) * gpa,
           0.5 * (C(0, 0) - C(0, 1)) * gpa);
      RMatrix ev = C;
      std::vector<double> lam;
      eigh_real(ev, lam);
      logf(lam[0] > 0.0 ? "  Mechanically stable (all eigenvalues of C positive, smallest %.3f GPa)\n"
                        : "  WARNING: mechanically unstable (smallest eigenvalue of C: %.3f GPa)\n", lam[0] * gpa);
      write_elastic_results(inp.prefix + ".elastic.json", inp.crystal, r);
      logf("\n  Elastic constants written to %s.elastic.json\n", inp.prefix.c_str());
      print_timings();
      logf("\n  Total wall time: %.1f s\n", wall_time());
      return 0;
    }
    if (inp.task == "phonons") {
      const int* nsc = inp.phonon_supercell;
      const bool supercell = nsc[0] * nsc[1] * nsc[2] > 1;
      const Crystal prim = inp.crystal;
      Crystal cell = prim;
      Params params = inp.params;
      if (supercell) {  // force constants from a supercell of the (primitive) input cell
        cell = make_supercell(prim, nsc);
        for (int d = 0; d < 3; ++d) params.kgrid[d] = std::max(1, (inp.params.kgrid[d] + nsc[d] - 1) / nsc[d]);
        logf("\n  Phonons: %d x %d x %d supercell with %d atoms, k-point grid %d x %d x %d\n", nsc[0], nsc[1], nsc[2], cell.natoms(),
             params.kgrid[0], params.kgrid[1], params.kgrid[2]);
      }
      std::vector<double> masses, masses_prim;
      for (const Atom& a : prim.atoms) {
        const Species& sp = prim.species[a.species];
        masses_prim.push_back(sp.mass > 0.0 ? sp.mass : atomic_mass(static_cast<int>(std::lround(sp.setup.Z))));
      }
      for (int i = 0; i < cell.natoms(); ++i) masses.push_back(masses_prim[i % prim.natoms()]);
      const PhononResult r = run_phonons(cell, params, inp.phonons, masses);
      if (r.stopped) {
        logf("\n  Stopped by the wall-time limit; rerun with 'restart = true' (completed displacements are kept).\n");
        return 4;
      }
      logf("\n  Force constants: acoustic sum rule violated by %.2e, asymmetry %.2e Ha/bohr^2 (both removed)\n", r.asr_violation, r.asymmetry);
      logf("\n  Gamma-point phonons of the %scell\n   mode     THz        cm^-1       meV\n", supercell ? "super" : "");
      int imaginary = 0;
      for (size_t i = 0; i < r.frequency_thz.size(); ++i) {
        const double nu = r.frequency_thz[i];
        if (r.frequency_thz.size() <= 60 || nu < -0.05)
          logf("  %5zu %10.4f %10.3f %10.4f%s\n", i + 1, nu, nu * 33.35640951982, nu * 4.135667696, nu < -0.05 ? "   imaginary" : "");
        if (nu < -0.05) ++imaginary;
      }
      if (imaginary > 0) logf("  WARNING: %d imaginary modes (structure not at a minimum, or numerical noise)\n", imaginary);
      write_phonon_results(inp.prefix + ".phonons.json", cell, masses, r);
      logf("\n  Frequencies written to %s.phonons.json\n", inp.prefix.c_str());
      if (supercell && par::root()) {
        // ---- dispersion along the path ---------------------------------------------------------------
        if (inp.kpath.size() >= 2) {
          const std::string df = inp.prefix + ".phdisp";
          std::FILE* f = std::fopen(df.c_str(), "w");
          LSDA_CHECK(f != nullptr, "cannot write '" + df + "'");
          std::fprintf(f, "# phonon dispersion: path length (1/bohr), %d frequencies in THz (imaginary as negative)\n# special points:", 3 * prim.natoms());
          double d = 0.0;
          std::string ticks = strf("  %s = %.6f", inp.kpath[0].first.c_str(), 0.0);
          std::vector<std::pair<double, Vec3>> pts;
          for (size_t seg = 0; seg + 1 < inp.kpath.size(); ++seg) {
            const Vec3 &q0 = inp.kpath[seg].second, &q1 = inp.kpath[seg + 1].second;
            Vec3 dq{};
            for (int j = 0; j < 3; ++j)
              for (int i = 0; i < 3; ++i) dq[j] += (q1[i] - q0[i]) * prim.B[i][j];
            const double len = norm(dq);
            for (int i = (seg == 0 ? 0 : 1); i <= inp.kpath_points; ++i) {
              const double t = static_cast<double>(i) / inp.kpath_points;
              pts.push_back({d + t * len, Vec3{q0[0] + t * (q1[0] - q0[0]), q0[1] + t * (q1[1] - q0[1]), q0[2] + t * (q1[2] - q0[2])}});
            }
            d += len;
            ticks += strf("  %s = %.6f", inp.kpath[seg + 1].first.c_str(), d);
          }
          std::fprintf(f, "%s\n", ticks.c_str());
          for (const auto& [x, q] : pts) {
            std::fprintf(f, "%10.6f", x);
            for (double nu : phonon_frequencies(prim, cell, r.force_constants, masses_prim, q)) std::fprintf(f, " %10.5f", nu);
            std::fprintf(f, "\n");
          }
          std::fclose(f);
          logf("  Phonon dispersion (%zu q-points) written to %s\n", pts.size(), df.c_str());
        }
        // ---- frequencies on a q-mesh, energy and volume per primitive cell (for tools/qha.py) --------------
        const int* nq = inp.phonon_qgrid;
        const std::string mf = inp.prefix + ".phmesh.json";
        std::FILE* f = std::fopen(mf.c_str(), "w");
        LSDA_CHECK(f != nullptr, "cannot write '" + mf + "'");
        const int ncell = nsc[0] * nsc[1] * nsc[2];
        std::fprintf(f, "{\n  \"n_atoms\": %d,\n  \"volume\": %.10f,\n  \"free_energy\": %.12f,\n  \"qgrid\": [%d, %d, %d],\n", prim.natoms(),
                     prim.volume, r.energy / ncell, nq[0], nq[1], nq[2]);
        std::fprintf(f, "  \"comment\": \"per primitive cell; Hartree, bohr^3; one list of mode frequencies (THz) per q-point of the mesh\",\n");
        std::fprintf(f, "  \"frequencies_THz\": [\n");
        int n_imag = 0;
        for (int i = 0; i < nq[0]; ++i)
          for (int j = 0; j < nq[1]; ++j)
            for (int k = 0; k < nq[2]; ++k) {
              const Vec3 q{(i + 0.5) / nq[0], (j + 0.5) / nq[1], (k + 0.5) / nq[2]};  // shifted mesh avoids Gamma
              const std::vector<double> nu = phonon_frequencies(prim, cell, r.force_constants, masses_prim, q);
              std::fprintf(f, "    [");
              for (size_t m = 0; m < nu.size(); ++m) {
                std::fprintf(f, "%.6f%s", nu[m], m + 1 < nu.size() ? ", " : "]");
                if (nu[m] < -0.05) ++n_imag;
              }
              std::fprintf(f, "%s\n", (i + 1 == nq[0] && j + 1 == nq[1] && k + 1 == nq[2]) ? "" : ",");
            }
        std::fprintf(f, "  ]\n}\n");
        std::fclose(f);
        logf("  Frequencies on a %d x %d x %d q-mesh written to %s%s\n", nq[0], nq[1], nq[2], mf.c_str(),
             n_imag ? strf("  (WARNING: %d imaginary values)", n_imag).c_str() : "");
      }
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
    if (inp.write_density) {
      calc.write_cube(inp.prefix + ".rho.cube", false);
      if (calc.n_spin() == 2) calc.write_cube(inp.prefix + ".mag.cube", true);
      logf("\n  Pseudo density written to %s.rho.cube%s\n", inp.prefix.c_str(), calc.n_spin() == 2 ? " (magnetization: .mag.cube)" : "");
    }
    if (inp.task == "bands" && converged) {
      // non-self-consistent eigenvalues along the k-path in the converged potential
      const Crystal& c = calc.crystal();
      std::vector<Vec3> ks;
      std::vector<double> dist;
      std::vector<std::pair<double, std::string>> ticks;
      double d = 0.0;
      for (size_t seg = 0; seg + 1 < inp.kpath.size(); ++seg) {
        const Vec3 &k0 = inp.kpath[seg].second, &k1 = inp.kpath[seg + 1].second;
        Vec3 dk{};
        for (int j = 0; j < 3; ++j)
          for (int i = 0; i < 3; ++i) dk[j] += (k1[i] - k0[i]) * c.B[i][j];
        const double len = norm(dk);
        if (seg == 0) ticks.push_back({0.0, inp.kpath[0].first});
        for (int i = (seg == 0 ? 0 : 1); i <= inp.kpath_points; ++i) {
          const double t = static_cast<double>(i) / inp.kpath_points;
          ks.push_back({k0[0] + t * (k1[0] - k0[0]), k0[1] + t * (k1[1] - k0[1]), k0[2] + t * (k1[2] - k0[2])});
          dist.push_back(d + t * len);
        }
        d += len;
        ticks.push_back({d, inp.kpath[seg + 1].first});
      }
      const int nb = calc.n_bands(), ns = calc.n_spin();
      std::vector<double> e(ks.size() * ns * nb, 0.0);
      double check = 0.0;
      int n_checked = 0;
      for (size_t ik = 0; ik < ks.size(); ++ik) {
        if (!par::owns(static_cast<int>(ik))) continue;
        const auto ev = calc.eigenvalues_at(ks[ik]);
        for (int s = 0; s < ns; ++s)
          for (int n = 0; n < nb; ++n) e[(ik * ns + s) * nb + n] = ev[s][n];
        // points of the path that belong to the SCF k-point set must reproduce the SCF eigenvalues
        for (size_t jk = 0; jk < calc.kpoints().size(); ++jk) {
          const Vec3& q = calc.kpoints()[jk].frac;
          if (std::abs(q[0] - ks[ik][0]) + std::abs(q[1] - ks[ik][1]) + std::abs(q[2] - ks[ik][2]) > 1e-9) continue;
          ++n_checked;
          for (int s = 0; s < ns; ++s)
            for (int n = 0; n < nb / 2; ++n) check = std::max(check, std::abs(ev[s][n] - calc.scf_eigenvalues(static_cast<int>(jk), s)[n]));
        }
      }
      par::sum(e.data(), e.size());
      par::max(&check, 1);
      par::sum(&n_checked, 1);
      if (par::root()) {
        const std::string bf = inp.prefix + ".bands";
        std::FILE* f = std::fopen(bf.c_str(), "w");
        LSDA_CHECK(f != nullptr, "cannot write '" + bf + "'");
        std::fprintf(f, "# band structure: column 1 = path length (1/bohr), then %d bands%s; energies in eV relative to E_F\n", nb,
                     ns == 2 ? " spin up followed by the bands of spin down" : "");
        std::fprintf(f, "# the highest bands are not converged (increase nbands if they are needed)\n# special points:");
        for (const auto& [x, label] : ticks) std::fprintf(f, "  %s = %.6f", label.c_str(), x);
        std::fprintf(f, "\n");
        for (size_t ik = 0; ik < ks.size(); ++ik) {
          std::fprintf(f, "%10.6f", dist[ik]);
          for (int i = 0; i < ns * nb; ++i) std::fprintf(f, " %10.5f", (e[ik * ns * nb + i] - calc.fermi_level()) * hartree_in_ev);
          std::fprintf(f, "\n");
        }
        std::fclose(f);
        logf("\n  Band structure (%zu k-points) written to %s\n", ks.size(), bf.c_str());
        if (n_checked > 0) logf("  Consistency with the SCF eigenvalues at %d common k-point(s): max deviation %.2e Ha\n", n_checked, check);
      }
    }
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
    if (par::size() > 1) par::abort(2);  // take the other processes down as well
    return 2;
  }
}

int main(int argc, char** argv) {
  par::init(&argc, &argv);
  const int code = run(argc, argv);
  par::finalize();
  return code;
}
