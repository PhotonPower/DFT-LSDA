#include "dft/neb.hpp"

#include <cmath>
#include <fstream>
#include <memory>

namespace lsda {

namespace {

using Coords = std::vector<Vec3>;  // Cartesian positions or forces of one image

double dotc(const Coords& a, const Coords& b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) s += dot(a[i], b[i]);
  return s;
}
Coords diff(const Coords& a, const Coords& b) {
  Coords d(a.size());
  for (size_t i = 0; i < a.size(); ++i)
    for (int k = 0; k < 3; ++k) d[i][k] = a[i][k] - b[i][k];
  return d;
}
void axpy(double f, const Coords& x, Coords& y) {
  for (size_t i = 0; i < x.size(); ++i)
    for (int k = 0; k < 3; ++k) y[i][k] += f * x[i][k];
}

}  // namespace

NebResult optimize_band(std::vector<BandCoords>& R, std::vector<double>& energy,
                        const std::function<bool(int, const BandCoords&, double&, BandCoords&)>& evaluate,
                        const NebParams& np, const std::function<void(int)>& on_step, bool verbose, double energy_unit) {
  const int ntot = static_cast<int>(R.size()), nim = ntot - 2, nat = static_cast<int>(R[0].size());
  LSDA_CHECK(nim >= 1 && static_cast<int>(energy.size()) == ntot, "optimize_band: inconsistent band");
  NebResult res;
  std::vector<Coords> F(ntot, Coords(nat, Vec3{0, 0, 0})), V(ntot, Coords(nat, Vec3{0, 0, 0}));
  double dt = 1.0, alpha = 0.1;
  int n_pos = 0;
  if (verbose) {
    logf("\n  NEB with %d images, spring constant %.3f%s\n", nim, np.spring, np.climb ? ", climbing image" : "");
    logf("  step   max NEB force        barrier   saddle image   energies relative to the first end point\n");
  }
  for (int step = 0; step <= np.max_steps; ++step) {
    res.steps = step;
    for (int i = 1; i <= nim; ++i)
      if (!evaluate(i, R[i], energy[i], F[i])) {
        res.stopped = true;
        res.energy = energy;
        return res;
      }
    if (on_step) on_step(step);
    res.saddle = 1;
    for (int i = 1; i <= nim; ++i)
      if (energy[i] > energy[res.saddle]) res.saddle = i;

    // ---- NEB forces: true force perpendicular to the path, spring force along it ----------------------
    const bool climbing = np.climb && step >= 2;
    std::vector<Coords> G(ntot, Coords(nat, Vec3{0, 0, 0}));
    double fmax = 0.0;
    for (int i = 1; i <= nim; ++i) {
      const Coords dp = diff(R[i + 1], R[i]), dm = diff(R[i], R[i - 1]);
      const double ep = energy[i + 1] - energy[i], em = energy[i] - energy[i - 1];
      Coords tau(nat, Vec3{0, 0, 0});
      if (ep > 0 && em > 0) {
        tau = dp;
      } else if (ep < 0 && em < 0) {
        tau = dm;
      } else {  // extremum: energy-weighted average of the two segments
        const double hi = std::max(std::abs(ep), std::abs(em)), lo = std::min(std::abs(ep), std::abs(em));
        const bool up = energy[i + 1] > energy[i - 1];
        axpy(up ? hi : lo, dp, tau);
        axpy(up ? lo : hi, dm, tau);
      }
      const double tn = std::sqrt(dotc(tau, tau));
      for (auto& v : tau)
        for (double& x : v) x /= tn;
      const double fpar = dotc(F[i], tau);
      G[i] = F[i];
      if (climbing && i == res.saddle) {
        axpy(-2.0 * fpar, tau, G[i]);  // invert the force along the path: climb to the saddle point
      } else {
        axpy(-fpar, tau, G[i]);
        axpy(np.spring * (std::sqrt(dotc(dp, dp)) - std::sqrt(dotc(dm, dm))), tau, G[i]);
      }
      for (const Vec3& g : G[i]) fmax = std::max(fmax, norm(g));
    }
    res.max_force = fmax;
    if (verbose) {
      std::string line;
      for (int i = 0; i < ntot; ++i) line += strf(" %8.4f", (energy[i] - energy[0]) * energy_unit);
      logf("  %4d   %12.6f   %12.5f   %8d      %s\n", step, fmax, (energy[res.saddle] - energy[0]) * energy_unit, res.saddle, line.c_str());
    }
    if (fmax < np.force_tol && (climbing || !np.climb)) {
      res.converged = true;
      break;
    }
    if (step == np.max_steps) break;

    // ---- FIRE step on all images ------------------------------------------------------------------------
    double p = 0.0, vv = 0.0, gg = 0.0;
    for (int i = 1; i <= nim; ++i) {
      p += dotc(G[i], V[i]);
      vv += dotc(V[i], V[i]);
      gg += dotc(G[i], G[i]);
    }
    if (p > 0.0) {
      const double mix = alpha * std::sqrt(vv / std::max(gg, 1e-300));
      for (int i = 1; i <= nim; ++i)
        for (int a = 0; a < nat; ++a)
          for (int k = 0; k < 3; ++k) V[i][a][k] = (1.0 - alpha) * V[i][a][k] + mix * G[i][a][k];
      if (++n_pos > 5) {
        dt = std::min(dt * 1.1, 10.0);
        alpha *= 0.99;
      }
    } else {
      for (int i = 1; i <= nim; ++i) V[i].assign(nat, Vec3{0, 0, 0});
      dt *= 0.5;
      alpha = 0.1;
      n_pos = 0;
    }
    double dmax = 0.0;
    for (int i = 1; i <= nim; ++i) {
      axpy(dt, G[i], V[i]);
      for (const Vec3& v : V[i]) dmax = std::max(dmax, dt * norm(v));
    }
    // limit the largest displacement; the velocities are reduced accordingly, so that the time step adapts
    if (dmax > np.max_step) {
      const double scale = np.max_step / dmax;
      for (int i = 1; i <= nim; ++i)
        for (auto& v : V[i])
          for (double& x : v) x *= scale;
    }
    for (int i = 1; i <= nim; ++i) axpy(dt, V[i], R[i]);
  }
  res.energy = energy;
  res.distance.assign(ntot, 0.0);
  for (int i = 1; i < ntot; ++i) {
    const Coords d = diff(R[i], R[i - 1]);
    res.distance[i] = res.distance[i - 1] + std::sqrt(dotc(d, d));
  }
  return res;
}

NebResult run_neb(const Crystal& initial, const std::vector<Vec3>& frac_final, const Params& params, const NebParams& np) {
  const int nat = initial.natoms(), nim = np.n_images, ntot = nim + 2;
  LSDA_CHECK(static_cast<int>(frac_final.size()) == nat, "neb: final structure has a different number of atoms");
  LSDA_CHECK(nim >= 1, "neb: at least one intermediate image is required");
  std::vector<double> energy(ntot, 0.0);
  std::vector<Coords> frac(ntot, Coords(nat));
  for (int i = 0; i < ntot; ++i)
    for (int a = 0; a < nat; ++a)
      for (int k = 0; k < 3; ++k)
        frac[i][a][k] = initial.atoms[a].frac[k] + (frac_final[a][k] - initial.atoms[a].frac[k]) * i / (ntot - 1.0);

  // restart: positions of the last completed step
  const std::string pos_file = np.prefix + ".neb.pos";
  if (np.restart) {
    std::ifstream in(pos_file);
    int n1 = 0, n2 = 0;
    if (in >> n1 >> n2 && n1 == ntot && n2 == nat) {
      std::vector<Coords> f(ntot, Coords(nat));
      bool ok = true;
      for (auto& img : f)
        for (auto& v : img)
          for (double& x : v) ok = ok && static_cast<bool>(in >> x);
      if (ok) {
        frac = f;
        logf("  NEB restart: image positions read from %s\n", pos_file.c_str());
      }
    }
  }

  // all images use the symmetry operations that the two end points have in common
  auto make_crystal = [&](int i) {
    Crystal c = initial;
    for (int a = 0; a < nat; ++a) c.atoms[a].frac = frac[i][a];
    c.sym_constraints = {frac[0], frac[ntot - 1]};
    return c;
  };
  Params quiet = params;
  quiet.verbosity = 0;
  auto chk_name = [&](int i) { return np.prefix + ".img" + std::to_string(i) + ".chk"; };
  NebResult stopped;
  stopped.stopped = true;

  // ---- end points (fixed) -----------------------------------------------------------------------
  for (int i : {0, ntot - 1}) {
    Calculation calc(make_crystal(i), quiet);
    if (np.restart) calc.load_checkpoint(chk_name(i));
    calc.scf();
    calc.save_checkpoint(chk_name(i));
    if (calc.stopped_by_time()) return stopped;
    energy[i] = calc.energies().free;
    double fmax = 0.0;
    for (const Vec3& f : calc.forces()) fmax = std::max(fmax, norm(f));
    logf("  NEB end point %d: free energy = %.10f Ha, max |F| = %.6f Ha/bohr%s\n", i, energy[i], fmax,
         fmax > 5 * np.force_tol ? "  (not relaxed!)" : "");
  }

  // ---- intermediate images ------------------------------------------------------------------------
  std::vector<std::unique_ptr<Calculation>> calc(ntot);
  for (int i = 1; i <= nim; ++i) {
    calc[i] = std::make_unique<Calculation>(make_crystal(i), quiet);
    if (np.restart) calc[i]->load_checkpoint(chk_name(i));
  }
  std::vector<Coords> R(ntot, Coords(nat));
  for (int i = 0; i < ntot; ++i)
    for (int a = 0; a < nat; ++a) R[i][a] = initial.cart(frac[i][a]);

  auto evaluate = [&](int i, const Coords& r, double& e, Coords& f) {
    bool moved = false;
    for (int a = 0; a < nat; ++a) {
      const Vec3 x = initial.frac_from_cart(r[a]);
      for (int k = 0; k < 3; ++k)
        if (std::abs(x[k] - frac[i][a][k]) > 1e-14) moved = true;
      frac[i][a] = x;
    }
    if (moved) calc[i]->move_atoms(frac[i]);
    calc[i]->scf();
    calc[i]->save_checkpoint(chk_name(i));
    if (calc[i]->stopped_by_time()) return false;
    f = calc[i]->forces();
    e = calc[i]->energies().free;
    return true;
  };
  auto on_step = [&](int) {  // positions of this step (consistent with the checkpoints) for restarts
    std::ofstream out(pos_file);
    out.precision(17);
    out << ntot << " " << nat << "\n";
    for (const auto& img : frac)
      for (const auto& v : img) out << v[0] << " " << v[1] << " " << v[2] << "\n";
  };
  NebResult res = optimize_band(R, energy, evaluate, np, on_step);
  res.frac = frac;
  return res;
}

void write_neb_results(const std::string& path, const Crystal& c, const NebResult& r) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  LSDA_CHECK(f != nullptr, "cannot write '" + path + "'");
  const int n = static_cast<int>(r.energy.size());
  std::fprintf(f, "{\n  \"converged\": %s,\n  \"steps\": %d,\n  \"max_force\": %.10f,\n", r.converged ? "true" : "false", r.steps, r.max_force);
  std::fprintf(f, "  \"units\": {\"energy\": \"Hartree\", \"length\": \"bohr\"},\n  \"saddle_image\": %d,\n", r.saddle);
  std::fprintf(f, "  \"barrier_forward_eV\": %.8f,\n  \"barrier_backward_eV\": %.8f,\n", (r.energy[r.saddle] - r.energy[0]) * hartree_in_ev,
               (r.energy[r.saddle] - r.energy[n - 1]) * hartree_in_ev);
  std::fprintf(f, "  \"images\": [\n");
  for (int i = 0; i < n; ++i) {
    std::fprintf(f, "    {\"free_energy\": %.12f, \"path_length\": %.8f, \"atoms\": [", r.energy[i], r.distance.empty() ? 0.0 : r.distance[i]);
    for (int a = 0; a < c.natoms(); ++a)
      std::fprintf(f, "{\"species\": \"%s\", \"frac\": [%.12f, %.12f, %.12f]}%s", c.species[c.atoms[a].species].name.c_str(), r.frac[i][a][0],
                   r.frac[i][a][1], r.frac[i][a][2], a + 1 < c.natoms() ? ", " : "");
    std::fprintf(f, "]}%s\n", i + 1 < n ? "," : "");
  }
  std::fprintf(f, "  ]\n}\n");
  std::fclose(f);
}

}  // namespace lsda
