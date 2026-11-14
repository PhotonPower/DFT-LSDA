#include "dft/relax.hpp"

#include <cmath>

namespace lsda {

RelaxResult relax(Calculation& calc, const RelaxParams& rp) {
  const int nat = calc.crystal().natoms(), n = 3 * nat;
  RelaxResult res;
  RMatrix B(n, n);  // Hessian approximation
  for (int i = 0; i < n; ++i) B(i, i) = rp.hessian0;
  std::vector<double> x(n), g(n), x_old, g_old;
  for (int a = 0; a < nat; ++a) {
    const Vec3 r = calc.crystal().atom_cart(a);
    for (int d = 0; d < 3; ++d) x[3 * a + d] = r[d];
  }
  for (int step = 0; step <= rp.max_steps; ++step) {
    const bool ok = calc.scf();
    if (!ok) logf("  WARNING: SCF not converged during relaxation step %d\n", step);
    res.forces = calc.forces();
    res.steps = step;
    if (rp.on_step) rp.on_step(calc, res.forces, ok);
    double fmax = 0.0;
    for (int a = 0; a < nat; ++a) {
      fmax = std::max(fmax, norm(res.forces[a]));
      for (int d = 0; d < 3; ++d) g[3 * a + d] = -res.forces[a][d];
    }
    res.max_force = fmax;
    logf("\n  Relaxation step %3d:  free energy = %.10f Ha   max |F| = %.6f Ha/bohr\n", step, calc.energies().free, fmax);
    if (fmax < rp.force_tol) {
      res.converged = true;
      break;
    }
    if (step == rp.max_steps) break;

    // BFGS update of the Hessian
    if (!x_old.empty()) {
      std::vector<double> s(n), y(n), Bs(n, 0.0);
      double sy = 0.0, sBs = 0.0;
      for (int i = 0; i < n; ++i) {
        s[i] = x[i] - x_old[i];
        y[i] = g[i] - g_old[i];
        sy += s[i] * y[i];
      }
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) Bs[i] += B(i, j) * s[j];
      for (int i = 0; i < n; ++i) sBs += s[i] * Bs[i];
      if (sy > 1e-10 && sBs > 1e-10)
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j) B(i, j) += y[i] * y[j] / sy - Bs[i] * Bs[j] / sBs;
    }
    // step = -B^{-1} g using |eigenvalues| (guards against negative curvature)
    RMatrix V = B;
    std::vector<double> lam;
    eigh_real(V, lam);
    std::vector<double> dx(n, 0.0);
    for (int k = 0; k < n; ++k) {
      double proj = 0.0;
      for (int i = 0; i < n; ++i) proj += V(i, k) * g[i];
      const double c = -proj / std::max(std::abs(lam[k]), 1e-3);
      for (int i = 0; i < n; ++i) dx[i] += c * V(i, k);
    }
    double dmax = 0.0;
    for (int a = 0; a < nat; ++a)
      dmax = std::max(dmax, std::sqrt(dx[3 * a] * dx[3 * a] + dx[3 * a + 1] * dx[3 * a + 1] + dx[3 * a + 2] * dx[3 * a + 2]));
    const double scale = dmax > rp.max_step ? rp.max_step / dmax : 1.0;
    x_old = x;
    g_old = g;
    std::vector<Vec3> frac(nat);
    for (int a = 0; a < nat; ++a) {
      Vec3 r;
      for (int d = 0; d < 3; ++d) {
        x[3 * a + d] += scale * dx[3 * a + d];
        r[d] = x[3 * a + d];
      }
      frac[a] = calc.crystal().frac_from_cart(r);
    }
    calc.move_atoms(frac);
  }
  return res;
}

}  // namespace lsda
