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

CellRelaxResult relax_cell(std::unique_ptr<Calculation>& calc, const RelaxParams& rp, const CellRelaxParams& cp) {
  const double gpa = 29421.02648438959;
  const int ca[6] = {0, 1, 2, 1, 0, 0}, cb[6] = {0, 1, 2, 2, 2, 1};  // xx yy zz yz xz xy
  CellRelaxResult res;
  RMatrix B(6, 6);  // Hessian with respect to the six strain variables (off-diagonal: eps_ij = eps_ji = x)
  bool hessian_set = false;
  std::vector<double> x(6, 0.0), g(6), x_old, g_old;
  // A reference lattice given by the user keeps the basis set fixed; otherwise the basis is rebuilt for
  // every cell at constant cutoff (requires ecut_smooth > 0 for a differentiable energy surface).
  bool fixed_basis = false;
  {
    const Crystal& c = calc->crystal();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        if (std::abs(c.A[i][j] - c.A_cut[i][j]) > 1e-12) fixed_basis = true;
  }
  if (!fixed_basis && calc->params().ecut_smooth <= 0.0)
    logf("  WARNING: cell relaxation at constant cutoff without ecut_smooth: the stress is not the derivative of the energy\n");
  for (int step = 0; step <= cp.max_steps; ++step) {
    res.steps = step;
    // ---- atoms at fixed cell, then the stress -----------------------------------------------------
    if (calc->crystal().natoms() > 1) {
      const RelaxResult rr = relax(*calc, rp);
      res.forces = rr.forces;
    } else {
      calc->scf();
      res.forces = calc->forces();
    }
    res.stress = calc->stress();
    const double vol = calc->crystal().volume;
    double dev = 0.0;
    for (int c = 0; c < 6; ++c) {
      const double d = res.stress[ca[c]][cb[c]] + (c < 3 ? cp.pressure : 0.0);
      dev = std::max(dev, std::abs(d));
      g[c] = vol * d * (c < 3 ? 1.0 : 2.0);
    }
    logf("\n  Cell step %3d:  free energy = %.10f Ha   volume = %.6f bohr^3   pressure = %.4f GPa   max |stress - target| = %.4f GPa\n",
         step, calc->energies().free, vol, -(res.stress[0][0] + res.stress[1][1] + res.stress[2][2]) / 3.0 * gpa, dev * gpa);
    if (cp.on_step) cp.on_step(*calc, res.forces, res.stress);

    if (dev < cp.stress_tol) {
      res.converged = true;
      break;
    }
    if (step == cp.max_steps) break;

    // ---- quasi-Newton step in strain space ----------------------------------------------------------
    if (!hessian_set) {
      for (int c = 0; c < 6; ++c) B(c, c) = cp.modulus * vol * (c < 3 ? 1.0 : 2.0);
      hessian_set = true;
    }
    if (!x_old.empty()) {
      std::vector<double> s(6), y(6), Bs(6, 0.0);
      double sy = 0.0, sBs = 0.0;
      for (int i = 0; i < 6; ++i) {
        s[i] = x[i] - x_old[i];
        y[i] = g[i] - g_old[i];
        sy += s[i] * y[i];
      }
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) Bs[i] += B(i, j) * s[j];
      for (int i = 0; i < 6; ++i) sBs += s[i] * Bs[i];
      if (sy > 1e-14 && sBs > 1e-14)
        for (int i = 0; i < 6; ++i)
          for (int j = 0; j < 6; ++j) B(i, j) += y[i] * y[j] / sy - Bs[i] * Bs[j] / sBs;
    }
    RMatrix V = B;
    std::vector<double> lam, dx(6, 0.0);
    eigh_real(V, lam);
    for (int k = 0; k < 6; ++k) {
      double proj = 0.0;
      for (int i = 0; i < 6; ++i) proj += V(i, k) * g[i];
      const double coef = -proj / std::max(std::abs(lam[k]), 1e-3 * cp.modulus * vol);
      for (int i = 0; i < 6; ++i) dx[i] += coef * V(i, k);
    }
    double dmax = 0.0;
    for (double v : dx) dmax = std::max(dmax, std::abs(v));
    const double scale = dmax > cp.max_strain ? cp.max_strain / dmax : 1.0;
    x_old = x;
    g_old = g;
    Mat3 eps{};
    for (int c = 0; c < 6; ++c) {
      dx[c] *= scale;
      x[c] += dx[c];
      eps[ca[c]][cb[c]] = eps[cb[c]][ca[c]] = dx[c];
    }
    // a_i' = (1 + eps) a_i, fractional coordinates unchanged
    Crystal next = calc->crystal();
    const Mat3 a_basis = next.A_cut;
    Mat3 a_new{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        a_new[i][j] = next.A[i][j];
        for (int k = 0; k < 3; ++k) a_new[i][j] += eps[j][k] * next.A[i][k];
      }
    next.set_lattice(a_new);
    if (fixed_basis) next.set_basis_lattice(a_basis);
    auto fresh = std::make_unique<Calculation>(std::move(next), calc->params());
    fresh->adopt_state(std::move(*calc));
    calc = std::move(fresh);
  }
  return res;
}

}  // namespace lsda
