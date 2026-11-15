// Core kinetic energy density for meta-GGA functionals.
//
// The JTH datasets contain the frozen core density but neither the core orbitals nor their kinetic energy density.
// The core orbitals are therefore recomputed: the all-electron potential of the reference atom is rebuilt from
// the core density and the bound all-electron partial waves (with the functional of the dataset), and the
// scalar-relativistic radial equation (Koelling and Harmon, without spin-orbit coupling)
//   P'' = [l(l+1)/r^2 + 2 M (V - eps)] P - V'/(2 M c^2) (P' - P/r),   M = 1 + (eps - V)/(2 c^2),   P = r R
// is solved on the grid of the dataset (second-order differences in the index variable, eigenvalues by node
// counting and bisection, eigenfunctions by inverse iteration). Checks: the recomputed core density and the core
// kinetic energy are compared with the values stored in the dataset (check_core_density, check_core_kinetic).
//   tau_c(r) = sum_c f_c/(4 pi) 1/2 [R_c'^2 + l(l+1) R_c^2 / r^2]
// The pseudo core kinetic energy density equals tau_c outside the matching radius of the pseudo core density and
// continues it inside by a polynomial in r^2 (value and two derivatives continuous); any smooth choice is valid
// because it is added on the plane-wave grid and subtracted inside the augmentation spheres.
#include <algorithm>
#include <cmath>

#include "paw/setup.hpp"

namespace lsda {

void PawSetup::prepare_mgga() {
  if (has_core_tau) return;
  const int n = grid.size();
  const double s4pi = std::sqrt(4 * pi), c_light = 137.035999084;
  tauc.assign(n, 0.0);
  tauct.assign(n, 0.0);
  check_core_density = check_core_kinetic = 0.0;
  if (n_core > 0.5) {
    // ---- all-electron potential of the reference atom -------------------------------------------------
    std::vector<double> dens(n + 2, 0.0);  // f(r) Y_00 convention
    for (int i = 0; i < n; ++i) {
      dens[i] = nc[i];
      for (int j = 0; j < nj; ++j) dens[i] += f_j[j] * phi[j][i] * phi[j][i] / s4pi;
    }
    const std::vector<double> vh = grid.hartree(0, std::vector<double>(dens.begin(), dens.begin() + n), n);
    const std::unique_ptr<XCFunctional> xc = make_xc(xc_name);
    LSDA_CHECK(xc->exx_fraction() == 0.0 && !xc->is_mgga(), "core states: unsupported functional of the dataset");
    std::vector<double> half(n), e(n), vxc(n), vdn(n);
    for (int i = 0; i < n; ++i) half[i] = 0.5 * dens[i] / s4pi;
    if (xc->is_gga()) {
      std::vector<double> dn = grid.derivative(dens, n);
      std::vector<double> sig(n), vs1(n), vs2(n), vs3(n), flux(n + 2, 0.0);
      for (int i = 0; i < n; ++i) {
        dn[i] /= s4pi;  // derivative of the total density
        sig[i] = 0.25 * dn[i] * dn[i];
      }
      xc->evaluate_gga(n, half.data(), half.data(), sig.data(), sig.data(), sig.data(), e.data(), vxc.data(), vdn.data(), vs1.data(),
                       vs2.data(), vs3.data());
      // v = de/dn - 1/r^2 d/dr [r^2 2 (de/d|grad n|^2) n'],  de/d|grad n|^2 = (v_uu + v_ud + v_dd)/4
      for (int i = 0; i < n; ++i) flux[i] = grid.r[i] * grid.r[i] * 0.5 * (vs1[i] + vs2[i] + vs3[i]) * dn[i];
      const std::vector<double> dflux = grid.derivative(flux, n);
      for (int i = 1; i < n; ++i) vxc[i] -= dflux[i] / (grid.r[i] * grid.r[i]);
      vxc[0] = vxc[1];
    } else {
      xc->evaluate(n, half.data(), half.data(), e.data(), vxc.data(), vdn.data());
    }
    std::vector<double> vsmooth(n + 2, 0.0);  // potential without the nuclear attraction
    for (int i = 0; i < n; ++i) vsmooth[i] = vh[i] / s4pi + vxc[i];
    vsmooth[n] = vsmooth[n + 1] = vsmooth[n - 1];
    const std::vector<double> dvs = grid.derivative(vsmooth, n);
    // range of the core states and grid quantities in the index variable
    int nmax = n - 3;
    while (nmax > 20 && grid.r[nmax] > 20.0) --nmax;
    std::vector<double> V(nmax + 1, 0.0), dV(nmax + 1, 0.0), rp(nmax + 1), rpp(nmax + 1);
    for (int i = 1; i <= nmax; ++i) {
      V[i] = -Z / grid.r[i] + vsmooth[i];
      dV[i] = Z / (grid.r[i] * grid.r[i]) + dvs[i];
      rp[i] = grid.dr[i];
      rpp[i] = 0.5 * (grid.dr[i + 1] - grid.dr[i - 1]);
    }
    // difference equation (1 - g/2) P(i-1) - (2 + h) P(i) + (1 + g/2) P(i+1) = 0 for the trial energy eps
    std::vector<double> g(nmax + 1), h(nmax + 1);
    auto coefficients = [&](int l, double eps) {
      for (int i = 1; i <= nmax; ++i) {
        const double r = grid.r[i], M = 1.0 + (eps - V[i]) / (2.0 * c_light * c_light);
        const double A = l * (l + 1) / (r * r) + 2.0 * M * (V[i] - eps), B = dV[i] / (2.0 * M * c_light * c_light);
        g[i] = -rpp[i] / rp[i] + B * rp[i];
        h[i] = rp[i] * rp[i] * (A + B / r);
      }
    };
    auto nodes = [&](int l, double eps) {
      coefficients(l, eps);
      double pm = 0.0, p = 1e-20;
      int count = 0;
      for (int i = 1; i < nmax; ++i) {
        const double pn = ((2.0 + h[i]) * p - (1.0 - 0.5 * g[i]) * pm) / (1.0 + 0.5 * g[i]);
        if (pn * p < 0.0) ++count;
        pm = p;
        p = pn;
        if (std::abs(p) > 1e200) {
          pm *= 1e-200;
          p *= 1e-200;
        }
      }
      return count;
    };
    // core shells: Aufbau order without the shells that are treated as valence states
    static const int order[15][2] = {{1, 0}, {2, 0}, {2, 1}, {3, 0}, {3, 1}, {4, 0}, {3, 2}, {4, 1}, {5, 0}, {4, 2}, {5, 1}, {6, 0}, {4, 3}, {5, 2}, {6, 1}};
    double electrons = 0.0, ekin = 0.0;
    std::vector<double> nc_new(n, 0.0);
    const std::vector<double> w = grid.weights(nmax + 1);
    for (const auto& shell : order) {
      if (electrons > n_core - 0.5) break;
      const int nq = shell[0], l = shell[1];
      bool valence = false;
      for (int j = 0; j < nj; ++j) valence = valence || (n_j[j] == nq && l_j[j] == l);
      if (valence) continue;
      const double occ = 2.0 * (2 * l + 1);
      electrons += occ;
      // eigenvalue: lowest energy with more than nq - l - 1 nodes
      const int target = nq - l - 1;
      double lo = -0.75 * Z * Z - 20.0, hi = 0.0;
      for (int it = 0; it < 100 && hi - lo > 1e-13 * (1.0 + std::abs(lo)); ++it) {
        const double mid = 0.5 * (lo + hi);
        if (nodes(l, mid) > target) hi = mid; else lo = mid;
      }
      const double eps = 0.5 * (lo + hi);
      // eigenfunction: inverse iteration with the tridiagonal matrix of the difference equation
      coefficients(l, eps - 1e-9 * (1.0 + std::abs(eps)));
      std::vector<double> P(nmax + 1, 0.0), cp(nmax + 1), dp(nmax + 1);
      for (int i = 1; i < nmax; ++i) P[i] = grid.r[i] * std::exp(-std::sqrt(-2.0 * eps) * grid.r[i] / (target + 1));
      for (int it = 0; it < 4; ++it) {
        // Thomas algorithm for rows i = 1 .. nmax-1 with P(0) = P(nmax) = 0
        for (int i = 1; i < nmax; ++i) {
          const double a = 1.0 - 0.5 * g[i], b = -(2.0 + h[i]), c = 1.0 + 0.5 * g[i];
          const double m = i == 1 ? b : b - a * cp[i - 1];
          cp[i] = c / m;
          dp[i] = (P[i] - (i == 1 ? 0.0 : a * dp[i - 1])) / m;
        }
        P[nmax - 1] = dp[nmax - 1];
        for (int i = nmax - 2; i >= 1; --i) P[i] = dp[i] - cp[i] * P[i + 1];
        double nrm = 0.0;
        for (int i = 1; i < nmax; ++i) nrm += w[i] * P[i] * P[i];
        nrm = 1.0 / std::sqrt(nrm);
        for (int i = 1; i < nmax; ++i) P[i] *= nrm;
      }
      double vexp = 0.0;
      for (int i = 1; i < nmax; ++i) vexp += w[i] * P[i] * P[i] * V[i];
      ekin += occ * (eps - vexp);
      for (int i = 1; i < nmax; ++i) {
        const double r = grid.r[i], R = P[i] / r, dP = (P[i + 1] - P[i - 1]) / (2.0 * rp[i]), dR = (dP - R) / r;
        nc_new[i] += occ / (4 * pi) * R * R;
        tauc[i] += s4pi * occ / (4 * pi) * 0.5 * (dR * dR + l * (l + 1) * R * R / (r * r));
      }
    }
    LSDA_CHECK(std::abs(electrons - n_core) < 0.5, "core states of " + symbol + ": cannot assign the core shells");
    tauc[0] = tauc[1];
    nc_new[0] = nc_new[1];
    double dev = 0.0;
    const std::vector<double> wf = grid.weights(n);
    for (int i = 0; i < n; ++i) dev += wf[i] * 4 * pi * grid.r[i] * grid.r[i] * std::abs(nc_new[i] - nc[i] / s4pi);
    check_core_density = dev / n_core;
    check_core_kinetic = e_kin_core != 0.0 ? std::abs(ekin / e_kin_core - 1.0) : 0.0;

    // ---- pseudo core kinetic energy density ---------------------------------------------------------
    double nmaxc = 0.0;
    for (int i = 0; i < n; ++i) nmaxc = std::max(nmaxc, std::abs(nc[i]));
    int im = ncut - 1;  // matching point: where the pseudo core density joins the core density, inside the sphere
    while (im > 4 && std::abs(nc[im] - nct[im]) < 1e-7 * nmaxc) --im;
    im = std::min(im + 2, grid.index_of(r_paw) - 1);
    tauct = tauc;
    const double rm = grid.r[im];
    std::vector<double> padded(tauc);
    padded.resize(n + 2, 0.0);
    const std::vector<double> d1 = grid.derivative(padded, n);
    std::vector<double> d1p(d1);
    d1p.resize(n + 2, 0.0);
    const std::vector<double> d2 = grid.derivative(d1p, n);
    const double f0 = tauc[im], f1 = d1[im], f2 = d2[im];
    const double C = (f2 - f1 / rm) / (8 * rm * rm), B = (f1 - 4 * C * rm * rm * rm) / (2 * rm), A = f0 - B * rm * rm - C * rm * rm * rm * rm;
    for (int i = 0; i < im; ++i) {
      const double r2 = grid.r[i] * grid.r[i];
      tauct[i] = std::max(0.0, A + B * r2 + C * r2 * r2);
    }
  }
  has_core_tau = true;

  // ---- radial pair functions of the valence kinetic energy density ---------------------------------------
  //   tau_ij(r, Omega) = 1/2 grad(phi_i Y_i) . grad(phi_j Y_j)
  //                    = sum_L G^L_ij Y_L 1/2 [phi_i' phi_j' + (l_i(l_i+1) + l_j(l_j+1) - l(l+1))/2 phi_i phi_j / r^2]
  const int nl = 2 * lmax + 1;
  tq_ae_.assign(nl, std::vector<std::vector<double>>(nq_, std::vector<double>(ncut, 0.0)));
  tq_ps_ = tq_ae_;
  std::vector<std::vector<double>> dphi(nj), dphit(nj);
  for (int j = 0; j < nj; ++j) {
    dphi[j] = grid.derivative(phi[j], ncut);
    dphit[j] = grid.derivative(phit[j], ncut);
  }
  for (int j1 = 0; j1 < nj; ++j1)
    for (int j2 = j1; j2 < nj; ++j2) {
      const int q = pair(j1, j2), l1 = l_j[j1], l2 = l_j[j2];
      for (int l = 0; l < nl; ++l) {
        const double ang = 0.5 * (l1 * (l1 + 1) + l2 * (l2 + 1) - l * (l + 1));
        for (int i = 0; i < ncut; ++i) {
          const double inv_r2 = grid.r[i] > 0.0 ? 1.0 / (grid.r[i] * grid.r[i]) : 0.0;
          tq_ae_[l][q][i] = 0.5 * (dphi[j1][i] * dphi[j2][i] + ang * inv_r2 * phi[j1][i] * phi[j2][i]);
          tq_ps_[l][q][i] = 0.5 * (dphit[j1][i] * dphit[j2][i] + ang * inv_r2 * phit[j1][i] * phit[j2][i]);
        }
      }
    }
}

}  // namespace lsda
