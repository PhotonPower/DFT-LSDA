// r2SCAN meta-GGA: Furness, Kaplan, Ning, Perdew, Sun, J. Phys. Chem. Lett. 11, 8208 (2020) (regularized and
// restored form of SCAN, Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015)).
//
// Variables: spin densities, gradient invariants and the positive kinetic energy densities
//   tau_s = 1/2 sum_i f_i |grad psi_i,s|^2.
// Iso-orbital indicator (regularized): alpha = (tau - tau_W) / (tau_unif + eta tau_W), tau_W = |grad n|^2 / (8 n).
//   exchange   : F_x = { h1x(p) + f_x(alpha) [h0x - h1x(p)] } g_x(p), spin scaling E_x = (E_x[2n_up] + E_x[2n_dn])/2
//   correlation: eps_c = eps_c1 + f_c(alpha) (eps_c0 - eps_c1)
// All derivatives are obtained with dual numbers.
#include <algorithm>
#include <cmath>

#include "base/dual.hpp"
#include "xc/lda.hpp"

namespace lsda {

namespace {

constexpr double eta = 0.001, dp2 = 0.361;
constexpr double k0 = 0.174, k1 = 0.065, a1x = 4.9479, c1x = 0.667, c2x = 0.8, dx = 1.24;
constexpr double c1c = 0.64, c2c = 1.5, dc = 0.7;
constexpr double b1c = 0.0285764, b2c = 0.0889, b3c = 0.125541, chi_inf = 0.128026;
constexpr double cx[8] = {1.0, -0.667, -0.4445555, -0.663086601049, 1.451297044490, -0.887998041597, 0.234528941479, -0.023185843322};
constexpr double cc[8] = {1.0, -0.64, -0.4352, -1.535685604549, 3.061560252175, -1.915710236206, 0.516884468372, -0.051848879792};

// interpolation function between the single-orbital (alpha = 0) and the slowly varying (alpha = 1) limits
template <class T>
T switching(const T& a, const double* c, double c1, double c2, double d) {
  if (a.v < 0.0) return exp(-c1 * a / (1.0 - a));
  if (a.v <= 2.5) {
    T f = a * c[7] + c[6];
    for (int i = 5; i >= 0; --i) f = f * a + c[i];
    return f;
  }
  return -d * exp(c2 / (1.0 - a));
}

struct PWParams { double A, a1, b1, b2, b3, b4; };
constexpr PWParams pw_para{0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294};
constexpr PWParams pw_ferro{0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517};
constexpr PWParams pw_alpha{0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671};
constexpr double fz20 = 1.709921;

// Perdew-Wang function G(rs) and its derivative dG/drs
template <class T>
void pw_g(const PWParams& p, const T& rs, T& g, T& dg) {
  const T srs = sqrt(rs);
  const T q1 = (2.0 * p.A) * srs * (p.b1 + srs * (p.b2 + srs * (p.b3 + srs * p.b4)));
  const T dq1 = p.A * (p.b1 / srs + 2.0 * p.b2 + 3.0 * p.b3 * srs + 4.0 * p.b4 * rs);
  const T lg = log1p(1.0 / q1);
  g = (-2.0 * p.A) * (1.0 + p.a1 * rs) * lg;
  dg = (-2.0 * p.A * p.a1) * lg + (2.0 * p.A) * (1.0 + p.a1 * rs) * dq1 / (q1 * (q1 + 1.0));
}

// exchange energy per volume of a spin-unpolarized system with density r, |grad r|^2 = g2 and kinetic energy density t
template <class T>
T r2scan_exchange(const T& r, const T& g2, const T& t) {
  const double pi2 = pi * pi;
  const T r13 = cbrt(r), r23 = r13 * r13, r53 = r * r23, r83 = r53 * r;
  const T p = g2 / (4.0 * std::pow(3.0 * pi2, 2.0 / 3.0) * r83);
  const T tw = g2 / (8.0 * r), tu = 0.3 * std::pow(3.0 * pi2, 2.0 / 3.0) * r53;
  const T alpha = (t - tw) / (tu + eta * tw);
  const double c_eta = 20.0 / 27.0 + eta * 5.0 / 3.0;
  double c2 = 0.0;
  for (int i = 1; i < 8; ++i) c2 -= i * cx[i] * (1.0 - (1.0 + k0));
  const double mu = 10.0 / 81.0, dp4 = dp2 * dp2 * dp2 * dp2;
  const T x = (c_eta * c2 * exp(-p * p / dp4) + mu) * p;
  const T h1 = 1.0 + k1 - k1 / (1.0 + x / k1);
  const T fx = switching(alpha, cx, c1x, c2x, dx);
  T gx(1.0);
  if (p.v > 1e-12) gx = 1.0 - exp(-a1x / sqrt(sqrt(p)));
  const T F = (h1 + fx * ((1.0 + k0) - h1)) * gx;
  return (-0.75 * std::cbrt(3.0 / pi)) * r * r13 * F;
}

// correlation energy per volume; a, b: spin densities, g2 = |grad(a + b)|^2, t: total kinetic energy density
template <class T>
T r2scan_correlation(const T& a, const T& b, const T& g2, const T& t) {
  const double pi2 = pi * pi;
  const T n = a + b;
  const T n13 = cbrt(n), n23 = n13 * n13, n53 = n * n23, n83 = n53 * n;
  const T rs = std::cbrt(3.0 / (4.0 * pi)) / n13, srs = sqrt(rs);
  const T zeta = (a - b) / n;
  const T opz = 1.0 + zeta, omz = 1.0 - zeta;
  const T p13 = cbrt(opz), m13 = cbrt(omz);
  const T phi = 0.5 * (p13 * p13 + m13 * m13), phi3 = phi * phi * phi;
  const T dxz = 0.5 * (opz * p13 + omz * m13);                      // [(1+z)^(4/3) + (1-z)^(4/3)]/2
  const T dsz = 0.5 * (opz * p13 * p13 + omz * m13 * m13);          // [(1+z)^(5/3) + (1-z)^(5/3)]/2
  const T p = g2 / (4.0 * std::pow(3.0 * pi2, 2.0 / 3.0) * n83);    // s^2
  const T tw = g2 / (8.0 * n), tu = 0.3 * std::pow(3.0 * pi2, 2.0 / 3.0) * n53;
  const T alpha = (t - tw) / (tu * dsz + eta * tw);

  // ---- alpha = 0 limit ---------------------------------------------------------------------------
  const T den0 = 1.0 + b2c * srs + b3c * rs;
  const T e_lda0 = -b1c / den0;
  const T de_lda0 = b1c * (0.5 * b2c / srs + b3c) / (den0 * den0);
  const T w0 = expm1(-e_lda0 / b1c);
  const T ginf = 1.0 / sqrt(sqrt(1.0 + 4.0 * chi_inf * p));
  const T h0 = b1c * log1p(w0 * (1.0 - ginf));
  const T z2 = zeta * zeta, z4 = z2 * z2, z12 = z4 * z4 * z4;
  const T gc = (1.0 - 2.363 * (dxz - 1.0)) * (1.0 - z12);
  const T ec0 = (e_lda0 + h0) * gc;

  // ---- alpha = 1 limit (PW92 + gradient correction) -------------------------------------------------
  T g0, dg0, g1, dg1, ga, dga;
  pw_g(pw_para, rs, g0, dg0);
  pw_g(pw_ferro, rs, g1, dg1);
  pw_g(pw_alpha, rs, ga, dga);
  const T fz = (opz * p13 + omz * m13 - 2.0) / (2.0 * std::cbrt(2.0) - 2.0);
  const T e_lsda1 = g0 - ga * fz / fz20 * (1.0 - z4) + (g1 - g0) * fz * z4;
  const T de_lsda1 = dg0 - dga * fz / fz20 * (1.0 - z4) + (dg1 - dg0) * fz * z4;  // d/drs at fixed zeta
  const double gamma = (1.0 - std::log(2.0)) / pi2;
  const T w1 = expm1(-e_lsda1 / (gamma * phi3));
  const T beta = 0.066725 * (1.0 + 0.1 * rs) / (1.0 + 0.1778 * rs);
  const T kf = std::cbrt(3.0 * pi2) * n13;
  const T t2 = g2 / (4.0 * phi * phi * (4.0 / pi) * kf * n * n);
  const T y = beta * t2 / (gamma * w1);
  double dfc2 = 0.0;
  for (int i = 1; i < 8; ++i) dfc2 += i * cc[i];
  const double dp4 = dp2 * dp2 * dp2 * dp2;
  const T e_lsda0 = e_lda0 * gc;
  const T dy = dfc2 / (27.0 * gamma * dsz * phi3 * w1) *
               (20.0 * rs * (de_lda0 * gc - de_lsda1) - 45.0 * eta * (e_lsda0 - e_lsda1)) * p * exp(-p * p / dp4);
  const T g = 1.0 / sqrt(sqrt(1.0 + 4.0 * (y - dy)));
  const T h1 = gamma * phi3 * log1p(w1 * (1.0 - g));
  const T ec1 = e_lsda1 + h1;

  const T fc = switching(alpha, cc, c1c, c2c, dc);
  return n * (ec1 + fc * (ec0 - ec1));
}

}  // namespace

void MGGA_R2SCAN::evaluate_mgga(size_t n, const double* nup, const double* ndn, const double* suu, const double* sud,
                                const double* sdd, const double* tup, const double* tdn, double* exc, double* vup,
                                double* vdn, double* vsuu, double* vsud, double* vsdd, double* vtup, double* vtdn) const {
  const double tiny = 1e-12;
#pragma omp parallel for schedule(static) if (n > 2000)
  for (size_t i = 0; i < n; ++i) {
    const double rho_raw = (nup[i] > 0.0 ? nup[i] : 0.0) + (ndn[i] > 0.0 ? ndn[i] : 0.0);
    if (rho_raw < tiny) {
      exc[i] = vup[i] = vdn[i] = vsuu[i] = vsud[i] = vsdd[i] = vtup[i] = vtdn[i] = 0.0;
      continue;
    }
    // a vanishing spin density is replaced by a tiny fraction of the total (as in the PBE routine)
    const double floor = 1e-10 * rho_raw;
    const double dens[2] = {std::max(nup[i], floor), std::max(ndn[i], floor)};
    const double sig[2] = {std::max(suu[i], 0.0), std::max(sdd[i], 0.0)};
    const double tau[2] = {std::max(tup[i], 0.0), std::max(tdn[i], 0.0)};
    double e = 0.0, vn[2], vs[2], vt[2];
    using D3 = Dual<3>;
    for (int s = 0; s < 2; ++s) {  // exchange: E_x = (E_x[2 n_up] + E_x[2 n_dn]) / 2
      const D3 ex = r2scan_exchange(D3::variable(2.0 * dens[s], 0), D3::variable(4.0 * sig[s], 1), D3::variable(2.0 * tau[s], 2));
      e += 0.5 * ex.v;
      vn[s] = ex.d[0];
      vs[s] = 2.0 * ex.d[1];
      vt[s] = ex.d[2];
    }
    using D4 = Dual<4>;
    const double g2tot = std::max(suu[i] + 2.0 * sud[i] + sdd[i], 0.0);
    const D4 ec = r2scan_correlation(D4::variable(dens[0], 0), D4::variable(dens[1], 1), D4::variable(g2tot, 2), D4::variable(tau[0] + tau[1], 3));
    exc[i] = e + ec.v;
    vup[i] = vn[0] + ec.d[0];
    vdn[i] = vn[1] + ec.d[1];
    vsuu[i] = vs[0] + ec.d[2];
    vsdd[i] = vs[1] + ec.d[2];
    vsud[i] = 2.0 * ec.d[2];
    vtup[i] = vt[0] + ec.d[3];
    vtdn[i] = vt[1] + ec.d[3];
  }
}

}  // namespace lsda
