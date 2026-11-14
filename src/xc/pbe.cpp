// Perdew-Burke-Ernzerhof generalized gradient approximation (spin-polarized).
//   exchange   : E_x[n_up, n_dn] = (E_x[2 n_up] + E_x[2 n_dn]) / 2,  e_x = e_x^LDA(n) F_x(s),
//                F_x = 1 + kappa - kappa / (1 + mu s^2 / kappa),  s = |grad n| / (2 k_F n)
//   correlation: e_c = n [eps_c^PW92(rs, zeta) + H(rs, zeta, t)],
//                H = gamma phi^3 ln[1 + beta/gamma t^2 (1 + A t^2)/(1 + A t^2 + A^2 t^4)],
//                A = beta/gamma / (exp(-eps_c/(gamma phi^3)) - 1),  t = |grad n| / (2 phi k_s n)
// The exchange derivatives are coded analytically; the correlation part is differentiated with
// dual numbers (variables n_up, n_dn, |grad n|^2).
#include <algorithm>
#include <cmath>

#include "base/dual.hpp"
#include "xc/lda.hpp"

namespace lsda {

namespace {

constexpr double pbe_kappa = 0.804;
constexpr double pbe_beta = 0.06672455060314922;
const double pbe_mu = pbe_beta * pi * pi / 3.0;
const double pbe_gamma = (1.0 - std::log(2.0)) / (pi * pi);

struct PWParams { double A, a1, b1, b2, b3, b4; };
constexpr PWParams pw_para{0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294};
constexpr PWParams pw_ferro{0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517};
constexpr PWParams pw_alpha{0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671};
constexpr double fz20 = 1.709921;

template <class T>
T pw_g(const PWParams& p, const T& rs, const T& srs) {
  const T q1 = (2.0 * p.A) * srs * (p.b1 + srs * (p.b2 + srs * (p.b3 + srs * p.b4)));
  return (-2.0 * p.A) * (1.0 + p.a1 * rs) * log1p(1.0 / q1);
}

// correlation energy per volume; a, b: spin densities (> 0), g2 = |grad(a + b)|^2
template <class T>
T pbe_correlation(const T& a, const T& b, const T& g2) {
  const T rho = a + b;
  const T rho13 = cbrt(rho);
  const T rs = std::cbrt(3.0 / (4.0 * pi)) / rho13, srs = sqrt(rs);
  const T zeta = (a - b) / rho;
  const T opz = 1.0 + zeta, omz = 1.0 - zeta;
  const T p13 = cbrt(opz), m13 = cbrt(omz);
  // PW92 uniform-gas correlation energy per particle
  const T e0 = pw_g(pw_para, rs, srs), e1 = pw_g(pw_ferro, rs, srs), ac = -pw_g(pw_alpha, rs, srs);
  const T f = (opz * p13 + omz * m13 - 2.0) / (2.0 * std::cbrt(2.0) - 2.0);
  const T z2 = zeta * zeta, z4 = z2 * z2;
  const T eps = e0 + ac * f / fz20 * (1.0 - z4) + (e1 - e0) * f * z4;
  // gradient correction
  const T phi = 0.5 * (p13 * p13 + m13 * m13), phi3 = phi * phi * phi;
  const T kf = std::cbrt(3.0 * pi * pi) * rho13;
  const T ks2 = (4.0 / pi) * kf;
  const T t2 = g2 / (4.0 * phi * phi * ks2 * rho * rho);
  const T A = (pbe_beta / pbe_gamma) / expm1(-eps / (pbe_gamma * phi3));
  const T at2 = A * t2;
  const T H = pbe_gamma * phi3 * log1p((pbe_beta / pbe_gamma) * t2 * (1.0 + at2) / (1.0 + at2 + at2 * at2));
  return rho * (eps + H);
}

// Short-range PBE exchange enhancement factor of Henderson, Janesko and Scuseria (HJS), Eqs. (37)-(43) of
// J. Chem. Phys. 128, 194105 (2008), with the parameters of the PBE exchange hole.
template <class T>
T hjs_enhancement(const T& s, const T& nu) {
  const double A = 0.757211, B = -0.106364, C = -0.118649, D = 0.609650;
  static const double a[6] = {0.0159941, 0.0852995, -0.160368, 0.152645, -0.0971263, 0.0422061};  // s^2 ... s^7
  static const double b[9] = {5.33319, -12.4780, 11.0988, -5.11013, 1.71468, -0.610380, 0.307555, -0.0770547, 0.0334840};
  const T s2 = s * s;
  T num = s * a[5] + a[4];
  for (int i = 3; i >= 0; --i) num = num * s + a[i];
  num = num * s2;
  T den = s * b[8] + b[7];
  for (int i = 6; i >= 0; --i) den = den * s + b[i];
  den = den * s + 1.0;
  const T H = num / den;
  const T zeta = s2 * H, eta = zeta + A, lam = zeta + D;
  const T nu2 = nu * nu;
  const T sl_nu = sqrt(lam + nu2), sz_nu = sqrt(zeta + nu2), se_nu = sqrt(eta + nu2);
  const T chi = nu / sl_nu, chi2 = chi * chi, chi3 = chi2 * chi, chi5 = chi3 * chi2;
  const T Fbar = 1.0 - s2 / (27.0 * C * (1.0 + 0.25 * s2)) - zeta / (2.0 * C);
  const T lam2 = lam * lam, lam3 = lam2 * lam, lam72 = lam3 * sqrt(lam);
  const T EG = -0.4 * C * Fbar * lam - (4.0 / 15.0) * B * lam2 - 1.2 * A * lam3 - 0.8 * std::sqrt(pi) * lam72 -
               2.4 * lam72 * (sqrt(zeta) - sqrt(eta));
  return A - (4.0 / 9.0) * B * (1.0 - chi) / lam - (4.0 / 9.0) * C * Fbar * (1.0 - 1.5 * chi + 0.5 * chi3) / lam2 -
         (8.0 / 9.0) * EG * (1.0 - 1.875 * chi + 1.25 * chi3 - 0.375 * chi5) / lam3 + 2.0 * nu * (sz_nu - se_nu) +
         2.0 * zeta * log((nu + sz_nu) / (nu + sl_nu)) - 2.0 * eta * log((nu + se_nu) / (nu + sl_nu));
}

}  // namespace

double sr_pbe_enhancement(double s, double nu, double* dF_ds, double* dF_dnu) {
  // the model is used for s <= 8 (beyond: constant in s; such gradients occur only in density tails)
  const double s_max = 8.0, s_min = 1e-7;
  const bool capped = s > s_max;
  using D2 = Dual<2>;
  const D2 F = hjs_enhancement(D2::variable(std::clamp(s, s_min, s_max), 0), D2::variable(nu, 1));
  if (dF_ds) *dF_ds = capped ? 0.0 : F.d[0];
  if (dF_dnu) *dF_dnu = F.d[1];
  return F.v;
}

std::string GGA_PBE::name() const {
  if (exx_ == 0.0) return "GGA (Perdew-Burke-Ernzerhof)";
  if (omega_ == 0.0) return strf("hybrid PBE0 type (%.4g exact exchange + PBE)", exx_);
  return strf("hybrid HSE type (%.4g screened exact exchange, omega = %.4g/bohr, HJS short-range PBE exchange)", exx_, omega_);
}

void GGA_PBE::evaluate_gga(size_t n, const double* nup, const double* ndn, const double* suu, const double* sud,
                           const double* sdd, double* exc, double* vup, double* vdn, double* vsuu, double* vsud,
                           double* vsdd) const {
  const double ax = -0.75 * std::cbrt(3.0 / pi);
  const double kf2 = std::pow(3.0 * pi * pi, 2.0 / 3.0);
  const double tiny = 1e-12;
#pragma omp parallel for schedule(static) if (n > 5000)
  for (size_t i = 0; i < n; ++i) {
    const double rho_raw = (nup[i] > 0.0 ? nup[i] : 0.0) + (ndn[i] > 0.0 ? ndn[i] : 0.0);
    if (rho_raw < tiny) {
      exc[i] = vup[i] = vdn[i] = vsuu[i] = vsud[i] = vsdd[i] = 0.0;
      continue;
    }
    // a vanishing spin density is replaced by a tiny fraction of the total (the zeta -> +-1 limits of the
    // derivatives are singular); this does not change the energy to working precision
    const double floor = 1e-10 * rho_raw;
    const double dens[2] = {std::max(nup[i], floor), std::max(ndn[i], floor)};
    const double sig[2] = {std::max(suu[i], 0.0), std::max(sdd[i], 0.0)};
    double e = 0.0, vn[2], vs[2];
    // ---- exchange: spin scaling ---------------------------------------------------------------
    for (int s = 0; s < 2; ++s) {
      const double r2 = 2.0 * dens[s], g2 = 4.0 * sig[s];
      const double r13 = std::cbrt(r2), r43 = r2 * r13;
      const double s2 = g2 / (4.0 * kf2 * r43 * r43);
      const double den = 1.0 + pbe_mu * s2 / pbe_kappa;
      double F = 1.0 + pbe_kappa - pbe_kappa / den, dF = pbe_mu / (den * den);  // dF/d(s^2)
      double nu_dF = 0.0;  // nu dF/dnu (hybrid with screened exchange)
      if (exx_ != 0.0 && omega_ == 0.0) {
        F *= 1.0 - exx_;
        dF *= 1.0 - exx_;
      } else if (exx_ != 0.0) {  // subtract the fraction of short-range exchange that is treated exactly
        const double sv = std::sqrt(s2), nu = omega_ / (std::sqrt(kf2) * r13);
        double fs, fnu;
        const double Fsr = sr_pbe_enhancement(sv, nu, &fs, &fnu);
        F -= exx_ * Fsr;
        dF -= exx_ * fs / (2.0 * std::max(sv, 1e-7));
        nu_dF = -exx_ * nu * fnu;
      }
      e += 0.5 * ax * r43 * F;
      vn[s] = ax * r13 * ((4.0 / 3.0) * F - (8.0 / 3.0) * s2 * dF - (1.0 / 3.0) * nu_dF);
      vs[s] = 0.5 * ax * dF / (kf2 * r43);
    }
    // ---- correlation ----------------------------------------------------------------------------
    const double g2tot = std::max(suu[i] + 2.0 * sud[i] + sdd[i], 0.0);
    using D3 = Dual<3>;
    const D3 ec = pbe_correlation(D3::variable(dens[0], 0), D3::variable(dens[1], 1), D3::variable(g2tot, 2));
    exc[i] = e + ec.v;
    vup[i] = vn[0] + ec.d[0];
    vdn[i] = vn[1] + ec.d[1];
    vsuu[i] = vs[0] + ec.d[2];
    vsdd[i] = vs[1] + ec.d[2];
    vsud[i] = 2.0 * ec.d[2];
  }
}

void GGA_PBE::evaluate(size_t n, const double* nup, const double* ndn, double* exc, double* vup, double* vdn) const {
  const std::vector<double> zero(n, 0.0);
  std::vector<double> a(n), b(n), c(n);
  evaluate_gga(n, nup, ndn, zero.data(), zero.data(), zero.data(), exc, vup, vdn, a.data(), b.data(), c.data());
}

}  // namespace lsda
