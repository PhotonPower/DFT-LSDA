#include "xc/lda.hpp"

#include <cmath>

namespace lsda {

namespace {

// Perdew-Wang 92 interpolation G(rs) = -2A(1+a1 rs) ln(1 + 1/(2A(b1 rs^1/2 + b2 rs + b3 rs^3/2 + b4 rs^2)))
struct PWParams { double A, a1, b1, b2, b3, b4; };
constexpr PWParams pw_para{0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294};    // eps_c(rs, 0)
constexpr PWParams pw_ferro{0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517};  // eps_c(rs, 1)
constexpr PWParams pw_alpha{0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671};  // -alpha_c(rs)
constexpr double fz20 = 1.709921;                                                  // f''(0)

inline void pw_g(const PWParams& p, double rs, double srs, double& g, double& dg) {
  const double q0 = -2.0 * p.A * (1.0 + p.a1 * rs);
  const double q1 = 2.0 * p.A * srs * (p.b1 + srs * (p.b2 + srs * (p.b3 + srs * p.b4)));
  const double dq1 = p.A * (p.b1 / srs + 2.0 * p.b2 + 3.0 * p.b3 * srs + 4.0 * p.b4 * rs);
  const double lg = std::log1p(1.0 / q1);
  g = q0 * lg;
  dg = -2.0 * p.A * p.a1 * lg - q0 * dq1 / (q1 * q1 + q1);
}

}  // namespace

void LSDA_PW92::evaluate(size_t n, const double* nup, const double* ndn, double* exc, double* vup,
                         double* vdn) const {
  const double cx = -0.75 * std::cbrt(3.0 / pi);          // e_x = cx * 2^{1/3} (nup^{4/3} + ndn^{4/3})
  const double two13 = std::cbrt(2.0);
  const double fden = 2.0 * two13 - 2.0;                  // 2^{4/3} - 2
  const double rs_fac = std::cbrt(3.0 / (4.0 * pi));
  const double tiny = 1e-20;
#pragma omp parallel for schedule(static) if (n > 20000)
  for (size_t i = 0; i < n; ++i) {
    const double a = nup[i] > 0.0 ? nup[i] : 0.0;
    const double b = ndn[i] > 0.0 ? ndn[i] : 0.0;
    const double rho = a + b;
    if (rho < tiny) {
      exc[i] = vup[i] = vdn[i] = 0.0;
      continue;
    }
    // exchange (spin scaling relation)
    const double a13 = std::cbrt(a), b13 = std::cbrt(b);
    double e = cx * two13 * (a * a13 + b * b13);
    double va = cx * two13 * (4.0 / 3.0) * a13;
    double vb = cx * two13 * (4.0 / 3.0) * b13;
    // correlation
    const double rs = rs_fac / std::cbrt(rho), srs = std::sqrt(rs);
    const double zeta = (a - b) / rho;
    double e0, de0, e1, de1, mac, dmac;
    pw_g(pw_para, rs, srs, e0, de0);
    pw_g(pw_ferro, rs, srs, e1, de1);
    pw_g(pw_alpha, rs, srs, mac, dmac);
    const double ac = -mac, dac = -dmac;
    const double p13 = std::cbrt(1.0 + zeta), m13 = std::cbrt(1.0 - zeta);
    const double f = ((1.0 + zeta) * p13 + (1.0 - zeta) * m13 - 2.0) / fden;
    const double df = (4.0 / 3.0) * (p13 - m13) / fden;
    const double z3 = zeta * zeta * zeta, z4 = z3 * zeta;
    const double ec = e0 + ac * f / fz20 * (1.0 - z4) + (e1 - e0) * f * z4;
    const double dec_rs = de0 + dac * f / fz20 * (1.0 - z4) + (de1 - de0) * f * z4;
    const double dec_z = 4.0 * z3 * f * (e1 - e0 - ac / fz20) + df * (z4 * (e1 - e0) + (1.0 - z4) * ac / fz20);
    const double common = ec - rs / 3.0 * dec_rs;
    e += rho * ec;
    va += common + (1.0 - zeta) * dec_z;
    vb += common - (1.0 + zeta) * dec_z;
    exc[i] = e;
    vup[i] = va;
    vdn[i] = vb;
  }
}

std::unique_ptr<XCFunctional> make_xc(const std::string& name) {
  if (name == "lda" || name == "lsda" || name == "pw92" || name == "lda_pw92" || name == "PW")
    return std::make_unique<LSDA_PW92>();
  fail("unknown xc functional '" + name + "' (available: pw92)");
}

}  // namespace lsda
