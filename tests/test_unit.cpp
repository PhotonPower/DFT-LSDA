// Unit tests for the numerical building blocks. Usage: test_unit <pseudo-dir>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

#include "base/fft.hpp"
#include "base/linalg.hpp"
#include "crystal/crystal.hpp"
#include "dft/basis.hpp"
#include "dft/eigensolver.hpp"
#include "dft/hamiltonian.hpp"
#include "dft/mixer.hpp"
#include "dft/neb.hpp"
#include "dft/occupations.hpp"
#include "math/radial.hpp"
#include "math/special.hpp"
#include "paw/setup.hpp"
#include "xc/lda.hpp"

using namespace lsda;

static int n_fail = 0;
static void check(bool ok, const std::string& what, double value = 0.0) {
  std::printf("  [%s] %-58s %.3e\n", ok ? " ok " : "FAIL", what.c_str(), value);
  if (!ok) ++n_fail;
}

static void test_special() {
  std::printf("special functions\n");
  // spherical Bessel functions against closed forms
  double err = 0.0;
  for (double x : {1e-3, 0.3, 0.99, 1.0, 1.7, 2.5, 3.01, 5.5, 9.9, 23.0, 150.0}) {
    double j[8];
    sph_bessel(6, x, j);
    const double s = std::sin(x), c = std::cos(x);
    const double j0 = s / x, j1 = s / (x * x) - c / x, j2 = (3 / (x * x) - 1) * s / x - 3 * c / (x * x);
    const double j3 = (15 / (x * x * x) - 6 / x) * s / x - (15 / (x * x) - 1) * c / x;
    err = std::max({err, std::abs(j[0] - j0), std::abs(j[1] - j1) / (1 + 1 / x), std::abs(j[2] - j2) * std::min(1.0, x * x),
                    std::abs(j[3] - j3) * std::min(1.0, x * x * x)});
    // recurrence check for higher l: j_{l-1} + j_{l+1} = (2l+1)/x j_l
    for (int l = 1; l < 6; ++l) err = std::max(err, std::abs(j[l - 1] + j[l + 1] - (2 * l + 1) / x * j[l]) * std::min(1.0, x));
  }
  check(err < 1e-12, "spherical Bessel j_l, l <= 6", err);

  // orthonormality of real spherical harmonics, l <= 6
  const int lmax = 6, n = (lmax + 1) * (lmax + 1);
  const SphereQuadrature q = make_sphere_quadrature(lmax + 1);
  std::vector<double> y(n), ov(n * n, 0.0);
  double wsum = 0.0;
  for (size_t k = 0; k < q.points.size(); ++k) {
    real_ylm(lmax, q.points[k].data(), y.data());
    wsum += q.weights[k];
    for (int a = 0; a < n; ++a)
      for (int b = 0; b < n; ++b) ov[a * n + b] += q.weights[k] * y[a] * y[b];
  }
  err = std::abs(wsum - 4 * pi);
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) err = std::max(err, std::abs(ov[a * n + b] - (a == b)));
  check(err < 1e-12, "orthonormality of real Y_lm, l <= 6", err);

  // Gaunt coefficients: G(00, L1, L2) = delta/sqrt(4 pi); sum rule Y_L1 Y_L2 = sum_L G Y_L
  const GauntTable g(2);
  err = 0.0;
  for (int a = 0; a < 9; ++a)
    for (int b = 0; b < 9; ++b) err = std::max(err, std::abs(g(0, a, b) - (a == b) / std::sqrt(4 * pi)));
  const Vec3 u{0.3, -0.5, 0.81};
  double y4[25];
  real_ylm(4, u.data(), y4);
  for (int a = 0; a < 9; ++a)
    for (int b = 0; b < 9; ++b) {
      double s = 0.0;
      for (int L = 0; L < 25; ++L) s += g(L, a, b) * y4[L];
      err = std::max(err, std::abs(s - y4[a] * y4[b]));
    }
  check(err < 1e-12, "Gaunt coefficients (product expansion)", err);

  // rotation matrices: Y(Ru) = sum D Y(u), orthogonality
  const double ca = std::cos(0.7), sa = std::sin(0.7);
  const Mat3 R{{{ca, -sa, 0}, {sa, ca, 0}, {0, 0, -1}}};  // improper rotation
  const RMatrix D = ylm_rotation(2, R);
  const Vec3 ru = matvec3(R, u);
  double yu[9], yr[9];
  real_ylm(2, u.data(), yu);
  real_ylm(2, ru.data(), yr);
  err = 0.0;
  for (int m = 0; m < 9; ++m) {
    double s = 0.0;
    for (int mp = 0; mp < 9; ++mp) s += D(mp, m) * yu[mp];
    err = std::max(err, std::abs(s - yr[m]));
  }
  check(err < 1e-12, "rotation of real Y_lm", err);

  // gradient of Y_lm(v/|v|) with respect to v against central differences
  err = 0.0;
  const Vec3 w{0.7, -1.3, 0.45};
  double yv[49], gv[147], yp[49], ym[49];
  real_ylm_grad(6, w.data(), yv, gv);
  for (int al = 0; al < 3; ++al) {
    Vec3 wp = w, wm = w;
    wp[al] += 1e-5;
    wm[al] -= 1e-5;
    real_ylm(6, wp.data(), yp);
    real_ylm(6, wm.data(), ym);
    for (int i = 0; i < 49; ++i) err = std::max(err, std::abs((yp[i] - ym[i]) / 2e-5 - gv[3 * i + al]));
  }
  check(err < 1e-9, "gradient of real Y_lm, l <= 6", err);
}

static void test_radial() {
  std::printf("radial grids\n");
  // logarithmic grid r = a (exp(d i) - 1)
  RadialGrid g;
  const int n = 1801;
  const double a = 5e-4, d = 0.006;
  for (int i = 0; i < n; ++i) {
    g.r.push_back(a * (std::exp(d * i) - 1));
    g.dr.push_back(a * d * std::exp(d * i));
  }
  // Gaussian density n(r) = exp(-r^2) Y_00-type: int r^2 exp(-r^2) = sqrt(pi)/4
  std::vector<double> f(n), w = g.weights(n);
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    f[i] = std::exp(-g.r[i] * g.r[i]);
    s += w[i] * g.r[i] * g.r[i] * f[i];
  }
  check(std::abs(s - std::sqrt(pi) / 4) < 1e-10, "Simpson quadrature on log grid", std::abs(s - std::sqrt(pi) / 4));
  // Hartree potential (l = 0) of exp(-r^2): v(r) = pi^{3/2} erf(r)/r
  const std::vector<double> v = g.hartree(0, f, n);
  double err = 0.0;
  for (int i = 1; i < n; ++i)
    if (g.r[i] < 8.0) err = std::max(err, std::abs(v[i] - std::pow(pi, 1.5) * std::erf(g.r[i]) / g.r[i]));
  check(err < 1e-8, "radial Hartree potential, l = 0", err);
  // l = 2: n(r) = r^2 exp(-r^2); Hartree energy int r^2 n v dr = (4pi/5) * 2 * int int ... compare with
  // the analytic value (4 pi/5) * 15 pi/(64 sqrt(2)) * ... -> verified by self-consistency of the two forms:
  std::vector<double> f2(n);
  for (int i = 0; i < n; ++i) f2[i] = g.r[i] * g.r[i] * f[i];
  const std::vector<double> v2 = g.hartree(2, f2, n);
  // analytic: v2(r) = (4 pi/5) [ r^-3 A(r) + r^2 B(r) ], B(r) = int_r^inf r' exp(-r'^2) = exp(-r^2)/2
  err = 0.0;
  for (int i = 200; i < n; ++i) {
    const double r = g.r[i];
    if (r > 6.0) break;
    // A(r) = int_0^r r'^6 exp(-r'^2) dr' = (15 sqrt(pi)/16) erf(r) - exp(-r^2) (r^5/2 + 5 r^3/4 + 15 r/8)
    const double A = 15 * std::sqrt(pi) / 16 * std::erf(r) - std::exp(-r * r) * (std::pow(r, 5) / 2 + 5 * r * r * r / 4 + 15 * r / 8);
    err = std::max(err, std::abs(v2[i] - 4 * pi / 5 * (A / (r * r * r) + r * r * std::exp(-r * r) / 2)));
  }
  check(err < 1e-8, "radial Hartree potential, l = 2", err);
  // Bessel transform of exp(-r^2): l=0: sqrt(pi)/4 exp(-q^2/4); l=2 of r^2 exp(-r^2): sqrt(pi)/16 q^2 exp(-q^2/4)
  const RadialTable t0 = bessel_transform(g, f, n, 0, 12.0, 0.01), t2 = bessel_transform(g, f2, n, 2, 12.0, 0.01);
  err = 0.0;
  for (double q = 0.0; q < 12.0; q += 0.0937) {
    err = std::max(err, std::abs(t0(q) - std::sqrt(pi) / 4 * std::exp(-q * q / 4)));
    err = std::max(err, std::abs(t2(q) - std::sqrt(pi) / 16 * q * q * std::exp(-q * q / 4)));
  }
  check(err < 1e-10, "spherical Bessel transform + table interpolation", err);
  err = 0.0;
  for (double q = 0.05; q < 12.0; q += 0.0937)
    err = std::max(err, std::abs(t0.deriv(q) + std::sqrt(pi) / 8 * q * std::exp(-q * q / 4)));
  check(err < 1e-9, "derivative of the interpolated table", err);
}

static void test_xc() {
  std::printf("exchange-correlation\n");
  LSDA_PW92 xc;
  // reference: eps_c(rs, zeta) of PW92 (Perdew & Wang 1992, Table/known values)
  auto eps_c = [&](double rs, double zeta) {
    const double rho = 3.0 / (4 * pi * rs * rs * rs);
    const double a = 0.5 * rho * (1 + zeta), b = 0.5 * rho * (1 - zeta);
    double e, va, vb;
    xc.evaluate(1, &a, &b, &e, &va, &vb);
    const double ex = -0.75 * std::cbrt(3 / pi) * 0.5 * (std::pow(2 * a, 4.0 / 3) + std::pow(2 * b, 4.0 / 3));
    return (e - ex) / rho;
  };
  double err = std::max({std::abs(eps_c(1.0, 0) + 0.059774), std::abs(eps_c(2.0, 0) + 0.044760),
                         std::abs(eps_c(5.0, 0) + 0.028216)});
  check(err < 1e-6, "PW92 correlation energy, zeta = 0 (rs = 1, 2, 5)", err);
  // fully polarized gas: Ceperley-Alder data (the PW92 fit reproduces them within ~1e-4 Ha)
  err = std::max({std::abs(eps_c(2.0, 1) + 0.02405), std::abs(eps_c(5.0, 1) + 0.01555), std::abs(eps_c(10.0, 1) + 0.01045)});
  check(err < 2e-4, "PW92 correlation energy, zeta = 1 vs. Ceperley-Alder", err);
  // potentials against numerical derivatives
  err = 0.0;
  for (double a : {1e-4, 0.01, 0.3, 5.0})
    for (double b : {2e-4, 0.02, 0.1, 5.0}) {
      const double h = 1e-4 * std::min(a, b);
      double e, va, vb, ep, em, t1, t2;
      xc.evaluate(1, &a, &b, &e, &va, &vb);
      double ap = a + h, am = a - h, bp = b + h, bm = b - h;
      xc.evaluate(1, &ap, &b, &ep, &t1, &t2);
      xc.evaluate(1, &am, &b, &em, &t1, &t2);
      err = std::max(err, std::abs((ep - em) / (2 * h) - va) / std::abs(va));
      xc.evaluate(1, &a, &bp, &ep, &t1, &t2);
      xc.evaluate(1, &a, &bm, &em, &t1, &t2);
      err = std::max(err, std::abs((ep - em) / (2 * h) - vb) / std::abs(vb));
    }
  check(err < 1e-7, "v_xc = d e_xc / d n_sigma (finite differences)", err);
}

static void test_pbe() {
  std::printf("PBE functional\n");
  GGA_PBE pbe;
  LSDA_PW92 lda;
  // zero gradient: PBE reduces to LSDA
  double err = 0.0;
  for (double a : {1e-3, 0.05, 0.7, 4.0})
    for (double b : {2e-3, 0.03, 0.7, 2.0}) {
      const double z = 0.0;
      double e1, v1a, v1b, e2, v2a, v2b, s1, s2, s3;
      lda.evaluate(1, &a, &b, &e1, &v1a, &v1b);
      pbe.evaluate_gga(1, &a, &b, &z, &z, &z, &e2, &v2a, &v2b, &s1, &s2, &s3);
      err = std::max({err, std::abs(e1 - e2) / std::abs(e1), std::abs(v1a - v2a), std::abs(v1b - v2b)});
    }
  check(err < 1e-12, "PBE with zero gradient equals LSDA", err);
  // exchange enhancement factor: unpolarized density, F_x(s) = 1 + kappa - kappa/(1 + mu s^2/kappa)
  {
    const double n = 0.3, s = 1.3, kf = std::cbrt(3 * pi * pi * n), g2 = std::pow(2 * kf * n * s, 2);
    const double a = n / 2, q = g2 / 4, z = 0.0;
    double e, va, vb, s1, s2, s3, e0;
    pbe.evaluate_gga(1, &a, &a, &q, &q, &q, &e, &va, &vb, &s1, &s2, &s3);
    // subtract correlation by evaluating at the same point through the exchange-free combination:
    // exchange depends only on s_uu and s_dd, correlation only on the total gradient -> set s_ud = -q
    const double mq = -q;
    pbe.evaluate_gga(1, &a, &a, &q, &mq, &q, &e0, &va, &vb, &s1, &s2, &s3);  // total gradient zero: e0 = e_x F_x + e_c^LDA
    double el, t1, t2;
    lda.evaluate(1, &a, &a, &el, &t1, &t2);
    const double ex_lda = -0.75 * std::cbrt(3 / pi) * std::pow(n, 4.0 / 3), mu = 0.06672455060314922 * pi * pi / 3;
    const double fx = (e0 - (el - ex_lda)) / ex_lda;
    err = std::abs(fx - (1 + 0.804 - 0.804 / (1 + mu * s * s / 0.804)));
    check(err < 1e-12, "PBE exchange enhancement factor F_x(s)", err);
    // correlation limits: H -> beta phi^3 t^2 for t -> 0 and e_c -> 0 for t -> infinity
    const double small = 1e-10 * g2, big = 1e12 * g2, qs = small / 4, qb = big / 4;
    double es, eb;
    pbe.evaluate_gga(1, &a, &a, &z, &qs, &z, &es, &va, &vb, &s1, &s2, &s3);   // only s_ud: total gradient 2 qs
    const double ks2 = 4 * kf / pi, tt = 2 * qs / (4 * ks2 * n * n);
    err = std::abs((es - el) / (n * 0.06672455060314922 * tt) - 1.0);
    check(err < 1e-5, "PBE correlation: H -> beta t^2 for small gradients", err);
    pbe.evaluate_gga(1, &a, &a, &z, &qb, &z, &eb, &va, &vb, &s1, &s2, &s3);
    err = std::abs(eb - ex_lda) / std::abs(el - ex_lda);
    check(err < 1e-6, "PBE correlation vanishes for large gradients", err);
  }
  // all five partial derivatives against central differences
  err = 0.0;
  for (double a : {0.02, 0.4, 3.0})
    for (double b : {0.01, 0.4, 1.0})
      for (double g : {0.0, 0.3, 5.0}) {
        const double x0[5] = {a, b, g * a * a * 3, g * a * b * 2, g * b * b * 4};  // n_up, n_dn, s_uu, s_ud, s_dd
        double e, d[5];
        pbe.evaluate_gga(1, &x0[0], &x0[1], &x0[2], &x0[3], &x0[4], &e, &d[0], &d[1], &d[2], &d[3], &d[4]);
        for (int k = 0; k < 5; ++k) {
          if (k >= 2 && g == 0.0) continue;  // one-sided at zero gradient
          double xp[5], xm[5], ep, em, t[5];
          for (int j = 0; j < 5; ++j) xp[j] = xm[j] = x0[j];
          const double h = 1e-5 * std::max(std::abs(x0[k]), 1e-3);
          xp[k] += h;
          xm[k] -= h;
          pbe.evaluate_gga(1, &xp[0], &xp[1], &xp[2], &xp[3], &xp[4], &ep, &t[0], &t[1], &t[2], &t[3], &t[4]);
          pbe.evaluate_gga(1, &xm[0], &xm[1], &xm[2], &xm[3], &xm[4], &em, &t[0], &t[1], &t[2], &t[3], &t[4]);
          err = std::max(err, std::abs((ep - em) / (2 * h) - d[k]) / std::max(std::abs(d[k]), 1e-3));
        }
      }
  check(err < 1e-5, "PBE derivatives (finite differences)", err);
}

static void test_fft_linalg() {
  std::printf("FFT and linear algebra\n");
  FFT3D fft(12, 10, 9);
  CVec f(fft.size());
  // plane wave with m = (2,-3,1): forward transform must be N at that index only
  for (int i = 0; i < 12; ++i)
    for (int j = 0; j < 10; ++j)
      for (int k = 0; k < 9; ++k)
        f[(i * 10 + j) * 9 + k] = std::exp(cplx(0, 2 * pi * (2.0 * i / 12 - 3.0 * j / 10 + 1.0 * k / 9)));
  fft.forward(f.data());
  double err = 0.0;
  for (size_t i = 0; i < f.size(); ++i)
    err = std::max(err, std::abs(f[i] - (i == fft.index(2, -3, 1) ? cplx(static_cast<double>(f.size())) : cplx(0))));
  check(err < 1e-10, "FFT convention and index mapping", err);

  // pruned transforms agree with the full ones for data supported on a small sphere
  {
    FFT3D g(15, 18, 16);
    const int M = 3;
    std::vector<int> cols, planes;
    std::vector<size_t> idx;
    for (int m1 = -M; m1 <= M; ++m1) {
      planes.push_back(m1 < 0 ? m1 + 15 : m1);
      for (int m2 = -M; m2 <= M; ++m2) {
        if (m1 * m1 + m2 * m2 > M * M) continue;
        cols.push_back((m1 < 0 ? m1 + 15 : m1) * 18 + (m2 < 0 ? m2 + 18 : m2));
        for (int m3 = -M; m3 <= M; ++m3)
          if (m1 * m1 + m2 * m2 + m3 * m3 <= M * M) idx.push_back(g.index(m1, m2, m3));
      }
    }
    CVec a(g.size(), cplx(0.0)), b(g.size(), cplx(0.0));
    for (size_t i = 0; i < idx.size(); ++i) a[idx[i]] = b[idx[i]] = cplx(std::sin(0.37 * i), std::cos(1.1 * i));
    g.backward(a.data());
    g.backward_pruned(b.data(), cols, planes);
    err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) err = std::max(err, std::abs(a[i] - b[i]));
    for (size_t i = 0; i < a.size(); ++i) a[i] = b[i] = a[i] * std::cos(0.01 * i);  // "potential times psi"
    g.forward(a.data());
    g.forward_pruned(b.data(), cols, planes);
    for (size_t i : idx) err = std::max(err, std::abs(a[i] - b[i]));
    check(err < 1e-10, "pruned FFT equals full FFT on the plane-wave sphere", err);
  }

  // generalized Hermitian eigenproblem
  const int n = 30;
  CMatrix A(n, n), B(n, n);
  unsigned seed = 12345;
  auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return ((seed >> 8) & 0xffff) / 65536.0 - 0.5; };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j) {
      A(i, j) = cplx(rnd(), i == j ? 0.0 : rnd());
      A(j, i) = std::conj(A(i, j));
      B(i, j) = 0.1 * cplx(rnd(), i == j ? 0.0 : rnd());
      B(j, i) = std::conj(B(i, j));
    }
  for (int i = 0; i < n; ++i) B(i, i) += 2.0;
  CMatrix A0 = A, B0 = B;
  std::vector<double> w;
  check(eigh_gen(A, B, w), "generalized eigensolver returns");
  err = 0.0;
  for (int k = 0; k < n; ++k)
    for (int i = 0; i < n; ++i) {
      cplx r = 0.0;
      for (int j = 0; j < n; ++j) r += (A0(i, j) - w[k] * B0(i, j)) * A(j, k);
      err = std::max(err, std::abs(r));
    }
  check(err < 1e-11, "residual of A c = w B c", err);
}

static void test_symmetry() {
  std::printf("symmetry and k-points\n");
  Crystal c;
  const double a = 5.4;
  c.set_lattice({{{a, 0, 0}, {0, a, 0}, {0, 0, a}}});
  c.species.resize(2);
  c.atoms = {{0, {0, 0, 0}, 0}, {1, {0.5, 0.5, 0.5}, 0}};
  std::vector<SymOp> ops = find_symmetry(c, true);
  check(ops.size() == 48, "B2 structure has 48 symmetry operations", static_cast<double>(ops.size()));
  const int n[3] = {4, 4, 4};
  auto k0 = make_kpoints(n, {0, 0, 0}, ops);
  check(k0.size() == 10, "4x4x4 Gamma-centred grid -> 10 irreducible k-points", static_cast<double>(k0.size()));
  auto k1 = make_kpoints(n, {0.5, 0.5, 0.5}, ops);
  check(k1.size() == 4, "4x4x4 shifted grid -> 4 irreducible k-points", static_cast<double>(k1.size()));
  double wsum = 0.0;
  for (const auto& k : k1) wsum += k.weight;
  check(std::abs(wsum - 1) < 1e-12, "k-point weights sum to one", std::abs(wsum - 1));
  // fcc primitive cell
  Crystal f;
  f.set_lattice({{{0, a / 2, a / 2}, {a / 2, 0, a / 2}, {a / 2, a / 2, 0}}});
  f.species.resize(1);
  f.atoms = {{0, {0, 0, 0}, 0}};
  ops = find_symmetry(f, true);
  check(ops.size() == 48, "fcc primitive cell has 48 symmetry operations", static_cast<double>(ops.size()));
  const int n8[3] = {8, 8, 8};
  auto kf = make_kpoints(n8, {0, 0, 0}, ops);
  check(kf.size() == 29, "fcc 8x8x8 Gamma-centred grid -> 29 irreducible k-points", static_cast<double>(kf.size()));
}

static void test_paw(const std::string& dir, bool gga) {
  std::printf(gga ? "PAW datasets (PBE)\n" : "PAW datasets (LDA)\n");
  const std::unique_ptr<XCFunctional> xcp = make_xc(gga ? "pbe" : "pw92");
  const XCFunctional& xc = *xcp;
  for (const char* el : {"Al", "Fe", "Ni"}) {
    PawSetup s = PawSetup::load(dir + "/" + el + (gga ? ".GGA_PBE-JTH.xml" : ".LDA_PW-JTH.xml"));
    if (!gga && std::string(el) != "Al") {  // the LDA runs of Fe and Ni also cover the Hubbard term (d shell)
      s.hubbard_l = 2;
      s.hubbard_u = 0.15;
    }
    check(std::abs(s.check_core_charge - s.n_core) < 1e-6, std::string(el) + ": core charge", std::abs(s.check_core_charge - s.n_core));
    check(s.check_duality < 5e-5, std::string(el) + ": <p_i|phi~_j> = delta_ij", s.check_duality);
    // neutrality of the reference atom: sqrt(4 pi) Q_00 + pseudo valence + pseudo core charge = 0
    const RMatrix D = s.atomic_density_matrix();
    std::vector<double> Q(s.nL);
    s.multipoles(D, Q.data());
    double nval = 0.0;
    for (int i = 0; i < s.ni; ++i) nval += D(i, i);
    check(std::abs(nval - s.n_valence) < 1e-10, std::string(el) + ": reference occupations", std::abs(nval - s.n_valence));
    // on-site Hamiltonian = derivative of the on-site energy (random symmetric D, both spins)
    RMatrix Ds[2] = {D, D}, dH[2];
    unsigned seed = 4711;
    auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return ((seed >> 8) & 0xffff) / 65536.0 - 0.5; };
    for (int sp = 0; sp < 2; ++sp)
      for (int i = 0; i < s.ni; ++i)
        for (int j = 0; j <= i; ++j) {
          const double v = (sp == 0 ? 0.6 : 0.4) * D(i, j) + 1e-5 * rnd();
          Ds[sp](i, j) = Ds[sp](j, i) = v;
        }
    s.onsite(2, Ds, dH, xc);
    double err = 0.0, asym = 0.0;
    for (int sp = 0; sp < 2; ++sp)
      for (auto [i, j] : {std::pair{0, 0}, std::pair{1, 0}, std::pair{s.ni - 1, s.ni - 1}, std::pair{s.ni - 1, s.ni - 3}, std::pair{2, 5}}) {
        const double h = 1e-5;
        RMatrix Dp[2] = {Ds[0], Ds[1]}, Dm[2] = {Ds[0], Ds[1]};
        Dp[sp](i, j) += h;
        Dm[sp](i, j) -= h;
        const double fd = (s.onsite(2, Dp, nullptr, xc) - s.onsite(2, Dm, nullptr, xc)) / (2 * h);
        err = std::max(err, std::abs(fd - dH[sp](i, j)) / std::max(1.0, std::abs(fd)));
        asym = std::max(asym, std::abs(dH[sp](i, j) - dH[sp](j, i)));
      }
    check(err < (gga ? 2e-5 : 2e-6), std::string(el) + ": dH_ij = dE/dD_ij (finite differences)", err);
    check(asym < 1e-10, std::string(el) + ": dH symmetric", asym);
    // spin-unpolarized path consistent with the polarized one
    RMatrix Dt = D, dH1[1], Dh[2] = {D, D}, dH2[2];
    for (int ij = 0; ij < s.ni * s.ni; ++ij) Dh[0].data()[ij] = Dh[1].data()[ij] = 0.5 * D.data()[ij];
    const double e1 = s.onsite(1, &Dt, dH1, xc), e2 = s.onsite(2, Dh, dH2, xc);
    check(std::abs(e1 - e2) < 1e-9, std::string(el) + ": nspin = 1 and 2 agree", std::abs(e1 - e2));
    if (!gga) {  // non-collinear on-site term: collinear limit, rotation invariance, derivative
      RMatrix D4[4] = {D, D, D, D}, dH4[4], Drot[4] = {D, D, D, D};
      for (int ij = 0; ij < s.ni * s.ni; ++ij) {
        const double t = Ds[0].data()[ij] + Ds[1].data()[ij], m = Ds[0].data()[ij] - Ds[1].data()[ij];
        D4[0].data()[ij] = Drot[0].data()[ij] = t;
        D4[1].data()[ij] = D4[2].data()[ij] = 0.0;
        D4[3].data()[ij] = m;                        // magnetization along z
        Drot[1].data()[ij] = 0.6 * m;                // the same, rotated to (0.6, 0, 0.8)
        Drot[2].data()[ij] = 0.0;
        Drot[3].data()[ij] = 0.8 * m;
      }
      const double ecol = s.onsite(2, Ds, dH, xc), e4 = s.onsite(4, D4, dH4, xc), erot = s.onsite(4, Drot, nullptr, xc);
      double dev = 0.0;
      for (int ij = 0; ij < s.ni * s.ni; ++ij)
        dev = std::max({dev, std::abs(dH4[0].data()[ij] - 0.5 * (dH[0].data()[ij] + dH[1].data()[ij])),
                        std::abs(dH4[3].data()[ij] - 0.5 * (dH[0].data()[ij] - dH[1].data()[ij]))});
      check(std::abs(e4 - ecol) < 1e-9 && dev < 1e-9, std::string(el) + ": non-collinear on-site term, collinear limit", std::max(std::abs(e4 - ecol), dev));
      check(std::abs(erot - ecol) < 1e-9, std::string(el) + ": non-collinear on-site energy is rotation invariant", std::abs(erot - ecol));
      s.onsite(4, Drot, dH4, xc);
      double derr = 0.0;
      for (int cmp = 0; cmp < 4; ++cmp) {
        const double h = 1e-5;
        RMatrix Dp[4] = {Drot[0], Drot[1], Drot[2], Drot[3]}, Dm[4] = {Drot[0], Drot[1], Drot[2], Drot[3]};
        Dp[cmp](s.ni - 1, s.ni - 1) += h;
        Dm[cmp](s.ni - 1, s.ni - 1) -= h;
        const double fd = (s.onsite(4, Dp, nullptr, xc) - s.onsite(4, Dm, nullptr, xc)) / (2 * h);
        derr = std::max(derr, std::abs(fd - dH4[cmp](s.ni - 1, s.ni - 1)) / std::max(1.0, std::abs(fd)));
      }
      check(derr < 2e-6, std::string(el) + ": non-collinear dH_c = dE/dD_c (finite differences)", derr);
    }
    std::printf("         %s: Z=%g core=%g valence=%g ni=%d r_paw=%.3f  on-site energy(ref) = %.6f Ha\n", el, s.Z, s.n_core,
                s.n_valence, s.ni, s.r_paw, e1);
  }
}

static void test_occupations_mixer() {
  std::printf("occupations and mixing\n");
  // variational smearing: entropy function s and occupation f satisfy s'(x) = x f'(x)
  double err = 0.0;
  for (Smearing t : {Smearing::FermiDirac, Smearing::Gaussian, Smearing::MethfesselPaxton1, Smearing::Cold})
    for (double x = -6.0; x <= 6.0; x += 0.37) {
      const double h = 1e-5;
      const double df = (smearing_occupation(t, x + h) - smearing_occupation(t, x - h)) / (2 * h);
      const double ds = (smearing_entropy(t, x + h) - smearing_entropy(t, x - h)) / (2 * h);
      err = std::max(err, std::abs(ds - x * df));
    }
  check(err < 1e-9, "smearing functions: s'(x) = x f'(x)", err);
  // Fermi level search: electron count and fixed-moment mode
  std::vector<std::vector<double>> eig(4);
  for (int ks = 0; ks < 4; ++ks)
    for (int n = 0; n < 12; ++n) eig[ks].push_back(0.05 * n + 0.013 * ks - (ks % 2) * 0.02);
  const std::vector<double> kw = {0.25, 0.75};
  Occupations o = compute_occupations(eig, kw, 2, 9.3, Smearing::Cold, 0.01, std::nullopt);
  check(std::abs(o.n_spin[0] + o.n_spin[1] - 9.3) < 1e-10, "Fermi level: electron count", std::abs(o.n_spin[0] + o.n_spin[1] - 9.3));
  o = compute_occupations(eig, kw, 2, 9.3, Smearing::FermiDirac, 0.01, 1.7);
  check(std::abs(o.n_spin[0] - o.n_spin[1] - 1.7) < 1e-10, "Fermi level: fixed spin moment", std::abs(o.n_spin[0] - o.n_spin[1] - 1.7));
  // tetrahedron method for free electrons eps = k^2/2 on a cubic grid: electron number and band energy
  // inside the Fermi sphere, n = k_F^3/(6 pi^2) Omega_cell and E = (3/5) eps_F n (per spin)
  for (int ngrid : {8, 16}) {
    TetraGrid g;
    const double a = 5.0, kf = 0.9 * pi / a;  // sphere well inside the first Brillouin zone
    g.n[0] = g.n[1] = g.n[2] = ngrid;
    g.B = Mat3{{{2 * pi / a, 0, 0}, {0, 2 * pi / a, 0}, {0, 0, 2 * pi / a}}};
    const int nt = ngrid * ngrid * ngrid;
    g.map.resize(nt);
    std::vector<std::vector<double>> e(nt, std::vector<double>(1));
    for (int i = 0; i < nt; ++i) {
      g.map[i] = i;
      const int m[3] = {i / (ngrid * ngrid), (i / ngrid) % ngrid, i % ngrid};
      double k2 = 0.0;
      for (int d = 0; d < 3; ++d) {
        double x = static_cast<double>(m[d]) / ngrid;
        if (x > 0.5) x -= 1.0;
        k2 += std::pow(2 * pi / a * x, 2);
      }
      e[i][0] = 0.5 * k2;
    }
    const double nexact = 2.0 * kf * kf * kf / (6 * pi * pi) * a * a * a;  // both spins
    const Occupations ot = compute_occupations(e, std::vector<double>(nt, 1.0 / nt), 1, nexact, Smearing::Tetrahedron, 0.0, std::nullopt, &g);
    const double eband = 0.6 * (0.5 * kf * kf) * nexact;
    check(std::abs(ot.efermi[0] - 0.5 * kf * kf) < (ngrid == 8 ? 5e-2 : 1.2e-2) * 0.5 * kf * kf, strf("tetrahedron: Fermi level of free electrons, %d^3 grid (relative)", ngrid),
          std::abs(ot.efermi[0] / (0.5 * kf * kf) - 1.0));
    check(std::abs(ot.band_energy - eband) < (ngrid == 8 ? 3e-3 : 1e-4) * eband, strf("tetrahedron: band energy with Bloechl correction, %d^3 grid (relative)", ngrid),
          std::abs(ot.band_energy / eband - 1.0));
  }
  // Pulay mixing solves a linear fixed-point problem x = A x + b in (dimension + 1) steps
  const int n = 6;
  std::vector<double> A(n * n), b(n), x(n, 0.0);
  unsigned seed = 99;
  auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return ((seed >> 8) & 0xffff) / 65536.0 - 0.5; };
  for (double& v : A) v = 0.9 * rnd();
  for (double& v : b) v = rnd();
  Mixer mixer(std::vector<double>(n, 1.0), std::vector<double>(n, 0.3), 10);
  double res = 1.0;
  for (int it = 0; it < n + 2; ++it) {
    std::vector<double> y(b);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) y[i] += A[i * n + j] * x[j];
    res = 0.0;
    for (int i = 0; i < n; ++i) res = std::max(res, std::abs(y[i] - x[i]));
    if (std::getenv("LSDA_TEST_VERBOSE")) std::printf("         mixer iteration %d residual %.3e\n", it, res);
    x = mixer.mix(x, y);
  }
  check(res < 1e-5, "Pulay mixer: linear problem solved after n+1 steps", res);
}

// Nudged elastic band with climbing image on the Mueller-Brown surface (scaled by 1/100):
// minima at (-0.558, 1.442) and (-0.050, 0.467), saddle point between them at (-0.822, 0.624), V = -40.665
static void test_neb() {
  std::printf("nudged elastic band\n");
  const double A[4] = {-200, -100, -170, 15}, a[4] = {-1, -1, -6.5, 0.7}, b[4] = {0, 0, 11, 0.6}, c[4] = {-10, -10, -6.5, 0.7};
  const double x0[4] = {1, 0, -0.5, -1}, y0[4] = {0, 0.5, 1.5, 1};
  auto potential = [&](const BandCoords& r, BandCoords& f) {
    double v = 0.0;
    f.assign(1, Vec3{0, 0, 0});
    for (int k = 0; k < 4; ++k) {
      const double dx = r[0][0] - x0[k], dy = r[0][1] - y0[k];
      const double e = 0.01 * A[k] * std::exp(a[k] * dx * dx + b[k] * dx * dy + c[k] * dy * dy);
      v += e;
      f[0][0] -= e * (2 * a[k] * dx + b[k] * dy);
      f[0][1] -= e * (b[k] * dx + 2 * c[k] * dy);
    }
    return v;
  };
  const int ntot = 9;
  std::vector<BandCoords> R(ntot, BandCoords(1));
  const Vec3 ra{-0.5582236346, 1.4417258418, 0}, rb{-0.0500108229, 0.4666941049, 0};
  for (int i = 0; i < ntot; ++i)
    for (int k = 0; k < 3; ++k) R[i][0][k] = ra[k] + (rb[k] - ra[k]) * i / (ntot - 1.0);
  std::vector<double> E(ntot);
  BandCoords ftmp;
  E[0] = potential(R[0], ftmp);
  E[ntot - 1] = potential(R[ntot - 1], ftmp);
  NebParams np;
  np.spring = 1.0;
  np.force_tol = 1e-5;
  np.max_step = 0.05;
  np.max_steps = 3000;
  int evaluations = 0;
  const NebResult r = optimize_band(R, E, [&](int, const BandCoords& x, double& e, BandCoords& f) {
    e = potential(x, f);
    ++evaluations;
    return true;
  }, np, nullptr, false);
  check(r.converged, strf("NEB converged (%d steps)", r.steps), r.max_force);
  check(std::abs(r.energy[r.saddle] + 0.40665) < 1e-5, "NEB: saddle-point energy of the Mueller-Brown surface", std::abs(r.energy[r.saddle] + 0.40665));
  const double dpos = std::hypot(R[r.saddle][0][0] + 0.8220, R[r.saddle][0][1] - 0.6243);
  check(dpos < 1e-3, "NEB: saddle-point position", dpos);
  check(std::abs(E[0] + 1.46700) < 1e-5 && std::abs(E[ntot - 1] + 0.80768) < 1e-5, "NEB: end-point energies", std::abs(E[0] + 1.46700));
}

// Spinor Hamiltonian: Hermiticity, iterative solver against dense diagonalization, and the collinear limit
// (magnetic field along z or x: spectrum = union of the spectra of the two collinear spin channels)
static void test_spinor_hamiltonian(const std::string& dir) {
  std::printf("spinor Hamiltonian\n");
  Crystal c;
  const double a = 6.2;
  c.set_lattice({{{a, 0, 0}, {0.2, a, 0}, {0, -0.1, a}}});
  c.species.resize(1);
  c.species[0].setup = PawSetup::load(dir + "/Fe.LDA_PW-JTH.xml");
  c.atoms = {{0, {0.03, -0.02, 0.01}, 0}};
  const double ecut = 3.0;
  c.species[0].setup.build_tables(std::sqrt(2 * ecut) + 1.0, 2 * std::sqrt(2 * ecut) + 1.0);
  int nc[3];
  fft_dimensions(c, 2 * std::sqrt(2 * ecut), nc);
  FFT3D fft(nc[0], nc[1], nc[2]);
  const PWBasis pw = make_pw_basis(c, KPoint{{0.2, -0.1, 0.3}, 1.0}, ecut, fft);
  const int n = pw.npw, ni = c.setup(0).ni;
  const size_t ng = fft.size();
  std::vector<double> v0(ng), b(ng), zero(ng, 0.0), bx(ng), by(ng), bz(ng), vup(ng), vdn(ng);
  for (int i = 0; i < nc[0]; ++i)
    for (int j = 0; j < nc[1]; ++j)
      for (int k = 0; k < nc[2]; ++k) {
        const size_t g = (static_cast<size_t>(i) * nc[1] + j) * nc[2] + k;
        v0[g] = -0.4 * std::cos(2 * pi * i / nc[0]) + 0.2 * std::sin(2 * pi * (j + k) / nc[1]);
        b[g] = 0.15 + 0.1 * std::cos(2 * pi * k / nc[2]);
        bx[g] = 0.6 * b[g] * std::cos(2 * pi * j / nc[1]);   // a genuinely non-collinear field
        by[g] = 0.5 * b[g] * std::sin(2 * pi * i / nc[0]);
        bz[g] = b[g];
        vup[g] = v0[g] + b[g];
        vdn[g] = v0[g] - b[g];
      }
  unsigned seed = 11;
  auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return ((seed >> 8) & 0xffff) / 65536.0 - 0.5; };
  RMatrix h0(ni, ni), hm(ni, ni), hx(ni, ni), hy(ni, ni), hup(ni, ni), hdn(ni, ni), nul(ni, ni);
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j <= i; ++j) {
      h0(i, j) = h0(j, i) = 0.3 * rnd() + (i == j ? 0.2 : 0.0);
      hm(i, j) = hm(j, i) = 0.1 * rnd();
      hx(i, j) = hx(j, i) = 0.05 * rnd();
      hy(i, j) = hy(j, i) = 0.05 * rnd();
      hup(i, j) = hup(j, i) = h0(i, j) + hm(i, j);
      hdn(i, j) = hdn(j, i) = h0(i, j) - hm(i, j);
    }
  const RMatrix& dS = c.setup(0).dS;
  const Projectors proj = make_projectors(c, pw);
  auto dense = [&](const Hamiltonian& h, std::vector<double>& w) {
    const int d = h.npw();
    CMatrix I(d, d), Hd(d, d), Sd(d, d);
    for (int i = 0; i < d; ++i) I(i, i) = 1.0;
    h.apply(I.data(), d, Hd.data(), Sd.data());
    double herm = 0.0;
    for (int i = 0; i < d; ++i)
      for (int j = 0; j < d; ++j) herm = std::max({herm, std::abs(Hd(i, j) - std::conj(Hd(j, i))), std::abs(Sd(i, j) - std::conj(Sd(j, i)))});
    eigh_gen(Hd, Sd, w);
    return herm;
  };
  std::vector<double> wup, wdn, wz, wx, wnc;
  dense(Hamiltonian(pw, fft, vup.data(), proj, {&hup}, {&dS}), wup);
  dense(Hamiltonian(pw, fft, vdn.data(), proj, {&hdn}, {&dS}), wdn);
  std::vector<double> both = wup;
  both.insert(both.end(), wdn.begin(), wdn.end());
  std::sort(both.begin(), both.end());
  using A4 = std::array<const double*, 4>;
  using H4 = std::array<const RMatrix*, 4>;
  double herm = dense(Hamiltonian(pw, fft, A4{v0.data(), zero.data(), zero.data(), b.data()}, proj, {H4{&h0, &nul, &nul, &hm}}, {&dS}), wz);
  dense(Hamiltonian(pw, fft, A4{v0.data(), b.data(), zero.data(), zero.data()}, proj, {H4{&h0, &hm, &nul, &nul}}, {&dS}), wx);
  double ez = 0.0, ex = 0.0;
  for (int i = 0; i < 2 * n; ++i) {
    ez = std::max(ez, std::abs(wz[i] - both[i]));
    ex = std::max(ex, std::abs(wx[i] - both[i]));
  }
  check(herm < 1e-11, "spinor H and S are Hermitian", herm);
  check(ez < 1e-10, "field along z: spectrum = collinear up and down spectra", ez);
  check(ex < 1e-10, "field along x: same spectrum (spin-rotation invariance)", ex);
  const Hamiltonian hnc(pw, fft, A4{v0.data(), bx.data(), by.data(), bz.data()}, proj, {H4{&h0, &hx, &hy, &hm}}, {&dS});
  herm = dense(hnc, wnc);
  check(herm < 1e-11, "non-collinear H is Hermitian", herm);
  const int nb = 12;
  CMatrix X(2 * n, nb);
  for (int j = 0; j < nb; ++j)
    for (int i = 0; i < 2 * n; ++i) X(i, j) = cplx(rnd(), rnd()) / (1.0 + hnc.kinetic(i));
  std::vector<double> eps;
  const EigensolverResult r = ppcg(hnc, X, eps, std::vector<double>(nb, 1e-9), nb - 3, 400);
  double err = 0.0;
  for (int j = 0; j < nb - 3; ++j) err = std::max(err, std::abs(eps[j] - wnc[j]));
  check(err < 1e-9, strf("PPCG on spinors equals dense diagonalization (%d iterations)", r.iterations), err);
}

// Iterative eigensolvers against dense diagonalization of the full PAW Hamiltonian matrix
static void test_eigensolvers(const std::string& dir) {
  std::printf("eigensolvers\n");
  Crystal c;
  const double a = 7.6;
  c.set_lattice({{{a, 0, 0}, {0.3, a, 0}, {0, -0.2, a}}});
  c.species.resize(2);
  c.species[0].setup = PawSetup::load(dir + "/Al.LDA_PW-JTH.xml");
  c.species[1].setup = PawSetup::load(dir + "/Ni.LDA_PW-JTH.xml");
  c.atoms = {{0, {0.02, 0.01, -0.03}, 0}, {1, {0.52, 0.47, 0.55}, 0}};
  const double ecut = 4.0;
  for (auto& sp : c.species) sp.setup.build_tables(std::sqrt(2 * ecut) + 1.0, 2 * std::sqrt(2 * ecut) + 1.0);
  int nc[3];
  fft_dimensions(c, 2 * std::sqrt(2 * ecut), nc);
  FFT3D fft(nc[0], nc[1], nc[2]);
  const PWBasis pw = make_pw_basis(c, KPoint{{0.25, -0.125, 0.375}, 1.0}, ecut, fft);
  const int n = pw.npw, nb = 14;
  // smooth local potential and on-site matrices (atomic dS; dH = symmetric test matrix)
  std::vector<double> v(fft.size());
  for (int i = 0; i < nc[0]; ++i)
    for (int j = 0; j < nc[1]; ++j)
      for (int k = 0; k < nc[2]; ++k)
        v[(i * nc[1] + j) * nc[2] + k] = -0.4 * std::cos(2 * pi * i / nc[0]) - 0.3 * std::sin(2 * pi * (j + 2.0 * k) / nc[1]) +
                                         0.2 * std::cos(2 * pi * (i - j + k) / nc[2]);
  unsigned seed = 7;
  auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return ((seed >> 8) & 0xffff) / 65536.0 - 0.5; };
  std::vector<RMatrix> dH(2), dS(2);
  for (int at = 0; at < 2; ++at) {
    const PawSetup& s = c.setup(at);
    dS[at] = s.dS;
    dH[at].resize(s.ni, s.ni);
    for (int i = 0; i < s.ni; ++i)
      for (int j = 0; j <= i; ++j) dH[at](i, j) = dH[at](j, i) = 0.3 * rnd() + (i == j ? 0.2 : 0.0);
  }
  const Projectors proj = make_projectors(c, pw);
  const Hamiltonian h(pw, fft, v.data(), proj, {&dH[0], &dH[1]}, {&dS[0], &dS[1]});
  // dense reference
  CMatrix I(n, n), Hd(n, n), Sd(n, n);
  for (int i = 0; i < n; ++i) I(i, i) = 1.0;
  h.apply(I.data(), n, Hd.data(), Sd.data());
  double herm = 0.0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) herm = std::max({herm, std::abs(Hd(i, j) - std::conj(Hd(j, i))), std::abs(Sd(i, j) - std::conj(Sd(j, i)))});
  check(herm < 1e-11, "H and S matrices are Hermitian", herm);
  std::vector<double> wref;
  check(eigh_gen(Hd, Sd, wref), "overlap matrix S is positive definite");
  // iterative solvers from random start vectors
  for (int solver = 0; solver < 3; ++solver) {
    CMatrix X(n, nb);
    for (int j = 0; j < nb; ++j)
      for (int i = 0; i < n; ++i) X(i, j) = cplx(rnd(), rnd()) / (1.0 + pw.kin[i]);
    std::vector<double> eps;
    const std::vector<double> tol(nb, 1e-9);
    CMatrix PX;
    const EigensolverResult r = solver == 0   ? ppcg(h, X, eps, tol, nb - 3, 300, &PX, 1)
                                : solver == 1 ? ppcg(h, X, eps, tol, nb - 3, 300, &PX, 4)
                                              : lobpcg(h, X, eps, tol, nb - 3, 300, &PX);
    double err = 0.0;
    for (int j = 0; j < nb - 3; ++j) err = std::max(err, std::abs(eps[j] - wref[j]));
    const std::string name = solver == 0 ? "PPCG(block 1)" : solver == 1 ? "PPCG(block 4)" : "LOBPCG";
    check(err < 1e-10, name + ": eigenvalues equal dense diagonalization", err);
    check(r.max_residual < 1e-8, name + strf(": residual norm (%d iterations)", r.iterations), r.max_residual);
    const CMatrix Pref = h.project(X.data(), nb);
    double perr = 0.0;
    for (size_t i = 0; i < Pref.size(); ++i) perr = std::max(perr, std::abs(Pref.data()[i] - PX.data()[i]));
    check(perr < 1e-10, name + ": returned projections <p|psi> are consistent", perr);
  }
}


// ---- meta-GGA r2SCAN: limits and derivatives ---------------------------------------------------------------
static void test_r2scan(const std::string& dir) {
  std::printf("meta-GGA r2SCAN\n");
  const MGGA_R2SCAN mgga;
  const LSDA_PW92 lda;
  auto eval = [&](double a, double b, double suu, double sud, double sdd, double ta, double tb, double* out) {
    double e, v[7];
    mgga.evaluate_mgga(1, &a, &b, &suu, &sud, &sdd, &ta, &tb, &e, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]);
    if (out) std::copy(v, v + 7, out);
    return e;
  };
  // uniform electron gas (no gradient, tau = tau_unif of each spin channel): LSDA
  double err = 0.0;
  for (const auto& d : {std::array<double, 2>{0.02, 0.02}, std::array<double, 2>{0.3, 0.1}, std::array<double, 2>{1.5, 0.2}}) {
    const double cf = 0.3 * std::pow(6.0 * pi * pi, 2.0 / 3.0);  // tau_s = cf n_s^(5/3)
    const double e = eval(d[0], d[1], 0, 0, 0, cf * std::pow(d[0], 5.0 / 3.0), cf * std::pow(d[1], 5.0 / 3.0), nullptr);
    double el, vu, vd;
    lda.evaluate(1, &d[0], &d[1], &el, &vu, &vd);
    err = std::max(err, std::abs(e - el) / std::abs(el));
  }
  check(err < 1e-9, "r2SCAN equals LSDA for the uniform gas", err);
  // hydrogen atom (exact density, fully polarized, tau = tau_W): E_x = -5/16 Ha, E_c = 0
  {
    double ex = 0.0, ec_part = 0.0;
    const int nr = 4000;
    const double rmax = 30.0, h = rmax / nr;
    for (int i = 1; i <= nr; ++i) {
      const double r = (i - 0.5) * h, n = std::exp(-2 * r) / pi, g2 = 4 * n * n, tw = g2 / (8 * n);
      const double e = eval(n, 0.0, g2, 0.0, 0.0, tw, 0.0, nullptr);
      ex += 4 * pi * r * r * h * e;
      // correlation alone: difference to the same point with the exchange of a hypothetical unpolarized pair removed
      (void)ec_part;
    }
    check(std::abs(ex + 0.3125) < 2e-4, "r2SCAN hydrogen atom: E_xc = -5/16 Ha (E_c = 0)", std::abs(ex + 0.3125));
  }
  // recomputed core states (needed for the core kinetic energy density) against the datasets
  for (const char* name : {"Al.LDA_PW-JTH.xml", "Al.GGA_PBE-JTH.xml", "Fe.GGA_PBE-JTH.xml", "Ni.GGA_PBE-JTH.xml"}) {
    PawSetup s = PawSetup::load(dir + "/" + name);
    s.prepare_mgga();
    std::printf("    %-20s core density deviation %.2e, core kinetic energy deviation %.2e\n", name, s.check_core_density, s.check_core_kinetic);
    check(s.check_core_density < 1e-4, std::string(name) + ": recomputed core density", s.check_core_density);
    check(s.check_core_kinetic < 1e-4, std::string(name) + ": recomputed core kinetic energy", s.check_core_kinetic);
    double tneg = 0.0, jump = 0.0;
    for (size_t i = 0; i < s.tauct.size(); ++i) tneg = std::min(tneg, s.tauct[i]);
    const int ip = s.grid.index_of(s.r_paw);
    for (int i = ip; i < s.grid.size(); ++i) jump = std::max(jump, std::abs(s.tauct[i] - s.tauc[i]));
    check(tneg >= 0.0 && jump == 0.0, std::string(name) + ": pseudo core tau >= 0, equal to tau_c outside", jump - tneg);
  }
  // on-site meta-GGA term: dH = dE/dD (Fe, spin-polarized) and spin-paired limit
  {
    PawSetup s = PawSetup::load(dir + "/Fe.GGA_PBE-JTH.xml");
    s.prepare_mgga();
    const int ni = s.ni;
    RMatrix D[2], dD[2], dH[2];
    const RMatrix d0 = s.atomic_density_matrix();
    unsigned seed = 4711;
    auto rnd = [&]() {
      seed = seed * 1664525u + 1013904223u;
      return static_cast<double>(seed >> 8) / 16777216.0 - 0.5;
    };
    for (int sp = 0; sp < 2; ++sp) {
      D[sp] = d0;
      dD[sp].resize(ni, ni);
      for (int i = 0; i < ni; ++i)
        for (int j = 0; j <= i; ++j) {
          D[sp](i, j) = D[sp](j, i) = (sp == 0 ? 0.6 : 0.4) * d0(i, j) + 0.002 * rnd();
          dD[sp](i, j) = dD[sp](j, i) = 0.02 * rnd();
        }
    }
    s.onsite(2, D, dH, mgga);
    double lin = 0.0;
    for (int sp = 0; sp < 2; ++sp)
      for (int i = 0; i < ni; ++i)
        for (int j = 0; j < ni; ++j) lin += dH[sp](i, j) * dD[sp](i, j);
    const double h = 1e-4;
    RMatrix Dp[2] = {D[0], D[1]}, Dm[2] = {D[0], D[1]};
    for (int sp = 0; sp < 2; ++sp)
      for (size_t i = 0; i < D[sp].size(); ++i) {
        Dp[sp].data()[i] += h * dD[sp].data()[i];
        Dm[sp].data()[i] -= h * dD[sp].data()[i];
      }
    const double fd = (s.onsite(2, Dp, nullptr, mgga) - s.onsite(2, Dm, nullptr, mgga)) / (2 * h);
    check(std::abs(fd - lin) < 1e-5 * std::max(1.0, std::abs(lin)), "Fe on-site r2SCAN: dH = dE/dD", std::abs(fd - lin));
    RMatrix half[2] = {D[0], D[0]};
    for (int sp = 0; sp < 2; ++sp)
      for (size_t i = 0; i < half[sp].size(); ++i) half[sp].data()[i] *= 0.5;
    const double e1 = s.onsite(1, D, nullptr, mgga), e2 = s.onsite(2, half, nullptr, mgga);
    check(std::abs(e1 - e2) < 1e-9 * std::abs(e1), "Fe on-site r2SCAN: spin-paired limit", std::abs(e1 - e2));
  }
  // derivatives against finite differences
  {
    const double x0[7] = {0.21, 0.13, 0.05, 0.02, 0.03, 0.19, 0.11};
    double v[7];
    eval(x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], v);
    const int map[7] = {0, 1, 2, 3, 4, 5, 6};  // output order: vup, vdn, vsuu, vsud, vsdd, vtup, vtdn
    err = 0.0;
    for (int k = 0; k < 7; ++k) {
      double xp[7], xm[7];
      std::copy(x0, x0 + 7, xp);
      std::copy(x0, x0 + 7, xm);
      const double h = 1e-5 * x0[k];
      xp[k] += h;
      xm[k] -= h;
      const double fd = (eval(xp[0], xp[1], xp[2], xp[3], xp[4], xp[5], xp[6], nullptr) - eval(xm[0], xm[1], xm[2], xm[3], xm[4], xm[5], xm[6], nullptr)) / (2 * h);
      err = std::max(err, std::abs(fd - v[map[k]]));
    }
    check(err < 1e-7, "r2SCAN derivatives (finite differences)", err);
  }
}

// ---- hybrid functionals: short-range PBE exchange, on-site exact exchange, dataset conventions -------------
static void test_exchange(const std::string& dir) {
  std::printf("Hybrid functionals\n");
  // HJS enhancement factor: omega = 0 reproduces PBE, s = 0 the exact short-range LDA exchange
  double dev_pbe = 0.0, dev_lda = 0.0;
  for (double sv = 0.0; sv <= 3.0001; sv += 0.25) {
    const double f_pbe = 1.0 + 0.804 - 0.804 / (1.0 + 0.2195149727645171 * sv * sv / 0.804);
    dev_pbe = std::max(dev_pbe, std::abs(sr_pbe_enhancement(sv, 0.0) - f_pbe));
  }
  for (double nu : {0.05, 0.2, 0.5, 1.0, 2.0, 5.0}) {
    // F(a) = 1 - 8/3 a [sqrt(pi) erf(1/(2a)) + (2a - 4a^3) exp(-1/(4a^2)) - 3a + 4a^3],  a = omega/(2 k_F)
    const double a = 0.5 * nu, ex = std::exp(-1.0 / (4 * a * a));
    const double exact = 1.0 - 8.0 / 3.0 * a * (std::sqrt(pi) * std::erf(1.0 / (2 * a)) + (2 * a - 4 * a * a * a) * ex - 3 * a + 4 * a * a * a);
    dev_lda = std::max(dev_lda, std::abs(sr_pbe_enhancement(0.0, nu) - exact));
  }
  check(dev_pbe < 3e-3, "HJS factor at omega = 0 equals the PBE factor (s <= 3)", dev_pbe);
  // the model exchange hole reproduces the exact short-range LDA exchange to a few 1e-3 only (by construction)
  check(dev_lda < 5e-3, "HJS factor at s = 0 close to short-range LDA exchange", dev_lda);
  {
    double fs, fn;
    const double s0 = 1.3, n0 = 0.4, h = 1e-5;
    sr_pbe_enhancement(s0, n0, &fs, &fn);
    const double ds = (sr_pbe_enhancement(s0 + h, n0) - sr_pbe_enhancement(s0 - h, n0)) / (2 * h);
    const double dn = (sr_pbe_enhancement(s0, n0 + h) - sr_pbe_enhancement(s0, n0 - h)) / (2 * h);
    check(std::abs(fs - ds) + std::abs(fn - dn) < 1e-8, "HJS factor: derivatives", std::abs(fs - ds) + std::abs(fn - dn));
  }

  // on-site exact exchange: dH = dE/dD for Hartree-Fock and for the screened hybrid
  for (int kind = 0; kind < 2; ++kind) {
    const std::unique_ptr<XCFunctional> xc = make_xc(kind == 0 ? "hf" : "hse06");
    PawSetup s = PawSetup::load(dir + "/Fe.GGA_PBE-JTH.xml");
    s.prepare_exchange(xc->exx_screening());
    const int ni = s.ni;
    RMatrix D[2], dD[2], dH[2];
    const RMatrix d0 = s.atomic_density_matrix();
    unsigned seed = 12345;
    auto rnd = [&]() {
      seed = seed * 1664525u + 1013904223u;
      return static_cast<double>(seed >> 8) / 16777216.0 - 0.5;
    };
    for (int sp = 0; sp < 2; ++sp) {
      D[sp] = d0;
      dD[sp].resize(ni, ni);
      for (int i = 0; i < ni; ++i)
        for (int j = 0; j <= i; ++j) {
          D[sp](i, j) = D[sp](j, i) = (sp == 0 ? 0.6 : 0.4) * d0(i, j) + 0.01 * rnd();
          dD[sp](i, j) = dD[sp](j, i) = 0.02 * rnd();
        }
    }
    s.onsite(2, D, dH, *xc);
    double lin = 0.0;
    for (int sp = 0; sp < 2; ++sp)
      for (int i = 0; i < ni; ++i)
        for (int j = 0; j < ni; ++j) lin += dH[sp](i, j) * dD[sp](i, j);
    const double h = 1e-3;
    RMatrix Dp[2] = {D[0], D[1]}, Dm[2] = {D[0], D[1]};
    for (int sp = 0; sp < 2; ++sp)
      for (size_t i = 0; i < D[sp].size(); ++i) {
        Dp[sp].data()[i] += h * dD[sp].data()[i];
        Dm[sp].data()[i] -= h * dD[sp].data()[i];
      }
    const double fd = (s.onsite(2, Dp, nullptr, *xc) - s.onsite(2, Dm, nullptr, *xc)) / (2 * h);
    check(std::abs(fd - lin) < 2e-6 * std::max(1.0, std::abs(lin)), kind == 0 ? "Fe on-site exchange (HF): dH = dE/dD" : "Fe on-site (HSE06): dH = dE/dD",
          std::abs(fd - lin));
    // spin-paired limit: nspin = 1 with D equals nspin = 2 with D/2 in both channels
    RMatrix half[2] = {D[0], D[0]};
    for (int sp = 0; sp < 2; ++sp)
      for (size_t i = 0; i < half[sp].size(); ++i) half[sp].data()[i] *= 0.5;
    const double e1 = s.onsite(1, D, nullptr, *xc), e2 = s.onsite(2, half, nullptr, *xc);
    check(std::abs(e1 - e2) < 1e-9 * std::abs(e1), kind == 0 ? "Fe on-site exchange (HF): spin-paired limit" : "Fe on-site (HSE06): spin-paired limit", std::abs(e1 - e2));
    if (kind == 1) {
      // screening removes long-range exchange only: the tensor changes little, and not at all for omega -> 0
      double diff = 0.0, scale = 0.0;
      for (size_t i = 0; i < s.C.size(); ++i) {
        diff = std::max(diff, std::abs(s.C.data()[i] - s.Cx.data()[i]));
        scale = std::max(scale, std::abs(s.C.data()[i]));
      }
      check(diff > 0.0 && diff < 0.02 * scale, "Fe: screened on-site Coulomb tensor close to the bare one", diff / scale);
      s.prepare_exchange(1e-3);
      diff = 0.0;
      for (size_t i = 0; i < s.C.size(); ++i) diff = std::max(diff, std::abs(s.C.data()[i] - s.Cx.data()[i]));
      check(diff < 1e-8, "Fe: screened tensor -> bare tensor for omega -> 0", diff);
    }
  }

  // Convention of the valence-core exchange matrix of the datasets, X_ij = sum_c (i c|c j) over the core orbitals
  // of one spin direction: the core orbitals are recomputed here (non-relativistic radial Schroedinger equation in
  // the reconstructed LDA potential of the reference atom) and the matrix elements of the bound valence states
  // are compared with the dataset.
  for (const char* el : {"Al", "Fe"}) {
    const PawSetup s = PawSetup::load(dir + "/" + el + ".LDA_PW-JTH.xml");
    const RadialGrid& g = s.grid;
    const int nfull = g.size();
    std::vector<double> dens(nfull);  // spherical density times sqrt(4 pi) (convention of the stored densities)
    for (int i = 0; i < nfull; ++i) {
      dens[i] = s.nc[i];
      for (int j = 0; j < s.nj; ++j) dens[i] += s.f_j[j] * s.phi[j][i] * s.phi[j][i] / std::sqrt(4 * pi);
    }
    const std::vector<double> vh = g.hartree(0, dens, nfull);
    std::vector<double> half(nfull), exc(nfull), vxc(nfull), vdn(nfull);
    for (int i = 0; i < nfull; ++i) half[i] = 0.5 * dens[i] / std::sqrt(4 * pi);
    LSDA_PW92().evaluate(nfull, half.data(), half.data(), exc.data(), vxc.data(), vdn.data());
    // every second grid point with 0 < r < 25
    std::vector<int> idx;
    for (int i = 1; i < nfull && g.r[i] < 25.0; i += 2) idx.push_back(i);
    const int m = static_cast<int>(idx.size());
    std::vector<double> x(m), v(m), hm(m), hp(m);
    for (int i = 0; i < m; ++i) {
      x[i] = g.r[idx[i]];
      v[i] = -s.Z / x[i] + vh[idx[i]] / std::sqrt(4 * pi) + vxc[idx[i]];
    }
    for (int i = 0; i < m; ++i) {
      hm[i] = i == 0 ? x[0] : x[i] - x[i - 1];
      hp[i] = i == m - 1 ? hm[i] : x[i + 1] - x[i];
    }
    const int ncore = static_cast<int>(std::lround(s.n_core));
    std::vector<std::pair<int, int>> shells;  // (l, number of core states)
    if (ncore == 2) shells = {{0, 1}};
    if (ncore == 10) shells = {{0, 2}, {1, 1}};
    if (ncore == 18) shells = {{0, 3}, {1, 2}};
    if (shells.empty()) continue;
    std::vector<std::pair<int, std::vector<double>>> core;  // (l, u(r) = r R(r))
    for (const auto& [l, count] : shells) {
      RMatrix H(m, m);
      for (int i = 0; i < m; ++i) {
        H(i, i) = 1.0 / (hm[i] * hp[i]) + 0.5 * l * (l + 1) / (x[i] * x[i]) + v[i];
        if (i + 1 < m) H(i, i + 1) = H(i + 1, i) = -1.0 / (hp[i] * std::sqrt((hm[i] + hp[i]) * (hm[i + 1] + hp[i + 1])));
      }
      std::vector<double> e;
      eigh_real(H, e);
      for (int k = 0; k < count; ++k) {
        std::vector<double> u(m);
        double nrm = 0.0;
        for (int i = 0; i < m; ++i) u[i] = H(i, k) / std::sqrt(hm[i] + hp[i]);
        for (int i = 0; i < m; ++i) nrm += 0.5 * (hm[i] + hp[i]) * u[i] * u[i];
        for (double& ui : u) ui /= std::sqrt(nrm);
        core.push_back({l, u});
      }
    }
    auto cumulative = [&](const std::vector<double>& y) {
      std::vector<double> out(m, 0.0);
      out[0] = 0.5 * x[0] * y[0];
      for (int i = 1; i < m; ++i) out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
      return out;
    };
    auto slater = [&](int k, const std::vector<double>& f1, const std::vector<double>& f2) {  // int int f1(r) r<^k/r>^(k+1) f2(r')
      std::vector<double> a(m), b(m);
      for (int i = 0; i < m; ++i) {
        a[i] = f2[i] * std::pow(x[i], k);
        b[i] = f2[i] / std::pow(x[i], k + 1);
      }
      const std::vector<double> ca = cumulative(a), cb = cumulative(b);
      std::vector<double> y(m);
      for (int i = 0; i < m; ++i) y[i] = f1[i] * (ca[i] / std::pow(x[i], k + 1) + std::pow(x[i], k) * (cb[m - 1] - cb[i]));
      return cumulative(y)[m - 1];
    };
    auto w3j0sq = [](int a, int b, int c) {  // squared Wigner 3j symbol (a b c; 0 0 0)
      const int J = a + b + c;
      if (J % 2 != 0 || c < std::abs(a - b) || c > a + b) return 0.0;
      auto f = [](int n) { return std::tgamma(n + 1.0); };
      const int h = J / 2;
      const double t = f(h) / (f(h - a) * f(h - b) * f(h - c));
      return f(J - 2 * a) * f(J - 2 * b) * f(J - 2 * c) / f(J + 1) * t * t;
    };
    double worst = 0.0;
    for (int j = 0; j < s.nj; ++j) {
      if (!s.is_bound(j) || s.f_j[j] <= 0.0) continue;
      const int l = s.l_j[j];
      double own = 0.0;
      for (const auto& [lc, uc] : core) {
        std::vector<double> prod(m);
        for (int i = 0; i < m; ++i) prod[i] = x[i] * s.phi[j][idx[i]] * uc[i];
        for (int k = std::abs(l - lc); k <= l + lc; ++k) own += (2 * lc + 1) * w3j0sq(l, k, lc) * slater(k, prod, prod);
      }
      int i_first = 0;
      while (s.j_i[i_first] != j) ++i_first;
      worst = std::max(worst, std::abs(s.X(i_first, i_first) / own - 1.0));
    }
    check(worst < 0.03, std::string(el) + ": valence-core exchange matrix = sum over core orbitals", worst);
  }
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "pseudos";
  try {
    test_special();
    test_radial();
    test_xc();
    test_pbe();
    test_fft_linalg();
    test_symmetry();
    test_occupations_mixer();
    test_neb();
    test_eigensolvers(dir);
    test_spinor_hamiltonian(dir);
    test_paw(dir, false);
    test_paw(dir, true);
    test_exchange(dir);
    test_r2scan(dir);
  } catch (const std::exception& e) {
    std::printf("exception: %s\n", e.what());
    return 2;
  }
  std::printf("%s (%d failures)\n", n_fail == 0 ? "ALL TESTS PASSED" : "TESTS FAILED", n_fail);
  return n_fail == 0 ? 0 : 1;
}
