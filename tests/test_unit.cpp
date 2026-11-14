// Unit tests for the numerical building blocks. Usage: test_unit <pseudo-dir>
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
    test_paw(dir, false);
    test_paw(dir, true);
  } catch (const std::exception& e) {
    std::printf("exception: %s\n", e.what());
    return 2;
  }
  std::printf("%s (%d failures)\n", n_fail == 0 ? "ALL TESTS PASSED" : "TESTS FAILED", n_fail);
  return n_fail == 0 ? 0 : 1;
}
