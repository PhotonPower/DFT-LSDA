#include "math/radial.hpp"

#include <cmath>

#include "math/special.hpp"

namespace lsda {

int RadialGrid::index_of(double radius) const {
  for (int i = 0; i < size(); ++i)
    if (r[i] >= radius) return i;
  fail("RadialGrid::index_of: radius outside of grid");
}

static std::vector<double> simpson_coefficients(int npts) {
  LSDA_CHECK(npts >= 5, "radial quadrature needs at least 5 points");
  std::vector<double> c(npts, 0.0);
  int nsimp = npts;  // number of points treated by Simpson's rule
  if ((npts - 1) % 2 == 1) nsimp = npts - 3;
  for (int i = 0; i + 2 < nsimp; i += 2) {
    c[i] += 1.0 / 3.0;
    c[i + 1] += 4.0 / 3.0;
    c[i + 2] += 1.0 / 3.0;
  }
  if (nsimp != npts) {  // Simpson 3/8 on the last three intervals
    const int i = npts - 4;
    c[i] += 3.0 / 8.0;
    c[i + 1] += 9.0 / 8.0;
    c[i + 2] += 9.0 / 8.0;
    c[i + 3] += 3.0 / 8.0;
  }
  return c;
}

std::vector<double> RadialGrid::weights(int npts) const {
  LSDA_CHECK(npts <= size(), "radial weights: too many points");
  std::vector<double> w = simpson_coefficients(npts);
  for (int i = 0; i < npts; ++i) w[i] *= dr[i];
  return w;
}

void RadialGrid::cumulative(const double* f, int npts, double* F) const {
  LSDA_CHECK(npts >= 4, "cumulative integral needs at least 4 points");
  auto g = [&](int i) { return f[i] * dr[i]; };
  F[0] = 0.0;
  F[1] = (9 * g(0) + 19 * g(1) - 5 * g(2) + g(3)) / 24.0;
  for (int i = 1; i + 2 < npts; ++i) F[i + 1] = F[i] + (-g(i - 1) + 13 * g(i) + 13 * g(i + 1) - g(i + 2)) / 24.0;
  const int m = npts - 1;
  F[m] = F[m - 1] + (g(m - 3) - 5 * g(m - 2) + 19 * g(m - 1) + 9 * g(m)) / 24.0;
}

std::vector<double> RadialGrid::hartree(int l, const std::vector<double>& n, int npts) const {
  std::vector<double> fa(npts), fb(npts), A(npts), B(npts), v(npts);
  for (int i = 0; i < npts; ++i) {
    fa[i] = n[i] * std::pow(r[i], l + 2);
    fb[i] = r[i] > 0.0 ? n[i] * std::pow(r[i], 1 - l) : 0.0;
  }
  cumulative(fa.data(), npts, A.data());
  cumulative(fb.data(), npts, B.data());
  const double btot = B[npts - 1];
  for (int i = 0; i < npts; ++i) {
    const double outer = btot - B[i];
    if (r[i] > 0.0)
      v[i] = 4 * pi / (2 * l + 1) * (A[i] * std::pow(r[i], -l - 1) + outer * std::pow(r[i], l));
    else
      v[i] = l == 0 ? 4 * pi * outer : 0.0;
  }
  return v;
}

double lagrange6(const double* f, int n, double x) {
  int i0 = static_cast<int>(std::floor(x)) - 2;
  if (i0 < 0) i0 = 0;
  if (i0 > n - 6) i0 = n - 6;
  const double t = x - i0;
  double sum = 0.0;
  for (int k = 0; k < 6; ++k) {
    double w = 1.0;
    for (int j = 0; j < 6; ++j)
      if (j != k) w *= (t - j) / (k - j);
    sum += w * f[i0 + k];
  }
  return sum;
}

double RadialTable::operator()(double q) const {
  const double x = q / dq;
  const int n = static_cast<int>(f.size());
  LSDA_CHECK(x <= n - 3, "RadialTable: argument beyond tabulated range");
  // inline 6-point Lagrange with precomputed denominators
  int i0 = static_cast<int>(x) - 2;
  if (i0 < 0) i0 = 0;
  if (i0 > n - 6) i0 = n - 6;
  const double t = x - i0;
  const double t0 = t, t1 = t - 1, t2 = t - 2, t3 = t - 3, t4 = t - 4, t5 = t - 5;
  const double* p = f.data() + i0;
  return -p[0] * t1 * t2 * t3 * t4 * t5 / 120.0 + p[1] * t0 * t2 * t3 * t4 * t5 / 24.0 -
         p[2] * t0 * t1 * t3 * t4 * t5 / 12.0 + p[3] * t0 * t1 * t2 * t4 * t5 / 12.0 -
         p[4] * t0 * t1 * t2 * t3 * t5 / 24.0 + p[5] * t0 * t1 * t2 * t3 * t4 / 120.0;
}

double RadialTable::deriv(double q) const {
  const double x = q / dq;
  const int n = static_cast<int>(f.size());
  LSDA_CHECK(x <= n - 3, "RadialTable: argument beyond tabulated range");
  int i0 = static_cast<int>(x) - 2;
  if (i0 < 0) i0 = 0;
  if (i0 > n - 6) i0 = n - 6;
  const double t = x - i0;
  double sum = 0.0;
  for (int k = 0; k < 6; ++k) {
    double den = 1.0, dl = 0.0;
    for (int j = 0; j < 6; ++j)
      if (j != k) den *= (k - j);
    for (int j = 0; j < 6; ++j) {  // derivative of prod_{i != k} (t - i)
      if (j == k) continue;
      double prod = 1.0;
      for (int i = 0; i < 6; ++i)
        if (i != k && i != j) prod *= (t - i);
      dl += prod;
    }
    sum += f[i0 + k] * dl / den;
  }
  return sum / dq;
}

RadialTable bessel_transform(const RadialGrid& grid, const std::vector<double>& f, int npts, int l, double qmax,
                             double dq, int refine) {
  LSDA_CHECK(npts >= 6 && npts <= grid.size(), "bessel_transform: bad number of points");
  // refined mesh in the index variable: x_k = k / refine
  const int nf = (npts - 1) * refine + 1;
  std::vector<double> rf(nf), wf(nf);
  const std::vector<double> c = simpson_coefficients(nf);
  for (int k = 0; k < nf; ++k) {
    const double x = static_cast<double>(k) / refine;
    const bool on_node = (k % refine == 0);
    const double rr = on_node ? grid.r[k / refine] : lagrange6(grid.r.data(), npts, x);
    const double dd = on_node ? grid.dr[k / refine] : lagrange6(grid.dr.data(), npts, x);
    const double ff = on_node ? f[k / refine] : lagrange6(f.data(), npts, x);
    rf[k] = rr;
    wf[k] = c[k] * dd / refine * rr * rr * ff;
  }
  RadialTable t;
  t.dq = dq;
  const int nq = static_cast<int>(std::ceil(qmax / dq)) + 6;
  t.f.assign(nq, 0.0);
#pragma omp parallel for schedule(static)
  for (int iq = 0; iq < nq; ++iq) {
    const double q = iq * dq;
    double jl[16], sum = 0.0;
    for (int k = 0; k < nf; ++k) {
      sph_bessel(l, q * rf[k], jl);
      sum += wf[k] * jl[l];
    }
    t.f[iq] = sum;
  }
  return t;
}

}  // namespace lsda
