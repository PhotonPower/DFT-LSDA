#include "math/special.hpp"

#include <algorithm>
#include <cmath>

#include "base/dual.hpp"

namespace lsda {

void sph_bessel(int lmax, double x, double* jl) {
  x = std::abs(x);
  // Upward recurrence is stable for x > l; below that use the ascending series.
  const int nrec = std::min(lmax, static_cast<int>(std::floor(x)) - 1);  // highest l from recurrence
  if (x >= 1.0) {
    const double s = std::sin(x), c = std::cos(x);
    double jm = s / x, j = s / (x * x) - c / x;
    jl[0] = jm;
    if (lmax >= 1 && nrec >= 1) jl[1] = j;
    for (int l = 1; l < nrec; ++l) {
      const double jn = (2 * l + 1) / x * j - jm;
      jm = j;
      j = jn;
      jl[l + 1] = j;
    }
  }
  const int lstart = x >= 1.0 ? std::max(1, nrec + 1) : 0;
  const double x2 = 0.5 * x * x;
  for (int l = lstart; l <= lmax; ++l) {
    double pref = 1.0;  // x^l / (2l+1)!!
    for (int k = 1; k <= l; ++k) pref *= x / (2 * k + 1);
    double term = 1.0, sum = 1.0;
    for (int k = 1; k < 200; ++k) {
      term *= -x2 / (k * (2 * l + 2 * k + 1));
      sum += term;
      if (std::abs(term) < 1e-17 * std::abs(sum)) break;
    }
    jl[l] = pref * sum;
  }
}

double sph_bessel(int l, double x) {
  double jl[16];
  LSDA_CHECK(l < 16, "sph_bessel: l too large");
  sph_bessel(l, x, jl);
  return jl[l];
}

namespace {

using Dual3 = Dual<3>;

// Y_lm(v/|v|) for a scalar type T (double or Dual3); vx, vy, vz are the components of v != 0
template <class T>
void real_ylm_impl(int lmax, const T& vx, const T& vy, const T& vz, T* ylm) {
  using std::sqrt;
  const T r = sqrt(vx * vx + vy * vy + vz * vz);
  const T x = vx / r, y = vy / r, z = vz / r;
  // (x + i y)^m = cm + i sm  (= sin^m(theta) e^{i m phi})
  T cm(1.0), sm(0.0);
  double pmm = 1.0;  // P_m^m / sin^m(theta) = (2m-1)!!
  for (int m = 0; m <= lmax; ++m) {
    if (m > 0) {
      const T c2 = cm * x - sm * y;
      sm = cm * y + sm * x;
      cm = c2;
      pmm *= (2 * m - 1);
    }
    T p0(pmm), p1(0.0);  // P_l^m and P_{l-1}^m (divided by sin^m)
    for (int l = m; l <= lmax; ++l) {
      if (l == m + 1) {
        p1 = p0;
        p0 = z * T((2 * m + 1) * pmm);
      } else if (l > m + 1) {
        const T pn = (T(2.0 * l - 1.0) * z * p0 - T(l + m - 1.0) * p1) / T(static_cast<double>(l - m));
        p1 = p0;
        p0 = pn;
      }
      // normalization sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!)
      double ratio = 1.0;
      for (int k = l - m + 1; k <= l + m; ++k) ratio /= k;
      const double nrm = std::sqrt((2 * l + 1) / (4 * pi) * ratio);
      if (m == 0) {
        ylm[lm_index(l, 0)] = T(nrm) * p0;
      } else {
        ylm[lm_index(l, m)] = T(std::sqrt(2.0) * nrm) * p0 * cm;
        ylm[lm_index(l, -m)] = T(std::sqrt(2.0) * nrm) * p0 * sm;
      }
    }
  }
}

}  // namespace

void real_ylm(int lmax, const double* v, double* ylm) {
  const int n = (lmax + 1) * (lmax + 1);
  ylm[0] = 0.5 / std::sqrt(pi);
  if (lmax == 0) return;
  if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < 1e-28) {
    for (int i = 1; i < n; ++i) ylm[i] = 0.0;
    return;
  }
  real_ylm_impl<double>(lmax, v[0], v[1], v[2], ylm);
}

void real_ylm_grad(int lmax, const double* v, double* ylm, double* grad) {
  const int n = (lmax + 1) * (lmax + 1);
  if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < 1e-28) {
    real_ylm(lmax, v, ylm);
    std::fill(grad, grad + 3 * n, 0.0);
    return;
  }
  const Dual3 x = Dual3::variable(v[0], 0), y = Dual3::variable(v[1], 1), z = Dual3::variable(v[2], 2);
  Dual3 out[49];
  LSDA_CHECK(n <= 49, "real_ylm_grad: lmax too large");
  real_ylm_impl<Dual3>(lmax, x, y, z, out);
  for (int i = 0; i < n; ++i) {
    ylm[i] = out[i].v;
    for (int a = 0; a < 3; ++a) grad[3 * i + a] = out[i].d[a];
  }
}

void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
  x.assign(n, 0.0);
  w.assign(n, 0.0);
  for (int i = 0; i < (n + 1) / 2; ++i) {
    double z = std::cos(pi * (i + 0.75) / (n + 0.5)), pp = 0.0;
    for (int it = 0; it < 100; ++it) {
      double p1 = 1.0, p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      const double dz = p1 / pp;
      z -= dz;
      if (std::abs(dz) < 1e-15) break;
    }
    x[i] = -z;
    x[n - 1 - i] = z;
    w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
  }
}

SphereQuadrature make_sphere_quadrature(int ntheta) {
  std::vector<double> x, w;
  gauss_legendre(ntheta, x, w);
  const int nphi = 2 * ntheta;
  SphereQuadrature q;
  for (int i = 0; i < ntheta; ++i) {
    const double st = std::sqrt(std::max(0.0, 1.0 - x[i] * x[i]));
    for (int j = 0; j < nphi; ++j) {
      const double phi = 2 * pi * (j + 0.5) / nphi;
      q.points.push_back({st * std::cos(phi), st * std::sin(phi), x[i]});
      q.weights.push_back(w[i] * 2 * pi / nphi);
    }
  }
  return q;
}

GauntTable::GauntTable(int lmax) : lmax_(lmax), nL_((2 * lmax + 1) * (2 * lmax + 1)), n1_((lmax + 1) * (lmax + 1)) {
  g_.assign(static_cast<size_t>(nL_) * n1_ * n1_, 0.0);
  const SphereQuadrature q = make_sphere_quadrature(2 * lmax + 2);  // exact to degree 4*lmax+3
  std::vector<double> y(nL_);
  for (size_t k = 0; k < q.points.size(); ++k) {
    real_ylm(2 * lmax, q.points[k].data(), y.data());
    for (int L = 0; L < nL_; ++L)
      for (int L1 = 0; L1 < n1_; ++L1) {
        const double f = q.weights[k] * y[L] * y[L1];
        double* row = &g_[(static_cast<size_t>(L) * n1_ + L1) * n1_];
        for (int L2 = 0; L2 < n1_; ++L2) row[L2] += f * y[L2];
      }
  }
  for (double& v : g_)
    if (std::abs(v) < 1e-13) v = 0.0;
}

RMatrix ylm_rotation(int lmax, const Mat3& R) {
  const int n = (lmax + 1) * (lmax + 1);
  RMatrix d(n, n);
  const SphereQuadrature q = make_sphere_quadrature(lmax + 1);
  std::vector<double> y(n), yr(n);
  for (size_t k = 0; k < q.points.size(); ++k) {
    const Vec3 ru = matvec3(R, q.points[k]);
    real_ylm(lmax, q.points[k].data(), y.data());
    real_ylm(lmax, ru.data(), yr.data());
    for (int l = 0; l <= lmax; ++l)
      for (int m = -l; m <= l; ++m)
        for (int mp = -l; mp <= l; ++mp)
          d(lm_index(l, mp), lm_index(l, m)) += q.weights[k] * y[lm_index(l, mp)] * yr[lm_index(l, m)];
  }
  return d;
}

}  // namespace lsda
