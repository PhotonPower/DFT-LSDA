// Radial grids (arbitrary r(i) with dr/di), quadrature, multipole Hartree potentials,
// spherical Bessel transforms and uniform q-space tables.
#pragma once
#include "base/common.hpp"

namespace lsda {

struct RadialGrid {
  std::vector<double> r;   // r(i), i = 0..n-1 (r(0) may be 0)
  std::vector<double> dr;  // dr/di
  int size() const { return static_cast<int>(r.size()); }
  // first index with r(i) >= radius
  int index_of(double radius) const;
  // Quadrature weights w(i) (including dr/di) for int_{r(0)}^{r(npts-1)} f dr; Simpson rule in the
  // index variable (3/8 rule on the last three intervals if the number of intervals is odd).
  std::vector<double> weights(int npts) const;
  // F(i) = int_{r(0)}^{r(i)} f dr, i < npts (4-point formula, error O(h^5) per interval)
  void cumulative(const double* f, int npts, double* F) const;
  // Radial Hartree potential of the multipole density n(r) Y_L:
  //   v(r) = 4 pi/(2l+1) [ r^{-l-1} int_0^r n r'^{l+2} dr' + r^l int_r^R n r'^{1-l} dr' ],  R = r(npts-1)
  std::vector<double> hartree(int l, const std::vector<double>& n, int npts) const;
  // df/dr at the first npts grid points (5-point differences in the index variable; f must have npts + 2 values)
  std::vector<double> derivative(const std::vector<double>& f, int npts) const;
};

// 6-point Lagrange interpolation of samples f(0..n-1) at the fractional index x
double lagrange6(const double* f, int n, double x);

// Function tabulated on the uniform grid q = i*dq, interpolated with 6-point Lagrange polynomials.
struct RadialTable {
  double dq = 0.0;
  std::vector<double> f;
  double operator()(double q) const;
  double deriv(double q) const;  // derivative of the interpolating polynomial
  double qmax() const { return dq * (static_cast<int>(f.size()) - 4); }
};

// F(q) = int_0^{r(npts-1)} f(r) j_l(q r) r^2 dr on the uniform grid q = 0, dq, ..., >= qmax.
// The integrand is evaluated on a radial mesh refined by `refine` (Lagrange interpolation in the
// index variable) so that the oscillating Bessel factor is integrated accurately.
RadialTable bessel_transform(const RadialGrid& grid, const std::vector<double>& f, int npts, int l, double qmax,
                             double dq, int refine = 4);

}  // namespace lsda
