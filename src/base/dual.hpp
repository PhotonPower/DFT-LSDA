// Forward-mode automatic differentiation: first-order dual numbers in N variables.
// Used where analytic derivatives are lengthy and error-prone (gradients of spherical harmonics,
// derivatives of gradient-corrected functionals).
#pragma once
#include <cmath>

namespace lsda {

template <int N>
struct Dual {
  double v = 0.0;
  double d[N] = {};
  Dual() = default;
  Dual(double x) : v(x) {}  // NOLINT: implicit conversion from constants is intended
  static Dual variable(double x, int index) {
    Dual r(x);
    r.d[index] = 1.0;
    return r;
  }
};

template <int N>
inline Dual<N> chain(const Dual<N>& a, double f, double df) {  // f(a) with derivative df = f'(a.v)
  Dual<N> r(f);
  for (int i = 0; i < N; ++i) r.d[i] = df * a.d[i];
  return r;
}
template <int N>
inline Dual<N> operator+(const Dual<N>& a, const Dual<N>& b) {
  Dual<N> r(a.v + b.v);
  for (int i = 0; i < N; ++i) r.d[i] = a.d[i] + b.d[i];
  return r;
}
template <int N>
inline Dual<N> operator-(const Dual<N>& a, const Dual<N>& b) {
  Dual<N> r(a.v - b.v);
  for (int i = 0; i < N; ++i) r.d[i] = a.d[i] - b.d[i];
  return r;
}
template <int N>
inline Dual<N> operator-(const Dual<N>& a) {
  Dual<N> r(-a.v);
  for (int i = 0; i < N; ++i) r.d[i] = -a.d[i];
  return r;
}
template <int N>
inline Dual<N> operator*(const Dual<N>& a, const Dual<N>& b) {
  Dual<N> r(a.v * b.v);
  for (int i = 0; i < N; ++i) r.d[i] = a.d[i] * b.v + a.v * b.d[i];
  return r;
}
template <int N>
inline Dual<N> operator/(const Dual<N>& a, const Dual<N>& b) {
  Dual<N> r(a.v / b.v);
  for (int i = 0; i < N; ++i) r.d[i] = (a.d[i] - r.v * b.d[i]) / b.v;
  return r;
}
// mixed operations with plain numbers
template <int N> inline Dual<N> operator+(const Dual<N>& a, double b) { Dual<N> r = a; r.v += b; return r; }
template <int N> inline Dual<N> operator+(double b, const Dual<N>& a) { return a + b; }
template <int N> inline Dual<N> operator-(const Dual<N>& a, double b) { Dual<N> r = a; r.v -= b; return r; }
template <int N> inline Dual<N> operator-(double b, const Dual<N>& a) { return -a + b; }
template <int N> inline Dual<N> operator*(const Dual<N>& a, double b) { return chain(a, a.v * b, b); }
template <int N> inline Dual<N> operator*(double b, const Dual<N>& a) { return chain(a, a.v * b, b); }
template <int N> inline Dual<N> operator/(const Dual<N>& a, double b) { return chain(a, a.v / b, 1.0 / b); }
template <int N> inline Dual<N> operator/(double b, const Dual<N>& a) { return chain(a, b / a.v, -b / (a.v * a.v)); }

template <int N> inline Dual<N> sqrt(const Dual<N>& a) { const double s = std::sqrt(a.v); return chain(a, s, 0.5 / s); }
template <int N> inline Dual<N> cbrt(const Dual<N>& a) { const double c = std::cbrt(a.v); return chain(a, c, c / (3.0 * a.v)); }
template <int N> inline Dual<N> exp(const Dual<N>& a) { const double e = std::exp(a.v); return chain(a, e, e); }
template <int N> inline Dual<N> expm1(const Dual<N>& a) { return chain(a, std::expm1(a.v), std::exp(a.v)); }
template <int N> inline Dual<N> log(const Dual<N>& a) { return chain(a, std::log(a.v), 1.0 / a.v); }
template <int N> inline Dual<N> log1p(const Dual<N>& a) { return chain(a, std::log1p(a.v), 1.0 / (1.0 + a.v)); }

}  // namespace lsda
