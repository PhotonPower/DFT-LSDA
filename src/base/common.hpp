// Basic types, physical constants, error handling and timers.
#pragma once
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace lsda {

using cplx = std::complex<double>;
using Vec3 = std::array<double, 3>;
using Mat3 = std::array<Vec3, 3>;

inline constexpr double pi = 3.14159265358979323846;
inline constexpr double bohr_in_angstrom = 0.529177210903;
inline constexpr double hartree_in_ev = 27.211386245988;
inline constexpr double hartree_in_kelvin = 315775.02480407;

[[noreturn]] void fail(const std::string& msg);
#define LSDA_CHECK(cond, msg) \
  do { if (!(cond)) ::lsda::fail(std::string(msg) + "  [" #cond "]"); } while (0)

// printf-style logging to stdout (flushed, so that long runs can be followed)
void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// ---- small 3x3 helpers ------------------------------------------------------
inline double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
double det3(const Mat3& m);
Mat3 inverse3(const Mat3& m);
Mat3 transpose3(const Mat3& m);
Mat3 matmul3(const Mat3& a, const Mat3& b);
inline Vec3 matvec3(const Mat3& m, const Vec3& v) { return {dot(m[0], v), dot(m[1], v), dot(m[2], v)}; }

// ---- wall-clock timers ------------------------------------------------------
// Use only outside of OpenMP parallel regions.
class ScopedTimer {
 public:
  explicit ScopedTimer(const char* name);
  ~ScopedTimer();
 private:
  const char* name_;
  double t0_;
};
double wall_time();
void add_time(const char* name, double seconds);  // accumulate into the timer table
void print_timings();

int num_threads();

}  // namespace lsda
