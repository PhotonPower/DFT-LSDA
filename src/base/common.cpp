#include "base/common.hpp"

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <map>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace lsda {

void fail(const std::string& msg) { throw std::runtime_error(msg); }

namespace {
bool g_logging = true;
}
void set_logging(bool enabled) { g_logging = enabled; }

void logf(const char* fmt, ...) {
  if (!g_logging) return;
  va_list ap;
  va_start(ap, fmt);
  std::vprintf(fmt, ap);
  va_end(ap);
  std::fflush(stdout);
}

std::string strf(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  return buf;
}

double det3(const Mat3& m) { return dot(m[0], cross(m[1], m[2])); }

Mat3 inverse3(const Mat3& m) {
  const double d = det3(m);
  LSDA_CHECK(std::abs(d) > 1e-14, "singular 3x3 matrix");
  const Vec3 c0 = cross(m[1], m[2]), c1 = cross(m[2], m[0]), c2 = cross(m[0], m[1]);
  Mat3 inv;
  for (int i = 0; i < 3; ++i) inv[i] = {c0[i] / d, c1[i] / d, c2[i] / d};
  return inv;
}

Mat3 transpose3(const Mat3& m) {
  Mat3 t;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) t[i][j] = m[j][i];
  return t;
}

Mat3 matmul3(const Mat3& a, const Mat3& b) {
  Mat3 c{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) c[i][j] += a[i][k] * b[k][j];
  return c;
}

double wall_time() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point t0 = clock::now();
  return std::chrono::duration<double>(clock::now() - t0).count();
}

namespace {
struct TimerEntry { double total = 0; long calls = 0; };
std::map<std::string, TimerEntry>& registry() {
  static std::map<std::string, TimerEntry> r;
  return r;
}
}  // namespace

ScopedTimer::ScopedTimer(const char* name) : name_(name), t0_(wall_time()) {}
ScopedTimer::~ScopedTimer() {
  auto& e = registry()[name_];
  e.total += wall_time() - t0_;
  e.calls += 1;
}

void add_time(const char* name, double seconds) {
  auto& e = registry()[name];
  e.total += seconds;
  e.calls += 1;
}

void print_timings() {
  logf("\n  Timings (wall clock)\n  %-28s %12s %10s\n", "section", "seconds", "calls");
  for (const auto& [name, e] : registry()) logf("  %-28s %12.2f %10ld\n", name.c_str(), e.total, e.calls);
}

int num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

}  // namespace lsda
