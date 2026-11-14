#include "dft/mixer.hpp"

#include <cmath>

#include "base/linalg.hpp"

namespace lsda {

double Mixer::dot(const std::vector<double>& a, const std::vector<double>& b) const {
  double s = 0.0;
  const size_t n = a.size();
#pragma omp parallel for schedule(static) reduction(+ : s)
  for (size_t i = 0; i < n; ++i) s += weight_[i] * a[i] * b[i];
  return s;
}

std::vector<double> Mixer::mix(const std::vector<double>& x_in, const std::vector<double>& x_out) {
  const size_t n = x_in.size();
  LSDA_CHECK(weight_.size() == n && precond_.size() == n, "mixer: inconsistent vector length");
  std::vector<double> r(n);
  for (size_t i = 0; i < n; ++i) r[i] = x_out[i] - x_in[i];

  if (have_prev_) {
    std::vector<double> dx(n), dr(n);
    for (size_t i = 0; i < n; ++i) {
      dx[i] = x_in[i] - x_prev_[i];
      dr[i] = r[i] - r_prev_[i];
    }
    dx_.push_back(std::move(dx));
    dr_.push_back(std::move(dr));
    if (static_cast<int>(dx_.size()) > history_) {
      dx_.erase(dx_.begin());
      dr_.erase(dr_.begin());
    }
  }
  x_prev_ = x_in;
  r_prev_ = r;
  have_prev_ = true;

  // least-squares coefficients: minimize | r - sum_j gamma_j dr_j |
  int m = static_cast<int>(dr_.size());
  std::vector<double> gamma;
  while (m > 0) {
    RMatrix A(m, m);
    std::vector<double> b(m);
    const int off = static_cast<int>(dr_.size()) - m;
    for (int i = 0; i < m; ++i) {
      b[i] = dot(dr_[off + i], r);
      for (int j = 0; j <= i; ++j) A(i, j) = A(j, i) = dot(dr_[off + i], dr_[off + j]);
    }
    // Solve the normal equations in scaled form (unit diagonal) with a tiny ridge term;
    // if the history is numerically singular, drop the oldest vectors.
    std::vector<double> scale(m);
    bool ok = true;
    for (int i = 0; i < m; ++i) {
      if (!(A(i, i) > 0.0)) ok = false;
      scale[i] = ok ? 1.0 / std::sqrt(A(i, i)) : 0.0;
    }
    for (int i = 0; i < m && ok; ++i) {
      b[i] *= scale[i];
      for (int j = 0; j < m; ++j) A(i, j) *= scale[i] * scale[j];
      A(i, i) += 1e-13;
    }
    ok = ok && solve_real(A, b);
    for (int i = 0; i < m; ++i) {
      b[i] *= scale[i];
      if (!std::isfinite(b[i])) ok = false;
    }
    if (ok) {
      gamma = b;
      break;
    }
    --m;
  }

  std::vector<double> x_new(n);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i) x_new[i] = x_in[i] + precond_[i] * r[i];
  const int off = static_cast<int>(dr_.size()) - m;
  for (int j = 0; j < m; ++j) {
    const std::vector<double>&dx = dx_[off + j], &dr = dr_[off + j];
    const double g = gamma[j];
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) x_new[i] -= g * (dx[i] + precond_[i] * dr[i]);
  }
  return x_new;
}

bool Mixer::save(std::FILE* f) const {
  const long n = static_cast<long>(weight_.size()), m = have_prev_ ? static_cast<long>(dx_.size()) : -1;
  bool ok = std::fwrite(&n, sizeof n, 1, f) == 1 && std::fwrite(&m, sizeof m, 1, f) == 1;
  if (m < 0) return ok;
  auto put = [&](const std::vector<double>& v) { ok = ok && std::fwrite(v.data(), sizeof(double), v.size(), f) == v.size(); };
  put(x_prev_);
  put(r_prev_);
  for (long i = 0; i < m; ++i) {
    put(dx_[i]);
    put(dr_[i]);
  }
  return ok;
}

bool Mixer::load(std::FILE* f) {
  reset();
  long n = 0, m = 0;
  if (std::fread(&n, sizeof n, 1, f) != 1 || std::fread(&m, sizeof m, 1, f) != 1) return false;
  if (n != static_cast<long>(weight_.size()) || m > 64) return false;
  if (m < 0) return true;
  bool ok = true;
  auto get = [&](std::vector<double>& v) {
    v.resize(n);
    ok = ok && std::fread(v.data(), sizeof(double), v.size(), f) == v.size();
  };
  std::vector<double> xp, rp;
  std::vector<std::vector<double>> dx(m), dr(m);
  get(xp);
  get(rp);
  for (long i = 0; i < m; ++i) {
    get(dx[i]);
    get(dr[i]);
  }
  if (!ok) return false;
  x_prev_ = std::move(xp);
  r_prev_ = std::move(rp);
  dx_ = std::move(dx);
  dr_ = std::move(dr);
  have_prev_ = true;
  return true;
}

}  // namespace lsda
