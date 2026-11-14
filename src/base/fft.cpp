#include "base/fft.hpp"

#include <fftw3.h>
#include <utility>

namespace lsda {

void* aligned_malloc(size_t bytes) {
  void* p = fftw_malloc(bytes ? bytes : 16);
  LSDA_CHECK(p != nullptr, "out of memory");
  return p;
}
void aligned_free(void* p) { fftw_free(p); }

FFT3D::FFT3D(int n1, int n2, int n3) : n_{n1, n2, n3}, size_(static_cast<size_t>(n1) * n2 * n3) {
  CVec tmp(size_);
  auto* p = reinterpret_cast<fftw_complex*>(tmp.data());
  fwd_ = fftw_plan_dft_3d(n1, n2, n3, p, p, FFTW_FORWARD, FFTW_MEASURE);
  bwd_ = fftw_plan_dft_3d(n1, n2, n3, p, p, FFTW_BACKWARD, FFTW_MEASURE);
  LSDA_CHECK(fwd_ && bwd_, "FFTW planning failed");
  // batched 1D plans for the pruned transforms: z lines (contiguous), all y lines of one i1 plane,
  // all x lines of the box
  for (int dir = 0; dir < 2; ++dir) {
    const int sign = dir == 0 ? FFTW_FORWARD : FFTW_BACKWARD;
    const unsigned flags = FFTW_MEASURE | FFTW_UNALIGNED;
    line_[dir][0] = fftw_plan_many_dft(1, &n3, 1, p, nullptr, 1, n3, p, nullptr, 1, n3, sign, flags);
    line_[dir][1] = fftw_plan_many_dft(1, &n2, n3, p, nullptr, n3, 1, p, nullptr, n3, 1, sign, flags);
    line_[dir][2] = fftw_plan_many_dft(1, &n1, n2 * n3, p, nullptr, n2 * n3, 1, p, nullptr, n2 * n3, 1, sign, flags);
    LSDA_CHECK(line_[dir][0] && line_[dir][1] && line_[dir][2], "FFTW planning failed");
  }
}

FFT3D::~FFT3D() {
  if (fwd_) fftw_destroy_plan(static_cast<fftw_plan>(fwd_));
  if (bwd_) fftw_destroy_plan(static_cast<fftw_plan>(bwd_));
  for (auto& dir : line_)
    for (void*& pl : dir)
      if (pl) fftw_destroy_plan(static_cast<fftw_plan>(pl));
}

FFT3D::FFT3D(FFT3D&& o) noexcept { *this = std::move(o); }
FFT3D& FFT3D::operator=(FFT3D&& o) noexcept {
  std::swap(n_, o.n_);
  std::swap(size_, o.size_);
  std::swap(fwd_, o.fwd_);
  std::swap(bwd_, o.bwd_);
  std::swap(line_, o.line_);
  return *this;
}

void FFT3D::forward(cplx* data) const {
  auto* p = reinterpret_cast<fftw_complex*>(data);
  fftw_execute_dft(static_cast<fftw_plan>(fwd_), p, p);
}
void FFT3D::backward(cplx* data) const {
  auto* p = reinterpret_cast<fftw_complex*>(data);
  fftw_execute_dft(static_cast<fftw_plan>(bwd_), p, p);
}

void FFT3D::backward_pruned(cplx* data, const std::vector<int>& columns, const std::vector<int>& planes) const {
  auto* p = reinterpret_cast<fftw_complex*>(data);
  const size_t n3 = n_[2], plane = static_cast<size_t>(n_[1]) * n_[2];
  for (int c : columns) fftw_execute_dft(static_cast<fftw_plan>(line_[1][0]), p + c * n3, p + c * n3);
  for (int i1 : planes) fftw_execute_dft(static_cast<fftw_plan>(line_[1][1]), p + i1 * plane, p + i1 * plane);
  fftw_execute_dft(static_cast<fftw_plan>(line_[1][2]), p, p);
}

void FFT3D::forward_pruned(cplx* data, const std::vector<int>& columns, const std::vector<int>& planes) const {
  auto* p = reinterpret_cast<fftw_complex*>(data);
  const size_t n3 = n_[2], plane = static_cast<size_t>(n_[1]) * n_[2];
  fftw_execute_dft(static_cast<fftw_plan>(line_[0][2]), p, p);
  for (int i1 : planes) fftw_execute_dft(static_cast<fftw_plan>(line_[0][1]), p + i1 * plane, p + i1 * plane);
  for (int c : columns) fftw_execute_dft(static_cast<fftw_plan>(line_[0][0]), p + c * n3, p + c * n3);
}

int good_fft_size(int n) {
  for (int m = n < 2 ? 2 : n;; ++m) {
    int k = m;
    for (int p : {2, 3, 5, 7})
      while (k % p == 0) k /= p;
    if (k == 1) return m;
  }
}

}  // namespace lsda
