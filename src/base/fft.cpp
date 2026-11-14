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
}

FFT3D::~FFT3D() {
  if (fwd_) fftw_destroy_plan(static_cast<fftw_plan>(fwd_));
  if (bwd_) fftw_destroy_plan(static_cast<fftw_plan>(bwd_));
}

FFT3D::FFT3D(FFT3D&& o) noexcept { *this = std::move(o); }
FFT3D& FFT3D::operator=(FFT3D&& o) noexcept {
  std::swap(n_, o.n_);
  std::swap(size_, o.size_);
  std::swap(fwd_, o.fwd_);
  std::swap(bwd_, o.bwd_);
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

int good_fft_size(int n) {
  for (int m = n < 2 ? 2 : n;; ++m) {
    int k = m;
    for (int p : {2, 3, 5, 7})
      while (k % p == 0) k /= p;
    if (k == 1) return m;
  }
}

}  // namespace lsda
