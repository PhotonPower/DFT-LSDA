// 3D complex FFT (FFTW3 backend). Conventions:
//   forward : F(G) = sum_r f(r) exp(-i G r)   (unnormalized)
//   backward: f(r) = sum_G F(G) exp(+i G r)
// Grid index = (i1 * n2 + i2) * n3 + i3. execute() is thread-safe on distinct arrays.
#pragma once
#include "base/common.hpp"

namespace lsda {

// Allocator returning SIMD-aligned memory (required for FFTW new-array execution).
template <class T>
struct AlignedAllocator {
  using value_type = T;
  AlignedAllocator() = default;
  template <class U>
  AlignedAllocator(const AlignedAllocator<U>&) {}
  T* allocate(size_t n);
  void deallocate(T* p, size_t);
  template <class U>
  bool operator==(const AlignedAllocator<U>&) const { return true; }
};
void* aligned_malloc(size_t bytes);
void aligned_free(void* p);
template <class T>
T* AlignedAllocator<T>::allocate(size_t n) { return static_cast<T*>(aligned_malloc(n * sizeof(T))); }
template <class T>
void AlignedAllocator<T>::deallocate(T* p, size_t) { aligned_free(p); }

using CVec = std::vector<cplx, AlignedAllocator<cplx>>;

class FFT3D {
 public:
  FFT3D() = default;
  FFT3D(int n1, int n2, int n3);
  ~FFT3D();
  FFT3D(const FFT3D&) = delete;
  FFT3D& operator=(const FFT3D&) = delete;
  FFT3D(FFT3D&& o) noexcept;
  FFT3D& operator=(FFT3D&& o) noexcept;

  void forward(cplx* data) const;   // in place
  void backward(cplx* data) const;  // in place

  // Pruned transforms for data whose reciprocal-space support is a sphere much smaller than the box
  // (plane-wave coefficients on the wavefunction grid). `columns` lists the (i1 * n2 + i2) columns and
  // `planes` the i1 planes that contain non-zero coefficients.
  //   backward_pruned: data must be zero outside the listed columns; result is the full real-space array.
  //   forward_pruned : input is a full real-space array; on exit only the listed columns are valid.
  // Only the 1D transforms that can contribute are carried out (z for the columns, y for the planes, x for all).
  void backward_pruned(cplx* data, const std::vector<int>& columns, const std::vector<int>& planes) const;
  void forward_pruned(cplx* data, const std::vector<int>& columns, const std::vector<int>& planes) const;
  int n(int d) const { return n_[d]; }
  size_t size() const { return size_; }
  // FFT-box index of the integer reciprocal vector (m1,m2,m3); |m_d| must be < n_d/2 (+1)
  size_t index(int m1, int m2, int m3) const {
    const int i1 = m1 < 0 ? m1 + n_[0] : m1, i2 = m2 < 0 ? m2 + n_[1] : m2, i3 = m3 < 0 ? m3 + n_[2] : m3;
    return (static_cast<size_t>(i1) * n_[1] + i2) * n_[2] + i3;
  }

 private:
  int n_[3] = {0, 0, 0};
  size_t size_ = 0;
  void* fwd_ = nullptr;
  void* bwd_ = nullptr;
  void* line_[2][3] = {{nullptr, nullptr, nullptr}, {nullptr, nullptr, nullptr}};  // [direction][z, y-plane, x-all]
};

// smallest integer >= n whose prime factors are all in {2,3,5,7}
int good_fft_size(int n);

}  // namespace lsda
