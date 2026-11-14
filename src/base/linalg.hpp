// Dense column-major matrices and thin wrappers around BLAS/LAPACK.
#pragma once
#include <algorithm>
#include "base/common.hpp"

namespace lsda {

template <class T>
class Matrix {
 public:
  Matrix() = default;
  Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(static_cast<size_t>(rows) * cols, T{}) {}
  void resize(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;
    data_.assign(static_cast<size_t>(rows) * cols, T{});
  }
  int rows() const { return rows_; }
  int cols() const { return cols_; }
  T& operator()(int i, int j) { return data_[i + static_cast<size_t>(j) * rows_]; }
  const T& operator()(int i, int j) const { return data_[i + static_cast<size_t>(j) * rows_]; }
  T* data() { return data_.data(); }
  const T* data() const { return data_.data(); }
  T* col(int j) { return data_.data() + static_cast<size_t>(j) * rows_; }
  const T* col(int j) const { return data_.data() + static_cast<size_t>(j) * rows_; }
  void fill(T v) { std::fill(data_.begin(), data_.end(), v); }
  size_t size() const { return data_.size(); }

 private:
  int rows_ = 0, cols_ = 0;
  std::vector<T> data_;
};

using CMatrix = Matrix<cplx>;
using RMatrix = Matrix<double>;

// C = alpha * op(A) * op(B) + beta * C, op = 'N' or 'C' (conjugate transpose).
void gemm(char opa, char opb, int m, int n, int k, cplx alpha, const cplx* a, int lda, const cplx* b, int ldb,
          cplx beta, cplx* c, int ldc);

// Convenience: C(ma x nb) = A^H B for A (npw x ma), B (npw x nb)
CMatrix overlap(const cplx* a, int ma, const cplx* b, int nb, int npw);

// In-place Cholesky M = L L^H (lower). Returns false if M is not positive definite.
bool cholesky_lower(CMatrix& m);
// L^{-H} as a dense (upper triangular) matrix
CMatrix inverse_lh(const CMatrix& l);
// X <- X L^{-H} for X (npw x n), L lower triangular (n x n).
void solve_right_lh(const CMatrix& l, cplx* x, int npw, int n);
// B <- L^{-1} B (left = true) or B <- B L^{-H} (left = false) for small matrices
void trsm_lower(const CMatrix& l, CMatrix& b, bool left_inverse);

// Hermitian eigenproblem: on exit columns of a hold eigenvectors, w ascending eigenvalues.
void eigh(CMatrix& a, std::vector<double>& w);
// Generalized Hermitian eigenproblem A c = w B c. Returns false if B is not positive definite.
bool eigh_gen(CMatrix& a, CMatrix& b, std::vector<double>& w);
// Real symmetric eigenproblem
void eigh_real(RMatrix& a, std::vector<double>& w);
// Solve real linear system A x = b (A destroyed); returns false if singular.
bool solve_real(RMatrix& a, std::vector<double>& b);

}  // namespace lsda
