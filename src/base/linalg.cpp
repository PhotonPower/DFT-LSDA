#include "base/linalg.hpp"

extern "C" {
void zgemm_(const char*, const char*, const int*, const int*, const int*, const lsda::cplx*, const lsda::cplx*,
            const int*, const lsda::cplx*, const int*, const lsda::cplx*, lsda::cplx*, const int*);
void zpotrf_(const char*, const int*, lsda::cplx*, const int*, int*);
void ztrsm_(const char*, const char*, const char*, const char*, const int*, const int*, const lsda::cplx*,
            const lsda::cplx*, const int*, lsda::cplx*, const int*);
void zheevd_(const char*, const char*, const int*, lsda::cplx*, const int*, double*, lsda::cplx*, const int*,
             double*, const int*, int*, const int*, int*);
void zhegst_(const int*, const char*, const int*, lsda::cplx*, const int*, const lsda::cplx*, const int*, int*);
void dsyev_(const char*, const char*, const int*, double*, const int*, double*, double*, const int*, int*);
void dgesv_(const int*, const int*, double*, const int*, int*, double*, const int*, int*);
}

namespace lsda {

void gemm(char opa, char opb, int m, int n, int k, cplx alpha, const cplx* a, int lda, const cplx* b, int ldb,
          cplx beta, cplx* c, int ldc) {
  if (m == 0 || n == 0) return;
  zgemm_(&opa, &opb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

CMatrix overlap(const cplx* a, int ma, const cplx* b, int nb, int npw) {
  CMatrix c(ma, nb);
  gemm('C', 'N', ma, nb, npw, 1.0, a, npw, b, npw, 0.0, c.data(), ma);
  return c;
}

bool cholesky_lower(CMatrix& m) {
  int n = m.rows(), info = 0;
  if (n == 0) return true;
  zpotrf_("L", &n, m.data(), &n, &info);
  if (info != 0) return false;
  for (int j = 1; j < n; ++j)
    for (int i = 0; i < j; ++i) m(i, j) = 0.0;
  return true;
}

CMatrix inverse_lh(const CMatrix& l) {
  const int n = l.rows();
  CMatrix t(n, n);
  for (int i = 0; i < n; ++i) t(i, i) = 1.0;
  if (n == 0) return t;
  const cplx one = 1.0;
  ztrsm_("R", "L", "C", "N", &n, &n, &one, l.data(), &n, t.data(), &n);  // t = L^{-H} (upper triangular)
  return t;
}

void solve_right_lh(const CMatrix& l, cplx* x, int npw, int n) {
  if (n == 0) return;
  // X L^{-H} as a matrix product with the explicit inverse: level-3 BLAS gemm is much better
  // threaded than trsm, and the Cholesky factors met here are well conditioned.
  const CMatrix t = inverse_lh(l);
  std::vector<cplx> tmp(static_cast<size_t>(npw) * n);
  gemm('N', 'N', npw, n, n, 1.0, x, npw, t.data(), n, 0.0, tmp.data(), npw);
  std::copy(tmp.begin(), tmp.end(), x);
}

void trsm_lower(const CMatrix& l, CMatrix& b, bool left_inverse) {
  const cplx one = 1.0;
  int m = b.rows(), n = b.cols(), nl = l.rows();
  if (m == 0 || n == 0) return;
  if (left_inverse)
    ztrsm_("L", "L", "N", "N", &m, &n, &one, l.data(), &nl, b.data(), &m);
  else
    ztrsm_("R", "L", "C", "N", &m, &n, &one, l.data(), &nl, b.data(), &m);
}

void eigh(CMatrix& a, std::vector<double>& w) {
  int n = a.rows(), info = 0;
  w.assign(n, 0.0);
  if (n == 0) return;
  int lwork = -1, lrwork = -1, liwork = -1, iwq = 0;
  cplx wq;
  double rwq;
  zheevd_("V", "L", &n, a.data(), &n, w.data(), &wq, &lwork, &rwq, &lrwork, &iwq, &liwork, &info);
  lwork = static_cast<int>(wq.real()) + 1;
  lrwork = static_cast<int>(rwq) + 1;
  liwork = iwq + 1;
  std::vector<cplx> work(lwork);
  std::vector<double> rwork(lrwork);
  std::vector<int> iwork(liwork);
  zheevd_("V", "L", &n, a.data(), &n, w.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork,
          &info);
  LSDA_CHECK(info == 0, "zheevd failed");
}

bool eigh_gen(CMatrix& a, CMatrix& b, std::vector<double>& w) {
  int n = a.rows(), info = 0;
  if (n == 0) return true;
  zpotrf_("L", &n, b.data(), &n, &info);
  if (info != 0) return false;
  const int itype = 1;
  zhegst_(&itype, "L", &n, a.data(), &n, b.data(), &n, &info);
  LSDA_CHECK(info == 0, "zhegst failed");
  eigh(a, w);
  const cplx one = 1.0;  // back-transform: c = L^{-H} y
  ztrsm_("L", "L", "C", "N", &n, &n, &one, b.data(), &n, a.data(), &n);
  return true;
}

void eigh_real(RMatrix& a, std::vector<double>& w) {
  int n = a.rows(), info = 0, lwork = -1;
  w.assign(n, 0.0);
  if (n == 0) return;
  double wq;
  dsyev_("V", "L", &n, a.data(), &n, w.data(), &wq, &lwork, &info);
  lwork = static_cast<int>(wq) + 1;
  std::vector<double> work(lwork);
  dsyev_("V", "L", &n, a.data(), &n, w.data(), work.data(), &lwork, &info);
  LSDA_CHECK(info == 0, "dsyev failed");
}

bool solve_real(RMatrix& a, std::vector<double>& b) {
  int n = a.rows(), nrhs = 1, info = 0;
  if (n == 0) return true;
  std::vector<int> ipiv(n);
  dgesv_(&n, &nrhs, a.data(), &n, ipiv.data(), b.data(), &n, &info);
  return info == 0;
}

}  // namespace lsda
