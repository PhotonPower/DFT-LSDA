#include "dft/eigensolver.hpp"

#include <algorithm>
#include <cmath>

namespace lsda {

namespace {

// Y <- Y * M for Y (n x k), M (k x m); result has m columns
void right_multiply(CMatrix& Y, const CMatrix& M, CMatrix& tmp) {
  tmp.resize(Y.rows(), M.cols());
  gemm('N', 'N', Y.rows(), M.cols(), Y.cols(), 1.0, Y.data(), Y.rows(), M.data(), M.rows(), 0.0, tmp.data(), Y.rows());
  std::swap(Y, tmp);
}

void column_norms(const CMatrix& R, std::vector<double>& nrm) {
  nrm.assign(R.cols(), 0.0);
#pragma omp parallel for schedule(static)
  for (int j = 0; j < R.cols(); ++j) {
    double s = 0.0;
    const cplx* r = R.col(j);
    for (int i = 0; i < R.rows(); ++i) s += std::norm(r[i]);
    nrm[j] = std::sqrt(s);
  }
}

// S-orthonormalize the columns of W (with images HW, SW). Returns the number of columns kept.
int orthonormalize(CMatrix& W, CMatrix& HW, CMatrix& SW, CMatrix& tmp) {
  const int n = W.rows(), k = W.cols();
  CMatrix M = overlap(W.data(), k, SW.data(), k, n);
  for (int i = 0; i < k; ++i)
    for (int j = 0; j < i; ++j) M(i, j) = std::conj(M(j, i));
  CMatrix L = M;
  if (cholesky_lower(L)) {
    solve_right_lh(L, W.data(), n, k);
    solve_right_lh(L, HW.data(), n, k);
    solve_right_lh(L, SW.data(), n, k);
    return k;
  }
  // (nearly) linearly dependent columns: canonical orthonormalization, dropping the null space
  std::vector<double> lam;
  eigh(M, lam);
  int first = 0;
  while (first < k && lam[first] < 1e-12 * lam[k - 1]) ++first;
  const int kept = k - first;
  if (kept == 0) return 0;
  CMatrix T(k, kept);
  for (int j = 0; j < kept; ++j) {
    const double s = 1.0 / std::sqrt(lam[first + j]);
    for (int i = 0; i < k; ++i) T(i, j) = M(i, first + j) * s;
  }
  right_multiply(W, T, tmp);
  right_multiply(HW, T, tmp);
  right_multiply(SW, T, tmp);
  return kept;
}

}  // namespace

EigensolverResult lobpcg(const Hamiltonian& h, CMatrix& X, std::vector<double>& eps, const std::vector<double>& tol,
                         int n_required, int max_iter, CMatrix* PX) {
  const int n = h.npw(), nb = X.cols();
  LSDA_CHECK(X.rows() == n && nb <= n, "lobpcg: inconsistent dimensions");
  EigensolverResult res;
  CMatrix HX(n, nb), SX(n, nb), tmp;
  CMatrix W, HW, SW, P, HP, SP;
  CMatrix Gxp, Gpp;  // X^H H P and P^H H P carried along with the basis
  int np = 0;

  // ---- initial Rayleigh-Ritz step --------------------------------------------------
  h.apply(X.data(), nb, HX.data(), SX.data());
  {
    ScopedTimer t_rr("  lobpcg: initial RR");
    const int kept = orthonormalize(X, HX, SX, tmp);
    LSDA_CHECK(kept == nb, "lobpcg: start vectors are linearly dependent");
    CMatrix G = overlap(X.data(), nb, HX.data(), nb, n);
    for (int i = 0; i < nb; ++i)
      for (int j = 0; j < i; ++j) G(i, j) = std::conj(G(j, i));
    eigh(G, eps);
    right_multiply(X, G, tmp);
    right_multiply(HX, G, tmp);
    right_multiply(SX, G, tmp);
  }

  CMatrix R(n, nb);
  std::vector<double> rnorm;
  double t_prec = 0, t_orth = 0, t_rr = 0, t_upd = 0;
  auto residuals = [&]() {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < nb; ++j) {
      const cplx *hx = HX.col(j), *sx = SX.col(j);
      cplx* r = R.col(j);
      for (int i = 0; i < n; ++i) r[i] = hx[i] - eps[j] * sx[i];
    }
    column_norms(R, rnorm);
    res.max_residual = 0.0;
    res.not_converged = 0;
    for (int j = 0; j < std::min(nb, n_required); ++j) {
      res.max_residual = std::max(res.max_residual, rnorm[j]);
      if (rnorm[j] > tol[j]) ++res.not_converged;
    }
  };

  for (int iter = 0; iter < max_iter; ++iter) {
    residuals();
    std::vector<int> active;
    for (int j = 0; j < nb; ++j)
      if (rnorm[j] > tol[j]) active.push_back(j);
    if (active.empty()) break;
    res.iterations = iter + 1;
    int na = static_cast<int>(active.size());

    // ---- preconditioned residuals (Teter-Payne-Allan) --------------------------------
    double t0 = wall_time();
    W.resize(n, na);
#pragma omp parallel for schedule(static)
    for (int a = 0; a < na; ++a) {
      const int j = active[a];
      const cplx* x = X.col(j);
      double ek = 0.0, nrm = 0.0;
      for (int i = 0; i < n; ++i) {
        ek += h.kinetic(i) * std::norm(x[i]);
        nrm += std::norm(x[i]);
      }
      ek = std::max(ek / nrm, 1e-3);
      const cplx* r = R.col(j);
      cplx* w = W.col(a);
      for (int i = 0; i < n; ++i) {
        const double t = h.kinetic(i) / (1.5 * ek);
        const double num = 27.0 + t * (18.0 + t * (12.0 + 8.0 * t));
        w[i] = r[i] * (num / (num + 16.0 * t * t * t * t));
      }
    }
    // ---- S-orthogonalize against X and P ---------------------------------------------
    {
      CMatrix c = overlap(SX.data(), nb, W.data(), na, n);
      gemm('N', 'N', n, na, nb, -1.0, X.data(), n, c.data(), nb, 1.0, W.data(), n);
      if (np > 0) {
        CMatrix c2 = overlap(SP.data(), np, W.data(), na, n);
        gemm('N', 'N', n, na, np, -1.0, P.data(), n, c2.data(), np, 1.0, W.data(), n);
      }
    }
    HW.resize(n, na);
    SW.resize(n, na);
    t_prec += wall_time() - t0;
    h.apply(W.data(), na, HW.data(), SW.data());
    t0 = wall_time();
    na = orthonormalize(W, HW, SW, tmp);
    t_orth += wall_time() - t0;
    if (na == 0) break;
    t0 = wall_time();

    // ---- Rayleigh-Ritz in the basis [X, W, P] ----------------------------------------
    bool solved = false;
    CMatrix C;  // eigenvectors, dim x nb
    std::vector<double> theta;
    CMatrix G, O;
    int dim = 0;
    for (int attempt = 0; attempt < 2 && !solved; ++attempt) {
      dim = nb + na + np;
      G.resize(dim, dim);
      O.resize(dim, dim);
      for (int i = 0; i < dim; ++i) O(i, i) = 1.0;
      for (int i = 0; i < nb; ++i) G(i, i) = eps[i];
      auto set_block = [&](CMatrix& M, const CMatrix& blk, int r0, int c0) {
        for (int j = 0; j < blk.cols(); ++j)
          for (int i = 0; i < blk.rows(); ++i) {
            M(r0 + i, c0 + j) = blk(i, j);
            M(c0 + j, r0 + i) = std::conj(blk(i, j));
          }
      };
      set_block(G, overlap(X.data(), nb, HW.data(), na, n), 0, nb);
      set_block(O, overlap(X.data(), nb, SW.data(), na, n), 0, nb);
      {
        CMatrix gww = overlap(W.data(), na, HW.data(), na, n), oww = overlap(W.data(), na, SW.data(), na, n);
        for (int i = 0; i < na; ++i)
          for (int j = 0; j < na; ++j) {
            G(nb + i, nb + j) = 0.5 * (gww(i, j) + std::conj(gww(j, i)));
            O(nb + i, nb + j) = 0.5 * (oww(i, j) + std::conj(oww(j, i)));
          }
      }
      if (np > 0) {
        set_block(G, overlap(P.data(), np, HW.data(), na, n), nb + na, nb);  // P^H H W -> (P, W) block
        set_block(O, overlap(P.data(), np, SW.data(), na, n), nb + na, nb);
        set_block(G, Gxp, 0, nb + na);
        for (int i = 0; i < np; ++i)
          for (int j = 0; j < np; ++j) G(nb + na + i, nb + na + j) = Gpp(i, j);
      }
      CMatrix Gc = G, Oc = O;
      if (eigh_gen(Gc, Oc, theta)) {
        C.resize(dim, nb);
        for (int j = 0; j < nb; ++j)
          for (int i = 0; i < dim; ++i) C(i, j) = Gc(i, j);
        solved = true;
      } else {
        np = 0;  // ill-conditioned basis: restart without the P block
      }
    }
    if (!solved) break;
    for (int j = 0; j < nb; ++j) eps[j] = theta[j];
    t_rr += wall_time() - t0;
    t0 = wall_time();

    // ---- new search directions P = [W, P] part of the active Ritz vectors, O-orthonormalized
    const int np_new_max = static_cast<int>(active.size());
    CMatrix Cp(dim, np_new_max);
    for (int a = 0; a < np_new_max; ++a)
      for (int i = nb; i < dim; ++i) Cp(i, a) = C(i, active[a]);
    int np_new = np_new_max;
    {
      // Cp <- Cp - C (C^H O Cp)
      CMatrix OCp(dim, np_new);
      gemm('N', 'N', dim, np_new, dim, 1.0, O.data(), dim, Cp.data(), dim, 0.0, OCp.data(), dim);
      CMatrix proj(nb, np_new);
      gemm('C', 'N', nb, np_new, dim, 1.0, C.data(), dim, OCp.data(), dim, 0.0, proj.data(), nb);
      gemm('N', 'N', dim, np_new, nb, -1.0, C.data(), dim, proj.data(), nb, 1.0, Cp.data(), dim);
      gemm('N', 'N', dim, np_new, dim, 1.0, O.data(), dim, Cp.data(), dim, 0.0, OCp.data(), dim);
      CMatrix M(np_new, np_new);
      gemm('C', 'N', np_new, np_new, dim, 1.0, Cp.data(), dim, OCp.data(), dim, 0.0, M.data(), np_new);
      for (int i = 0; i < np_new; ++i)
        for (int j = 0; j < i; ++j) M(i, j) = std::conj(M(j, i));
      if (cholesky_lower(M)) {
        trsm_lower(M, Cp, false);  // Cp <- Cp L^{-H}
      } else {
        np_new = 0;
      }
    }
    if (np_new > 0) {
      // reduced Hamiltonian blocks for the next iteration
      CMatrix GCp(dim, np_new);
      gemm('N', 'N', dim, np_new, dim, 1.0, G.data(), dim, Cp.data(), dim, 0.0, GCp.data(), dim);
      Gxp.resize(nb, np_new);
      gemm('C', 'N', nb, np_new, dim, 1.0, C.data(), dim, GCp.data(), dim, 0.0, Gxp.data(), nb);
      Gpp.resize(np_new, np_new);
      gemm('C', 'N', np_new, np_new, dim, 1.0, Cp.data(), dim, GCp.data(), dim, 0.0, Gpp.data(), np_new);
      for (int i = 0; i < np_new; ++i)
        for (int j = 0; j <= i; ++j) {
          const cplx v = 0.5 * (Gpp(i, j) + std::conj(Gpp(j, i)));
          Gpp(i, j) = v;
          Gpp(j, i) = std::conj(v);
        }
    }

    // ---- update vectors: Y_new = X Cx + W Cw + P Cp for Y in {X, HX, SX} and the P block ----
    auto combine = [&](const CMatrix& A, const CMatrix& Bw, const CMatrix& Bp, const CMatrix& coef, bool with_x,
                       CMatrix& out) {
      const int m = coef.cols();
      out.resize(n, m);
      if (with_x) gemm('N', 'N', n, m, nb, 1.0, A.data(), n, coef.data(), dim, 0.0, out.data(), n);
      gemm('N', 'N', n, m, na, 1.0, Bw.data(), n, coef.data() + nb, dim, with_x ? 1.0 : 0.0, out.data(), n);
      if (np > 0) gemm('N', 'N', n, m, np, 1.0, Bp.data(), n, coef.data() + nb + na, dim, 1.0, out.data(), n);
    };
    CMatrix Xn, HXn, SXn, Pn, HPn, SPn;
    combine(X, W, P, C, true, Xn);
    combine(HX, HW, HP, C, true, HXn);
    combine(SX, SW, SP, C, true, SXn);
    if (np_new > 0) {
      // Cp has a nonzero X block after orthogonalization against C
      combine(X, W, P, Cp, true, Pn);
      combine(HX, HW, HP, Cp, true, HPn);
      combine(SX, SW, SP, Cp, true, SPn);
    }
    std::swap(X, Xn);
    std::swap(HX, HXn);
    std::swap(SX, SXn);
    std::swap(P, Pn);
    std::swap(HP, HPn);
    std::swap(SP, SPn);
    np = np_new;
    t_upd += wall_time() - t0;
  }
  residuals();
  add_time("  lobpcg: precond+project", t_prec);
  add_time("  lobpcg: orthonormalize W", t_orth);
  add_time("  lobpcg: Rayleigh-Ritz", t_rr);
  add_time("  lobpcg: update vectors", t_upd);
  if (PX) *PX = h.project(X.data(), nb);
  return res;
}

// =====================================================================================
//  PPCG
// =====================================================================================
namespace {

inline cplx vdot(const cplx* a, const cplx* b, int n) {  // a^H b
  double re = 0.0, im = 0.0;
  for (int i = 0; i < n; ++i) {
    re += a[i].real() * b[i].real() + a[i].imag() * b[i].imag();
    im += a[i].real() * b[i].imag() - a[i].imag() * b[i].real();
  }
  return {re, im};
}

}  // namespace

EigensolverResult ppcg(const Hamiltonian& h, CMatrix& X, std::vector<double>& eps, const std::vector<double>& tol,
                       int n_required, int max_iter, CMatrix* PX_out, int block_size, int rr_period, bool px_valid) {
  const int n = h.npw(), nb = X.cols(), nproj = h.nproj();
  LSDA_CHECK(X.rows() == n && nb <= n, "ppcg: inconsistent dimensions");
  LSDA_CHECK(block_size >= 1, "ppcg: block size must be positive");
  EigensolverResult res;
  double t_orth = 0, t_rr = 0, t_res = 0, t_proj = 0, t_band = 0;
  CMatrix tmp, KX(n, nb), PX;
  if (px_valid && PX_out && PX_out->rows() == nproj && PX_out->cols() == nb)
    PX = std::move(*PX_out);
  else
    PX = h.project(X.data(), nb);
  h.apply_local(X.data(), nb, KX.data());
  eps.assign(nb, 0.0);

  // X <- X L^{-H} with L L^H = X^H S X (also applied to K X and the projections)
  auto s_orthonormalize = [&]() {
    const double t0 = wall_time();
    CMatrix SPX;
    h.block_multiply(PX, nullptr, &SPX);
    CMatrix M = overlap(X.data(), nb, X.data(), nb, n);
    if (nproj > 0) gemm('C', 'N', nb, nb, nproj, 1.0, PX.data(), nproj, SPX.data(), nproj, 1.0, M.data(), nb);
    for (int i = 0; i < nb; ++i)
      for (int j = 0; j < i; ++j) M(i, j) = std::conj(M(j, i));
    CMatrix L = M;
    if (cholesky_lower(L)) {
      const CMatrix T = inverse_lh(L);
      right_multiply(X, T, tmp);
      right_multiply(KX, T, tmp);
      if (nproj > 0) right_multiply(PX, T, tmp);
    } else {
      // (nearly) dependent vectors: canonical orthonormalization with clamped eigenvalues
      std::vector<double> lam;
      eigh(M, lam);
      const double floor = 1e-14 * std::max(lam[nb - 1], 1e-300);
      for (int j = 0; j < nb; ++j) {
        const double sc = 1.0 / std::sqrt(std::max(lam[j], floor));
        for (int i = 0; i < nb; ++i) M(i, j) *= sc;
      }
      right_multiply(X, M, tmp);
      right_multiply(KX, M, tmp);
      if (nproj > 0) right_multiply(PX, M, tmp);
    }
    t_orth += wall_time() - t0;
  };
  // full Rayleigh-Ritz step in span(X); X must be S-orthonormal
  auto rayleigh_ritz = [&]() {
    const double t0 = wall_time();
    CMatrix DPX;
    h.block_multiply(PX, &DPX, nullptr);
    CMatrix G = overlap(X.data(), nb, KX.data(), nb, n);
    if (nproj > 0) gemm('C', 'N', nb, nb, nproj, 1.0, PX.data(), nproj, DPX.data(), nproj, 1.0, G.data(), nb);
    for (int i = 0; i < nb; ++i) {
      G(i, i) = G(i, i).real();
      for (int j = 0; j < i; ++j) {
        const cplx v = 0.5 * (G(i, j) + std::conj(G(j, i)));
        G(i, j) = v;
        G(j, i) = std::conj(v);
      }
    }
    eigh(G, eps);
    right_multiply(X, G, tmp);
    right_multiply(KX, G, tmp);
    if (nproj > 0) right_multiply(PX, G, tmp);
    t_rr += wall_time() - t0;
  };
  // Orthonormalization and Rayleigh-Ritz in one step: generalized problem (X^H H X) c = theta (X^H S X) c
  auto rayleigh_ritz_general = [&]() {
    const double t0 = wall_time();
    CMatrix DPX, SPX;
    h.block_multiply(PX, &DPX, &SPX);
    CMatrix G = overlap(X.data(), nb, KX.data(), nb, n), M = overlap(X.data(), nb, X.data(), nb, n);
    if (nproj > 0) {
      gemm('C', 'N', nb, nb, nproj, 1.0, PX.data(), nproj, DPX.data(), nproj, 1.0, G.data(), nb);
      gemm('C', 'N', nb, nb, nproj, 1.0, PX.data(), nproj, SPX.data(), nproj, 1.0, M.data(), nb);
    }
    for (int i = 0; i < nb; ++i) {
      G(i, i) = G(i, i).real();
      M(i, i) = M(i, i).real();
      for (int j = 0; j < i; ++j) {
        const cplx g = 0.5 * (G(i, j) + std::conj(G(j, i))), m = 0.5 * (M(i, j) + std::conj(M(j, i)));
        G(i, j) = g;
        G(j, i) = std::conj(g);
        M(i, j) = m;
        M(j, i) = std::conj(m);
      }
    }
    if (eigh_gen(G, M, eps)) {
      right_multiply(X, G, tmp);
      right_multiply(KX, G, tmp);
      if (nproj > 0) right_multiply(PX, G, tmp);
      t_rr += wall_time() - t0;
    } else {  // (nearly) dependent vectors
      t_rr += wall_time() - t0;
      s_orthonormalize();
      rayleigh_ritz();
    }
  };
  // R = K X - X eps + beta (dH - eps dS) P_X ; returns residual norms
  CMatrix R(n, nb);
  std::vector<double> rnorm;
  auto residuals = [&]() {
    const double t0 = wall_time();
    CMatrix DPX, SPX;
    h.block_multiply(PX, &DPX, &SPX);
#pragma omp parallel for schedule(static)
    for (int j = 0; j < nb; ++j) {
      cplx *d = DPX.col(j);
      const cplx* s = SPX.col(j);
      for (int i = 0; i < nproj; ++i) d[i] -= eps[j] * s[i];
      const cplx *kx = KX.col(j), *x = X.col(j);
      cplx* r = R.col(j);
      for (int i = 0; i < n; ++i) r[i] = kx[i] - eps[j] * x[i];
    }
    if (nproj > 0) h.add_nonlocal(DPX, R.data(), nb);
    column_norms(R, rnorm);
    res.max_residual = 0.0;
    res.not_converged = 0;
    for (int j = 0; j < std::min(nb, n_required); ++j) {
      res.max_residual = std::max(res.max_residual, rnorm[j]);
      if (rnorm[j] > tol[j]) ++res.not_converged;
    }
    t_res += wall_time() - t0;
  };

  rayleigh_ritz_general();

  CMatrix W, KW, PW, P, KP, PP;       // search directions with local images and projections
  std::vector<int> p_col(nb, -1);     // column of P belonging to each band (-1: none)
  bool ritz = true;                   // X currently holds Ritz vectors
  bool have_stats = false;
  for (int iter = 0; iter < max_iter; ++iter) {
    residuals();
    have_stats = true;
    std::vector<int> active;
    for (int j = 0; j < nb; ++j)
      if (rnorm[j] > tol[j]) active.push_back(j);
    if (active.empty()) break;
    res.iterations = iter + 1;
    const int na = static_cast<int>(active.size());

    // ---- preconditioned residuals, S-orthogonal to X ---------------------------------------
    double t0 = wall_time();
    W.resize(n, na);
#pragma omp parallel for schedule(static)
    for (int a = 0; a < na; ++a) {
      const int j = active[a];
      const cplx* x = X.col(j);
      double ek = 0.0, nrm = 0.0;
      for (int i = 0; i < n; ++i) {
        ek += h.kinetic(i) * std::norm(x[i]);
        nrm += std::norm(x[i]);
      }
      ek = std::max(ek / nrm, 1e-3);
      const cplx* r = R.col(j);
      cplx* w = W.col(a);
      for (int i = 0; i < n; ++i) {
        const double t = h.kinetic(i) / (1.5 * ek);
        const double num = 27.0 + t * (18.0 + t * (12.0 + 8.0 * t));
        w[i] = r[i] * (num / (num + 16.0 * t * t * t * t));
      }
    }
    PW = h.project(W.data(), na);
    CMatrix SPX;
    h.block_multiply(PX, nullptr, &SPX);
    auto project_out_x = [&](CMatrix& V, CMatrix* KV, CMatrix& PV) {  // V <- (1 - X X^H S) V
      const int m = V.cols();
      CMatrix c = overlap(X.data(), nb, V.data(), m, n);
      if (nproj > 0) gemm('C', 'N', nb, m, nproj, 1.0, SPX.data(), nproj, PV.data(), nproj, 1.0, c.data(), nb);
      gemm('N', 'N', n, m, nb, -1.0, X.data(), n, c.data(), nb, 1.0, V.data(), n);
      if (KV) gemm('N', 'N', n, m, nb, -1.0, KX.data(), n, c.data(), nb, 1.0, KV->data(), n);
      if (nproj > 0) gemm('N', 'N', nproj, m, nb, -1.0, PX.data(), nproj, c.data(), nb, 1.0, PV.data(), nproj);
    };
    project_out_x(W, nullptr, PW);
    t_proj += wall_time() - t0;
    KW.resize(n, na);
    h.apply_local(W.data(), na, KW.data());

    // previous search directions of the active bands, projected as well
    t0 = wall_time();
    std::vector<int> pa(na, -1);
    {
      int np = 0;
      for (int a = 0; a < na; ++a)
        if (p_col[active[a]] >= 0) ++np;
      if (np > 0) {
        CMatrix P2(n, np), KP2(n, np), PP2(nproj, np);
        int q = 0;
        for (int a = 0; a < na; ++a) {
          const int c = p_col[active[a]];
          if (c < 0) continue;
          std::copy(P.col(c), P.col(c) + n, P2.col(q));
          std::copy(KP.col(c), KP.col(c) + n, KP2.col(q));
          std::copy(PP.col(c), PP.col(c) + nproj, PP2.col(q));
          pa[a] = q++;
        }
        std::swap(P, P2);
        std::swap(KP, KP2);
        std::swap(PP, PP2);
        project_out_x(P, &KP, PP);
      }
    }
    t_proj += wall_time() - t0;

    // ---- Rayleigh-Ritz within sub-blocks of bands: span{x_j, w_j, p_j : j in block} ---------------
    t0 = wall_time();
    CMatrix DPX, DPW, SPW, DPP, SPP;
    h.block_multiply(PX, &DPX, nullptr);
    h.block_multiply(PW, &DPW, &SPW);
    if (P.cols() > 0) h.block_multiply(PP, &DPP, &SPP);
    CMatrix Pn(n, na), KPn(n, na), PPn(nproj, na);
    std::vector<char> updated(na, 0);
    const int ngroups = (na + block_size - 1) / block_size;
#pragma omp parallel for schedule(dynamic, 1)
    for (int g = 0; g < ngroups; ++g) {
      const int a0 = g * block_size, s = std::min(block_size, na - a0);
      // basis vectors of the block: x (s), w (s), p (those available)
      std::vector<const cplx*> vec, kvec, pvec, dvec, svec;
      for (int i = 0; i < s; ++i) {
        const int j = active[a0 + i];
        vec.push_back(X.col(j)), kvec.push_back(KX.col(j)), pvec.push_back(PX.col(j));
        dvec.push_back(DPX.col(j)), svec.push_back(SPX.col(j));
      }
      for (int i = 0; i < s; ++i) {
        const int a = a0 + i;
        vec.push_back(W.col(a)), kvec.push_back(KW.col(a)), pvec.push_back(PW.col(a));
        dvec.push_back(DPW.col(a)), svec.push_back(SPW.col(a));
      }
      for (int i = 0; i < s; ++i) {
        const int q = pa[a0 + i];
        if (q < 0) continue;
        vec.push_back(P.col(q)), kvec.push_back(KP.col(q)), pvec.push_back(PP.col(q));
        dvec.push_back(DPP.col(q)), svec.push_back(SPP.col(q));
      }
      const int mfull = static_cast<int>(vec.size());
      CMatrix A(mfull, mfull), B(mfull, mfull);
      for (int u = 0; u < mfull; ++u)
        for (int v = u; v < mfull; ++v) {
          A(u, v) = vdot(vec[u], kvec[v], n) + (nproj > 0 ? vdot(pvec[u], dvec[v], nproj) : cplx(0.0));
          B(u, v) = vdot(vec[u], vec[v], n) + (nproj > 0 ? vdot(pvec[u], svec[v], nproj) : cplx(0.0));
          A(v, u) = std::conj(A(u, v));
          B(v, u) = std::conj(B(u, v));
        }
      // scale to unit S-norm, then solve the projected problem; drop the p block if it is ill-conditioned
      std::vector<double> sc(mfull, 0.0);
      bool ok = true;
      for (int u = 0; u < mfull; ++u) {
        const double d = B(u, u).real();
        if (!(d > 1e-300)) ok = false;
        sc[u] = ok ? 1.0 / std::sqrt(d) : 0.0;
      }
      CMatrix C;
      int m = mfull;
      auto solve = [&](int mm) {
        CMatrix As(mm, mm), Bs(mm, mm);
        for (int u = 0; u < mm; ++u)
          for (int v = 0; v < mm; ++v) {
            As(u, v) = A(u, v) * sc[u] * sc[v];
            Bs(u, v) = B(u, v) * sc[u] * sc[v];
          }
        std::vector<double> w;
        CMatrix Bc = Bs;
        eigh(Bc, w);
        if (w[0] < 1e-8) return false;
        if (!eigh_gen(As, Bs, w)) return false;
        for (int i = 0; i < s; ++i)
          if (!std::isfinite(w[i])) return false;
        C.resize(mm, s);
        for (int i = 0; i < s; ++i)
          for (int u = 0; u < mm; ++u) C(u, i) = As(u, i) * sc[u];
        m = mm;
        return true;
      };
      if (ok) ok = solve(mfull) || (mfull > 2 * s && solve(2 * s));
      if (!ok) continue;
      // new directions p_i = sum_{u in w,p} C(u,i) v_u and vectors x_i = sum_{u in x} C(u,i) x_u + p_i
      std::vector<cplx> xn(static_cast<size_t>(n) * s), kxn(static_cast<size_t>(n) * s), pxn(static_cast<size_t>(nproj) * s);
      for (int i = 0; i < s; ++i) {
        const int a = a0 + i;
        cplx *pn = Pn.col(a), *kpn = KPn.col(a), *ppn = PPn.col(a);
        cplx *xi = &xn[static_cast<size_t>(i) * n], *kxi = &kxn[static_cast<size_t>(i) * n], *pxi = &pxn[static_cast<size_t>(i) * nproj];
        std::fill(pn, pn + n, cplx(0.0));
        std::fill(kpn, kpn + n, cplx(0.0));
        std::fill(ppn, ppn + nproj, cplx(0.0));
        for (int u = s; u < m; ++u) {
          const cplx cu = C(u, i);
          const cplx *vu = vec[u], *kvu = kvec[u], *pvu = pvec[u];
          for (int r = 0; r < n; ++r) {
            pn[r] += cu * vu[r];
            kpn[r] += cu * kvu[r];
          }
          for (int r = 0; r < nproj; ++r) ppn[r] += cu * pvu[r];
        }
        std::copy(pn, pn + n, xi);
        std::copy(kpn, kpn + n, kxi);
        std::copy(ppn, ppn + nproj, pxi);
        for (int u = 0; u < s; ++u) {
          const cplx cu = C(u, i);
          const cplx *vu = vec[u], *kvu = kvec[u], *pvu = pvec[u];
          for (int r = 0; r < n; ++r) {
            xi[r] += cu * vu[r];
            kxi[r] += cu * kvu[r];
          }
          for (int r = 0; r < nproj; ++r) pxi[r] += cu * pvu[r];
        }
      }
      for (int i = 0; i < s; ++i) {
        const int j = active[a0 + i];
        std::copy(&xn[static_cast<size_t>(i) * n], &xn[static_cast<size_t>(i) * n] + n, X.col(j));
        std::copy(&kxn[static_cast<size_t>(i) * n], &kxn[static_cast<size_t>(i) * n] + n, KX.col(j));
        std::copy(&pxn[static_cast<size_t>(i) * nproj], &pxn[static_cast<size_t>(i) * nproj] + nproj, PX.col(j));
        updated[a0 + i] = 1;
      }
    }
    std::fill(p_col.begin(), p_col.end(), -1);
    for (int a = 0; a < na; ++a)
      if (updated[a]) p_col[active[a]] = a;
    std::swap(P, Pn);
    std::swap(KP, KPn);
    std::swap(PP, PPn);
    t_band += wall_time() - t0;
    ritz = false;

    // ---- restore S-orthonormality; Rayleigh quotients or periodic full Rayleigh-Ritz ----------
    if (iter + 1 >= max_iter) {  // last update: orthonormalization and Rayleigh-Ritz in one step
      rayleigh_ritz_general();
      ritz = true;
      break;
    }
    s_orthonormalize();
    if ((iter + 1) % rr_period == 0) {
      rayleigh_ritz();
      std::fill(p_col.begin(), p_col.end(), -1);  // restart the conjugate directions
      ritz = true;
    } else {
      CMatrix D2;
      h.block_multiply(PX, &D2, nullptr);
#pragma omp parallel for schedule(static)
      for (int j = 0; j < nb; ++j)
        eps[j] = (vdot(X.col(j), KX.col(j), n) + (nproj > 0 ? vdot(PX.col(j), D2.col(j), nproj) : cplx(0.0))).real();
    }
  }
  if (!ritz) rayleigh_ritz();
  if (!have_stats) residuals();
  add_time("  ppcg: orthonormalize", t_orth);
  add_time("  ppcg: Rayleigh-Ritz", t_rr);
  add_time("  ppcg: residuals", t_res);
  add_time("  ppcg: precond+project", t_proj);
  add_time("  ppcg: band updates", t_band);
  if (PX_out) *PX_out = std::move(PX);
  return res;
}

bool subspace_start(const Hamiltonian& h, CMatrix& X, int nkeep) {
  const int n = h.npw(), m = X.cols();
  CMatrix HX(n, m), SX(n, m), tmp;
  h.apply(X.data(), m, HX.data(), SX.data());
  CMatrix M = overlap(X.data(), m, SX.data(), m, n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < i; ++j) M(i, j) = std::conj(M(j, i));
  // canonical orthonormalization, discarding (nearly) dependent directions
  std::vector<double> lam;
  eigh(M, lam);
  int first = 0;
  while (first < m && lam[first] < 1e-7 * lam[m - 1]) ++first;
  const int kept = m - first;
  if (kept < nkeep) return false;
  CMatrix T(m, kept);
  for (int j = 0; j < kept; ++j) {
    const double sc = 1.0 / std::sqrt(lam[first + j]);
    for (int i = 0; i < m; ++i) T(i, j) = M(i, first + j) * sc;
  }
  right_multiply(X, T, tmp);
  right_multiply(HX, T, tmp);
  CMatrix G = overlap(X.data(), kept, HX.data(), kept, n);
  for (int i = 0; i < kept; ++i)
    for (int j = 0; j < i; ++j) G(i, j) = std::conj(G(j, i));
  eigh(G, lam);
  CMatrix C(kept, nkeep);
  for (int j = 0; j < nkeep; ++j)
    for (int i = 0; i < kept; ++i) C(i, j) = G(i, j);
  right_multiply(X, C, tmp);
  return true;
}

}  // namespace lsda
