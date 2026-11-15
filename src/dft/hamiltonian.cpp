#include "dft/hamiltonian.hpp"

#include <cmath>

#include "math/special.hpp"

namespace lsda {

Projectors make_projectors(const Crystal& c, const PWBasis& pw) {
  Projectors p;
  const int nat = c.natoms(), npw = pw.npw;
  p.offset.resize(nat);
  for (int a = 0; a < nat; ++a) {
    p.offset[a] = p.nproj;
    p.nproj += c.setup(a).ni;
  }
  p.beta.resize(npw, p.nproj);
  const double pref = 4 * pi / std::sqrt(c.volume);
  const cplx il[4] = {cplx(1, 0), cplx(0, -1), cplx(-1, 0), cplx(0, 1)};  // (-i)^l
  for (size_t is = 0; is < c.species.size(); ++is) {
    const PawSetup& s = c.species[is].setup;
    // form factors without the structure phase
    CMatrix form(npw, s.ni);
#pragma omp parallel for schedule(static)
    for (int ig = 0; ig < npw; ++ig) {
      const double* q = &pw.kg[3 * ig];
      const double qn = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
      double ylm[49], rad[16];
      real_ylm(s.lmax, q, ylm);
      for (int j = 0; j < s.nj; ++j) rad[j] = s.proj_q[j](qn);
      for (int i = 0; i < s.ni; ++i) form(ig, i) = pref * il[s.l_i[i] % 4] * ylm[s.L_i[i]] * rad[s.j_i[i]];
    }
    for (int a = 0; a < nat; ++a) {
      if (c.atoms[a].species != static_cast<int>(is)) continue;
      const Vec3 r = c.atom_cart(a);
#pragma omp parallel for schedule(static)
      for (int ig = 0; ig < npw; ++ig) {
        const double* q = &pw.kg[3 * ig];
        const double arg = -(q[0] * r[0] + q[1] * r[1] + q[2] * r[2]);
        const cplx phase(std::cos(arg), std::sin(arg));
        for (int i = 0; i < s.ni; ++i) p.beta(ig, p.offset[a] + i) = form(ig, i) * phase;
      }
    }
  }
  return p;
}

CMatrix make_projector_strain_derivative(const Crystal& c, const PWBasis& pw, int a, int b) {
  const int nat = c.natoms(), npw = pw.npw;
  int nproj = 0;
  std::vector<int> offset(nat);
  for (int at = 0; at < nat; ++at) {
    offset[at] = nproj;
    nproj += c.setup(at).ni;
  }
  CMatrix d(npw, nproj);
  const double pref = 4 * pi / std::sqrt(c.volume);
  const cplx il[4] = {cplx(1, 0), cplx(0, -1), cplx(-1, 0), cplx(0, 1)};
  for (size_t is = 0; is < c.species.size(); ++is) {
    const PawSetup& s = c.species[is].setup;
    CMatrix form(npw, s.ni);
#pragma omp parallel for schedule(static)
    for (int ig = 0; ig < npw; ++ig) {
      const double* q = &pw.kg[3 * ig];
      const double qn = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
      double ylm[49], grad[147], rad[16], drad[16];
      real_ylm_grad(s.lmax, q, ylm, grad);
      for (int j = 0; j < s.nj; ++j) {
        rad[j] = s.proj_q[j](qn);
        drad[j] = s.proj_q[j].deriv(qn);
      }
      for (int i = 0; i < s.ni; ++i) {
        const int L = s.L_i[i], j = s.j_i[i];
        // d/dq_a [Y(q^) p(q)] = p'(q) q_a/q Y + p(q) dY/dq_a ; the q -> 0 limit of q_b d/dq_a(...) vanishes
        double val = 0.0;
        if (qn > 1e-12)
          val = -(drad[j] * q[a] * q[b] / qn * ylm[L] + 0.5 * rad[j] * (q[b] * grad[3 * L + a] + q[a] * grad[3 * L + b]));
        form(ig, i) = pref * il[s.l_i[i] % 4] * val;
      }
    }
    for (int at = 0; at < nat; ++at) {
      if (c.atoms[at].species != static_cast<int>(is)) continue;
      const Vec3 r = c.atom_cart(at);
#pragma omp parallel for schedule(static)
      for (int ig = 0; ig < npw; ++ig) {
        const double* q = &pw.kg[3 * ig];
        const double arg = -(q[0] * r[0] + q[1] * r[1] + q[2] * r[2]);
        const cplx phase(std::cos(arg), std::sin(arg));
        for (int i = 0; i < s.ni; ++i) d(ig, offset[at] + i) = form(ig, i) * phase;
      }
    }
  }
  return d;
}

void Hamiltonian::apply_local(const cplx* x, int nb, cplx* kx) const {
  const int npw = pw_.npw, dim = this->npw();
  const size_t ngrid = fft_.size();
  const double inv_n = 1.0 / static_cast<double>(ngrid);
  n_applications += nb;
  const double t0 = wall_time();
#pragma omp parallel
  {
    CVec work(ngrid), work2(spinor_ ? ngrid : 0);
#pragma omp for schedule(dynamic, 1)
    for (int ib = 0; ib < nb; ++ib) {
      const cplx* xb = x + static_cast<size_t>(ib) * dim;
      cplx* hb = kx + static_cast<size_t>(ib) * dim;
      std::fill(work.begin(), work.end(), cplx(0.0));
      for (int ig = 0; ig < npw; ++ig) work[pw_.idx[ig]] = xb[ig];
      fft_.backward_pruned(work.data(), pw_.columns, pw_.planes);
      if (!spinor_) {
        for (size_t i = 0; i < ngrid; ++i) work[i] *= vloc_[0][i];
        fft_.forward_pruned(work.data(), pw_.columns, pw_.planes);
        for (int ig = 0; ig < npw; ++ig) hb[ig] = pw_.kin[ig] * xb[ig] + inv_n * work[pw_.idx[ig]];
        if (vtau_)  // sum_d (k+G)_d FFT[v_tau FFT^-1[(k+G)_d x]] / 2
          for (int d = 0; d < 3; ++d) {
            std::fill(work.begin(), work.end(), cplx(0.0));
            for (int ig = 0; ig < npw; ++ig) work[pw_.idx[ig]] = pw_.kg[3 * ig + d] * xb[ig];
            fft_.backward_pruned(work.data(), pw_.columns, pw_.planes);
            for (size_t i = 0; i < ngrid; ++i) work[i] *= vtau_[i];
            fft_.forward_pruned(work.data(), pw_.columns, pw_.planes);
            for (int ig = 0; ig < npw; ++ig) hb[ig] += 0.5 * inv_n * pw_.kg[3 * ig + d] * work[pw_.idx[ig]];
          }
        continue;
      }
      // spinor: (V psi)_up = (v + Bz) psi_up + (Bx - i By) psi_dn,  (V psi)_dn = (Bx + i By) psi_up + (v - Bz) psi_dn
      std::fill(work2.begin(), work2.end(), cplx(0.0));
      for (int ig = 0; ig < npw; ++ig) work2[pw_.idx[ig]] = xb[npw + ig];
      fft_.backward_pruned(work2.data(), pw_.columns, pw_.planes);
      const double *v0 = vloc_[0], *bx = vloc_[1], *by = vloc_[2], *bz = vloc_[3];
      for (size_t i = 0; i < ngrid; ++i) {
        const cplx u = work[i], d = work2[i];
        work[i] = (v0[i] + bz[i]) * u + cplx(bx[i], -by[i]) * d;
        work2[i] = cplx(bx[i], by[i]) * u + (v0[i] - bz[i]) * d;
      }
      fft_.forward_pruned(work.data(), pw_.columns, pw_.planes);
      fft_.forward_pruned(work2.data(), pw_.columns, pw_.planes);
      for (int ig = 0; ig < npw; ++ig) {
        hb[ig] = pw_.kin[ig] * xb[ig] + inv_n * work[pw_.idx[ig]];
        hb[npw + ig] = pw_.kin[ig] * xb[npw + ig] + inv_n * work2[pw_.idx[ig]];
      }
    }
  }
  add_time("  H: local part (FFT)", wall_time() - t0);
  if (!xi_.empty()) {
    const double t1 = wall_time();
    for (const auto& [xi, coef] : xi_) {
      if (xi->cols() == 0 || coef == 0.0) continue;
      const CMatrix c = overlap(xi->data(), xi->cols(), x, nb, dim);
      gemm('N', 'N', dim, nb, xi->cols(), -coef, xi->data(), dim, c.data(), xi->cols(), 1.0, kx, dim);
    }
    add_time("  H: exact exchange (ACE)", wall_time() - t1);
  }
}

CMatrix Hamiltonian::project(const cplx* x, int nb) const {
  const double t0 = wall_time();
  const int npw = pw_.npw, np = proj_.nproj, dim = this->npw();
  CMatrix P(nproj(), nb);
  if (np > 0) {
    // the projectors act on each spin component separately
    gemm('C', 'N', np, nb, npw, 1.0, proj_.beta.data(), npw, x, dim, 0.0, P.data(), nproj());
    if (spinor_) gemm('C', 'N', np, nb, npw, 1.0, proj_.beta.data(), npw, x + npw, dim, 0.0, P.data() + np, nproj());
  }
  add_time("  H: projections", wall_time() - t0);
  return P;
}

void Hamiltonian::add_nonlocal(const CMatrix& T, cplx* r, int nb) const {
  const int npw = pw_.npw, np = proj_.nproj, dim = this->npw();
  if (np == 0) return;
  gemm('N', 'N', npw, nb, np, 1.0, proj_.beta.data(), npw, T.data(), nproj(), 1.0, r, dim);
  if (spinor_) gemm('N', 'N', npw, nb, np, 1.0, proj_.beta.data(), npw, T.data() + np, nproj(), 1.0, r + npw, dim);
}

void Hamiltonian::block_multiply(const CMatrix& P, CMatrix* DP, CMatrix* SP) const {
  const int nb = P.cols(), nat = static_cast<int>(dH_.size()), np = proj_.nproj;
  if (DP) DP->resize(nproj(), nb);
  if (SP) SP->resize(nproj(), nb);
#pragma omp parallel for schedule(static)
  for (int ib = 0; ib < nb; ++ib)
    for (int a = 0; a < nat; ++a) {
      const RMatrix& ds = *dS_[a];
      const int ni = ds.rows(), off = proj_.offset[a];
      const cplx* p = P.col(ib) + off;
      if (!spinor_) {
        const RMatrix& dh = *dH_[a][0];
        for (int i = 0; i < ni; ++i) {
          cplx sh = 0.0, ss = 0.0;
          for (int j = 0; j < ni; ++j) {
            sh += dh(i, j) * p[j];
            ss += ds(i, j) * p[j];
          }
          if (DP) (*DP)(off + i, ib) = sh;
          if (SP) (*SP)(off + i, ib) = ss;
        }
        continue;
      }
      const cplx* q = p + np;  // spin-down projections
      const RMatrix &h0 = *dH_[a][0], &hx = *dH_[a][1], &hy = *dH_[a][2], &hz = *dH_[a][3];
      for (int i = 0; i < ni; ++i) {
        cplx su = 0.0, sd = 0.0, ou = 0.0, od = 0.0;
        for (int j = 0; j < ni; ++j) {
          const cplx hud(hx(i, j), -hy(i, j)), hdu(hx(i, j), hy(i, j));
          su += (h0(i, j) + hz(i, j)) * p[j] + hud * q[j];
          sd += hdu * p[j] + (h0(i, j) - hz(i, j)) * q[j];
          ou += ds(i, j) * p[j];
          od += ds(i, j) * q[j];
        }
        if (DP) {
          (*DP)(off + i, ib) = su;
          (*DP)(np + off + i, ib) = sd;
        }
        if (SP) {
          (*SP)(off + i, ib) = ou;
          (*SP)(np + off + i, ib) = od;
        }
      }
    }
}

void Hamiltonian::apply(const cplx* x, int nb, cplx* hx, cplx* sx) const {
  const int dim = this->npw();
  apply_local(x, nb, hx);
  std::copy(x, x + static_cast<size_t>(nb) * dim, sx);
  if (proj_.nproj == 0) return;
  const CMatrix P = project(x, nb);
  CMatrix HP, SP;
  block_multiply(P, &HP, &SP);
  const double t0 = wall_time();
  add_nonlocal(HP, hx, nb);
  add_nonlocal(SP, sx, nb);
  add_time("  H: nonlocal update", wall_time() - t0);
}

}  // namespace lsda
