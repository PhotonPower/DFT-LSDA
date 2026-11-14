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

void Hamiltonian::apply_local(const cplx* x, int nb, cplx* kx) const {
  const int npw = pw_.npw;
  const size_t ngrid = fft_.size();
  const double inv_n = 1.0 / static_cast<double>(ngrid);
  n_applications += nb;
  const double t0 = wall_time();
#pragma omp parallel
  {
    CVec work(ngrid);
#pragma omp for schedule(dynamic, 1)
    for (int ib = 0; ib < nb; ++ib) {
      const cplx* xb = x + static_cast<size_t>(ib) * npw;
      cplx* hb = kx + static_cast<size_t>(ib) * npw;
      std::fill(work.begin(), work.end(), cplx(0.0));
      for (int ig = 0; ig < npw; ++ig) work[pw_.idx[ig]] = xb[ig];
      fft_.backward(work.data());
      for (size_t i = 0; i < ngrid; ++i) work[i] *= vloc_[i];
      fft_.forward(work.data());
      for (int ig = 0; ig < npw; ++ig) hb[ig] = pw_.kin[ig] * xb[ig] + inv_n * work[pw_.idx[ig]];
    }
  }
  add_time("  H: local part (FFT)", wall_time() - t0);
}

CMatrix Hamiltonian::project(const cplx* x, int nb) const {
  const double t0 = wall_time();
  CMatrix P = overlap(proj_.beta.data(), proj_.nproj, x, nb, pw_.npw);
  add_time("  H: projections", wall_time() - t0);
  return P;
}

void Hamiltonian::block_multiply(const CMatrix& P, CMatrix* DP, CMatrix* SP) const {
  const int nb = P.cols(), nat = static_cast<int>(dH_.size());
  if (DP) DP->resize(proj_.nproj, nb);
  if (SP) SP->resize(proj_.nproj, nb);
#pragma omp parallel for schedule(static)
  for (int ib = 0; ib < nb; ++ib)
    for (int a = 0; a < nat; ++a) {
      const RMatrix& dh = *dH_[a];
      const RMatrix& ds = *dS_[a];
      const int ni = dh.rows(), off = proj_.offset[a];
      const cplx* p = P.col(ib) + off;
      for (int i = 0; i < ni; ++i) {
        cplx sh = 0.0, ss = 0.0;
        for (int j = 0; j < ni; ++j) {
          sh += dh(i, j) * p[j];
          ss += ds(i, j) * p[j];
        }
        if (DP) (*DP)(off + i, ib) = sh;
        if (SP) (*SP)(off + i, ib) = ss;
      }
    }
}

void Hamiltonian::apply(const cplx* x, int nb, cplx* hx, cplx* sx) const {
  const int npw = pw_.npw, nproj = proj_.nproj;
  apply_local(x, nb, hx);
  std::copy(x, x + static_cast<size_t>(nb) * npw, sx);
  if (nproj == 0) return;
  const CMatrix P = project(x, nb);
  CMatrix HP, SP;
  block_multiply(P, &HP, &SP);
  const double t0 = wall_time();
  gemm('N', 'N', npw, nb, nproj, 1.0, proj_.beta.data(), npw, HP.data(), nproj, 1.0, hx, npw);
  gemm('N', 'N', npw, nb, nproj, 1.0, proj_.beta.data(), npw, SP.data(), nproj, 1.0, sx, npw);
  add_time("  H: nonlocal update", wall_time() - t0);
}

}  // namespace lsda
