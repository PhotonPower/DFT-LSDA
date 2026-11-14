// Hellmann-Feynman forces of the PAW total energy (Mortensen et al., PRB 71, 035109, Eq. (50) ff.):
//   F_a = - int [ v~ dn~c_a/dR + n~ dvbar_a/dR + v~_H sum_L Q_L dg_L^a/dR ] d^3r
//         - sum_n f_n sum_ij (dH_ij - eps_n dS_ij) d(P*_ni P_nj)/dR
#include <cmath>

#include "dft/calculation.hpp"
#include "dft/hamiltonian.hpp"
#include "math/special.hpp"

namespace lsda {

std::vector<Vec3> Calculation::forces() {
  ScopedTimer timer("forces");
  LSDA_CHECK(have_occ_, "forces: run scf() first");
  const int nat = c_.natoms(), ng = gs_.ng, nL = (lmax_comp_ + 1) * (lmax_comp_ + 1);
  std::vector<Vec3> pos(nat);
  for (int a = 0; a < nat; ++a) pos[a] = c_.atom_cart(a);

  // compensation multipoles
  std::vector<std::vector<double>> Q(nat);
  for (int a = 0; a < nat; ++a) {
    const PawSetup& s = c_.setup(a);
    RMatrix dt = D_[a][0];
    if (nspin_ == 2)
      for (size_t i = 0; i < dt.size(); ++i) dt.data()[i] += D_[a][1].data()[i];
    Q[a].resize(s.nL);
    s.multipoles(dt, Q[a].data());
  }

  // ---- local terms (dense G-sphere) ------------------------------------------------------
  std::vector<double> floc(3 * static_cast<size_t>(nat), 0.0);
#pragma omp parallel
  {
    std::vector<double> ylm(nL), t(nL * c_.species.size()), loc(3 * static_cast<size_t>(nat), 0.0);
#pragma omp for schedule(static) nowait
    for (int ig = 1; ig < ng; ++ig) {
      const double* g = &gs_.g[3 * ig];
      real_ylm(lmax_comp_, g, ylm.data());
      for (size_t is = 0; is < c_.species.size(); ++is) {
        const int nLs = c_.species[is].setup.nL;
        for (int L = 0; L < nLs; ++L) {
          const int l = static_cast<int>(std::floor(std::sqrt(static_cast<double>(L)) + 1e-9));
          t[is * nL + L] = ((l / 2) % 2 == 0 ? 1.0 : -1.0) * ylm[L] * ff_g_[is][static_cast<size_t>(l) * ng + ig];
        }
      }
      const cplx vavg = nspin_ == 2 ? 0.5 * (veff_G_[0][ig] + veff_G_[1][ig]) : veff_G_[0][ig];
      const cplx ntot = rho_[0][ig] + (nspin_ == 2 ? rho_[1][ig] : cplx(0.0)) + nct_G_[ig];
      const cplx cv = std::conj(vavg), cn = std::conj(ntot), ch = std::conj(vH_G_[ig]);
      for (int a = 0; a < nat; ++a) {
        const int is = c_.atoms[a].species;
        const int nLs = c_.species[is].setup.nL;
        const double* ta = &t[is * nL];
        double even = 0.0, odd = 0.0;
        for (int L = 0; L < nLs; ++L) {
          const int l = static_cast<int>(std::floor(std::sqrt(static_cast<double>(L)) + 1e-9));
          (l % 2 == 0 ? even : odd) += Q[a][L] * ta[L];
        }
        const double arg = -(g[0] * pos[a][0] + g[1] * pos[a][1] + g[2] * pos[a][2]);
        const cplx phase(std::cos(arg), std::sin(arg));
        const cplx z = cv * ff_nct_[is][ig] + cn * ff_vbar_[is][ig] + ch * (4 * pi * cplx(even, -odd));
        const double val = (phase * z).imag();
        for (int d = 0; d < 3; ++d) loc[3 * a + d] -= g[d] * val;
      }
    }
#pragma omp critical
    for (size_t i = 0; i < floc.size(); ++i) floc[i] += loc[i];
  }

  // ---- projector terms ---------------------------------------------------------------------
  std::vector<double> fnl(3 * static_cast<size_t>(nat), 0.0);
  const int nk = static_cast<int>(kpts_.size());
  for (int ik = 0; ik < nk; ++ik) {
    const PWBasis& b = basis_[ik];
    const Projectors proj = make_projectors(c_, b);
    for (int s = 0; s < nspin_; ++s) {
      const size_t ks = static_cast<size_t>(ik) * nspin_ + s;
      const CMatrix& x = psi_[ks];
      const CMatrix& P = proj_psi_[ks];
      CMatrix xd(b.npw, nbands_);
      for (int d = 0; d < 3; ++d) {
#pragma omp parallel for schedule(static)
        for (int n = 0; n < nbands_; ++n)
          for (int ig = 0; ig < b.npw; ++ig) xd(ig, n) = cplx(0.0, b.kg[3 * ig + d]) * x(ig, n);
        const CMatrix dP = overlap(proj.beta.data(), proj.nproj, xd.data(), nbands_, b.npw);
        for (int a = 0; a < nat; ++a) {
          const PawSetup& su = c_.setup(a);
          const int ni = su.ni, off = proj.offset[a];
          const RMatrix& dh = dH_[a][s];
          double sum = 0.0;
          for (int n = 0; n < nbands_; ++n) {
            const double w = occ_.f[ks][n] * b.weight;
            if (std::abs(w) < 1e-16) continue;
            const double e = eig_[ks][n];
            cplx acc = 0.0;
            for (int j = 0; j < ni; ++j) {
              cplx row = 0.0;
              for (int i = 0; i < ni; ++i) row += (dh(i, j) - e * su.dS(i, j)) * std::conj(P(off + i, n));
              acc += row * dP(off + j, n);
            }
            sum += w * acc.real();
          }
          fnl[3 * a + d] -= 2.0 * sum;
        }
      }
    }
  }

  // ---- total, symmetrized --------------------------------------------------------------------
  std::vector<Vec3> F(nat), Fs(nat, Vec3{0, 0, 0});
  for (int a = 0; a < nat; ++a)
    for (int d = 0; d < 3; ++d) F[a][d] = floc[3 * a + d] + fnl[3 * a + d];
  const int nops = static_cast<int>(ops_.size());
  for (int bq = 0; bq < nat; ++bq) {
    for (const SymOp& op : ops_) {
      const Vec3& f = F[op.atom_map[bq]];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) Fs[bq][i] += op.R[j][i] * f[j];  // R^T f
    }
    for (int i = 0; i < 3; ++i) Fs[bq][i] /= nops;
  }
  return Fs;
}

}  // namespace lsda
