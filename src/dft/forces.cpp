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

// =====================================================================================
//  Stress tensor
// =====================================================================================
// Homogeneous strain r -> (1 + eps) r at fixed plane-wave coefficients (Hellmann-Feynman): G -> (1 - eps) G,
// Omega -> Omega (1 + tr eps), structure-factor phases unchanged. With N(G) = Omega * n(G):
//   kinetic      : - sum_n f_n sum_G |c|^2 T'(q^2/2) q_a q_b   (T' = 1 without cutoff smoothing)
//   Hartree      : - delta_ab E_H + Omega/(4 pi) sum_G |v_H|^2 G_a G_b + sum_G Re[v_H^* dN_comp/d eps]
//   xc           :   delta_ab (E_xc - int v_xc n~)
//   zero pot.    : - delta_ab E_zero + sum_G Re[n~^* dVbar/d eps]
//   pseudo core  :   sum_G Re[v~^* dN_core/d eps]
//   projectors   :   sum_n f_n sum_ij (dH_ij - eps_n dS_ij) d(P*_ni P_nj)/d eps
// where the form-factor derivatives are d f(G)/d eps_ab = -G_b df/dG_a.
Mat3 Calculation::stress() {
  ScopedTimer timer("stress");
  LSDA_CHECK(have_occ_, "stress: run scf() first");
  const int nat = c_.natoms(), ng = gs_.ng, nL = (lmax_comp_ + 1) * (lmax_comp_ + 1);
  const int ca[6] = {0, 1, 2, 1, 0, 0}, cb[6] = {0, 1, 2, 2, 2, 1};  // xx yy zz yz xz xy
  double sig[6] = {0, 0, 0, 0, 0, 0};                                 // dF/d eps (energy units)
  std::vector<Vec3> pos(nat);
  for (int a = 0; a < nat; ++a) pos[a] = c_.atom_cart(a);

  // ---- kinetic and projector terms -------------------------------------------------------------
  const int nk = static_cast<int>(kpts_.size());
  double diag_proj = 0.0;
  for (int ik = 0; ik < nk; ++ik) {
    const PWBasis& b = basis_[ik];
    std::vector<int> offset(nat);
    for (int a = 0, off = 0; a < nat; ++a) {
      offset[a] = off;
      off += c_.setup(a).ni;
    }
    for (int s = 0; s < nspin_; ++s) {
      const size_t ks = static_cast<size_t>(ik) * nspin_ + s;
      const CMatrix& x = psi_[ks];
      double kin[6] = {0, 0, 0, 0, 0, 0};
#pragma omp parallel for schedule(static) reduction(+ : kin[:6])
      for (int n = 0; n < nbands_; ++n) {
        const double w = occ_.f[ks][n] * b.weight;
        if (std::abs(w) < 1e-16) continue;
        const cplx* xn = x.col(n);
        for (int ig = 0; ig < b.npw; ++ig) {
          const double c2 = w * std::norm(xn[ig]) * b.dkin[ig];
          const double* q = &b.kg[3 * ig];
          for (int c = 0; c < 6; ++c) kin[c] -= c2 * q[ca[c]] * q[cb[c]];
        }
      }
      for (int c = 0; c < 6; ++c) sig[c] += kin[c];
    }
    for (int c = 0; c < 6; ++c) {
      const CMatrix dbeta = make_projector_strain_derivative(c_, b, ca[c], cb[c]);
      const int nproj = dbeta.cols();
      for (int s = 0; s < nspin_; ++s) {
        const size_t ks = static_cast<size_t>(ik) * nspin_ + s;
        const CMatrix& P = proj_psi_[ks];
        const CMatrix X = overlap(dbeta.data(), nproj, psi_[ks].data(), nbands_, b.npw);
        double sum = 0.0, dsum = 0.0;
        for (int a = 0; a < nat; ++a) {
          const PawSetup& su = c_.setup(a);
          const int ni = su.ni, off = offset[a];
          const RMatrix& dh = dH_[a][s];
          for (int n = 0; n < nbands_; ++n) {
            const double w = occ_.f[ks][n] * b.weight;
            if (std::abs(w) < 1e-16) continue;
            const double e = eig_[ks][n];
            cplx acc = 0.0, dacc = 0.0;
            for (int j = 0; j < ni; ++j) {
              cplx row = 0.0;
              for (int i = 0; i < ni; ++i) row += (dh(i, j) - e * su.dS(i, j)) * std::conj(P(off + i, n));
              acc += row * X(off + j, n);
              dacc += row * P(off + j, n);
            }
            sum += w * acc.real();
            dsum += w * dacc.real();
          }
        }
        sig[c] += 2.0 * sum;
        if (c == 0) diag_proj -= dsum;  // volume factor of the projectors, -1/2 delta_ab * 2
      }
    }
  }

  // ---- local terms on the dense sphere -----------------------------------------------------------
  std::vector<std::vector<double>> Q(nat);
  for (int a = 0; a < nat; ++a) {
    const PawSetup& s = c_.setup(a);
    RMatrix dt = D_[a][0];
    if (nspin_ == 2)
      for (size_t i = 0; i < dt.size(); ++i) dt.data()[i] += D_[a][1].data()[i];
    Q[a].resize(s.nL);
    s.multipoles(dt, Q[a].data());
  }
  const size_t nsp = c_.species.size();
  double loc[6] = {0, 0, 0, 0, 0, 0};
#pragma omp parallel
  {
    std::vector<double> ylm(nL), grad(3 * nL), t(6 * nL * nsp), dn(nsp), dv(nsp);
#pragma omp for schedule(static) reduction(+ : loc[:6])
    for (int ig = 1; ig < ng; ++ig) {
      const double* g = &gs_.g[3 * ig];
      const double gn = std::sqrt(gs_.g2[ig]);
      real_ylm_grad(lmax_comp_, g, ylm.data(), grad.data());
      for (size_t is = 0; is < nsp; ++is) {
        const PawSetup& s = c_.species[is].setup;
        dn[is] = s.nct_q.deriv(gn);
        dv[is] = s.vbar_q.deriv(gn);
        for (int L = 0; L < s.nL; ++L) {
          const int l = static_cast<int>(std::floor(std::sqrt(static_cast<double>(L)) + 1e-9));
          const double sign = (l / 2) % 2 == 0 ? 1.0 : -1.0;
          const double gl = ff_g_[is][static_cast<size_t>(l) * ng + ig], dgl = s.g_q[l].deriv(gn);
          for (int c = 0; c < 6; ++c)
            t[(is * nL + L) * 6 + c] = -sign * (dgl * g[ca[c]] * g[cb[c]] / gn * ylm[L] +
                                                0.5 * gl * (g[cb[c]] * grad[3 * L + ca[c]] + g[ca[c]] * grad[3 * L + cb[c]]));
        }
      }
      const cplx vavg = nspin_ == 2 ? 0.5 * (veff_G_[0][ig] + veff_G_[1][ig]) : veff_G_[0][ig];
      const cplx ntot = rho_[0][ig] + (nspin_ == 2 ? rho_[1][ig] : cplx(0.0)) + nct_G_[ig];
      const cplx cv = std::conj(vavg), cn = std::conj(ntot), ch = std::conj(vH_G_[ig]);
      // Hartree term from 1/G^2
      const double h2 = c_.volume * std::norm(vH_G_[ig]) / (4 * pi);
      for (int c = 0; c < 6; ++c) loc[c] += h2 * g[ca[c]] * g[cb[c]];
      for (int a = 0; a < nat; ++a) {
        const int is = c_.atoms[a].species;
        const int nLs = c_.species[is].setup.nL;
        const double arg = -(g[0] * pos[a][0] + g[1] * pos[a][1] + g[2] * pos[a][2]);
        const cplx phase(std::cos(arg), std::sin(arg));
        const double core = (cv * phase).real() * dn[is] + (cn * phase).real() * dv[is];
        const cplx hp = ch * phase;
        for (int c = 0; c < 6; ++c) {
          double even = 0.0, odd = 0.0;
          for (int L = 0; L < nLs; ++L) {
            const int l = static_cast<int>(std::floor(std::sqrt(static_cast<double>(L)) + 1e-9));
            (l % 2 == 0 ? even : odd) += Q[a][L] * t[(is * nL + L) * 6 + c];
          }
          loc[c] += -core * g[ca[c]] * g[cb[c]] / gn + 4 * pi * (hp * cplx(even, -odd)).real();
        }
      }
    }
  }
  for (int c = 0; c < 6; ++c) sig[c] += loc[c];
  const double diag = -f_pot_.hartree - f_pot_.zero + f_pot_.xc - vxc_n_ + diag_proj;
  for (int c = 0; c < 3; ++c) sig[c] += diag;

  // ---- symmetrize and convert to energy per volume -------------------------------------------------
  Mat3 raw{}, out{};
  for (int c = 0; c < 6; ++c) raw[ca[c]][cb[c]] = raw[cb[c]][ca[c]] = sig[c] / c_.volume;
  for (const SymOp& op : ops_)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
          for (int l = 0; l < 3; ++l) out[i][j] += op.R[i][k] * raw[k][l] * op.R[j][l];
  for (auto& row : out)
    for (double& v : row) v /= static_cast<double>(ops_.size());
  return out;
}

}  // namespace lsda
