// Exact (Fock) exchange for hybrid functionals in the PAW method.
//
// Energy (per spin channel s, occupations f in [0, 1], k-point weights w):
//   E_x = -a/2 sum_{nk,mq} w_k w_q f_nk f_mq (rho~_mn | v | rho~_mn) + sum_a dE_x^a(D)
//   rho~_mn(r) = psi~*_mq(r) psi~_nk(r) + sum_aL Q_aL[m,n] g_aL(r),   Q_aL[m,n] = sum_ij Delta_Lij <psi~_m|p_i><p_j|psi~_n>
// The pair densities carry compensation charges so that their multipole moments are those of the all-electron
// pair densities; the atom-centred remainder dE_x^a depends on the density matrices only and is part of
// PawSetup::onsite. Operator acting on psi~_nk:
//   V_x psi~_n = -a sum_mq w_q f_mq [ v~_mn(r) psi~_m(r) + sum_a sum_ij |p_i> (sum_L Delta_Lij int g_aL v~_mn) <p_j|psi~_m> ]
// with v~_mn = v * rho~_mn. Kernel v(G): full Coulomb (spherically truncated or auxiliary-function treatment of the
// G -> 0 singularity) or the short-range kernel 4 pi/G^2 (1 - exp(-G^2/4 omega^2)).
//
// The operator is not applied inside the eigensolver. Instead W = V_x psi is computed once per outer iteration for
// all bands and replaced by the adaptively compressed exchange operator (Lin, JCTC 12, 2242 (2016))
//   V_ACE = -xi xi^H,  xi = W U lambda^-1/2,  U lambda U^H = -psi^H W,
// which agrees with V_x on the span of the orbitals it was built from.
#include <algorithm>
#include <cmath>

#include "base/parallel.hpp"
#include "dft/calculation.hpp"
#include "dft/hamiltonian.hpp"

namespace lsda {

namespace {
inline int l_of_L(int L) { return static_cast<int>(std::floor(std::sqrt(static_cast<double>(L)) + 1e-9)); }
const cplx minus_i_pow[4] = {cplx(1, 0), cplx(0, -1), cplx(-1, 0), cplx(0, 1)};  // (-i)^l
}  // namespace

// Treatment of the Coulomb singularity and the kernel parameters
void Calculation::setup_exchange() {
  exx_fraction_ = xc_->exx_fraction();
  exx_omega_ = xc_->exx_screening();
  const double nk_total = static_cast<double>(p_.kgrid[0]) * p_.kgrid[1] * p_.kgrid[2];
  std::string mode = p_.exx_singularity;
  // radius of the sphere with the volume of the Born-von-Karman supercell and of its inscribed sphere
  exx_rc_ = std::cbrt(3.0 * nk_total * c_.volume / (4.0 * pi));
  double r_in = 1e300;
  for (int i = 0; i < 3; ++i) r_in = std::min(r_in, pi * p_.kgrid[i] / norm(Vec3{c_.B[i][0], c_.B[i][1], c_.B[i][2]}));
  if (mode == "auto") mode = exx_rc_ <= 1.3 * r_in ? "truncated" : "auxiliary";
  LSDA_CHECK(mode == "truncated" || mode == "auxiliary", "exx_singularity must be auto, truncated or auxiliary");
  exx_truncated_ = mode == "truncated";
  exx_v0_ = 0.0;
  if (exx_omega_ == 0.0 && !exx_truncated_) {
    // auxiliary function exp(-alpha p^2)/p^2 (Gygi-Baldereschi type): the G = 0 term is replaced by the difference
    // between the exact integral and the discrete sum over the q-grid, plus the limit of 4 pi (1 - exp(-alpha p^2))/p^2
    const double alpha = 10.0 / (2.0 * p_.ecut);
    double sum = 0.0;
    for (int i = 0; i < p_.kgrid[0]; ++i)
      for (int j = 0; j < p_.kgrid[1]; ++j)
        for (int k = 0; k < p_.kgrid[2]; ++k) {
          const Vec3 qf{static_cast<double>(i) / p_.kgrid[0], static_cast<double>(j) / p_.kgrid[1], static_cast<double>(k) / p_.kgrid[2]};
          Vec3 q{0, 0, 0};
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) q[b] += qf[a] * c_.B[a][b];
          for (int ig = 0; ig < gs_.ng; ++ig) {
            const double px = q[0] + gs_.g[3 * ig], py = q[1] + gs_.g[3 * ig + 1], pz = q[2] + gs_.g[3 * ig + 2];
            const double p2 = px * px + py * py + pz * pz;
            if (p2 > 1e-12) sum += 4 * pi * std::exp(-alpha * p2) / p2;
          }
        }
    exx_v0_ = nk_total * c_.volume / std::sqrt(pi * alpha) - sum + 4 * pi * alpha;
  }
  if (p_.verbosity > 0) {
    if (exx_omega_ > 0.0)
      logf("  Exact exchange         : fraction %.4f, short-range kernel erfc(omega r)/r with omega = %.4f/bohr\n", exx_fraction_, exx_omega_);
    else if (exx_truncated_)
      logf("  Exact exchange         : fraction %.4f, Coulomb kernel truncated at R_c = %.3f bohr\n", exx_fraction_, exx_rc_);
    else
      logf("  Exact exchange         : fraction %.4f, Coulomb kernel with auxiliary-function correction (v(0) = %.4f)\n", exx_fraction_, exx_v0_);
  }
}

double Calculation::exchange_kernel(double p2) const {
  if (exx_omega_ > 0.0) {
    const double t = p2 / (4.0 * exx_omega_ * exx_omega_);
    return t < 1e-8 ? pi / (exx_omega_ * exx_omega_) * (1.0 - 0.5 * t) : -4 * pi / p2 * std::expm1(-t);
  }
  if (exx_truncated_) {
    const double x = std::sqrt(p2) * exx_rc_;
    return x < 1e-4 ? 2 * pi * exx_rc_ * exx_rc_ * (1.0 - x * x / 12.0) : 4 * pi / p2 * (1.0 - std::cos(x));
  }
  return p2 < 1e-12 ? exx_v0_ : 4 * pi / p2;
}

// sum_k w_k sum_n f_n <psi_n|V|psi_n> for the current orbitals and a compressed operator V = -xi xi^H
double Calculation::exchange_cross_energy(const std::vector<CMatrix>& ace) const {
  double e = 0.0;
  const int nk = static_cast<int>(kpts_.size());
  for (int ik = 0; ik < nk; ++ik) {
    if (!par::owns(ik)) continue;
    for (int s = 0; s < nspin_; ++s) {
      const size_t ks = static_cast<size_t>(ik) * nspin_ + s;
      const CMatrix& xi = ace[ks];
      if (xi.cols() == 0) continue;
      const CMatrix c = overlap(xi.data(), xi.cols(), psi_[ks].data(), nbands_, basis_[ik].npw);
      for (int n = 0; n < nbands_; ++n) {
        double sum = 0.0;
        for (int j = 0; j < xi.cols(); ++j) sum += std::norm(c(j, n));
        e -= basis_[ik].weight * occ_.f[ks][n] * sum;
      }
    }
  }
  par::sum(&e, 1);
  return e;
}

// One pass over all pairs of orbitals. Returns the plane-wave part of the exchange energy for the current
// wavefunctions and occupations. build: construct the compressed operators (ace_). forces: add the forces of
// this energy term (explicit dependence on the atomic positions through projectors and compensation charges).
double Calculation::exchange_pass(bool build, std::vector<double>* forces) {
  ScopedTimer timer(build ? "exact exchange (operator)" : "exact exchange (forces)");
  LSDA_CHECK(have_occ_, "exchange: no occupations");
  const int nk = static_cast<int>(kpts_.size()), nat = c_.natoms(), ng = gs_.ng, nsp = static_cast<int>(c_.species.size());
  const size_t nc = fft_c_.size();
  const double vol = c_.volume, deg = nspin_ == 1 ? 2.0 : 1.0, alpha = exx_fraction_, inv_nc = 1.0 / static_cast<double>(nc);
  const double f_min = 1e-10;
  const int nLmax = (lmax_comp_ + 1) * (lmax_comp_ + 1);

  // ---- occupied orbitals and their projections at all k-points, replicated on all processes -------------
  std::vector<std::vector<int>> occ_idx(psi_.size());
  std::vector<CMatrix> xocc(psi_.size()), pocc(psi_.size());
  for (int ik = 0; ik < nk; ++ik) {
    const PWBasis& b = basis_[ik];
    Projectors proj;
    if (par::owns(ik)) proj = make_projectors(c_, b);
    for (int s = 0; s < nspin_; ++s) {
      const size_t ks = static_cast<size_t>(ik) * nspin_ + s;
      for (int n = 0; n < nbands_; ++n)
        if (occ_.f[ks][n] / deg > f_min) occ_idx[ks].push_back(n);
      const int no = static_cast<int>(occ_idx[ks].size());
      xocc[ks].resize(b.npw, no);
      int nproj = 0;
      for (int a = 0; a < nat; ++a) nproj += c_.setup(a).ni;
      pocc[ks].resize(nproj, no);
      if (par::owns(ik)) {
        for (int m = 0; m < no; ++m) std::copy(psi_[ks].col(occ_idx[ks][m]), psi_[ks].col(occ_idx[ks][m]) + b.npw, xocc[ks].col(m));
        if (no > 0 && nproj > 0) pocc[ks] = overlap(proj.beta.data(), nproj, xocc[ks].data(), no, b.npw);
      }
      if (par::size() > 1) {
        par::sum(xocc[ks].data(), xocc[ks].size());
        par::sum(pocc[ks].data(), pocc[ks].size());
      }
    }
  }
  std::vector<int> offset(nat);
  int nproj = 0;
  for (int a = 0; a < nat; ++a) {
    offset[a] = nproj;
    nproj += c_.setup(a).ni;
  }
  std::vector<Vec3> pos(nat);
  for (int a = 0; a < nat; ++a) pos[a] = c_.atom_cart(a);

  if (build) ace_.assign(psi_.size(), CMatrix());
  std::vector<double> fx(3 * static_cast<size_t>(nat), 0.0);
  double energy = 0.0;

  for (int ik = 0; ik < nk; ++ik) {
    if (!par::owns(ik)) continue;
    const PWBasis& bk = basis_[ik];
    const Projectors proj = make_projectors(c_, bk);
    std::vector<CMatrix> Pk(nspin_), W(nspin_), Cnl(nspin_);
    std::vector<std::array<CMatrix, 3>> dPk(nspin_);
    for (int s = 0; s < nspin_; ++s) {
      const size_t ks = static_cast<size_t>(ik) * nspin_ + s;
      Pk[s] = overlap(proj.beta.data(), nproj, psi_[ks].data(), nbands_, bk.npw);
      if (build) {
        W[s].resize(bk.npw, nbands_);
        Cnl[s].resize(nproj, nbands_);
      }
      if (forces) {  // projections of the derivative with respect to the atomic position: <dp/dR|psi>
        CMatrix xd(bk.npw, nbands_);
        for (int d = 0; d < 3; ++d) {
          for (int n = 0; n < nbands_; ++n)
            for (int ig = 0; ig < bk.npw; ++ig) xd(ig, n) = cplx(0.0, bk.kg[3 * ig + d]) * psi_[ks](ig, n);
          dPk[s][d] = overlap(proj.beta.data(), nproj, xd.data(), nbands_, bk.npw);
        }
      }
    }

    for (int iq = 0; iq < nk; ++iq) {
      const PWBasis& bq = basis_[iq];
      // ---- tables for the momentum transfers p = k - q + G ---------------------------------------------
      const Vec3 dk{bk.kcart[0] - bq.kcart[0], bk.kcart[1] - bq.kcart[1], bk.kcart[2] - bq.kcart[2]};
      std::vector<double> kern(ng), pvec(3 * static_cast<size_t>(ng));
      std::vector<std::vector<double>> T(nsp, std::vector<double>(static_cast<size_t>(nLmax) * ng, 0.0));  // 4 pi/Omega Y_L g_l
      std::vector<std::vector<cplx>> phase(nat, std::vector<cplx>(ng));
#pragma omp parallel
      {
        std::vector<double> ylm(nLmax);
#pragma omp for schedule(static)
        for (int ig = 0; ig < ng; ++ig) {
          const double p[3] = {dk[0] + gs_.g[3 * ig], dk[1] + gs_.g[3 * ig + 1], dk[2] + gs_.g[3 * ig + 2]};
          const double p2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2], pn = std::sqrt(p2);
          for (int d = 0; d < 3; ++d) pvec[3 * static_cast<size_t>(ig) + d] = p[d];
          kern[ig] = exchange_kernel(p2);
          real_ylm(lmax_comp_, p, ylm.data());
          for (int is = 0; is < nsp; ++is) {
            const PawSetup& su = c_.species[is].setup;
            double gl[16];
            for (int l = 0; l <= 2 * su.lmax; ++l) gl[l] = pn <= su.g_q[l].qmax() ? su.g_q[l](pn) : 0.0;
            for (int L = 0; L < su.nL; ++L) T[is][static_cast<size_t>(L) * ng + ig] = 4 * pi / vol * ylm[L] * gl[l_of_L(L)];
          }
          for (int a = 0; a < nat; ++a) {
            const double arg = -(p[0] * pos[a][0] + p[1] * pos[a][1] + p[2] * pos[a][2]);
            phase[a][ig] = cplx(std::cos(arg), std::sin(arg));
          }
        }
      }

      for (int s = 0; s < nspin_; ++s) {
        const size_t ks = static_cast<size_t>(ik) * nspin_ + s, qs = static_cast<size_t>(iq) * nspin_ + s;
        const int no = static_cast<int>(occ_idx[qs].size());
        if (no == 0) continue;
        // occupied orbitals at q on the real-space grid
        std::vector<CVec> uq(no, CVec(nc));
#pragma omp parallel for schedule(dynamic, 1)
        for (int m = 0; m < no; ++m) {
          std::fill(uq[m].begin(), uq[m].end(), cplx(0.0));
          const cplx* x = xocc[qs].col(m);
          for (int ig = 0; ig < bq.npw; ++ig) uq[m][bq.idx[ig]] = x[ig];
          fft_c_.backward_pruned(uq[m].data(), bq.columns, bq.planes);
        }
        const CMatrix& Pq = pocc[qs];
        const CMatrix& Pn = Pk[s];
        double e_sum = 0.0;
#pragma omp parallel reduction(+ : e_sum)
        {
          CVec un(nc), work(nc), wr(nc), rho(ng), vg(ng), tmp(ng);
          std::vector<cplx> Q(static_cast<size_t>(nat) * nLmax), V(static_cast<size_t>(nat) * nLmax), cnl(nproj);
          std::vector<double> floc(3 * static_cast<size_t>(nat), 0.0);
#pragma omp for schedule(dynamic, 1)
          for (int n = 0; n < nbands_; ++n) {
            const double fn = occ_.f[ks][n];  // both spin channels for nspin = 1
            if (!build && fn / deg <= f_min) continue;
            std::fill(un.begin(), un.end(), cplx(0.0));
            const cplx* x = psi_[ks].col(n);
            for (int ig = 0; ig < bk.npw; ++ig) un[bk.idx[ig]] = x[ig];
            fft_c_.backward_pruned(un.data(), bk.columns, bk.planes);
            if (build) {
              std::fill(wr.begin(), wr.end(), cplx(0.0));
              std::fill(cnl.begin(), cnl.end(), cplx(0.0));
            }
            for (int m = 0; m < no; ++m) {
              const double fm = bq.weight * occ_.f[qs][occ_idx[qs][m]] / deg;
              // pair density: smooth part ...
              const double inv = inv_nc / vol;
              for (size_t i = 0; i < nc; ++i) work[i] = std::conj(uq[m][i]) * un[i];
              fft_c_.forward(work.data());
              for (int ig = 0; ig < ng; ++ig) rho[ig] = gs_.idx_coarse[ig] >= 0 ? work[gs_.idx_coarse[ig]] * inv : cplx(0.0);
              // ... plus compensation charges
              for (int a = 0; a < nat; ++a) {
                const PawSetup& su = c_.setup(a);
                const int is = c_.atoms[a].species, ni = su.ni, off = offset[a];
                cplx* Qa = &Q[static_cast<size_t>(a) * nLmax];
                for (int L = 0; L < su.nL; ++L) {
                  const double* dl = &su.Delta_Lij[static_cast<size_t>(L) * ni * ni];
                  cplx q = 0.0;
                  for (int j = 0; j < ni; ++j) {
                    cplx t = 0.0;
                    for (int i = 0; i < ni; ++i) t += dl[i * ni + j] * std::conj(Pq(off + i, m));
                    q += t * Pn(off + j, n);
                  }
                  Qa[L] = q;
                }
                std::fill(tmp.begin(), tmp.end(), cplx(0.0));
                for (int L = 0; L < su.nL; ++L) {
                  const cplx c = minus_i_pow[l_of_L(L) % 4] * Qa[L];
                  if (c == cplx(0.0)) continue;
                  const double* t = &T[is][static_cast<size_t>(L) * ng];
                  for (int ig = 0; ig < ng; ++ig) tmp[ig] += c * t[ig];
                }
                const cplx* ph = phase[a].data();
                for (int ig = 0; ig < ng; ++ig) rho[ig] += ph[ig] * tmp[ig];
              }
              double e_pair = 0.0;
              for (int ig = 0; ig < ng; ++ig) {
                vg[ig] = kern[ig] * rho[ig];
                e_pair += kern[ig] * std::norm(rho[ig]);
              }
              e_pair *= vol;
              e_sum += -0.5 * alpha * bk.weight * fn * fm * e_pair;
              // V[a][L] = int g_aL(r) v~(r) d^3r
              for (int a = 0; a < nat; ++a) {
                const PawSetup& su = c_.setup(a);
                const int is = c_.atoms[a].species;
                const cplx* ph = phase[a].data();
                for (int ig = 0; ig < ng; ++ig) tmp[ig] = std::conj(ph[ig]) * vg[ig];
                for (int L = 0; L < su.nL; ++L) {
                  const double* t = &T[is][static_cast<size_t>(L) * ng];
                  cplx sum = 0.0;
                  for (int ig = 0; ig < ng; ++ig) sum += t[ig] * tmp[ig];
                  V[static_cast<size_t>(a) * nLmax + L] = vol * std::conj(minus_i_pow[l_of_L(L) % 4]) * sum;
                }
              }
              if (build) {
                std::fill(work.begin(), work.end(), cplx(0.0));
                for (int ig = 0; ig < ng; ++ig)
                  if (gs_.idx_coarse[ig] >= 0) work[gs_.idx_coarse[ig]] = vg[ig];
                fft_c_.backward(work.data());
                for (size_t i = 0; i < nc; ++i) wr[i] += fm * work[i] * uq[m][i];
                for (int a = 0; a < nat; ++a) {
                  const PawSetup& su = c_.setup(a);
                  const int ni = su.ni, off = offset[a];
                  for (int L = 0; L < su.nL; ++L) {
                    const cplx v = fm * V[static_cast<size_t>(a) * nLmax + L];
                    if (v == cplx(0.0)) continue;
                    const double* dl = &su.Delta_Lij[static_cast<size_t>(L) * ni * ni];
                    for (int j = 0; j < ni; ++j) {
                      const cplx vp = v * Pq(off + j, m);
                      for (int i = 0; i < ni; ++i) cnl[off + i] += dl[i * ni + j] * vp;
                    }
                  }
                }
              }
              if (forces && fn / deg > f_min) {
                const double pref = alpha * bk.weight * fn * fm;
                for (int a = 0; a < nat; ++a) {
                  const PawSetup& su = c_.setup(a);
                  const int is = c_.atoms[a].species, ni = su.ni, off = offset[a];
                  const cplx* Qa = &Q[static_cast<size_t>(a) * nLmax];
                  // (i) projector positions: 2 Re sum_L conj(V_L) sum_ij Delta_Lij conj(P_m,i) dP_n,j
                  for (int d = 0; d < 3; ++d) {
                    const CMatrix& dP = dPk[s][d];
                    cplx sum = 0.0;
                    for (int L = 0; L < su.nL; ++L) {
                      const double* dl = &su.Delta_Lij[static_cast<size_t>(L) * ni * ni];
                      cplx q = 0.0;
                      for (int j = 0; j < ni; ++j) {
                        cplx t = 0.0;
                        for (int i = 0; i < ni; ++i) t += dl[i * ni + j] * std::conj(Pq(off + i, m));
                        q += t * dP(off + j, n);
                      }
                      sum += std::conj(V[static_cast<size_t>(a) * nLmax + L]) * q;
                    }
                    floc[3 * a + d] += pref * 2.0 * sum.real();
                  }
                  // (ii) position of the compensation charge: Re Omega sum_G conj(v~(G)) (-i p) Q_L g_aL(G)
                  std::fill(tmp.begin(), tmp.end(), cplx(0.0));
                  for (int L = 0; L < su.nL; ++L) {
                    const cplx c = minus_i_pow[l_of_L(L) % 4] * Qa[L];
                    if (c == cplx(0.0)) continue;
                    const double* t = &T[is][static_cast<size_t>(L) * ng];
                    for (int ig = 0; ig < ng; ++ig) tmp[ig] += c * t[ig];
                  }
                  const cplx* ph = phase[a].data();
                  double g[3] = {0, 0, 0};
                  for (int ig = 0; ig < ng; ++ig) {
                    const double im = (std::conj(vg[ig]) * ph[ig] * tmp[ig]).imag();  // Re[-i z] = Im z
                    for (int d = 0; d < 3; ++d) g[d] += pvec[3 * static_cast<size_t>(ig) + d] * im;
                  }
                  for (int d = 0; d < 3; ++d) floc[3 * a + d] += pref * vol * g[d];
                }
              }
            }
            if (build) {
              fft_c_.forward_pruned(wr.data(), bk.columns, bk.planes);
              cplx* w = W[s].col(n);
              for (int ig = 0; ig < bk.npw; ++ig) w[ig] -= alpha * inv_nc * wr[bk.idx[ig]];
              cplx* cn = Cnl[s].col(n);
              for (int i = 0; i < nproj; ++i) cn[i] -= alpha * cnl[i];
            }
          }
          if (forces) {
#pragma omp critical
            for (size_t i = 0; i < fx.size(); ++i) fx[i] += floc[i];
          }
        }
        energy += e_sum;
      }
    }

    if (build)
      for (int s = 0; s < nspin_; ++s) {
        const size_t ks = static_cast<size_t>(ik) * nspin_ + s;
        if (nproj > 0)
          gemm('N', 'N', bk.npw, nbands_, nproj, 1.0, proj.beta.data(), bk.npw, Cnl[s].data(), nproj, 1.0, W[s].data(), bk.npw);
        // compression: M = -psi^H W = U diag(lambda) U^H (positive semidefinite), xi = W U lambda^-1/2
        CMatrix M = overlap(psi_[ks].data(), nbands_, W[s].data(), nbands_, bk.npw);
        for (int i = 0; i < nbands_; ++i)
          for (int j = 0; j <= i; ++j) {
            const cplx v = -0.5 * (M(i, j) + std::conj(M(j, i)));
            M(i, j) = v;
            M(j, i) = std::conj(v);
          }
        std::vector<double> lam;
        eigh(M, lam);
        const double lmax = lam.empty() ? 0.0 : lam.back();
        int first = 0;
        while (first < nbands_ && lam[first] <= 1e-12 * lmax) ++first;
        const int nkeep = nbands_ - first;
        CMatrix& xi = ace_[ks];
        xi.resize(bk.npw, nkeep);
        if (nkeep > 0) {
          CMatrix U(nbands_, nkeep);
          for (int j = 0; j < nkeep; ++j) {
            const double sc = 1.0 / std::sqrt(lam[first + j]);
            for (int i = 0; i < nbands_; ++i) U(i, j) = M(i, first + j) * sc;
          }
          gemm('N', 'N', bk.npw, nkeep, nbands_, 1.0, W[s].data(), bk.npw, U.data(), nbands_, 0.0, xi.data(), bk.npw);
        }
      }
  }
  par::sum(&energy, 1);
  if (forces) {
    par::sum(fx.data(), fx.size());
    *forces = fx;
  }
  return energy;
}

}  // namespace lsda
