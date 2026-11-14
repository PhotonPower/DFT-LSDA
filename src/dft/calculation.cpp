#include "dft/calculation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

#include "base/parallel.hpp"
#include "dft/eigensolver.hpp"
#include "dft/hamiltonian.hpp"
#include "math/special.hpp"

namespace lsda {

namespace {
// sign of (-i)^l written as s_l * (1 or -i): s_l = (-1)^{floor(l/2)}
inline double il_sign(int l) { return (l / 2) % 2 == 0 ? 1.0 : -1.0; }
inline int l_of_L(int L) { return static_cast<int>(std::floor(std::sqrt(static_cast<double>(L)) + 1e-9)); }
}  // namespace

// =====================================================================================
//  Construction
// =====================================================================================
Calculation::Calculation(Crystal crystal, const Params& params) : c_(std::move(crystal)), p_(params) {
  ScopedTimer timer("setup");
  LSDA_CHECK(p_.nspin == 1 || p_.nspin == 2, "nspin must be 1 or 2");
  LSDA_CHECK(p_.eigensolver == "ppcg" || p_.eigensolver == "lobpcg", "eigensolver must be ppcg or lobpcg");
  noncol_ = p_.noncollinear;
  nspin_ = noncol_ ? 1 : p_.nspin;
  ncomp_ = noncol_ ? 4 : nspin_;
  // exchange-correlation functional: by default the one the PAW datasets were generated with
  {
    std::string from_data;
    for (const Species& sp : c_.species) {
      if (from_data.empty()) from_data = sp.setup.xc_name;
      LSDA_CHECK(from_data == sp.setup.xc_name, "PAW datasets were generated with different xc functionals");
    }
    if (p_.xc.empty()) p_.xc = from_data;
    xc_ = make_xc(p_.xc, p_.exx_fraction, p_.exx_screening);
    hybrid_ = xc_->exx_fraction() != 0.0;
    std::string semilocal = xc_->name();
    if (hybrid_) {
      // semilocal functional of the start phase: the parent GGA, for Hartree-Fock the functional of the datasets
      const std::string start = xc_->is_gga() ? "pbe" : (from_data.empty() ? "pw92" : from_data);
      xc_other_ = make_xc(start);
      semilocal = xc_other_->name();
      exx_active_ = true;
      LSDA_CHECK(!noncol_, "hybrid functionals are not implemented for non-collinear magnetism");
      for (Species& sp : c_.species) {
        LSDA_CHECK(sp.setup.has_exx_data, "PAW dataset of " + sp.name + " contains no exact-exchange data");
        sp.setup.prepare_exchange(xc_->exx_screening());
      }
    }
    if (!from_data.empty() && make_xc(from_data)->name() != semilocal && p_.verbosity > 0)
      logf("\n  WARNING: xc functional '%s' differs from the one of the PAW datasets (%s)\n", p_.xc.c_str(), from_data.c_str());
  }
  smearing_ = parse_smearing(p_.smearing);
  nelec_ = c_.n_electrons();

  // ---- cutoffs ------------------------------------------------------------------------
  if (p_.ecut <= 0.0) {
    for (const Species& s : c_.species) p_.ecut = std::max(p_.ecut, s.setup.ecut_high);
    LSDA_CHECK(p_.ecut > 0.0, "ecut not given and no recommendation in the PAW datasets");
  }
  LSDA_CHECK(p_.dense_factor >= 4.0, "dense_factor must be >= 4");
  if (p_.ecut_dense <= 0.0) p_.ecut_dense = p_.dense_factor * p_.ecut;
  LSDA_CHECK(p_.ecut_dense >= 4.0 * p_.ecut - 1e-9, "ecut_dense must be at least 4 * ecut");

  // ---- symmetry and k-points ----------------------------------------------------------
  LSDA_CHECK(!(noncol_ && xc_->is_gga()), "non-collinear magnetism is implemented for the LSDA only");
  LSDA_CHECK(!(noncol_ && p_.fixed_moment), "fixed_moment is not available for non-collinear calculations");
  // spin rotations are not part of the symmetry analysis: non-collinear runs use no symmetry and no time reversal
  // exact exchange couples all k-points of the full grid: hybrid runs use no symmetry either
  ops_ = find_symmetry(c_, p_.symmetry && !noncol_ && !hybrid_);
  kpts_ = make_kpoints(p_.kgrid, p_.kshift, ops_, &tetra_.map, !noncol_ && !hybrid_);
  for (int d = 0; d < 3; ++d) tetra_.n[d] = p_.kgrid[d];
  tetra_.B = c_.B;
  for (const KPoint& k : kpts_) kweight_.push_back(k.weight);
  rot_.resize(ops_.size());
  for (size_t io = 0; io < ops_.size(); ++io)
    for (const Species& sp : c_.species) {
      const PawSetup& s = sp.setup;
      const RMatrix d = ylm_rotation(s.lmax, ops_[io].R);
      RMatrix r(s.ni, s.ni);
      for (int i = 0; i < s.ni; ++i)
        for (int j = 0; j < s.ni; ++j)
          if (s.j_i[i] == s.j_i[j]) r(i, j) = d(s.L_i[i], s.L_i[j]);
      rot_[io].push_back(std::move(r));
    }

  // ---- FFT grids and basis sets -------------------------------------------------------
  int nc[3], nf[3];
  fft_dimensions(c_, 2.0 * std::sqrt(2.0 * p_.ecut), nc);
  fft_dimensions(c_, std::sqrt(2.0 * p_.ecut_dense), nf);
  for (int d = 0; d < 3; ++d) nf[d] = std::max(nf[d], nc[d]);
  fft_c_ = FFT3D(nc[0], nc[1], nc[2]);
  fft_f_ = FFT3D(nf[0], nf[1], nf[2]);
  gs_ = make_gsphere(c_, p_.ecut_dense, fft_f_, fft_c_);
  for (const KPoint& k : kpts_) basis_.push_back(make_pw_basis(c_, k, p_.ecut, fft_c_, p_.ecut_smooth));

  // ---- bands ---------------------------------------------------------------------------
  nbands_ = p_.nbands;
  if (nbands_ <= 0) {
    double mom = 0.0;
    for (const Atom& a : c_.atoms) mom += std::abs(a.magmom);
    if (p_.fixed_moment) mom = std::max(mom, std::abs(*p_.fixed_moment));
    nbands_ = static_cast<int>(std::ceil(0.5 * nelec_ * 1.2 + (nspin_ == 2 ? 0.5 * mom : 0.0))) + 4;
    if (noncol_) nbands_ = static_cast<int>(std::ceil(nelec_ * 1.2)) + 8;  // spinor bands hold one electron each
  }
  int npw_min = basis_[0].npw, npw_max = 0;
  for (const PWBasis& b : basis_) {
    npw_min = std::min(npw_min, b.npw);
    npw_max = std::max(npw_max, b.npw);
  }
  LSDA_CHECK(nbands_ <= (noncol_ ? 2 : 1) * npw_min, "more bands than plane waves");
  LSDA_CHECK((noncol_ ? 1.0 : 2.0) * nbands_ >= nelec_, "nbands too small");

  // ---- reciprocal-space tables of the PAW datasets ------------------------------------
  // table ranges from the actual vector lengths (they differ from the cutoffs if a reference lattice is used)
  double qmax_wfc = 0.0, qmax_dense = 0.0;
  for (const PWBasis& b : basis_)
    for (int ig = 0; ig < b.npw; ++ig)
      qmax_wfc = std::max(qmax_wfc, std::sqrt(b.kg[3 * ig] * b.kg[3 * ig] + b.kg[3 * ig + 1] * b.kg[3 * ig + 1] +
                                              b.kg[3 * ig + 2] * b.kg[3 * ig + 2]));
  for (double g2 : gs_.g2) qmax_dense = std::max(qmax_dense, std::sqrt(g2));
  qmax_wfc += 0.05;
  qmax_dense += 0.05;
  const int ng = gs_.ng;
  ff_g_.resize(c_.species.size());
  ff_nct_.resize(c_.species.size());
  ff_vbar_.resize(c_.species.size());
  ff_nvt_.resize(c_.species.size());
  for (size_t is = 0; is < c_.species.size(); ++is) {
    PawSetup& s = c_.species[is].setup;
    s.build_tables(qmax_wfc, qmax_dense);
    lmax_comp_ = std::max(lmax_comp_, 2 * s.lmax);
    const int nl = 2 * s.lmax + 1;
    ff_g_[is].resize(static_cast<size_t>(nl) * ng);
    ff_nct_[is].resize(ng);
    ff_vbar_[is].resize(ng);
    ff_nvt_[is].resize(ng);
    LSDA_CHECK(!s.nvt.empty(), "PAW dataset without pseudo valence density: " + s.path);
#pragma omp parallel for schedule(static)
    for (int ig = 0; ig < ng; ++ig) {
      const double q = std::sqrt(gs_.g2[ig]);
      for (int l = 0; l < nl; ++l) ff_g_[is][static_cast<size_t>(l) * ng + ig] = s.g_q[l](q);
      ff_nct_[is][ig] = s.nct_q(q);
      ff_vbar_[is][ig] = s.vbar_q(q);
      ff_nvt_[is][ig] = s.nvt_q(q);
    }
  }
  LSDA_CHECK(lmax_comp_ <= 6, "angular momentum of the compensation charges too large");

  dS_.resize(c_.natoms());
  for (int a = 0; a < c_.natoms(); ++a) dS_[a] = c_.setup(a).dS;

  // ---- report --------------------------------------------------------------------------
  if (p_.verbosity > 0) {
    logf("\n  Cell volume            : %.6f bohr^3\n", c_.volume);
    logf("  Atoms / species        : %d / %zu\n", c_.natoms(), c_.species.size());
    logf("  Valence electrons      : %.4f\n", nelec_);
    logf("  Exchange-correlation   : %s\n", xc_->name().c_str());
    if (noncol_)
      logf("  Non-collinear spinors  : %d bands\n", nbands_);
    else
      logf("  Spin channels / bands  : %d / %d\n", nspin_, nbands_);
    logf("  Cutoff wavefunctions   : %.2f Ha   (%d - %d plane waves)\n", p_.ecut, npw_min, npw_max);
    if (p_.ecut_smooth > 0.0) logf("  Cutoff smoothing       : %.2f Ha\n", p_.ecut_smooth);
    logf("  Cutoff dense sphere    : %.2f Ha   (%d G-vectors)\n", p_.ecut_dense, ng);
    logf("  FFT grid coarse / fine : %d x %d x %d  /  %d x %d x %d\n", nc[0], nc[1], nc[2], nf[0], nf[1], nf[2]);
    logf("  Symmetry operations    : %zu\n", ops_.size());
    logf("  k-points               : %d x %d x %d grid -> %zu irreducible\n", p_.kgrid[0], p_.kgrid[1], p_.kgrid[2],
         kpts_.size());
    if (smearing_ == Smearing::Tetrahedron)
      logf("  Brillouin-zone integral: tetrahedron method with Bloechl correction (forces and stresses are not\n"
           "                           exactly consistent with this energy; use a smearing method for relaxations)\n");
    else
      logf("  Smearing               : %s, width %.5f Ha\n", p_.smearing.c_str(), p_.width);
    logf("  Threads                : %d\n", num_threads());
    const double mem = 16.0 * npw_max * nbands_ * kpts_.size() * (noncol_ ? 2 : nspin_) / 1048576.0;
    logf("  Wavefunction storage   : %.1f MB\n", mem);
  }

  setup_structure();
  setup_mixer();
  if (hybrid_) setup_exchange();
  init_wavefunctions();
  init_density();
}

// Structure-dependent sums: pseudo core density and zero potential of all atoms on the dense sphere.
void Calculation::setup_structure() {
  const int ng = gs_.ng, nat = c_.natoms();
  // The PAW transformation is exact only for non-overlapping augmentation spheres: report overlaps.
  {
    double worst = 0.0;
    int wa = -1, wb = -1;
    for (int a = 0; a < nat; ++a)
      for (int b = a; b < nat; ++b)
        for (int i = -2; i <= 2; ++i)
          for (int j = -2; j <= 2; ++j)
            for (int k = -2; k <= 2; ++k) {
              if (a == b && i == 0 && j == 0 && k == 0) continue;
              const Vec3 df{c_.atoms[b].frac[0] - c_.atoms[a].frac[0] + i, c_.atoms[b].frac[1] - c_.atoms[a].frac[1] + j,
                            c_.atoms[b].frac[2] - c_.atoms[a].frac[2] + k};
              const double d = norm(c_.cart(df));
              const double overlap = (c_.setup(a).r_paw + c_.setup(b).r_paw - d) / d;
              if (overlap > worst) {
                worst = overlap;
                wa = a;
                wb = b;
              }
            }
    if (worst > 0.0 && p_.verbosity > 0)
      logf("  WARNING: PAW spheres of atoms %d and %d overlap by %.1f %% of their distance\n", wa + 1, wb + 1, 100 * worst);
  }
  nct_G_.assign(ng, 0.0);
  vbar_G_.assign(ng, 0.0);
  std::vector<Vec3> pos(nat);
  for (int a = 0; a < nat; ++a) pos[a] = c_.atom_cart(a);
  const double inv_vol = 1.0 / c_.volume;
#pragma omp parallel for schedule(static)
  for (int ig = 0; ig < ng; ++ig) {
    const double* g = &gs_.g[3 * ig];
    cplx sn = 0.0, sv = 0.0;
    for (int a = 0; a < nat; ++a) {
      const double arg = -(g[0] * pos[a][0] + g[1] * pos[a][1] + g[2] * pos[a][2]);
      const cplx phase(std::cos(arg), std::sin(arg));
      const int is = c_.atoms[a].species;
      sn += phase * ff_nct_[is][ig];
      sv += phase * ff_vbar_[is][ig];
    }
    nct_G_[ig] = sn * inv_vol;
    vbar_G_[ig] = sv * inv_vol;
  }
}

void Calculation::init_wavefunctions() {
  const int nk = static_cast<int>(kpts_.size());
  psi_.resize(static_cast<size_t>(nk) * nspin_);
  proj_psi_.resize(psi_.size());
  eig_.assign(psi_.size(), std::vector<double>(nbands_, 0.0));
  std::uniform_real_distribution<double> u(-0.5, 0.5);
  for (int ik = 0; ik < nk; ++ik) {
    if (!par::owns(ik)) continue;  // wavefunctions live on the process that owns the k-point
    for (int s = 0; s < nspin_; ++s) {
      const size_t ks0 = static_cast<size_t>(ik) * nspin_ + s;
      std::mt19937_64 rng(p_.seed + 7919 * ks0);  // independent of the number of processes
      CMatrix& x = psi_[ks0];
      const PWBasis& b = basis_[ik];
      const int dim = (noncol_ ? 2 : 1) * b.npw;
      x.resize(dim, nbands_);
      for (int n = 0; n < nbands_; ++n)
        for (int ig = 0; ig < dim; ++ig) x(ig, n) = cplx(u(rng), u(rng)) / (1.0 + 2.0 * b.kin[ig % b.npw]);
    }
  }
  first_solve_ = true;
}

// Pseudo partial waves of the bound atomic states of all atoms in the plane-wave basis of k-point ik.
CMatrix Calculation::atomic_orbitals(const PWBasis& b) const {
  int nao = 0;
  for (int a = 0; a < c_.natoms(); ++a) {
    const PawSetup& s = c_.setup(a);
    for (int j = 0; j < s.nj; ++j)
      if (s.is_bound(j)) nao += 2 * s.l_j[j] + 1;
  }
  CMatrix ao(b.npw, nao);
  const cplx il[4] = {cplx(1, 0), cplx(0, -1), cplx(-1, 0), cplx(0, 1)};
  int col0 = 0;
  for (int a = 0; a < c_.natoms(); ++a) {
    const PawSetup& s = c_.setup(a);
    const Vec3 r = c_.atom_cart(a);
    int ncol = 0;
    for (int j = 0; j < s.nj; ++j)
      if (s.is_bound(j)) ncol += 2 * s.l_j[j] + 1;
#pragma omp parallel for schedule(static)
    for (int ig = 0; ig < b.npw; ++ig) {
      const double* q = &b.kg[3 * ig];
      const double qn = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
      double ylm[49];
      real_ylm(s.lmax, q, ylm);
      const double arg = -(q[0] * r[0] + q[1] * r[1] + q[2] * r[2]);
      const cplx phase(std::cos(arg), std::sin(arg));
      int col = col0;
      for (int j = 0; j < s.nj; ++j) {
        if (!s.is_bound(j)) continue;
        const int l = s.l_j[j];
        const cplx f = phase * il[l % 4] * s.phit_q[j](qn);
        for (int m = -l; m <= l; ++m) ao(ig, col++) = f * ylm[lm_index(l, m)];
      }
    }
    col0 += ncol;
  }
  return ao;
}

// Superposition of atomic pseudo valence densities and atomic density matrices.
void Calculation::init_density() {
  const int ng = gs_.ng, nat = c_.natoms();
  rho_.assign(ncomp_, std::vector<cplx>(ng, 0.0));
  D_.assign(nat, std::vector<RMatrix>(ncomp_));
  // relative spin polarization of every atom: scalar (collinear) or vector (non-collinear)
  std::vector<Vec3> zeta(nat, Vec3{0, 0, 0});
  std::vector<Vec3> pos(nat);
  for (int a = 0; a < nat; ++a) {
    pos[a] = c_.atom_cart(a);
    const PawSetup& s = c_.setup(a);
    if (noncol_) {
      const double m = norm(c_.atoms[a].mvec), z = std::clamp(m / s.n_valence, 0.0, 0.95);
      if (m > 0.0)
        for (int d = 0; d < 3; ++d) zeta[a][d] = z * c_.atoms[a].mvec[d] / m;
    } else if (nspin_ == 2) {
      zeta[a][2] = std::clamp(c_.atoms[a].magmom / s.n_valence, -0.95, 0.95);
    }
    const RMatrix d0 = s.atomic_density_matrix();
    for (int c = 0; c < ncomp_; ++c) {
      D_[a][c] = d0;
      double f = 1.0;
      if (noncol_)
        f = c == 0 ? 1.0 : zeta[a][c - 1];
      else if (nspin_ == 2)
        f = 0.5 * (1.0 + (c == 0 ? zeta[a][2] : -zeta[a][2]));
      for (size_t i = 0; i < d0.size(); ++i) D_[a][c].data()[i] *= f;
    }
  }
  const double inv_vol = 1.0 / c_.volume;
#pragma omp parallel for schedule(static)
  for (int ig = 0; ig < ng; ++ig) {
    const double* g = &gs_.g[3 * ig];
    cplx tot = 0.0, m[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < nat; ++a) {
      const double arg = -(g[0] * pos[a][0] + g[1] * pos[a][1] + g[2] * pos[a][2]);
      const cplx v = cplx(std::cos(arg), std::sin(arg)) * (ff_nvt_[c_.atoms[a].species][ig] * inv_vol);
      tot += v;
      for (int d = 0; d < 3; ++d) m[d] += v * zeta[a][d];
    }
    if (noncol_) {
      rho_[0][ig] = tot;
      for (int d = 0; d < 3; ++d) rho_[1 + d][ig] = m[d];
    } else if (nspin_ == 1) {
      rho_[0][ig] = tot;
    } else {
      rho_[0][ig] = 0.5 * (tot + m[2]);
      rho_[1][ig] = 0.5 * (tot - m[2]);
    }
  }
  // enforce exact charge neutrality of the start density (rescale the valence part)
  double comp = 0.0;  // total compensation + pseudo core charge
  for (int a = 0; a < nat; ++a) {
    const PawSetup& s = c_.setup(a);
    RMatrix dt = D_[a][0];
    if (ncomp_ == 2)
      for (size_t i = 0; i < dt.size(); ++i) dt.data()[i] += D_[a][1].data()[i];
    std::vector<double> Q(s.nL);
    s.multipoles(dt, Q.data());
    comp += std::sqrt(4 * pi) * Q[0];
  }
  comp += nct_G_[0].real() * c_.volume;
  double nval = 0.0;
  for (int sp = 0; sp < (ncomp_ == 2 ? 2 : 1); ++sp) nval += rho_[sp][0].real() * c_.volume;
  const double scale = -comp / nval;
  if (p_.verbosity > 0) logf("  Start density          : pseudo valence charge %.6f, rescaled by %.8f\n", nval, scale);
  LSDA_CHECK(std::abs(scale - 1.0) < 0.05, "inconsistent atomic start density");
  for (auto& r : rho_)
    for (cplx& v : r) v *= scale;
  have_density_ = true;
  have_occ_ = false;
  mixer_.reset();
}

// New atomic positions. The density is carried along rigidly with the atoms in first order:
// rho <- rho - sum_a n_atom(r - R_a_old) + sum_a n_atom(r - R_a_new); wavefunctions and density
// matrices are kept as start values.
void Calculation::move_atoms(const std::vector<Vec3>& frac) {
  exx_ready_ = false;  // hybrid functionals: the exchange operators belong to the old positions
  exx_hist_.clear();
  exx_coef_.clear();
  const int nat = c_.natoms(), ng = gs_.ng;
  LSDA_CHECK(static_cast<int>(frac.size()) == nat, "move_atoms: wrong number of positions");
  std::vector<Vec3> r_old(nat), r_new(nat);
  for (int a = 0; a < nat; ++a) {
    r_old[a] = c_.atom_cart(a);
    c_.atoms[a].frac = frac[a];
    r_new[a] = c_.atom_cart(a);
  }
  if (have_density_) {
    const double f = (ncomp_ == 2 ? 0.5 : 1.0) / c_.volume;
#pragma omp parallel for schedule(static)
    for (int ig = 0; ig < ng; ++ig) {
      const double* g = &gs_.g[3 * ig];
      cplx d = 0.0;
      for (int a = 0; a < nat; ++a) {
        const double a_new = -(g[0] * r_new[a][0] + g[1] * r_new[a][1] + g[2] * r_new[a][2]);
        const double a_old = -(g[0] * r_old[a][0] + g[1] * r_old[a][1] + g[2] * r_old[a][2]);
        d += ff_nvt_[c_.atoms[a].species][ig] * (cplx(std::cos(a_new), std::sin(a_new)) - cplx(std::cos(a_old), std::sin(a_old)));
      }
      for (int s = 0; s < (ncomp_ == 2 ? 2 : 1); ++s) rho_[s][ig] += f * d;  // charge component(s) only
    }
  }
  setup_structure();
  mixer_.reset();
  proj_valid_ = false;  // the projectors move with the atoms
}

// =====================================================================================
//  Compensation charges
// =====================================================================================
// out(G) += 1/Omega sum_a exp(-iG.R_a) sum_L Q^a_L 4 pi (-i)^l Y_L(G) g_l(|G|)
void Calculation::add_compensation(const std::vector<std::vector<double>>& Q, std::vector<cplx>& out) const {
  const int ng = gs_.ng, nat = c_.natoms(), nL = (lmax_comp_ + 1) * (lmax_comp_ + 1);
  std::vector<Vec3> pos(nat);
  for (int a = 0; a < nat; ++a) pos[a] = c_.atom_cart(a);
  const double pref = 4 * pi / c_.volume;
#pragma omp parallel
  {
    std::vector<double> ylm(nL), t(nL * c_.species.size());
#pragma omp for schedule(static)
    for (int ig = 0; ig < ng; ++ig) {
      const double* g = &gs_.g[3 * ig];
      real_ylm(lmax_comp_, g, ylm.data());
      for (size_t is = 0; is < c_.species.size(); ++is) {
        const int nLs = c_.species[is].setup.nL;
        for (int L = 0; L < nLs; ++L) {
          const int l = l_of_L(L);
          t[is * nL + L] = il_sign(l) * ylm[L] * ff_g_[is][static_cast<size_t>(l) * ng + ig];
        }
      }
      cplx sum = 0.0;
      for (int a = 0; a < nat; ++a) {
        const int is = c_.atoms[a].species;
        const int nLs = c_.species[is].setup.nL;
        const double* ta = &t[is * nL];
        double even = 0.0, odd = 0.0;
        for (int L = 0; L < nLs; ++L) {
          if (l_of_L(L) % 2 == 0)
            even += Q[a][L] * ta[L];
          else
            odd += Q[a][L] * ta[L];
        }
        const double arg = -(g[0] * pos[a][0] + g[1] * pos[a][1] + g[2] * pos[a][2]);
        sum += cplx(std::cos(arg), std::sin(arg)) * cplx(even, -odd);
      }
      out[ig] += pref * sum;
    }
  }
}

// W[a][L] = int v(r) g_L^a(r - R_a) d^3r = sum_G Re[ conj(v(G)) exp(-iG.R_a) 4 pi (-i)^l Y_L(G) g_l(|G|) ]
void Calculation::project_on_shape(const std::vector<cplx>& v, std::vector<std::vector<double>>& W) const {
  const int ng = gs_.ng, nat = c_.natoms(), nL = (lmax_comp_ + 1) * (lmax_comp_ + 1);
  std::vector<Vec3> pos(nat);
  for (int a = 0; a < nat; ++a) pos[a] = c_.atom_cart(a);
  std::vector<double> acc(static_cast<size_t>(nat) * nL, 0.0);
#pragma omp parallel
  {
    std::vector<double> ylm(nL), t(nL * c_.species.size()), loc(static_cast<size_t>(nat) * nL, 0.0);
#pragma omp for schedule(static) nowait
    for (int ig = 0; ig < ng; ++ig) {
      const double* g = &gs_.g[3 * ig];
      real_ylm(lmax_comp_, g, ylm.data());
      for (size_t is = 0; is < c_.species.size(); ++is) {
        const int nLs = c_.species[is].setup.nL;
        for (int L = 0; L < nLs; ++L) {
          const int l = l_of_L(L);
          t[is * nL + L] = il_sign(l) * ylm[L] * ff_g_[is][static_cast<size_t>(l) * ng + ig];
        }
      }
      for (int a = 0; a < nat; ++a) {
        const int is = c_.atoms[a].species;
        const int nLs = c_.species[is].setup.nL;
        const double arg = -(g[0] * pos[a][0] + g[1] * pos[a][1] + g[2] * pos[a][2]);
        const cplx z = std::conj(v[ig]) * cplx(std::cos(arg), std::sin(arg));
        const double* ta = &t[is * nL];
        double* w = &loc[static_cast<size_t>(a) * nL];
        // (-i)^l = s_l for even l and -i s_l for odd l:  Re[z] and Re[-i z] = Im[z]
        for (int L = 0; L < nLs; ++L) w[L] += ta[L] * (l_of_L(L) % 2 == 0 ? z.real() : z.imag());
      }
    }
#pragma omp critical
    for (size_t i = 0; i < acc.size(); ++i) acc[i] += loc[i];
  }
  W.assign(nat, {});
  for (int a = 0; a < nat; ++a) {
    const int nLs = c_.setup(a).nL;
    W[a].resize(nLs);
    for (int L = 0; L < nLs; ++L) W[a][L] = 4 * pi * acc[static_cast<size_t>(a) * nL + L];
  }
}

// =====================================================================================
//  Energy functional and potentials for a given (pseudo density, density matrices)
// =====================================================================================
Calculation::Functional Calculation::evaluate(const RhoG& rho, const DMat& D, bool build) {
  ScopedTimer timer(build ? "potential" : "energy functional");
  const int ng = gs_.ng, nat = c_.natoms();
  const size_t nfine = fft_f_.size();
  const double vol = c_.volume;
  Functional e;

  // ---- total smooth charge: valence + core + compensation -------------------------------
  std::vector<std::vector<double>> Q(nat);
  for (int a = 0; a < nat; ++a) {
    const PawSetup& s = c_.setup(a);
    RMatrix dt = D[a][0];
    if (ncomp_ == 2)
      for (size_t i = 0; i < dt.size(); ++i) dt.data()[i] += D[a][1].data()[i];
    Q[a].resize(s.nL);
    s.multipoles(dt, Q[a].data());
  }
  std::vector<cplx> ntot(ng), rho_t(ng, 0.0);
  for (int ig = 0; ig < ng; ++ig) ntot[ig] = rho[0][ig] + (ncomp_ == 2 ? rho[1][ig] : cplx(0.0)) + nct_G_[ig];
  add_compensation(Q, rho_t);
  for (int ig = 0; ig < ng; ++ig) rho_t[ig] += ntot[ig];
  LSDA_CHECK(std::abs(rho_t[0]) * vol < 1e-5, strf("smooth charge is not neutral (%.3e)", std::abs(rho_t[0]) * vol));

  // ---- Hartree energy/potential and zero-potential energy --------------------------------
  std::vector<cplx> vh(ng, 0.0);
  double eh = 0.0, ez = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : eh, ez)
  for (int ig = 0; ig < ng; ++ig) {
    if (ig > 0) {
      vh[ig] = 4 * pi * rho_t[ig] / gs_.g2[ig];
      eh += 2 * pi * std::norm(rho_t[ig]) / gs_.g2[ig];
    }
    ez += (std::conj(ntot[ig]) * vbar_G_[ig]).real();
  }
  e.hartree = eh * vol;
  e.zero = ez * vol;

  // ---- exchange-correlation on the fine grid ----------------------------------------------
  std::vector<double> nup(nfine), ndn, exc(nfine), vup(nfine), vdn;
  CVec work(nfine);
  std::vector<std::vector<double>> vnc;  // non-collinear: dE/dn and dE/dm_c on the fine grid
  if (noncol_) {
    // local spin frame: n_up/down = (n +- |m|)/2;  dE/dn = (v_up + v_dn)/2,  dE/dm = (v_up - v_dn)/2 m/|m|
    std::vector<std::vector<double>> comp(4, std::vector<double>(nfine));
    for (int c = 0; c < 4; ++c) {
      std::fill(work.begin(), work.end(), cplx(0.0));
      for (int ig = 0; ig < ng; ++ig) work[gs_.idx_fine[ig]] = rho[c][ig] + (c == 0 ? nct_G_[ig] : cplx(0.0));
      fft_f_.backward(work.data());
      for (size_t i = 0; i < nfine; ++i) comp[c][i] = work[i].real();
    }
    ndn.resize(nfine);
    vdn.resize(nfine);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nfine; ++i) {
      const double m = std::sqrt(comp[1][i] * comp[1][i] + comp[2][i] * comp[2][i] + comp[3][i] * comp[3][i]);
      nup[i] = 0.5 * (comp[0][i] + m);
      ndn[i] = 0.5 * (comp[0][i] - m);
    }
    xc_->evaluate(nfine, nup.data(), ndn.data(), exc.data(), vup.data(), vdn.data());
    vnc.assign(4, std::vector<double>(nfine));
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nfine; ++i) {
      const double m = nup[i] - ndn[i], b = 0.5 * (vup[i] - vdn[i]);
      vnc[0][i] = 0.5 * (vup[i] + vdn[i]);
      for (int c = 1; c < 4; ++c) vnc[c][i] = m > 1e-14 ? b * comp[c][i] / m : 0.0;
    }
  }
  for (int s = 0; s < (noncol_ ? 0 : nspin_); ++s) {
    std::fill(work.begin(), work.end(), cplx(0.0));
    for (int ig = 0; ig < ng; ++ig) work[gs_.idx_fine[ig]] = rho[s][ig] + (nspin_ == 2 ? 0.5 : 1.0) * nct_G_[ig];
    fft_f_.backward(work.data());
    std::vector<double>& dst = s == 0 ? nup : ndn;
    dst.resize(nfine);
    const double f = nspin_ == 2 ? 1.0 : 0.5;
    for (size_t i = 0; i < nfine; ++i) dst[i] = f * work[i].real();
  }
  if (noncol_) {
    // already evaluated above
  } else if (xc_->is_gga()) {
    // ---- gradient-corrected functional (White & Bird scheme) --------------------------------
    // grad n_s from i G n_s(G) on the dense sphere; potential v_s = de/dn_s - div(de/d grad n_s)
    const double fs = nspin_ == 2 ? 1.0 : 0.5;
    std::vector<std::vector<double>> grad(3 * nspin_, std::vector<double>(nfine));
    for (int s = 0; s < nspin_; ++s)
      for (int a = 0; a < 3; ++a) {
        std::fill(work.begin(), work.end(), cplx(0.0));
        for (int ig = 0; ig < ng; ++ig)
          work[gs_.idx_fine[ig]] = cplx(0.0, gs_.g[3 * ig + a]) * (rho[s][ig] + (nspin_ == 2 ? 0.5 : 1.0) * nct_G_[ig]);
        fft_f_.backward(work.data());
        std::vector<double>& g = grad[3 * s + a];
        for (size_t i = 0; i < nfine; ++i) g[i] = fs * work[i].real();
      }
    auto gr = [&](int s, int a) -> const std::vector<double>& { return grad[3 * (nspin_ == 2 ? s : 0) + a]; };
    std::vector<double> suu(nfine), sud(nfine), sdd(nfine), vsuu(nfine), vsud(nfine), vsdd(nfine);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nfine; ++i) {
      double uu = 0.0, ud = 0.0, dd = 0.0;
      for (int a = 0; a < 3; ++a) {
        const double gu = gr(0, a)[i], gd = gr(1, a)[i];
        uu += gu * gu;
        ud += gu * gd;
        dd += gd * gd;
      }
      suu[i] = uu;
      sud[i] = ud;
      sdd[i] = dd;
    }
    const std::vector<double>& nd = nspin_ == 2 ? ndn : nup;
    vdn.resize(nfine);
    xc_->evaluate_gga(nfine, nup.data(), nd.data(), suu.data(), sud.data(), sdd.data(), exc.data(), vup.data(), vdn.data(),
                      vsuu.data(), vsud.data(), vsdd.data());
    if (build) {  // integrals for the stress tensor: sum_s (de/d grad n_s)_a (grad n_s)_b
      const int ca[6] = {0, 1, 2, 1, 0, 0}, cb[6] = {0, 1, 2, 2, 2, 1};
      for (int c = 0; c < 6; ++c) {
        double sum = 0.0;
        const std::vector<double>&ua = gr(0, ca[c]), &ub = gr(0, cb[c]), &da = gr(1, ca[c]), &db = gr(1, cb[c]);
#pragma omp parallel for schedule(static) reduction(+ : sum)
        for (size_t i = 0; i < nfine; ++i)
          sum += 2.0 * vsuu[i] * ua[i] * ub[i] + 2.0 * vsdd[i] * da[i] * db[i] + vsud[i] * (ua[i] * db[i] + da[i] * ub[i]);
        gga_stress_[c] = sum * vol / static_cast<double>(nfine);
      }
    }
    // divergence term, evaluated on the dense sphere
    std::vector<cplx> div(ng);
    for (int s = 0; s < nspin_; ++s) {
      std::fill(div.begin(), div.end(), cplx(0.0));
      const std::vector<double>& vss = s == 0 ? vsuu : vsdd;
      for (int a = 0; a < 3; ++a) {
        const std::vector<double>&gs = gr(s, a), &go = gr(1 - s, a);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nfine; ++i) work[i] = 2.0 * vss[i] * gs[i] + vsud[i] * go[i];
        fft_f_.forward(work.data());
        for (int ig = 0; ig < ng; ++ig) div[ig] += cplx(0.0, gs_.g[3 * ig + a]) * work[gs_.idx_fine[ig]];
      }
      std::fill(work.begin(), work.end(), cplx(0.0));
      const double inv = 1.0 / static_cast<double>(nfine);
      for (int ig = 0; ig < ng; ++ig) work[gs_.idx_fine[ig]] = div[ig] * inv;
      fft_f_.backward(work.data());
      std::vector<double>& v = s == 0 ? vup : vdn;
      for (size_t i = 0; i < nfine; ++i) v[i] -= work[i].real();
    }
    if (nspin_ == 1) vdn.clear();
  } else if (nspin_ == 1) {
    xc_->evaluate(nfine, nup.data(), nup.data(), exc.data(), vup.data(), vup.data());
  } else {
    vdn.resize(nfine);
    xc_->evaluate(nfine, nup.data(), ndn.data(), exc.data(), vup.data(), vdn.data());
  }
  if (build && !xc_->is_gga()) std::fill(gga_stress_, gga_stress_ + 6, 0.0);
  double exc_sum = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : exc_sum)
  for (size_t i = 0; i < nfine; ++i) exc_sum += exc[i];
  e.xc = exc_sum * vol / static_cast<double>(nfine);
  if (build) {
    double vn = 0.0;
    const std::vector<double>& nd = (nspin_ == 2 || noncol_) ? ndn : nup;
    const std::vector<double>& vd = (nspin_ == 2 || noncol_) ? vdn : vup;
#pragma omp parallel for schedule(static) reduction(+ : vn)
    for (size_t i = 0; i < nfine; ++i) vn += vup[i] * nup[i] + vd[i] * nd[i];
    vxc_n_ = vn * vol / static_cast<double>(nfine);
  }

  // ---- potentials ---------------------------------------------------------------------------
  std::vector<std::vector<double>> W;
  if (build) {
    vH_G_ = vh;
    veff_G_.assign(ncomp_, std::vector<cplx>(ng));
    vcoarse_.assign(ncomp_, std::vector<double>(fft_c_.size()));
    std::vector<double> vsmooth(nfine);  // Hartree + zero potential in real space
    std::fill(work.begin(), work.end(), cplx(0.0));
    for (int ig = 0; ig < ng; ++ig) work[gs_.idx_fine[ig]] = vh[ig] + vbar_G_[ig];
    fft_f_.backward(work.data());
    for (size_t i = 0; i < nfine; ++i) vsmooth[i] = work[i].real();
    CVec wc(fft_c_.size());
    for (int s = 0; s < ncomp_; ++s) {
      // collinear: one potential per spin; non-collinear: v (with Hartree and zero potential) and B_x, B_y, B_z
      const std::vector<double>& vxc = noncol_ ? vnc[s] : (s == 0 ? vup : vdn);
      const double smooth = (noncol_ && s > 0) ? 0.0 : 1.0;
      for (size_t i = 0; i < nfine; ++i) work[i] = smooth * vsmooth[i] + vxc[i];
      fft_f_.forward(work.data());
      const double inv = 1.0 / static_cast<double>(nfine);
      std::fill(wc.begin(), wc.end(), cplx(0.0));
      for (int ig = 0; ig < ng; ++ig) {
        veff_G_[s][ig] = work[gs_.idx_fine[ig]] * inv;
        if (gs_.idx_coarse[ig] >= 0) wc[gs_.idx_coarse[ig]] = veff_G_[s][ig];
      }
      fft_c_.backward(wc.data());
      for (size_t i = 0; i < wc.size(); ++i) vcoarse_[s][i] = wc[i].real();
    }
    project_on_shape(vh, W);
    dH_.assign(nat, std::vector<RMatrix>(ncomp_));
  }

  // ---- atomic (on-site) corrections ---------------------------------------------------------
  double eon = 0.0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : eon)
  for (int a = 0; a < nat; ++a) {
    const PawSetup& s = c_.setup(a);
    eon += s.onsite(ncomp_, D[a].data(), build ? dH_[a].data() : nullptr, *xc_);
    if (build)
      for (int sp = 0; sp < (noncol_ ? 1 : nspin_); ++sp)  // the electrostatic term acts on the charge only
        for (int L = 0; L < s.nL; ++L) {
          const double w = W[a][L];
          const double* d = &s.Delta_Lij[static_cast<size_t>(L) * s.ni * s.ni];
          double* h = dH_[a][sp].data();
          for (int ij = 0; ij < s.ni * s.ni; ++ij) h[ij] += w * d[ij];
        }
  }
  e.onsite = eon;
  if (build) f_pot_ = e;
  return e;
}

// int v~ n~_val + sum dH D  (the potential-energy part contained in the band energy)
double Calculation::double_counting(const RhoG& rho, const DMat& D) const {
  double dc = 0.0;
  for (int s = 0; s < ncomp_; ++s) {
    double sum = 0.0;
    const int ng = gs_.ng;
#pragma omp parallel for schedule(static) reduction(+ : sum)
    for (int ig = 0; ig < ng; ++ig) sum += (std::conj(veff_G_[s][ig]) * rho[s][ig]).real();
    dc += sum * c_.volume;
  }
  for (int a = 0; a < c_.natoms(); ++a)
    for (int s = 0; s < ncomp_; ++s)
      for (size_t i = 0; i < D[a][s].size(); ++i) dc += dH_[a][s].data()[i] * D[a][s].data()[i];
  return dc;
}

// =====================================================================================
//  Band structure
// =====================================================================================
Hamiltonian Calculation::hamiltonian(int ik, int s, const Projectors& proj) const {
  const int nat = c_.natoms();
  std::vector<const RMatrix*> ds(nat);
  for (int a = 0; a < nat; ++a) ds[a] = &dS_[a];
  if (noncol_) {
    std::vector<std::array<const RMatrix*, 4>> dh(nat);
    for (int a = 0; a < nat; ++a) dh[a] = {&dH_[a][0], &dH_[a][1], &dH_[a][2], &dH_[a][3]};
    return Hamiltonian(basis_[ik], fft_c_, std::array<const double*, 4>{vcoarse_[0].data(), vcoarse_[1].data(), vcoarse_[2].data(), vcoarse_[3].data()},
                       proj, std::move(dh), std::move(ds));
  }
  std::vector<const RMatrix*> dh(nat);
  for (int a = 0; a < nat; ++a) dh[a] = &dH_[a][s];
  Hamiltonian h(basis_[ik], fft_c_, vcoarse_[s].data(), proj, std::move(dh), std::move(ds));
  if (exx_active_)
    for (size_t i = 0; i < exx_hist_.size(); ++i)
      if (exx_coef_[i] != 0.0) h.add_exchange(&exx_hist_[i].ace[static_cast<size_t>(ik) * nspin_ + s], exx_coef_[i]);
  return h;
}

void Calculation::solve_bands(double tol, int max_iter, double& max_residual, int& not_converged) {
  ScopedTimer timer("eigensolver");
  const int nk = static_cast<int>(kpts_.size());
  max_residual = 0.0;
  not_converged = 0;
  const double deg = (nspin_ == 1 && !noncol_) ? 2.0 : 1.0;
  for (int ik = 0; ik < nk; ++ik) {
    if (!par::owns(ik)) {  // eigenvalues of other processes arrive through the reduction below
      for (int s = 0; s < nspin_; ++s) std::fill(eig_[static_cast<size_t>(ik) * nspin_ + s].begin(), eig_[static_cast<size_t>(ik) * nspin_ + s].end(), 0.0);
      continue;
    }
    const double tp = wall_time();
    const Projectors proj = make_projectors(c_, basis_[ik]);
    add_time("  projectors (k-point)", wall_time() - tp);
    for (int s = 0; s < nspin_; ++s) {
      const size_t ks = static_cast<size_t>(ik) * nspin_ + s;
      const Hamiltonian h = hamiltonian(ik, s, proj);
      if (first_solve_) {
        // start vectors: lowest states in the span of the atomic orbitals, padded with random vectors
        const double t0 = wall_time();
        const CMatrix ao = atomic_orbitals(basis_[ik]);
        const int npw = basis_[ik].npw, dim = h.npw(), nao = (noncol_ ? 2 : 1) * ao.cols();
        const int nrand = std::max(0, nbands_ - nao) + (nao < nbands_ ? 0 : std::min(4, nbands_));
        CMatrix all(dim, nao + nrand);
        for (int j = 0; j < ao.cols(); ++j) {
          std::copy(ao.col(j), ao.col(j) + npw, all.col(j));  // spin up (or the only component)
          if (noncol_) std::copy(ao.col(j), ao.col(j) + npw, all.col(ao.cols() + j) + npw);  // spin down
        }
        for (int n = 0; n < nrand; ++n) std::copy(psi_[ks].col(n), psi_[ks].col(n) + dim, all.col(nao + n));
        if (nao >= nbands_ / 2 && nao + nrand <= dim && subspace_start(h, all, nbands_))
          psi_[ks] = std::move(all);
        else
          lcao_start_ok_ = false;
        add_time("  start wavefunctions", wall_time() - t0);
      }
      // All bands are iterated; the convergence statistics cover the bands that carry charge. Bands with
      // negligible occupation converge slowly and do not enter the density: counting them would keep the
      // SCF loop running long after the density has converged.
      const std::vector<double> tb(nbands_, tol);
      int n_required = nbands_ - std::max(2, nbands_ / 8);
      if (have_occ_) {
        int last_occ = -1;
        for (int n = 0; n < nbands_; ++n)
          if (std::abs(occ_.f[ks][n]) > 1e-6 * deg) last_occ = n;
        n_required = std::min(nbands_, last_occ + 1);
      }
      const int iters = first_solve_ && !lcao_start_ok_ ? 5 * max_iter : max_iter;
      const EigensolverResult r = p_.eigensolver == "lobpcg"
                                      ? lobpcg(h, psi_[ks], eig_[ks], tb, n_required, iters, &proj_psi_[ks])
                                      : ppcg(h, psi_[ks], eig_[ks], tb, n_required, iters, &proj_psi_[ks], p_.ppcg_block, 5,
                                             proj_valid_ && !first_solve_);
      max_residual = std::max(max_residual, r.max_residual);
      not_converged += r.not_converged;
    }
  }
  proj_valid_ = true;
  if (par::size() > 1) {  // collect eigenvalues and statistics of all processes
    std::vector<double> buf;
    for (const auto& e : eig_) buf.insert(buf.end(), e.begin(), e.end());
    par::sum(buf.data(), buf.size());
    size_t pos = 0;
    for (auto& e : eig_)
      for (double& v : e) v = buf[pos++];
    par::max(&max_residual, 1);
    par::sum(&not_converged, 1);
  }
}

// =====================================================================================
//  Density from wavefunctions
// =====================================================================================
void Calculation::compute_density(RhoG& rho, DMat& D) const {
  ScopedTimer timer("density");
  const int nk = static_cast<int>(kpts_.size()), nat = c_.natoms(), ng = gs_.ng;
  const size_t nc = fft_c_.size();
  rho.assign(ncomp_, std::vector<cplx>(ng, 0.0));
  D.assign(nat, std::vector<RMatrix>(ncomp_));
  for (int a = 0; a < nat; ++a)
    for (int s = 0; s < ncomp_; ++s) D[a][s].resize(c_.setup(a).ni, c_.setup(a).ni);
  std::vector<int> offset(nat);
  for (int a = 0, off = 0; a < nat; ++a) {
    offset[a] = off;
    off += c_.setup(a).ni;
  }

  if (noncol_) {
    // spinors: n = |psi_up|^2 + |psi_dn|^2, m = psi^+ sigma psi; D^c_ij = Re sum_ab P*_ia sigma^c_ab P_jb
    std::vector<std::vector<double>> rr(4, std::vector<double>(nc, 0.0));
    for (int ik = 0; ik < nk; ++ik) {
      if (!par::owns(ik)) continue;
      const CMatrix& x = psi_[ik];
      const PWBasis& b = basis_[ik];
      const std::vector<double>& f = occ_.f[ik];
#pragma omp parallel
      {
        std::vector<std::vector<double>> acc(4, std::vector<double>(nc, 0.0));
        CVec wu(nc), wd(nc);
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < nbands_; ++n) {
          const double w = f[n] * b.weight / c_.volume;
          if (std::abs(w) < 1e-16) continue;
          std::fill(wu.begin(), wu.end(), cplx(0.0));
          std::fill(wd.begin(), wd.end(), cplx(0.0));
          const cplx* xn = x.col(n);
          for (int ig = 0; ig < b.npw; ++ig) {
            wu[b.idx[ig]] = xn[ig];
            wd[b.idx[ig]] = xn[b.npw + ig];
          }
          fft_c_.backward_pruned(wu.data(), b.columns, b.planes);
          fft_c_.backward_pruned(wd.data(), b.columns, b.planes);
          for (size_t i = 0; i < nc; ++i) {
            const double uu = std::norm(wu[i]), dd = std::norm(wd[i]);
            const cplx ud = std::conj(wu[i]) * wd[i];
            acc[0][i] += w * (uu + dd);
            acc[1][i] += w * 2.0 * ud.real();
            acc[2][i] += w * 2.0 * ud.imag();
            acc[3][i] += w * (uu - dd);
          }
        }
#pragma omp critical
        for (int c = 0; c < 4; ++c)
          for (size_t i = 0; i < nc; ++i) rr[c][i] += acc[c][i];
      }
      const CMatrix& P = proj_psi_[ik];
      const int nph = P.rows() / 2;  // projections: spin-up block followed by the spin-down block
      for (int a = 0; a < nat; ++a) {
        const int ni = c_.setup(a).ni, off = offset[a];
        for (int n = 0; n < nbands_; ++n) {
          const double w = f[n] * b.weight;
          if (std::abs(w) < 1e-16) continue;
          for (int j = 0; j < ni; ++j) {
            const cplx ju = P(off + j, n), jd = P(nph + off + j, n);
            for (int i = 0; i < ni; ++i) {
              const cplx iu = std::conj(P(off + i, n)), id = std::conj(P(nph + off + i, n));
              D[a][0](i, j) += w * (iu * ju + id * jd).real();
              D[a][1](i, j) += w * (iu * jd + id * ju).real();
              D[a][2](i, j) += w * (cplx(0.0, -1.0) * iu * jd + cplx(0.0, 1.0) * id * ju).real();
              D[a][3](i, j) += w * (iu * ju - id * jd).real();
            }
          }
        }
      }
    }
    for (int c = 0; c < 4; ++c) {
      par::sum(rr[c].data(), nc);
      CVec work(nc);
      for (size_t i = 0; i < nc; ++i) work[i] = rr[c][i];
      fft_c_.forward(work.data());
      const double inv = 1.0 / static_cast<double>(nc);
      for (int ig = 0; ig < ng; ++ig)
        if (gs_.idx_coarse[ig] >= 0) rho[c][ig] = work[gs_.idx_coarse[ig]] * inv;
    }
  }
  for (int s = 0; s < (noncol_ ? 0 : nspin_); ++s) {
    std::vector<double> rho_r(nc, 0.0);
    for (int ik = 0; ik < nk; ++ik) {
      if (!par::owns(ik)) continue;
      const size_t ks = static_cast<size_t>(ik) * nspin_ + s;
      const CMatrix& x = psi_[ks];
      const PWBasis& b = basis_[ik];
      const std::vector<double>& f = occ_.f[ks];
#pragma omp parallel
      {
        std::vector<double> acc(nc, 0.0);
        CVec work(nc);
        bool used = false;
#pragma omp for schedule(dynamic, 1) nowait
        for (int n = 0; n < nbands_; ++n) {
          const double w = f[n] * b.weight / c_.volume;
          if (std::abs(w) < 1e-16) continue;
          used = true;
          std::fill(work.begin(), work.end(), cplx(0.0));
          const cplx* xn = x.col(n);
          for (int ig = 0; ig < b.npw; ++ig) work[b.idx[ig]] = xn[ig];
          fft_c_.backward_pruned(work.data(), b.columns, b.planes);
          for (size_t i = 0; i < nc; ++i) acc[i] += w * std::norm(work[i]);
        }
        if (used) {
#pragma omp critical
          for (size_t i = 0; i < nc; ++i) rho_r[i] += acc[i];
        }
      }
      // density matrices
      const CMatrix& P = proj_psi_[ks];
      for (int a = 0; a < nat; ++a) {
        const int ni = c_.setup(a).ni, off = offset[a];
        RMatrix& d = D[a][s];
        for (int n = 0; n < nbands_; ++n) {
          const double w = f[n] * b.weight;
          if (std::abs(w) < 1e-16) continue;
          for (int j = 0; j < ni; ++j) {
            const cplx pj = P(off + j, n);
            for (int i = 0; i < ni; ++i) d(i, j) += w * (std::conj(P(off + i, n)) * pj).real();
          }
        }
      }
    }
    par::sum(rho_r.data(), nc);  // contributions of the k-points of all processes
    CVec work(nc);
    for (size_t i = 0; i < nc; ++i) work[i] = rho_r[i];
    fft_c_.forward(work.data());
    const double inv = 1.0 / static_cast<double>(nc);
    for (int ig = 0; ig < ng; ++ig)
      if (gs_.idx_coarse[ig] >= 0) rho[s][ig] = work[gs_.idx_coarse[ig]] * inv;
  }
  if (par::size() > 1) {
    std::vector<double> buf;
    for (const auto& da : D)
      for (const RMatrix& d : da) buf.insert(buf.end(), d.data(), d.data() + d.size());
    par::sum(buf.data(), buf.size());
    size_t pos = 0;
    for (auto& da : D)
      for (RMatrix& d : da)
        for (size_t i = 0; i < d.size(); ++i) d.data()[i] = buf[pos++];
  }
  symmetrize_density(rho);
  symmetrize_D(D);
}

// rho_s(m') = 1/N sum_ops rho(Wk m') exp(2 pi i (Wk m').t)
void Calculation::symmetrize_density(RhoG& rho) const {
  const int nops = static_cast<int>(ops_.size()), ng = gs_.ng;
  if (nops == 1) return;
  for (auto& r : rho) {
    std::vector<cplx> out(ng);
#pragma omp parallel for schedule(static)
    for (int ig = 0; ig < ng; ++ig) {
      const int* mp = &gs_.m[3 * ig];
      cplx sum = 0.0;
      for (const SymOp& op : ops_) {
        int m[3];
        for (int i = 0; i < 3; ++i) m[i] = op.Wk[i][0] * mp[0] + op.Wk[i][1] * mp[1] + op.Wk[i][2] * mp[2];
        const int j = gs_.lookup[fft_f_.index(m[0], m[1], m[2])];
        if (j < 0) continue;  // only possible if the basis reference lattice breaks the symmetry (checked below)
        const double arg = 2 * pi * (m[0] * op.t[0] + m[1] * op.t[1] + m[2] * op.t[2]);
        sum += r[j] * cplx(std::cos(arg), std::sin(arg));
      }
      out[ig] = sum / static_cast<double>(nops);
    }
    r.swap(out);
  }
}

// D_s^b = 1/N sum_g rot(g) D^{g(b)} rot(g)^T
void Calculation::symmetrize_D(DMat& D) const {
  const int nops = static_cast<int>(ops_.size()), nat = c_.natoms();
  if (nops == 1) return;
  DMat out(nat, std::vector<RMatrix>(ncomp_));
  for (int b = 0; b < nat; ++b) {
    const int is = c_.atoms[b].species, ni = c_.setup(b).ni;
    for (int s = 0; s < nspin_; ++s) {
      RMatrix acc(ni, ni), tmp(ni, ni);
      for (int io = 0; io < nops; ++io) {
        const RMatrix& r = rot_[io][is];
        const RMatrix& d = D[ops_[io].atom_map[b]][s];
        tmp.fill(0.0);
        for (int j = 0; j < ni; ++j)       // tmp = r d
          for (int k = 0; k < ni; ++k) {
            const double dkj = d(k, j);
            if (dkj == 0.0) continue;
            for (int i = 0; i < ni; ++i) tmp(i, j) += r(i, k) * dkj;
          }
        for (int j = 0; j < ni; ++j)       // acc += tmp r^T
          for (int k = 0; k < ni; ++k) {
            const double rjk = r(j, k);
            if (rjk == 0.0) continue;
            for (int i = 0; i < ni; ++i) acc(i, j) += tmp(i, k) * rjk;
          }
      }
      for (size_t i = 0; i < acc.size(); ++i) acc.data()[i] /= nops;
      out[b][s] = std::move(acc);
    }
  }
  D.swap(out);
}

// =====================================================================================
//  Mixing
// =====================================================================================
// Mixing vector: [rho_tot(G) (re, im)] [rho_mag(G) (re, im)] [D_tot] [D_mag]
void Calculation::setup_mixer() {
  const int ng = gs_.ng;
  size_t nd = 0;
  for (int a = 0; a < c_.natoms(); ++a) nd += static_cast<size_t>(c_.setup(a).ni) * c_.setup(a).ni;
  const int nmag = ncomp_ == 2 ? 1 : (ncomp_ == 4 ? 3 : 0);  // magnetization components
  std::vector<double> w, pre;
  const double vol = c_.volume, q02 = p_.kerker_q0 * p_.kerker_q0;
  // G = 0 of the total density carries no Hartree energy (weight 0) but must be mixed with the same
  // factor as the density matrices: charge neutrality, Omega rho(0) + sum dS_ij D_ij = N, is linear.
  for (int c = 0; c < 2; ++c) {
    w.push_back(0.0);
    pre.push_back(p_.mix_alpha);
  }
  for (int ig = 1; ig < ng; ++ig) {
    const double g2 = gs_.g2[ig];
    for (int c = 0; c < 2; ++c) {
      w.push_back(4 * pi * vol / g2);
      pre.push_back(p_.mix_alpha * g2 / (g2 + q02));
    }
  }
  const double wmag = 4 * pi * vol / gs_.g2[1];
  for (int m = 0; m < nmag; ++m)
    for (int ig = 0; ig < ng; ++ig)
      for (int c = 0; c < 2; ++c) {
        w.push_back(wmag);
        pre.push_back(p_.mix_alpha_mag);
      }
  for (size_t i = 0; i < nd; ++i) {
    w.push_back(1.0);
    pre.push_back(p_.mix_alpha);
  }
  for (int m = 0; m < nmag; ++m)
    for (size_t i = 0; i < nd; ++i) {
      w.push_back(1.0);
      pre.push_back(p_.mix_alpha_mag);
    }
  mixer_ = Mixer(std::move(w), std::move(pre), p_.mix_history);
}

// charge and magnetization components of the stored quantities: collinear data are kept as (up, down),
// non-collinear data as (n, m_x, m_y, m_z)
std::vector<double> Calculation::pack(const RhoG& rho, const DMat& D) const {
  const int ng = gs_.ng, nmag = ncomp_ == 2 ? 1 : (ncomp_ == 4 ? 3 : 0);
  std::vector<double> x;
  for (int ig = 0; ig < ng; ++ig) {
    const cplx t = rho[0][ig] + (ncomp_ == 2 ? rho[1][ig] : cplx(0.0));
    x.push_back(t.real());
    x.push_back(t.imag());
  }
  for (int m = 0; m < nmag; ++m)
    for (int ig = 0; ig < ng; ++ig) {
      const cplx v = ncomp_ == 2 ? rho[0][ig] - rho[1][ig] : rho[1 + m][ig];
      x.push_back(v.real());
      x.push_back(v.imag());
    }
  for (int a = 0; a < c_.natoms(); ++a)
    for (size_t i = 0; i < D[a][0].size(); ++i) x.push_back(D[a][0].data()[i] + (ncomp_ == 2 ? D[a][1].data()[i] : 0.0));
  for (int m = 0; m < nmag; ++m)
    for (int a = 0; a < c_.natoms(); ++a)
      for (size_t i = 0; i < D[a][0].size(); ++i)
        x.push_back(ncomp_ == 2 ? D[a][0].data()[i] - D[a][1].data()[i] : D[a][1 + m].data()[i]);
  return x;
}

void Calculation::unpack(const std::vector<double>& x, RhoG& rho, DMat& D) const {
  const int ng = gs_.ng, nmag = ncomp_ == 2 ? 1 : (ncomp_ == 4 ? 3 : 0);
  size_t p = 0;
  std::vector<cplx> tot(ng);
  std::vector<std::vector<cplx>> mag(nmag, std::vector<cplx>(ng));
  for (int ig = 0; ig < ng; ++ig, p += 2) tot[ig] = cplx(x[p], x[p + 1]);
  for (int m = 0; m < nmag; ++m)
    for (int ig = 0; ig < ng; ++ig, p += 2) mag[m][ig] = cplx(x[p], x[p + 1]);
  for (int ig = 0; ig < ng; ++ig) {
    if (ncomp_ == 2) {
      rho[0][ig] = 0.5 * (tot[ig] + mag[0][ig]);
      rho[1][ig] = 0.5 * (tot[ig] - mag[0][ig]);
    } else {
      rho[0][ig] = tot[ig];
      for (int m = 0; m < nmag; ++m) rho[1 + m][ig] = mag[m][ig];
    }
  }
  size_t nd = 0;
  for (int a = 0; a < c_.natoms(); ++a) nd += D[a][0].size();
  size_t q = 0;
  for (int a = 0; a < c_.natoms(); ++a)
    for (size_t i = 0; i < D[a][0].size(); ++i, ++q) {
      const double t = x[p + q];
      if (ncomp_ == 2) {
        const double m = x[p + nd + q];
        D[a][0].data()[i] = 0.5 * (t + m);
        D[a][1].data()[i] = 0.5 * (t - m);
      } else {
        D[a][0].data()[i] = t;
        for (int m = 0; m < nmag; ++m) D[a][1 + m].data()[i] = x[p + (1 + m) * nd + q];
      }
    }
}

// =====================================================================================
//  Self-consistency loop
// =====================================================================================
void Calculation::set_exchange_active(bool on) {
  if (!hybrid_ || on == exx_active_) return;
  std::swap(xc_, xc_other_);
  exx_active_ = on;
}

// Hybrid functionals: outer loop over the exchange operator. Each outer iteration builds the compressed
// operator P_k = V_x[psi_k] from the current orbitals and runs an ordinary SCF cycle with a fixed operator U.
// In the inner cycle the exchange energy is linearized, E_x ~ Tr[gamma U] - 1/2 Tr[gamma_in U]; the difference to
// the true functional for the resulting orbitals, dE = E_x[new] - Tr[gamma_new U] + 1/2 Tr[gamma_in U], measures
// the convergence. The fixed-point iteration U = P_k converges slowly for metals and magnets; therefore U is an
// Anderson combination of the operators of the last iterations, U = sum_i a_i P_i with sum a_i = 1. V_x is linear
// in the density matrix, so this corresponds to mixing density matrices; the coefficients minimize the residual
// of the packed densities (smooth density and PAW density matrices), which serve as a proxy.
bool Calculation::scf() {
  if (!hybrid_) return scf_cycle();
  const bool verbose = p_.verbosity > 0;
  if (!exx_ready_) {  // start orbitals from the semilocal parent functional
    if (verbose) logf("\n  Hybrid functional: start with %s\n", xc_other_->name().c_str());
    set_exchange_active(false);
    exx_shift_ = 0.0;
    const bool ok = scf_cycle();
    set_exchange_active(true);
    if (stopped_by_time_) return false;
    if (!ok && verbose) logf("  WARNING: the semilocal start calculation did not converge\n");
    exx_hist_.clear();
    exx_coef_.clear();
  }
  const int keep = std::max(1, p_.exx_history);
  // input_pure: the inner cycle ran with the operator of a single set of orbitals (no mixture); polish: the
  // mixture has converged and one plain step follows. Convergence is accepted for plain steps only: a compressed
  // operator is exact on the span of its own orbitals, so only then the orbitals are eigenstates of the true
  // Hamiltonian to first order (needed for the forces).
  bool converged = false, inner_ok = true, input_pure = true, polish = false;
  // The mixture is used only close to convergence: far from it the map is strongly nonlinear and an extrapolation
  // can lead into another (e.g. magnetic) solution; until then plain fixed-point steps are taken.
  bool linear_regime = false;
  const double mix_threshold = 1e-5 * std::max(1.0, nelec_);
  for (int outer = 0;; ++outer) {
    // Tr[gamma_new P_i] for the stored operators, then the operator of the current orbitals
    int nold = static_cast<int>(exx_hist_.size());
    std::vector<double> cross(nold);
    for (int i = 0; i < nold; ++i) cross[i] = exchange_cross_energy(exx_hist_[i].ace);
    const double e_new = exchange_pass(true, nullptr);
    exx_energy_ = e_new;
    ExxEntry entry;
    entry.ace = std::move(ace_);
    entry.x = pack(rho_, D_);
    if (nold > 0) {
      double e_lin = 0.0;
      for (int i = 0; i < nold; ++i) e_lin += exx_coef_[i] * cross[i];
      const double dexx = e_new - e_lin - exx_shift_;
      // energies of the true functional for the current orbitals
      en_.total += dexx;
      en_.free += dexx;
      en_.sigma0 += dexx;
      en_.harris += dexx;
      en_.exx = e_new;
      if (verbose)
        logf("\n  Exact exchange, outer iteration %d: E_x(plane waves) = %.10f Ha, change of the energy %.3e Ha, free energy %.10f Ha\n",
             outer, e_new, dexx, en_.free);
      converged = std::abs(dexx) < p_.exx_tol && inner_ok && input_pure;
      polish = std::abs(dexx) < p_.exx_tol && inner_ok && !input_pure;
      if (!linear_regime && std::abs(dexx) < mix_threshold) {
        linear_regime = true;
        exx_hist_.clear();  // residuals of the nonlinear phase are not used
        nold = 0;
      }
      entry.has_residual = true;
      entry.r.resize(entry.x.size());
      for (size_t i = 0; i < entry.x.size(); ++i) entry.r[i] = entry.x[i] - exx_xin_[i];
    } else if (verbose) {
      logf("\n  Exact exchange: E_x(plane waves) = %.10f Ha for the start orbitals\n", e_new);
    }
    // append to the history; entries without residual (start orbitals) and the oldest ones are dropped
    exx_hist_.push_back(std::move(entry));
    RMatrix e2(nold + 1, nold + 1);
    for (int i = 0; i < nold; ++i) {
      for (int j = 0; j < nold; ++j) e2(i, j) = exx_e2_(i, j);
      e2(i, nold) = e2(nold, i) = cross[i];
    }
    e2(nold, nold) = 2.0 * e_new;
    while (static_cast<int>(exx_hist_.size()) > 1 && (!exx_hist_.front().has_residual || static_cast<int>(exx_hist_.size()) > keep)) {
      exx_hist_.erase(exx_hist_.begin());
      const int n = e2.rows() - 1;
      RMatrix t(n, n);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) t(i, j) = e2(i + 1, j + 1);
      e2 = t;
    }
    exx_e2_ = e2;
    const int n = static_cast<int>(exx_hist_.size());
    exx_coef_.assign(n, 0.0);
    exx_coef_[n - 1] = 1.0;  // operator of the current orbitals
    exx_ready_ = true;
    if (converged || outer >= p_.exx_max_iter) {
      exx_shift_ = -e_new;
      break;
    }
    // Anderson coefficients: minimize |sum_i a_i r_i| with sum_i a_i = 1
    input_pure = true;
    if (n > 1 && exx_hist_.front().has_residual && !polish && linear_regime) {
      RMatrix g(n, n);
      double tr = 0.0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) g(i, j) = g(j, i) = mixer_.dot(exx_hist_[i].r, exx_hist_[j].r);
        tr += g(i, i);
      }
      for (int i = 0; i < n; ++i) g(i, i) += 1e-10 * tr / n;
      std::vector<double> a(n, 1.0);
      double sum = 0.0, amax = 0.0;
      if (solve_real(g, a)) {
        for (double v : a) sum += v;
        for (double& v : a) {
          v /= sum;
          amax = std::max(amax, std::abs(v));
        }
        if (std::isfinite(amax) && amax < 10.0) {
          exx_coef_ = a;
          input_pure = false;
        }
      }
    }
    exx_xin_.assign(exx_hist_.back().x.size(), 0.0);
    exx_shift_ = 0.0;
    for (int i = 0; i < n; ++i) {
      for (size_t k = 0; k < exx_xin_.size(); ++k) exx_xin_[k] += exx_coef_[i] * exx_hist_[i].x[k];
      for (int j = 0; j < n; ++j) exx_shift_ -= 0.5 * exx_coef_[i] * exx_coef_[j] * exx_e2_(i, j);
    }
    setup_mixer();  // the Pulay history of the previous operator is of no use
    inner_ok = scf_cycle();
    if (stopped_by_time_) return false;
  }
  scf_converged_ = converged;
  if (verbose) {
    logf(converged ? "\n  Hybrid functional converged\n" : "\n  WARNING: hybrid functional NOT converged\n");
    logf("    exact exchange (plane-wave part) %18.10f\n", en_.exx);
    logf("    total energy E         %20.10f\n", en_.total);
    logf("    free energy  E-TS      %20.10f   (%.8f eV)\n", en_.free, en_.free * hartree_in_ev);
    logf("    E(sigma -> 0)          %20.10f\n", en_.sigma0);
  }
  return converged;
}

bool Calculation::scf_cycle() {
  ScopedTimer timer("scf");
  LSDA_CHECK(have_density_, "scf: no start density");
  const bool verbose = p_.verbosity > 0;
  Functional f_in = evaluate(rho_, D_, true);
  if (verbose)
    logf("\n  iter      free energy (Ha)     E_KS - E_Harris      |residual|^2    magnet.   eig.resid   time(s)\n");
  double dr2 = resume_dr2_ > 0.0 ? resume_dr2_ : 1.0, e_prev = 0.0;
  resume_dr2_ = -1.0;
  bool converged = false;
  stopped_by_time_ = false;
  scf_converged_ = false;
  const double t0 = wall_time();
  for (int iter = 1; iter <= p_.max_scf; ++iter) {
    n_scf_ = iter;
    const double t_iter = wall_time();
    // eigensolver tolerance follows the density residual
    // the tightest eigensolver tolerance must allow the requested SCF accuracy to be reached
    const double eig_floor = std::min(p_.eig_tol, 0.3 * p_.eig_tol_factor * std::sqrt(p_.scf_tol / std::max(1.0, nelec_)));
    double tol = std::clamp(p_.eig_tol_factor * std::sqrt(dr2 / std::max(1.0, nelec_)), eig_floor, p_.eig_tol_max);
    int eig_iters = first_solve_ ? p_.eig_max_iter_first : p_.eig_max_iter;
    if (first_solve_ && restarted_) {  // the density is already (nearly) converged: bands first
      tol = std::min(tol, 1e-5);
      eig_iters = std::max(eig_iters, 20);
    }
    double max_res;
    int nconv;
    solve_bands(tol, eig_iters, max_res, nconv);
    first_solve_ = false;
    restarted_ = false;

    occ_ = compute_occupations(eig_, kweight_, nspin_, nelec_, smearing_, p_.width, p_.fixed_moment, &tetra_, noncol_ ? 1.0 : 0.0);
    have_occ_ = true;

    RhoG rho_out;
    DMat D_out;
    compute_density(rho_out, D_out);
    const Functional f_out = evaluate(rho_out, D_out, false);
    // exx: linearized exact-exchange energy (the band energy contains sum f <psi|V_x|psi>)
    const double exx = exx_active_ ? exx_shift_ : 0.0;
    const double e_harris = occ_.band_energy - double_counting(rho_, D_) + f_in.total() + exx;
    const double e_ks = occ_.band_energy - double_counting(rho_out, D_out) + f_out.total() + exx;

    en_.band = occ_.band_energy;
    en_.hartree = f_out.hartree;
    en_.xc = f_out.xc;
    en_.zero = f_out.zero;
    en_.onsite = f_out.onsite;
    en_.entropy_term = occ_.entropy_term;
    en_.total = e_ks;
    en_.harris = e_harris;
    en_.free = e_ks + occ_.entropy_term;
    switch (smearing_) {
      case Smearing::FermiDirac:
      case Smearing::Gaussian: en_.sigma0 = e_ks + 0.5 * occ_.entropy_term; break;
      case Smearing::MethfesselPaxton1: en_.sigma0 = e_ks + (2.0 / 3.0) * occ_.entropy_term; break;
      case Smearing::Cold:
      case Smearing::Tetrahedron: en_.sigma0 = en_.free; break;
    }

    const std::vector<double> x_in = pack(rho_, D_), x_out = pack(rho_out, D_out);
    std::vector<double> r(x_in.size());
    for (size_t i = 0; i < r.size(); ++i) r[i] = x_out[i] - x_in[i];
    dr2 = mixer_.dot(r, r);
    if (par::size() > 1) {  // identical decisions on all processes: take the steering numbers from the root
      double v[12] = {dr2, max_res, en_.band, en_.hartree, en_.xc, en_.zero, en_.onsite, en_.entropy_term, en_.total, en_.free, en_.sigma0, en_.harris};
      par::bcast(v, 12);
      dr2 = v[0], max_res = v[1], en_.band = v[2], en_.hartree = v[3], en_.xc = v[4], en_.zero = v[5], en_.onsite = v[6];
      en_.entropy_term = v[7], en_.total = v[8], en_.free = v[9], en_.sigma0 = v[10], en_.harris = v[11];
    }
    last_dr2_ = dr2;

    if (verbose)
      logf("  %4d  %20.10f  %18.3e  %16.3e  %9.4f  %10.2e  %8.1f\n", iter, en_.free, e_ks - e_harris, dr2,
           total_magnetization(), max_res, wall_time() - t0);

    const bool energy_ok = iter > 1 && std::abs(en_.free - e_prev) < 1e-3 * std::max(1.0, std::sqrt(nelec_));
    e_prev = en_.free;
    if (dr2 < p_.scf_tol && energy_ok && max_res < 1e-4) {
      converged = true;
      // adopt the output density: wavefunctions, density and potential are then mutually consistent
      rho_ = std::move(rho_out);
      D_ = std::move(D_out);
      f_in = evaluate(rho_, D_, true);
      break;
    }
    std::vector<double> x_new = mixer_.mix(x_in, x_out);
    par::bcast(x_new.data(), x_new.size());  // keeps the replicated densities bitwise identical
    unpack(x_new, rho_, D_);
    f_in = evaluate(rho_, D_, true);
    // wall-time limit: stop if the next iteration would not fit (state is consistent for a restart)
    int stop = p_.max_seconds > 0.0 && wall_time() + 1.2 * (wall_time() - t_iter) > p_.max_seconds && iter < p_.max_scf;
    par::bcast(&stop, 1);  // the clock of the root process decides
    if (stop) {
      stopped_by_time_ = true;
      break;
    }
  }
  scf_converged_ = converged;
  if (stopped_by_time_) {
    if (verbose) logf("\n  Wall-time limit (max_seconds = %.0f) reached after %d SCF iterations: stopping for restart\n", p_.max_seconds, n_scf_);
    return false;
  }
  if (verbose) {
    logf(converged ? "\n  SCF converged in %d iterations\n" : "\n  WARNING: SCF NOT converged after %d iterations\n", n_scf_);
    logf("\n  Energies (Ha)\n");
    logf("    band energy            %20.10f\n", en_.band);
    logf("    Hartree (smooth)       %20.10f\n", en_.hartree);
    logf("    xc (smooth)            %20.10f\n", en_.xc);
    logf("    zero potential         %20.10f\n", en_.zero);
    logf("    on-site corrections    %20.10f\n", en_.onsite);
    logf("    -TS                    %20.10f\n", en_.entropy_term);
    logf("    total energy E         %20.10f\n", en_.total);
    logf("    free energy  E-TS      %20.10f   (%.8f eV)\n", en_.free, en_.free * hartree_in_ev);
    logf("    E(sigma -> 0)          %20.10f\n", en_.sigma0);
    if (nspin_ == 2 && p_.fixed_moment)
      logf("  Fermi levels             %12.6f / %12.6f Ha\n", occ_.efermi[0], occ_.efermi[1]);
    else
      logf("  Fermi level              %12.6f Ha  (%.4f eV)\n", occ_.efermi[0], occ_.efermi[0] * hartree_in_ev);
    for (int a = 0; a < c_.natoms(); ++a) {  // DFT+U: occupations of the Hubbard shells
      const PawSetup& su = c_.setup(a);
      if (su.hubbard_l < 0 || su.hubbard_u == 0.0) continue;
      std::string line = strf("  Hubbard shell atom %d (%s, l = %d, U_eff = %.3f eV): occupation", a + 1,
                              c_.species[c_.atoms[a].species].name.c_str(), su.hubbard_l, su.hubbard_u * hartree_in_ev);
      for (int s = 0; s < nspin_; ++s) {
        const RMatrix n = su.occupation_matrix(D_[a][s]);
        double tr = 0.0;
        for (int m = 0; m < n.rows(); ++m) tr += n(m, m);
        line += strf(nspin_ == 2 ? (s == 0 ? " up %.4f" : ", down %.4f") : " %.4f", tr);
      }
      logf("%s\n", line.c_str());
    }
    if (noncol_) {
      const Vec3 mt = magnetization_vector();
      logf("  Total magnetization      (%.6f, %.6f, %.6f) mu_B,  |M| = %.6f\n", mt[0], mt[1], mt[2], norm(mt));
      const std::vector<Vec3> mv = local_moment_vectors();
      logf("  Moment vectors inside the PAW spheres (mu_B):\n");
      for (int a = 0; a < c_.natoms(); ++a)
        logf("    %4d %-4s %10.5f %10.5f %10.5f   |m| = %.5f\n", a + 1, c_.species[c_.atoms[a].species].name.c_str(), mv[a][0], mv[a][1],
             mv[a][2], norm(mv[a]));
    }
    if (nspin_ == 2) {
      logf("  Total magnetization      %12.6f mu_B\n", total_magnetization());
      const std::vector<double> m = local_moments();
      logf("  Moments inside the PAW spheres (mu_B):\n");
      for (int a = 0; a < c_.natoms(); ++a)
        logf("    %4d %-4s %10.5f\n", a + 1, c_.species[c_.atoms[a].species].name.c_str(), m[a]);
    }
  }
  return converged;
}

void Calculation::write_dos(const std::string& path, double width, int npoints) const {
  LSDA_CHECK(have_occ_, "write_dos: run scf() first");
  LSDA_CHECK(!noncol_, "the density of states is not implemented for non-collinear calculations");
  LSDA_CHECK(width > 0.0 && npoints > 1, "write_dos: width and number of points must be positive");
  const int nat = c_.natoms(), nk = static_cast<int>(kpts_.size());
  int lmax = 0;
  std::vector<int> offset(nat);
  for (int a = 0, off = 0; a < nat; ++a) {
    offset[a] = off;
    off += c_.setup(a).ni;
    lmax = std::max(lmax, c_.setup(a).lmax);
  }
  const int nl = lmax + 1, ncol = nspin_ * (1 + nat * nl);  // per spin: total, then (atom, l)
  double emin = 1e300, emax = -1e300;
  for (const auto& e : eig_)
    for (double v : e) {
      emin = std::min(emin, v);
      emax = std::max(emax, v);
    }
  emin -= 6.0 * width;
  emax += 6.0 * width;
  const double de = (emax - emin) / (npoints - 1), deg = nspin_ == 1 ? 2.0 : 1.0;
  std::vector<double> dos(static_cast<size_t>(npoints) * ncol, 0.0);
  // sphere weights of every state for the atoms: sum_{ij in (a,l)} P_i^* <phi_i|phi_j> P_j; with reduced
  // k-points the weight of atom a is the average over the symmetry-equivalent atoms g(a)
  std::vector<double> w(static_cast<size_t>(nat) * nl), wsym(static_cast<size_t>(nat) * nl);
  for (int ik = 0; ik < nk; ++ik) {
    if (!par::owns(ik)) continue;
    for (int s = 0; s < nspin_; ++s) {
      const size_t ks = static_cast<size_t>(ik) * nspin_ + s;
      const CMatrix& P = proj_psi_[ks];
      for (int n = 0; n < nbands_; ++n) {
        std::fill(w.begin(), w.end(), 0.0);
        for (int a = 0; a < nat; ++a) {
          const PawSetup& su = c_.setup(a);
          for (int i = 0; i < su.ni; ++i)
            for (int j = 0; j < su.ni; ++j)
              if (su.L_i[i] == su.L_i[j])
                w[a * nl + su.l_i[i]] += su.dN(i, j) * (std::conj(P(offset[a] + i, n)) * P(offset[a] + j, n)).real();
        }
        std::fill(wsym.begin(), wsym.end(), 0.0);
        for (const SymOp& op : ops_)
          for (int a = 0; a < nat; ++a)
            for (int l = 0; l < nl; ++l) wsym[a * nl + l] += w[op.atom_map[a] * nl + l];
        const double norm = deg * kweight_[ik] / (std::sqrt(pi) * width) / static_cast<double>(ops_.size());
        const double e = eig_[ks][n];
        const int i0 = std::max(0, static_cast<int>((e - 6.0 * width - emin) / de));
        const int i1 = std::min(npoints - 1, static_cast<int>((e + 6.0 * width - emin) / de) + 1);
        for (int i = i0; i <= i1; ++i) {
          const double x = (emin + i * de - e) / width;
          const double g = norm * std::exp(-x * x);
          double* row = &dos[static_cast<size_t>(i) * ncol + s * (1 + nat * nl)];
          row[0] += g * static_cast<double>(ops_.size());
          for (int c = 0; c < nat * nl; ++c) row[1 + c] += g * wsym[c];
        }
      }
    }
  }
  par::sum(dos.data(), dos.size());
  if (!par::root()) return;
  std::FILE* f = std::fopen(path.c_str(), "w");
  LSDA_CHECK(f != nullptr, "cannot write '" + path + "'");
  const char lname[] = "spdfg";
  std::fprintf(f, "# density of states, Gaussian width %.5f Ha; energies in eV relative to the Fermi level (%.6f Ha)\n", width,
               occ_.efermi[0]);
  std::fprintf(f, "# states/eV per cell%s; projections: all-electron weight inside the PAW spheres\n",
               nspin_ == 1 ? " (both spins)" : " and spin");
  std::fprintf(f, "# columns: 1 E-E_F");
  int col = 2;
  for (int s = 0; s < nspin_; ++s) {
    const char* sp = nspin_ == 1 ? "" : (s == 0 ? "_up" : "_dn");
    std::fprintf(f, " | %d total%s", col++, sp);
    for (int a = 0; a < nat; ++a)
      for (int l = 0; l < nl; ++l)
        std::fprintf(f, " | %d %s%d_%c%s", col++, c_.species[c_.atoms[a].species].name.c_str(), a + 1, lname[std::min(l, 4)], sp);
  }
  std::fprintf(f, "\n");
  for (int i = 0; i < npoints; ++i) {
    std::fprintf(f, "%12.6f", (emin + i * de - occ_.efermi[0]) * hartree_in_ev);
    for (int c = 0; c < ncol; ++c) std::fprintf(f, " %12.6e", dos[static_cast<size_t>(i) * ncol + c] / hartree_in_ev);
    std::fprintf(f, "\n");
  }
  std::fclose(f);
}

void Calculation::write_cube(const std::string& path, bool magnetization) const {
  LSDA_CHECK(have_density_, "write_cube: no density");
  if (!par::root()) return;
  const size_t nf = fft_f_.size();
  CVec work(nf, cplx(0.0));
  for (int ig = 0; ig < gs_.ng; ++ig) {
    cplx v;
    if (magnetization)
      v = ncomp_ == 2 ? rho_[0][ig] - rho_[1][ig] : (noncol_ ? rho_[3][ig] : cplx(0.0));  // non-collinear: z component
    else
      v = rho_[0][ig] + (ncomp_ == 2 ? rho_[1][ig] : cplx(0.0)) + nct_G_[ig];
    work[gs_.idx_fine[ig]] = v;
  }
  fft_f_.backward(work.data());
  std::FILE* f = std::fopen(path.c_str(), "w");
  LSDA_CHECK(f != nullptr, "cannot write '" + path + "'");
  std::fprintf(f, "lsda: %s (electrons/bohr^3)\n", magnetization ? "pseudo magnetization density n_up - n_down" : "pseudo electron density, valence + pseudo core");
  std::fprintf(f, "smooth part only: the augmentation inside the PAW spheres is not included\n");
  std::fprintf(f, "%5d %12.6f %12.6f %12.6f\n", c_.natoms(), 0.0, 0.0, 0.0);
  for (int d = 0; d < 3; ++d)
    std::fprintf(f, "%5d %12.6f %12.6f %12.6f\n", fft_f_.n(d), c_.A[d][0] / fft_f_.n(d), c_.A[d][1] / fft_f_.n(d), c_.A[d][2] / fft_f_.n(d));
  for (int a = 0; a < c_.natoms(); ++a) {
    Vec3 fr = c_.atoms[a].frac;
    for (double& x : fr) x -= std::floor(x);
    const Vec3 r = c_.cart(fr);
    std::fprintf(f, "%5d %12.6f %12.6f %12.6f %12.6f\n", static_cast<int>(std::lround(c_.setup(a).Z)), c_.setup(a).n_valence, r[0], r[1], r[2]);
  }
  for (size_t i = 0; i < nf; ++i) std::fprintf(f, "%13.5e%s", work[i].real(), (i % 6 == 5 || (i + 1) % fft_f_.n(2) == 0) ? "\n" : "");
  std::fclose(f);
}

std::vector<std::vector<double>> Calculation::eigenvalues_at(const Vec3& kfrac, double tol, int max_iter) {
  LSDA_CHECK(have_occ_, "eigenvalues_at: run scf() first");
  LSDA_CHECK(!noncol_, "band structures are not implemented for non-collinear calculations");
  LSDA_CHECK(!hybrid_, "band structures are not implemented for hybrid functionals (the exchange operator exists on the k-grid only)");
  const int nat = c_.natoms();
  const PWBasis b = make_pw_basis(c_, KPoint{kfrac, 0.0}, p_.ecut, fft_c_, p_.ecut_smooth);
  LSDA_CHECK(b.npw >= nbands_, "eigenvalues_at: fewer plane waves than bands");
  const Projectors proj = make_projectors(c_, b);
  std::vector<std::vector<double>> out(nspin_);
  std::mt19937_64 rng(p_.seed);
  std::uniform_real_distribution<double> u(-0.5, 0.5);
  for (int s = 0; s < nspin_; ++s) {
    std::vector<const RMatrix*> dh(nat), ds(nat);
    for (int a = 0; a < nat; ++a) {
      dh[a] = &dH_[a][s];
      ds[a] = &dS_[a];
    }
    const Hamiltonian h(b, fft_c_, vcoarse_[s].data(), proj, dh, ds);
    // start: lowest states spanned by the atomic orbitals, padded with random vectors
    CMatrix ao = atomic_orbitals(b);
    const int nao = ao.cols(), nrand = std::max(0, nbands_ - nao) + (nao < nbands_ ? 0 : std::min(4, nbands_));
    CMatrix x(b.npw, nao + nrand);
    std::copy(ao.data(), ao.data() + ao.size(), x.data());
    for (int n = 0; n < nrand; ++n)
      for (int ig = 0; ig < b.npw; ++ig) x(ig, nao + n) = cplx(u(rng), u(rng)) / (1.0 + 2.0 * b.kin[ig]);
    if (!(nao + nrand <= b.npw && subspace_start(h, x, nbands_))) {
      x.resize(b.npw, nbands_);
      for (int n = 0; n < nbands_; ++n)
        for (int ig = 0; ig < b.npw; ++ig) x(ig, n) = cplx(u(rng), u(rng)) / (1.0 + 2.0 * b.kin[ig]);
    }
    const std::vector<double> tb(nbands_, tol);
    ppcg(h, x, out[s], tb, std::max(1, nbands_ - std::max(2, nbands_ / 8)), max_iter, nullptr, p_.ppcg_block);
  }
  return out;
}

std::vector<Vec3> Calculation::local_moment_vectors() const {
  std::vector<Vec3> m(c_.natoms(), Vec3{0, 0, 0});
  for (int a = 0; a < c_.natoms(); ++a) {
    const PawSetup& s = c_.setup(a);
    for (int ij = 0; ij < s.ni * s.ni; ++ij) {
      if (noncol_)
        for (int d = 0; d < 3; ++d) m[a][d] += D_[a][1 + d].data()[ij] * s.dN.data()[ij];
      else if (ncomp_ == 2)
        m[a][2] += (D_[a][0].data()[ij] - D_[a][1].data()[ij]) * s.dN.data()[ij];
    }
  }
  return m;
}

std::vector<double> Calculation::local_moments() const {
  std::vector<double> m(c_.natoms(), 0.0);
  const std::vector<Vec3> v = local_moment_vectors();
  for (int a = 0; a < c_.natoms(); ++a) m[a] = noncol_ ? norm(v[a]) : v[a][2];
  return m;
}

// total moment of the all-electron density: smooth part plus the augmentation, M_c = Omega m~_c(G=0) + sum D^c_ij dS_ij
Vec3 Calculation::magnetization_vector() const {
  Vec3 m{0, 0, 0};
  if (!noncol_) {
    m[2] = occ_.n_spin[0] - occ_.n_spin[1];
    return m;
  }
  for (int d = 0; d < 3; ++d) {
    m[d] = rho_[1 + d][0].real() * c_.volume;
    for (int a = 0; a < c_.natoms(); ++a) {
      const PawSetup& s = c_.setup(a);
      for (int ij = 0; ij < s.ni * s.ni; ++ij) m[d] += D_[a][1 + d].data()[ij] * s.dS.data()[ij];
    }
  }
  return m;
}

double Calculation::total_magnetization() const { return noncol_ ? norm(magnetization_vector()) : occ_.n_spin[0] - occ_.n_spin[1]; }

std::vector<double> Calculation::local_charges() const {
  std::vector<double> q(c_.natoms(), 0.0);
  for (int a = 0; a < c_.natoms(); ++a) {
    const PawSetup& s = c_.setup(a);
    for (int sp = 0; sp < (ncomp_ == 2 ? 2 : 1); ++sp)
      for (int ij = 0; ij < s.ni * s.ni; ++ij) q[a] += D_[a][sp].data()[ij] * s.dN.data()[ij];
  }
  return q;
}

bool Calculation::adopt_state(Calculation&& o) {
  if (noncol_ || o.noncol_) return false;  // spinor layout: start from scratch
  if (o.nspin_ != nspin_ || o.nbands_ != nbands_ || o.basis_.size() != basis_.size() || o.c_.natoms() != c_.natoms() ||
      !o.have_density_ || o.psi_.size() != psi_.size())
    return false;
  for (int a = 0; a < c_.natoms(); ++a)
    if (o.c_.setup(a).ni != c_.setup(a).ni) return false;
  for (size_t ik = 0; ik < basis_.size(); ++ik)
    for (int d = 0; d < 3; ++d)
      if (std::abs(o.basis_[ik].kfrac[d] - basis_[ik].kfrac[d]) > 1e-10) return false;

  // wavefunctions: copy the coefficients of common plane waves (identified by their integer G vectors)
  for (size_t ik = 0; ik < basis_.size(); ++ik) {
    if (!par::owns(static_cast<int>(ik))) continue;
    const PWBasis &bn = basis_[ik], &bo = o.basis_[ik];
    std::vector<int> map(bn.npw, -1);
    if (bn.mill == bo.mill) {
      for (int ig = 0; ig < bn.npw; ++ig) map[ig] = ig;
    } else {
      std::vector<int> where(o.fft_c_.size(), -1);
      for (int ig = 0; ig < bo.npw; ++ig) where[bo.idx[ig]] = ig;
      for (int ig = 0; ig < bn.npw; ++ig) {
        const int* m = &bn.mill[3 * ig];
        if (2 * std::abs(m[0]) < o.fft_c_.n(0) && 2 * std::abs(m[1]) < o.fft_c_.n(1) && 2 * std::abs(m[2]) < o.fft_c_.n(2))
          map[ig] = where[o.fft_c_.index(m[0], m[1], m[2])];
      }
    }
    for (int s = 0; s < nspin_; ++s) {
      const size_t ks = ik * nspin_ + s;
      const CMatrix& xo = o.psi_[ks];
      CMatrix xn(bn.npw, nbands_);
      for (int n = 0; n < nbands_; ++n)
        for (int ig = 0; ig < bn.npw; ++ig)
          if (map[ig] >= 0) xn(ig, n) = xo(map[ig], n);
      psi_[ks] = std::move(xn);
    }
  }
  // density: Omega * n(G), the number of electrons per plane-wave component, is carried over
  const double scale = o.c_.volume / c_.volume;
  for (int s = 0; s < nspin_; ++s) {
    std::vector<cplx> r(gs_.ng, 0.0);
    for (int ig = 0; ig < gs_.ng; ++ig) {
      const int* m = &gs_.m[3 * ig];
      if (2 * std::abs(m[0]) >= o.fft_f_.n(0) || 2 * std::abs(m[1]) >= o.fft_f_.n(1) || 2 * std::abs(m[2]) >= o.fft_f_.n(2)) continue;
      const int j = o.gs_.lookup[o.fft_f_.index(m[0], m[1], m[2])];
      if (j >= 0) r[ig] = o.rho_[s][j] * scale;
    }
    rho_[s] = std::move(r);
  }
  D_ = std::move(o.D_);
  eig_ = std::move(o.eig_);
  occ_ = std::move(o.occ_);
  have_occ_ = o.have_occ_;
  have_density_ = true;
  first_solve_ = o.first_solve_;
  proj_valid_ = false;
  mixer_.reset();
  return true;
}

bool Calculation::adopt_density(const Calculation& o) {
  if (o.ncomp_ != ncomp_ || o.gs_.ng != gs_.ng || o.gs_.m != gs_.m || o.c_.natoms() != c_.natoms() || !o.have_density_) return false;
  for (int a = 0; a < c_.natoms(); ++a)
    if (o.c_.setup(a).ni != c_.setup(a).ni) return false;
  rho_ = o.rho_;
  D_ = o.D_;
  have_density_ = true;
  // the adopted density is (nearly) self-consistent while the wavefunctions start from scratch: converge
  // the bands thoroughly in the first step, otherwise eigensolver noise dominates the first residuals
  restarted_ = true;
  mixer_.reset();
  return true;
}

// =====================================================================================
//  Checkpoints
// =====================================================================================
namespace {
const char chk_magic[8] = {'L', 'S', 'D', 'A', 'C', 'H', 'K', '2'};
struct ChkHeader {
  char magic[8];
  int nspin, nat, ng, scf_converged;
  double ecut_dense;
  double lattice[9];
};
struct FileCloser {
  std::FILE* f;
  ~FileCloser() { if (f) std::fclose(f); }
};
}  // namespace

void Calculation::save_checkpoint(const std::string& path) const {
  const bool with_wfc = p_.checkpoint_wfc && !first_solve_ && !noncol_;  // spinor wavefunctions are not stored
  if (!par::root()) {  // the root process writes the file; the others hand over their wavefunctions
    if (with_wfc)
      for (size_t ik = 0; ik < basis_.size(); ++ik)
        if (par::owns(static_cast<int>(ik)))
          for (int s = 0; s < nspin_; ++s) {
            const size_t ks = ik * nspin_ + s;
            par::send(psi_[ks].data(), psi_[ks].size(), 0, static_cast<int>(ks));
          }
    return;
  }
  const std::string tmp = path + ".tmp";
  {
    FileCloser fc{std::fopen(tmp.c_str(), "wb")};
    LSDA_CHECK(fc.f != nullptr, "cannot write checkpoint '" + tmp + "'");
    ChkHeader h{};
    std::memcpy(h.magic, chk_magic, 8);
    h.nspin = ncomp_;
    h.nat = c_.natoms();
    h.ng = gs_.ng;
    h.scf_converged = scf_converged_ ? 1 : 0;
    h.ecut_dense = p_.ecut_dense;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) h.lattice[3 * i + j] = c_.A[i][j];
    bool ok = std::fwrite(&h, sizeof h, 1, fc.f) == 1;
    for (const Atom& a : c_.atoms) ok = ok && std::fwrite(a.frac.data(), sizeof(double), 3, fc.f) == 3;
    for (int s = 0; s < ncomp_; ++s) ok = ok && std::fwrite(rho_[s].data(), sizeof(cplx), gs_.ng, fc.f) == static_cast<size_t>(gs_.ng);
    for (int a = 0; a < c_.natoms(); ++a) {
      const int ni = c_.setup(a).ni;
      ok = ok && std::fwrite(&ni, sizeof ni, 1, fc.f) == 1;
      for (int s = 0; s < ncomp_; ++s) ok = ok && std::fwrite(D_[a][s].data(), sizeof(double), D_[a][s].size(), fc.f) == D_[a][s].size();
    }
    // wavefunctions (optional): per k-point the integer G vectors, per spin eigenvalues and coefficients
    const int has_wfc = with_wfc ? 1 : 0, nk = static_cast<int>(basis_.size());
    ok = ok && std::fwrite(&has_wfc, sizeof has_wfc, 1, fc.f) == 1;
    if (has_wfc) {
      ok = ok && std::fwrite(&nk, sizeof nk, 1, fc.f) == 1 && std::fwrite(&nbands_, sizeof nbands_, 1, fc.f) == 1;
      for (int ik = 0; ik < nk && ok; ++ik) {
        const PWBasis& b = basis_[ik];
        ok = ok && std::fwrite(b.kfrac.data(), sizeof(double), 3, fc.f) == 3 && std::fwrite(&b.npw, sizeof b.npw, 1, fc.f) == 1 &&
             std::fwrite(b.mill.data(), sizeof(int), b.mill.size(), fc.f) == b.mill.size();
        for (int s = 0; s < nspin_ && ok; ++s) {
          const size_t ks = static_cast<size_t>(ik) * nspin_ + s;
          const cplx* data = psi_[ks].data();
          const size_t count = static_cast<size_t>(b.npw) * nbands_;
          std::vector<cplx> remote;
          if (!par::owns(ik)) {  // wavefunctions of another process
            remote.resize(count);
            par::recv(remote.data(), count, par::owner(ik), static_cast<int>(ks));
            data = remote.data();
          }
          ok = ok && std::fwrite(eig_[ks].data(), sizeof(double), nbands_, fc.f) == static_cast<size_t>(nbands_) &&
               std::fwrite(data, sizeof(cplx), count, fc.f) == count;
        }
      }
    }
    // unfinished SCF cycle: residual and Pulay history, so that a restart continues where it stopped
    const int has_mixer = has_wfc && !scf_converged_ ? 1 : 0;
    ok = ok && std::fwrite(&has_mixer, sizeof has_mixer, 1, fc.f) == 1;
    if (has_mixer) ok = ok && std::fwrite(&last_dr2_, sizeof last_dr2_, 1, fc.f) == 1 && mixer_.save(fc.f);
    LSDA_CHECK(ok, "error while writing checkpoint '" + tmp + "'");
  }
  LSDA_CHECK(std::rename(tmp.c_str(), path.c_str()) == 0, "cannot rename checkpoint to '" + path + "'");
}

bool Calculation::read_checkpoint_structure(const std::string& path, int natoms, std::vector<Vec3>& frac, Mat3& lattice) {
  FileCloser fc{std::fopen(path.c_str(), "rb")};
  if (!fc.f) return false;
  ChkHeader h{};
  if (std::fread(&h, sizeof h, 1, fc.f) != 1 || std::memcmp(h.magic, chk_magic, 8) != 0 || h.nat != natoms) return false;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) lattice[i][j] = h.lattice[3 * i + j];
  frac.resize(natoms);
  for (Vec3& v : frac)
    if (std::fread(v.data(), sizeof(double), 3, fc.f) != 3) return false;
  return true;
}

bool Calculation::load_checkpoint(const std::string& path) {
  FileCloser fc{std::fopen(path.c_str(), "rb")};
  if (!fc.f) return false;
  ChkHeader h{};
  if (std::fread(&h, sizeof h, 1, fc.f) != 1 || std::memcmp(h.magic, chk_magic, 8) != 0) return false;
  if (h.nspin != ncomp_ || h.nat != c_.natoms() || h.ng != gs_.ng || std::abs(h.ecut_dense - p_.ecut_dense) > 1e-9) return false;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (std::abs(h.lattice[3 * i + j] - c_.A[i][j]) > 1e-9) return false;
  for (const Atom& a : c_.atoms) {  // the density belongs to the stored positions
    Vec3 v;
    if (std::fread(v.data(), sizeof(double), 3, fc.f) != 3) return false;
    for (int d = 0; d < 3; ++d)
      if (std::abs(v[d] - a.frac[d]) > 1e-9) return false;
  }
  RhoG rho(ncomp_, std::vector<cplx>(gs_.ng));
  DMat D(c_.natoms(), std::vector<RMatrix>(ncomp_));
  for (int s = 0; s < ncomp_; ++s)
    if (std::fread(rho[s].data(), sizeof(cplx), gs_.ng, fc.f) != static_cast<size_t>(gs_.ng)) return false;
  for (int a = 0; a < c_.natoms(); ++a) {
    int ni = 0;
    if (std::fread(&ni, sizeof ni, 1, fc.f) != 1 || ni != c_.setup(a).ni) return false;
    for (int s = 0; s < ncomp_; ++s) {
      D[a][s].resize(ni, ni);
      if (std::fread(D[a][s].data(), sizeof(double), D[a][s].size(), fc.f) != D[a][s].size()) return false;
    }
  }
  rho_ = std::move(rho);
  D_ = std::move(D);
  have_density_ = true;
  have_occ_ = false;
  restarted_ = h.scf_converged != 0;  // converged density: converge the bands thoroughly in the first step
  mixer_.reset();

  // wavefunctions, if stored and compatible (otherwise atomic-orbital start vectors are used)
  int has_wfc = 0, nk = 0, nb = 0;
  if (std::fread(&has_wfc, sizeof has_wfc, 1, fc.f) != 1 || !has_wfc) return true;
  if (std::fread(&nk, sizeof nk, 1, fc.f) != 1 || std::fread(&nb, sizeof nb, 1, fc.f) != 1) return true;
  if (nk != static_cast<int>(basis_.size()) || nb != nbands_) return true;
  std::vector<CMatrix> psi(psi_.size());
  std::vector<std::vector<double>> eig(psi_.size(), std::vector<double>(nbands_));
  for (int ik = 0; ik < nk; ++ik) {
    const PWBasis& b = basis_[ik];
    Vec3 kf;
    int npw = 0;
    if (std::fread(kf.data(), sizeof(double), 3, fc.f) != 3 || std::fread(&npw, sizeof npw, 1, fc.f) != 1 || npw <= 0) return true;
    for (int d = 0; d < 3; ++d)
      if (std::abs(kf[d] - b.kfrac[d]) > 1e-10) return true;
    std::vector<int> mill(3 * static_cast<size_t>(npw));
    if (std::fread(mill.data(), sizeof(int), mill.size(), fc.f) != mill.size()) return true;
    // position of every stored plane wave in the present basis (-1: not in the basis)
    std::vector<int> where(fft_c_.size(), -1), map(npw, -1);
    for (int ig = 0; ig < b.npw; ++ig) where[b.idx[ig]] = ig;
    for (int ig = 0; ig < npw; ++ig) {
      const int* m = &mill[3 * static_cast<size_t>(ig)];
      if (2 * std::abs(m[0]) < fft_c_.n(0) && 2 * std::abs(m[1]) < fft_c_.n(1) && 2 * std::abs(m[2]) < fft_c_.n(2))
        map[ig] = where[fft_c_.index(m[0], m[1], m[2])];
    }
    std::vector<cplx> buf(static_cast<size_t>(npw) * nbands_);
    for (int s = 0; s < nspin_; ++s) {
      const size_t ks = static_cast<size_t>(ik) * nspin_ + s;
      if (std::fread(eig[ks].data(), sizeof(double), nbands_, fc.f) != static_cast<size_t>(nbands_)) return true;
      if (!par::owns(ik)) {  // belongs to another process
        if (std::fseek(fc.f, static_cast<long>(buf.size() * sizeof(cplx)), SEEK_CUR) != 0) return true;
        continue;
      }
      if (std::fread(buf.data(), sizeof(cplx), buf.size(), fc.f) != buf.size()) return true;
      psi[ks].resize(b.npw, nbands_);
      for (int n = 0; n < nbands_; ++n)
        for (int ig = 0; ig < npw; ++ig)
          if (map[ig] >= 0) psi[ks](map[ig], n) = buf[static_cast<size_t>(n) * npw + ig];
    }
  }
  psi_ = std::move(psi);
  eig_ = std::move(eig);
  first_solve_ = false;
  proj_valid_ = false;
  int has_mixer = 0;
  double dr2 = 0.0;
  if (std::fread(&has_mixer, sizeof has_mixer, 1, fc.f) == 1 && has_mixer && std::fread(&dr2, sizeof dr2, 1, fc.f) == 1 &&
      mixer_.load(fc.f))
    resume_dr2_ = dr2;
  return true;
}

}  // namespace lsda
