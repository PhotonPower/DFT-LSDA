// PAW dataset (PAW-XML format, e.g. the JTH table) and all atom-centred ("on-site") quantities.
//
// Formalism: Bloechl's PAW method in the formulation of Mortensen, Hansen, Jacobsen, PRB 71, 035109 (2005):
//   E = E~[n~, rho~] + sum_a dE_a(D_a),
//   rho~ = n~ + sum_a sum_L Q_L^a g_L^a(r)   (neutral smooth charge, no Ewald sum required),
//   dE_a = E0 + sum_ij B_ij D_ij + sum_ijkl D_ij C_ijkl D_kl + dExc_a(D).
// Radial functions follow the PAW-XML convention: partial waves/projectors are phi(r) with
// phi(r) Y_lm, densities and the zero potential are stored as f(r) with f(r) Y_00.
#pragma once
#include "base/common.hpp"
#include "base/linalg.hpp"
#include "math/radial.hpp"
#include "math/special.hpp"
#include "xc/lda.hpp"

namespace lsda {

class PawSetup {
 public:
  static PawSetup load(const std::string& path, int ntheta = 7);

  // ---- raw data ---------------------------------------------------------------
  std::string symbol, path, xc_name;
  double Z = 0, n_core = 0, n_valence = 0;
  double e_total_ae = 0;   // all-electron total energy of the reference atom
  double e_kin_core = 0;
  double r_paw = 0, rc_shape = 0;
  double ecut_low = 0, ecut_high = 0;  // recommended plane-wave cutoffs (Ha), 0 if unknown
  RadialGrid grid;
  int ncut = 0;            // number of radial points used for integrals over the augmentation sphere
  int nj = 0;              // radial partial waves
  std::vector<int> l_j;
  std::vector<double> f_j, e_j;
  std::vector<std::vector<double>> phi, phit, proj;
  std::vector<double> nc, nct, nvt, vbar;

  // ---- projector functions i = (j, m) ------------------------------------------
  int ni = 0, lmax = 0, nL = 0;  // nL = (2 lmax + 1)^2 multipoles of the compensation charge
  std::vector<int> j_i, l_i, L_i;

  // ---- on-site tensors ---------------------------------------------------------
  RMatrix dS;                     // overlap correction <phi_i|phi_j> - <phi~_i|phi~_j>
  RMatrix dN;                     // all-electron norm inside the sphere (for local moments)
  RMatrix B;                      // linear energy term
  RMatrix C;                      // (ni*ni) x (ni*ni) Coulomb tensor
  std::vector<double> Delta_Lij;  // multipole moments Delta(L, i, j), index (L*ni + i)*ni + j
  double Delta0 = 0;              // constant part of Q_00
  double E0 = 0;                  // constant energy
  double delta(int L, int i, int j) const { return Delta_Lij[(static_cast<size_t>(L) * ni + i) * ni + j]; }

  // Compensation multipoles Q_L = Delta0 delta_L0 + sum_ij Delta_Lij D_ij (D summed over spin)
  void multipoles(const RMatrix& Dtot, double* Q) const;

  // On-site energy and Hamiltonian. D[s] are the spin density matrices (nspin = 1: D[0] is the total).
  // dH[s] is set to dE_a/dD^s_ij (without the term sum_L Delta_Lij int v~_H g_L, which lives on the grid).
  // If dH == nullptr only the energy is computed.
  double onsite(int nspin, const RMatrix* D, RMatrix* dH, const XCFunctional& xc) const;

  // Spin-summed density matrix of the spherical reference atom
  RMatrix atomic_density_matrix() const;

  // ---- reciprocal-space form factors -------------------------------------------
  // Fourier transforms: f(G) = int f(r) exp(-iG.r) d^3r for the spherical functions,
  // radial Bessel transforms int f(r) j_l(qr) r^2 dr for projectors and shape functions.
  void build_tables(double qmax_wfc, double qmax_dense);
  std::vector<RadialTable> proj_q;  // per radial function j
  std::vector<RadialTable> phit_q;  // pseudo partial waves of bound states (start wavefunctions), else empty
  bool is_bound(int j) const { return e_j[j] < 0.0; }
  std::vector<RadialTable> g_q;     // per l = 0..2 lmax
  RadialTable nct_q, nvt_q, vbar_q;

  // diagnostics (filled by load)
  double check_core_charge = 0, check_duality = 0;

 private:
  int nq_ = 0;  // radial pairs
  int pair(int j1, int j2) const { return j1 <= j2 ? j1 * nj - j1 * (j1 - 1) / 2 + (j2 - j1) : pair(j2, j1); }
  std::vector<std::vector<double>> nq_ae_, nq_ps_;  // pair densities phi_j1 phi_j2 on r < r(ncut)
  std::vector<std::vector<double>> dnq_ae_, dnq_ps_;  // their radial derivatives (gradient-corrected functionals)
  std::vector<double> dnc_, dnct_;                  // radial derivatives of the core densities
  std::vector<double> ylm_grad_quad_;               // tangential gradients of Y_L at the quadrature points (3 nL per point)
  double onsite_xc_gga(int nspin, const std::vector<RMatrix>& dlq, std::vector<RMatrix>* vlq, const XCFunctional& xc) const;
  std::vector<double> wr2_;                         // radial weights * r^2
  std::vector<std::vector<double>> shape_;          // g_l(r), l = 0..2 lmax
  SphereQuadrature quad_;
  RMatrix ylm_quad_;                                // nL x npoints
  struct GauntEntry { int L, q; double g; };
  std::vector<std::vector<GauntEntry>> gaunt_ij_;   // per (i*ni + j)
  void compute_onsite_tensors();
};

}  // namespace lsda
