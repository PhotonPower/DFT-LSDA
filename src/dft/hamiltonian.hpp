// PAW Hamiltonian and overlap operator for one k-point and spin channel in the plane-wave basis:
//   H = -1/2 nabla^2 + v~(r) + sum_a sum_ij |p_i^a> dH_ij^a <p_j^a|,   S = 1 + sum_a sum_ij |p_i^a> dS_ij^a <p_j^a|
#pragma once
#include <array>

#include "base/common.hpp"
#include "base/fft.hpp"
#include "base/linalg.hpp"
#include "crystal/crystal.hpp"
#include "dft/basis.hpp"

namespace lsda {

// Projector functions of all atoms in the plane-wave basis of one k-point:
//   beta(G, i) = <k+G|p_i^a> = 4 pi/sqrt(Omega) (-i)^l Y_lm(k+G) p_j(|k+G|) exp(-i (k+G).R_a)
struct Projectors {
  CMatrix beta;              // npw x nproj
  std::vector<int> offset;   // first projector of each atom
  int nproj = 0;
};
Projectors make_projectors(const Crystal& c, const PWBasis& pw);
// Strain derivative of the projector form factors (symmetrized in a, b; the volume factor
// -1/2 delta_ab beta is not included):  dbeta = -1/2 (q_b d/dq_a + q_a d/dq_b) beta|_{fixed phase}
CMatrix make_projector_strain_derivative(const Crystal& c, const PWBasis& pw, int a, int b);

class Hamiltonian {
 public:
  // collinear case: one spin channel with its local potential and on-site matrices dH[atom]
  Hamiltonian(const PWBasis& pw, const FFT3D& fft, const double* vloc, const Projectors& proj,
              std::vector<const RMatrix*> dH, std::vector<const RMatrix*> dS)
      : pw_(pw), fft_(fft), vloc_{vloc, nullptr, nullptr, nullptr}, proj_(proj), dS_(std::move(dS)) {
    dH_.resize(dH.size());
    for (size_t a = 0; a < dH.size(); ++a) dH_[a] = {dH[a], nullptr, nullptr, nullptr};
  }
  // non-collinear case: two-component spinors x = (x_up, x_down), stored one after the other.
  // Local potential V = v[0] + sum_c v[c] sigma_c and on-site matrices dH = dH[0] + sum_c dH[c] sigma_c
  // (c = 1, 2, 3 for x, y, z; sigma: Pauli matrices).
  Hamiltonian(const PWBasis& pw, const FFT3D& fft, std::array<const double*, 4> vloc, const Projectors& proj,
              std::vector<std::array<const RMatrix*, 4>> dH, std::vector<const RMatrix*> dS)
      : pw_(pw), fft_(fft), vloc_(vloc), proj_(proj), dH_(std::move(dH)), dS_(std::move(dS)), spinor_(true) {}

  int npw() const { return spinor_ ? 2 * pw_.npw : pw_.npw; }  // dimension of the coefficient vectors
  bool spinor() const { return spinor_; }
  const PWBasis& basis() const { return pw_; }
  double kinetic(int i) const { return pw_.kin[spinor_ && i >= pw_.npw ? i - pw_.npw : i]; }
  // hx = H x, sx = S x for nb vectors stored column-wise (leading dimension npw())
  void apply(const cplx* x, int nb, cplx* hx, cplx* sx) const;
  // Building blocks, used by solvers that work with the projections <p_i|x> instead of H x and S x:
  //   <a|H|b> = a^H (K b) + (beta^H a)^H dH (beta^H b),   <a|S|b> = a^H b + (beta^H a)^H dS (beta^H b)
  void apply_local(const cplx* x, int nb, cplx* kx) const;   // K x = (-1/2 nabla^2 + v~) x
  CMatrix project(const cplx* x, int nb) const;              // beta^H x  (nproj() x nb)
  void block_multiply(const CMatrix& P, CMatrix* DP, CMatrix* SP) const;  // dH P and/or dS P
  void add_nonlocal(const CMatrix& T, cplx* r, int nb) const;             // r += beta T
  int nproj() const { return spinor_ ? 2 * proj_.nproj : proj_.nproj; }
  // exact exchange in compressed form, V_x = -sum_t c_t xi_t xi_t^H (hybrid functionals); part of apply_local
  void add_exchange(const CMatrix* xi, double coefficient) { xi_.push_back({xi, coefficient}); }
  // meta-GGA: additional term -1/2 div(v_tau(r) grad) with v_tau on the coarse grid (collinear case only)
  void set_tau_potential(const double* vtau) { vtau_ = vtau; }
  mutable long n_applications = 0;

 private:
  const PWBasis& pw_;
  const FFT3D& fft_;
  std::array<const double*, 4> vloc_;  // local potential on the coarse real-space grid
  const Projectors& proj_;
  std::vector<std::array<const RMatrix*, 4>> dH_;  // per atom
  std::vector<const RMatrix*> dS_;
  bool spinor_ = false;
  std::vector<std::pair<const CMatrix*, double>> xi_;
  const double* vtau_ = nullptr;
};

}  // namespace lsda
