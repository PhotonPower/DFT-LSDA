// PAW Hamiltonian and overlap operator for one k-point and spin channel in the plane-wave basis:
//   H = -1/2 nabla^2 + v~(r) + sum_a sum_ij |p_i^a> dH_ij^a <p_j^a|,   S = 1 + sum_a sum_ij |p_i^a> dS_ij^a <p_j^a|
#pragma once
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
  Hamiltonian(const PWBasis& pw, const FFT3D& fft, const double* vloc, const Projectors& proj,
              std::vector<const RMatrix*> dH, std::vector<const RMatrix*> dS)
      : pw_(pw), fft_(fft), vloc_(vloc), proj_(proj), dH_(std::move(dH)), dS_(std::move(dS)) {}

  int npw() const { return pw_.npw; }
  const PWBasis& basis() const { return pw_; }
  // hx = H x, sx = S x for nb vectors stored column-wise (leading dimension npw)
  void apply(const cplx* x, int nb, cplx* hx, cplx* sx) const;
  // Building blocks, used by solvers that work with the projections <p_i|x> instead of H x and S x:
  //   <a|H|b> = a^H (K b) + (beta^H a)^H dH (beta^H b),   <a|S|b> = a^H b + (beta^H a)^H dS (beta^H b)
  void apply_local(const cplx* x, int nb, cplx* kx) const;   // K x = (-1/2 nabla^2 + v~) x
  CMatrix project(const cplx* x, int nb) const;              // beta^H x  (nproj x nb)
  void block_multiply(const CMatrix& P, CMatrix* DP, CMatrix* SP) const;  // dH P and/or dS P
  const CMatrix& beta() const { return proj_.beta; }
  int nproj() const { return proj_.nproj; }
  mutable long n_applications = 0;

 private:
  const PWBasis& pw_;
  const FFT3D& fft_;
  const double* vloc_;  // local potential on the coarse real-space grid
  const Projectors& proj_;
  std::vector<const RMatrix*> dH_, dS_;  // per atom
};

}  // namespace lsda
