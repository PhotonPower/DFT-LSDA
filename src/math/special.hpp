// Spherical Bessel functions, real spherical harmonics, Gaunt coefficients,
// sphere quadrature and rotation matrices of real spherical harmonics.
#pragma once
#include "base/common.hpp"
#include "base/linalg.hpp"

namespace lsda {

// j_l(x) for l = 0..lmax
void sph_bessel(int lmax, double x, double* jl);
double sph_bessel(int l, double x);

// Real spherical harmonics Y_lm(v/|v|) for l <= lmax, index lm = l*l + l + m (m = -l..l).
// m > 0: ~cos(m phi), m < 0: ~sin(|m| phi), no Condon-Shortley phase. For v = 0 only Y_00 is nonzero.
void real_ylm(int lmax, const double* v, double* ylm);
inline int lm_index(int l, int m) { return l * l + l + m; }

// Product quadrature on the unit sphere (Gauss-Legendre in cos(theta), uniform in phi),
// exact for spherical harmonics up to degree 2*ntheta-1. Weights sum to 4 pi.
struct SphereQuadrature {
  std::vector<Vec3> points;
  std::vector<double> weights;
};
SphereQuadrature make_sphere_quadrature(int ntheta);
void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w);  // on [-1,1]

// Real Gaunt coefficients G(L, L1, L2) = int Y_L Y_L1 Y_L2 dOmega,
// L < (2*lmax+1)^2 and L1, L2 < (lmax+1)^2.
class GauntTable {
 public:
  explicit GauntTable(int lmax);
  double operator()(int L, int L1, int L2) const { return g_[(static_cast<size_t>(L) * n1_ + L1) * n1_ + L2]; }
  int lmax() const { return lmax_; }
  int nL() const { return nL_; }

 private:
  int lmax_, nL_, n1_;
  std::vector<double> g_;
};

// Rotation matrix of real spherical harmonics for the Cartesian orthogonal matrix R (proper or improper):
//   Y_lm(R u) = sum_m' D(lm', lm) Y_lm'(u).  Returned matrix has dimension (lmax+1)^2, block diagonal in l.
RMatrix ylm_rotation(int lmax, const Mat3& R);

}  // namespace lsda
