// Plane-wave basis sets and FFT grids.
//   wavefunctions : |k+G|^2/2 <= ecut, stored as coefficient vectors per k-point
//   coarse grid   : holds all products psi* psi and v psi without aliasing (G up to 2 k_max)
//   dense sphere  : |G|^2/2 <= ecut_dense, used for densities, compensation charges and potentials
#pragma once
#include "base/common.hpp"
#include "base/fft.hpp"
#include "crystal/crystal.hpp"

namespace lsda {

struct GSphere {
  int ng = 0;
  std::vector<int> m;             // integer components (3 per vector)
  std::vector<double> g;          // Cartesian components (3 per vector)
  std::vector<double> g2;         // |G|^2; G = 0 is the first vector
  std::vector<size_t> idx_fine;   // position in the fine FFT box
  std::vector<long> idx_coarse;   // position in the coarse FFT box, -1 if not representable there
  std::vector<int> lookup;        // fine-box index -> sphere index (-1 outside the sphere)
};

struct PWBasis {
  Vec3 kfrac{}, kcart{};
  double weight = 0;
  int npw = 0;
  std::vector<size_t> idx;   // position of each plane wave in the coarse FFT box
  std::vector<int> mill;     // integer components of G (3 per plane wave)
  std::vector<int> columns;  // FFT-box columns (i1 * n2 + i2) and
  std::vector<int> planes;   // i1 planes that contain plane waves (for the pruned FFT)
  std::vector<double> kin;   // kinetic energy T(|k+G|^2/2); equals |k+G|^2/2 unless the cutoff is smoothed
  std::vector<double> dkin;  // dT/d(|k+G|^2/2)  (1 without smoothing); needed for the stress
  std::vector<double> kg;    // Cartesian k+G (3 per plane wave)
};

// FFT dimensions able to represent all G with |G| <= gmax
void fft_dimensions(const Crystal& c, double gmax, int n[3]);
GSphere make_gsphere(const Crystal& c, double ecut_dense, const FFT3D& fine, const FFT3D& coarse);
// ecut_smooth > 0: the kinetic energy t = |k+G|^2/2 is replaced by t / (x^2 (3 + x - 6x^2 + 3x^3)),
// x = (ecut - t)/ecut_smooth, for t within ecut_smooth of the cutoff (Bernasconi et al., J. Phys. Chem.
// Solids 56, 501 (1995); "ecutsm" in ABINIT). Plane waves then enter and leave the basis smoothly when the
// cell changes, so that E(cell) at constant cutoff is differentiable and the stress is its derivative.
PWBasis make_pw_basis(const Crystal& c, const KPoint& k, double ecut, const FFT3D& coarse, double ecut_smooth = 0.0);

}  // namespace lsda
