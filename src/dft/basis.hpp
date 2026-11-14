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
  std::vector<double> kin;   // |k+G|^2/2
  std::vector<double> kg;    // Cartesian k+G (3 per plane wave)
};

// FFT dimensions able to represent all G with |G| <= gmax
void fft_dimensions(const Crystal& c, double gmax, int n[3]);
GSphere make_gsphere(const Crystal& c, double ecut_dense, const FFT3D& fine, const FFT3D& coarse);
PWBasis make_pw_basis(const Crystal& c, const KPoint& k, double ecut, const FFT3D& coarse);

}  // namespace lsda
